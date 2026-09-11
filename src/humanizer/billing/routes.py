"""Auth and billing routes: `/api/auth/*`, `/api/billing/*`, `/api/me`.

Registered onto an app built by `humanizer.api.server.create_app` by
`billing.app.install()`, using the same append-then-move-the-mount trick the
server's own feature blocks use, so nothing in `server.py` changes.

Login is by magic link. There are no passwords to store or leak. A link is a
signed token (`tokens.sign`) that also carries a single-use nonce recorded in
the database, so forwarding the email to someone else after use does nothing.
Verifying it sets an HttpOnly session cookie; API clients can instead create a
bearer key at `POST /api/auth/keys`.

Endpoints
---------
    GET  /api/billing/health          paywall on/off, packs, mail transport
    POST /api/auth/request-link       {email} -> sends the link (dev: returns it)
    POST /api/auth/verify             {token} -> sets cookie, returns user
    GET  /api/auth/verify?token=&next=  same, then redirects (for the email link)
    POST /api/auth/logout
    GET  /api/me                      user, credits, session expiry
    DELETE /api/me                    delete the account
    GET  /api/auth/keys               list API keys
    POST /api/auth/keys               {label} -> {key} (shown once)
    DELETE /api/auth/keys/{id}
    GET  /api/billing/packs           purchasable packs
    POST /api/billing/checkout        {pack} -> {url} Stripe Checkout
    POST /api/billing/webhook         Stripe events (signature verified)
    GET  /api/billing/ledger          the caller's credit history
"""

from __future__ import annotations

import json
import logging
import re
from html import escape as html_escape
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .config import BillingConfig
from .gate import SESSION_COOKIE
from .store import Store
from . import mail as mail_mod
from . import stripe_client
from . import tokens

log = logging.getLogger("humanizer.billing.routes")

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")


class EmailBody(BaseModel):
    email: str = Field(default="", max_length=320)


class TokenBody(BaseModel):
    token: str = Field(default="", max_length=2000)


class KeyBody(BaseModel):
    label: str = Field(default="", max_length=80)


class CheckoutBody(BaseModel):
    pack: str = Field(default="", max_length=32)


#: Served by GET /api/auth/verify. Self-contained: no app assets, no fonts.
_VERIFY_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex"><title>Signing in to {product}</title>
<style>
html{background:#0a0b0d;color:#ece8df;font:16px/1.6 "Fira Sans",ui-sans-serif,-apple-system,"Segoe UI",Arial,sans-serif}
body{margin:0;display:grid;min-height:100vh;place-items:center;padding:20px}
main{max-width:34rem;background:#15181d;border:1px solid rgba(230,220,200,.22);border-radius:4px;padding:24px}
h1{font-size:1.125rem;font-weight:500;margin:0 0 8px}p{color:#b5b0a5;margin:8px 0}
button{height:36px;padding:0 16px;border-radius:4px;border:0;background:#f0a63a;color:#1a1204;font:inherit;font-weight:500;cursor:pointer}
button:focus-visible{outline:2px solid #f0a63a;outline-offset:2px}
</style></head><body><main>
<h1 id="h">Signing you in to {product}</h1>
<p id="p">One moment.</p>
<form id="f"><button type="submit">Continue</button></form>
<noscript><p>JavaScript is off. Press Continue to finish signing in.</p></noscript>
<script>
(function(){var t={token},n={next},h=document.getElementById('h'),p=document.getElementById('p'),f=document.getElementById('f');
function go(){fetch('/api/auth/verify',{method:'POST',credentials:'same-origin',headers:{'content-type':'application/json'},body:JSON.stringify({token:t})})
.then(function(r){return r.json().then(function(b){return {ok:r.ok,b:b}})})
.then(function(x){if(x.ok){location.replace(n)}else{h.textContent='This link did not work';p.textContent=(x.b&&x.b.detail)||'Request a new sign-in link.';f.style.display='none'}})
.catch(function(){h.textContent='Could not reach the server';p.textContent='Check your connection and press Continue.'})}
f.addEventListener('submit',function(e){e.preventDefault();go()});go();})();
</script></main></body></html>"""


def _err(status: int, error: str, detail: str, **extra: Any) -> JSONResponse:
    body: Dict[str, Any] = {"error": error, "detail": detail}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


def _user_payload(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "email": user["email"],
        "credits": int(user.get("credits") or 0),
        "is_admin": bool(user.get("is_admin")),
        "created": user.get("created"),
        "session_expires": user.get("session_expires"),
    }


def _current_user(request: Request) -> Optional[Dict[str, Any]]:
    # Set by PaywallMiddleware for every /api/ request.
    return getattr(request.state, "billing_user", None)


def _set_session_cookie(response: Any, cfg: BillingConfig, session_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=cfg.session_days * 86400,
        httponly=True,
        secure=cfg.secure_cookies,
        samesite="lax",
        path="/",
    )


def register_billing_routes(app: FastAPI, cfg: BillingConfig, store: Store) -> FastAPI:
    app.state.billing_config = cfg
    app.state.billing_store = store

    # -- health -----------------------------------------------------------

    @app.get("/api/billing/health")
    def billing_health() -> JSONResponse:
        return JSONResponse(cfg.public())

    # -- login ------------------------------------------------------------

    @app.post("/api/auth/request-link")
    def request_link(request: Request, body: EmailBody) -> JSONResponse:
        email = body.email.strip().lower()
        if not _EMAIL_RE.match(email):
            return _err(400, "bad_email", "Enter a valid email address.")
        domain = email.rsplit("@", 1)[1]
        if domain in cfg.blocked_domains:
            return _err(400, "blocked_email", "Disposable email addresses cannot be used. Use a regular address.")
        ip = getattr(request.state, "client_ip", None) or (request.client.host if request.client else "unknown")
        if store.user_by_email(email) is None and cfg.signups_per_ip_per_day > 0:
            if store.signups_from(ip, 86400.0) >= cfg.signups_per_ip_per_day:
                return _err(
                    429,
                    "signup_limit",
                    "Too many new accounts from this address today. Sign in to an existing account or try tomorrow.",
                )
        store.create_or_get(email, cfg.free_credits, ip=ip)
        nonce = store.issue_nonce(email)
        token = tokens.sign(
            cfg.secret, "login", {"email": email, "nonce": nonce}, cfg.magic_link_minutes * 60
        )
        link = "%s/api/auth/verify?token=%s&next=/" % (cfg.public_url, token)
        try:
            transport = mail_mod.send_magic_link(
                cfg.smtp_url, cfg.mail_from, email, link, cfg.magic_link_minutes, cfg.product_name
            )
        except Exception as exc:  # noqa: BLE001 - report, do not 500
            log.exception("could not send magic link")
            return _err(502, "mail_failed", "Could not send the sign-in email: %s" % exc)
        payload: Dict[str, Any] = {
            "sent": True,
            "email": email,
            "expires_in_minutes": cfg.magic_link_minutes,
            "transport": transport,
        }
        if cfg.dev_links:
            payload["dev_link"] = link
            payload["dev_token"] = token
        return JSONResponse(payload)

    def _verify(token: str) -> Any:
        data = tokens.verify(cfg.secret, "login", token)
        if data is None:
            return _err(400, "bad_token", "This sign-in link is invalid or has expired. Request a new one.")
        email = str(data.get("email") or "")
        nonce = str(data.get("nonce") or "")
        if not store.consume_nonce(nonce, email):
            return _err(400, "used_token", "This sign-in link was already used. Request a new one.")
        user = store.get_or_create_user(email, cfg.free_credits)
        session = store.create_session(user["id"], cfg.session_days * 86400)
        return user, session

    @app.post("/api/auth/verify")
    def verify_post(body: TokenBody) -> JSONResponse:
        result = _verify(body.token.strip())
        if isinstance(result, JSONResponse):
            return result
        user, session = result
        user = dict(user, session_expires=session["expires"])
        response = JSONResponse({"user": _user_payload(user)})
        _set_session_cookie(response, cfg, session["id"])
        return response

    @app.get("/api/auth/verify")
    def verify_get(token: str = "", next: str = "/") -> Any:
        """The link in the email. Renders a tiny page that POSTs the token.

        A GET must not consume the nonce: Outlook SafeLinks, Gmail and
        corporate proxies fetch links in email before the person clicks, and
        a consuming GET would hand them a dead link. The page posts with
        `fetch`, which those scanners do not run, then goes to `next`.
        """
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        if not token or tokens.verify(cfg.secret, "login", token) is None:
            return RedirectResponse(url="/?login_error=bad_token", status_code=303)
        page = _VERIFY_PAGE.replace("{product}", html_escape(cfg.product_name)).replace(
            "{token}", json.dumps(token)
        ).replace("{next}", json.dumps(target))
        return HTMLResponse(page, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})

    @app.post("/api/auth/logout")
    def logout(request: Request) -> JSONResponse:
        sid = request.cookies.get(SESSION_COOKIE)
        if sid:
            store.revoke_session(sid)
        response = JSONResponse({"logged_out": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/api/me")
    def me(request: Request) -> JSONResponse:
        user = _current_user(request)
        if user is None:
            return JSONResponse({"user": None, "paywall": cfg.enabled})
        fresh = store.user(user["id"]) or user
        fresh = dict(fresh, session_expires=user.get("session_expires"))
        return JSONResponse({"user": _user_payload(fresh), "paywall": cfg.enabled})

    @app.delete("/api/me")
    def delete_me(request: Request) -> JSONResponse:
        """Delete the caller's account: sessions and keys revoked, email removed.

        Credits are forfeited and the ledger is kept under an anonymous id for
        accounting, which is what the privacy policy promises.
        """
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in first.")
        store.delete_user(user["id"])
        response = JSONResponse({"deleted": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    # -- API keys ---------------------------------------------------------

    @app.get("/api/auth/keys")
    def list_keys(request: Request) -> JSONResponse:
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in first.")
        return JSONResponse({"keys": store.list_api_keys(user["id"])})

    @app.post("/api/auth/keys")
    def create_key(request: Request, body: KeyBody) -> JSONResponse:
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in first.")
        if len(store.list_api_keys(user["id"])) >= 10:
            return _err(400, "too_many_keys", "Revoke an existing key first (limit 10).")
        created = store.create_api_key(user["id"], body.label)
        created["note"] = "Store this key now; it is not shown again. Send it as 'Authorization: Bearer <key>'."
        return JSONResponse(created, status_code=201)

    @app.delete("/api/auth/keys/{key_id}")
    def revoke_key(request: Request, key_id: str) -> JSONResponse:
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in first.")
        if not store.revoke_api_key(user["id"], key_id):
            return _err(404, "unknown_key", "No such key.")
        return JSONResponse({"revoked": key_id})

    # -- billing ----------------------------------------------------------

    @app.get("/api/billing/packs")
    def packs() -> JSONResponse:
        return JSONResponse(
            {
                "stripe": cfg.stripe_configured,
                "words_per_credit": cfg.words_per_credit,
                "packs": [
                    {"name": p.name, "credits": p.credits, "label": p.label} for p in cfg.packs
                ],
            }
        )

    @app.post("/api/billing/checkout")
    def checkout(request: Request, body: CheckoutBody) -> JSONResponse:
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in before buying credits.")
        if not cfg.stripe_configured:
            return _err(503, "stripe_unconfigured", "Payments are not configured on this server.")
        pack = cfg.pack(body.pack.strip().lower())
        if pack is None:
            return _err(400, "unknown_pack", "Choose one of: %s" % ", ".join(p.name for p in cfg.packs))
        try:
            session = stripe_client.create_checkout_session(
                cfg.stripe_secret_key,
                pack.price_id,
                1,
                success_url="%s/?purchase=success" % cfg.public_url,
                cancel_url="%s/?purchase=cancelled" % cfg.public_url,
                customer_email=user["email"],
                metadata={"user_id": user["id"], "credits": str(pack.credits), "pack": pack.name},
            )
        except stripe_client.StripeError as exc:
            log.error("checkout failed: %s", exc)
            return _err(502, "stripe_error", str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("checkout failed")
            return _err(502, "stripe_error", "Could not reach Stripe: %s" % exc)
        return JSONResponse({"url": session.get("url"), "session_id": session.get("id"), "pack": pack.name})

    @app.post("/api/billing/webhook")
    async def webhook(request: Request) -> JSONResponse:
        raw = await request.body()
        event = stripe_client.verify_webhook(
            cfg.stripe_webhook_secret, raw, request.headers.get("stripe-signature", "")
        )
        if event is None:
            return _err(400, "bad_signature", "Webhook signature could not be verified.")
        event_id = str(event.get("id") or "")
        event_type = str(event.get("type") or "")
        if not event_id:
            return _err(400, "bad_event", "Event has no id.")

        if event_type not in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
            store.record_webhook(event_id, event_type)
            return JSONResponse({"received": True, "ignored": event_type})

        obj = (event.get("data") or {}).get("object") or {}
        if obj.get("payment_status") not in (None, "paid"):
            return JSONResponse({"received": True, "ignored": "unpaid"})
        meta = obj.get("metadata") or {}
        user_id = str(meta.get("user_id") or obj.get("client_reference_id") or "")
        try:
            credits = int(meta.get("credits") or 0)
        except ValueError:
            credits = 0
        if not user_id or credits <= 0 or store.user(user_id) is None:
            # A paid session we cannot credit. Answer 500 so Stripe retries and
            # the failing endpoint shows in its dashboard; the operator can
            # then grant by hand from the session id.
            log.error("webhook %s: cannot credit (user=%r credits=%r session=%r)", event_id, user_id, credits, obj.get("id"))
            return _err(500, "cannot_credit", "Paid session has no matching user or credit amount; operator alerted.")
        try:
            balance = store.credit_once(event_id, event_type, user_id, credits, kind="purchase", ref=str(obj.get("id") or event_id))
        except Exception as exc:  # noqa: BLE001 - Stripe must retry
            log.exception("webhook %s: credit failed", event_id)
            return _err(500, "credit_failed", "Could not apply the credit: %s" % exc)
        if balance is None:
            return JSONResponse({"received": True, "duplicate": True})
        log.info("credited %d to %s (balance %d) for %s", credits, user_id, balance, event_id)
        return JSONResponse({"received": True, "credited": credits, "balance": balance})

    @app.get("/api/billing/ledger")
    def ledger(request: Request) -> JSONResponse:
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in first.")
        return JSONResponse({"balance": store.balance(user["id"]), "entries": store.ledger(user["id"])})

    # Same fix `server._register_llm_routes` applies: the static Mount at "/"
    # matches everything, so it must come after every route just added.
    from starlette.routing import Mount

    routes = app.router.routes
    mounts = [r for r in routes if isinstance(r, Mount)]
    if mounts:
        app.router.routes = [r for r in routes if not isinstance(r, Mount)] + mounts
    return app
