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
    GET  /api/billing/plans           plans, packs, pricing in words
    POST /api/billing/checkout        {plan} or {pack} -> {url} Stripe Checkout
    POST /api/billing/portal          {url} Stripe billing portal (subscribers)
    POST /api/billing/webhook         Stripe events (signature verified)
    GET  /api/billing/ledger          the caller's credit history

Plans and the webhook
---------------------
A plan checkout carries `metadata.plan`; a pack checkout carries
`metadata.credits`. The webhook branches on that first, so a plan can never
fall into the pack path (which answers 500 for Stripe to retry when it
cannot credit). Monthly and yearly plans are Stripe subscriptions: the
checkout event grants a provisional 35 days, then `customer.subscription.*`
and `invoice.paid` events move `plan_until` to the real period end plus a
grace. Lifetime is a one-off payment with `plan_until` NULL. Every event is
applied through `Store.apply_once`, so Stripe's retries are no-ops.
"""

from __future__ import annotations

import json
import logging
import re
import time
from html import escape as html_escape
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .config import BillingConfig
from .gate import SESSION_COOKIE
from .store import GRACE_DAYS, Store
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
    plan: str = Field(default="", max_length=32)


#: Provisional plan length granted by a subscription checkout before the
#: first `customer.subscription.*` event reports the real period end.
PROVISIONAL_DAYS = 35


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


def _user_payload(user: Dict[str, Any], words_per_credit: int = 1) -> Dict[str, Any]:
    credits = int(user.get("credits") or 0)
    plan = str(user.get("plan") or "")
    return {
        "id": user["id"],
        "email": user["email"],
        "credits": credits,
        "words_left": credits * words_per_credit,
        "words_per_credit": words_per_credit,
        "plan": plan,
        "plan_active": Store.plan_active(user) if plan else False,
        "plan_until": user.get("plan_until"),
        "plan_allowance": int(user.get("plan_allowance") or 0),
        "has_billing_portal": bool(user.get("stripe_customer_id")),
        "is_admin": bool(user.get("is_admin")),
        # Admins (the operator) are never charged; the page shows "Unlimited".
        "unlimited": bool(user.get("is_admin")),
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
        response = JSONResponse({"user": _user_payload(user, cfg.words_per_credit)})
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

    def _fresh(user: Dict[str, Any]) -> Dict[str, Any]:
        """Re-read a user after applying any allowance that fell due."""
        store.topup_if_due(user["id"])
        fresh = store.user(user["id"]) or user
        return dict(fresh, session_expires=user.get("session_expires"))

    @app.get("/api/me")
    def me(request: Request) -> JSONResponse:
        """Who is paying: the signed-in user, or the guest row for this
        address (never created here; a visitor who has spent nothing sees the
        full guest grant), or nobody when guests are off."""
        user = _current_user(request)
        if user is None:
            guest = getattr(request.state, "guest_user", None)
            if guest is None:
                return JSONResponse({"user": None, "paywall": cfg.enabled})
            if guest.get("virtual"):
                payload = _user_payload(dict(guest, id="guest"), cfg.words_per_credit)
            else:
                payload = _user_payload(_fresh(guest), cfg.words_per_credit)
            # The address is the key; it is not the visitor's business to see it echoed.
            payload.update({"guest": True, "email": "", "id": "guest"})
            return JSONResponse({"user": payload, "paywall": cfg.enabled, "guest": True})
        return JSONResponse({"user": dict(_user_payload(_fresh(user), cfg.words_per_credit), guest=False), "paywall": cfg.enabled})

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
            {"stripe": cfg.stripe_configured, "words_per_credit": cfg.words_per_credit, "packs": cfg.public_packs()}
        )

    @app.get("/api/billing/plans")
    def plans() -> JSONResponse:
        return JSONResponse(
            {
                "stripe": cfg.stripe_configured,
                "paywall": cfg.enabled,
                "words_per_credit": cfg.words_per_credit,
                "free_credits": cfg.free_credits,
                "free_words": cfg.free_credits * cfg.words_per_credit,
                "plans": cfg.public_plans(),
                "packs": cfg.public_packs(),
            }
        )

    @app.post("/api/billing/checkout")
    def checkout(request: Request, body: CheckoutBody) -> JSONResponse:
        """Start a Stripe Checkout for `{plan}` or `{pack}` (plan wins).

        Subscriptions carry the user id and plan name on the Subscription
        object too (`subscription_data[metadata]`), so every later renewal
        event can be matched without the checkout session.
        """
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in before buying credits.", signin_url="/signin")
        if not cfg.stripe_configured:
            return _err(503, "stripe_unconfigured", "Payments are not configured on this server.")
        plan_name = body.plan.strip().lower()
        pack_name = body.pack.strip().lower()
        if not plan_name and not pack_name:
            return _err(400, "nothing_chosen", "Send {plan} or {pack}.")
        fresh = store.user(user["id"]) or user
        customer = fresh.get("stripe_customer_id") or None
        kwargs: Dict[str, Any] = dict(
            success_url="%s/app?purchase=success" % cfg.public_url,
            cancel_url="%s/app?purchase=cancelled" % cfg.public_url,
            customer_email=user["email"],
            customer=customer,
        )
        if plan_name:
            plan = cfg.plan(plan_name)
            if plan is None:
                return _err(400, "unknown_plan", "Choose one of: %s" % ", ".join(p.name for p in cfg.plans))
            if plan.recurring and Store.plan_active(fresh) and fresh.get("stripe_subscription_id"):
                return _err(
                    409,
                    "already_subscribed",
                    "You already have an active subscription. Change or cancel it in the billing portal first.",
                    portal="/api/billing/portal",
                )
            meta = {"user_id": user["id"], "plan": plan.name}
            price_id = plan.price_id
            kwargs["metadata"] = meta
            if plan.recurring:
                kwargs["mode"] = "subscription"
                kwargs["subscription_metadata"] = meta
            else:
                kwargs["mode"] = "payment"
            label: Dict[str, Any] = {"plan": plan.name, "mode": kwargs["mode"]}
        else:
            pack = cfg.pack(pack_name)
            if pack is None:
                return _err(400, "unknown_pack", "Choose one of: %s" % ", ".join(p.name for p in cfg.packs))
            price_id = pack.price_id
            kwargs["mode"] = "payment"
            kwargs["metadata"] = {"user_id": user["id"], "credits": str(pack.credits), "pack": pack.name}
            label = {"pack": pack.name, "mode": "payment"}
        try:
            session = stripe_client.create_checkout_session(cfg.stripe_secret_key, price_id, 1, **kwargs)
        except stripe_client.StripeError as exc:
            log.error("checkout failed: %s", exc)
            return _err(502, "stripe_error", str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("checkout failed")
            return _err(502, "stripe_error", "Could not reach Stripe: %s" % exc)
        out = {"url": session.get("url"), "session_id": session.get("id")}
        out.update(label)
        return JSONResponse(out)

    @app.post("/api/billing/portal")
    def portal(request: Request) -> JSONResponse:
        """A Stripe billing portal link for the caller's stored customer."""
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in first.", signin_url="/signin")
        if not cfg.stripe_secret_key:
            return _err(503, "stripe_unconfigured", "Payments are not configured on this server.")
        fresh = store.user(user["id"]) or user
        customer = fresh.get("stripe_customer_id")
        if not customer:
            return _err(404, "no_customer", "No Stripe customer is linked to this account yet. Buy a plan first.")
        try:
            session = stripe_client.create_portal_session(cfg.stripe_secret_key, customer, "%s/app" % cfg.public_url)
        except stripe_client.StripeError as exc:
            log.error("portal failed: %s", exc)
            return _err(502, "stripe_error", str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("portal failed")
            return _err(502, "stripe_error", "Could not reach Stripe: %s" % exc)
        return JSONResponse({"url": session.get("url")})

    # -- webhook ----------------------------------------------------------

    def _find_user(meta: Dict[str, Any], customer: Any, subscription: Any = None) -> Optional[Dict[str, Any]]:
        """By metadata.user_id, then the stored customer id, then the subscription id."""
        uid = str(meta.get("user_id") or "")
        user = store.user(uid) if uid else None
        if user is None and isinstance(customer, str):
            user = store.user_by_customer(customer)
        if user is None and isinstance(subscription, str):
            user = store.user_by_subscription(subscription)
        return user

    def _period_end(sub: Dict[str, Any]) -> Optional[float]:
        """`current_period_end` from the subscription, wherever this API version puts it."""
        end = sub.get("current_period_end")
        if end is None:
            items = ((sub.get("items") or {}).get("data") or [])
            if items and isinstance(items[0], dict):
                end = items[0].get("current_period_end")
        try:
            return float(end) if end is not None else None
        except (TypeError, ValueError):
            return None

    def _apply_plan_checkout(event_id: str, event_type: str, obj: Dict[str, Any], plan_name: str) -> JSONResponse:
        plan = cfg.plan(plan_name)
        meta = obj.get("metadata") or {}
        customer = obj.get("customer") if isinstance(obj.get("customer"), str) else None
        subscription = obj.get("subscription") if isinstance(obj.get("subscription"), str) else None
        user = _find_user(meta, customer)
        if plan is None or user is None:
            # Paid but unmatched: 500 so Stripe retries and the dashboard
            # shows it; the operator can grant with `admin plan` meanwhile.
            log.error("webhook %s: cannot apply plan (user=%r plan=%r session=%r)", event_id, meta.get("user_id"), plan_name, obj.get("id"))
            return _err(500, "cannot_apply_plan", "Paid session has no matching user or plan; operator alerted.")
        now = time.time()
        until = None if not plan.recurring else now + PROVISIONAL_DAYS * 86400

        def apply() -> Dict[str, Any]:
            return store.set_plan(user["id"], plan.name, plan.allowance_credits, until, customer_id=customer, subscription_id=subscription, now=now)

        try:
            row = store.apply_once(event_id, event_type, apply)
        except Exception as exc:  # noqa: BLE001 - Stripe must retry
            log.exception("webhook %s: plan failed", event_id)
            return _err(500, "plan_failed", "Could not apply the plan: %s" % exc)
        if row is None:
            return JSONResponse({"received": True, "duplicate": True})
        log.info("plan %s for %s until %r (balance %d) for %s", plan.name, user["id"], until, row["credits"], event_id)
        return JSONResponse({"received": True, "plan": plan.name, "plan_until": until, "balance": int(row["credits"])})

    def _apply_subscription(event_id: str, event_type: str, sub: Dict[str, Any]) -> JSONResponse:
        meta = sub.get("metadata") or {}
        customer = sub.get("customer") if isinstance(sub.get("customer"), str) else None
        sub_id = sub.get("id") if isinstance(sub.get("id"), str) else None
        user = _find_user(meta, customer, sub_id)
        if user is None:
            store.record_webhook(event_id, event_type)
            log.warning("webhook %s: subscription %r for no known user", event_id, sub_id)
            return JSONResponse({"received": True, "ignored": "unknown_user"})
        status = str(sub.get("status") or "")
        live = event_type != "customer.subscription.deleted" and status in ("active", "trialing")
        now = time.time()
        end = _period_end(sub)
        until = (end + GRACE_DAYS * 86400) if (live and end is not None) else now
        plan = cfg.plan(str(meta.get("plan") or ""))

        def apply() -> Optional[float]:
            store.set_stripe_ids(user["id"], customer, sub_id)
            if live and plan is not None and (user.get("plan") != plan.name or not Store.plan_active(user, now)):
                # The subscription event beat (or replaced) the checkout
                # event: put the user on the plan from the subscription's
                # own metadata so the allowance starts regardless of order.
                store.set_plan(user["id"], plan.name, plan.allowance_credits, until, customer_id=customer, subscription_id=sub_id, now=now)
            else:
                store.set_plan_until(user["id"], until)
            return until

        try:
            result = store.apply_once(event_id, event_type, apply)
        except Exception as exc:  # noqa: BLE001
            log.exception("webhook %s: subscription update failed", event_id)
            return _err(500, "subscription_failed", "Could not apply the subscription event: %s" % exc)
        if result is None:
            return JSONResponse({"received": True, "duplicate": True})
        return JSONResponse({"received": True, "status": status, "plan_until": result})

    def _apply_invoice(event_id: str, event_type: str, inv: Dict[str, Any]) -> JSONResponse:
        customer = inv.get("customer") if isinstance(inv.get("customer"), str) else None
        sub_id = inv.get("subscription") if isinstance(inv.get("subscription"), str) else None
        if sub_id is None:
            details = ((inv.get("parent") or {}).get("subscription_details") or {})
            sub_id = details.get("subscription") if isinstance(details.get("subscription"), str) else None
        ends = []
        for line in ((inv.get("lines") or {}).get("data") or []):
            period = (line.get("period") or {}) if isinstance(line, dict) else {}
            try:
                ends.append(float(period.get("end")))
            except (TypeError, ValueError):
                pass
        user = _find_user({}, customer, sub_id)
        if user is None or not ends or not user.get("plan"):
            store.record_webhook(event_id, event_type)
            return JSONResponse({"received": True, "ignored": "no_subscriber"})
        until = max(ends) + GRACE_DAYS * 86400

        def apply() -> Optional[float]:
            current = user.get("plan_until")
            if current is None or float(current) >= until:
                return None
            store.set_plan_until(user["id"], until)
            return until

        try:
            result = store.apply_once(event_id, event_type, apply)
        except Exception as exc:  # noqa: BLE001
            log.exception("webhook %s: invoice failed", event_id)
            return _err(500, "invoice_failed", "Could not apply the invoice: %s" % exc)
        return JSONResponse({"received": True, "plan_until": result})

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
        obj = (event.get("data") or {}).get("object") or {}
        if not isinstance(obj, dict):
            obj = {}

        if event_type in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
            return _apply_subscription(event_id, event_type, obj)
        if event_type == "invoice.paid":
            return _apply_invoice(event_id, event_type, obj)
        if event_type not in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
            store.record_webhook(event_id, event_type)
            return JSONResponse({"received": True, "ignored": event_type})

        if obj.get("payment_status") not in (None, "paid"):
            return JSONResponse({"received": True, "ignored": "unpaid"})
        meta = obj.get("metadata") or {}
        plan_name = str(meta.get("plan") or "").strip().lower()
        if plan_name:
            return _apply_plan_checkout(event_id, event_type, obj, plan_name)

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
        customer = obj.get("customer")
        if isinstance(customer, str) and customer:
            store.set_stripe_ids(user_id, customer_id=customer)
        log.info("credited %d to %s (balance %d) for %s", credits, user_id, balance, event_id)
        return JSONResponse({"received": True, "credited": credits, "balance": balance})

    @app.get("/api/billing/ledger")
    def ledger(request: Request) -> JSONResponse:
        user = _current_user(request)
        if user is None:
            return _err(401, "login_required", "Sign in first.", signin_url="/signin")
        fresh = _fresh(user)
        return JSONResponse(
            {
                "balance": int(fresh.get("credits") or 0),
                "words_left": int(fresh.get("credits") or 0) * cfg.words_per_credit,
                "plan": fresh.get("plan") or "",
                "plan_until": fresh.get("plan_until"),
                "entries": store.ledger(user["id"]),
            }
        )

    # Same fix `server._register_llm_routes` applies: the static Mount at "/"
    # matches everything, so it must come after every route just added.
    from starlette.routing import Mount

    routes = app.router.routes
    mounts = [r for r in routes if isinstance(r, Mount)]
    if mounts:
        app.router.routes = [r for r in routes if not isinstance(r, Mount)] + mounts
    return app
