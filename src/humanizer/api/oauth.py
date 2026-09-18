"""Sign in with Google or Apple, alongside the email-and-password accounts.

Adds to `humanizer.api.auth` the two buttons every sign-in page has, using the
standard OpenID Connect authorization-code flow:

    GET  /api/auth/providers                      {"google": bool, "apple": bool}
    GET  /api/auth/oauth/{provider}/start?next=   302 to the provider
    GET  /api/auth/oauth/google/callback          code + state from Google
    POST /api/auth/oauth/apple/callback           code + state (+ user) from Apple

On success the person is signed in exactly as `/api/auth/signin` signs them
in: a row in `users` (created on first sign-in with an unusable random
password), a row in `sessions`, the `rh_session` cookie, then a 303 to `next`.
On any failure the browser is sent to `/signin?error=<code>` and the page
explains it; nothing is half-created.

Configuration, all environment variables:

    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET      from Google Cloud console
    APPLE_CLIENT_ID     the Services ID (e.g. com.readshuman.web)
    APPLE_TEAM_ID       10-character team id
    APPLE_KEY_ID        the Sign in with Apple key id
    APPLE_PRIVATE_KEY   the .p8 key, PEM text, or a path to the file
    HUMANIZER_PUBLIC_URL  https://readshuman.example.com; the redirect URIs
                        registered with both providers are this plus
                        /api/auth/oauth/google/callback and /apple/callback.
                        Unset: derived from the request (fine for localhost
                        with Google; Apple requires https).

A provider whose variables are missing is reported `false` by /providers and
its start URL redirects to `/signin?error=<provider>_unavailable`.

Why the id_token is trusted without a JWKS signature check: it is received
directly from the provider's token endpoint over TLS, in exchange for a code
and our client secret, so its origin is authenticated by the channel. The
claims are still checked (issuer, audience, expiry, nonce, verified email).
Apple's client secret is itself a JWT signed with ES256, which needs the
`cryptography` package (`pip install -e '.[oauth]'`); without it Apple is
reported unavailable and Google still works.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from . import auth as auth_mod

log = logging.getLogger("humanizer.api.oauth")

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
APPLE_AUTH = "https://appleid.apple.com/auth/authorize"
APPLE_TOKEN = "https://appleid.apple.com/auth/token"
STATE_TTL_S = 600.0
PROVIDERS = ("google", "apple")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """The claims of a JWT, without verifying its signature (see module doc)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a JWT")
    data = json.loads(_unb64url(parts[1]).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JWT payload is not an object")
    return data


# ----------------------------------------------------------------- config


class OAuthConfig:
    def __init__(self, env: Optional[Dict[str, str]] = None) -> None:
        e = os.environ if env is None else env
        g = lambda k: (e.get(k) or "").strip()  # noqa: E731
        self.public_url = g("HUMANIZER_PUBLIC_URL").rstrip("/")
        self.google_client_id = g("GOOGLE_CLIENT_ID")
        self.google_client_secret = g("GOOGLE_CLIENT_SECRET")
        self.apple_client_id = g("APPLE_CLIENT_ID")
        self.apple_team_id = g("APPLE_TEAM_ID")
        self.apple_key_id = g("APPLE_KEY_ID")
        key = g("APPLE_PRIVATE_KEY")
        if key and "BEGIN" not in key and os.path.isfile(os.path.expanduser(key)):
            key = open(os.path.expanduser(key)).read()
        self.apple_private_key = key

    @property
    def google(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def apple(self) -> bool:
        return bool(
            self.apple_client_id and self.apple_team_id and self.apple_key_id
            and self.apple_private_key and es256_available()
        )

    def enabled(self, provider: str) -> bool:
        return {"google": self.google, "apple": self.apple}.get(provider, False)

    def redirect_uri(self, request: Request, provider: str) -> str:
        base = self.public_url or str(request.base_url).rstrip("/")
        return "%s/api/auth/oauth/%s/callback" % (base, provider)


# ------------------------------------------------------------------ ES256


def es256_available() -> bool:
    try:
        from cryptography.hazmat.primitives import hashes  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import ec  # noqa: F401
        return True
    except ImportError:
        return False


def es256_jwt(private_key_pem: str, header: Dict[str, Any], payload: Dict[str, Any]) -> str:
    """A compact JWT signed with ES256, the format Apple's client secret needs."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":"), sort_keys=True).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    )
    der = key.sign(signing_input.encode("ascii"), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return signing_input + "." + _b64url(sig)


def apple_client_secret(cfg: OAuthConfig, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    return es256_jwt(
        cfg.apple_private_key,
        {"alg": "ES256", "kid": cfg.apple_key_id, "typ": "JWT"},
        {
            "iss": cfg.apple_team_id,
            "iat": int(now),
            "exp": int(now) + 3600,
            "aud": "https://appleid.apple.com",
            "sub": cfg.apple_client_id,
        },
    )


# ------------------------------------------------------------- state store


class StateStore:
    """Pending sign-ins by `state`, with a TTL. One process, so a dict."""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def put(self, data: Dict[str, Any]) -> str:
        state = secrets.token_urlsafe(24)
        with self._lock:
            now = time.time()
            for k in [k for k, v in self._items.items() if v["exp"] < now]:
                del self._items[k]
            self._items[state] = dict(data, exp=now + STATE_TTL_S)
        return state

    def pop(self, state: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.pop(state or "", None)
        if item is None or item["exp"] < time.time():
            return None
        return item


# ---------------------------------------------------------------- flows


class OAuthError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


def _check_claims(claims: Dict[str, Any], issuers: Tuple[str, ...], audience: str, nonce: str) -> Tuple[str, Optional[str]]:
    if claims.get("iss") not in issuers:
        raise OAuthError("oauth_failed", "unexpected issuer %r" % claims.get("iss"))
    aud = claims.get("aud")
    if aud != audience and not (isinstance(aud, list) and audience in aud):
        raise OAuthError("oauth_failed", "token audience mismatch")
    try:
        if float(claims.get("exp", 0)) < time.time() - 60:
            raise OAuthError("oauth_failed", "token expired")
    except (TypeError, ValueError):
        raise OAuthError("oauth_failed", "bad exp")
    if nonce and claims.get("nonce") != nonce:
        raise OAuthError("oauth_state", "nonce mismatch")
    email = str(claims.get("email") or "").strip().lower()
    verified = claims.get("email_verified")
    if verified in (False, "false", "0", 0):
        raise OAuthError("oauth_email_missing", "email not verified by the provider")
    if not email:
        raise OAuthError("oauth_email_missing", "the provider returned no email")
    name = claims.get("name")
    return email, (str(name).strip() if name else None)


def exchange_google(cfg: OAuthConfig, code: str, redirect_uri: str, code_verifier: str) -> Dict[str, Any]:
    resp = requests.post(
        GOOGLE_TOKEN,
        data={
            "code": code,
            "client_id": cfg.google_client_id,
            "client_secret": cfg.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        },
        timeout=20,
    )
    if resp.status_code >= 400:
        raise OAuthError("oauth_failed", "Google token endpoint %s: %s" % (resp.status_code, resp.text[:200]))
    return resp.json()


def exchange_apple(cfg: OAuthConfig, code: str, redirect_uri: str) -> Dict[str, Any]:
    resp = requests.post(
        APPLE_TOKEN,
        data={
            "code": code,
            "client_id": cfg.apple_client_id,
            "client_secret": apple_client_secret(cfg),
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    if resp.status_code >= 400:
        raise OAuthError("oauth_failed", "Apple token endpoint %s: %s" % (resp.status_code, resp.text[:200]))
    return resp.json()


# ------------------------------------------------------------- registration


def register_oauth(app: FastAPI, store: auth_mod.AuthStore, config: Optional[OAuthConfig] = None) -> FastAPI:
    cfg = config or OAuthConfig()
    states = StateStore()
    app.state.oauth_config = cfg

    def fail(code: str, detail: str = "") -> RedirectResponse:
        if detail:
            log.warning("oauth %s: %s", code, detail)
        return RedirectResponse(url="/signin?error=%s" % code, status_code=303)

    def sign_in(request: Request, email: str, name: Optional[str], next_url: str) -> RedirectResponse:
        user = store.user_by_email(email)
        if user is None:
            # No password can ever match this; the account is provider-only
            # until the owner adds a password-reset flow.
            user = store.create_user(email, secrets.token_urlsafe(32), name)
        token = store.create_session(user["id"])
        response = RedirectResponse(url=next_url, status_code=303)
        auth_mod.set_session_cookie(response, request, token)
        return response

    @app.get("/api/auth/providers")
    def providers() -> JSONResponse:
        return JSONResponse({"google": cfg.google, "apple": cfg.apple})

    @app.get("/api/auth/oauth/{provider}/start")
    def start(request: Request, provider: str, next: str = "/app") -> Any:
        if provider not in PROVIDERS:
            return fail("oauth_failed", "unknown provider %r" % provider)
        if not cfg.enabled(provider):
            return fail("%s_unavailable" % provider)
        next_url = auth_mod._safe_next(next)
        nonce = secrets.token_urlsafe(16)
        redirect_uri = cfg.redirect_uri(request, provider)
        if provider == "google":
            verifier = secrets.token_urlsafe(48)
            challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
            state = states.put({"provider": provider, "nonce": nonce, "next": next_url, "verifier": verifier, "redirect_uri": redirect_uri})
            params = {
                "client_id": cfg.google_client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "nonce": nonce,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "prompt": "select_account",
            }
            return RedirectResponse(url=GOOGLE_AUTH + "?" + urlencode(params), status_code=302)
        state = states.put({"provider": provider, "nonce": nonce, "next": next_url, "redirect_uri": redirect_uri})
        params = {
            "client_id": cfg.apple_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "response_mode": "form_post",
            "scope": "name email",
            "state": state,
            "nonce": nonce,
        }
        return RedirectResponse(url=APPLE_AUTH + "?" + urlencode(params), status_code=302)

    @app.get("/api/auth/oauth/google/callback")
    def google_callback(request: Request, code: str = "", state: str = "", error: str = "") -> Any:
        if error:
            return fail("oauth_denied", "google: %s" % error)
        pending = states.pop(state)
        if pending is None or pending.get("provider") != "google":
            return fail("oauth_state", "unknown or expired state")
        if not code:
            return fail("oauth_failed", "no code")
        try:
            tokens = exchange_google(cfg, code, pending["redirect_uri"], pending["verifier"])
            claims = decode_jwt_payload(str(tokens.get("id_token") or ""))
            email, name = _check_claims(
                claims, ("accounts.google.com", "https://accounts.google.com"), cfg.google_client_id, pending["nonce"]
            )
        except OAuthError as exc:
            return fail(exc.code, str(exc))
        except Exception as exc:  # noqa: BLE001 - network or parse failure
            return fail("oauth_failed", "google: %s" % exc)
        return sign_in(request, email, name, pending["next"])

    @app.post("/api/auth/oauth/apple/callback")
    async def apple_callback(request: Request) -> Any:
        form = await request.form()
        if form.get("error"):
            return fail("oauth_denied", "apple: %s" % form.get("error"))
        pending = states.pop(str(form.get("state") or ""))
        if pending is None or pending.get("provider") != "apple":
            return fail("oauth_state", "unknown or expired state")
        code = str(form.get("code") or "")
        if not code:
            return fail("oauth_failed", "no code")
        name: Optional[str] = None
        raw_user = form.get("user")
        if raw_user:
            try:
                u = json.loads(str(raw_user))
                n = u.get("name") or {}
                name = " ".join(x for x in (n.get("firstName"), n.get("lastName")) if x) or None
            except (ValueError, AttributeError):
                name = None
        try:
            tokens = exchange_apple(cfg, code, pending["redirect_uri"])
            claims = decode_jwt_payload(str(tokens.get("id_token") or ""))
            email, _ = _check_claims(claims, ("https://appleid.apple.com",), cfg.apple_client_id, pending["nonce"])
        except OAuthError as exc:
            return fail(exc.code, str(exc))
        except Exception as exc:  # noqa: BLE001
            return fail("oauth_failed", "apple: %s" % exc)
        return sign_in(request, email, name, pending["next"])

    return app
