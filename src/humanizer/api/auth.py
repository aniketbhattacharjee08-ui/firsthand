"""Accounts and sign-in gating for the web app.

The owner's first instruction (September): "you shouldn't be able to just
go to humanizer right away there should be like a sign-in / create account
page and then you should have to move into this humanizer page". The
October revision: "people get the 1000 words for free, don't have to sign
in, make it public and then block IPs; after a couple tries charge people".

So the site has three pages, and the gate is a switch:

    GET /            web/home.html        public landing page
    GET /signin      web/auth.html        one page; it reads the path for mode
    GET /signup      web/auth.html
    GET /app         web/index.html       the humanizer. Public by default
                                          (HUMANIZER_PUBLIC_APP=1); with the
                                          switch off, signed-in users only,
                                          otherwise 303 to /signin?next=/app
    GET /index.html  same rule as /app

and a JSON API under /api/auth/ (signup, signin, signout, me). With the
switch off, every product route (/api/humanize*, /api/analyze, /api/detect,
/api/plan, /api/features, /api/models*) answers 401
`{"error": "sign_in_required"}` without a valid session cookie. With it on
(the default) the product routes are open here and the paywall in
`humanizer.billing.gate` is what limits a visitor: one free allowance per
client address, then 402 and the sign-up page. /api/health, /api/auth/*
and /api/references are always public.

Storage is one SQLite file (`data/auth.sqlite`, or `HUMANIZER_AUTH_DB`) with
`users` and `sessions`. Passwords are hashed with `hashlib.scrypt` (n=2**14,
r=8, p=1, 16-byte random salt) and stored as `scrypt$<salt_hex>$<hash_hex>`;
on a Python whose OpenSSL lacks scrypt they are PBKDF2-HMAC-SHA256 (600k
iterations) tagged `pbkdf2_sha256$`, and either kind verifies anywhere.
The session cookie `rh_session` is HttpOnly, SameSite=Lax, Secure over https,
30 days, and its value is a random 32-byte urlsafe token that is only ever
stored server-side, so a leaked database does not leak passwords and a
leaked cookie can be revoked by deleting one row.

Nothing here has a dependency beyond the standard library and FastAPI, which
the API extra already requires. The gate is a middleware rather than a
dependency on each route because the product routes are closures inside
`create_app` and this module is appended, not woven in.

This is separate from `humanizer.billing`, the paid tier's magic-link login
and credit paywall. The two do not share a cookie, a table or a route path;
`humanizer.billing.app.create_app` builds the server with `auth=False` and
keeps its own gate. Merging them is a later decision.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

SESSION_COOKIE = "rh_session"
#: `HUMANIZER_PUBLIC_APP`: unset or anything but 0/false/no/off serves /app
#: and the product API to visitors who have not signed in.
PUBLIC_APP_ENV = "HUMANIZER_PUBLIC_APP"
SESSION_TTL_S = 30 * 24 * 3600
MIN_PASSWORD_LENGTH = 8

#: scrypt parameters, fixed so a stored hash can always be re-verified. The
#: cost is ~16MB of memory and tens of milliseconds per check, which is the
#: point: a stolen database is expensive to brute-force.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_BYTES = 16
SCRYPT_MAXMEM = 64 * 1024 * 1024

#: `hashlib.scrypt` exists only when Python was linked against an OpenSSL
#: with scrypt (LibreSSL builds and some macOS pythons lack it). Where it is
#: missing, new hashes use PBKDF2-HMAC-SHA256 at the OWASP 2023 iteration
#: count and are tagged `pbkdf2_sha256$` so a database written on one build
#: still verifies on the other: `verify_password` dispatches on the tag.
SCRYPT_AVAILABLE = hasattr(hashlib, "scrypt")
PBKDF2_ITERATIONS = 600_000

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: Path prefixes that need a session. Prefix match, so `/api/humanize/llm`,
#: `/api/humanize/guided/stream` and `/api/models/release` are all covered.
GATED_PREFIXES = (
    "/api/humanize",
    "/api/analyze",
    "/api/detect",
    "/api/plan",
    "/api/features",
    "/api/models",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user'
);
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
"""


# ------------------------------------------------------------------ passwords


def _derive(scheme: str, password: str, salt: bytes) -> Optional[bytes]:
    """The raw digest for one scheme, or None if this build cannot compute it."""
    data = password.encode("utf-8")
    if scheme == "scrypt":
        if not SCRYPT_AVAILABLE:
            return None
        return hashlib.scrypt(
            data, salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM
        )
    if scheme == "pbkdf2_sha256":
        return hashlib.pbkdf2_hmac("sha256", data, salt, PBKDF2_ITERATIONS, dklen=64)
    return None


def hash_password(password: str) -> str:
    """`scrypt$<salt_hex>$<hash_hex>` with a fresh random salt.

    `pbkdf2_sha256$...` instead when this interpreter has no scrypt; see
    `SCRYPT_AVAILABLE`.
    """
    scheme = "scrypt" if SCRYPT_AVAILABLE else "pbkdf2_sha256"
    salt = secrets.token_bytes(SCRYPT_SALT_BYTES)
    digest = _derive(scheme, password, salt)
    assert digest is not None
    return "%s$%s$%s" % (scheme, salt.hex(), digest.hex())


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of `password` against a `hash_password` string."""
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    digest = _derive(scheme, password, salt)
    if digest is None:
        return False
    return hmac.compare_digest(digest, expected)


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email)) and len(email) <= 254


# -------------------------------------------------------------------- storage


def default_auth_db_path() -> Path:
    """`HUMANIZER_AUTH_DB`, else `data/auth.sqlite` next to the other data.

    Mirrors `server.default_reference_dir`: the working directory first, since
    the CLI runs from the project root, then the repo root inferred from this
    file.
    """
    env = os.environ.get("HUMANIZER_AUTH_DB")
    if env:
        return Path(env)
    for base in (Path.cwd() / "data", _REPO_ROOT / "data"):
        if base.is_dir():
            return base / "auth.sqlite"
    return Path.cwd() / "data" / "auth.sqlite"


class AuthStore:
    """Users and sessions in one SQLite file, one connection, one lock.

    Adequate for the single-process server this repo runs (the LLM job store
    is an in-memory dict for the same reason). Every method is safe to call
    from any thread.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            # Databases created before roles existed get the column added.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(users)")}
            if "role" not in cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- users

    def create_user(
        self, email: str, password: str, name: Optional[str] = None, role: str = "user"
    ) -> Dict[str, Any]:
        """Insert a user. Raises `ValueError("email_exists")` on a duplicate."""
        email = normalise_email(email)
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO users (email, name, password_hash, created_at, role) VALUES (?, ?, ?, ?, ?)",
                    (email, name or None, hash_password(password), now, role),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError("email_exists")
            return {"id": cur.lastrowid, "email": email, "name": name or None, "created_at": now, "role": role}

    def set_master(self, email: str, password: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Create or update the owner's account: role `master`, this password.

        Idempotent, so it can run at every deploy. An existing account with
        that email keeps its id and sessions but gets the new password, the
        name if given, and the master role.
        """
        email = normalise_email(email)
        with self._lock:
            row = self._conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if row is None:
                cur = self._conn.execute(
                    "INSERT INTO users (email, name, password_hash, created_at, role) VALUES (?, ?, ?, ?, 'master')",
                    (email, name or None, hash_password(password), time.time()),
                )
                user_id = cur.lastrowid
            else:
                user_id = row["id"]
                if name:
                    self._conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
                self._conn.execute(
                    "UPDATE users SET password_hash = ?, role = 'master' WHERE id = ?",
                    (hash_password(password), user_id),
                )
            self._conn.commit()
        user = self.user_by_email(email)
        assert user is not None
        return user

    def user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, email, name, password_hash, created_at, role FROM users WHERE email = ?",
                (normalise_email(email),),
            ).fetchone()
        return dict(row) if row else None

    def authenticate(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """The user row on a correct email and password, else None.

        A missing email and a wrong password take the same path, including
        an scrypt run against a dummy hash, so the response time does not
        say which one it was.
        """
        user = self.user_by_email(email)
        stored = user["password_hash"] if user else _DUMMY_HASH
        ok = verify_password(password, stored)
        if user is None or not ok:
            return None
        return user

    # -- sessions

    def create_session(self, user_id: int, ttl_s: float = SESSION_TTL_S) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, now, now + ttl_s),
            )
            self._conn.commit()
        return token

    def user_for_session(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        """The user behind a live session token, else None. Expired rows are deleted."""
        if not token:
            return None
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT s.expires_at, u.id, u.email, u.name, u.created_at, u.role "
                "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                self._conn.commit()
                return None
        return {"id": row["id"], "email": row["email"], "name": row["name"], "created_at": row["created_at"], "role": row["role"] or "user"}

    def delete_session(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            self._conn.commit()

    def purge_expired(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
            self._conn.commit()
            return cur.rowcount


#: Verified against when the email does not exist, so both failure branches
#: cost one scrypt run. Computed once at import; the password is irrelevant.
_DUMMY_HASH = hash_password("not-a-real-password")


# ------------------------------------------------------------------- helpers


def user_payload(user: Dict[str, Any]) -> Dict[str, Any]:
    """The `user` object on the wire: email, name and role, never the id or hash."""
    return {"email": user["email"], "name": user.get("name"), "role": user.get("role") or "user"}


def _is_https(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    return request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"


def set_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_S,
        httponly=True,
        samesite="lax",
        secure=_is_https(request),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def current_user(request: Request) -> Optional[Dict[str, Any]]:
    store: Optional[AuthStore] = getattr(request.app.state, "auth_store", None)
    if store is None:
        return None
    return store.user_for_session(request.cookies.get(SESSION_COOKIE))


def _err(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def _safe_next(value: str) -> str:
    """Only same-site relative paths survive; anything else falls back to /app."""
    if value and value.startswith("/") and not value.startswith("//"):
        return value
    return "/app"


def _page(path: Path) -> FileResponse:
    # The HTML pages are edited often and the URL does not end in .html, so
    # the `_no_stale_assets` middleware in server.py would not catch them.
    return FileResponse(str(path), media_type="text/html", headers={"Cache-Control": "no-cache, must-revalidate"})


# ---------------------------------------------------------------- request DTOs


class SignupBody(BaseModel):
    email: str = Field(default="", max_length=254)
    password: str = Field(default="", max_length=1024)
    name: Optional[str] = Field(default=None, max_length=200)


class SigninBody(BaseModel):
    email: str = Field(default="", max_length=254)
    password: str = Field(default="", max_length=1024)


# -------------------------------------------------------------------- install


def public_app_from_env() -> bool:
    """`HUMANIZER_PUBLIC_APP` as a bool; unset is on."""
    return os.environ.get(PUBLIC_APP_ENV, "1").strip().lower() not in ("0", "false", "no", "off", "")


def register_auth(
    app: FastAPI, store: AuthStore, web_root: Optional[Path], public_app: Optional[bool] = None
) -> FastAPI:
    """Add the pages, the /api/auth routes and the gate to a built app.

    `public_app=None` reads `HUMANIZER_PUBLIC_APP` (default on): the app page
    and the product API are served to visitors who have not signed in, and
    the paywall is what limits them. False restores the September gate.
    Idempotent per app instance. The caller is responsible for moving any
    Mount back to the end of the router afterwards (server.py does).
    """
    if getattr(app.state, "auth_installed", False):
        return app
    public = public_app_from_env() if public_app is None else bool(public_app)
    app.state.auth_store = store
    app.state.auth_enabled = True
    app.state.auth_installed = True
    app.state.public_app = public

    web = Path(web_root) if web_root else None

    def page_path(name: str) -> Optional[Path]:
        if web is None:
            return None
        candidate = web / name
        return candidate if candidate.is_file() else None

    # -- pages

    if web is not None:

        @app.get("/", include_in_schema=False)
        def home_page(request: Request) -> Any:
            home = page_path("home.html")
            if home is not None:
                return _page(home)
            # No landing page on disk yet: send visitors through the gate.
            return RedirectResponse("/app", status_code=303)

        @app.get("/pricing", include_in_schema=False)
        def pricing_page(request: Request) -> Any:
            # Public, like the landing: the plans and prices need no account.
            page = page_path("pricing.html")
            if page is None:
                return _err(404, "no_pricing_page", "web/pricing.html is missing.")
            return _page(page)

        @app.get("/signin", include_in_schema=False)
        @app.get("/signup", include_in_schema=False)
        def auth_page(request: Request) -> Any:
            page = page_path("auth.html")
            if page is None:
                return _err(404, "no_auth_page", "web/auth.html is missing; the API at /api/auth/* is up.")
            return _page(page)

        @app.get("/app", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def app_page(request: Request) -> Any:
            if not public and current_user(request) is None:
                return RedirectResponse("/signin?next=" + quote("/app", safe="/"), status_code=303)
            index = page_path("index.html")
            if index is None:
                return _err(404, "no_app_page", "web/index.html is missing.")
            return _page(index)

    # -- JSON API

    @app.post("/api/auth/signup")
    def signup(request: Request, body: SignupBody) -> JSONResponse:
        """Create an account and sign it in. 201, or 409 / 422 with `error`."""
        email = normalise_email(body.email)
        if not valid_email(email):
            return _err(422, "invalid_email", "Enter a valid email address.")
        if len(body.password) < MIN_PASSWORD_LENGTH:
            return _err(422, "invalid_password", "Use at least %d characters." % MIN_PASSWORD_LENGTH)
        name = (body.name or "").strip() or None
        try:
            user = store.create_user(email, body.password, name)
        except ValueError:
            return _err(409, "email_exists", "An account with that email already exists. Sign in instead.")
        token = store.create_session(user["id"])
        response = JSONResponse(status_code=201, content={"ok": True, "user": user_payload(user)})
        set_session_cookie(response, request, token)
        return response

    @app.post("/api/auth/signin")
    def signin(request: Request, body: SigninBody) -> JSONResponse:
        """200 with the user, or 401 `invalid_credentials` whether or not the email exists."""
        user = store.authenticate(body.email, body.password)
        if user is None:
            return _err(401, "invalid_credentials", "That email and password do not match.")
        token = store.create_session(user["id"])
        response = JSONResponse(status_code=200, content={"ok": True, "user": user_payload(user)})
        set_session_cookie(response, request, token)
        return response

    @app.post("/api/auth/signout", status_code=204)
    def signout(request: Request) -> Response:
        """Delete the session and clear the cookie. Always 204."""
        store.delete_session(request.cookies.get(SESSION_COOKIE))
        response = Response(status_code=204)
        clear_session_cookie(response)
        return response

    @app.get("/api/auth/me")
    def me(request: Request) -> JSONResponse:
        user = current_user(request)
        if user is None:
            return JSONResponse({"signed_in": False})
        return JSONResponse({"signed_in": True, "user": user_payload(user)})

    # -- the gate

    @app.middleware("http")
    async def _require_sign_in(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        if not public and path.startswith(GATED_PREFIXES) and request.method != "OPTIONS":
            if store.user_for_session(request.cookies.get(SESSION_COOKIE)) is None:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "sign_in_required",
                        "detail": "Sign in at /signin, or POST /api/auth/signin, then retry.",
                        "signin_url": "/signin",
                    },
                )
        return await call_next(request)

    return app
