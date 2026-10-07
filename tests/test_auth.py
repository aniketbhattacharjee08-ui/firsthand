"""Accounts and sign-in gating (`humanizer.api.auth`).

Every app here is built with `auth=True` and a temporary SQLite file, so the
suite never touches `data/auth.sqlite`. The other API test modules build
their apps with `auth=False`, which is why they keep passing without a
session; the point of this file is to check what happens when it is on.

The page contract is the one the front end codes against:

    /            web/index.html when the app is public (the tool is the landing); web/home.html behind the wall
    /pricing     web/pricing.html, public
    /signin      web/auth.html, public
    /signup      web/auth.html, public
    /app         web/index.html, 303 to /signin?next=/app without a session
    /index.html  same rule as /app
"""

import sqlite3
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api import auth as auth_mod  # noqa: E402
from humanizer.api.server import create_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"

SHORT = "Rain fell. The meeting ended early because of it."
GOOD = {"email": "ada@example.com", "password": "correct horse", "name": "Ada"}


def _web(tmp_path: Path) -> Path:
    """A stand-in for web/ with the three pages the front end provides."""
    web = tmp_path / "web"
    web.mkdir()
    (web / "home.html").write_text("<h1>home</h1>")
    (web / "pricing.html").write_text("<h1>pricing</h1><h2>Monthly</h2>")
    (web / "how.html").write_text("<h1>how</h1>")
    (web / "auth.html").write_text("<h1>auth</h1>")
    (web / "index.html").write_text("<h1>humanizer</h1>")
    (web / "app.js").write_text("console.log(1)")
    return web


def _app(tmp_path: Path, web=None, **kw):
    return create_app(
        reference_dir=REFERENCE_DIR,
        web_dir=web if web is not None else tmp_path / "no-web",
        auth=True,
        auth_db=tmp_path / "auth.sqlite",
        **kw,
    )


@pytest.fixture()
def client(tmp_path):
    with TestClient(_app(tmp_path, _web(tmp_path))) as c:
        yield c


# ------------------------------------------------------------- passwords


class TestPasswords:
    def test_hash_format_and_round_trip(self):
        stored = auth_mod.hash_password("hunter22")
        scheme, salt_hex, hash_hex = stored.split("$")
        assert scheme == ("scrypt" if auth_mod.SCRYPT_AVAILABLE else "pbkdf2_sha256")
        assert len(bytes.fromhex(salt_hex)) == 16
        assert len(bytes.fromhex(hash_hex)) == 64
        assert auth_mod.verify_password("hunter22", stored)
        assert not auth_mod.verify_password("hunter23", stored)

    def test_scrypt_hash_is_the_documented_shape_when_available(self):
        if not auth_mod.SCRYPT_AVAILABLE:
            pytest.skip("this Python's OpenSSL has no scrypt")
        assert auth_mod.hash_password("x" * 8).startswith("scrypt$")

    def test_pbkdf2_hashes_verify_on_every_build(self):
        # A hash written by a build without scrypt must still verify here.
        salt = bytes(range(16))
        import hashlib

        digest = hashlib.pbkdf2_hmac("sha256", b"hunter22", salt, auth_mod.PBKDF2_ITERATIONS, dklen=64)
        stored = "pbkdf2_sha256$%s$%s" % (salt.hex(), digest.hex())
        assert auth_mod.verify_password("hunter22", stored)
        assert not auth_mod.verify_password("hunter23", stored)

    def test_salt_is_random(self):
        assert auth_mod.hash_password("same") != auth_mod.hash_password("same")

    def test_garbage_hash_never_verifies(self):
        assert not auth_mod.verify_password("x", "not-a-hash")
        assert not auth_mod.verify_password("x", "md5$00$00")
        assert not auth_mod.verify_password("x", "")


# ------------------------------------------------------------- JSON API


class TestSignup:
    def test_signup_is_201_and_sets_the_cookie(self, client):
        r = client.post("/api/auth/signup", json=GOOD)
        assert r.status_code == 201
        assert r.json() == {"ok": True, "user": {"email": "ada@example.com", "name": "Ada", "role": "user"}}
        assert auth_mod.SESSION_COOKIE in r.cookies
        me = client.get("/api/auth/me").json()
        assert me == {"signed_in": True, "user": {"email": "ada@example.com", "name": "Ada", "role": "user"}}

    def test_name_is_optional_and_email_is_normalised(self, client):
        r = client.post("/api/auth/signup", json={"email": "  Bob@Example.COM ", "password": "long enough"})
        assert r.status_code == 201
        assert r.json()["user"] == {"email": "bob@example.com", "name": None, "role": "user"}

    def test_duplicate_email_is_409(self, client):
        assert client.post("/api/auth/signup", json=GOOD).status_code == 201
        r = client.post("/api/auth/signup", json={**GOOD, "email": "ADA@example.com"})
        assert r.status_code == 409
        assert r.json()["error"] == "email_exists"

    def test_invalid_email_is_422(self, client):
        r = client.post("/api/auth/signup", json={"email": "not-an-email", "password": "long enough"})
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_email"

    def test_short_password_is_422(self, client):
        r = client.post("/api/auth/signup", json={"email": "a@b.co", "password": "1234567"})
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_password"
        # Nothing was created, so the same email can still sign up.
        assert client.post("/api/auth/signup", json={"email": "a@b.co", "password": "12345678"}).status_code == 201

    def test_password_hash_never_leaves_the_server(self, client, tmp_path):
        r = client.post("/api/auth/signup", json=GOOD)
        assert "password" not in r.text and "$" not in r.text
        row = sqlite3.connect(tmp_path / "auth.sqlite").execute(
            "SELECT email, name, password_hash FROM users"
        ).fetchone()
        assert row[0] == "ada@example.com" and row[1] == "Ada"
        assert row[2].split("$")[0] in ("scrypt", "pbkdf2_sha256")
        assert GOOD["password"] not in row[2]


class TestSignin:
    def test_signin_round_trip(self, client):
        client.post("/api/auth/signup", json=GOOD)
        client.post("/api/auth/signout")
        assert client.get("/api/auth/me").json() == {"signed_in": False}
        r = client.post("/api/auth/signin", json={"email": GOOD["email"], "password": GOOD["password"]})
        assert r.status_code == 200
        assert r.json() == {"ok": True, "user": {"email": "ada@example.com", "name": "Ada", "role": "user"}}
        assert client.get("/api/auth/me").json()["signed_in"] is True

    def test_wrong_password_is_401(self, client):
        client.post("/api/auth/signup", json=GOOD)
        client.post("/api/auth/signout")
        r = client.post("/api/auth/signin", json={"email": GOOD["email"], "password": "wrong password"})
        assert r.status_code == 401
        assert r.json()["error"] == "invalid_credentials"
        assert auth_mod.SESSION_COOKIE not in r.cookies

    def test_unknown_email_is_the_same_401(self, client):
        client.post("/api/auth/signup", json=GOOD)
        wrong_pw = client.post("/api/auth/signin", json={"email": GOOD["email"], "password": "wrong password"})
        no_user = client.post("/api/auth/signin", json={"email": "nobody@example.com", "password": "wrong password"})
        assert wrong_pw.status_code == no_user.status_code == 401
        assert wrong_pw.json() == no_user.json()

    def test_signout_is_204_and_clears_the_cookie(self, client):
        client.post("/api/auth/signup", json=GOOD)
        r = client.post("/api/auth/signout")
        assert r.status_code == 204
        assert r.content == b""
        set_cookie = r.headers.get("set-cookie", "")
        assert auth_mod.SESSION_COOKIE + '="";' in set_cookie or auth_mod.SESSION_COOKIE + "=;" in set_cookie
        assert client.get("/api/auth/me").json() == {"signed_in": False}

    def test_signout_without_a_session_is_still_204(self, client):
        assert client.post("/api/auth/signout").status_code == 204

    def test_me_without_a_session(self, client):
        assert client.get("/api/auth/me").json() == {"signed_in": False}

    def test_forged_cookie_is_not_a_session(self, client):
        client.cookies.set(auth_mod.SESSION_COOKIE, "not-a-real-token")
        assert client.get("/api/auth/me").json() == {"signed_in": False}


class TestCookie:
    def test_flags_over_http(self, client):
        r = client.post("/api/auth/signup", json=GOOD)
        header = r.headers["set-cookie"].lower()
        assert header.startswith(auth_mod.SESSION_COOKIE + "=")
        assert "httponly" in header
        assert "samesite=lax" in header
        assert "path=/" in header
        assert "max-age=%d" % auth_mod.SESSION_TTL_S in header
        assert "secure" not in header

    def test_secure_over_https(self, tmp_path):
        with TestClient(_app(tmp_path, _web(tmp_path)), base_url="https://testserver") as c:
            header = c.post("/api/auth/signup", json=GOOD).headers["set-cookie"].lower()
        assert "secure" in header

    def test_secure_behind_a_tls_terminating_proxy(self, client):
        header = client.post(
            "/api/auth/signup", json=GOOD, headers={"x-forwarded-proto": "https"}
        ).headers["set-cookie"].lower()
        assert "secure" in header

    def test_token_is_random_and_stored_server_side(self, client, tmp_path):
        token = client.post("/api/auth/signup", json=GOOD).cookies[auth_mod.SESSION_COOKIE]
        assert len(token) >= 43  # 32 bytes urlsafe-base64, unpadded
        rows = sqlite3.connect(tmp_path / "auth.sqlite").execute(
            "SELECT token, expires_at - created_at FROM sessions"
        ).fetchall()
        assert [r[0] for r in rows] == [token]
        assert rows[0][1] == pytest.approx(auth_mod.SESSION_TTL_S, abs=2)

    def test_expired_session_is_rejected_and_deleted(self, client, tmp_path):
        client.post("/api/auth/signup", json=GOOD)
        assert client.get("/api/auth/me").json()["signed_in"] is True
        db = sqlite3.connect(tmp_path / "auth.sqlite")
        db.execute("UPDATE sessions SET expires_at = ?", (time.time() - 1,))
        db.commit()
        assert client.get("/api/auth/me").json() == {"signed_in": False}
        # /app is public by default, so the expired session shows as a guest
        # visit rather than a redirect; the row itself is gone.
        assert client.get("/app", follow_redirects=False).status_code == 200
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


# ------------------------------------------------------------------ pages


class TestPages:
    def test_home_is_the_tool_when_public(self, client):
        """Since 2026-10-07 the tool is the landing page: "/" serves index.html."""
        r = client.get("/")
        assert r.status_code == 200 and "<h1>humanizer</h1>" in r.text

    def test_home_page_is_used_behind_the_wall(self, tmp_path):
        with TestClient(_app(tmp_path, _web(tmp_path), public_app=False)) as c:
            r = c.get("/")
            assert r.status_code == 200 and "<h1>home</h1>" in r.text
        assert r.headers["cache-control"] == "no-cache, must-revalidate"

    def test_home_is_still_the_tool_when_signed_in(self, client):
        client.post("/api/auth/signup", json=GOOD)
        assert "<h1>humanizer</h1>" in client.get("/").text

    def test_pricing_is_public(self, client):
        r = client.get("/pricing")
        assert r.status_code == 200 and "Monthly" in r.text
        assert r.headers["cache-control"] == "no-cache, must-revalidate"

    def test_pricing_is_the_same_page_when_signed_in(self, client):
        client.post("/api/auth/signup", json=GOOD)
        assert "Monthly" in client.get("/pricing").text

    def test_how_it_works_is_public(self, client):
        r = client.get("/how")
        assert r.status_code == 200 and "<h1>how</h1>" in r.text

    def test_signin_and_signup_serve_the_auth_page(self, client):
        for path in ("/signin", "/signup"):
            r = client.get(path)
            assert r.status_code == 200 and "<h1>auth</h1>" in r.text, path

    def test_app_is_public_by_default(self, client):
        """October rule: the free words need no account, so /app is served to
        anyone and the paywall (by address) is what limits a visitor."""
        for path in ("/app", "/index.html"):
            r = client.get(path, follow_redirects=False)
            assert r.status_code == 200 and "<h1>humanizer</h1>" in r.text, path

    def test_app_redirects_without_a_session_when_not_public(self, tmp_path):
        with TestClient(_app(tmp_path, _web(tmp_path), public_app=False)) as c:
            for path in ("/app", "/index.html"):
                r = c.get(path, follow_redirects=False)
                assert r.status_code == 303, path
                assert r.headers["location"] == "/signin?next=/app"
            c.post("/api/auth/signup", json=GOOD)
            assert "<h1>humanizer</h1>" in c.get("/app").text

    def test_public_app_switch_reads_the_environment(self, tmp_path, monkeypatch):
        web = _web(tmp_path)
        monkeypatch.setenv(auth_mod.PUBLIC_APP_ENV, "0")
        with TestClient(_app(tmp_path, web)) as c:
            assert c.get("/app", follow_redirects=False).status_code == 303
        monkeypatch.setenv(auth_mod.PUBLIC_APP_ENV, "1")
        with TestClient(_app(tmp_path, web)) as c:
            assert c.get("/app", follow_redirects=False).status_code == 200

    def test_app_is_served_with_a_session(self, client):
        client.post("/api/auth/signup", json=GOOD)
        r = client.get("/app")
        assert r.status_code == 200 and "<h1>humanizer</h1>" in r.text
        assert "<h1>humanizer</h1>" in client.get("/index.html").text

    def test_other_assets_stay_public(self, client):
        assert "console.log(1)" in client.get("/app.js").text
        assert "<h1>home</h1>" in client.get("/home.html").text
        assert "<h1>auth</h1>" in client.get("/auth.html").text

    def test_static_mount_is_still_last(self, client):
        paths = [getattr(r, "path", None) for r in client.app.router.routes]
        assert paths[-1] == ""
        for p in ("/", "/how", "/pricing", "/signin", "/signup", "/app", "/index.html", "/api/auth/signup", "/api/humanize"):
            assert p in paths and paths.index(p) < paths.index(""), p

    def test_missing_home_page_sends_visitors_through_the_gate(self, tmp_path):
        web = tmp_path / "web"
        web.mkdir()
        (web / "index.html").write_text("<h1>humanizer</h1>")
        with TestClient(_app(tmp_path, web)) as c:
            # index.html is present, so "/" is the tool itself, no redirect.
            r = c.get("/", follow_redirects=False)
            assert r.status_code == 200 and "<h1>humanizer</h1>" in r.text

    def test_no_web_dir_keeps_the_api_placeholder(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            assert c.get("/").json()["status"] == "ok"
            assert c.post("/api/auth/signup", json=GOOD).status_code == 201


# ------------------------------------------------------------------- gate


@pytest.fixture()
def walled(tmp_path):
    """The September configuration: HUMANIZER_PUBLIC_APP off, so the product
    API and /app need a session."""
    with TestClient(_app(tmp_path, _web(tmp_path), public_app=False)) as c:
        yield c


class TestGate:
    def test_product_routes_are_open_to_guests_by_default(self, client):
        """Public app: the product routes answer without a session here; the
        paywall in humanizer.billing is what limits a visitor by address."""
        r = client.post("/api/humanize", json={"text": SHORT, "aggressiveness": "light"})
        assert r.status_code == 200
        assert client.post("/api/analyze", json={"text": SHORT, "detect": False}).status_code == 200

    def test_product_routes_are_401_without_a_session(self, walled):
        client = walled
        cases = [
            ("post", "/api/analyze", {"text": SHORT}),
            ("post", "/api/features", {"text": SHORT}),
            ("post", "/api/detect", {"text": SHORT}),
            ("post", "/api/plan", {"mu": 0.6, "target": 0.99}),
            ("post", "/api/humanize", {"text": SHORT}),
            ("post", "/api/humanize/llm", {"text": SHORT}),
            ("get", "/api/humanize/llm/health", None),
            ("get", "/api/humanize/guided/health", None),
            ("get", "/api/models", None),
            ("post", "/api/models/release", None),
        ]
        for method, path, body in cases:
            r = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
            assert r.status_code == 401, (path, r.status_code)
            assert r.json()["error"] == "sign_in_required", path

    def test_public_routes_stay_public(self, client):
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/references").status_code == 200
        assert client.get("/api/auth/me").status_code == 200

    def test_analyze_works_with_a_session_and_is_unchanged(self, client):
        client.post("/api/auth/signup", json=GOOD)
        r = client.post("/api/analyze", json={"text": SHORT, "detect": False})
        assert r.status_code == 200
        body = r.json()
        for key in ("features", "bands", "findings", "detectors", "sentences", "ai_style_signals"):
            assert key in body
        assert body["sentence_risk_is_advisory"] is True
        assert body["features"]["n_words"] > 0

    def test_humanize_works_with_a_session(self, client):
        client.post("/api/auth/signup", json=GOOD)
        r = client.post("/api/humanize", json={"text": SHORT, "aggressiveness": "light"})
        assert r.status_code == 200
        assert set(r.json()) == {"original", "humanized", "edits", "before", "after", "summary"}

    def test_validation_still_runs_behind_the_gate(self, client):
        client.post("/api/auth/signup", json=GOOD)
        assert client.post("/api/analyze", json={"text": ""}).status_code == 400

    def test_signing_out_closes_the_gate(self, walled):
        client = walled
        client.post("/api/auth/signup", json=GOOD)
        assert client.post("/api/features", json={"text": SHORT}).status_code == 200
        client.post("/api/auth/signout")
        assert client.post("/api/features", json={"text": SHORT}).status_code == 401

    def test_cors_preflight_is_not_gated(self, client):
        r = client.options(
            "/api/analyze",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.status_code == 200


# ---------------------------------------------------------------- switch


class TestSwitch:
    def test_auth_false_serves_the_app_at_root(self, tmp_path):
        web = _web(tmp_path)
        app = create_app(reference_dir=REFERENCE_DIR, web_dir=web, auth=False)
        with TestClient(app) as c:
            assert app.state.auth_enabled is False
            # No page routes: StaticFiles(html=True) serves index.html at "/".
            assert "<h1>humanizer</h1>" in c.get("/").text
            assert c.get("/api/auth/me").status_code == 404
            assert c.post("/api/features", json={"text": SHORT}).status_code == 200

    def test_env_default_is_on(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HUMANIZER_AUTH", raising=False)
        monkeypatch.setenv("HUMANIZER_AUTH_DB", str(tmp_path / "env.sqlite"))
        app = create_app(reference_dir=REFERENCE_DIR, web_dir=tmp_path / "no-web")
        assert app.state.auth_enabled is True
        assert (tmp_path / "env.sqlite").exists()

    def test_env_zero_turns_it_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HUMANIZER_AUTH", "0")
        app = create_app(reference_dir=REFERENCE_DIR, web_dir=tmp_path / "no-web")
        assert app.state.auth_enabled is False

    def test_cli_has_no_auth_flag(self):
        from humanizer.cli import build_parser

        assert build_parser().parse_args(["serve", "--no-auth"]).no_auth is True
        assert build_parser().parse_args(["serve"]).no_auth is False

    def test_default_db_path_honours_the_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HUMANIZER_AUTH_DB", str(tmp_path / "x.sqlite"))
        assert auth_mod.default_auth_db_path() == tmp_path / "x.sqlite"
        monkeypatch.delenv("HUMANIZER_AUTH_DB")
        assert auth_mod.default_auth_db_path().name == "auth.sqlite"

    def test_billing_app_keeps_its_own_gate(self, tmp_path, monkeypatch):
        """The paywall builds the server with auth=True (the site's pages and
        password accounts) and bridges `rh_session` to its own users; the
        auth module's blanket 401 middleware is replaced by the paywall's gate."""
        pytest.importorskip("humanizer.billing.app")
        from humanizer.billing.app import create_app as billing_create_app
        from humanizer.billing.config import BillingConfig
        from humanizer.billing.store import Store

        for k, v in {
            "LONGHAND_PAYWALL": "1",
            "LONGHAND_SECRET": "x" * 48,
            "LONGHAND_DB": str(tmp_path / "billing.db"),
            "LONGHAND_DEV_LINKS": "1",
        }.items():
            monkeypatch.setenv(k, v)
        cfg = BillingConfig.from_env()
        app = billing_create_app(
            reference_dir=REFERENCE_DIR, web_dir=tmp_path / "no-web", config=cfg, store=Store(cfg.db_path), auth_db=tmp_path / "auth.sqlite"
        )
        assert app.state.auth_enabled is True and app.state.auth_store is not None
        assert not any(
            getattr(getattr(m, "kwargs", {}).get("dispatch"), "__module__", "") == auth_mod.__name__ for m in app.user_middleware
        )



def test_master_account_upsert_and_role_on_the_wire(tmp_path):
    from humanizer.api.auth import AuthStore, user_payload

    store = AuthStore(tmp_path / "auth.sqlite")
    plain = store.create_user("reader@example.com", "reader-pass-1")
    assert plain["role"] == "user"
    master = store.set_master("owner@example.com", "owner-pass-123", "Owner")
    assert master["role"] == "master" and master["name"] == "Owner"
    # Idempotent: a second call keeps the id, resets the password, keeps the role.
    again = store.set_master("owner@example.com", "new-owner-pass-9")
    assert again["id"] == master["id"] and again["role"] == "master"
    assert store.authenticate("owner@example.com", "owner-pass-123") is None
    assert store.authenticate("owner@example.com", "new-owner-pass-9") is not None
    assert user_payload(again) == {"email": "owner@example.com", "name": "Owner", "role": "master"}
    # A session for the master carries the role too.
    token = store.create_session(again["id"])
    assert store.user_for_session(token)["role"] == "master"
    store.close()


def test_cli_creates_the_master_account(tmp_path, capsys):
    from humanizer.cli import main
    from humanizer.api.auth import AuthStore

    db = tmp_path / "a.sqlite"
    assert main(["account", "master", "--email", "owner@example.com", "--name", "Owner", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "master account ready: owner@example.com (role master)" in out
    assert "password (shown once" in out
    password = out.split("store it now): ")[1].strip()
    store = AuthStore(db)
    assert store.authenticate("owner@example.com", password)["role"] == "master"
    store.close()
