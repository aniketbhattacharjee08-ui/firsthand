"""Google and Apple sign-in on top of the accounts module.

No network: the token endpoints are replaced by fakes that return an
id_token whose payload is built here. The Apple flow is exercised end to end
only when `cryptography` is installed (the client secret needs ES256);
otherwise the test asserts the graceful "unavailable" path.
"""

import base64
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api import oauth  # noqa: E402
from humanizer.api.server import create_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _jwt(payload):
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()  # noqa: E731
    return "%s.%s.%s" % (b64(b'{"alg":"RS256"}'), b64(json.dumps(payload).encode()), b64(b"sig"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "APPLE_CLIENT_ID", "APPLE_TEAM_ID", "APPLE_KEY_ID", "APPLE_PRIVATE_KEY", "HUMANIZER_PUBLIC_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "gid.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "gsecret")
    monkeypatch.setenv("HUMANIZER_AUTH", "1")
    app = create_app(reference_dir=REPO_ROOT / "data" / "reference", web_dir=None, auth=True, auth_db=tmp_path / "auth.sqlite")
    with TestClient(app) as c:
        yield c


def test_providers_reports_configuration(client):
    r = client.get("/api/auth/providers")
    assert r.status_code == 200
    assert r.json()["google"] is True
    assert r.json()["apple"] is False  # no Apple keys in this test


def test_unconfigured_provider_redirects_with_error(client):
    r = client.get("/api/auth/oauth/apple/start", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/signin?error=apple_unavailable"
    r = client.get("/api/auth/oauth/nope/start", follow_redirects=False)
    assert r.headers["location"] == "/signin?error=oauth_failed"


def test_google_start_builds_a_pkce_request(client):
    r = client.get("/api/auth/oauth/google/start?next=/app", follow_redirects=False)
    assert r.status_code == 302
    url = urlparse(r.headers["location"])
    assert url.netloc == "accounts.google.com"
    q = parse_qs(url.query)
    assert q["client_id"] == ["gid.apps.googleusercontent.com"]
    assert q["code_challenge_method"] == ["S256"] and q["state"] and q["nonce"]
    assert q["redirect_uri"][0].endswith("/api/auth/oauth/google/callback")
    assert "openid" in q["scope"][0]


def test_google_callback_signs_in_and_creates_the_account(client, monkeypatch):
    r = client.get("/api/auth/oauth/google/start?next=/app", follow_redirects=False)
    q = parse_qs(urlparse(r.headers["location"]).query)
    state, nonce = q["state"][0], q["nonce"][0]
    seen = {}

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"id_token": _jwt({
                "iss": "https://accounts.google.com", "aud": "gid.apps.googleusercontent.com",
                "exp": time.time() + 3600, "nonce": nonce, "email": "Person@Example.com",
                "email_verified": True, "name": "A Person",
            })}

    def fake_post(url, data=None, timeout=None):
        seen["url"] = url
        seen["data"] = data
        return FakeResp()

    monkeypatch.setattr(oauth.requests, "post", fake_post)
    r = client.get("/api/auth/oauth/google/callback?code=abc&state=%s" % state, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/app", r.text
    assert "rh_session" in r.headers.get("set-cookie", "")
    assert seen["url"] == oauth.GOOGLE_TOKEN and seen["data"]["code"] == "abc" and seen["data"]["code_verifier"]
    me = client.get("/api/auth/me").json()
    assert me["signed_in"] is True and me["user"]["email"] == "person@example.com"
    # Password sign-in cannot hijack a provider-only account.
    r = client.post("/api/auth/signin", json={"email": "person@example.com", "password": "anything"})
    assert r.status_code == 401
    # The state is single-use.
    r = client.get("/api/auth/oauth/google/callback?code=abc&state=%s" % state, follow_redirects=False)
    assert r.headers["location"] == "/signin?error=oauth_state"


def test_google_callback_rejects_bad_claims(client, monkeypatch):
    def start():
        r = client.get("/api/auth/oauth/google/start", follow_redirects=False)
        q = parse_qs(urlparse(r.headers["location"]).query)
        return q["state"][0], q["nonce"][0]

    def with_token(payload):
        class FakeResp:
            status_code = 200
            text = ""

            def json(self):
                return {"id_token": _jwt(payload)}

        monkeypatch.setattr(oauth.requests, "post", lambda *a, **k: FakeResp())

    base = {"iss": "https://accounts.google.com", "aud": "gid.apps.googleusercontent.com", "exp": time.time() + 600, "email": "x@example.com", "email_verified": True}
    state, nonce = start()
    with_token(dict(base, nonce="wrong"))
    assert client.get("/api/auth/oauth/google/callback?code=c&state=" + state, follow_redirects=False).headers["location"] == "/signin?error=oauth_state"
    state, nonce = start()
    with_token(dict(base, nonce=nonce, aud="someone-else"))
    assert client.get("/api/auth/oauth/google/callback?code=c&state=" + state, follow_redirects=False).headers["location"] == "/signin?error=oauth_failed"
    state, nonce = start()
    with_token(dict(base, nonce=nonce, email_verified=False))
    assert client.get("/api/auth/oauth/google/callback?code=c&state=" + state, follow_redirects=False).headers["location"] == "/signin?error=oauth_email_missing"
    state, nonce = start()
    assert client.get("/api/auth/oauth/google/callback?error=access_denied&state=" + state, follow_redirects=False).headers["location"] == "/signin?error=oauth_denied"
    assert client.get("/api/auth/me").json()["signed_in"] is False


def test_apple_flow_when_cryptography_is_present(tmp_path, monkeypatch):
    if not oauth.es256_available():
        pytest.skip("cryptography not installed; Apple reports unavailable, covered above")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    monkeypatch.setenv("APPLE_CLIENT_ID", "com.example.web")
    monkeypatch.setenv("APPLE_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("APPLE_KEY_ID", "KEYID12345")
    monkeypatch.setenv("APPLE_PRIVATE_KEY", pem)
    monkeypatch.setenv("HUMANIZER_PUBLIC_URL", "https://readshuman.example.com")
    app = create_app(reference_dir=REPO_ROOT / "data" / "reference", web_dir=None, auth=True, auth_db=tmp_path / "a.sqlite")
    with TestClient(app) as c:
        assert c.get("/api/auth/providers").json()["apple"] is True
        r = c.get("/api/auth/oauth/apple/start?next=/app", follow_redirects=False)
        q = parse_qs(urlparse(r.headers["location"]).query)
        assert q["response_mode"] == ["form_post"] and q["redirect_uri"] == ["https://readshuman.example.com/api/auth/oauth/apple/callback"]
        state, nonce = q["state"][0], q["nonce"][0]
        secret_seen = {}

        class FakeResp:
            status_code = 200
            text = ""

            def json(self):
                return {"id_token": _jwt({"iss": "https://appleid.apple.com", "aud": "com.example.web", "exp": time.time() + 600, "nonce": nonce, "email": "apple@example.com", "email_verified": "true"})}

        def fake_post(url, data=None, timeout=None):
            secret_seen["secret"] = data["client_secret"]
            return FakeResp()

        monkeypatch.setattr(oauth.requests, "post", fake_post)
        r = c.post("/api/auth/oauth/apple/callback", data={"code": "c", "state": state, "user": json.dumps({"name": {"firstName": "Ada", "lastName": "L"}})}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/app", r.text
        header = json.loads(base64.urlsafe_b64decode(secret_seen["secret"].split(".")[0] + "==="))
        assert header["alg"] == "ES256" and header["kid"] == "KEYID12345"
        me = c.get("/api/auth/me").json()
        assert me["user"]["email"] == "apple@example.com" and me["user"]["name"] == "Ada L"
