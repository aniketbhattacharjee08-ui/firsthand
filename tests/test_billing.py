"""Paywall tests: auth, credits, refunds, Stripe webhook, rate limits.

Two apps are used. The real one (`humanizer.billing.app.create_app`) checks
that free routes stay free and that a charged route which the server refuses
(the LLM engine is unavailable without MLX, a 503) is refunded. A stub app
with a fake `/api/humanize/llm` that returns 200 checks the happy path of
charging, since no test machine is guaranteed to have a model.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from humanizer.billing import stripe_client, tokens  # noqa: E402
from humanizer.billing.app import create_app, install  # noqa: E402
from humanizer.billing.config import BillingConfig, parse_packs, parse_rate  # noqa: E402
from humanizer.billing.store import Store  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "x" * 48
WHSEC = "whsec_test_secret"

TEXT_300 = " ".join(["word"] * 300)
TEXT_301 = " ".join(["word"] * 301)


def _env(tmp_path, **over):
    base = {
        "LONGHAND_PAYWALL": "1",
        "LONGHAND_SECRET": SECRET,
        "LONGHAND_DB": str(tmp_path / "billing.db"),
        "LONGHAND_DEV_LINKS": "1",
        "LONGHAND_FREE_CREDITS": "3",
        "LONGHAND_WORDS_PER_CREDIT": "300",
        "LONGHAND_RATE_LIMIT": "1000/minute",
        "LONGHAND_AUTH_RATE_LIMIT": "1000/minute",
        "STRIPE_SECRET_KEY": "sk_test_x",
        "STRIPE_WEBHOOK_SECRET": WHSEC,
        "STRIPE_PRICES": "starter:price_1:50:$5 for 50;pro:price_2:250:$20 for 250",
        "LONGHAND_ADMIN_TOKEN": "adm",
    }
    base.update(over)
    return BillingConfig.from_env(base)


def _stub_app(cfg):
    """A tiny app with the routes the gate cares about, all returning 200."""
    app = FastAPI()

    @app.post("/api/humanize/llm")
    def llm(body: dict):
        return {"ok": True, "words": len(body.get("text", "").split())}

    @app.post("/api/humanize")
    def rule(body: dict):
        return {"ok": True}

    @app.post("/api/detect")
    def detect(body: dict):
        return {"ok": True, "detectors": body.get("detectors")}

    @app.get("/api/models")
    def models():
        return {"ok": True}

    @app.post("/api/fail")
    def fail():
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"error": "down"})

    # /api/fail is not in RULES; register it as charged for the refund test.
    from humanizer.billing import gate
    import re

    if not any(p.pattern == r"^/api/fail$" for _m, p, _r in gate.RULES):
        gate.RULES.insert(0, ("POST", re.compile(r"^/api/fail$"), "charge"))
    return install(app, config=cfg, store=Store(cfg.db_path))


def _login(client):
    r = client.post("/api/auth/request-link", json={"email": "a@example.com"})
    assert r.status_code == 200, r.text
    r = client.post("/api/auth/verify", json={"token": r.json()["dev_token"]})
    assert r.status_code == 200, r.text
    return r.json()["user"]


# ------------------------------------------------------------------ config


def test_parse_packs_and_rates():
    packs = parse_packs("starter:price_1:50:$5 for 50;pro:price_2:250")
    assert [p.name for p in packs] == ["starter", "pro"]
    assert packs[0].label == "$5 for 50" and packs[1].label == "pro"
    assert parse_rate("60/minute", (1, 1.0)) == (60, 60.0)
    assert parse_rate("", (7, 9.0)) == (7, 9.0)
    with pytest.raises(ValueError):
        parse_packs("bad")
    with pytest.raises(ValueError):
        parse_rate("sixty", (1, 1.0))


def test_paywall_on_requires_secret(tmp_path):
    with pytest.raises(ValueError):
        BillingConfig.from_env({"LONGHAND_PAYWALL": "1", "LONGHAND_SECRET": "short"})


def test_tokens_round_trip_and_kind_isolation():
    t = tokens.sign(SECRET, "login", {"email": "a@b.co"}, 60)
    assert tokens.verify(SECRET, "login", t)["email"] == "a@b.co"
    assert tokens.verify(SECRET, "session", t) is None
    assert tokens.verify(SECRET + "y", "login", t) is None
    assert tokens.verify(SECRET, "login", t[:-2] + "zz") is None
    expired = tokens.sign(SECRET, "login", {}, -1)
    assert tokens.verify(SECRET, "login", expired) is None


# ------------------------------------------------------------------- store


def test_store_charge_is_atomic_and_ledger_reconciles(tmp_path):
    s = Store(tmp_path / "s.db")
    u = s.get_or_create_user("A@Example.com", 3)
    assert u["email"] == "a@example.com" and u["credits"] == 3
    assert s.charge(u["id"], 2, "charge") == 1
    assert s.charge(u["id"], 2, "charge") is None
    assert s.balance(u["id"]) == 1
    assert s.credit(u["id"], 5, "purchase") == 6
    entries = s.ledger(u["id"])
    assert [e["delta"] for e in entries] == [5, -2, 3]
    assert entries[0]["balance_after"] == 6
    # Same email again is the same user, no second signup grant.
    assert s.get_or_create_user("a@example.com", 3)["credits"] == 6


def test_store_nonce_single_use_and_api_keys(tmp_path):
    s = Store(tmp_path / "s.db")
    u = s.get_or_create_user("k@example.com", 0)
    n = s.issue_nonce("k@example.com")
    assert s.consume_nonce(n, "k@example.com")
    assert not s.consume_nonce(n, "k@example.com")
    key = s.create_api_key(u["id"], "ci")
    assert key["key"].startswith("lh_")
    assert s.api_key_user(key["key"])["id"] == u["id"]
    assert s.api_key_user("lh_nope") is None
    assert s.revoke_api_key(u["id"], key["id"])
    assert s.api_key_user(key["key"]) is None


# -------------------------------------------------------------- real app


@pytest.fixture()
def real_client(tmp_path):
    cfg = _env(tmp_path)
    app = create_app(reference_dir=REPO_ROOT / "data" / "reference", web_dir=None, config=cfg, store=Store(cfg.db_path))
    with TestClient(app) as c:
        yield c


def test_free_routes_stay_free(real_client):
    assert real_client.get("/api/health").status_code == 200
    r = real_client.post("/api/analyze", json={"text": TEXT_300, "detect": False})
    assert r.status_code == 200
    r = real_client.post("/api/humanize", json={"text": TEXT_300})
    assert r.status_code == 200
    h = real_client.get("/api/billing/health").json()
    assert h["paywall"] is True and [p["name"] for p in h["packs"]] == ["starter", "pro"]


def test_llm_route_requires_login(real_client):
    r = real_client.post("/api/humanize/llm", json={"text": TEXT_300})
    assert r.status_code == 401
    assert r.json()["error"] == "login_required"


def test_refused_request_is_refunded(real_client):
    user = _login(real_client)
    assert user["credits"] == 3
    # An unknown style is refused by `llm_precheck` with a 400 before any
    # model is touched, so this is deterministic on every machine.
    r = real_client.post("/api/humanize/llm", json={"text": TEXT_300, "style": "bogus"})
    assert r.status_code == 400, r.text
    assert r.headers["x-longhand-credits-charged"] == "0"
    me = real_client.get("/api/me").json()["user"]
    assert me["credits"] == 3
    kinds = [e["kind"] for e in real_client.get("/api/billing/ledger").json()["entries"]]
    assert kinds[:2] == ["refund", "charge"]


def test_admin_routes_blocked_without_token(real_client):
    assert real_client.get("/api/models").status_code == 403
    assert real_client.post("/api/models/release").status_code == 403
    assert real_client.get("/api/models", headers={"X-Admin-Token": "adm"}).status_code == 200


def test_verify_get_renders_page_without_consuming_link(real_client):
    """Mail scanners prefetch links; the GET must not burn the nonce."""
    r = real_client.post("/api/auth/request-link", json={"email": "g@example.com"})
    link = r.json()["dev_link"]
    token = r.json()["dev_token"]
    path = link.split("http://127.0.0.1:8000", 1)[1]
    r = real_client.get(path, follow_redirects=False)
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert json.dumps(token) in r.text and "set-cookie" not in r.headers
    # Fetch it twice more (a scanner), then the real POST still works.
    real_client.get(path)
    real_client.get(path)
    r = real_client.post("/api/auth/verify", json={"token": token})
    assert r.status_code == 200
    assert real_client.get("/api/me").json()["user"]["email"] == "g@example.com"
    # Reusing the link now fails, and a bad token on GET redirects.
    assert real_client.post("/api/auth/verify", json={"token": token}).status_code == 400
    r = real_client.get("/api/auth/verify?token=garbage", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?login_error=bad_token"


def test_logout_revokes_session(real_client):
    _login(real_client)
    assert real_client.get("/api/me").json()["user"] is not None
    real_client.post("/api/auth/logout")
    assert real_client.get("/api/me").json()["user"] is None


# -------------------------------------------------------------- stub app


@pytest.fixture()
def stub_client(tmp_path):
    cfg = _env(tmp_path)
    with TestClient(_stub_app(cfg)) as c:
        yield c


def test_charge_by_word_count(stub_client):
    user = _login(stub_client)
    assert user["credits"] == 3
    r = stub_client.post("/api/humanize/llm", json={"text": TEXT_300})
    assert r.status_code == 200
    assert r.headers["x-longhand-credits-charged"] == "1"
    assert r.headers["x-longhand-credits-balance"] == "2"
    r = stub_client.post("/api/humanize/llm", json={"text": TEXT_301})
    assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "2"
    assert stub_client.get("/api/me").json()["user"]["credits"] == 0
    r = stub_client.post("/api/humanize/llm", json={"text": "one more"})
    assert r.status_code == 402
    body = r.json()
    assert body["error"] == "insufficient_credits" and body["needed"] == 1 and body["balance"] == 0


def test_server_error_refunds(stub_client):
    _login(stub_client)
    r = stub_client.post("/api/fail", json={"text": TEXT_300})
    assert r.status_code == 503
    assert stub_client.get("/api/me").json()["user"]["credits"] == 3


def test_detect_free_unless_operator_gptzero(stub_client):
    r = stub_client.post("/api/detect", json={"text": TEXT_300, "detectors": ["surrogate"]})
    assert r.status_code == 200
    r = stub_client.post("/api/detect", json={"text": TEXT_300, "detectors": ["gptzero"]})
    assert r.status_code == 401
    _login(stub_client)
    r = stub_client.post("/api/detect", json={"text": TEXT_300, "detectors": ["gptzero"]})
    assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "1"


def test_api_key_bearer_auth(stub_client):
    _login(stub_client)
    r = stub_client.post("/api/auth/keys", json={"label": "ci"})
    assert r.status_code == 201
    key = r.json()["key"]
    stub_client.post("/api/auth/logout")
    anon = stub_client.post("/api/humanize/llm", json={"text": "hi"})
    assert anon.status_code == 401
    r = stub_client.post("/api/humanize/llm", json={"text": "hi"}, headers={"Authorization": "Bearer " + key})
    assert r.status_code == 200
    r = stub_client.get("/api/auth/keys", headers={"Authorization": "Bearer " + key})
    assert len(r.json()["keys"]) == 1 and "key" not in r.json()["keys"][0]


def test_webhook_credits_once(stub_client):
    user = _login(stub_client)
    event = {
        "id": "evt_1",
        "type": "checkout.session.completed",
        "data": {"object": {"id": "cs_1", "payment_status": "paid", "metadata": {"user_id": user["id"], "credits": "50"}}},
    }
    raw = json.dumps(event).encode()
    sig = stripe_client.sign_webhook(WHSEC, raw)
    r = stub_client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": sig, "Content-Type": "application/json"})
    assert r.status_code == 200 and r.json()["credited"] == 50
    assert stub_client.get("/api/me").json()["user"]["credits"] == 53
    r = stub_client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": sig, "Content-Type": "application/json"})
    assert r.json().get("duplicate") is True
    assert stub_client.get("/api/me").json()["user"]["credits"] == 53


def test_webhook_rejects_bad_signature_and_stale_timestamp(stub_client):
    raw = json.dumps({"id": "evt_2", "type": "checkout.session.completed"}).encode()
    r = stub_client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": "t=1,v1=00"})
    assert r.status_code == 400
    stale = stripe_client.sign_webhook(WHSEC, raw, ts=1)
    r = stub_client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": stale})
    assert r.status_code == 400
    r = stub_client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": stripe_client.sign_webhook("other", raw)})
    assert r.status_code == 400


def test_checkout_requires_login_and_known_pack(stub_client, monkeypatch):
    assert stub_client.post("/api/billing/checkout", json={"pack": "starter"}).status_code == 401
    _login(stub_client)
    assert stub_client.post("/api/billing/checkout", json={"pack": "nope"}).status_code == 400
    monkeypatch.setattr(stripe_client, "create_checkout_session", lambda *a, **k: {"url": "https://checkout.stripe.com/x", "id": "cs_x"})
    r = stub_client.post("/api/billing/checkout", json={"pack": "pro"})
    assert r.status_code == 200 and r.json()["url"].startswith("https://checkout.stripe.com/")


def test_rate_limit(tmp_path):
    cfg = _env(tmp_path, LONGHAND_RATE_LIMIT="3/minute")
    with TestClient(_stub_app(cfg)) as c:
        codes = [c.post("/api/humanize", json={"text": "x"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_paywall_off_passes_everything(tmp_path):
    cfg = _env(tmp_path, LONGHAND_PAYWALL="0")
    with TestClient(_stub_app(cfg)) as c:
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}).status_code == 200
        assert c.get("/api/models").status_code == 200
        assert c.get("/api/billing/health").json()["paywall"] is False
        assert c.get("/api/me").json() == {"user": None, "paywall": False}


# ------------------------------------------------ phase 1: gate additions

import threading  # noqa: E402
import time  # noqa: E402

from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from humanizer.billing import gate as gate_mod  # noqa: E402
from humanizer.billing.gate import looks_unchanged  # noqa: E402


def _phase1_app(cfg, hold=None):
    """Routes shaped like the server's engine routes, with controllable behaviour."""
    app = FastAPI()
    jobs = {}

    @app.post("/api/humanize/llm")
    def llm(body: dict):
        if hold is not None:
            hold.wait(5)
        text = body.get("text", "")
        if body.get("style") == "noop":
            return {"original": text, "humanized": text, "summary": {"paragraphs_unchanged": 1}, "paragraphs": [{}]}
        return {"original": text, "humanized": text + " rewritten", "summary": {"paragraphs_unchanged": 0}, "paragraphs": [{}]}

    @app.post("/api/humanize/stream")
    def llm_stream(body: dict):
        text = body.get("text", "")
        out = text if body.get("style") == "noop" else text + " rewritten"

        def frames():
            yield "event: progress\ndata: {\"stage\":\"analyze\"}\n\n"
            yield "event: result\ndata: %s\n\n" % json.dumps({"original": text, "humanized": out})
            yield "event: done\ndata: {\"ok\":true}\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    @app.post("/api/humanize/jobs")
    def park(body: dict):
        jid = "job%d" % (len(jobs) + 1)
        jobs[jid] = body
        return {"job_id": jid, "stream_url": "/api/humanize/stream?job=%s" % jid}

    @app.get("/api/humanize/stream")
    def resume(job: str = ""):
        body = jobs.pop(job, None)
        if body is None:
            return JSONResponse(status_code=404, content={"error": "unknown_job"})
        text = body.get("text", "")
        out = text if body.get("style") == "noop" else text + " rewritten"

        def frames():
            yield "event: result\ndata: %s\n\n" % json.dumps({"original": text, "humanized": out})
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    return install(app, config=cfg, store=Store(cfg.db_path))


def _middleware(app):
    """The PaywallMiddleware instance inside a built app's stack."""
    mw = app.middleware_stack
    while mw is not None and not isinstance(mw, gate_mod.PaywallMiddleware):
        mw = getattr(mw, "app", None)
    assert mw is not None
    return mw


def test_looks_unchanged():
    same = json.dumps({"original": "a b", "humanized": "a b "}).encode()
    diff = json.dumps({"original": "a b", "humanized": "a c"}).encode()
    assert looks_unchanged("application/json", same) is True
    assert looks_unchanged("application/json", diff) is False
    assert looks_unchanged("application/json", b'{"error": "x"}') is None
    sse = b"event: progress\ndata: {}\n\nevent: result\ndata: " + same + b"\n\nevent: done\ndata: {}\n\n"
    assert looks_unchanged("text/event-stream", sse) is True
    assert looks_unchanged("text/event-stream", b"event: progress\ndata: {}\n\n") is None
    by_summary = json.dumps({"paragraphs": [{}, {}], "summary": {"paragraphs_unchanged": 2}}).encode()
    assert looks_unchanged("application/json", by_summary) is True


def test_unchanged_rewrite_is_refunded_json_and_sse(tmp_path):
    cfg = _env(tmp_path)
    with TestClient(_phase1_app(cfg)) as c:
        _login(c)
        r = c.post("/api/humanize/llm", json={"text": TEXT_300, "style": "noop"})
        assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "1"
        assert c.get("/api/me").json()["user"]["credits"] == 3  # refunded after the fact
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert c.get("/api/me").json()["user"]["credits"] == 2
        r = c.post("/api/humanize/stream", json={"text": TEXT_300, "style": "noop"})
        assert r.status_code == 200 and "event: result" in r.text
        assert c.get("/api/me").json()["user"]["credits"] == 2
        r = c.post("/api/humanize/stream", json={"text": TEXT_300})
        assert c.get("/api/me").json()["user"]["credits"] == 1
        kinds = [e["kind"] for e in c.get("/api/billing/ledger").json()["entries"]]
        assert kinds.count("refund") == 2


def test_unchanged_refund_can_be_disabled(tmp_path):
    cfg = _env(tmp_path, LONGHAND_REFUND_UNCHANGED="0")
    with TestClient(_phase1_app(cfg)) as c:
        _login(c)
        c.post("/api/humanize/llm", json={"text": TEXT_300, "style": "noop"})
        assert c.get("/api/me").json()["user"]["credits"] == 2


def test_parked_job_charged_once_and_consumed_on_resume(tmp_path):
    cfg = _env(tmp_path)
    app = _phase1_app(cfg)
    with TestClient(app) as c:
        _login(c)
        r = c.post("/api/humanize/jobs", json={"text": TEXT_300})
        assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "1"
        jid = r.json()["job_id"]
        mw = _middleware(app)
        assert jid in mw.jobs
        # The GET needs no cookie: the job id is the paid capability.
        c.cookies.clear()
        r = c.get("/api/humanize/stream", params={"job": jid})
        assert r.status_code == 200 and "rewritten" in r.text
        assert jid not in mw.jobs
        # Unknown job: server 404s, nothing to refund, no crash.
        assert c.get("/api/humanize/stream", params={"job": "nope"}).status_code == 404
    store = Store(cfg.db_path)
    u = store.user_by_email("a@example.com")
    assert store.balance(u["id"]) == 2


def test_parked_job_unclaimed_is_refunded_after_ttl(tmp_path):
    cfg = _env(tmp_path, LONGHAND_JOB_TTL_S="1")
    app = _phase1_app(cfg)
    with TestClient(app) as c:
        _login(c)
        r = c.post("/api/humanize/jobs", json={"text": TEXT_300})
        jid = r.json()["job_id"]
        assert c.get("/api/me").json()["user"]["credits"] == 2
        mw = _middleware(app)
        assert mw.sweep_jobs(now=time.time() + 2) == 1
        assert jid not in mw.jobs
        assert c.get("/api/me").json()["user"]["credits"] == 3
        # A late GET after the refund gets the server's 404 and no double refund.
        c.get("/api/humanize/stream", params={"job": jid})
        assert c.get("/api/me").json()["user"]["credits"] == 3


def test_parked_job_unchanged_is_refunded(tmp_path):
    cfg = _env(tmp_path)
    with TestClient(_phase1_app(cfg)) as c:
        _login(c)
        jid = c.post("/api/humanize/jobs", json={"text": TEXT_300, "style": "noop"}).json()["job_id"]
        c.get("/api/humanize/stream", params={"job": jid})
        assert c.get("/api/me").json()["user"]["credits"] == 3


def test_busy_when_slot_taken(tmp_path):
    cfg = _env(tmp_path, LONGHAND_MAX_INFLIGHT="1", LONGHAND_QUEUE_WAIT_S="0.3")
    hold = threading.Event()
    app = _phase1_app(cfg, hold=hold)
    with TestClient(app) as c:
        _login(c)
        cookies = dict(c.cookies)
        results = {}

        def first():
            with TestClient(app, cookies=cookies) as c1:
                results["first"] = c1.post("/api/humanize/llm", json={"text": TEXT_300})

        t = threading.Thread(target=first)
        t.start()
        time.sleep(0.15)  # let the first request take the slot
        second = c.post("/api/humanize/llm", json={"text": TEXT_300})
        hold.set()
        t.join(5)
        assert results["first"].status_code == 200
        assert second.status_code == 503
        assert second.json()["error"] == "busy" and "retry-after" in second.headers
        # First charged one credit, second refunded: 3 - 1 = 2.
        assert c.get("/api/me").json()["user"]["credits"] == 2


def test_request_log_written(tmp_path):
    cfg = _env(tmp_path, LONGHAND_LOG_DIR=str(tmp_path / "logs"))
    with TestClient(_phase1_app(cfg)) as c:
        _login(c)
        c.post("/api/humanize/llm", json={"text": TEXT_300})
        c.post("/api/humanize/llm", json={"text": TEXT_300, "style": "noop"})
    lines = [json.loads(l) for l in (tmp_path / "logs" / "requests.log").read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["path"] == "/api/humanize/llm" and lines[0]["credits"] == 1 and lines[0]["status"] == 200
    assert lines[1]["unchanged"] is True and lines[1]["refunded"] == 1
    assert all("text" not in l for l in lines)


def test_signup_limits_and_blocklist(tmp_path):
    cfg = _env(tmp_path, LONGHAND_SIGNUPS_PER_IP_PER_DAY="2")
    with TestClient(_phase1_app(cfg)) as c:
        r = c.post("/api/auth/request-link", json={"email": "x@mailinator.com"})
        assert r.status_code == 400 and r.json()["error"] == "blocked_email"
        assert c.post("/api/auth/request-link", json={"email": "u1@example.com"}).status_code == 200
        assert c.post("/api/auth/request-link", json={"email": "u2@example.com"}).status_code == 200
        r = c.post("/api/auth/request-link", json={"email": "u3@example.com"})
        assert r.status_code == 429 and r.json()["error"] == "signup_limit"
        # An existing account is never blocked by the cap.
        assert c.post("/api/auth/request-link", json={"email": "u1@example.com"}).status_code == 200


def test_custom_blocklist_file(tmp_path):
    bl = tmp_path / "blocked.txt"
    bl.write_text("# comment\nexample.org\n")
    cfg = _env(tmp_path, LONGHAND_EMAIL_BLOCKLIST=str(bl))
    assert "example.org" in cfg.blocked_domains and "mailinator.com" in cfg.blocked_domains


def test_delete_account(tmp_path):
    cfg = _env(tmp_path)
    with TestClient(_phase1_app(cfg)) as c:
        _login(c)
        c.post("/api/auth/keys", json={"label": "k"})
        r = c.delete("/api/me")
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert c.get("/api/me").json()["user"] is None
    store = Store(cfg.db_path)
    assert store.user_by_email("a@example.com") is None
    st = store.stats()
    assert st["deleted_users"] == 1 and st["users"] == 0


def test_admin_cli(tmp_path, monkeypatch, capsys):
    from humanizer.billing.__main__ import main

    for k, v in {
        "LONGHAND_PAYWALL": "1", "LONGHAND_SECRET": SECRET, "LONGHAND_DB": str(tmp_path / "b.db"),
        "STRIPE_PRICES": "", "LONGHAND_EMAIL_BLOCKLIST": "",
    }.items():
        monkeypatch.setenv(k, v)
    assert main(["admin", "grant", "ops@example.com", "20", "--note", "promo"]) == 0
    assert main(["admin", "set-admin", "ops@example.com"]) == 0
    assert main(["admin", "user", "ops@example.com"]) == 0
    out = capsys.readouterr().out
    assert "balance 20" in out and "admin=True" in out and "manual" in out
    assert main(["admin", "stats"]) == 0
    out = capsys.readouterr().out
    import re as _re

    assert _re.search(r"credits_granted\s+20", out)
    assert main(["check"]) == 0
    assert main(["admin", "delete", "ops@example.com"]) == 0
    assert main(["admin", "user", "ops@example.com"]) == 1


def test_real_app_with_static_mount_keeps_routes_reachable(tmp_path):
    """Production mounts web/ at "/"; billing routes must still win the match."""
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>t</title>ok")
    cfg = _env(tmp_path)
    app = create_app(reference_dir=REPO_ROOT / "data" / "reference", web_dir=web, config=cfg, store=Store(cfg.db_path))
    with TestClient(app) as c:
        assert c.get("/").status_code == 200 and "ok" in c.get("/").text
        assert c.get("/api/billing/health").json()["paywall"] is True
        assert c.get("/api/me").json()["user"] is None
        assert c.post("/api/auth/request-link", json={"email": "s@example.com"}).status_code == 200
        assert c.get("/api/humanize/llm/health").status_code == 200


# ------------------------------------------------ security review fixes


def _phase2_app(cfg):
    """Engine-shaped routes with error and slow behaviours for the review fixes."""
    app = FastAPI()
    jobs = {}

    @app.post("/api/humanize/llm")
    def llm(body: dict):
        return {"original": body.get("text", ""), "humanized": body.get("text", "") + " rewritten", "n": body.get("n_candidates")}

    @app.post("/api/humanize/stream")
    def llm_stream(body: dict):
        fail = body.get("style") == "explode"

        def frames():
            yield "event: progress\ndata: {}\n\n"
            if fail:
                yield "event: error\ndata: {\"error\":\"pipeline_failed\"}\n\n"
            else:
                yield "event: result\ndata: %s\n\n" % json.dumps({"original": "a", "humanized": "b"})
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    @app.post("/api/humanize/jobs")
    def park(body: dict):
        jid = "job%d" % (len(jobs) + 1)
        jobs[jid] = body
        return {"job_id": jid}

    @app.get("/api/humanize/stream")
    def resume(job: str = ""):
        body = jobs.pop(job, None)
        if body is None:
            return JSONResponse(status_code=404, content={"error": "unknown_job", "server": True})

        def frames():
            yield "event: result\ndata: %s\n\n" % json.dumps({"original": "a", "humanized": "b"})

        return StreamingResponse(frames(), media_type="text/event-stream")

    @app.post("/api/humanize")
    def rule(body: dict):
        return {"ok": True}

    return install(app, config=cfg, store=Store(cfg.db_path))


def test_duplicate_job_param_is_rejected_and_unknown_job_never_forwards(tmp_path):
    cfg = _env(tmp_path)
    app = _phase2_app(cfg)
    with TestClient(app) as c:
        _login(c)
        jid = c.post("/api/humanize/jobs", json={"text": TEXT_300}).json()["job_id"]
        r = c.get("/api/humanize/stream?job=%s&job=nope" % jid)
        assert r.status_code == 400 and r.json()["error"] == "bad_job"
        # The job is still tracked and still paid for.
        assert jid in _middleware(app).jobs
        assert c.get("/api/me").json()["user"]["credits"] == 2
        # Unknown ids are answered by the gate, never by the server.
        r = c.get("/api/humanize/stream?job=nope")
        assert r.status_code == 404 and "server" not in r.json()
        # The real one still works once.
        assert c.get("/api/humanize/stream?job=%s" % jid).status_code == 200
        assert c.get("/api/humanize/stream?job=%s" % jid).status_code == 404


def test_sse_error_frame_is_refunded(tmp_path):
    cfg = _env(tmp_path)
    with TestClient(_phase2_app(cfg)) as c:
        _login(c)
        r = c.post("/api/humanize/stream", json={"text": TEXT_300, "style": "explode"})
        assert r.status_code == 200 and "event: error" in r.text
        assert c.get("/api/me").json()["user"]["credits"] == 3
        r = c.post("/api/humanize/stream", json={"text": TEXT_300})
        assert c.get("/api/me").json()["user"]["credits"] == 2
        refs = [e["ref"] for e in c.get("/api/billing/ledger").json()["entries"] if e["kind"] == "refund"]
        assert any("stream error" in x for x in refs)


def test_pricing_matches_server_word_count_and_rejects_bad_json(tmp_path):
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="20")
    with TestClient(_phase2_app(cfg)) as c:
        _login(c)
        digits = " ".join(str(i) for i in range(301))
        r = c.post("/api/humanize/llm", json={"text": digits})
        assert r.headers["x-longhand-credits-charged"] == "2"
        bom = b"\xef\xbb\xbf" + json.dumps({"text": TEXT_301}).encode()
        r = c.post("/api/humanize/llm", content=bom, headers={"content-type": "application/json"})
        assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "2"
        r = c.post("/api/humanize/llm", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 400 and r.json()["error"] == "bad_json"
        r = c.post("/api/humanize/llm", content=b"[1,2]", headers={"content-type": "application/json"})
        assert r.status_code == 400
        assert c.get("/api/me").json()["user"]["credits"] == 16


def test_effort_multiplier_prices_candidates_and_rounds(tmp_path):
    from humanizer.billing.quota import effort_multiplier

    assert effort_multiplier(6, 1) == 1
    assert effort_multiplier(16, 1) == 3
    assert effort_multiplier(6, 3) == 3
    assert effort_multiplier(6, 1, 6) == 3
    assert effort_multiplier("junk", None) == 1
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="10")
    with TestClient(_phase2_app(cfg)) as c:
        _login(c)
        r = c.post("/api/humanize/llm", json={"text": TEXT_300, "n_candidates": 16, "rounds": 2})
        assert r.headers["x-longhand-credits-charged"] == "6"
        r = c.post("/api/humanize/llm", json={"text": TEXT_300, "n_candidates": 16, "rounds": 3})
        assert r.status_code == 402 and r.json()["needed"] == 9 and r.json()["effort"] == 9


def test_unchanged_refund_capped_per_day(tmp_path):
    cfg = _env(tmp_path, LONGHAND_UNCHANGED_REFUNDS_PER_DAY="2", LONGHAND_FREE_CREDITS="10")
    with TestClient(_phase1_app(cfg)) as c:
        _login(c)
        for _ in range(4):
            c.post("/api/humanize/llm", json={"text": TEXT_300, "style": "noop"})
        # Two refunded, two kept.
        assert c.get("/api/me").json()["user"]["credits"] == 8


def test_free_post_routes_have_body_limit(tmp_path):
    cfg = _env(tmp_path, LONGHAND_MAX_BODY_BYTES="500")
    with TestClient(_phase2_app(cfg)) as c:
        r = c.post("/api/humanize", json={"text": "x" * 1000})
        assert r.status_code == 413
        assert c.post("/api/humanize", json={"text": "short"}).status_code == 200


def test_trust_proxy_uses_real_ip_or_rightmost_forwarded(tmp_path):
    from humanizer.billing.gate import client_ip

    scope = {"client": ("10.0.0.1", 1), "headers": [(b"x-forwarded-for", b"1.1.1.1, 2.2.2.2")]}
    assert client_ip(scope, False) == "10.0.0.1"
    assert client_ip(scope, True) == "2.2.2.2"
    scope["headers"].append((b"x-real-ip", b"3.3.3.3"))
    assert client_ip(scope, True) == "3.3.3.3"


def test_store_recovers_from_failed_transaction(tmp_path):
    s = Store(tmp_path / "s.db")
    u = s.get_or_create_user("r@example.com", 3)
    with pytest.raises(KeyError):
        s.credit("nobody", 5, "manual")
    # The connection is not wedged.
    assert s.charge(u["id"], 1, "charge") == 2
    assert s.credit(u["id"], 1, "refund") == 3
    # credit_once is atomic and idempotent.
    assert s.credit_once("evt_a", "checkout.session.completed", u["id"], 10, "purchase") == 13
    assert s.credit_once("evt_a", "checkout.session.completed", u["id"], 10, "purchase") is None
    with pytest.raises(KeyError):
        s.credit_once("evt_b", "checkout.session.completed", "nobody", 10, "purchase")
    # A failed credit_once leaves the event unrecorded so a retry can succeed.
    assert s.record_webhook("evt_b", "x") is True


def test_webhook_unmatched_user_returns_500_for_retry(stub_client):
    _login(stub_client)
    event = {"id": "evt_9", "type": "checkout.session.completed", "data": {"object": {"id": "cs_9", "payment_status": "paid", "metadata": {"user_id": "ghost", "credits": "50"}}}}
    raw = json.dumps(event).encode()
    r = stub_client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": stripe_client.sign_webhook(WHSEC, raw)})
    assert r.status_code == 500
    # Stripe retries; a later fix (user exists) then credits exactly once.
    event["data"]["object"]["metadata"]["user_id"] = stub_client.get("/api/me").json()["user"]["id"]
    raw = json.dumps(event).encode()
    r = stub_client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": stripe_client.sign_webhook(WHSEC, raw)})
    assert r.status_code == 200 and r.json()["credited"] == 50


def test_dev_links_refused_on_https_with_paywall(tmp_path):
    with pytest.raises(ValueError):
        _env(tmp_path, LONGHAND_PUBLIC_URL="https://readshuman.example.com")
    cfg = _env(tmp_path, LONGHAND_PUBLIC_URL="https://readshuman.example.com", LONGHAND_DEV_LINKS="0")
    assert cfg.secure_cookies


def test_admin_token_still_works_with_compare_digest(real_client):
    assert real_client.get("/api/models", headers={"X-Admin-Token": "adm"}).status_code == 200
    assert real_client.get("/api/models", headers={"X-Admin-Token": "adn"}).status_code == 403
