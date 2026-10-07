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


def _real_app(tmp_path, cfg, web_dir=None):
    return create_app(
        reference_dir=REPO_ROOT / "data" / "reference",
        web_dir=web_dir,
        config=cfg,
        store=Store(cfg.db_path),
        auth_db=tmp_path / "auth.sqlite",
    )


@pytest.fixture()
def real_client(tmp_path):
    cfg = _env(tmp_path)
    with TestClient(_real_app(tmp_path, cfg)) as c:
        yield c


def test_free_routes_stay_free(real_client):
    assert real_client.get("/api/health").status_code == 200
    r = real_client.post("/api/analyze", json={"text": TEXT_300, "detect": False})
    assert r.status_code == 200
    r = real_client.post("/api/humanize", json={"text": TEXT_300})
    assert r.status_code == 200
    h = real_client.get("/api/billing/health").json()
    assert h["paywall"] is True and [p["name"] for p in h["packs"]] == ["starter", "pro"]


def test_llm_route_charges_a_guest_by_address(real_client):
    """Nobody signed in is a guest keyed by client address: the request is
    charged against the address's free allowance instead of refused."""
    me = real_client.get("/api/me").json()
    assert me["user"]["guest"] is True and me["user"]["words_left"] == 900 and me["user"]["email"] == ""
    r = real_client.post("/api/humanize/llm", json={"text": TEXT_300})
    # The route itself may refuse (no model on this machine) but the gate let
    # it through and charged, then refunded on a refusal; either way, not 401.
    assert r.status_code != 401
    assert "x-longhand-credits-charged" in r.headers or r.status_code >= 400


def test_llm_route_requires_login_when_guests_are_off(tmp_path):
    """With LONGHAND_GUESTS=0 the real app keeps the September wall: the 401
    carries the code the site's app.js redirects on."""
    cfg = _env(tmp_path, LONGHAND_GUESTS="0")
    with TestClient(_real_app(tmp_path, cfg)) as c:
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert r.status_code == 401
        assert r.json()["error"] == "sign_in_required" and r.json()["signin_url"] == "/signin"
        assert c.get("/api/me").json() == {"user": None, "paywall": True}


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
    # Signed out, the caller is the address's guest, not the account.
    me = real_client.get("/api/me").json()["user"]
    assert me["guest"] is True and me["email"] == ""


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
    # The operator's GPTZero key costs a credit; a guest spends the address's free words on it.
    r = stub_client.post("/api/detect", json={"text": TEXT_300, "detectors": ["gptzero"]})
    assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "1"
    _login(stub_client)
    r = stub_client.post("/api/detect", json={"text": TEXT_300, "detectors": ["gptzero"]})
    assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "1"


def test_api_key_bearer_auth(stub_client):
    _login(stub_client)
    r = stub_client.post("/api/auth/keys", json={"label": "ci"})
    assert r.status_code == 201
    key = r.json()["key"]
    stub_client.post("/api/auth/logout")
    # Signed out, the same address is a guest and still gets through on its free words.
    anon = stub_client.post("/api/humanize/llm", json={"text": "hi"})
    assert anon.status_code == 200 and stub_client.get("/api/me").json()["user"]["guest"] is True
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
        after = c.get("/api/me").json()["user"]
        assert after is None or after["guest"] is True
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
    (web / "index.html").write_text("<!doctype html><title>t</title>app")
    (web / "home.html").write_text("<!doctype html><title>t</title>ok")
    cfg = _env(tmp_path)
    app = _real_app(tmp_path, cfg, web_dir=web)
    with TestClient(app) as c:
        assert c.get("/").status_code == 200 and "ok" in c.get("/").text
        # /app is public (the paywall limits visitors by address); the site's
        # pages still win over the static mount.
        r = c.get("/app", follow_redirects=False)
        assert r.status_code == 200 and "app" in r.text
        assert c.get("/api/billing/health").json()["paywall"] is True
        assert c.get("/api/me").json()["user"]["guest"] is True
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

    # The product default (8 candidates, 2 rounds) is 1x, so a plan's words
    # are priced as words; only callers asking for more pay more.
    assert effort_multiplier(8, 2) == 1
    assert effort_multiplier(6, 1) == 1
    assert effort_multiplier(16, 2) == 2
    assert effort_multiplier(8, 4) == 2
    assert effort_multiplier(16, 4) == 4
    assert effort_multiplier(8, 2, 6) == 3
    assert effort_multiplier("junk", None) == 1
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="10")
    with TestClient(_phase2_app(cfg)) as c:
        _login(c)
        r = c.post("/api/humanize/llm", json={"text": TEXT_300, "n_candidates": 16, "rounds": 2})
        assert r.headers["x-longhand-credits-charged"] == "2"
        r = c.post("/api/humanize/llm", json={"text": TEXT_300, "n_candidates": 24, "rounds": 6})
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


# ------------------------------------------------------------ plans


import calendar  # noqa: E402

from humanizer.api import auth as auth_mod  # noqa: E402
from humanizer.billing.config import parse_plans  # noqa: E402

PLANS = "monthly:price_m:300:month:$9.99 a month;yearly:price_y:300:year:$79.99 a year;lifetime:price_l:200:lifetime:$299 once"


def _ts(year, month, day=1):
    return float(calendar.timegm((year, month, day, 12, 0, 0)))


def _post_event(client, event):
    raw = json.dumps(event).encode()
    sig = stripe_client.sign_webhook(WHSEC, raw)
    return client.post("/api/billing/webhook", content=raw, headers={"Stripe-Signature": sig, "Content-Type": "application/json"})


def test_parse_plans():
    plans = parse_plans(PLANS)
    assert [p.name for p in plans] == ["monthly", "yearly", "lifetime"]
    assert plans[0].recurring and plans[1].recurring and not plans[2].recurring
    assert plans[2].allowance_credits == 200 and plans[1].label == "$79.99 a year"
    assert parse_plans("") == []
    for bad in ("x:price:300:weekly", "x:price:0:month", "x:price:300", "a:p:1:month;a:p:1:month"):
        with pytest.raises(ValueError):
            parse_plans(bad)
    cfg = _env(pathlib_tmp(), STRIPE_PLANS=PLANS, STRIPE_PRICES="")
    assert cfg.stripe_configured and cfg.plan("yearly").interval == "year" and cfg.plan("nope") is None
    pub = cfg.public()["plans"]
    assert pub[0] == {"name": "monthly", "label": "$9.99 a month", "interval": "month", "allowance_credits": 300, "words_per_month": 90000}


def pathlib_tmp():
    import tempfile

    return Path(tempfile.mkdtemp())


def test_topup_once_per_month_up_to_allowance(tmp_path):
    s = Store(tmp_path / "s.db")
    u = s.get_or_create_user("p@example.com", 3)
    jan, feb = _ts(2030, 1, 15), _ts(2030, 2, 2)
    assert s.topup_if_due(u["id"], jan) is None  # no plan
    row = s.set_plan(u["id"], "monthly", 300, jan + 75 * 86400, customer_id="cus_1", subscription_id="sub_1", now=jan)
    assert row["credits"] == 300 and row["plan_period"] == "2030-01"
    # Same month again: nothing, and no second ledger row.
    assert s.topup_if_due(u["id"], jan + 86400) is None
    assert [e["kind"] for e in s.ledger(u["id"])] == ["allowance", "signup"]
    assert s.ledger(u["id"])[0]["ref"] == "plan:monthly 2030-01" and s.ledger(u["id"])[0]["delta"] == 297
    # Spend some, next month tops back up to 300 exactly.
    s.charge(u["id"], 120, "charge")
    assert s.topup_if_due(u["id"], feb) == 120
    assert s.balance(u["id"]) == 300
    # Purchased credits above the allowance are kept; the month still records as renewed.
    s.credit(u["id"], 500, "purchase")
    mar = _ts(2030, 3, 1)
    assert s.topup_if_due(u["id"], mar) == 0
    assert s.balance(u["id"]) == 800
    assert s.ledger(u["id"])[0]["kind"] == "allowance" and s.ledger(u["id"])[0]["delta"] == 0
    # Expired plan (ended late March): no top-up in June.
    s.charge(u["id"], 800, "charge")
    assert s.topup_if_due(u["id"], _ts(2030, 6, 1)) is None
    assert s.balance(u["id"]) == 0
    assert not Store.plan_active(s.user(u["id"]), _ts(2030, 6, 1))
    # Lifetime keeps renewing across months and years.
    s.set_plan(u["id"], "lifetime", 200, None, now=_ts(2030, 4, 1))
    assert s.balance(u["id"]) == 200
    s.charge(u["id"], 200, "charge")
    assert s.topup_if_due(u["id"], _ts(2031, 7, 1)) == 200
    assert s.topup_if_due(u["id"], _ts(2031, 7, 30)) is None
    # Expire keeps the name for display; clear removes it.
    s.expire_plan(u["id"], _ts(2031, 8, 1))
    assert s.user(u["id"])["plan"] == "lifetime" and s.topup_if_due(u["id"], _ts(2031, 9, 1)) is None
    s.clear_plan(u["id"])
    assert s.user(u["id"])["plan"] == "" and s.user(u["id"])["stripe_customer_id"] == "cus_1"
    assert s.user_by_customer("cus_1")["id"] == u["id"] and s.user_by_subscription("sub_1") is None
    assert s.stats()["credits_allowance"] > 0


def test_apply_once_is_atomic(tmp_path):
    s = Store(tmp_path / "s.db")
    u = s.get_or_create_user("o@example.com", 0)

    def boom():
        s.credit(u["id"], 5, "purchase")
        raise RuntimeError("after the credit")

    with pytest.raises(RuntimeError):
        s.apply_once("evt_x", "t", boom)
    # Neither the credit nor the event survived, so Stripe's retry can apply it.
    assert s.balance(u["id"]) == 0
    assert s.apply_once("evt_x", "t", lambda: s.credit(u["id"], 5, "purchase")) == 5
    assert s.apply_once("evt_x", "t", lambda: s.credit(u["id"], 5, "purchase")) is None
    assert s.balance(u["id"]) == 5


@pytest.fixture()
def plan_client(tmp_path):
    cfg = _env(tmp_path, STRIPE_PLANS=PLANS)
    with TestClient(_stub_app(cfg)) as c:
        yield c


def test_plans_endpoint_and_checkout_modes(plan_client, monkeypatch):
    r = plan_client.get("/api/billing/plans").json()
    assert r["stripe"] is True and r["words_per_credit"] == 300 and r["free_words"] == 900
    assert [p["name"] for p in r["plans"]] == ["monthly", "yearly", "lifetime"]
    assert [p["name"] for p in r["packs"]] == ["starter", "pro"]
    assert plan_client.post("/api/billing/checkout", json={"plan": "monthly"}).status_code == 401
    user = _login(plan_client)
    calls = []

    def fake(secret, price_id, qty, **kw):
        calls.append((price_id, kw))
        return {"url": "https://checkout.stripe.com/x", "id": "cs_%d" % len(calls)}

    monkeypatch.setattr(stripe_client, "create_checkout_session", fake)
    assert plan_client.post("/api/billing/checkout", json={}).status_code == 400
    assert plan_client.post("/api/billing/checkout", json={"plan": "nope"}).status_code == 400
    r = plan_client.post("/api/billing/checkout", json={"plan": "monthly"})
    assert r.status_code == 200 and r.json()["plan"] == "monthly" and r.json()["mode"] == "subscription"
    price, kw = calls[-1]
    assert price == "price_m" and kw["mode"] == "subscription"
    assert kw["metadata"] == {"user_id": user["id"], "plan": "monthly"}
    assert kw["subscription_metadata"] == {"user_id": user["id"], "plan": "monthly"}
    assert kw["customer"] is None and kw["customer_email"] == "a@example.com"
    assert kw["success_url"].endswith("/app?purchase=success") and kw["cancel_url"].endswith("/app?purchase=cancelled")
    r = plan_client.post("/api/billing/checkout", json={"plan": "lifetime"})
    assert r.json()["mode"] == "payment" and calls[-1][1]["metadata"] == {"user_id": user["id"], "plan": "lifetime"}
    assert "subscription_metadata" not in calls[-1][1]
    r = plan_client.post("/api/billing/checkout", json={"pack": "starter"})
    assert r.json()["pack"] == "starter" and calls[-1][1]["metadata"]["credits"] == "50"
    assert calls[-1][1]["success_url"].endswith("/app?purchase=success")
    # Portal needs a stored customer.
    assert plan_client.post("/api/billing/portal").status_code == 404


def test_plan_checkout_webhook_never_takes_pack_path(plan_client):
    user = _login(plan_client)
    # No `credits` in metadata: the pack path would 500. The plan path applies it.
    event = {
        "id": "evt_p1",
        "type": "checkout.session.completed",
        "data": {"object": {"id": "cs_p1", "payment_status": "paid", "customer": "cus_9", "subscription": "sub_9",
                            "metadata": {"user_id": user["id"], "plan": "monthly"}}},
    }
    r = _post_event(plan_client, event)
    assert r.status_code == 200 and r.json()["plan"] == "monthly" and r.json()["balance"] == 300
    me = plan_client.get("/api/me").json()["user"]
    assert me["plan"] == "monthly" and me["plan_allowance"] == 300 and me["credits"] == 300
    assert me["words_left"] == 90000 and me["words_per_credit"] == 300 and me["plan_active"] is True
    assert me["has_billing_portal"] is True and me["is_admin"] is False
    assert 30 * 86400 < me["plan_until"] - time.time() < 36 * 86400
    assert _post_event(plan_client, event).json()["duplicate"] is True
    assert plan_client.get("/api/me").json()["user"]["credits"] == 300
    kinds = [e["kind"] for e in plan_client.get("/api/billing/ledger").json()["entries"]]
    assert kinds.count("allowance") == 1 and "purchase" not in kinds
    # Lifetime: plan_until is None and the allowance is the lifetime one.
    event = {"id": "evt_p2", "type": "checkout.session.completed",
             "data": {"object": {"id": "cs_p2", "payment_status": "paid", "metadata": {"user_id": user["id"], "plan": "lifetime"}}}}
    assert _post_event(plan_client, event).json()["plan_until"] is None
    me = plan_client.get("/api/me").json()["user"]
    assert me["plan"] == "lifetime" and me["plan_until"] is None and me["credits"] == 300  # 300 already above 200
    # A paid plan for an unknown user still 500s so Stripe retries; unknown plan names too.
    event = {"id": "evt_p3", "type": "checkout.session.completed",
             "data": {"object": {"id": "cs_p3", "payment_status": "paid", "metadata": {"user_id": "ghost", "plan": "monthly"}}}}
    assert _post_event(plan_client, event).status_code == 500
    # Ledger shows the plan and the portal works once a customer is stored.
    assert plan_client.get("/api/billing/ledger").json()["plan"] == "lifetime"


def test_subscription_lifecycle_events(plan_client, monkeypatch):
    user = _login(plan_client)
    now = time.time()
    end = now + 30 * 86400
    # The subscription event can arrive before the checkout event: the plan
    # starts from the subscription's own metadata.
    sub = {"id": "sub_1", "customer": "cus_1", "status": "active", "current_period_end": end,
           "metadata": {"user_id": user["id"], "plan": "yearly"}}
    r = _post_event(plan_client, {"id": "evt_s1", "type": "customer.subscription.created", "data": {"object": sub}})
    assert r.status_code == 200 and abs(r.json()["plan_until"] - (end + 3 * 86400)) < 1
    me = plan_client.get("/api/me").json()["user"]
    assert me["plan"] == "yearly" and me["credits"] == 300 and abs(me["plan_until"] - (end + 3 * 86400)) < 1
    # The checkout event then lands: provisional 35 days must not shorten the real period end.
    # (It resets the period so the first month lands at once; balance is already 300.)
    event = {"id": "evt_c1", "type": "checkout.session.completed",
             "data": {"object": {"id": "cs_1", "payment_status": "paid", "customer": "cus_1", "subscription": "sub_1",
                                 "metadata": {"user_id": user["id"], "plan": "yearly"}}}}
    assert _post_event(plan_client, event).status_code == 200
    # Renewal: period end moves out, found by customer id only (no metadata).
    end2 = now + 60 * 86400
    sub2 = {"id": "sub_1", "customer": "cus_1", "status": "active", "items": {"data": [{"current_period_end": end2}]}}
    r = _post_event(plan_client, {"id": "evt_s2", "type": "customer.subscription.updated", "data": {"object": sub2}})
    assert abs(r.json()["plan_until"] - (end2 + 3 * 86400)) < 1
    # invoice.paid extends from the line period when later, never shortens.
    inv = {"id": "in_1", "customer": "cus_1", "subscription": "sub_1", "lines": {"data": [{"period": {"end": now + 90 * 86400}}]}}
    r = _post_event(plan_client, {"id": "evt_i1", "type": "invoice.paid", "data": {"object": inv}})
    assert abs(r.json()["plan_until"] - (now + 93 * 86400)) < 1
    inv_old = {"id": "in_0", "customer": "cus_1", "lines": {"data": [{"period": {"end": now + 10 * 86400}}]}}
    r = _post_event(plan_client, {"id": "evt_i0", "type": "invoice.paid", "data": {"object": inv_old}})
    assert r.json()["plan_until"] is None
    assert abs(plan_client.get("/api/me").json()["user"]["plan_until"] - (now + 93 * 86400)) < 1
    # Duplicate delivery is a no-op.
    assert _post_event(plan_client, {"id": "evt_s2", "type": "customer.subscription.updated", "data": {"object": sub2}}).json()["duplicate"] is True
    # Past due: allowance stops now, the name stays for display.
    sub3 = dict(sub2, status="past_due")
    _post_event(plan_client, {"id": "evt_s3", "type": "customer.subscription.updated", "data": {"object": sub3}})
    me = plan_client.get("/api/me").json()["user"]
    assert me["plan"] == "yearly" and me["plan_active"] is False and me["plan_until"] <= time.time()
    # Reactivated, then deleted.
    _post_event(plan_client, {"id": "evt_s4", "type": "customer.subscription.updated", "data": {"object": sub2}})
    assert plan_client.get("/api/me").json()["user"]["plan_active"] is True
    _post_event(plan_client, {"id": "evt_s5", "type": "customer.subscription.deleted", "data": {"object": dict(sub2, status="canceled")}})
    me = plan_client.get("/api/me").json()["user"]
    assert me["plan_active"] is False and me["plan"] == "yearly"
    # An event for nobody we know is recorded and ignored, not 500.
    r = _post_event(plan_client, {"id": "evt_s6", "type": "customer.subscription.updated", "data": {"object": {"id": "sub_x", "customer": "cus_x", "status": "active"}}})
    assert r.status_code == 200 and r.json()["ignored"] == "unknown_user"
    # The portal opens for the stored customer.
    monkeypatch.setattr(stripe_client, "create_portal_session", lambda key, cust, ret: {"url": "https://billing.stripe.com/p/" + cust})
    assert plan_client.post("/api/billing/portal").json()["url"].endswith("cus_1")


def test_402_body_names_plans_and_words(plan_client):
    _login(plan_client)
    r = plan_client.post("/api/humanize/llm", json={"text": " ".join(["w"] * 3000)})
    assert r.status_code == 402
    b = r.json()
    assert b["needed"] == 10 and b["balance"] == 3 and b["words"] == 3000
    assert b["words_needed"] == 3000 and b["words_left"] == 900 and b["words_per_credit"] == 300
    assert b["plans"] == "/api/billing/plans" and b["packs"] == "/api/billing/packs"


def test_admin_plan_cli(tmp_path, monkeypatch, capsys):
    from humanizer.billing.__main__ import main

    for k, v in {
        "LONGHAND_PAYWALL": "1", "LONGHAND_SECRET": SECRET, "LONGHAND_DB": str(tmp_path / "b.db"),
        "STRIPE_PRICES": "", "STRIPE_PLANS": PLANS, "LONGHAND_EMAIL_BLOCKLIST": "",
    }.items():
        monkeypatch.setenv(k, v)
    assert main(["admin", "plan", "sub@example.com", "monthly", "--until", "2031-01-31"]) == 0
    out = capsys.readouterr().out
    assert "plan=monthly allowance=300" in out and "2031-01-31" in out and "balance=300" in out
    assert main(["admin", "plan", "sub@example.com", "lifetime", "--lifetime"]) == 0
    assert "until=lifetime" in capsys.readouterr().out
    assert main(["admin", "plan", "sub@example.com", "custom"]) == 2
    assert main(["admin", "plan", "sub@example.com", "custom", "--allowance", "50", "--lifetime"]) == 0
    assert main(["admin", "stats"]) == 0
    out = capsys.readouterr().out
    assert "credits_allowance" in out and "allowance" in out and "subscribers" in out
    assert main(["admin", "plan", "sub@example.com", "--off"]) == 0
    assert "plan=off" in capsys.readouterr().out
    store = Store(tmp_path / "b.db")
    assert store.user_by_email("sub@example.com")["plan"] == ""


# ------------------------------------------------ the rh_session bridge


def _site_app(cfg, tmp_path):
    """Stub engine routes plus the site's password accounts, under the paywall."""
    app = FastAPI()

    @app.post("/api/humanize/llm")
    def llm(body: dict):
        return {"original": body.get("text", ""), "humanized": body.get("text", "") + " rewritten"}

    @app.post("/api/humanize")
    def rule(body: dict):
        return {"ok": True}

    auth_mod.register_auth(app, auth_mod.AuthStore(tmp_path / "auth.sqlite"), None)
    return install(app, config=cfg, store=Store(cfg.db_path))


def test_site_signup_is_bridged_and_charged(tmp_path):
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="2", LONGHAND_GUESTS="0")
    app = _site_app(cfg, tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert r.status_code == 401
        assert r.json()["error"] == "sign_in_required" and r.json()["signin_url"] == "/signin"
        # Free routes stay free for anonymous callers; the site's blanket gate is gone.
        assert c.post("/api/humanize", json={"text": "x"}).status_code == 200
        r = c.post("/api/auth/signup", json={"email": "Site@Example.com", "password": "longenough"})
        assert r.status_code == 201 and "rh_session" in c.cookies
        me = c.get("/api/me").json()["user"]
        assert me["email"] == "site@example.com" and me["credits"] == 2 and me["plan"] == ""
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "1"
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}).status_code == 200
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert r.status_code == 402
        b = r.json()
        assert b["error"] == "insufficient_credits" and b["words_left"] == 0 and b["words_needed"] == 300 and b["plans"] == "/api/billing/plans"
        # Signing out drops the bridge; signing back in finds the same billing row.
        c.post("/api/auth/signout")
        assert c.get("/api/me").json()["user"] is None
        c.post("/api/auth/signin", json={"email": "site@example.com", "password": "longenough"})
        assert c.get("/api/me").json()["user"]["credits"] == 0
    store = Store(cfg.db_path)
    u = store.user_by_email("site@example.com")
    assert u["auth_user_id"] and store.user_by_auth_id(u["auth_user_id"])["id"] == u["id"]
    assert [e["kind"] for e in store.ledger(u["id"])] == ["charge", "charge", "signup"]


def test_guest_spends_the_address_allowance_then_is_sent_to_sign_up(tmp_path):
    """The October rule: free words without an account, limited by address,
    then 402 with `guest: true` and the sign-up URL."""
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="2")
    app = _site_app(cfg, tmp_path)
    with TestClient(app) as c:
        # Before anything is spent, /api/me shows the full grant without creating a row.
        me = c.get("/api/me").json()
        assert me["guest"] is True and me["user"]["guest"] is True and me["user"]["words_left"] == 600
        assert Store(cfg.db_path).user_by_email("guest:testclient") is None
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert r.status_code == 200 and r.headers["x-longhand-credits-charged"] == "1"
        assert c.get("/api/me").json()["user"]["words_left"] == 300
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}).status_code == 200
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert r.status_code == 402
        b = r.json()
        assert b["error"] == "insufficient_credits" and b["guest"] is True
        assert b["signup_url"] == "/signup?next=/pricing&error=free_used" and b["words_left"] == 0
        assert "Create an account" in b["detail"]
        # Free routes never needed anyone.
        assert c.post("/api/humanize", json={"text": "x"}).status_code == 200
    store = Store(cfg.db_path)
    g = store.user_by_email("guest:testclient")
    assert g is not None and g["credits"] == 0
    assert [e["kind"] for e in store.ledger(g["id"])] == ["charge", "charge", "signup"]


def test_account_created_from_a_used_address_inherits_what_the_guest_had_left(tmp_path):
    """One free allowance per address: the guest's remainder moves to the new
    account and the guest row is emptied, so nothing is spent twice."""
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="3")
    app = _site_app(cfg, tmp_path)
    with TestClient(app) as c:
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}).status_code == 200  # guest: 3 -> 2
        r = c.post("/api/auth/signup", json={"email": "new@example.com", "password": "longenough"})
        assert r.status_code == 201
        me = c.get("/api/me").json()["user"]
        assert me["guest"] is False and me["email"] == "new@example.com" and me["credits"] == 2
    store = Store(cfg.db_path)
    g = store.user_by_email("guest:testclient")
    assert g["credits"] == 0
    assert [e["kind"] for e in store.ledger(g["id"])] == ["transfer", "charge", "signup"]
    u = store.user_by_email("new@example.com")
    assert [e["kind"] for e in store.ledger(u["id"])] == ["signup"] and store.ledger(u["id"])[0]["delta"] == 2


def test_account_from_an_exhausted_address_starts_at_zero(tmp_path):
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="1")
    app = _site_app(cfg, tmp_path)
    with TestClient(app) as c:
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}).status_code == 200
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}).status_code == 402
        c.post("/api/auth/signup", json={"email": "late@example.com", "password": "longenough"})
        assert c.get("/api/me").json()["user"]["credits"] == 0
        r = c.post("/api/humanize/llm", json={"text": TEXT_300})
        assert r.status_code == 402 and r.json()["guest"] is False and r.json()["signup_url"] is None


def test_guests_are_separate_by_address(tmp_path):
    """Behind the proxy the address is X-Real-IP; each one has its own allowance."""
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="1", LONGHAND_TRUST_PROXY="1")
    app = _site_app(cfg, tmp_path)
    with TestClient(app) as c:
        a = {"X-Real-IP": "203.0.113.5"}
        b = {"X-Real-IP": "198.51.100.9"}
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}, headers=a).status_code == 200
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}, headers=a).status_code == 402
        assert c.post("/api/humanize/llm", json={"text": TEXT_300}, headers=b).status_code == 200
        assert c.get("/api/me", headers=a).json()["user"]["words_left"] == 0
        assert c.get("/api/me", headers=b).json()["user"]["words_left"] == 0
    store = Store(cfg.db_path)
    assert store.user_by_email("guest:203.0.113.5")["credits"] == 0
    assert store.user_by_email("guest:198.51.100.9")["credits"] == 0
    # Guest rows never count toward the per-address sign-up cap.
    assert store.signups_from("203.0.113.5", 86400.0) == 0


def test_site_bridge_mirrors_master_and_keeps_magic_link_credits(tmp_path):
    cfg = _env(tmp_path, LONGHAND_FREE_CREDITS="3")
    app = _site_app(cfg, tmp_path)
    auth_store = app.state.auth_store
    auth_store.set_master("boss@example.com", "bosspassword")
    with TestClient(app) as c:
        # A magic-link account that later signs in by password keeps its credits (one row by email).
        _login(c)  # a@example.com, 3 credits via longhand_session
        c.post("/api/humanize/llm", json={"text": TEXT_300})
        c.post("/api/auth/logout")
        c.post("/api/auth/signup", json={"email": "a@example.com", "password": "longenough"})
        assert c.get("/api/me").json()["user"]["credits"] == 2
        c.post("/api/auth/signout")
        c.post("/api/auth/signin", json={"email": "boss@example.com", "password": "bosspassword"})
        me = c.get("/api/me").json()["user"]
        assert me["is_admin"] is True and me["credits"] == 3


def test_site_bridge_applies_signup_limits(tmp_path):
    cfg = _env(tmp_path, LONGHAND_SIGNUPS_PER_IP_PER_DAY="1", LONGHAND_FREE_CREDITS="3")
    app = _site_app(cfg, tmp_path)
    with TestClient(app) as c:
        c.post("/api/auth/signup", json={"email": "one@example.com", "password": "longenough"})
        assert c.get("/api/me").json()["user"]["credits"] == 3
        c.post("/api/auth/signout")
        # Second account from the same address: created, usable, but no free credits.
        c.post("/api/auth/signup", json={"email": "two@example.com", "password": "longenough"})
        assert c.get("/api/me").json()["user"]["credits"] == 0
        assert c.post("/api/humanize/llm", json={"text": "hi"}).status_code == 402
        c.post("/api/auth/signout")
        c.post("/api/auth/signup", json={"email": "x@mailinator.com", "password": "longenough"})
        assert c.get("/api/me").json()["user"]["credits"] == 0


def test_real_app_site_signup_bridges(tmp_path):
    """End to end on the production app: password sign-up, then the paywall
    sees the account, charges it and refunds a refused request."""
    cfg = _env(tmp_path)
    with TestClient(_real_app(tmp_path, cfg)) as c:
        r = c.post("/api/auth/signup", json={"email": "real@example.com", "password": "longenough"})
        assert r.status_code == 201
        assert c.get("/api/auth/me").json()["signed_in"] is True
        me = c.get("/api/me").json()["user"]
        assert me["credits"] == 3 and me["words_left"] == 900
        r = c.post("/api/humanize/llm", json={"text": TEXT_300, "style": "bogus"})
        assert r.status_code == 400 and r.headers["x-longhand-credits-charged"] == "0"
        assert c.get("/api/me").json()["user"]["credits"] == 3
        assert c.get("/api/billing/plans").status_code == 200
