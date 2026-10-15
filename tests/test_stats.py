"""First-party statistics (api.stats): the store, the beacon, the summary.

The TestClient drives the whole app in-process so the beacon's path through
the router (behind the auth block and the StaticFiles mount) is the real one.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api import stats as stats_mod  # noqa: E402
from humanizer.api.server import create_app  # noqa: E402

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"


def _web(tmp_path: Path) -> Path:
    web = tmp_path / "web"
    web.mkdir(exist_ok=True)
    for name in ("index.html", "how.html", "pricing.html", "auth.html", "check.html", "voice.html"):
        (web / name).write_text("<h1>%s</h1>" % name[:-5])
    return web


def _app(tmp_path: Path, **kw):
    return create_app(
        reference_dir=REFERENCE_DIR,
        web_dir=_web(tmp_path),
        auth=True,
        auth_db=tmp_path / "auth.sqlite",
        stats_db=tmp_path / "stats.sqlite",
        **kw,
    )


# ------------------------------------------------------------------- store


class TestStore:
    def test_visitor_hash_is_stable_within_a_day_and_differs_across_days(self, tmp_path):
        s = stats_mod.StatsStore(tmp_path / "s.sqlite")
        now = time.time()
        a = s.visitor_id("1.2.3.4", "UA", t=now)
        b = s.visitor_id("1.2.3.4", "UA", t=now)
        c = s.visitor_id("1.2.3.4", "UA", t=now + 86400 * 3)
        d = s.visitor_id("1.2.3.5", "UA", t=now)
        assert a == b and len(a) == 16
        assert a != c and a != d
        assert s.visitor_id("", "") == ""

    def test_prune_drops_old_rows_and_old_salts(self, tmp_path):
        s = stats_mod.StatsStore(tmp_path / "s.sqlite")
        now = time.time()
        s.record("pageview", t=now - 500 * 86400)
        s.record("pageview", t=now)
        s.visitor_id("ip", "ua", t=now - 10 * 86400)
        s.visitor_id("ip", "ua", t=now)
        dropped = s.prune(now=now)
        assert dropped == 1
        assert s.summary(days=600, now=now)["pageviews"] == 1
        days = [r[0] for r in s._conn.execute("SELECT day FROM salts")]
        assert len(days) == 1

    def test_summary_counts_the_funnel(self, tmp_path):
        s = stats_mod.StatsStore(tmp_path / "s.sqlite")
        now = time.time()
        v1 = s.visitor_id("1.1.1.1", "ua", t=now)
        v2 = s.visitor_id("2.2.2.2", "ua", t=now)
        s.record("pageview", path="/", ref="https://www.reddit.com/r/college/x", source="reddit", visitor=v1, t=now)
        s.record("pageview", path="/check", visitor=v2, t=now)
        s.record("pageview", path="/", visitor=v1, t=now)
        s.record("check", path="/check", visitor=v2, props={"pct": 80}, t=now)
        s.record("run_started", visitor=v1, t=now)
        s.record("run_finished", visitor=v1, props={"flipped": True}, t=now)
        s.record("download", visitor=v1, t=now)
        s.record("run", user="u1", props={"rule": "charge", "status": 200, "words": 120, "seconds": 31.0}, t=now)
        s.record("run", user="u1", props={"rule": "charge", "status": 503, "busy": True}, t=now)
        s.record("run", user="u1", props={"rule": "resume", "status": 500, "error": True}, t=now)
        s.record("paywall", user="guest:1.1.1.1", props={"guest": True}, t=now)
        s.record("signup", user="u1", props={"method": "password"}, t=now)
        s.record("signup", user="u2", props={"method": "google"}, t=now)
        s.record("purchase", user="u1", props={"kind": "plan", "credits": 400}, t=now)
        out = s.summary(days=7, now=now + 1)
        assert out["pageviews"] == 3
        assert out["visitors"] == 2
        assert out["top_pages"][0] == {"key": "/", "n": 2}
        assert out["top_referrers"] == [{"key": "reddit.com", "n": 1}]
        assert out["top_sources"] == [{"key": "reddit", "n": 1}]
        f = out["funnel"]
        assert f["visitors"] == 2 and f["checked"] == 1 and f["started_run"] == 1 and f["finished_run"] == 1
        assert f["kept_result"] == 1
        assert f["paywall_hits"] == 1 and f["paywall_hits_guest"] == 1
        assert f["signups"] == 2 and f["signups_by_method"] == {"password": 1, "google": 1}
        assert f["purchases"] == 1 and f["purchase_credits"] == 400
        r = out["runs"]
        assert r["finished"] == 1 and r["busy"] == 1 and r["failed"] == 1 and r["words"] == 120
        assert r["median_seconds"] == 31.0
        assert r["page_reported_flipped"] == 1
        assert len(out["by_day"]) == 1 and out["by_day"][0]["runs"] == 1
        text = stats_mod.format_summary(out)
        assert "signed up" in text and "reddit.com" in text

    def test_props_too_large_are_dropped_not_stored(self, tmp_path):
        s = stats_mod.StatsStore(tmp_path / "s.sqlite")
        s.record("pageview", props={"x": "y" * 5000})
        row = s._conn.execute("SELECT props FROM events").fetchone()
        assert row["props"] == "{}"


# ------------------------------------------------------------------ routes


class TestRoutes:
    def test_beacon_records_a_pageview_without_a_cookie(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            r = c.post("/api/stats/event", json={"name": "pageview", "path": "/how", "ref": "https://news.ycombinator.com/item?id=1", "source": "HN"})
            assert r.status_code == 204
            assert "set-cookie" not in {k.lower() for k in r.headers}
            s = c.app.state.stats
            out = s.summary(days=1)
            assert out["pageviews"] == 1 and out["visitors"] == 1
            assert out["top_referrers"] == [{"key": "news.ycombinator.com", "n": 1}]
            assert out["top_sources"] == [{"key": "hn", "n": 1}]

    def test_beacon_ignores_unknown_and_server_only_names(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            for name in ("signup", "purchase", "run", "nonsense"):
                assert c.post("/api/stats/event", json={"name": name}).status_code == 204
            assert c.app.state.stats.summary(days=1)["events"] == {}

    def test_dnt_counts_the_view_without_a_visitor(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            c.post("/api/stats/event", json={"name": "pageview", "path": "/"}, headers={"DNT": "1"})
            out = c.app.state.stats.summary(days=1)
            assert out["pageviews"] == 1 and out["visitors"] == 0

    def test_beacon_keeps_only_small_scalar_props(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            c.post("/api/stats/event", json={"name": "check", "props": {"pct": 72, "band": "ai", "nested": {"a": 1}, "long": "z" * 500}})
            row = c.app.state.stats._conn.execute("SELECT props FROM events").fetchone()
            assert row["props"] == '{"band":"ai","pct":72}'

    def test_summary_is_404_without_a_token_or_with_the_wrong_one(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HUMANIZER_STATS_TOKEN", raising=False)
        monkeypatch.delenv("LONGHAND_ADMIN_TOKEN", raising=False)
        with TestClient(_app(tmp_path)) as c:
            assert c.get("/api/stats/summary").status_code == 404
        monkeypatch.setenv("HUMANIZER_STATS_TOKEN", "s3cret")
        with TestClient(_app(tmp_path)) as c:
            assert c.get("/api/stats/summary").status_code == 404
            assert c.get("/api/stats/summary", headers={"X-Admin-Token": "wrong"}).status_code == 404
            r = c.get("/api/stats/summary?days=7", headers={"X-Admin-Token": "s3cret"})
            assert r.status_code == 200 and r.json()["days"] == 7
            assert r.headers["cache-control"] == "no-store"

    def test_signup_records_a_server_event(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            r = c.post("/api/auth/signup", json={"email": "ada@example.com", "password": "correct horse", "name": "Ada"})
            assert r.status_code == 201
            out = c.app.state.stats.summary(days=1)
            assert out["funnel"]["signups"] == 1
            assert out["funnel"]["signups_by_method"] == {"password": 1}

    def test_stats_off_by_switch(self, tmp_path):
        with TestClient(_app(tmp_path, stats=False)) as c:
            assert getattr(c.app.state, "stats", None) is None
            # the route does not exist; the static mount answers 404
            assert c.post("/api/stats/event", json={"name": "pageview"}).status_code in (404, 405)

    def test_record_event_never_raises(self):
        class App:
            class state:
                stats = None
        stats_mod.record_event(App, "signup")
        stats_mod.record_event(None, "signup")
        stats_mod.record_event_from_scope({}, "run")


# ------------------------------------------------------------- site files


class TestSiteFiles:
    def test_robots_and_sitemap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HUMANIZER_PUBLIC_URL", "https://example.test/")
        with TestClient(_app(tmp_path)) as c:
            r = c.get("/robots.txt")
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
            assert "Disallow: /api/" in r.text and "Sitemap: https://example.test/sitemap.xml" in r.text
            s = c.get("/sitemap.xml")
            assert s.status_code == 200 and "application/xml" in s.headers["content-type"]
            assert "<loc>https://example.test/</loc>" in s.text
            assert "<loc>https://example.test/check</loc>" in s.text
            assert "<loc>https://example.test/how</loc>" in s.text

    def test_sitemap_lists_only_pages_on_disk(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            (Path(c.app.state.web_dir) / "voice.html").unlink()
            s = c.get("/sitemap.xml").text
            assert "/voice</loc>" not in s and "/pricing</loc>" in s

    def test_check_page_is_public(self, tmp_path):
        with TestClient(_app(tmp_path)) as c:
            r = c.get("/check")
            assert r.status_code == 200 and "<h1>check</h1>" in r.text
            assert r.headers["cache-control"] == "no-cache, must-revalidate"
