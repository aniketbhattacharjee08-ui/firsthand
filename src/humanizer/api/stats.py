"""First-party usage statistics: page views and the funnel, without cookies.

Added 2026-10-14 because the site had no way to answer the only growth
questions that matter: how many people land, how many run a rewrite, how
many hit the paywall, how many sign up, how many pay. Nothing third-party:
one small script (`web/stats.js`) posts a page view and a few named events
to `POST /api/stats/event`; the server records its own events (a finished
run, a 402, a sign-up, a purchase) through `record_event`. Everything lands
in one SQLite file beside the auth database (`HUMANIZER_STATS_DB`, else
`stats.sqlite`), and `GET /api/stats/summary` reads it back for the owner.

What is kept per event: the time, the event name, the page path, the
referrer's host, the UTM source/medium/campaign, a visitor hash, the user id
when the server knows it, and a small JSON bag of properties (a word count,
a verdict). Never the text, never the full referrer URL, never the address.

The visitor hash is Plausible's scheme: sha256(daily salt | client address |
user agent), truncated. The salt is random per UTC day and is deleted with
the day, so the hash cannot be reversed or joined across days; it exists
only to count unique visitors within a day. A request with `DNT: 1` or
`Sec-GPC: 1` is counted without a visitor hash at all. No cookie is set, so
the privacy page's "no analytics cookies" stays true.

The summary route answers 404 unless a token is configured
(`HUMANIZER_STATS_TOKEN`, else `LONGHAND_ADMIN_TOKEN`) and presented in
`X-Admin-Token`; `humanizer stats` prints the same summary from the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("humanizer.api.stats")

STATS_DB_ENV = "HUMANIZER_STATS_DB"
STATS_ENABLED_ENV = "HUMANIZER_STATS"
STATS_TOKEN_ENV = "HUMANIZER_STATS_TOKEN"
ADMIN_TOKEN_ENV = "LONGHAND_ADMIN_TOKEN"
ORIGIN_SECRET_ENV = "LONGHAND_ORIGIN_SECRET"
TRUST_PROXY_ENV = "LONGHAND_TRUST_PROXY"

#: Events the page may post. Anything else is dropped with a 204, so a stale
#: script can never grow the schema. Server-side events (`run`, `paywall`,
#: `signup`, `purchase`) are not in this set on purpose: a visitor must not
#: be able to forge a sign-up.
CLIENT_EVENTS = frozenset(
    ("pageview", "check", "run_started", "run_finished", "use", "copy", "download", "upload", "handoff", "plans", "signup_view")
)
SERVER_EVENTS = frozenset(("run", "paywall", "signup", "signin", "purchase"))

#: Field size caps for the beacon body. The limits are generous for real
#: values and tiny for abuse.
MAX_PATH = 200
MAX_REF = 120
MAX_TAG = 80
MAX_PROPS_BYTES = 1024
#: Rows older than this are pruned on start-up and once a day after.
KEEP_DAYS = 400


def stats_enabled_from_env() -> bool:
    """`HUMANIZER_STATS` as a bool; unset is on."""
    return os.environ.get(STATS_ENABLED_ENV, "1").strip().lower() not in ("0", "false", "no", "off", "")


def default_stats_db_path() -> Path:
    """`HUMANIZER_STATS_DB`, else `stats.sqlite` beside the auth database."""
    env = os.environ.get(STATS_DB_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    from . import auth as auth_mod

    return auth_mod.default_auth_db_path().with_name("stats.sqlite")


def summary_token_from_env() -> str:
    return (os.environ.get(STATS_TOKEN_ENV) or os.environ.get(ADMIN_TOKEN_ENV) or "").strip()


def _day(t: Optional[float] = None) -> str:
    return datetime.fromtimestamp(t if t is not None else time.time(), tz=timezone.utc).strftime("%Y-%m-%d")


def client_ip(request: Request) -> str:
    """The visitor's address, trusting the proxy headers only from the edge.

    Mirrors `billing.gate.client_ip`: with `LONGHAND_ORIGIN_SECRET` set, the
    forwarded headers are believed only on requests that carry the secret;
    otherwise `LONGHAND_TRUST_PROXY` decides. Both unset: the socket peer.
    """
    headers = request.headers
    secret = os.environ.get(ORIGIN_SECRET_ENV, "").strip()
    trust = os.environ.get(TRUST_PROXY_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    if secret:
        trust = secrets.compare_digest(headers.get("x-origin-secret", ""), secret)
    if trust:
        real = headers.get("x-real-ip", "").strip()
        if real:
            return real
        fwd = headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    client = request.client
    return client.host if client else ""


def _host_of(value: str) -> str:
    """The host of a referrer URL (or a bare host), lower-cased, without www."""
    v = (value or "").strip()
    if not v:
        return ""
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].split("@")[-1]
    v = v.lower()
    if v.startswith("www."):
        v = v[4:]
    return v[:MAX_REF]


class StatsStore:
    """The events table and the daily salts, in one SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY,
              t REAL NOT NULL,
              name TEXT NOT NULL,
              path TEXT NOT NULL DEFAULT '',
              ref TEXT NOT NULL DEFAULT '',
              source TEXT NOT NULL DEFAULT '',
              medium TEXT NOT NULL DEFAULT '',
              campaign TEXT NOT NULL DEFAULT '',
              visitor TEXT NOT NULL DEFAULT '',
              user TEXT NOT NULL DEFAULT '',
              props TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS events_t ON events (t);
            CREATE INDEX IF NOT EXISTS events_name_t ON events (name, t);
            CREATE TABLE IF NOT EXISTS salts (
              day TEXT PRIMARY KEY,
              salt TEXT NOT NULL
            );
            """
        )
        self._last_prune = 0.0
        self.prune()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- visitors

    def _salt(self, day: str) -> str:
        row = self._conn.execute("SELECT salt FROM salts WHERE day = ?", (day,)).fetchone()
        if row is not None:
            return str(row["salt"])
        salt = secrets.token_hex(16)
        self._conn.execute("INSERT OR IGNORE INTO salts (day, salt) VALUES (?, ?)", (day, salt))
        self._conn.commit()
        row = self._conn.execute("SELECT salt FROM salts WHERE day = ?", (day,)).fetchone()
        return str(row["salt"]) if row is not None else salt

    def visitor_id(self, ip: str, user_agent: str, t: Optional[float] = None) -> str:
        """A per-day hash of address and user agent; empty when both are empty."""
        if not ip and not user_agent:
            return ""
        with self._lock:
            salt = self._salt(_day(t))
        digest = hashlib.sha256(("%s|%s|%s" % (salt, ip, user_agent)).encode("utf-8")).hexdigest()
        return digest[:16]

    # -- writing

    def record(
        self,
        name: str,
        *,
        path: str = "",
        ref: str = "",
        source: str = "",
        medium: str = "",
        campaign: str = "",
        visitor: str = "",
        user: str = "",
        props: Optional[Dict[str, Any]] = None,
        t: Optional[float] = None,
    ) -> None:
        now = time.time() if t is None else float(t)
        bag = json.dumps(props or {}, separators=(",", ":"), sort_keys=True, default=str)
        if len(bag) > MAX_PROPS_BYTES:
            bag = "{}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (t, name, path, ref, source, medium, campaign, visitor, user, props)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    round(now, 3),
                    str(name)[:MAX_TAG],
                    str(path or "")[:MAX_PATH],
                    _host_of(ref),
                    str(source or "")[:MAX_TAG].lower(),
                    str(medium or "")[:MAX_TAG].lower(),
                    str(campaign or "")[:MAX_TAG].lower(),
                    str(visitor or "")[:32],
                    str(user or "")[:64],
                    bag,
                ),
            )
            self._conn.commit()
        if now - self._last_prune > 86400:
            self.prune(now=now)

    def prune(self, keep_days: int = KEEP_DAYS, now: Optional[float] = None) -> int:
        """Drop events older than `keep_days` and every salt but today's and yesterday's."""
        now = time.time() if now is None else now
        cutoff = now - keep_days * 86400
        keep = {_day(now), _day(now - 86400)}
        with self._lock:
            cur = self._conn.execute("DELETE FROM events WHERE t < ?", (cutoff,))
            self._conn.execute(
                "DELETE FROM salts WHERE day NOT IN (%s)" % ",".join("?" for _ in keep), tuple(sorted(keep))
            )
            self._conn.commit()
            self._last_prune = now
        return int(cur.rowcount or 0)

    # -- reading

    def rows(self, since: float, until: Optional[float] = None) -> List[sqlite3.Row]:
        until = time.time() if until is None else until
        with self._lock:
            return list(
                self._conn.execute(
                    "SELECT t, name, path, ref, source, medium, campaign, visitor, user, props"
                    " FROM events WHERE t >= ? AND t < ? ORDER BY t",
                    (since, until),
                )
            )

    def summary(self, days: int = 30, now: Optional[float] = None) -> Dict[str, Any]:
        """Counts an owner reads once a day: traffic, where it came from, the funnel.

        `funnel` counts *visitors* (daily hashes, so a person over several
        days is several) for the page-side steps and *events* for the
        server-side ones. The server's `run` rows are the authoritative count
        of finished rewrites: one row per charged request that returned 2xx.
        """
        now = time.time() if now is None else now
        days = max(1, min(int(days), KEEP_DAYS))
        since = now - days * 86400
        rows = self.rows(since, now)

        by_day: Dict[str, Dict[str, Any]] = {}
        events: Counter = Counter()
        paths: Counter = Counter()
        refs: Counter = Counter()
        sources: Counter = Counter()
        campaigns: Counter = Counter()
        visitors_all: set = set()
        visitors_by: Dict[str, set] = {k: set() for k in ("pageview", "check", "run_started", "run_finished", "use", "download", "copy")}
        runs_ok = runs_failed = runs_busy = runs_unchanged = 0
        run_words = 0
        run_seconds: List[float] = []
        paywall = paywall_guest = 0
        signups: Counter = Counter()
        purchases = 0
        purchase_credits = 0
        flipped = finished = 0

        for r in rows:
            name = str(r["name"])
            day = _day(float(r["t"]))
            d = by_day.setdefault(day, {"day": day, "pageviews": 0, "visitors": set(), "runs": 0, "signups": 0, "purchases": 0})
            events[name] += 1
            vis = str(r["visitor"] or "")
            if vis:
                visitors_all.add(vis)
                if name in visitors_by:
                    visitors_by[name].add(vis)
            try:
                props = json.loads(r["props"] or "{}")
            except ValueError:
                props = {}
            if name == "pageview":
                d["pageviews"] += 1
                if vis:
                    d["visitors"].add(vis)
                paths[str(r["path"] or "/")] += 1
                if r["ref"]:
                    refs[str(r["ref"])] += 1
                if r["source"]:
                    sources[str(r["source"])] += 1
                if r["campaign"]:
                    campaigns[str(r["campaign"])] += 1
            elif name == "run":
                status = int(props.get("status") or 0)
                if props.get("busy"):
                    runs_busy += 1
                elif 200 <= status < 400:
                    runs_ok += 1
                    d["runs"] += 1
                    run_words += int(props.get("words") or 0)
                    if props.get("seconds") is not None:
                        run_seconds.append(float(props["seconds"]))
                    if props.get("unchanged"):
                        runs_unchanged += 1
                else:
                    runs_failed += 1
            elif name == "run_finished":
                finished += 1
                if props.get("flipped"):
                    flipped += 1
            elif name == "paywall":
                paywall += 1
                if props.get("guest"):
                    paywall_guest += 1
            elif name == "signup":
                d["signups"] += 1
                signups[str(props.get("method") or "password")] += 1
            elif name == "purchase":
                purchases += 1
                d["purchases"] += 1
                purchase_credits += int(props.get("credits") or 0)

        series = []
        for day in sorted(by_day):
            d = by_day[day]
            series.append(
                {"day": day, "pageviews": d["pageviews"], "visitors": len(d["visitors"]), "runs": d["runs"], "signups": d["signups"], "purchases": d["purchases"]}
            )
        run_seconds.sort()
        median = run_seconds[len(run_seconds) // 2] if run_seconds else None

        def top(c: Counter, n: int = 15) -> List[Dict[str, Any]]:
            return [{"key": k, "n": v} for k, v in c.most_common(n)]

        return {
            "days": days,
            "since": datetime.fromtimestamp(since, tz=timezone.utc).isoformat(timespec="seconds"),
            "until": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds"),
            "pageviews": events["pageview"],
            "visitors": len(visitors_all),
            "events": dict(sorted(events.items())),
            "by_day": series,
            "top_pages": top(paths),
            "top_referrers": top(refs),
            "top_sources": top(sources),
            "top_campaigns": top(campaigns),
            "funnel": {
                "visitors": len(visitors_by["pageview"]),
                "checked": len(visitors_by["check"]),
                "started_run": len(visitors_by["run_started"]),
                "finished_run": len(visitors_by["run_finished"]),
                "kept_result": len(visitors_by["use"] | visitors_by["download"] | visitors_by["copy"]),
                "paywall_hits": paywall,
                "paywall_hits_guest": paywall_guest,
                "signups": sum(signups.values()),
                "signups_by_method": dict(signups),
                "purchases": purchases,
                "purchase_credits": purchase_credits,
            },
            "runs": {
                "finished": runs_ok,
                "failed": runs_failed,
                "busy": runs_busy,
                "unchanged": runs_unchanged,
                "words": run_words,
                "median_seconds": median,
                "page_reported_finished": finished,
                "page_reported_flipped": flipped,
            },
        }


# ------------------------------------------------------------------ helpers


def record_event(app: Any, name: str, **kw: Any) -> None:
    """Record a server-side event on `app.state.stats`, if stats are on.

    Never raises: a statistics failure must not fail the request it rides on.
    """
    store = getattr(getattr(app, "state", None), "stats", None)
    if store is None:
        return
    try:
        store.record(name, **kw)
    except Exception:  # noqa: BLE001 - statistics are best effort
        log.exception("stats: could not record %s", name)


def record_event_from_scope(scope: Dict[str, Any], name: str, **kw: Any) -> None:
    """The ASGI-middleware form of `record_event` (the paywall has no `app` handle)."""
    record_event(scope.get("app"), name, **kw)


# ------------------------------------------------------------------- routes


class EventBody(BaseModel):
    name: str = Field(default="", max_length=MAX_TAG)
    path: str = Field(default="", max_length=MAX_PATH)
    ref: str = Field(default="", max_length=600)
    source: str = Field(default="", max_length=MAX_TAG)
    medium: str = Field(default="", max_length=MAX_TAG)
    campaign: str = Field(default="", max_length=MAX_TAG)
    props: Optional[Dict[str, Any]] = None


def _do_not_track(request: Request) -> bool:
    h = request.headers
    return h.get("dnt", "").strip() == "1" or h.get("sec-gpc", "").strip() == "1"


def register_stats(app: FastAPI, store: StatsStore, token: Optional[str] = None) -> FastAPI:
    """Add the beacon and the summary routes; put the store on `app.state.stats`."""
    if getattr(app.state, "stats_installed", False):
        return app
    app.state.stats = store
    app.state.stats_installed = True
    summary_token = summary_token_from_env() if token is None else token

    @app.post("/api/stats/event", status_code=204, include_in_schema=False)
    def stats_event(request: Request, body: EventBody) -> Response:
        """The page's beacon. Always 204: a visitor's browser is never told
        whether an event was kept."""
        name = body.name.strip().lower()
        if name not in CLIENT_EVENTS:
            return Response(status_code=204)
        visitor = ""
        if not _do_not_track(request):
            visitor = store.visitor_id(client_ip(request), request.headers.get("user-agent", ""))
        props: Dict[str, Any] = {}
        if isinstance(body.props, dict):
            # Scalars only, a handful of them, short keys: a counter's bag.
            for k, v in list(body.props.items())[:12]:
                if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= MAX_TAG):
                    props[str(k)[:40]] = v
        try:
            store.record(
                name,
                path=body.path or "",
                ref=body.ref,
                source=body.source,
                medium=body.medium,
                campaign=body.campaign,
                visitor=visitor,
                props=props,
            )
        except Exception:  # noqa: BLE001
            log.exception("stats: beacon not recorded")
        return Response(status_code=204)

    @app.get("/api/stats/summary", include_in_schema=False)
    def stats_summary(request: Request, days: int = 30) -> JSONResponse:
        """The owner's numbers. 404 unless a token is configured and presented."""
        if not summary_token:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        given = request.headers.get("x-admin-token", "") or request.query_params.get("token", "")
        if not given or not secrets.compare_digest(given, summary_token):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        payload = store.summary(days=days)
        return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})

    return app


def format_summary(s: Dict[str, Any]) -> str:
    """The summary as plain text, for `humanizer stats`."""
    f = s["funnel"]
    r = s["runs"]
    lines: List[str] = []
    lines.append("Vervly, last %d days (%s to %s)" % (s["days"], s["since"][:10], s["until"][:10]))
    lines.append("")
    lines.append("  page views        %6d" % s["pageviews"])
    lines.append("  visitors          %6d   (daily hashes, no cookies)" % s["visitors"])
    lines.append("")
    lines.append("  funnel (visitors)")
    for key, label in (
        ("visitors", "landed"),
        ("checked", "ran the free check"),
        ("started_run", "started a rewrite"),
        ("finished_run", "saw a rewrite finish"),
        ("kept_result", "kept, copied or downloaded"),
    ):
        lines.append("    %-28s %6d" % (label, f[key]))
    lines.append("  funnel (events)")
    lines.append("    %-28s %6d   (%d from guests)" % ("hit the paywall", f["paywall_hits"], f["paywall_hits_guest"]))
    lines.append("    %-28s %6d   %s" % ("signed up", f["signups"], json.dumps(f["signups_by_method"])))
    lines.append("    %-28s %6d   (%d credits)" % ("paid", f["purchases"], f["purchase_credits"]))
    lines.append("")
    lines.append("  rewrites finished %6d   failed %d, busy %d, unchanged %d, %d words" % (r["finished"], r["failed"], r["busy"], r["unchanged"], r["words"]))
    if r["median_seconds"] is not None:
        lines.append("  median seconds    %6.1f" % r["median_seconds"])
    if r["page_reported_finished"]:
        lines.append("  verdict flipped   %6d of %d the page reported" % (r["page_reported_flipped"], r["page_reported_finished"]))

    def block(title: str, items: Iterable[Dict[str, Any]]) -> None:
        items = list(items)
        if not items:
            return
        lines.append("")
        lines.append("  " + title)
        for it in items:
            lines.append("    %-40s %6d" % (it["key"][:40], it["n"]))

    block("pages", s["top_pages"])
    block("referrers", s["top_referrers"])
    block("utm sources", s["top_sources"])
    block("campaigns", s["top_campaigns"])
    if s["by_day"]:
        lines.append("")
        lines.append("  day         views  visitors  runs  signups  paid")
        for d in s["by_day"]:
            lines.append("  %s %6d  %8d  %4d  %7d  %4d" % (d["day"], d["pageviews"], d["visitors"], d["runs"], d["signups"], d["purchases"]))
    return "\n".join(lines)
