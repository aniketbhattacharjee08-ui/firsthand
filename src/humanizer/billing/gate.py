"""The paywall itself: one ASGI middleware that authenticates, charges and
polices the GPU.

Why a middleware and not a dependency on each route: the constraint was to add
the paywall without editing `humanizer.api.server`. Its routes are closures
inside `create_app`, so nothing can be injected into them, but every request
passes through the middleware stack first, and that is enough.

Per request under `/api/`:

1. **Rate limit** by client IP (`LONGHAND_RATE_LIMIT`), stricter on the login
   endpoints (`LONGHAND_AUTH_RATE_LIMIT`).
2. **Classify** (method, path) with `RULES`. Measurement and the rule-based
   rewrite are free: they cost milliseconds and are what a visitor tries
   before paying. The LLM and guided engines, and any request that would
   spend the operator's GPTZero key, need a signed-in user and are charged
   in credits by word count.
3. **Charge** atomically before the work runs, after applying any plan
   allowance that fell due this month (`Store.topup_if_due`). 401 without a
   session, 402 without enough credits, each with a body that names the fix.
4. **One GPU slot at a time** (`LONGHAND_MAX_INFLIGHT`). A request that
   cannot get a slot within `LONGHAND_QUEUE_WAIT_S` is answered
   `503 busy` with `Retry-After`, and refunded.
5. **Refund** when the server refuses (any 4xx or 5xx), when a parked job is
   never claimed within `LONGHAND_JOB_TTL_S`, and, if
   `LONGHAND_REFUND_UNCHANGED` is on, when the finished rewrite returned the
   whole document unchanged. The last one is detected from the response body
   (`humanized == original`, or every paragraph fell back), for JSON and for
   the SSE `result` frame alike.
6. **Log** one JSON line per gated request (`LONGHAND_LOG_DIR/requests.log`)
   with the user id, path, words, credits, status and seconds. Never the text.

Job routes need a word on their own. `POST /api/humanize/jobs` only parks a
body; the compute happens when `GET /api/humanize/stream?job=` connects. So
the POST is charged (rule `park`) and recorded, and the GET (rule `resume`)
is the one that takes the GPU slot. The GET needs no session: the job id is
random, single-use and was paid for. If the GET never comes, a sweep refunds
the park charge after the TTL.

Who the caller is comes from one of three places, in this order: a bearer
`lh_` API key, the paywall's own magic-link cookie (`longhand_session`), or
the shipped site's password session (`rh_session`, resolved through the
`humanizer.api.auth` store the server mounted at `app.state.auth_store`).
The last is bridged to a billing user by email on first sight, with the
same free-credit grant and the same per-IP and disposable-domain limits the
magic-link path applies, and the link is cached in `users.auth_user_id`.

With `LONGHAND_PAYWALL` off, only step 1 runs.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import hmac
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

from .config import BillingConfig
from .quota import RateLimiter, credits_for_words, effort_multiplier, word_count
from .store import Store

log = logging.getLogger("humanizer.billing.gate")
request_log = logging.getLogger("humanizer.billing.requests")

SESSION_COOKIE = "longhand_session"
#: The shipped site's cookie (`humanizer.api.auth.SESSION_COOKIE`). Named
#: here rather than imported so this module keeps no dependency on the
#: server package.
SITE_COOKIE = "rh_session"

#: Largest response body kept for the unchanged check. A result payload with
#: 16 candidates per paragraph is a few hundred KB; anything past this is
#: forwarded but not inspected.
CAPTURE_LIMIT = 8 * 1024 * 1024

#: (method, path regex, rule). First match wins.
#:   free    no auth
#:   charge  auth + credits by word count; takes a GPU slot
#:   park    auth + credits; records the job id from the response
#:   resume  takes a GPU slot; consumes the recorded job (no auth: paid id)
#:   detect  free unless body["detectors"] names "gptzero" (operator's key)
#:   admin   X-Admin-Token must equal LONGHAND_ADMIN_TOKEN, or an admin user
RULES: List[Tuple[str, "re.Pattern[str]", str]] = [
    ("POST", re.compile(r"^/api/humanize/(llm|stream)$"), "charge"),
    ("POST", re.compile(r"^/api/humanize/jobs$"), "park"),
    ("GET", re.compile(r"^/api/humanize/stream$"), "resume"),
    ("POST", re.compile(r"^/api/humanize/guided(/stream)?$"), "charge"),
    ("POST", re.compile(r"^/api/humanize/guided/jobs$"), "park"),
    ("GET", re.compile(r"^/api/humanize/guided/stream$"), "resume"),
    ("POST", re.compile(r"^/api/detect$"), "detect"),
    ("GET", re.compile(r"^/api/models$"), "admin"),
    ("POST", re.compile(r"^/api/models/release$"), "admin"),
]

_AUTH_PATHS = re.compile(r"^/api/auth/(request-link|verify)$")
_SSE_RESULT_RE = re.compile(rb"event: result\ndata: (.*?)\n\n", re.DOTALL)


def rule_for(method: str, path: str) -> str:
    for m, pattern, rule in RULES:
        if m == method and pattern.match(path):
            return rule
    return "free"


def client_ip(scope: Dict[str, Any], trust_proxy: bool) -> str:
    """The address to rate-limit on.

    With `trust_proxy`, `X-Real-IP` (which `deploy/Caddyfile` sets by
    replacement to the peer address) wins; failing that, the *rightmost*
    `X-Forwarded-For` element, which is the one the trusted proxy appended.
    The leftmost element is whatever the client sent and must not be used.
    """
    headers = _headers(scope)
    if trust_proxy:
        real = headers.get("x-real-ip")
        if real:
            return real.strip()
        fwd = headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[-1].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


def _headers(scope: Dict[str, Any]) -> Dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


def _cookies(headers: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in headers.get("cookie", "").split(";"):
        k, _, v = part.strip().partition("=")
        if k:
            out[k] = v
    return out


def authenticate(store: Store, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """The user for this request's bearer API key or magic-link cookie.

    The site cookie is bridged separately (`bridge_site_user`) because it
    needs the auth store and the config, which this helper never had.
    """
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        raw = auth[7:].strip()
        if raw.startswith("lh_"):
            return store.api_key_user(raw)
    sid = _cookies(headers).get(SESSION_COOKIE)
    if sid:
        return store.session_user(sid)
    return None


def bridge_site_user(
    store: Store, config: BillingConfig, auth_store: Any, headers: Dict[str, str], ip: str
) -> Optional[Dict[str, Any]]:
    """The billing user behind the site's `rh_session` cookie, or None.

    A site account meets billing on first sight: the row is found by the
    cached `auth_user_id`, else by email (so an account that already logged
    in by magic link keeps its credits), else created. The free-credit
    grant follows the magic-link rules: none for a disposable domain and
    none past the per-IP daily sign-up cap, but the account itself is still
    created so the person can buy. A `master` site account is an admin
    here; the flag is only ever raised, never lowered, so an operator grant
    made with the CLI survives.
    """
    token = _cookies(headers).get(SITE_COOKIE)
    if not token or auth_store is None:
        return None
    try:
        site = auth_store.user_for_session(token)
    except Exception:  # noqa: BLE001 - a broken auth db must not 500 every request
        log.exception("site session lookup failed")
        return None
    if not site:
        return None
    auth_id = str(site.get("id"))
    email = str(site.get("email") or "").strip().lower()
    if not email:
        return None
    user = store.user_by_auth_id(auth_id)
    if user is None:
        free = config.free_credits
        domain = email.rsplit("@", 1)[-1]
        if domain in config.blocked_domains:
            free = 0
        elif config.signups_per_ip_per_day > 0 and store.user_by_email(email) is None:
            if store.signups_from(ip, 86400.0) >= config.signups_per_ip_per_day:
                free = 0
        user, _created = store.create_or_get(email, free, ip=ip)
        store.link_auth_user(user["id"], auth_id)
        user = dict(user, auth_user_id=auth_id)
    if site.get("role") == "master" and not user.get("is_admin"):
        store.set_admin(user["id"], True)
        user = dict(user, is_admin=1)
    return dict(user, site_user=True, name=site.get("name"))


def looks_unchanged(content_type: str, body: bytes) -> Optional[bool]:
    """Whether a finished rewrite returned the document unchanged.

    None when the body carries no rewrite result (an error payload, a
    truncated capture, a stream that never reached `result`).
    """
    data: Any = None
    try:
        if "text/event-stream" in content_type:
            frames = _SSE_RESULT_RE.findall(body)
            if not frames:
                return None
            data = json.loads(frames[-1].decode("utf-8"))
        elif "application/json" in content_type:
            data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    original, humanized = data.get("original"), data.get("humanized")
    if isinstance(original, str) and isinstance(humanized, str):
        return original.strip() == humanized.strip()
    paragraphs = data.get("paragraphs")
    summary = data.get("summary") or {}
    if isinstance(paragraphs, list) and paragraphs and isinstance(summary, dict):
        unchanged = summary.get("paragraphs_unchanged")
        if isinstance(unchanged, int):
            return unchanged >= len(paragraphs)
    return None


_SSE_ERROR_RE = re.compile(rb"event: error\ndata: ")


def stream_failed(content_type: str, body: bytes) -> bool:
    """An SSE response that reported an error and never delivered a result.

    The server emits `event: error` inside an HTTP 200 once headers are out,
    so status alone cannot see a failed run. Such a run is refunded.
    """
    if "text/event-stream" not in content_type:
        return False
    return bool(_SSE_ERROR_RE.search(body)) and not _SSE_RESULT_RE.search(body)


class RequestLog:
    """Append-only JSON lines. Falls back to `logging` when no directory is set."""

    def __init__(self, directory: str) -> None:
        self.path: Optional[Path] = None
        if directory:
            d = Path(directory).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / "requests.log"
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        if self.path is None:
            request_log.info(line)
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


async def _json_response(
    send: Callable[..., Awaitable[None]],
    status: int,
    body: Dict[str, Any],
    extra_headers: Optional[List[Tuple[bytes, bytes]]] = None,
) -> None:
    raw = json.dumps(body).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(raw)).encode("ascii")),
    ] + (extra_headers or [])
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": raw})


class PaywallMiddleware:
    def __init__(self, app: Any, config: BillingConfig, store: Store) -> None:
        self.app = app
        self.config = config
        self.store = store
        self.api_limiter = RateLimiter(*config.api_rate)
        self.auth_limiter = RateLimiter(*config.auth_rate)
        self.request_log = RequestLog(config.log_dir)
        #: Parked, paid jobs awaiting their GET. job_id -> record.
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()
        # A threading semaphore, not an asyncio one: it is loop-agnostic (the
        # test suite drives one app from several event loops) and is waited
        # on from a worker thread so the event loop stays free.
        self._slots = threading.Semaphore(config.max_inflight)
        # Bound on requests waiting for a slot, and a private executor for the
        # blocking waits so the default threadpool stays free for the app.
        self._waiters = threading.Semaphore(config.max_waiting)
        self._wait_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_waiting, thread_name_prefix="gpu-slot-wait"
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _auth_store(scope: Dict[str, Any]) -> Any:
        """The site's `AuthStore`, when the server mounted one; else None."""
        app = scope.get("app")
        state = getattr(app, "state", None)
        return getattr(state, "auth_store", None) if state is not None else None

    @staticmethod
    def _sign_in_body(auth_store: Any) -> Dict[str, Any]:
        """The 401 body. The shipped site redirects on `sign_in_required`
        (what `humanizer.api.auth` answers), so that is the code whenever
        its accounts exist; a bare paywall keeps the magic-link code."""
        if auth_store is not None:
            return {
                "error": "sign_in_required",
                "detail": "Sign in at /signin, or POST /api/auth/signin, then retry.",
                "signin_url": "/signin",
                "login": "/api/auth/request-link",
            }
        return {
            "error": "login_required",
            "detail": "Sign in to use this engine. POST /api/auth/request-link with your email.",
            "login": "/api/auth/request-link",
        }

    def _refund(self, user_id: str, cost: int, why: str) -> int:
        return self.store.credit(user_id, cost, kind="refund", ref=why)

    def _refund_unchanged_allowed(self, user_id: str) -> bool:
        cap = self.config.unchanged_refunds_per_day
        if cap <= 0:
            return False
        return self.store.count_refunds(user_id, "unchanged", 86400.0) < cap

    def sweep_jobs(self, now: Optional[float] = None) -> int:
        """Refund parked jobs nobody claimed within the TTL. Returns the count."""
        now = time.time() if now is None else now
        expired: List[Tuple[str, Dict[str, Any]]] = []
        with self._jobs_lock:
            for job_id, rec in list(self.jobs.items()):
                if now - rec["created"] > self.config.job_ttl_s:
                    expired.append((job_id, self.jobs.pop(job_id)))
        for job_id, rec in expired:
            self._refund(rec["user_id"], rec["cost"], "%s job %s unclaimed" % (rec["path"], job_id))
            self.request_log.write(
                {
                    "t": round(now, 3),
                    "event": "job_expired",
                    "user": rec["user_id"],
                    "path": rec["path"],
                    "credits": rec["cost"],
                    "refunded": rec["cost"],
                }
            )
        return len(expired)

    # -- entry -------------------------------------------------------------

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[[], Awaitable[Dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        method: str = scope.get("method", "GET").upper()
        headers = _headers(scope)
        ip = client_ip(scope, self.config.trust_proxy)

        limiter = self.auth_limiter if _AUTH_PATHS.match(path) and method == "POST" else self.api_limiter
        allowed, retry = limiter.allow(ip)
        if not allowed:
            wait = max(1, int(retry) + 1)
            await _json_response(
                send,
                429,
                {
                    "error": "rate_limited",
                    "detail": "Too many requests from this address. Try again in %d seconds." % wait,
                    "retry_after_s": round(retry, 1),
                },
                [(b"retry-after", str(wait).encode("ascii"))],
            )
            return

        auth_store = self._auth_store(scope)
        user = authenticate(self.store, headers)
        if user is None:
            user = bridge_site_user(self.store, self.config, auth_store, headers, ip)
        scope.setdefault("state", {})
        scope["state"]["billing_user"] = user
        scope["state"]["billing_config"] = self.config
        scope["state"]["client_ip"] = ip

        if not self.config.enabled:
            await self.app(scope, receive, send)
            return

        rule = rule_for(method, path)
        if rule == "free":
            if method in ("POST", "PUT", "PATCH"):
                # Cheap protection for the open routes: cap the body before
                # Starlette buffers it.
                body, too_large = await self._read_body(receive)
                if too_large:
                    await _json_response(send, 413, {"error": "body_too_large", "detail": "Request body exceeds %d bytes." % self.config.max_body_bytes})
                    return
                await self.app(scope, self._replay(body, receive), send)
                return
            await self.app(scope, receive, send)
            return

        self.sweep_jobs()

        if rule == "admin":
            token = headers.get("x-admin-token", "")
            ok = bool(self.config.admin_token) and hmac.compare_digest(token, self.config.admin_token)
            if not ok and user is not None and user.get("is_admin"):
                ok = True
            if not ok:
                await _json_response(send, 403, {"error": "admin_only", "detail": "This endpoint is restricted to the operator."})
                return
            await self.app(scope, receive, send)
            return

        if rule == "resume":
            await self._resume(scope, receive, send, ip)
            return

        # charge / park / detect all need the body.
        body, too_large = await self._read_body(receive)
        if too_large:
            await _json_response(send, 413, {"error": "body_too_large", "detail": "Request body exceeds %d bytes." % self.config.max_body_bytes})
            return
        # `utf-8-sig` so a BOM prices the same document Starlette will parse;
        # anything unparseable is refused rather than priced as empty.
        try:
            data = json.loads(body.decode("utf-8-sig")) if body else {}
        except (ValueError, UnicodeDecodeError):
            await _json_response(send, 400, {"error": "bad_json", "detail": "Request body must be a JSON object."})
            return
        if not isinstance(data, dict):
            await _json_response(send, 400, {"error": "bad_json", "detail": "Request body must be a JSON object."})
            return

        takes_slot = rule in ("charge",)
        if rule == "detect":
            names = data.get("detectors") or []
            if not (isinstance(names, list) and "gptzero" in names):
                await self.app(scope, self._replay(body, receive), send)
                return
            # A GPTZero call on the operator's key is charged; no GPU involved.

        if user is None:
            await _json_response(send, 401, self._sign_in_body(auth_store))
            return

        n_words = word_count(str(data.get("text") or ""))
        wpc = self.config.words_per_credit
        cost = credits_for_words(n_words, wpc)
        effort = 1
        if rule in ("charge", "park"):
            effort = effort_multiplier(data.get("n_candidates", 6), data.get("rounds", 1), data.get("repair_attempts", 0))
            cost *= effort
        ref = uuid.uuid4().hex[:12]
        # A plan allowance that fell due this month lands before the charge,
        # so a subscriber is never refused on the first of the month.
        self.store.topup_if_due(user["id"])
        balance = self.store.charge(user["id"], cost, kind="charge", ref="%s %s" % (path, ref))
        if balance is None:
            have = self.store.balance(user["id"])
            have = int(user.get("credits") or 0) if have is None else int(have)
            await _json_response(
                send,
                402,
                {
                    "error": "insufficient_credits",
                    "detail": "This request costs %d credit%s (%d words at %d words per credit%s); you have %d." % (
                        cost, "" if cost == 1 else "s", n_words, wpc,
                        "" if effort == 1 else ", x%d for extra candidates or rounds" % effort, have
                    ),
                    "needed": cost,
                    "effort": effort,
                    "balance": have,
                    "words": n_words,
                    "words_needed": cost * wpc,
                    "words_left": have * wpc,
                    "words_per_credit": wpc,
                    "packs": "/api/billing/packs",
                    "plans": "/api/billing/plans",
                    "checkout": "/api/billing/checkout",
                },
            )
            return

        scope["state"]["billing_charge"] = {"credits": cost, "words": n_words, "ref": ref, "balance": balance}
        record: Dict[str, Any] = {
            "t": round(time.time(), 3),
            "ip": ip,
            "user": user["id"],
            "path": path,
            "rule": rule,
            "words": n_words,
            "effort": effort,
            "credits": cost,
            "refunded": 0,
        }
        await self._run_charged(
            scope,
            self._replay(body, receive),
            send,
            user_id=user["id"],
            cost=cost,
            balance=balance,
            ref=ref,
            record=record,
            takes_slot=takes_slot,
            park=(rule == "park"),
            check_unchanged=(rule == "charge" and self.config.refund_unchanged),
        )

    # -- charged execution -------------------------------------------------

    async def _acquire_slot(self) -> bool:
        """Wait up to `queue_wait_s` for a GPU slot, off the event loop.

        At most `max_waiting` requests wait at once; the rest are busy at
        once, so a burst cannot pin every worker thread for the wait period.
        """
        if not self._waiters.acquire(blocking=False):
            return False
        try:
            loop = asyncio.get_running_loop()
            acquire = functools.partial(self._slots.acquire, True, self.config.queue_wait_s)
            return bool(await loop.run_in_executor(self._wait_pool, acquire))
        finally:
            self._waiters.release()

    async def _busy(self, send: Callable[..., Awaitable[None]]) -> None:
        wait = max(5, int(self.config.queue_wait_s))
        await _json_response(
            send,
            503,
            {
                "error": "busy",
                "detail": "The rewriter is busy with another document. Nothing was charged; try again in about %d seconds." % wait,
                "retry_after_s": wait,
            },
            [(b"retry-after", str(wait).encode("ascii"))],
        )

    async def _run_charged(
        self,
        scope: Dict[str, Any],
        receive: Callable[[], Awaitable[Dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
        *,
        user_id: str,
        cost: int,
        balance: int,
        ref: str,
        record: Dict[str, Any],
        takes_slot: bool,
        park: bool,
        check_unchanged: bool,
    ) -> None:
        path = scope["path"]
        started = time.monotonic()
        got_slot = False
        if takes_slot:
            got_slot = await self._acquire_slot()
            if not got_slot:
                self._refund(user_id, cost, "%s %s busy" % (path, ref))
                record.update(status=503, refunded=cost, seconds=round(time.monotonic() - started, 3), busy=True)
                self.request_log.write(record)
                await self._busy(send)
                return

        state: Dict[str, Any] = {"status": None, "content_type": "", "captured": bytearray(), "overflow": False, "refunded": 0}
        capture = True

        async def send_wrapper(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status = int(message.get("status", 200))
                state["status"] = status
                hdrs = list(message.get("headers", []))
                for k, v in hdrs:
                    if k.lower() == b"content-type":
                        state["content_type"] = v.decode("latin-1")
                if status >= 400:
                    self._refund(user_id, cost, "%s %s status %d" % (path, ref, status))
                    state["refunded"] = cost
                hdrs.append((b"x-longhand-credits-charged", str(0 if status >= 400 else cost).encode("ascii")))
                hdrs.append((b"x-longhand-credits-balance", str(balance + (cost if status >= 400 else 0)).encode("ascii")))
                message = dict(message, headers=hdrs)
            elif message["type"] == "http.response.body" and capture and not state["overflow"]:
                chunk = message.get("body", b"") or b""
                if len(state["captured"]) + len(chunk) > CAPTURE_LIMIT:
                    state["overflow"] = True
                else:
                    state["captured"].extend(chunk)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            if state["status"] is None:
                self._refund(user_id, cost, "%s %s exception" % (path, ref))
                state["refunded"] = cost
            record.update(status=500, refunded=state["refunded"], seconds=round(time.monotonic() - started, 3), error=True)
            self.request_log.write(record)
            raise
        finally:
            if got_slot:
                self._slots.release()

        status = state["status"] or 0
        body = bytes(state["captured"])
        if park and status < 400 and body:
            try:
                job_id = str(json.loads(body.decode("utf-8")).get("job_id") or "")
            except (ValueError, UnicodeDecodeError, AttributeError):
                job_id = ""
            if job_id:
                with self._jobs_lock:
                    self.jobs[job_id] = {"user_id": user_id, "cost": cost, "created": time.time(), "path": path, "ref": ref}
                record["job"] = job_id
        unchanged: Optional[bool] = None
        if status < 400 and not state["overflow"] and not park:
            if stream_failed(state["content_type"], body):
                self._refund(user_id, cost, "%s %s stream error" % (path, ref))
                state["refunded"] = cost
                record["stream_error"] = True
            elif check_unchanged:
                unchanged = looks_unchanged(state["content_type"], body)
                if unchanged and self._refund_unchanged_allowed(user_id):
                    self._refund(user_id, cost, "%s %s unchanged" % (path, ref))
                    state["refunded"] = cost
        record.update(status=status, refunded=state["refunded"], seconds=round(time.monotonic() - started, 3))
        if unchanged is not None:
            record["unchanged"] = unchanged
        self.request_log.write(record)

    async def _resume(
        self,
        scope: Dict[str, Any],
        receive: Callable[[], Awaitable[Dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
        ip: str,
    ) -> None:
        """GET /api/humanize/stream?job= and its guided twin."""
        path = scope["path"]
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        values = query.get("job") or [""]
        if len(values) != 1:
            # Starlette takes the last value, `parse_qs` the first; agreeing on
            # one id is the only safe answer.
            await _json_response(send, 400, {"error": "bad_job", "detail": "Exactly one job parameter is required."})
            return
        job_id = values[0]
        with self._jobs_lock:
            rec = self.jobs.pop(job_id, None)
        if rec is None:
            # Unknown or already-consumed job. The server would 404 too, but
            # answering here means an anonymous GET never touches the GPU slot.
            await _json_response(
                send,
                404,
                {
                    "error": "unknown_job",
                    "detail": "No queued job %r. Ids are single-use and expire after %.0f seconds." % (job_id, self.config.job_ttl_s),
                },
            )
            return
        started = time.monotonic()
        record: Dict[str, Any] = {
            "t": round(time.time(), 3),
            "ip": ip,
            "user": rec["user_id"] if rec else None,
            "path": path,
            "rule": "resume",
            "job": job_id,
            "credits": rec["cost"] if rec else 0,
            "refunded": 0,
        }

        if not await self._acquire_slot():
            if rec:
                self._refund(rec["user_id"], rec["cost"], "%s job %s busy" % (path, job_id))
                record["refunded"] = rec["cost"]
            record.update(status=503, busy=True, seconds=round(time.monotonic() - started, 3))
            self.request_log.write(record)
            await self._busy(send)
            return

        state: Dict[str, Any] = {"status": None, "content_type": "", "captured": bytearray(), "overflow": False}

        async def send_wrapper(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                state["status"] = int(message.get("status", 200))
                for k, v in message.get("headers", []):
                    if k.lower() == b"content-type":
                        state["content_type"] = v.decode("latin-1")
            elif message["type"] == "http.response.body" and not state["overflow"]:
                chunk = message.get("body", b"") or b""
                if len(state["captured"]) + len(chunk) > CAPTURE_LIMIT:
                    state["overflow"] = True
                else:
                    state["captured"].extend(chunk)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            if rec and state["status"] is None:
                self._refund(rec["user_id"], rec["cost"], "%s job %s exception" % (path, job_id))
                record["refunded"] = rec["cost"]
            record.update(status=500, error=True, seconds=round(time.monotonic() - started, 3))
            self.request_log.write(record)
            raise
        finally:
            self._slots.release()

        status = state["status"] or 0
        if rec:
            body = bytes(state["captured"])
            if status >= 400:
                self._refund(rec["user_id"], rec["cost"], "%s job %s status %d" % (path, job_id, status))
                record["refunded"] = rec["cost"]
            elif not state["overflow"] and stream_failed(state["content_type"], body):
                self._refund(rec["user_id"], rec["cost"], "%s job %s stream error" % (path, job_id))
                record["refunded"] = rec["cost"]
                record["stream_error"] = True
            elif self.config.refund_unchanged and not state["overflow"]:
                unchanged = looks_unchanged(state["content_type"], body)
                if unchanged is not None:
                    record["unchanged"] = unchanged
                if unchanged and self._refund_unchanged_allowed(rec["user_id"]):
                    self._refund(rec["user_id"], rec["cost"], "%s job %s unchanged" % (path, job_id))
                    record["refunded"] = rec["cost"]
        record.update(status=status, seconds=round(time.monotonic() - started, 3))
        self.request_log.write(record)

    # -- body plumbing -----------------------------------------------------

    async def _read_body(self, receive: Callable[[], Awaitable[Dict[str, Any]]]) -> Tuple[bytes, bool]:
        chunks: List[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.config.max_body_bytes:
                return b"", True
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        return b"".join(chunks), False

    @staticmethod
    def _replay(
        body: bytes, receive: Callable[[], Awaitable[Dict[str, Any]]]
    ) -> Callable[[], Awaitable[Dict[str, Any]]]:
        """A receive channel that yields `body` once, then the real channel.

        The second part matters: `StreamingResponse` listens on `receive()`
        for `http.disconnect` while it streams, so returning a synthetic
        disconnect here would cancel every SSE response after its first
        frame. Delegating means a real client disconnect is still seen.
        """
        sent = {"done": False}

        async def replay() -> Dict[str, Any]:
            if not sent["done"]:
                sent["done"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return replay
