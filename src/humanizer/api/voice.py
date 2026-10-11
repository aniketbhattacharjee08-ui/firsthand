"""Your voice: the routes, the store and the hook into the humanize request.

    GET    /api/voice                        the user's samples, liked pile, profile, switch
    POST   /api/voice/samples                add a sample (scored, then prepared in the background)
    POST   /api/voice/samples/{id}/prepare   retry a failed preparation
    DELETE /api/voice/samples/{id}
    DELETE /api/voice/liked/{id}
    POST   /api/voice/settings               {"enabled", "strength"}
    POST   /api/voice/feedback               thumbs up or down on a rewrite

The math is in `humanize.voice`; this module is the account-scoped storage
and the HTTP surface. See `web/VOICE.md` for the wire contract the front end
builds against.

**Who the caller is.** The paywall middleware (`billing.gate`) sets
`request.state.billing_user` on every `/api/` request for a signed-in user,
whichever way they signed in; without the paywall, the site's own session
cookie resolves through `api.auth.current_user`. Guests have no voice: a
sample is personal data and needs an account to be deleted from. A guest
row (`guest:<ip>`) is refused even if one is handed in.

**What is stored.** The sample text, its generic AI twin, its notes, the
free detector's reading at upload, and a weight that thumbs move. Liked
rewrites are stored with the draft they came from so they can serve as a
(draft, rewrite) pair with no model call. Nothing here is ever used to
train anything shared; it is read only for the owning user's own runs.

**Preparation.** A sample is usable as an exemplar only once its twin
exists. That is one instruct-model call per sample, run on a single
background thread so uploads return at once; the page polls `GET
/api/voice` while any sample is "preparing". The run lock the LLM routes
hold is taken for the call, so a preparation never contends with a
rewrite on a local GPU. Without a model (no MLX, no remote URL) the sample
stays "failed" with a reason, the profile still works, and the ranking
tier still runs; only the exemplar tier waits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..humanize import voice as voice_mod
from ..text import words
from . import auth as auth_mod

log = logging.getLogger("humanizer.api.voice")

__all__ = ["VoiceStore", "register_voice_routes", "default_voice_db_path", "attach_voice"]

#: Characters, not words: a 2,500-word sample is about 15,000 characters;
#: this stops a pathological upload before the feature extractor sees it.
MAX_SAMPLE_CHARS = 24000
MAX_TITLE_CHARS = 80
#: A document in parts can run past the single-request word limit; the
#: whole thing is accepted and the stored copy is cut to `KEEP_FEEDBACK_CHARS`.
MAX_FEEDBACK_CHARS = 160000
KEEP_FEEDBACK_CHARS = 40000
MAX_LIKED = 40
#: The free detector's reading above which a sample is refused as AI-written.
AI_SAMPLE_THRESHOLD = 0.5
#: How a thumb moves the weight of the samples that were used.
WEIGHT_UP = 0.25
WEIGHT_DOWN = -0.35
WEIGHT_MIN, WEIGHT_MAX = 0.2, 2.0
EXCERPT_CHARS = 160

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_samples (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    n_words     INTEGER NOT NULL,
    ai_probability REAL,
    draft       TEXT NOT NULL DEFAULT '',
    notes       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'preparing',
    detail      TEXT NOT NULL DEFAULT '',
    weight      REAL NOT NULL DEFAULT 1.0,
    uses        INTEGER NOT NULL DEFAULT 0,
    created     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS voice_samples_user ON voice_samples(user_id, created);
CREATE TABLE IF NOT EXISTS voice_liked (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    run_id      TEXT NOT NULL DEFAULT '',
    original    TEXT NOT NULL,
    text        TEXT NOT NULL,
    n_words     INTEGER NOT NULL,
    created     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS voice_liked_user ON voice_liked(user_id, created);
CREATE TABLE IF NOT EXISTS voice_votes (
    user_id     TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    vote        TEXT NOT NULL,
    sample_ids  TEXT NOT NULL DEFAULT '[]',
    created     REAL NOT NULL,
    PRIMARY KEY (user_id, run_id)
);
CREATE TABLE IF NOT EXISTS voice_settings (
    user_id     TEXT PRIMARY KEY,
    enabled     INTEGER NOT NULL DEFAULT 1,
    strength    TEXT NOT NULL DEFAULT 'balanced',
    profile     TEXT NOT NULL DEFAULT '',
    updated     REAL NOT NULL
);
"""


def default_voice_db_path() -> Path:
    """`HUMANIZER_VOICE_DB`, else `voice.sqlite` beside the auth database."""
    env = os.environ.get("HUMANIZER_VOICE_DB")
    if env:
        return Path(env)
    return auth_mod.default_auth_db_path().with_name("voice.sqlite")


def _excerpt(text: str, n: int = EXCERPT_CHARS) -> str:
    t = " ".join((text or "").split())
    if len(t) <= n:
        return t
    return t[: n - 1].rsplit(" ", 1)[0] + "..."


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


class VoiceStore:
    """SQLite, one connection, one lock. Small enough not to need more."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # A preparation in flight when the process stopped is not coming
            # back; say so, so the page offers Try again instead of waiting.
            self._conn.execute(
                "UPDATE voice_samples SET status = 'failed', detail = ? WHERE status = 'preparing'",
                ("The preparation was interrupted by a restart. Try again.",),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- samples -----------------------------------------------------------

    def samples(self, user_id: str, with_text: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM voice_samples WHERE user_id = ? ORDER BY created DESC", (user_id,)
            ).fetchall()
        return [self._sample_dict(r, with_text) for r in rows]

    def sample(self, user_id: str, sample_id: str, with_text: bool = False) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM voice_samples WHERE user_id = ? AND id = ?", (user_id, sample_id)
            ).fetchone()
        return self._sample_dict(row, with_text) if row is not None else None

    def _sample_dict(self, row: sqlite3.Row, with_text: bool) -> Dict[str, Any]:
        d = {
            "id": row["id"],
            "title": row["title"],
            "n_words": row["n_words"],
            "excerpt": _excerpt(row["text"]),
            "status": row["status"],
            "detail": row["detail"],
            "ai_probability": row["ai_probability"],
            "weight": round(float(row["weight"]), 3),
            "uses": row["uses"],
            "created": row["created"],
        }
        if with_text:
            d["text"] = row["text"]
            d["draft"] = row["draft"]
            d["notes"] = row["notes"]
        return d

    def count_samples(self, user_id: str) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM voice_samples WHERE user_id = ?", (user_id,)).fetchone()[0])

    def has_sample_text(self, user_id: str, text: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM voice_samples WHERE user_id = ? AND text_hash = ?", (user_id, _hash(text))
            ).fetchone() is not None

    def add_sample(self, user_id: str, title: str, text: str, ai_probability: Optional[float], status: str, detail: str = "") -> Dict[str, Any]:
        sample_id = "s_" + uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO voice_samples (id, user_id, title, text, text_hash, n_words, ai_probability, draft, notes, status, detail, weight, uses, created) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, 1.0, 0, ?)",
                (sample_id, user_id, title, text, _hash(text), len(words(text)), ai_probability, voice_mod.notes_from(text), status, detail, now),
            )
            self._conn.commit()
        self._refresh_profile(user_id)
        return self.sample(user_id, sample_id)  # type: ignore[return-value]

    def set_prepared(self, user_id: str, sample_id: str, draft: Optional[str], detail: str = "") -> None:
        with self._lock:
            if draft:
                self._conn.execute(
                    "UPDATE voice_samples SET draft = ?, status = 'ready', detail = '' WHERE user_id = ? AND id = ?",
                    (draft, user_id, sample_id),
                )
            else:
                self._conn.execute(
                    "UPDATE voice_samples SET status = 'failed', detail = ? WHERE user_id = ? AND id = ?",
                    (detail, user_id, sample_id),
                )
            self._conn.commit()

    def set_status(self, user_id: str, sample_id: str, status: str, detail: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE voice_samples SET status = ?, detail = ? WHERE user_id = ? AND id = ?",
                (status, detail, user_id, sample_id),
            )
            self._conn.commit()

    def delete_sample(self, user_id: str, sample_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM voice_samples WHERE user_id = ? AND id = ?", (user_id, sample_id))
            self._conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            self._refresh_profile(user_id)
        return deleted

    def note_uses(self, user_id: str, sample_ids: Sequence[str]) -> None:
        if not sample_ids:
            return
        with self._lock:
            for sid in sample_ids:
                self._conn.execute(
                    "UPDATE voice_samples SET uses = uses + 1 WHERE user_id = ? AND id = ?", (user_id, sid)
                )
            self._conn.commit()

    def move_weights(self, user_id: str, sample_ids: Sequence[str], delta: float) -> None:
        if not sample_ids:
            return
        with self._lock:
            for sid in sample_ids:
                self._conn.execute(
                    "UPDATE voice_samples SET weight = MAX(?, MIN(?, weight + ?)) WHERE user_id = ? AND id = ?",
                    (WEIGHT_MIN, WEIGHT_MAX, delta, user_id, sid),
                )
            self._conn.commit()

    # -- liked --------------------------------------------------------------

    def liked(self, user_id: str, with_text: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM voice_liked WHERE user_id = ? ORDER BY created DESC", (user_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = {"id": r["id"], "excerpt": _excerpt(r["text"]), "n_words": r["n_words"], "created": r["created"], "run_id": r["run_id"]}
            if with_text:
                d["text"] = r["text"]
                d["original"] = r["original"]
            out.append(d)
        return out

    def add_liked(self, user_id: str, run_id: str, original: str, text: str) -> str:
        liked_id = "l_" + uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            if run_id:
                self._conn.execute("DELETE FROM voice_liked WHERE user_id = ? AND run_id = ?", (user_id, run_id))
            self._conn.execute(
                "INSERT INTO voice_liked (id, user_id, run_id, original, text, n_words, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (liked_id, user_id, run_id, original, text, len(words(text)), now),
            )
            # Keep the pile bounded: the oldest go first.
            extra = self._conn.execute(
                "SELECT id FROM voice_liked WHERE user_id = ? ORDER BY created DESC LIMIT -1 OFFSET ?", (user_id, MAX_LIKED)
            ).fetchall()
            for r in extra:
                self._conn.execute("DELETE FROM voice_liked WHERE id = ?", (r["id"],))
            self._conn.commit()
        return liked_id

    def remove_liked_for_run(self, user_id: str, run_id: str) -> None:
        if not run_id:
            return
        with self._lock:
            self._conn.execute("DELETE FROM voice_liked WHERE user_id = ? AND run_id = ?", (user_id, run_id))
            self._conn.commit()

    def delete_liked(self, user_id: str, liked_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM voice_liked WHERE user_id = ? AND id = ?", (user_id, liked_id))
            self._conn.commit()
            return cur.rowcount > 0

    # -- votes --------------------------------------------------------------

    def record_vote(self, user_id: str, run_id: str, vote: str, sample_ids: Sequence[str]) -> Tuple[Optional[str], List[str]]:
        """Store the vote; return the previous vote for this run and the
        sample ids it moved, so the caller can undo exactly that move."""
        with self._lock:
            prev = self._conn.execute(
                "SELECT vote, sample_ids FROM voice_votes WHERE user_id = ? AND run_id = ?", (user_id, run_id)
            ).fetchone()
            self._conn.execute(
                "INSERT INTO voice_votes (user_id, run_id, vote, sample_ids, created) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, run_id) DO UPDATE SET vote = excluded.vote, sample_ids = excluded.sample_ids, created = excluded.created",
                (user_id, run_id, vote, json.dumps(list(sample_ids)), time.time()),
            )
            self._conn.commit()
        if prev is None:
            return None, []
        try:
            prev_ids = [str(x) for x in json.loads(prev["sample_ids"] or "[]")]
        except ValueError:
            prev_ids = []
        return prev["vote"], prev_ids

    # -- settings and profile ------------------------------------------------

    def settings(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM voice_settings WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return {"enabled": True, "strength": voice_mod.DEFAULT_STRENGTH, "profile": None}
        profile = None
        if row["profile"]:
            try:
                profile = json.loads(row["profile"])
            except ValueError:
                profile = None
        return {"enabled": bool(row["enabled"]), "strength": row["strength"], "profile": profile}

    def set_settings(self, user_id: str, enabled: Optional[bool] = None, strength: Optional[str] = None) -> Dict[str, Any]:
        current = self.settings(user_id)
        if enabled is not None:
            current["enabled"] = bool(enabled)
        if strength is not None:
            current["strength"] = strength
        self._write_settings(user_id, current)
        return current

    def _write_settings(self, user_id: str, current: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO voice_settings (user_id, enabled, strength, profile, updated) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET enabled = excluded.enabled, strength = excluded.strength, profile = excluded.profile, updated = excluded.updated",
                (
                    user_id,
                    1 if current.get("enabled", True) else 0,
                    current.get("strength") or voice_mod.DEFAULT_STRENGTH,
                    json.dumps(current["profile"]) if current.get("profile") else "",
                    time.time(),
                ),
            )
            self._conn.commit()

    def _refresh_profile(self, user_id: str) -> None:
        """Rebuild the fingerprint from every sample the user has. Samples
        count toward the profile whether or not their twin is ready: the
        fingerprint needs only the user's real text."""
        texts = [s["text"] for s in self.samples(user_id, with_text=True)]
        profile = voice_mod.build_profile(texts)
        current = self.settings(user_id)
        current["profile"] = profile.as_dict() if not profile.empty else None
        self._write_settings(user_id, current)

    def profile(self, user_id: str) -> Optional[voice_mod.VoiceProfile]:
        raw = self.settings(user_id).get("profile")
        if not raw:
            return None
        profile = voice_mod.VoiceProfile.from_dict(raw)
        return None if profile.empty else profile

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            for table in ("voice_samples", "voice_liked", "voice_votes", "voice_settings"):
                self._conn.execute("DELETE FROM %s WHERE user_id = ?" % table, (user_id,))
            self._conn.commit()


# ------------------------------------------------------------ preparation


class Preparer:
    """One background thread that writes each sample's generic twin.

    `backend_factory` returns an `llm.Backend` (the instruct checkpoint);
    `run_lock` is the server's LLM run lock, held for the call. Both are
    injected so tests can hand in an `EchoBackend` and a dummy lock.
    """

    def __init__(self, store: VoiceStore, backend_factory: Any = None, run_lock: Any = None) -> None:
        self.store = store
        self.backend_factory = backend_factory
        self.run_lock = run_lock
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending: set = set()
        self._available_at = 0.0
        self._available_value = False

    #: How long one availability answer is trusted. The page polls every
    #: few seconds while a sample prepares; probing the backend registry on
    #: every poll would reorder it under a running rewrite.
    AVAILABILITY_TTL_S = 30.0

    def available(self) -> bool:
        if self.backend_factory is None:
            return False
        now = time.time()
        if now - self._available_at < self.AVAILABILITY_TTL_S:
            return self._available_value
        try:
            backend = self.backend_factory()
            value = backend is not None and bool(backend.available())
        except Exception:  # noqa: BLE001 - "not available" is the answer, not a crash
            value = False
        self._available_at = now
        self._available_value = value
        return value

    def pending(self, user_id: str, sample_id: str) -> bool:
        with self._lock:
            return (user_id, sample_id) in self._pending

    def unavailable_reason(self) -> str:
        if self.backend_factory is None:
            return "No language model is configured on this server, so the sample cannot be prepared."
        try:
            backend = self.backend_factory()
            reason = backend.unavailable_reason() if backend is not None else None
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
        return reason or "The language model is not available right now."

    def submit(self, user_id: str, sample_id: str) -> bool:
        """Queue one preparation. False when it is already queued or running."""
        key = (user_id, sample_id)
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
            self._queue.put(key)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="voice-preparer", daemon=True)
                self._thread.start()
        return True

    def _run(self) -> None:
        while True:
            try:
                user_id, sample_id = self._queue.get(timeout=30.0)
            except queue.Empty:
                # Retire under the lock so a submit racing this exit either
                # sees the live thread and a queue it will drain, or no
                # thread at all and starts one.
                with self._lock:
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            try:
                self.prepare_now(user_id, sample_id)
            except Exception as exc:  # noqa: BLE001 - one failure must not kill the thread
                log.warning("voice: preparing %s failed: %s", sample_id, exc)
                try:
                    self.store.set_prepared(user_id, sample_id, None, "Preparation failed: %s" % exc)
                except Exception:  # noqa: BLE001
                    pass
            finally:
                with self._lock:
                    self._pending.discard((user_id, sample_id))
                self._queue.task_done()

    def prepare_now(self, user_id: str, sample_id: str) -> None:
        """Synchronous: write the twin, or the reason there is none."""
        sample = self.store.sample(user_id, sample_id, with_text=True)
        if sample is None:
            return
        if not self.available():
            self.store.set_prepared(user_id, sample_id, None, self.unavailable_reason())
            return
        from ..humanize import llm as llm_mod

        backend = self.backend_factory()
        # Not `estimate_max_tokens`: its 900-token ceiling is for a paragraph,
        # and a twin must be allowed to run the whole sample's length.
        n_words = max(1, len(words(sample["text"])))
        request = llm_mod.GenerationRequest(
            prompt=voice_mod.flatten_prompt(sample["text"]),
            temperature=0.7,
            max_tokens=int(n_words * 1.35 * 1.8) + 64,
            tag=("voice", sample_id),
        )
        lock = self.run_lock
        if lock is not None:
            lock.acquire()
        try:
            batch = backend.generate([request])
        finally:
            if lock is not None:
                lock.release()
        raw = batch.texts[0] if batch.texts else ""
        twin = voice_mod.clean_flattened(raw, sample["text"])
        if twin is None:
            self.store.set_prepared(
                user_id, sample_id, None,
                "The model did not produce a usable generic version. Try again, or try a different passage.",
            )
            return
        self.store.set_prepared(user_id, sample_id, twin)


# ------------------------------------------------------------------ identity


def _voice_user_id(request: Request) -> Optional[str]:
    """The account that owns the voice, or None for a guest or a visitor."""
    billing_user = getattr(request.state, "billing_user", None)
    if isinstance(billing_user, dict) and billing_user.get("id"):
        uid = str(billing_user["id"])
        if uid.startswith("guest:") or billing_user.get("guest"):
            return None
        # A site account bridged to billing keeps its site id as the key, so
        # the same person owns the same rows whether or not the paywall is
        # mounted. Magic-link and API-key users have only a billing id.
        if billing_user.get("auth_user_id"):
            return "a:" + str(billing_user["auth_user_id"])
        return "b:" + uid
    site_user = auth_mod.current_user(request)
    if site_user and site_user.get("id"):
        return "a:" + str(site_user["id"])
    return None


def _err(status: int, error: str, detail: str, **extra: Any) -> JSONResponse:
    body: Dict[str, Any] = {"error": error, "detail": detail}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


# ------------------------------------------------------------------ scoring


def _sample_ai_probability(text: str) -> Optional[float]:
    """The free detector's reading, or None when none is available."""
    try:
        from ..detectors import yardstick as yardstick_mod

        judge = yardstick_mod.proxy()
        if judge is None or not judge.available():
            return None
        p = float(judge.score(text).ai_probability)
        return p if p == p else None
    except Exception:  # noqa: BLE001 - scoring is a check, not the feature
        return None


# ----------------------------------------------------------------- the spec


def build_spec(store: VoiceStore, user_id: str, run_id: str) -> voice_mod.VoiceSpec:
    """What one run gets: the ready samples as a per-paragraph picker, the
    liked pile as the fallback exemplar, the profile and the strength."""
    settings = store.settings(user_id)
    if not settings.get("enabled", True):
        return voice_mod.VoiceSpec(reason="disabled", run_id=run_id)
    samples = [s for s in store.samples(user_id, with_text=True) if s["status"] == "ready" and s["draft"]]
    liked = store.liked(user_id, with_text=True)
    profile = store.profile(user_id)
    if not samples and profile is None:
        return voice_mod.VoiceSpec(reason="no_samples", run_id=run_id)

    spec = voice_mod.VoiceSpec(
        profile=profile,
        sample_ids=tuple(s["id"] for s in samples),
        strength=settings.get("strength") or voice_mod.DEFAULT_STRENGTH,
        run_id=run_id,
        on_used=lambda ids: store.note_uses(user_id, [i for i in ids if i.startswith("s_")]),
    )

    def shots_for(paragraph: str) -> Sequence[Tuple[str, str, str, str]]:
        shots, ids = voice_mod.select_shots(samples, paragraph, liked=liked)
        for sid in ids:
            if sid not in spec.used_ids:
                spec.used_ids.append(sid)
        return shots

    if samples or liked:
        spec.shots_for = shots_for
    return spec


def attach_voice(app: FastAPI, body: Any, request: Request) -> None:
    """Called by the LLM routes at POST time: resolve the caller's voice into
    `body._voice`. Nothing is attached when `body.voice` is false. A job
    resumed later carries the spec with it, so the GET needs no session."""
    if not getattr(body, "voice", False):
        return
    store: Optional[VoiceStore] = getattr(app.state, "voice_store", None)
    if store is None:
        return
    user_id = _voice_user_id(request)
    run_id = "r_" + uuid.uuid4().hex[:12]
    if user_id is None:
        body._voice = voice_mod.VoiceSpec(reason="login_required", run_id=run_id)
        return
    body._voice = build_spec(store, user_id, run_id)


# ------------------------------------------------------------------- bodies


class SampleBody(BaseModel):
    title: str = Field(default="", max_length=MAX_TITLE_CHARS)
    text: str = Field(default="", max_length=MAX_SAMPLE_CHARS)


class SettingsBody(BaseModel):
    enabled: Optional[bool] = None
    strength: Optional[str] = None


class FeedbackBody(BaseModel):
    vote: str = Field(default="")
    original: str = Field(default="", max_length=MAX_FEEDBACK_CHARS)
    text: str = Field(default="", max_length=MAX_FEEDBACK_CHARS)
    run_id: str = Field(default="", max_length=40)
    sample_ids: List[str] = Field(default_factory=list)


_ID_OK = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def _clean_id(value: str, prefix: str) -> Optional[str]:
    v = (value or "").strip()
    if not v.startswith(prefix) or len(v) > 24 or any(ch not in _ID_OK for ch in v):
        return None
    return v


# ------------------------------------------------------------------- routes


def register_voice_routes(
    app: FastAPI,
    store: Optional[VoiceStore] = None,
    backend_factory: Any = None,
    run_lock: Any = None,
) -> FastAPI:
    """Add the voice routes and the attach hook. Idempotent per app."""
    if getattr(app.state, "voice_installed", False):
        return app
    db = store or VoiceStore(default_voice_db_path())
    preparer = Preparer(db, backend_factory=backend_factory, run_lock=run_lock)
    app.state.voice_store = db
    app.state.voice_preparer = preparer
    app.state.voice_installed = True

    limits = {
        "min_words": voice_mod.MIN_SAMPLE_WORDS,
        "max_words": voice_mod.MAX_SAMPLE_WORDS,
        "max_samples": voice_mod.MAX_SAMPLES,
        "part_words": voice_mod.PART_WORDS,
        "min_ready": 1,
    }

    def payload(user_id: Optional[str]) -> Dict[str, Any]:
        if user_id is None:
            return {
                "signed_in": False,
                "enabled": False,
                "strength": voice_mod.DEFAULT_STRENGTH,
                "ready": 0,
                "samples": [],
                "liked": [],
                "profile": None,
                "llm_available": preparer.available(),
                "limits": limits,
            }
        settings = db.settings(user_id)
        samples = db.samples(user_id)
        profile = db.profile(user_id)
        return {
            "signed_in": True,
            "enabled": bool(settings.get("enabled", True)),
            "strength": settings.get("strength") or voice_mod.DEFAULT_STRENGTH,
            "ready": sum(1 for s in samples if s["status"] == "ready"),
            "samples": samples,
            "liked": db.liked(user_id),
            "profile": {
                "n_samples": profile.n_samples,
                "n_words": profile.n_words,
                "traits": voice_mod.describe_profile(profile),
            }
            if profile is not None
            else None,
            "llm_available": preparer.available(),
            "limits": limits,
        }

    @app.get("/api/voice")
    def voice_state(request: Request) -> JSONResponse:
        """The caller's voice: samples, the liked pile, the profile, the switch.
        Always 200; `signed_in: false` for a guest or a visitor."""
        return JSONResponse(payload(_voice_user_id(request)))

    @app.post("/api/voice/samples")
    def add_sample(request: Request, body: SampleBody) -> JSONResponse:
        """Add one piece of the user's own writing. 201 with the sample, or
        422 naming what is wrong with it (too short, too long, too many,
        a duplicate, or reads as AI-written to the free detector)."""
        user_id = _voice_user_id(request)
        if user_id is None:
            return _err(401, "login_required", "Sign in to add your writing.")
        text = voice_mod.neutralise_markers((body.text or "").strip())
        title = " ".join((body.title or "").split())[:MAX_TITLE_CHARS]
        n = len(words(text))
        if n < voice_mod.MIN_SAMPLE_WORDS:
            return _err(422, "too_short", "A sample needs at least %d words; this one has %d." % (voice_mod.MIN_SAMPLE_WORDS, n), n_words=n)
        if n > voice_mod.MAX_SAMPLE_WORDS:
            return _err(422, "too_long", "A sample can be at most %d words; this one has %d. Split it into pieces." % (voice_mod.MAX_SAMPLE_WORDS, n), n_words=n)
        have = db.count_samples(user_id)
        if have >= voice_mod.MAX_SAMPLES:
            return _err(422, "too_many", "You already have %d samples, the most the profile uses. Remove one to add another." % voice_mod.MAX_SAMPLES)
        if db.has_sample_text(user_id, text):
            return _err(422, "duplicate", "That sample is already in your list.")
        p = _sample_ai_probability(text)
        if p is not None and p > AI_SAMPLE_THRESHOLD:
            return _err(
                422,
                "reads_as_ai",
                "This reads as AI-written to the detector (%d percent), so it would teach the humanizer the wrong habits. "
                "Use something you wrote yourself without AI help." % round(p * 100),
                ai_probability=p,
            )
        # A long upload becomes several paragraph-sized parts: exemplars
        # have to fit the model's window three at a time, and a part is
        # what the per-paragraph picker matches against.
        parts = voice_mod.split_sample(text)
        if have + len(parts) > voice_mod.MAX_SAMPLES:
            return _err(
                422,
                "too_many",
                "That piece would be split into %d parts, and you have room for %d more. Remove a sample or upload a shorter piece."
                % (len(parts), voice_mod.MAX_SAMPLES - have),
            )
        parts = [part for part in parts if not db.has_sample_text(user_id, part)] or parts
        available = preparer.available()
        reason = "" if available else preparer.unavailable_reason()
        added = []
        for i, part in enumerate(parts):
            part_title = title if len(parts) == 1 else ("%s (part %d)" % (title or "Untitled", i + 1))[:MAX_TITLE_CHARS]
            sample = db.add_sample(user_id, part_title, part, p, "preparing" if available else "failed", reason)
            if available:
                preparer.submit(user_id, sample["id"])
            added.append(sample)
        return JSONResponse(status_code=201, content={"sample": added[0], "samples": added, "n_parts": len(added)})

    @app.post("/api/voice/samples/{sample_id}/prepare")
    def prepare_sample(request: Request, sample_id: str) -> JSONResponse:
        """Queue a failed sample's preparation again."""
        user_id = _voice_user_id(request)
        if user_id is None:
            return _err(401, "login_required", "Sign in first.")
        sid = _clean_id(sample_id, "s_")
        sample = db.sample(user_id, sid) if sid else None
        if sample is None:
            return _err(404, "unknown_sample", "No such sample.")
        if sample["status"] == "ready" or preparer.pending(user_id, sid):
            return JSONResponse({"sample": sample})
        if not preparer.available():
            db.set_status(user_id, sid, "failed", preparer.unavailable_reason())
            return JSONResponse({"sample": db.sample(user_id, sid)})
        db.set_status(user_id, sid, "preparing", "")
        preparer.submit(user_id, sid)
        return JSONResponse({"sample": db.sample(user_id, sid)})

    @app.delete("/api/voice/samples/{sample_id}")
    def delete_sample(request: Request, sample_id: str) -> JSONResponse:
        user_id = _voice_user_id(request)
        if user_id is None:
            return _err(401, "login_required", "Sign in first.")
        sid = _clean_id(sample_id, "s_")
        if not sid or not db.delete_sample(user_id, sid):
            return _err(404, "unknown_sample", "No such sample.")
        return JSONResponse({"deleted": True})

    @app.delete("/api/voice/liked/{liked_id}")
    def delete_liked(request: Request, liked_id: str) -> JSONResponse:
        user_id = _voice_user_id(request)
        if user_id is None:
            return _err(401, "login_required", "Sign in first.")
        lid = _clean_id(liked_id, "l_")
        if not lid or not db.delete_liked(user_id, lid):
            return _err(404, "unknown_liked", "No such rewrite.")
        return JSONResponse({"deleted": True})

    @app.post("/api/voice/settings")
    def set_settings(request: Request, body: SettingsBody) -> JSONResponse:
        user_id = _voice_user_id(request)
        if user_id is None:
            return _err(401, "login_required", "Sign in first.")
        if body.strength is not None and body.strength not in voice_mod.STRENGTHS:
            return _err(422, "bad_strength", "Strength must be one of %s." % ", ".join(voice_mod.STRENGTHS))
        current = db.set_settings(user_id, enabled=body.enabled, strength=body.strength)
        return JSONResponse({"enabled": current["enabled"], "strength": current["strength"]})

    @app.post("/api/voice/feedback")
    def feedback(request: Request, body: FeedbackBody) -> JSONResponse:
        """A thumb on a rewrite. Up keeps the (draft, rewrite) pair in the
        liked pile and raises the used samples' weight; down lowers it. A
        second vote on the same run replaces the first."""
        user_id = _voice_user_id(request)
        if user_id is None:
            return _err(401, "login_required", "Sign in first.")
        vote = (body.vote or "").strip().lower()
        if vote not in ("up", "down"):
            return _err(422, "bad_vote", "Vote must be up or down.")
        text = voice_mod.neutralise_markers((body.text or "").strip())[:KEEP_FEEDBACK_CHARS]
        original = voice_mod.neutralise_markers((body.original or "").strip())[:KEEP_FEEDBACK_CHARS]
        if vote == "up" and (not text or not original):
            return _err(422, "missing_text", "A thumbs up needs the draft and the rewrite.")
        run_id = _clean_id(body.run_id, "r_") or ""
        own = {s["id"] for s in db.samples(user_id)}
        sample_ids = [sid for sid in (_clean_id(s, "s_") for s in body.sample_ids[:voice_mod.MAX_SAMPLES]) if sid and sid in own]
        prev, prev_ids = db.record_vote(user_id, run_id, vote, sample_ids) if run_id else (None, [])
        # Undo the earlier vote's move on the ids it actually moved.
        prev_ids = [sid for sid in prev_ids if sid in own]
        if prev == "up":
            db.move_weights(user_id, prev_ids, -WEIGHT_UP)
        elif prev == "down":
            db.move_weights(user_id, prev_ids, -WEIGHT_DOWN)
        if vote == "up":
            db.move_weights(user_id, sample_ids, WEIGHT_UP)
            p = _sample_ai_probability(text)
            if p is None or p <= AI_SAMPLE_THRESHOLD:
                db.add_liked(user_id, run_id, original, text)
        else:
            db.move_weights(user_id, sample_ids, WEIGHT_DOWN)
            db.remove_liked_for_run(user_id, run_id)
        return JSONResponse({"recorded": True, "vote": vote, "liked_count": len(db.liked(user_id))})

    return app
