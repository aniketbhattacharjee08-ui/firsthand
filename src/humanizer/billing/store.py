"""SQLite persistence for users, sessions, API keys and the credit ledger.

One file, WAL mode, one connection guarded by a lock. That is adequate for a
single-process deployment, which is what the rest of the server assumes (the
LLM job store is an in-memory dict). Every balance change goes through
`charge()` or `credit()`, both of which write a ledger row in the same
transaction, so the ledger always reconciles to the balance.

`charge()` is atomic: it is one `UPDATE ... WHERE credits >= ?` and reports
failure by rowcount, so two concurrent requests cannot both spend the last
credit.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    credits     INTEGER NOT NULL DEFAULT 0,
    is_admin    INTEGER NOT NULL DEFAULT 0,
    created     REAL NOT NULL,
    last_login  REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    created     REAL NOT NULL,
    expires     REAL NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    key_hash    TEXT NOT NULL UNIQUE,
    prefix      TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    created     REAL NOT NULL,
    last_used   REAL,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ledger (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(id),
    delta         INTEGER NOT NULL,
    balance_after INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    ref           TEXT NOT NULL DEFAULT '',
    created       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_user ON ledger(user_id, id);
CREATE TABLE IF NOT EXISTS webhook_events (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    received    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS magic_links (
    nonce       TEXT PRIMARY KEY,
    email       TEXT NOT NULL,
    created     REAL NOT NULL,
    used        REAL
);
CREATE TABLE IF NOT EXISTS signups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ip          TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    created     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS signups_ip ON signups(ip, created);
"""


def _row(cur: sqlite3.Cursor) -> Optional[Dict[str, Any]]:
    r = cur.fetchone()
    return dict(r) if r is not None else None


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=10.0
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _begin(self) -> None:
        # A previous transaction that died mid-way must not wedge the
        # connection for every later write.
        if self._conn.in_transaction:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        self._conn.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        try:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    # -- users -----------------------------------------------------------

    def get_or_create_user(self, email: str, free_credits: int) -> Dict[str, Any]:
        return self.create_or_get(email, free_credits)[0]

    def create_or_get(
        self, email: str, free_credits: int, ip: str = ""
    ) -> "Tuple[Dict[str, Any], bool]":
        """(user, created). Records the sign-up address when a user is new."""
        email = email.strip().lower()
        with self._lock:
            row = _row(self._conn.execute("SELECT * FROM users WHERE email = ?", (email,)))
            if row is not None:
                return row, False
            uid = secrets.token_hex(12)
            now = time.time()
            self._begin()
            try:
                self._conn.execute(
                    "INSERT INTO users (id, email, credits, created) VALUES (?, ?, ?, ?)",
                    (uid, email, int(free_credits), now),
                )
                if free_credits > 0:
                    self._conn.execute(
                        "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
                        " VALUES (?, ?, ?, 'signup', '', ?)",
                        (uid, int(free_credits), int(free_credits), now),
                    )
                if ip:
                    self._conn.execute(
                        "INSERT INTO signups (ip, user_id, created) VALUES (?, ?, ?)", (ip, uid, now)
                    )
                self._conn.execute("COMMIT")
            except sqlite3.IntegrityError:
                # Lost a race with a concurrent sign-up of the same email.
                self._rollback()
                row = _row(self._conn.execute("SELECT * FROM users WHERE email = ?", (email,)))
                if row is not None:
                    return row, False
                raise
            except Exception:
                self._rollback()
                raise
            return _row(self._conn.execute("SELECT * FROM users WHERE id = ?", (uid,))), True  # type: ignore[return-value]

    def signups_from(self, ip: str, since_seconds: float) -> int:
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM signups WHERE ip = ? AND created > ?",
                    (ip, time.time() - since_seconds),
                ).fetchone()[0]
            )

    def delete_user(self, user_id: str) -> bool:
        """Anonymise an account: sessions and keys revoked, email tombstoned.

        Ledger rows stay under the same id for accounting; nothing personal
        remains on them.
        """
        with self._lock:
            row = _row(self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)))
            if row is None:
                return False
            now = time.time()
            self._begin()
            try:
                self._conn.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (user_id,))
                self._conn.execute("UPDATE api_keys SET revoked = 1 WHERE user_id = ?", (user_id,))
                self._conn.execute(
                    "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
                    " VALUES (?, ?, 0, 'deleted', '', ?)",
                    (user_id, -int(row["credits"]), now),
                )
                self._conn.execute(
                    "UPDATE users SET email = ?, credits = 0, is_admin = 0 WHERE id = ?",
                    ("deleted:%s" % user_id, user_id),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
            return True

    def stats(self, days: int = 14) -> Dict[str, Any]:
        """Operator summary: totals and a per-day table for the last `days`."""
        with self._lock:
            c = self._conn
            total_users = c.execute("SELECT COUNT(*) FROM users WHERE email NOT LIKE 'deleted:%'").fetchone()[0]
            deleted = c.execute("SELECT COUNT(*) FROM users WHERE email LIKE 'deleted:%'").fetchone()[0]
            balance = c.execute("SELECT COALESCE(SUM(credits), 0) FROM users").fetchone()[0]

            def total(kind: str) -> int:
                return int(c.execute("SELECT COALESCE(SUM(delta), 0) FROM ledger WHERE kind = ?", (kind,)).fetchone()[0])

            since = time.time() - days * 86400
            per_day: Dict[str, Dict[str, int]] = {}
            for kind, delta, created in c.execute(
                "SELECT kind, delta, created FROM ledger WHERE created > ? ORDER BY created", (since,)
            ):
                day = time.strftime("%Y-%m-%d", time.gmtime(created))
                d = per_day.setdefault(day, {"signups": 0, "purchased": 0, "spent": 0, "refunded": 0})
                if kind == "signup":
                    d["signups"] += 1
                elif kind == "purchase":
                    d["purchased"] += int(delta)
                elif kind == "charge":
                    d["spent"] += -int(delta)
                elif kind == "refund":
                    d["refunded"] += int(delta)
            return {
                "users": int(total_users),
                "deleted_users": int(deleted),
                "credits_outstanding": int(balance),
                "credits_purchased": total("purchase"),
                "credits_granted": total("signup") + total("manual"),
                "credits_spent": -total("charge"),
                "credits_refunded": total("refund"),
                "days": per_day,
            }

    def user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return _row(self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)))

    def user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return _row(
                self._conn.execute(
                    "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
                )
            )

    def set_admin(self, user_id: str, is_admin: bool = True) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET is_admin = ? WHERE id = ?", (1 if is_admin else 0, user_id)
            )

    # -- magic links -----------------------------------------------------

    def issue_nonce(self, email: str) -> str:
        nonce = secrets.token_urlsafe(24)
        with self._lock:
            self._conn.execute(
                "INSERT INTO magic_links (nonce, email, created) VALUES (?, ?, ?)",
                (nonce, email.strip().lower(), time.time()),
            )
            # Keep the table small; links older than a day are dead anyway.
            self._conn.execute(
                "DELETE FROM magic_links WHERE created < ?", (time.time() - 86400,)
            )
        return nonce

    def consume_nonce(self, nonce: str, email: str) -> bool:
        """True exactly once per nonce. Makes login links single-use."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE magic_links SET used = ? WHERE nonce = ? AND email = ? AND used IS NULL",
                (time.time(), nonce, email.strip().lower()),
            )
            return cur.rowcount == 1

    # -- sessions --------------------------------------------------------

    def create_session(self, user_id: str, ttl_seconds: float) -> Dict[str, Any]:
        sid = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, user_id, created, expires) VALUES (?, ?, ?, ?)",
                (sid, user_id, now, now + ttl_seconds),
            )
            self._conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?", (now, user_id)
            )
        return {"id": sid, "user_id": user_id, "created": now, "expires": now + ttl_seconds}

    def session_user(self, session_id: str) -> Optional[Dict[str, Any]]:
        """The user for a live session, or None."""
        with self._lock:
            row = _row(
                self._conn.execute(
                    "SELECT u.*, s.expires AS session_expires FROM sessions s"
                    " JOIN users u ON u.id = s.user_id"
                    " WHERE s.id = ? AND s.revoked = 0 AND s.expires > ?",
                    (session_id, time.time()),
                )
            )
        return row

    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET revoked = 1 WHERE id = ?", (session_id,))

    # -- API keys --------------------------------------------------------

    def create_api_key(self, user_id: str, label: str = "") -> Dict[str, Any]:
        """Create a key. The raw key is returned once and never stored."""
        raw = "lh_" + secrets.token_urlsafe(32)
        kid = secrets.token_hex(8)
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_keys (id, user_id, key_hash, prefix, label, created)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (kid, user_id, hash_key(raw), raw[:11], label[:80], now),
            )
        return {"id": kid, "key": raw, "prefix": raw[:11], "label": label[:80], "created": now}

    def api_key_user(self, raw: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = _row(
                self._conn.execute(
                    "SELECT u.*, k.id AS api_key_id FROM api_keys k"
                    " JOIN users u ON u.id = k.user_id"
                    " WHERE k.key_hash = ? AND k.revoked = 0",
                    (hash_key(raw),),
                )
            )
            if row is not None:
                self._conn.execute(
                    "UPDATE api_keys SET last_used = ? WHERE id = ?",
                    (time.time(), row["api_key_id"]),
                )
        return row

    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, prefix, label, created, last_used FROM api_keys"
                " WHERE user_id = ? AND revoked = 0 ORDER BY created",
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def revoke_api_key(self, user_id: str, key_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?",
                (key_id, user_id),
            )
            return cur.rowcount == 1

    # -- credits ---------------------------------------------------------

    def charge(self, user_id: str, amount: int, kind: str, ref: str = "") -> Optional[int]:
        """Deduct `amount`. Returns the new balance, or None if insufficient."""
        amount = int(amount)
        if amount <= 0:
            return self.balance(user_id)
        with self._lock:
            self._begin()
            try:
                cur = self._conn.execute(
                    "UPDATE users SET credits = credits - ? WHERE id = ? AND credits >= ?",
                    (amount, user_id, amount),
                )
                if cur.rowcount != 1:
                    self._conn.execute("ROLLBACK")
                    return None
                bal = self._conn.execute(
                    "SELECT credits FROM users WHERE id = ?", (user_id,)
                ).fetchone()[0]
                self._conn.execute(
                    "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, -amount, bal, kind, ref, time.time()),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
            return int(bal)

    def _credit_in_txn(self, user_id: str, amount: int, kind: str, ref: str) -> int:
        """Body of a credit; the caller owns the transaction and the lock."""
        cur = self._conn.execute(
            "UPDATE users SET credits = credits + ? WHERE id = ?", (amount, user_id)
        )
        if cur.rowcount != 1:
            raise KeyError("no such user %r" % user_id)
        bal = self._conn.execute(
            "SELECT credits FROM users WHERE id = ?", (user_id,)
        ).fetchone()[0]
        self._conn.execute(
            "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, amount, bal, kind, ref, time.time()),
        )
        return int(bal)

    def credit(self, user_id: str, amount: int, kind: str, ref: str = "") -> int:
        amount = int(amount)
        with self._lock:
            self._begin()
            try:
                bal = self._credit_in_txn(user_id, amount, kind, ref)
                self._conn.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
            return bal

    def credit_once(
        self, event_id: str, event_type: str, user_id: str, amount: int, kind: str, ref: str = ""
    ) -> Optional[int]:
        """Record a webhook event and apply its credit in one transaction.

        Returns the new balance, or None when `event_id` was already
        processed. Either both rows land or neither does, so a crash between
        "seen" and "credited" cannot lose a purchase.
        """
        with self._lock:
            self._begin()
            try:
                try:
                    self._conn.execute(
                        "INSERT INTO webhook_events (id, type, received) VALUES (?, ?, ?)",
                        (event_id, event_type, time.time()),
                    )
                except sqlite3.IntegrityError:
                    self._conn.execute("ROLLBACK")
                    return None
                bal = self._credit_in_txn(user_id, int(amount), kind, ref)
                self._conn.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
            return bal

    def count_refunds(self, user_id: str, marker: str, since_seconds: float) -> int:
        """Refund rows for `user_id` whose ref contains `marker` in the window."""
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM ledger WHERE user_id = ? AND kind = 'refund'"
                    " AND ref LIKE ? AND created > ?",
                    (user_id, "%" + marker + "%", time.time() - since_seconds),
                ).fetchone()[0]
            )

    def balance(self, user_id: str) -> Optional[int]:
        row = self.user(user_id)
        return int(row["credits"]) if row else None

    def ledger(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, delta, balance_after, kind, ref, created FROM ledger"
                " WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, int(limit)),
            )
            return [dict(r) for r in cur.fetchall()]

    # -- webhooks --------------------------------------------------------

    def record_webhook(self, event_id: str, event_type: str) -> bool:
        """True if this event id has not been seen before (idempotency)."""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO webhook_events (id, type, received) VALUES (?, ?, ?)",
                    (event_id, event_type, time.time()),
                )
            except sqlite3.IntegrityError:
                return False
            return True
