"""SQLite persistence for users, sessions, API keys and the credit ledger.

One file, WAL mode, one connection guarded by a lock. That is adequate for a
single-process deployment, which is what the rest of the server assumes (the
LLM job store is an in-memory dict). Every balance change goes through
`charge()` or `credit()`, both of which write a ledger row in the same
transaction, so the ledger always reconciles to the balance.

`charge()` is atomic: it is one `UPDATE ... WHERE credits >= ?` and reports
failure by rowcount, so two concurrent requests cannot both spend the last
credit.

Plans sit on top of the ledger rather than beside it. A user on a plan has an
`allowance` (credits per UTC calendar month); `topup_if_due()` raises the
balance *up to* that number once per month and writes an `allowance` ledger
row, so purchased credits above the allowance are never clipped and a
renewal can never be applied twice (the guard is the `plan_period` column,
checked inside the same `BEGIN IMMEDIATE`).

Transactions nest: `_txn()` opens `BEGIN IMMEDIATE` only at depth zero, so a
webhook handler can record the event id and apply several plan changes as
one unit (`apply_once`). An exception anywhere inside rolls the whole unit
back.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

T = TypeVar("T")

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

#: Columns added after the first release. Applied with `ALTER TABLE ADD
#: COLUMN` when `PRAGMA table_info` does not list them, so a database from
#: before plans existed keeps working without a migration step.
_USER_COLUMNS = (
    ("plan", "TEXT NOT NULL DEFAULT ''"),
    ("plan_allowance", "INTEGER NOT NULL DEFAULT 0"),
    ("plan_until", "REAL"),
    ("plan_period", "TEXT"),
    ("stripe_customer_id", "TEXT"),
    ("stripe_subscription_id", "TEXT"),
    ("auth_user_id", "TEXT"),
)

#: Days past a subscription's period end during which the plan still counts
#: as active: Stripe retries a failed card for a while and the renewal
#: webhook can lag the period boundary.
GRACE_DAYS = 3


def month_key(t: float) -> str:
    """The UTC calendar month `t` falls in, as `YYYY-MM`."""
    return time.strftime("%Y-%m", time.gmtime(t))


class _Abort(Exception):
    """Raised inside `_txn` to roll back and return without an error."""


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
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(users)")}
            for name, decl in _USER_COLUMNS:
                if name not in cols:
                    self._conn.execute("ALTER TABLE users ADD COLUMN %s %s" % (name, decl))
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS users_customer ON users(stripe_customer_id)"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS users_auth ON users(auth_user_id)")
        self._depth = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _rollback(self) -> None:
        try:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    @contextlib.contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        """One write transaction, nestable.

        The outermost entry takes the lock and `BEGIN IMMEDIATE`; inner
        entries just join it. An exception rolls everything back at the
        outermost level, so a nested helper that fails cannot leave half of
        its caller's work committed. `_Abort` is the quiet way out: roll back
        and let the caller return a value.
        """
        with self._lock:
            if self._depth == 0:
                # A previous transaction that died mid-way must not wedge the
                # connection for every later write.
                self._rollback()
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self._conn
            except BaseException:
                self._depth -= 1
                if self._depth == 0:
                    self._rollback()
                raise
            else:
                self._depth -= 1
                if self._depth == 0:
                    self._conn.execute("COMMIT")

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
            try:
                with self._txn() as c:
                    c.execute(
                        "INSERT INTO users (id, email, credits, created) VALUES (?, ?, ?, ?)",
                        (uid, email, int(free_credits), now),
                    )
                    if free_credits > 0:
                        c.execute(
                            "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
                            " VALUES (?, ?, ?, 'signup', '', ?)",
                            (uid, int(free_credits), int(free_credits), now),
                        )
                    if ip:
                        c.execute(
                            "INSERT INTO signups (ip, user_id, created) VALUES (?, ?, ?)", (ip, uid, now)
                        )
            except sqlite3.IntegrityError:
                # Lost a race with a concurrent sign-up of the same email.
                row = _row(self._conn.execute("SELECT * FROM users WHERE email = ?", (email,)))
                if row is not None:
                    return row, False
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
            with self._txn() as c:
                c.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (user_id,))
                c.execute("UPDATE api_keys SET revoked = 1 WHERE user_id = ?", (user_id,))
                c.execute(
                    "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
                    " VALUES (?, ?, 0, 'deleted', '', ?)",
                    (user_id, -int(row["credits"]), now),
                )
                # The plan and the Stripe link go too; the site account that
                # was bridged to this row gets a fresh row next time it signs in.
                c.execute(
                    "UPDATE users SET email = ?, credits = 0, is_admin = 0, plan = '', plan_allowance = 0,"
                    " plan_until = NULL, plan_period = NULL, stripe_subscription_id = NULL, auth_user_id = NULL"
                    " WHERE id = ?",
                    ("deleted:%s" % user_id, user_id),
                )
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
                d = per_day.setdefault(day, {"signups": 0, "purchased": 0, "allowance": 0, "spent": 0, "refunded": 0})
                if kind == "signup":
                    d["signups"] += 1
                elif kind == "purchase":
                    d["purchased"] += int(delta)
                elif kind == "allowance":
                    d["allowance"] += int(delta)
                elif kind == "charge":
                    d["spent"] += -int(delta)
                elif kind == "refund":
                    d["refunded"] += int(delta)
            return {
                "users": int(total_users),
                "deleted_users": int(deleted),
                "credits_outstanding": int(balance),
                "credits_purchased": total("purchase"),
                "credits_allowance": total("allowance"),
                "subscribers": int(
                    c.execute(
                        "SELECT COUNT(*) FROM users WHERE plan != '' AND (plan_until IS NULL OR plan_until > ?)",
                        (time.time(),),
                    ).fetchone()[0]
                ),
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
        try:
            with self._txn() as c:
                cur = c.execute(
                    "UPDATE users SET credits = credits - ? WHERE id = ? AND credits >= ?",
                    (amount, user_id, amount),
                )
                if cur.rowcount != 1:
                    raise _Abort()
                bal = c.execute("SELECT credits FROM users WHERE id = ?", (user_id,)).fetchone()[0]
                c.execute(
                    "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, -amount, bal, kind, ref, time.time()),
                )
        except _Abort:
            return None
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
        with self._txn():
            return self._credit_in_txn(user_id, amount, kind, ref)

    def apply_once(self, event_id: str, event_type: str, fn: Callable[[], T]) -> Optional[T]:
        """Record a webhook event and run `fn` in the same transaction.

        Returns what `fn` returned, or None when `event_id` was already
        processed. Either the event row and every write `fn` makes land, or
        none of them do, so a crash between "seen" and "applied" cannot lose
        a purchase or a renewal, and Stripe's retries are harmless.
        """
        try:
            with self._txn() as c:
                try:
                    c.execute(
                        "INSERT INTO webhook_events (id, type, received) VALUES (?, ?, ?)",
                        (event_id, event_type, time.time()),
                    )
                except sqlite3.IntegrityError:
                    raise _Abort()
                return fn()
        except _Abort:
            return None

    def credit_once(
        self, event_id: str, event_type: str, user_id: str, amount: int, kind: str, ref: str = ""
    ) -> Optional[int]:
        """`apply_once` for a one-off purchase: the new balance, or None if seen."""
        return self.apply_once(
            event_id, event_type, lambda: self._credit_in_txn(user_id, int(amount), kind, ref)
        )

    # -- plans -----------------------------------------------------------

    @staticmethod
    def plan_active(user: Dict[str, Any], now: Optional[float] = None) -> bool:
        """Whether `user` is on a plan whose allowance still renews."""
        now = time.time() if now is None else now
        if not user.get("plan"):
            return False
        until = user.get("plan_until")
        return until is None or float(until) > now

    def topup_if_due(self, user_id: str, now: Optional[float] = None) -> Optional[int]:
        """Raise the balance up to the plan allowance, once per UTC month.

        Returns the credits added (0 when the balance was already at or
        above the allowance; the ledger row is written either way so the
        month shows as renewed), or None when nothing was due: no plan, an
        expired plan, or this month already applied. Called from the gate
        before every charge, so a renewal is never missed and never doubled.
        """
        now = time.time() if now is None else now
        period = month_key(now)
        try:
            with self._txn() as c:
                row = _row(c.execute("SELECT * FROM users WHERE id = ?", (user_id,)))
                if row is None or not self.plan_active(row, now) or row.get("plan_period") == period:
                    raise _Abort()
                delta = max(0, int(row["plan_allowance"]) - int(row["credits"]))
                cur = c.execute(
                    "UPDATE users SET credits = credits + ?, plan_period = ?"
                    " WHERE id = ? AND (plan_period IS NULL OR plan_period != ?)",
                    (delta, period, user_id, period),
                )
                if cur.rowcount != 1:
                    raise _Abort()
                bal = c.execute("SELECT credits FROM users WHERE id = ?", (user_id,)).fetchone()[0]
                c.execute(
                    "INSERT INTO ledger (user_id, delta, balance_after, kind, ref, created)"
                    " VALUES (?, ?, ?, 'allowance', ?, ?)",
                    (user_id, delta, bal, "plan:%s %s" % (row["plan"], period), now),
                )
        except _Abort:
            return None
        return delta

    def set_plan(
        self,
        user_id: str,
        plan: str,
        allowance: int,
        until: Optional[float],
        customer_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Put a user on `plan` and top them up at once.

        `until` None means lifetime. `plan_period` is reset so the first
        month's allowance lands immediately even if an earlier plan already
        renewed this month. Stripe ids are stored when given and kept
        otherwise. Returns the fresh user row.
        """
        with self._txn() as c:
            cur = c.execute(
                "UPDATE users SET plan = ?, plan_allowance = ?, plan_until = ?, plan_period = NULL,"
                " stripe_customer_id = COALESCE(?, stripe_customer_id),"
                " stripe_subscription_id = COALESCE(?, stripe_subscription_id) WHERE id = ?",
                (plan, int(allowance), until, customer_id or None, subscription_id or None, user_id),
            )
            if cur.rowcount != 1:
                raise KeyError("no such user %r" % user_id)
            self.topup_if_due(user_id, now)
            return _row(c.execute("SELECT * FROM users WHERE id = ?", (user_id,)))  # type: ignore[return-value]

    def set_plan_until(self, user_id: str, until: Optional[float]) -> None:
        """Move the plan's end. The name and allowance stay for display."""
        with self._txn() as c:
            c.execute("UPDATE users SET plan_until = ? WHERE id = ?", (until, user_id))

    def expire_plan(self, user_id: str, now: Optional[float] = None) -> None:
        """Stop the allowance now; the plan name is kept so the UI can say so."""
        self.set_plan_until(user_id, time.time() if now is None else now)

    def clear_plan(self, user_id: str) -> None:
        """Remove the plan entirely (operator `--off`). The customer id stays
        so the billing portal still opens."""
        with self._txn() as c:
            c.execute(
                "UPDATE users SET plan = '', plan_allowance = 0, plan_until = NULL, plan_period = NULL,"
                " stripe_subscription_id = NULL WHERE id = ?",
                (user_id,),
            )

    def set_stripe_ids(
        self, user_id: str, customer_id: Optional[str] = None, subscription_id: Optional[str] = None
    ) -> None:
        with self._txn() as c:
            c.execute(
                "UPDATE users SET stripe_customer_id = COALESCE(?, stripe_customer_id),"
                " stripe_subscription_id = COALESCE(?, stripe_subscription_id) WHERE id = ?",
                (customer_id or None, subscription_id or None, user_id),
            )

    def user_by_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        if not customer_id:
            return None
        with self._lock:
            return _row(
                self._conn.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,))
            )

    def user_by_subscription(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        if not subscription_id:
            return None
        with self._lock:
            return _row(
                self._conn.execute(
                    "SELECT * FROM users WHERE stripe_subscription_id = ?", (subscription_id,)
                )
            )

    # -- site accounts ---------------------------------------------------

    def user_by_auth_id(self, auth_user_id: str) -> Optional[Dict[str, Any]]:
        """The billing row linked to a `humanizer.api.auth` account id."""
        with self._lock:
            return _row(
                self._conn.execute(
                    "SELECT * FROM users WHERE auth_user_id = ? AND email NOT LIKE 'deleted:%'",
                    (str(auth_user_id),),
                )
            )

    def link_auth_user(self, user_id: str, auth_user_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET auth_user_id = ? WHERE id = ?", (str(auth_user_id), user_id)
            )

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
