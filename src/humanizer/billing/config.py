"""Paywall configuration, read from the environment once at app build time.

Every knob is an environment variable with a `LONGHAND_` or `STRIPE_` prefix.
Nothing here is read at import time; `BillingConfig.from_env()` is called by
`billing.app.create_app`, so tests can pass a dict and the production process
can be configured entirely from `deploy/.env` loaded by the service manager.

The paywall is **off** unless `LONGHAND_PAYWALL` is a truthy value. Off means
every existing route behaves exactly as `humanizer.api.server` serves it; the
auth and billing routes are still mounted so the frontend can be built
against them before any money changes hands.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

_TRUTHY = {"1", "true", "yes", "on"}

#: Default database location. `data/cache/` is already in `.gitignore`, so a
#: local database cannot be committed by accident.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _REPO_ROOT / "data" / "cache" / "billing.db"

#: Disposable-email domains refused at sign-up. Extend with a file of one
#: domain per line in `LONGHAND_EMAIL_BLOCKLIST`.
DEFAULT_BLOCKED_DOMAINS: FrozenSet[str] = frozenset(
    """
    mailinator.com guerrillamail.com 10minutemail.com tempmail.com yopmail.com
    sharklasers.com trashmail.com dispostable.com temp-mail.org getnada.com
    maildrop.cc fakeinbox.com throwawaymail.com mohmal.com
    """.split()
)


@dataclass(frozen=True)
class Pack:
    """One purchasable credit pack, mapped to a Stripe Price."""

    name: str
    price_id: str
    credits: int
    label: str


def parse_packs(raw: str) -> List[Pack]:
    """Parse `STRIPE_PRICES`.

    Format: packs separated by `;`, fields by `:` in the order
    ``name:price_id:credits:label``. The label is free text shown by the
    frontend, e.g. ``starter:price_1Abc:50:$5 for 50 credits``.
    """
    packs: List[Pack] = []
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":", 3)
        if len(parts) < 3:
            raise ValueError(
                "STRIPE_PRICES entry %r must be name:price_id:credits[:label]" % chunk
            )
        name, price_id, credits = parts[0].strip(), parts[1].strip(), parts[2].strip()
        label = parts[3].strip() if len(parts) == 4 else ""
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", name):
            raise ValueError("pack name %r must be lowercase [a-z0-9_-]" % name)
        try:
            n = int(credits)
        except ValueError:
            raise ValueError("pack %r: credits %r is not an integer" % (name, credits))
        if n <= 0:
            raise ValueError("pack %r: credits must be positive" % name)
        packs.append(Pack(name=name, price_id=price_id, credits=n, label=label or name))
    return packs


PLAN_INTERVALS = ("month", "year", "lifetime")


@dataclass(frozen=True)
class Plan:
    """One subscription or lifetime plan, mapped to a Stripe Price.

    `allowance_credits` is the monthly allowance: the balance is topped up
    to this number once per UTC calendar month while the plan is active
    (`Store.topup_if_due`). A yearly plan is still a monthly allowance paid
    for twelve months at once; lifetime is the same allowance forever.
    """

    name: str
    price_id: str
    allowance_credits: int
    interval: str
    label: str

    @property
    def recurring(self) -> bool:
        return self.interval in ("month", "year")


def parse_plans(raw: str) -> List[Plan]:
    """Parse `STRIPE_PLANS`.

    Format: plans separated by `;`, fields by `:` in the order
    ``name:price_id:allowance_credits:interval[:label]`` where interval is
    `month`, `year` or `lifetime`, e.g.
    ``monthly:price_1Abc:300:month:$9.99 a month``.
    """
    plans: List[Plan] = []
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":", 4)
        if len(parts) < 4:
            raise ValueError(
                "STRIPE_PLANS entry %r must be name:price_id:allowance_credits:interval[:label]" % chunk
            )
        name, price_id, credits, interval = (x.strip() for x in parts[:4])
        label = parts[4].strip() if len(parts) == 5 else ""
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", name):
            raise ValueError("plan name %r must be lowercase [a-z0-9_-]" % name)
        try:
            n = int(credits)
        except ValueError:
            raise ValueError("plan %r: allowance %r is not an integer" % (name, credits))
        if n <= 0:
            raise ValueError("plan %r: allowance must be positive" % name)
        interval = interval.lower()
        if interval not in PLAN_INTERVALS:
            raise ValueError("plan %r: interval %r must be one of %s" % (name, interval, ", ".join(PLAN_INTERVALS)))
        if any(p.name == name for p in plans):
            raise ValueError("plan %r is listed twice" % name)
        plans.append(Plan(name=name, price_id=price_id, allowance_credits=n, interval=interval, label=label or name))
    return plans


def parse_rate(raw: str, default: Tuple[int, float]) -> Tuple[int, float]:
    """Parse ``N/second|minute|hour`` into (N, window_seconds)."""
    raw = (raw or "").strip().lower()
    if not raw:
        return default
    m = re.fullmatch(r"(\d+)\s*/\s*(second|sec|s|minute|min|m|hour|hr|h)", raw)
    if not m:
        raise ValueError("rate %r must look like 60/minute" % raw)
    n = int(m.group(1))
    unit = m.group(2)[0]
    window = {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return n, window


def load_blocklist(path: str) -> FrozenSet[str]:
    domains = set(DEFAULT_BLOCKED_DOMAINS)
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise ValueError("LONGHAND_EMAIL_BLOCKLIST %r is not a file" % path)
        for line in p.read_text().splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                domains.add(line)
    return frozenset(domains)


@dataclass(frozen=True)
class BillingConfig:
    enabled: bool = False
    #: Shown in emails and by the frontend. The page calls itself Vervly;
    #: the environment-variable prefix stays LONGHAND_ for stability.
    product_name: str = "Vervly"
    secret: str = ""
    db_path: Path = _DEFAULT_DB
    public_url: str = "http://127.0.0.1:8000"
    free_credits: int = 3
    session_days: int = 30
    magic_link_minutes: int = 15
    words_per_credit: int = 300
    max_body_bytes: int = 1_000_000
    #: Per-IP limit on every `/api/` request.
    api_rate: Tuple[int, float] = (120, 60.0)
    #: Per-IP limit on the login endpoints, which send email.
    auth_rate: Tuple[int, float] = (5, 60.0)
    #: New accounts one address may create in 24 hours (free-credit farming).
    signups_per_ip_per_day: int = 3
    blocked_domains: FrozenSet[str] = DEFAULT_BLOCKED_DOMAINS
    #: GPU work in flight at once, and how long a request waits for a slot
    #: before it is refused as busy (and refunded).
    max_inflight: int = 1
    queue_wait_s: float = 20.0
    #: How long a parked job (`POST /api/humanize/jobs`) may go unclaimed
    #: before its charge is refunded. Matches the server's job TTL.
    job_ttl_s: float = 900.0
    #: Refund a rewrite that returned the whole document unchanged, at most
    #: this many times per user per day (a deliberate no-op rewrite still
    #: occupies the GPU).
    refund_unchanged: bool = True
    unchanged_refunds_per_day: int = 3
    #: Requests allowed to wait for a GPU slot at once; beyond this the
    #: answer is busy immediately rather than tying up a thread.
    max_waiting: int = 8
    #: Directory for `requests.log`; empty logs through `logging` instead.
    log_dir: str = ""
    trust_proxy: bool = False
    dev_links: bool = False
    smtp_url: str = ""
    mail_from: str = "Vervly <no-reply@localhost>"
    admin_token: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    packs: Tuple[Pack, ...] = field(default_factory=tuple)
    plans: Tuple[Plan, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "BillingConfig":
        e = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            return (e.get(name) or default).strip()

        def get_int(name: str, default: int) -> int:
            raw = get(name)
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                raise ValueError("%s must be an integer, got %r" % (name, raw))

        def get_float(name: str, default: float) -> float:
            raw = get(name)
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                raise ValueError("%s must be a number, got %r" % (name, raw))

        def get_bool(name: str, default: bool) -> bool:
            raw = get(name).lower()
            if not raw:
                return default
            return raw in _TRUTHY

        enabled = get_bool("LONGHAND_PAYWALL", False)
        secret = get("LONGHAND_SECRET")
        if enabled and len(secret) < 32:
            raise ValueError(
                "LONGHAND_PAYWALL is on but LONGHAND_SECRET is missing or shorter "
                "than 32 characters. Generate one with: "
                "python -c 'import secrets; print(secrets.token_urlsafe(48))'"
            )
        public_url = get("LONGHAND_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")
        if enabled and get_bool("LONGHAND_DEV_LINKS", False) and public_url.startswith("https://"):
            raise ValueError(
                "LONGHAND_DEV_LINKS returns sign-in tokens in HTTP responses, which is "
                "account takeover for any email. It cannot be on with the paywall "
                "enabled on an https deployment. Unset it."
            )
        if not secret:
            # Off mode still signs dev sessions; an ephemeral secret is fine
            # because nothing behind it is gated.
            import secrets

            secret = secrets.token_urlsafe(48)

        db_raw = get("LONGHAND_DB")
        return cls(
            enabled=enabled,
            product_name=get("LONGHAND_PRODUCT_NAME", "Vervly"),
            secret=secret,
            db_path=Path(db_raw).expanduser() if db_raw else _DEFAULT_DB,
            public_url=public_url,
            free_credits=get_int("LONGHAND_FREE_CREDITS", 3),
            session_days=get_int("LONGHAND_SESSION_DAYS", 30),
            magic_link_minutes=get_int("LONGHAND_MAGIC_LINK_MINUTES", 15),
            words_per_credit=max(1, get_int("LONGHAND_WORDS_PER_CREDIT", 300)),
            max_body_bytes=get_int("LONGHAND_MAX_BODY_BYTES", 1_000_000),
            api_rate=parse_rate(get("LONGHAND_RATE_LIMIT"), (120, 60.0)),
            auth_rate=parse_rate(get("LONGHAND_AUTH_RATE_LIMIT"), (5, 60.0)),
            signups_per_ip_per_day=get_int("LONGHAND_SIGNUPS_PER_IP_PER_DAY", 3),
            blocked_domains=load_blocklist(get("LONGHAND_EMAIL_BLOCKLIST")),
            max_inflight=max(1, get_int("LONGHAND_MAX_INFLIGHT", 1)),
            queue_wait_s=max(0.0, get_float("LONGHAND_QUEUE_WAIT_S", 20.0)),
            job_ttl_s=max(1.0, get_float("LONGHAND_JOB_TTL_S", 900.0)),
            refund_unchanged=get_bool("LONGHAND_REFUND_UNCHANGED", True),
            unchanged_refunds_per_day=max(0, get_int("LONGHAND_UNCHANGED_REFUNDS_PER_DAY", 3)),
            max_waiting=max(1, get_int("LONGHAND_MAX_WAITING", 8)),
            log_dir=get("LONGHAND_LOG_DIR"),
            trust_proxy=get_bool("LONGHAND_TRUST_PROXY", False),
            dev_links=get_bool("LONGHAND_DEV_LINKS", False),
            smtp_url=get("LONGHAND_SMTP_URL"),
            mail_from=get("LONGHAND_MAIL_FROM", "Vervly <no-reply@localhost>"),
            admin_token=get("LONGHAND_ADMIN_TOKEN"),
            stripe_secret_key=get("STRIPE_SECRET_KEY"),
            stripe_webhook_secret=get("STRIPE_WEBHOOK_SECRET"),
            packs=tuple(parse_packs(get("STRIPE_PRICES"))),
            plans=tuple(parse_plans(get("STRIPE_PLANS"))),
        )

    @property
    def secure_cookies(self) -> bool:
        return self.public_url.startswith("https://")

    @property
    def stripe_configured(self) -> bool:
        return bool(self.stripe_secret_key and self.stripe_webhook_secret and (self.packs or self.plans))

    def pack(self, name: str) -> Optional[Pack]:
        for p in self.packs:
            if p.name == name:
                return p
        return None

    def plan(self, name: str) -> Optional[Plan]:
        for p in self.plans:
            if p.name == name:
                return p
        return None

    def public_packs(self) -> List[Dict[str, object]]:
        return [{"name": p.name, "credits": p.credits, "label": p.label} for p in self.packs]

    def public_plans(self) -> List[Dict[str, object]]:
        return [
            {
                "name": p.name,
                "label": p.label,
                "interval": p.interval,
                "allowance_credits": p.allowance_credits,
                "words_per_month": p.allowance_credits * self.words_per_credit,
            }
            for p in self.plans
        ]

    def public(self) -> Dict[str, object]:
        """The subset safe to show a browser (`GET /api/billing/health`)."""
        return {
            "product": self.product_name,
            "paywall": self.enabled,
            "free_credits": self.free_credits,
            "magic_link_minutes": self.magic_link_minutes,
            "words_per_credit": self.words_per_credit,
            "refund_unchanged": self.refund_unchanged,
            "max_inflight": self.max_inflight,
            "free_words": self.free_credits * self.words_per_credit,
            "stripe": self.stripe_configured,
            "packs": self.public_packs(),
            "plans": self.public_plans(),
            "mail": "smtp" if self.smtp_url else ("dev-links" if self.dev_links else "log"),
        }
