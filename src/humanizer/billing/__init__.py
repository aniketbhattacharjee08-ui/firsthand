"""Accounts, credits and a Stripe paywall for the public Longhand service.

This package was added without editing anything else in the repository. It
wraps the existing FastAPI app (`humanizer.api.server.create_app`) with:

* magic-link login and HttpOnly session cookies, or bearer API keys;
* a SQLite ledger of credits, atomic charge and refund;
* Stripe Checkout for credit packs and a signature-verified webhook;
* one ASGI middleware that rate-limits every `/api/` request and, when
  `LONGHAND_PAYWALL` is on, requires a signed-in user with enough credits for
  the LLM and guided engines and for any request that would spend the
  operator's GPTZero key. Measurement and the rule-based rewrite stay free.

Run with ``python -m humanizer.billing``. See `deploy/README.md` for the
environment variables and the hosting runbook, and `billing.app` for the
one-line merge into `humanizer serve` when the frontend is ready.
"""

from .app import create_app, install
from .config import BillingConfig

__all__ = ["create_app", "install", "BillingConfig"]
