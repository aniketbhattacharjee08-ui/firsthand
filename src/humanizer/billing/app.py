"""The public app: `humanizer.api.server.create_app` plus the paywall.

Nothing in `humanizer.api` is edited. This module builds the existing app,
registers the auth and billing routes, and wraps the whole thing in
`PaywallMiddleware`. Run it instead of `humanizer serve`::

    python -m humanizer.billing --host 0.0.0.0 --port 8000
    # or
    uvicorn --factory humanizer.billing.app:create_app --host 0.0.0.0 --port 8000

Merging later is one line: `server.run()` (or `cli.cmd_serve`) calls
`humanizer.billing.app.create_app` instead of `humanizer.api.server.create_app`.
Until then the two entry points coexist and the tests for each stay
independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from ..api import server
from .config import BillingConfig
from .gate import PaywallMiddleware
from .routes import register_billing_routes
from .store import Store


def install(app: FastAPI, config: Optional[BillingConfig] = None, store: Optional[Store] = None) -> FastAPI:
    """Add the paywall to any FastAPI app. Idempotent per app instance."""
    if getattr(app.state, "billing_installed", False):
        return app
    cfg = config or BillingConfig.from_env()
    db = store or Store(cfg.db_path)
    register_billing_routes(app, cfg, db)
    app.add_middleware(PaywallMiddleware, config=cfg, store=db)
    app.state.billing_installed = True
    return app


def create_app(
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
    default_reference: Optional[str] = None,
    config: Optional[BillingConfig] = None,
    store: Optional[Store] = None,
) -> FastAPI:
    """The full application with auth, billing and the paywall middleware.

    Built with `auth=False`: the magic-link login and `longhand_session`
    cookie here are the paywall's own, and the password accounts in
    `humanizer.api.auth` (the `humanizer serve` default) would gate the same
    routes twice with a different cookie.
    """
    app = server.create_app(
        reference_dir=reference_dir,
        web_dir=web_dir,
        default_reference=default_reference,
        auth=False,
    )
    return install(app, config=config, store=store)


def run(host: str = "127.0.0.1", port: int = 8000, reference: Optional[str] = None) -> None:
    import uvicorn

    cfg = BillingConfig.from_env()
    app = create_app(default_reference=reference, config=cfg)
    print(
        "%s on http://%s:%d  paywall=%s  stripe=%s  db=%s"
        % (cfg.product_name, host, port, "on" if cfg.enabled else "off", "on" if cfg.stripe_configured else "off", cfg.db_path)
    )
    # proxy_headers lets uvicorn honour X-Forwarded-* from Caddy or nginx; the
    # gate still only trusts them when LONGHAND_TRUST_PROXY is set.
    uvicorn.run(app, host=host, port=port, log_level="info", proxy_headers=cfg.trust_proxy)
