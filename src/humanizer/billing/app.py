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

from ..api import auth as _auth_mod
from ..api import server
from .config import BillingConfig
from .gate import PaywallMiddleware
from .routes import register_billing_routes
from .store import Store


def _drop_auth_gate(app: FastAPI) -> None:
    """Remove `humanizer.api.auth`'s own 401 middleware when the paywall is on.

    The paywall is the gate: it knows which routes are free (measuring, the
    rule-based rewrite), accepts bearer keys and the magic-link cookie as
    well as the site's `rh_session`, and charges before the work runs. The
    auth module's blanket gate would answer 401 to a key-holder or a paid
    job resume that never carries `rh_session`, after the credit was taken.
    The pages, `/api/auth/*` and the session store stay.
    """
    keep = []
    for m in app.user_middleware:
        dispatch = getattr(m, "kwargs", {}).get("dispatch")
        if dispatch is not None and getattr(dispatch, "__module__", "") == _auth_mod.__name__:
            continue
        keep.append(m)
    app.user_middleware[:] = keep


def install(app: FastAPI, config: Optional[BillingConfig] = None, store: Optional[Store] = None) -> FastAPI:
    """Add the paywall to any FastAPI app. Idempotent per app instance."""
    if getattr(app.state, "billing_installed", False):
        return app
    cfg = config or BillingConfig.from_env()
    db = store or Store(cfg.db_path)
    register_billing_routes(app, cfg, db)
    if getattr(app.state, "auth_installed", False):
        _drop_auth_gate(app)
    app.add_middleware(PaywallMiddleware, config=cfg, store=db)
    app.state.billing_installed = True
    return app


def create_app(
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
    default_reference: Optional[str] = None,
    config: Optional[BillingConfig] = None,
    store: Optional[Store] = None,
    auth_db: Optional[Path] = None,
) -> FastAPI:
    """The full application with auth, billing and the paywall middleware.

    Built with `auth=True` so the shipped site's pages (`/`, `/signin`,
    `/signup`, `/app`) and password accounts (`/api/auth/signup|signin|
    signout|me`, cookie `rh_session`) exist. The paywall bridges that cookie
    to a billing user by email (`gate.PaywallMiddleware._authenticate`), so
    one sign-in both opens the app and carries the credits. The auth
    module's own 401 gate is dropped in `install()`; see `_drop_auth_gate`.
    The magic-link login and `longhand_session` cookie keep working for API
    users and the existing tests.
    """
    app = server.create_app(
        reference_dir=reference_dir,
        web_dir=web_dir,
        default_reference=default_reference,
        auth=True,
        auth_db=auth_db,
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
