"""`python -m humanizer.billing`: serve the paywalled app, or administer it.

    python -m humanizer.billing [serve] --host 0.0.0.0 --port 8000
    python -m humanizer.billing check
    python -m humanizer.billing admin user <email>
    python -m humanizer.billing admin grant <email> <credits> [--note promo]
    python -m humanizer.billing admin set-admin <email> [--off]
    python -m humanizer.billing admin plan <email> <name> [--until YYYY-MM-DD | --lifetime | --off]
    python -m humanizer.billing admin delete <email>
    python -m humanizer.billing admin stats [--days 14]
    python -m humanizer.billing admin sweep    (refund nothing; prints parked-job policy)

The admin commands open the same SQLite file the server uses (`LONGHAND_DB`)
and are safe while it runs: the store is WAL-mode and every write is one
short transaction.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from typing import List, Optional

from .config import BillingConfig
from .store import Store

_SUBCOMMANDS = ("serve", "check", "admin")


def _config() -> BillingConfig:
    try:
        return BillingConfig.from_env()
    except ValueError as exc:
        print("config error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)


def cmd_serve(args: argparse.Namespace) -> int:
    from .app import run

    run(host=args.host, port=args.port, reference=args.reference)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    cfg = _config()
    for k, v in cfg.public().items():
        print("%-18s %s" % (k, v))
    print("%-18s %s" % ("db", cfg.db_path))
    print("%-18s %s" % ("public_url", cfg.public_url))
    print("%-18s %s" % ("log_dir", cfg.log_dir or "(stdout)"))
    print("%-18s %s / wait %.0fs" % ("gpu_slots", cfg.max_inflight, cfg.queue_wait_s))
    print("%-18s %s" % ("trust_proxy", cfg.trust_proxy))
    if cfg.enabled and cfg.dev_links:
        print("WARNING: LONGHAND_DEV_LINKS is on with the paywall enabled; never in production.", file=sys.stderr)
    if cfg.enabled and not cfg.secure_cookies:
        print("WARNING: LONGHAND_PUBLIC_URL is not https; session cookies will not be Secure.", file=sys.stderr)
    return 0


def _fmt_time(t: Optional[float]) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)) if t else "-"


def cmd_admin(args: argparse.Namespace) -> int:
    cfg = _config()
    store = Store(cfg.db_path)
    try:
        if args.action == "user":
            u = store.user_by_email(args.email)
            if u is None:
                print("no such user", file=sys.stderr)
                return 1
            print(json.dumps({**u, "created": _fmt_time(u.get("created")), "last_login": _fmt_time(u.get("last_login"))}, indent=2))
            for e in store.ledger(u["id"], limit=args.limit):
                print("%s  %+5d  bal %5d  %-9s %s" % (_fmt_time(e["created"]), e["delta"], e["balance_after"], e["kind"], e["ref"]))
            return 0
        if args.action == "grant":
            u = store.user_by_email(args.email)
            if u is None:
                u = store.get_or_create_user(args.email, 0)
                print("created %s" % u["id"])
            bal = store.credit(u["id"], args.credits, kind="manual", ref=args.note or "grant")
            print("balance %d" % bal)
            return 0
        if args.action == "set-admin":
            u = store.user_by_email(args.email)
            if u is None:
                u = store.get_or_create_user(args.email, cfg.free_credits)
                print("created %s" % u["id"])
            store.set_admin(u["id"], not args.off)
            print("%s admin=%s" % (u["email"], not args.off))
            return 0
        if args.action == "plan":
            u = store.user_by_email(args.email)
            if u is None:
                u = store.get_or_create_user(args.email, 0)
                print("created %s" % u["id"])
            if args.off:
                store.clear_plan(u["id"])
                print("%s plan=off" % u["email"])
                return 0
            plan = cfg.plan(args.name)
            allowance = args.allowance if args.allowance is not None else (plan.allowance_credits if plan else None)
            if allowance is None:
                print("plan %r is not in STRIPE_PLANS; pass --allowance N" % args.name, file=sys.stderr)
                return 2
            if args.lifetime:
                until = None
            elif args.until:
                try:
                    # Inclusive: the plan lasts through the last second of that day.
                    until = float(calendar.timegm(time.strptime(args.until, "%Y-%m-%d"))) + 86400.0 - 1.0
                except ValueError:
                    print("--until must be YYYY-MM-DD", file=sys.stderr)
                    return 2
            elif plan is not None and plan.interval == "lifetime":
                until = None
            else:
                # No end given: one period of the configured interval, with
                # the same grace a Stripe renewal gets.
                days = 366 if plan is not None and plan.interval == "year" else 31
                until = time.time() + (days + 3) * 86400
            row = store.set_plan(u["id"], args.name, allowance, until)
            print("%s plan=%s allowance=%d until=%s balance=%d" % (
                u["email"], args.name, allowance, "lifetime" if until is None else _fmt_time(until), row["credits"]))
            return 0
        if args.action == "delete":
            u = store.user_by_email(args.email)
            if u is None:
                print("no such user", file=sys.stderr)
                return 1
            store.delete_user(u["id"])
            print("deleted %s" % u["id"])
            return 0
        if args.action == "stats":
            st = store.stats(days=args.days)
            days = st.pop("days")
            for k, v in st.items():
                print("%-20s %s" % (k, v))
            if days:
                print("\n%-12s %8s %10s %10s %8s %9s" % ("day", "signups", "purchased", "allowance", "spent", "refunded"))
                for day in sorted(days):
                    d = days[day]
                    print("%-12s %8d %10d %10d %8d %9d" % (day, d["signups"], d["purchased"], d.get("allowance", 0), d["spent"], d["refunded"]))
            return 0
    finally:
        store.close()
    print("unknown admin action", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m humanizer.billing", description="Longhand: serve or administer the paywalled API.")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="run the HTTP API with the paywall installed")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reference", help="default reference distribution for /api/analyze")
    s.set_defaults(func=cmd_serve)

    c = sub.add_parser("check", help="validate the environment and exit")
    c.set_defaults(func=cmd_check)

    a = sub.add_parser("admin", help="operator commands against the billing database")
    asub = a.add_subparsers(dest="action")
    u = asub.add_parser("user", help="show a user and their ledger")
    u.add_argument("email")
    u.add_argument("--limit", type=int, default=20)
    g = asub.add_parser("grant", help="add credits to a user")
    g.add_argument("email")
    g.add_argument("credits", type=int)
    g.add_argument("--note", default="")
    m = asub.add_parser("set-admin", help="make a user an admin (or --off)")
    m.add_argument("email")
    m.add_argument("--off", action="store_true")
    pl = asub.add_parser("plan", help="put a user on a plan, or take them off one")
    pl.add_argument("email")
    pl.add_argument("name", nargs="?", default="", help="plan name from STRIPE_PLANS (any name with --allowance)")
    pl.add_argument("--until", default="", help="last day, YYYY-MM-DD (UTC, inclusive)")
    pl.add_argument("--lifetime", action="store_true", help="no end date")
    pl.add_argument("--off", action="store_true", help="remove the plan")
    pl.add_argument("--allowance", type=int, default=None, help="credits per month; defaults to the configured plan's")
    d = asub.add_parser("delete", help="delete (anonymise) a user")
    d.add_argument("email")
    t = asub.add_parser("stats", help="totals and a per-day table")
    t.add_argument("--days", type=int, default=14)
    a.set_defaults(func=cmd_admin)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `python -m humanizer.billing --port 8000` keeps working: default to serve.
    if not argv or argv[0].startswith("-"):
        if "--check" in argv:
            argv = ["check"]
        else:
            argv = ["serve"] + argv
    args = build_parser().parse_args(argv)
    if args.command == "admin" and not getattr(args, "action", None):
        build_parser().parse_args(["admin", "--help"])
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
