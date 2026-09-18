"""Concurrency and refund rehearsal against a running server.

    .venv/bin/python deploy/loadtest.py http://127.0.0.1:8000 --concurrency 3

Needs `LONGHAND_DEV_LINKS=1` on the server (staging only) so it can sign in
without email, and enough free credits. It fires N simultaneous LLM
requests, prints each status and the credit headers, then reads the ledger
and checks that exactly one request per available GPU slot was charged and
every busy or refused request was refunded. It also times the SSE stream:
the first frame should arrive well before the last, or a proxy is buffering.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import requests

SAMPLE = (
    "In today's rapidly evolving landscape, organisations must leverage robust "
    "frameworks to navigate multifaceted challenges. Furthermore, stakeholders "
    "should delve into comprehensive strategies that underscore the importance of "
    "resilience. Consequently, a holistic approach is pivotal for sustained success. "
) * 3


def sign_in(base: str, email: str) -> requests.Session:
    s = requests.Session()
    r = s.post(base + "/api/auth/request-link", json={"email": email}, timeout=20)
    r.raise_for_status()
    token = r.json().get("dev_token")
    if not token:
        sys.exit("server did not return dev_token; set LONGHAND_DEV_LINKS=1 on staging")
    s.post(base + "/api/auth/verify", json={"token": token}, timeout=20).raise_for_status()
    return s


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base", nargs="?", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--email", default="loadtest@example.com")
    p.add_argument("--skip-sse", action="store_true")
    args = p.parse_args()
    base = args.base.rstrip("/")

    s = sign_in(base, args.email)
    me = s.get(base + "/api/me", timeout=20).json()["user"]
    print("signed in as %s with %d credits" % (me["email"], me["credits"]))
    health = s.get(base + "/api/billing/health", timeout=20).json()
    print("paywall=%s max_inflight=%s words_per_credit=%s" % (health["paywall"], health.get("max_inflight"), health["words_per_credit"]))

    results = [None] * args.concurrency

    def fire(i: int) -> None:
        t = time.time()
        try:
            r = s.post(base + "/api/humanize/llm", json={"text": SAMPLE, "n_candidates": 2, "time_budget_s": 60}, timeout=300)
            results[i] = (r.status_code, r.headers.get("x-longhand-credits-charged"), r.headers.get("x-longhand-credits-balance"), round(time.time() - t, 1), (r.json().get("error") if r.headers.get("content-type", "").startswith("application/json") else None))
        except Exception as exc:  # noqa: BLE001
            results[i] = ("EXC", None, None, round(time.time() - t, 1), str(exc))

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(args.concurrency)]
    for th in threads:
        th.start()
        time.sleep(0.2)
    for th in threads:
        th.join()
    print("\n%-4s %-7s %-8s %-8s %-7s %s" % ("#", "status", "charged", "balance", "secs", "error"))
    for i, r in enumerate(results):
        print("%-4d %-7s %-8s %-8s %-7s %s" % (i + 1, r[0], r[1], r[2], r[3], r[4] or ""))

    ledger = s.get(base + "/api/billing/ledger", timeout=20).json()
    charges = [e for e in ledger["entries"] if e["kind"] == "charge"]
    refunds = [e for e in ledger["entries"] if e["kind"] == "refund"]
    ok = [r for r in results if r[0] == 200]
    print("\nledger: %d charges, %d refunds, balance %d" % (len(charges), len(refunds), ledger["balance"]))
    print("expected: %d successful (one per slot per ~minute), the rest 503 busy and refunded" % max(1, int(health.get("max_inflight") or 1)))
    print("check: successes %d, busy %d" % (len(ok), sum(1 for r in results if r[0] == 503)))

    if args.skip_sse:
        return 0
    print("\nSSE timing (a buffering proxy shows first ~= last):")
    t0 = time.time()
    first = None
    frames = 0
    with s.post(base + "/api/humanize/stream", json={"text": SAMPLE, "n_candidates": 2, "time_budget_s": 60}, stream=True, timeout=300) as r:
        print("status", r.status_code, "charged", r.headers.get("x-longhand-credits-charged"))
        for line in r.iter_lines():
            if line.startswith(b"event:"):
                frames += 1
                if first is None:
                    first = time.time() - t0
    print("first frame %.2fs, last frame %.2fs, %d frames" % (first or -1, time.time() - t0, frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
