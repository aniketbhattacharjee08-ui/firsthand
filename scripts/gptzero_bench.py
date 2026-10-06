"""Score the shipped pipeline against GPTZero itself, when a key is available.

The product target is a GPTZero "human" verdict (research/06, /07, /13), and
research/08 §2.2 is blunt that the local proxy's probability carries no
information about that verdict until calibrated. This script is the
calibration step: it runs the shipped LLM pipeline on a set of paragraphs,
scores the before and after text with GPTZero, and writes rows that
`humanizer calibrate` can read.

Usage
-----
    export GPTZERO_API_KEY=...          # without it the script only prices the run
    python scripts/gptzero_bench.py texts.json [--facts facts.json] [--out rows.json]

`texts.json` is a list of {"name": ..., "text": ...}. `facts.json`, optional,
maps name -> author-supplied facts (see PipelineConfig.facts). Every GPTZero
response is cached on disk under data/cache/gptzero, so re-runs are free.

Cost: about $0.00046 per word, charged for the before AND the after text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanizer.detectors.gptzero import GPTZeroClient, estimate_cost  # noqa: E402


def _score_when_reachable(client, text, max_wait_s: float = 1800.0):
    """Score, retrying through network outages for up to `max_wait_s`."""
    import time

    deadline = time.monotonic() + max_wait_s
    while True:
        try:
            return client.score(text)
        except Exception as exc:  # noqa: BLE001 - connection errors of every shape
            if time.monotonic() > deadline or "40" in str(exc)[:3]:
                raise
            print("  network: %s; retrying in 60 s" % type(exc).__name__, flush=True)
            time.sleep(60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", help="JSON list of {name, text}")
    ap.add_argument("--facts", help="JSON object name -> facts text", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "cache" / "gptzero_bench_rows.json"))
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--style", default="register")
    args = ap.parse_args()

    items = json.loads(Path(args.texts).read_text())
    facts = json.loads(Path(args.facts).read_text()) if args.facts else {}
    texts = [it["text"] for it in items]
    print("paragraphs: %d, estimated GPTZero cost for before+after: $%.2f" % (len(texts), 2 * estimate_cost(texts)))

    client = GPTZeroClient()
    if not client.available():
        print("GPTZERO_API_KEY is not set. Nothing was scored. Set it and re-run; "
              "the yardstick in the app switches to GPTZero automatically when it is present.")
        return 2

    from humanizer.humanize.pipeline import PipelineConfig, humanize_llm

    rows = []
    for it in items:
        cfg = PipelineConfig(n_candidates=args.n_candidates, rounds=args.rounds,
                             style=args.style, facts=facts.get(it["name"], ""))
        res = humanize_llm(it["text"], config=cfg)
        # This machine's DNS drops for minutes at a time; a bench must wait
        # the outage out rather than lose the paragraphs already rewritten.
        before = _score_when_reachable(client, it["text"])
        after = _score_when_reachable(client, res.humanized)
        row = {
            "name": it["name"],
            "changed": res.humanized.strip() != it["text"].strip(),
            "gptzero_before": before.ai_probability,
            "gptzero_after": after.ai_probability,
            "gptzero_label_before": getattr(before, "label", None),
            "gptzero_label_after": getattr(after, "label", None),
            "local_before": res.summary.get("ai_probability_before"),
            "local_after": res.summary.get("ai_probability_after"),
            "unverified_specifics": res.summary.get("unverified_specifics", []),
            # the shape `humanizer calibrate` reads
            "local_human_score": 1.0 - (res.paragraphs[0].ai_probability_after
                                        if res.paragraphs and res.paragraphs[0].ai_probability_after is not None
                                        else 1.0),
            "gptzero_human": (after.ai_probability is not None and after.ai_probability < 0.5),
            # The repair stage (humanize/repair.py): attempts, accepted
            # rewrites, whether this paragraph was rescued by it, and the log.
            "repairs_attempted": res.summary.get("repairs_attempted", 0),
            "repairs_accepted": res.summary.get("repairs_accepted", 0),
            "repairs_rescued": res.summary.get("repairs_rescued", 0),
            "repair_log": res.summary.get("repair_log", []),
            "seconds": res.summary.get("seconds"),
        }
        rows.append(row)
        print("%-24s changed=%s gptzero %.3f -> %.3f (%s -> %s) repairs %d/%d%s %.0fs" % (
            it["name"], row["changed"], before.ai_probability or float("nan"),
            after.ai_probability or float("nan"), row["gptzero_label_before"], row["gptzero_label_after"],
            row["repairs_accepted"], row["repairs_attempted"],
            " rescued" if row["repairs_rescued"] else "", row["seconds"] or 0.0), flush=True)
        for rec in row["repair_log"]:
            print("    try %d: %s -> %s -> %s" % (rec["attempt"], rec["diagnosis"], rec["action"], rec["result"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    flips = sum(1 for r in rows if (r["gptzero_before"] or 0) >= 0.5 and (r["gptzero_after"] or 1) < 0.5)
    print("GPTZero verdict flips: %d of %d. Rows written to %s (feed them to `humanizer calibrate`)." % (flips, len(rows), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
