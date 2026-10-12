#!/usr/bin/env python
"""The authorship bench for Your voice: voice on versus voice off, for real.

Two questions, measured on the shipped pipeline in the product's own mode
(the free surrogate ranks candidates, GPTZero judges the output with the
owner's key):

1. Does the voice cost detector performance? The 9 AI bench paragraphs are
   rewritten twice, once plain and once in a borrowed author's voice, and
   GPTZero reads both outputs. Pass condition: the human rate with the voice
   on is no worse than with it off.
2. Does the output actually move toward the author? Each voice-on output is
   placed by fingerprint distance against every author's profile; the share
   whose nearest author is the one whose voice was used is the authorship
   accuracy, against chance of 1/K and against the voice-off outputs placed
   the same way. The held-out human paragraph of each author gives the
   ceiling the fingerprint can reach on real writing.

Authors are PMC articles from data/raw/pmc (one article, one author): the
first N usable paragraphs are the samples, the next one is held out. Twins
are prepared with the base model through the same `Preparer` the
product uses, so the whole path is exercised.

    source ~/.zshenv   # GPTZERO_API_KEY
    .venv/bin/python .voice/bench.py [--authors 4] [--samples 4] [--limit 9] [--tag voice-8x2]

Results: .voice/results/<tag>.json, rewritten after every paragraph; --resume
skips what is done. Python 3.9.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIP = ROOT / ".hip7b"
sys.path.insert(0, str(HIP))
import common  # noqa: E402  (the .hip7b bench helpers)

RESULTS = HERE / "results"
PMC = ROOT / "data" / "raw" / "pmc"

#: Articles with many paragraphs in the 80-450 word band, chosen so the
#: authors differ in subject and in how they write.
DEFAULT_AUTHORS = ["PMC8786337", "PMC8802128", "PMC8809424", "PMC8762392", "PMC9214341", "PMC9115682"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authors", type=int, default=4)
    ap.add_argument("--samples", type=int, default=4, help="sample paragraphs per author (plus one held out)")
    ap.add_argument("--limit", type=int, default=9, help="how many of the 9 AI bench paragraphs")
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--strength", default="balanced")
    ap.add_argument("--time-budget", type=float, default=600.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--off-only", action="store_true", help="only the voice-off condition")
    ap.add_argument("--judge", choices=("auto", "gptzero", "surrogate"), default="auto",
                    help="who reads the output: GPTZero with the owner's key, or the product's free surrogate; auto tries GPTZero first")
    return ap


def configure_env() -> Dict[str, str]:
    env = {
        "HUMANIZER_PROXY": "surrogate",
        "HUMANIZER_YARDSTICK": "surrogate",
        "HUMANIZER_PASS_THRESHOLD": "0.15",
        "HUMANIZER_REPAIR_ATTEMPTS": "0",
        "HUMANIZER_REPAIR_CRITIC": "0",
        "HUMANIZER_ASSUME_AI": "1",
        "HUMANIZER_SCORE_CANDIDATES": "1",
    }
    os.environ.update(env)
    return env


def usable_paragraphs(doc_id: str, lo: int = 80, hi: int = 450) -> List[str]:
    text = (PMC / ("%s.txt" % doc_id)).read_text()
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        n = len(para.split())
        if lo <= n <= hi and not para.lower().startswith(("table", "fig", "figure")):
            out.append(para)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tag = args.tag or ("voice-%dx%d-a%d-s%d" % (args.n_candidates, args.rounds, args.authors, args.samples))
    out_path = RESULTS / ("%s.json" % tag)
    RESULTS.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    env = configure_env()
    common.ensure_src_on_path()

    from humanizer.api import voice as api_voice
    from humanizer.detectors import yardstick
    from humanizer.detectors.gptzero import GPTZeroClient
    from humanizer.humanize import llm as llm_mod
    from humanizer.humanize import voice as voice_mod
    from humanizer.humanize.pipeline import PipelineConfig, humanize_llm

    yardstick.reset()
    client = GPTZeroClient(cache_dir=common.CACHE_DIR)
    judge = args.judge
    if judge in ("auto", "gptzero"):
        ok = client.available()
        if ok:
            try:
                client.score("The library was shut on Tuesday, so I walked. It rained; I did not mind. Nobody tells you how long the road is until you are on it, and by then the question is moot. We argued about whether the bridge counted. It did not. In 2019 the council spent 12,000 dollars on the sign alone.")
            except Exception as exc:  # noqa: BLE001
                print("GPTZero refused the key: %s" % str(exc)[:120])
                ok = False
        if not ok and judge == "gptzero":
            print("GPTZERO_API_KEY is missing or rejected (source ~/.zshenv; reset at app.gptzero.me/api). Nothing was run.")
            return 2
        judge = "gptzero" if ok else "surrogate"
    if judge == "surrogate":
        print("JUDGE: the free surrogate reads the outputs (GPTZero key unavailable). The authorship half is exact; the detector half is the surrogate's opinion, about 92 percent in step with GPTZero at 0.5 (research/24).", flush=True)
        surrogate = yardstick.proxy()

        class _SurrogateJudge:
            def cached(self, text):
                return None

            def score(self, text):
                return surrogate.score(text)

        client = _SurrogateJudge()

    # ---- authors: samples, held-out paragraph, twins, profile -------------
    store = api_voice.VoiceStore(Path(tempfile.mkdtemp(prefix="voice-bench-")) / "voice.sqlite")
    preparer = api_voice.Preparer(store, backend_factory=lambda: llm_mod.backend_for("freeform", llm_mod.FREEFORM_MODEL, adapter=""), run_lock=None)
    authors: List[Dict[str, Any]] = []
    for doc_id in DEFAULT_AUTHORS[: args.authors]:
        paras = usable_paragraphs(doc_id)
        if len(paras) < args.samples + 1:
            print("skipping %s: only %d usable paragraphs" % (doc_id, len(paras)))
            continue
        user_id = "bench:" + doc_id
        t0 = time.perf_counter()
        for i, para in enumerate(paras[: args.samples]):
            sample = store.add_sample(user_id, "%s %d" % (doc_id, i + 1), para, None, "preparing")
            preparer.prepare_now(user_id, sample["id"])
        ready = [s for s in store.samples(user_id) if s["status"] == "ready"]
        failed = [s for s in store.samples(user_id) if s["status"] != "ready"]
        authors.append({
            "id": doc_id,
            "user_id": user_id,
            "held_out": paras[args.samples],
            "n_ready": len(ready),
            "failed": [s["detail"] for s in failed],
            "profile": store.profile(user_id),
        })
        print("author %s: %d of %d twins ready in %.0fs%s" % (
            doc_id, len(ready), args.samples, time.perf_counter() - t0,
            ("; failed: " + failed[0]["detail"][:80]) if failed else ""), flush=True)
    if not authors:
        print("no authors")
        return 2

    def nearest_author(text: str) -> Dict[str, Any]:
        d = {a["id"]: voice_mod.profile_distance(a["profile"], text) for a in authors if a["profile"] is not None}
        best = min(d, key=d.get) if d else None
        return {"distances": {k: round(v, 4) for k, v in d.items()}, "nearest": best}

    # The ceiling: does a real held-out paragraph land on its own author?
    ceiling = []
    for a in authors:
        placed = nearest_author(a["held_out"])
        ceiling.append({"author": a["id"], "nearest": placed["nearest"], "correct": placed["nearest"] == a["id"],
                        "own_distance": placed["distances"].get(a["id"])})
    print("held-out ceiling: %d of %d placed on their own author" % (sum(1 for c in ceiling if c["correct"]), len(ceiling)), flush=True)

    # ---- the runs ---------------------------------------------------------
    items = common.load_bench(ai_only=True, limit=args.limit)
    rows: List[Dict[str, Any]] = []
    done = set()
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text())
        rows = list(prev.get("rows") or [])
        done = {(r["name"], r["condition"]) for r in rows}
    ledger = common.CostLedger(common.CACHE_DIR)
    meta: Dict[str, Any] = {
        "tag": tag, "env": env, "judge": judge, "n_candidates": args.n_candidates, "rounds": args.rounds, "strength": args.strength,
        "authors": [{"id": a["id"], "n_ready": a["n_ready"], "failed": a["failed"]} for a in authors],
        "ceiling": ceiling, "started": common.utc_now(),
        "base_model": llm_mod.FREEFORM_MODEL, "adapter": os.environ.get("HUMANIZER_BASE_ADAPTER") or llm_mod.DEFAULT_BASE_ADAPTER,
    }

    def write() -> None:
        meta["gptzero_cost"] = ledger.as_dict()
        out_path.write_text(json.dumps({"meta": meta, "rows": rows, "summary": summarize(rows, authors)}, indent=1))

    conditions = ["off"] if args.off_only else ["off", "on"]
    for i, it in enumerate(items):
        author = authors[i % len(authors)]
        for condition in conditions:
            if (it["name"], condition) in done:
                continue
            cfg = PipelineConfig(n_candidates=args.n_candidates, rounds=args.rounds, style="freeform",
                                 facts=it["facts"], time_budget_s=args.time_budget)
            if condition == "on":
                spec = api_voice.build_spec(store, author["user_id"], "r_bench")
                spec.strength = args.strength
                cfg.voice = spec
            t0 = time.perf_counter()
            res = humanize_llm(it["text"], config=cfg)
            secs = time.perf_counter() - t0
            before = common.score_with_retry(client, it["text"], ledger)
            after = common.score_with_retry(client, res.humanized, ledger)
            row = common.assemble_row(it, res.as_dict(), before, after, secs)
            placed = nearest_author(res.humanized)
            row.update({
                "condition": condition,
                "author": author["id"],
                "nearest_author": placed["nearest"],
                "author_correct": placed["nearest"] == author["id"],
                "own_distance": placed["distances"].get(author["id"]),
                "distances": placed["distances"],
                "voice": (res.summary or {}).get("voice"),
            })
            rows.append(row)
            write()
            print("%-22s %-3s %s  %s -> %s  own-dist %.3f  nearest %s%s  pass %d/%d  %.0fs%s" % (
                it["name"], condition, author["id"], common.fmt_p(row["p_before"]), common.fmt_p(row["p_after"]),
                row["own_distance"] or 0.0, row["nearest_author"], " *" if row["author_correct"] else "",
                row["n_passed"], row["n_candidates"], secs,
                (" fallback: " + row["fallback_reason"][:60]) if row["fallback_reason"] else ""), flush=True)

    meta["finished"] = common.utc_now()
    write()
    print()
    print(report(rows, authors, ceiling))
    print("wrote %s" % out_path)
    return 0


def summarize(rows: List[Dict[str, Any]], authors: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"chance": round(1.0 / max(1, len(authors)), 3)}
    for condition in ("off", "on"):
        sub = [r for r in rows if r["condition"] == condition]
        if not sub:
            continue
        scored = [r for r in sub if r["human_after"] is not None]
        out[condition] = {
            "n": len(sub),
            "human_after": sum(1 for r in scored if r["human_after"]),
            "n_scored": len(scored),
            "human_rate": round(sum(1 for r in scored if r["human_after"]) / len(scored), 3) if scored else None,
            "author_accuracy": round(sum(1 for r in sub if r["author_correct"]) / len(sub), 3),
            "mean_own_distance": round(sum(r["own_distance"] or 0.0 for r in sub) / len(sub), 4),
            "mean_p_after": round(sum(r["p_after"] or 0.0 for r in scored) / len(scored), 3) if scored else None,
            "fallbacks": sum(1 for r in sub if r["fallback_reason"]),
        }
    return out


def report(rows: List[Dict[str, Any]], authors: List[Dict[str, Any]], ceiling: List[Dict[str, Any]]) -> str:
    s = summarize(rows, authors)
    lines = ["| condition | GPTZero human | author placed right | mean distance to own author | fallbacks |", "|---|---|---|---|---|"]
    for condition in ("off", "on"):
        c = s.get(condition)
        if not c:
            continue
        lines.append("| voice %s | %d of %d | %d of %d (chance %.0f%%) | %.3f | %d |" % (
            condition, c["human_after"], c["n_scored"], round(c["author_accuracy"] * c["n"]), c["n"], s["chance"] * 100,
            c["mean_own_distance"], c["fallbacks"]))
    lines.append("")
    lines.append("held-out human paragraphs placed on their own author: %d of %d" % (
        sum(1 for c in ceiling if c["correct"]), len(ceiling)))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
