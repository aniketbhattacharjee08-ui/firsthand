#!/usr/bin/env python
"""Bench one base checkpoint (plus optional LoRA) through the shipped pipeline,
scoring GPTZero pass rate and writing quality together.

    .venv/bin/python .hip7b/bench.py --model mlx-community/Qwen2.5-7B-4bit \\
        [--adapter .hip7b/adapters/step300] --mode free|ownkey --tag hip7b-300 \\
        [--n-candidates 8] [--rounds 2] [--limit N] [--ai-only]

Modes (research/24 6.13):
  ownkey  GPTZero judges and ranks candidates (HUMANIZER_PROXY=gptzero,
          HUMANIZER_YARDSTICK=gptzero, pass threshold 0.5). About $1 per
          paragraph at 8 candidates x 2 rounds. Measured baseline 7-8 of 9.
  free    the local surrogate ranks (HUMANIZER_PROXY=surrogate, threshold 0.15);
          GPTZero is called only on the before/after text here, about $0.14 per
          paragraph, mostly cached. Measured baseline 5-7 of 9.

The environment variables the pipeline reads are set *before* any humanizer
module is imported, because `humanize/llm.py` reads HUMANIZER_BASE_MODEL at
import time. Everything model-shaped is imported inside main(); `--help` and
`--selftest` never touch mlx, torch or the network.

Results: .hip7b/results/<tag>.json, rewritten after every paragraph so a crash
keeps what was measured; `--resume` skips paragraphs already in that file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-4bit"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="MLX base checkpoint id or path (HUMANIZER_BASE_MODEL)")
    ap.add_argument("--adapter", default=None, help="LoRA directory on the base checkpoint (HUMANIZER_BASE_ADAPTER); omit for the plain base")
    ap.add_argument("--mode", choices=("free", "ownkey"), default="ownkey")
    ap.add_argument("--tag", required=False, default=None, help="results name; default derived from model/adapter/mode")
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N bench paragraphs")
    ap.add_argument("--ai-only", action="store_true", help="skip the 5 human PMC controls")
    ap.add_argument("--texts", default=str(common.DEFAULT_TEXTS))
    ap.add_argument("--facts", default=str(common.DEFAULT_FACTS))
    ap.add_argument("--style", default="freeform", help="PipelineConfig.style (freeform = base checkpoint few-shot)")
    ap.add_argument("--objective", default="evade", choices=("evade", "faithful"))
    ap.add_argument("--proxy", default="surrogate", help="free mode candidate scorer (HUMANIZER_PROXY)")
    ap.add_argument("--pass-threshold", type=float, default=None, help="HUMANIZER_PASS_THRESHOLD; default 0.5 ownkey, 0.15 free")
    ap.add_argument("--repair-attempts", type=int, default=0,
                    help="HUMANIZER_REPAIR_ATTEMPTS; 0 (default) matches the research/24 6.8 baseline runs")
    ap.add_argument("--retry-adapter", default=None, help="HUMANIZER_BASE_ADAPTER_RETRY; unset by default so only --adapter is measured")
    ap.add_argument("--time-budget", type=float, default=600.0, help="PipelineConfig.time_budget_s per paragraph")
    ap.add_argument("--quality-weight", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="skip paragraphs already in the results file")
    ap.add_argument("--out", default=None, help="results path; default .hip7b/results/<tag>.json")
    ap.add_argument("--selftest", action="store_true", help="exercise the metric and table code on a fake result; no model, no network")
    return ap


def _default_tag(args: argparse.Namespace) -> str:
    base = Path(args.model).name.replace("/", "-")
    ad = ("+" + Path(args.adapter).name) if args.adapter else ""
    return "%s%s-%s" % (base, ad, args.mode)


def configure_env(args: argparse.Namespace) -> Dict[str, str]:
    """Set the pipeline's environment. Must run before importing humanizer."""
    env: Dict[str, str] = {
        "HUMANIZER_BASE_MODEL": args.model,
        # "" means the plain base: default_backend() does `environ.get(...) or None`
        "HUMANIZER_BASE_ADAPTER": args.adapter or "",
        "HUMANIZER_BASE_ADAPTER_RETRY": args.retry_adapter or "",
        "HUMANIZER_REPAIR_ATTEMPTS": str(args.repair_attempts),
        "HUMANIZER_REPAIR_CRITIC": "0",
    }
    if args.mode == "ownkey":
        env["HUMANIZER_PROXY"] = "gptzero"
        env["HUMANIZER_YARDSTICK"] = "gptzero"
        env["HUMANIZER_PASS_THRESHOLD"] = str(args.pass_threshold if args.pass_threshold is not None else 0.5)
    else:
        env["HUMANIZER_PROXY"] = args.proxy
        env["HUMANIZER_YARDSTICK"] = args.proxy
        env["HUMANIZER_PASS_THRESHOLD"] = str(args.pass_threshold if args.pass_threshold is not None else 0.15)
    os.environ.update(env)
    return env


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    tag = args.tag or _default_tag(args)
    out_path = Path(args.out) if args.out else common.RESULTS_DIR / ("%s.json" % tag)

    # The GPTZero client's default cache path is relative (data/cache/gptzero),
    # and the pipeline builds its own client in own-key mode: run from the root.
    os.chdir(common.ROOT)
    env = configure_env(args)
    common.ensure_src_on_path()

    # ---- model-side imports, only now -------------------------------------
    from humanizer.detectors import yardstick
    from humanizer.detectors.gptzero import GPTZeroClient
    yardstick.reset()
    client = GPTZeroClient(cache_dir=common.CACHE_DIR)
    if not client.available():
        print("GPTZERO_API_KEY is not set (try: source ~/.zshenv). Both modes score the result with GPTZero; nothing was run.")
        return 2
    if args.mode == "ownkey" and yardstick.configured_name() != "gptzero":
        print("yardstick did not resolve to gptzero: %s" % yardstick.configured_name())
        return 2
    from humanizer.humanize import llm as llm_mod
    from humanizer.humanize.pipeline import PipelineConfig, humanize_llm
    if llm_mod.FREEFORM_MODEL != args.model:
        print("humanizer.humanize.llm was imported before the environment was set "
              "(FREEFORM_MODEL=%s, wanted %s). Run this script directly." % (llm_mod.FREEFORM_MODEL, args.model))
        return 2

    items = common.load_bench(Path(args.texts), Path(args.facts), ai_only=args.ai_only, limit=args.limit)
    ledger = common.CostLedger(common.CACHE_DIR)
    rows: List[Dict[str, Any]] = []
    done = set()
    if args.resume and out_path.exists():
        prev = common.load_results(out_path)
        rows = list(prev.get("paragraphs") or [])
        done = {r["name"] for r in rows}
        if prev.get("gptzero_cost"):
            ledger.calls = prev["gptzero_cost"].get("calls", 0)
            ledger.misses = prev["gptzero_cost"].get("cache_misses", 0)
            ledger.words_charged = prev["gptzero_cost"].get("words_charged", 0)
    todo = [it for it in items if it["name"] not in done]

    meta: Dict[str, Any] = {
        "tag": tag,
        "mode": args.mode,
        "model": args.model,
        "adapter": args.adapter,
        "retry_adapter": args.retry_adapter,
        "config": {
            "style": args.style, "objective": args.objective, "n_candidates": args.n_candidates,
            "rounds": args.rounds, "pass_threshold": float(env["HUMANIZER_PASS_THRESHOLD"]),
            "repair_attempts": args.repair_attempts, "time_budget_s": args.time_budget,
            "quality_weight": args.quality_weight, "proxy": env["HUMANIZER_PROXY"],
            "yardstick": env["HUMANIZER_YARDSTICK"],
        },
        "env": env,
        "texts": str(args.texts),
        "facts": str(args.facts),
        "started": common.utc_now(),
        "n_paragraphs_planned": len(items),
    }

    words_uncached = sum(it_words for it_words in (
        common.n_words(it["text"]) for it in todo if not common.cache_has(it["text"])))
    print("bench %s: %d paragraphs (%d to run), mode=%s model=%s adapter=%s" % (
        tag, len(items), len(todo), args.mode, args.model, args.adapter or "none"))
    print("  before-texts not yet cached: %d words ($%.2f); after-texts: ~$%.2f; %s" % (
        words_uncached, words_uncached * common.COST_PER_WORD_USD,
        sum(common.n_words(it["text"]) for it in todo) * common.COST_PER_WORD_USD,
        "candidates ranked by GPTZero, about $1/paragraph at 8x2" if args.mode == "ownkey"
        else "candidates ranked by the free surrogate"), flush=True)

    for it in todo:
        cfg = PipelineConfig(
            n_candidates=args.n_candidates, rounds=args.rounds, style=args.style,
            facts=it["facts"], objective=args.objective, time_budget_s=args.time_budget,
            quality_weight=args.quality_weight,
        )
        snap = ledger.snapshot()
        t0 = time.perf_counter()
        res = humanize_llm(it["text"], config=cfg)
        secs = time.perf_counter() - t0
        # own-key mode: the pipeline's own GPTZero calls (candidates, before,
        # after) land in the cache; charge every new file exactly once.
        ledger.charge_new_files(snap)
        before = common.score_with_retry(client, it["text"], ledger)
        after = common.score_with_retry(client, res.humanized, ledger)
        row = common.assemble_row(it, res.as_dict(), before, after, secs)
        rows.append(row)
        meta["gptzero_cost"] = ledger.as_dict()
        common.write_results(out_path, meta, rows, None)
        q = row["quality"]
        print("%-22s %s %s -> %s (%s -> %s) ov %.2f len %.2f FRE %s>%s spec %d pass %d/%d %.0fs%s" % (
            it["name"], "chg" if row["changed"] else "   ",
            common.fmt_p(row["p_before"]), common.fmt_p(row["p_after"]),
            row["label_before"], row["label_after"], q["content_overlap"] or 0.0, q["length_ratio"] or 0.0,
            common.fmt_f(q["fre_source"], "%.0f"), common.fmt_f(q["fre_after"], "%.0f"),
            q["n_new_specifics"], row["n_passed"], row["n_candidates"], secs,
            (" fallback: " + row["fallback_reason"]) if row["fallback_reason"] else ""), flush=True)

    summary = common.summarize(rows, ledger, mode=args.mode)
    meta["finished"] = common.utc_now()
    meta["gptzero_cost"] = ledger.as_dict()
    common.write_results(out_path, meta, rows, summary)
    print()
    print(common.paragraph_table(rows))
    print()
    print(common.summary_line(summary))
    print("wrote %s" % out_path)
    return 0


# ------------------------------------------------------------------ selftest


def _fake_result(source: str, chosen: str, cands: List[Dict[str, Any]], winner: Optional[int]) -> Dict[str, Any]:
    """The shape of `LlmHumanizeResult.as_dict()` that assemble_row reads."""
    return {
        "original": source,
        "humanized": chosen,
        "model": "fake-base",
        "backend": "echo",
        "summary": {
            "ai_probability_before": 0.9, "ai_probability_after": 0.1,
            "candidate_scorer": "fake", "yardstick": {"name": "fake", "kind": "local-proxy"},
            "gate_rejections": {g: sum(1 for c in cands if g in c["gates"]) for c in cands for g in c["gates"]},
            "unverified_specifics": [], "seconds": 1.2, "generation_seconds": 1.0,
            "repairs_attempted": 0, "repairs_accepted": 0, "repairs_rescued": 0, "repair_log": [],
        },
        "paragraphs": [{
            "index": 0, "original": source, "humanized": chosen, "winning_candidate": winner,
            "ai_probability_before": 0.9, "ai_probability_after": 0.1, "changed": chosen != source,
            "fallback_reason": None if winner is not None else "every candidate failed a gate",
            "unverified_specifics": [], "repairs": [],
            "candidates": [dict(c, paragraph_index=0, index=i, persona="p", constraint="c",
                                temperature=0.8, style="freeform", round=1 + i // 4,
                                passed=not c["gates"], quality={"content_overlap": 0.5, "flags": []})
                           for i, c in enumerate(cands)],
        }],
    }


def selftest() -> int:
    common.ensure_src_on_path()
    src = ("Furthermore, the urban heat island effect is a critical phenomenon in modern cities. "
           "Moreover, it increases energy demand and undermines public health. "
           "Consequently, planners must adopt green infrastructure to mitigate it. "
           "Therefore, cities that act now will be more resilient in the future.")
    chosen = ("Cities run hotter than the land around them, by 1 to 3 degrees on a typical afternoon. "
              "That gap raises electricity demand in summer, and it isn't good for anyone with a heart condition. "
              "Phoenix opened a heat office in 2021; the first results aren't in yet. "
              "Trees and pale roofs help, but slowly.")
    facts = "Phoenix opened a municipal heat office in 2021. Cities are 1 to 3 degrees warmer."
    cands = [
        {"proxy_probability": 0.02, "gates": [], "text": chosen},
        {"proxy_probability": 0.98, "gates": [], "text": src.replace("Furthermore, ", "")},
        {"proxy_probability": 0.01, "gates": ["content_drift", "too_short"], "text": "Short. Off topic entirely."},
        {"proxy_probability": None, "gates": ["invariant:numbers"], "text": chosen + " About 8500 deaths followed."},
    ]
    item = {"name": "ai-selftest", "kind": "ai", "text": src, "facts": facts}
    before = {"ai_probability": 1.0, "label": "ai", "cached": True, "words": common.n_words(src)}
    after = {"ai_probability": 0.03, "label": "human", "cached": False, "words": common.n_words(chosen)}
    row = common.assemble_row(item, _fake_result(src, chosen, cands, 0), before, after, seconds=12.3)
    # a human control that was left unchanged and scored under the 250-char floor
    hsrc = "We measured the samples twice. The second run agreed with the first within 2%."
    hitem = {"name": "PMC-selftest", "kind": "human", "text": hsrc, "facts": ""}
    hrow = common.assemble_row(hitem, _fake_result(hsrc, hsrc, [], None),
                               {"ai_probability": None, "label": None, "cached": True, "words": 14, "error": "too_short_for_gptzero"},
                               {"ai_probability": 0.0, "label": "human", "cached": True, "words": 14}, seconds=0.5)
    rows = [row, hrow]
    ledger = common.CostLedger(common.CACHE_DIR)
    ledger.record(chosen, was_cached=False)
    ledger.record(src, was_cached=True)
    summary = common.summarize(rows, ledger, mode="ownkey")

    q = row["quality"]
    checks = [
        ("changed", row["changed"] is True),
        ("human_after", row["human_after"] is True),
        ("overlap in (0,1)", 0.0 < q["content_overlap"] < 1.0),
        ("length ratio", q["length_ratio"] is not None and 0.5 < q["length_ratio"] < 1.6),
        ("FRE both", q["fre_source"] is not None and q["fre_after"] is not None and q["fre_delta"] is not None),
        ("CV both", q["sent_len_cv_source"] is not None and q["sent_len_cv_after"] is not None),
        ("formal connectives counted on source", q["connective_openers_source"]["formal"] == 4),
        ("formal connectives gone after", q["connective_openers_after"]["formal"] == 0),
        ("contractions after", q["contractions_after"] == 2),
        ("contractions source", q["contractions_source"] == 0),
        ("new specifics licensed by facts", q["n_new_specifics"] == 0),
        ("structure flags are lists", isinstance(q["structure_flags_after"], list) and isinstance(q["structure_flags_source"], list)),
        ("candidates flattened", row["n_candidates"] == 4 and row["n_passed"] == 2),
        ("chosen flag", sum(1 for c in row["candidates"] if c["chosen"]) == 1),
        ("summary counts", summary["ai_paragraphs_human"] == 1 and summary["humans_unharmed"] == 1),
        ("candidate-level (ownkey)", summary["candidate_level"]["human"] == 2 and summary["candidate_level"]["passed_and_human"] == 1),
        ("cost only for misses", ledger.misses == 1 and ledger.words_charged == common.n_words(chosen)),
        ("json serialisable", json.dumps({"paragraphs": rows, "summary": summary}) is not None),
    ]
    # an unlicensed specific must be reported
    q2 = common.quality_metrics(src, chosen + " Some 8500 people died in Paris.", facts)
    checks.append(("unlicensed specifics reported", "8500" in q2["new_specifics"] and "Paris" in q2["new_specifics"]))

    print(common.paragraph_table(rows))
    print()
    print(common.summary_line(summary))
    print()
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))
    if failed:
        print("selftest FAILED: %s" % ", ".join(failed))
        return 1
    print("selftest passed (%d checks); no model loaded, no network used" % len(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
