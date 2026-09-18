#!/usr/bin/env python
"""Candidate-level view of one bench results file (.hip7b/results/<tag>.json).

    .venv/bin/python .hip7b/candidates.py results/<tag>.json
    .venv/bin/python .hip7b/candidates.py results/<tag>.json --score-candidates [--yes] [--cached-only]

Without --score-candidates: gate pass rates per rejection reason, passers per
paragraph, and the proxy-score split between passers and non-passers, all read
from the results file. Free.

With --score-candidates: every distinct candidate text is scored with GPTZero
(cached where possible) so the candidate-level human rate, the metric
research/24 6.6 calls the informative one, and the "passed AND GPTZero-human"
supply of 6.14 can be computed. The run is priced first and refuses to spend
without --yes. --cached-only scores only texts already in the cache (free):
in own-key mode that is every candidate, since GPTZero ranked them.

Writes results/<tag>.candidates.json (the input file is left untouched).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="a .hip7b/results/<tag>.json written by bench.py")
    ap.add_argument("--score-candidates", action="store_true", help="score every candidate text with GPTZero (costs money; prices first)")
    ap.add_argument("--yes", action="store_true", help="confirm spending the estimated amount")
    ap.add_argument("--cached-only", action="store_true", help="with --score-candidates: only texts already in the cache (free)")
    ap.add_argument("--out", default=None, help="default results/<tag>.candidates.json")
    return ap


# ------------------------------------------------------------ gate report


def _mean(vals: List[float]) -> Optional[float]:
    return round(statistics.mean(vals), 3) if vals else None


def gate_report(data: Dict[str, Any]) -> str:
    rows = data.get("paragraphs") or []
    cands = [dict(c, paragraph=r["name"], kind=r["kind"]) for r in rows for c in r.get("candidates") or []]
    n = len(cands)
    lines: List[str] = []
    lines.append("results: %s  mode=%s  model=%s  adapter=%s" % (
        data.get("tag"), data.get("mode"), data.get("model"), data.get("adapter") or "none"))
    lines.append("candidates: %d over %d paragraphs; passed every gate: %d (%.0f%%)" % (
        n, len(rows), sum(1 for c in cands if c["passed"]), 100.0 * sum(1 for c in cands if c["passed"]) / n if n else 0.0))
    lines.append("")
    # per gate
    gate_counts: Dict[str, int] = {}
    gate_only: Dict[str, int] = {}
    for c in cands:
        for g in c["gates"]:
            gate_counts[g] = gate_counts.get(g, 0) + 1
        if len(c["gates"]) == 1:
            gate_only[c["gates"][0]] = gate_only.get(c["gates"][0], 0) + 1
    body = []
    for g, k in sorted(gate_counts.items(), key=lambda kv: -kv[1]):
        body.append([g, k, "%.0f%%" % (100.0 * k / n) if n else "-", gate_only.get(g, 0)])
    lines.append("rejections by gate (a candidate can fail several; 'only' = the sole failure):")
    lines.append(common.text_table(["gate", "candidates", "share", "only"], body, right=(1, 2, 3)))
    lines.append("")
    # per paragraph
    body = []
    for r in rows:
        cs = r.get("candidates") or []
        passers = [c for c in cs if c["passed"]]
        pp = [c["proxy_probability"] for c in passers if c["proxy_probability"] is not None]
        npp = [c["proxy_probability"] for c in cs if not c["passed"] and c["proxy_probability"] is not None]
        gates: Dict[str, int] = {}
        for c in cs:
            for g in c["gates"]:
                gates[g] = gates.get(g, 0) + 1
        top = ", ".join("%s %d" % (g, k) for g, k in sorted(gates.items(), key=lambda kv: -kv[1])[:3]) or "-"
        body.append([r["name"][:22], r["kind"], "%d/%d" % (len(passers), len(cs)),
                     common.fmt_f(_mean(pp), "%.3f"), common.fmt_f(_mean(npp), "%.3f"),
                     common.fmt_p(r.get("p_after")), r.get("label_after") or "-", top])
    lines.append("per paragraph (proxy = the in-loop scorer: GPTZero in own-key mode, surrogate in free mode):")
    lines.append(common.text_table(["paragraph", "kind", "passed", "proxy passers", "proxy rejected", "GPTZero after", "verdict", "top gates"], body, right=(2, 3, 4, 5)))
    return "\n".join(lines)


# ------------------------------------------------------- candidate scoring


def _distinct_candidate_texts(data: Dict[str, Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for r in data.get("paragraphs") or []:
        for c in r.get("candidates") or []:
            t = (c.get("text") or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def score_candidates(data: Dict[str, Any], client: Any, cached_only: bool) -> Dict[str, Any]:
    ledger = common.CostLedger(common.CACHE_DIR)
    scores: Dict[str, Dict[str, Any]] = {}
    for text in _distinct_candidate_texts(data):
        if len(text) < common.MIN_CHARS:
            scores[text] = {"ai_probability": None, "label": None, "error": "too_short_for_gptzero"}
            continue
        if cached_only and not common.cache_has(text):
            scores[text] = {"ai_probability": None, "label": None, "error": "not_cached"}
            continue
        if client is None:
            payload = common.read_cache_payload(text)
            scores[text] = common.parse_payload(payload) if payload else {"ai_probability": None, "label": None, "error": "not_cached"}
            continue
        scores[text] = common.score_with_retry(client, text, ledger)
    for r in data.get("paragraphs") or []:
        for c in r.get("candidates") or []:
            s = scores.get((c.get("text") or "").strip()) or {}
            c["gptzero"] = {"ai_probability": s.get("ai_probability"), "label": s.get("label"), "error": s.get("error")}
            c["gptzero_human"] = common.is_human(s.get("ai_probability"), s.get("label"))
    data["candidate_scoring_cost"] = ledger.as_dict()
    return data


def candidate_study(data: Dict[str, Any]) -> Dict[str, Any]:
    rows = data.get("paragraphs") or []
    ai_rows = [r for r in rows if r["kind"] == "ai"]
    per_para = []
    tot = {"scored": 0, "human": 0, "mixed": 0, "ai": 0, "passed": 0, "passed_and_human": 0, "unscored": 0}
    for r in ai_rows:
        cs = r.get("candidates") or []
        scored = [c for c in cs if c.get("gptzero_human") is not None]
        human = [c for c in scored if c["gptzero_human"]]
        both = [c for c in human if c["passed"]]
        labels = [(c.get("gptzero") or {}).get("label") for c in scored]
        entry = {
            "name": r["name"], "candidates": len(cs), "scored": len(scored), "human": len(human),
            "mixed": labels.count("mixed"), "ai": labels.count("ai"),
            "passed": sum(1 for c in cs if c["passed"]), "passed_and_human": len(both),
            "chosen_gptzero_human": r.get("human_after"),
            "oracle_available": bool(both),
        }
        per_para.append(entry)
        for k in ("scored", "human", "mixed", "ai", "passed", "passed_and_human"):
            tot[k] += entry[k]
        tot["unscored"] += len(cs) - len(scored)
    tot["human_share"] = round(tot["human"] / tot["scored"], 3) if tot["scored"] else None
    tot["passed_and_human_share"] = round(tot["passed_and_human"] / tot["scored"], 3) if tot["scored"] else None
    tot["oracle_bound_paragraphs"] = sum(1 for e in per_para if e["oracle_available"])
    tot["ai_paragraphs"] = len(ai_rows)
    tot["chosen_human_paragraphs"] = sum(1 for e in per_para if e["chosen_gptzero_human"])
    return {"per_paragraph": per_para, "total": tot}


def study_table(study: Dict[str, Any]) -> str:
    body = []
    for e in study["per_paragraph"]:
        body.append([e["name"][:22], e["candidates"], e["scored"], e["human"], e["mixed"], e["ai"],
                     e["passed"], e["passed_and_human"],
                     "human" if e["chosen_gptzero_human"] else ("ai" if e["chosen_gptzero_human"] is False else "-")])
    t = study["total"]
    body.append(["TOTAL", sum(e["candidates"] for e in study["per_paragraph"]), t["scored"], t["human"], t["mixed"], t["ai"],
                 t["passed"], t["passed_and_human"], "%d/%d" % (t["chosen_human_paragraphs"], t["ai_paragraphs"])])
    lines = [common.text_table(["paragraph", "cands", "scored", "human", "mixed", "ai", "passed", "pass+human", "chosen"],
                               body, right=(1, 2, 3, 4, 5, 6, 7))]
    lines.append("")
    lines.append("candidate-level GPTZero-human: %d/%d (%s); passed AND human: %d (%s); "
                 "paragraphs with at least one passed+human candidate (oracle bound): %d/%d; chosen text human: %d/%d" % (
                     t["human"], t["scored"], ("%.0f%%" % (100 * t["human_share"])) if t["human_share"] is not None else "-",
                     t["passed_and_human"], ("%.0f%%" % (100 * t["passed_and_human_share"])) if t["passed_and_human_share"] is not None else "-",
                     t["oracle_bound_paragraphs"], t["ai_paragraphs"], t["chosen_human_paragraphs"], t["ai_paragraphs"]))
    if t["unscored"]:
        lines.append("unscored candidates: %d (under 250 characters, or not cached with --cached-only)" % t["unscored"])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.results)
    data = common.load_results(path)
    print(gate_report(data))
    if not args.score_candidates:
        if data.get("mode") == "ownkey":
            print()
            print("own-key run: proxy_probability above IS GPTZero's 1 - P(human); run with "
                  "--score-candidates --cached-only for the candidate-level table at no cost.")
        return 0

    texts = _distinct_candidate_texts(data)
    scorable = [t for t in texts if len(t) >= common.MIN_CHARS]
    misses = [t for t in scorable if not common.cache_has(t)]
    words = sum(common.n_words(t) for t in misses)
    print()
    print("candidate texts: %d distinct, %d scorable (>= 250 chars), %d already cached, %d would be charged: %d words, about $%.2f" % (
        len(texts), len(scorable), len(scorable) - len(misses), len(misses), words, words * common.COST_PER_WORD_USD))
    client = None
    if misses and not args.cached_only:
        if not args.yes:
            print("refusing to spend without --yes (or pass --cached-only to use the cache alone)")
            return 3
        os.chdir(common.ROOT)
        common.ensure_src_on_path()
        from humanizer.detectors.gptzero import GPTZeroClient
        client = GPTZeroClient(cache_dir=common.CACHE_DIR)
        if not client.available():
            print("GPTZERO_API_KEY is not set; use --cached-only or set the key")
            return 2
    data = score_candidates(data, client, cached_only=(client is None))
    study = candidate_study(data)
    data["candidate_study"] = study
    out = Path(args.out) if args.out else path.with_name(path.stem + ".candidates.json")
    out.write_text(json.dumps(data, indent=1))
    print()
    print(study_table(study))
    cost = data.get("candidate_scoring_cost") or {}
    print("GPTZero spent by this step: $%.2f (%d cache misses)" % (cost.get("estimated_usd", 0.0), cost.get("cache_misses", 0)))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
