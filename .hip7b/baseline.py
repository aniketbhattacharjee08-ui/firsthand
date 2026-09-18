#!/usr/bin/env python
"""The baseline a new adapter has to beat, from the existing bench rows.

    .venv/bin/python .hip7b/baseline.py [--no-cache-window] [--out results/baseline.md]

CPU only, no network, no model. Reads every .kriukow/gptzero_rows*.json
(the rows scripts/gptzero_bench.py wrote for research/24 section 6) and
prints one table: file, configuration guessed from the tag (cross-checked
with research/24), AI paragraphs GPTZero called human N/9, human controls
unharmed N/5, and the candidate-level human share.

The row files carry no candidate information. For the own-key runs (GPTZero
was the candidate scorer, research/24 6.4-6.8) the candidate-level share is
recovered the way 6.6 did it: every GPTZero cache file written between the
previous rows file and this one, minus the bench source texts, is a candidate
of this run. Free-mode runs ranked candidates with the local surrogate, so
their windows contain labelling probes rather than candidates and the column
says n/a. The windows for 7b (111: 55 human) and hip3b (112: 71 human)
reproduce the numbers printed in 6.6 exactly.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

#: tag -> (config guess, own-key mode?, research/24 section)
CONFIGS: Dict[str, Tuple[str, bool, str]] = {
    "": ("register adapter r4 (Qwen3-4B-Instruct+LoRA), facts, 8 cand x 1 round, GPTZero judge", True, "6.1"),
    "v2": ("3B base freeform, exemplar triples + notes, 8x2, GPTZero ranks", True, "6.4"),
    "7b_hard": ("7B base freeform, 3 holdout paragraphs only, 8x2, GPTZero ranks", True, "6.5"),
    "7b": ("7B base freeform, 8x2, GPTZero ranks", True, "6.5"),
    "hip3b": ("3B base + HIP LoRA step 300, 8x2, GPTZero ranks", True, "6.6"),
    "hip3b_it200": ("3B base + HIP LoRA step 200, 8x2, GPTZero ranks", True, "6.6"),
    "base_r3": ("3B base, 8 cand x 3 rounds, GPTZero ranks, old gates", True, "6.7"),
    "c1": ("3B base, 8x2, GPTZero ranks, corrected gates (overlap 0.25, len 0.55-1.45), run 1", True, "6.8"),
    "c2": ("3B base, 8x2, GPTZero ranks, corrected gates, run 2", True, "6.8"),
    "surrogate": ("free mode: surrogate v1 ranks, threshold 0.5, 8x2", False, "6.11"),
    "surrogate2": ("free mode: surrogate v1 ranks, run 2", False, "6.11"),
    "surrogate3": ("free mode: surrogate v3 (1,136 labels), threshold 0.15, 8x3", False, "6.12"),
    "surrogate4": ("free mode: surrogate v4 (1,265 labels), 0.15, 8x3", False, "6.14"),
    "surrogate5": ("free mode: surrogate v4, consistency run", False, "6.14"),
    "surrogate6": ("free mode: surrogate v5 (1,368 labels), 8x3", False, "6.14"),
    "surrogate7": ("free mode: surrogate v6 (1,452 labels), 12 cand x 3", False, "6.14"),
    "surrogate8": ("free mode: surrogate v7/v8 (fixed split), 8x3", False, "6.14"),
    "r5": ("free mode: surrogate v8, 8 cand x 5 rounds", False, "6.14"),
    "repair": ("free mode: surrogate, 8x3 + repair stage (3 attempts)", False, "6.16"),
}


def tag_of(path: Path) -> str:
    stem = path.stem  # gptzero_rows_c1
    return stem[len("gptzero_rows"):].lstrip("_")


def guess_config(tag: str) -> Tuple[str, Optional[bool], str]:
    if tag in CONFIGS:
        return CONFIGS[tag]
    if tag.startswith("surrogate") or tag.startswith("free"):
        return ("free mode (unlisted tag)", False, "?")
    if "7b" in tag or "hip" in tag or "base" in tag:
        return ("base checkpoint run (unlisted tag)", True, "?")
    return ("unknown", None, "?")


def row_human(r: Dict[str, Any]) -> Optional[bool]:
    return common.is_human(r.get("gptzero_after"), r.get("gptzero_label_after"))


def load_cache_index(cache_dir: Path) -> List[Tuple[float, Optional[str], str]]:
    out: List[Tuple[float, Optional[str], str]] = []
    for p in glob.glob(str(cache_dir / "*.json")):
        try:
            payload = json.loads(Path(p).read_text())
        except Exception:  # noqa: BLE001
            continue
        parsed = common.parse_payload(payload)
        doc = (payload.get("documents") or [{}])[0]
        out.append((os.path.getmtime(p), parsed["label"], (doc.get("inputText") or "").strip()))
    out.sort(key=lambda t: t[0])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-glob", default=str(common.KRIUKOW / "gptzero_rows*.json"))
    ap.add_argument("--texts", default=str(common.DEFAULT_TEXTS))
    ap.add_argument("--no-cache-window", action="store_true", help="skip the cache-mtime candidate estimate")
    ap.add_argument("--out", default=str(common.RESULTS_DIR / "baseline.md"))
    args = ap.parse_args(argv)

    files = sorted((Path(p) for p in glob.glob(args.rows_glob)), key=os.path.getmtime)
    if not files:
        print("no row files match %s" % args.rows_glob)
        return 1
    kinds = {it["name"]: it["kind"] for it in common.load_bench(Path(args.texts), None)}
    sources = {it["text"].strip() for it in common.load_bench(Path(args.texts), None)}
    cache = [] if args.no_cache_window else load_cache_index(common.CACHE_DIR)

    headers = ["file", "when", "config (research/24)", "AI human", "humans unharmed", "changed", "candidates GPTZero-human (cache window)"]
    table: List[List[Any]] = []
    prev_mtime = 0.0
    best_ownkey: Optional[Tuple[int, str]] = None
    best_free: Optional[Tuple[int, str]] = None
    for path in files:
        rows = json.loads(path.read_text())
        tag = tag_of(path)
        config, ownkey, section = guess_config(tag)
        ai = [r for r in rows if kinds.get(r["name"], common.guess_kind(r["name"])) == "ai"]
        hu = [r for r in rows if kinds.get(r["name"], common.guess_kind(r["name"])) == "human"]
        ai_human = sum(1 for r in ai if row_human(r))
        hu_ok = sum(1 for r in hu if row_human(r))
        changed = sum(1 for r in rows if r.get("changed"))
        mtime = os.path.getmtime(path)
        cand_col = "n/a"
        if cache and ownkey:
            win = [c for c in cache if prev_mtime < c[0] <= mtime and c[2] not in sources]
            if win:
                cl = Counter(c[1] for c in win)
                cand_col = "%d/%d human (%.0f%%), %d mixed" % (cl.get("human", 0), len(win), 100.0 * cl.get("human", 0) / len(win), cl.get("mixed", 0))
        elif cache and ownkey is False:
            cand_col = "n/a (surrogate ranked; no GPTZero on candidates)"
        prev_mtime = mtime
        table.append([
            path.name, time.strftime("%m-%d %H:%M", time.localtime(mtime)),
            "%s [%s]" % (config, section),
            "%d/%d" % (ai_human, len(ai)), "%d/%d" % (hu_ok, len(hu)), "%d/%d" % (changed, len(rows)), cand_col,
        ])
        if len(ai) == 9:
            if ownkey and (best_ownkey is None or ai_human > best_ownkey[0]):
                best_ownkey = (ai_human, tag or "rows")
            if ownkey is False and (best_free is None or ai_human > best_free[0]):
                best_free = (ai_human, tag)

    text = common.text_table(headers, table)
    print(text)
    print()
    notes = [
        "Baseline to beat (from research/24 6.8 and 6.14): own-key mode 7-8 of 9 (22/27 = 81% over c-runs), "
        "free mode 5-7 of 9 (61-67%), humans 5/5 in every run.",
        "Best own-key row here: %s of 9 (%s). Best free-mode row here: %s of 9 (%s)." % (
            best_ownkey[0] if best_ownkey else "-", best_ownkey[1] if best_ownkey else "-",
            best_free[0] if best_free else "-", best_free[1] if best_free else "-"),
        "Candidate-level human share to beat (6.6): 3B base 45%, 7B base 50%, 3B+HIP-300 63% (but more gate failures). "
        "The product of candidate-level human rate and gate pass rate is what lifts paragraphs.",
        "The 'cache window' column counts GPTZero cache files written between consecutive row files, minus the 14 source texts. "
        "It reproduces research/24 6.6 exactly for v2 (217, 97 human), 7b (111, 55) and hip3b (112, 71); "
        "the c1 window also contains the 6.7 diagnostic run (.kriukow/diag_candidates.json), so its N is an upper bound. "
        "Own-key runs only: in free mode the surrogate ranked candidates and GPTZero never saw them.",
    ]
    for n in notes:
        print(n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Baseline: existing GPTZero bench rows (.kriukow/gptzero_rows*.json)", "",
          "Generated %s by .hip7b/baseline.py. 9 AI paragraphs with author facts, 5 human PMC controls." % common.utc_now(), "",
          common.markdown_table(headers, table), ""]
    md.extend("- " + n for n in notes)
    out.write_text("\n".join(md) + "\n")
    print()
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
