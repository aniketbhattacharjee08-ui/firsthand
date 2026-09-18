"""CPU-only audit of .hiplora/pairs.jsonl as consumed by .hip7b/build_data.py.

Writes rows.jsonl (per-pair metrics), summary.json, contamination.json and
hand_review_sample.md into this directory. Imports nothing heavier than the
pipeline's pure-python helpers.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent

from humanizer.humanize.pipeline import (  # noqa: E402
    content_overlap, flesch_reading_ease, new_specifics, _invariant_failures,
)
from humanizer.humanize.kriukow import structure_report  # noqa: E402
from humanizer.humanize.llm import _FREEFORM_NO_NOTES  # noqa: E402
from humanizer.text import split_sentences  # noqa: E402

spec = importlib.util.spec_from_file_location("build_data", ROOT / ".hip7b/build_data.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)

BUILDER = sys.argv[1] if len(sys.argv) > 1 else None
if BUILDER:
    spec2 = importlib.util.spec_from_file_location("build_data_v2", ROOT / BUILDER)
    bd2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(bd2)
else:
    bd2 = None

RESIDUE = {
    "dangling_paren": re.compile(r"\(\s*[,;.]|\(\s*\)|\(\s*and\b|,\s*\)|\(\s*$"),
    "fig_table_ref": re.compile(r"\b(Fig\.|Figs\.|Figure \d|Table \d|Supplementary|see Table|see Fig)", re.I),
    "et_al_bare": re.compile(r"\bet al\.?\s*(?:[,;.)]|$)"),
    "numeric_cite": re.compile(r"\[\d+(?:[,–-]\s*\d+)*\]|\(\d{1,3}(?:[,–-]\s*\d{1,3})+\)"),
    "orphan_punct": re.compile(r"\s[.,;:](?:\s|$)|,\s*[.,]"),
    "double_space": re.compile(r"\S {2,}\S"),
    "no_terminal": re.compile(r"[^.!?\"')\]]\s*$"),
    "lowercase_start": re.compile(r"^[a-z]"),
    "table_like": re.compile(r"(?:\d[\d.,%/]*\s*){8,}"),
    "asterisk_markup": re.compile(r"\*[^*]+\*"),
    "url_or_doi": re.compile(r"https?://|doi\.org|www\."),
    "we_our_study": re.compile(r"\b(our study|the present study|this study|our results|our findings|in this paper|we (?:found|observed|report|show|hypothesi[sz]e))\b", re.I),
}


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def cv(text: str):
    lens = [len(s.split()) for s in split_sentences(text) if s.strip()]
    if len(lens) < 2:
        return 0.0, lens
    m = st.mean(lens)
    return (st.pstdev(lens) / m if m else 0.0), lens


def profile_row(r, rng):
    draft, human = " ".join(r["ai"].split()), " ".join(r["human"].split())
    why = bd.keep(draft, human)
    notes = bd.build_notes(draft, human, rng)
    hw, dw = len(human.split()), len(draft.split())
    cv_h, lens_h = cv(human)
    cv_d, _ = cv(draft)
    srep_h = structure_report(human)
    srep_d = structure_report(draft)
    ns_notes = new_specifics(draft, human, facts=notes if notes != _FREEFORM_NO_NOTES else "")
    ns_bare = new_specifics(draft, human)
    inv = _invariant_failures(draft, human, notes if notes != _FREEFORM_NO_NOTES else "")
    ratio = hw / max(1, dw)
    ov = content_overlap(draft, human)
    evade_ok = (0.55 <= ratio <= 1.45) and ov >= 0.25 and not inv
    flags = [k for k, rx in RESIDUE.items() if rx.search(human)]
    sents = split_sentences(human)
    conn = sum(1 for s in sents if bd.CONNECTIVE_RE.match(s))
    row = dict(
        source=r["source"], doc=r["doc"], kept=(why == ""), drop_reason=why or None,
        hw=hw, dw=dw, ratio=round(ratio, 3), overlap=round(ov, 3),
        fre_h=round(flesch_reading_ease(human), 1), fre_d=round(flesch_reading_ease(draft), 1),
        cv_h=round(cv_h, 3), cv_d=round(cv_d, 3), n_sent_h=len(sents), n_sent_d=len(split_sentences(draft)),
        min_sent=min(lens_h) if lens_h else 0, max_sent=max(lens_h) if lens_h else 0,
        connectives_h=conn,
        penalty_h=srep_h["penalty"], penalty_d=srep_d["penalty"],
        hedges_h=srep_h["hedges_per_1k"], hedges_d=srep_d["hedges_per_1k"],
        the_share_h=srep_h["the_this_it_share"],
        notes=notes, no_notes=(notes == _FREEFORM_NO_NOTES),
        n_notes=0 if notes == _FREEFORM_NO_NOTES else notes.count("\n- ") + 1,
        notes_words=0 if notes == _FREEFORM_NO_NOTES else len(notes.split()),
        new_specifics_with_notes=ns_notes, n_new_specifics_with_notes=len(ns_notes),
        n_new_specifics_bare=len(ns_bare),
        invariant_failures=inv, evade_gate_ok=evade_ok,
        residue_flags=flags,
        draft=draft, human=human,
    )
    if bd2 is not None:
        row["v2_drop_reason"] = bd2.keep(draft, human) or None
    return row


def summarise(rows):
    def q(vals, p):
        vals = sorted(vals)
        if not vals:
            return None
        k = (len(vals) - 1) * p
        f, c = int(k), min(int(k) + 1, len(vals) - 1)
        return round(vals[f] + (vals[c] - vals[f]) * (k - f), 3)

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {}
        return {"n": len(vals), "mean": round(st.mean(vals), 3), "median": q(vals, .5), "p10": q(vals, .1), "p90": q(vals, .9)}

    out = {"n": len(rows)}
    for k in ("hw", "dw", "ratio", "overlap", "fre_h", "fre_d", "cv_h", "cv_d", "n_sent_h", "min_sent", "max_sent",
              "connectives_h", "penalty_h", "penalty_d", "hedges_h", "hedges_d", "the_share_h",
              "n_notes", "notes_words", "n_new_specifics_with_notes", "n_new_specifics_bare"):
        out[k] = stats([r[k] for r in rows])
    n = max(1, len(rows))
    out["share_no_notes"] = round(sum(r["no_notes"] for r in rows) / n, 3)
    out["share_with_new_specifics_despite_notes"] = round(sum(1 for r in rows if r["n_new_specifics_with_notes"] > 0) / n, 3)
    out["share_evade_gate_ok"] = round(sum(r["evade_gate_ok"] for r in rows) / n, 3)
    out["invariant_failure_kinds"] = dict(Counter(f for r in rows for f in r["invariant_failures"]))
    out["residue_flag_counts"] = dict(Counter(f for r in rows for f in r["residue_flags"]))
    out["share_any_residue"] = round(sum(1 for r in rows if r["residue_flags"]) / n, 3)
    out["share_fre_h_below_30"] = round(sum(1 for r in rows if r["fre_h"] < 30) / n, 3)
    out["share_human_shorter_than_draft"] = round(sum(1 for r in rows if r["ratio"] < 1.0) / n, 3)
    out["share_closer_summary"] = round(sum(1 for r in rows if bd.CLOSER_RE.match(split_sentences(r["human"])[-1] if split_sentences(r["human"]) else "")) / n, 3)
    return out


def main():
    rows_in = [json.loads(l) for l in open(ROOT / ".hiplora/pairs.jsonl")]
    rng = random.Random(0)
    rows = [profile_row(r, rng) for r in rows_in]
    with open(HERE / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    kept = [r for r in rows if r["kept"]]
    dropped = [r for r in rows if not r["kept"]]
    summary = {
        "all": summarise(rows), "kept": summarise(kept), "dropped": summarise(dropped),
        "drop_reasons": dict(Counter(r["drop_reason"] for r in dropped)),
        "drop_reasons_by_source": {s: dict(Counter(r["drop_reason"] for r in dropped if r["source"] == s)) for s in ("pmc", "pmc2", "wiki")},
        "kept_by_source": {s: summarise([r for r in kept if r["source"] == s]) for s in ("pmc", "pmc2", "wiki")},
        "dropped_by_source": {s: summarise([r for r in dropped if r["source"] == s]) for s in ("pmc", "pmc2", "wiki")},
        "all_by_source": {s: summarise([r for r in rows if r["source"] == s]) for s in ("pmc", "pmc2", "wiki")},
        "kept_docs": len({r["doc"] for r in kept}),
        "kept_docs_by_source": {s: len({r["doc"] for r in kept if r["source"] == s}) for s in ("pmc", "pmc2", "wiki")},
        "kept_per_doc_max": max(Counter(r["doc"] for r in kept).values()),
    }
    if bd2 is not None:
        v2k = [r for r in rows if r["v2_drop_reason"] is None]
        summary["v2"] = {
            "kept": summarise(v2k),
            "drop_reasons": dict(Counter(r["v2_drop_reason"] for r in rows if r["v2_drop_reason"])),
            "kept_by_source": {s: summarise([r for r in v2k if r["source"] == s]) for s in ("pmc", "pmc2", "wiki")},
            "kept_docs": len({r["doc"] for r in v2k}),
        }
    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)

    # ---- contamination: 60-char shingles of normalised text
    bench = json.load(open(ROOT / ".kriukow/gptzero_texts.json"))
    facts = json.load(open(ROOT / ".kriukow/gptzero_facts.json"))
    bench_texts = {b["name"]: norm(b["text"]) for b in bench}
    bench_texts.update({"facts:" + k: norm(v) for k, v in facts.items()})
    K = 60
    bench_sh = {}
    for name, t in bench_texts.items():
        for i in range(0, max(1, len(t) - K + 1)):
            bench_sh.setdefault(t[i:i + K], set()).add(name)
    hits = []
    bench_ids = {b["name"].replace(".txt", "") for b in bench}
    for r in rows:
        for field in ("human", "draft"):
            t = norm(r[field])
            found = set()
            for i in range(0, max(1, len(t) - K + 1)):
                s = t[i:i + K]
                if s in bench_sh:
                    found |= bench_sh[s]
            if found:
                hits.append({"doc": r["doc"], "source": r["source"], "field": field, "bench": sorted(found), "kept": r["kept"]})
    doc_hits = sorted({r["doc"] for r in rows if r["doc"].replace(".txt", "") in bench_ids})
    contamination = {"shingle_k": K, "n_bench_texts": len(bench_texts), "shingle_hits": hits, "doc_name_hits": doc_hits,
                     "rows_checked": len(rows) * 2}
    json.dump(contamination, open(HERE / "contamination.json", "w"), indent=1)

    # ---- hand review sample
    srng = random.Random(7)
    sample_kept = srng.sample(kept, 30)
    sample_dropped = srng.sample(dropped, 15)
    with open(HERE / "hand_review_sample.md", "w") as f:
        for title, group in (("KEPT", sample_kept), ("DROPPED", sample_dropped)):
            f.write(f"# {title}\n\n")
            for i, r in enumerate(group, 1):
                f.write(f"## {title[0]}{i} {r['source']} {r['doc']} drop={r['drop_reason']} hw={r['hw']} dw={r['dw']} ratio={r['ratio']} ov={r['overlap']} fre_h={r['fre_h']} fre_d={r['fre_d']} cv_h={r['cv_h']} cv_d={r['cv_d']} pen_h={r['penalty_h']} pen_d={r['penalty_d']} newspec={r['new_specifics_with_notes']} inv={r['invariant_failures']} residue={r['residue_flags']}\n\n")
                f.write(f"DRAFT: {r['draft']}\n\nNOTES:\n{r['notes']}\n\nHUMAN: {r['human']}\n\n")
    print(json.dumps({k: summary[k] for k in ("drop_reasons", "kept_docs", "kept_docs_by_source", "kept_per_doc_max")}, indent=1))
    print("kept", summary["kept"]["n"], "dropped", summary["dropped"]["n"])
    print("contamination hits:", len(hits), "doc-name hits:", doc_hits)


if __name__ == "__main__":
    main()
