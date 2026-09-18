"""Compare two built datasets (train+valid) on the audit metrics. CPU only.
usage: compare.py DIR [DIR ...]"""
import json, re, statistics as st, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from humanizer.humanize.pipeline import content_overlap, flesch_reading_ease, new_specifics, _invariant_failures
from humanizer.humanize.llm import _FREEFORM_NO_NOTES
from humanizer.text import split_sentences

def parts(row):
    p = row["prompt"]; i = p.rfind("DRAFT:\n"); j = p.rfind("\n\nNOTES:\n"); k = p.rfind("\n\nHUMAN:\n")
    return p[i + 7:j], p[j + 9:k], row["completion"].strip()

def cv(t):
    l = [len(s.split()) for s in split_sentences(t) if s.strip()]
    return st.pstdev(l) / st.mean(l) if len(l) > 1 and st.mean(l) else 0.0

def norm(t): return " ".join(t.lower().split())

bench = json.load(open(ROOT / ".kriukow/gptzero_texts.json")); facts = json.load(open(ROOT / ".kriukow/gptzero_facts.json"))
K = 60; sh = set()
for t in [b["text"] for b in bench] + list(facts.values()):
    t = norm(t); sh.update(t[i:i + K] for i in range(max(1, len(t) - K + 1)))

for d in sys.argv[1:]:
    rows = [json.loads(l) for f in ("train.jsonl", "valid.jsonl") for l in open(Path(d) / f)]
    m = Counter(); vals = {k: [] for k in ("hw", "dw", "ratio", "ov", "fre_h", "fre_d", "cv_h", "n_notes")}
    contam = 0; docs = set(); src = Counter()
    for r in rows:
        draft, notes, human = parts(r); docs.add(r["doc"]); src[r["source"]] += 1
        hw, dw = len(human.split()), len(draft.split())
        vals["hw"].append(hw); vals["dw"].append(dw); vals["ratio"].append(hw / dw); vals["ov"].append(content_overlap(draft, human))
        vals["fre_h"].append(flesch_reading_ease(human)); vals["fre_d"].append(flesch_reading_ease(draft)); vals["cv_h"].append(cv(human))
        nn = notes == _FREEFORM_NO_NOTES; vals["n_notes"].append(0 if nn else notes.count("\n- ") + 1)
        m["no_notes"] += nn
        f = "" if nn else notes
        m["new_specifics>0"] += bool(new_specifics(draft, human, facts=f))
        inv = _invariant_failures(draft, human, f)
        ratio = hw / dw
        m["evade_gate_ok"] += (0.55 <= ratio <= 1.45 and content_overlap(draft, human) >= 0.25 and not inv)
        m["inv_fail"] += bool(inv)
        m["fre_h<30"] += flesch_reading_ease(human) < 30
        m["human_shorter"] += ratio < 1
        m["residue_spacing"] += bool(re.search(r"\s[.,;](\s|$)|shown in\.|\(\s*and", human))
        m["html_entity"] += bool(re.search(r"&(gt|lt|amp);", human + draft))
        m["draft_markdown"] += bool(re.search(r"###|\*\*|The passage|This passage", draft))
        t = norm(human); contam += any(t[i:i + K] in sh for i in range(max(1, len(t) - K + 1)))
    n = len(rows)
    print(f"\n== {d}: rows {n}, docs {len(docs)}, sources {dict(src)}, bench-shingle hits {contam}")
    for k, v in vals.items(): print(f"  {k:8s} mean {st.mean(v):7.3f}  median {st.median(v):7.3f}")
    for k, v in sorted(m.items()): print(f"  {k:18s} {v:4d}  ({v / n:.1%})")
