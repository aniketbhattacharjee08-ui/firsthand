"""Build (prompt, completion) pairs for the 7B HIP adapter in the pipeline's
exact freeform format.

Sources: .hiplora/pairs.jsonl (1,500 human paragraphs from PMC, PMC2 and
pre-2022 Wikipedia, each with a Qwen2.5-3B-Instruct paraphrase as the draft).
The bench paragraphs' PMC files were excluded when that file was built.

What is different from the 3B round (research/24 §6.6), and why:

* The prompt is the shipped inference format: header, ONE exemplar triple
  (rotating), DRAFT, NOTES, HUMAN. The 3B round trained on an older header
  with no NOTES slot, so the adapter never learned to use the author's
  particulars and invented them instead.
* NOTES carry the particulars (numbers, dates, names) that appear in the
  human original but not in the draft, as short clauses from the original.
  When the draft already has them all, NOTES says so. This trains the one
  behaviour the gates enforce: add a particular only if the notes license it.
* The loss is masked to the HUMAN completion. The 3B round trained on the
  whole text, which spent a third of the gradient learning to reproduce the
  instruct paraphrase, the exact register the product is trying to leave.
* Pairs are filtered for fidelity and shape (research/23 §7: the passing
  rewriters shift distribution without expansion or register drop):
  length ratio human/draft 0.70-1.35, content overlap >= 0.40, human target
  50-260 words, FRE 15-80, at most one sentence-initial formal connective,
  no summary closer. The 3B adapter drifted because nothing stopped it
  learning to write a PMC paragraph rather than a rewrite of the draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanizer.humanize.engine import _NUMBER_RE, _entities  # noqa: E402
from humanizer.humanize.exemplars import FREEFORM_SHOTS  # noqa: E402
from humanizer.humanize.llm import _FREEFORM_HEADER, _FREEFORM_NO_NOTES  # noqa: E402
from humanizer.humanize.pipeline import content_overlap, flesch_reading_ease  # noqa: E402
from humanizer.text import split_sentences  # noqa: E402

CONNECTIVE_RE = re.compile(
    r"^(However|Moreover|Furthermore|Additionally|Consequently|Therefore|Thus|Hence|"
    r"In addition|In conclusion|Overall|Ultimately|Notably|Importantly|Similarly|"
    r"Nevertheless|Nonetheless|Accordingly|Subsequently|Indeed|For example|For instance),?\s",
    re.IGNORECASE,
)
CLOSER_RE = re.compile(r"^(In conclusion|Overall|In summary|To summarize|Ultimately|In short)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:1[89]|20)\d{2}\b")


def particulars(text: str):
    nums = set(_NUMBER_RE.findall(text))
    ents = set(_entities(text))
    return nums, ents


def clause_around(sentence: str, needle: str, max_words: int = 18) -> str:
    """The clause of `sentence` containing `needle`, trimmed to `max_words`."""
    parts = re.split(r"(?<=[;:,])\s+", sentence)
    for part in parts:
        if needle in part:
            words = part.split()
            if len(words) <= max_words:
                return part.strip(" ,;:")
            i = next((k for k, w in enumerate(words) if needle in w), 0)
            lo = max(0, i - max_words // 2)
            return " ".join(words[lo : lo + max_words]).strip(" ,;:")
    words = sentence.split()
    return " ".join(words[:max_words]).strip(" ,;:")


def build_notes(draft: str, human: str, rng: random.Random, max_items: int = 5) -> str:
    d_nums, d_ents = particulars(draft)
    h_nums, h_ents = particulars(human)
    missing = sorted(h_nums - d_nums, key=len, reverse=True) + sorted(h_ents - d_ents, key=len, reverse=True)
    if not missing:
        return _FREEFORM_NO_NOTES
    sentences = split_sentences(human)
    items, used = [], set()
    for needle in missing:
        for s in sentences:
            if needle in s and s not in used:
                items.append("- " + clause_around(s, needle))
                used.add(s)
                break
        if len(items) >= max_items:
            break
    if not items:
        return _FREEFORM_NO_NOTES
    rng.shuffle(items)
    return "\n".join(items)


def shot_block(shot) -> str:
    _topic, draft, notes, rewrite = shot
    return f"DRAFT:\n{draft}\n\nNOTES:\n{notes}\n\nHUMAN:\n{rewrite}\n\n"


def keep(draft: str, human: str) -> str:
    """Empty string to keep, else the reason to drop."""
    hw, dw = len(human.split()), len(draft.split())
    if not (50 <= hw <= 260):
        return "human_length"
    ratio = hw / max(1, dw)
    if not (0.70 <= ratio <= 1.35):
        return "ratio"
    if content_overlap(draft, human) < 0.40:
        return "overlap"
    fre = flesch_reading_ease(human)
    if not (15.0 <= fre <= 80.0):
        return "fre"
    sents = split_sentences(human)
    if len(sents) < 3:
        return "sentences"
    if sum(1 for s in sents if CONNECTIVE_RE.match(s)) > 1:
        return "connectives"
    if CLOSER_RE.match(sents[-1]):
        return "closer"
    if re.search(r"\[\d+(?:[,–-]\s*\d+)*\]|\(\s*(?:Fig|Table|Supplementary)", human):
        return "citation_marks"
    if human.count("  ") > 2 or " ." in human or " ," in human:
        return "tokenised_spacing"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / ".hiplora/pairs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / ".hip7b/data"))
    ap.add_argument("--valid-share", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-wiki-share", type=float, default=0.0, help="0 keeps every pair; otherwise cap PMC so wiki is at least this share")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.pairs)]
    kept, reasons = [], {}
    for r in rows:
        draft, human = " ".join(r["ai"].split()), " ".join(r["human"].split())
        why = keep(draft, human)
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        kept.append({"source": r["source"], "doc": r["doc"], "draft": draft, "human": human})
    print("kept %d of %d; dropped: %s" % (len(kept), len(rows), json.dumps(reasons, sort_keys=True)), file=sys.stderr)

    wiki = [k for k in kept if k["source"] == "wiki"]
    pmc = [k for k in kept if k["source"] != "wiki"]
    rng.shuffle(pmc)
    if args.max_wiki_share > 0 and wiki:
        max_pmc = int(len(wiki) * (1 - args.max_wiki_share) / args.max_wiki_share)
        pmc = pmc[:max_pmc] if len(pmc) > max_pmc else pmc
    kept = wiki + pmc
    rng.shuffle(kept)
    print("after balancing: %d (wiki %d, pmc %d)" % (len(kept), len(wiki), len(pmc)), file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_f = (out / "train.jsonl").open("w")
    valid_f = (out / "valid.jsonl").open("w")
    n_train = n_valid = n_no_notes = 0
    for i, k in enumerate(kept):
        notes = build_notes(k["draft"], k["human"], rng)
        if notes == _FREEFORM_NO_NOTES:
            n_no_notes += 1
        shot = FREEFORM_SHOTS[i % len(FREEFORM_SHOTS)]
        prompt = _FREEFORM_HEADER + shot_block(shot) + "DRAFT:\n" + k["draft"] + "\n\nNOTES:\n" + notes + "\n\nHUMAN:\n"
        row = {"prompt": prompt, "completion": k["human"] + "\n\n", "source": k["source"], "doc": k["doc"]}
        # Fixed split by hash of the human text so reruns are comparable.
        h = int(hashlib.sha1(k["human"].encode()).hexdigest(), 16) % 1000
        if h < args.valid_share * 1000:
            valid_f.write(json.dumps(row) + "\n")
            n_valid += 1
        else:
            train_f.write(json.dumps(row) + "\n")
            n_train += 1
    train_f.close()
    valid_f.close()
    print("train %d, valid %d, no-notes rows %d" % (n_train, n_valid, n_no_notes), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
