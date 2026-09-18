"""Build (prompt, completion) pairs for the 7B HIP adapter: second builder.

Same CLI shape and byte-identical prompt format as build_data.py (header, one
exemplar triple, DRAFT, NOTES, HUMAN); output to .hip7b/data_v2/. What changes
is which pairs survive and what NOTES carry, following the audit in
.hip7b/audit/REPORT.md:

* Gate-consistent targets. The human target must pass the product's own
  evade gates against its draft with the NOTES as facts: length ratio,
  content overlap, every draft number/quotation/citation/date preserved,
  draft entities preserved (relaxed to "some token of the name survives", so
  the paraphraser's acronym expansions do not count), and
  `pipeline.new_specifics(draft, human, facts=notes)` empty. 63% of the v1
  rows failed this, which taught the adapter the behaviour the gate rejects
  (research/24 §6.6, §6.7).
* NOTES cover every particular the target uses that the draft lacks
  (numbers, names, quotations, citations, dates), as short clause windows
  (<= 12 words) rather than whole sentences. Rows needing more than
  --max-notes items are dropped: the product's users write 3-6 bullets.
* Residue filters for the PMC scrape: stripped figure/table references
  ("shown in.", "( and)", "( a)"), bracket citations, orphan punctuation,
  URLs, lowercase starts, unterminated paragraphs; HTML entities are
  unescaped. Drafts with markdown, LaTeX or meta text ("The passage
  discusses") are dropped: they are not the product's input distribution.
* Shape: ratio 0.60-1.25 (research/23 §0, §7: passers do not expand;
  research/24 §6.4: GPTZero-human candidates are shorter), overlap
  0.35-0.85, FRE >= 10 and not more than 8 points denser than the draft
  (the bench inputs sit at FRE -1 to 19, so an absolute floor of 15 threw
  away the band the product runs in), sentence-length CV >= 0.15
  (research/21 band A floor 0.18), <= 1 formal connective opener, no summary
  closer.
* Token budget. Prompt + completion must fit train.py's --max-seq-length
  (default 1536) or the trainer truncates the loss-bearing completion; 161
  of the 593 v1 rows did not fit. The shortest exemplar is used for pairs
  that would not fit with the rotating one; pairs that still do not fit are
  dropped.
* Wiki section headings glued onto paragraph starts by
  `build_pairs.paragraphs()` ("Purification High thorium...") are removed by
  looking the paragraph up in the raw file.
* Train/valid split by document (hash of the file name), so no article
  contributes to both.
* Per-document cap (--per-doc, default 6), a wiki floor (--max-wiki-share,
  default 0.25) and wiki rows written twice to train.jsonl (--wiki-repeat 2):
  wiki targets had 3% residue against 34-39% for PMC and all seven wiki pairs
  in the hand review were prose a product should imitate, but only ~64
  survive, so they are weighted rather than PMC discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanizer.humanize.engine import invariants  # noqa: E402
from humanizer.humanize.exemplars import FREEFORM_SHOTS  # noqa: E402
from humanizer.humanize.llm import _FREEFORM_HEADER, _FREEFORM_NO_NOTES  # noqa: E402
from humanizer.humanize.pipeline import content_overlap, flesch_reading_ease, new_specifics  # noqa: E402
from humanizer.text import split_sentences  # noqa: E402

CONNECTIVE_RE = re.compile(
    r"^(However|Moreover|Furthermore|Additionally|Consequently|Therefore|Thus|Hence|"
    r"In addition|In conclusion|Overall|Ultimately|Notably|Importantly|Similarly|"
    r"Nevertheless|Nonetheless|Accordingly|Subsequently|Indeed|For example|For instance),?\s",
    re.IGNORECASE,
)
CLOSER_RE = re.compile(r"^(In conclusion|Overall|In summary|To summarize|Ultimately|In short)\b", re.IGNORECASE)

#: Residue of stripped figure, table and reference markers in the PMC scrape,
#: plus scrape damage. Each alternative was checked against samples in the
#: audit; "(P)" and "(i)" style parentheticals are deliberately not matched.
HUMAN_RESIDUE = re.compile(
    r"\b(?:shown|presented|summari[sz]ed|listed|depicted|illustrated|described|seen|given|displayed|"
    r"provided|reported|indicated|found|observed|detailed|outlined|demonstrated)\s+in\s*[.,;)]"
    r"|(?<!\d\s)(?<!\d)\bin\s*\)"
    r"|\(\s*(?:and|see|;|,)\b|\(\s*[;,]"
    r"|\bsee\s*[.,;)]"
    r"|\b(?:as|and|or)\s*[.,;]\s"
    r"|\bet al\.?,?\s*\d{4}\s*;\s*\)"
    r"|\bin\s+(?:Table|Fig)\b"
    r"|\[\d"
    r"|\s[.,;](?:\s|$)"
    r"|\(\s+[A-Za-z]\s*\)|\(\s*[a-z]\s*\)"
    r"|\bFig(?:ure|s)?\.?\s*\d|\bTable\s*\d|\bSupplementary\b"
    r"|https?://|doi\.org"
    r"|,\s*\)"
)
DRAFT_BAD = re.compile(
    r"(?:^|\s)###|(?:^|\s)---(?:\s|$)|\*\*|(?:^|\.\s|:\s)- [A-Z]|\\\(|\\text|\\\["
    r"|\b(?:The passage|This passage|the original (?:text|passage)|the given (?:text|passage)|"
    r"paraphrased? (?:version|text)|rewritten version|This (?:paraphrase|rewrite|version))\b"
    r"|^(?:Here|Sure|Certainly)\b"
    r"|(?<!\*)\*[^*\s][^*]*\*(?!\*)",
)
#: Capitalised words the paraphraser's formatting turns into "entities".
COMMON_CAPS = frozenset(
    "additionally however this these that those before after although while moreover furthermore "
    "therefore thus hence overall finally first second third notably importantly similarly conversely "
    "specifically in the a an it there for from with when where which as such also then here".split()
)

_TOKENIZER = None


def token_count(text: str) -> int:
    """Exact Qwen token count when the cached tokenizer.json is available
    (the `tokenizers` package, no model load), else chars / 3.6."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = False
        try:
            from tokenizers import Tokenizer  # standalone Rust package

            cands = sorted((Path.home() / ".cache/huggingface/hub").glob("models--mlx-community--Qwen2.5-7B-4bit/snapshots/*/tokenizer.json"))
            if cands:
                _TOKENIZER = Tokenizer.from_file(str(cands[-1]))
        except Exception:  # noqa: BLE001
            _TOKENIZER = False
    if _TOKENIZER:
        return len(_TOKENIZER.encode(text).ids)
    return int(len(text) / 3.6) + 1


def clean(text: str) -> str:
    return " ".join(html.unescape(text).split())


META_SENTENCE = re.compile(
    r"\b(?:The passage|This passage|the original (?:text|passage)|the given (?:text|passage)|"
    r"paraphrased? (?:version|text)|rewritten version|This (?:paraphrase|rewrite|version))\b"
)


def repair_draft(draft: str) -> str:
    """Strip the paraphraser's markdown (headings, bold, rules, bullets,
    italics) and its meta sentences ("The passage discusses..."). Content is
    untouched; what remains is judged by `keep` like any other draft."""
    d = re.sub(r"(?:^|\s)###\s*", " ", draft)
    d = re.sub(r"\*\*", "", d)
    d = re.sub(r"(?:^|\s)---(?=\s|$)", " ", d)
    d = re.sub(r"(?<!\*)\*([^*\s][^*]*)\*(?!\*)", r"\1", d)
    d = re.sub(r"(?:(?<=^)|(?<=[.:;]\s)|(?<=\s))- (?=[A-Z])", "", d)
    sents = [x for x in split_sentences(d) if not META_SENTENCE.search(x)]
    return " ".join(" ".join(sents).split())


def repair_human(human: str) -> str:
    """Undo the scrape damage that has a clean inverse: a space before
    punctuation where a reference was cut ("described ."), and empty or
    marker-only parentheses left by a stripped figure reference ("( and, and)",
    "( a)"). Sentences whose grammar was broken by the cut ("shown in.") are
    not repairable and are dropped by `keep`."""
    h = re.sub(r"\(\s*(?:and|see|;|,)?[\s,;]*(?:and)?[\s,;]*\)", "", human)
    h = re.sub(r"\(\s+[A-Za-z]\s*\)|\(\s*[a-z]\s*\)", "", h)
    h = re.sub(r"\s+([.,;:])", r"\1", h)
    return " ".join(h.split())


_RAW_DIRS = (ROOT / "data/raw/pmc", ROOT / ".hiplora/raw/pmc2", ROOT / ".hiplora/raw/wikipedia")
_HEADINGS = None


def _heading_index():
    """Map from a paragraph's whitespace-normalised text to the section
    heading `build_pairs.paragraphs()` glued onto its front: the raw files
    keep the heading on its own line inside the paragraph block."""
    global _HEADINGS
    if _HEADINGS is not None:
        return _HEADINGS
    _HEADINGS = {}
    for d in _RAW_DIRS:
        for path in d.glob("*.txt") if d.exists() else []:
            for block in re.split(r"\n\s*\n", path.read_text(errors="ignore")):
                lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
                if len(lines) < 2:
                    continue
                head = lines[0]
                if len(head.split()) <= 8 and not re.search(r"[.!?:;,]$", head) and head[:1].isupper():
                    joined = " ".join(block.split())
                    _HEADINGS[joined] = head
    return _HEADINGS


def strip_heading(human: str) -> str:
    """Drop a glued section heading ("Soft parts Females of L. canarium...")
    when the raw file shows one; otherwise return the text unchanged."""
    head = _heading_index().get(human)
    if head and human.startswith(head + " "):
        rest = human[len(head) + 1:]
        if rest[:1].isupper():
            return rest
    return human


def _strip_quote(q: str) -> str:
    return re.sub(r"[\"“”'‘’.,;:!?\s]+", " ", q).strip().lower()


def sentence_cv(text: str) -> float:
    lens = [len(s.split()) for s in split_sentences(text) if s.strip()]
    if len(lens) < 2:
        return 0.0
    m = statistics.mean(lens)
    return statistics.pstdev(lens) / m if m else 0.0


def missing_particulars(draft: str, human: str):
    """Everything in `human` that `_invariant_failures` would refuse without
    notes: numbers, quotations, citations (multiset extras) and mid-sentence
    capitalised runs the draft lacks. Quotations first so the window can
    carry the whole quotation."""
    b, a = invariants(draft), invariants(human)
    out = []
    for key in ("quotations", "citations", "numbers"):
        extra = a[key] - b[key]
        out += sorted(extra, key=len, reverse=True)
    # Same test `pipeline.new_specifics` applies: a name that is only
    # sentence-initial in the draft counts as new when the target uses it
    # mid-sentence, so the notes must carry it.
    out += sorted((e for e in a["entities"] if e not in b["entities"]), key=len, reverse=True)
    return out


def window(sentence: str, needle: str, max_words: int = 12) -> str:
    """Up to `max_words` words of `sentence` around `needle`; the whole needle
    always survives, so a long quotation is returned intact."""
    n_words = needle.split()
    if len(n_words) >= max_words:
        return needle
    words = sentence.split()
    first = n_words[0]
    # Exact token first ("2" must not land on "2019", "Louis" not on
    # "Louisiana"), then a run of tokens, then substring as a last resort.
    idx = next((k for k in range(len(words) - len(n_words) + 1)
                if [re.sub(r"^\W+|\W+$", "", w) for w in words[k:k + len(n_words)]] == [re.sub(r"^\W+|\W+$", "", x) for x in n_words]), None)
    if idx is None:
        idx = next((k for k, w in enumerate(words) if re.sub(r"^\W+|\W+$", "", w) == first), None)
    if idx is None:
        idx = next((k for k, w in enumerate(words) if first in w), None)
    if idx is None:
        return needle
    span = len(n_words)
    lo = max(0, idx - (max_words - span) // 2)
    hi = min(len(words), lo + max_words)
    lo = max(0, hi - max_words)
    text = " ".join(words[lo:hi]).strip(" ,;:")
    return text if needle in text else needle


def build_notes(draft: str, human: str, rng: random.Random, max_items: int):
    """NOTES text, or None when the target needs more than `max_items`."""
    missing = missing_particulars(draft, human)
    if not missing:
        return _FREEFORM_NO_NOTES
    sentences = split_sentences(human)
    items = []  # (sentence index, set of needles)
    for needle in missing:
        for i, s in enumerate(sentences):
            if needle in s:
                for it in items:
                    if it[0] == i:
                        it[1].append(needle)
                        break
                else:
                    items.append((i, [needle]))
                break
    if len(items) > max_items:
        return None
    lines = []
    for i, needles in items:
        s = sentences[i]
        # One window per sentence, wide enough to hold every needle in it.
        text = window(s, needles[0])
        for n in needles[1:]:
            if n not in text:
                text = text + " ... " + window(s, n, 8) if n not in s[: s.find(text)] else window(s, n) + " ... " + text
        lines.append("- " + text)
    joined = "\n".join(lines)
    for needle in missing:
        if needle not in joined:
            for sent in sentences:
                if needle in sent:
                    lines.append("- " + window(sent, needle))
                    joined = "\n".join(lines)
                    break
    if len(lines) > max_items:
        return None
    rng.shuffle(lines)
    return "\n".join(lines)


def entity_loss(draft: str, human: str) -> bool:
    """True when a draft name vanished from the target entirely."""
    low = human.lower()
    for ent in invariants(draft)["entities"]:
        if ent in human:
            continue
        toks = [t for t in re.findall(r"[A-Za-z0-9&'’-]+", ent) if t.lower() not in COMMON_CAPS]
        if not toks:
            continue
        if any(t.lower() in low for t in toks if len(t) > 2):
            continue
        return True
    return False


def keep(draft: str, human: str, args) -> str:
    """Empty string to keep, else the reason to drop."""
    if DRAFT_BAD.search(draft):
        return "draft_markdown_or_meta"
    if HUMAN_RESIDUE.search(human):
        return "citation_residue"
    if human[:1].islower():
        return "lowercase_start"
    if not re.search(r"[.!?\"'”’)\]]$", human):
        return "unterminated"
    hw, dw = len(human.split()), len(draft.split())
    if hw < args.min_words:
        return "human_length"
    ratio = hw / max(1, dw)
    if not (args.min_ratio <= ratio <= args.max_ratio):
        return "ratio"
    ov = content_overlap(draft, human)
    if not (args.min_overlap <= ov <= args.max_overlap):
        return "overlap"
    fre_h, fre_d = flesch_reading_ease(human), flesch_reading_ease(draft)
    if fre_h < args.min_fre or fre_h > 80.0 or fre_h < fre_d - args.max_fre_drop:
        return "fre"
    sents = split_sentences(human)
    if len(sents) < 3:
        return "sentences"
    if sentence_cv(human) < args.min_cv:
        return "flat_cv"
    if sum(1 for s in sents if CONNECTIVE_RE.match(s)) > 1:
        return "connectives"
    if CLOSER_RE.match(sents[-1]):
        return "closer"
    b, a = invariants(draft), invariants(human)
    # Numbers: every distinct draft number must survive at least once. The
    # paraphraser repeats years the target states once; a rewrite that
    # de-duplicates is not a factual change. Citations and dates stay exact.
    if any(a["numbers"][item] == 0 for item in b["numbers"]):
        return "draft_particular_lost"
    for key in ("citations", "dates"):
        if any(a[key][item] < n for item, n in b[key].items()):
            return "draft_particular_lost"
    # The paraphraser shortens quotations; the target gives them in full. A
    # draft quotation counts as kept when its words survive inside the
    # target (the full quotation reaches the model through NOTES).
    h_low = human.lower()
    for q in b["quotations"]:
        if a["quotations"][q] < b["quotations"][q] and _strip_quote(q) not in re.sub(r"[\"“”'‘’.,;:!?\s]+", " ", h_low):
            return "draft_particular_lost"
    if entity_loss(draft, human):
        return "draft_entity_lost"
    return ""


def shot_block(shot) -> str:
    _topic, draft, notes, rewrite = shot
    return f"DRAFT:\n{draft}\n\nNOTES:\n{notes}\n\nHUMAN:\n{rewrite}\n\n"


def make_prompt(shot, draft: str, notes: str) -> str:
    return _FREEFORM_HEADER + shot_block(shot) + "DRAFT:\n" + draft + "\n\nNOTES:\n" + notes + "\n\nHUMAN:\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / ".hiplora/pairs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / ".hip7b/data_v2"))
    ap.add_argument("--valid-share", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-wiki-share", type=float, default=0.25, help="0 keeps every pair; otherwise cap PMC so wiki is at least this share")
    ap.add_argument("--per-doc", type=int, default=6, help="max pairs kept from one source document")
    ap.add_argument("--max-notes", type=int, default=6)
    ap.add_argument("--max-seq-length", type=int, default=1536, help="must match train.py")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--min-ratio", type=float, default=0.60)
    ap.add_argument("--max-ratio", type=float, default=1.25)
    ap.add_argument("--min-overlap", type=float, default=0.35)
    ap.add_argument("--max-overlap", type=float, default=0.85)
    ap.add_argument("--min-fre", type=float, default=10.0)
    ap.add_argument("--min-cv", type=float, default=0.15)
    ap.add_argument("--max-fre-drop", type=float, default=8.0, help="drop targets denser than their draft by more FRE points than this")
    ap.add_argument("--wiki-repeat", type=int, default=2, help="emit each wiki training row this many times")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.pairs)]
    kept, reasons = [], Counter()
    shortest_shot = min(FREEFORM_SHOTS, key=lambda s: len(shot_block(s)))
    for i, r in enumerate(rows):
        human_raw = " ".join(r["human"].split())
        stripped = strip_heading(human_raw)
        if stripped != human_raw:
            reasons["heading_stripped(kept)"] += 1
        draft, human = repair_draft(clean(r["ai"])), repair_human(clean(stripped))
        why = keep(draft, human, args)
        if why:
            reasons[why] += 1
            continue
        notes = build_notes(draft, human, rng, args.max_notes)
        if notes is None:
            reasons["too_many_particulars"] += 1
            continue
        if new_specifics(draft, human, facts="" if notes == _FREEFORM_NO_NOTES else notes):
            reasons["unlicensed_specific"] += 1
            continue
        shot = FREEFORM_SHOTS[i % len(FREEFORM_SHOTS)]
        completion = human + "\n\n"
        n_tok = token_count(make_prompt(shot, draft, notes) + completion) + 1
        if n_tok > args.max_seq_length:
            shot = shortest_shot
            n_tok = token_count(make_prompt(shot, draft, notes) + completion) + 1
            if n_tok > args.max_seq_length:
                reasons["too_long_for_seq"] += 1
                continue
        kept.append({"source": r["source"], "doc": r["doc"], "draft": draft, "human": human, "notes": notes, "shot": shot, "tokens": n_tok})
    print("kept %d of %d; dropped: %s" % (len(kept), len(rows), json.dumps(dict(sorted(reasons.items())))), file=sys.stderr)

    # Per-document cap, then wiki floor.
    rng.shuffle(kept)
    per_doc, capped = Counter(), []
    for k in kept:
        if per_doc[k["doc"]] >= args.per_doc:
            reasons["per_doc_cap"] += 1
            continue
        per_doc[k["doc"]] += 1
        capped.append(k)
    kept = capped
    wiki = [k for k in kept if k["source"] == "wiki"]
    pmc = [k for k in kept if k["source"] != "wiki"]
    if args.max_wiki_share > 0 and wiki:
        max_pmc = int(len(wiki) * (1 - args.max_wiki_share) / args.max_wiki_share)
        pmc = pmc[:max_pmc] if len(pmc) > max_pmc else pmc
    kept = wiki + pmc
    rng.shuffle(kept)
    print("after per-doc cap %d and balancing: %d (wiki %d, pmc %d); docs %d" % (args.per_doc, len(kept), len(wiki), len(pmc), len({k["doc"] for k in kept})), file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_f = (out / "train.jsonl").open("w")
    valid_f = (out / "valid.jsonl").open("w")
    n_train = n_valid = n_no_notes = 0
    max_tok = 0
    for k in kept:
        if k["notes"] == _FREEFORM_NO_NOTES:
            n_no_notes += 1
        prompt = make_prompt(k["shot"], k["draft"], k["notes"])
        row = {"prompt": prompt, "completion": k["human"] + "\n\n", "source": k["source"], "doc": k["doc"]}
        max_tok = max(max_tok, k["tokens"])
        # Split by document, not by paragraph: with a per-paragraph hash 30 of
        # the 31 v1 validation documents also had paragraphs in train, so the
        # validation loss measured memorisation of the source article.
        h = int(hashlib.sha1(k["doc"].encode()).hexdigest(), 16) % 1000
        if h < args.valid_share * 1000:
            valid_f.write(json.dumps(row) + "\n")
            n_valid += 1
        else:
            reps = args.wiki_repeat if k["source"] == "wiki" else 1
            for _ in range(reps):
                train_f.write(json.dumps(row) + "\n")
                n_train += 1
    train_f.close()
    valid_f.close()
    print("train %d, valid %d, no-notes rows %d, longest row %d tokens" % (n_train, n_valid, n_no_notes, max_tok), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
