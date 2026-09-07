"""Sentence-structure features: cadence, openers, clause placement.

`shape.py` measures how long sentences are and how the length series moves.
`syntax.py` measures what is inside a clause. This module measures the thing
between the two, which the feature inventory found missing and which
research/06 says GPTZero's own cadence heuristics look at: how sentences
*begin*, whether their lengths park in a narrow band for runs at a time,
where the commas fall, whether a very short sentence ever sits next to a very
long one, and (with spaCy) where the dependent clause sits relative to the
main verb.

Why these and not more word-level features
------------------------------------------
research/06 §3 and research/13 §1 put word-choice levers near the floor:
synonym swapping moves GPTZero by about -5.5 points at 5% FPR, and vocabulary
swaps "change highlights but do not directly affect probability scores". The
one structural lever with a peer-reviewed effect is sentence-length variance
(Perkins: detector accuracy 39.5% -> 15.9%). research/06 §1 and §6 list the
cadence patterns GPTZero's own material names as flags:

  * three or more consecutive sentences of 17-23 words ("metronome");
  * a mean sentence length parked at 18-24 words;
  * more than half a paragraph's sentences opening "The", "This", "It" or "In";
  * three-part lists and "not just X, but Y" parallelism.

Those are the features here, plus the opener taxonomy and comma-rhythm series
a stylometrician would ask for next to them.

What this module does NOT claim
-------------------------------
None of these is a detector or a score. They describe the text. The opener
taxonomy is a word list, not a parse; the fragment test is a heuristic that
will call some elliptical human sentences fragments and miss some verbless
AI ones. The spaCy features are absent, not zero, when the parser is not
installed, exactly as `syntax.py` degrades.

Human bands, measured
---------------------
Every band is the 20th-80th percentile measured over `data/raw/pmc/*.txt`: 60
human research articles, biomedical register, 2026-09-07. They describe *that*
register; an essay corpus would move several of them (research/16: sample, do
not centre). The three research/06 heuristics are kept separately in
`GPTZERO_HEURISTICS`, because **two of them are violated by most of the human
corpus**: PMC authors run three or more 17-23-word sentences in a row in half
the articles (median run 3), and in a typical article the worst paragraph opens
more than half its sentences on The/This/It/In (median 0.667). Treat those two
as "what GPTZero's material says it looks for", not as what humans do.

    feature                                       p20      p50      p80
    -------------------------------------------------------------------
    struct_opener_repeat_share                  0.051    0.072    0.098
    struct_opener_distinct_ratio                0.415    0.464    0.548
    struct_opener_first_pos_entropy             1.651    1.718    1.835
    struct_opener_det_share                     0.222    0.283    0.361
    struct_opener_pronoun_share                 0.016    0.044    0.084
    struct_opener_adverbial_share               0.241    0.312    0.397
    struct_opener_conj_share                    0.000    0.000    0.001
    struct_opener_content_share                 0.262    0.339    0.414
    struct_the_this_it_in_share                 0.226    0.293    0.386
    struct_para_max_the_this_it_in_share        0.620    0.667    0.867
    struct_metronome_share                      0.233    0.283    0.332
    struct_metronome_run_max                    2        3        5
    struct_metronome_runs_3plus_per_1k          0.000    0.547    1.075
    struct_fragment_share                       0.000    0.000    0.011
    struct_comma_per_sentence_mean              1.030    1.312    1.618
    struct_comma_per_sentence_cv                1.047    1.185    1.363
    struct_first_comma_position_mean            0.291    0.321    0.365
    struct_comma_free_share                     0.142    0.199    0.286
    struct_len_contrast_adjacent_max            1.313    1.440    1.588
    struct_short_after_long_per_1k              0.000    0.357    0.909
    struct_para_open_close_len_ratio_mean       1.001    1.210    1.387
    struct_tricolon_share                       0.028    0.071    0.115
    struct_question_share                       0.000    0.000    0.000
    struct_sentence_type_simple_share           0.528    0.600    0.679   (spaCy)
    struct_sentence_type_compound_share         0.029    0.052    0.077   (spaCy)
    struct_sentence_type_complex_share          0.240    0.303    0.382   (spaCy)
    struct_sentence_type_compound_complex_share 0.008    0.025    0.051   (spaCy)
    struct_initial_subordinate_share            0.016    0.029    0.055   (spaCy)
    struct_final_subordinate_share              0.030    0.054    0.077   (spaCy)
    struct_dep_depth_cv                         0.341    0.374    0.410   (spaCy)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..text import Document

try:  # the same pattern register.py uses; mirrored below if it ever moves
    from .register import _TRICOLON_RE as TRICOLON_RE
except ImportError:  # pragma: no cover - defensive
    TRICOLON_RE = re.compile(
        r"\b[\w'-]+(?:\s+[\w'-]+){0,3},\s+[\w'-]+(?:\s+[\w'-]+){0,3},\s+and\s+[\w'-]+",
    )

__all__ = [
    "extract",
    "extract_syntax",
    "band_report",
    "opener_class",
    "METRONOME_BAND",
    "HUMAN_BANDS",
    "GPTZERO_HEURISTICS",
]

# research/06 §1: GPTZero's cadence heuristic band.
METRONOME_BAND = (17, 23)
#: Fragment heuristic floor: fewer words than this is a fragment regardless.
FRAGMENT_MAX_WORDS = 3
#: "Short after long" contrast thresholds.
SHORT_AFTER_LONG = (8, 25)
#: Paragraphs shorter than this are skipped by the per-paragraph opener share.
MIN_PARA_SENTENCES = 3

# --------------------------------------------------------- opener taxonomy

_DETERMINER_OPENERS = frozenset(
    {"the", "a", "an", "this", "that", "these", "those", "it", "there"}
)
_PRONOUN_OPENERS = frozenset(
    {"i", "we", "you", "he", "she", "they", "one", "my", "our", "their", "his", "her"}
)
_ADVERBIAL_OPENERS = frozenset(
    {
        "in", "on", "at", "by", "for", "with", "from", "after", "before", "during",
        "despite", "although", "though", "while", "when", "whenever", "if",
        "unless", "because", "as", "since", "until", "once", "however", "thus",
        "hence", "therefore", "moreover", "furthermore", "nevertheless",
        "nonetheless", "meanwhile", "instead", "otherwise", "first", "second",
        "third", "finally", "then", "now", "here", "yet", "still", "also",
        "even", "perhaps", "of", "to", "under", "over", "among", "between",
        "through", "within", "without", "across", "against", "beyond", "given",
        "unlike", "like", "according",
    }
)
_CONJUNCTION_OPENERS = frozenset({"and", "but", "or", "so", "nor"})
_THE_THIS_IT_IN = frozenset({"the", "this", "it", "in"})

#: Words that count as a verb for the fragment heuristic even without a
#: -s/-ed/-ing ending.
_COMMON_VERBS = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "being", "am", "has", "have",
        "had", "do", "does", "did", "can", "could", "will", "would", "may",
        "might", "must", "should", "shall", "get", "got", "go", "went", "gone",
        "make", "made", "take", "took", "taken", "come", "came", "see", "saw",
        "seen", "know", "knew", "known", "think", "thought", "say", "said",
        "find", "found", "give", "gave", "given", "show", "shown", "become",
        "became", "remain", "seem", "keep", "kept", "let", "put", "set", "run",
        "ran", "lead", "led", "hold", "held", "mean", "meant", "need", "want",
        "tell", "told", "begin", "began", "begun", "bring", "brought", "write",
        "wrote", "written", "read", "leave", "left", "feel", "felt", "lose",
        "lost", "pay", "paid", "meet", "met", "grow", "grew", "grown", "cut",
        "fall", "fell", "fallen", "rise", "rose", "risen", "spend", "spent",
        "build", "built", "send", "sent", "stand", "stood", "understand",
        "understood", "draw", "drew", "drawn", "occur", "suggest", "indicate",
        "require", "include", "involve", "allow", "provide", "use", "cause",
        "depend", "differ", "vary", "tend", "appear", "exist", "matter",
        "yield", "reflect", "reveal", "predict", "reduce", "increase",
    }
)

_INTERNAL_PUNCT_RE = re.compile(r"[,;:—–()\-]")
_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


def opener_class(word: Optional[str]) -> str:
    """Cheap opener taxonomy: det, pronoun, adverbial, conj or content.

    A word list, not a parse. `-ly` words are treated as adverbial because a
    sentence-initial adverb is the fronted-adverbial move this class exists to
    count; "Only" and "Early" are counted with them, which is acceptable
    noise for a shares-and-entropy feature.
    """
    if not word:
        return "content"
    w = word.lower()
    if w in _DETERMINER_OPENERS:
        return "det"
    if w in _PRONOUN_OPENERS:
        return "pronoun"
    if w in _CONJUNCTION_OPENERS:
        return "conj"
    if w in _ADVERBIAL_OPENERS or (w.endswith("ly") and len(w) > 3):
        return "adverbial"
    return "content"


def _first_word(sentence_words: Sequence[str]) -> Optional[str]:
    for w in sentence_words:
        cleaned = _STRIP_RE.sub("", w).lower()
        if cleaned:
            return cleaned
    return None


def _entropy(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return float("nan")
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _cv(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    mean = arr.mean()
    if mean == 0:
        return float("nan")
    return float(arr.std(ddof=1) / mean)


def _looks_like_verb(word: str) -> bool:
    w = word.lower()
    if w in _COMMON_VERBS:
        return True
    return len(w) > 3 and (w.endswith("ed") or w.endswith("ing") or w.endswith("s"))


def _is_fragment(sentence_words: Sequence[str]) -> bool:
    """Heuristic: very short, or nothing that looks like a verb.

    The -s test means any plural noun rescues a sentence from being called a
    fragment, so this under-counts fragments in noun-heavy prose and over-counts
    them in short imperative or elliptical sentences. It is a signal about the
    *rate*, not a grammar check.
    """
    if len(sentence_words) <= FRAGMENT_MAX_WORDS:
        return True
    return not any(_looks_like_verb(w) for w in sentence_words)


def _runs_in_band(lengths: Sequence[int], band) -> List[int]:
    """Lengths of every maximal run of consecutive sentences inside `band`."""
    runs: List[int] = []
    current = 0
    for n in lengths:
        if band[0] <= n <= band[1]:
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


# ------------------------------------------------------------- extraction


def extract(doc: Document) -> Dict[str, float]:
    """All `struct_*` features. Never raises; spaCy features only when present."""
    feats: Dict[str, float] = {}
    sents = doc.sentences
    n = len(sents)
    if n == 0:
        return feats
    n_words = doc.n_words
    per_1k = 1000.0 / n_words if n_words else float("nan")

    lengths = [s.length for s in sents]
    firsts = [_first_word(s.words) for s in sents]

    # -- openers ----------------------------------------------------------
    repeats = sum(1 for i in range(1, n) if firsts[i] and firsts[i] == firsts[i - 1])
    feats["struct_opener_repeat_share"] = repeats / (n - 1) if n > 1 else 0.0
    feats["struct_opener_distinct_ratio"] = len({f for f in firsts if f}) / n

    classes = Counter(opener_class(f) for f in firsts)
    for cls in ("det", "pronoun", "adverbial", "conj", "content"):
        feats["struct_opener_%s_share" % cls] = classes.get(cls, 0) / n
    feats["struct_opener_first_pos_entropy"] = _entropy(list(classes.values()))

    ttii = [1 if f in _THE_THIS_IT_IN else 0 for f in firsts]
    feats["struct_the_this_it_in_share"] = sum(ttii) / n
    per_para: Dict[int, List[int]] = {}
    for s, flag in zip(sents, ttii):
        per_para.setdefault(s.paragraph_index, []).append(flag)
    # Only paragraphs of three or more sentences: a one-sentence paragraph
    # opening "The" would otherwise make the maximum 1.0 trivially.
    eligible = [v for v in per_para.values() if len(v) >= MIN_PARA_SENTENCES]
    feats["struct_para_max_the_this_it_in_share"] = (
        max(sum(v) / len(v) for v in eligible) if eligible else float("nan")
    )

    # -- metronome --------------------------------------------------------
    in_band = [1 for L in lengths if METRONOME_BAND[0] <= L <= METRONOME_BAND[1]]
    feats["struct_metronome_share"] = len(in_band) / n
    runs = _runs_in_band(lengths, METRONOME_BAND)
    feats["struct_metronome_run_max"] = float(max(runs) if runs else 0)
    feats["struct_metronome_runs_3plus_per_1k"] = (
        sum(1 for r in runs if r >= 3) * per_1k if n_words else float("nan")
    )

    # -- fragments --------------------------------------------------------
    feats["struct_fragment_share"] = sum(1 for s in sents if _is_fragment(s.words)) / n

    # -- comma rhythm -----------------------------------------------------
    commas = [s.text.count(",") for s in sents]
    feats["struct_comma_per_sentence_mean"] = float(np.mean(commas))
    feats["struct_comma_per_sentence_cv"] = _cv(commas)
    positions: List[float] = []
    for s in sents:
        if "," in s.text and s.length:
            before = len(re.findall(r"[A-Za-z0-9][\w'-]*", s.text.split(",", 1)[0]))
            positions.append(min(1.0, before / s.length))
    feats["struct_first_comma_position_mean"] = (
        float(np.mean(positions)) if positions else float("nan")
    )
    feats["struct_comma_free_share"] = (
        sum(1 for s in sents if not _INTERNAL_PUNCT_RE.search(s.text)) / n
    )

    # -- length contrast ---------------------------------------------------
    if n > 1:
        contrasts = [
            abs(a - b) / ((a + b) / 2.0) if (a + b) else 0.0
            for a, b in zip(lengths[:-1], lengths[1:])
        ]
        feats["struct_len_contrast_adjacent_max"] = float(max(contrasts))
        short_after_long = sum(
            1
            for a, b in zip(lengths[:-1], lengths[1:])
            if a >= SHORT_AFTER_LONG[1] and b <= SHORT_AFTER_LONG[0]
        )
        feats["struct_short_after_long_per_1k"] = (
            short_after_long * per_1k if n_words else float("nan")
        )
    else:
        feats["struct_len_contrast_adjacent_max"] = float("nan")
        feats["struct_short_after_long_per_1k"] = float("nan")

    # -- paragraph open/close ---------------------------------------------
    ratios: List[float] = []
    by_para: Dict[int, List[int]] = {}
    for s in sents:
        by_para.setdefault(s.paragraph_index, []).append(s.length)
    for seq in by_para.values():
        if len(seq) > 1 and seq[-1] > 0:
            ratios.append(seq[0] / seq[-1])
    feats["struct_para_open_close_len_ratio_mean"] = (
        float(np.mean(ratios)) if ratios else float("nan")
    )

    # -- parallelism and mood ---------------------------------------------
    feats["struct_tricolon_share"] = sum(1 for s in sents if TRICOLON_RE.search(s.text)) / n
    feats["struct_question_share"] = sum(1 for s in sents if s.text.rstrip().endswith("?")) / n

    try:
        feats.update(extract_syntax(doc))
    except Exception:  # noqa: BLE001 - the parser must never take the rest down
        pass
    return feats


def extract_syntax(doc: Document) -> Dict[str, float]:
    """Sentence-type mix and clause placement. `{}` without spaCy.

    Sentence typology follows the school taxonomy because that is what the
    research reports and the writing guides use: one finite clause is simple;
    coordinated finite clauses with no dependent clause is compound; one or
    more dependent clauses and no coordination is complex; both is
    compound-complex. Dependent clause heads are `advcl`, `ccomp`, `xcomp`
    (finite only), `relcl` and `acl` with a subject. Clause placement is read
    off `advcl` heads only, because "initial vs final subordinate clause" is a
    statement about adverbial clauses, not about relatives.
    """
    from . import syntax  # optional extra

    if not syntax.available():
        return {}
    if not doc.text.strip():
        return {}
    nlp = syntax.load_model()
    parsed = nlp(doc.text[: nlp.max_length])
    sentences = [s for s in parsed.sents if any(not t.is_space for t in s)]
    if not sentences:
        return {}

    dep_heads = ("advcl", "ccomp", "relcl", "acl", "csubj", "csubjpass")
    types = Counter()
    initial = 0
    final = 0
    with_advcl = 0
    depths: List[int] = []
    for sent in sentences:
        depths.append(syntax._dependency_depth(sent))
        root = next((t for t in sent if t.dep_ == "ROOT"), sent.root)
        coordinated = syntax._t_unit_count(sent) > 1
        dependent = any(
            t.dep_ in dep_heads and syntax._is_finite_clause_head(t) for t in sent
        )
        if coordinated and dependent:
            types["compound_complex"] += 1
        elif coordinated:
            types["compound"] += 1
        elif dependent:
            types["complex"] += 1
        else:
            types["simple"] += 1
        advcls = [t for t in sent if t.dep_ == "advcl" and syntax._is_finite_clause_head(t)]
        if advcls:
            with_advcl += 1
            first = min(advcls, key=lambda t: t.i)
            # The clause "precedes the root" when its leftmost token does.
            left_edge = min(tok.i for tok in first.subtree)
            if left_edge < root.i:
                initial += 1
            else:
                final += 1

    n = len(sentences)
    out: Dict[str, float] = {}
    for key in ("simple", "compound", "complex", "compound_complex"):
        out["struct_sentence_type_%s_share" % key] = types.get(key, 0) / n
    out["struct_initial_subordinate_share"] = initial / n
    out["struct_final_subordinate_share"] = final / n
    out["struct_dep_depth_cv"] = _cv(depths)
    return out


# ------------------------------------------------------------------ bands


#: research/06 §1, §6: the cadence patterns GPTZero's own material names as
#: flags. `(threshold, direction)`: a value past the threshold in that direction
#: is what the heuristic flags. Reported for explanation only; see the module
#: docstring for why these are not human bands.
GPTZERO_HEURISTICS: Dict[str, tuple] = {
    "struct_metronome_run_max": (3.0, "high"),
    "struct_para_max_the_this_it_in_share": (0.5, "high"),
    "struct_opener_repeat_share": (0.15, "high"),
}

#: PMC p20-p80, 60 documents. See the module docstring table.
HUMAN_BANDS: Dict[str, tuple] = {
    "struct_opener_repeat_share": (0.051, 0.098),
    "struct_opener_distinct_ratio": (0.415, 0.548),
    "struct_opener_first_pos_entropy": (1.651, 1.835),
    "struct_opener_det_share": (0.222, 0.361),
    "struct_opener_pronoun_share": (0.016, 0.084),
    "struct_opener_adverbial_share": (0.241, 0.397),
    "struct_opener_content_share": (0.262, 0.414),
    "struct_the_this_it_in_share": (0.226, 0.386),
    "struct_para_max_the_this_it_in_share": (0.620, 0.867),
    "struct_metronome_share": (0.233, 0.332),
    "struct_metronome_run_max": (2.0, 5.0),
    "struct_metronome_runs_3plus_per_1k": (0.0, 1.075),
    "struct_fragment_share": (0.0, 0.011),
    "struct_comma_per_sentence_mean": (1.030, 1.618),
    "struct_comma_per_sentence_cv": (1.047, 1.363),
    "struct_first_comma_position_mean": (0.291, 0.365),
    "struct_comma_free_share": (0.142, 0.286),
    "struct_len_contrast_adjacent_max": (1.313, 1.588),
    "struct_short_after_long_per_1k": (0.0, 0.909),
    "struct_para_open_close_len_ratio_mean": (1.001, 1.387),
    "struct_tricolon_share": (0.028, 0.115),
    "struct_sentence_type_simple_share": (0.528, 0.679),
    "struct_sentence_type_compound_share": (0.029, 0.077),
    "struct_sentence_type_complex_share": (0.240, 0.382),
    "struct_sentence_type_compound_complex_share": (0.008, 0.051),
    "struct_initial_subordinate_share": (0.016, 0.055),
    "struct_final_subordinate_share": (0.030, 0.077),
    "struct_dep_depth_cv": (0.341, 0.410),
}


def band_report(feats: Dict[str, float]) -> Dict[str, str]:
    """"low" / "in band" / "high" / "unknown" per banded structure feature."""
    out: Dict[str, str] = {}
    for key, band in HUMAN_BANDS.items():
        value = feats.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out[key] = "unknown"
        elif value < band[0]:
            out[key] = "low"
        elif value > band[1]:
            out[key] = "high"
        else:
            out[key] = "in band"
    return out
