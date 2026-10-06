"""Individually testable rewrite transforms.

Every transform is a pure function ``(Document, HumanizeConfig) -> List[Edit]``.
Nothing here mutates text; the engine collects edits, resolves overlaps, runs
the meaning-preservation gate and only then splices. That separation is what
makes each rule testable in isolation and makes a bad rule removable without
touching the pipeline.

What this module implements, and why only this
----------------------------------------------
research/00 section 4 is a conflict matrix that rates every candidate humanizing
edit by detector benefit *and* by cost to an academic grade. The product thesis
is the asymmetry it documents: the four highest-value edits carry zero or
negative grade cost, while the most grade-damaging edits carry only medium
detector benefit. So we implement the safe region and refuse the rest.

Implemented (all rated "Do" or "Do aggressively", grade cost 0 or negative):

==================================  ===============  ==========
Edit                                Detector benefit Grade cost
==================================  ===============  ==========
Strip sentence-initial connectives  High             Raises it
Kill AI vocabulary                  High             0/positive
Break the paragraph template        High             0/positive
Sentence-length variance into band  High             0/positive
Break tricolons / "not X but Y"     Medium-High      0
Hedge absolute claims               Unmeasured*      0 (academic norm)
Relocate contrastive openers        Low-Medium*      0
Front a trailing clause             Low-Medium*      0
==================================  ===============  ==========

\* The last three implement the manual method transcribed in research/24
(hedging as "intellectual hesitation"; moving "Conversely," to after the
subject and moving a trailing "because ..." clause to the front as
"reversing the order", applied only where consecutive sentences open the
same way). research/24 §2 measured that method at 0 of 5 verdict flips on
the local bench, so they are shipped as quality and de-templating edits with
an honest "no pass-rate claim" label, exactly as research/22 §7.2 labels
every stance and punctuation edit.

REFUSED. These are not unimplemented; they are refused on the evidence, and
they must stay refused. Each line is the matrix verdict and the reason.

* **Contractions in academic body text** - matrix: medium-high detector
  benefit, *medium* grade cost, verdict "Never in body". Contractions are a
  register violation in every academic rubric we surveyed (research/15), and
  `features.punctuation` already caps the acceptable rate near zero.
* **Spelling, subject-verb agreement and wrong-word errors** - medium benefit,
  *high* cost. research/06 also lists deliberate typos under "things that used
  to work but no longer do": detectors now read them as a paraphraser
  signature, so this trades a real grade penalty for no detector gain.
* **Simplifying vocabulary broadly** - medium benefit, *high* cost, verdict
  "Never; only de-Latinize verbs". We therefore replace *AI-marked* words with
  plainer equivalents one by one from a curated map, and never run a general
  simplifier. `significant`, `efficacy`, `paradigm`, `dynamic` and `trajectory`
  are deliberately absent from that map because they are load-bearing technical
  terms in at least one discipline.
* **Adding subordination to sound scholarly** - low benefit, *medium* cost.
  research/15: academic complexity is phrasal, not clausal. Adding dependent
  clauses moves the text toward conversation, away from the target register.
  (`front_trailing_clauses` is not this: it moves a clause the writer already
  wrote to the front of its own sentence and adds none.)
* **Unpacking phrasal syntax into finite clauses** - low-medium benefit,
  *high* cost, for the same reason in reverse.
* **"I think" / "I feel" / "in my opinion"** - medium benefit, *high* cost.
  These read as unsupported opinion, not argumentative stance. (First person
  for argumentative acts, "I argue", is rated safe, but it requires knowing
  what the writer argues, which a surface transform does not.)
* **Reordering or dropping evidence; trimming commentary to shorten** - medium
  and low benefit, *very high* cost. The paragraph-closer rule below is the one
  deletion we perform, and it fires only when the sentence introduces no
  content word that the paragraph has not already used, which is the strictest
  mechanical proxy we have for "drops no evidence".
* **Any edit inside a quotation, a citation, or the first or last sentence** -
  *very high* cost. Enforced structurally: every transform filters its
  candidates through `Document.protected_spans()`.

Offsets
-------
`Edit.start` and `Edit.end` are character offsets into the `Document.text` that
the transform was handed. The engine runs four passes (see `PASSES`), each
over the text the previous one produced, so offsets from different passes index
different strings and are not comparable. Only `before`/`after` are stable
across passes, which is why they are what the API puts on the wire.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..features.shape import (
    HUMAN_LAG1_AUTOCORR,
    HUMAN_LONG_SHARE,
    HUMAN_SENTENCE_CV,
    HUMAN_SHORT_SHARE,
    LONG_SENTENCE,
    SHORT_SENTENCE,
    lag_autocorrelation,
)
from ..text import Document, words
from .kriukow import (
    ABSOLUTE_ADJECTIVE_HEDGES,
    ABSOLUTE_VERB_HEDGES,
    CONTRASTIVE_RELOCATIONS,
    HEDGE_WORDS,
    HUMAN_HEDGES_PER_1K,
    OPENER_BARE,
    OPENER_PRONOUN,
    opener_class,
)

__all__ = [
    "Edit",
    "HumanizeConfig",
    "CONFIGS",
    "TRANSFORMS",
    "PASSES",
    "CLOSED_VOCABULARY",
    "config_for",
    "sentence_spans",
    "apply_edits",
    "strip_formal_connectives",
    "replace_ai_vocabulary",
    "break_paragraph_template",
    "break_parallelism",
    "vary_sentence_length",
    "hedge_absolutes",
    "relocate_contrastive_openers",
    "front_trailing_clauses",
    "FRONTABLE_SUBORDINATORS",
]


# --------------------------------------------------------------------- model


#: Grade-cost labels, using the vocabulary of the research/00 matrix.
#: "negative" means the edit *improves* the grade, which is the matrix verdict
#: for connective stripping. Nothing above "none" is implemented.
GRADE_COST_NEGATIVE = "negative"
GRADE_COST_NONE = "none"


@dataclass(frozen=True)
class Edit:
    """One atomic, reversible rewrite.

    `before` and `after` are the exact text at `[start, end)` and its
    replacement, so an edit can be inspected, displayed in a diff, or dropped
    by the meaning gate without re-running the transform that produced it.
    """

    kind: str
    start: int
    end: int
    before: str
    after: str
    sentence_index: int
    rationale: str
    grade_cost: str = GRADE_COST_NONE

    @property
    def delta(self) -> int:
        """Character-length change if this edit is applied."""
        return len(self.after) - (self.end - self.start)

    def as_dict(self) -> Dict[str, object]:
        """The wire shape used by the API (offsets deliberately omitted)."""
        return {
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "sentence_index": self.sentence_index,
            "rationale": self.rationale,
            "grade_cost": self.grade_cost,
        }


@dataclass(frozen=True)
class HumanizeConfig:
    """Which transforms run, and how hard they push.

    Three levels, matching the tiers of the research/00 matrix rather than an
    arbitrary slider:

    ``light``     lexicon and connectives only. Word-for-word substitutions and
                  deletions of discourse glue; no sentence is restructured.
    ``balanced``  adds parallelism breaking, paragraph-template breaking and
                  the research/24 method (hedging, opener relocation, clause
                  fronting). Sentence boundaries still never move.
    ``strong``    adds sentence-length restructuring, which splits and merges
                  sentences. The only tier that changes the segmentation.
    """

    name: str
    transforms: Tuple[str, ...]
    #: Upper bound on shape edits, as a share of sentences. Shape edits are the
    #: only ones that move sentence boundaries, so they carry the most risk of
    #: an awkward join; the cap stops a pathological document being rebuilt.
    max_shape_edit_share: float = 0.25
    #: Minimum sentences before the shape transform will act at all.
    #: research/10's CV band was measured on documents, not on five-sentence
    #: fragments, and `eval.report` already warns under 300 words.
    min_sentences_for_shape: int = 8


CONFIGS: Dict[str, HumanizeConfig] = {
    "light": HumanizeConfig(
        name="light",
        transforms=("connective", "vocabulary"),
    ),
    "balanced": HumanizeConfig(
        name="balanced",
        transforms=(
            "paragraph_template",
            "connective",
            "opener",
            "vocabulary",
            "parallelism",
            "hedge",
            "fronting",
        ),
    ),
    "strong": HumanizeConfig(
        name="strong",
        transforms=(
            "paragraph_template",
            "connective",
            "opener",
            "vocabulary",
            "parallelism",
            "hedge",
            "shape",
            "fronting",
        ),
    ),
}

DEFAULT_AGGRESSIVENESS = "balanced"


def config_for(aggressiveness: str) -> HumanizeConfig:
    """Look up a level by name, raising a readable error for a bad one."""
    key = (aggressiveness or DEFAULT_AGGRESSIVENESS).strip().lower()
    if key not in CONFIGS:
        raise ValueError(
            f"Unknown aggressiveness {aggressiveness!r}. "
            f"Valid levels: {sorted(CONFIGS)}."
        )
    return CONFIGS[key]


# ------------------------------------------------------------------- helpers


def sentence_spans(doc: Document) -> List[Tuple[int, int]]:
    """Character spans of each sentence in `doc.text`.

    `Document` stores sentence strings but not offsets, and every transform
    here needs offsets. Sentences are located by a forward scan so that a
    repeated sentence resolves to the right occurrence.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for sent in doc.sentences:
        pos = doc.text.find(sent.text, cursor)
        if pos < 0:  # pragma: no cover - only reachable if parse() changed text
            pos = doc.text.find(sent.text)
            if pos < 0:
                spans.append((cursor, cursor))
                continue
        spans.append((pos, pos + len(sent.text)))
        cursor = pos + len(sent.text)
    return spans


def _overlaps(span: Tuple[int, int], spans: Sequence[Tuple[int, int]]) -> bool:
    lo, hi = span
    return any(lo < b and a < hi for a, b in spans)


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    """Splice edits into `text`. Assumes non-overlapping spans."""
    out = text
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        out = out[: edit.start] + edit.after + out[edit.end :]
    return out


def _match_case(source: str, replacement: str) -> str:
    """Copy the capitalisation of `source` onto `replacement`."""
    if not source or not replacement:
        return replacement
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


# "an" before a vowel *sound*. A leading "u" is almost always consonantal in
# academic prose ("a university", "a useful", "a unified"), and the h-words
# that take "an" are a closed list, so both are special-cased rather than
# guessed from the spelling.
_H_VOWEL_WORDS = frozenset({"hour", "hours", "honest", "honestly", "honour",
                            "honor", "honorary", "heir", "heirs"})


def _takes_an(word: str) -> bool:
    lower = word.lower().strip("\"'“‘(")
    if not lower:
        return False
    if lower in _H_VOWEL_WORDS:
        return True
    if lower[0] == "u":
        return False
    return lower[0] in "aeio"


_WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def _sentence_index_at(offset: int, spans: Sequence[Tuple[int, int]]) -> int:
    """Index of the sentence containing `offset`, or the nearest preceding one."""
    best = 0
    for i, (lo, hi) in enumerate(spans):
        if lo <= offset < hi:
            return i
        if lo <= offset:
            best = i
    return best


def _content_words(text: str) -> FrozenSet[str]:
    """Lowercased content words, for the "introduces nothing new" test."""
    return frozenset(
        w.lower() for w in words(text) if len(w) > 3 and w.lower() not in _STOPWORDS
    )


_STOPWORDS = frozenset(
    """
    the a an and or but if then than that this these those there here of to in
    on at by for with from as is are was were be been being has have had do
    does did will would can could may might must shall should not no nor so
    such very more most much many some any each every both either neither
    which who whom whose what when where why how while about into over under
    between among during before after above below again further once also
    other others another same own too only just its it they them their we our
    us you your he she his her him
    """.split()
)


# ------------------------------------------------- 1. sentence-initial glue


#: The matrix's top-rated edit. These are additive, resultative and summative
#: connectives whose relation is recoverable from the two adjacent claims, so
#: deleting them costs no information.
#:
#: Deliberately absent: *However*, *Nevertheless*, *Conversely*, *In contrast*,
#: *By contrast*. Those mark a genuine reversal, and deleting one inverts the
#: argument. research/10 also measured *however* as *more* frequent in human
#: prose than in AI prose (11.40 vs 8.50 per 10k), so removing it would push
#: the text the wrong way on the one connective where AI under-uses.
STRIPPABLE_CONNECTIVES: Tuple[str, ...] = (
    "in conclusion",
    "in summary",
    "furthermore",
    "moreover",
    "additionally",
    "consequently",
    "therefore",
    "thus",
    "hence",
    "notably",
    "importantly",
    "ultimately",
    "overall",
)

#: These double as ordinary adverbs or adjectives ("Overall performance rose",
#: "Notably absent was ..."), so we only treat them as discourse glue when a
#: comma marks them off. The rest read as connectives with or without one.
_COMMA_REQUIRED = frozenset(
    {"overall", "notably", "importantly", "ultimately", "in conclusion", "in summary"}
)

_CONNECTIVE_RE = re.compile(
    r"^(" + "|".join(sorted((re.escape(c) for c in STRIPPABLE_CONNECTIVES),
                            key=len, reverse=True)) + r")(\s*,\s+|\s+)",
    re.IGNORECASE,
)

#: Below this the sentence is too short to survive losing its opener without
#: reading as a fragment.
_MIN_WORDS_AFTER_STRIP = 5


def strip_formal_connectives(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Delete sentence-initial formal connectives, recapitalising after.

    research/00 section 4 rates this the single best joint move: high detector
    benefit (research/04 feature 9, the formal/informal sentence-initial
    connective ratio) and a *negative* grade cost, because top-band coherence
    is carried by the content of adjacent claims and "attracts no attention",
    whereas signposting every sentence is a mid-band tell.
    """
    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    edits: List[Edit] = []

    for sent, (s0, s1) in zip(doc.sentences, spans):
        match = _CONNECTIVE_RE.match(sent.text)
        if match is None:
            continue
        connective = match.group(1).lower()
        has_comma = "," in match.group(2)
        if connective in _COMMA_REQUIRED and not has_comma:
            continue

        rest = sent.text[match.end() :]
        if not rest or not rest[0].isalpha():
            continue
        if len(words(rest)) < _MIN_WORDS_AFTER_STRIP:
            continue

        # Span covers the connective plus the first letter of what follows, so
        # that deletion and recapitalisation are one atomic edit.
        start, end = s0, s0 + match.end() + 1
        if _overlaps((start, end), protected):
            continue
        edits.append(
            Edit(
                kind="connective",
                start=start,
                end=end,
                before=doc.text[start:end],
                after=rest[0].upper(),
                sentence_index=sent.index,
                rationale=(
                    f"research/00 §4: sentence-initial '{match.group(1)}' is a "
                    "formal-connective tell (research/04 feature 9). Removing "
                    "it is the one edit the matrix rates as *raising* the "
                    "grade, since the claim order already carries the "
                    "coherence."
                ),
                grade_cost=GRADE_COST_NEGATIVE,
            )
        )
    return edits


# ------------------------------------------------------- 2. AI vocabulary


#: Curated one-for-one replacements for the words `features.ai_lexicon` weights
#: highest. The rule for entry is strict: a replacement is listed only when it
#: is a *safe* synonym in academic prose, i.e. it preserves truth conditions in
#: every context we could think of. Where no such synonym exists the word is
#: left alone rather than guessed at - see `_DELIBERATELY_UNMAPPED`.
#:
#: All keys are lowercase. Longest match wins, so phrases are tried first.
PHRASE_REPLACEMENTS: Dict[str, str] = {
    "delve into": "examine",
    "delves into": "examines",
    "delving into": "examining",
    "delved into": "examined",
    "a myriad of": "many",
    "a plethora of": "many",
    "a wide range of": "many",
    "in the realm of": "in",
    "plays a crucial role in": "is central to",
    "plays a pivotal role in": "is central to",
    "plays a vital role in": "is central to",
    "play a crucial role in": "are central to",
    "play a pivotal role in": "are central to",
    "play a vital role in": "are central to",
    "shed light on": "clarify",
    "sheds light on": "clarifies",
    "shedding light on": "clarifying",
    "a testament to": "evidence of",
    "pave the way for": "enable",
    "paves the way for": "enables",
    "paving the way for": "enabling",
    "paved the way for": "enabled",
    "gain a deeper understanding of": "better understand",
    "gaining a deeper understanding of": "better understanding",
    "harness the power of": "use",
    "harnessing the power of": "using",
    "navigating the complexities of": "dealing with",
    "navigate the complexities of": "deal with",
    "rich tapestry of": "range of",
    "comprehensive overview": "overview",
    "when it comes to": "for",
    "in today's fast-paced": "in the fast-moving",
}

WORD_REPLACEMENTS: Dict[str, str] = {
    # De-Latinized verbs. The matrix permits exactly this narrow case and
    # forbids broad simplification.
    "delve": "examine", "delves": "examines", "delving": "examining",
    "delved": "examined",
    "utilize": "use", "utilizes": "uses", "utilizing": "using",
    "utilized": "used", "utilization": "use",
    "utilise": "use", "utilises": "uses", "utilising": "using",
    "utilised": "used", "utilisation": "use",
    "leverage": "use", "leverages": "uses", "leveraging": "using",
    "leveraged": "used",
    "elucidate": "explain", "elucidates": "explains",
    "elucidating": "explaining", "elucidated": "explained",
    "facilitate": "enable", "facilitates": "enables",
    "facilitating": "enabling", "facilitated": "enabled",
    "showcase": "show", "showcases": "shows", "showcasing": "showing",
    "showcased": "showed",
    "underscore": "highlight", "underscores": "highlights",
    "underscoring": "highlighting", "underscored": "highlighted",
    "illuminate": "clarify", "illuminates": "clarifies",
    "illuminating": "clarifying", "illuminated": "clarified",
    "bolster": "strengthen", "bolsters": "strengthens",
    "bolstering": "strengthening", "bolstered": "strengthened",
    "foster": "encourage", "fosters": "encourages",
    "fostering": "encouraging", "fostered": "encouraged",
    "streamline": "simplify", "streamlines": "simplifies",
    "streamlining": "simplifying", "streamlined": "simplified",
    "empower": "enable", "empowers": "enables", "empowering": "enabling",
    "empowered": "enabled",
    "amplify": "increase", "amplifies": "increases",
    "amplifying": "increasing", "amplified": "increased",
    "navigate": "handle", "navigates": "handles", "navigating": "handling",
    "navigated": "handled",
    "spearheaded": "led", "spearheading": "leading",
    "underpin": "support", "underpins": "supports",
    "underpinning": "supporting", "underpinned": "supported",
    "encompass": "include", "encompasses": "includes",
    "encompassing": "including", "encompassed": "included",
    # Adjectives and nouns.
    "pivotal": "central",
    "crucial": "essential",
    "vital": "essential",
    "paramount": "essential",
    "imperative": "essential",
    "integral": "central",
    "myriad": "many",
    "plethora": "many",
    "meticulous": "careful",
    "meticulously": "carefully",
    "robust": "strong",
    "realm": "area", "realms": "areas",
    "intricate": "complex",
    "intricacies": "details",
    "multifaceted": "complex",
    "burgeoning": "growing",
    "garnered": "received",
    "seamless": "smooth", "seamlessly": "smoothly",
    "comprehensive": "thorough",
    "holistic": "overall",
    "innovative": "new",
    "transformative": "far-reaching",
    "groundbreaking": "major",
    "cutting-edge": "advanced",
    "compelling": "strong",
    "cornerstone": "basis",
    "interplay": "interaction",
    "pertinent": "relevant",
    "salient": "key",
    "endeavors": "efforts", "endeavours": "efforts",
    "adept": "skilled",
    "evolving": "changing",
    "ever-evolving": "changing",
}

#: Words the lexicon weights highly that we refuse to substitute, and why. Kept
#: as data so a test can assert we never quietly start replacing them.
_DELIBERATELY_UNMAPPED: Dict[str, str] = {
    "significant": "statistical term of art; 'important' would misstate a result",
    "efficacy": "distinct from 'effectiveness' in clinical writing",
    "paradigm": "technical in philosophy of science and in psychology methods",
    "dynamic": "technical in systems, fluids and psychology",
    "trajectory": "technical in physics, medicine and developmental research",
    "nuanced": "no synonym preserves the meaning; 'subtle' is not the same claim",
    "profound": "'deep' shifts register without preserving the intensifier",
    "landscape": "figurative or literal depending on field",
    "align": "technical in sequence analysis, optics and management",
    "novel": "means 'not previously reported'; no plainer equivalent",
}

#: Context guards. A word is skipped when the following (or preceding) token is
#: listed, because the sense there is not the sense the replacement carries.
_SKIP_IF_NEXT: Dict[str, FrozenSet[str]] = {
    # "robust to misspecification", "robust standard errors" are technical.
    "robust": frozenset(
        {"to", "standard", "standards", "estimator", "estimators",
         "regression", "regressions", "inference", "check", "checks"}
    ),
}
_SKIP_IF_PREV: Dict[str, FrozenSet[str]] = {
    # Noun senses.
    "harness": frozenset({"a", "the", "this", "that"}),
    "utilization": frozenset({"capacity"}),
}

_PHRASE_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b",
                   re.IGNORECASE),
        phrase,
        replacement,
    )
    for phrase, replacement in sorted(
        PHRASE_REPLACEMENTS.items(), key=lambda kv: len(kv[0]), reverse=True
    )
]


def _article_fix(
    text: str, start: int, replacement: str
) -> Tuple[int, str]:
    """Extend an edit backwards over a mismatched a/an, if one precedes it.

    Substituting a word can break article agreement ("an intricate" ->
    "an complex"). Rather than emit a second edit that the overlap resolver
    would have to reconcile, the article is folded into this edit's span.
    """
    prefix = text[max(0, start - 4) : start]
    match = re.search(r"\b(a|an|A|An)(\s+)$", prefix)
    if match is None:
        return start, replacement
    article = match.group(1)
    wanted = "an" if _takes_an(replacement) else "a"
    if article.lower() == wanted:
        return start, replacement
    new_start = start - (len(match.group(0)))
    return new_start, _match_case(article, wanted) + match.group(2) + replacement


def replace_ai_vocabulary(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Swap AI-marked vocabulary for plainer equivalents, one word at a time.

    research/04 ranks AI-vocabulary density the single highest-value feature
    (published excess ratios of 10x to 270x, and the top cue for expert human
    raters). research/06 is honest that GPTZero says this feature does not move
    its probability by itself, so the payoff is mostly the red highlights a
    grader sees - and the matrix rates the edit as zero-to-positive for the
    grade, which is why it stays in the "do" tier.

    This is a curated map, not a simplifier. The matrix rates broad vocabulary
    simplification as *high* grade cost, so a word with no safe synonym is
    skipped (see `_DELIBERATELY_UNMAPPED`).
    """
    text = doc.text
    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    edits: List[Edit] = []
    taken: List[Tuple[int, int]] = []

    def emit(start: int, end: int, replacement: str, surface: str, weightword: str) -> None:
        if _overlaps((start, end), protected) or _overlaps((start, end), taken):
            return
        cased = _match_case(surface, replacement)
        start, cased = _article_fix(text, start, cased)
        if _overlaps((start, end), protected) or _overlaps((start, end), taken):
            return
        before = text[start:end]
        if before == cased:
            return
        taken.append((start, end))
        edits.append(
            Edit(
                kind="vocabulary",
                start=start,
                end=end,
                before=before,
                after=cased,
                sentence_index=_sentence_index_at(start, spans),
                rationale=(
                    f"research/04 feature 1: '{weightword}' is in the weighted "
                    "AI-vocabulary lexicon. Replaced with a curated safe "
                    "synonym; research/00 §4 rates this zero-to-positive for "
                    "the grade, unlike broad simplification."
                ),
                grade_cost=GRADE_COST_NONE,
            )
        )

    for pattern, phrase, replacement in _PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            emit(match.start(), match.end(), replacement, match.group(0), phrase)

    for match in _WORD_TOKEN_RE.finditer(text):
        surface = match.group(0)
        key = surface.lower().rstrip("’'")
        replacement = WORD_REPLACEMENTS.get(key)
        if replacement is None:
            continue
        next_match = _WORD_TOKEN_RE.search(text, match.end())
        next_word = next_match.group(0).lower() if next_match else ""
        if next_word in _SKIP_IF_NEXT.get(key, frozenset()):
            continue
        prev_words = _WORD_TOKEN_RE.findall(text[max(0, match.start() - 30) : match.start()])
        prev_word = prev_words[-1].lower() if prev_words else ""
        if prev_word in _SKIP_IF_PREV.get(key, frozenset()):
            continue
        # Never create an accidental repetition ("use use").
        if prev_word == replacement.lower() or next_word == replacement.lower():
            continue
        emit(match.start(), match.end(), replacement, surface, key)

    return edits


# ------------------------------------------------ 3. the paragraph template


#: Formulaic frames that announce a claim instead of making it. Removing the
#: frame leaves the claim untouched, so no evidence moves. All are matched only
#: at a sentence start.
TEMPLATE_FRAMES: Tuple[str, ...] = (
    "it is important to note that",
    "it's important to note that",
    "it is worth noting that",
    "it's worth noting that",
    "it is important to recognize that",
    "it is important to recognise that",
    "it should be noted that",
    "it must be noted that",
    "it is essential to note that",
    "it is crucial to note that",
    "it is worth mentioning that",
    "as previously mentioned,",
    "as previously noted,",
    "as discussed earlier,",
    "as we have seen,",
    "needless to say,",
    "in essence,",
    "at its core,",
)

_FRAME_RE = re.compile(
    r"^(?:" + "|".join(sorted((re.escape(f) for f in TEMPLATE_FRAMES),
                              key=len, reverse=True)) + r")\s+",
    re.IGNORECASE,
)

#: Markers that open a formulaic paragraph closer or a thesis-restating opener.
_SUMMATIVE_MARKERS: Tuple[str, ...] = (
    "overall", "ultimately", "in conclusion", "in summary", "in short",
    "taken together", "collectively", "all in all", "on the whole",
    "in essence", "to summarize", "to summarise", "thus", "therefore",
    "as such", "in sum",
)

_SUMMATIVE_RE = re.compile(
    r"^(?:" + "|".join(sorted((re.escape(m) for m in _SUMMATIVE_MARKERS),
                              key=len, reverse=True)) + r")\b[,\s]",
    re.IGNORECASE,
)

_HAS_DIGIT_RE = re.compile(r"\d")

#: A paragraph needs at least this many sentences before its last one can be
#: called redundant. In a two-sentence paragraph the closer *is* the content.
_MIN_PARA_SENTENCES = 3


def break_paragraph_template(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Remove formulaic paragraph closers and de-frame announced claims.

    research/04 feature 15 names paragraph-closer detection explicitly
    ("Overall/Ultimately ..., highlighting"); research/06 section 9 item 8 asks
    for paragraph *function* to vary rather than every paragraph running
    topic-sentence -> evidence -> restatement. The matrix rates breaking that
    template as high detector benefit, zero grade cost, with the proviso
    "preserve claim order".

    Two rules, and only two, because deleting text is the one place this
    package can destroy evidence:

    1. **Frame removal.** "It is important to note that X" -> "X". The claim is
       untouched; only the announcement goes.
    2. **Closer removal.** A paragraph-final sentence is deleted only when the
       paragraph has at least three sentences, the sentence opens with a
       summative marker, it contains no digit, citation or quotation, and every
       content word in it already appears earlier in the same paragraph. That
       last condition is the strongest mechanical proxy available for "this
       sentence introduces no evidence"; without an entailment model it is what
       we have, and it is deliberately strict.
    """
    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    edits: List[Edit] = []

    # Rule 1: frames.
    for sent, (s0, _s1) in zip(doc.sentences, spans):
        match = _FRAME_RE.match(sent.text)
        if match is None:
            continue
        rest = sent.text[match.end() :]
        if not rest or not rest[0].isalpha():
            continue
        if len(words(rest)) < _MIN_WORDS_AFTER_STRIP:
            continue
        start, end = s0, s0 + match.end() + 1
        if _overlaps((start, end), protected):
            continue
        edits.append(
            Edit(
                kind="template_frame",
                start=start,
                end=end,
                before=doc.text[start:end],
                after=rest[0].upper(),
                sentence_index=sent.index,
                rationale=(
                    "research/04 feature 10: formulaic hedge frames such as "
                    "'it is important to note' are over-represented in AI text "
                    "(GPTZero reports up to 269x enrichment for this family). "
                    "The frame is removed; the claim it announced is kept "
                    "verbatim."
                ),
                grade_cost=GRADE_COST_NONE,
            )
        )

    # Rule 2: redundant closers.
    by_para: Dict[int, List[int]] = {}
    for sent in doc.sentences:
        by_para.setdefault(sent.paragraph_index, []).append(sent.index)

    for para_idx, indices in by_para.items():
        if len(indices) < _MIN_PARA_SENTENCES:
            continue
        last = indices[-1]
        if last in (0, len(doc.sentences) - 1):
            continue  # first/last sentence of the document: protected
        sent = doc.sentences[last]
        marker = _SUMMATIVE_RE.match(sent.text)
        if marker is None:
            continue
        if _HAS_DIGIT_RE.search(sent.text) or '"' in sent.text or "“" in sent.text:
            continue
        earlier = " ".join(doc.sentences[i].text for i in indices[:-1])
        # The marker itself ("Overall", "Collectively") is discourse glue, not
        # content, and it is exactly the word the paragraph will not have used
        # before - so testing it would veto every closer.
        body = sent.text[marker.end() :]
        new_words = _content_words(body) - _content_words(earlier)
        if new_words:
            continue
        start = spans[last - 1][1]
        end = spans[last][1]
        if _overlaps((start, end), protected):
            continue
        edits.append(
            Edit(
                kind="paragraph_closer",
                start=start,
                end=end,
                before=doc.text[start:end],
                after="",
                sentence_index=sent.index,
                rationale=(
                    "research/04 feature 15 / research/06 §9.8: a summative "
                    "paragraph closer that repeats the paragraph's own content "
                    "words is the AI paragraph template. It introduces no "
                    "content word the paragraph has not already used, so "
                    "removing it drops no evidence "
                    f"(paragraph {para_idx + 1})."
                ),
                grade_cost=GRADE_COST_NONE,
            )
        )

    return edits


# ------------------------------------------------------- 4. parallelism


#: Inverted "Not only *does* X ..." requires subject-auxiliary inversion, and
#: flattening it to "X and ..." leaves an ungrammatical clause. Skip those.
_INVERSION_TRIGGERS = frozenset(
    {"does", "do", "did", "is", "are", "was", "were", "has", "have", "had",
     "can", "could", "will", "would", "may", "might", "should", "must"}
)

_NEG_PARALLEL_RE = re.compile(
    r"\bnot\s+(?:only|just|merely|simply)\s+"
    r"(?P<a>[^.;:!?]{1,90}?)"
    r"\s*,?\s*but\s+(?:also\s+)?",
    re.IGNORECASE,
)

#: The list must end the clause. Measured failure, kept as a comment because it
#: is the reason this rule is narrow: rewriting a list that modifies a head that
#: follows it produces garbage. "meaningful, sustainable, and equitable reform"
#: became "meaningful and sustainable, as well as equitable reform", which
#: attaches the wrong way, and "how technology, pedagogy, and policy interact"
#: became "how technology and pedagogy, as well as policy interact", which
#: strands the verb. Requiring clause-final position removes both.
_TRICOLON_RE = re.compile(
    r"(?P<a>\b[A-Za-z][\w'-]*),\s+(?P<b>[A-Za-z][\w'-]*),\s+and\s+(?P<c>[A-Za-z][\w'-]*)\b"
    r"(?=\s*(?:[.;:!?,)\]]|$))"
)


def break_parallelism(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Flatten "not only X but also Y" and three-item parallel lists.

    research/04 features 6 and 7 measure tricolon density at about 2x human and
    the negative-parallelism family at about 3x. research/06 section 9 item 7
    asks for "unequal lists or single points"; the matrix rates the edit as
    medium-high benefit at zero grade cost.

    Both rewrites are conservative and keep every item:

    * "not only A but also B" -> "A and B". This drops rhetorical emphasis, not
      content: the truth conditions are identical.
    * "A, B, and C" -> "A and B, as well as C", which turns a balanced triple
      into an unequal 2 + 1 list. Two guards, both of them the result of bad
      output during evaluation rather than caution in advance: every item must
      be a single word (so "as well as" coordinates a bare constituent), and the
      list must be clause-final (so it cannot be a stack of premodifiers on a
      head noun that follows). Coverage is low by design and measured to be low;
      the alternative rewrites all drop or reorder an item, which the matrix
      rates as *very high* grade cost.
    """
    text = doc.text
    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    edits: List[Edit] = []
    taken: List[Tuple[int, int]] = []
    sentence_starts = {s0 for s0, _ in spans}

    for match in _NEG_PARALLEL_RE.finditer(text):
        first = match.group("a").strip()
        if not first:
            continue
        head = _WORD_TOKEN_RE.match(first)
        if head is not None and head.group(0).lower() in _INVERSION_TRIGGERS:
            continue
        if re.search(r"\bbut\b", first, re.IGNORECASE):
            continue
        span = (match.start(), match.end())
        if _overlaps(span, protected) or _overlaps(span, taken):
            continue
        after = first + " and "
        if match.start() in sentence_starts:
            after = after[0].upper() + after[1:]
        taken.append(span)
        edits.append(
            Edit(
                kind="negative_parallel",
                start=span[0],
                end=span[1],
                before=text[span[0] : span[1]],
                after=after,
                sentence_index=_sentence_index_at(span[0], spans),
                rationale=(
                    "research/04 feature 7: the 'not only X but also Y' family "
                    "runs about 3x the human rate. Flattening to 'X and Y' "
                    "keeps both terms and the truth conditions, and drops only "
                    "the rhetorical emphasis."
                ),
                grade_cost=GRADE_COST_NONE,
            )
        )

    for match in _TRICOLON_RE.finditer(text):
        a, b, c = match.group("a"), match.group("b"), match.group("c")
        span = (match.start(), match.end())
        if _overlaps(span, protected) or _overlaps(span, taken):
            continue
        # A run of capitalised single words mid-sentence is an author or place
        # list, not a rhetorical tricolon. Reordering names would be a factual
        # edit, so leave it alone.
        if (
            match.start() not in sentence_starts
            and a[0].isupper()
            and b[0].isupper()
            and c[0].isupper()
        ):
            continue
        taken.append(span)
        edits.append(
            Edit(
                kind="tricolon",
                start=span[0],
                end=span[1],
                before=text[span[0] : span[1]],
                after=f"{a} and {b}, as well as {c}",
                sentence_index=_sentence_index_at(span[0], spans),
                rationale=(
                    "research/04 feature 6: balanced three-item lists run about "
                    "2x the human rate. Rebalanced to an unequal 2 + 1 list "
                    "(research/06 §9.7); all three items are kept."
                ),
                grade_cost=GRADE_COST_NONE,
            )
        )

    return edits


# ------------------------------------------------------- 5. sentence shape


#: Words that can safely open a clause promoted to its own sentence, or be
#: lowercased when a sentence is merged into the one before it. Restricted to
#: determiners, pronouns and quantifiers so that no proper noun is ever
#: recapitalised, which would trip the meaning gate and, worse, be wrong.
_SAFE_CLAUSE_HEADS = frozenset(
    {"this", "these", "that", "those", "it", "they", "we", "the", "a", "an",
     "such", "its", "their", "our", "his", "her", "both", "each", "many",
     "most", "some", "few", "several", "other", "others", "all", "one",
     "there", "here"}
)

#: A promoted clause must contain a finite verb or the result is a fragment.
#: This is a closed list rather than a suffix rule, because "-s" and "-ed"
#: endings are ambiguous between plural nouns and finite verbs without a POS
#: tagger, and a false positive here produces ungrammatical output.
_FINITE_VERBS = frozenset(
    """
    is are was were be being been has have had do does did can could may might
    will would shall should must remains remain remained appears appear
    appeared seems seem seemed becomes become became provides provide provided
    shows show showed shown suggests suggest suggested indicates indicate
    indicated requires require required includes include included represents
    represent represented allows allow allowed enables enable enabled offers
    offer offered reflects reflect reflected supports support supported
    explains explain explained produces produce produced yields yield yielded
    contains contain contained involves involve involved depends depend
    depended varies vary varied differs differ differed exists exist existed
    occurs occur occurred tends tend tended makes make made gives give gave
    takes take took holds hold held leads lead led means mean meant
    """.split()
)

_SPLIT_CONJUNCTIONS: Tuple[str, ...] = (", and ", ", but ", ", yet ", "; ")

#: Only sentences at least this long are candidates for splitting, and each
#: half must reach `_MIN_SPLIT_HALF`. research/10 puts the human short tail at
#: under 10 words, so a 6-8 word second half is the target, not an accident.
_MIN_SPLIT_SOURCE = 24
_MIN_SPLIT_HALF = 5
#: Merge candidates: two adjacent shortish sentences. The ceiling is 42 words,
#: not the ~30 a first pass used, because research/10 wants 18-28% of sentences
#: *over* 30 words and two 15-word sentences can only ever make a 31-word one.
#: With the tighter cap the measured long-tail share stayed at 0% on every
#: sample; the band needs merges that clear 30 by a real margin.
_MAX_MERGE_PART = 22
_MAX_MERGE_TOTAL = 42


def _band_distance(value: float, band: Tuple[float, float]) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    lo, hi = band
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


def _cv(lengths: Sequence[float]) -> float:
    if len(lengths) < 2:
        return float("nan")
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return float("nan")
    var = sum((x - mean) ** 2 for x in lengths) / (len(lengths) - 1)
    return math.sqrt(var) / mean


def _shape_cost(lengths: Sequence[float]) -> float:
    """Distance from the research/10 human shape bands. Lower is better.

    CV is weighted 3x because it is the feature the detector scores and the one
    research/10 measured most tightly (0.42-0.60). The tail shares are weighted
    at 1 because they are the *mechanism* by which CV should move: research/10
    found humans widen the tails rather than jitter the middle. The lag-1 term
    is a one-sided penalty that only fires below -0.10, which is the signature
    of an alternating long/short humanizer and is itself an anomaly.
    """
    n = len(lengths)
    if n < 2:
        return 0.0
    cv = _cv(lengths)
    short_share = sum(1 for x in lengths if x < SHORT_SENTENCE) / n
    long_share = sum(1 for x in lengths if x > LONG_SENTENCE) / n
    cost = 3.0 * _band_distance(cv, HUMAN_SENTENCE_CV)
    cost += _band_distance(short_share, HUMAN_SHORT_SHARE)
    cost += _band_distance(long_share, HUMAN_LONG_SHARE)
    lag1 = lag_autocorrelation(lengths, 1)
    if math.isfinite(lag1):
        cost += 2.0 * max(0.0, HUMAN_LAG1_AUTOCORR[0] - lag1)
    return cost


@dataclass
class _ShapeCandidate:
    """A pending split or merge, with the sentence lengths it would produce."""

    kind: str  # "split" | "merge"
    index: int  # first sentence involved
    parts: Tuple[int, ...]
    edit: Edit


def _simulate(lengths: Sequence[int], chosen: Dict[int, _ShapeCandidate]) -> List[int]:
    """Sentence lengths after applying the chosen, pairwise-disjoint edits."""
    out: List[int] = []
    i = 0
    n = len(lengths)
    while i < n:
        cand = chosen.get(i)
        if cand is None:
            out.append(lengths[i])
            i += 1
        elif cand.kind == "split":
            out.extend(cand.parts)
            i += 1
        else:
            out.append(sum(cand.parts))
            i += 2
    return out


def _split_candidates(
    doc: Document, spans: Sequence[Tuple[int, int]], protected: Sequence[Tuple[int, int]]
) -> List[_ShapeCandidate]:
    out: List[_ShapeCandidate] = []
    for sent, (s0, s1) in zip(doc.sentences, spans):
        if sent.length < _MIN_SPLIT_SOURCE:
            continue
        if _overlaps((s0, s1), protected):
            continue
        best: Optional[_ShapeCandidate] = None
        for conj in _SPLIT_CONJUNCTIONS:
            pos = sent.text.find(conj)
            while pos != -1:
                head = sent.text[:pos]
                tail = sent.text[pos + len(conj) :]
                head_words, tail_words = words(head), words(tail)
                pos = sent.text.find(conj, pos + 1)
                if len(head_words) < _MIN_SPLIT_HALF or len(tail_words) < _MIN_SPLIT_HALF:
                    continue
                tail_lower = [w.lower() for w in tail_words]
                if tail_lower[0] not in _SAFE_CLAUSE_HEADS:
                    continue
                if not any(w in _FINITE_VERBS for w in tail_lower):
                    continue
                start = s0 + len(head)
                end = start + len(conj) + 1
                if _overlaps((start, end), protected):
                    continue
                replacement = ". " + tail[0].upper()
                cand = _ShapeCandidate(
                    kind="split",
                    index=sent.index,
                    parts=(len(head_words), len(tail_words)),
                    edit=Edit(
                        kind="sentence_split",
                        start=start,
                        end=end,
                        before=doc.text[start:end],
                        after=replacement,
                        sentence_index=sent.index,
                        rationale=(
                            "research/10: the human sentence-length CV band is "
                            "0.42-0.60 and the tails carry it (9-14% under 10 "
                            "words, 18-28% over 30). This document sits below "
                            f"the band, so a {sent.length}-word sentence is "
                            "split at a coordinating conjunction into "
                            f"{len(head_words)} + {len(tail_words)} words to "
                            "widen the short tail."
                        ),
                        grade_cost=GRADE_COST_NONE,
                    ),
                )
                # Prefer the split that leaves the shortest well-formed tail:
                # research/10 says widen the tails, do not jitter the middle.
                if best is None or min(cand.parts) < min(best.parts):
                    best = cand
        if best is not None:
            out.append(best)
    return out


def _merge_candidates(
    doc: Document, spans: Sequence[Tuple[int, int]], protected: Sequence[Tuple[int, int]]
) -> List[_ShapeCandidate]:
    out: List[_ShapeCandidate] = []
    for i in range(len(doc.sentences) - 1):
        first, second = doc.sentences[i], doc.sentences[i + 1]
        if first.paragraph_index != second.paragraph_index:
            continue
        if first.length > _MAX_MERGE_PART or second.length > _MAX_MERGE_PART:
            continue
        if first.length + second.length > _MAX_MERGE_TOTAL:
            continue
        if not first.text.endswith("."):
            continue
        if not second.words or second.words[0].lower() not in _SAFE_CLAUSE_HEADS:
            continue
        start = spans[i][1] - 1  # the full stop
        end = spans[i + 1][0] + 1  # through the first letter of the next
        if end <= start or _overlaps((start, end), protected):
            continue
        joiner = "; " if ", and " in first.text else ", and "
        out.append(
            _ShapeCandidate(
                kind="merge",
                index=i,
                parts=(first.length, second.length),
                edit=Edit(
                    kind="sentence_merge",
                    start=start,
                    end=end,
                    before=doc.text[start:end],
                    after=joiner + second.words[0][0].lower(),
                    sentence_index=first.index,
                    rationale=(
                        "research/10: sentence-length CV is below the human "
                        "0.42-0.60 band. Two adjacent short sentences "
                        f"({first.length} + {second.length} words) are merged "
                        "so the long tail widens; the short tail is left to "
                        "the split rule."
                    ),
                    grade_cost=GRADE_COST_NONE,
                ),
            )
        )
    return out


def vary_sentence_length(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Move sentence-length CV into the human band by widening the tails.

    research/10 is the reason this is not a jitter. It measured the human CV
    band at 0.42-0.60 (not "as high as possible": overshooting to 0.85 is as
    anomalous as undershooting to 0.30), found lag-1 autocorrelation of only
    0.01-0.10 (so humans do *not* alternate long and short), and located the
    signal in the tails: about 9-14% of academic sentences run under 10 words
    and 18-28% over 30.

    So the algorithm is greedy and measured, not stochastic:

    1. If CV is already inside the band, return no edits at all.
    2. If CV is *above* the band, also return no edits. Pulling CV down means
       padding or merging text that is already varied, and research/10 gives no
       safe recipe for it; a no-op is honest.
    3. Otherwise, enumerate splits (a long sentence at a coordinating
       conjunction, choosing the point that leaves the shortest well-formed
       tail) and merges (two adjacent short sentences). Score every candidate
       by `_shape_cost`, apply the single best one, re-measure, and repeat.
    4. Stop as soon as CV reaches the bottom of the band. Never accept an edit
       that would push CV above 0.60, and never accept one that drives lag-1
       autocorrelation below -0.10, which is what alternating long and short
       looks like.
    """
    lengths = list(doc.sentence_lengths)
    if len(lengths) < config.min_sentences_for_shape:
        return []
    cv = _cv(lengths)
    lo, hi = HUMAN_SENTENCE_CV
    if not math.isfinite(cv) or cv >= lo:
        return []

    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    candidates = _split_candidates(doc, spans, protected) + _merge_candidates(
        doc, spans, protected
    )
    if not candidates:
        return []

    chosen: Dict[int, _ShapeCandidate] = {}
    used: set = set()
    max_edits = max(1, int(round(config.max_shape_edit_share * len(lengths))))
    current_cost = _shape_cost(lengths)

    while len(chosen) < max_edits:
        best: Optional[Tuple[float, _ShapeCandidate]] = None
        for cand in candidates:
            touched = {cand.index} if cand.kind == "split" else {cand.index, cand.index + 1}
            if touched & used:
                continue
            trial = dict(chosen)
            trial[cand.index] = cand
            new_lengths = _simulate(lengths, trial)
            new_cv = _cv(new_lengths)
            if not math.isfinite(new_cv) or new_cv > hi:
                continue
            lag1 = lag_autocorrelation(new_lengths, 1)
            if math.isfinite(lag1) and lag1 < HUMAN_LAG1_AUTOCORR[0]:
                continue
            cost = _shape_cost(new_lengths)
            if cost >= current_cost:
                continue
            if best is None or cost < best[0]:
                best = (cost, cand)
        if best is None:
            break
        current_cost, cand = best
        chosen[cand.index] = cand
        used.update({cand.index} if cand.kind == "split" else {cand.index, cand.index + 1})
        if _cv(_simulate(lengths, chosen)) >= lo:
            break

    return [c.edit for c in chosen.values()]


# ------------------------------------------------------------------ registry


Transform = Callable[[Document, HumanizeConfig], List[Edit]]

#: Every transform by name, for direct use and for tests.
# ------------------------------------------- 6. stance and openers (research/24)


#: Modals and hedges that, within a few tokens before a claim verb, already
#: soften it. A second hedge would stack ("may can play"), which research/22
#: lists under "hedge stacking: no human/LLM gap, do not add".
_PRE_HEDGE_WINDOW = 3
_MODALS = frozenset(
    {"can", "could", "may", "might", "would", "should", "will", "shall", "must",
     "to", "not", "never", "often", "usually", "always"}
)
_ADJECTIVE_HEDGE_RE = re.compile(
    r"\b(?P<cop>is|are)\s+(?P<adj>" + "|".join(ABSOLUTE_ADJECTIVE_HEDGES) + r")\b",
    re.IGNORECASE,
)
_HAS_DIGIT_OR_QUOTE_RE = re.compile(r"\d|[\"“”]")

#: The closed set of tokens every transform in this module is allowed to
#: introduce. `tests.test_humanize.test_engine_never_injects_surface_errors`
#: asserts that no output token falls outside the source plus this set, which
#: is the mechanical form of the refusal to inject anything unplanned.
def _base_form(verb: str) -> str:
    """Third-person -s form to bare infinitive, for the closed set below."""
    if verb.endswith("es") and not verb.endswith(("ses", "zes", "ches", "shes")):
        return verb[:-2]
    return verb[:-1] if verb.endswith("s") else verb


CLOSED_VOCABULARY: FrozenSet[str] = frozenset(
    {"and", "as", "well", "a", "an"}
    | {w for v in ABSOLUTE_VERB_HEDGES.values() for w in v.lower().split()}
    | {"appears", "appear"}
    | {w for v in CONTRASTIVE_RELOCATIONS.values() for w in v.lower().split()}
    # Hedged forms of lexicon-mapped verbs ("can encourage").
    | {_base_form(WORD_REPLACEMENTS[k].lower()) for k in ABSOLUTE_VERB_HEDGES if k in WORD_REPLACEMENTS}
)


def hedge_absolutes(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Turn unhedged evaluative claims into hedged ones.

    research/24 §1.2, the presenter's first principle ("intellectual
    hesitation"): AI "writes in absolutes", scholars "hedge, suggest, suspect";
    "plays a critical role" becomes "can play a critical role". research/18 §5
    measures hedges as genuinely depleted in instruct-model text (GPT-4 at 0.00
    per sentence) and research/15 gives Hyland's band of 8-20 per 1,000 words,
    which is the ceiling this transform will not push past.

    Conservative by construction:

    * only third-person -s claim verbs from `ABSOLUTE_VERB_HEDGES` and the
      "is/are + evaluative adjective" frame are touched, so no infinitive,
      past tense or reportive verb changes;
    * a sentence that already contains a hedge or modal, a digit or a
      quotation is skipped (numbers and quoted claims are facts, not claims);
    * at most one hedge per sentence, at most one sentence in three, and never
      once the document's hedge density reaches the top of Hyland's band;
    * protected spans (quotations, citations, first and last sentence) are
      never edited.

    Detector evidence: none at the verdict level. research/19 §4 measured
    hedge *insertion* as noise at 6 of 30 flips on boundary text and 0 of 40 on
    saturated text. This is a quality-realism edit.
    """
    text = doc.text
    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    n_sent = len(doc.sentences)
    if n_sent == 0:
        return []
    n_words = max(1, doc.n_words)
    existing = sum(1 for w in doc.lower_words if w in HEDGE_WORDS)
    ceiling = HUMAN_HEDGES_PER_1K[1] * n_words / 1000.0
    budget = max(1, math.ceil(n_sent / 3))
    edits: List[Edit] = []

    for sent, (s0, s1) in zip(doc.sentences, spans):
        if len(edits) >= budget or existing + len(edits) >= ceiling:
            break
        if _HAS_DIGIT_OR_QUOTE_RE.search(sent.text):
            continue
        toks = [w.lower() for w in sent.words]
        if any(t in HEDGE_WORDS for t in toks):
            continue
        made = False
        for match in _WORD_TOKEN_RE.finditer(sent.text):
            word = match.group(0)
            low = word.lower()
            replacement = ABSOLUTE_VERB_HEDGES.get(low)
            if replacement is None:
                continue
            # No stacking: skip if a modal or hedge sits just before the verb.
            before_toks = [w.lower() for w in words(sent.text[: match.start()])]
            if any(t in _MODALS or t in HEDGE_WORDS for t in before_toks[-_PRE_HEDGE_WINDOW:]):
                continue
            # The verb must have a subject before it: not sentence-initial.
            if not before_toks:
                continue
            start, end = s0 + match.start(), s0 + match.end()
            if _overlaps((start, end), protected):
                continue
            # If the verb is also on the AI-vocabulary map ("fosters" ->
            # "encourages"), hedge the *mapped* verb so one edit does both
            # jobs: "fosters" -> "can encourage". The hedge pass runs before
            # the lexicon pass and wins the overlap, so without this the
            # lexicon edit would be lost.
            mapped = WORD_REPLACEMENTS.get(low)
            if mapped is not None and " " in replacement:
                modal, _, _verb = replacement.partition(" ")
                replacement = f"{modal} {_base_form(mapped.lower())}"
            edits.append(
                Edit(
                    kind="hedge",
                    start=start,
                    end=end,
                    before=word,
                    after=_match_case(word, replacement),
                    sentence_index=sent.index,
                    rationale=(
                        f"research/24 §1.2 (intellectual hesitation): '{word}' "
                        "states a causal or evaluative claim as fact; "
                        f"'{replacement}' hedges it. Hedges are depleted in "
                        "instruct-model prose (research/18 §5) and are the "
                        "academic norm (Hyland, research/15). No pass-rate claim."
                    ),
                    grade_cost=GRADE_COST_NONE,
                )
            )
            made = True
            break
        if made:
            continue
        match = _ADJECTIVE_HEDGE_RE.search(sent.text)
        if match is None:
            continue
        before_toks = [w.lower() for w in words(sent.text[: match.start()])]
        if not before_toks or any(
            t in _MODALS or t in HEDGE_WORDS for t in before_toks[-_PRE_HEDGE_WINDOW:]
        ):
            continue
        start, end = s0 + match.start("cop"), s0 + match.end("cop")
        if _overlaps((start, end), protected):
            continue
        cop = match.group("cop")
        replacement = "appears" if cop.lower() == "is" else "appear"
        edits.append(
            Edit(
                kind="hedge",
                start=start,
                end=end,
                before=cop,
                after=_match_case(cop, replacement),
                sentence_index=sent.index,
                rationale=(
                    f"research/24 §1.2: '{cop} {match.group('adj')}' asserts an "
                    f"evaluation as fact; '{replacement} {match.group('adj')}' "
                    "hedges it while keeping the copula's agreement. No pass-rate claim."
                ),
                grade_cost=GRADE_COST_NONE,
            )
        )
    return edits


_CONTRASTIVE_RE = re.compile(
    r"^(" + "|".join(sorted((re.escape(c) for c in CONTRASTIVE_RELOCATIONS),
                            key=len, reverse=True)) + r")\s*,\s+",
    re.IGNORECASE,
)
_MAX_SUBJECT_WORDS = 8
_MIN_WORDS_FOR_RELOCATION = 8


def relocate_contrastive_openers(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Move a sentence-initial contrastive connective to after the subject.

    "Conversely, low self-esteem may hinder communication" becomes "Low
    self-esteem, on the other hand, may hinder communication". This is the
    presenter's "reverse the order" move (research/24 §1.2 principle 3, §1.3
    sentence 3), and it addresses the opener-class monotony research/21 §10.1
    checks 6-9 measure: skilled prose opens 1.3 sentences per 1,000 words on a
    connective, instruct-model prose about 1 per 200 words.

    Unlike `strip_formal_connectives`, nothing is deleted: However, Conversely,
    In contrast, Nevertheless and Similarly carry a logical relation the
    argument needs (research/23 §7 warns against removing connectives that a
    rubric scorer reads), so they are moved, not cut.

    Guards: the subject must be one to eight words with no comma and must be
    followed by a finite verb or modal from the closed `_FINITE_VERBS` list
    (so the insertion point is certain to be the subject/verb boundary); the
    sentence must have at least eight words; protected spans are never edited.

    Detector evidence: research/19 §4 measured connective *removal* at 2 of 30
    flips; relocation is unmeasured. Quality and de-templating edit.
    """
    text = doc.text
    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    edits: List[Edit] = []

    for sent, (s0, s1) in zip(doc.sentences, spans):
        match = _CONTRASTIVE_RE.match(sent.text)
        if match is None:
            continue
        if len(sent.words) < _MIN_WORDS_FOR_RELOCATION:
            continue
        rest = sent.text[match.end() :]
        # Find the subject/verb boundary: the first finite verb or modal.
        verb_pos: Optional[int] = None
        subject_words = 0
        for tok in _WORD_TOKEN_RE.finditer(rest):
            low = tok.group(0).lower()
            if subject_words >= 1 and low in _FINITE_VERBS:
                verb_pos = tok.start()
                break
            subject_words += 1
            if subject_words > _MAX_SUBJECT_WORDS:
                break
        if verb_pos is None or subject_words == 0:
            continue
        subject = rest[:verb_pos].rstrip()
        if "," in subject or not subject or not subject[0].isalpha():
            continue
        start, end = s0, s0 + match.end() + verb_pos
        if _overlaps((start, end), protected):
            continue
        relocated = CONTRASTIVE_RELOCATIONS[match.group(1).lower()]
        after = subject[0].upper() + subject[1:] + ", " + relocated + ", "
        edits.append(
            Edit(
                kind="opener",
                start=start,
                end=end,
                before=text[start:end],
                after=after,
                sentence_index=sent.index,
                rationale=(
                    f"research/24 §1.2 (vary sentence openings): '{match.group(1)},' "
                    "opens the sentence on a connective, the opener class AI "
                    "over-uses (research/21 §10.1 check 6). Moved after the "
                    f"subject as ', {relocated},'. Nothing deleted. No pass-rate claim."
                ),
                grade_cost=GRADE_COST_NONE,
            )
        )
    return edits




# ------------------------------------ 7. fronting trailing clauses (research/24)


#: Subordinators whose trailing clause can move to the front of its sentence
#: without changing what the sentence asserts ("X rose because Y" and "Because
#: Y, X rose" make the same claim). "as" is absent because it is ambiguous
#: (causal, temporal, comparative, "as well as"); "so that", "whether" and
#: "that" are absent because they introduce complements, not adjuncts.
FRONTABLE_SUBORDINATORS: Tuple[str, ...] = (
    "even though", "even if", "although", "though", "because", "whereas",
    "while", "whenever", "when", "unless", "until", "after", "before",
    "once", "since", "if",
)

#: Greedy `pre` so the match lands on the *last* subordinator, which is the
#: one that heads the final clause. The clause may not contain a comma,
#: semicolon, colon, bracket or quotation mark: a trailing clause with
#: internal structure is left alone rather than risk fronting half of it.
_FRONTABLE_RE = re.compile(
    r"^(?P<pre>\S.*[^\s,])(?P<sep>,?\s+)(?P<sub>"
    + "|".join(re.escape(s) for s in sorted(FRONTABLE_SUBORDINATORS, key=len, reverse=True))
    + r")\s+(?P<clause>[^,;:()\[\]\"“”]+?)(?P<end>[.!?]+)$",
    re.IGNORECASE | re.DOTALL,
)

#: If the word before the subordinator is one of these, the "clause" is a
#: complement or a relative ("is unclear if ...", "the year when ..."), and
#: fronting it would change the meaning or leave a fragment.
_COMPLEMENT_HEADS = frozenset(
    """
    is are was were be been being am unclear uncertain unknown clear obvious
    know knows knew known ask asks asked wonder wonders wondered determine
    determines determined see sees saw seen decide decides decided question
    questions doubt doubts doubted matter matters mattered whether
    time times period periods era eras day days year years decade decades
    century centuries moment moments point points stage stages phase phases
    age ages a an the this that these those
    """.split()
)

#: A clause that opens on a pronoun refers back to the main clause; fronting
#: it makes the pronoun point forward ("Because it arrived late, the funding
#: ..."), which reads as a slip. Such clauses are left where they are.
_CATAPHORA_HEADS = frozenset(
    {"it", "its", "they", "their", "them", "this", "these", "those", "that",
     "he", "she", "his", "her", "we", "our", "such", "there", "so"}
)

_MIN_FRONT_CLAUSE = 3
_MAX_FRONT_CLAUSE = 18
_MIN_FRONT_MAIN = 5


def _same_opener_run(
    i: int, classes: Sequence[str], firsts: Sequence[str], lengths: Sequence[int]
) -> bool:
    """Whether sentence `i` repeats the opener pattern of its neighbours.

    The presenter's rule (research/24 §1.3): a sentence may keep its shape "as
    long as the following sentence does not follow the exact same pattern".
    Repetition here is either the same first word as the sentence before, or
    the same opener class across three consecutive sentences, or the same
    class as the sentence before at the same "beat" (within three words).
    Only bare-subject and pronoun openers count: those are the two classes
    research/21 §10.1 checks 8-9 measure as over-represented.
    """
    if i == 0 or classes[i] not in (OPENER_BARE, OPENER_PRONOUN):
        return False
    if firsts[i] and firsts[i] == firsts[i - 1]:
        return True
    if classes[i - 1] != classes[i]:
        return False
    if i + 1 < len(classes) and classes[i + 1] == classes[i]:
        return True
    if i >= 2 and classes[i - 2] == classes[i]:
        return True
    return abs(lengths[i] - lengths[i - 1]) <= 3


def front_trailing_clauses(doc: Document, config: HumanizeConfig) -> List[Edit]:
    """Move a sentence's trailing subordinate clause to its front, where the
    sentence repeats the opener pattern of the one before it.

    research/24 §1.2, the presenter's third principle and the one he calls
    "very important": AI text "will often start sentences in the exact same
    way" and the fix is "introductory clauses, dependent clauses, inverted
    structures rather than just repeating that same exact structure". His own
    example is a fronted clause ("Although widely cited, this study ...") and
    in the demonstration he fronts a causal clause ("As low self-assessed
    English skills are believed to ..., it is crucial ...").

    This transform performs exactly that move and nothing more: a clause the
    writer already wrote, headed by one of `FRONTABLE_SUBORDINATORS`, moves
    from the end of its sentence to the beginning. No word is added or
    removed, so the closed-vocabulary guarantee holds trivially and the
    module docstring's refusal of *added* subordination is untouched.

    Fires only on a sentence that repeats the opener pattern of its
    neighbours (`_same_opener_run`), because the aim is to break a run, not
    to front every clause: research/21 §10.1 puts human bare-subject openers
    at 45-60%, so most sentences should keep the subject first.

    Guards: the clause is 3-18 words with no internal punctuation; the main
    clause has at least five words and no semicolon, colon or dash; the word
    before the subordinator is not a complement or time-noun head
    (`_COMPLEMENT_HEADS`); the clause does not open on a pronoun
    (`_CATAPHORA_HEADS`); the main clause's first word can be lowercased
    safely (a determiner, pronoun or quantifier, or a word the document also
    uses in lowercase); protected spans are never touched; at most one
    sentence in four per document and never two in a row.

    Detector evidence: none at the verdict level (research/24 §2, research/19
    §4). Quality and de-templating edit; no pass-rate claim.
    """
    sents = doc.sentences
    if len(sents) < 3:
        return []
    text = doc.text
    protected = doc.protected_spans()
    spans = sentence_spans(doc)
    lower_elsewhere = {w for w in doc.words if w[:1].islower()}

    budget = max(1, len(sents) // 4)
    edits: List[Edit] = []
    last_fronted = -2

    # Opener runs are a within-paragraph notion.
    by_para: Dict[int, List[int]] = {}
    for idx, sent in enumerate(sents):
        by_para.setdefault(sent.paragraph_index, []).append(idx)

    for indices in by_para.values():
        classes = [opener_class(sents[i].text) for i in indices]
        firsts = []
        for i in indices:
            m = _WORD_TOKEN_RE.search(sents[i].text)
            firsts.append(m.group(0).lower() if m else "")
        lengths = [sents[i].length for i in indices]
        for local_i, idx in enumerate(indices):
            if len(edits) >= budget:
                return edits
            if idx - last_fronted < 2:
                continue
            if not _same_opener_run(local_i, classes, firsts, lengths):
                continue
            sent = sents[idx]
            match = _FRONTABLE_RE.match(sent.text.strip())
            if match is None:
                continue
            pre, sub, clause, end = (
                match.group("pre"), match.group("sub").lower(),
                match.group("clause").strip(), match.group("end"),
            )
            if any(ch in pre for ch in ";:—–") or " -- " in pre or " - " in pre:
                continue
            pre_words = words(pre)
            clause_words = words(clause)
            if len(pre_words) < _MIN_FRONT_MAIN:
                continue
            if not _MIN_FRONT_CLAUSE <= len(clause_words) <= _MAX_FRONT_CLAUSE:
                continue
            if pre_words[-1].lower() in _COMPLEMENT_HEADS:
                continue
            if clause_words[0].lower() in _CATAPHORA_HEADS:
                continue
            # "because of", "since then": a prepositional tail is fine to
            # front, but "if so" / "if any" are elliptical and are not.
            if sub == "if" and clause_words[0].lower() in {"so", "any", "not", "necessary", "possible"}:
                continue
            first = pre_words[0]
            if not pre[0].isalpha() or (first.isupper() and len(first) > 1):
                continue
            low_first = first.lower()
            if low_first not in _SAFE_CLAUSE_HEADS and first not in lower_elsewhere and low_first not in lower_elsewhere:
                continue
            s0, s1 = spans[idx]
            if _overlaps((s0, s1), protected):
                continue
            main = pre[0].lower() + pre[1:] if first[0].isupper() else pre
            after = sub[0].upper() + sub[1:] + " " + clause + ", " + main + end
            edits.append(
                Edit(
                    kind="fronting",
                    start=s0,
                    end=s1,
                    before=text[s0:s1],
                    after=after,
                    sentence_index=sent.index,
                    rationale=(
                        "research/24 §1.2 (vary sentence openings): this sentence "
                        "opens the way the one before it does, the repeated "
                        "opener structure the presenter calls the single most "
                        f"important tell. Its trailing '{sub} ...' clause is moved "
                        "to the front, which is his own move ('Although widely "
                        "cited, this study ...'). No word is added or removed. "
                        "No pass-rate claim."
                    ),
                    grade_cost=GRADE_COST_NONE,
                )
            )
            last_fronted = idx
    return edits

TRANSFORMS: Dict[str, Transform] = {
    "paragraph_template": break_paragraph_template,
    "vocabulary": replace_ai_vocabulary,
    "parallelism": break_parallelism,
    "hedge": hedge_absolutes,
    "connective": strip_formal_connectives,
    "opener": relocate_contrastive_openers,
    "shape": vary_sentence_length,
    "fronting": front_trailing_clauses,
}

#: The pipeline, as an ordered sequence of passes. The engine applies each pass
#: in full and re-parses before the next, so a later pass sees the text the
#: earlier one produced. Order within a pass is the tie-break for the overlap
#: resolver ("skip the later one").
#:
#: Why three passes and not one:
#:
#: * Paragraph-closer deletion must beat every edit inside the sentence it is
#:   deleting, so it leads pass 1.
#: * Connective stripping is pass 2 because its edit span has to swallow the
#:   first letter of the following word in order to recapitalise it atomically.
#:   In one pass that span would block a vocabulary edit on exactly that word,
#:   and "Additionally, robust frameworks" would keep `robust`. Running the
#:   lexicon first and the strip second fixes both.
#: * Shape is pass 3 because sentence lengths only mean something once the
#:   deletions have landed.
#: * Hedging leads pass 1 (ahead of the lexicon) because both may target the
#:   same claim verb ("fosters"), and the hedge edit folds the lexicon's
#:   replacement into its own output ("can encourage"), so it must win the
#:   overlap. It also has to read the original word order for its
#:   "subject before verb" check, which pass 2's opener move would change.
#: * Opener relocation joins pass 2 next to connective stripping: the two act
#:   on disjoint connective sets (formal vs contrastive), and both need the
#:   pass-1 lexicon edits to have landed first.
#: * Clause fronting is pass 4, alone. Which sentences repeat an opener is
#:   only known once the connective strips (pass 2) and the boundary moves
#:   (pass 3) have landed: a stripped "Furthermore," exposes a bare subject,
#:   and a merge removes a sentence from the run.
PASSES: Tuple[Tuple[Tuple[str, Transform], ...], ...] = (
    (
        ("paragraph_template", break_paragraph_template),
        ("hedge", hedge_absolutes),
        ("vocabulary", replace_ai_vocabulary),
        ("parallelism", break_parallelism),
    ),
    (
        ("connective", strip_formal_connectives),
        ("opener", relocate_contrastive_openers),
    ),
    (("shape", vary_sentence_length),),
    (("fronting", front_trailing_clauses),),
)
