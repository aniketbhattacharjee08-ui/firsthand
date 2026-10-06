"""The manual humanising method of research/24, as checkable structure.

research/24 transcribes the most-viewed manual humanising tutorial for
academic writing (Dr Kriukow, 2025) and scores its demonstration paragraph on
the local bench. The method has six moves. Four are about *shape*, which the
presenter calls "the same number of beats": AI sentences share opener kind,
length and closing pattern even when the words differ. The other two are about
*stance*: hedge absolute claims, and note a limit where the draft states a
claim flat. The seventh instruction, "forget the wording and say the meaning in
your own words", is the whole-paragraph re-expression that research/00
finding 2 and research/19 §6 already require.

This module turns the six moves into numbers so that

* the LLM pipeline can rank candidates on them (`pipeline._rank_key`),
* the quality report can name them (`pipeline._quality` flags), and
* the deterministic transforms that implement the three mechanical moves
  (`transforms.hedge_absolutes`, `transforms.relocate_contrastive_openers`,
  `transforms.front_trailing_clauses`) can be tested against the same
  definitions.

Nothing here rewrites text. Everything is a pure function of a string.

What the bench says about the method, so nobody over-reads these numbers:
research/24 §2 measured the presenter's own before/after paragraph at
desklib 0.9999 -> 0.9829 with 0 of 5 detectors flipped. These checks are
quality and de-templating features with a small measured detector effect, not a
pass-rate lever. The honest label for every one of them is research/22 §7.2's
"quality-realism, no pass-rate claim" until the bench shows otherwise.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from ..text import Document, words

__all__ = [
    "OPENER_CONNECTIVE",
    "OPENER_FRONTED",
    "OPENER_PRONOUN",
    "OPENER_BARE",
    "CONTRASTIVE_RELOCATIONS",
    "ABSOLUTE_VERB_HEDGES",
    "ABSOLUTE_ADJECTIVE_HEDGES",
    "HEDGE_WORDS",
    "SURFACE_FRAMES",
    "HUMAN_HEDGES_PER_1K",
    "opener_class",
    "syllables",
    "structure_report",
    "structure_flags",
]


# ------------------------------------------------------------ vocabularies


OPENER_CONNECTIVE = "connective"
OPENER_FRONTED = "fronted"
OPENER_PRONOUN = "pronoun"
OPENER_BARE = "bare"

#: Sentence-initial contrastive or additive connectives and where the
#: presenter's "reverse the order" move puts them: after the subject, set off
#: by commas. "Conversely, low self-esteem may hinder" became "Low self-esteem,
#: on the other hand, may affect" in the demonstration (research/24 §1.3).
#: `strip_formal_connectives` deletes the *formal* set (Furthermore, Moreover,
#: Therefore ...) outright; these carry a logical relation a reader needs, so
#: they are moved, not deleted.
CONTRASTIVE_RELOCATIONS: Dict[str, str] = {
    "however": "however",
    "conversely": "on the other hand",
    "in contrast": "by contrast",
    "by contrast": "by contrast",
    "nevertheless": "nevertheless",
    "nonetheless": "nonetheless",
    "similarly": "similarly",
    "likewise": "likewise",
}

#: Formal connectives that mark a sentence as glue-opened. Mirrors
#: `transforms.STRIPPABLE_CONNECTIVES` plus the contrastive set above, kept
#: here as a plain tuple so this module does not import `transforms`.
_CONNECTIVE_OPENERS: Tuple[str, ...] = (
    "in conclusion", "in summary", "in addition", "in contrast", "by contrast",
    "on the other hand", "for example", "for instance", "as a result",
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "thus", "hence", "notably", "importantly", "ultimately", "overall",
    "however", "conversely", "nevertheless", "nonetheless", "similarly",
    "likewise", "indeed", "specifically", "finally", "first", "second",
    "firstly", "secondly", "lastly",
)

#: Words that begin a fronted dependent clause or adverbial: the "introductory
#: clauses, dependent clauses, inverted structures" of research/24 §1.2.
_SUBORDINATORS = frozenset(
    """
    although though while whereas because since as when whenever once until
    after before if unless given despite unlike where wherever
    """.split()
)
_PREPOSITIONS = frozenset(
    """
    in on at for from by with within without under over across among between
    through during against beyond toward towards among amid via per
    """.split()
)
_PRONOUN_OPENERS = frozenset(
    {"it", "this", "these", "that", "those", "they", "there", "such"}
)

#: Third-person present verbs that state a causal or evaluative claim as fact,
#: and the hedged form the presenter's "intellectual hesitation" move produces
#: (research/24 §1.2: "plays a critical role" -> "can play a critical role").
#: Only -s forms are listed so that no bare infinitive ("to play") is ever
#: touched. Verbs that are definitional or reportive (is, requires, includes,
#: represents, contains, shows-as-in-a-figure) are deliberately absent.
ABSOLUTE_VERB_HEDGES: Dict[str, str] = {
    "plays": "can play",
    "fosters": "can foster",
    "ensures": "helps ensure",
    "guarantees": "helps guarantee",
    "demonstrates": "suggests",
    "proves": "suggests",
    "leads": "can lead",
    "causes": "can cause",
    "hinders": "may hinder",
    "undermines": "may undermine",
    "improves": "can improve",
    "reduces": "can reduce",
    "increases": "can increase",
    "enhances": "can enhance",
    "strengthens": "can strengthen",
    "weakens": "can weaken",
    "affects": "may affect",
    "shapes": "can shape",
    "drives": "can drive",
    "transforms": "can transform",
    "promotes": "can promote",
    "creates": "can create",
    "generates": "can generate",
    "enables": "can enable",
    "prevents": "can prevent",
    "determines": "largely determines",
    "dictates": "largely dictates",
    "revolutionizes": "may change",
    "revolutionises": "may change",
}

#: "X is essential for Y" -> "X appears essential for Y". The copula is kept
#: and only the evaluative adjective is hedged, so agreement is untouched.
ABSOLUTE_ADJECTIVE_HEDGES: Tuple[str, ...] = (
    "essential", "vital", "crucial", "critical", "indispensable",
    "fundamental", "paramount", "imperative", "key",
)

#: Hedges and modals. A sentence that already contains one is left alone by
#: the hedging transform, and the density they produce is measured against
#: Hyland's band. Superset of `features.register._HEDGES` plus the phrasal
#: forms the presenter uses ("it is believed", "is suspected").
HEDGE_WORDS = frozenset(
    """
    may might could would should can possibly perhaps probably likely unlikely
    appear appears appeared seem seems seemed suggest suggests suggested
    indicate indicates indicated propose proposes proposed assume assumes
    assumed estimate estimates estimated approximately roughly somewhat
    relatively generally typically often sometimes potentially presumably
    arguably apparently largely partially partly tend tends tended suggesting
    indicating believed suspected thought argued reportedly plausibly
    """.split()
)

#: Hyland's per-discipline hedge band (research/15, research/00 §5): 8.2
#: (mechanical engineering) to 20.0 (marketing) per 1,000 words.
HUMAN_HEDGES_PER_1K: Tuple[float, float] = (8.0, 20.0)

#: Clauses that "seem meaningful but are not" (research/24 §1.1 factor 7;
#: research/21 finding 3, announced importance; research/22 §2, emphasis
#: metadiscourse at 0.52 vs 0.12 per 1,000 words). Counted, never deleted by
#: a transform: deletion is a content decision the matrix reserves for the
#: LLM path under the overlap gate.
SURFACE_FRAMES: Tuple[str, ...] = (
    r"\bit is (?:important|worth|crucial|essential|vital) to (?:note|consider|"
    r"recognize|recognise|remember|understand|acknowledge)\b",
    r"\bplays? an? (?:critical|crucial|vital|key|significant|important|pivotal|"
    r"central|major|essential) role\b",
    r"\bin (?:various|different|diverse|multiple|numerous|a variety of|a range of) "
    r"(?:contexts|settings|ways|fields|areas|domains|situations)\b",
    r"\b(?:overall|general) well-?being\b",
    r"\bcannot be (?:overlooked|overstated|ignored|underestimated)\b",
    r"\bis (?:essential|vital|crucial) (?:for|to) (?:ensuring|promoting|"
    r"fostering|achieving|understanding)\b",
    r"\bit is (?:clear|evident|undeniable) that\b",
    r"\b(?:a|the) (?:wide|broad) range of\b",
)
_SURFACE_RES = tuple(re.compile(p, re.IGNORECASE) for p in SURFACE_FRAMES)

#: A sentence that closes on a coordinated pair or triple: "..., expressing
#: needs, and participating in social contexts." The presenter's diagnosis is
#: that "almost every time it lists things it is two things divided by *and*",
#: and consecutive sentences closing this way are the template he merges
#: (research/24 §1.3).
_PAIRED_ENDING_RE = re.compile(
    r"\b[\w'-]+(?:\s+[\w'-]+){0,3},?\s+(?:and|or)\s+(?:also\s+)?[\w'-]+(?:\s+[\w'-]+){0,3}\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_NEG_PARALLEL_ENDING_RE = re.compile(
    r"\bnot\s+(?:only|just|merely)\b.*\bbut\s+(?:also\s+)?", re.IGNORECASE
)

_FIRST_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SYLLABLE_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


# ----------------------------------------------------------------- helpers


def syllables(word: str) -> int:
    """Cheap syllable estimate, identical in spirit to `pipeline._syllables`."""
    w = word.lower().strip("'’")
    if not w:
        return 0
    n = len(_SYLLABLE_RE.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def opener_class(sentence: str) -> str:
    """Classify how a sentence opens.

    ``connective``  a discourse connective, formal or contrastive, with or
                    without a comma ("However, ...", "Furthermore ...").
    ``fronted``     a dependent clause, adverbial or prepositional phrase
                    arrives before the subject ("Although widely cited, ...",
                    "In 2019, ...", "Central to this argument is ...").
    ``pronoun``     It / This / These / That / Those / They / There / Such.
    ``bare``        the subject noun phrase comes first ("The study ...",
                    "Migrants who ...", "Self-esteem ...").
    """
    text = sentence.strip()
    low = text.lower()
    for conn in sorted(_CONNECTIVE_OPENERS, key=len, reverse=True):
        if low.startswith(conn) and (
            len(low) == len(conn) or not low[len(conn)].isalpha()
        ):
            return OPENER_CONNECTIVE
    first_match = _FIRST_WORD_RE.match(text)
    if first_match is None:
        return OPENER_BARE
    first = first_match.group(0).lower()
    if first in _SUBORDINATORS:
        return OPENER_FRONTED
    head = words(text)[:8]
    # A sentence cannot open on a bare-subject preposition, so a prepositional
    # first word is a fronted adverbial whether or not a comma follows
    # ("In 2019 the court held otherwise").
    if first in _PREPOSITIONS and len(head) >= 4:
        return OPENER_FRONTED
    # A prepositional or participial opener set off by a comma before the
    # subject arrives: "In the years since, ..." / "Having failed twice, ...".
    comma = text.find(",")
    if 0 < comma < 60:
        pre = words(text[:comma])
        if pre and (pre[0].lower() in _PREPOSITIONS or pre[0].lower().endswith("ing")):
            return OPENER_FRONTED
    # Inversion of the "Central to this argument is ..." kind: an adjective or
    # participle followed by "to"/"among"/"in" before any verb.
    if len(head) >= 3 and head[1].lower() in {"to", "among", "in", "for"} and head[0].lower() not in _PRONOUN_OPENERS and head[0].lower() not in {"the", "a", "an"}:
        return OPENER_FRONTED
    if first in _PRONOUN_OPENERS:
        return OPENER_PRONOUN
    return OPENER_BARE


def _first_word(sentence: str) -> str:
    m = _FIRST_WORD_RE.search(sentence)
    return m.group(0).lower() if m else ""


def _ends_on_pair(sentence: str) -> bool:
    tail = sentence.strip()
    if _NEG_PARALLEL_ENDING_RE.search(tail):
        return True
    # Look only at the final clause so a mid-sentence "and" does not count.
    last_clause = re.split(r"[;:]|,\s+(?:which|who|that|because|although|while)\b", tail)[-1]
    return _PAIRED_ENDING_RE.search(last_clause) is not None


def _absolute_claims(sentence: str) -> int:
    """Unhedged evaluative claims in one sentence (0 or 1: the sentence is the unit)."""
    low = sentence.lower()
    toks = words(low)
    if any(t in HEDGE_WORDS for t in toks):
        return 0
    if re.search(r"\d", sentence):
        return 0
    if any(tok in ABSOLUTE_VERB_HEDGES for tok in toks):
        return 1
    if re.search(r"\b(?:is|are)\s+(?:" + "|".join(ABSOLUTE_ADJECTIVE_HEDGES) + r")\b", low):
        return 1
    return 0


def _surface_hits(sentence: str) -> int:
    return sum(1 for rx in _SURFACE_RES if rx.search(sentence))


# ------------------------------------------------------------------ report


def structure_report(text: str) -> Dict[str, object]:
    """Measure the six research/24 moves on one paragraph or document.

    Returns plain numbers so the result can go on the wire unchanged. Keys:

    ``n_sentences``
    ``opener_classes``            list, one of the four classes per sentence
    ``the_this_it_share``         share of sentences opening on a determiner
                                  or pronoun (research/21 §10.1 check 8;
                                  human bands 0.45-0.60)
    ``same_first_token_pairs``    adjacent sentences sharing a first word
                                  (check 9; human 0)
    ``max_same_opener_run``       longest run of one opener class
    ``similar_beat_pairs``        adjacent sentences within 3 words and 0.12
                                  syllables per word of each other
    ``max_beat_run``              longest run of such sentences
    ``paired_ending_share``       share of sentences closing on "X and Y"
    ``consecutive_paired_endings`` adjacent pairs that both do
    ``hedges_per_1k``             hedge words per 1,000 words
    ``absolute_claims``           sentences with an unhedged evaluative claim
    ``surface_statements``        matches of the empty-frame patterns
    ``penalty``                   0-1, mean of five normalised terms; the
                                  ranking term in `pipeline._rank_key`
    """
    doc = Document.parse(text or "")
    sents = [s.text for s in doc.sentences]
    n = len(sents)
    n_words = max(1, doc.n_words)
    if n == 0:
        return {
            "n_sentences": 0, "opener_classes": [], "the_this_it_share": 0.0,
            "same_first_token_pairs": 0, "max_same_opener_run": 0,
            "similar_beat_pairs": 0, "max_beat_run": 0,
            "paired_ending_share": 0.0, "consecutive_paired_endings": 0,
            "hedges_per_1k": 0.0, "absolute_claims": 0,
            "surface_statements": 0, "penalty": 0.0,
        }

    classes = [opener_class(s) for s in sents]
    firsts = [_first_word(s) for s in sents]
    det_pron = {"the", "a", "an", "this", "these", "it", "in", "there", "that", "those", "they"}
    the_share = sum(1 for f in firsts if f in det_pron) / n
    same_first = sum(1 for a, b in zip(firsts, firsts[1:]) if a and a == b)

    def _max_run(flags: Sequence[bool]) -> int:
        best = run = 0
        for f in flags:
            run = run + 1 if f else 0
            best = max(best, run)
        return best

    same_class_pairs = [a == b for a, b in zip(classes, classes[1:])]
    max_class_run = (_max_run(same_class_pairs) + 1) if same_class_pairs else 1

    lengths = [len(words(s)) for s in sents]
    spw = [
        (sum(syllables(w) for w in words(s)) / len(words(s))) if words(s) else 0.0
        for s in sents
    ]
    # "The same number of beats": word count is the beat; syllables per word
    # is a loose secondary check so that one polysyllable does not hide a
    # metronomic run.
    beat_pairs = [
        abs(l1 - l2) <= 3 and abs(s1 - s2) <= 0.35
        for (l1, l2), (s1, s2) in zip(zip(lengths, lengths[1:]), zip(spw, spw[1:]))
    ]
    similar_beats = sum(beat_pairs)
    max_beat_run = (_max_run(beat_pairs) + 1) if beat_pairs else 1

    pairs = [_ends_on_pair(s) for s in sents]
    paired_share = sum(pairs) / n
    consecutive_pairs = sum(1 for a, b in zip(pairs, pairs[1:]) if a and b)

    hedges = sum(1 for w in doc.lower_words if w in HEDGE_WORDS)
    hedges_per_1k = hedges * 1000.0 / n_words
    absolutes = sum(_absolute_claims(s) for s in sents)
    surface = sum(_surface_hits(s) for s in sents)

    denom = max(1, n - 1)
    terms = [
        min(1.0, same_first / denom + max(0.0, the_share - 0.5)),
        min(1.0, similar_beats / denom),
        min(1.0, consecutive_pairs / denom + max(0.0, paired_share - 0.5)),
        # An absolute claim is a target whatever the paragraph's overall hedge
        # density: the presenter hedged sentences 1, 2 and 5 of a paragraph
        # that already had "may" twice (research/24 §1.3).
        min(1.0, absolutes / n),
        min(1.0, surface / n),
    ]
    penalty = sum(terms) / len(terms)

    return {
        "n_sentences": n,
        "opener_classes": classes,
        "the_this_it_share": round(the_share, 4),
        "same_first_token_pairs": same_first,
        "max_same_opener_run": max_class_run,
        "similar_beat_pairs": similar_beats,
        "max_beat_run": max_beat_run,
        "paired_ending_share": round(paired_share, 4),
        "consecutive_paired_endings": consecutive_pairs,
        "hedges_per_1k": round(hedges_per_1k, 2),
        "absolute_claims": absolutes,
        "surface_statements": surface,
        "penalty": round(penalty, 4),
    }


def structure_flags(report: Dict[str, object]) -> List[str]:
    """Named findings for the quality report, in the vocabulary of research/24."""
    flags: List[str] = []
    n = int(report.get("n_sentences", 0) or 0)
    if n < 2:
        return flags
    if int(report.get("same_first_token_pairs", 0)) >= 1 or float(report.get("the_this_it_share", 0.0)) > 0.6:
        flags.append("repeated_openers")
    if int(report.get("max_beat_run", 1)) >= 3:
        flags.append("uniform_beats")
    # One adjacent pair of "X and Y" closers is ordinary prose; a run of three,
    # or a paragraph where most sentences close that way, is the template.
    consecutive = int(report.get("consecutive_paired_endings", 0))
    share = float(report.get("paired_ending_share", 0.0))
    if consecutive >= 2 or (consecutive >= 1 and share >= 0.6):
        flags.append("paired_coordination_template")
    if int(report.get("absolute_claims", 0)) >= 2:
        flags.append("unhedged_absolutes")
    if float(report.get("hedges_per_1k", 0.0)) > HUMAN_HEDGES_PER_1K[1] * 1.5:
        flags.append("over_hedged")
    if int(report.get("surface_statements", 0)) >= 1:
        flags.append("surface_statements")
    return flags
