"""Register and grammatical-density features, without a parser.

Targets these against the research/15 numbers for real academic prose:
  * passives ~18.5 per 1,000 words (about 25% of finite verbs)
  * nominalizations 61.0-72.1 per 1,000 by sub-register
  * hedges 8.2 (mechanical engineering) to 20.0 (marketing) per 1,000
  * noun-noun premodification 76.6 (specialist science) vs 24.3 (history)

The load-bearing point from research/15 and research/09 is that **academic
complexity is phrasal, not clausal**. There are about twice as many dependent
clauses in conversation as in academic writing. A humanizer that adds
subordination to sound scholarly is moving toward conversation and away from
the target register.

research/04 also flags sentence-final participial tails as a strong AI tell:
5.3x human rate, Cohen's d = 1.38.

These are regex heuristics. Install the `syntax` extra for spaCy-backed
versions in `features/syntax.py`, which supersede the passive and participial
estimates here.
"""

from __future__ import annotations

import re
from typing import Dict

from ..text import Document

HUMAN_PASSIVE_PER_1K = (12.0, 25.0)
HUMAN_NOMINALIZATION_PER_1K = (55.0, 80.0)

_BE_FORMS = r"(?:is|are|was|were|be|been|being|am)"
# Irregular past participles common in academic prose.
_IRREGULAR_PP = (
    r"(?:shown|given|taken|seen|known|found|made|held|drawn|written|built|"
    r"chosen|driven|grown|led|left|lost|meant|met|paid|put|read|said|sent|"
    r"set|shot|sold|spent|told|thought|understood|proven|proved)"
)
_PASSIVE_RE = re.compile(
    rf"\b{_BE_FORMS}\b(?:\s+\w+ly)?\s+(?:\w+ed|{_IRREGULAR_PP})\b", re.IGNORECASE
)
_AGENTLESS_PASSIVE_RE = re.compile(
    rf"\b{_BE_FORMS}\b(?:\s+\w+ly)?\s+(?:\w+ed|{_IRREGULAR_PP})\b(?!\s+by\b)",
    re.IGNORECASE,
)

_NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "ancy",
    "ency", "ship", "hood", "age", "ure",
)

# Sentence-final participial clause: ", ...ing ..." closing the sentence.
_PARTICIPIAL_TAIL_RE = re.compile(r",\s+\w+ing\b[^,;:]{0,120}[.!?]\s*$")

_HEDGES = {
    "may", "might", "could", "would", "should", "possibly", "perhaps",
    "probably", "likely", "unlikely", "appear", "appears", "appeared",
    "seem", "seems", "seemed", "suggest", "suggests", "suggested",
    "indicate", "indicates", "indicated", "propose", "proposes", "proposed",
    "assume", "assumes", "assumed", "estimate", "estimates", "estimated",
    "approximately", "roughly", "somewhat", "relatively", "generally",
    "typically", "often", "sometimes", "potentially", "presumably",
    "arguably", "apparently", "largely", "partially", "tend", "tends",
    "tended", "suggesting", "indicating",
}

_BOOSTERS = {
    "clearly", "obviously", "certainly", "definitely", "undoubtedly",
    "indeed", "always", "never", "must", "demonstrate", "demonstrates",
    "demonstrated", "prove", "proves", "proved", "proven", "establish",
    "establishes", "established", "show", "shows", "shown", "evident",
    "conclusively", "invariably", "surely", "absolutely",
}

_SELF_MENTION_SINGULAR = {"i", "me", "my", "mine", "myself"}
_SELF_MENTION_PLURAL = {"we", "us", "our", "ours", "ourselves"}

# research/09 rates these as "never" edits: they read as unsupported opinion
# rather than argumentative stance, and cost grade under AP Row A.
_OPINION_MARKERS = (
    "i think", "i feel", "i believe", "in my opinion", "personally,",
    "to me,", "i guess", "if you ask me",
)

_DOWNTONERS = {
    "barely", "hardly", "scarcely", "nearly", "almost", "slightly",
    "marginally", "mildly", "somewhat",
}

_CONCESSIVES = {"although", "though", "whereas", "albeit", "notwithstanding"}

_TRICOLON_RE = re.compile(
    r"\b[\w'-]+(?:\s+[\w'-]+){0,3},\s+[\w'-]+(?:\s+[\w'-]+){0,3},\s+and\s+[\w'-]+",
)
_NEG_PARALLEL_RE = re.compile(
    r"\bnot\s+(?:just|only|merely|simply)\b[^.!?]{0,120}?\bbut\s+(?:also\s+)?",
    re.IGNORECASE,
)


def extract(doc: Document) -> Dict[str, float]:
    text = doc.text
    lower = doc.lower_words
    n = len(lower)
    feats: Dict[str, float] = {}
    if n == 0:
        return feats
    per_1k = 1000.0 / n

    passives = len(_PASSIVE_RE.findall(text))
    agentless = len(_AGENTLESS_PASSIVE_RE.findall(text))
    feats["passive_per_1k"] = passives * per_1k
    feats["agentless_passive_per_1k"] = agentless * per_1k

    nominalizations = sum(
        1
        for w in lower
        if len(w) > 5 and w.endswith(_NOMINALIZATION_SUFFIXES)
    )
    feats["nominalization_per_1k"] = nominalizations * per_1k

    tails = sum(
        1 for s in doc.sentences if _PARTICIPIAL_TAIL_RE.search(s.text)
    )
    feats["participial_tail_per_1k"] = tails * per_1k
    feats["participial_tail_share"] = (
        tails / len(doc.sentences) if doc.sentences else float("nan")
    )

    hedges = sum(1 for w in lower if w in _HEDGES)
    boosters = sum(1 for w in lower if w in _BOOSTERS)
    feats["hedge_per_1k"] = hedges * per_1k
    feats["booster_per_1k"] = boosters * per_1k

    singular = sum(1 for w in lower if w in _SELF_MENTION_SINGULAR)
    plural = sum(1 for w in lower if w in _SELF_MENTION_PLURAL)
    feats["self_mention_singular_per_10k"] = singular * per_1k * 10
    feats["self_mention_plural_per_10k"] = plural * per_1k * 10

    lowered_text = text.lower()
    feats["opinion_marker_count"] = float(
        sum(lowered_text.count(m) for m in _OPINION_MARKERS)
    )

    feats["downtoner_per_1k"] = sum(1 for w in lower if w in _DOWNTONERS) * per_1k
    feats["concessive_per_1k"] = sum(1 for w in lower if w in _CONCESSIVES) * per_1k

    feats["tricolon_per_1k"] = len(_TRICOLON_RE.findall(text)) * per_1k
    feats["negative_parallel_per_1k"] = len(_NEG_PARALLEL_RE.findall(text)) * per_1k

    # Paragraph openers. research/10: at most 3-4% of academic paragraph
    # openers should be a formal connective, and essentially none a coordinator.
    from .ai_lexicon import FORMAL_CONNECTIVES, INFORMAL_CONNECTIVES

    openers_formal = 0
    openers_coord = 0
    seen_paras = set()
    for sent in doc.sentences:
        if sent.paragraph_index in seen_paras:
            continue
        seen_paras.add(sent.paragraph_index)
        first = sent.words[0].lower() if sent.words else ""
        if first in FORMAL_CONNECTIVES:
            openers_formal += 1
        if first in INFORMAL_CONNECTIVES:
            openers_coord += 1
    n_paras = max(len(seen_paras), 1)
    feats["para_opener_formal_share"] = openers_formal / n_paras
    feats["para_opener_coordinator_share"] = openers_coord / n_paras

    # Sentence-initial coordinators across the whole document (research/10:
    # falls 5.1% -> 2.3% from essay score 1 to 6).
    sent_coord = sum(
        1
        for s in doc.sentences
        if s.words and s.words[0].lower() in {"and", "but", "so"}
    )
    feats["sent_initial_coordinator_share"] = (
        sent_coord / len(doc.sentences) if doc.sentences else float("nan")
    )

    return feats
