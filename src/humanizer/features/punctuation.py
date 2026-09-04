"""Punctuation-profile features.

Punctuation carries far more weight than its size suggests:

  * research/16: Grieve's 40-author single-register study found a punctuation
    profile of only 8 features reaching 95% accuracy on 2 authors and 63% on
    40, against 2.5% chance. It is one of the strongest single indicators in
    that study, while mean sentence length manages 69% and 6%.
  * research/16: punctuation n-grams are the *worst* feature in-domain and the
    *best* cross-domain. Punctuation habits survive topic and genre shift.
  * research/10: punctuation habits are the most author-stable family measured,
    with intraclass correlations of 0.63 (question marks), 0.52 (em dashes),
    0.49 (parentheses) and 0.44 (commas). Everything else sits at 0.07-0.20.

That ICC ordering is why the sampler freezes punctuation per persona and lets
hedges, passives and lexical diversity float per document.

Reference values: academic comma density 57-65 per 1,000 words; contractions
under 1.4 per 1,000 in research articles; em dashes 3.23 per 1,000 for human
essays versus 10.62 for GPT-4.1 (Freeburg). Do not calibrate em dashes against
XML-derived corpora, which normalise dashes away.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict

import numpy as np

from ..text import Document

HUMAN_COMMA_PER_1K = (57.0, 65.0)
HUMAN_EM_DASH_PER_1K = (1.0, 6.0)
MAX_ACADEMIC_CONTRACTIONS_PER_1K = 1.4

# High-ICC features: freeze these per persona (research/10).
PERSONA_STABLE = (
    "question_mark_per_1k",
    "em_dash_per_1k",
    "paren_per_1k",
    "colon_per_1k",
    "comma_per_1k",
    "sent_len_mean",
)

MARKS = {
    "comma": ",",
    "period": ".",
    "semicolon": ";",
    "colon": ":",
    "question_mark": "?",
    "exclamation": "!",
    "em_dash": "—",
    "en_dash": "–",
    "hyphen": "-",
}

# Contractions only. A naive `\w+'s` pattern counts possessives such as
# "today's" and "the study's", which would report contractions in text that has
# none, so the 's form is restricted to the closed set of pronouns and
# wh-words where 's is genuinely a contracted verb.
_CONTRACTION_RE = re.compile(
    r"\b(?:"
    r"\w+['’](?:t|re|ve|ll|m)"              # don't, they're, we've, I'll, I'm
    r"|(?:i|you|he|she|we|they)['’]d"       # I'd, they'd
    r"|(?:it|that|there|here|what|who|where|how|let|he|she|one)['’]s"
    r")\b",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\(")
_QUOTE_RE = re.compile(r"[\"“”]")
_ELLIPSIS_RE = re.compile(r"\.{3}|…")


def _entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy over the punctuation-mark distribution, in bits.

    research/04 notes AI text has lower punctuation variety. Entropy captures
    that in one number without committing to any single mark.
    """
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    probs = np.array([c / total for c in counts.values() if c > 0], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def extract(doc: Document) -> Dict[str, float]:
    text = doc.text
    n_words = doc.n_words
    feats: Dict[str, float] = {}
    if n_words == 0:
        return feats
    per_1k = 1000.0 / n_words

    counts: Dict[str, int] = {}
    for name, char in MARKS.items():
        counts[name] = text.count(char)
        feats[f"{name}_per_1k"] = counts[name] * per_1k

    counts["paren"] = len(_PAREN_RE.findall(text))
    counts["quote"] = len(_QUOTE_RE.findall(text)) // 2
    counts["ellipsis"] = len(_ELLIPSIS_RE.findall(text))
    feats["paren_per_1k"] = counts["paren"] * per_1k
    feats["quote_per_1k"] = counts["quote"] * per_1k
    feats["ellipsis_per_1k"] = counts["ellipsis"] * per_1k

    feats["punct_entropy"] = _entropy(counts)
    feats["punct_total_per_1k"] = sum(counts.values()) * per_1k

    contractions = len(_CONTRACTION_RE.findall(text))
    feats["contraction_per_1k"] = contractions * per_1k

    # Sentence-final mark distribution: AI prose is overwhelmingly declarative.
    finals = Counter()
    for sent in doc.sentences:
        stripped = sent.text.rstrip("\"'”’)]")
        finals[stripped[-1] if stripped else ""] += 1
    total_sents = max(sum(finals.values()), 1)
    feats["sent_final_question_share"] = finals.get("?", 0) / total_sents
    feats["sent_final_exclaim_share"] = finals.get("!", 0) / total_sents

    return feats


def band_report(feats: Dict[str, float]) -> Dict[str, str]:
    """Flag punctuation features that sit outside academic human ranges."""
    out: Dict[str, str] = {}
    checks = {
        "comma_per_1k": HUMAN_COMMA_PER_1K,
        "em_dash_per_1k": HUMAN_EM_DASH_PER_1K,
    }
    for key, band in checks.items():
        value = feats.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out[key] = "unknown"
        elif value < band[0]:
            out[key] = "low"
        elif value > band[1]:
            out[key] = "high"
        else:
            out[key] = "in band"

    contractions = feats.get("contraction_per_1k")
    if contractions is None or (isinstance(contractions, float) and math.isnan(contractions)):
        out["contraction_per_1k"] = "unknown"
    else:
        out["contraction_per_1k"] = (
            "high" if contractions > MAX_ACADEMIC_CONTRACTIONS_PER_1K else "in band"
        )
    return out
