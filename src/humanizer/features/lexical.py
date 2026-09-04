"""Lexical diversity, sophistication and AI-vocabulary features.

Implements the research/04 lexical-richness bundle (MTLD, HD-D, hapax rate,
MATTR) plus AI-vocabulary density, which the catalog ranks first overall.

Two cautions carried from the research:
  * research/15: type-token ratio in academic prose is *lower* than in fiction
    and news. Raising lexical diversity to seem sophisticated moves academic
    text away from its real distribution.
  * research/16: TTR and Yule's K are among the weakest author discriminators
    (8% and 6% accuracy across 40 same-register authors). Track them because
    detectors compute them cheaply, not because they individuate.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from ..text import Document
from . import ai_lexicon

MTLD_THRESHOLD = 0.72
HDD_SAMPLE_SIZE = 42


def _mtld_pass(tokens: Sequence[str], threshold: float) -> float:
    factors = 0.0
    types: set = set()
    count = 0
    for token in tokens:
        types.add(token)
        count += 1
        ttr = len(types) / count
        if ttr <= threshold:
            factors += 1
            types = set()
            count = 0
    if count > 0:
        ttr = len(types) / count
        # Partial factor for the trailing segment.
        denom = 1.0 - threshold
        if denom > 0:
            factors += (1.0 - ttr) / denom
    if factors == 0:
        return float(len(tokens))
    return len(tokens) / factors


def mtld(tokens: Sequence[str], threshold: float = MTLD_THRESHOLD) -> float:
    """Measure of Textual Lexical Diversity, bidirectional mean.

    Length-robust unlike raw TTR. research/16 gives across-writer SD of roughly
    11-20 on a mean of 72-88 for student academic writing.
    """
    if len(tokens) < 50:
        return float("nan")
    forward = _mtld_pass(tokens, threshold)
    backward = _mtld_pass(list(reversed(tokens)), threshold)
    return float((forward + backward) / 2.0)


def hdd(tokens: Sequence[str], sample_size: int = HDD_SAMPLE_SIZE) -> float:
    """HD-D: expected type contribution from hypergeometric sampling.

    Equivalent to vocd-D but computed analytically rather than by curve fitting.
    """
    n = len(tokens)
    if n < sample_size:
        return float("nan")
    counts = Counter(tokens)
    total = 0.0
    for freq in counts.values():
        # P(at least one token of this type in a random sample of `sample_size`)
        # = 1 - C(n-freq, k) / C(n, k), computed in log space for stability.
        if n - freq < sample_size:
            prob = 1.0
        else:
            log_p_absent = (
                math.lgamma(n - freq + 1)
                - math.lgamma(n - freq - sample_size + 1)
                - math.lgamma(n + 1)
                + math.lgamma(n - sample_size + 1)
            )
            prob = 1.0 - math.exp(log_p_absent)
        total += prob
    return float(total / sample_size)


def mattr(tokens: Sequence[str], window: int = 50) -> float:
    """Moving-average type-token ratio."""
    n = len(tokens)
    if n < window:
        return float("nan")
    ratios = []
    counts: Counter = Counter(tokens[:window])
    ratios.append(len(counts) / window)
    for i in range(window, n):
        out_tok = tokens[i - window]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
        counts[tokens[i]] += 1
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


def extract(doc: Document) -> Dict[str, float]:
    lower = doc.lower_words
    n = len(lower)
    feats: Dict[str, float] = {}
    if n == 0:
        return feats

    per_1k = 1000.0 / n
    counts = Counter(lower)

    feats["ttr"] = len(counts) / n
    feats["mattr_50"] = mattr(lower, 50)
    feats["mtld"] = mtld(lower)
    feats["hdd"] = hdd(lower)
    feats["hapax_rate"] = sum(1 for c in counts.values() if c == 1) / n
    feats["dis_legomena_rate"] = sum(1 for c in counts.values() if c == 2) / n

    lengths = np.array([len(w) for w in lower], dtype=float)
    feats["word_len_mean"] = float(lengths.mean())
    feats["word_len_sd"] = float(lengths.std(ddof=1)) if n > 1 else float("nan")
    feats["long_word_share"] = float((lengths > 6).mean())

    # AI vocabulary. Two views: raw rate (how many hits) and weighted density
    # (how *incriminating* the hits are, using published excess ratios).
    word_hits, phrase_hits = ai_lexicon.find_hits(doc.text, lower)
    all_hits = word_hits + phrase_hits
    feats["ai_vocab_hits_per_1k"] = len(all_hits) * per_1k
    feats["ai_vocab_weighted_per_1k"] = sum(w for _, w in all_hits) * per_1k
    feats["ai_phrase_hits_per_1k"] = len(phrase_hits) * per_1k
    feats["ai_vocab_distinct"] = float(len({h for h, _ in all_hits}))

    # Formal vs informal connective balance (research/04 feature 9).
    formal = sum(counts[c] for c in ai_lexicon.FORMAL_CONNECTIVES if " " not in c)
    informal = sum(counts[c] for c in ai_lexicon.INFORMAL_CONNECTIVES)
    feats["formal_connective_per_1k"] = formal * per_1k
    feats["informal_connective_per_1k"] = informal * per_1k
    total_conn = formal + informal
    feats["formal_connective_ratio"] = (
        float(formal / total_conn) if total_conn else float("nan")
    )

    return feats
