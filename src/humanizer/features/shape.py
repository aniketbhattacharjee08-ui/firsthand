"""Sentence- and paragraph-shape features.

These are the features research/10 measured directly on five human corpora and
where the standard humanizer folk wisdom turned out to be wrong:

  * Human sentence-length CV sits in a *band* of 0.42-0.60, not "as high as
    possible". A measured GPT baseline is ~0.50, inside that band. Overshooting
    to 0.85 is as anomalous as undershooting to 0.30.
  * Humans do not alternate long and short sentences. Lag-1 autocorrelation is
    0.01-0.10. The real structure is long-range 1/f noise with spectral
    exponent beta ~= 0.5 (range 0.25-0.75).
  * The tails matter, not the middle: ~9-14% of academic sentences are under 10
    words and ~18-28% are over 30.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from ..text import Document

# Bands from research/10. Used by the scorer to flag over- and under-shoot.
HUMAN_SENTENCE_CV = (0.42, 0.60)
HUMAN_LAG1_AUTOCORR = (-0.10, 0.25)
HUMAN_SPECTRAL_BETA = (0.25, 0.75)
HUMAN_SHORT_SHARE = (0.09, 0.14)
HUMAN_LONG_SHARE = (0.18, 0.28)

SHORT_SENTENCE = 10
LONG_SENTENCE = 30


def _cv(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    mean = arr.mean()
    if mean == 0:
        return float("nan")
    # Sample standard deviation: these are samples of a writer's habit, not a
    # complete population.
    return float(arr.std(ddof=1) / mean)


def lag_autocorrelation(values: Sequence[float], lag: int = 1) -> float:
    """Autocorrelation of a series at the given lag.

    Near zero for human prose. A strongly negative value is the signature of a
    naive "alternate long and short" humanizer and is itself anomalous.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size <= lag + 1:
        return float("nan")
    centred = arr - arr.mean()
    denom = float((centred**2).sum())
    if denom == 0:
        return float("nan")
    return float((centred[:-lag] * centred[lag:]).sum() / denom)


def spectral_beta(values: Sequence[float]) -> float:
    """Estimate beta in a 1/f^beta power spectrum by log-log regression.

    Human sentence-length series show long-range correlation with beta ~= 0.5.
    White noise (independent draws) gives beta ~= 0, which is what naive
    per-sentence jitter produces.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 16:
        return float("nan")
    centred = arr - arr.mean()
    spectrum = np.abs(np.fft.rfft(centred)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Drop DC and the Nyquist bin; fit the low-frequency half where the
    # power-law holds.
    mask = (freqs > 0) & (freqs <= 0.25) & (spectrum > 0)
    if mask.sum() < 5:
        return float("nan")
    slope, _ = np.polyfit(np.log(freqs[mask]), np.log(spectrum[mask]), 1)
    return float(-slope)


def in_band(value: float, band: tuple) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return band[0] <= value <= band[1]


def extract(doc: Document) -> Dict[str, float]:
    lengths = doc.sentence_lengths
    para_words = doc.paragraph_word_lengths
    para_sents = doc.paragraph_lengths

    arr = np.asarray(lengths, dtype=float)
    n = arr.size

    feats: Dict[str, float] = {
        "n_words": float(doc.n_words),
        "n_sentences": float(n),
        "n_paragraphs": float(len(doc.paragraphs)),
    }

    if n:
        feats["sent_len_mean"] = float(arr.mean())
        feats["sent_len_median"] = float(np.median(arr))
        feats["sent_len_sd"] = float(arr.std(ddof=1)) if n > 1 else float("nan")
        feats["sent_len_cv"] = _cv(lengths)
        feats["sent_len_min"] = float(arr.min())
        feats["sent_len_max"] = float(arr.max())
        feats["sent_short_share"] = float((arr < SHORT_SENTENCE).mean())
        feats["sent_long_share"] = float((arr > LONG_SENTENCE).mean())
        feats["sent_lag1_autocorr"] = lag_autocorrelation(lengths, 1)
        feats["sent_lag2_autocorr"] = lag_autocorrelation(lengths, 2)
        feats["sent_spectral_beta"] = spectral_beta(lengths)
        if n > 1:
            diffs = np.abs(np.diff(arr))
            feats["sent_adjacent_absdiff_mean"] = float(diffs.mean())
            # Normalised so it is comparable across documents of different pace.
            feats["sent_adjacent_absdiff_norm"] = (
                float(diffs.mean() / arr.mean()) if arr.mean() else float("nan")
            )

    if para_words:
        feats["para_words_mean"] = float(np.mean(para_words))
        feats["para_words_cv"] = _cv(para_words)
        feats["para_sents_mean"] = float(np.mean(para_sents))
        feats["para_sents_cv"] = _cv(para_sents)
        feats["para_single_sentence_share"] = float(
            np.mean([1.0 if c <= 1 else 0.0 for c in para_sents])
        )

    return feats


def band_report(feats: Dict[str, float]) -> Dict[str, str]:
    """Compare shape features against the human bands from research/10.

    Returns "low", "in band" or "high" per feature. This is what stops the
    system overshooting into unnaturally bursty output, which is the specific
    failure mode the research flagged.
    """
    checks = {
        "sent_len_cv": HUMAN_SENTENCE_CV,
        "sent_lag1_autocorr": HUMAN_LAG1_AUTOCORR,
        "sent_spectral_beta": HUMAN_SPECTRAL_BETA,
        "sent_short_share": HUMAN_SHORT_SHARE,
        "sent_long_share": HUMAN_LONG_SHARE,
    }
    out: Dict[str, str] = {}
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
    return out
