"""A transparent, dependency-free baseline detector.

This is not competitive with a trained classifier and is not meant to be. It
exists so the pipeline has a working scorer on day one, and so feature changes
can be regression-tested without a GPU or a paid API call.

It scores the features research/04 ranks highest and which research/13 shows
survive paraphrasing: AI-vocabulary density, sentence-length variation against
the human band, formal-connective load, participial tails, tricolons and
negative parallelism.

Deliberate design choice: sentence-length variation is scored as distance from
the *middle of the human band*, not as "more is better". research/10 found the
human CV band is 0.42-0.60 and that a GPT baseline sits at ~0.50 inside it, so
a detector that rewards extreme burstiness would mark hand-tuned humanizer
output as more human than real prose. Ours penalises both tails.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from ..features import extract_features
from ..features.shape import HUMAN_SENTENCE_CV
from .base import Detector, DetectorResult

# Weights are hand-set from the research/04 ranking, not fitted. Positive
# weights push toward "AI".
WEIGHTS: Dict[str, float] = {
    "ai_vocab_weighted": 1.30,
    "cv_band_distance": 0.90,
    "formal_connective": 0.70,
    "participial_tail": 0.60,
    "tricolon": 0.45,
    "negative_parallel": 0.45,
    "para_opener_formal": 0.50,
    "uniform_paragraphs": 0.40,
    "no_contractions": 0.15,
}
BIAS = -1.55


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _sat(value: Optional[float], scale: float) -> float:
    """Saturating transform to [0, 1); keeps one loud feature from dominating."""
    if value is None or not math.isfinite(value) or value <= 0:
        return 0.0
    return value / (value + scale)


class HeuristicDetector(Detector):
    name = "heuristic"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, text: str) -> DetectorResult:
        feats = extract_features(text)
        return self.score_features(feats)

    def score_features(self, feats: Dict[str, float]) -> DetectorResult:
        signals: Dict[str, float] = {}

        signals["ai_vocab_weighted"] = _sat(
            feats.get("ai_vocab_weighted_per_1k"), 25.0
        )
        signals["formal_connective"] = _sat(
            feats.get("formal_connective_per_1k"), 12.0
        )
        signals["participial_tail"] = _sat(
            feats.get("participial_tail_per_1k"), 6.0
        )
        signals["tricolon"] = _sat(feats.get("tricolon_per_1k"), 4.0)
        signals["negative_parallel"] = _sat(
            feats.get("negative_parallel_per_1k"), 2.0
        )
        signals["para_opener_formal"] = _sat(
            feats.get("para_opener_formal_share"), 0.12
        )

        # Distance from the centre of the human CV band, penalising both tails.
        cv = feats.get("sent_len_cv")
        lo, hi = HUMAN_SENTENCE_CV
        if cv is not None and math.isfinite(cv):
            centre = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            signals["cv_band_distance"] = _sat(abs(cv - centre) / half - 1.0, 1.0)
        else:
            signals["cv_band_distance"] = 0.0

        para_cv = feats.get("para_words_cv")
        if para_cv is not None and math.isfinite(para_cv):
            # Uniform paragraph lengths are an AI tell; human para CV runs
            # 0.42-0.71 (research/10).
            signals["uniform_paragraphs"] = _sat(max(0.0, 0.42 - para_cv), 0.2)
        else:
            signals["uniform_paragraphs"] = 0.0

        contractions = feats.get("contraction_per_1k", 0.0) or 0.0
        signals["no_contractions"] = 1.0 if contractions == 0 else 0.0

        logit = BIAS + sum(WEIGHTS[k] * v for k, v in signals.items())
        prob = _sigmoid(4.0 * logit)

        return DetectorResult(
            detector=self.name,
            ai_probability=prob,
            label="ai" if prob >= self.threshold else "human",
            raw={"signals": signals, "logit": logit},
        )
