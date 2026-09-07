"""Explanatory style signals. **Not a detector.**

WHY THIS IS NOT A DETECTOR ANY MORE
-----------------------------------
This module used to export `HeuristicDetector`: hand-set weights over nine
features, a hand-tuned bias, and a logistic squash, producing an
"ai_probability". Every number in that pipeline was invented in this repo. It
was fitted to nothing, validated against nothing, and it sat next to real
detector scores in the same response, which invited callers to read it as
comparable evidence.

Detection now comes exclusively from published, pretrained checkpoints
(`humanizer.detectors.local`). This repo writes no AI-detection arithmetic.

What survives is the part that was actually useful and was never a
classifier: the individual **style signals**. They answer "which known AI
tells are present in this text, and how strongly", which is a different
question from "was this generated". research/04 ranks these features and
research/13 shows they survive paraphrasing, so they are good *explanations*
to put next to a detector's verdict. They are not evidence for it.

WHAT CHANGED, CONCRETELY
------------------------
  * `HeuristicDetector` is gone. There is no `score()`, no `name`, no
    `label`, no `threshold`, no `DetectorResult`, and no subclass of
    `Detector`. It cannot be passed to `analyze(detectors=[...])`, it is not
    in `DETECTOR_FACTORIES`, and it cannot appear in `/api/detect`.
  * The `WEIGHTS` dict, the `BIAS` constant and the logistic are **deleted**,
    not commented out. There is no aggregate score here to resurrect.
  * `signals()` returns the per-feature values on a 0-1 scale. A caller that
    wants to rank them may; a caller that sums them is reinventing the thing
    this module was rewritten to remove.

WHAT THE SIGNALS MEAN
---------------------
Each value is a saturating normalisation of one measured feature onto [0, 1),
`v / (v + scale)`, so a signal near 1 means "far more of this than the scale
constant" and 0 means "none". The scale constants are per-feature reading
aids, chosen so a typical document lands mid-range; they are not thresholds
and nothing is classified against them.

`cv_band_distance` is the one signal that is not a simple count. It measures
how far the sentence-length coefficient of variation sits *outside* the human
band, in either direction. research/10 measured the human CV band at
0.42-0.60 with a GPT baseline at ~0.50, i.e. inside it, so both an unnaturally
flat document and an unnaturally spiky one are worth flagging. A signal that
rewarded extreme burstiness would tell a user to make their prose weirder,
which research/10 shows moves it *out* of the human band.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from ..features import extract_features
from ..features.shape import HUMAN_SENTENCE_CV

__all__ = ["AiStyleSignals", "SIGNAL_SCALES", "SIGNAL_SOURCES"]

#: Per-signal reading scale for the saturating transform. Not thresholds:
#: changing one rescales a display, it cannot change a verdict, because there
#: is no verdict in this module.
SIGNAL_SCALES: Dict[str, float] = {
    "ai_vocab_weighted": 25.0,
    "formal_connective": 12.0,
    "participial_tail": 6.0,
    "tricolon": 4.0,
    "negative_parallel": 2.0,
    "para_opener_formal": 0.12,
    "uniform_paragraphs": 0.2,
    "metronome_run": 3.0,
    "opener_repeat": 0.10,
}

#: Where each signal comes from, so a UI can cite it rather than assert it.
SIGNAL_SOURCES: Dict[str, str] = {
    "ai_vocab_weighted": "research/04: AI-vocabulary density, top-ranked tell",
    "cv_band_distance": "research/10: human sentence-length CV band 0.42-0.60",
    "formal_connective": "research/04: formal-connective load",
    "participial_tail": "research/04: participial tail clauses",
    "tricolon": "research/04: three-part lists",
    "negative_parallel": "research/04: 'not X, but Y' parallelism",
    "para_opener_formal": "research/04: formal paragraph openers",
    "uniform_paragraphs": "research/10: human paragraph-length CV 0.42-0.71",
    "no_contractions": "research/04: contraction absence",
    "metronome_run": "research/06: 3+ consecutive 17-23-word sentences",
    "opener_repeat": "research/06: consecutive sentences sharing a first word",
}


def _sat(value: Optional[float], scale: float) -> float:
    """Saturating normalisation onto [0, 1). Missing or non-finite reads 0."""
    if value is None or not math.isfinite(value) or value <= 0:
        return 0.0
    return value / (value + scale)


class AiStyleSignals:
    """Named, cited style signals for one document. Explanation, not scoring.

    Deliberately not a `Detector` and deliberately without an aggregate: see
    the module docstring. Use it to tell a reader *which* AI tells their draft
    contains; use a published detector from `humanizer.detectors.local` to
    decide whether the draft reads as generated.
    """

    def signals(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Signal name -> value in [0, 1], from an extracted feature vector."""
        out: Dict[str, float] = {}

        for name, feature in (
            ("ai_vocab_weighted", "ai_vocab_weighted_per_1k"),
            ("formal_connective", "formal_connective_per_1k"),
            ("participial_tail", "participial_tail_per_1k"),
            ("tricolon", "tricolon_per_1k"),
            ("negative_parallel", "negative_parallel_per_1k"),
            ("para_opener_formal", "para_opener_formal_share"),
            ("metronome_run", "struct_metronome_run_max"),
            ("opener_repeat", "struct_opener_repeat_share"),
        ):
            out[name] = _sat(feats.get(feature), SIGNAL_SCALES[name])

        # Distance outside the measured human band, penalising both tails.
        cv = feats.get("sent_len_cv")
        lo, hi = HUMAN_SENTENCE_CV
        if cv is not None and math.isfinite(cv):
            centre = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            out["cv_band_distance"] = _sat(abs(cv - centre) / half - 1.0, 1.0)
        else:
            out["cv_band_distance"] = 0.0

        para_cv = feats.get("para_words_cv")
        if para_cv is not None and math.isfinite(para_cv):
            # Uniform paragraph lengths are an AI tell; human paragraph CV
            # runs 0.42-0.71 (research/10).
            out["uniform_paragraphs"] = _sat(
                max(0.0, 0.42 - para_cv), SIGNAL_SCALES["uniform_paragraphs"]
            )
        else:
            out["uniform_paragraphs"] = 0.0

        contractions = feats.get("contraction_per_1k", 0.0) or 0.0
        out["no_contractions"] = 1.0 if contractions == 0 else 0.0
        return out

    def signals_for_text(self, text: str) -> Dict[str, float]:
        """`signals()` over a freshly extracted feature vector."""
        return self.signals(extract_features(text))

    def present(self, feats: Dict[str, float], floor: float = 0.25) -> Dict[str, float]:
        """Only the signals above `floor`, strongest first.

        `floor` is a display convenience with no semantics beyond "quiet
        signals are not worth a line of UI".
        """
        found = {k: v for k, v in self.signals(feats).items() if v >= floor}
        return dict(sorted(found.items(), key=lambda kv: -kv[1]))
