"""Detector interface.

Every detector returns a `DetectorResult` carrying both a probability and,
where the detector exposes one, a discrete verdict.

The verdict matters more than the probability. research/13 documents the DUPE
result: attacks that cut false-negative rates to 22-51% still produced only
1 document out of 400 actually labelled "Human". Optimising a probability delta
is not the same as earning a human verdict, so `label` is a first-class field
and the evaluation harness reports verdict rates, not mean probabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DetectorResult:
    """One detector's opinion about one document."""

    detector: str
    ai_probability: float
    label: Optional[str] = None  # "human" | "ai" | "mixed" | None
    confidence: Optional[str] = None
    sentence_scores: List[float] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def human_probability(self) -> float:
        return 1.0 - self.ai_probability

    @property
    def says_human(self) -> bool:
        """Verdict-level judgement, falling back to a 0.5 threshold."""
        if self.label is not None:
            return self.label == "human"
        return self.ai_probability < 0.5

    def __repr__(self) -> str:
        label = self.label or ("human" if self.says_human else "ai")
        return (
            f"<{self.detector}: p_ai={self.ai_probability:.3f} "
            f"label={label}>"
        )


class Detector(ABC):
    """Base class for anything that scores text for AI authorship."""

    name: str = "detector"
    #: True when the detector is a remote paid API, so callers can budget.
    remote: bool = False

    @abstractmethod
    def score(self, text: str) -> DetectorResult:
        """Score a single document."""

    def score_batch(self, texts: Sequence[str]) -> List[DetectorResult]:
        """Score many documents. Override when the backend supports batching."""
        return [self.score(t) for t in texts]

    def available(self) -> bool:
        """Whether this detector can run right now (deps present, key set)."""
        return True
