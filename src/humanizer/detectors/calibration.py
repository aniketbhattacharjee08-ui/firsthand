"""Calibrating a local detector against GPTZero, and best-of-N pass-rate math.

Two things live here, both from research/08.

**1. The only number that matters.** A local detector saying "0.99 human"
carries no information about GPTZero. What carries information is the measured
conditional

    q(tau) = P(GPTZero verdict == human | local human-score >= tau)

estimated on our own outputs, with an interval. `calibrate()` builds that curve
and `select_threshold()` picks the operating point using the *lower* confidence
bound, so a threshold is never chosen on the strength of a handful of lucky
samples.

**2. Best-of-N does not follow 1-(1-p)^N.** research/08 identifies two
corrections:

  * *Heterogeneity.* The fleet pass rate is E_x[1-(1-p(x))^N], which saturates
    at 1 - P(p(x)=0). If 1.5% of inputs can never pass, no N reaches 98.5%.
  * *Correlation.* Candidates from one model share failure modes. Modelling
    p ~ Beta with intra-class correlation rho gives 99.93% at rho=0 but 87% at
    rho=0.5 for mu=0.6, N=8, and the tail decays as a power law in N rather
    than exponentially.

`pass_rate()` implements the Beta-binomial version so a target N can be chosen
against a realistic rho instead of the optimistic independent case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple:
    """Wilson score interval: well behaved at small n and at p near 0 or 1."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    ) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class CalibrationPoint:
    threshold: float
    n: int
    n_human: int
    rate: float
    lower: float
    upper: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "threshold": self.threshold,
            "n": float(self.n),
            "n_human": float(self.n_human),
            "rate": self.rate,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass
class Calibration:
    """The measured local-score to GPTZero-verdict mapping."""

    points: List[CalibrationPoint]
    n_total: int
    base_rate: float

    def select_threshold(self, target: float = 0.95) -> Optional[CalibrationPoint]:
        """Lowest threshold whose *lower* 95% bound clears `target`.

        Using the lower bound rather than the point estimate is what stops a
        threshold being chosen because three of three samples happened to pass.
        """
        for point in self.points:
            if point.lower >= target:
                return point
        return None

    def rate_at(self, threshold: float) -> Optional[CalibrationPoint]:
        best: Optional[CalibrationPoint] = None
        for point in self.points:
            if point.threshold <= threshold:
                best = point
        return best

    def summary(self) -> str:
        lines = [
            f"Calibration on {self.n_total} documents "
            f"(base human-verdict rate {self.base_rate:.1%})",
            f"{'threshold':>10} {'n':>6} {'human':>6} {'rate':>8} {'95% CI':>18}",
        ]
        for p in self.points:
            lines.append(
                f"{p.threshold:>10.2f} {p.n:>6d} {p.n_human:>6d} "
                f"{p.rate:>8.1%} [{p.lower:.1%}, {p.upper:.1%}]"
            )
        return "\n".join(lines)


def calibrate(
    local_human_scores: Sequence[float],
    gptzero_says_human: Sequence[bool],
    thresholds: Optional[Sequence[float]] = None,
    min_samples: int = 5,
) -> Calibration:
    """Measure P(GPTZero human | local score >= tau) across thresholds.

    `local_human_scores` should be *human*-oriented (higher means more human),
    matching `DetectorResult.human_probability`.
    """
    scores = np.asarray(local_human_scores, dtype=float)
    labels = np.asarray(gptzero_says_human, dtype=bool)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same length")
    if scores.size == 0:
        raise ValueError("no samples supplied")

    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0, 0.05), 2)

    points: List[CalibrationPoint] = []
    for tau in thresholds:
        mask = scores >= tau
        n = int(mask.sum())
        if n < min_samples:
            continue
        n_human = int(labels[mask].sum())
        lower, upper = wilson_interval(n_human, n)
        points.append(
            CalibrationPoint(
                threshold=float(tau),
                n=n,
                n_human=n_human,
                rate=n_human / n,
                lower=lower,
                upper=upper,
            )
        )

    return Calibration(
        points=points,
        n_total=int(scores.size),
        base_rate=float(labels.mean()),
    )


# --------------------------------------------------------------- best-of-N


def pass_rate(mu: float, n: int, rho: float = 0.0) -> float:
    """Expected best-of-N pass rate under a Beta-distributed per-candidate p.

    With p ~ Beta(a, b), the mean is mu = a/(a+b) and the intra-class
    correlation is rho = 1/(a+b+1), so a+b = 1/rho - 1. Then

        E[(1-p)^N] = prod_{i=0}^{N-1} (b+i) / (a+b+i)

    and the pass rate is one minus that. rho=0 recovers the familiar
    1-(1-mu)^N; rho>0 reproduces the power-law decay research/08 describes.
    """
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must be in [0, 1]")
    if n < 1:
        raise ValueError("n must be at least 1")
    if rho <= 0:
        return 1.0 - (1.0 - mu) ** n
    if rho >= 1:
        return mu
    total = 1.0 / rho - 1.0
    a = mu * total
    b = (1.0 - mu) * total
    log_fail = 0.0
    for i in range(n):
        log_fail += math.log(b + i) - math.log(a + b + i)
    return 1.0 - math.exp(log_fail)


def fleet_pass_rate(
    per_document_mu: Sequence[float], n: int, rho: float = 0.0
) -> float:
    """Pass rate across a heterogeneous document population.

    This is the saturation correction: documents whose per-candidate success
    probability is zero never pass at any N, so the fleet rate is capped at the
    share of documents with mu > 0 regardless of how much compute is spent.
    """
    mus = np.asarray(per_document_mu, dtype=float)
    if mus.size == 0:
        return float("nan")
    return float(np.mean([pass_rate(float(m), n, rho) for m in mus]))


def required_n(
    mu: float, target: float, rho: float = 0.0, max_n: int = 512
) -> Optional[int]:
    """Smallest N reaching `target`, or None if unreachable within max_n."""
    for n in range(1, max_n + 1):
        if pass_rate(mu, n, rho) >= target:
            return n
    return None


def correlation_penalty_table(
    mu: float = 0.6, n: int = 8, rhos: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5)
) -> List[Dict[str, float]]:
    """Reproduce the research/08 illustration for a chosen operating point."""
    return [
        {"rho": float(r), "pass_rate": pass_rate(mu, n, float(r))} for r in rhos
    ]
