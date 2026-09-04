"""Per-genre human reference distributions.

research/10 is explicit about why this cannot be a table of means:

    "Features are correlated (comma density with sentence length,
     nominalization with word length), so sampling them independently produces
     documents that are individually plausible on every axis and jointly
     impossible."

So a ReferenceDistribution stores the full feature matrix, not just summary
statistics, and offers three things:

  1. `summary()` - mean, SD and the 10/25/50/75/90 percentiles per feature.
  2. `mahalanobis()` - joint distance of a candidate from the human cloud,
     using a shrunk covariance estimate. This is the scorer that prevents
     overshooting into "unnaturally bursty": research/10 notes a CV of 0.85 in
     a research-article target is as far outside the distribution as 0.30.
  3. `sample_profile()` - draw a target vector by resampling a real document's
     whole profile, which preserves covariance for free and is the cheapest
     correct implementation.

The two-level persona model comes from the measured intraclass correlations in
research/10: no feature exceeds ICC 0.63, punctuation is most author-stable
(question marks 0.63, em dashes 0.52, commas 0.44), while hedges, passives and
lexical diversity sit at 0.07-0.20 and are mostly document-level noise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

PERCENTILES = (10, 25, 50, 75, 90)

# Measured intraclass correlations (research/10, 1,833 texts / 130 authors).
# Features absent here default to LOW_ICC_DEFAULT and float per document.
FEATURE_ICC: Dict[str, float] = {
    "question_mark_per_1k": 0.63,
    "em_dash_per_1k": 0.52,
    "paren_per_1k": 0.49,
    "comma_per_1k": 0.44,
    "colon_per_1k": 0.40,
    "semicolon_per_1k": 0.20,
    "sent_len_mean": 0.35,
    "sent_len_cv": 0.25,
    "mtld": 0.15,
    "hedge_per_1k": 0.12,
    "passive_per_1k": 0.10,
    "nominalization_per_1k": 0.18,
}
LOW_ICC_DEFAULT = 0.10


@dataclass
class ReferenceDistribution:
    """Feature distribution for one genre, backed by the raw feature matrix."""

    genre: str
    feature_names: List[str]
    matrix: np.ndarray  # shape (n_documents, n_features)
    doc_ids: List[str]
    source: str = ""

    # ---------------------------------------------------------------- build

    @classmethod
    def from_feature_dicts(
        cls,
        genre: str,
        rows: Sequence[Dict[str, float]],
        doc_ids: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
        source: str = "",
    ) -> "ReferenceDistribution":
        if not rows:
            raise ValueError(f"no documents supplied for genre {genre!r}")
        if feature_names is None:
            # Keep only features present and finite in every row, so the
            # covariance matrix has no holes.
            common = set(rows[0])
            for row in rows[1:]:
                common &= set(row)
            feature_names = sorted(
                name
                for name in common
                if all(_finite(row.get(name)) for row in rows)
            )
        names = list(feature_names)
        if not names:
            raise ValueError(
                f"genre {genre!r}: no feature is finite across all "
                f"{len(rows)} documents"
            )
        matrix = np.array(
            [[float(row[name]) for name in names] for row in rows], dtype=float
        )
        ids = list(doc_ids) if doc_ids is not None else [
            f"doc{i}" for i in range(len(rows))
        ]
        return cls(
            genre=genre,
            feature_names=names,
            matrix=matrix,
            doc_ids=ids,
            source=source,
        )

    # ------------------------------------------------------------ summarise

    @property
    def n_documents(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def mean(self) -> np.ndarray:
        return self.matrix.mean(axis=0)

    @property
    def sd(self) -> np.ndarray:
        if self.n_documents < 2:
            return np.zeros(self.matrix.shape[1])
        return self.matrix.std(axis=0, ddof=1)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Mean, SD, CV and percentiles per feature."""
        mean = self.mean
        sd = self.sd
        pct = np.percentile(self.matrix, PERCENTILES, axis=0)
        out: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.feature_names):
            entry = {
                "mean": float(mean[i]),
                "sd": float(sd[i]),
                "cv": float(sd[i] / mean[i]) if mean[i] else float("nan"),
            }
            for j, p in enumerate(PERCENTILES):
                entry[f"p{p}"] = float(pct[j, i])
            out[name] = entry
        return out

    # -------------------------------------------------------------- distance

    #: A feature must be non-zero in at least this share of reference
    #: documents to be trusted for distance scoring.
    MIN_NONZERO_SHARE = 0.15
    #: ...and take at least this many distinct values.
    MIN_DISTINCT_VALUES = 4

    def _active_mask(self, feats: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Features usable for a distance computation.

        Three conditions. The feature must vary in the reference corpus, since
        constant features carry no information and make the covariance
        singular. It must be *populated* often enough to support a stable
        variance estimate. And when a candidate is supplied it must be present
        and finite there too, so short or single-paragraph documents are still
        scored on the features they do have rather than collapsing to NaN.

        The population test is not decoration. On a 60-document corpus,
        `question_mark_per_1k` is zero in nearly every article, so its variance
        is a rounding artifact and a single question mark in a candidate
        produced a z-score of +22 and pushed an otherwise ordinary excerpt to
        the 100th distance percentile. Rare features need a much larger
        reference corpus before they mean anything.
        """
        sd = self.sd
        scale = np.maximum(np.abs(self.mean), 1.0)
        mask = sd > (1e-6 * scale)

        if self.n_documents >= 10:
            nonzero_share = (np.abs(self.matrix) > 0).mean(axis=0)
            mask = mask & (nonzero_share >= self.MIN_NONZERO_SHARE)
            distinct = np.array(
                [
                    len(np.unique(np.round(self.matrix[:, i], 9)))
                    for i in range(self.matrix.shape[1])
                ]
            )
            mask = mask & (distinct >= self.MIN_DISTINCT_VALUES)

        if feats is not None:
            available = np.array(
                [_finite(feats.get(name)) for name in self.feature_names]
            )
            mask = mask & available
        return mask

    def excluded_features(self) -> List[str]:
        """Reference features too sparse or too constant to score against."""
        mask = self._active_mask()
        return [n for n, keep in zip(self.feature_names, mask) if not keep]

    def _whitener(self, mask: Optional[np.ndarray] = None, shrinkage: float = 0.15):
        """Return (mask, mean, sd, inverse correlation matrix).

        Standardising before inverting keeps the problem well conditioned: the
        feature vector mixes counts per 1,000 words with shares in [0, 1], so
        the raw covariance spans many orders of magnitude. Shrinking the
        correlation matrix toward the identity handles the small-n regime,
        where the reference corpus is a few hundred documents against ~68
        features.
        """
        if mask is None:
            mask = self._active_mask()
        mean = self.mean[mask]
        sd = self.sd[mask]
        if mask.sum() == 0 or self.n_documents < 2:
            return mask, mean, sd, np.zeros((0, 0))
        standardized = (self.matrix[:, mask] - mean) / sd
        corr = np.cov(standardized, rowvar=False)
        corr = np.atleast_2d(corr)
        corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(corr.shape[0])
        return mask, mean, sd, _safe_pinv(corr)

    def mahalanobis(self, feats: Dict[str, float]) -> float:
        """Joint distance of a candidate document from this human cloud.

        Computed over the features the candidate actually has. Distances from
        different candidates are only comparable when their coverage matches,
        so use `score()` when you need the percentile too.
        """
        mask = self._active_mask(feats)
        if mask.sum() == 0:
            return float("nan")
        mean, sd, inv = self._whitener(mask)[1:]
        if inv.size == 0:
            return float("nan")
        vec = np.array(
            [float(feats[name]) for name, m in zip(self.feature_names, mask) if m]
        )
        delta = (vec - mean) / sd
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            value = float(delta @ inv @ delta)
        if not math.isfinite(value):
            return float("nan")
        return math.sqrt(max(value, 0.0))

    def reference_mahalanobis(
        self, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Distance of every reference document, for calibrating a threshold."""
        if mask is None:
            mask = self._active_mask()
        if mask.sum() == 0 or self.n_documents < 2:
            return np.zeros(self.n_documents)
        mean, sd, inv = self._whitener(mask)[1:]
        if inv.size == 0:
            return np.zeros(self.n_documents)
        deltas = (self.matrix[:, mask] - mean) / sd
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            vals = np.einsum("ij,jk,ik->i", deltas, inv, deltas)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sqrt(np.clip(vals, 0.0, None))

    def score(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Distance, percentile and coverage in one pass.

        The reference distances are recomputed on the *same* feature subset as
        the candidate, so the percentile is a fair comparison rather than a
        candidate scored on 20 features against a reference scored on 24.
        """
        mask = self._active_mask(feats)
        n_active = int(mask.sum())
        n_possible = int(self._active_mask().sum())
        if n_active == 0:
            return {
                "distance": float("nan"),
                "percentile": float("nan"),
                "n_features": 0.0,
                "coverage": 0.0,
            }
        distance = self.mahalanobis(feats)
        ref = self.reference_mahalanobis(mask)
        percentile = float((ref < distance).mean() * 100.0)
        return {
            "distance": distance,
            "percentile": percentile,
            "n_features": float(n_active),
            "coverage": float(n_active / n_possible) if n_possible else 0.0,
        }

    def distance_percentile(self, feats: Dict[str, float]) -> float:
        """Where a candidate falls in the reference distance distribution.

        research/10 recommends regenerating beyond roughly the 90th percentile,
        which automatically catches overshoot as well as undershoot.
        """
        return self.score(feats)["percentile"]

    def zscores(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Per-feature z-scores, for explaining *why* a document looks off."""
        mean, sd = self.mean, self.sd
        out: Dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            value = feats.get(name)
            if not _finite(value) or sd[i] == 0:
                continue
            out[name] = float((value - mean[i]) / sd[i])
        return out

    def vectorize(self, feats: Dict[str, float]) -> Optional[np.ndarray]:
        values = []
        for name in self.feature_names:
            value = feats.get(name)
            if not _finite(value):
                return None
            values.append(float(value))
        return np.asarray(values, dtype=float)

    # -------------------------------------------------------------- sampling

    def sample_profile(
        self, rng: Optional[np.random.Generator] = None, jitter: float = 0.0
    ) -> Dict[str, float]:
        """Draw a target vector by resampling a real document's whole profile.

        This is research/10 step 2. Resampling preserves the covariance
        structure exactly, unlike drawing each feature from its own marginal.
        `jitter` adds a small fraction of each feature's SD if you want targets
        that are near, but not identical to, a real document.
        """
        rng = rng or np.random.default_rng()
        idx = int(rng.integers(0, self.n_documents))
        vec = self.matrix[idx].copy()
        if jitter > 0:
            vec = vec + rng.normal(0.0, jitter * self.sd)
        return dict(zip(self.feature_names, (float(v) for v in vec)))

    def sample_persona(
        self, rng: Optional[np.random.Generator] = None
    ) -> Dict[str, float]:
        """Draw the persona-level offsets held fixed across one user's work.

        research/10 step 3: a persona offset of N(0, sd*sqrt(rho)) with a
        per-document residual of N(0, sd*sqrt(1-rho)), where rho is the measured
        intraclass correlation. High-ICC punctuation habits are frozen; low-ICC
        features float.
        """
        rng = rng or np.random.default_rng()
        sd = self.sd
        offsets: Dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            rho = FEATURE_ICC.get(name, LOW_ICC_DEFAULT)
            offsets[name] = float(rng.normal(0.0, sd[i] * math.sqrt(rho)))
        return offsets

    # ------------------------------------------------------------ persistence

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "genre": self.genre,
            "source": self.source,
            "feature_names": self.feature_names,
            "doc_ids": self.doc_ids,
            "matrix": self.matrix.tolist(),
            "summary": self.summary(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ReferenceDistribution":
        payload = json.loads(Path(path).read_text())
        return cls(
            genre=payload["genre"],
            feature_names=payload["feature_names"],
            matrix=np.asarray(payload["matrix"], dtype=float),
            doc_ids=payload.get("doc_ids", []),
            source=payload.get("source", ""),
        )


def _safe_pinv(matrix: np.ndarray) -> np.ndarray:
    """Pseudo-inverse, tolerating spurious BLAS warnings.

    numpy's pinv emits "divide by zero"/"overflow" RuntimeWarnings from inside
    matmul on Apple Silicon's Accelerate BLAS even for well-conditioned inputs
    (observed at condition number ~42 with a finite, correct result). Those
    warnings are suppressed here and the result is validated instead; if it
    really is non-finite we fall back to the diagonal inverse, which degrades
    the Mahalanobis distance to a plain z-score sum rather than failing.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inv = np.linalg.pinv(matrix)
    if not np.isfinite(inv).all():
        diag = np.diag(matrix).copy()
        diag[diag <= 0] = 1.0
        inv = np.diag(1.0 / diag)
    return inv


def _finite(value) -> bool:
    return (
        value is not None
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def build_from_texts(
    genre: str,
    texts: Iterable[str],
    doc_ids: Optional[Sequence[str]] = None,
    source: str = "",
    min_words: int = 300,
) -> ReferenceDistribution:
    """Build a reference distribution from raw documents.

    `min_words` defaults to 300 because research/16 puts weak attributability
    below ~2,000 words and near-nothing below ~500, while research/01 notes
    detectors refuse or hedge below ~50 words. Short documents contribute noisy
    shape features, especially the spectral estimate, which needs 16 sentences.
    """
    from ..clean import clean_markup, looks_like_markup
    from ..features import extract_features

    rows: List[Dict[str, float]] = []
    ids: List[str] = []
    supplied = list(doc_ids) if doc_ids is not None else None
    for i, text in enumerate(texts):
        if looks_like_markup(text):
            text = clean_markup(text)
        feats = extract_features(text)
        if feats.get("n_words", 0) < min_words:
            continue
        rows.append(feats)
        ids.append(supplied[i] if supplied and i < len(supplied) else f"{genre}-{i}")
    return ReferenceDistribution.from_feature_dicts(
        genre=genre, rows=rows, doc_ids=ids, source=source
    )
