"""GPTZero API client.

Schema reconstructed in research/07 from GPTZero's support articles and four
community wrappers, because their Stoplight docs are JavaScript-only.

Why this class is the centre of Stage 0: research/08 shows a local detector
score carries *zero* information about GPTZero on its own. The only number that
matters is the measured joint probability that GPTZero returns a human verdict
given our local score exceeds a threshold, calibrated on our own outputs. That
calibration is what `humanizer.detectors.calibration` computes, and this client
is how it buys the labels.

Cost control matters. Overage is about $0.00046 per word, so 10,000 evaluations
of 500-word documents runs to roughly $2,100. Every response is cached on disk
keyed by a hash of the text, so re-running an experiment is free.

Terms note from research/07: GPTZero's terms of service (21 Aug 2026) forbid
scraping, reverse engineering and competing use. They name neither adversarial
testing nor training on outputs. Get legal review before building a
distillation loop on top of this.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import Detector, DetectorResult

API_URL = "https://api.gptzero.me/v2/predict/text"
DEFAULT_CACHE = Path("data/cache/gptzero")
COST_PER_WORD_USD = 0.00046
MIN_CHARS = 250  # API rejects shorter inputs (research/07)


def estimate_cost(texts: Sequence[str]) -> float:
    """Estimated USD cost of scoring these documents at the overage rate."""
    return sum(len(t.split()) for t in texts) * COST_PER_WORD_USD


class GPTZeroClient(Detector):
    """Thin, cached client for the GPTZero text endpoint."""

    name = "gptzero"
    remote = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = DEFAULT_CACHE,
        multilingual: bool = False,
        min_interval: float = 0.2,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("GPTZERO_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.multilingual = multilingual
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_call = 0.0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------ cache

    def _cache_path(self, text: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(
            (text + f"|multilingual={self.multilingual}").encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def cached(self, text: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(text)
        if path and path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
        return None

    # ------------------------------------------------------------------- call

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def _request(self, text: str) -> Dict[str, Any]:
        import requests  # imported lazily so the package works offline

        self._throttle()
        response = requests.post(
            API_URL,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"document": text, "multilingual": self.multilingual},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def score(self, text: str, use_cache: bool = True) -> DetectorResult:
        if not self.api_key:
            raise RuntimeError(
                "No GPTZero API key. Set GPTZERO_API_KEY or pass api_key=."
            )
        if len(text) < MIN_CHARS:
            raise ValueError(
                f"GPTZero requires at least {MIN_CHARS} characters; got {len(text)}."
            )

        payload = self.cached(text) if use_cache else None
        if payload is None:
            payload = self._request(text)
            path = self._cache_path(text)
            if path:
                path.write_text(json.dumps(payload))
        return self.parse(payload)

    # ------------------------------------------------------------------ parse

    @staticmethod
    def parse(payload: Dict[str, Any]) -> DetectorResult:
        """Turn an API payload into a DetectorResult.

        Field names follow research/07. `class_probabilities` and
        `document_classification` are the modern fields;
        `completely_generated_prob` is the legacy one kept for compatibility.
        """
        docs = payload.get("documents") or []
        doc = docs[0] if docs else {}

        probs = doc.get("class_probabilities") or {}
        ai_prob = probs.get("ai")
        if ai_prob is None:
            ai_prob = doc.get("completely_generated_prob", float("nan"))

        classification = (
            doc.get("document_classification")
            or doc.get("predicted_class")
            or ""
        )
        label = _normalize_label(classification)

        sentences = [
            s.get("generated_prob", float("nan"))
            for s in (doc.get("sentences") or [])
        ]

        return DetectorResult(
            detector="gptzero",
            ai_probability=float(ai_prob) if ai_prob is not None else float("nan"),
            label=label,
            confidence=doc.get("confidence_category"),
            sentence_scores=sentences,
            raw={
                "version": payload.get("version"),
                "scanId": payload.get("scanId"),
                "class_probabilities": probs,
                "document_classification": classification,
                "confidence_score": doc.get("confidence_score"),
                "overall_burstiness": doc.get("overall_burstiness"),
                "average_generated_prob": doc.get("average_generated_prob"),
                "subclass": doc.get("subclass"),
                "result_message": doc.get("result_message"),
            },
        )

    def score_batch(
        self, texts: Sequence[str], use_cache: bool = True
    ) -> List[DetectorResult]:
        return [self.score(t, use_cache=use_cache) for t in texts]


def _normalize_label(classification: str) -> Optional[str]:
    """Map GPTZero's class strings onto human / ai / mixed.

    research/07: the taxonomy is a document softmax over {Human, AI, Mixed}
    with a nested subclass of {Pure AI, Polished, AI Paraphrased}. Anything
    paraphrase-flagged is emphatically not a human verdict, which is the whole
    point of research/12: "humanized" is a third detectable class.
    """
    if not classification:
        return None
    text = classification.strip().upper().replace(" ", "_")
    if "HUMAN" in text:
        return "human"
    if "MIXED" in text:
        return "mixed"
    if "AI" in text:
        return "ai"
    return None
