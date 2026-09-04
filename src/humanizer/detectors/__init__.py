"""Detector interfaces, published detector wrappers, and the GPTZero harness.

Every detector exported here scores with somebody else's pretrained
checkpoint. This repo writes no AI-detection arithmetic: the one piece of it
that ever existed, `HeuristicDetector`, was removed, and what was useful about
it survives as `AiStyleSignals` -- named, cited *explanatory* style signals
with no aggregate score, which is deliberately not a `Detector` and cannot be
passed to `analyze(detectors=[...])` or reached through `/api/detect`.

`PerplexityDetector` is the one borderline case and is labelled as such: gpt2
is published but the mapping from its perplexities to a probability is ours.
It is excluded from every default and reports `is_published_detector: false`.

`local` is imported eagerly, but it only pulls in stdlib plus `humanizer.text`.
Importing this package must never import torch: the CLI and the HTTP API have
to start on a machine without the `detectors` extra, and `available()` has to
answer False rather than raising. `tests/test_local_detectors.py` asserts that
"torch" is absent from `sys.modules` after importing this package.
"""

from .base import Detector, DetectorResult
from .calibration import (
    Calibration,
    calibrate,
    fleet_pass_rate,
    pass_rate,
    required_n,
)
from .gptzero import GPTZeroClient, estimate_cost
from .heuristic import AiStyleSignals
from .local import (
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_MODERN_MODEL,
    DEFAULT_PERPLEXITY_MODEL,
    FAST_PERPLEXITY_MODEL,
    ClassifierDetector,
    EnsembleDetector,
    ModernDetector,
    PerplexityDetector,
    backend_available,
    clear_model_cache,
    loaded_models,
    resolve_ai_index,
    resolve_head,
    check_load_integrity,
    MODEL_HEADS,
    PUBLISHED_DETECTORS,
    SHIPPED_DETECTORS,
)

__all__ = [
    "Detector",
    "DetectorResult",
    # Explanatory style signals. NOT a detector; see `detectors.heuristic`.
    "AiStyleSignals",
    "GPTZeroClient",
    "estimate_cost",
    "Calibration",
    "calibrate",
    "pass_rate",
    "fleet_pass_rate",
    "required_n",
    # Local open-weight detectors; need the `detectors` extra to actually run.
    # `ModernDetector` is the default engine: a current fine-tuned transformer
    # classifier, the same architectural class every 2026 commercial detector
    # uses. The other two are the 2023-era methods, kept for comparison.
    "ModernDetector",
    "PerplexityDetector",
    "ClassifierDetector",
    "EnsembleDetector",
    "backend_available",
    "clear_model_cache",
    "loaded_models",
    "DEFAULT_PERPLEXITY_MODEL",
    "FAST_PERPLEXITY_MODEL",
    "DEFAULT_CLASSIFIER_MODEL",
    "DEFAULT_MODERN_MODEL",
    "PUBLISHED_DETECTORS",
    "SHIPPED_DETECTORS",
    "MODEL_HEADS",
    "resolve_ai_index",
    "resolve_head",
    "check_load_integrity",
]
