"""Detector interfaces, local baselines, and the GPTZero calibration harness.

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
from .heuristic import HeuristicDetector
from .local import (
    ALTERNATIVE_MODERN_MODELS,
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
)

__all__ = [
    "Detector",
    "DetectorResult",
    "HeuristicDetector",
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
    "ALTERNATIVE_MODERN_MODELS",
    "resolve_ai_index",
    "resolve_head",
]
