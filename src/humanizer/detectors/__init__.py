"""Detector interfaces, a local baseline, and the GPTZero calibration harness."""

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
]
