"""Humanizer: feature extraction, reference distributions and detector bench."""

from .text import Document
from .features import extract_features, band_report

__version__ = "0.1.0"
__all__ = ["Document", "extract_features", "band_report"]
