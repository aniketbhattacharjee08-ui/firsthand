"""Feature extraction for AI-vs-human text analysis.

Implements the ranked catalog from research/04 plus the shape features
research/10 measured directly on human corpora.

Usage:
    from humanizer.features import extract_features
    feats = extract_features(text)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..text import Document
from . import lexical, punctuation, register, shape, structure

__all__ = [
    "extract_features",
    "extract_from_document",
    "feature_names",
    "band_report",
    "syntax_available",
    "FEATURE_ORDER",
]

_MODULES = (shape, lexical, punctuation, register, structure)

# Non-stylistic bookkeeping fields: reported, but excluded from the feature
# vector used for distance and sampling, since document size is not a style.
SIZE_FIELDS = frozenset({"n_words", "n_sentences", "n_paragraphs"})


def extract_from_document(doc: Document, with_syntax: bool = False) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for module in _MODULES:
        feats.update(module.extract(doc))
    if with_syntax:
        try:
            from . import syntax  # optional, requires the `syntax` extra
        except ImportError:
            pass
        else:
            feats.update(syntax.extract(doc))
    return feats


def extract_features(text: str, with_syntax: bool = False) -> Dict[str, float]:
    """Extract the full feature vector from raw text."""
    return extract_from_document(Document.parse(text), with_syntax=with_syntax)


def feature_names(feats: Dict[str, float], include_size: bool = False) -> List[str]:
    """Stable sorted feature names, size fields optionally excluded."""
    names = sorted(feats)
    if not include_size:
        names = [n for n in names if n not in SIZE_FIELDS]
    return names


def band_report(feats: Dict[str, float]) -> Dict[str, str]:
    """Human-band checks across every module that defines them.

    Syntax bands are added only when the parsed features are present, so the
    report stays consistent whether or not the optional backend ran.
    """
    out: Dict[str, str] = {}
    out.update(shape.band_report(feats))
    out.update(punctuation.band_report(feats))
    out.update(structure.band_report(feats))
    if any(name.startswith("syn_") for name in feats):
        try:
            from . import syntax
        except ImportError:
            pass
        else:
            out.update(syntax.band_report(feats))
    return out


def syntax_available() -> bool:
    """Whether the optional spaCy clause-level backend can run."""
    try:
        from . import syntax
    except ImportError:
        return False
    return syntax.available()


# Canonical ordering for the reference-distribution vectors. Computed lazily
# from a probe document so the list always matches what the extractors emit.
FEATURE_ORDER: Optional[List[str]] = None


def canonical_order() -> List[str]:
    """The feature-vector ordering used by reference distributions."""
    global FEATURE_ORDER
    if FEATURE_ORDER is None:
        probe = (
            "The study examines a question. It does so carefully, and at "
            "length, over several sentences. Researchers have shown that the "
            "effect is robust; others disagree. We therefore propose a model.\n\n"
            "A second paragraph follows here. It adds detail, evidence, and "
            "commentary. The result may indicate something important."
        )
        FEATURE_ORDER = feature_names(extract_features(probe))
    return FEATURE_ORDER
