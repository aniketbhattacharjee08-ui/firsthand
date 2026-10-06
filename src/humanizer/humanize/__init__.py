"""The humanizing engine.

Everything in `humanizer` up to this point measures text. This package is the
part that changes it, and it changes only what research/00 section 4 rates as
safe: edits with high detector benefit and zero or negative cost to an academic
grade. The refusal list - contractions in body text, injected errors, broad
vocabulary simplification, added subordination, unpacked phrasal syntax,
"I think", reordered or dropped evidence, any edit inside a quote, citation or
the first/last sentence - is in the `transforms` module docstring together with
the matrix verdict that justifies each refusal.

Usage:
    from humanizer.humanize import humanize
    result = humanize(text, aggressiveness="balanced")
    print(result.humanized, result.summary["risk_delta"])
"""

from __future__ import annotations

from .engine import HumanizeResult, check_invariants, humanize, invariants
from .kriukow import structure_flags, structure_report
from .transforms import (
    CONFIGS,
    PASSES,
    TRANSFORMS,
    Edit,
    HumanizeConfig,
    apply_edits,
    break_paragraph_template,
    break_parallelism,
    config_for,
    hedge_absolutes,
    relocate_contrastive_openers,
    replace_ai_vocabulary,
    sentence_spans,
    strip_formal_connectives,
    vary_sentence_length,
)

__all__ = [
    "humanize",
    "HumanizeResult",
    "invariants",
    "check_invariants",
    "Edit",
    "HumanizeConfig",
    "CONFIGS",
    "TRANSFORMS",
    "PASSES",
    "config_for",
    "apply_edits",
    "sentence_spans",
    "strip_formal_connectives",
    "replace_ai_vocabulary",
    "break_paragraph_template",
    "break_parallelism",
    "vary_sentence_length",
    "hedge_absolutes",
    "relocate_contrastive_openers",
    "structure_report",
    "structure_flags",
]
