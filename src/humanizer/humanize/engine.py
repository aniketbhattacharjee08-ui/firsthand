"""The humanizing pipeline: collect edits, gate them, apply them, re-measure.

Design
------
Transforms never touch text. Each one returns `Edit` objects against the
document it was handed; the engine is the only thing that splices, and it does
so only after two filters have run:

1. **Overlap resolution.** Edits are ordered by (start offset, transform rank)
   and any edit that starts inside an already-accepted span is skipped. The
   rank is the transform's position within its pass in `transforms.PASSES`,
   which makes the outcome deterministic: a paragraph-closer deletion beats a
   vocabulary swap inside the sentence it is deleting, because the deletion is
   listed first in that pass.
2. **The meaning gate** (`check_invariants`). Every edit is applied on its own
   and the result is compared against the input on a set of exact-match
   invariants. Any edit that changes one is dropped, then the full result is
   re-checked in case two individually safe edits interact.

Three passes, not one (`transforms.PASSES` documents why each boundary is
there). Each pass re-parses the text the previous one produced, so a
consequence documented in `transforms` holds here too: `Edit.start`/`Edit.end`
from different passes index different strings.

On the strength of the meaning guarantee
----------------------------------------
**This is weaker than entailment.** There is no NLI model in this package, so
"meaning preserved" is approximated by an exact-match invariant over numbers,
quoted spans, citation spans and capitalised multiword sequences. It will catch
a mangled figure, a broken citation, an edited quotation and a corrupted name.
It will *not* catch a paraphrase that reverses a claim, drops a hedge, or swaps
a synonym with different truth conditions. The defence against that class is
upstream and structural: every transform is a deletion of discourse glue, a
substitution from a hand-checked synonym map, or a sentence boundary move, and
none of them rewrites a proposition. A future NLI or QA-consistency check
belongs here, and the invariants should stay either way, since they are cheap
and catch the failures an entailment model is worst at.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..features import extract_from_document
from ..text import Document, normalize
# Imported from `text` rather than redeclared so that the meaning gate and
# `Document.protected_spans()` can never disagree about what a citation is.
from ..text import _NUMERIC_CITATION, _PAREN_CITATION, _QUOTE_SPAN
from .transforms import (
    PASSES,
    Edit,
    HumanizeConfig,
    apply_edits,
    config_for,
)

__all__ = [
    "HumanizeResult",
    "humanize",
    "invariants",
    "check_invariants",
]


# ------------------------------------------------------------ meaning gate


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_CAP_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")

#: Month and weekday names, so that a date is protected even when it carries no
#: digit ("in early March" vs "in March 2019").
_DATE_WORDS = frozenset(
    """
    january february march april may june july august september october
    november december monday tuesday wednesday thursday friday saturday sunday
    """.split()
)


def _entities(text: str) -> Counter:
    """Capitalised sequences that are not sentence-initial.

    Sentence-initial capitalisation is orthography, not identity: every
    transform here either deletes a sentence opener and recapitalises what
    follows, or moves a sentence boundary, so counting the first token of a
    sentence would flag correct edits as entity corruption. Everything after
    the first token is a real signal - a capitalised mid-sentence run is a name,
    a place, an instrument or a proper title.
    """
    found: Counter = Counter()
    for sent in Document.parse(text).sentences:
        run: List[str] = []
        for i, match in enumerate(_CAP_TOKEN_RE.finditer(sent.text)):
            token = match.group(0)
            if i == 0:
                continue
            if token[0].isupper():
                run.append(token)
            elif run:
                found[" ".join(run)] += 1
                run = []
        if run:
            found[" ".join(run)] += 1
    return found


def invariants(text: str) -> Dict[str, Any]:
    """Everything an edit is forbidden to alter.

    Counts are exact multisets for numbers, quotations and citations: those are
    evidence, and losing or duplicating one is a factual change. Entities are
    compared as a *set*, because the one deletion this package performs (a
    paragraph closer that introduces no new content word) legitimately removes a
    repeated mention. The set test still catches the failure that matters: an
    entity appearing, vanishing, or being spelled differently.
    """
    return {
        "numbers": Counter(_NUMBER_RE.findall(text)),
        "quotations": Counter(m.group(0) for m in _QUOTE_SPAN.finditer(text)),
        "citations": Counter(
            m.group(0)
            for pattern in (_PAREN_CITATION, _NUMERIC_CITATION)
            for m in pattern.finditer(text)
        ),
        "dates": Counter(
            w.lower() for w in _CAP_TOKEN_RE.findall(text) if w.lower() in _DATE_WORDS
        ),
        "entities": set(_entities(text)),
    }


def check_invariants(before: str, after: str) -> List[str]:
    """Names of the invariants that `after` violates. Empty means it is safe."""
    a, b = invariants(before), invariants(after)
    return [key for key in sorted(a) if a[key] != b[key]]


# ---------------------------------------------------------------- pipeline


def _resolve_overlaps(ranked: Sequence[Tuple[int, Edit]]) -> List[Edit]:
    """Keep the earlier of any two overlapping edits.

    Sorted by start offset, then by transform rank, then by span length
    descending so that the more decisive of two co-located edits wins. Ties
    beyond that are broken by kind so the order is total and the pipeline is
    reproducible.
    """
    ordered = sorted(
        ranked,
        key=lambda pair: (
            pair[1].start,
            pair[0],
            -(pair[1].end - pair[1].start),
            pair[1].kind,
        ),
    )
    kept: List[Edit] = []
    last_end = -1
    for _rank, edit in ordered:
        if edit.start < last_end:
            continue
        if edit.end < edit.start:  # pragma: no cover - defensive
            continue
        kept.append(edit)
        last_end = edit.end
    return kept


def _gate(base: str, edits: Sequence[Edit]) -> Tuple[List[Edit], List[Edit]]:
    """Split edits into (accepted, dropped) under the meaning invariants.

    Each edit is tested alone first, which localises the blame: if two edits
    together break an invariant only the second is dropped, not both. The
    combined result is then re-checked, because an interaction is possible in
    principle even when neither edit fails alone.
    """
    accepted: List[Edit] = []
    dropped: List[Edit] = []
    for edit in edits:
        if check_invariants(base, apply_edits(base, [edit])):
            dropped.append(edit)
        else:
            accepted.append(edit)

    while accepted and check_invariants(base, apply_edits(base, accepted)):
        dropped.append(accepted.pop())
    return accepted, dropped


@dataclass
class HumanizeResult:
    """Before/after text, the edit list, and the measurement of both."""

    original: str
    humanized: str
    aggressiveness: str
    edits: List[Edit] = field(default_factory=list)
    dropped: List[Edit] = field(default_factory=list)
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.humanized != self.original

    def edit_counts(self) -> Dict[str, int]:
        """Edit count by kind, for the CLI table and the evaluation report."""
        return dict(Counter(e.kind for e in self.edits))

    def as_dict(self) -> Dict[str, Any]:
        """Exactly the POST /api/humanize response body.

        `dropped` and `aggressiveness` are deliberately not on the wire: the
        endpoint contract is pinned by a test and by the frontend, and extra
        keys are how contracts rot.
        """
        return {
            "original": self.original,
            "humanized": self.humanized,
            "edits": [e.as_dict() for e in self.edits],
            "before": self.before,
            "after": self.after,
            "summary": self.summary,
        }


def _detector() -> Any:
    """The process-wide scoring model, built on first use.

    `ModernDetector` wraps `desklib/ai-text-detector-v1.01`, a published
    fine-tuned DeBERTa-v3-large trained on RAID. It is used here rather than
    any score this repo wrote itself: hand-set weights measure our own opinion
    of what looks generated, which is exactly the thing a before/after number
    must not do. On this repo's own corpus the published model separates human
    from AI paragraphs at 1.000 pairwise against 0.740 for the perplexity
    engine, so it is also simply the better yardstick.

    Imported inside the function, never at module scope: `humanizer.humanize`
    has to import on a machine with no torch, and `tests/test_local_detectors`
    asserts the detectors package itself stays torch-free at import time.

    `score_sentences=False` because only the document score is needed here.
    Per-sentence rows cost roughly 5x and research/01 puts single sentences
    well under every vendor's stated reliability floor anyway.
    """
    global _DETECTOR
    if _DETECTOR is None:
        # Since 2026-09-07 the yardstick is `detectors.yardstick.resolve()`:
        # GPTZero when a key is configured, the local proxy otherwise. The
        # `Yardstick` object has the same `available()`/`score()` surface.
        from ..detectors import yardstick

        _DETECTOR = yardstick.resolve()
    return _DETECTOR


#: Cached across calls: the model is ~1.74GB and loading it per request would
#: dominate a rewrite that is otherwise a few milliseconds of regex work.
_DETECTOR: Optional[Any] = None


def _measure(doc: Document) -> Dict[str, Any]:
    """Feature vector plus the detector's AI probability for one document.

    `ai_probability` is **null** when the model cannot run - torch missing, the
    weights not downloaded, an empty document. That is a degraded response, not
    an error: the rewrite and the feature-level before/after (sentence-length
    CV against the research/10 band, AI-vocabulary density) need no model at
    all and are still worth returning. Callers must handle null.
    """
    feats = extract_from_document(doc)
    probability: Optional[float] = None
    label: Optional[str] = None
    yardstick_block: Optional[Dict[str, Any]] = None
    if doc.text.strip():
        try:
            detector = _detector()
            yardstick_block = {
                "name": getattr(detector, "name", None),
                "kind": getattr(detector, "kind", None),
                "model": getattr(detector, "model", None),
            }
            if detector.available():
                result = detector.score(doc.text)
                probability = result.ai_probability
                if probability is not None and probability != probability:
                    probability = None
                label = getattr(result, "label", None)
                if label not in ("human", "ai", "mixed") and probability is not None:
                    label = "ai" if probability >= 0.5 else "human"
        except Exception:  # noqa: BLE001 - a missing model must not 500
            probability = None
            label = None
    return {
        "ai_probability": probability,
        "label": label,
        "features": feats,
        "yardstick": yardstick_block,
    }


def _delta(before: Dict[str, Any], after: Dict[str, Any]) -> Optional[float]:
    """after - before on the AI probability, or None if either is unscored."""
    a, b = before.get("ai_probability"), after.get("ai_probability")
    if a is None or b is None:
        return None
    return b - a


def humanize(
    text: str,
    aggressiveness: str = "balanced",
    config: Optional[HumanizeConfig] = None,
) -> HumanizeResult:
    """Rewrite `text` toward the human band and report what changed.

    Args:
        text: the document. Normalised on entry, so the `original` field is the
            normalised form and offsets in the edit list line up with it.
        aggressiveness: ``light`` (lexicon and connectives), ``balanced``
            (adds parallelism and paragraph template) or ``strong`` (adds
            sentence-length restructuring). See `transforms.HumanizeConfig`.
        config: an explicit configuration, overriding `aggressiveness`. Used by
            tests to enable a single transform.

    Returns:
        A `HumanizeResult`. `summary["risk_delta"]` is ``after - before`` on
        the detector's AI probability, so a **negative** number means the
        document moved toward "human". It is **null** when the detector could
        not run, which is a degraded but valid response - the rewrite and the
        feature-level before/after do not need a model.

        The detector is `ModernDetector`, not GPTZero. It is the same
        architectural family as every post-2023 commercial detector, but its
        training data and calibration are its own, so there is no relationship
        between this number and the one GPTZero would return. Read the delta as
        "this document moved", not as "this document will pass".
    """
    cfg = config if config is not None else config_for(aggressiveness)
    original = normalize(text or "")

    doc = Document.parse(original)
    before = _measure(doc)

    working = doc.text
    edits: List[Edit] = []
    dropped: List[Edit] = []

    for stage in PASSES:
        active = [(name, fn) for name, fn in stage if name in cfg.transforms]
        if not active:
            continue
        # Re-parse: every pass must see the text the previous one produced,
        # not a prediction of it.
        stage_doc = Document.parse(working)
        ranked: List[Tuple[int, Edit]] = []
        for rank, (_name, transform) in enumerate(active):
            for edit in transform(stage_doc, cfg):
                ranked.append((rank, edit))
        accepted, stage_dropped = _gate(stage_doc.text, _resolve_overlaps(ranked))
        working = apply_edits(stage_doc.text, accepted)
        edits.extend(accepted)
        dropped.extend(stage_dropped)

    after_doc = Document.parse(working)
    after = _measure(after_doc)

    return HumanizeResult(
        original=original,
        humanized=after_doc.text,
        aggressiveness=cfg.name,
        edits=edits,
        dropped=dropped,
        before=before,
        after=after,
        summary={"n_edits": len(edits), "risk_delta": _delta(before, after)},
    )
