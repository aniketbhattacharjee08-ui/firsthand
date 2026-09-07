"""The yardstick: the one detector whose verdict this project is trying to beat.

WHY THIS MODULE EXISTS
----------------------
Until 2026-09-07 every rewriting path in `humanizer.humanize` scored itself
against `desklib/ai-text-detector-v1.01`, and the module docstrings of
`pipeline`, `guided` and `pretrained` all record the same number against it:
0 verdict flips of 9. The product goal is not "beat desklib". It is "pass
GPTZero" (research/06, /07, /13), and research/08 §2.2 is blunt that a local
detector's probability carries no information about GPTZero's verdict on its
own. So the yardstick is now GPTZero itself, and this module is the single
place that decides what "the yardstick" is, so that the three engines, the
HTTP health endpoints and the benchmark cannot drift apart about it.

TWO ROLES, KEPT APART ON PURPOSE
--------------------------------
* **The yardstick** scores the document *before* and *after*. Its `label` is
  the verdict that `summary["verdict_flipped"]` reports. When GPTZero is
  configured, that is a real GPTZero verdict from the paid API.

* **The proxy** scores every *candidate* inside the loop, where there may be
  fifty of them per document. It is a local model, free and fast. research/07
  §7 recommends exactly this posture -- GPTZero as a periodic validator, never
  an inner-loop reward -- for three reasons: cost (about $0.00046 a word, so
  a 500-word document with six candidates per paragraph is roughly $1.40 per
  run), the 250-character minimum that most single paragraphs' candidates
  fail, and the terms of service, which permit the API but not automation
  against the web app.

  The proxy's number is never written into a field called `ai_probability` on
  the document. Candidates carry a `proxy_probability`; the document carries
  the yardstick's. research/08 §2.2 again: surfacing the local probability as
  if it were GPTZero's is the specific mistake that module exists to prevent.

When no GPTZero key is configured, the yardstick *is* the proxy, and
`describe()["kind"]` says `"local-proxy"` so nobody mistakes the number for a
GPTZero verdict. That is the state this machine was in when the module was
written, and it is what every measurement in this repo so far was taken in.

CONFIGURATION
-------------
::

    HUMANIZER_YARDSTICK   auto (default) | gptzero | modern | fakespot |
                          academic | fast | radar
    HUMANIZER_PROXY       modern (default) | fakespot | academic | fast | radar
    GPTZERO_API_KEY       enables `gptzero`; `auto` picks gptzero when set

`auto` means: GPTZero if a key is present, otherwise the proxy. `gptzero`
without a key resolves to an *unavailable* yardstick rather than silently
falling back, because a caller who asked for GPTZero and got desklib would
read the wrong number.

The whole thing is memoised per process. Call `reset()` in tests.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import DetectorResult

__all__ = [
    "YARDSTICK_ENV",
    "PROXY_ENV",
    "DEFAULT_PROXY",
    "Yardstick",
    "resolve",
    "proxy",
    "describe",
    "reset",
    "configured_name",
    "unavailable_reason",
]

YARDSTICK_ENV = "HUMANIZER_YARDSTICK"
PROXY_ENV = "HUMANIZER_PROXY"
DEFAULT_PROXY = "modern"
#: The names `HUMANIZER_YARDSTICK` accepts besides `auto` and `gptzero`.
LOCAL_NAMES = ("modern", "fakespot", "academic", "fast", "radar")


@dataclass
class Yardstick:
    """A detector plus the bookkeeping the response needs about it.

    `detector` is anything with `.score(text) -> DetectorResult` and
    `.available()`. `kind` is `"gptzero"` or `"local-proxy"`. `model` is the
    checkpoint or API name shown to the user. `remote` mirrors
    `Detector.remote` so callers can budget.
    """

    name: str
    kind: str
    model: str
    detector: Any = None
    remote: bool = False
    reason: Optional[str] = None
    #: Running totals, for the response's cost line. Only meaningful for a
    #: remote yardstick; local scoring is free and is not counted.
    calls: int = 0
    words_scored: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- availability --------------------------------------------------------

    def available(self) -> bool:
        if self.detector is None:
            return False
        try:
            return bool(self.detector.available())
        except Exception:  # noqa: BLE001 - availability must not raise
            return False

    def unavailable_reason(self) -> Optional[str]:
        if self.available():
            return None
        if self.reason:
            return self.reason
        fn = getattr(self.detector, "unavailable_reason", None)
        if callable(fn):
            try:
                return fn() or "the yardstick detector is unavailable"
            except Exception:  # noqa: BLE001
                pass
        return "the yardstick detector is unavailable"

    # -- scoring -------------------------------------------------------------

    def score(self, text: str) -> DetectorResult:
        """Score one document. Raises whatever the detector raises.

        Callers that want `None` on failure use `probability()`.
        """
        if self.detector is None:
            raise RuntimeError(self.unavailable_reason() or "no yardstick")
        if self.remote:
            cached = getattr(self.detector, "cached", None)
            hit = bool(cached(text)) if callable(cached) else False
            result = self.detector.score(text)
            with self._lock:
                self.calls += 1
                self.words_scored += len(text.split())
                if hit:
                    self.cache_hits += 1
            return result
        return self.detector.score(text)

    def probability(self, text: str) -> Optional[float]:
        """AI probability or `None` when the detector cannot or will not run.

        `None` covers: no detector, detector unavailable, empty text, GPTZero's
        250-character minimum, network or quota errors. The pipeline treats
        `None` as "unscored", which is honest, rather than as 0.5.
        """
        if not text or not text.strip() or not self.available():
            return None
        try:
            value = float(self.score(text).ai_probability)
        except Exception:  # noqa: BLE001 - a failed score is a null, not a 500
            return None
        if value != value:  # NaN
            return None
        return value

    def result(self, text: str) -> Optional[DetectorResult]:
        """The full `DetectorResult`, or `None` on any failure."""
        if not text or not text.strip() or not self.available():
            return None
        try:
            return self.score(text)
        except Exception:  # noqa: BLE001
            return None

    def label(self, text: str, threshold: float = 0.5) -> Optional[str]:
        """`"human"` / `"ai"` / `"mixed"` / `None`. The detector's own label
        when it gives one (GPTZero does), else the threshold on probability."""
        res = self.result(text)
        if res is None:
            return None
        if res.label:
            return res.label
        return "ai" if res.ai_probability >= threshold else "human"

    # -- reporting -----------------------------------------------------------

    def estimated_cost_usd(self) -> Optional[float]:
        if not self.remote:
            return None
        from .gptzero import COST_PER_WORD_USD

        return round(self.words_scored * COST_PER_WORD_USD, 4)

    def describe(self) -> Dict[str, Any]:
        """The block the health endpoints and `summary["yardstick"]` carry."""
        out: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "remote": self.remote,
            "available": self.available(),
            "reason": self.unavailable_reason(),
            "role": (
                "the reported verdict. `before`/`after` and `verdict_flipped` "
                "come from here and nowhere else."
            ),
        }
        if self.kind == "local-proxy":
            out["note"] = (
                "This is NOT GPTZero. No GPTZERO_API_KEY is configured, so a "
                "local published checkpoint stands in. research/08 section 2.2: "
                "its probability carries no information about GPTZero's verdict "
                "until calibrated against real GPTZero labels. Set "
                "GPTZERO_API_KEY to make the yardstick GPTZero."
            )
        if self.remote:
            out.update(
                {
                    "calls": self.calls,
                    "words_scored": self.words_scored,
                    "cache_hits": self.cache_hits,
                    "estimated_cost_usd": self.estimated_cost_usd(),
                    "min_chars": 250,
                }
            )
        return out


# --------------------------------------------------------------- resolution


_LOCK = threading.Lock()
_YARDSTICK: Optional[Yardstick] = None
_PROXY: Optional[Yardstick] = None


def configured_name() -> str:
    """The `HUMANIZER_YARDSTICK` value after `auto` is resolved."""
    raw = (os.environ.get(YARDSTICK_ENV) or "auto").strip().lower()
    if raw == "auto":
        return "gptzero" if os.environ.get("GPTZERO_API_KEY") else _proxy_name()
    return raw


def _proxy_name() -> str:
    raw = (os.environ.get(PROXY_ENV) or DEFAULT_PROXY).strip().lower()
    return raw if raw in LOCAL_NAMES else DEFAULT_PROXY


def _local(name: str) -> Yardstick:
    """A `Yardstick` over one published local checkpoint.

    Imports the torch-backed class lazily: `humanizer.detectors` is asserted
    torch-free at import time, and this module is imported by the health
    endpoints on machines that may not have torch at all.
    """
    from .local import PUBLISHED_DETECTORS, ModernDetector

    spec = PUBLISHED_DETECTORS.get(name) or PUBLISHED_DETECTORS[DEFAULT_PROXY]
    det = ModernDetector(
        model_name=spec["model"], head=spec.get("head"), score_sentences=False
    )
    return Yardstick(name=name, kind="local-proxy", model=spec["model"], detector=det)


def _gptzero() -> Yardstick:
    from .gptzero import GPTZeroClient

    client = GPTZeroClient()
    reason = None
    if not client.available():
        reason = (
            "HUMANIZER_YARDSTICK=gptzero but GPTZERO_API_KEY is not set. "
            "Set the key, or set HUMANIZER_YARDSTICK=auto to fall back to the "
            "local proxy (and have the response say so)."
        )
    return Yardstick(
        name="gptzero",
        kind="gptzero",
        model="GPTZero API (api.gptzero.me/v2/predict/text)",
        detector=client if reason is None else None,
        remote=True,
        reason=reason,
    )


def resolve() -> Yardstick:
    """The process-wide yardstick, built on first use."""
    global _YARDSTICK
    with _LOCK:
        if _YARDSTICK is None:
            name = configured_name()
            if name == "gptzero":
                _YARDSTICK = _gptzero()
            elif name in LOCAL_NAMES:
                _YARDSTICK = _local(name)
            else:
                # An unknown name is a configuration error worth surfacing, but
                # not worth taking the server down for: fall back to the proxy
                # and say so in `reason`.
                _YARDSTICK = _local(_proxy_name())
                _YARDSTICK.reason = None
                _YARDSTICK.name = _proxy_name()
        return _YARDSTICK


def proxy() -> Yardstick:
    """The in-loop candidate scorer. Local, free, always a published checkpoint.

    When the yardstick is itself local, this is the *same object*, so a
    pipeline that scores candidates with the proxy and the document with the
    yardstick reports numbers from one configured load, not two.
    """
    global _PROXY
    with _LOCK:
        if _PROXY is None:
            y = _YARDSTICK
            if y is not None and y.kind == "local-proxy" and y.name == _proxy_name():
                _PROXY = y
            else:
                _PROXY = _local(_proxy_name())
        return _PROXY


def describe() -> Dict[str, Any]:
    """`{"yardstick": ..., "proxy": ...}` for health endpoints."""
    y = resolve()
    p = proxy()
    return {
        "yardstick": y.describe(),
        "proxy": {
            "name": p.name,
            "model": p.model,
            "available": p.available(),
            "reason": p.unavailable_reason(),
            "role": (
                "in-loop candidate scorer. Free and local; its numbers appear "
                "on candidates as `proxy_probability` and are never the "
                "reported verdict."
            ),
            "same_as_yardstick": p is y,
        },
    }


def unavailable_reason() -> Optional[str]:
    return resolve().unavailable_reason()


def reset() -> None:
    """Forget the memoised yardstick and proxy. Tests, and after a key is set."""
    global _YARDSTICK, _PROXY
    with _LOCK:
        _YARDSTICK = None
        _PROXY = None
