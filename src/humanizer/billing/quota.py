"""Per-IP rate limiting and the credit price of a request.

The limiter is an in-memory token bucket, which is exactly as far as the rest
of the server scales today: the LLM job store is also a process-local dict.
When the deployment grows past one process, this is the class to back with
Redis; its interface is one method.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Tuple


class RateLimiter:
    """Allow `limit` events per `window` seconds per key, refilled continuously."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit = max(1, int(limit))
        self.window = float(window)
        self._rate = self.limit / self.window  # tokens per second
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def allow(self, key: str) -> Tuple[bool, float]:
        """(allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.limit), now))
            tokens = min(float(self.limit), tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                allowed, retry = True, 0.0
            else:
                self._buckets[key] = (tokens, now)
                allowed, retry = False, (1.0 - tokens) / self._rate
            if now - self._last_prune > self.window * 2 and len(self._buckets) > 1024:
                stale = [k for k, (t, l) in self._buckets.items() if now - l > self.window * 2]
                for k in stale:
                    del self._buckets[k]
                self._last_prune = now
        return allowed, retry


def credits_for_words(n_words: int, words_per_credit: int) -> int:
    """Credits charged for an LLM rewrite of `n_words`; never less than one."""
    return max(1, int(math.ceil(n_words / float(max(1, words_per_credit)))))


def word_count(text: str) -> int:
    """Whitespace-separated tokens, the same count `llm_precheck` enforces.

    Not `humanizer.text.words`: that regex counts only Latin letters, so a
    document of digits, code or CJK would be priced as empty while the server
    happily rewrites all of it.
    """
    return len((text or "").split())


def effort_multiplier(n_candidates: Any, rounds: Any, repair_attempts: Any = 0) -> int:
    """How many times the default amount of GPU work a request asks for.

    Six candidates and one round is the baseline (1x). Sixteen candidates is
    3x, three rounds is 3x, and every three repair attempts add 1x. Without
    this, one credit could buy ten minutes of exclusive GPU.
    """

    def as_int(v: Any, default: int) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    cands = max(1, as_int(n_candidates, 6))
    rnds = max(1, as_int(rounds, 1))
    repairs = max(0, as_int(repair_attempts, 0))
    return int(math.ceil(cands / 6.0)) * rnds * (1 + repairs // 3)
