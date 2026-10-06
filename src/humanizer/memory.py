"""Process-memory hygiene shared by every model cache in the package.

The API server keeps detector checkpoints (`detectors.local`), paraphraser
checkpoints (`humanize.pretrained`) and MLX language models (`humanize.llm`)
resident between requests, because loading any of them costs seconds. Each of
those modules bounds its own cache; this module is the one place that knows
how to *give the memory back* once a cache decides to drop something, and how
to read the caps from the environment.

It must import on a machine with neither torch nor MLX installed, so both are
touched only when they are already in `sys.modules` -- if a library was never
imported, nothing was allocated through it and there is nothing to free.
"""

from __future__ import annotations

import gc
import os
import resource
import sys
from typing import Any


def max_resident(env_var: str, default: int) -> int:
    """The cap named by `env_var`, read at call time.

    Returns `0` for "unlimited" when the variable is zero, negative or not an
    integer-shaped string, so callers can write ``cap and len(cache) > cap``.
    """
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw.strip())
        except ValueError:
            value = default
    return value if value > 0 else 0


def release_memory(*objects: Any) -> None:
    """Drop `objects`, collect garbage and return allocator caches to the OS.

    Every step is best-effort: a missing backend, an old torch without
    `torch.mps`, or an MLX build that predates `mx.clear_cache` must never turn
    a cache eviction into an error.
    """
    del objects
    gc.collect()
    _empty_torch_caches()
    _empty_mlx_cache()


def _empty_torch_caches() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _empty_mlx_cache() -> None:
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def rss_mb() -> int:
    """Peak resident set size of this process in MB, rounded.

    `resource.getrusage` reports the high-water mark, not the instantaneous
    figure, in bytes on macOS and kilobytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(peak / (1024 * 1024))
    return round(peak / 1024)
