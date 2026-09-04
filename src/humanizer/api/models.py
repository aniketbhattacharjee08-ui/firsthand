"""Request bodies and JSON-safety helpers for the HTTP API.

Two things live here that the rest of the package does not need:

1. Pydantic request models. `text` defaults to the empty string rather than
   being required, so a caller who omits it gets our own 400 with a readable
   message instead of pydantic's 422 validation blob.

2. `json_safe()`. The feature dicts legitimately contain NaN and +/-inf --
   `sent_len_cv` is NaN for a one-sentence document, `summary()` emits NaN for
   a CV whose mean is zero, and `mahalanobis()` returns NaN when no feature is
   shared with the reference. NaN and Infinity are not valid JSON (RFC 8259),
   and starlette's JSONResponse serialises with `allow_nan=False`, so an
   unsanitised report raises at render time. Every payload goes through
   `json_safe()` on the way out.
"""

from __future__ import annotations

import math
from typing import Any, Optional

try:  # pragma: no cover - exercised only when the `api` extra is absent
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    raise ImportError(
        "The HTTP API needs the `api` extra: "
        "pip install -e '.[api]'  (fastapi, uvicorn)"
    )


class AnalyzeRequest(BaseModel):
    """Body for POST /api/analyze."""

    text: str = Field(default="", description="Raw document text.")
    reference: Optional[str] = Field(
        default=None,
        description=(
            "Name of a reference distribution from GET /api/references, or a "
            "path to a distribution JSON. Null uses the server default."
        ),
    )


class FeaturesRequest(BaseModel):
    """Body for POST /api/features."""

    text: str = Field(default="", description="Raw document text.")


class PlanRequest(BaseModel):
    """Body for POST /api/plan."""

    mu: float = Field(default=0.6, description="Per-candidate success rate.")
    target: float = Field(default=0.99, description="Desired overall pass rate.")


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None and unwrap numpy scalars.

    NaN/Infinity are not representable in JSON. Dropping the keys instead of
    nulling them would be worse for the frontend: a missing `sent_len_cv` and a
    `sent_len_cv` that could not be computed look identical to a client, and
    the second case is the common one for short documents.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    # numpy scalars and 0-d arrays expose .item(); numpy arrays expose .tolist().
    if hasattr(obj, "tolist"):
        return json_safe(obj.tolist())
    if hasattr(obj, "item"):
        return json_safe(obj.item())
    return obj
