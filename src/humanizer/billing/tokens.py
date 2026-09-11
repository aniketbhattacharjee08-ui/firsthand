"""Signed, expiring tokens with the standard library only.

Used for magic-link login tokens and the session cookie. A token is
``base64url(json payload) . base64url(hmac-sha256)``. The payload always
carries `exp` (unix seconds) and `kind`, and `verify()` refuses a token whose
kind does not match what the caller expects, so a login link can never be
replayed as a session cookie or the other way round.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def sign(secret: str, kind: str, payload: Dict[str, Any], ttl_seconds: float) -> str:
    body = dict(payload)
    body["kind"] = kind
    body["exp"] = int(time.time() + ttl_seconds)
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).digest()
    return _b64(raw) + "." + _b64(mac)


def verify(secret: str, kind: str, token: str) -> Optional[Dict[str, Any]]:
    """The payload if the token is authentic, unexpired and of `kind`; else None."""
    if not token or "." not in token:
        return None
    head, _, sig = token.partition(".")
    try:
        raw = _unb64(head)
        expected = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64(expected), sig):
            return None
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict) or body.get("kind") != kind:
        return None
    exp = body.get("exp")
    if not isinstance(exp, (int, float)) or exp < time.time():
        return None
    return body
