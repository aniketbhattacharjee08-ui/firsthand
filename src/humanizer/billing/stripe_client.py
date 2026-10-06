"""Stripe Checkout and webhook verification over plain HTTPS.

No Stripe SDK: `requests` is already a core dependency and the three calls
this product needs (create a Checkout Session, open a billing portal
session, verify a webhook signature) are form-encoded POSTs and an HMAC. Keeping the dependency list unchanged is the
point of this package's "no edits to existing code" constraint.

Webhook signatures follow Stripe's documented scheme: the `Stripe-Signature`
header carries `t=<unix>` and one or more `v1=<hex>`; the signed payload is
``f"{t}.{raw_body}"`` under HMAC-SHA256 with the endpoint secret, and the
timestamp must be within `tolerance` seconds.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Dict, Optional

import requests

API = "https://api.stripe.com/v1"


class StripeError(RuntimeError):
    pass


def _post(secret_key: str, path: str, form: Dict[str, str], timeout: float) -> Dict[str, Any]:
    resp = requests.post(
        API + path,
        data=form,
        auth=(secret_key, ""),
        headers={"Idempotency-Key": uuid.uuid4().hex},
        timeout=timeout,
    )
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if resp.status_code >= 400:
        msg = (payload.get("error") or {}).get("message") or resp.text[:200]
        raise StripeError("Stripe %s: %s" % (resp.status_code, msg))
    return payload


def create_checkout_session(
    secret_key: str,
    price_id: str,
    quantity: int,
    success_url: str,
    cancel_url: str,
    customer_email: str,
    metadata: Dict[str, str],
    timeout: float = 20.0,
    mode: str = "payment",
    customer: Optional[str] = None,
    subscription_metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a Checkout Session. Returns Stripe's JSON.

    `mode` is `payment` for packs and lifetime plans, `subscription` for
    monthly and yearly ones. A stored Stripe customer id is passed as
    `customer` so repeat purchases and the billing portal share one
    customer; otherwise `customer_email` prefills the form.
    `subscription_metadata` lands on the Subscription object itself, which
    is what `customer.subscription.*` events carry, so a renewal can be
    matched to a user without a second lookup.
    """
    form: Dict[str, str] = {
        "mode": mode,
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": str(int(quantity)),
        "success_url": success_url,
        "cancel_url": cancel_url,
        "client_reference_id": metadata.get("user_id", ""),
    }
    if customer:
        form["customer"] = customer
    else:
        form["customer_email"] = customer_email
    for k, v in metadata.items():
        form["metadata[%s]" % k] = str(v)
    for k, v in (subscription_metadata or {}).items():
        form["subscription_data[metadata][%s]" % k] = str(v)
    return _post(secret_key, "/checkout/sessions", form, timeout)


def create_portal_session(
    secret_key: str, customer: str, return_url: str, timeout: float = 20.0
) -> Dict[str, Any]:
    """Open a Stripe billing portal session (cancel, change card, invoices)."""
    return _post(
        secret_key, "/billing_portal/sessions", {"customer": customer, "return_url": return_url}, timeout
    )


def verify_webhook(
    secret: str, payload: bytes, signature_header: str, tolerance: float = 300.0
) -> Optional[Dict[str, Any]]:
    """The parsed event if the signature is valid and fresh, else None."""
    if not signature_header or not secret:
        return None
    ts: Optional[str] = None
    sigs = []
    for part in signature_header.split(","):
        k, _, v = part.strip().partition("=")
        if k == "t":
            ts = v
        elif k == "v1":
            sigs.append(v)
    if ts is None or not sigs:
        return None
    try:
        t = int(ts)
    except ValueError:
        return None
    if abs(time.time() - t) > tolerance:
        return None
    signed = ("%s." % ts).encode("utf-8") + payload
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in sigs):
        return None
    try:
        event = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return event if isinstance(event, dict) else None


def sign_webhook(secret: str, payload: bytes, ts: Optional[int] = None) -> str:
    """Build a `Stripe-Signature` header. For tests and the local runbook."""
    t = int(time.time()) if ts is None else int(ts)
    signed = ("%d." % t).encode("utf-8") + payload
    return "t=%d,v1=%s" % (t, hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest())
