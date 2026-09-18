"""Sending the magic-link email.

`LONGHAND_SMTP_URL` selects the transport::

    smtp://user:password@smtp.example.com:587       STARTTLS
    smtps://user:password@smtp.example.com:465      implicit TLS
    (unset)                                        log the link to stdout

The unset mode is for development and for the first hours of a deployment:
the link is printed by the server process, so a single operator can log in
without a mail provider. `LONGHAND_DEV_LINKS=1` additionally returns the link
in the HTTP response, which must never be on in production.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional
from urllib.parse import unquote, urlparse

log = logging.getLogger("humanizer.billing.mail")


def send_magic_link(
    smtp_url: str, mail_from: str, to: str, link: str, minutes: int, product: str = "Firsthand"
) -> Optional[str]:
    """Deliver the login link. Returns the transport used, or raises."""
    subject = "Your %s sign-in link" % product
    body = (
        "Click to sign in to %s:\n\n%s\n\n"
        "The link works once and expires in %d minutes. If you did not ask "
        "for it, ignore this email.\n" % (product, link, minutes)
    )
    if not smtp_url:
        log.warning("magic link for %s: %s", to, link)
        return "log"

    u = urlparse(smtp_url)
    if u.scheme not in ("smtp", "smtps"):
        raise ValueError("LONGHAND_SMTP_URL must start with smtp:// or smtps://")
    host = u.hostname or "localhost"
    port = u.port or (465 if u.scheme == "smtps" else 587)
    user = unquote(u.username) if u.username else None
    password = unquote(u.password) if u.password else None

    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    ctx = ssl.create_default_context()
    if u.scheme == "smtps":
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as s:
            if user:
                s.login(user, password or "")
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls(context=ctx)
            if user:
                s.login(user, password or "")
            s.send_message(msg)
    return "smtp"
