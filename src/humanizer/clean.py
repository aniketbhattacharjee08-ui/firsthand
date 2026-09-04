"""Strip markup before feature extraction.

Feature extraction assumes running prose. Markdown tables, headings, code
fences and list bullets each parse as very short "sentences", which inflates
sentence-length CV and the short-sentence share far outside any human band.
The smoke test on a research report produced a CV of 1.95 against a human band
of 0.42-0.60 purely from table rows, so this step is not optional.
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)
_HR = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_FOOTNOTE = re.compile(r"\[\^[^\]]+\]")
_BLANKS = re.compile(r"\n{3,}")


def clean_markup(text: str, drop_tables: bool = True) -> str:
    """Remove Markdown and HTML markup, keeping prose and its paragraphing."""
    text = _CODE_FENCE.sub("\n\n", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _FOOTNOTE.sub("", text)
    if drop_tables:
        text = _TABLE_SEP.sub("", text)
        text = _TABLE_ROW.sub("", text)
    text = _HEADING.sub("", text)
    text = _HR.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _HTML_TAG.sub("", text)
    text = _BARE_URL.sub("", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def looks_like_markup(text: str) -> bool:
    """Heuristic: is this document markup-heavy enough to need cleaning?"""
    if not text.strip():
        return False
    lines = text.splitlines()
    if not lines:
        return False
    marked = sum(
        1
        for line in lines
        if _TABLE_ROW.match(line)
        or _HEADING.match(line)
        or _BULLET.match(line)
        or line.strip().startswith("```")
    )
    return marked / len(lines) > 0.15
