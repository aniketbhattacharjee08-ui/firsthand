"""Document model: paragraph, sentence and token segmentation.

Pure stdlib so it runs anywhere. Sentence splitting is tuned for academic
prose, where naive splitters break on abbreviations, initials, decimals and
parenthetical citations. Getting this right matters more than it looks: every
shape feature in research/10 (sentence-length CV, autocorrelation, tail shares)
is computed off this segmentation, so a splitter that fragments "Smith et al.
(2020)" into three sentences would corrupt the whole feature vector.

Known limitations, both deliberate trade-offs:

  * A period followed by a *lowercase* continuation is treated as a boundary
    only when the token before it is longer than three characters. This keeps
    unlisted short abbreviations ("Ave.", "Rd.") intact at the cost of merging
    genuinely lowercase-continued sentences. Real edited prose capitalises
    after a full stop, so this only bites on all-lowercase input.
  * Text without terminal punctuation, such as Markdown table rows, collapses
    into one very long pseudo-sentence rather than many short ones. Run
    `humanizer.clean.clean_markup` first; `eval.analyze` does so automatically.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Sequence, Tuple

# Abbreviations that end in a period but rarely end a sentence.
_ABBREVIATIONS = {
    # Titles
    "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st", "sr", "jr",
    # Academic / citation
    "et", "al", "eds", "ed", "vol", "no", "pp", "p", "ch", "fig", "figs",
    "tbl", "eq", "ref", "refs", "cf", "viz", "ibid", "op", "cit", "trans",
    "repr", "rev", "suppl", "ser", "pt", "sec", "app",
    # Latin / general
    "e.g", "i.e", "etc", "vs", "approx", "est", "cca", "circa",
    # Units and organizations
    "inc", "ltd", "co", "corp", "dept", "univ", "assn", "bros",
    # Months and days
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept", "sep", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    # US states / directions commonly abbreviated in citations
    "u.s", "u.k", "n.y", "d.c", "ph.d", "m.a", "b.a", "m.d", "b.s", "m.s",
}

_SENT_END = re.compile(r"[.!?]+[\"'”’\)\]]*")
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’][A-Za-z]+)*")
_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")
_PARA_SPLIT = re.compile(r"\n\s*\n+")

# Spans that must never be edited or have errors injected into them
# (research/09: any error inside a quote or citation is "very high" grade cost).
_QUOTE_SPAN = re.compile(r"[\"“][^\"”]{0,600}[\"”]")
_PAREN_CITATION = re.compile(
    r"\((?:[^()]*?\b(?:19|20)\d{2}[a-z]?\b[^()]*?)\)"
)
_NUMERIC_CITATION = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")


def normalize(text: str) -> str:
    """Normalize Unicode without destroying stylistic punctuation.

    Applies NFC and collapses the confusable-space characters that humanizer
    tools inject (research/12 lists U+2009 thin space as a known artifact), but
    deliberately preserves em dashes, curly quotes and ellipses because those
    are measured features, not noise.
    """
    text = unicodedata.normalize("NFC", text)
    # Space-like characters that are not a plain space or newline.
    text = re.sub(r"[   -   　]", " ", text)
    # Zero-width and bidi controls: pure obfuscation, never stylistic.
    text = re.sub(r"[​-‏  ⁠﻿]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _is_abbreviation(text: str, period_idx: int) -> bool:
    """True if the period at `period_idx` closes a known abbreviation."""
    start = period_idx
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    token = text[start:period_idx].lower().rstrip(".")
    if not token:
        return False
    if token in _ABBREVIATIONS:
        return True
    # Single initial, e.g. "J. R. R. Tolkien".
    if len(token) == 1 and token.isalpha():
        return True
    # Dotted acronym such as "U.S.A" or "Ph.D".
    if "." in text[start:period_idx] and len(token) <= 4:
        return True
    return False


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, protecting academic-prose edge cases."""
    text = text.strip()
    if not text:
        return []

    protected: List[Tuple[int, int]] = []
    for pattern in (_PAREN_CITATION, _NUMERIC_CITATION):
        protected.extend(m.span() for m in pattern.finditer(text))

    def inside_protected(idx: int) -> bool:
        return any(lo <= idx < hi for lo, hi in protected)

    boundaries: List[int] = []
    for match in _SENT_END.finditer(text):
        end = match.end()
        first_period = match.start()

        if inside_protected(first_period):
            continue
        # Decimal number or version string: "3.14", "Section 2.1".
        if (
            text[first_period] == "."
            and first_period > 0
            and text[first_period - 1].isdigit()
            and end < len(text)
            and text[end : end + 1].isdigit()
        ):
            continue
        if text[first_period] == "." and _is_abbreviation(text, first_period):
            continue
        # A boundary must be followed by whitespace or end of text.
        rest = text[end:]
        if rest and not rest[0].isspace():
            continue
        stripped = rest.lstrip()
        # A lowercase continuation usually means an abbreviation we do not
        # know. Treat it as a boundary anyway unless the token before the
        # period is short enough to plausibly be one, since requiring an
        # uppercase continuation outright would fail on all-lowercase input.
        if stripped and not (
            stripped[0].isupper()
            or stripped[0].isdigit()
            or stripped[0] in "\"'“‘([—"
        ):
            token_start = first_period
            while token_start > 0 and text[token_start - 1].isalnum():
                token_start -= 1
            if (first_period - token_start) <= 3:
                continue
        boundaries.append(end)

    sentences: List[str] = []
    prev = 0
    for boundary in boundaries:
        chunk = text[prev:boundary].strip()
        if chunk:
            sentences.append(chunk)
        prev = boundary
    tail = text[prev:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_paragraphs(text: str) -> List[str]:
    """Split on blank lines; fall back to single newlines when there are none."""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    if len(parts) == 1 and "\n" in text:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


def words(text: str) -> List[str]:
    """Alphabetic word tokens, apostrophes preserved (so "don't" is one word)."""
    return _WORD_RE.findall(text)


def tokens(text: str) -> List[str]:
    """Words and punctuation marks as separate tokens."""
    return _TOKEN_RE.findall(text)


@dataclass
class Sentence:
    text: str
    index: int
    paragraph_index: int

    @cached_property
    def words(self) -> List[str]:
        return words(self.text)

    @property
    def length(self) -> int:
        return len(self.words)


@dataclass
class Document:
    """A segmented document. Construct with `Document.parse(text)`."""

    text: str
    paragraphs: List[str] = field(default_factory=list)
    sentences: List[Sentence] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> "Document":
        text = normalize(text)
        paragraphs = split_paragraphs(text)
        sentences: List[Sentence] = []
        idx = 0
        for p_idx, para in enumerate(paragraphs):
            for sent in split_sentences(para):
                sentences.append(Sentence(text=sent, index=idx, paragraph_index=p_idx))
                idx += 1
        return cls(text=text, paragraphs=paragraphs, sentences=sentences)

    @cached_property
    def words(self) -> List[str]:
        return words(self.text)

    @cached_property
    def lower_words(self) -> List[str]:
        return [w.lower() for w in self.words]

    @cached_property
    def tokens(self) -> List[str]:
        return tokens(self.text)

    @cached_property
    def sentence_lengths(self) -> List[int]:
        return [s.length for s in self.sentences]

    @cached_property
    def paragraph_lengths(self) -> List[int]:
        """Sentences per paragraph."""
        counts = [0] * len(self.paragraphs)
        for sent in self.sentences:
            counts[sent.paragraph_index] += 1
        return counts

    @cached_property
    def paragraph_word_lengths(self) -> List[int]:
        counts = [0] * len(self.paragraphs)
        for sent in self.sentences:
            counts[sent.paragraph_index] += sent.length
        return counts

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    def protected_spans(self) -> List[Tuple[int, int]]:
        """Character spans that must never be edited.

        research/09 rates any error inside a quotation, citation, thesis or the
        first/last sentence as "very high" grade cost, so those spans are
        off-limits to the error-injection and rewriting stages alike.
        """
        spans: List[Tuple[int, int]] = []
        for pattern in (_QUOTE_SPAN, _PAREN_CITATION, _NUMERIC_CITATION):
            spans.extend(m.span() for m in pattern.finditer(self.text))
        if self.sentences:
            for sent in (self.sentences[0], self.sentences[-1]):
                pos = self.text.find(sent.text)
                if pos >= 0:
                    spans.append((pos, pos + len(sent.text)))
        return merge_spans(spans)


def merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping character spans."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]
