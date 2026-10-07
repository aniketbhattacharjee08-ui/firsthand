"""Documents in, documents out.

The humanizer works on plain text. This module is the bridge for people who
have a file rather than a paragraph in the clipboard: it pulls the text out of
a `.docx`, `.pdf`, `.txt` or `.md` upload, and it writes a rewrite back out as
a document again.

Reading
-------
`extract(filename, data)` returns an `Extracted`: the text with one blank
line between paragraphs (the shape `humanizer.text.split_paragraphs` and the
rewrite pipeline expect), the paragraphs themselves with a `locked` flag for
the ones that look like headings, a word count and, for a PDF, the page
count. Word files are read with the standard library (`zipfile` plus
`xml.etree`), so the API image carries nothing new for them. PDFs need the
`docs` extra (`pypdf`); without it `extract` raises `PdfSupportMissing` and
the route says so in plain words rather than failing.

Writing
-------
`export(text, fmt, source=...)` returns the bytes of a document holding
`text`. For `fmt="docx"` with the original `.docx` bytes as `source`, the
rewrite is written *into* that file: each paragraph keeps its paragraph
properties (style, alignment, spacing, numbering) and the formatting of its
first run, and only the words change. That mapping holds when the rewrite
has as many paragraphs as the original; when it does not (the pipeline
drops an empty paragraph, or the user edited the draft), the body is
rebuilt with plain paragraphs inside the same package, so the document's
styles, fonts, page size and margins still come from the source. Without a
source the result is a small fresh `.docx` with Normal and Heading 1 styles.

Why ElementTree and not a `.docx` library: the write-back touches only
`word/document.xml`; every other part is copied byte for byte. ElementTree
would drop the root's unused namespace declarations on serialisation, and
Word rejects a document whose `mc:Ignorable` names a prefix that is no
longer declared, so `_serialise_document` keeps the original root start tag.

Headings
--------
`looks_like_heading(text)` is deliberately conservative: eight words or
fewer, no sentence-ending punctuation, not a list fragment. The front end
keeps such paragraphs out of the rewrite request (rewriting "Introduction"
is never an improvement), and the fresh `.docx` writer styles them as
Heading 1. In a `.docx` source the style wins: a paragraph whose style id
starts with "Heading" or "Title" is locked whatever its length.
"""

from __future__ import annotations

import io
import posixpath
import re
import statistics
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape as _xml_escape

from .text import split_paragraphs, words

#: Largest upload the extract route accepts. A 40-page essay with a few
#: figures is well under a megabyte; this is a ceiling against accidents,
#: not a budget. Kept under the edge proxy's own body limit.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

#: Most PDF pages read in one upload. The rewrite pipeline caps the words
#: anyway; this stops a scanned textbook from tying the CPU up for nothing.
MAX_PDF_PAGES = 80

#: Extension to kind. The kind is what the API reports and what `export`
#: takes as `fmt`.
KINDS: Dict[str, str] = {
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
    ".docx": "docx",
    ".pdf": "pdf",
}

#: Media types browsers send for the same files, for an upload whose name
#: carries no extension.
MEDIA_KINDS: Dict[str, str] = {
    "text/plain": "txt",
    "text/markdown": "md",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/pdf": "pdf",
}

EXPORT_MEDIA: Dict[str, str] = {
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _w(tag: str) -> str:
    return "{%s}%s" % (W_NS, tag)


class DocumentError(ValueError):
    """Base class; `code` is the API's `error` field, the message its detail."""

    code = "bad_document"
    status = 400


class UnsupportedDocument(DocumentError):
    code = "unsupported_type"
    status = 415


class DocumentTooLarge(DocumentError):
    code = "too_large"
    status = 413


class EmptyDocument(DocumentError):
    code = "empty_document"
    status = 422


class PdfSupportMissing(DocumentError):
    code = "pdf_unavailable"
    status = 501


@dataclass
class Paragraph:
    text: str
    #: True for headings and other lines the rewrite should leave alone.
    locked: bool = False
    #: The source style id for a `.docx` paragraph ("Heading1", "Normal"),
    #: else None.
    style: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {"text": self.text, "locked": self.locked, "style": self.style}


@dataclass
class Extracted:
    kind: str
    filename: str
    paragraphs: List[Paragraph]
    n_pages: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def n_words(self) -> int:
        return len(words(self.text))

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "filename": self.filename,
            "text": self.text,
            "paragraphs": [p.as_dict() for p in self.paragraphs],
            "n_paragraphs": len(self.paragraphs),
            "n_locked": sum(1 for p in self.paragraphs if p.locked),
            "n_words": self.n_words,
            "n_pages": self.n_pages,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

_TERMINAL = re.compile(r"[.!?;:,]['\"”’)\]]*$")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-zA-Z][.)])\s+")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")


def looks_like_heading(text: str, max_words: int = 8) -> bool:
    """A short line with no sentence-ending punctuation reads as a heading."""
    t = text.strip()
    if not t or "\n" in t:
        return False
    if _TERMINAL.search(t):
        return False
    if _LIST_MARK.match(t):
        return False
    n = len(words(t))
    return 0 < n <= max_words


def _style_locks(style_id: Optional[str]) -> bool:
    if not style_id:
        return False
    s = style_id.lower()
    return s.startswith("heading") or s.startswith("title") or s.startswith("subtitle")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def kind_of(filename: str, content_type: str = "") -> Optional[str]:
    """The document kind for a filename, else for its media type, else None."""
    ext = posixpath.splitext((filename or "").strip().lower())[1]
    if ext in KINDS:
        return KINDS[ext]
    ct = (content_type or "").split(";")[0].strip().lower()
    return MEDIA_KINDS.get(ct)


def extract(filename: str, data: bytes, content_type: str = "") -> Extracted:
    """Pull the text out of an uploaded document. Raises `DocumentError`."""
    name = (filename or "").strip() or "document"
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentTooLarge(
            "That file is %.1f MB; the largest accepted is %d MB."
            % (len(data) / (1024.0 * 1024.0), MAX_UPLOAD_BYTES // (1024 * 1024))
        )
    kind = kind_of(name, content_type)
    if kind is None:
        raise UnsupportedDocument(
            "Only .docx, .pdf, .txt and .md files are read. Save the document as one of those and try again."
        )
    if not data:
        raise EmptyDocument("That file is empty.")

    warnings: List[str] = []
    n_pages: Optional[int] = None
    if kind == "docx":
        paragraphs = [Paragraph(t, locked=_style_locks(s) or looks_like_heading(t), style=s) for t, s in docx_paragraphs(data)]
    elif kind == "pdf":
        pdf_paras, n_pages, pdf_warnings = pdf_paragraphs(data)
        warnings.extend(pdf_warnings)
        paragraphs = [Paragraph(t, locked=looks_like_heading(t)) for t in pdf_paras]
    else:
        text = decode_text(data)
        if kind == "md":
            text = strip_markdown(text)
        paragraphs = [Paragraph(t, locked=looks_like_heading(t)) for t in split_paragraphs(text)]

    paragraphs = [p for p in paragraphs if p.text.strip()]
    if not paragraphs:
        raise EmptyDocument(
            "No text was found in that file."
            + (" A scanned PDF holds pictures of pages, not words; this reads only real text." if kind == "pdf" else "")
        )
    return Extracted(kind=kind, filename=name, paragraphs=paragraphs, n_pages=n_pages, warnings=warnings)


def decode_text(data: bytes) -> str:
    """UTF-8 (with or without a BOM), else UTF-16 with a BOM, else cp1252."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return _clean_text(data.decode("utf-16"))
        except UnicodeDecodeError:
            pass
    try:
        return _clean_text(data.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return _clean_text(data.decode("cp1252", errors="replace"))


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = text.replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_markdown(text: str) -> str:
    """Take the marks off headings and emphasis; leave the words alone.

    Lists keep their markers (the heading heuristic skips them) and links
    keep their text, not their address.
    """
    out: List[str] = []
    for line in text.split("\n"):
        line = _MD_HEADING.sub("", line)
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
        line = re.sub(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)", r"\2", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        if re.match(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$", line):
            line = ""
        out.append(line)
    return _clean_text("\n".join(out))


# -- .docx ------------------------------------------------------------------


def _read_docx_part(data: bytes, part: str) -> Optional[bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            try:
                return zf.read(part)
            except KeyError:
                return None
    except zipfile.BadZipFile:
        raise DocumentError("That .docx file could not be opened. Open it in Word and save it again, or paste the text.")


def _docx_document_xml(data: bytes) -> bytes:
    xml = _read_docx_part(data, "word/document.xml")
    if xml is None:
        raise DocumentError("That file is not a Word document (it has no word/document.xml inside).")
    return xml


def _paragraph_text(p: ET.Element) -> str:
    """The text of a `w:p`: runs' `w:t`, tabs as tabs, breaks as newlines.

    Deleted text (`w:delText`, tracked changes) is not text the reader sees
    and is skipped. Field codes (`w:instrText`) likewise.
    """
    parts: List[str] = []
    for node in p.iter():
        if node.tag == _w("t"):
            parts.append(node.text or "")
        elif node.tag == _w("tab"):
            parts.append("\t")
        elif node.tag in (_w("br"), _w("cr")):
            parts.append("\n")
    text = "".join(parts).replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _paragraph_style(p: ET.Element) -> Optional[str]:
    ppr = p.find(_w("pPr"))
    if ppr is None:
        return None
    ps = ppr.find(_w("pStyle"))
    if ps is None:
        return None
    return ps.get(_w("val"))


def _body_paragraphs(root: ET.Element) -> List[ET.Element]:
    """Every `w:p` in the body in document order, tables and text boxes included."""
    body = root.find(_w("body"))
    if body is None:
        raise DocumentError("That Word document has no body.")
    return [p for p in body.iter(_w("p"))]


def docx_paragraphs(data: bytes) -> List[Tuple[str, Optional[str]]]:
    """(text, style id) for each non-empty paragraph of a `.docx`, in order."""
    xml = _docx_document_xml(data)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        raise DocumentError("That Word document's XML could not be read.")
    out: List[Tuple[str, Optional[str]]] = []
    for p in _body_paragraphs(root):
        text = _paragraph_text(p)
        if text:
            out.append((text, _paragraph_style(p)))
    return out


# -- .pdf -------------------------------------------------------------------


def pdf_available() -> bool:
    try:
        import pypdf  # noqa: F401
    except ImportError:
        return False
    return True


def pdf_paragraphs(data: bytes) -> Tuple[List[str], int, List[str]]:
    """Paragraphs, page count and warnings for a PDF.

    PDF text has no paragraphs, only positioned lines. `reflow_lines` puts
    the paragraphs back with the usual heuristics: a blank line is a break,
    a line that ends short of the column with a full stop is a break, a
    hyphen at a line end is a hyphenated word. The warning says so.
    """
    try:
        import pypdf
    except ImportError:
        raise PdfSupportMissing(
            "This server cannot read PDFs yet. Save the document as .docx or .txt, or paste the text."
        )
    warnings: List[str] = []
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # pragma: no cover - depends on pypdf's crypto extras
                raise DocumentError("That PDF is password protected. Remove the password and try again.")
        n_pages = len(reader.pages)
        pages = reader.pages[:MAX_PDF_PAGES]
        texts = []
        for page in pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("That PDF could not be read. Export it again from the program that made it, or paste the text.")
    if n_pages > MAX_PDF_PAGES:
        warnings.append("Only the first %d of %d pages were read." % (MAX_PDF_PAGES, n_pages))
    paragraphs: List[str] = []
    for t in texts:
        paragraphs.extend(reflow_lines(t))
    if paragraphs:
        warnings.append("The text came out of a PDF, so paragraph breaks are a best guess. Check them before you rewrite.")
    return paragraphs, n_pages, warnings


_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)
_SENTENCE_END = re.compile(r"[.!?]['\"”’)\]]*$")


def reflow_lines(page_text: str) -> List[str]:
    """Join a page's lines back into paragraphs."""
    lines = [l.rstrip() for l in page_text.replace("\r", "").split("\n")]
    lines = [l for l in lines if not _PAGE_NUMBER.match(l)]
    lengths = [len(l) for l in lines if l.strip()]
    if not lengths:
        return []
    typical = statistics.median(lengths)
    short = typical * 0.6

    paragraphs: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            text = " ".join(current)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                paragraphs.append(text)
            del current[:]

    for line in lines:
        s = line.strip()
        if not s:
            flush()
            continue
        if not current and looks_like_heading(s) and len(s) < short:
            paragraphs.append(s)
            continue
        if current and current[-1].endswith("-") and current[-1][-2:-1].isalpha() and s[:1].islower():
            current[-1] = current[-1][:-1] + s
        else:
            current.append(s)
        if len(s) < short and _SENTENCE_END.search(s):
            flush()
    flush()
    return paragraphs


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def export(text: str, fmt: str, source: Optional[bytes] = None) -> Tuple[bytes, str, str]:
    """(bytes, media type, extension) for `text` as a document of `fmt`."""
    fmt = (fmt or "").strip().lower().lstrip(".")
    if fmt == "text":
        fmt = "txt"
    if fmt not in EXPORT_MEDIA:
        raise UnsupportedDocument("Documents are written as .docx, .txt or .md.")
    paragraphs = split_paragraphs(text or "")
    if not paragraphs:
        raise EmptyDocument("There is no text to write.")
    if fmt in ("txt", "md"):
        body = "\n\n".join(paragraphs) + "\n"
        return body.encode("utf-8"), EXPORT_MEDIA[fmt], fmt
    if source:
        data = rewrite_docx(source, paragraphs)
    else:
        data = build_docx(paragraphs)
    return data, EXPORT_MEDIA["docx"], "docx"


def _text_runs(p: ET.Element) -> List[ET.Element]:
    """The runs of a paragraph that carry visible text, in order."""
    runs = []
    for node in p.iter(_w("r")):
        for child in node:
            if child.tag in (_w("t"), _w("tab"), _w("br"), _w("cr")):
                runs.append(node)
                break
    return runs


def _set_run_text(run: ET.Element, text: str) -> None:
    """Make `run` carry exactly `text`, keeping its `w:rPr`."""
    for child in list(run):
        if child.tag != _w("rPr"):
            run.remove(child)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i:
            ET.SubElement(run, _w("br"))
        t = ET.SubElement(run, _w("t"))
        t.text = line
        if line != line.strip() or "  " in line:
            t.set("{%s}space" % XML_NS, "preserve")


def _parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def _register_namespaces(xml: bytes) -> None:
    for event, (prefix, uri) in ET.iterparse(io.BytesIO(xml), events=("start-ns",)):
        if prefix and not re.match(r"^ns\d+$", prefix):
            try:
                ET.register_namespace(prefix, uri)
            except ValueError:
                pass


_ROOT_TAG = re.compile(rb"<w:document\b[^>]*>")


def _serialise_document(root: ET.Element, original: bytes) -> bytes:
    """Serialise `root` with the source's root start tag, declarations intact."""
    out = ET.tostring(root, encoding="unicode")
    m_new = re.match(r"<w:document\b[^>]*>", out)
    m_old = _ROOT_TAG.search(original)
    if m_new and m_old:
        out = m_old.group(0).decode("utf-8") + out[m_new.end():]
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n' + out).encode("utf-8")


def _heading_style_available(source: bytes) -> bool:
    styles = _read_docx_part(source, "word/styles.xml")
    return bool(styles) and b'w:styleId="Heading1"' in styles


def _new_paragraph(text: str, heading: bool) -> ET.Element:
    p = ET.Element(_w("p"))
    if heading:
        ppr = ET.SubElement(p, _w("pPr"))
        ET.SubElement(ppr, _w("pStyle")).set(_w("val"), "Heading1")
    run = ET.SubElement(p, _w("r"))
    _set_run_text(run, text)
    return p


def rewrite_docx(source: bytes, paragraphs: Sequence[str]) -> bytes:
    """`source` with its paragraph text replaced by `paragraphs`.

    One to one when the counts match (formatting kept per paragraph); else
    the body is rebuilt with plain paragraphs in the same package.
    """
    xml = _docx_document_xml(source)
    _register_namespaces(xml)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        raise DocumentError("That Word document's XML could not be read.")
    parents = _parent_map(root)
    body = root.find(_w("body"))
    if body is None:
        raise DocumentError("That Word document has no body.")

    filled = [p for p in _body_paragraphs(root) if _paragraph_text(p)]
    if len(filled) == len(paragraphs):
        for p, text in zip(filled, paragraphs):
            runs = _text_runs(p)
            if not runs:
                continue
            _set_run_text(runs[0], text)
            for run in runs[1:]:
                parent = parents.get(run)
                if parent is not None:
                    parent.remove(run)
    else:
        sect = body.find(_w("sectPr"))
        for child in list(body):
            body.remove(child)
        heading_ok = _heading_style_available(source)
        for text in paragraphs:
            body.append(_new_paragraph(text, heading_ok and looks_like_heading(text)))
        if sect is not None:
            body.append(sect)

    new_xml = _serialise_document(root, xml)
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, new_xml)
            else:
                zout.writestr(item, zin.read(item.filename))
    return out.getvalue()


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    "</Types>"
)

_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)

_DOC_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    "</Relationships>"
)

#: Normal in a 12pt serif with a 12pt space after; Heading 1 bold, 16pt.
_STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<w:styles xmlns:w="%s">'
    "<w:docDefaults><w:rPrDefault><w:rPr>"
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
    '<w:sz w:val="24"/><w:szCs w:val="24"/><w:lang w:val="en-US"/>'
    "</w:rPr></w:rPrDefault>"
    '<w:pPrDefault><w:pPr><w:spacing w:after="240" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
    "</w:docDefaults>"
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>'
    '<w:rPr><w:b/><w:bCs/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>'
    "</w:styles>"
) % W_NS


def _run_xml(text: str) -> str:
    pieces = []
    for i, line in enumerate(text.split("\n")):
        if i:
            pieces.append("<w:br/>")
        pieces.append('<w:t xml:space="preserve">%s</w:t>' % _xml_escape(line))
    return "<w:r>%s</w:r>" % "".join(pieces)


def build_docx(paragraphs: Iterable[str]) -> bytes:
    """A fresh, minimal `.docx` holding `paragraphs`; short lines as Heading 1."""
    body = []
    for text in paragraphs:
        text = text.strip()
        if not text:
            continue
        if looks_like_heading(text):
            body.append('<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>%s</w:p>' % _run_xml(text))
        else:
            body.append("<w:p>%s</w:p>" % _run_xml(text))
    # US Letter, one inch margins.
    sect = (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<w:document xmlns:w="%s"><w:body>%s%s</w:body></w:document>' % (W_NS, "".join(body), sect)
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/styles.xml", _STYLES)
    return out.getvalue()


def export_filename(original: Optional[str], ext: str) -> str:
    """`essay.docx` -> `essay humanized.docx`; nothing -> `humanized.docx`."""
    stem = posixpath.splitext(posixpath.basename((original or "").replace("\\", "/").strip()))[0]
    stem = re.sub(r"[\x00-\x1f\"\\/:*?<>|]+", "", stem).strip().rstrip(".")
    stem = re.sub(r"\s+humanized$", "", stem, flags=re.I)
    name = (stem + " humanized") if stem else "humanized"
    return "%s.%s" % (name[:120], ext)
