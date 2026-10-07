"""Documents in, documents out: `humanizer.documents` and its API routes.

The module is the bridge between a file and the plain text the rewrite
pipeline reads, and back. Three things matter more than smoke coverage:

  * The `.docx` write-back keeps the source's formatting. A heading stays a
    heading, a run's bold stays bold, and the root element's namespace
    declarations survive serialisation (Word refuses a document whose
    `mc:Ignorable` names an undeclared prefix).
  * Headings are locked, by style when there is one and by shape when not,
    so the front end can keep them out of the rewrite.
  * The routes never 500 for a bad upload: 415, 413, 422, 400 and 501 each
    say what to do instead.
"""

import io
import re
import zipfile
from pathlib import Path

import pytest

from humanizer import documents as d

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"

W = d.W_NS

PROSE_1 = "In today's rapidly evolving research landscape, it is important to note that computational methods play a pivotal role."
PROSE_2 = "We surveyed 40 sites in Kerala in 2021. The audit found a 12 percent shortfall."


def word_docx(paragraphs):
    """A `.docx` the way Word writes one: many namespaces, `mc:Ignorable`,
    a `Heading1` style, two runs in the first body paragraph (one bold)."""
    body = []
    for text, style in paragraphs:
        ppr = '<w:pPr><w:pStyle w:val="%s"/></w:pPr>' % style if style else ""
        if text == PROSE_1:
            head, tail = text.split(" it is important", 1)
            runs = (
                '<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>'
                '<w:r><w:t xml:space="preserve"> it is important%s</w:t></w:r>' % (head, tail)
            )
        else:
            runs = "<w:r><w:t>%s</w:t></w:r>" % text
        body.append("<w:p>%s%s</w:p>" % (ppr, runs))
    body.append("<w:p/>")  # an empty paragraph, as Word leaves between sections
    body.append('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>')
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:w="%s" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'mc:Ignorable="w14 wpc"><w:body>%s</w:body></w:document>' % (W, "".join(body))
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="%s">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
        "</w:styles>" % W
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("[Content_Types].xml", d._CONTENT_TYPES)
        zf.writestr("_rels/.rels", d._RELS)
        zf.writestr("word/_rels/document.xml.rels", d._DOC_RELS)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/styles.xml", styles)
        zf.writestr("docProps/app.xml", "<Properties/>")
    return out.getvalue()


SOURCE = word_docx([("Introduction", "Heading1"), (PROSE_1, None), ("Short closing line", None), (PROSE_2, None)])


def document_xml(data):
    return zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml")


class TestHeadings:
    @pytest.mark.parametrize("text", ["Introduction", "Methods and Materials", "A Study of Four Rivers in Kerala"])
    def test_short_lines_without_a_full_stop_are_headings(self, text):
        assert d.looks_like_heading(text)

    @pytest.mark.parametrize(
        "text",
        [
            "This is a sentence.",
            "Is it a question?",
            "- a list item",
            "1. numbered item",
            "One two three four five six seven eight nine",
            "A line,",
            "",
        ],
    )
    def test_sentences_lists_and_long_lines_are_not(self, text):
        assert not d.looks_like_heading(text)


class TestExtract:
    def test_docx_paragraphs_in_order_with_style_and_lock(self):
        ex = d.extract("essay.docx", SOURCE)
        assert ex.kind == "docx"
        assert [p.text for p in ex.paragraphs] == ["Introduction", PROSE_1, "Short closing line", PROSE_2]
        assert [p.style for p in ex.paragraphs] == ["Heading1", None, None, None]
        # the heading by style, the short line by shape, the prose by neither
        assert [p.locked for p in ex.paragraphs] == [True, False, True, False]
        assert ex.text == "Introduction\n\n" + PROSE_1 + "\n\nShort closing line\n\n" + PROSE_2
        assert ex.n_words == len(d.words(ex.text))
        assert ex.n_pages is None
        payload = ex.as_dict()
        assert payload["n_paragraphs"] == 4 and payload["n_locked"] == 2
        assert payload["paragraphs"][0] == {"text": "Introduction", "locked": True, "style": "Heading1"}

    def test_docx_runs_are_joined_and_tracked_deletions_skipped(self):
        xml = (
            '<w:document xmlns:w="%s"><w:body><w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> </w:t></w:r>'
            "<w:del><w:r><w:delText>gone</w:delText></w:r></w:del><w:r><w:t>world.</w:t></w:r><w:r><w:tab/><w:t>Tabbed</w:t></w:r></w:p>"
            "</w:body></w:document>" % W
        )
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("word/document.xml", xml)
        assert d.docx_paragraphs(out.getvalue()) == [("Hello world. Tabbed", None)]

    def test_text_file_with_bom_and_windows_newlines(self):
        data = "\ufeffFirst paragraph.\r\n\r\nSecond paragraph.\r\n".encode("utf-8")
        ex = d.extract("notes.txt", data)
        assert ex.kind == "txt"
        assert [p.text for p in ex.paragraphs] == ["First paragraph.", "Second paragraph."]

    def test_markdown_marks_come_off_and_the_heading_locks(self):
        ex = d.extract("draft.md", b"## Methods\n\nWe used **forty** sites and `code`. See [the paper](http://x).\n\n---\n")
        assert [p.text for p in ex.paragraphs] == ["Methods", "We used forty sites and code. See the paper."]
        assert ex.paragraphs[0].locked and not ex.paragraphs[1].locked

    def test_kind_falls_back_to_the_media_type(self):
        assert d.kind_of("upload", "application/pdf") == "pdf"
        assert d.kind_of("essay.DOCX", "") == "docx"
        assert d.kind_of("photo.png", "image/png") is None

    def test_unsupported_too_large_and_empty(self):
        with pytest.raises(d.UnsupportedDocument):
            d.extract("photo.png", b"\x89PNG")
        with pytest.raises(d.DocumentTooLarge):
            d.extract("big.txt", b"x" * (d.MAX_UPLOAD_BYTES + 1))
        with pytest.raises(d.EmptyDocument):
            d.extract("empty.txt", b"")
        with pytest.raises(d.EmptyDocument):
            d.extract("blank.txt", b"   \n\n  ")

    def test_a_zip_that_is_not_a_word_file_and_a_broken_zip(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("hello.txt", "hi")
        with pytest.raises(d.DocumentError):
            d.extract("x.docx", out.getvalue())
        with pytest.raises(d.DocumentError):
            d.extract("x.docx", b"not a zip at all")


class TestPdf:
    def test_reflow_joins_lines_breaks_paragraphs_and_fixes_hyphens(self):
        page = (
            "Introduction\n"
            "The quick brown fox jumps over the lazy dog and keeps on run-\n"
            "ning until the line is long enough to be a typical line here.\n"
            "A short last line.\n"
            "The next paragraph begins here and also runs across the page\n"
            "to the typical width before it ends.\n"
            "\n"
            "After a blank line comes another one that is long enough too.\n"
            "7\n"
        )
        paras = d.reflow_lines(page)
        assert paras[0] == "Introduction"
        assert paras[1].startswith("The quick brown fox") and "running until" in paras[1]
        assert paras[1].endswith("A short last line.")
        assert paras[2].startswith("The next paragraph") and paras[2].endswith("it ends.")
        assert paras[3].startswith("After a blank line")
        assert len(paras) == 4

    def test_extract_reads_a_pdf_or_says_pdfs_are_unavailable(self):
        if not d.pdf_available():
            with pytest.raises(d.PdfSupportMissing):
                d.extract("x.pdf", b"%PDF-1.4")
            return
        pytest.importorskip("pypdf")
        from pypdf import PdfWriter

        w = PdfWriter()
        w.add_blank_page(width=300, height=300)
        buf = io.BytesIO()
        w.write(buf)
        # a page with no text is an empty document, with the scanned-PDF hint
        with pytest.raises(d.EmptyDocument) as info:
            d.extract("blank.pdf", buf.getvalue())
        assert "scanned" in str(info.value)
        with pytest.raises(d.DocumentError):
            d.extract("broken.pdf", b"%PDF-1.4 garbage")


class TestExport:
    def test_write_back_keeps_styles_runs_and_namespaces(self):
        new = ["Introduction", "Rewritten first paragraph.", "Short closing line", "Rewritten second paragraph."]
        out = d.rewrite_docx(SOURCE, new)
        assert d.docx_paragraphs(out) == [
            ("Introduction", "Heading1"),
            ("Rewritten first paragraph.", None),
            ("Short closing line", None),
            ("Rewritten second paragraph.", None),
        ]
        xml = document_xml(out)
        # the root start tag is the source's, declarations and mc:Ignorable included
        assert re.search(rb'<w:document [^>]*xmlns:wpc="[^"]+"[^>]*mc:Ignorable="w14 wpc"', xml)
        assert xml.startswith(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
        # the bold run survived with the new text; the second run is gone
        assert b"<w:rPr><w:b /></w:rPr>" in xml or b"<w:rPr><w:b/></w:rPr>" in xml
        assert xml.count(b"Rewritten first paragraph.") == 1
        # the empty paragraph and the section properties are untouched
        assert b"<w:p />" in xml or b"<w:p/>" in xml
        assert b'<w:pgSz w:w="11906"' in xml
        # every other part is copied byte for byte
        with zipfile.ZipFile(io.BytesIO(SOURCE)) as a, zipfile.ZipFile(io.BytesIO(out)) as b:
            assert a.namelist() == b.namelist()
            assert a.read("docProps/app.xml") == b.read("docProps/app.xml")

    def test_write_back_rebuilds_the_body_when_the_counts_differ(self):
        out = d.rewrite_docx(SOURCE, ["Title", "One paragraph is left."])
        assert d.docx_paragraphs(out) == [("Title", "Heading1"), ("One paragraph is left.", None)]
        xml = document_xml(out)
        assert b'<w:pgSz w:w="11906"' in xml
        assert xml.count(b"<w:p>") == 2

    def test_fresh_docx_opens_and_styles_headings(self):
        data = d.build_docx(["Introduction", "A paragraph with a line\nbreak in it.", "", "Closing."])
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert set(zf.namelist()) >= {"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml"}
        assert d.docx_paragraphs(data) == [("Introduction", "Heading1"), ("A paragraph with a line\nbreak in it.", None), ("Closing.", None)]

    def test_export_formats_and_names(self):
        body, media, ext = d.export("One.\n\n\nTwo.", "txt")
        assert (body, media, ext) == (b"One.\n\nTwo.\n", "text/plain; charset=utf-8", "txt")
        body, media, ext = d.export("Heading\n\nBody.", "md")
        assert ext == "md" and body == b"Heading\n\nBody.\n"
        body, media, ext = d.export("Heading\n\nBody.", ".DOCX")
        assert ext == "docx" and body[:2] == b"PK"
        with pytest.raises(d.UnsupportedDocument):
            d.export("x", "pdf")
        with pytest.raises(d.EmptyDocument):
            d.export("  ", "docx")
        assert d.export_filename("essay.docx", "docx") == "essay humanized.docx"
        assert d.export_filename("C:\\Users\\me\\My Essay.pdf", "docx") == "My Essay humanized.docx"
        assert d.export_filename("essay humanized.docx", "docx") == "essay humanized.docx"
        assert d.export_filename("", "txt") == "humanized.txt"
        assert d.export_filename('bad"name<>.md', "md") == "badname humanized.md"


pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api.server import create_app  # noqa: E402

DOCX_MEDIA = d.EXPORT_MEDIA["docx"]


@pytest.fixture(scope="module")
def client():
    app = create_app(reference_dir=REFERENCE_DIR, web_dir=REPO_ROOT / "does-not-exist-web", auth=False)
    with TestClient(app) as c:
        yield c


class TestRoutes:
    def test_health_names_the_kinds_and_the_ceiling(self, client):
        r = client.get("/api/documents/health")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert {"docx", "txt", "md"} <= set(body["kinds"])
        assert body["pdf"] == d.pdf_available()
        assert body["max_bytes"] == d.MAX_UPLOAD_BYTES
        assert body["export_formats"] == ["docx", "md", "txt"]
        assert isinstance(body["rewrite_max_words"], int)

    def test_extract_docx(self, client):
        r = client.post("/api/documents/extract", files={"file": ("essay.docx", SOURCE, DOCX_MEDIA)})
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "docx" and body["filename"] == "essay.docx"
        assert body["text"].startswith("Introduction\n\n")
        assert body["n_paragraphs"] == 4 and body["n_locked"] == 2
        assert body["paragraphs"][1]["locked"] is False
        assert body["n_pages"] is None and body["warnings"] == []

    def test_extract_text_without_an_extension_uses_the_media_type(self, client):
        r = client.post("/api/documents/extract", files={"file": ("upload", b"Hello there.\n\nAgain.", "text/plain")})
        assert r.status_code == 200
        assert r.json()["text"] == "Hello there.\n\nAgain."

    def test_extract_refusals_have_status_codes_and_plain_words(self, client):
        r = client.post("/api/documents/extract", files={"file": ("photo.png", b"\x89PNG", "image/png")})
        assert r.status_code == 415 and r.json()["error"] == "unsupported_type"
        r = client.post("/api/documents/extract", files={"file": ("empty.txt", b"", "text/plain")})
        assert r.status_code == 422 and r.json()["error"] == "empty_document"
        r = client.post("/api/documents/extract", files={"file": ("x.docx", b"not a zip", DOCX_MEDIA)})
        assert r.status_code == 400 and r.json()["error"] == "bad_document"
        big = b"x" * (d.MAX_UPLOAD_BYTES + 10)
        r = client.post("/api/documents/extract", files={"file": ("big.txt", big, "text/plain")})
        assert r.status_code == 413 and r.json()["error"] == "too_large"
        assert "MB" in r.json()["detail"]

    def test_export_writes_into_the_uploaded_word_file(self, client):
        text = "Introduction\n\nRewritten first paragraph.\n\nShort closing line\n\nRewritten second paragraph."
        r = client.post(
            "/api/documents/export",
            data={"text": text, "format": "docx", "filename": "essay.docx"},
            files={"source": ("essay.docx", SOURCE, DOCX_MEDIA)},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(DOCX_MEDIA)
        assert r.headers["content-disposition"].startswith('attachment; filename="essay humanized.docx"')
        assert r.headers["x-document-filename"] == "essay humanized.docx"
        assert r.headers["cache-control"] == "no-store"
        assert d.docx_paragraphs(r.content)[0] == ("Introduction", "Heading1")
        assert d.docx_paragraphs(r.content)[1] == ("Rewritten first paragraph.", None)

    def test_export_without_a_source_is_a_fresh_docx_and_txt_is_txt(self, client):
        r = client.post("/api/documents/export", data={"text": "Heading\n\nBody.", "format": "docx"})
        assert r.status_code == 200
        assert r.headers["content-disposition"].startswith('attachment; filename="humanized.docx"')
        assert d.docx_paragraphs(r.content) == [("Heading", "Heading1"), ("Body.", None)]
        r = client.post("/api/documents/export", data={"text": "Plain.", "format": "txt", "filename": "notes.txt"})
        assert r.status_code == 200
        assert r.content == b"Plain.\n"
        assert r.headers["content-type"].startswith("text/plain")
        assert "notes humanized.txt" in r.headers["content-disposition"]

    def test_export_ignores_a_source_that_is_not_a_word_file(self, client):
        r = client.post(
            "/api/documents/export",
            data={"text": "Heading\n\nBody.", "format": "docx", "filename": "essay.pdf"},
            files={"source": ("essay.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 200
        assert d.docx_paragraphs(r.content) == [("Heading", "Heading1"), ("Body.", None)]
        assert "essay humanized.docx" in r.headers["content-disposition"]

    def test_export_refusals(self, client):
        r = client.post("/api/documents/export", data={"text": "", "format": "docx"})
        assert r.status_code == 422 and r.json()["error"] == "empty_document"
        r = client.post("/api/documents/export", data={"text": "x", "format": "pdf"})
        assert r.status_code == 415 and r.json()["error"] == "unsupported_type"

    def test_unicode_download_name(self, client):
        r = client.post("/api/documents/export", data={"text": "Body.", "format": "txt", "filename": "résumé.txt"})
        assert r.status_code == 200
        cd = r.headers["content-disposition"]
        assert 'filename="rsum humanized.txt"' in cd
        assert "filename*=UTF-8''r%C3%A9sum%C3%A9%20humanized.txt" in cd
