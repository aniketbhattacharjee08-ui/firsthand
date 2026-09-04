"""Segmentation tests.

Sentence splitting is load-bearing: every shape feature is computed off it, so
a splitter that breaks on "et al." would corrupt the whole feature vector.
"""

import pytest

from humanizer.text import Document, normalize, split_paragraphs, split_sentences, words


class TestSentenceSplitting:
    def test_basic(self):
        assert len(split_sentences("One. Two. Three.")) == 3

    def test_citation_with_et_al_stays_whole(self):
        sents = split_sentences("Smith et al. (2020) found an effect. It held.")
        assert len(sents) == 2
        assert sents[0].startswith("Smith et al. (2020)")

    def test_initials_stay_whole(self):
        sents = split_sentences("We follow J. R. R. Tolkien here. He was right.")
        assert len(sents) == 2

    def test_decimals_stay_whole(self):
        assert len(split_sentences("The value was 3.14 units. That is small.")) == 2

    def test_numeric_citation_stays_whole(self):
        assert len(split_sentences("As shown [12, 14]. Others disagree.")) == 2

    def test_abbreviations(self):
        for abbr in ["Dr.", "Prof.", "vs.", "Fig.", "pp.", "e.g.", "i.e."]:
            text = f"See {abbr} something here. Next sentence."
            assert len(split_sentences(text)) == 2, abbr

    def test_question_and_exclamation(self):
        assert len(split_sentences("Really? Yes! Fine.")) == 3

    def test_quoted_sentence_end(self):
        sents = split_sentences('He said "it works." Then he left.')
        assert len(sents) == 2

    def test_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_no_terminal_punctuation(self):
        assert len(split_sentences("a sentence without a period")) == 1


class TestParagraphs:
    def test_blank_line_split(self):
        assert len(split_paragraphs("One para.\n\nTwo para.")) == 2

    def test_single_newline_fallback(self):
        assert len(split_paragraphs("One line.\nTwo line.")) == 2

    def test_collapses_extra_blanks(self):
        assert len(split_paragraphs("A.\n\n\n\nB.")) == 2


class TestWords:
    def test_apostrophe_is_one_word(self):
        assert words("don't stop") == ["don't", "stop"]

    def test_excludes_numbers_and_punctuation(self):
        assert words("There were 42 items, roughly.") == [
            "There", "were", "items", "roughly",
        ]


class TestNormalize:
    def test_strips_zero_width(self):
        assert "​" not in normalize("a​b")

    def test_collapses_thin_space(self):
        # A known humanizer artifact; must not survive into the feature stage.
        assert normalize("a b") == "a b"

    def test_preserves_em_dash(self):
        # Em dashes are a measured feature, not noise.
        assert "—" in normalize("a—b")

    def test_preserves_curly_quotes(self):
        assert "“" in normalize("“quoted”")


class TestDocument:
    def test_indices_and_counts(self):
        doc = Document.parse("A one. A two.\n\nB one.")
        assert len(doc.paragraphs) == 2
        assert doc.n_sentences == 3
        assert doc.paragraph_lengths == [2, 1]
        assert [s.paragraph_index for s in doc.sentences] == [0, 0, 1]

    def test_sentence_lengths(self):
        doc = Document.parse("One two three. Four five.")
        assert doc.sentence_lengths == [3, 2]

    def test_protected_spans_cover_first_and_last(self):
        doc = Document.parse("First sentence. Middle one. Final sentence.")
        spans = doc.protected_spans()
        assert any(lo == 0 for lo, _ in spans)
        assert any(hi == len(doc.text) for _, hi in spans)

    def test_protected_spans_cover_citations(self):
        doc = Document.parse(
            "Opening line here. A claim (Smith, 2020) follows. Closing line."
        )
        text = doc.text
        pos = text.find("(Smith, 2020)")
        assert any(lo <= pos < hi for lo, hi in doc.protected_spans())

    def test_empty_document(self):
        doc = Document.parse("")
        assert doc.n_sentences == 0
        assert doc.sentence_lengths == []
