"""Sentence-structure features: hand-built strings with known answers."""

import math

import pytest

from humanizer.features import band_report, extract_features
from humanizer.features import structure
from humanizer.text import Document


def _sent(n, opener="Word"):
    return opener + " " + " ".join(["is"] + ["word"] * max(n - 2, 0)) + "."


def _text(lengths, openers=None):
    openers = openers or ["Word"] * len(lengths)
    return " ".join(_sent(n, o) for n, o in zip(lengths, openers))


def extract(text):
    return structure.extract(Document.parse(text))


class TestOpeners:
    def test_repeat_share_counts_consecutive_identical_first_words(self):
        f = extract(_text([8, 8, 8, 8], ["The", "The", "Some", "The"]))
        # pairs: The-The (repeat), The-Some, Some-The -> 1 of 3
        assert f["struct_opener_repeat_share"] == pytest.approx(1 / 3)
        assert f["struct_opener_distinct_ratio"] == pytest.approx(2 / 4)

    def test_opener_classes_and_entropy(self):
        f = extract(_text([8] * 4, ["The", "We", "However", "Data"]))
        assert f["struct_opener_det_share"] == pytest.approx(0.25)
        assert f["struct_opener_pronoun_share"] == pytest.approx(0.25)
        assert f["struct_opener_adverbial_share"] == pytest.approx(0.25)
        assert f["struct_opener_content_share"] == pytest.approx(0.25)
        assert f["struct_opener_first_pos_entropy"] == pytest.approx(2.0)

    def test_the_this_it_in_share_and_paragraph_max(self):
        para1 = _text([8] * 4, ["The", "This", "It", "In"])
        para2 = _text([8] * 4, ["Data", "Results", "Cells", "Mice"])
        f = extract(para1 + "\n\n" + para2)
        assert f["struct_the_this_it_in_share"] == pytest.approx(0.5)
        assert f["struct_para_max_the_this_it_in_share"] == pytest.approx(1.0)

    def test_paragraph_max_ignores_short_paragraphs(self):
        f = extract("The end.\n\n" + _text([8] * 3, ["Data", "Cells", "Mice"]))
        assert f["struct_para_max_the_this_it_in_share"] == pytest.approx(0.0)

    def test_opener_class_ly_words_are_adverbial(self):
        assert structure.opener_class("Interestingly") == "adverbial"
        assert structure.opener_class("and") == "conj"
        assert structure.opener_class("Protein") == "content"


class TestMetronome:
    def test_run_of_three_in_band_is_counted(self):
        f = extract(_text([18, 20, 22, 5, 19, 40]))
        assert f["struct_metronome_share"] == pytest.approx(4 / 6)
        assert f["struct_metronome_run_max"] == 3
        assert f["struct_metronome_runs_3plus_per_1k"] > 0

    def test_no_run_when_lengths_alternate(self):
        f = extract(_text([18, 5, 20, 5, 22]))
        assert f["struct_metronome_run_max"] == 1
        assert f["struct_metronome_runs_3plus_per_1k"] == 0


class TestFragmentsAndRhythm:
    def test_fragment_share_flags_verbless_and_tiny_sentences(self):
        text = "Not now. The big red fox quick brown. The dog is running today."
        f = extract(text)
        assert f["struct_fragment_share"] == pytest.approx(2 / 3)

    def test_comma_rhythm(self):
        text = "One, two, three words here. No commas at all in this one. Alpha beta, gamma."
        f = extract(text)
        assert f["struct_comma_per_sentence_mean"] == pytest.approx(1.0)
        assert f["struct_comma_free_share"] == pytest.approx(1 / 3)
        # first-comma positions: 1/5 and 2/3
        assert f["struct_first_comma_position_mean"] == pytest.approx((0.2 + 2 / 3) / 2)
        assert f["struct_comma_per_sentence_cv"] > 0

    def test_length_contrast_and_short_after_long(self):
        f = extract(_text([30, 5, 12]))
        # |30-5| / 17.5
        assert f["struct_len_contrast_adjacent_max"] == pytest.approx(25 / 17.5)
        assert f["struct_short_after_long_per_1k"] > 0
        g = extract(_text([12, 13, 12]))
        assert g["struct_short_after_long_per_1k"] == 0

    def test_open_close_ratio(self):
        f = extract(_text([20, 12, 10]) + "\n\n" + "Single sentence paragraph here.")
        assert f["struct_para_open_close_len_ratio_mean"] == pytest.approx(2.0)

    def test_tricolon_and_question(self):
        text = "We measured mass, length, and width. Why does it matter? It does."
        f = extract(text)
        assert f["struct_tricolon_share"] == pytest.approx(1 / 3)
        assert f["struct_question_share"] == pytest.approx(1 / 3)


class TestIntegration:
    def test_probe_extraction_is_finite_or_nan_and_never_raises(self):
        feats = extract_features(
            "The study examines a question. It does so carefully, and at length. "
            "Researchers have shown that the effect is robust; others disagree.\n\n"
            "A second paragraph follows here. It adds detail, evidence, and commentary."
        )
        struct = {k: v for k, v in feats.items() if k.startswith("struct_")}
        assert len(struct) >= 20
        for v in struct.values():
            assert isinstance(v, float) or isinstance(v, int)
            assert math.isfinite(v) or math.isnan(v)

    def test_empty_document_yields_nothing(self):
        assert extract("") == {}

    def test_syntax_features_absent_without_spacy(self, monkeypatch):
        from humanizer.features import syntax

        monkeypatch.setattr(syntax, "available", lambda *a, **k: False)
        f = extract(_text([12, 14, 16]))
        assert not any(k.startswith("struct_sentence_type_") for k in f)
        assert "struct_dep_depth_cv" not in f
        assert "struct_metronome_share" in f

    def test_syntax_features_present_with_spacy(self):
        from humanizer.features import syntax

        if not syntax.available():
            pytest.skip("spaCy model not installed")
        f = extract(
            "Although the rain fell, we walked home. The dog barked and the cat ran. "
            "We slept. She said that she was tired because the day was long."
        )
        shares = [
            f["struct_sentence_type_%s_share" % k]
            for k in ("simple", "compound", "complex", "compound_complex")
        ]
        assert sum(shares) == pytest.approx(1.0)
        assert f["struct_initial_subordinate_share"] > 0
        assert 0 <= f["struct_dep_depth_cv"] or math.isnan(f["struct_dep_depth_cv"])

    def test_band_report_covers_every_band(self):
        report = band_report(extract_features(_text([18, 19, 20, 21, 5, 30] * 3)))
        for key in structure.HUMAN_BANDS:
            assert report[key] in ("low", "in band", "high", "unknown")
        assert report["struct_metronome_run_max"] == "in band"
