"""Feature extraction tests, anchored to the numbers in the research reports."""

import math

import numpy as np
import pytest

from humanizer.features import extract_features, band_report, canonical_order
from humanizer.features.lexical import hdd, mattr, mtld
from humanizer.features.shape import lag_autocorrelation, spectral_beta
from humanizer.text import Document


def _paragraphs(sentence_lengths, per_para=5, word="word"):
    """Build text with exactly the requested sentence lengths."""
    sents = [" ".join([word] * max(n, 1)) + "." for n in sentence_lengths]
    paras = [
        " ".join(sents[i : i + per_para]) for i in range(0, len(sents), per_para)
    ]
    return "\n\n".join(paras)


class TestShape:
    def test_sentence_lengths_exact(self):
        doc = Document.parse(_paragraphs([5, 10, 15]))
        assert doc.sentence_lengths == [5, 10, 15]

    def test_cv_of_constant_series_is_zero(self):
        feats = extract_features(_paragraphs([12] * 20))
        assert feats["sent_len_cv"] == pytest.approx(0.0, abs=1e-9)

    def test_cv_matches_numpy(self):
        lengths = [8, 22, 14, 31, 9, 19, 27, 12, 35, 16]
        feats = extract_features(_paragraphs(lengths))
        arr = np.array(lengths, dtype=float)
        assert feats["sent_len_cv"] == pytest.approx(
            arr.std(ddof=1) / arr.mean(), rel=1e-9
        )

    def test_tail_shares(self):
        # 2 of 10 under 10 words, 3 of 10 over 30.
        lengths = [5, 8, 12, 15, 20, 25, 33, 40, 45, 28]
        feats = extract_features(_paragraphs(lengths))
        assert feats["sent_short_share"] == pytest.approx(0.2)
        assert feats["sent_long_share"] == pytest.approx(0.3)

    def test_alternating_series_has_negative_lag1(self):
        # The signature of a naive burstiness rule, which research/10 shows
        # humans do NOT produce.
        alternating = [5, 35] * 12
        assert lag_autocorrelation(alternating, 1) < -0.8

    def test_human_like_series_has_near_zero_lag1(self):
        rng = np.random.default_rng(0)
        lengths = rng.normal(20, 9, 200)
        assert abs(lag_autocorrelation(lengths, 1)) < 0.2

    def test_white_noise_spectral_beta_near_zero(self):
        rng = np.random.default_rng(1)
        beta = spectral_beta(rng.normal(20, 8, 512))
        assert abs(beta) < 0.35

    def test_correlated_series_has_positive_beta(self):
        # Cumulative sums are strongly long-range correlated, so beta must rise
        # well above the white-noise case.
        rng = np.random.default_rng(2)
        series = np.cumsum(rng.normal(0, 1, 512))
        assert spectral_beta(series) > 1.0

    def test_spectral_beta_needs_enough_sentences(self):
        assert math.isnan(spectral_beta([10, 20, 30]))

    def test_bands_flag_uniform_text(self):
        feats = extract_features(_paragraphs([15] * 30))
        assert band_report(feats)["sent_len_cv"] == "low"

    def test_bands_flag_overshoot(self):
        # Overshooting is as anomalous as undershooting (research/10).
        lengths = [2, 60] * 15
        feats = extract_features(_paragraphs(lengths))
        assert band_report(feats)["sent_len_cv"] == "high"

    def test_band_accepts_human_range(self):
        rng = np.random.default_rng(3)
        lengths = np.clip(rng.normal(21, 11, 60), 3, 70).astype(int)
        feats = extract_features(_paragraphs(list(lengths)))
        assert band_report(feats)["sent_len_cv"] == "in band"


class TestLexical:
    def test_mtld_rises_with_diversity(self):
        diverse = [f"w{i}" for i in range(400)]
        repetitive = ["a", "b", "c", "d"] * 100
        assert mtld(diverse) > mtld(repetitive)

    def test_mtld_needs_length(self):
        assert math.isnan(mtld(["a", "b", "c"]))

    def test_hdd_in_unit_range(self):
        tokens = [f"w{i % 60}" for i in range(300)]
        assert 0.0 <= hdd(tokens) <= 1.0

    def test_mattr_of_all_unique_is_one(self):
        assert mattr([f"w{i}" for i in range(200)], 50) == pytest.approx(1.0)

    def test_hapax_rate(self):
        feats = extract_features(_paragraphs([300]) )
        # Single repeated word: no hapax at all.
        assert feats["hapax_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_ai_vocabulary_detected(self):
        text = (
            "This delves into the intricate tapestry of the realm. "
            "It underscores a pivotal and multifaceted landscape. " * 6
        )
        feats = extract_features(text)
        assert feats["ai_vocab_weighted_per_1k"] > 50
        assert feats["ai_vocab_hits_per_1k"] > 0

    def test_plain_text_has_low_ai_vocabulary(self):
        text = (
            "The cat sat on the mat and watched the rain fall past the window. "
            "Later it went outside to look for food near the shed. " * 6
        )
        assert extract_features(text)["ai_vocab_weighted_per_1k"] < 5

    def test_formal_connective_ratio(self):
        formal = "Furthermore, this holds. Moreover, that holds. " * 6
        informal = "But this holds. So that holds. And it holds. " * 6
        assert (
            extract_features(formal)["formal_connective_ratio"]
            > extract_features(informal)["formal_connective_ratio"]
        )


class TestPunctuation:
    def test_possessive_is_not_a_contraction(self):
        text = "In today's landscape the study's results matter greatly here. " * 6
        assert extract_features(text)["contraction_per_1k"] == pytest.approx(0.0)

    def test_real_contractions_counted(self):
        text = "It's clear we don't know and they're unsure about this. " * 6
        assert extract_features(text)["contraction_per_1k"] > 0

    def test_comma_rate(self):
        text = "One, two, three, four and five make a longer clause here. " * 10
        feats = extract_features(text)
        assert feats["comma_per_1k"] > 0

    def test_punctuation_entropy_higher_when_varied(self):
        plain = "This is a sentence. " * 30
        varied = (
            "This is a sentence; it continues: with more! Does it? "
            "Yes (indeed) — truly. " * 8
        )
        assert (
            extract_features(varied)["punct_entropy"]
            > extract_features(plain)["punct_entropy"]
        )


class TestRegister:
    def test_passive_detected(self):
        text = "The result was observed by the team in every trial run. " * 8
        assert extract_features(text)["passive_per_1k"] > 0

    def test_agentless_passive_excludes_by_phrase(self):
        feats = extract_features("The effect was measured by the team here. " * 8)
        assert feats["agentless_passive_per_1k"] < feats["passive_per_1k"]

    def test_nominalization(self):
        text = (
            "The implementation of the evaluation required consideration of "
            "the specification and its complexity. " * 6
        )
        assert extract_features(text)["nominalization_per_1k"] > 100

    def test_participial_tail(self):
        text = (
            "The model improved results, showing a clear gain across trials. "
            * 8
        )
        assert extract_features(text)["participial_tail_per_1k"] > 0

    def test_tricolon(self):
        text = "It was fast, cheap, and simple to run in every case here. " * 8
        assert extract_features(text)["tricolon_per_1k"] > 0

    def test_negative_parallelism(self):
        text = (
            "This is not just a change but also a genuine improvement here. "
            * 8
        )
        assert extract_features(text)["negative_parallel_per_1k"] > 0

    def test_opinion_markers_flagged(self):
        assert extract_features("I think this is right. " * 10)[
            "opinion_marker_count"
        ] > 0

    def test_argumentative_first_person_is_not_an_opinion_marker(self):
        # "I argue" is endorsed by style guides; "I think" is not.
        assert extract_features("I argue that this is right. " * 10)[
            "opinion_marker_count"
        ] == 0

    def test_paragraph_opener_share(self):
        text = "Furthermore, a claim here.\n\nMoreover, another claim here.\n\nA plain opener."
        feats = extract_features(text)
        assert feats["para_opener_formal_share"] == pytest.approx(2 / 3)


class TestVectorIntegrity:
    def test_canonical_order_is_stable(self):
        assert canonical_order() == canonical_order()

    def test_size_fields_excluded_from_vector(self):
        assert "n_words" not in canonical_order()

    def test_empty_text_does_not_crash(self):
        feats = extract_features("")
        assert feats["n_words"] == 0.0
        assert "sent_len_cv" not in feats

    def test_all_values_are_floats(self):
        feats = extract_features(_paragraphs([12, 20, 8, 30] * 5))
        assert all(isinstance(v, float) for v in feats.values())
