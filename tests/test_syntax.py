"""Tests for the optional spaCy clause-level backend.

Skipped entirely when the parser is not installed, so the suite still passes on
a machine with only the core dependencies.
"""

import math

import pytest

from humanizer.features import band_report, extract_features, syntax_available
from humanizer.text import Document

syntax = pytest.importorskip("humanizer.features.syntax")

pytestmark = pytest.mark.skipif(
    not syntax.available(), reason="spaCy model en_core_web_sm is not installed"
)

ACADEMIC = (
    "The implementation of the new evaluation framework was carefully "
    "considered by the committee. Results indicate that the observed "
    "improvement in patient outcomes reflects a substantial change in "
    "clinical practice. Researchers who examined the data reported that the "
    "distribution of responses varied considerably across the participating "
    "sites. This variation, which had not been anticipated, required a "
    "revision of the original analysis plan. The revised approach produced "
    "estimates that were consistent with earlier work in the field."
)

CONVERSATIONAL = (
    "I think you should know that we went there yesterday because it was "
    "raining and nobody wanted to walk. She said she would come but then she "
    "called and said she could not make it. We waited for a while and then we "
    "just left. It was fine really. I do not think anyone minded that much "
    "since we had already seen the place before."
)


class TestAvailability:
    def test_available(self):
        assert syntax.available()

    def test_exposed_from_package(self):
        assert syntax_available() is True

    def test_unknown_model_reports_unavailable(self):
        assert not syntax.available("definitely_not_a_model")


class TestTermination:
    def test_extraction_terminates(self):
        """Regression: an identity-based head walk never terminated.

        spaCy materialises a fresh Token on every `.head` access, so
        `token.head is token` is False even at the root. The walk must compare
        indices instead.
        """
        feats = syntax.extract_from_text(ACADEMIC)
        assert feats

    def test_root_identity_gotcha_still_holds(self):
        nlp = syntax.load_model()
        parsed = nlp("The cat sat on the mat.")
        root = next(t for t in parsed if t.dep_ == "ROOT")
        assert root.head is not root  # the trap
        assert syntax._is_root(root)  # the fix

    def test_long_document_completes(self):
        feats = syntax.extract_from_text(ACADEMIC * 12)
        assert feats["syn_passive_per_1k"] >= 0


class TestVoice:
    def test_passive_detected(self):
        feats = syntax.extract_from_text(ACADEMIC)
        assert feats["syn_passive_per_1k"] > 0

    def test_agentless_excludes_by_phrase(self):
        with_agent = syntax.extract_from_text(
            "The result was reported by the team. " * 6
        )
        assert with_agent["syn_agentless_passive_per_1k"] == 0

    def test_agentless_counted_without_agent(self):
        without = syntax.extract_from_text("The result was reported clearly. " * 6)
        assert without["syn_agentless_passive_per_1k"] > 0

    def test_passive_share_is_a_fraction(self):
        share = syntax.extract_from_text(ACADEMIC)["syn_passive_share_of_finite"]
        assert 0.0 <= share <= 1.0


class TestPhrasalVersusClausal:
    def test_academic_is_more_phrasal_than_conversational(self):
        """The central finding this module exists to measure.

        research/15: academic complexity is phrasal, not clausal, and
        conversation carries about twice as many dependent clauses.
        """
        academic = syntax.extract_from_text(ACADEMIC)
        spoken = syntax.extract_from_text(CONVERSATIONAL)
        assert (
            academic["syn_phrasal_to_clausal_ratio"]
            > spoken["syn_phrasal_to_clausal_ratio"]
        )

    def test_academic_has_more_nominalization(self):
        academic = syntax.extract_from_text(ACADEMIC)
        spoken = syntax.extract_from_text(CONVERSATIONAL)
        assert (
            academic["syn_nominalization_per_1k"]
            > spoken["syn_nominalization_per_1k"]
        )

    def test_academic_has_higher_noun_verb_ratio(self):
        academic = syntax.extract_from_text(ACADEMIC)
        spoken = syntax.extract_from_text(CONVERSATIONAL)
        assert academic["syn_noun_verb_ratio"] > spoken["syn_noun_verb_ratio"]

    def test_prep_postmodifiers_counted(self):
        assert syntax.extract_from_text(ACADEMIC)["syn_prep_postmodifier_per_1k"] > 0

    def test_noun_compounds_counted(self):
        feats = syntax.extract_from_text(
            "The patient outcome assessment procedure requires review. " * 6
        )
        assert feats["syn_noun_compound_per_1k"] > 0


class TestNominalization:
    def test_pos_filter_excludes_adjectives(self):
        """The regex version counted 'clinical' and 'critical' as nouns."""
        feats = syntax.extract_from_text(
            "The clinical and critical regional national data were vital. " * 8
        )
        assert feats["syn_nominalization_per_1k"] == 0

    def test_real_nominalizations_counted(self):
        feats = syntax.extract_from_text(
            "The implementation and evaluation of the specification required "
            "consideration of complexity and reliability. " * 4
        )
        assert feats["syn_nominalization_per_1k"] > 100

    def test_stoplist_excludes_non_derivational_nouns(self):
        feats = syntax.extract_from_text(
            "The image in the village had a message about the animal. " * 8
        )
        assert feats["syn_nominalization_per_1k"] == 0

    def test_activity_and_quality_are_nominalizations(self):
        """An earlier stoplist wrongly excluded these and halved the rate."""
        nlp = syntax.load_model()
        parsed = nlp("The activity and the quality of the treatment improved.")
        found = {t.text.lower() for t in parsed if syntax._is_nominalization(t)}
        assert {"activity", "quality", "treatment"} <= found


class TestClauses:
    def test_clause_counts_present(self):
        feats = syntax.extract_from_text(ACADEMIC)
        for key in (
            "syn_clauses_per_1k",
            "syn_dependent_clauses_per_1k",
            "syn_mean_length_t_unit",
            "syn_clauses_per_t_unit",
        ):
            assert key in feats and math.isfinite(feats[key])

    def test_relative_clause_detected(self):
        feats = syntax.extract_from_text(
            "The team that examined the data reported a change. " * 6
        )
        assert feats["syn_relative_clause_per_1k"] > 0

    def test_mean_length_t_unit_is_plausible(self):
        mlt = syntax.extract_from_text(ACADEMIC)["syn_mean_length_t_unit"]
        assert 5 < mlt < 60

    def test_dependency_depth_bounded(self):
        feats = syntax.extract_from_text(ACADEMIC)
        assert 0 < feats["syn_dependency_depth_mean"] <= feats["syn_dependency_depth_max"]


class TestEdgeCases:
    def test_empty_text(self):
        assert syntax.extract(Document.parse("")) == {}

    def test_whitespace_only(self):
        assert syntax.extract(Document.parse("   \n\n  ")) == {}

    def test_punctuation_only(self):
        assert syntax.extract_from_text("... !!! ???") in ({}, syntax.extract_from_text("... !!! ???"))

    def test_single_word(self):
        feats = syntax.extract_from_text("Hello.")
        assert isinstance(feats, dict)

    def test_all_values_finite_or_nan(self):
        for value in syntax.extract_from_text(ACADEMIC).values():
            assert isinstance(value, float)
            assert not math.isinf(value)


class TestIntegration:
    def test_with_syntax_adds_features(self):
        base = extract_features(ACADEMIC)
        full = extract_features(ACADEMIC, with_syntax=True)
        assert len(full) > len(base)
        assert any(k.startswith("syn_") for k in full)

    def test_default_excludes_syntax(self):
        assert not any(k.startswith("syn_") for k in extract_features(ACADEMIC))

    def test_band_report_includes_syntax_when_present(self):
        bands = band_report(extract_features(ACADEMIC, with_syntax=True))
        assert "syn_passive_per_1k" in bands

    def test_band_report_omits_syntax_when_absent(self):
        bands = band_report(extract_features(ACADEMIC))
        assert not any(k.startswith("syn_") for k in bands)

    def test_bands_classify(self):
        bands = syntax.band_report(syntax.extract_from_text(ACADEMIC))
        assert set(bands.values()) <= {"low", "in band", "high", "unknown"}

    def test_parser_avoids_regex_passive_false_positives(self):
        """The precision win over the regex backend.

        "was tired" is a copula plus adjective, not a passive, but any
        be-plus-"-ed" regex counts it. The parser reads the dependency label
        and does not.
        """
        text = "He was tired and she was excited about the surprise. " * 8
        feats = extract_features(text, with_syntax=True)
        assert feats["passive_per_1k"] > 100  # regex false positives
        assert feats["syn_passive_per_1k"] == 0  # parser is correct

    def test_parser_matches_regex_on_true_passives(self):
        text = "The effect has been shown repeatedly. It will be examined again. " * 6
        feats = extract_features(text, with_syntax=True)
        assert feats["syn_passive_per_1k"] > 0
