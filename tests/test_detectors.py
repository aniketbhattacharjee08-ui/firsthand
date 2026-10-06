"""Detector, calibration and best-of-N tests.

The pass-rate tests pin the figures reported in research/08, so a regression in
the Beta-binomial derivation shows up immediately.
"""

import math

import pytest

from humanizer.detectors import (
    AiStyleSignals,
    DetectorResult,
    GPTZeroClient,
    calibrate,
    estimate_cost,
    fleet_pass_rate,
    pass_rate,
    required_n,
)
from humanizer.detectors.calibration import wilson_interval
from humanizer.detectors.gptzero import _normalize_label
from humanizer.features import extract_features

AI_LIKE = (
    "In today's rapidly evolving landscape, it is important to note that this "
    "plays a pivotal role. Furthermore, the intricate tapestry of the realm "
    "underscores the importance of understanding. Moreover, we must delve into "
    "the multifaceted challenges. Additionally, robust frameworks are crucial. "
    "Consequently, stakeholders should leverage comprehensive strategies. "
) * 3

HUMAN_LIKE = (
    "The archive closed at four, so I had ninety minutes with the ledger. "
    "Most of it was unreadable. Water damage had taken the left margin of "
    "every third page, which is where the clerk wrote the dates. What "
    "survived was a column of names and a column of sums, and between them a "
    "narrow gutter where someone had later pencilled in corrections that "
    "disagree with the original entries by amounts too small to be errors of "
    "transcription and too large to be rounding. I copied out eleven of them "
    "before the bell went. The eleventh is the one that matters, though I did "
    "not know that yet, and I nearly skipped it because the ink had faded to "
    "the colour of the paper itself. It records a payment of nine shillings "
    "to a carter whose name appears nowhere else in the surviving records of "
    "the estate, on a date three weeks after the estate was supposedly sold."
) * 2


class TestDetectorResult:
    def test_human_probability_complements(self):
        r = DetectorResult(detector="t", ai_probability=0.3)
        assert r.human_probability == pytest.approx(0.7)

    def test_label_overrides_threshold(self):
        # research/13: a low probability is not the same as a human verdict.
        r = DetectorResult(detector="t", ai_probability=0.2, label="ai")
        assert not r.says_human

    def test_threshold_fallback_without_label(self):
        assert DetectorResult(detector="t", ai_probability=0.2).says_human
        assert not DetectorResult(detector="t", ai_probability=0.8).says_human


class TestAiStyleSignals:
    """The former `HeuristicDetector`, re-scoped to explanation only.

    It used to combine these signals with hand-set weights and a hand-tuned
    bias into an "ai_probability" that sat next to real detector scores in the
    same response. That arithmetic was invented in this repo, fitted to
    nothing, and is deleted. Detection now comes only from published
    checkpoints (`humanizer.detectors.local`).

    So what is tested here is that it is *not* a detector, and that the
    individual signals still measure what research/04 and research/10 say they
    measure.
    """

    def test_it_is_not_a_detector(self):
        """The policy, asserted rather than documented.

        No `score`, no `name`, no threshold, and not a `Detector`, so it
        cannot be handed to `analyze(detectors=[...])` or reached through
        `/api/detect`.
        """
        from humanizer.detectors.base import Detector

        sig = AiStyleSignals()
        assert not isinstance(sig, Detector)
        assert not issubclass(AiStyleSignals, Detector)
        for attr in ("score", "name", "threshold", "score_features"):
            assert not hasattr(sig, attr), attr

    def test_there_is_no_aggregate_to_read_as_a_score(self):
        """The deleted part: weights, bias, logistic. All of it, gone."""
        from humanizer.detectors import heuristic as mod

        for gone in ("WEIGHTS", "BIAS", "_sigmoid", "HeuristicDetector"):
            assert not hasattr(mod, gone), gone
        values = AiStyleSignals().signals_for_text(AI_LIKE)
        assert "ai_probability" not in values
        assert all(0.0 <= v <= 1.0 for v in values.values())

    def test_ai_like_prose_lights_up_more_signals_than_human_prose(self):
        """Still a useful explanation, which is why it was kept at all."""
        sig = AiStyleSignals()
        ai = sig.signals_for_text(AI_LIKE)
        human = sig.signals_for_text(HUMAN_LIKE)
        assert ai["ai_vocab_weighted"] > human["ai_vocab_weighted"]
        assert ai["formal_connective"] > human["formal_connective"]

    def test_overshot_variance_is_flagged_not_rewarded(self):
        """research/10's failure mode: burstiness is a band, not a direction.

        A signal that rewarded maximal variance would tell a user to make
        their prose weirder, which moves it *out* of the measured human band.
        """
        sig = AiStyleSignals()
        in_band = " ".join(
            ["word " * n + "." for n in (8, 22, 14, 31, 9, 19, 27, 12, 35, 16)] * 3
        )
        overshot = " ".join(["word ." if i % 2 else "word " * 70 + "." for i in range(30)])
        a = sig.signals_for_text(overshot)["cv_band_distance"]
        b = sig.signals_for_text(in_band)["cv_band_distance"]
        assert a >= b

    def test_every_signal_is_cited(self):
        """A UI showing these must be able to say where each one comes from."""
        from humanizer.detectors.heuristic import SIGNAL_SOURCES

        values = AiStyleSignals().signals(extract_features(AI_LIKE))
        assert set(values) == set(SIGNAL_SOURCES)
        assert all(SIGNAL_SOURCES[k].startswith("research/") for k in values)

    def test_present_filters_and_orders(self):
        found = AiStyleSignals().present(extract_features(AI_LIKE), floor=0.25)
        assert all(v >= 0.25 for v in found.values())
        assert list(found.values()) == sorted(found.values(), reverse=True)


class TestWilson:
    def test_full_success_has_upper_bound_one(self):
        lo, hi = wilson_interval(10, 10)
        assert hi == pytest.approx(1.0)
        assert lo < 1.0  # never certain from 10 samples

    def test_interval_narrows_with_n(self):
        narrow = wilson_interval(90, 100)
        wide = wilson_interval(9, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_total(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestCalibration:
    def test_rate_rises_with_threshold(self):
        scores = [i / 100 for i in range(100)]
        labels = [s > 0.5 for s in scores]
        cal = calibrate(scores, labels)
        rates = [p.rate for p in cal.points]
        assert rates == sorted(rates)

    def test_select_threshold_uses_lower_bound(self):
        # Three of three successes must not be enough to certify 95%.
        cal = calibrate([0.9, 0.95, 0.99], [True, True, True], min_samples=3)
        assert cal.select_threshold(0.95) is None

    def test_select_threshold_succeeds_with_evidence(self):
        scores = [0.9] * 200
        labels = [True] * 200
        cal = calibrate(scores, labels, min_samples=5)
        chosen = cal.select_threshold(0.95)
        assert chosen is not None and chosen.lower >= 0.95

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            calibrate([0.1, 0.2], [True])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calibrate([], [])


class TestBestOfN:
    def test_independent_case_matches_closed_form(self):
        assert pass_rate(0.6, 8, 0.0) == pytest.approx(1 - 0.4**8)

    def test_reproduces_research_figures(self):
        """research/08: mu=0.6, N=8 gives 99.93% at rho=0 and 87% at rho=0.5."""
        assert pass_rate(0.6, 8, 0.0) == pytest.approx(0.9993, abs=5e-4)
        assert pass_rate(0.6, 8, 0.2) == pytest.approx(0.975, abs=5e-3)
        assert pass_rate(0.6, 8, 0.5) == pytest.approx(0.87, abs=1e-2)

    def test_correlation_monotonically_hurts(self):
        rates = [pass_rate(0.6, 16, r) for r in (0.0, 0.1, 0.2, 0.3, 0.5)]
        assert rates == sorted(rates, reverse=True)

    def test_high_correlation_can_make_target_unreachable(self):
        # The power-law decay: more compute does not rescue correlated draws.
        assert required_n(0.6, 0.99, 0.5) is None

    def test_required_n_independent(self):
        assert required_n(0.6, 0.99, 0.0) == 6

    def test_rho_one_degenerates_to_single_draw(self):
        assert pass_rate(0.6, 100, 1.0) == pytest.approx(0.6)

    def test_fleet_saturates_on_impossible_documents(self):
        # 1.5% impossible documents cap the fleet rate at 98.5% for any N.
        mus = [0.0] * 15 + [0.9] * 985
        assert fleet_pass_rate(mus, 1024, 0.0) <= 0.985 + 1e-9

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            pass_rate(1.5, 4)
        with pytest.raises(ValueError):
            pass_rate(0.5, 0)


class TestGPTZeroClient:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("GPTZERO_API_KEY", raising=False)
        assert not GPTZeroClient(api_key=None, cache_dir=None).available()

    def test_score_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("GPTZERO_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            GPTZeroClient(api_key=None, cache_dir=None).score("x" * 300)

    def test_short_text_rejected(self):
        with pytest.raises(ValueError):
            GPTZeroClient(api_key="fake", cache_dir=None).score("too short")

    def test_label_normalization(self):
        assert _normalize_label("HUMAN_ONLY") == "human"
        assert _normalize_label("MIXED") == "mixed"
        assert _normalize_label("AI_ONLY") == "ai"
        assert _normalize_label("") is None

    def test_paraphrased_is_not_a_human_verdict(self):
        # research/12: "humanized" is a third detectable class.
        assert _normalize_label("AI_PARAPHRASED") == "ai"

    def test_parse_modern_schema(self):
        payload = {
            "version": "4.9b",
            "documents": [
                {
                    "class_probabilities": {"human": 0.9, "ai": 0.08, "mixed": 0.02},
                    "document_classification": "HUMAN_ONLY",
                    "confidence_category": "high",
                    "sentences": [{"generated_prob": 0.1}, {"generated_prob": 0.2}],
                }
            ],
        }
        result = GPTZeroClient.parse(payload)
        # 1 - P(human): the mixed / paraphrased class counts against a pass.
        assert result.ai_probability == pytest.approx(0.10)
        assert result.says_human
        assert len(result.sentence_scores) == 2

    def test_parse_legacy_schema(self):
        payload = {
            "documents": [
                {"completely_generated_prob": 0.77, "predicted_class": "ai"}
            ]
        }
        result = GPTZeroClient.parse(payload)
        assert result.ai_probability == pytest.approx(0.77)
        assert not result.says_human

    def test_parse_empty_payload(self):
        assert math.isnan(GPTZeroClient.parse({}).ai_probability)

    def test_cost_estimate(self):
        cost = estimate_cost(["word " * 500] * 10)
        assert cost == pytest.approx(500 * 10 * 0.00046, rel=1e-6)
