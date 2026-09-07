"""Tests for the published-checkpoint engine.

Three things here are load-bearing rather than ordinary unit-test hygiene:

  * **The content gate is the whole point.** The AuthorMist paper's headline
    evasion number was reproduced here only in a configuration that dropped
    numbers and dates and pushed content overlap to 0.52. `_gate` is what
    stops that from counting as a success, so it is asserted directly, per
    failure mode, not just end to end.
  * **`available()` may never raise and may never load.** It is called from a
    health endpoint on machines with no torch and no 12GB checkpoint. A test
    that a broken environment returns `False` instead of an exception is the
    only thing keeping `/api/health` up.
  * **The notice must survive to the wire.** This engine flipped 0 of 9
    verdicts. `summary["notice"]` saying so is a correctness requirement of the
    response body, not a nicety, so it is asserted on every path -- success,
    unavailable, and empty input.

Nothing here downloads or loads the 12GB checkpoint. The generation path is
exercised through injected stubs; the real measurement lives in `.weightshunt/`
and is quoted in the module docstring.
"""

import pytest

from humanizer.humanize import pretrained as pt
from humanizer.humanize.pretrained import (
    DEFAULT_MODEL,
    KNOWN_CHECKPOINTS,
    NOTICE,
    PretrainedCandidate,
    PretrainedConfig,
    available,
    humanize_pretrained,
    unavailable_reason,
)

SOURCE = (
    "The study enrolled 240 participants between March 2019 and June 2020. "
    "Researchers at Stanford measured a 17% reduction in reported symptoms. "
    "As Chen (2021) notes, the effect persisted at twelve months."
)

#: A faithful rewrite: every number, date, citation and name survives.
FAITHFUL = (
    "Between March 2019 and June 2020 the trial signed up 240 people. "
    "A 17% drop in reported symptoms was measured by researchers at Stanford. "
    "The effect was still there at twelve months, as Chen (2021) notes."
)


class _StubDetector:
    """Returns a scripted probability per exact string, default 0.9."""

    def __init__(self, scores=None, default=0.9):
        self.scores = scores or {}
        self.default = default
        self.calls = []

    def available(self):
        return True

    def score(self, text):
        self.calls.append(text)
        value = self.scores.get(text.strip(), self.default)
        return type("S", (), {"ai_probability": value})()


def _install_stub_generator(monkeypatch, samples):
    """Make `_generate` hand back `samples` in order, no model involved."""
    box = {"i": 0}

    def fake_generate(tokenizer, model, text, temperature, cfg):
        i = box["i"]
        box["i"] += 1
        return samples[i] if i < len(samples) else ""

    monkeypatch.setattr(pt, "_generate", fake_generate)
    monkeypatch.setattr(pt, "_load", lambda cfg: ("tok", "model"))
    monkeypatch.setattr(pt, "unavailable_reason", lambda config=None: None)
    return box


# ------------------------------------------------------------ the registry


class TestKnownCheckpoints:
    """The hunt's negative results are data, and data can rot silently."""

    def test_default_is_the_evasion_trained_checkpoint(self):
        assert DEFAULT_MODEL == "authormist/authormist-originality"
        assert KNOWN_CHECKPOINTS[DEFAULT_MODEL]["trained_against_detector"]

    def test_every_entry_says_what_happened(self):
        for name, entry in KNOWN_CHECKPOINTS.items():
            assert "status" in entry, "%s has no status" % name

    def test_the_unpublished_ones_are_recorded_as_unpublished(self):
        """MASH, SICO, RAFT, HMGC and ToBlend released code, not weights.

        Recorded so the next reader does not repeat the search.
        """
        for name in ("MASH", "SICO", "RAFT", "HMGC", "ToBlend"):
            assert "no weights published" in KNOWN_CHECKPOINTS[name]["status"]

    def test_dipper_xxl_records_why_it_was_skipped(self):
        entry = KNOWN_CHECKPOINTS["kalpeshk2011/dipper-paraphraser-xxl"]
        assert entry["status"].startswith("skipped")
        assert "45GB" in entry["status"]


# ---------------------------------------------------------------- the gate


class TestGate:
    """`_gate` is the defence against calling content destruction a success."""

    def test_faithful_rewrite_passes(self):
        assert pt._gate(SOURCE, FAITHFUL, PretrainedConfig()) == []

    def test_empty_is_rejected(self):
        assert pt._gate(SOURCE, "   ", PretrainedConfig()) == ["empty"]

    def test_dropped_number_is_an_invariant_failure(self):
        broken = FAITHFUL.replace("240 people", "many people")
        assert "invariant:numbers" in pt._gate(SOURCE, broken, PretrainedConfig())

    def test_dropped_date_is_an_invariant_failure(self):
        broken = FAITHFUL.replace("March 2019 and June 2020", "the study period")
        failures = pt._gate(SOURCE, broken, PretrainedConfig())
        assert any(f.startswith("invariant:") for f in failures)

    def test_dropped_citation_is_an_invariant_failure(self):
        broken = FAITHFUL.replace("Chen (2021)", "one researcher")
        failures = pt._gate(SOURCE, broken, PretrainedConfig())
        assert "invariant:citations" in failures

    def test_a_rewrite_about_a_different_subject_is_content_drift(self):
        """The failure the previous agent was caught by: fluent, and unrelated."""
        other = (
            "Urban planners increasingly favour green corridors because they "
            "moderate street temperature, absorb rainfall and give commuters "
            "somewhere pleasant to walk during the summer months of July."
        )
        assert "content_drift" in pt._gate(SOURCE, other, PretrainedConfig())

    def test_truncation_is_too_short(self):
        failures = pt._gate(SOURCE, "The study enrolled 240 participants.",
                            PretrainedConfig())
        assert "too_short" in failures

    def test_padding_is_too_long(self):
        failures = pt._gate(SOURCE, FAITHFUL + " " + FAITHFUL, PretrainedConfig())
        assert "too_long" in failures

    def test_thresholds_match_the_prompted_pipeline(self):
        """Both engines are held to the same standard, on purpose."""
        from humanizer.humanize.pipeline import PipelineConfig

        assert (PretrainedConfig().min_content_overlap
                == PipelineConfig().min_content_overlap)


# ----------------------------------------------------------- configuration


class TestConfig:
    def test_default_prompt_style_is_the_content_preserving_one(self):
        """`card` moves the detector more but only by dropping facts."""
        assert PretrainedConfig().prompt_style == "chat"

    def test_card_prompt_is_verbatim_from_the_model_card(self):
        assert pt.CARD_PROMPT.startswith("Please paraphrase the following text")
        assert pt.CARD_PROMPT.endswith("Paraphrased text:")

    def test_temperatures_are_spread_across_candidates(self):
        temps = pt._temperatures(PretrainedConfig(n_candidates=4))
        assert len(temps) == 4
        assert len(set(temps)) == 4
        assert temps == sorted(temps)

    def test_single_candidate_uses_the_configured_temperature(self):
        cfg = PretrainedConfig(n_candidates=1, temperature=0.42)
        assert pt._temperatures(cfg) == [0.42]

    def test_resolved_device_is_a_real_string(self):
        assert PretrainedConfig().resolved_device() in ("mps", "cuda", "cpu")

    def test_explicit_device_wins(self):
        assert PretrainedConfig(device="cpu").resolved_device() == "cpu"


# ------------------------------------------------------------ availability


class TestAvailability:
    def test_available_never_raises(self, monkeypatch):
        def boom(config=None):
            raise RuntimeError("hub is on fire")

        monkeypatch.setattr(pt, "unavailable_reason", boom)
        assert available() is False

    def test_missing_torch_is_a_readable_sentence(self, monkeypatch):
        monkeypatch.setattr(pt, "_torch_unavailable_reason",
                            lambda: "no torch here")
        assert unavailable_reason() == "no torch here"

    def test_unavailable_reason_names_the_fallback(self, monkeypatch):
        monkeypatch.setattr(
            pt, "_torch_unavailable_reason",
            lambda: "The pretrained-checkpoint engine needs torch and "
                    "transformers. The rule-based engine at "
                    "POST /api/humanize still works.")
        assert "/api/humanize" in unavailable_reason()

    def test_unavailable_engine_returns_the_source_unchanged(self, monkeypatch):
        monkeypatch.setattr(pt, "unavailable_reason",
                            lambda config=None: "weights not downloaded")
        result = humanize_pretrained(SOURCE)
        assert result.humanized == SOURCE
        assert result.changed is False
        assert result.summary["available"] is False
        assert result.summary["reason"] == "weights not downloaded"
        assert result.summary["notice"] == NOTICE


# --------------------------------------------------------------- rewriting


class TestHumanizePretrained:
    def test_empty_input_short_circuits(self):
        result = humanize_pretrained("   ")
        assert result.humanized.strip() == ""
        assert result.summary["notice"] == NOTICE

    def test_picks_the_lowest_scoring_candidate_that_passes(self, monkeypatch):
        good_low = FAITHFUL
        good_high = FAITHFUL.replace("signed up", "recruited")
        _install_stub_generator(monkeypatch, [good_high, good_low])
        detector = _StubDetector({good_high: 0.8, good_low: 0.3, SOURCE: 0.99})

        result = humanize_pretrained(
            SOURCE, PretrainedConfig(n_candidates=2), detector=detector)

        assert result.humanized == good_low
        assert result.summary["n_candidates_passed"] == 2
        assert result.summary["verdict_flipped"] is True

    def test_a_gate_failing_candidate_never_wins_on_score(self, monkeypatch):
        """The core rule: a lower detector score does not buy a dropped fact."""
        cheating = "Some people were studied and things improved a lot overall."
        _install_stub_generator(monkeypatch, [cheating, FAITHFUL])
        detector = _StubDetector({cheating: 0.01, FAITHFUL: 0.85, SOURCE: 0.99})

        result = humanize_pretrained(
            SOURCE, PretrainedConfig(n_candidates=2), detector=detector)

        assert result.humanized == FAITHFUL
        assert result.summary["n_candidates_passed"] == 1
        assert result.summary["all_gates_failed"] is False

    def test_all_gates_failing_is_reported_not_hidden(self, monkeypatch):
        junk = "Things happened."
        _install_stub_generator(monkeypatch, [junk, junk])
        detector = _StubDetector({junk: 0.1, SOURCE: 0.99})

        result = humanize_pretrained(
            SOURCE, PretrainedConfig(n_candidates=2), detector=detector)

        assert result.summary["all_gates_failed"] is True
        assert result.summary["gate_failures"]

    def test_dropped_samples_do_not_fail_the_request(self, monkeypatch):
        """MPS hands back NaN occasionally; `_generate` returns "" for that."""
        _install_stub_generator(monkeypatch, ["", FAITHFUL, ""])
        detector = _StubDetector({FAITHFUL: 0.4, SOURCE: 0.99})

        result = humanize_pretrained(
            SOURCE, PretrainedConfig(n_candidates=3), detector=detector)

        assert result.humanized == FAITHFUL
        assert result.summary["n_candidates_generated"] == 1
        assert result.summary["n_candidates_total"] == 3

    def test_every_sample_dropped_returns_the_source(self, monkeypatch):
        _install_stub_generator(monkeypatch, ["", "", ""])
        detector = _StubDetector({SOURCE: 0.99})

        result = humanize_pretrained(
            SOURCE, PretrainedConfig(n_candidates=3), detector=detector)

        assert result.humanized == SOURCE
        assert result.changed is False
        assert result.summary["n_candidates_generated"] == 0

    def test_verdict_flipped_is_false_when_it_did_not_flip(self, monkeypatch):
        _install_stub_generator(monkeypatch, [FAITHFUL])
        detector = _StubDetector({FAITHFUL: 0.97, SOURCE: 0.99})

        result = humanize_pretrained(
            SOURCE, PretrainedConfig(n_candidates=1), detector=detector)

        assert result.summary["verdict_flipped"] is False


# ------------------------------------------------------------- the payload


class TestResponseShape:
    """The API layer treats this like `LlmHumanizeResult`, so the keys matter."""

    def _result(self, monkeypatch):
        _install_stub_generator(monkeypatch, [FAITHFUL])
        detector = _StubDetector({FAITHFUL: 0.4, SOURCE: 0.99})
        return humanize_pretrained(
            SOURCE, PretrainedConfig(n_candidates=1), detector=detector)

    def test_as_dict_carries_the_pipeline_keys(self, monkeypatch):
        body = self._result(monkeypatch).as_dict()
        for key in ("original", "humanized", "model", "backend",
                    "before", "after", "summary", "candidates"):
            assert key in body, key

    def test_before_and_after_carry_probability_and_label(self, monkeypatch):
        body = self._result(monkeypatch).as_dict()
        assert body["before"]["ai_probability"] == pytest.approx(0.99)
        assert body["before"]["label"] == "ai"
        assert body["after"]["ai_probability"] == pytest.approx(0.4)
        assert body["after"]["label"] == "human"

    def test_the_notice_reaches_the_wire(self, monkeypatch):
        """0 of 9 flips. A UI that hides this would be misleading the user."""
        summary = self._result(monkeypatch).as_dict()["summary"]
        assert summary["notice"] == NOTICE
        assert "0 of 9" in summary["notice"]

    def test_candidates_are_json_shaped(self, monkeypatch):
        candidates = self._result(monkeypatch).as_dict()["candidates"]
        assert len(candidates) == 1
        row = candidates[0]
        for key in ("index", "text", "ai_probability", "overlap",
                    "gate_failures", "passed", "chosen"):
            assert key in row, key
        assert row["chosen"] is True

    def test_candidate_text_can_be_withheld(self):
        candidate = PretrainedCandidate(index=0, text="x", temperature=0.7)
        assert "text" not in candidate.as_dict(with_text=False)

    def test_result_is_json_serialisable(self, monkeypatch):
        import json

        from humanizer.api.models import json_safe

        json.dumps(json_safe(self._result(monkeypatch).as_dict()))


class TestModelCache:
    def test_clear_cache_empties_loaded_models(self):
        pt._MODEL_CACHE[("m", "cpu", "float16")] = ("tok", "model")
        assert pt.loaded_models() == ["m"]
        pt.clear_model_cache()
        assert pt.loaded_models() == []


class TestReachabilityMemo:
    """`available()` is a health-endpoint call; it must not hit the Hub twice."""

    def test_verdict_is_memoised(self, monkeypatch):
        pt.clear_model_cache()
        calls = []

        def fake_snapshot(model, **kwargs):
            calls.append(kwargs.get("local_files_only", False))
            return "/tmp/%s" % model

        monkeypatch.setattr(pt, "_torch_unavailable_reason", lambda: None)
        monkeypatch.setitem(
            __import__("sys").modules, "huggingface_hub",
            type("M", (), {"snapshot_download": staticmethod(fake_snapshot)}))

        assert unavailable_reason() is None
        assert unavailable_reason() is None
        assert len(calls) == 1, "the Hub was queried twice"
        pt.clear_model_cache()

    def test_cached_weights_are_checked_before_the_network(self, monkeypatch):
        """A box with the 12GB checkpoint keeps working with the network down."""
        pt.clear_model_cache()
        seen = []

        def fake_snapshot(model, **kwargs):
            local = kwargs.get("local_files_only", False)
            seen.append(local)
            if not local:
                raise OSError("no network")
            return "/tmp/x"

        monkeypatch.setattr(pt, "_torch_unavailable_reason", lambda: None)
        monkeypatch.setitem(
            __import__("sys").modules, "huggingface_hub",
            type("M", (), {"snapshot_download": staticmethod(fake_snapshot)}))

        assert unavailable_reason() is None
        assert seen == [True]
        pt.clear_model_cache()

    def test_unreachable_checkpoint_is_a_readable_sentence(self, monkeypatch):
        pt.clear_model_cache()

        def fake_snapshot(model, **kwargs):
            raise OSError("nope")

        monkeypatch.setattr(pt, "_torch_unavailable_reason", lambda: None)
        monkeypatch.setitem(
            __import__("sys").modules, "huggingface_hub",
            type("M", (), {"snapshot_download": staticmethod(fake_snapshot)}))

        reason = unavailable_reason()
        assert reason is not None
        assert DEFAULT_MODEL in reason
        assert "/api/humanize" in reason
        assert available() is False
        pt.clear_model_cache()


class TestOverGenerationStops:
    """The one "verdict flip" this engine produced was a triple-paraphrase.

    In completion mode the checkpoint runs past its own answer and restates the
    paragraph two or three times, narrating between attempts. The result is
    incoherent, three times the source length, and desklib scores incoherent
    text as human -- which is how a rambling failure masquerades as evasion.
    Truncating at the stage direction is what keeps that out of the product.
    """

    def test_stage_directions_are_stop_strings(self):
        for tell in ("Paraphrase with", "Let me refine"):
            assert tell in pt.STOP_STRINGS

    def test_a_triple_paraphrase_is_truncated_to_the_first(self):
        from humanizer.humanize.llm import clean_completion

        rambling = (
            FAITHFUL
            + " Paraphrase with less formal writing and friendly approach: "
            + FAITHFUL
            + " Let me refine the passage using more human friendly language: "
            + FAITHFUL
        )
        assert clean_completion(rambling, pt.STOP_STRINGS).strip() == FAITHFUL

    def test_over_generation_fails_the_length_gate_even_if_stops_miss(self):
        """Belt and braces: the gate catches it when the stop list does not."""
        tripled = " ".join([FAITHFUL] * 3)
        assert "too_long" in pt._gate(SOURCE, tripled, PretrainedConfig())
