"""Tests for the LLM humanizing pipeline.

The pipeline has a GPU in the middle of it, so almost everything here runs
against `llm.EchoBackend` and a fake detector instead. That is not a compromise
for CI's benefit: the interesting behaviour is the *ordering* (scrub after the
model, not before), the *gates* (hard rejection, not soft weighting) and the
*event protocol*, and none of those needs a model to be wrong.

Four groups are load-bearing rather than ordinary hygiene:

  * **Scrub-after-LLM ordering.** The instruct checkpoint was measured
    reintroducing the exact AI vocabulary the deterministic pass removes, so
    scrubbing the input instead of the output silently throws the whole pass
    away. `test_scrub_runs_after_the_model` asserts the backend saw unscrubbed
    text *and* that the winner came out scrubbed.
  * **Gates reject, they do not down-weight.** A candidate that mangles a
    number or drops a third of the paragraph must be unusable, not merely
    ranked lower. Both failures are asserted at the gate and end to end.
  * **The event protocol.** The frontend renders a checklist from these stage
    names and a circular fill from `progress`, so stage order, monotonicity and
    the "exactly one result frame" rule are contract tests.
  * **Degradation without MLX.** This package is expected to run on machines
    with no Apple Silicon. Every LLM endpoint must answer 503 with the
    rule-based fallback named, never 500.

MLX-dependent tests are marked `requires_mlx` and skip cleanly when `mlx_lm`
cannot be imported.
"""

from __future__ import annotations

import json
import math
import re
from types import SimpleNamespace

import pytest

from humanizer.humanize import llm
from humanizer.humanize import pipeline as pipe
from humanizer.humanize.engine import check_invariants

requires_mlx = pytest.mark.skipif(
    not llm.mlx_available(),
    reason="mlx-lm is not importable (needs Apple Silicon)",
)


# --------------------------------------------------------------- fixtures

#: Two paragraphs of AI-flavoured prose carrying a number, a date, a citation
#: and a proper name, so the meaning invariants have something to protect.
SOURCE = (
    "In today's rapidly evolving research landscape, machine learning plays a "
    "pivotal role in clinical decision support. Furthermore, a 2019 study by "
    "Chen et al. reported that diagnostic accuracy improved by 12% when "
    "clinicians were shown model confidence alongside each prediction [4]. It "
    "is important to note that these gains were not uniform across "
    "specialties. Moreover, the intricate tapestry of deployment constraints "
    "must be considered.\n\n"
    "Consequently, hospitals must delve into the multifaceted question of "
    "governance. Additionally, robust oversight frameworks are essential for "
    "maintaining trust. Ultimately, success depends on a commitment to "
    "transparency and continuous improvement across the entire organisation."
)

FIRST_PARAGRAPH = SOURCE.split("\n\n")[0]


def echo_source(request: llm.GenerationRequest) -> str:
    """A backend that returns the draft it was given, unchanged."""
    rendered = request.prompt.rendered()
    if "DRAFT TO REVISE:\n" in rendered:
        return rendered.split("DRAFT TO REVISE:\n")[-1]
    return rendered.split("\n\nDRAFT:\n")[-1].split("\n\nHUMAN:\n")[0]


def fixed_backend(text_for=None, **kwargs):
    """An `EchoBackend` whose responses come from a callable on the request."""
    return llm.EchoBackend(text_for or echo_source, **kwargs)


class FakeDetector:
    """Scores text with a supplied function. No torch, no weights."""

    def __init__(self, fn=None, default=0.9):
        self.fn = fn
        self.default = default
        self.seen = []

    def available(self):
        return True

    def score(self, text):
        self.seen.append(text)
        value = self.fn(text) if self.fn is not None else self.default
        return SimpleNamespace(ai_probability=float(value))


def run(text=SOURCE, backend=None, detector=False, **config_kwargs):
    """Drain the pipeline and return `(events, result)`."""
    config = pipe.PipelineConfig(**config_kwargs)
    events = []
    result = None
    for event in pipe.stream(
        text, config=config, backend=backend or fixed_backend(), detector=detector
    ):
        events.append(event)
        if event.result_object is not None:
            result = event.result_object
    return events, result


# ------------------------------------------------------- prompt construction


def test_stage_names_are_the_documented_eight():
    assert pipe.STAGES == (
        "analyze",
        "plan",
        "generate",
        "scrub",
        "score",
        "verify",
        "select",
        "finalize",
    )


def test_faithful_prompt_carries_draft_contract_persona_and_banned_words():
    spec = llm.CANDIDATE_SPECS[2]
    prompt = llm.build_prompt(FIRST_PARAGRAPH, spec)
    rendered = prompt.rendered()

    assert prompt.kind == "chat"
    assert [role for role, _ in prompt.messages] == ["system", "user"]
    assert FIRST_PARAGRAPH in rendered
    assert spec.persona in rendered
    assert spec.constraint in rendered
    assert "STYLE CONTRACT" in rendered
    # A sample of the banned list, and the exemplar the model is matching.
    for banned in ("delve", "multifaceted", "furthermore", "in conclusion"):
        assert banned in rendered.lower()
    assert llm.STYLE_EXEMPLAR in rendered


def test_faithful_prompt_puts_the_draft_last():
    """Instruction-following decays with distance from the end of the context."""
    rendered = llm.build_prompt(FIRST_PARAGRAPH, llm.CANDIDATE_SPECS[0]).rendered()
    assert rendered.rstrip().endswith(FIRST_PARAGRAPH.strip())
    assert rendered.index("STYLE CONTRACT") < rendered.index("DRAFT TO REVISE")


def test_freeform_prompt_is_a_fewshot_continuation():
    spec = llm.FREEFORM_SPECS[0]
    prompt = llm.build_prompt(FIRST_PARAGRAPH, spec)
    assert prompt.kind == "text"
    assert prompt.text.count("DRAFT:") == 2  # the example and the real one
    assert prompt.text.count("HUMAN:") == 2
    assert prompt.text.rstrip().endswith("HUMAN:")
    assert llm.STYLE_EXEMPLAR in prompt.text


def test_build_prompt_refuses_an_empty_paragraph():
    with pytest.raises(ValueError):
        llm.build_prompt("   ", llm.CANDIDATE_SPECS[0])


# ---------------------------------------------------------------- diversity


def test_every_candidate_has_a_distinct_persona_and_constraint():
    """research/08: correlated candidates make best-of-N decay as a power law."""
    specs = llm.specs_for(8)
    assert len({s.persona for s in specs}) == 8
    assert len({s.constraint for s in specs}) == 8
    assert len({(s.persona, s.constraint) for s in specs}) == 8


def test_candidate_temperatures_span_a_range():
    specs = llm.specs_for(6)
    temperatures = [s.temperature for s in specs]
    assert min(temperatures) >= 0.7
    assert max(temperatures) <= 1.1
    assert max(temperatures) - min(temperatures) >= 0.2


def test_candidate_prompts_differ_from_each_other():
    """The diversity has to reach the prompt, not just the spec dataclass."""
    rendered = {
        llm.build_prompt(FIRST_PARAGRAPH, s).rendered() for s in llm.specs_for(6)
    }
    assert len(rendered) == 6


def test_specs_cycle_with_a_raised_temperature_past_the_ladder():
    specs = llm.specs_for(len(llm.CANDIDATE_SPECS) + 2)
    first, wrapped = specs[0], specs[len(llm.CANDIDATE_SPECS)]
    assert wrapped.persona == first.persona
    assert wrapped.temperature > first.temperature
    assert all(s.temperature <= 1.3 for s in specs)
    assert [s.index for s in specs] == list(range(len(specs)))


def test_specs_for_rejects_a_nonpositive_count():
    with pytest.raises(ValueError):
        llm.specs_for(0)


# ------------------------------------------------------ completion cleaning


def test_clean_completion_strips_a_chat_preamble():
    raw = "Sure! Here is the revised paragraph:\n\nThe study reported 12% gains."
    assert llm.clean_completion(raw) == "The study reported 12% gains."


def test_clean_completion_strips_a_code_fence_and_a_label():
    assert llm.clean_completion("```\nHUMAN: A short line.\n```") == "A short line."


def test_clean_completion_truncates_at_a_stop_string():
    raw = "The real rewrite.\n\nDRAFT:\nSomething the base model invented."
    assert llm.clean_completion(raw, llm.FREEFORM_STOPS) == "The real rewrite."


def test_clean_completion_leaves_an_internal_quotation_alone():
    """Stripping a real quotation would break `check_invariants`."""
    raw = 'The author wrote "a pivotal moment" in 1998 and never revised it.'
    assert llm.clean_completion(raw) == raw
    assert check_invariants(raw, llm.clean_completion(raw)) == []


def test_clean_completion_drops_a_trailing_self_commentary_block():
    raw = "The rewritten text.\n\nNote: I removed the transitions as requested."
    assert llm.clean_completion(raw) == "The rewritten text."


def test_estimate_max_tokens_scales_with_length_and_is_clamped():
    short = llm.estimate_max_tokens("one two three")
    long = llm.estimate_max_tokens(" ".join(["word"] * 400))
    assert short == 96
    assert long == 900
    assert llm.estimate_max_tokens(" ".join(["word"] * 120)) > short


# -------------------------------------------------------------- the gates


CONFIG = pipe.PipelineConfig()


def test_gate_rejects_a_mangled_number():
    """A changed figure is a factual change, not a stylistic one."""
    mangled = FIRST_PARAGRAPH.replace("12%", "21%")
    failures = pipe._gate_candidate(FIRST_PARAGRAPH, mangled, CONFIG)
    assert any(f.startswith("invariant:numbers") for f in failures)


def test_gate_rejects_a_dropped_citation():
    stripped = FIRST_PARAGRAPH.replace(" [4]", "")
    failures = pipe._gate_candidate(FIRST_PARAGRAPH, stripped, CONFIG)
    assert any(f.startswith("invariant:citations") for f in failures)


def test_gate_rejects_a_dropped_name():
    """The entity gate exists to catch a name vanishing."""
    without = FIRST_PARAGRAPH.replace("Chen et al.", "one group")
    failures = pipe._gate_candidate(FIRST_PARAGRAPH, without, CONFIG)
    assert "invariant:entities" in failures


def test_gate_allows_a_name_that_only_moved():
    """A reordering rewrite is not a factual change.

    `engine.invariants` compares capitalised runs as sets and skips each
    sentence's first token, so a name that moves to the front of a sentence
    reads as corrupted. That is right for span edits and wrong here; see
    `pipeline._invariant_failures`.
    """
    moved = (
        "Chen et al. reported in a 2019 study that diagnostic accuracy "
        "improved by 12% when clinicians were shown model confidence alongside "
        "each prediction [4]. The gains were not uniform across specialties. "
        "Machine learning shapes clinical decision support, and deployment "
        "constraints still have to be weighed against that."
    )
    assert "invariant:entities" not in pipe._gate_candidate(
        FIRST_PARAGRAPH, moved, CONFIG
    )
    # The exact-multiset invariants are untouched by that change.
    assert pipe._invariant_failures(FIRST_PARAGRAPH, moved) == []


def test_gate_rejects_a_truncated_candidate():
    """A candidate that stops halfway is short, not concise."""
    tokens = FIRST_PARAGRAPH.split()
    truncated = " ".join(tokens[: len(tokens) // 2])
    failures = pipe._gate_candidate(FIRST_PARAGRAPH, truncated, CONFIG)
    assert "too_short" in failures
    assert "length_ratio" in failures


def test_gate_rejects_an_inflated_candidate():
    inflated = FIRST_PARAGRAPH + " " + FIRST_PARAGRAPH
    failures = pipe._gate_candidate(FIRST_PARAGRAPH, inflated, CONFIG)
    assert "length_ratio" in failures
    assert "too_short" not in failures


def test_gate_rejects_fluent_but_off_topic_text():
    """The base checkpoint's characteristic failure: fluent, wrong subject."""
    drifted = (
        "The harbour master kept careful notes through the winter of 1911, and "
        "the ledger survives in the county archive, where a volunteer "
        "transcribed it over four years. Nobody has checked the arithmetic. "
        "Most of the entries concern coal, some concern timber, and a handful "
        "concern a dispute over moorings that ran for two seasons and was "
        "settled out of court by a solicitor whose name appears nowhere else."
    )
    failures = pipe._gate_candidate(FIRST_PARAGRAPH, drifted, CONFIG)
    assert "content_drift" in failures


def test_gate_rejects_an_unchanged_candidate():
    assert "unchanged" in pipe._gate_candidate(
        FIRST_PARAGRAPH, FIRST_PARAGRAPH, CONFIG
    )


def test_gate_rejects_an_empty_candidate():
    assert pipe._gate_candidate(FIRST_PARAGRAPH, "   ", CONFIG) == ["empty"]


def test_gate_accepts_a_faithful_restructuring():
    """The safe operating region has to actually be reachable."""
    faithful = (
        "Machine learning shapes clinical decision support. A 2019 study by "
        "Chen et al. reported that diagnostic accuracy improved by 12% when "
        "clinicians were shown model confidence alongside each prediction [4]. "
        "The gains were not uniform across specialties. Deployment constraints "
        "still have to be weighed, and that is where most of the argument sits."
    )
    assert pipe._gate_candidate(FIRST_PARAGRAPH, faithful, CONFIG) == []


def test_content_overlap_is_bounded_and_ordered():
    assert pipe.content_overlap(FIRST_PARAGRAPH, FIRST_PARAGRAPH) == 1.0
    assert pipe.content_overlap(FIRST_PARAGRAPH, "") == 0.0
    assert pipe.content_overlap("", "anything") == 1.0
    half = " ".join(FIRST_PARAGRAPH.split()[: len(FIRST_PARAGRAPH.split()) // 2])
    assert 0.0 < pipe.content_overlap(FIRST_PARAGRAPH, half) < 1.0


# ------------------------------------------------- scrub-after-LLM ordering


def test_scrub_runs_after_the_model_not_before():
    """The model reintroduces the vocabulary the scrub removes.

    Two assertions, and both matter: the backend must have been handed the
    *unscrubbed* source (otherwise the scrub was wasted before generation), and
    the candidate's stored text must be scrubbed (otherwise it never ran).
    """
    backend = fixed_backend()
    _events, result = run(backend=backend, n_candidates=2)

    sent = "\n".join(
        r.prompt.rendered() for call in backend.calls for r in call
    )
    assert "Furthermore," in sent, "the model must see the raw draft"
    assert "Moreover," in sent

    winner = result.paragraphs[0]
    assert winner.chosen_index is not None
    assert "Furthermore," in winner.original
    assert "Furthermore," not in winner.chosen

    # "Moreover," survives, and that is correct rather than a miss: it opens
    # the paragraph's final sentence, and research/09 rates any edit in the
    # first or last sentence as very-high grade cost, so `Document.
    # protected_spans()` keeps every transform out of it.
    assert "Moreover," in winner.chosen


def test_scrub_reports_its_edits_on_the_winning_candidate():
    _events, result = run(n_candidates=2)
    assert result.summary["n_edits"] > 0
    assert len(result.edits) == result.summary["n_edits"]


def test_scrub_preserves_the_meaning_invariants():
    scrubbed, edits = pipe._scrub(FIRST_PARAGRAPH)
    assert edits, "the sample is full of connectives; something should fire"
    assert check_invariants(FIRST_PARAGRAPH, scrubbed) == []


def test_scrub_does_not_call_the_detector(monkeypatch):
    """`engine.humanize` scores twice per call; at 50 candidates that is fatal."""

    def explode():
        raise AssertionError("the scrub must not touch the detector")

    monkeypatch.setattr(pipe, "_detector", explode)
    scrubbed, edits = pipe._scrub(FIRST_PARAGRAPH)
    assert edits and scrubbed != FIRST_PARAGRAPH


# --------------------------------------------------------- progress events


def _stage_runs(events):
    """Collapse the event list into contiguous per-stage runs."""
    runs = []
    for event in events:
        if not runs or runs[-1][0] != event.stage:
            runs.append((event.stage, [event]))
        else:
            runs[-1][1].append(event)
    return runs


def test_events_arrive_in_the_documented_stage_order():
    # One round: a second round repeats generate..select for the paragraphs
    # the proxy still calls AI, which is covered by the round tests below.
    events, _result = run(n_candidates=2, rounds=1, detector=FakeDetector())
    order = [stage for stage, _ in _stage_runs(events)]
    assert order == [
        "analyze",
        "plan",
        "generate",
        "scrub",
        "score",
        "verify",
        "select",
        "finalize",
    ]


def test_progress_is_monotonic_within_every_stage():
    events, _result = run(n_candidates=3, detector=FakeDetector())
    for stage, run_events in _stage_runs(events):
        values = [e.progress for e in run_events]
        assert values == sorted(values), "%s went backwards: %s" % (stage, values)
        assert all(0.0 <= v <= 1.0 for v in values)
        assert run_events[0].status == "start" or run_events[0].status == "skip"
        assert run_events[-1].status in ("done", "skip")
        assert run_events[-1].progress == 1.0


def test_elapsed_never_goes_backwards():
    events, _result = run(n_candidates=2)
    elapsed = [e.elapsed for e in events]
    assert elapsed == sorted(elapsed)


def test_every_event_carries_the_contract_keys():
    events, _result = run(n_candidates=2)
    for event in events:
        payload = event.as_dict()
        assert set(payload) >= {"stage", "status", "progress", "detail", "elapsed"}
        assert payload["stage"] in pipe.STAGES
        assert payload["status"] in ("start", "progress", "done", "skip")
        assert isinstance(payload["detail"], str) and payload["detail"]


def test_details_name_the_candidate_and_the_paragraph():
    events, _result = run(n_candidates=3, detector=FakeDetector())
    details = [e.detail for e in events]
    assert any(re.search(r"candidate \d+ of \d+, paragraph \d+ of \d+", d) for d in details)
    assert any("paragraph 1 of 2" in d for d in details)


def test_exactly_one_event_carries_the_result():
    events, _result = run(n_candidates=2)
    with_result = [e for e in events if e.result is not None]
    assert len(with_result) == 1
    assert (with_result[0].stage, with_result[0].status) == ("finalize", "done")
    assert with_result[0] is events[-1]


def test_the_score_stage_is_skipped_rather_than_faked_without_a_detector():
    events, _result = run(n_candidates=2, detector=False)
    score_events = [e for e in events if e.stage == "score"]
    assert [e.status for e in score_events] == ["skip"]
    assert "no detector" in score_events[0].detail


def test_stage_table_includes_finalize_and_matches_the_stream():
    _events, result = run(n_candidates=2, detector=FakeDetector())
    stages = [row["stage"] for row in result.stages]
    assert stages[-1] == "finalize"
    assert set(stages) == set(pipe.STAGES)
    assert all(row["seconds"] >= 0 for row in result.stages)


# ------------------------------------------------------------- selection


def test_selection_takes_the_lowest_scoring_survivor():
    """Ranking is by detector score, over the candidates that passed the gates."""
    marks = {}

    def responder(request):
        para = echo_source(request)
        # Give each candidate a distinguishable, faithful rewrite.
        tag = "%0.2f" % request.temperature
        marks[tag] = True
        return para.replace("Furthermore,", "Separately,") + " Tagged %s." % tag

    def scorer(text):
        match = re.search(r"Tagged (\d\.\d\d)\.", text)
        return float(match.group(1)) / 2.0 if match else 0.99

    detector = FakeDetector(scorer)
    _events, result = run(
        backend=fixed_backend(responder), detector=detector, n_candidates=4
    )
    for outcome in result.paragraphs:
        survivors = [c for c in outcome.candidates if c.passed]
        if not survivors:
            continue
        assert outcome.ai_probability_after == min(
            c.ai_probability for c in survivors
        )


def test_a_paragraph_whose_candidates_all_fail_keeps_its_original():
    """Falling back is reported, never silently swallowed."""
    _events, result = run(backend=fixed_backend(lambda r: "No."), n_candidates=2)
    for outcome in result.paragraphs:
        assert outcome.chosen == outcome.original
        assert outcome.chosen_index is None
        assert outcome.fallback_reason and "no candidate passed" in outcome.fallback_reason
    assert result.humanized.strip() == result.original.strip()
    assert result.summary["paragraphs_unchanged"] == len(result.paragraphs)


def test_summary_breaks_rejections_down_by_gate():
    _events, result = run(backend=fixed_backend(lambda r: "No."), n_candidates=2)
    rejections = result.summary["gate_rejections"]
    assert rejections
    assert sum(rejections.values()) >= result.summary["n_candidates_rejected"]
    assert result.summary["n_candidates_passed"] == 0


def test_summary_reports_the_verdict_not_only_the_delta():
    """research/13: probability deltas without verdict flips mean nothing."""
    detector = FakeDetector(lambda t: 0.99)
    _events, result = run(n_candidates=2, detector=detector)
    assert result.summary["label_before"] == "ai"
    assert result.summary["verdict_flipped"] is False
    assert set(result.summary) >= {
        "risk_delta",
        "label_before",
        "label_after",
        "verdict_flipped",
        "gate_rejections",
        "quality_flags",
    }


def test_quality_flags_surface_a_possible_semantic_drift():
    """A candidate can pass every gate and still be worth flagging."""
    quality = pipe._quality(FIRST_PARAGRAPH, "Chen reported gains in 2019. Deployment is hard.")
    assert "possible_semantic_drift" in quality["flags"]
    assert quality["content_overlap"] < 0.55


def test_quality_block_measures_the_shape_band_and_readability():
    quality = pipe._quality(FIRST_PARAGRAPH, FIRST_PARAGRAPH)
    assert quality["content_overlap"] == 1.0
    assert quality["sent_len_cv"] is not None
    assert quality["flesch_reading_ease"] is not None
    assert quality["ai_vocab_hits_per_1k"] > 0


def test_flesch_reading_ease_orders_two_obvious_cases():
    simple = "The cat sat. The dog ran. Birds sing. We went home."
    dense = (
        "The heterogeneity of institutional accountability mechanisms "
        "necessitates a correspondingly differentiated evaluative "
        "infrastructure, particularly where jurisdictional overlap obtains."
    )
    assert pipe.flesch_reading_ease(simple) > pipe.flesch_reading_ease(dense)
    assert math.isnan(pipe.flesch_reading_ease(""))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        list(pipe.stream("   "))


def test_a_document_with_no_blank_lines_is_one_paragraph():
    _events, result = run(text=FIRST_PARAGRAPH, n_candidates=2)
    assert len(result.paragraphs) == 1


def test_result_payload_is_a_superset_of_the_rule_based_body():
    _events, result = run(n_candidates=2, detector=FakeDetector())
    payload = result.as_dict()
    assert set(payload) >= {
        "original",
        "humanized",
        "edits",
        "before",
        "after",
        "summary",
        "model",
        "stages",
        "paragraphs",
    }
    assert json.dumps(payload, default=str)


def test_a_second_round_repeats_the_stages_with_a_higher_round_number():
    events, result = run(
        n_candidates=2, rounds=2, detector=FakeDetector(lambda t: 0.99)
    )
    rounds = {e.round for e in events}
    assert rounds == {1, 2}
    generate_rounds = [e.round for e in events if e.stage == "generate"]
    assert generate_rounds == sorted(generate_rounds)
    assert result.summary["n_candidates_total"] > 2 * len(result.paragraphs)


def test_a_second_round_is_skipped_when_every_paragraph_is_already_human():
    events, _result = run(
        n_candidates=2, rounds=2, detector=FakeDetector(lambda t: 0.01)
    )
    skips = [e for e in events if e.status == "skip" and e.stage == "generate"]
    assert len(skips) == 1
    assert "not needed" in skips[0].detail


# ------------------------------------------------------ backend behaviour


def test_backend_receives_one_request_per_candidate_per_paragraph():
    backend = fixed_backend()
    _events, result = run(backend=backend, n_candidates=3)
    assert sum(len(call) for call in backend.calls) == 3 * len(result.paragraphs)
    # One batched call per style per paragraph: the instruct and base
    # checkpoints are different backends, so a batch cannot span them.
    n_styles = len(pipe.PipelineConfig(n_candidates=3).styles())
    assert len(backend.calls) == n_styles * len(result.paragraphs)


def test_generation_requests_carry_a_per_candidate_temperature():
    backend = fixed_backend()
    run(backend=backend, n_candidates=4)
    temperatures = {r.temperature for call in backend.calls for r in call}
    assert len(temperatures) >= 3


def test_cohorts_honour_exact_temperatures_when_they_fit():
    backend = llm.MlxBackend(temperature_groups=2)
    requests = [
        llm.GenerationRequest(llm.build_prompt("A short paragraph here.", s), t, 32)
        for s, t in zip(llm.specs_for(4), (0.8, 0.8, 0.8, 0.8))
    ]
    assert backend._cohorts(requests) == [(0.8, [0, 1, 2, 3])]


def test_cohorts_collapse_many_temperatures_into_the_configured_groups():
    """Six temperatures would otherwise mean six unbatched calls."""
    backend = llm.MlxBackend(temperature_groups=2)
    specs = llm.specs_for(6)
    requests = [
        llm.GenerationRequest(llm.build_prompt("A short paragraph here.", s), s.temperature, 32)
        for s in specs
    ]
    cohorts = backend._cohorts(requests)
    assert len(cohorts) == 2
    assert sorted(i for _t, members in cohorts for i in members) == list(range(6))
    assert cohorts[0][0] < cohorts[1][0]
    assert all(len(members) == 3 for _t, members in cohorts)


def test_cohorts_can_be_disabled_for_maximum_batching():
    backend = llm.MlxBackend(temperature_groups=1)
    requests = [
        llm.GenerationRequest(llm.build_prompt("A short paragraph here.", s), s.temperature, 32)
        for s in llm.specs_for(6)
    ]
    assert len(backend._cohorts(requests)) == 1


def test_progress_detail_is_grammatical_for_a_single_paragraph():
    events, _result = run(text=FIRST_PARAGRAPH, n_candidates=2)
    details = " ".join(e.detail for e in events)
    assert "1 paragraph," in details
    assert "1 paragraphs" not in details


def test_mlx_unavailable_reason_is_actionable_when_import_fails(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_lm":
            raise ImportError("No module named 'mlx_lm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert llm.mlx_available() is False
    reason = llm.mlx_unavailable_reason()
    assert "mlx-lm is not installed" in reason
    assert "/api/humanize" in reason
    assert pipe.pipeline_available() is False


def test_backend_load_failure_is_a_runtime_error_naming_the_fallback(monkeypatch):
    backend = llm.MlxBackend("nonexistent/model-that-is-not-there")
    monkeypatch.setattr(llm, "mlx_unavailable_reason", lambda: None)
    monkeypatch.setattr(llm, "mlx_available", lambda: True)

    def explode(*_args, **_kwargs):
        raise OSError("not found on the hub")

    monkeypatch.setattr(backend, "_encode", explode)
    import sys as _sys

    if "mlx_lm" not in _sys.modules:
        pytest.skip("mlx_lm is not importable, so load() fails earlier")
    with pytest.raises(RuntimeError) as excinfo:
        backend.load()
    assert "/api/humanize" in str(excinfo.value)


@requires_mlx
def test_mlx_backend_reports_whether_it_batched():
    """The measured speedup is ~3x, so which path ran is worth asserting."""
    backend = llm.MlxBackend()
    backend.load()
    assert isinstance(backend.batched, bool)
    requests = [
        llm.GenerationRequest(
            prompt=llm.build_prompt("The sky is blue on most days.", spec),
            temperature=0.8,
            max_tokens=24,
        )
        for spec in llm.specs_for(2)
    ]
    batch = backend.generate(requests)
    assert len(batch.texts) == 2
    assert batch.model == llm.DEFAULT_MODEL
    assert batch.batched == backend.batched
    assert batch.seconds > 0


# ------------------------------------------------------------- HTTP layer

fastapi = pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api import server as server_mod  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(server_mod.create_app(web_dir=None))


def _fake_stream(text, config=None, backend=None, detector=None):
    """A pipeline stream that needs no model, for the HTTP tests."""
    yield pipe.ProgressEvent("analyze", "start", 0.0, "reading", 0.001)
    yield pipe.ProgressEvent("analyze", "done", 1.0, "1 paragraph", 0.002)
    payload = {
        "original": text,
        "humanized": text.upper(),
        "edits": [],
        "before": {"ai_probability": 0.99, "label": "ai", "features": {}},
        "after": {"ai_probability": 0.98, "label": "ai", "features": {}},
        "summary": {"verdict_flipped": False, "risk_delta": -0.01},
        "model": "test/model",
        "backend": "echo",
        "stages": [{"stage": "analyze", "seconds": 0.002}],
        "paragraphs": [],
    }
    yield pipe.ProgressEvent(
        "finalize", "done", 1.0, "done", 0.003, result=payload
    )


def test_sse_frame_has_event_data_and_a_blank_terminator():
    frame = sse = server_mod.sse_frame("progress", {"stage": "analyze"})
    assert frame.startswith("event: progress\ndata: {")
    assert frame.endswith("\n\n")
    lines = frame.split("\n")
    assert len(lines) == 4 and lines[2] == "" and lines[3] == ""
    assert json.loads(lines[1][len("data: ") :]) == {"stage": "analyze"}


def test_sse_frame_survives_nan_in_a_feature_vector():
    """Feature dicts really do contain NaN, and NaN is not JSON."""
    frame = server_mod.sse_frame("result", {"features": {"sent_len_cv": float("nan")}})
    assert "NaN" not in frame
    assert json.loads(frame.split("data: ")[1])["features"]["sent_len_cv"] is None


def test_llm_health_reports_stages_and_the_fallback(client):
    body = client.get("/api/humanize/llm/health").json()
    assert body["stages"] == list(pipe.STAGES)
    assert body["fallback_endpoint"] == "/api/humanize"
    assert isinstance(body["available"], bool)
    if not body["available"]:
        assert body["reason"]


def test_llm_endpoint_rejects_empty_text(client):
    response = client.post("/api/humanize/llm", json={"text": "  "})
    assert response.status_code == 400
    assert response.json()["error"] == "empty_text"


def test_llm_endpoint_rejects_an_oversized_document(client):
    long_text = " ".join(["word"] * (server_mod.LLM_MAX_WORDS + 10))
    response = client.post("/api/humanize/llm", json={"text": long_text})
    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "too_long"
    assert body["fallback_endpoint"] == "/api/humanize"


def test_llm_endpoint_rejects_an_unknown_style(client):
    response = client.post(
        "/api/humanize/llm", json={"text": SOURCE, "style": "interpretive-dance"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "unknown_style"


def test_llm_endpoint_degrades_to_503_without_mlx(client, monkeypatch):
    monkeypatch.setattr(
        pipe, "pipeline_unavailable_reason", lambda config=None: "mlx-lm is not installed."
    )
    response = client.post("/api/humanize/llm", json={"text": SOURCE})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "llm_unavailable"
    assert body["available"] is False
    assert "/api/humanize" in body["fallback"]
    assert body["fallback_endpoint"] == "/api/humanize"


def test_stream_endpoint_degrades_to_503_without_mlx(client, monkeypatch):
    monkeypatch.setattr(
        pipe, "pipeline_unavailable_reason", lambda config=None: "mlx-lm is not installed."
    )
    response = client.post("/api/humanize/stream", json={"text": SOURCE})
    assert response.status_code == 503
    assert response.json()["fallback_endpoint"] == "/api/humanize"


def test_stream_endpoint_emits_progress_then_result_then_done(client, monkeypatch):
    monkeypatch.setattr(pipe, "pipeline_unavailable_reason", lambda config=None: None)
    monkeypatch.setattr(pipe, "stream", _fake_stream)
    response = client.post("/api/humanize/stream", json={"text": SOURCE})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"].startswith("no-cache")

    names = re.findall(r"^event: (\w+)$", response.text, re.MULTILINE)
    assert names == ["progress", "progress", "progress", "result", "done"]

    frames = [f for f in response.text.split("\n\n") if f.strip()]
    result_frame = [f for f in frames if f.startswith("event: result")][0]
    payload = json.loads(result_frame.split("data: ")[1])
    assert payload["model"] == "test/model"
    assert payload["summary"]["verdict_flipped"] is False

    first = json.loads(frames[0].split("data: ")[1])
    assert set(first) >= {"stage", "status", "progress", "detail", "elapsed"}
    assert "result" not in first


def test_stream_endpoint_reports_a_mid_stream_failure_as_an_error_frame(
    client, monkeypatch
):
    """Once the headers are flushed there is no status code left to use."""

    def exploding_stream(text, config=None, backend=None, detector=None):
        yield pipe.ProgressEvent("analyze", "start", 0.0, "reading", 0.0)
        raise RuntimeError("Could not load the model with mlx-lm")

    monkeypatch.setattr(pipe, "pipeline_unavailable_reason", lambda config=None: None)
    monkeypatch.setattr(pipe, "stream", exploding_stream)
    response = client.post("/api/humanize/stream", json={"text": SOURCE})

    assert response.status_code == 200
    names = re.findall(r"^event: (\w+)$", response.text, re.MULTILINE)
    assert names == ["progress", "error"]
    error = json.loads(response.text.split("event: error\ndata: ")[1].strip())
    assert "/api/humanize" in error["fallback"]


def test_job_then_eventsource_get_streams_the_same_frames(client, monkeypatch):
    monkeypatch.setattr(pipe, "pipeline_unavailable_reason", lambda config=None: None)
    monkeypatch.setattr(pipe, "stream", _fake_stream)

    job = client.post("/api/humanize/jobs", json={"text": SOURCE}).json()
    assert job["stream_url"] == "/api/humanize/stream?job=%s" % job["job_id"]

    response = client.get(job["stream_url"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    names = re.findall(r"^event: (\w+)$", response.text, re.MULTILINE)
    assert names[-2:] == ["result", "done"]

    # Job ids are single-use.
    assert client.get(job["stream_url"]).status_code == 404


def test_unknown_job_id_is_a_404_before_the_stream_opens(client):
    response = client.get("/api/humanize/stream?job=deadbeef")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_job"


def test_llm_routes_sit_in_front_of_the_static_mount(tmp_path):
    """A Mount at "/" matches every path, so ordering is the whole contract."""
    (tmp_path / "index.html").write_text("<html></html>")
    app = server_mod.create_app(web_dir=tmp_path)
    paths = [getattr(r, "path", None) for r in app.router.routes]
    assert "/api/humanize/llm" in paths
    assert paths.index("/api/humanize/llm") < paths.index("")
    with TestClient(app) as static_client:
        assert static_client.get("/api/humanize/llm/health").status_code == 200


def test_rule_based_endpoint_still_works_alongside_the_llm_one(client):
    """The LLM block is append-only; the existing contract must be untouched."""
    response = client.post("/api/humanize", json={"text": SOURCE})
    assert response.status_code == 200
    assert set(response.json()) == {
        "original",
        "humanized",
        "edits",
        "before",
        "after",
        "summary",
    }
