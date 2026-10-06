"""Tests for detector-guided decoding.

The module has a GPU and two transformer detectors in the middle of it, so
almost everything here runs against `llm.EchoBackend` and two *deliberately
disagreeing* fake detectors. That is not a concession to CI. The behaviour
worth protecting is structural, and none of it needs weights to be wrong:

  * **The guide is not the yardstick.** This is the whole design. The in-loop
    guide (`MayZhou/e5-small-lora`, 28 ms, and by this repo's own bench a
    detector that calls 8 of 14 human paragraphs AI) orders the search; the
    yardstick (desklib) produces every number the response reports. The fake
    detectors here are built to disagree *maximally* -- the guide loves what
    the yardstick hates -- so any code path that quietly substituted one for
    the other fails loudly. See `TestGuideIsNotTheYardstick`.
  * **Beam bookkeeping.** `select_beam` is pure and is tested directly:
    duplicate collapse, unscored-last ordering, tie-breaking toward the
    conservative candidate, and the width contract. A beam that silently
    collapses to greedy is the difference between this module and
    `pipeline.py`.
  * **Gate reuse, not gate reimplementation.** The paragraph gates are
    `pipeline._gate_candidate` and the invariants are
    `pipeline._invariant_failures`. The tests assert the *call*, by
    monkeypatching the pipeline's function and watching it fire, so a future
    copy-paste of the gate logic into `guided.py` breaks a test.
  * **The event protocol and the SSE wire format.** The frontend builds its
    checklist from the published `stages` list and renders `progress`
    verbatim, so stage order, per-stage terminal events, monotonicity and the
    exactly-one-`result`-frame rule are contract tests.
  * **Degradation without MLX.** Every guided endpoint must answer 503 naming
    both fallbacks, never 500.

MLX-dependent tests are marked `requires_mlx` and skip cleanly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from humanizer.humanize import guided
from humanizer.humanize import llm
from humanizer.humanize import pipeline as pipe

requires_mlx = pytest.mark.skipif(
    not llm.mlx_available(),
    reason="mlx-lm is not importable (needs Apple Silicon)",
)


# --------------------------------------------------------------- fixtures

#: Three sentences carrying a date, a percentage, a citation and a proper
#: name, so the invariants have something real to protect.
SOURCE = (
    "In today's rapidly evolving research landscape, machine learning plays a "
    "pivotal role in clinical decision support. A 2019 study by Chen et al. "
    "reported that diagnostic accuracy improved by 12% when clinicians were "
    "shown model confidence [4]. Furthermore, these gains were not uniform "
    "across specialties."
)


class FakeDetector:
    """Anything with `.score(text).ai_probability`. Records what it saw.

    `scorer` maps text to a probability. The default is a constant, which is
    what most tests want; the interesting tests pass a function that separates
    two candidates so the ranking is observable.
    """

    def __init__(self, scorer=0.9, name="fake"):
        self.scorer = scorer
        self.name = name
        self.seen = []

    def available(self):
        return True

    def unavailable_reason(self):
        return None

    def score(self, text):
        self.seen.append(text)
        value = self.scorer(text) if callable(self.scorer) else float(self.scorer)
        return SimpleNamespace(ai_probability=float(value))


def echo_source_sentence(request):
    """An `EchoBackend` responder that returns the sentence it was asked to
    rewrite, verbatim. Faithful by construction, so gates pass and the test is
    about the plumbing rather than about the model."""
    user = request.prompt.messages[-1][1]
    return user.rsplit("Rewrite this sentence:", 1)[-1].strip()


#: Lead-ins keyed to `SENTENCE_MOVES` by position, so `echo_variant` produces a
#: *different* sentence per sample index. `echo_source_sentence` returns the
#: same string for every sample, and `select_beam` correctly collapses
#: duplicates -- which makes it useless for testing that the beam has width.
_LEAD_INS = ("plainly", "in short", "notably", "broadly", "in fact", "again",
             "still", "even so")


def echo_variant(request):
    """Echo the source sentence behind a lead-in chosen by structural move.

    Faithful (every number, name and citation survives, and the first token is
    only lower-cased, which `engine._entities` already ignores) but distinct
    per sample, so the beam has genuinely different hypotheses to rank.
    """
    user = request.prompt.messages[-1][1]
    base = user.rsplit("Rewrite this sentence:", 1)[-1].strip()
    move = ""
    if "How to write it: " in user:
        move = user.split("How to write it: ", 1)[1].split("\n", 1)[0]
    names = [m for m, _t in guided.SENTENCE_MOVES]
    index = names.index(move) if move in names else 0
    lead = _LEAD_INS[index % len(_LEAD_INS)]
    return "%s, %s%s" % (lead.capitalize(), base[:1].lower(), base[1:])


def run(text=SOURCE, config=None, backend=None, detector=None, guide=False, **kw):
    """`humanize_guided` with test-friendly defaults."""
    cfg = config or guided.GuidedConfig(beam_width=2, n_samples=2)
    be = backend or llm.EchoBackend(responses=echo_source_sentence)
    return guided.humanize_guided(
        text, config=cfg, backend=be, detector=detector, guide=guide, **kw
    )


# ------------------------------------------------------- 1. beam bookkeeping


class TestBeamBookkeeping:
    def test_select_beam_keeps_the_lowest_scoring_width(self):
        hyps = [guided.Hypothesis(sentences=(t,)) for t in ("a", "b", "c", "d")]
        scored = [(0.9, 0, hyps[0]), (0.1, 1, hyps[1]), (0.5, 2, hyps[2]), (0.7, 3, hyps[3])]
        kept = guided.select_beam(scored, 2)
        assert [h.sentences[0] for h in kept] == ["b", "c"]

    def test_select_beam_collapses_duplicate_texts(self):
        """Three beams that rewrite a sentence identically must not eat the
        beam. Without the dedupe the search silently becomes greedy."""
        same = [guided.Hypothesis(sentences=("same",)) for _ in range(3)]
        other = guided.Hypothesis(sentences=("different",))
        scored = [
            (0.1, 0, same[0]),
            (0.1, 1, same[1]),
            (0.1, 2, same[2]),
            (0.4, 3, other),
        ]
        kept = guided.select_beam(scored, 3)
        assert [h.sentences[0] for h in kept] == ["same", "different"]

    def test_select_beam_sorts_unscored_last_but_keeps_them(self):
        """No guide installed must degrade to 'first width candidates', not to
        an empty beam."""
        scored = [
            (None, 0, guided.Hypothesis(sentences=("unscored",))),
            (0.8, 1, guided.Hypothesis(sentences=("scored",))),
        ]
        kept = guided.select_beam(scored, 2)
        assert [h.sentences[0] for h in kept] == ["scored", "unscored"]
        assert guided.select_beam(scored, 1)[0].sentences[0] == "scored"

    def test_select_beam_ties_break_toward_the_earlier_candidate(self):
        """A tie resolves to the lower generation index, which is the cooler
        temperature and the more conservative structural move."""
        scored = [
            (0.5, 7, guided.Hypothesis(sentences=("late",))),
            (0.5, 1, guided.Hypothesis(sentences=("early",))),
        ]
        assert guided.select_beam(scored, 1)[0].sentences[0] == "early"

    def test_select_beam_rejects_a_non_positive_width(self):
        with pytest.raises(ValueError, match="width must be positive"):
            guided.select_beam([], 0)

    def test_hypothesis_extend_records_its_parent_and_fallbacks(self):
        h = guided.Hypothesis()
        h = h.extend("one.", 0.4, parent=0, fallback=False)
        h = h.extend("two.", 0.3, parent=1, fallback=True)
        assert h.text == "one. two."
        assert h.origin == (0, 1)
        assert h.n_fallbacks == 1
        assert h.as_dict()["guide_score"] == 0.3

    def test_beam_width_one_is_greedy_and_still_produces_a_result(self):
        result = run(config=guided.GuidedConfig(beam_width=1, n_samples=2))
        assert len(result.paragraphs[0].finalists) == 1

    def test_the_beam_carries_at_most_width_hypotheses_between_sentences(self):
        """The observable consequence: batch n+1 asks for exactly
        `beam_width x n_samples` continuations, never `n_samples^step`."""
        backend = llm.EchoBackend(responses=echo_variant)
        cfg = guided.GuidedConfig(beam_width=2, n_samples=3)
        guide = FakeDetector(scorer=lambda t: 0.5 + 0.001 * len(t))
        guided.humanize_guided(
            SOURCE, config=cfg, backend=backend, detector=False, guide=guide
        )
        sizes = [len(call) for call in backend.calls]
        assert sizes[0] == 3, "the first sentence starts from one empty hypothesis"
        assert all(n == 6 for n in sizes[1:]), sizes


# ------------------------------------------- 2. guide versus yardstick


class TestGuideIsNotTheYardstick:
    def test_reported_probability_comes_from_the_yardstick(self):
        yardstick = FakeDetector(scorer=0.11, name="desklib")
        guide = FakeDetector(scorer=0.97, name="e5")
        result = run(detector=yardstick, guide=guide)
        assert result.after["ai_probability"] == pytest.approx(0.11)
        assert result.before["ai_probability"] == pytest.approx(0.11)
        assert result.guide["probability_after"] == pytest.approx(0.97)

    def test_verdict_is_the_yardsticks_even_when_the_guide_disagrees(self):
        """Guide says 0.02 ('human'), yardstick says 0.99 ('ai'). The verdict
        must be 'ai' and must not be flipped."""
        result = run(
            detector=FakeDetector(scorer=0.99),
            guide=FakeDetector(scorer=0.02),
        )
        assert result.after["label"] == "ai"
        assert result.summary["verdict_flipped"] is False
        # The summary names whatever actually judged, not the configured default.
        assert result.summary["yardstick_model"] == "fake"

    def test_risk_delta_is_computed_from_yardstick_scores_only(self):
        """Yardstick constant, guide plunging. `risk_delta` must be ~0 and the
        guide's large drop must live only in the `guide` block."""
        yardstick = FakeDetector(scorer=0.95)
        guide = FakeDetector(scorer=lambda t: 0.95 if t.startswith("In today") else 0.05)
        result = run(
            backend=llm.EchoBackend(responses=echo_variant),
            detector=yardstick,
            guide=guide,
        )
        assert result.summary["risk_delta"] == pytest.approx(0.0, abs=1e-9)
        assert result.guide["delta"] < -0.5

    def test_selection_among_finalists_uses_the_yardstick_not_the_guide(self):
        """Two finalists survive. The guide prefers the one containing 'alpha';
        the yardstick prefers 'beta'. The chosen text must contain 'beta'."""
        sentences = ["Alpha holds in 2019.", "Beta holds in 2019."]
        counter = {"n": 0}

        def responder(request):
            out = sentences[counter["n"] % 2]
            counter["n"] += 1
            return out

        backend = llm.EchoBackend(responses=responder)
        guide = FakeDetector(scorer=lambda t: 0.1 if "Alpha" in t else 0.9)
        yardstick = FakeDetector(scorer=lambda t: 0.1 if "Beta" in t else 0.9)
        result = guided.humanize_guided(
            "Something holds in 2019.",
            config=guided.GuidedConfig(
                beam_width=2, n_samples=2, min_content_overlap=0.0,
                min_sentence_overlap=0.0, max_length_ratio_delta=1.0,
                min_length_ratio=0.4,
            ),
            backend=backend,
            detector=yardstick,
            guide=guide,
        )
        assert "Beta" in result.humanized
        assert result.after["ai_probability"] == pytest.approx(0.1)

    def test_guide_block_is_labelled_as_a_heuristic_and_names_the_yardstick(self):
        result = run(detector=FakeDetector(0.9), guide=FakeDetector(0.4))
        note = result.guide["note"]
        assert "not a verdict" in note
        assert guided.YARDSTICK_MODEL in note

    def test_guide_transfer_is_reported_with_the_gap_between_the_two(self):
        result = run(detector=FakeDetector(0.90), guide=FakeDetector(0.20))
        transfer = result.summary["guide_transfer"]
        assert transfer["n"] >= 1
        assert transfer["mean_signed_gap"] == pytest.approx(-0.70, abs=1e-6)
        assert transfer["mean_abs_gap"] == pytest.approx(0.70, abs=1e-6)
        assert transfer["verdict_disagreements"] == transfer["n"]

    def test_transfer_helper_handles_an_empty_and_a_partial_input(self):
        assert guided._transfer([])["n"] == 0
        assert guided._transfer([(None, 0.5), (0.4, None)])["n"] == 0
        one = guided._transfer([(0.6, 0.4), (None, 0.9)])
        assert one["n"] == 1 and one["mean_signed_gap"] == pytest.approx(0.2)

    def test_the_search_runs_unguided_when_the_guide_is_disabled(self):
        result = run(detector=FakeDetector(0.5), guide=False)
        assert result.summary["guide_available"] is False
        assert result.guide["probability_after"] is None
        assert "disabled by the caller" in result.summary["guide_unavailable_reason"]
        # ...and still produces a scored result from the yardstick.
        assert result.after["ai_probability"] == pytest.approx(0.5)


# ------------------------------------------------------------ 3. gate reuse


class TestGateReuse:
    def test_sentence_gate_calls_the_pipelines_invariant_check(self, monkeypatch):
        """Not a copy of the invariant logic -- the pipeline's own function."""
        calls = []
        original = pipe._invariant_failures

        def spy(source, candidate):
            calls.append((source, candidate))
            return original(source, candidate)

        monkeypatch.setattr(pipe, "_invariant_failures", spy)
        guided.sentence_gate("Growth hit 12% in 2019.", "Growth hit 12% in 2019.")
        assert calls, "sentence_gate must delegate to pipeline._invariant_failures"

    def test_sentence_gate_rejects_a_mangled_number(self):
        gates = guided.sentence_gate(
            "Accuracy improved by 12% in 2019.", "Accuracy improved by 21% in 2019."
        )
        assert "invariant:numbers" in gates

    def test_sentence_gate_rejects_a_dropped_proper_name(self):
        gates = guided.sentence_gate(
            "A study by Chen reported the effect.", "A study reported the effect."
        )
        assert "invariant:entities" in gates

    def test_sentence_gate_rejects_content_drift(self):
        gates = guided.sentence_gate(
            "Microplastic concentrations rose sharply in coastal sediment.",
            "Nothing whatsoever about that.",
        )
        assert "content_drift" in gates

    def test_sentence_gate_allows_the_length_swings_the_moves_ask_for(self):
        """'Make it short and blunt' and 'let it run long' are two of eight
        `SENTENCE_MOVES`; the sentence gate must not reject its own moves."""
        source = "The measured effect was small but consistent across the cohort."
        short = "The effect was small but consistent."
        assert "length_ratio" not in guided.sentence_gate(source, short)

    def test_sentence_gate_rejects_empty_output(self):
        assert guided.sentence_gate("A real sentence here.", "   ") == ["empty"]

    def test_paragraph_gates_are_the_pipelines(self, monkeypatch):
        calls = []
        original = pipe._gate_candidate

        def spy(source, candidate, config):
            calls.append(type(config).__name__)
            return original(source, candidate, config)

        monkeypatch.setattr(pipe, "_gate_candidate", spy)
        run(detector=False)
        assert calls, "the verify stage must call pipeline._gate_candidate"
        assert set(calls) == {"GuidedConfig"}, (
            "GuidedConfig must be accepted by the pipeline's gate directly; "
            "that shared field vocabulary is the reuse"
        )

    def test_a_paragraph_that_mangles_a_number_falls_back_to_the_original(self):
        """A candidate that drops the score by breaking a fact is a failure,
        not a success."""
        backend = llm.EchoBackend(
            responses=lambda r: "Accuracy improved by 99% in 1066 according to nobody."
        )
        result = guided.humanize_guided(
            "Accuracy improved by 12% in 2019 according to Chen.",
            config=guided.GuidedConfig(beam_width=1, n_samples=1),
            backend=backend,
            detector=FakeDetector(0.01),
            guide=FakeDetector(0.01),
        )
        assert result.humanized.strip() == result.original.strip()
        assert result.paragraphs[0].fallback_reason
        assert result.summary["paragraphs_unchanged"] == 1

    def test_a_sentence_whose_candidates_all_fail_keeps_the_source_sentence(self):
        backend = llm.EchoBackend(responses=lambda r: "")
        result = guided.humanize_guided(
            SOURCE,
            config=guided.GuidedConfig(beam_width=1, n_samples=2),
            backend=backend,
            detector=False,
            guide=False,
        )
        assert result.summary["n_fallback_sentences"] == result.summary["n_sentences"]
        assert result.humanized.strip() == result.original.strip()

    def test_scrub_reuses_the_pipelines_transform_passes(self, monkeypatch):
        calls = []
        original = pipe._scrub

        def spy(text, aggressiveness="strong"):
            calls.append(aggressiveness)
            return original(text, aggressiveness)

        monkeypatch.setattr(pipe, "_scrub", spy)
        run(detector=False)
        assert calls and set(calls) == {"strong"}


# --------------------------------------------------- 4. the event protocol


def stage_sequence(events):
    """Stage names in first-seen order."""
    out = []
    for ev in events:
        if not out or out[-1] != ev.stage:
            out.append(ev.stage)
    return out


class TestProgressEvents:
    def test_stages_match_the_pipelines_exactly(self):
        """The frontend keys its checklist off these strings and both
        pipelines feed the same checklist."""
        assert guided.STAGES == pipe.STAGES

    def test_events_arrive_in_stage_order(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=1, n_samples=1),
                backend=llm.EchoBackend(responses=echo_source_sentence),
                detector=False,
                guide=False,
            )
        )
        assert stage_sequence(events) == list(guided.STAGES)

    def test_every_stage_that_runs_emits_exactly_one_terminal_event(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=1, n_samples=1),
                backend=llm.EchoBackend(responses=echo_source_sentence),
                detector=FakeDetector(0.7),
                guide=False,
            )
        )
        terminals = {}
        for ev in events:
            if ev.status in ("done", "skip"):
                terminals[ev.stage] = terminals.get(ev.stage, 0) + 1
        assert set(terminals) == set(guided.STAGES)
        assert set(terminals.values()) == {1}

    def test_progress_never_decreases_inside_a_stage_and_elapsed_never_does(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=2, n_samples=2),
                backend=llm.EchoBackend(responses=echo_source_sentence),
                detector=FakeDetector(0.7),
                guide=FakeDetector(0.4),
            )
        )
        last = {}
        for ev in events:
            assert 0.0 <= ev.progress <= 1.0
            if ev.status == "start":
                last[ev.stage] = 0.0
            assert ev.progress >= last.get(ev.stage, 0.0) - 1e-9, ev
            last[ev.stage] = ev.progress
        elapsed = [ev.elapsed for ev in events]
        assert elapsed == sorted(elapsed)

    def test_exactly_one_event_carries_the_result_and_it_is_the_last(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=1, n_samples=1),
                backend=llm.EchoBackend(responses=echo_source_sentence),
                detector=False,
                guide=False,
            )
        )
        with_result = [ev for ev in events if ev.result is not None]
        assert len(with_result) == 1
        assert with_result[0] is events[-1]
        assert (events[-1].stage, events[-1].status) == ("finalize", "done")

    def test_the_score_stage_is_skipped_and_says_so_without_a_yardstick(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=1, n_samples=1),
                backend=llm.EchoBackend(responses=echo_source_sentence),
                detector=False,
                guide=False,
            )
        )
        skipped = [ev for ev in events if ev.stage == "score" and ev.status == "skip"]
        assert len(skipped) == 1
        assert "yardstick" in skipped[0].detail

    def test_the_score_stage_names_the_yardstick_when_it_runs(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=1, n_samples=1),
                backend=llm.EchoBackend(responses=echo_source_sentence),
                detector=FakeDetector(0.7),
                guide=False,
            )
        )
        start = [ev for ev in events if ev.stage == "score" and ev.status == "start"]
        # The detail names the detector that is doing the scoring.
        assert "fake" in start[0].detail

    def test_events_are_the_pipelines_own_class_not_a_lookalike(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=1, n_samples=1),
                backend=llm.EchoBackend(responses=echo_source_sentence),
                detector=False,
                guide=False,
            )
        )
        assert all(isinstance(ev, pipe.ProgressEvent) for ev in events)
        payload = events[0].as_dict()
        assert set(payload) == {
            "stage", "status", "progress", "detail", "elapsed", "round"
        }

    def test_generate_events_stream_live_rather_than_in_a_burst(self):
        """The search is 20-30 seconds of the user's life. Its events must
        arrive as each sentence finishes, not all at once when the paragraph
        does -- a callback that buffered them would render as a frozen bar."""
        backend = llm.EchoBackend(responses=echo_variant)
        events = guided.stream(
            SOURCE,
            config=guided.GuidedConfig(beam_width=1, n_samples=1),
            backend=backend,
            detector=False,
            guide=False,
        )
        seen_first_sentence = False
        for ev in events:
            if ev.stage == "generate" and "sentence 1 of" in ev.detail:
                seen_first_sentence = True
                assert len(backend.calls) == 0, (
                    "the 'sentence 1' event must reach the client before the "
                    "batch for sentence 1 is even issued"
                )
            if ev.stage == "generate" and "sentence 2 of" in ev.detail:
                assert len(backend.calls) == 1, (
                    "events are buffered: the whole paragraph ran before any "
                    "event was forwarded"
                )
                break
        assert seen_first_sentence

    def test_one_generate_event_per_sentence(self):
        events = list(
            guided.stream(
                SOURCE,
                config=guided.GuidedConfig(beam_width=1, n_samples=1),
                backend=llm.EchoBackend(responses=echo_variant),
                detector=False,
                guide=False,
            )
        )
        per_sentence = [
            ev for ev in events if ev.stage == "generate" and "sentence " in ev.detail
        ]
        assert len(per_sentence) == 3, [ev.detail for ev in per_sentence]

    def test_stage_timings_include_finalize(self):
        result = run(detector=False)
        assert [row["stage"] for row in result.stages] == list(guided.STAGES)


# ------------------------------------------------------ 5. pure helpers


class TestPureHelpers:
    def test_guide_context_full_keeps_the_sources_remaining_sentences(self):
        text = guided.guide_context("A rewritten start.", "A candidate.", ["Tail one.", "Tail two."])
        assert text == "A rewritten start. A candidate. Tail one. Tail two."

    def test_guide_context_prefix_and_sentence_modes(self):
        assert guided.guide_context(
            "Head.", "Cand.", ["Tail."], guided.GuideContext.PREFIX
        ) == "Head. Cand."
        assert guided.guide_context(
            "Head.", "Cand.", ["Tail."], guided.GuideContext.SENTENCE
        ) == "Cand."

    def test_the_guide_sees_a_full_length_document_at_every_step(self):
        """`GuideContext.FULL` keeps the source's untouched tail in view, so
        the guide always scores something document-length. Every detector here
        is document-trained (`MIN_RELIABLE_CHARS` is 250) and a 20-word prefix
        is a question it was never trained on."""
        guide = FakeDetector(0.5)
        guided.humanize_guided(
            SOURCE,
            config=guided.GuidedConfig(beam_width=1, n_samples=1),
            backend=llm.EchoBackend(responses=echo_variant),
            detector=False,
            guide=guide,
        )
        in_loop = [t for t in guide.seen[2:] if "specialties" in t]
        assert len(in_loop) >= 2, (
            "the source's last sentence must still be in view while the first "
            "sentence is being chosen"
        )

    def test_prefix_context_shows_the_guide_only_what_is_written(self):
        guide = FakeDetector(0.5)
        guided.humanize_guided(
            SOURCE,
            config=guided.GuidedConfig(
                beam_width=1, n_samples=1, guide_context=guided.GuideContext.PREFIX
            ),
            backend=llm.EchoBackend(responses=echo_variant),
            detector=False,
            guide=guide,
        )
        # seen[0] is the document at `analyze`, seen[1] the paragraph at
        # `plan`; the search's own calls start at seen[2].
        in_loop = guide.seen[2:]
        assert in_loop
        assert "specialties" not in in_loop[0]
        assert len(in_loop[0].split()) < len(SOURCE.split())

    def test_reweight_logprobs_with_zero_strength_is_the_identity(self):
        lp = [-1.0, -2.0, -3.0]
        assert guided.reweight_logprobs(lp, [0.9, 0.1, 0.5], strength=0.0) == lp

    def test_reweight_logprobs_penalises_the_guides_favourite_ai_token(self):
        out = guided.reweight_logprobs([-1.0, -1.0], [0.9, 0.1], strength=4.0)
        assert out[1] > out[0]
        assert out[1] == pytest.approx(-1.0)
        assert out[0] == pytest.approx(-1.0 - 4.0 * 0.8)

    def test_reweight_logprobs_leaves_unscored_tokens_alone(self):
        out = guided.reweight_logprobs([-1.0, -2.0], [None, 0.5], strength=4.0)
        assert out[0] == pytest.approx(-1.0)

    def test_reweight_logprobs_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            guided.reweight_logprobs([-1.0], [0.1, 0.2])

    def test_clean_sentence_strips_quotes_bullets_and_runaway_text(self):
        assert guided.clean_sentence('"Just the sentence."') == "Just the sentence."
        assert guided.clean_sentence("- A bulleted one.") == "A bulleted one."
        assert guided.clean_sentence("One. Two. Three. Four.") == "One. Two."
        assert guided.clean_sentence("First line.\nSecond line.") == "First line."
        assert guided.clean_sentence("") == ""

    def test_build_sentence_prompt_puts_the_target_last(self):
        prompt = guided.build_sentence_prompt(
            "Already written.", "Rewrite me.", "Be blunt.", "Comes next."
        )
        user = prompt.messages[-1][1]
        assert user.rstrip().endswith("Rewrite me.")
        assert "Already written." in user and "Comes next." in user

    def test_build_sentence_prompt_rejects_an_empty_target(self):
        with pytest.raises(ValueError, match="must not be empty"):
            guided.build_sentence_prompt("prefix", "   ")

    def test_sentence_moves_cycle_with_a_capped_rising_temperature(self):
        moves = guided.sentence_moves(20)
        assert len(moves) == 20
        assert all(t <= 1.4 for _m, t in moves)
        assert moves[0][0] == moves[len(guided.SENTENCE_MOVES)][0]
        assert moves[len(guided.SENTENCE_MOVES)][1] > moves[0][1]
        with pytest.raises(ValueError):
            guided.sentence_moves(0)

    def test_sentence_max_tokens_is_far_below_the_paragraph_estimate(self):
        short = "A short one."
        assert guided.sentence_max_tokens(short) < llm.estimate_max_tokens(short)
        assert 28 <= guided.sentence_max_tokens(short) <= 200

    def test_config_validation_rejects_impossible_settings(self):
        for kwargs in (
            {"beam_width": 0},
            {"n_samples": 0},
            {"mode": "nope"},
            {"guide_context": "nope"},
            {"guidance_interval": 0},
            {"top_k": 0},
        ):
            with pytest.raises(ValueError):
                guided.GuidedConfig(**kwargs).validate()
        guided.GuidedConfig().validate()

    def test_config_fanout_is_the_generations_per_sentence(self):
        assert guided.GuidedConfig(beam_width=3, n_samples=4).fanout == 12

    def test_empty_text_is_a_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            list(guided.stream("   ", detector=False, guide=False))


# ------------------------------------------------------ 6. result payload


class TestResultPayload:
    def test_payload_is_shaped_like_the_llm_pipelines(self):
        result = run(detector=FakeDetector(0.8), guide=FakeDetector(0.3))
        payload = result.as_dict()
        for key in (
            "original", "humanized", "edits", "before", "after", "summary",
            "model", "backend", "stages", "paragraphs",
        ):
            assert key in payload, key
        assert payload["edits"] == [], "a regeneration has no honest span-edit list"
        assert "guide" in payload

    def test_payload_is_json_serialisable_with_nan_handling(self):
        from humanizer.api.models import json_safe

        result = run(detector=FakeDetector(0.8), guide=FakeDetector(0.3))
        text = json.dumps(json_safe(result.as_dict()), allow_nan=False)
        assert "NaN" not in text

    def test_paragraph_rows_expose_the_search(self):
        result = run(detector=False, guide=FakeDetector(0.5))
        para = result.paragraphs[0].as_dict()
        assert para["steps"], "every step's candidates must be inspectable"
        assert para["finalists"]
        first = para["steps"][0][0]
        assert {"gates", "guide_score", "move", "temperature"} <= set(first)

    def test_summary_reports_the_budget_and_the_search_shape(self):
        result = run(
            config=guided.GuidedConfig(beam_width=2, n_samples=3),
            detector=FakeDetector(0.8),
            guide=FakeDetector(0.3),
        )
        s = result.summary
        assert s["engine"] == "guided"
        assert s["beam_width"] == 2 and s["n_samples"] == 3
        assert s["guide_calls"] > 0
        assert s["n_sentences"] >= 3
        assert 0.0 <= s["content_overlap"] <= 1.0


# ------------------------------------------------------ 7. degradation


class TestDegradation:
    def test_guided_unavailable_reason_names_a_fallback(self, monkeypatch):
        monkeypatch.setattr(
            llm, "mlx_unavailable_reason", lambda: "mlx-lm is not installed."
        )
        assert guided.guided_unavailable_reason() == "mlx-lm is not installed."

    def test_guided_available_tracks_mlx(self, monkeypatch):
        monkeypatch.setattr(llm, "mlx_available", lambda: False)
        assert guided.guided_available() is False

    def test_guide_unavailable_reason_never_raises(self, monkeypatch):
        def boom(model_name=guided.GUIDE_MODEL):
            raise OSError("no such checkpoint")

        monkeypatch.setattr(guided, "_guide_detector", boom)
        reason = guided.guide_unavailable_reason()
        assert "OSError" in reason and "no such checkpoint" in reason

    def test_a_run_without_a_guide_still_produces_gated_text(self, monkeypatch):
        """No torch means no guide. The search must still run, the gates must
        still hold, and the result must say the guide was missing."""
        monkeypatch.setattr(
            guided, "guide_unavailable_reason", lambda model_name=None: "no torch"
        )
        result = guided.humanize_guided(
            SOURCE,
            config=guided.GuidedConfig(beam_width=1, n_samples=2),
            backend=llm.EchoBackend(responses=echo_source_sentence),
            detector=False,
            guide=None,
        )
        assert result.summary["guide_available"] is False
        assert result.summary["guide_unavailable_reason"] == "no torch"
        assert result.humanized

    def test_token_mode_degrades_to_segment_without_a_guide(self):
        result = guided.humanize_guided(
            SOURCE,
            config=guided.GuidedConfig(
                beam_width=1, n_samples=1, mode=guided.GuideMode.TOKEN
            ),
            backend=llm.EchoBackend(responses=echo_source_sentence),
            detector=False,
            guide=False,
        )
        assert result.summary["mode"] == guided.GuideMode.SEGMENT
        assert "segment-guided search" in result.summary["guide_unavailable_reason"]

    def test_token_guided_sentence_raises_a_named_error_on_a_dumb_backend(self):
        with pytest.raises(RuntimeError, match="mode='segment'"):
            guided.token_guided_sentence(
                llm.EchoBackend(),
                guided.build_sentence_prompt("", "Rewrite me."),
                0.8,
                32,
                guided.GuidedConfig(),
                FakeDetector(0.5),
            )

    def test_a_detector_that_throws_is_survivable(self):
        class Exploding(FakeDetector):
            def score(self, text):
                raise RuntimeError("model file is corrupt")

        result = run(detector=Exploding(), guide=Exploding())
        assert result.after["ai_probability"] is None
        assert result.summary["verdict_flipped"] is False

    def test_the_time_budget_stops_the_search_and_keeps_the_rest_as_written(self):
        backend = llm.EchoBackend(responses=echo_source_sentence, delay=0.05)
        result = guided.humanize_guided(
            SOURCE,
            config=guided.GuidedConfig(
                beam_width=1, n_samples=1, time_budget_s=0.06
            ),
            backend=backend,
            detector=False,
            guide=False,
        )
        assert result.humanized
        assert result.summary["n_fallback_sentences"] >= 1


# ------------------------------------------------------------- 8. the API


fastapi = pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api import server as server_mod  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server_mod.create_app(web_dir=None, auth=False))


def fake_stream(text, config=None, backend=None, detector=None, guide=None):
    """Two progress events and a result, without touching a GPU."""
    yield pipe.ProgressEvent("analyze", "start", 0.0, "reading", 0.001)
    yield pipe.ProgressEvent("analyze", "done", 1.0, "scored", 0.002)
    yield pipe.ProgressEvent(
        "finalize",
        "done",
        1.0,
        "verdict unchanged (ai)",
        0.003,
        1,
        {"original": text, "humanized": text, "summary": {"verdict_flipped": False}},
    )


class TestGuidedEndpoints:
    def test_health_publishes_the_stage_list_the_frontend_renders(self, client):
        body = client.get("/api/humanize/guided/health").json()
        assert body["stages"] == list(guided.STAGES)
        assert body["modes"] == list(guided.GuideMode.ALL)

    def test_health_separates_the_guide_from_the_yardstick(self, client):
        body = client.get("/api/humanize/guided/health").json()
        assert body["guide"]["model"] == guided.GUIDE_MODEL
        assert body["yardstick"]["model"] == guided.YARDSTICK_MODEL
        assert body["guide"]["model"] != body["yardstick"]["model"]
        assert "never reported as a verdict" in body["guide"]["role"].lower()

    def test_health_is_200_even_when_nothing_can_run(self, client, monkeypatch):
        monkeypatch.setattr(
            guided, "guided_unavailable_reason", lambda: "mlx-lm is not installed."
        )
        response = client.get("/api/humanize/guided/health")
        assert response.status_code == 200
        assert response.json()["available"] is False

    def test_endpoint_degrades_to_503_naming_both_fallbacks(self, client, monkeypatch):
        monkeypatch.setattr(
            guided, "guided_unavailable_reason", lambda: "mlx-lm is not installed."
        )
        response = client.post("/api/humanize/guided", json={"text": SOURCE})
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "guided_unavailable"
        assert body["fallback_endpoint"] == "/api/humanize/llm"
        assert body["fallback_endpoint_no_gpu"] == "/api/humanize"

    def test_empty_text_is_400_not_500(self, client):
        response = client.post("/api/humanize/guided", json={"text": "   "})
        assert response.status_code == 400
        assert response.json()["error"] == "empty_text"

    def test_an_over_long_document_is_413_with_the_limit(self, client):
        response = client.post(
            "/api/humanize/guided",
            json={"text": "word " * (server_mod.GUIDED_MAX_WORDS + 5)},
        )
        assert response.status_code == 413
        assert response.json()["limit"] == server_mod.GUIDED_MAX_WORDS

    def test_an_unknown_mode_is_400(self, client, monkeypatch):
        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        response = client.post(
            "/api/humanize/guided", json={"text": SOURCE, "mode": "telepathy"}
        )
        assert response.status_code == 400
        assert response.json()["error"] == "unknown_mode"

    def test_blocking_endpoint_returns_the_payload(self, client, monkeypatch):
        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        monkeypatch.setattr(
            guided,
            "humanize_guided",
            lambda text, config=None, guide=None: guided.GuidedResult(
                original=text, humanized=text, model="fake",
                after={"ai_probability": 0.42, "label": "human"},
                guide={"probability_after": 0.99},
                summary={"verdict_flipped": True},
            ),
        )
        body = client.post("/api/humanize/guided", json={"text": SOURCE}).json()
        assert body["after"]["ai_probability"] == 0.42
        assert body["guide"]["probability_after"] == 0.99

    def test_unknown_job_id_is_404(self, client):
        response = client.get("/api/humanize/guided/stream?job=nope")
        assert response.status_code == 404
        assert response.json()["error"] == "unknown_job"

    def test_llm_routes_still_work_after_the_guided_block_was_appended(self, client):
        assert client.get("/api/humanize/llm/health").status_code == 200
        assert client.get("/api/health").status_code == 200


class TestSseWireFormat:
    def parse(self, raw):
        """SSE text -> [(event, data)]."""
        out = []
        for chunk in raw.split("\n\n"):
            if not chunk.strip():
                continue
            lines = dict(
                line.split(": ", 1) for line in chunk.splitlines() if ": " in line
            )
            out.append((lines["event"], json.loads(lines["data"])))
        return out

    def test_frames_match_the_llm_streams_documented_shape(self, client, monkeypatch):
        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        monkeypatch.setattr(guided, "stream", fake_stream)
        raw = client.post(
            "/api/humanize/guided/stream", json={"text": SOURCE}
        ).text
        frames = self.parse(raw)
        names = [name for name, _ in frames]
        assert names == ["progress", "progress", "progress", "result", "done"]
        assert frames[-1][1] == {"ok": True}
        for _name, payload in frames[:3]:
            assert set(payload) == {
                "stage", "status", "progress", "detail", "elapsed", "round"
            }

    def test_the_result_frame_is_stripped_from_the_progress_frame(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        monkeypatch.setattr(guided, "stream", fake_stream)
        frames = self.parse(
            client.post("/api/humanize/guided/stream", json={"text": SOURCE}).text
        )
        progress = [p for n, p in frames if n == "progress"]
        assert all("result" not in p for p in progress)
        result = [p for n, p in frames if n == "result"]
        assert len(result) == 1 and result[0]["humanized"] == SOURCE

    def test_a_failure_after_the_headers_is_an_error_frame_at_http_200(
        self, client, monkeypatch
    ):
        def exploding(*args, **kwargs):
            raise KeyError("something deep")
            yield  # pragma: no cover

        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        monkeypatch.setattr(guided, "stream", exploding)
        response = client.post("/api/humanize/guided/stream", json={"text": SOURCE})
        assert response.status_code == 200
        frames = self.parse(response.text)
        assert [n for n, _ in frames] == ["error"]
        assert frames[0][1]["error"] == "guided_failed"
        assert frames[0][1]["fallback_endpoint"] == "/api/humanize/llm"

    def test_a_runtime_error_names_the_fallback(self, client, monkeypatch):
        def exploding(*args, **kwargs):
            raise RuntimeError("the weights will not load")
            yield  # pragma: no cover

        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        monkeypatch.setattr(guided, "stream", exploding)
        frames = self.parse(
            client.post("/api/humanize/guided/stream", json={"text": SOURCE}).text
        )
        assert frames[0][1]["error"] == "guided_unavailable"
        assert "the weights will not load" in frames[0][1]["detail"]

    def test_a_job_streams_the_same_frames_and_is_single_use(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        monkeypatch.setattr(guided, "stream", fake_stream)
        job = client.post("/api/humanize/guided/jobs", json={"text": SOURCE}).json()
        assert job["stream_url"].startswith("/api/humanize/guided/stream?job=")
        first = client.get(job["stream_url"])
        assert [n for n, _ in self.parse(first.text)][-1] == "done"
        assert client.get(job["stream_url"]).status_code == 404

    def test_sse_frames_are_ascii_and_terminated(self, client, monkeypatch):
        monkeypatch.setattr(guided, "guided_unavailable_reason", lambda: None)
        monkeypatch.setattr(guided, "stream", fake_stream)
        raw = client.post(
            "/api/humanize/guided/stream", json={"text": "Curly “quotes” here."}
        ).text
        assert raw.endswith("\n\n")
        assert all(
            block.startswith("event: ")
            for block in raw.split("\n\n")
            if block.strip()
        )


# ------------------------------------------------------------- 9. with MLX


@requires_mlx
def test_token_guided_sentence_runs_on_a_real_backend():
    """The published variant, end to end, on one short sentence.

    Marked `requires_mlx` and kept to one sentence: token-level guidance
    cannot batch, so this is the slowest test in the file by an order of
    magnitude and it exists to prove the code path executes, not to measure
    anything.
    """
    backend = llm.default_backend("faithful")
    cfg = guided.GuidedConfig(guidance_interval=16, top_k=4, guidance_strength=4.0)
    text, stats = guided.token_guided_sentence(
        backend,
        guided.build_sentence_prompt("", "The trial began in 2019."),
        0.8,
        24,
        cfg,
        FakeDetector(scorer=lambda t: 0.5),
    )
    assert isinstance(text, str)
    assert stats["tokens"] <= 24
    assert stats["guide_calls"] >= 0
