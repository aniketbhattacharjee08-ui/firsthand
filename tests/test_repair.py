"""Tests for the self-correcting repair stage (`humanize.repair`).

Everything runs against `llm.EchoBackend` and a fake detector. The backend
is a callable on the request, so a test can make the model "learn" from the
repair stage's instruction: it returns off-topic text until the prompt
carries the NOTES line the diagnosis chose, then a faithful rewrite. That is
the behaviour the stage exists to produce, and it needs no GPU to check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from humanizer.humanize import llm
from humanizer.humanize import pipeline as pipe
from humanizer.humanize import repair


# --------------------------------------------------------------- fixtures

PARAGRAPH = (
    "In today's rapidly evolving research landscape, machine learning plays a "
    "pivotal role in clinical decision support. Furthermore, a 2019 study by "
    "Chen et al. reported that diagnostic accuracy improved by 12% when "
    "clinicians were shown model confidence alongside each prediction [4]. It "
    "is important to note that these gains were not uniform across "
    "specialties. Moreover, the intricate tapestry of deployment constraints "
    "must be considered."
)

#: Same numbers, same name, same citation, different subject: fails only
#: `content_drift`, which is the measured dominant failure (research/24 §6.14).
DRIFT = (
    "The allotment behind the station has been planted since 2019, and Chen et al. "
    "still argue about the tomatoes, which came in 12% heavier after the new compost [4]. "
    "Slugs took the lettuces early. Nobody waters the far bed, so the beans there "
    "fail every summer, and the rhubarb does whatever it likes. The shed door sticks "
    "after rain. Someone keeps leaving the gate open and the fox gets in."
)

#: A faithful rewrite that passes every gate.
GOOD = (
    "Machine learning now plays a real part in clinical decision support. A 2019 "
    "study by Chen et al. reported that diagnostic accuracy improved by 12% when "
    "clinicians saw the model's confidence next to each prediction [4]. The gains "
    "were not uniform across specialties. Deployment constraints matter too."
)


class FakeDetector:
    def __init__(self, fn=None, default=0.9):
        self.fn = fn
        self.default = default

    def available(self):
        return True

    def score(self, text):
        value = self.fn(text) if self.fn is not None else self.default
        return SimpleNamespace(ai_probability=float(value))


def judge(text):
    """Reads the faithful rewrite as human and everything else as AI."""
    return 0.05 if "real part" in text else 0.95


def learning_backend(request):
    """Off topic until the repair stage asks for fidelity, then faithful."""
    rendered = request.prompt.rendered()
    if repair.NOTE_FIDELITY in rendered:
        return GOOD
    return DRIFT


def stubborn_backend(request):
    return DRIFT


def run(backend_fn, detector=None, **config_kwargs):
    # repair_attempts is 0 by default since 2026-09-20; these tests exercise
    # the stage, so they turn it on explicitly.
    kwargs = dict(n_candidates=4, rounds=1, time_budget_s=60.0, repair_attempts=3)
    kwargs.update(config_kwargs)
    config = pipe.PipelineConfig(**kwargs)
    events, result = [], None
    for event in pipe.stream(
        PARAGRAPH,
        config=config,
        backend=llm.EchoBackend(backend_fn),
        detector=detector if detector is not None else FakeDetector(judge),
    ):
        events.append(event)
        if event.result_object is not None:
            result = event.result_object
    return events, result


def cand(index, gates, p=None):
    c = pipe.Candidate(paragraph_index=0, index=index, persona="p", constraint="c", temperature=0.9)
    c.scrubbed = GOOD
    c.gates = list(gates)
    c.ai_probability = p
    return c


def outcome_with(*cands):
    o = pipe.ParagraphOutcome(index=0, original=PARAGRAPH, chosen=PARAGRAPH)
    o.candidates = list(cands)
    o.ai_probability_before = 0.95
    return o


# ------------------------------------------------------------- diagnosis


class TestDiagnosis:
    def test_invented_specifics_dominating_picks_restrict_particulars(self):
        o = outcome_with(
            *[cand(i, ["invariant:numbers"]) for i in range(6)],
            cand(6, [], 0.7),
            cand(7, [], 0.8),
        )
        d = repair.diagnose(o, pipe.PipelineConfig())
        assert d.category == repair.CAT_INVENTED
        assert d.text.startswith("6 of 8 candidates added figures")
        assert "the 2 faithful ones read as AI (best p 0.70)" in d.text
        assert repair.choose_action(d) == repair.ACT_RESTRICT

    def test_invented_wording_names_the_notes_when_facts_were_supplied(self):
        o = outcome_with(*[cand(i, ["invariant:dates"]) for i in range(3)])
        d = repair.diagnose(o, pipe.PipelineConfig(facts="Chen (2019)"))
        assert "not in your notes" in d.text
        assert "none was faithful" in d.text

    def test_content_drift_dominating_picks_raise_fidelity(self):
        o = outcome_with(*[cand(i, ["content_drift"]) for i in range(7)], cand(7, [], 0.6))
        d = repair.diagnose(o, pipe.PipelineConfig())
        assert d.category == repair.CAT_DRIFT
        assert "7 of 8 candidates drifted" in d.text
        assert repair.choose_action(d) == repair.ACT_FIDELITY

    def test_length_failures_pick_match_length(self):
        o = outcome_with(
            *[cand(i, ["too_short", "length_ratio"]) for i in range(5)],
            cand(5, ["content_drift"]),
        )
        d = repair.diagnose(o, pipe.PipelineConfig())
        assert d.category == repair.CAT_LENGTH
        assert repair.choose_action(d) == repair.ACT_LENGTH

    def test_passers_that_all_read_as_ai_pick_change_distribution(self):
        o = outcome_with(*[cand(i, [], 0.4 + i / 100) for i in range(5)], cand(5, ["content_drift"]))
        d = repair.diagnose(o, pipe.PipelineConfig())
        assert d.category == repair.CAT_AI_PASSERS
        assert "kept the meaning but every one reads as AI" in d.text
        assert repair.choose_action(d) == repair.ACT_DISTRIBUTION

    def test_no_dominant_cause_and_no_passer_picks_split(self):
        # One rejection per cause: nothing reaches half, nothing passed.
        o = outcome_with(
            cand(0, ["content_drift"]),
            cand(1, ["too_short"]),
            cand(2, ["invariant:entities"]),
            cand(3, ["unchanged"]),
        )
        d = repair.diagnose(o, pipe.PipelineConfig())
        assert d.category == repair.CAT_NOTHING
        assert d.text.startswith("no candidate passed the gates in 4 tries")
        assert repair.choose_action(d) == repair.ACT_SPLIT

    def test_no_candidates_at_all(self):
        d = repair.diagnose(outcome_with(), pipe.PipelineConfig())
        assert d.category == repair.CAT_NOTHING
        assert "no candidate was written" in d.text

    def test_never_the_same_action_twice_in_a_row(self):
        d = repair.Diagnosis(category=repair.CAT_DRIFT, text="")
        first = repair.choose_action(d, None)
        second = repair.choose_action(d, first)
        third = repair.choose_action(d, second)
        assert first != second and second != third
        assert {first, second, third} <= set(repair.LADDER[repair.CAT_DRIFT])

    def test_every_ladder_action_has_a_label_and_a_handler_name(self):
        for actions in repair.LADDER.values():
            for a in actions:
                assert a in repair.ACTIONS
                assert a in repair.ACTION_LABELS
                assert hasattr(repair._Repairer, a)


# --------------------------------------------------------------- the stage


def test_stages_include_repair_between_select_and_finalize():
    assert "repair" in pipe.STAGES
    i = pipe.STAGES.index("repair")
    assert pipe.STAGES[i - 1] == "select" and pipe.STAGES[i + 1] == "finalize"


def test_a_paragraph_fixed_by_a_repair_action_ends_accepted_with_a_record():
    events, result = run(learning_backend, repair_attempts=3)
    para = result.paragraphs[0]
    assert para.changed
    assert "real part" in para.chosen
    assert len(para.repairs) == 1
    rec = para.repairs[0]
    assert rec["attempt"] == 1
    assert rec["action"] == repair.ACT_FIDELITY
    assert "drifted" in rec["diagnosis"]
    assert rec["result"]["accepted"] is True
    assert rec["result"]["passed"] >= 1
    assert rec["result"]["best"] == pytest.approx(0.05)
    assert para.ai_probability_after == pytest.approx(0.05)
    assert para.fallback_reason is None

    sm = result.summary
    assert sm["repairs_attempted"] == 1
    assert sm["repairs_accepted"] == 1
    assert sm["repairs_rescued"] == 1
    assert sm["repair_log"][0]["paragraph"] == 0
    assert sm["paragraphs_rewritten"] == 1

    stages = [e.stage for e in events]
    assert "repair" in stages
    assert stages.index("repair") > stages.index("select")
    assert stages.index("repair") < stages.index("finalize")
    done = [e for e in events if e.stage == "repair" and e.status == "done"]
    assert len(done) == 1 and "rescued 1 of 1" in done[0].detail


def test_the_wire_format_carries_repairs_on_the_paragraph():
    _events, result = run(learning_backend, repair_attempts=2)
    payload = result.as_dict()
    repairs = payload["paragraphs"][0]["repairs"]
    assert isinstance(repairs, list) and repairs
    rec = repairs[0]
    assert set(rec) >= {"attempt", "diagnosis", "action", "result"}
    assert set(rec["result"]) >= {"passed", "best", "accepted"}
    assert payload["summary"]["repair_log"][0]["action"] == rec["action"]


def test_a_paragraph_that_never_passes_ships_the_closest_rewrite_with_n_records():
    """Since 2026-09-22 nothing is left unrewritten: the drifting candidate
    ships with its reason stated, and every repair attempt is on record."""
    events, result = run(stubborn_backend, repair_attempts=3)
    para = result.paragraphs[0]
    assert para.changed
    assert para.chosen != PARAGRAPH
    assert para.fallback_reason and "shipped the closest rewrite" in para.fallback_reason
    assert len(para.repairs) == 3
    assert [r["attempt"] for r in para.repairs] == [1, 2, 3]
    actions = [r["action"] for r in para.repairs]
    assert all(a != b for a, b in zip(actions, actions[1:]))
    assert all(r["result"]["accepted"] is False for r in para.repairs)
    assert result.summary["repairs_attempted"] == 3
    assert result.summary["repairs_accepted"] == 0
    assert result.summary["repairs_rescued"] == 0
    done = [e for e in events if e.stage == "repair" and e.status == "done"][0]
    assert "rescued 0 of 1" in done.detail
    # Every repair candidate is on the paragraph, not hidden.
    assert any(c.round > 1 for c in para.candidates)


def test_repair_attempts_zero_skips_the_stage():
    events, result = run(stubborn_backend, repair_attempts=0)
    repair_events = [e for e in events if e.stage == "repair"]
    assert [e.status for e in repair_events] == ["skip"]
    assert "disabled" in repair_events[0].detail
    assert result.paragraphs[0].repairs == []
    assert result.summary["repairs_attempted"] == 0
    assert result.summary["repair_log"] == []
    # The stage table still has the row, so the checklist completes.
    assert "repair" in [row["stage"] for row in result.stages]


def test_nothing_to_repair_is_a_skip():
    def good_backend(request):
        return GOOD

    events, result = run(good_backend, repair_attempts=3)
    repair_events = [e for e in events if e.stage == "repair"]
    assert [e.status for e in repair_events] == ["skip"]
    assert "nothing to repair" in repair_events[0].detail
    assert result.paragraphs[0].repairs == []


def test_the_time_budget_stops_the_stage():
    events, result = run(stubborn_backend, repair_attempts=3, repair_time_budget_s=0.0)
    para = result.paragraphs[0]
    assert para.repairs == []
    done = [e for e in events if e.stage == "repair" and e.status == "done"][0]
    assert "budget" in done.detail
    assert result.summary["repairs_attempted"] == 0


def test_an_already_human_paragraph_is_still_humanized():
    """Owner's rule (2026-09-22): the scanner calling the draft human is not a
    reason to leave it alone, because the scanner is sometimes wrong. The
    drifting candidate ships and the repair stage still works the paragraph."""
    events, result = run(stubborn_backend, detector=FakeDetector(lambda t: 0.02), repair_attempts=3)
    para = result.paragraphs[0]
    assert para.changed and para.chosen != PARAGRAPH
    assert len(para.repairs) == 3
    assert "skip" not in [e.status for e in events if e.stage == "repair"]


def test_repair_candidates_are_gated_never_relaxed():
    """A repair action's output that changes a number is rejected exactly as
    the main loop would reject it."""

    def inventing_backend(request):
        rendered = request.prompt.rendered()
        if repair.NOTE_FIDELITY in rendered:
            return GOOD.replace("12%", "15%")
        return DRIFT

    _events, result = run(inventing_backend, repair_attempts=1)
    para = result.paragraphs[0]
    # The drifting main-loop candidate ships as the fallback; the repair
    # attempt's number-changing output is still rejected, not taken over it.
    assert para.changed and "15%" not in para.chosen
    assert para.repairs[0]["result"]["passed"] == 0
    assert any("invariant:numbers" in c.gates for c in para.candidates if c.round > 1)


def test_dropping_the_sentence_with_an_invented_figure_rescues_a_candidate():
    cfg = pipe.PipelineConfig()
    c = cand(0, ["invariant:numbers"])
    c.scrubbed = GOOD + " Roughly 40 hospitals took part."
    text = repair._drop_invented_sentences(PARAGRAPH, c, cfg)
    assert text is not None
    assert "40" not in text and "Chen" in text
    assert pipe._gate_candidate(PARAGRAPH, text, cfg) == []
    # Not for a candidate that failed on anything else.
    c2 = cand(1, ["invariant:numbers", "content_drift"])
    c2.scrubbed = c.scrubbed
    assert repair._drop_invented_sentences(PARAGRAPH, c2, cfg) is None


def test_restrict_particulars_rescues_through_the_stage():
    """Main-loop candidates that only invented a figure are repaired
    deterministically by the first ladder step, with no new generation
    needed to pass."""

    def inventing_backend(request):
        return GOOD + " Roughly 40 hospitals took part."

    _events, result = run(inventing_backend, repair_attempts=2)
    para = result.paragraphs[0]
    assert para.repairs[0]["action"] == repair.ACT_RESTRICT
    assert "added figures" in para.repairs[0]["diagnosis"]
    assert "40" in para.repairs[0]["diagnosis"]
    assert para.repairs[0]["result"]["accepted"] is True
    assert para.changed and "40" not in para.chosen


def test_the_critic_is_asked_once_per_attempt_and_fed_forward():
    asked = []

    def backend_fn(request):
        rendered = request.prompt.rendered()
        if "ONE LINE:" in rendered:
            asked.append(rendered)
            return "Keep the clinicians and the 12% figure in the same sentence."
        return DRIFT

    _events, result = run(backend_fn, repair_attempts=2, repair_critic=True)
    para = result.paragraphs[0]
    assert len(para.repairs) == 2
    assert para.repairs[0]["critic"].startswith("Keep the clinicians")
    assert para.repairs[1]["notes_from_critic"] == para.repairs[0]["critic"]
    # One call for attempt 1; none after the last attempt.
    assert len(asked) == 1


def test_critic_off_by_default():
    _events, result = run(stubborn_backend, repair_attempts=1)
    assert "critic" not in result.paragraphs[0].repairs[0]


# ------------------------------------------------------------ the prompt


def test_extra_notes_land_in_the_freeform_notes_slot_and_the_chat_turn():
    free = llm.build_prompt(PARAGRAPH, llm.FREEFORM_SPECS[0], extra_notes="Match the draft's length.")
    text = free.text
    assert "NOTES:\n(none: use only what the draft already says)\n- Match the draft's length." in text
    assert text.index("Match the draft's length.") < text.rindex("\n\nHUMAN:\n")
    chat = llm.build_prompt(PARAGRAPH, llm.CANDIDATE_SPECS[0], extra_notes="Match the draft's length.")
    user = chat.messages[1][1]
    assert "NOTES FROM THE EDITOR" in user
    assert user.index("Match the draft's length.") < user.index("DRAFT TO REVISE:")


def test_shots_override_selects_and_reorders_the_exemplars():
    from humanizer.humanize.exemplars import FREEFORM_SHOTS

    short = repair._shortest_shots(2)
    assert len(short) == 2
    prompt = llm.build_prompt(PARAGRAPH, llm.FREEFORM_SPECS[0], shots=short).text
    assert prompt.count("HUMAN:") == 3  # two exemplars plus the draft's own
    swapped = repair._swapped_shots()
    assert swapped[0] == FREEFORM_SHOTS[-1]


# ------------------------------------------------------------------ config


def test_config_defaults_and_env(monkeypatch):
    cfg = pipe.PipelineConfig()
    assert cfg.repair_attempts == 0  # off by default since 2026-09-20
    assert cfg.repair_time_budget_s == 240.0
    assert cfg.repair_critic is False
    monkeypatch.setenv("HUMANIZER_REPAIR_ATTEMPTS", "5")
    monkeypatch.setenv("HUMANIZER_REPAIR_CRITIC", "true")
    cfg = pipe.PipelineConfig()
    assert cfg.repair_attempts == 5 and cfg.repair_critic is True
    with pytest.raises(ValueError):
        pipe.PipelineConfig(repair_attempts=-1)


def test_api_field_flows_into_the_config():
    pytest.importorskip("fastapi")
    from pydantic import ValidationError

    from humanizer.api.server import LlmHumanizeRequest, _llm_config

    assert _llm_config(LlmHumanizeRequest(text="A draft.")).repair_attempts == 0
    assert _llm_config(LlmHumanizeRequest(text="A draft.", repair_attempts=0)).repair_attempts == 0
    assert _llm_config(LlmHumanizeRequest(text="A draft.", repair_attempts=6)).repair_attempts == 6
    with pytest.raises(ValidationError):
        LlmHumanizeRequest(text="A draft.", repair_attempts=7)
