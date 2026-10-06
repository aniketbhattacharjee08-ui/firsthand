"""Tests for the research/24 manual method: structure checks, the two
deterministic transforms, the prompt rules, and the ranking term.

The demonstration paragraph from the video is the fixture, because research/24
§1.3 records exactly what a skilled editor did to it, so every check here has a
ground truth: the *before* text must trip the checks and the *after* text must
clear them.
"""

from __future__ import annotations

import pytest

from humanizer.humanize import kriukow
from humanizer.humanize import llm
from humanizer.humanize import pipeline as pipe
from humanizer.humanize import transforms as transforms_mod
from humanizer.humanize.engine import humanize
from humanizer.humanize.transforms import (
    HumanizeConfig,
    apply_edits,
    front_trailing_clauses,
    hedge_absolutes,
    relocate_contrastive_openers,
)
from humanizer.text import Document

BEFORE = (
    "Self-esteem plays a critical role in shaping the communicative experiences "
    "of migrants using English as a second language. High self-esteem fosters "
    "confidence, which is essential for engaging in conversations, expressing "
    "needs, and participating in social, educational, and professional contexts. "
    "Conversely, low self-esteem may hinder communication by increasing anxiety, "
    "fear of judgment, and reluctance to speak, ultimately affecting language "
    "development and social integration. Migrants who perceive their English "
    "skills as inadequate may internalize feelings of inferiority, reinforcing "
    "exclusion and marginalization. Therefore, supporting self-esteem is vital "
    "not only for linguistic competence but also for promoting migrants' overall "
    "well-being and inclusion."
)

AFTER = (
    "Self-esteem can play a critical role in shaping the communicative "
    "experiences of migrants using English as a second language. It has also "
    "been shown that high self-esteem can increase confidence and, as a result, "
    "migrants' willingness to participate in various conversational exchanges. "
    "Low self-esteem, on the other hand, may affect social integration, mainly "
    "due to increased anxiety and fear of judgment. As low self-assessed English "
    "skills are believed to lead to an inferiority complex and exclusion, it is "
    "crucial to support self-esteem building to promote migrants' linguistic and "
    "social inclusion."
)

ONLY = lambda *names: HumanizeConfig(name="test", transforms=tuple(names))  # noqa: E731


# ------------------------------------------------------------ opener class


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("However, the data are thin.", kriukow.OPENER_CONNECTIVE),
        ("Conversely, low self-esteem may hinder communication.", kriukow.OPENER_CONNECTIVE),
        ("Although widely cited, this study has had little follow-up.", kriukow.OPENER_FRONTED),
        ("In 2019 the court held otherwise.", kriukow.OPENER_FRONTED),
        ("Central to this argument is the notion of consent.", kriukow.OPENER_FRONTED),
        ("This study shows a clear effect.", kriukow.OPENER_PRONOUN),
        ("It is important to note that costs rose.", kriukow.OPENER_PRONOUN),
        ("Self-esteem plays a critical role.", kriukow.OPENER_BARE),
        ("The migrants who arrived first fared best.", kriukow.OPENER_BARE),
    ],
)
def test_opener_class(sentence, expected):
    assert kriukow.opener_class(sentence) == expected


# ------------------------------------------------------------ the report


def test_before_paragraph_trips_the_checks():
    report = kriukow.structure_report(BEFORE)
    assert report["n_sentences"] == 5
    # Three unhedged absolutes: "plays a critical role", "fosters ... is
    # essential", "is vital", exactly the three sentences the presenter hedged.
    assert report["absolute_claims"] == 3
    assert "unhedged_absolutes" in kriukow.structure_flags(report)
    # "plays a critical role" and "overall well-being" are surface frames.
    assert report["surface_statements"] >= 2
    # Sentences 2, 3 and 4 close on coordinated pairs or "not only ... but also".
    assert report["paired_ending_share"] >= 0.4
    assert report["consecutive_paired_endings"] >= 1
    assert report["penalty"] > 0.15
    flags = kriukow.structure_flags(report)
    assert "paired_coordination_template" in flags
    assert "surface_statements" in flags


def test_after_paragraph_clears_most_checks():
    before = kriukow.structure_report(BEFORE)
    after = kriukow.structure_report(AFTER)
    assert after["absolute_claims"] == 0
    assert after["hedges_per_1k"] > before["hedges_per_1k"]
    # He merged the run of "X and Y" closers (3 consecutive) down to one pair.
    assert after["consecutive_paired_endings"] < before["consecutive_paired_endings"]
    assert after["penalty"] < before["penalty"]
    assert "unhedged_absolutes" not in kriukow.structure_flags(after)
    assert "paired_coordination_template" not in kriukow.structure_flags(after)


def test_repeated_first_tokens_are_counted():
    text = (
        "This study measured twelve sites over a decade. This result held in every "
        "season. This pattern was strongest in the north. Rainfall explained most of it."
    )
    report = kriukow.structure_report(text)
    assert report["same_first_token_pairs"] == 2
    assert "repeated_openers" in kriukow.structure_flags(report)


def test_uniform_beats_are_counted():
    text = (
        "The first cohort completed the survey in March. "
        "The second cohort completed the survey in April. "
        "The third cohort completed the survey in May. "
        "Response rates fell."
    )
    report = kriukow.structure_report(text)
    assert report["max_beat_run"] >= 3
    assert "uniform_beats" in kriukow.structure_flags(report)


def test_report_is_wire_safe_numbers():
    report = kriukow.structure_report(BEFORE)
    for key, value in report.items():
        if key == "opener_classes":
            assert all(isinstance(v, str) for v in value)
        else:
            assert isinstance(value, (int, float)), key
    assert kriukow.structure_report("") ["n_sentences"] == 0
    assert kriukow.structure_flags(kriukow.structure_report("One sentence only.")) == []


# ----------------------------------------------------------- hedging


def test_hedge_turns_absolute_verbs_into_hedged_ones():
    # Pad with a neutral opener and closer so the target sentences are not the
    # protected first and last sentences.
    text = (
        "The sample is described below. "
        "Self-esteem plays a critical role in communication. "
        "High self-esteem fosters confidence among migrants. "
        "The next section reports the results."
    )
    doc = Document.parse(text)
    edits = hedge_absolutes(doc, ONLY("hedge"))
    kinds = {e.kind for e in edits}
    assert kinds == {"hedge"}
    out = apply_edits(doc.text, edits)
    assert "can play a critical role" in out
    # Budget: at most one sentence in three on a four-sentence document.
    assert len(edits) <= 2
    for e in edits:
        assert e.grade_cost == transforms_mod.GRADE_COST_NONE
        assert "research/24" in e.rationale


def test_hedge_uses_appears_for_evaluative_adjectives():
    text = (
        "The sample is described below. "
        "Confidence is essential for engaging in conversation. "
        "The next section reports the results."
    )
    doc = Document.parse(text)
    out = apply_edits(doc.text, hedge_absolutes(doc, ONLY("hedge")))
    assert "Confidence appears essential" in out


def test_hedge_never_stacks_on_an_existing_hedge():
    text = (
        "The sample is described below. "
        "Low self-esteem may hinder communication in class. "
        "Self-esteem can play a role in learning. "
        "The next section reports the results."
    )
    doc = Document.parse(text)
    assert hedge_absolutes(doc, ONLY("hedge")) == []


def test_hedge_skips_numbers_quotes_and_protected_spans():
    text = (
        "Self-esteem plays a critical role in communication. "
        "Confidence improves outcomes by 20 percent in this sample. "
        "The author states that \"confidence fosters fluency\" in the interviews. "
        "Support plays a critical role at the end."
    )
    doc = Document.parse(text)
    edits = hedge_absolutes(doc, ONLY("hedge"))
    # First and last sentences are protected; the middle two carry a number and
    # a quotation. Nothing may be hedged.
    assert edits == []


def test_hedge_respects_hylands_ceiling():
    hedged = " ".join(
        ["This may be true and it might hold, which could matter and would seem likely."] * 6
    )
    text = "The sample is described below. " + hedged + " Costs rose sharply in the trial. The next section reports the results."
    doc = Document.parse(text)
    # Density is already far above 20 per 1,000 words, so no hedge is added
    # even to the unhedged sentence.
    assert hedge_absolutes(doc, ONLY("hedge")) == []


# ---------------------------------------------------- opener relocation


def test_contrastive_opener_moves_after_the_subject():
    text = (
        "The sample is described below. "
        "Conversely, low self-esteem may hinder communication by increasing anxiety. "
        "The next section reports the results."
    )
    doc = Document.parse(text)
    edits = relocate_contrastive_openers(doc, ONLY("opener"))
    assert len(edits) == 1
    out = apply_edits(doc.text, edits)
    assert "Low self-esteem, on the other hand, may hinder" in out
    assert "Conversely" not in out
    assert edits[0].kind == "opener"
    assert edits[0].grade_cost == transforms_mod.GRADE_COST_NONE


def test_however_is_kept_not_deleted():
    text = (
        "The sample is described below. "
        "However, the second cohort showed no such effect in any season. "
        "The next section reports the results."
    )
    doc = Document.parse(text)
    out = apply_edits(doc.text, relocate_contrastive_openers(doc, ONLY("opener")))
    assert "The second cohort, however, showed" in out


def test_relocation_skips_short_sentences_and_complex_subjects():
    text = (
        "The sample is described below. "
        "However, costs rose. "
        "Conversely, the group that arrived later, which was smaller, did well in every season. "
        "The next section reports the results."
    )
    doc = Document.parse(text)
    assert relocate_contrastive_openers(doc, ONLY("opener")) == []


def test_relocation_leaves_formal_connectives_to_the_strip_pass():
    text = (
        "The sample is described below. "
        "Furthermore, the second cohort showed no such effect in any season. "
        "The next section reports the results."
    )
    doc = Document.parse(text)
    assert relocate_contrastive_openers(doc, ONLY("opener")) == []


# ------------------------------------------------------ end to end


def test_engine_applies_the_method_at_balanced_and_strong():
    for level in ("balanced", "strong"):
        result = humanize(BEFORE, level)
        kinds = {e.kind for e in result.edits}
        # The first and last sentences are protected, so the hedge lands on
        # sentence 2 or 3 and the opener move on sentence 3.
        assert "opener" in kinds, (level, kinds)
        assert "hedge" in kinds, (level, kinds)
        out = result.humanized
        assert "Low self-esteem, on the other hand, may hinder" in out
        # Every number, name and citation survives (none here, so the check is
        # that the text is still five sentences of the same subject).
        assert Document.parse(out).n_sentences == 5
        assert "migrants" in out.lower()


def test_light_level_does_not_run_the_method():
    result = humanize(BEFORE, "light")
    assert not {e.kind for e in result.edits} & {"hedge", "opener"}


def test_new_tokens_stay_inside_the_closed_vocabulary():
    result = humanize(BEFORE, "strong")
    source_words = set(Document.parse(BEFORE).lower_words)
    new_words = set(Document.parse(result.humanized).lower_words) - source_words
    allowed = set(transforms_mod.CLOSED_VOCABULARY)
    for value in transforms_mod.WORD_REPLACEMENTS.values():
        allowed.update(value.lower().split())
    for value in transforms_mod.PHRASE_REPLACEMENTS.values():
        allowed.update(value.lower().split())
    assert new_words <= allowed, new_words


def test_method_never_adds_a_citation_or_a_number():
    """research/24 §2: the presenter's only real flip came from a fabricated
    citation. The deterministic method must never manufacture one."""
    out = humanize(BEFORE, "strong").humanized
    assert "(" not in out.replace("(", "", BEFORE.count("("))
    assert not any(ch.isdigit() for ch in out)


# ------------------------------------------------------------ the prompt


def test_style_contract_carries_the_method_rules():
    rendered = llm.build_prompt(BEFORE, llm.CANDIDATE_SPECS[0]).rendered()
    for phrase in (
        "Read the draft for its meaning first",
        "Hedge every claim",
        "invent a study, a source or a citation",
        "No two consecutive sentences may share a shape",
        "adds no proposition",
        "put it aside",
    ):
        assert phrase in rendered, phrase


def test_method_specs_lead_the_ladder_and_reach_the_default_run():
    specs = llm.specs_for(6, "mixed")
    faithful = [s for s in specs if s.style == "faithful"]
    constraints = " ".join(s.constraint for s in faithful)
    assert "saying it back from memory" in specs[0].persona
    assert "hedged" in constraints
    assert "different shapes" in constraints
    assert [s.index for s in llm.CANDIDATE_SPECS] == list(range(len(llm.CANDIDATE_SPECS)))


# ------------------------------------------------------------- ranking


def test_quality_report_carries_structure_and_flags():
    quality = pipe._quality(BEFORE, BEFORE)
    assert "structure" in quality
    assert "penalty" in quality["structure"]
    assert "opener_classes" not in quality["structure"]
    assert "paired_coordination_template" in quality["flags"]


def test_rank_prefers_the_structurally_better_candidate_at_equal_probability():
    """Same probability, same overlap and CV, different structure penalty."""
    cfg = pipe.PipelineConfig(quality_weight=0.5)
    worse = pipe.Candidate(0, 0, "p", "c", 0.8, scrubbed=BEFORE, ai_probability=0.9)
    better = pipe.Candidate(0, 1, "p", "c", 0.8, scrubbed=BEFORE, ai_probability=0.9)
    worse.quality = pipe._quality(BEFORE, BEFORE)
    assert worse.quality["structure"]["penalty"] > 0.2
    better.quality = dict(worse.quality)
    better.quality["structure"] = dict(worse.quality["structure"], penalty=0.0)
    assert pipe._rank_key(better, cfg) < pipe._rank_key(worse, cfg)
    # The gap is exactly the weighted penalty over the three terms.
    gap = pipe._rank_key(worse, cfg) - pipe._rank_key(better, cfg)
    assert abs(gap - 0.5 * worse.quality["structure"]["penalty"] / 3.0) < 1e-9


def test_rank_still_penalises_the_presenters_content_loss():
    """research/24 §3: his rewrite cut content overlap to 0.43. The drift term
    must outweigh the structure gain, so the ranker does not learn his one bad
    habit from his good ones."""
    cfg = pipe.PipelineConfig(quality_weight=0.5)
    faithful = pipe.Candidate(0, 0, "p", "c", 0.8, scrubbed=BEFORE, ai_probability=0.9)
    lossy = pipe.Candidate(0, 1, "p", "c", 0.8, scrubbed=AFTER, ai_probability=0.9)
    faithful.quality = pipe._quality(BEFORE, BEFORE)
    lossy.quality = pipe._quality(BEFORE, AFTER)
    assert lossy.quality["structure"]["penalty"] < faithful.quality["structure"]["penalty"]
    assert lossy.quality["content_overlap"] < 0.5
    assert pipe._rank_key(lossy, cfg) > pipe._rank_key(faithful, cfg)


def test_rank_is_probability_alone_at_zero_weight():
    cfg = pipe.PipelineConfig(quality_weight=0.0)
    cand = pipe.Candidate(0, 0, "p", "c", 0.8, scrubbed=BEFORE, ai_probability=0.42)
    cand.quality = pipe._quality(BEFORE, BEFORE)
    assert pipe._rank_key(cand, cfg) == 0.42


def test_paragraph_cv_band_is_the_measured_one():
    assert pipe.HUMAN_SENTENCE_CV == (0.30, 0.49)


# ------------------------------------------------------- clause fronting

# Three consecutive interior sentences open on "The ..." at the same beat: the
# repeated opener structure the presenter calls the most important tell.
RUN = (
    "The survey covered twelve sites across the region. "
    "The first cohort completed the questionnaire in March. "
    "The second cohort answered the same questions in April because the funding arrived late. "
    "The third cohort was surveyed in May. "
    "Response rates fell over the three rounds."
)


def test_fronting_moves_the_trailing_clause_in_a_repeated_opener_run():
    doc = Document.parse(RUN)
    edits = front_trailing_clauses(doc, ONLY("fronting"))
    assert len(edits) == 1
    assert edits[0].kind == "fronting"
    assert edits[0].grade_cost == transforms_mod.GRADE_COST_NONE
    assert "research/24" in edits[0].rationale
    out = apply_edits(doc.text, edits)
    assert (
        "Because the funding arrived late, the second cohort answered the same "
        "questions in April."
    ) in out
    # No word added or removed: the bags of lowercased words are identical.
    assert sorted(Document.parse(out).lower_words) == sorted(doc.lower_words)
    # The run is broken: one fewer adjacent pair shares a first token.
    before, after = kriukow.structure_report(RUN), kriukow.structure_report(out)
    assert after["same_first_token_pairs"] < before["same_first_token_pairs"]
    assert kriukow.opener_class("Because the funding arrived late, the cohort answered.") == kriukow.OPENER_FRONTED


def test_fronting_leaves_varied_openers_alone():
    text = (
        "The survey covered twelve sites across the region. "
        "In March, the first cohort completed the questionnaire. "
        "The second cohort answered the same questions in April because the funding arrived late. "
        "Although the third cohort was smaller, it was surveyed in May. "
        "Response rates fell over the three rounds."
    )
    assert front_trailing_clauses(Document.parse(text), ONLY("fronting")) == []


def test_fronting_skips_complements_pronoun_clauses_and_proper_noun_subjects():
    text = (
        "The survey covered twelve sites across the region. "
        "The first cohort completed the questionnaire in March. "
        # "because it ...": fronting would make the pronoun point forward.
        "The second cohort answered the same questions in April because it arrived late. "
        # "unclear if ...": a complement, not an adjunct.
        "The third cohort was unclear if the funding arrived late. "
        # A proper-noun subject cannot be lowercased mid-sentence.
        "NASA funded the fourth cohort because the state ran short. "
        "Response rates fell over the three rounds."
    )
    assert front_trailing_clauses(Document.parse(text), ONLY("fronting")) == []


def test_fronting_never_touches_the_first_or_last_sentence():
    text = (
        "The survey ran for a year because the funding arrived late. "
        "The cohort completed the questionnaire in March. "
        "The cohort answered the same questions in April because the forms were translated."
    )
    edits = front_trailing_clauses(Document.parse(text), ONLY("fronting"))
    assert edits == []


def test_engine_runs_fronting_at_balanced_and_strong_but_not_light():
    for level in ("balanced", "strong"):
        result = humanize(RUN, level)
        assert "fronting" in {e.kind for e in result.edits}, level
        assert "Because the funding arrived late, the second cohort" in result.humanized
    assert "fronting" not in {e.kind for e in humanize(RUN, "light").edits}


def test_fronting_adds_no_tokens_outside_the_closed_vocabulary():
    result = humanize(RUN, "strong")
    source_words = set(Document.parse(RUN).lower_words)
    new_words = set(Document.parse(result.humanized).lower_words) - source_words
    allowed = set(transforms_mod.CLOSED_VOCABULARY)
    for value in transforms_mod.WORD_REPLACEMENTS.values():
        allowed.update(value.lower().split())
    for value in transforms_mod.PHRASE_REPLACEMENTS.values():
        allowed.update(value.lower().split())
    assert new_words <= allowed, new_words


def test_fronting_is_its_own_pass_after_shape():
    names = [[name for name, _ in stage] for stage in transforms_mod.PASSES]
    assert names[-1] == ["fronting"]
    assert "shape" in names[-2]
