"""Tests for the humanizing engine.

Three things here are load-bearing and not ordinary unit-test hygiene:

  * **The refusal list.** research/00 §4 rates contractions, injected errors,
    broad vocabulary simplification and first-person opinion markers as medium
    detector benefit for high grade cost. A test that the engine never produces
    them is the only thing stopping a future "just add contractions, it helps"
    change from shipping.
  * **Protected spans.** research/09 rates any error inside a quotation, a
    citation or the first/last sentence as *very high* grade cost. Asserted per
    transform, not just end to end, because a transform is the thing that would
    regress.
  * **The shape band, from both sides.** research/10 found the human CV band is
    0.42-0.60 and that overshooting to 0.85 is as anomalous as undershooting to
    0.30, and that humans do *not* alternate long and short (lag-1
    autocorrelation 0.01-0.10). So the tests assert an upper bound and a lag-1
    floor, not just "CV went up".
"""

import math

import pytest

from humanizer.features import extract_features
from humanizer.features.shape import HUMAN_SENTENCE_CV, lag_autocorrelation
from humanizer.humanize import (
    CONFIGS,
    Edit,
    apply_edits,
    break_paragraph_template,
    break_parallelism,
    check_invariants,
    config_for,
    humanize,
    replace_ai_vocabulary,
    sentence_spans,
    strip_formal_connectives,
    vary_sentence_length,
)
from humanizer.humanize import engine as engine_mod
from humanizer.humanize import transforms as transforms_mod
from humanizer.text import Document

BALANCED = config_for("balanced")
STRONG = config_for("strong")


class _NoDetector:
    """Stands in for the scoring model in tests that are about edits."""

    def available(self):
        return False

    def score(self, text):  # pragma: no cover - never reached
        raise AssertionError("the stub detector must not be scored")


@pytest.fixture(scope="session")
def real_detector():
    """The published model, or a skip.

    `ModernDetector` wraps `desklib/ai-text-detector-v1.01`, ~1.74GB of weights
    fetched from the hub on first use and roughly 300ms per document on CPU.
    Only the handful of tests that genuinely measure a score ask for it.
    """
    pytest.importorskip("torch", reason="the detectors extra is not installed")
    from humanizer.detectors.local import ModernDetector

    detector = ModernDetector(score_sentences=False)
    if not detector.available():
        pytest.skip(detector.unavailable_reason() or "detector unavailable")
    try:
        detector.load()
    except Exception as exc:  # noqa: BLE001 - a missing download is not a bug
        pytest.skip(f"could not load the detector: {exc}")
    return detector


@pytest.fixture(autouse=True)
def _scoring(request, monkeypatch):
    """Route every `humanize()` call to the right scorer.

    Tests that do not ask for `real_detector` get the stub, so the suite does
    not pay a 1.74GB model load and two forward passes per call to assert
    something about a regex.
    """
    if "real_detector" in request.fixturenames:
        detector = request.getfixturevalue("real_detector")
    else:
        detector = _NoDetector()
    monkeypatch.setattr(engine_mod, "_DETECTOR", detector)


def run(transform, text, config=BALANCED):
    """Run one transform over `text` and return (edits, rewritten text)."""
    doc = Document.parse(text)
    edits = transform(doc, config)
    return edits, apply_edits(doc.text, edits)


#: An AI-flavoured document long enough for the shape features to be defined.
#: First and last sentences are plain on purpose, since both are protected and
#: would otherwise absorb every interesting case.
AI_TEXT = (
    "This paper reviews classroom technology. "
    "Furthermore, the intricate tapestry of pedagogical theory underscores the "
    "importance of rigour. "
    "Moreover, researchers must delve into the multifaceted challenges that "
    "arise when resources are constrained. "
    "It is important to note that robust frameworks are crucial for reform. "
    "Consequently, institutions should leverage comprehensive strategies. "
    "Additionally, the team utilized a meticulous protocol. "
    "Therefore, the findings are compelling.\n\n"
    "The evidence base remains uneven. "
    "Furthermore, studies in well-resourced districts report gains that "
    "smaller schools have not replicated. "
    "Teachers describe a gap between the promise of software and the reality "
    "of deployment. "
    "Moreover, training budgets are the first line cut when funding tightens. "
    "The approach is not only faster but also more accurate. "
    "Notably, several districts reported that rollouts left practice "
    "unchanged. "
    "The results were consistent, reproducible, and clear. "
    "Ultimately, the evidence base remains uneven and gains are not "
    "replicated. "
    "This review draws no further conclusions."
)


# ------------------------------------------------- 1. formal connectives


def test_connective_strip_removes_opener_and_recapitalises():
    edits, out = run(
        strip_formal_connectives,
        "A first sentence sits here. Furthermore, the results were clear "
        "enough to report. A last sentence sits here.",
    )
    assert [e.kind for e in edits] == ["connective"]
    assert "Furthermore" not in out
    assert "The results were clear enough to report." in out


def test_connective_strip_is_rated_as_raising_the_grade():
    edits, _ = run(
        strip_formal_connectives,
        "A first sentence sits here. Moreover, the second finding was also "
        "reported by the team. A last sentence sits here.",
    )
    # research/00 §4: the only edit in the matrix with a negative grade cost.
    assert edits[0].grade_cost == "negative"
    assert "research/00" in edits[0].rationale


def test_connective_strip_leaves_contrastive_connectives_alone():
    # research/10 measured *however* as more frequent in human prose than in AI
    # prose, and deleting it inverts the argument.
    for word in ("However", "Nevertheless", "Conversely", "In contrast"):
        text = (
            f"A first sentence sits here. {word}, the second finding "
            "contradicted the first one entirely. A last sentence sits here."
        )
        edits, out = run(strip_formal_connectives, text)
        assert edits == [], word
        assert word in out


def test_connective_strip_requires_a_comma_for_adverb_homographs():
    """"Overall performance" is an adjective, "Overall, performance" is glue."""
    edits, _ = run(
        strip_formal_connectives,
        "A first sentence sits here. Overall performance improved across every "
        "measured condition. A last sentence sits here.",
    )
    assert edits == []

    edits, out = run(
        strip_formal_connectives,
        "A first sentence sits here. Overall, performance improved across "
        "every measured condition. A last sentence sits here.",
    )
    assert len(edits) == 1
    assert "Performance improved across every measured condition." in out


def test_connective_strip_skips_sentences_that_would_become_stubs():
    edits, _ = run(
        strip_formal_connectives,
        "A first sentence sits here. Therefore, it failed. A last sentence "
        "sits here.",
    )
    assert edits == []


# --------------------------------------------------------- 2. vocabulary


def test_vocabulary_replaces_lexicon_words_with_curated_synonyms():
    _edits, out = run(
        replace_ai_vocabulary,
        "An opening sentence. The team utilized a meticulous protocol and "
        "leveraged robust methods to delve into the topic. A closing sentence.",
    )
    assert "utilized" not in out and "used" in out
    assert "meticulous" not in out and "careful" in out
    assert "leveraged" not in out
    assert "delve into" not in out and "examine" in out


def test_vocabulary_preserves_capitalisation():
    _edits, out = run(
        replace_ai_vocabulary,
        "An opening sentence. Meticulous work was done here by the team. A "
        "closing sentence.",
    )
    assert out.count("Careful work was done here") == 1


def test_vocabulary_repairs_article_agreement():
    """Substitution must not leave "an complex" behind."""
    _edits, out = run(
        replace_ai_vocabulary,
        "An opening sentence. The authors describe an intricate mechanism in "
        "some detail. A closing sentence.",
    )
    assert "a complex mechanism" in out
    assert "an complex" not in out


def test_vocabulary_skips_technical_senses():
    """"robust to" and "robust standard errors" are terms of art, not style."""
    _edits, out = run(
        replace_ai_vocabulary,
        "An opening sentence. The estimate is robust to misspecification and "
        "we report robust standard errors throughout. A closing sentence.",
    )
    assert "robust to misspecification" in out
    assert "robust standard errors" in out


def test_vocabulary_leaves_unmapped_words_untouched():
    """research/00 §4 rates broad simplification as *high* grade cost."""
    text = (
        "An opening sentence. The effect was significant and the efficacy of "
        "the paradigm shifted the trajectory of a nuanced debate. A closing "
        "sentence."
    )
    _edits, out = run(replace_ai_vocabulary, text)
    for word in transforms_mod._DELIBERATELY_UNMAPPED:
        assert word not in transforms_mod.WORD_REPLACEMENTS
    for word in ("significant", "efficacy", "paradigm", "trajectory", "nuanced"):
        assert word in out


def test_vocabulary_never_creates_a_repeated_word():
    _edits, out = run(
        replace_ai_vocabulary,
        "An opening sentence. Researchers use leverage points when they can "
        "and they use them often. A closing sentence.",
    )
    assert "use use" not in out


# ------------------------------------------------- 3. paragraph template


def test_template_frame_removal_keeps_the_claim():
    _edits, out = run(
        break_paragraph_template,
        "An opening sentence. It is important to note that the sample was "
        "drawn from a single hospital. A closing sentence.",
    )
    assert "It is important to note" not in out
    assert "The sample was drawn from a single hospital." in out


def test_paragraph_closer_removed_only_when_it_adds_nothing():
    redundant = (
        "An opening sentence sits here.\n\n"
        "The survey covered three regions. Response rates varied by region. "
        "Rural respondents replied less often than urban respondents. "
        "Overall, response rates varied by region.\n\n"
        "A closing sentence sits here."
    )
    edits, out = run(break_paragraph_template, redundant)
    assert [e.kind for e in edits] == ["paragraph_closer"]
    assert "Overall, response rates varied by region." not in out

    # Same shape, but the closer introduces a new claim: it must survive.
    substantive = redundant.replace(
        "Overall, response rates varied by region.",
        "Overall, nonresponse correlated with household income.",
    )
    edits, out = run(break_paragraph_template, substantive)
    assert [e for e in edits if e.kind == "paragraph_closer"] == []
    assert "household income" in out


def test_paragraph_closer_left_alone_in_short_paragraphs():
    text = (
        "An opening sentence sits here.\n\n"
        "The survey covered three regions. Overall, the survey covered three "
        "regions.\n\n"
        "A closing sentence sits here."
    )
    edits, _ = run(break_paragraph_template, text)
    assert [e for e in edits if e.kind == "paragraph_closer"] == []


# --------------------------------------------------------- 4. parallelism


def test_negative_parallelism_is_flattened_keeping_both_terms():
    _edits, out = run(
        break_parallelism,
        "An opening sentence. The method is not only faster but also more "
        "accurate than the baseline. A closing sentence.",
    )
    assert "not only" not in out
    assert "The method is faster and more accurate than the baseline." in out


def test_negative_parallelism_skips_subject_auxiliary_inversion():
    text = (
        "An opening sentence. Not only does the model converge faster, but it "
        "also generalises better. A closing sentence."
    )
    edits, out = run(break_parallelism, text)
    assert [e for e in edits if e.kind == "negative_parallel"] == []
    assert "Not only does the model converge" in out


def test_tricolon_becomes_an_unequal_list_and_keeps_every_item():
    _edits, out = run(
        break_parallelism,
        "An opening sentence. The results were consistent, reproducible, and "
        "clear. A closing sentence.",
    )
    assert "consistent and reproducible, as well as clear" in out
    for item in ("consistent", "reproducible", "clear"):
        assert item in out


def test_tricolon_left_alone_when_the_list_modifies_a_following_head():
    """Measured failure: rewriting a premodifier stack produced garbage."""
    text = (
        "An opening sentence. The plan called for meaningful, sustainable, "
        "and equitable reform. A closing sentence."
    )
    edits, out = run(break_parallelism, text)
    assert [e for e in edits if e.kind == "tricolon"] == []
    assert "meaningful, sustainable, and equitable reform" in out


def test_tricolon_left_alone_for_author_lists():
    text = (
        "An opening sentence. The replication was attempted by Smith, Jones, "
        "and Lee. A closing sentence."
    )
    edits, _ = run(break_parallelism, text)
    assert [e for e in edits if e.kind == "tricolon"] == []


# ---------------------------------------------------- 5. protected spans


def test_no_transform_touches_a_protected_span():
    text = (
        "An opening sentence sits here to be protected.\n\n"
        'The author wrote that "moreover, we must delve into the intricate '
        'tapestry of results" in the preface. Furthermore, the sample was '
        "small (Smith et al., 2019). Later work leveraged a robust design "
        "[12]. Additionally, the effect was clear.\n\n"
        "Ultimately, a closing sentence must survive intact."
    )
    doc = Document.parse(text)
    protected = doc.protected_spans()
    for transform in transforms_mod.TRANSFORMS.values():
        for edit in transform(doc, STRONG):
            for lo, hi in protected:
                assert not (edit.start < hi and lo < edit.end), (
                    transform.__name__,
                    edit,
                )


def test_first_and_last_sentence_survive_the_full_pipeline():
    result = humanize(AI_TEXT, "strong")
    assert result.humanized.startswith("This paper reviews classroom technology.")
    assert result.humanized.endswith("This review draws no further conclusions.")


def test_quotation_and_citations_survive_the_full_pipeline():
    text = (
        "An opening sentence sits here.\n\n"
        'Furthermore, one reviewer noted that "the intricate tapestry of '
        'evidence is compelling" in the report. Moreover, the sample was '
        "small (Smith et al., 2019). Additionally, later work leveraged a "
        "robust design [12].\n\n"
        "A closing sentence sits here."
    )
    out = humanize(text, "strong").humanized
    assert '"the intricate tapestry of evidence is compelling"' in out
    assert "(Smith et al., 2019)" in out
    assert "[12]" in out


# ----------------------------------------------------- 6. the meaning gate


def test_invariants_catch_a_mangled_number():
    before = "The cohort included 1,204 patients over 18 months."
    after = "The cohort included 1,240 patients over 18 months."
    assert "numbers" in check_invariants(before, after)
    assert check_invariants(before, before) == []


def test_invariants_catch_a_mangled_entity_and_citation():
    before = "We follow the Kaplan Meier method here (Smith et al., 2019)."
    assert "entities" in check_invariants(
        before, "We follow the Kaplan Mayer method here (Smith et al., 2019)."
    )
    assert "citations" in check_invariants(
        before, "We follow the Kaplan Meier method here (Smith et al., 2018)."
    )


def test_invariants_ignore_sentence_initial_capitalisation():
    """Recapitalising after a deleted connective is not entity corruption."""
    before = "An opening line. Furthermore, the trial continued for a year."
    after = "An opening line. The trial continued for a year."
    assert "entities" not in check_invariants(before, after)


def test_gate_drops_an_edit_that_breaks_an_invariant():
    base = "An opening line. The cohort included 1,204 patients. A closing line."
    bad = Edit(
        kind="test",
        start=base.index("1,204"),
        end=base.index("1,204") + 5,
        before="1,204",
        after="1,240",
        sentence_index=1,
        rationale="deliberately wrong",
        grade_cost="none",
    )
    accepted, dropped = engine_mod._gate(base, [bad])
    assert accepted == []
    assert dropped == [bad]


def test_overlapping_edits_keep_the_earlier_one():
    edits = [
        (0, Edit("a", 0, 10, "x", "y", 0, "r", "none")),
        (1, Edit("b", 5, 15, "x", "y", 0, "r", "none")),
        (1, Edit("c", 20, 25, "x", "y", 0, "r", "none")),
    ]
    kept = engine_mod._resolve_overlaps(edits)
    assert [e.kind for e in kept] == ["a", "c"]


# ------------------------------------------------------------- 7. shape


UNIFORM = " ".join(
    f"The {name} analysis produced an estimate that the research team recorded "
    f"in the shared table, and this estimate was reviewed by two readers in "
    f"round {i}."
    for i, name in enumerate(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    )
)


def _sentence(n_words: int) -> str:
    """A neutral sentence of exactly `n_words` words, with no rewrite hooks."""
    return "The " + " ".join(["team"] * (n_words - 2)) + " reported."


def test_shape_is_a_noop_when_cv_is_already_in_band():
    # Lengths chosen so the CV lands mid-band (0.42-0.60, research/10).
    text = " ".join(_sentence(n) for n in (8, 22, 15, 30, 12, 19, 9, 25, 17, 33))
    feats = extract_features(text)
    assert HUMAN_SENTENCE_CV[0] <= feats["sent_len_cv"] <= HUMAN_SENTENCE_CV[1]
    assert vary_sentence_length(Document.parse(text), STRONG) == []


def test_shape_is_a_noop_on_short_documents():
    assert vary_sentence_length(Document.parse("One. Two. Three. Four."), STRONG) == []


def test_shape_moves_cv_toward_the_band_without_overshooting():
    before = extract_features(UNIFORM)["sent_len_cv"]
    result = humanize(UNIFORM, "strong")
    after = result.after["features"]["sent_len_cv"]
    assert any(e.kind.startswith("sentence_") for e in result.edits)
    assert after > before
    assert after <= HUMAN_SENTENCE_CV[1]


def test_shape_never_overshoots_the_band_on_any_sample():
    for text in (UNIFORM, AI_TEXT):
        result = humanize(text, "strong")
        if any(e.kind.startswith("sentence_") for e in result.edits):
            assert result.after["features"]["sent_len_cv"] <= HUMAN_SENTENCE_CV[1]


def test_shape_does_not_introduce_alternating_long_and_short():
    """research/10: human lag-1 autocorrelation is 0.01-0.10, never strongly
    negative. A humanizer that alternates long and short is itself detectable."""
    for text in (UNIFORM, AI_TEXT):
        result = humanize(text, "strong")
        if not any(e.kind.startswith("sentence_") for e in result.edits):
            continue
        lengths = Document.parse(result.humanized).sentence_lengths
        lag1 = lag_autocorrelation(lengths, 1)
        if math.isfinite(lag1):
            assert lag1 >= -0.15


def test_shape_only_runs_at_strong():
    assert "shape" in CONFIGS["strong"].transforms
    assert "shape" not in CONFIGS["balanced"].transforms
    assert "shape" not in CONFIGS["light"].transforms
    balanced = humanize(UNIFORM, "balanced")
    assert not any(e.kind.startswith("sentence_") for e in balanced.edits)


# ------------------------------------------------------- 8. refusal list


def test_engine_never_introduces_contractions():
    """research/00 §4: medium-high detector benefit, *medium* grade cost,
    verdict "Never in body"."""
    for level in ("light", "balanced", "strong"):
        out = humanize(AI_TEXT, level).humanized
        before = extract_features(AI_TEXT).get("contraction_per_1k", 0.0)
        after = extract_features(out).get("contraction_per_1k", 0.0)
        assert after <= before


def test_engine_never_introduces_opinion_markers():
    for level in ("light", "balanced", "strong"):
        out = humanize(AI_TEXT, level).humanized.lower()
        for marker in ("i think", "i feel", "i believe", "in my opinion"):
            assert marker not in out


def test_engine_never_injects_surface_errors():
    """Every edit is either a deletion of glue, a mapped substitution, or a
    sentence-boundary move. Nothing produces a token the input did not have,
    except the small closed set the transforms are allowed to write."""
    result = humanize(AI_TEXT, "strong")
    source_words = set(Document.parse(AI_TEXT).lower_words)
    allowed = set()
    for value in transforms_mod.WORD_REPLACEMENTS.values():
        allowed.update(value.lower().split())
    for value in transforms_mod.PHRASE_REPLACEMENTS.values():
        allowed.update(value.lower().split())
    allowed.update(transforms_mod.CLOSED_VOCABULARY)
    new_words = set(Document.parse(result.humanized).lower_words) - source_words
    assert new_words <= allowed, new_words


def test_refused_transforms_are_absent_from_the_registry():
    names = set(transforms_mod.TRANSFORMS)
    for refused in (
        "contraction",
        "typo",
        "error_injection",
        "simplify",
        "subordination",
        "first_person",
        "fragment",
    ):
        assert refused not in names


def test_refusal_list_is_documented_with_reasons():
    doc = transforms_mod.__doc__
    assert doc is not None
    for phrase in (
        "REFUSED",
        "Contractions",
        "agreement",
        "Simplifying vocabulary broadly",
        "subordination",
        "phrasal syntax",
        "in my opinion",
        "dropping evidence",
    ):
        assert phrase in doc


# ------------------------------------------------------- 9. engine shape


def test_edit_records_every_contract_field():
    result = humanize(AI_TEXT, "balanced")
    assert result.edits
    for edit in result.edits:
        row = edit.as_dict()
        assert set(row) == {
            "kind",
            "before",
            "after",
            "sentence_index",
            "rationale",
            "grade_cost",
        }
        assert row["rationale"].startswith("research/")
        assert row["grade_cost"] in {"none", "negative"}
        assert isinstance(row["sentence_index"], int)


def test_result_as_dict_is_exactly_the_api_contract():
    payload = humanize(AI_TEXT, "balanced").as_dict()
    assert set(payload) == {
        "original",
        "humanized",
        "edits",
        "before",
        "after",
        "summary",
    }
    assert set(payload["before"]) == {"ai_probability", "label", "features", "yardstick"}
    assert set(payload["after"]) == {"ai_probability", "label", "features", "yardstick"}
    assert set(payload["summary"]) == {"n_edits", "risk_delta"}
    assert payload["summary"]["n_edits"] == len(payload["edits"])


def test_levels_enable_a_growing_set_of_transforms():
    light = set(CONFIGS["light"].transforms)
    balanced = set(CONFIGS["balanced"].transforms)
    strong = set(CONFIGS["strong"].transforms)
    assert light < balanced < strong
    assert light == {"connective", "vocabulary"}


def test_unknown_aggressiveness_is_rejected():
    with pytest.raises(ValueError):
        humanize("Some text here.", "nuclear")


def test_empty_and_tiny_inputs_do_not_raise():
    assert humanize("").humanized == ""
    assert humanize("Short.").summary["n_edits"] == 0


def test_running_twice_changes_almost_nothing_the_second_time():
    first = humanize(AI_TEXT, "strong")
    second = humanize(first.humanized, "strong")
    assert len(first.edits) >= 8
    assert len(second.edits) <= 1


def test_sentence_spans_line_up_with_the_document_text():
    doc = Document.parse(AI_TEXT)
    spans = sentence_spans(doc)
    assert len(spans) == len(doc.sentences)
    for sent, (lo, hi) in zip(doc.sentences, spans):
        assert doc.text[lo:hi] == sent.text


def test_humanizing_reduces_the_ai_lexicon_and_connective_load():
    """Feature-level effect, which needs no model at all."""
    result = humanize(AI_TEXT, "strong")
    before, after = result.before["features"], result.after["features"]
    assert after["ai_vocab_weighted_per_1k"] < before["ai_vocab_weighted_per_1k"]
    assert after["formal_connective_per_1k"] < before["formal_connective_per_1k"]


HUMAN_TEXT = (
    "Diagnosing cancer, as a life-threatening event, in children and "
    "adolescents stops the normal course of life for all family members. "
    "Families who have children diagnosed with cancer deal with distressing "
    "experiences. Given that the family is one of the primary caregivers of "
    "the child, the social and economic pressures caused by the disease have "
    "an effect on family life. Parents of children with cancer, especially "
    "mothers as primary caregivers, bear a heavy burden of care. Health-care "
    "providers usually focus on sick children and consider parents as "
    "assistants in the treatment process. Parents experience a difficult and "
    "distressing process and need help as well as support. Assessing parental "
    "stress is important because children can receive anxiety from their "
    "parents."
)


def test_human_prose_is_left_almost_alone():
    """Running the engine over real human academic prose must be close to a
    no-op; the failure mode to avoid is a tool that damages good writing."""
    result = humanize(HUMAN_TEXT, "strong")
    assert len(result.edits) <= 2


def test_score_is_null_and_delta_is_null_when_the_model_cannot_run():
    """Degraded, not broken: the rewrite is still worth returning."""
    result = humanize(AI_TEXT, "balanced")  # the stub detector is installed
    assert result.before["ai_probability"] is None
    assert result.after["ai_probability"] is None
    assert result.summary["risk_delta"] is None
    assert result.edits


def test_measured_score_drops_on_ai_text(real_detector):
    """The honest end-to-end check, against a published pretrained detector.

    `ModernDetector` is `desklib/ai-text-detector-v1.01`, a fine-tuned
    DeBERTa-v3-large trained on RAID. No score in this assertion is one this
    repo invented.
    """
    result = humanize(AI_TEXT, "strong")
    assert result.before["ai_probability"] is not None
    assert result.after["ai_probability"] is not None
    assert result.summary["risk_delta"] < 0


def test_measured_score_of_human_prose_is_not_damaged(real_detector):
    """The safety check. Human academic prose must not be pushed toward AI."""
    result = humanize(HUMAN_TEXT, "strong")
    assert result.summary["risk_delta"] <= 0.05


def test_dropped_edits_are_reported_not_silently_swallowed():
    result = humanize(AI_TEXT, "strong")
    assert isinstance(result.dropped, list)
    assert all(isinstance(e, Edit) for e in result.dropped)


# ---------------------------------------------------------- 10. endpoint


fastapi = pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(auth=False))


def test_endpoint_returns_the_documented_body(client):
    response = client.post(
        "/api/humanize", json={"text": AI_TEXT, "aggressiveness": "balanced"}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "original",
        "humanized",
        "edits",
        "before",
        "after",
        "summary",
    }
    assert set(body["summary"]) == {"n_edits", "risk_delta"}
    assert set(body["before"]) == {"ai_probability", "label", "features", "yardstick"}
    for edit in body["edits"]:
        assert set(edit) == {
            "kind",
            "before",
            "after",
            "sentence_index",
            "rationale",
            "grade_cost",
        }


def test_endpoint_rejects_empty_text(client):
    assert client.post("/api/humanize", json={"text": "   "}).status_code == 400
    assert client.post("/api/humanize", json={}).status_code == 400


def test_endpoint_rejects_an_unknown_level(client):
    response = client.post(
        "/api/humanize", json={"text": AI_TEXT, "aggressiveness": "nuclear"}
    )
    assert response.status_code == 400
    assert "aggressiveness" in response.json()["detail"]


def test_endpoint_defaults_to_balanced(client):
    response = client.post("/api/humanize", json={"text": AI_TEXT})
    assert response.status_code == 200
    assert response.json()["summary"]["n_edits"] > 0


def test_endpoint_emits_no_nan_tokens(client):
    """NaN is not valid JSON and starlette renders with allow_nan=False; a
    one-sentence document has a NaN sentence-length SD."""
    response = client.post("/api/humanize", json={"text": "A single sentence."})
    assert response.status_code == 200
    assert "NaN" not in response.text
    assert "Infinity" not in response.text


def test_invariants_treat_the_modal_may_and_chemical_formulae_as_words():
    from humanizer.humanize.engine import invariants

    src = "Emissions of CO2 rose in May 2019, and the trend may continue."
    # "May" the month counts once; the modal does not.
    assert invariants(src)["dates"] == {"may": 1}
    # "CO2" is not the number 2; "2019" is a number.
    assert invariants(src)["numbers"] == {"2019": 1}
    # A rewrite that drops the hedge, or adds one, keeps the invariants.
    assert invariants("Emissions of CO2 rose in May 2019, and the trend continues.")["dates"] == {"may": 1}
    # A trailing comma is punctuation, not part of the number.
    assert invariants("In 2017, costs fell.")["numbers"] == {"2017": 1}


def test_entities_break_on_punctuation():
    from humanizer.humanize.engine import _entities

    got = set(_entities("Gig platforms such as Uber, DoorDash and Upwork grew; the United Nations Environment Programme (UNEP) noted it."))
    assert {"Uber", "DoorDash", "Upwork", "United Nations Environment Programme", "UNEP"} <= got
    assert not any("Uber DoorDash" in e or "Programme UNEP" in e for e in got)


def test_entities_keep_digits_and_ampersands_inside_a_token():
    from humanizer.humanize.engine import _entities, invariants

    text = "Spending on R&D and on V2G charging rose; the H1N1 season was mild."
    got = set(_entities(text))
    assert {"R&D", "V2G", "H1N1"} <= got
    assert not any(e in {"R", "D", "V", "G", "H"} for e in got)
    # Deterministic across calls.
    assert invariants(text) == invariants(text)
