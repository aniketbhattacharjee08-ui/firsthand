"""Tests for the local open-weight detectors in `humanizer.detectors.local`.

Two rules shape this file.

**Nothing here may require torch to collect.** The `detectors` extra is a
~1.4GB download and the other 211 tests must pass without it, so every test
that needs a model is guarded by `requires_backend`, and the tests that can be
written against stubs (the ensemble's aggregation, the label resolver, the
availability contract) are written against stubs so they run everywhere.

**The laziness assertion is the important one.** `test_importing_package_does_not_import_torch`
runs in a fresh subprocess, because by the time the rest of this module has run
torch is in `sys.modules` and the check would pass vacuously. If that test ever
fails, `humanizer serve` starts taking five seconds and the API stops working
on machines without the extra.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from humanizer.detectors import Detector, DetectorResult
from humanizer.detectors.local import (
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_MODERN_MODEL,
    DEFAULT_PERPLEXITY_MODEL,
    HEAD_POOLED_SIGMOID,
    HEAD_SOFTMAX,
    DEFAULT_ENSEMBLE_WEIGHTS,
    ClassifierDetector,
    EnsembleDetector,
    ModernDetector,
    PerplexityDetector,
    MODEL_HEADS,
    PUBLISHED_DETECTORS,
    SHIPPED_DETECTORS,
    backend_available,
    check_load_integrity,
    resolve_ai_index,
    resolve_head,
)
from humanizer.text import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
PMC_DIR = REPO_ROOT / "data" / "raw" / "pmc"

requires_backend = pytest.mark.skipif(
    not backend_available(),
    reason="the detectors extra is not installed (pip install -e '.[detectors]')",
)

# Deliberately repetitive: the same claim restated four ways, then the block
# repeated. GPT-2 finds this almost free to predict, so its perplexity should
# be far below any real prose. This is the one perplexity ordering that is
# robust enough to assert on -- see `test_academic_prose_is_a_false_positive`
# for the ordering that is not.
REPETITIVE = (
    "The system is important. The system is important for users. "
    "The system is important for users and for the system. "
    "It is important to note that the system is important. "
) * 4

# Human narrative prose: specific, unhedged, with clause structure that does
# not repeat. This is the textbook case the 2023 method was demonstrated on.
HUMAN_NARRATIVE = (
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
)


# A templated LLM-style paragraph in the register the default detector is
# supposed to catch: hedged, tricolon-heavy, no specifics, closing synthesis.
# Distinct from REPETITIVE, which is degenerate rather than merely generic.
AI_STYLE = (
    "In today's rapidly evolving digital landscape, artificial intelligence has "
    "become an increasingly important tool for organizations of all sizes. It is "
    "important to note that the adoption of these technologies requires careful "
    "consideration of both the opportunities and the challenges involved. On the "
    "one hand, automation can significantly improve efficiency and reduce "
    "operational costs. On the other hand, organizations must ensure that "
    "appropriate safeguards are in place to protect user privacy and maintain "
    "trust. Furthermore, the ethical implications of these systems cannot be "
    "overlooked. By taking a balanced and thoughtful approach, organizations can "
    "harness the full potential of artificial intelligence while minimizing "
    "potential risks. Ultimately, success in this area depends on a commitment to "
    "transparency, accountability, and continuous improvement."
)


def pmc_paragraphs(limit=6):
    """Substantial human paragraphs from the shipped PMC corpus."""
    import re

    out = []
    for path in sorted(PMC_DIR.glob("*.txt"))[: limit * 3]:
        text = path.read_text(errors="ignore")
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if 800 < len(para) < 3000 and para.count(".") > 4:
                out.append((path.name, para))
                break
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------- laziness


def test_importing_package_does_not_import_torch():
    """`import humanizer.detectors` must not drag in torch.

    Run in a subprocess so the assertion is about a clean interpreter rather
    than about whatever this test session has already loaded. Also imports the
    `local` module by name, since that is the one with the torch calls in it.
    """
    code = (
        "import sys; "
        "import humanizer.detectors as d; "
        "import humanizer.detectors.local as l; "
        "import humanizer.api.server as s; "
        "print(','.join(sorted(m for m in ('torch', 'transformers') "
        "if m in sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    leaked = proc.stdout.strip()
    assert leaked == "", f"heavy modules imported at import time: {leaked}"


def test_constructing_detectors_does_not_load_a_model():
    """Constructors are free; `score()` is what costs 500MB."""
    p = PerplexityDetector()
    c = ClassifierDetector()
    m = ModernDetector()
    assert p.model_name == DEFAULT_PERPLEXITY_MODEL
    assert c.model_name == DEFAULT_CLASSIFIER_MODEL
    assert m.model_name == DEFAULT_MODERN_MODEL
    # 1.74GB of DeBERTa has not been touched: the head is not even resolved
    # until the config has been read inside `score()`.
    assert m.head is None
    # No forward pass has happened, so nothing about the model is known yet.
    assert p.ppl_center > 0


# ------------------------------------------------------------ availability


def test_available_matches_backend_and_never_raises():
    """`available()` answers a bool in every environment, extra or not."""
    for detector in (ModernDetector(), PerplexityDetector(), ClassifierDetector()):
        value = detector.available()
        assert isinstance(value, bool)
        if not backend_available():
            assert value is False
            assert "detectors" in (detector.unavailable_reason() or "")


def test_unavailable_reason_is_none_when_usable():
    detector = PerplexityDetector()
    if detector.available():
        assert detector.unavailable_reason() is None
    else:
        assert detector.unavailable_reason()


def test_missing_model_reports_unavailable_rather_than_raising_forever():
    """A model that cannot be fetched flips `available()` to False.

    The name is a valid-looking repo id that does not exist, so the hub returns
    a 404 offline or online. The first `score()` raises RuntimeError with a
    readable message; after that the detector reports itself unavailable
    instead of retrying the download on every request.
    """
    if not backend_available():
        pytest.skip("needs transformers to reach the failure path")
    detector = PerplexityDetector(
        model_name="humanizer-test/definitely-not-a-real-model"
    )
    with pytest.raises(RuntimeError) as exc:
        detector.score("Some text to score. And a second sentence.")
    assert "Could not load" in str(exc.value)
    assert detector.available() is False
    assert detector.unavailable_reason()


# ------------------------------------------------------ label-order resolver


class TestResolveAiIndex:
    """The classic bug this guards: assuming index 1 means "AI".

    `openai-community/roberta-base-openai-detector` publishes
    `{0: 'Fake', 1: 'Real'}`, so index 0 is the machine class. A detector that
    assumed 1 would report every score exactly inverted and still look
    plausible, which is why this is tested against the real config below as
    well as against these synthetic ones.
    """

    def test_openai_detector_order_is_fake_first(self):
        index, how = resolve_ai_index({0: "Fake", 1: "Real"})
        assert index == 0
        assert how == "id2label"

    def test_reversed_order_is_followed(self):
        assert resolve_ai_index({0: "Real", 1: "Fake"})[0] == 1

    def test_human_ai_wording(self):
        assert resolve_ai_index({0: "human", 1: "ai"})[0] == 1
        assert resolve_ai_index({0: "AI", 1: "Human"})[0] == 0

    def test_machine_generated_wording(self):
        assert resolve_ai_index({0: "human-written", 1: "machine-generated"})[0] == 1

    def test_real_does_not_match_the_ai_token(self):
        """"Real" must not match on a naive "ai" substring search."""
        index, how = resolve_ai_index({0: "Real", 1: "Fake"})
        assert (index, how) == (1, "id2label")

    def test_complement_when_only_the_human_label_is_recognised(self):
        index, how = resolve_ai_index({0: "LABEL_0", 1: "human"})
        assert index == 0
        assert how == "id2label:complement"

    def test_uninformative_labels_fall_back_to_the_known_model_map(self):
        index, how = resolve_ai_index(
            {0: "LABEL_0", 1: "LABEL_1"}, DEFAULT_CLASSIFIER_MODEL
        )
        assert index == 0
        assert how == "known-model"

    def test_uninformative_labels_and_unknown_model_say_so(self):
        index, how = resolve_ai_index({0: "LABEL_0", 1: "LABEL_1"}, "who/knows")
        assert how == "assumed"
        assert index in (0, 1)


class TestResolveHead:
    """Which classification head a checkpoint needs.

    Getting this wrong is quieter than getting the label index wrong.
    desklib's config.json advertises `num_labels: 2` with LABEL_0/LABEL_1, but
    the checkpoint's trained head is a single logit on a mean-pooled encoder.
    Loading it with `AutoModelForSequenceClassification` succeeds, discards the
    trained classifier, and returns confident-looking numbers from a randomly
    initialised head.
    """

    def test_desklib_architecture_needs_the_pooled_sigmoid_head(self):
        head, how = resolve_head(["DesklibAIDetectionModel"], DEFAULT_MODERN_MODEL)
        assert head == HEAD_POOLED_SIGMOID
        assert how == "architecture:DesklibAIDetectionModel"

    def test_an_ordinary_sequence_classifier_is_softmax(self):
        head, how = resolve_head(["RobertaForSequenceClassification"], "who/knows")
        assert head == HEAD_SOFTMAX
        assert how == "default"

    def test_no_architectures_at_all_defaults_to_softmax(self):
        assert resolve_head([], "")[0] == HEAD_SOFTMAX
        assert resolve_head(None, "")[0] == HEAD_SOFTMAX

    def test_every_published_checkpoint_is_registered_with_a_head(self):
        for model, head in MODEL_HEADS.items():
            assert head in (HEAD_SOFTMAX, HEAD_POOLED_SIGMOID), model
            assert resolve_head([], model) == (head, "known-model")


class TestCheckLoadIntegrity:
    """The guard for the desklib trap.

    `from_pretrained` will happily build a model whose classification head was
    not in the checkpoint, randomly initialise it, and return confident
    probabilities computed from noise. Nothing about that load looks like a
    failure, so the only defence is to read `missing_keys` and refuse.
    """

    def test_a_clean_load_records_the_zero_counts_as_evidence(self):
        check = check_load_integrity("some/model", [], [])
        assert check["n_missing"] == 0
        assert check["n_unexpected"] == 0
        assert check["head_weights_loaded"] is True

    def test_a_randomly_initialised_head_is_refused(self):
        with pytest.raises(RuntimeError) as exc:
            check_load_integrity("some/model", ["classifier.weight"], [])
        assert "randomly initialised" in str(exc.value)
        assert "classification head" in str(exc.value)

    def test_out_proj_counts_as_a_head_weight(self):
        """RoBERTa-style classification heads are named `...out_proj`."""
        with pytest.raises(RuntimeError):
            check_load_integrity("m", ["classifier.out_proj.bias"], [])

    def test_a_missing_encoder_weight_is_also_refused(self):
        """A half-initialised backbone is no more trustworthy than a bad head."""
        with pytest.raises(RuntimeError) as exc:
            check_load_integrity("m", ["encoder.layer.0.attention.self.query.weight"], [])
        assert "encoder" in str(exc.value)

    def test_unexpected_keys_are_recorded_but_allowed(self):
        """`roberta-large-openai-detector` ships an unused pooler. Harmless."""
        check = check_load_integrity("m", [], ["roberta.pooler.dense.bias"])
        assert check["n_unexpected"] == 1
        assert check["head_weights_loaded"] is True


class TestPublishedRegistry:
    """Every engine is somebody else's checkpoint, and carries its numbers."""

    def test_every_shipped_engine_has_a_measurement(self):
        for name, spec in PUBLISHED_DETECTORS.items():
            assert 0.0 <= spec["pairwise"] <= 1.0, name
            assert 0 <= spec["fpr"] <= 14, name
            assert 0 <= spec["tpr"] <= 14, name
            assert spec["note"], name

    def test_the_default_is_the_best_measured_shipped_engine(self):
        """No engine ships that beats the default on both axes at once."""
        default = PUBLISHED_DETECTORS["modern"]
        assert DEFAULT_MODERN_MODEL == default["model"]
        for name, spec in SHIPPED_DETECTORS.items():
            spec = PUBLISHED_DETECTORS[name]
            better = (
                spec["pairwise"] > default["pairwise"]
                and spec["fpr"] < default["fpr"]
            )
            assert not better, f"{name} beats the default and is not it"

    def test_rejected_checkpoints_keep_the_number_that_rejected_them(self):
        """A rejection with a measurement beats a silent omission."""
        rejected = {k: v for k, v in PUBLISHED_DETECTORS.items() if not v["ship"]}
        assert rejected
        # The cautionary pair: high separation, unusable false-positive rate.
        assert rejected["distilbert"]["pairwise"] > 0.9
        assert rejected["distilbert"]["fpr"] >= 13
        # And the opposite failure: safe because it never fires.
        assert rejected["piratexx"]["fpr"] == 0
        assert rejected["piratexx"]["tpr"] <= 1


# ---------------------------------------------------------- modern detector


@requires_backend
class TestModernDetector:
    """The default engine.

    Every assertion here is an ordering or a shape, never an absolute
    probability, with one exception: the AI-vs-human gap on the two fixtures is
    wide enough (1.000 vs 0.34 as measured) that asserting a 0.3 margin is a
    statement about the model working at all rather than a fitted threshold.
    """

    def test_the_head_weights_actually_loaded(self):
        """0 missing / 0 unexpected is the proof the wrapper matched.

        Loading desklib with `AutoModelForSequenceClassification` instead would
        succeed, silently discard the trained single-logit head, and score
        with a random one. This is what rules that out.
        """
        raw = ModernDetector(score_sentences=False).score(AI_STYLE).raw
        check = raw["load_check"]
        assert check["n_missing"] == 0, check["missing_keys"]
        assert check["n_unexpected"] == 0, check["unexpected_keys"]
        assert check["head_weights_loaded"] is True

    def test_it_is_flagged_as_a_published_checkpoint(self):
        assert ModernDetector.is_published_detector is True
        raw = ModernDetector(score_sentences=False).score(AI_STYLE).raw
        assert raw["is_published_detector"] is True

    def test_head_is_read_from_the_config_not_assumed(self):
        raw = ModernDetector(score_sentences=False).score(AI_STYLE).raw
        assert raw["head"] == HEAD_POOLED_SIGMOID
        assert raw["head_source"] == "architecture:DesklibAIDetectionModel"
        # A single-logit sigmoid head has no label index to resolve.
        assert raw["ai_index"] is None
        assert raw["model"] == DEFAULT_MODERN_MODEL

    def test_templated_ai_prose_outscores_human_narrative(self):
        detector = ModernDetector(score_sentences=False)
        ai = detector.score(AI_STYLE).ai_probability
        human = detector.score(HUMAN_NARRATIVE).ai_probability
        assert ai - human > 0.3, (ai, human)
        assert detector.score(AI_STYLE).label == "ai"

    def test_sentence_scores_match_the_document_segmentation(self):
        """The frontend heatmap indexes these against /api/analyze's rows."""
        result = ModernDetector().score(HUMAN_NARRATIVE)
        expected = Document.parse(HUMAN_NARRATIVE).n_sentences
        assert len(result.sentence_scores) == expected
        assert result.raw["n_sentences"] == expected
        assert result.raw["n_scored_sentences"] >= 1
        assert all(0.0 <= s <= 1.0 for s in result.sentence_scores)

    def test_sentence_scoring_can_be_switched_off(self):
        """Off still returns one row per sentence, inheriting the document."""
        result = ModernDetector(score_sentences=False).score(HUMAN_NARRATIVE)
        assert len(result.sentence_scores) == Document.parse(HUMAN_NARRATIVE).n_sentences
        assert result.raw["n_scored_sentences"] == 0
        assert set(result.sentence_scores) == {result.ai_probability}

    def test_short_sentences_inherit_the_document_score(self):
        text = "Yes. " + HUMAN_NARRATIVE
        result = ModernDetector().score(text)
        assert result.sentence_scores[0] == pytest.approx(result.ai_probability)

    def test_raw_carries_every_documented_field(self):
        raw = ModernDetector(score_sentences=False).score(HUMAN_NARRATIVE).raw
        for key in (
            "method",
            "model",
            "head",
            "head_source",
            "n_chunks",
            "chunk_probabilities",
            "mean_chunk_probability",
            "max_chunk_probability",
            "window_tokens",
            "n_tokens",
            "n_sentences",
            "n_scored_sentences",
            "sentence_scores_are_unreliable",
            "architecture_class",
            "is_gptzero",
            "gptzero_relationship",
            "thresholds_are_calibrated",
            "calibration_note",
            "below_reliable_length",
        ):
            assert key in raw, key
        assert raw["is_gptzero"] is False
        assert raw["thresholds_are_calibrated"] is False
        assert raw["sentence_scores_are_unreliable"] is True

    def test_it_never_claims_to_be_gptzero(self):
        """Non-negotiable, and asserted at the detector as well as the API."""
        raw = ModernDetector(score_sentences=False).score(AI_STYLE).raw
        assert raw["is_gptzero"] is False
        assert "proprietary" in raw["gptzero_relationship"]
        assert "No relationship" in raw["gptzero_relationship"] or (
            "None." in raw["gptzero_relationship"]
        )

    def test_long_document_is_chunked_and_every_token_is_scored(self):
        text = (HUMAN_NARRATIVE + " ") * 12
        result = ModernDetector(score_sentences=False).score(text)
        assert result.raw["n_chunks"] > 1
        assert sum(result.raw["chunk_tokens"]) == result.raw["n_tokens"]
        assert len(result.raw["chunk_probabilities"]) == result.raw["n_chunks"]
        # The headline number is the weighted mean; the max is reported beside
        # it because a partly generated document only shows up in one window.
        assert result.ai_probability == pytest.approx(
            result.raw["mean_chunk_probability"]
        )
        assert result.raw["max_chunk_probability"] >= result.ai_probability - 1e-9

    def test_short_input_is_a_single_chunk_and_flagged_as_short(self):
        result = ModernDetector(score_sentences=False).score("A short sentence here.")
        assert result.raw["n_chunks"] == 1
        assert result.raw["below_reliable_length"] is True
        assert result.confidence == "low"

    def test_empty_text_is_rejected(self):
        with pytest.raises(ValueError):
            ModernDetector().score("")

    def test_result_is_deterministic(self):
        detector = ModernDetector()
        a = detector.score(HUMAN_NARRATIVE)
        b = detector.score(HUMAN_NARRATIVE)
        assert a.ai_probability == pytest.approx(b.ai_probability)
        assert a.sentence_scores == pytest.approx(b.sentence_scores)

    def test_a_softmax_alternative_reads_its_label_order_from_the_config(self):
        """The other supported head shape, on a model that publishes labels."""
        name = SHIPPED_DETECTORS["academic"]
        detector = ModernDetector(model_name=name, score_sentences=False)
        if not detector.available():
            pytest.skip(detector.unavailable_reason() or "unavailable")
        try:
            raw = detector.score(AI_STYLE).raw
        except RuntimeError as exc:
            pytest.skip(f"could not fetch {name}: {exc}")
        assert raw["head"] == HEAD_SOFTMAX
        assert raw["ai_index_source"] == "id2label"
        assert "machine" in str(raw["ai_label"]).lower()

    def test_a_head_override_gets_its_own_cache_slot(self):
        """An override changes which weights are looked for, so it must not
        alias the default load, and a failure under it must be visible to
        `available()` -- otherwise every request retries a dead download.
        """
        plain = ModernDetector(model_name=DEFAULT_MODERN_MODEL)
        forced = ModernDetector(model_name=DEFAULT_MODERN_MODEL, head=HEAD_SOFTMAX)
        assert plain._cache_name() != forced._cache_name()
        # desklib has no softmax head, so this load cannot succeed.
        with pytest.raises(RuntimeError):
            forced.score(AI_STYLE)
        assert forced.available() is False
        assert plain.available() is True

    def test_missing_model_reports_unavailable_rather_than_raising_forever(self):
        detector = ModernDetector(model_name="humanizer-test/not-a-real-detector")
        with pytest.raises(RuntimeError) as exc:
            detector.score("Some text to score. And a second sentence.")
        assert "Could not load" in str(exc.value)
        assert detector.available() is False
        assert detector.unavailable_reason()


@requires_backend
@pytest.mark.skipif(not PMC_DIR.is_dir(), reason="PMC corpus not present")
def test_modern_detector_does_better_on_academic_prose_than_perplexity():
    """The reason the default moved, asserted rather than asserted-about.

    `test_academic_prose_is_a_false_positive` below pins the old default
    labelling genuine human PMC paragraphs as AI. This pins that the new
    default does not. Measured over 14 human paragraphs and 14 written AI-style
    paragraphs the modern detector separated them at 1.000 pairwise with 1 of
    14 human paragraphs over 0.5, against 0.740 and 13 of 14 for perplexity;
    here we check a cheaper five-paragraph slice of the same claim, so the
    assertion is "a majority of real human academic paragraphs are not
    flagged", not a reproduction of the benchmark.

    This is the failure mode that makes detectors harmful. research/01 §5:
    Liang et al. measured 61.3% false positives on TOEFL essays from the
    perplexity signal. A detector that flags formulaic or non-native human
    writing accuses real people, so the false-positive side is what is pinned.
    """
    paragraphs = pmc_paragraphs(limit=5)
    if len(paragraphs) < 3:
        pytest.skip("not enough usable PMC paragraphs")

    detector = ModernDetector(score_sentences=False)
    probabilities = [detector.score(text).ai_probability for _, text in paragraphs]
    flagged = sum(1 for p in probabilities if p >= 0.5)
    assert flagged <= len(probabilities) // 2, (
        "The modern detector started flagging human academic prose. That is "
        "the exact failure that motivated replacing the perplexity default, "
        f"so it is a regression, not a threshold to tune. probabilities="
        f"{probabilities}"
    )
    # And it must still fire on the templated AI paragraph, or the line above
    # is satisfied by a detector that simply says "human" to everything.
    assert detector.score(AI_STYLE).ai_probability >= 0.5


# ------------------------------------------------------ perplexity detector


def test_perplexity_is_labelled_as_this_repos_own_arithmetic():
    """The one engine whose scoring is not somebody else's, said out loud.

    gpt2 is published; the logistic, both centres and both slopes that turn
    its perplexities into a probability were written here and fitted to
    nothing. It stays for research comparison, so the label is what keeps it
    honest -- and it must never be in a default.
    """
    assert PerplexityDetector.is_published_detector is False
    assert ModernDetector.is_published_detector is True
    assert ClassifierDetector.is_published_detector is True

    from humanizer.api.server import DEFAULT_DETECTORS

    assert "perplexity" not in DEFAULT_DETECTORS


@requires_backend
def test_perplexity_says_so_on_the_wire_too():
    raw = PerplexityDetector().score(HUMAN_NARRATIVE).raw
    assert raw["is_published_detector"] is False
    assert "not" in raw["scoring_is_ours"].lower()


@requires_backend
class TestPerplexityDetector:
    def test_repetitive_text_has_far_lower_perplexity_than_human_prose(self):
        """The core signal, on the case where it genuinely works.

        Repetitive templated text is nearly free for GPT-2 to predict; varied
        human narrative is not. This is a large, robust gap -- measured at 2.3
        against 52.8 -- not a threshold squeaker.
        """
        detector = PerplexityDetector()
        repetitive = detector.score(REPETITIVE)
        human = detector.score(HUMAN_NARRATIVE)

        assert repetitive.raw["perplexity"] < human.raw["perplexity"]
        # Not a marginal difference: a whole order of magnitude.
        assert repetitive.raw["perplexity"] * 5 < human.raw["perplexity"]
        # Lower perplexity has to mean a higher AI probability, or the mapping
        # is wired backwards.
        assert repetitive.ai_probability > human.ai_probability
        assert repetitive.label == "ai"

    def test_burstiness_is_the_sd_of_the_sentence_perplexities(self):
        """Burstiness must be exactly what research/01 §1.2 says it is."""
        import statistics

        result = PerplexityDetector().score(HUMAN_NARRATIVE)
        values = [p for p in result.raw["sentence_perplexities"] if p is not None]
        assert len(values) >= 2
        assert result.raw["burstiness"] == pytest.approx(
            statistics.stdev(values), rel=1e-6
        )

    def test_sentence_scores_match_the_document_segmentation(self):
        """One score per sentence, indexed identically to Document.parse.

        The frontend heatmap indexes the response's sentence scores against the
        sentence list from `/api/analyze`, so any drift here silently paints
        the wrong sentences.
        """
        text = HUMAN_NARRATIVE + "\n\n" + REPETITIVE
        doc = Document.parse(text)
        result = PerplexityDetector().score(text)

        assert len(result.sentence_scores) == doc.n_sentences
        assert len(result.raw["sentence_perplexities"]) == doc.n_sentences
        assert result.raw["n_sentences"] == doc.n_sentences
        assert all(0.0 <= s <= 1.0 for s in result.sentence_scores)

    def test_short_sentences_inherit_the_document_score(self):
        """A three-word sentence has no measurable perplexity signal.

        Reporting a confident number for it would be a lie, so it falls back to
        the document probability, exactly as `api.server.sentence_rows` does.
        """
        text = "It rained. She left. He stayed."
        result = PerplexityDetector().score(text)
        assert len(result.sentence_scores) == Document.parse(text).n_sentences

        unscored = [
            i for i, p in enumerate(result.raw["sentence_perplexities"]) if p is None
        ]
        assert unscored, "expected at least one sentence below the token floor"
        for i in unscored:
            assert result.sentence_scores[i] == pytest.approx(result.ai_probability)

    def test_raw_carries_every_documented_statistic(self):
        raw = PerplexityDetector().score(HUMAN_NARRATIVE).raw
        for key in (
            "perplexity",
            "burstiness",
            "mean_sentence_perplexity",
            "max_sentence_perplexity",
            "sentence_perplexities",
            "model",
            "thresholds_are_calibrated",
        ):
            assert key in raw, key
        assert raw["model"] == DEFAULT_PERPLEXITY_MODEL
        # Never claim calibration we do not have.
        assert raw["thresholds_are_calibrated"] is False
        assert raw["min_sentence_perplexity"] <= raw["mean_sentence_perplexity"]
        assert raw["mean_sentence_perplexity"] <= raw["max_sentence_perplexity"]

    def test_result_is_deterministic(self):
        """eval mode and no_grad; two calls must agree exactly."""
        detector = PerplexityDetector()
        a = detector.score(HUMAN_NARRATIVE)
        b = detector.score(HUMAN_NARRATIVE)
        assert a.ai_probability == pytest.approx(b.ai_probability, rel=1e-9)
        assert a.raw["perplexity"] == pytest.approx(b.raw["perplexity"], rel=1e-9)

    def test_distilgpt2_override_works(self):
        """The fast model is a supported override, not just a docstring."""
        result = PerplexityDetector(model_name="distilgpt2").score(REPETITIVE)
        assert result.raw["model"] == "distilgpt2"
        assert result.raw["perplexity"] > 0

    def test_recentring_moves_the_probability_not_the_measurement(self):
        """`ppl_center` changes the mapping only; the statistics are fixed."""
        strict = PerplexityDetector(ppl_center=85.0).score(HUMAN_NARRATIVE)
        loose = PerplexityDetector(ppl_center=25.0).score(HUMAN_NARRATIVE)
        assert strict.raw["perplexity"] == pytest.approx(loose.raw["perplexity"])
        # A lower human centre means a given perplexity looks more human.
        assert loose.ai_probability < strict.ai_probability

    def test_long_document_is_windowed_not_truncated(self):
        """Every token must be scored, including those past the window."""
        long_text = HUMAN_NARRATIVE * 6
        result = PerplexityDetector(max_length=256).score(long_text)
        assert result.raw["n_tokens"] > 256
        assert result.raw["perplexity"] > 0


@requires_backend
@pytest.mark.skipif(not PMC_DIR.is_dir(), reason="PMC corpus not present")
def test_academic_prose_is_a_false_positive():
    """Pins the honest finding: this method fails on human academic prose.

    This is not a wish, it is a measurement. Human biomedical paragraphs from
    `data/raw/pmc/` are formulaic enough that GPT-2 finds them highly
    predictable, so the historical ~85 centre labels them "ai". research/01 §5
    records the same failure ("memorised or formulaic human text ... reads as
    AI to perplexity-based methods"; Liang et al. measured 61.3% false
    positives on TOEFL essays from this signal).

    The test asserts the false positives rather than asserting they are absent,
    because the alternative -- moving the threshold until this sample looks
    right -- would be fitting to twelve paragraphs and calling it calibration.
    If a future change makes this pass cleanly, that change needs a real
    labelled evaluation behind it, and this test should be replaced by one.
    """
    paragraphs = pmc_paragraphs(limit=5)
    if len(paragraphs) < 3:
        pytest.skip("not enough usable PMC paragraphs")

    detector = PerplexityDetector()
    probabilities = [detector.score(text).ai_probability for _, text in paragraphs]
    flagged = sum(1 for p in probabilities if p >= 0.5)
    assert flagged >= len(probabilities) // 2, (
        "The documented false-positive behaviour on academic prose changed. "
        f"probabilities={probabilities}"
    )


# ------------------------------------------------------ classifier detector


@requires_backend
class TestClassifierDetector:
    def test_label_order_is_read_from_the_model_config(self):
        """The real config, not a synthetic one: Fake=0, Real=1."""
        raw = ClassifierDetector().score(HUMAN_NARRATIVE).raw
        assert raw["id2label"] == {"0": "Fake", "1": "Real"}
        assert raw["ai_index"] == 0
        assert raw["ai_label"] == "Fake"
        assert raw["ai_index_source"] == "id2label"

    def test_human_narrative_scores_low_which_proves_the_order(self):
        """An end-to-end check that the mapping is not inverted.

        If `ai_index` were wrong, this human paragraph would come back at
        ~0.9998 instead of ~0.0002. A test on the index alone would not catch a
        softmax indexed on the wrong axis; this one would.
        """
        result = ClassifierDetector().score(HUMAN_NARRATIVE)
        assert result.ai_probability < 0.1
        assert result.label == "human"

    def test_explicit_ai_index_inverts_the_score(self):
        """Sanity check on the axis: forcing the other index complements it."""
        natural = ClassifierDetector().score(HUMAN_NARRATIVE)
        flipped = ClassifierDetector(ai_index=1).score(HUMAN_NARRATIVE)
        assert flipped.ai_probability == pytest.approx(
            1.0 - natural.ai_probability, abs=1e-5
        )
        assert flipped.raw["ai_index_source"] == "explicit"

    def test_long_input_is_chunked_and_every_token_is_scored(self):
        """No silent truncation at the 512-token context."""
        long_text = HUMAN_NARRATIVE * 4
        result = ClassifierDetector().score(long_text)
        raw = result.raw

        assert raw["n_chunks"] >= 2
        assert raw["n_tokens"] > raw["window_tokens"]
        # Chunks must partition the token stream exactly: nothing dropped,
        # nothing counted twice.
        assert sum(raw["chunk_tokens"]) == raw["n_tokens"]
        assert len(raw["chunk_probabilities"]) == raw["n_chunks"]
        assert all(c <= raw["window_tokens"] for c in raw["chunk_tokens"])

    def test_aggregation_reports_mean_and_max(self):
        result = ClassifierDetector().score(HUMAN_NARRATIVE * 4)
        raw = result.raw
        assert result.ai_probability == pytest.approx(raw["mean_chunk_probability"])
        assert raw["max_chunk_probability"] >= raw["mean_chunk_probability"]
        assert raw["max_chunk_probability"] == pytest.approx(
            max(raw["chunk_probabilities"])
        )

    def test_short_input_is_a_single_chunk(self):
        raw = ClassifierDetector().score(HUMAN_NARRATIVE).raw
        assert raw["n_chunks"] == 1

    def test_empty_text_is_rejected(self):
        with pytest.raises(ValueError):
            ClassifierDetector().score("")


# -------------------------------------------------------- ensemble detector

# Stubs so the aggregation logic is tested without a 1.4GB dependency. The
# arithmetic is the part worth pinning; whether gpt2 loads is tested above.


class StubDetector:
    """A detector with a fixed opinion, for testing the ensemble's maths."""

    def __init__(self, name, probability, sentence_scores=None, ok=True, boom=False):
        self.name = name
        self.probability = probability
        self.sentence_scores = list(sentence_scores or [])
        self.ok = ok
        self.boom = boom
        self.calls = 0

    def available(self):
        return self.ok

    def unavailable_reason(self):
        return None if self.ok else f"{self.name} is switched off"

    def score(self, text):
        self.calls += 1
        if self.boom:
            raise RuntimeError("model exploded")
        return DetectorResult(
            detector=self.name,
            ai_probability=self.probability,
            label="ai" if self.probability >= 0.5 else "human",
            sentence_scores=list(self.sentence_scores),
        )


class TestEnsembleDetector:
    def test_weighted_mean_of_member_probabilities(self):
        members = [
            StubDetector("perplexity", 0.9),
            StubDetector("classifier", 0.1),
            StubDetector("modern", 0.5),
        ]
        weights = {"perplexity": 1.0, "classifier": 1.5, "modern": 0.5}
        result = EnsembleDetector(members=members, weights=weights).score("text")

        expected = (1.0 * 0.9 + 1.5 * 0.1 + 0.5 * 0.5) / 3.0
        assert result.ai_probability == pytest.approx(expected)
        assert result.detector == "ensemble"

    def test_every_member_score_is_preserved_in_raw(self):
        members = [StubDetector("a", 0.8), StubDetector("b", 0.2)]
        result = EnsembleDetector(members=members, weights={"a": 1.0, "b": 1.0}).score(
            "text"
        )
        assert set(result.raw["members"]) == {"a", "b"}
        assert result.raw["members"]["a"]["ai_probability"] == pytest.approx(0.8)
        assert result.raw["members"]["b"]["ai_probability"] == pytest.approx(0.2)
        assert result.raw["members"]["a"]["weight"] == 1.0
        assert result.raw["n_scored"] == 2

    def test_unavailable_member_is_dropped_but_reported(self):
        members = [StubDetector("a", 0.8), StubDetector("b", 0.2, ok=False)]
        result = EnsembleDetector(members=members, weights={"a": 1.0, "b": 1.0}).score(
            "text"
        )
        # The unavailable member must not drag the mean toward 0.5.
        assert result.ai_probability == pytest.approx(0.8)
        assert result.raw["members"]["b"]["available"] is False
        assert "switched off" in result.raw["members"]["b"]["error"]
        assert members[1].calls == 0

    def test_a_raising_member_does_not_take_the_ensemble_down(self):
        members = [StubDetector("a", 0.8), StubDetector("b", 0.2, boom=True)]
        result = EnsembleDetector(members=members, weights={"a": 1.0, "b": 1.0}).score(
            "text"
        )
        assert result.ai_probability == pytest.approx(0.8)
        assert "model exploded" in result.raw["members"]["b"]["error"]

    def test_no_available_member_raises_rather_than_inventing_a_score(self):
        members = [StubDetector("a", 0.8, ok=False)]
        ensemble = EnsembleDetector(members=members)
        assert ensemble.available() is False
        with pytest.raises(RuntimeError):
            ensemble.score("text")

    def test_sentence_scores_are_blended_at_the_modal_length(self):
        members = [
            StubDetector("a", 0.8, sentence_scores=[0.2, 0.4]),
            StubDetector("b", 0.2, sentence_scores=[0.6, 0.8]),
            # Wrong length: a segmentation bug must not raise here.
            StubDetector("c", 0.5, sentence_scores=[0.1]),
        ]
        result = EnsembleDetector(
            members=members, weights={"a": 1.0, "b": 3.0, "c": 1.0}
        ).score("text")
        assert result.sentence_scores == pytest.approx([0.5, 0.7])

    def test_no_member_sentence_scores_gives_an_empty_list(self):
        members = [StubDetector("a", 0.8)]
        result = EnsembleDetector(members=members).score("text")
        assert result.sentence_scores == []

    def test_default_members_are_published_checkpoints_only(self):
        """The hand-written member is gone; the average is other people's."""
        ensemble = EnsembleDetector()
        assert [m.name for m in ensemble.members] == [
            "modern",
            "perplexity",
            "classifier",
        ]
        assert "heuristic" not in DEFAULT_ENSEMBLE_WEIGHTS
        # Every member now needs the extra, so without torch there is nothing
        # to average and the ensemble says so rather than inventing a number.
        assert ensemble.available() is backend_available()

    def test_a_single_member_ensemble_still_scores(self):
        """Graceful degradation: no torch, still a number."""
        ensemble = EnsembleDetector(members=[StubDetector("stub", 0.4)])
        result = ensemble.score(HUMAN_NARRATIVE)
        assert 0.0 <= result.ai_probability <= 1.0
        assert result.raw["n_scored"] == 1


@requires_backend
def test_real_ensemble_combines_the_published_members():
    result = EnsembleDetector().score(HUMAN_NARRATIVE)
    members = result.raw["members"]
    assert set(members) == {"modern", "perplexity", "classifier"}
    assert all(m["available"] for m in members.values())
    assert 0.0 <= result.ai_probability <= 1.0
    # In the default configuration only the perplexity member has sentence
    # scores, so the blend is that member's rows.
    assert len(result.sentence_scores) == Document.parse(HUMAN_NARRATIVE).n_sentences


# ------------------------------------------------------------- /api/detect

fastapi = pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(web_dir=Path("/nonexistent"))) as c:
        yield c


class TestDetectEndpoint:
    def test_the_hand_written_engine_is_gone(self, client):
        """`heuristic` was this repo's own arithmetic. It is not a detector.

        Not hidden, not disabled -- removed. Asking for it is the same error
        as asking for any other name that does not exist, and the valid-names
        list in the message is every published checkpoint plus the two
        labelled legacy engines.
        """
        r = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["heuristic"]}
        )
        assert r.status_code == 200
        row = r.json()["detectors"][0]
        assert row["available"] is False
        assert "Unknown detector" in row["error"]
        assert "heuristic" not in r.json()["known_detectors"]

    def test_response_shape_is_the_documented_contract(self, client):
        r = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["heuristic"]}
        )
        row = r.json()["detectors"][0]
        for key in (
            "name",
            "available",
            "ai_probability",
            "label",
            "sentence_scores",
            "raw",
            "error",
        ):
            assert key in row, key

    def test_null_detectors_runs_the_single_default_engine(self, client):
        """One engine by default, and it is the modern one.

        The UI shows a single score. Returning three disagreeing percentages,
        two of them from methods measured at or below chance on academic
        prose, invited callers to average them.
        """
        r = client.post("/api/detect", json={"text": HUMAN_NARRATIVE})
        assert r.status_code == 200
        names = [row["name"] for row in r.json()["detectors"]]
        assert names == ["modern"]

    def test_the_legacy_engines_are_still_reachable_by_name(self, client):
        """Kept, not deleted. The measured finding about them is the point."""
        r = client.post(
            "/api/detect",
            json={
                "text": HUMAN_NARRATIVE,
                "detectors": ["perplexity", "classifier", "ensemble"],
            },
        )
        assert r.status_code == 200
        names = [row["name"] for row in r.json()["detectors"]]
        assert names == ["perplexity", "classifier", "ensemble"]
        for row in r.json()["detectors"]:
            assert row["error"] is None or "Unknown detector" not in row["error"]

    def test_every_published_engine_is_reachable_by_name(self, client):
        """Five checkpoints, one wrapper, all addressable."""
        known = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["modern"]}
        ).json()["known_detectors"]
        for name in SHIPPED_DETECTORS:
            assert name in known

    def test_unknown_detector_is_a_row_not_a_500(self, client):
        r = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["nope"]}
        )
        assert r.status_code == 200
        row = r.json()["detectors"][0]
        assert row["available"] is False
        assert row["ai_probability"] is None
        assert "Unknown detector" in row["error"]

    def test_missing_model_never_500s(self, client):
        """The whole point of the endpoint's error contract.

        Whether or not torch is installed, a request for the model-backed
        detectors returns 200 with a usable row or a readable error string.
        """
        r = client.post(
            "/api/detect",
            json={"text": HUMAN_NARRATIVE, "detectors": ["perplexity", "classifier"]},
        )
        assert r.status_code == 200
        for row in r.json()["detectors"]:
            if row["available"]:
                assert 0.0 <= row["ai_probability"] <= 1.0
                assert row["error"] is None
            else:
                assert row["ai_probability"] is None
                assert row["error"]

    def test_empty_text_is_a_400(self, client):
        r = client.post("/api/detect", json={"text": "   "})
        assert r.status_code == 400

    def test_empty_detector_list_is_a_400(self, client):
        r = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": []}
        )
        assert r.status_code == 400

    def test_response_never_contains_nan(self, client):
        """NaN is not valid JSON and starlette renders with allow_nan=False.

        A one-sentence document makes several feature CVs NaN, and the
        perplexity detector emits NaN for a document it cannot score, so this
        is a live path rather than a hypothetical one.
        """
        r = client.post("/api/detect", json={"text": "One sentence only here."})
        assert r.status_code == 200
        assert "NaN" not in r.text
        assert "Infinity" not in r.text

    def test_disclaimer_denies_being_gptzero(self, client):
        """Non-negotiable: the wire format must not let a caller imply GPTZero."""
        body = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["heuristic"]}
        ).json()
        assert body["is_gptzero"] is False
        disclaimer = body["disclaimer"]
        assert "Not GPTZero" in disclaimer
        assert "proprietary" in disclaimer

    def test_disclaimer_names_the_default_model_and_its_relationship(self, client):
        """The disclaimer has to describe what actually ran.

        Naming the checkpoint is what makes the claim checkable: a reader can
        go and look at desklib/ai-text-detector-v1.01 and confirm both that it
        is a real fine-tuned transformer and that it is not GPTZero's.
        """
        body = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["heuristic"]}
        ).json()
        disclaimer = body["disclaimer"]
        assert "desklib/ai-text-detector-v1.01" in disclaimer
        assert "fine-tuned" in disclaimer
        assert "no relationship" in disclaimer.lower()
        assert "modern" in body["known_detectors"]

    def test_detect_reports_the_backend_capability(self, client):
        """Capability flags ride on /api/detect, not /api/health.

        /api/health's key set is pinned by an existing contract test in
        tests/test_api.py, so this endpoint carries its own flags rather than
        widening that response.
        """
        body = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["heuristic"]}
        ).json()
        assert body["backend_available"] == backend_available()
        assert "perplexity" in body["known_detectors"]

    @requires_backend
    def test_sentence_scores_align_with_the_reported_sentences(self, client):
        body = client.post(
            "/api/detect",
            json={"text": HUMAN_NARRATIVE, "detectors": ["perplexity"]},
        ).json()
        row = body["detectors"][0]
        assert row["available"] is True
        assert len(row["sentence_scores"]) == body["n_sentences"]
        assert len(body["sentences"]) == body["n_sentences"]

    @requires_backend
    def test_modern_row_carries_real_per_sentence_numbers(self, client):
        """The heatmap needs one row per sentence from the default engine."""
        body = client.post(
            "/api/detect",
            json={"text": HUMAN_NARRATIVE, "detectors": ["modern"]},
        ).json()
        row = body["detectors"][0]
        if not row["available"]:
            pytest.skip(row["error"])
        assert len(row["sentence_scores"]) == body["n_sentences"]
        assert row["raw"]["n_scored_sentences"] >= 1
        # Distinct values, i.e. actually scored rather than the document score
        # copied into every slot.
        assert len(set(row["sentence_scores"])) > 1

    @requires_backend
    def test_perplexity_row_exposes_burstiness(self, client):
        row = client.post(
            "/api/detect",
            json={"text": HUMAN_NARRATIVE, "detectors": ["perplexity"]},
        ).json()["detectors"][0]
        assert row["raw"]["burstiness"] > 0
        assert row["raw"]["perplexity"] > 0
        assert row["raw"]["thresholds_are_calibrated"] is False
