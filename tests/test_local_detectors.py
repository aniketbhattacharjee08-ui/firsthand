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

from humanizer.detectors import DetectorResult, HeuristicDetector
from humanizer.detectors.local import (
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_PERPLEXITY_MODEL,
    ClassifierDetector,
    EnsembleDetector,
    PerplexityDetector,
    backend_available,
    resolve_ai_index,
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
    assert p.model_name == DEFAULT_PERPLEXITY_MODEL
    assert c.model_name == DEFAULT_CLASSIFIER_MODEL
    # No forward pass has happened, so nothing about the model is known yet.
    assert p.ppl_center > 0


# ------------------------------------------------------------ availability


def test_available_matches_backend_and_never_raises():
    """`available()` answers a bool in every environment, extra or not."""
    for detector in (PerplexityDetector(), ClassifierDetector()):
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


# ------------------------------------------------------ perplexity detector


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
            StubDetector("heuristic", 0.5),
        ]
        weights = {"perplexity": 1.0, "classifier": 1.5, "heuristic": 0.5}
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

    def test_default_members_are_the_three_detectors(self):
        ensemble = EnsembleDetector()
        assert [m.name for m in ensemble.members] == [
            "perplexity",
            "classifier",
            "heuristic",
        ]
        # The heuristic has no dependencies, so the ensemble is always usable.
        assert ensemble.available() is True

    def test_heuristic_only_ensemble_still_scores(self):
        """Graceful degradation: no torch, still a number."""
        ensemble = EnsembleDetector(members=[HeuristicDetector()])
        result = ensemble.score(HUMAN_NARRATIVE)
        assert 0.0 <= result.ai_probability <= 1.0
        assert result.raw["n_scored"] == 1


@requires_backend
def test_real_ensemble_combines_all_three():
    result = EnsembleDetector().score(HUMAN_NARRATIVE)
    members = result.raw["members"]
    assert set(members) == {"perplexity", "classifier", "heuristic"}
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
    def test_heuristic_only_needs_no_extras(self, client):
        r = client.post(
            "/api/detect", json={"text": HUMAN_NARRATIVE, "detectors": ["heuristic"]}
        )
        assert r.status_code == 200
        rows = r.json()["detectors"]
        assert len(rows) == 1
        assert rows[0]["name"] == "heuristic"
        assert rows[0]["available"] is True
        assert 0.0 <= rows[0]["ai_probability"] <= 1.0

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

    def test_null_detectors_runs_the_default_set(self, client):
        r = client.post("/api/detect", json={"text": HUMAN_NARRATIVE})
        assert r.status_code == 200
        names = [row["name"] for row in r.json()["detectors"]]
        assert names == ["heuristic", "perplexity", "classifier"]

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
    def test_perplexity_row_exposes_burstiness(self, client):
        row = client.post(
            "/api/detect",
            json={"text": HUMAN_NARRATIVE, "detectors": ["perplexity"]},
        ).json()["detectors"][0]
        assert row["raw"]["burstiness"] > 0
        assert row["raw"]["perplexity"] > 0
        assert row["raw"]["thresholds_are_calibrated"] is False
