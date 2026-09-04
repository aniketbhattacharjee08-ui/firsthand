"""HTTP API tests.

Uses FastAPI's TestClient, which drives the ASGI app in-process, so nothing
here binds a port.

Two things are worth more than the usual smoke coverage:

  * NaN handling. Feature dicts genuinely contain NaN (a one-sentence document
    has no sentence-length SD), NaN is not valid JSON, and starlette renders
    with allow_nan=False. `test_nan_becomes_null_*` assert both that the field
    is null and that the literal token never reaches the wire.
  * The per-sentence risk contract. research/07 says the document verdict is
    not an aggregate of sentence scores, so the advisory flag is part of the
    API contract and is asserted, not assumed.
"""

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api.models import json_safe  # noqa: E402
from humanizer.api.server import create_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"

# ~330 words of plausibly AI-flavoured academic prose. Long enough that the
# shape features are defined and the report does not emit its "under 300 words"
# caveat for everything.
SAMPLE = (
    "In today's rapidly evolving research landscape, it is important to note "
    "that computational methods play a pivotal role in shaping outcomes. "
    "Furthermore, the intricate tapestry of statistical technique underscores "
    "the importance of methodological rigour across disciplines. Moreover, "
    "researchers must delve into the multifaceted challenges that arise when "
    "sample sizes remain small. Additionally, robust analytical frameworks are "
    "crucial for reproducible success in the modern academy.\n\n"
    "Consequently, stakeholders should leverage comprehensive strategies that "
    "align incentives across institutions. The study examined a narrow "
    "question. It did so carefully, and at some length, across a sequence of "
    "controlled experimental conditions that were specified in advance by the "
    "preregistration document. Researchers have shown that the effect is "
    "reasonably robust to reasonable perturbations of the analytic pipeline; "
    "others disagree, and their objections deserve a hearing. We therefore "
    "propose a model that accommodates both readings without collapsing the "
    "distinction between them.\n\n"
    "A third paragraph follows here to give the paragraph-level features "
    "something to measure. It adds detail, evidence, and commentary in roughly "
    "equal measure. The result may indicate something important about how "
    "measurement error propagates through a multi-stage estimation procedure, "
    "though we are careful not to overstate the strength of the inference. "
    "Notably, the observed variance was substantially larger than the "
    "simulation had anticipated, which suggests that the generative assumptions "
    "were not fully satisfied by the empirical data.\n\n"
    "Finally, we conclude by observing that the framework proposed here is not "
    "a replacement for careful theory but a complement to it. Future work "
    "should extend the analysis to larger corpora, examine sensitivity to the "
    "choice of prior, and report the full distribution of outcomes rather than "
    "a single summary statistic. In short, more work remains to be done before "
    "any strong claim can be defended. The limitations enumerated above are "
    "not incidental to the design; they follow directly from the decision to "
    "prioritise internal validity over generalisability, and a differently "
    "specified study would trade one for the other. Readers should weigh the "
    "evidence accordingly."
)

SHORT = "Rain fell. The meeting ended early because of it."
ONE_SENTENCE = "A single solitary sentence with no companion at all."


@pytest.fixture(scope="module")
def client():
    """App with the repo's real references and no web/ directory."""
    app = create_app(
        reference_dir=REFERENCE_DIR,
        web_dir=REPO_ROOT / "does-not-exist-web",
    )
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_shape(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["version"], str) and body["version"]
        assert set(body) == {"status", "version", "syntax_backend", "references"}

    def test_syntax_backend_is_a_bool_even_when_absent(self, client):
        # humanizer.features.syntax is the research/10 gap and may not exist.
        assert isinstance(client.get("/api/health").json()["syntax_backend"], bool)

    def test_health_lists_reference_names(self, client):
        names = client.get("/api/health").json()["references"]
        assert isinstance(names, list)
        assert "research-article-stem" in names


class TestAnalyze:
    def test_happy_path_returns_full_report(self, client):
        r = client.post("/api/analyze", json={"text": SAMPLE})
        assert r.status_code == 200
        body = r.json()
        for key in ("features", "bands", "findings", "detectors", "deviations"):
            assert key in body
        assert body["features"]["n_words"] > 300
        # `detectors` now carries the published default model, or is empty
        # with the reason in `detector_error`. There is no fallback score:
        # this repo writes no AI-detection arithmetic, so "the model would not
        # load" has to read as no number rather than as a different number.
        assert "detector_error" in body
        if body["detectors"]:
            row = next(iter(body["detectors"].values()))
            assert 0.0 <= row["ai_probability"] <= 1.0
            assert row["is_published_detector"] is True
            assert row["model"]
        else:
            assert body["detector_error"]

    def test_style_signals_are_explanation_and_say_so(self, client):
        """The former heuristic detector, re-scoped.

        It is still useful for telling a writer *which* AI tells their draft
        contains. It is no longer allowed to look like evidence, so it ships
        under its own key, with a citation per signal and no aggregate.
        """
        body = client.post("/api/analyze", json={"text": SAMPLE}).json()
        block = body["ai_style_signals"]
        assert block["is_a_detector"] is False
        assert "ai_probability" not in block["signals"]
        assert set(block["signals"]) == set(block["sources"])
        assert all(0.0 <= v <= 1.0 for v in block["signals"].values())
        assert "heuristic" not in body["detectors"]

    def test_findings_carry_grade_cost(self, client):
        findings = client.post("/api/analyze", json={"text": SAMPLE}).json()["findings"]
        assert findings, "AI-flavoured sample should trip at least one finding"
        assert all({"code", "severity", "message", "grade_cost"} <= set(f) for f in findings)

    def test_empty_text_is_400(self, client):
        r = client.post("/api/analyze", json={"text": ""})
        assert r.status_code == 400
        assert "text" in r.json()["detail"].lower()

    def test_whitespace_only_text_is_400(self, client):
        assert client.post("/api/analyze", json={"text": "   \n\t "}).status_code == 400

    def test_missing_text_field_is_400_not_422(self, client):
        # `text` defaults to "" so callers get our message, not a pydantic blob.
        assert client.post("/api/analyze", json={}).status_code == 400

    def test_short_text_still_analyzes(self, client):
        r = client.post("/api/analyze", json={"text": SHORT})
        assert r.status_code == 200
        assert r.json()["features"]["n_words"] > 0

    def test_with_reference_by_name(self, client):
        r = client.post(
            "/api/analyze",
            json={"text": SAMPLE, "reference": "research-article-stem"},
        )
        assert r.status_code == 200
        ref = r.json()["reference"]
        assert ref is not None
        assert 0.0 <= ref["percentile"] <= 100.0
        assert r.json()["deviations"]

    def test_with_reference_by_path(self, client):
        path = str(REFERENCE_DIR / "research-article-stem.json")
        r = client.post("/api/analyze", json={"text": SAMPLE, "reference": path})
        assert r.status_code == 200
        assert r.json()["reference"] is not None

    def test_unknown_reference_is_404(self, client):
        r = client.post("/api/analyze", json={"text": SAMPLE, "reference": "nope"})
        assert r.status_code == 404


class TestSentences:
    def test_sentence_array_shape(self, client):
        body = client.post("/api/analyze", json={"text": SAMPLE}).json()
        sentences = body["sentences"]
        assert len(sentences) == int(body["features"]["n_sentences"])
        for i, s in enumerate(sentences):
            assert set(s) == {"index", "text", "paragraph_index", "length", "risk"}
            assert s["index"] == i
            assert s["text"].strip()
            assert s["length"] > 0
            assert s["risk"] is None or 0.0 <= s["risk"] <= 1.0

    def test_paragraph_indices_are_non_decreasing(self, client):
        sentences = client.post("/api/analyze", json={"text": SAMPLE}).json()["sentences"]
        idx = [s["paragraph_index"] for s in sentences]
        assert idx == sorted(idx)
        assert max(idx) >= 3  # the sample has four paragraphs

    def test_sentence_risk_is_flagged_advisory(self, client):
        # research/07: the document verdict is not an aggregate of sentence
        # scores, and research/10 warns that repairing only the worst sentences
        # destroys the very burstiness the detector measures.
        body = client.post("/api/analyze", json={"text": SAMPLE}).json()
        assert body["sentence_risk_is_advisory"] is True

    def test_short_sentence_falls_back_to_document_risk(self, client):
        body = client.post("/api/analyze", json={"text": "Yes. " + SAMPLE}).json()
        if not body["detectors"]:
            pytest.skip(body["detector_error"])
        doc_risk = next(iter(body["detectors"].values()))["ai_probability"]
        first = body["sentences"][0]
        assert first["length"] < 5
        assert first["risk"] == pytest.approx(doc_risk)

    def test_sentence_risks_vary(self, client):
        """Real per-sentence numbers from the published model, not a constant."""
        body = client.post("/api/analyze", json={"text": SAMPLE}).json()
        if not body["detectors"]:
            pytest.skip(body["detector_error"])
        risks = {s["risk"] for s in body["sentences"]}
        assert len(risks) > 1, "per-sentence scoring should not be a constant"

    def test_risk_is_null_rather_than_invented_when_no_model_is_available(
        self, client
    ):
        """The honest degraded path.

        Before, an unavailable model was papered over by hand-written
        arithmetic that produced a number in the same field. Now the field is
        null and the reason is on the wire.
        """
        body = client.post("/api/analyze", json={"text": SAMPLE}).json()
        if body["detectors"]:
            pytest.skip("the detector is available in this environment")
        assert body["detector_error"]
        assert all(s["risk"] is None for s in body["sentences"])
        # And the explanation still works without any model at all.
        assert body["ai_style_signals"]["signals"]


class TestFeatures:
    def test_features_and_bands(self, client):
        r = client.post("/api/features", json={"text": SAMPLE})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"features", "bands"}
        assert body["features"]["n_sentences"] > 5
        assert "sent_len_cv" in body["features"]
        assert body["bands"]["sent_len_cv"] in {"low", "in band", "high", "unknown"}

    def test_features_empty_text_is_400(self, client):
        assert client.post("/api/features", json={"text": ""}).status_code == 400


class TestReferences:
    def test_listing_shape(self, client):
        r = client.get("/api/references")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list) and rows
        for row in rows:
            assert set(row) == {"name", "genre", "n_documents", "source"}
            assert row["n_documents"] > 0

    def test_detail_has_summary_and_count(self, client):
        r = client.get("/api/reference/research-article-stem")
        assert r.status_code == 200
        body = r.json()
        assert body["n_documents"] == 60
        summary = body["summary"]
        assert "sent_len_cv" in summary
        assert set(summary["sent_len_cv"]) >= {"mean", "sd", "p10", "p50", "p90"}

    def test_detail_unknown_is_404(self, client):
        assert client.get("/api/reference/not-a-genre").status_code == 404

    def test_empty_reference_dir_is_not_an_error(self, tmp_path):
        app = create_app(reference_dir=tmp_path, web_dir=tmp_path / "no-web")
        with TestClient(app) as c:
            assert c.get("/api/references").json() == []
            assert c.get("/api/health").json()["references"] == []


class TestPlan:
    def test_plan_rows(self, client):
        r = client.post("/api/plan", json={"mu": 0.6, "target": 0.99})
        assert r.status_code == 200
        rows = r.json()["rows"]
        assert [row["rho"] for row in rows] == [0.0, 0.1, 0.2, 0.3, 0.5]
        for row in rows:
            assert set(row) == {"rho", "required_n", "pass_at_8", "pass_at_32"}

    def test_plan_reproduces_the_published_figures(self, client):
        # research/08: mu=0.6 gives 99.93% at rho=0, 97.5% at 0.2, 87.3% at 0.5.
        rows = {r["rho"]: r for r in client.post("/api/plan", json={"mu": 0.6, "target": 0.99}).json()["rows"]}
        assert rows[0.0]["pass_at_8"] == pytest.approx(0.9993, abs=5e-4)
        assert rows[0.2]["pass_at_8"] == pytest.approx(0.975, abs=5e-3)
        assert rows[0.5]["pass_at_8"] == pytest.approx(0.873, abs=5e-3)

    def test_unreachable_target_reports_null_n(self, client):
        # A correlation of 0.5 makes a 99% target unreachable at any N.
        rows = {r["rho"]: r for r in client.post("/api/plan", json={"mu": 0.6, "target": 0.99}).json()["rows"]}
        assert rows[0.5]["required_n"] is None
        assert rows[0.0]["required_n"] is not None

    def test_bad_mu_is_400(self, client):
        assert client.post("/api/plan", json={"mu": 1.5, "target": 0.9}).status_code == 400

    def test_bad_target_is_400(self, client):
        assert client.post("/api/plan", json={"mu": 0.6, "target": 1.0}).status_code == 400


class TestJsonSafety:
    def test_json_safe_unit(self):
        out = json_safe(
            {
                "nan": float("nan"),
                "inf": float("inf"),
                "ninf": float("-inf"),
                "ok": 1.5,
                "nested": [float("nan"), {"deep": float("inf")}],
                "flag": True,
                "none": None,
            }
        )
        assert out["nan"] is None
        assert out["inf"] is None
        assert out["ninf"] is None
        assert out["ok"] == 1.5
        assert out["nested"] == [None, {"deep": None}]
        assert out["flag"] is True
        json.dumps(out, allow_nan=False)  # must not raise

    def test_json_safe_unwraps_numpy(self):
        np = pytest.importorskip("numpy")
        out = json_safe({"a": np.float64("nan"), "b": np.int64(3), "c": np.array([1.0, 2.0])})
        assert out == {"a": None, "b": 3, "c": [1.0, 2.0]}

    def test_nan_becomes_null_in_features_response(self, client):
        # One sentence => no sentence-length SD => sent_len_cv is NaN upstream.
        r = client.post("/api/features", json={"text": ONE_SENTENCE})
        assert r.status_code == 200
        assert r.json()["features"]["sent_len_cv"] is None
        assert "NaN" not in r.text and "Infinity" not in r.text

    def test_no_nan_token_anywhere_in_an_analysis(self, client):
        r = client.post(
            "/api/analyze",
            json={"text": ONE_SENTENCE, "reference": "research-article-stem"},
        )
        assert r.status_code == 200
        assert "NaN" not in r.text and "Infinity" not in r.text
        # json.loads accepts bare NaN/Infinity by default, which most other
        # JSON parsers do not, so parse strictly to match a real client.
        def reject(token):
            raise AssertionError(f"non-JSON constant on the wire: {token}")

        json.loads(r.text, parse_constant=reject)

    def test_upstream_really_does_emit_nan(self):
        # Guards the guard: if extract_features stops emitting NaN, the
        # conversion tests above would silently stop testing anything.
        from humanizer.features import extract_features

        feats = extract_features(ONE_SENTENCE)
        assert any(
            isinstance(v, float) and not math.isfinite(v) for v in feats.values()
        )


class TestStaticAndCors:
    def test_missing_web_dir_does_not_break_startup(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_web_dir_is_served_when_present(self, tmp_path):
        web = tmp_path / "web"
        web.mkdir()
        (web / "index.html").write_text("<h1>humanizer</h1>")
        (web / "app.js").write_text("console.log(1)")
        app = create_app(reference_dir=REFERENCE_DIR, web_dir=web)
        with TestClient(app) as c:
            assert "<h1>humanizer</h1>" in c.get("/").text
            assert "console.log(1)" in c.get("/app.js").text
            # The static mount must not shadow the API routes.
            assert c.get("/api/health").json()["status"] == "ok"

    def test_cors_allows_localhost(self, client):
        r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_rejects_other_origins(self, client):
        r = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in r.headers


class TestCli:
    def test_serve_is_registered(self):
        from humanizer.cli import build_parser

        args = build_parser().parse_args(["serve", "--port", "8137"])
        assert args.port == 8137
        assert args.command == "serve"

    def test_serve_defaults(self):
        from humanizer.cli import build_parser

        args = build_parser().parse_args(["serve"])
        assert args.port == 8000
        assert args.reference is None
