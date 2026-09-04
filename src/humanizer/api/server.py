"""FastAPI application exposing the Stage 0 analysis toolkit over HTTP.

Endpoints
---------
    GET  /api/health              version, syntax backend, reference names
    POST /api/analyze             full DocumentReport plus per-sentence rows
    POST /api/features            raw feature vector plus human-band verdicts
    GET  /api/references          reference distributions on disk
    GET  /api/reference/{name}    one distribution's summary statistics
    POST /api/detect              run the local detectors over one document
    POST /api/plan                best-of-N table under candidate correlation
    POST /api/humanize            rewrite a document and report every edit
    GET  /                        the static frontend in web/, when present

Design notes
------------
Reference distributions are loaded lazily and cached. The shipped
research-article-stem distribution is a 60x~68 float matrix; parsing it per
request would dominate the cost of an analysis that is otherwise a few
milliseconds of regex work.

Nothing here re-implements analysis. `analyze()`, `extract_features()`,
`band_report()`, `ReferenceDistribution.score()` and `pass_rate()`/`required_n()`
are called as-is so that the HTTP surface and the CLI can never drift apart.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..detectors import (
    ClassifierDetector,
    EnsembleDetector,
    HeuristicDetector,
    ModernDetector,
    PerplexityDetector,
    backend_available,
    pass_rate,
    required_n,
)
from ..eval import analyze
from ..features import band_report, extract_features
from ..reference import ReferenceDistribution
from ..text import Document
from .models import AnalyzeRequest, FeaturesRequest, PlanRequest, json_safe

try:  # pragma: no cover - `api.models` already raises a readable error first
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    raise

# research/08 illustration grid, matching `humanizer plan` so the web view and
# the CLI print the same rows.
PLAN_RHOS = (0.0, 0.1, 0.2, 0.3, 0.5)

# Below this, a sentence has too few words for the shape and lexical features
# to mean anything, so its own heuristic score is noise. research/01 notes
# detectors themselves refuse or hedge under ~50 words; five is simply the
# floor at which our per-sentence signals stop being all-zero.
MIN_WORDS_FOR_SENTENCE_RISK = 5

# Repo root, used to find data/ and web/ when the server is started from
# somewhere other than the project directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------- reference IO


class ReferenceStore:
    """Lazily loaded, cached reference distributions from a directory.

    Metadata for the listing endpoint is cached separately from the parsed
    distribution so that `GET /api/references` does not have to build a numpy
    matrix for every genre on disk just to report its name.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._loaded: Dict[str, ReferenceDistribution] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def paths(self) -> Dict[str, Path]:
        """Distribution name -> file path, for every JSON in the directory."""
        if not self.directory.is_dir():
            return {}
        return {p.stem: p for p in sorted(self.directory.glob("*.json"))}

    def names(self) -> List[str]:
        return list(self.paths())

    def metadata(self) -> List[Dict[str, Any]]:
        """Name, genre, document count and provenance for each distribution."""
        out: List[Dict[str, Any]] = []
        for name, path in self.paths().items():
            if name not in self._meta:
                try:
                    ref = self.get(name)
                except (OSError, ValueError, KeyError):
                    # A malformed file must not take the listing down with it.
                    continue
                self._meta[name] = {
                    "name": name,
                    "genre": ref.genre,
                    "n_documents": ref.n_documents,
                    "source": ref.source,
                }
            out.append(self._meta[name])
        return out

    def get(self, name: str) -> ReferenceDistribution:
        """Load and cache one distribution by name, or raise KeyError."""
        if name in self._loaded:
            return self._loaded[name]
        path = self.paths().get(name)
        if path is None:
            raise KeyError(name)
        ref = ReferenceDistribution.from_json(path)
        self._loaded[name] = ref
        return ref

    def resolve(self, spec: Optional[str]) -> Optional[ReferenceDistribution]:
        """Resolve a name *or* a filesystem path to a distribution.

        The CLI passes `--reference data/reference/research-article-stem.json`
        while the frontend passes the bare name from `GET /api/references`.
        Both work.
        """
        if not spec:
            return None
        try:
            return self.get(spec)
        except KeyError:
            pass
        path = Path(spec)
        if path.is_file():
            key = str(path.resolve())
            if key not in self._loaded:
                self._loaded[key] = ReferenceDistribution.from_json(path)
            return self._loaded[key]
        raise KeyError(spec)


def default_reference_dir() -> Path:
    """Where to look for `data/reference/*.json`.

    Checked in order: the HUMANIZER_REFERENCE_DIR override, the working
    directory (the normal case, since the CLI is run from the project root),
    then the repo root inferred from this file's location.
    """
    env = os.environ.get("HUMANIZER_REFERENCE_DIR")
    if env:
        return Path(env)
    candidates = [Path.cwd() / "data" / "reference", _REPO_ROOT / "data" / "reference"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def default_web_dir() -> Optional[Path]:
    """The static frontend directory, or None when it has not been built yet."""
    env = os.environ.get("HUMANIZER_WEB_DIR")
    candidates = [Path(env)] if env else [Path.cwd() / "web", _REPO_ROOT / "web"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


# ------------------------------------------------------------- syntax backend


def syntax_backend_available() -> bool:
    """Whether the optional spaCy clause parser is importable.

    `humanizer.features.syntax` is the known gap from research/10: clause-level
    features are regex heuristics until it lands. The module may not exist at
    all, so an ImportError here is an expected state, not an error.
    """
    try:
        from ..features import syntax  # noqa: F401
    except ImportError:
        return False
    return True


# ------------------------------------------------------------ sentence rows


def sentence_rows(text: str, document_risk: Optional[float]) -> List[Dict[str, Any]]:
    """Per-sentence rows for the editor gutter, with an ADVISORY risk score.

    Read this before using `risk` for anything automated.

    research/07 is explicit that GPTZero's document verdict is not an aggregate
    of its sentence highlights, so a per-sentence score here cannot predict
    which sentences moved the verdict. Worse, research/10 shows that repairing
    only the highest-scoring sentences is actively harmful: burstiness is a
    property of the *variance* of sentence lengths, and flattening the outliers
    is exactly how a document's CV collapses below the 0.42-0.60 human band.

    So this field exists to draw a reader's eye, and for nothing else. The
    response carries `sentence_risk_is_advisory: true` to say so on the wire.

    Sentences under `MIN_WORDS_FOR_SENTENCE_RISK` words fall back to the
    document score rather than reporting a spuriously clean 0.0 - a three-word
    sentence has no AI vocabulary and no measurable shape by construction.
    """
    detector = HeuristicDetector()
    rows: List[Dict[str, Any]] = []
    for sent in Document.parse(text).sentences:
        length = sent.length
        if length >= MIN_WORDS_FOR_SENTENCE_RISK:
            risk = detector.score_features(extract_features(sent.text)).ai_probability
        else:
            risk = document_risk
        if risk is not None and not math.isfinite(risk):
            risk = None
        rows.append(
            {
                "index": sent.index,
                "text": sent.text,
                "paragraph_index": sent.paragraph_index,
                "length": length,
                "risk": risk,
            }
        )
    return rows


# ------------------------------------------------------------------- helpers


def _require_text(text: str) -> str:
    """Reject empty submissions loudly; let short ones through.

    Short text is analysable - the report already warns under 300 words and the
    feature extractors return NaN where they cannot compute. Empty text is a
    client bug and deserves a 400.
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Field 'text' is required and must not be empty or whitespace.",
        )
    return text


def _ok(payload: Any) -> JSONResponse:
    """Serialise a payload after stripping NaN/Infinity (see models.json_safe)."""
    return JSONResponse(content=json_safe(payload))


# ------------------------------------------------------------ /api/detect

# Detector name -> zero-argument factory. Factories, not instances, so that
# nothing is constructed (and no 500MB of weights is touched) until a request
# actually names the detector. Instances are then cached on app.state so the
# model load is paid once per process rather than once per request.
#
# `modern`, `perplexity` and `classifier` need the `detectors` extra (torch,
# transformers). They are listed here unconditionally on purpose: the endpoint
# reports `available: false` with the install command rather than pretending
# the detector does not exist, which is what the frontend needs in order to
# show a disabled control instead of silently dropping a feature.
DETECTOR_FACTORIES: Dict[str, Any] = {
    # The default. A current fine-tuned transformer classifier
    # (desklib/ai-text-detector-v1.01, DeBERTa-v3-large, ~1.74GB) -- the same
    # architectural class GPTZero, Originality, Pangram and Turnitin all use
    # now, and not GPTZero. See `humanizer.detectors.local`.
    "modern": lambda: ModernDetector(),
    "heuristic": lambda: HeuristicDetector(),
    # 2023-era baselines, kept and reachable by name. Both were measured on
    # this repo's own PMC corpus and both are worse than `modern` there:
    # pairwise 0.740 and 0.444 against 1.000, with 13 of 14 and 3 of 14
    # genuine human academic paragraphs mislabelled AI against 1 of 14.
    "perplexity": lambda: PerplexityDetector(),
    "classifier": lambda: ClassifierDetector(),
    "ensemble": lambda: EnsembleDetector(),
}

#: Used when the request leaves `detectors` null.
#:
#: One engine, deliberately. The old default ran three and let the page show
#: three disagreeing percentages, two of which came from methods this repo has
#: measured as no better than chance on academic prose. Showing a caller a
#: number that is wrong, next to a number that is right, is worse than showing
#: one number: it invites them to average. `heuristic`, `perplexity`,
#: `classifier` and `ensemble` are all still one explicit request away.
DEFAULT_DETECTORS = ("modern",)


class DetectRequest(BaseModel):
    """Body for POST /api/detect."""

    text: str = Field(default="", description="Raw document text.")
    detectors: Optional[List[str]] = Field(
        default=None,
        description=(
            "Detector names to run. Null runs "
            f"{list(DEFAULT_DETECTORS)}. Valid names: "
            f"{sorted(DETECTOR_FACTORIES)}."
        ),
    )


def _detector_instance(app: FastAPI, name: str) -> Any:
    """Get or build the process-wide instance of one detector."""
    cache = app.state.detector_cache
    if name not in cache:
        cache[name] = DETECTOR_FACTORIES[name]()
    return cache[name]


def run_detector(app: FastAPI, name: str, text: str) -> Dict[str, Any]:
    """Run one detector and describe the outcome, never raising.

    A missing model is a normal state, not a server fault. research/01 §3.6 and
    the module docstring of `detectors.local` are blunt that these local models
    are weak; a caller that cannot get a score still needs the rest of the page
    to render, so every failure path returns `available: false` plus a message
    a human can act on, and the HTTP status stays 200.
    """
    row: Dict[str, Any] = {
        "name": name,
        "available": False,
        "ai_probability": None,
        "label": None,
        "confidence": None,
        "sentence_scores": [],
        "raw": {},
        "error": None,
    }
    if name not in DETECTOR_FACTORIES:
        row["error"] = (
            f"Unknown detector {name!r}. Valid names: "
            f"{sorted(DETECTOR_FACTORIES)}."
        )
        return row

    try:
        detector = _detector_instance(app, name)
    except Exception as exc:  # noqa: BLE001 - construction must not 500
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    if not detector.available():
        reason = getattr(detector, "unavailable_reason", lambda: None)()
        row["error"] = reason or (
            f"Detector {name!r} reports itself unavailable. "
            "Local model detectors need: pip install -e '.[detectors]'"
        )
        return row

    try:
        result = detector.score(text)
    except Exception as exc:  # noqa: BLE001 - a missing download is not a bug
        # `available()` for the local detectors flips to False after a failed
        # load, so the next request answers instantly instead of retrying a
        # download that is not going to work.
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row.update(
        {
            "available": True,
            "ai_probability": result.ai_probability,
            "label": result.label,
            "confidence": result.confidence,
            "sentence_scores": list(result.sentence_scores),
            "raw": result.raw,
        }
    )
    return row


# ---------------------------------------------------------- /api/humanize
#
# Body model for the appended humanize endpoint. It lives at module scope, next
# to `DetectRequest`, because FastAPI resolves a route's annotations against the
# defining module's globals: a request model declared inside `create_app()`
# cannot be found and every request 422s before it reaches the handler.


class HumanizeRequest(BaseModel):
    """Body for POST /api/humanize."""

    text: str = Field(default="", description="Raw document text.")
    aggressiveness: str = Field(
        default="balanced",
        description=(
            "'light' (AI lexicon and formal connectives only), 'balanced' "
            "(adds parallelism and paragraph-template breaking) or 'strong' "
            "(adds sentence-length restructuring)."
        ),
    )


# ----------------------------------------------------------------------- app


def create_app(
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
    default_reference: Optional[str] = None,
) -> FastAPI:
    """Build the ASGI application.

    `default_reference` is the name or path used by /api/analyze when the
    request body leaves `reference` null, so `humanizer serve --reference ...`
    behaves like `humanizer analyze --reference ...`.
    """
    store = ReferenceStore(reference_dir or default_reference_dir())
    web_root = web_dir if web_dir is not None else default_web_dir()

    app = FastAPI(
        title="humanizer",
        version=__version__,
        description="Stage 0 text analysis: features, human bands, detectors.",
    )
    app.state.store = store
    app.state.default_reference = default_reference
    # Detector instances are process-wide so that gpt2 and the RoBERTa
    # detector are loaded once, not per request. Built on first use.
    app.state.detector_cache = {}

    # The frontend is served from the same origin in production, but a dev
    # server on another port is the normal working setup, so localhost in any
    # form is allowed and nothing else is.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> JSONResponse:
        """Liveness plus the capability flags the frontend branches on."""
        return _ok(
            {
                "status": "ok",
                "version": __version__,
                "syntax_backend": syntax_backend_available(),
                "references": store.names(),
            }
        )

    @app.post("/api/analyze")
    def analyze_endpoint(body: AnalyzeRequest) -> JSONResponse:
        """Full document report, plus advisory per-sentence rows.

        The body of the response is exactly `DocumentReport.as_dict()` with two
        additions: `sentences` and `sentence_risk_is_advisory`. See
        `sentence_rows()` for why that flag is not decoration.
        """
        text = _require_text(body.text)
        spec = body.reference if body.reference is not None else app.state.default_reference
        try:
            reference = store.resolve(spec)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Unknown reference distribution: {spec!r}"
            )

        report = analyze(text, reference=reference, detectors=[HeuristicDetector()])
        payload = report.as_dict()

        heuristic = payload.get("detectors", {}).get("heuristic", {})
        document_risk = heuristic.get("ai_probability")
        payload["sentences"] = sentence_rows(text, document_risk)
        payload["sentence_risk_is_advisory"] = True
        payload["reference_name"] = spec if reference is not None else None
        return _ok(payload)

    @app.post("/api/features")
    def features_endpoint(body: FeaturesRequest) -> JSONResponse:
        """Raw feature vector and the human-band verdict for each banded feature."""
        text = _require_text(body.text)
        feats = extract_features(text)
        return _ok({"features": feats, "bands": band_report(feats)})

    @app.get("/api/references")
    def references_endpoint() -> JSONResponse:
        """Every reference distribution found on disk. May be an empty list."""
        return _ok(store.metadata())

    @app.get("/api/reference/{name}")
    def reference_endpoint(name: str) -> JSONResponse:
        """Per-feature mean, SD, CV and the 10/25/50/75/90 percentiles."""
        try:
            ref = store.get(name)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Unknown reference distribution: {name!r}"
            )
        return _ok(
            {
                "name": name,
                "genre": ref.genre,
                "source": ref.source,
                "n_documents": ref.n_documents,
                "summary": ref.summary(),
            }
        )

    @app.post("/api/detect")
    def detect_endpoint(body: DetectRequest) -> JSONResponse:
        """Score one document with each requested detector.

        This endpoint never returns 5xx because a model is missing. Each row
        carries its own `available` flag and `error` string, so a page can show
        the heuristic score while the 500MB gpt2 download is still failing.

        On what these detectors are. `modern` is the default and the only one
        worth reading a number off: `desklib/ai-text-detector-v1.01`, an open
        fine-tuned DeBERTa-v3-large classifier that led the RAID detection
        leaderboard. It is the same *architectural class* as every detector
        shipped after 2023 -- GPTZero, Originality, Pangram, Turnitin are all
        supervised fine-tuned transformers now -- which is exactly as far as
        the resemblance goes. `heuristic` is the dependency-free baseline.
        `perplexity` is GPTZero's *original* January-2023 method (per-sentence
        perplexity under gpt2, burstiness = the SD of those), kept because it
        is a historical reference and because this repo measured it failing on
        academic prose; `classifier` is OpenAI's 2019 RoBERTa GPT-2 output
        detector, which measured *below chance* on the same set.

        None of them is GPTZero, whose current model is a proprietary
        supervised transformer with a hierarchical multi-task head over
        {Human, AI, Mixed} plus a joint binary sentence head, trained on ~28.6M
        undisclosed documents (research/07 §3). The response says so on the
        wire in `disclaimer` rather than leaving it to the docs, because a
        percentage on a screen reads as authoritative.
        """
        text = _require_text(body.text)
        names = list(body.detectors) if body.detectors is not None else list(
            DEFAULT_DETECTORS
        )
        if not names:
            raise HTTPException(
                status_code=400,
                detail="'detectors' must be null or a non-empty list of names.",
            )

        results = [run_detector(app, name, text) for name in names]
        doc = Document.parse(text)
        return _ok(
            {
                "detectors": results,
                "n_sentences": doc.n_sentences,
                "n_words": doc.n_words,
                "sentences": [s.text for s in doc.sentences],
                # Capability flags live here rather than on /api/health,
                # whose key set is pinned by an existing contract test.
                # `backend_available` is torch+transformers being importable;
                # it says nothing about whether the weights are on disk, which
                # is what each row's own `available`/`error` reports.
                "backend_available": backend_available(),
                "known_detectors": sorted(DETECTOR_FACTORIES),
                "is_gptzero": False,
                "disclaimer": (
                    "Not GPTZero. The default engine 'modern' is "
                    "desklib/ai-text-detector-v1.01: an open fine-tuned "
                    "DeBERTa-v3-large sequence classifier (435M parameters) "
                    "trained on the RAID corpus, which led the RAID detection "
                    "leaderboard. It is the same architectural class GPTZero "
                    "uses today -- a supervised fine-tuned transformer "
                    "classifier rather than the 2023 perplexity-and-"
                    "burstiness statistic, which GPTZero itself abandoned as "
                    "a decision rule -- and that is the whole of the "
                    "resemblance. It is not GPTZero and has no relationship "
                    "to GPTZero's weights, training data, thresholds or "
                    "calibration. GPTZero's current model is proprietary and "
                    "unpublished: a supervised transformer with a "
                    "hierarchical multi-task head over {Human, AI, Mixed} "
                    "plus a jointly trained binary sentence head, trained on "
                    "~28.6M documents (research/07 s3). Nothing here is "
                    "calibrated. Measured on this repo's own 28-paragraph set "
                    "(14 human PMC paragraphs, 14 written AI-style "
                    "paragraphs), 'modern' separates them at 1.000 pairwise "
                    "with 1 of 14 genuine human academic paragraphs over 0.5; "
                    "the legacy 'perplexity' engine scored 0.740 pairwise "
                    "with 13 of 14, and 'classifier' scored 0.444 -- below "
                    "chance. n=28, one author on the AI side, no confidence "
                    "interval. Per-sentence rows are far less reliable than "
                    "the document score and exist for the heatmap only."
                ),
            }
        )

    @app.post("/api/plan")
    def plan_endpoint(body: PlanRequest) -> JSONResponse:
        """Best-of-N planning table.

        research/08: correlation between candidates, not N, is the binding
        constraint, so every row is a rho and `required_n` is null where the
        target is unreachable at any N.
        """
        if not 0.0 <= body.mu <= 1.0:
            raise HTTPException(status_code=400, detail="'mu' must be in [0, 1].")
        if not 0.0 < body.target < 1.0:
            raise HTTPException(status_code=400, detail="'target' must be in (0, 1).")
        rows = [
            {
                "rho": float(rho),
                "required_n": required_n(body.mu, body.target, rho),
                "pass_at_8": pass_rate(body.mu, 8, rho),
                "pass_at_32": pass_rate(body.mu, 32, rho),
            }
            for rho in PLAN_RHOS
        ]
        return _ok({"mu": body.mu, "target": body.target, "rows": rows})

    # ------------------------------------------------------ /api/humanize
    #
    # Appended, and deliberately registered *before* the StaticFiles mount
    # below: a mount at "/" matches every path, so a route added after it would
    # never be reached.

    @app.post("/api/humanize")
    def humanize_endpoint(body: HumanizeRequest) -> JSONResponse:
        """Rewrite one document and report every edit with its rationale.

        The response carries the full before/after feature vectors and the
        heuristic AI probability for each, so a caller can show the measurement
        that justified the rewrite rather than a bare score. `risk_delta` is
        after minus before, so **negative is an improvement**.

        Only edits that research/00 §4 rates as zero or negative grade cost are
        ever produced; `humanizer.humanize.transforms` documents the refusal
        list. Every edit is gated by an exact-match meaning invariant over
        numbers, quotations, citations and named entities before it is applied,
        and `Document.protected_spans()` keeps all of them out of quotes,
        citations and the first and last sentence.
        """
        text = _require_text(body.text)
        from ..humanize import humanize as run_humanize

        try:
            result = run_humanize(text, aggressiveness=body.aggressiveness)
        except ValueError as exc:  # unknown aggressiveness level
            raise HTTPException(status_code=400, detail=str(exc))
        return _ok(result.as_dict())

    # Mounted last on purpose: a StaticFiles mount at "/" matches every path,
    # so it must sit behind the API routes in the router's ordered list.
    # `html=True` serves web/index.html at "/". A missing web/ is normal while
    # the frontend is being built and must not stop the API from starting.
    if web_root is not None and web_root.is_dir():
        app.mount("/", StaticFiles(directory=str(web_root), html=True), name="web")
        app.state.web_dir = str(web_root)
    else:
        app.state.web_dir = None

        @app.get("/")
        def index_placeholder() -> JSONResponse:
            return _ok(
                {
                    "status": "ok",
                    "detail": "No web/ directory found; the API is up. See /docs.",
                }
            )

    return app


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    reference: Optional[str] = None,
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
) -> None:
    """Start the development server. Used by `humanizer serve`."""
    import uvicorn

    app = create_app(
        reference_dir=reference_dir, web_dir=web_dir, default_reference=reference
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
