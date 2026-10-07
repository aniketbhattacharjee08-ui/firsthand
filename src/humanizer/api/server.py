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
    GET  /api/models              resident model checkpoints and process RSS
    POST /api/models/release      unload every cached model
    POST /api/auth/signup         create an account and sign in (sets cookie)
    POST /api/auth/signin         sign in; 401 invalid_credentials
    POST /api/auth/signout        204, clears the cookie
    GET  /api/auth/me             {signed_in, user?}
    GET  /                        web/index.html, the tool as the landing page
    GET  /signin, /signup         web/auth.html
    GET  /app                     web/index.html, signed-in users only

Accounts are on by default (HUMANIZER_AUTH=1; `humanizer serve --no-auth` or
`create_app(auth=False)` turns them off). With auth on, every product route
(/api/humanize*, /api/analyze, /api/detect, /api/plan, /api/features,
/api/models*) answers 401 `sign_in_required` without a session cookie. See
`humanizer.api.auth`.

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
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..detectors import (
    PUBLISHED_DETECTORS,
    SHIPPED_DETECTORS,
    AiStyleSignals,
    ClassifierDetector,
    EnsembleDetector,
    ModernDetector,
    PerplexityDetector,
    backend_available,
    pass_rate,
    required_n,
)
from ..detectors.heuristic import SIGNAL_SOURCES
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


MAINTENANCE_ENV = "HUMANIZER_MAINTENANCE"


def maintenance_from_env() -> bool:
    """`HUMANIZER_MAINTENANCE` as a bool; unset is off."""
    return os.environ.get(MAINTENANCE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


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


def sentence_rows(
    text: str,
    document_risk: Optional[float],
    sentence_risks: Optional[Sequence[Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    """Per-sentence rows for the editor gutter, with an ADVISORY risk score.

    Read this before using `risk` for anything automated.

    `sentence_risks`, when given, is the published detector's own per-sentence
    output, positionally aligned with `Document.parse(text).sentences` because
    that is the segmentation the detector used too. When it is absent - no
    torch, weights not downloaded - every row's risk is `document_risk`, which
    is itself None in that case. **Nothing here computes a risk score.** This
    repo writes no detection arithmetic; the rows either carry a published
    model's number or they carry null.

    research/07 is explicit that GPTZero's document verdict is not an aggregate
    of its sentence highlights, so a per-sentence score here cannot predict
    which sentences moved the verdict. Worse, research/10 shows that repairing
    only the highest-scoring sentences is actively harmful: burstiness is a
    property of the *variance* of sentence lengths, and flattening the outliers
    is exactly how a document's CV collapses below the 0.42-0.60 human band.

    And the model itself is weakest here: a sentence is far under every
    vendor's stated reliability floor (research/01 §5). On this repo's corpus a
    human paragraph scoring 0.015 at document level had individual sentences up
    at 0.87.

    So this field exists to draw a reader's eye, and for nothing else. The
    response carries `sentence_risk_is_advisory: true` to say so on the wire.

    Sentences under `MIN_WORDS_FOR_SENTENCE_RISK` words fall back to the
    document score rather than reporting a spuriously clean 0.0 - a three-word
    sentence has no measurable shape by construction.
    """
    rows: List[Dict[str, Any]] = []
    for sent in Document.parse(text).sentences:
        length = sent.length
        risk = document_risk
        if length >= MIN_WORDS_FOR_SENTENCE_RISK and sentence_risks is not None:
            if sent.index < len(sentence_risks):
                risk = sentence_risks[sent.index]
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
# Every engine here needs the `detectors` extra (torch, transformers). They
# are listed unconditionally on purpose: the endpoint reports
# `available: false` with the install command rather than pretending the
# detector does not exist, which is what the frontend needs in order to show a
# disabled control instead of silently dropping a feature.
#
# The five published engines are one wrapper class with five checkpoints. The
# registry, the per-checkpoint measurements, and the five further checkpoints
# that were benchmarked and rejected live in
# `humanizer.detectors.local.PUBLISHED_DETECTORS`.
#
# There is no `heuristic` entry any more. It was this repo's own hand-written
# arithmetic and detection is now exclusively published models; the style
# signals it was built on survive as explanation on /api/analyze.
DETECTOR_FACTORIES: Dict[str, Any] = {
    name: (
        lambda m=model, thr=PUBLISHED_DETECTORS.get(name, {}).get("threshold", 0.5): ModernDetector(
            model_name=m, threshold=thr
        )
    )
    for name, model in SHIPPED_DETECTORS.items()
}
DETECTOR_FACTORIES.update(
    {
        # `perplexity` is the one engine whose scoring this repo wrote. It is
        # reachable by name for research comparison and excluded from every
        # default; its rows carry `is_published_detector: false`.
        "perplexity": lambda: PerplexityDetector(),
        # Published but obsolete: 0.444 pairwise, below chance, on this repo's
        # own corpus. Kept because that measurement is the evidence.
        "classifier": lambda: ClassifierDetector(),
        "ensemble": lambda: EnsembleDetector(),
        # The product's judge. Paid, cached on disk, unavailable without
        # GPTZERO_API_KEY; `run_detector` reports that as a normal state.
        "gptzero": lambda: _gptzero_detector(),
    }
)


def _gptzero_detector() -> Any:
    from ..detectors.gptzero import GPTZeroClient

    return GPTZeroClient()


#: Used when the request leaves `detectors` null.
#:
#: One engine, deliberately: the free surrogate trained on GPTZero's own
#: verdicts (detectors.local "surrogate", research/24 §6.10). GPTZero itself
#: is reachable as "gptzero" for benches but is never the default, because
#: every call costs the product owner money and the end user will re-scan
#: with their own GPTZero credits anyway. The older local checkpoints stay
#: reachable by name for research; they were measured rating GPTZero-human
#: rewrites 0.65-1.00 and are not shown to a writer.
DEFAULT_DETECTORS = ("surrogate",)


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


def _configured_yardstick() -> str:
    from ..detectors import yardstick

    return yardstick.configured_name()


def _detector_model_name(row: Dict[str, Any]) -> Optional[str]:
    """A display name for the checkpoint or service behind a detector row."""
    if row["name"] == "gptzero":
        version = row["raw"].get("version")
        return "GPTZero" + (f" ({version})" if version else "")
    if row["name"] == "surrogate":
        return "GPTZero estimate (local surrogate, free)"
    return row["raw"].get("model")


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
                # The judge, by configured name: "gptzero" when a key is set.
                "yardstick": _configured_yardstick(),
            }
        )

    @app.post("/api/analyze")
    def analyze_endpoint(body: AnalyzeRequest) -> JSONResponse:
        """Full document report, plus advisory per-sentence rows.

        The body is `DocumentReport.as_dict()` plus `sentences`,
        `sentence_risk_is_advisory`, `reference_name`, `detector_error` and
        `ai_style_signals`. See `sentence_rows()` for why the advisory flag is
        not decoration.

        `detectors` carries at most one entry, from the default published
        model, and is **empty** when that model cannot run - with the reason in
        `detector_error`. There is deliberately no fallback score: this repo
        writes no detection arithmetic, so the honest answer to "we could not
        load the detector" is no number at all. `ai_style_signals` is still
        populated in that case, because it is explanation rather than
        detection and needs no model.
        """
        text = _require_text(body.text)
        spec = body.reference if body.reference is not None else app.state.default_reference
        try:
            reference = store.resolve(spec)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Unknown reference distribution: {spec!r}"
            )

        report = analyze(text, reference=reference)
        payload = report.as_dict()

        # The AI score comes from the default *published* detector, the same
        # one /api/detect runs, so the two endpoints cannot disagree. It is
        # absent rather than substituted when the model cannot run: this repo
        # has no fallback scorer of its own any more, and a hand-written
        # number in the same field as a model's would be indistinguishable
        # from it on the wire.
        document_risk: Optional[float] = None
        sentence_risks: Optional[List[Optional[float]]] = None
        payload["detector"] = DEFAULT_DETECTORS[0]
        payload["detector_ran"] = bool(body.detect)
        if body.detect:
            row = run_detector(app, DEFAULT_DETECTORS[0], text)
            if row["available"]:
                document_risk = row["ai_probability"]
                sentence_risks = list(row["sentence_scores"]) or None
                payload["detectors"][row["name"]] = {
                    "ai_probability": row["ai_probability"],
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "model": _detector_model_name(row),
                    "is_published_detector": True,
                }
            payload["detector_error"] = row["error"]
        else:
            payload["detector_error"] = None

        payload["sentences"] = sentence_rows(text, document_risk, sentence_risks)
        payload["sentence_risk_is_advisory"] = True
        payload["reference_name"] = spec if reference is not None else None

        # Explanatory only, and named so on the wire. These are measured style
        # signals with citations (research/04, research/10), not evidence and
        # not a score: there is no aggregate to read off them. They answer
        # "which known AI tells are in this draft", which is the question a
        # writer can act on, next to a detector's verdict, which is not.
        payload["ai_style_signals"] = {
            "signals": AiStyleSignals().signals(payload["features"]),
            "sources": SIGNAL_SOURCES,
            "is_a_detector": False,
            "note": (
                "Explanatory style features, not a detection score. They are "
                "normalised counts of documented AI tells; nothing here was "
                "trained and nothing here votes on the verdict above."
            ),
        }
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

        Every engine here scores with a published, pretrained checkpoint. This
        repo writes no AI-detection arithmetic, with one labelled exception.

        `modern` is the default and the best measured: desklib/
        ai-text-detector-v1.01, an open fine-tuned DeBERTa-v3-large that led
        the RAID detection leaderboard. `fakespot`, `academic`, `radar` and
        `fast` are four more published checkpoints, all the same wrapper with
        a different model; `radar` is the only one trained adversarially
        against a paraphraser, and `fast` is 33M parameters but flagged 8 of
        14 genuine human academic paragraphs here, so its row says so.

        The exception is `perplexity`: gpt2 is published, but the mapping from
        its perplexities to a probability was written in this repo and fitted
        to nothing. It reports `is_published_detector: false` and is excluded
        from the default. `classifier` is OpenAI's 2019 RoBERTa detector,
        published but measured *below chance* on this repo's corpus.

        There is no `heuristic` engine any more.

        None of these is GPTZero, whose current model is a proprietary
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
                "is_gptzero": all(r["name"] == "gptzero" for r in results) and bool(results),
                "disclaimer": "GPTZero's own verdict, from its paid API; cached on disk per text." if results and all(r["name"] == "gptzero" for r in results) else (
                    "Not GPTZero. Every engine here runs a published, "
                    "pretrained checkpoint downloaded from Hugging Face; this "
                    "project wrote none of the detection arithmetic. The "
                    "default engine 'modern' is "
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
                    "with 1 of 14 genuine human academic paragraphs over 0.5. "
                    "The other published engines on the same set: 'fakespot' "
                    "1.000 pairwise but 3 of 14 human paragraphs flagged, "
                    "'academic' 0.980 and 1 of 14, 'fast' 0.944 and 8 of 14, "
                    "'radar' 0.867 and 1 of 14 but it missed half the AI. The "
                    "one engine whose scoring this project wrote itself, "
                    "'perplexity', is excluded from the default, reports "
                    "is_published_detector false, and scored 0.740 pairwise "
                    "with 13 of 14 human paragraphs called AI; 'classifier' "
                    "scored 0.444 -- below chance. n=28, one author on the AI "
                    "side, no confidence interval. Per-sentence rows are far "
                    "less reliable than the document score and exist for the "
                    "heatmap only."
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

        The response carries the full before/after feature vectors and an AI
        probability for each, so a caller can show the measurement that
        justified the rewrite rather than a bare score. `risk_delta` is after
        minus before, so **negative is an improvement**.

        The probability comes from `ModernDetector` (a published fine-tuned
        DeBERTa-v3-large), not from anything scored by hand in this repo. It
        costs roughly 300ms per document and needs the `detectors` extra, so
        `ai_probability` and `risk_delta` are **null** when the model cannot
        run. That is a degraded 200, not a 500: the rewrite and every
        feature-level number are still there, and the frontend should render
        them rather than blocking on a score.

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

        # The front end is three files served from disk and edited often. A
        # browser that keeps an old app.js against a new API calls routes the
        # new page no longer uses and shows nothing (2026-09-08). So every
        # page asset revalidates on each load; the API responses are unaffected.
        @app.middleware("http")
        async def _no_stale_assets(request, call_next):  # type: ignore[no-untyped-def]
            response = await call_next(request)
            path = request.url.path
            if path == "/" or path.endswith((".html", ".js", ".css")):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
            return response

        # Maintenance: with HUMANIZER_MAINTENANCE on, every page answers the
        # holding page (web/maintenance.html, 503 with Retry-After) while the
        # API keeps working for the operator. Set in the environment and
        # deployed like any other setting; unset to come back.
        holding = web_root / "maintenance.html"

        @app.middleware("http")
        async def _maintenance(request, call_next):  # type: ignore[no-untyped-def]
            if maintenance_from_env() and not request.url.path.startswith("/api/") and holding.is_file():
                return FileResponse(
                    str(holding), status_code=503, media_type="text/html",
                    headers={"Retry-After": "3600", "Cache-Control": "no-store"},
                )
            return await call_next(request)
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
    auth: Optional[bool] = None,
) -> None:
    """Start the development server. Used by `humanizer serve`.

    `auth=None` reads HUMANIZER_AUTH (default on); `humanizer serve --no-auth`
    passes False. `create_app` is looked up at call time, so this gets the
    fully wrapped factory defined at the bottom of the module.
    """
    import uvicorn

    app = create_app(
        reference_dir=reference_dir,
        web_dir=web_dir,
        default_reference=reference,
        auth=auth,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


# ===========================================================================
# LLM humanizer: POST /api/humanize/llm, POST /api/humanize/stream,
#                POST /api/humanize/jobs, GET /api/humanize/stream?job=<id>
#
# Appended below `create_app` rather than edited into it, and wired in by
# re-binding `create_app` at the bottom of this block. Two constraints force
# that shape: the routes above are defined inside the factory, and the
# StaticFiles mount at "/" matches every path, so anything registered after it
# is unreachable. `_register_llm_routes` therefore adds its routes and then
# moves any Mount back to the end of the router's ordered list.
# ===========================================================================

import json as _json
import time as _time
import uuid as _uuid
from typing import Iterator as _Iterator

from fastapi import Request
from fastapi.responses import StreamingResponse

#: What every LLM endpoint says when MLX or the model is not usable. The
#: rule-based engine is a real fallback, not a consolation: it is the same
#: transforms the LLM path runs as its scrub stage, and it needs no GPU.
LLM_FALLBACK_NOTE = (
    "The rule-based engine is available as a fallback at POST /api/humanize. "
    "It applies the same deterministic transforms this pipeline uses as its "
    "scrub stage (connectives, AI vocabulary, paragraph template) and needs "
    "neither MLX nor a model download. Measured, it does not flip a "
    "detector's verdict either -- see the response's own numbers."
)

#: Ceiling on documents accepted by the LLM path. At the measured batched rate
#: (~200 tok/s aggregate on an M4 Pro) a 2,000-word document with six
#: candidates per paragraph is already several minutes, far past the 30s the
#: user accepted, so it is refused with a number rather than silently truncated.
LLM_MAX_WORDS = 1200

#: Cap on the number of queued SSE jobs kept in memory at once. Jobs are tiny
#: (a string and a config) and are deleted when their stream starts, so this is
#: a guard against a client that POSTs jobs and never connects, not a queue.
LLM_MAX_JOBS = 64

#: A queued job is discarded after this many seconds even if never collected.
LLM_JOB_TTL_S = 900.0


class LlmHumanizeRequest(BaseModel):
    """Body for POST /api/humanize/llm, /api/humanize/stream and /jobs."""

    text: str = Field(default="", description="Raw document text.")
    gptzero_api_key: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Optional. The caller's own GPTZero API key. When present, GPTZero "
            "itself judges the document and ranks every candidate for this "
            "request, at the caller's expense (about $1 per paragraph at eight "
            "candidates). Without it the free local estimate does both. The "
            "key is used for the request and never stored or logged."
        ),
    )
    facts: str = Field(
        default="",
        max_length=6000,
        description=(
            "Optional. Names, dates, figures and sources the author vouches "
            "for, as free text. A rewrite may add a number, date, quotation "
            "or citation only if it appears here; anything else new is "
            "rejected and listed in `summary.unverified_specifics`. The "
            "register style anchors sentences in specifics and invents them "
            "without this, so without facts it usually returns the draft "
            "unchanged."
        ),
    )
    n_candidates: int = Field(
        default=8,
        ge=1,
        le=16,
        description=(
            "Diverse candidates per paragraph. Each gets its own persona and "
            "structural instruction (research/08: candidate correlation, not "
            "N, is the binding constraint)."
        ),
    )
    style: str = Field(
        default="freeform",
        description=(
            "'freeform' (default) uses the base checkpoint, few-shot, with "
            "`facts` as the author's notes: the only style whose output "
            "GPTZero has read as human (6 of 9 bench paragraphs at best-of-6 "
            "with GPTZero scoring candidates; set HUMANIZER_PROXY=gptzero). "
            "'register' uses Qwen3-4B-Instruct with the trained LoRA: flips "
            "the four local detectors, none of GPTZero's verdicts. 'faithful' "
            "uses the plain instruction-tuned checkpoint and moves neither."
        ),
    )
    aggressiveness: str = Field(
        default="strong",
        description="Aggressiveness of the deterministic scrub that runs after the model.",
    )
    time_budget_s: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=3600.0,
        description=(
            "Soft wall-clock budget for the main rounds, checked only before "
            "round 2. Null scales with the document: 240 s per paragraph, at "
            "least 300 s, at most 2400 s, enough for round 1 at eight candidates "
            "so that round 2 actually runs."
        ),
    )
    rounds: int = Field(
        default=2,
        ge=1,
        le=3,
        description=(
            "Refinement rounds. Round 2 re-runs only the paragraphs still "
            "above 0.5, and may not make a paragraph worse."
        ),
    )
    model: Optional[str] = Field(
        default=None, description="Override the MLX model id."
    )
    repair_attempts: int = Field(
        default=0,
        ge=0,
        le=6,
        description=(
            "Diagnose-and-retry attempts for every paragraph still failing "
            "after the rounds. Each attempt reads the gate rejections and "
            "judge scores, names the cause, picks one action from a ladder "
            "and retries; the log is `paragraphs[i].repairs` and "
            "`summary.repair_log`. 0 disables the stage."
        ),
    )


def _request_judge(body: Any) -> Any:
    """A GPTZero yardstick built from the caller's own key, or None.

    Passed to the pipeline as `detector`, which makes it both the document
    yardstick and the candidate scorer for that one request. The owner's key
    (GPTZERO_API_KEY) is never consulted here: the product's own judge is the
    free surrogate, and paid judging happens only on the caller's credits.
    """
    key = (getattr(body, "gptzero_api_key", None) or "").strip()
    if not key:
        return None
    from ..detectors import yardstick as yardstick_mod
    from ..detectors.gptzero import GPTZeroClient

    return yardstick_mod.Yardstick(
        name="gptzero",
        kind="gptzero",
        model="GPTZero API (api.gptzero.me/v2/predict/text), caller's key",
        detector=GPTZeroClient(api_key=key),
        remote=True,
    )


#: Seconds of main-round budget per paragraph when the request leaves
#: `time_budget_s` null. Eight 7B candidates plus GPTZero scoring took about
#: 100 s a paragraph per round on the 2026-09-11 bench, and the budget is
#: only checked before round 2, so it must cover round 1 in full or the
#: second round never runs.
LLM_BUDGET_PER_PARAGRAPH_S = 240.0
LLM_BUDGET_MIN_S = 300.0
LLM_BUDGET_MAX_S = 2400.0
#: Soft budget for the repair stage, checked between attempts. Each attempt
#: regenerates one paragraph (about 46 s), so this allows two to three.
LLM_REPAIR_BUDGET_S = 120.0

#: One rewrite at a time, process-wide. Two MLX decodes on the same GPU at
#: once took the server down on 2026-09-20 (Metal: "Completed handler
#: provided after commit call"), and launchd's restart killed the rewrite the
#: browser was waiting on. A second request now waits here, in a visible
#: `queue` stage, until the first releases the model.
_RUN_LOCK = threading.Lock()
#: How often a waiting stream tells the client it is still waiting.
LLM_QUEUE_POLL_S = 2.0


def _queue_frame(status: str, waited: float) -> Dict[str, Any]:
    """A `progress` payload for the `queue` stage, same shape as the pipeline's."""
    if status == "done":
        detail = "the model is free; starting your rewrite after %.0fs in the queue" % waited
    else:
        detail = (
            "another rewrite is using the model; yours starts when it finishes "
            "(waiting %.0fs)" % waited
        )
    return {
        "stage": "queue",
        "status": status,
        "progress": 1.0 if status == "done" else 0.0,
        "detail": detail,
        "elapsed": round(waited, 3),
        "round": 1,
    }


def _scaled_budget(text: str) -> float:
    from ..text import split_paragraphs

    n = max(1, len(split_paragraphs(text or "")))
    return max(LLM_BUDGET_MIN_S, min(LLM_BUDGET_MAX_S, LLM_BUDGET_PER_PARAGRAPH_S * n))


def _llm_config(body: LlmHumanizeRequest) -> Any:
    from ..humanize.pipeline import PipelineConfig

    budget = body.time_budget_s if body.time_budget_s is not None else _scaled_budget(body.text)
    return PipelineConfig(
        n_candidates=body.n_candidates,
        style=body.style,
        aggressiveness=body.aggressiveness,
        time_budget_s=budget,
        rounds=body.rounds,
        model=body.model,
        facts=body.facts,
        repair_attempts=body.repair_attempts,
        repair_time_budget_s=LLM_REPAIR_BUDGET_S,
    )


def _llm_unavailable_payload(reason: str) -> Dict[str, Any]:
    """The 503 body. Named so the two endpoints cannot drift apart."""
    return {
        "error": "llm_unavailable",
        "detail": reason,
        "available": False,
        "fallback": LLM_FALLBACK_NOTE,
        "fallback_endpoint": "/api/humanize",
    }


def llm_precheck(body: LlmHumanizeRequest) -> Optional[JSONResponse]:
    """Validate a request without touching the GPU. None means "go ahead".

    Returns a ready-to-send `JSONResponse` for every refusal so that the
    blocking endpoint and the SSE endpoint reject identically. A missing model
    is a 503 with the fallback named, never a 500: `humanizer` is expected to
    run on machines with no Apple Silicon and no weights on disk.
    """
    from ..humanize import pipeline as pipeline_mod

    if not body.text or not body.text.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": "empty_text",
                "detail": "Field 'text' is required and must not be empty.",
            },
        )
    n_words = len(body.text.split())
    if n_words > LLM_MAX_WORDS:
        return JSONResponse(
            status_code=413,
            content={
                "error": "too_long",
                "detail": (
                    "%d words exceeds the %d-word limit for the LLM path. "
                    "Split the document, or use POST /api/humanize."
                    % (n_words, LLM_MAX_WORDS)
                ),
                "n_words": n_words,
                "limit": LLM_MAX_WORDS,
                "fallback_endpoint": "/api/humanize",
            },
        )
    if body.style not in ("register", "faithful", "freeform"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "unknown_style",
                "detail": "'style' must be 'register', 'faithful' or 'freeform'.",
            },
        )
    reason = pipeline_mod.pipeline_unavailable_reason()
    if reason is not None:
        return JSONResponse(
            status_code=503, content=_llm_unavailable_payload(reason)
        )
    return None


def sse_frame(event: str, data: Any) -> str:
    """One Server-Sent Events frame.

    `event:` then a single `data:` line carrying compact JSON, then the blank
    line that terminates the frame. The JSON is emitted with `ensure_ascii` so
    a curly quote in the document cannot produce a byte sequence that some
    proxy re-chunks in the middle of a UTF-8 codepoint, and it is passed
    through `json_safe` first because feature vectors legitimately contain NaN
    (see `api.models.json_safe`) and NaN is not JSON.
    """
    body = _json.dumps(json_safe(data), ensure_ascii=True, separators=(",", ":"))
    return "event: %s\ndata: %s\n\n" % (event, body)


#: Headers that stop an SSE response being buffered into uselessness.
#: `X-Accel-Buffering: no` is for nginx; `no-transform` stops compressing
#: proxies from holding the stream until it closes.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


#: Seconds of silence on an event stream before a comment frame is sent so
#: the proxies between the browser and this process (Vercel's rewrite, Fly's
#: edge) do not take the connection for dead. The long batch decode in
#: "Writing candidate versions" can go three minutes without a pipeline
#: event; a friend's run on 2026-10-06 ended with "the stream ended before a
#: result arrived" for exactly that reason. 15 s is well inside every idle
#: timeout we know of and costs a dozen bytes.
SSE_KEEPALIVE_S = 15.0
SSE_KEEPALIVE = ": keepalive\n\n"


def _with_keepalive(events: _Iterator[Any], interval: float = SSE_KEEPALIVE_S) -> _Iterator[Any]:
    """Yield the items of `events`, and `None` whenever `interval` seconds pass
    without one, so the caller can send a comment frame.

    The source generator is driven on a worker thread and handed over through
    a queue; this generator waits on the queue with a timeout. Closing this
    generator (the client went away) asks the pump to stop and closes the
    source at its next event, then waits for the pump so the caller's
    `finally` (which releases the model lock) runs after the source is done.
    Exceptions from the source are re-raised here, in order.
    """
    import queue as _queue

    q: "_queue.Queue[tuple]" = _queue.Queue()
    stop = threading.Event()

    def pump() -> None:
        try:
            for item in events:
                q.put(("event", item))
                if stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            q.put(("error", exc))
        finally:
            try:
                events.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            q.put(("end", None))

    worker = threading.Thread(target=pump, name="sse-pump", daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, value = q.get(timeout=interval)
            except _queue.Empty:
                yield None
                continue
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stop.set()
        worker.join(timeout=600)


def llm_event_stream(body: LlmHumanizeRequest) -> _Iterator[str]:
    """The SSE body: one `progress` frame per pipeline event, then `result`.

    WIRE FORMAT -- this is the contract the frontend consumes.

        Content-Type: text/event-stream

        event: progress
        data: {"stage":"analyze","status":"start","progress":0.0,
               "detail":"reading the document","elapsed":0.004,"round":1}

        ... one `progress` frame per event, in stage order ...

        event: result
        data: {<exactly the POST /api/humanize/llm body>}

        event: done
        data: {"ok":true}

    Rules a client can rely on:

    * `stage` is one of `analyze plan generate scrub score verify select
      finalize`, always in that order. A stage may be absent (`score` is
      skipped when no detector is installed) and the whole run repeats from
      `generate` when `rounds > 1`, with `round` incremented.
    * `status` is `start`, `progress`, `done` or `skip`. Every stage that runs
      emits exactly one terminal event (`done` or `skip`).
    * `progress` is 0.0-1.0 and never decreases inside one stage occurrence.
    * `detail` is a human-readable sentence, safe to render verbatim.
    * `elapsed` is seconds since the run started, monotonically increasing
      across the whole stream.
    * Exactly one `result` frame is sent, immediately before `done`.
    * On failure a single `error` frame is sent instead of `result`, carrying
      `{"error", "detail", "fallback", "fallback_endpoint"}`, and the HTTP
      status is still 200 because the headers were already flushed. Clients
      must branch on the event name, not on the status code.
    """
    from ..humanize import pipeline as pipeline_mod

    # Wait for the model. Each poll that fails sends a frame so the client
    # can show the wait and the connection stays alive; a client that gives
    # up closes this generator at the `yield`, before the lock is held.
    waited = 0.0
    while not _RUN_LOCK.acquire(timeout=LLM_QUEUE_POLL_S):
        first = waited == 0.0
        waited += LLM_QUEUE_POLL_S
        yield sse_frame("progress", _queue_frame("start" if first else "progress", waited))
    try:
        if waited:
            yield sse_frame("progress", _queue_frame("done", waited))
        try:
            source = pipeline_mod.stream(body.text, config=_llm_config(body), detector=_request_judge(body))
            for event in _with_keepalive(source):
                if event is None:
                    yield SSE_KEEPALIVE
                    continue
                payload = event.as_dict()
                result = payload.pop("result", None)
                yield sse_frame("progress", payload)
                if result is not None:
                    yield sse_frame("result", result)
            yield sse_frame("done", {"ok": True})
        except (RuntimeError, ValueError) as exc:
            yield sse_frame("error", _llm_unavailable_payload(str(exc)))
        except Exception as exc:  # noqa: BLE001 - the stream must not just stop
            yield sse_frame(
                "error",
                {
                    "error": "pipeline_failed",
                    "detail": "%s: %s" % (type(exc).__name__, exc),
                    "fallback": LLM_FALLBACK_NOTE,
                    "fallback_endpoint": "/api/humanize",
                },
            )
    finally:
        _RUN_LOCK.release()


def _reap_jobs(jobs: Dict[str, Dict[str, Any]]) -> None:
    """Drop expired jobs, then the oldest ones if the table is still over cap."""
    now = _time.time()
    for job_id in [k for k, v in jobs.items() if now - v["created"] > LLM_JOB_TTL_S]:
        jobs.pop(job_id, None)
    while len(jobs) > LLM_MAX_JOBS:
        oldest = min(jobs, key=lambda k: jobs[k]["created"])
        jobs.pop(oldest, None)


def _register_llm_routes(app: FastAPI) -> FastAPI:
    """Add the LLM routes to an app built by the original `create_app`."""
    app.state.llm_jobs = {}

    @app.get("/api/humanize/llm/health")
    def llm_health() -> JSONResponse:
        """Whether the LLM path can run, and what it would run.

        Cheap: nothing is imported from MLX beyond the module itself and no
        weights are touched. The frontend uses this to decide whether to offer
        the LLM engine or fall straight through to the rule-based one.
        """
        from ..detectors import yardstick as yardstick_mod
        from ..humanize import llm as llm_mod
        from ..humanize import pipeline as pipeline_mod

        reason = pipeline_mod.pipeline_unavailable_reason()
        return _ok(
            {
                "available": reason is None,
                "reason": reason,
                # `model` is the default style's checkpoint: the base model
                # since research/24 §6 (the instruct one is `instruct_model`).
                "default_style": pipeline_mod.PipelineConfig().style,
                "model": llm_mod.FREEFORM_MODEL,
                "instruct_model": llm_mod.DEFAULT_MODEL,
                "freeform_model": llm_mod.FREEFORM_MODEL,
                # Where generation runs: a remote OpenAI-compatible server
                # (HUMANIZER_LLM_URL, so this host needs no GPU) or local MLX.
                "generation": "remote" if llm_mod.remote_llm_url() else "mlx",
                "generation_url": llm_mod.remote_llm_url(),
                # Which detector judges the document, by configured name only:
                # resolving it here would load weights inside a health call.
                "yardstick": yardstick_mod.configured_name(),
                "stages": list(pipeline_mod.STAGES),
                "max_words": LLM_MAX_WORDS,
                "fallback_endpoint": "/api/humanize",
                "fallback": LLM_FALLBACK_NOTE,
            }
        )

    @app.post("/api/humanize/llm")
    def humanize_llm_endpoint(body: LlmHumanizeRequest) -> JSONResponse:
        """Rewrite a document with the local LLM. Blocking; use /stream to watch.

        The body is a superset of `POST /api/humanize`: `original`,
        `humanized`, `edits`, `before`, `after` and `summary` mean the same
        things, plus:

        * `model` -- the MLX model id that produced the rewrite.
        * `stages` -- one row per pipeline stage with `seconds` and `detail`.
        * `paragraphs` -- per paragraph, the source, the winner, every
          candidate, each candidate's AI probability, the gates it failed and
          its quality numbers.

        `summary` carries the numbers worth reading: `risk_delta` (after minus
        before, so negative is an improvement), `label_before`/`label_after`,
        `verdict_flipped`, `n_candidates_rejected` with `gate_rejections`
        broken down by gate, and `quality_flags`.

        **Read `verdict_flipped`, not `risk_delta`.** research/13's DUPE result
        is that attacks which halve false-negative rates still produce almost
        no actual "human" verdicts, and that is what was measured here too: the
        candidates that score below 0.5 are the ones that drifted off the
        source's content, and the gates reject them. This endpoint reports the
        rejections rather than quietly returning the best-scoring text.

        Returns 503 with `fallback_endpoint` when MLX or the model is missing,
        413 when the document is over `LLM_MAX_WORDS`, never 500 for either.
        """
        refusal = llm_precheck(body)
        if refusal is not None:
            return refusal

        from ..humanize import pipeline as pipeline_mod

        try:
            with _RUN_LOCK:
                result = pipeline_mod.humanize_llm(body.text, config=_llm_config(body), detector=_request_judge(body))
        except RuntimeError as exc:
            # Raised by the backend when the weights will not load.
            return JSONResponse(
                status_code=503, content=json_safe(_llm_unavailable_payload(str(exc)))
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "bad_request", "detail": str(exc)},
            )
        return _ok(result.as_dict())

    @app.post("/api/humanize/stream")
    def humanize_llm_stream_post(body: LlmHumanizeRequest) -> Any:
        """Server-Sent Events over POST. See `llm_event_stream` for the format.

        Use this from `fetch()` + `ReadableStream`. A browser `EventSource` can
        only issue GET requests, so for that path POST to
        `/api/humanize/jobs` first and then open
        `GET /api/humanize/stream?job=<id>`; both routes emit byte-identical
        frames from the same generator.

        Refusals that can be detected before the stream opens (empty text, too
        long, unknown style, MLX missing) are returned as an ordinary JSON
        error with a real status code. Anything that fails after the first
        byte arrives as an `error` frame with HTTP 200, because the status line
        is long gone by then.
        """
        refusal = llm_precheck(body)
        if refusal is not None:
            return refusal
        return StreamingResponse(
            llm_event_stream(body),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.post("/api/humanize/jobs")
    def humanize_llm_job(body: LlmHumanizeRequest) -> JSONResponse:
        """Park a request and return a `job_id` for `EventSource`.

        The job holds only the request body. Nothing runs until the matching
        `GET /api/humanize/stream?job=<id>` connects, and the job is removed
        the moment it does, so an id is single-use.
        """
        refusal = llm_precheck(body)
        if refusal is not None:
            return refusal
        jobs = app.state.llm_jobs
        _reap_jobs(jobs)
        job_id = _uuid.uuid4().hex
        jobs[job_id] = {"body": body, "created": _time.time()}
        return _ok(
            {
                "job_id": job_id,
                "stream_url": "/api/humanize/stream?job=%s" % job_id,
                "expires_in_s": LLM_JOB_TTL_S,
            }
        )

    @app.get("/api/humanize/stream")
    def humanize_llm_stream_get(job: str = "") -> Any:
        """`EventSource`-compatible SSE for a job parked by POST /api/humanize/jobs.

        Emits exactly the frames documented on `llm_event_stream`. An unknown
        or already-consumed job id is a 404 before the stream opens.
        """
        jobs = app.state.llm_jobs
        _reap_jobs(jobs)
        entry = jobs.pop(job, None)
        if entry is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "unknown_job",
                    "detail": (
                        "No queued job %r. Ids are single-use and expire after "
                        "%.0f seconds; POST /api/humanize/jobs for a new one."
                        % (job, LLM_JOB_TTL_S)
                    ),
                },
            )
        return StreamingResponse(
            llm_event_stream(entry["body"]),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # A Mount at "/" matches every path, so the static frontend has to sit
    # behind the routes just added. Starlette matches in list order, so moving
    # the mounts to the end is the whole fix.
    from starlette.routing import Mount

    routes = app.router.routes
    mounts = [r for r in routes if isinstance(r, Mount)]
    if mounts:
        app.router.routes = [r for r in routes if not isinstance(r, Mount)] + mounts
    return app


_create_app_without_llm = create_app


def create_app(  # noqa: F811 - deliberate: wraps the definition above
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
    default_reference: Optional[str] = None,
) -> FastAPI:
    """Build the ASGI application, including the LLM humanizer routes.

    Identical to the definition above plus `_register_llm_routes`. It is a
    wrapper rather than an edit so that this whole LLM block is append-only
    against the file it landed in; `_create_app_without_llm` is kept as the
    unwrapped factory for tests that want the smaller surface.
    """
    app = _create_app_without_llm(
        reference_dir=reference_dir,
        web_dir=web_dir,
        default_reference=default_reference,
    )
    return _register_llm_routes(app)


# ===========================================================================
# Detector-guided decoding: POST /api/humanize/guided,
#                           POST /api/humanize/guided/stream,
#                           POST /api/humanize/guided/jobs,
#                           GET  /api/humanize/guided/stream?job=<id>,
#                           GET  /api/humanize/guided/health
#
# Appended below the LLM block for the same two reasons that block was
# appended below `create_app`: the routes live inside a factory, and the
# StaticFiles mount at "/" matches every path. `_register_guided_routes`
# therefore adds its routes and then moves any Mount back to the end, exactly
# as `_register_llm_routes` does, and `create_app` is re-bound once more at the
# bottom. Nothing above this line was edited.
#
# The frames are byte-identical in shape to `llm_event_stream`'s: same event
# names, same key set on `progress`, same `result`-then-`done` ordering, same
# `error` frame on failure. A client that renders one renders the other.
# ===========================================================================


#: What the guided endpoints say when they cannot run. Two fallbacks are named
#: because there are two different failures: no MLX means no generation at all
#: (rule-based engine), while a working MLX with a missing guide falls back to
#: the paragraph-level LLM path, which needs no guide.
GUIDED_FALLBACK_NOTE = (
    "Detector-guided decoding needs mlx-lm on Apple Silicon. Without it, POST "
    "/api/humanize/llm runs the paragraph-level best-of-N pipeline (also MLX), "
    "and POST /api/humanize runs the deterministic rule-based engine, which "
    "needs no GPU and no model download. Measured, none of the three flips a "
    "detector's verdict -- read the response's own numbers, not the marketing."
)

#: Guided decoding costs `beam x samples` generations *per sentence*, so the
#: word ceiling is lower than the LLM path's 1200. At the measured ~2.0s per
#: 12-prompt batch a 300-word paragraph is already ~20s.
GUIDED_MAX_WORDS = 600


class GuidedHumanizeRequest(BaseModel):
    """Body for POST /api/humanize/guided and its streaming variants."""

    text: str = Field(default="", description="Raw document text.")
    facts: str = Field(
        default="",
        max_length=6000,
        description=(
            "Optional. Same meaning as on /api/humanize/llm: names, dates and "
            "figures the author vouches for, which the paragraph gate then "
            "allows a candidate to contain."
        ),
    )
    beam_width: int = Field(
        default=3,
        ge=1,
        le=6,
        description=(
            "Hypotheses carried between sentences. 1 is greedy. research/08 "
            "§1.5: with an imperfect verifier the optimal number of sampling "
            "attempts is usually under 10, and beam x samples is that number."
        ),
    )
    n_samples: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Continuations proposed per hypothesis per sentence.",
    )
    mode: str = Field(
        default="segment",
        description=(
            "'segment' scores one candidate sentence at a time (tractable). "
            "'token' is the published per-token method with the guide "
            "throttled to every `guidance_interval` tokens; it cannot batch "
            "and is far slower."
        ),
    )
    guide_context: str = Field(
        default="full",
        description=(
            "What the guide sees: 'full' (rewritten prefix + candidate + the "
            "source's remaining sentences, constant length), 'prefix' "
            "(prefix + candidate only) or 'sentence' (the candidate alone)."
        ),
    )
    guidance_interval: int = Field(
        default=8, ge=1, le=64, description="Tokens between guide calls in 'token' mode."
    )
    top_k: int = Field(
        default=8, ge=2, le=32, description="Next tokens scored per guided step."
    )
    guidance_strength: float = Field(
        default=4.0,
        ge=0.0,
        le=50.0,
        description=(
            "How hard the guide overrides the LM's own next-token ranking. "
            "0.0 is plain sampling and is the control."
        ),
    )
    aggressiveness: str = Field(
        default="strong",
        description="Aggressiveness of the deterministic scrub that runs after the search.",
    )
    time_budget_s: float = Field(
        default=30.0, gt=0.0, le=600.0, description="Soft wall-clock budget."
    )
    model: Optional[str] = Field(
        default=None, description="Override the MLX generation model id."
    )
    guide_model: Optional[str] = Field(
        default=None, description="Override the in-loop guide detector."
    )
    use_guide: bool = Field(
        default=True,
        description=(
            "False runs the same beam search with the same gates and no "
            "guide. That is the control: it isolates how much of any movement "
            "came from the guide rather than from resampling."
        ),
    )


def _guided_config(body: GuidedHumanizeRequest) -> Any:
    from ..humanize.guided import GUIDE_MODEL, GuidedConfig

    return GuidedConfig(
        beam_width=body.beam_width,
        n_samples=body.n_samples,
        mode=body.mode,
        guide_context=body.guide_context,
        guidance_interval=body.guidance_interval,
        top_k=body.top_k,
        guidance_strength=body.guidance_strength,
        aggressiveness=body.aggressiveness,
        time_budget_s=body.time_budget_s,
        model=body.model,
        guide_model=body.guide_model or GUIDE_MODEL,
        facts=body.facts,
    )


def _guided_unavailable_payload(reason: str) -> Dict[str, Any]:
    """The 503 body. Named so every guided endpoint refuses identically."""
    return {
        "error": "guided_unavailable",
        "detail": reason,
        "available": False,
        "fallback": GUIDED_FALLBACK_NOTE,
        "fallback_endpoint": "/api/humanize/llm",
        "fallback_endpoint_no_gpu": "/api/humanize",
    }


def guided_precheck(body: GuidedHumanizeRequest) -> Optional[JSONResponse]:
    """Validate a guided request without touching the GPU. None means go.

    Mirrors `llm_precheck` deliberately, including returning a ready-made
    `JSONResponse` for every refusal so the blocking and streaming endpoints
    cannot drift apart. A missing MLX is a 503 naming two fallbacks, never a
    500.
    """
    from ..humanize import guided as guided_mod

    if not body.text or not body.text.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": "empty_text",
                "detail": "Field 'text' is required and must not be empty.",
            },
        )
    n_words = len(body.text.split())
    if n_words > GUIDED_MAX_WORDS:
        return JSONResponse(
            status_code=413,
            content={
                "error": "too_long",
                "detail": (
                    "%d words exceeds the %d-word limit for guided decoding, "
                    "which pays beam x samples generations per sentence. "
                    "Split the document, or use POST /api/humanize/llm."
                    % (n_words, GUIDED_MAX_WORDS)
                ),
                "n_words": n_words,
                "limit": GUIDED_MAX_WORDS,
                "fallback_endpoint": "/api/humanize/llm",
            },
        )
    if body.mode not in guided_mod.GuideMode.ALL:
        return JSONResponse(
            status_code=400,
            content={
                "error": "unknown_mode",
                "detail": "'mode' must be one of %s."
                % (", ".join(guided_mod.GuideMode.ALL)),
            },
        )
    if body.guide_context not in guided_mod.GuideContext.ALL:
        return JSONResponse(
            status_code=400,
            content={
                "error": "unknown_guide_context",
                "detail": "'guide_context' must be one of %s."
                % (", ".join(guided_mod.GuideContext.ALL)),
            },
        )
    reason = guided_mod.guided_unavailable_reason()
    if reason is not None:
        return JSONResponse(
            status_code=503, content=_guided_unavailable_payload(reason)
        )
    return None


def guided_event_stream(body: GuidedHumanizeRequest) -> _Iterator[str]:
    """The SSE body for guided decoding. Same wire format as `llm_event_stream`.

    Frames, in order:

        event: progress
        data: {"stage":"analyze","status":"start","progress":0.0,
               "detail":"reading the document","elapsed":0.004,"round":1}

        ... one `progress` frame per event, in stage order ...

        event: result
        data: {<exactly the POST /api/humanize/guided body>}

        event: done
        data: {"ok":true}

    Every rule documented on `llm_event_stream` holds here unchanged: the
    stage vocabulary is the same eight names in the same order (guided
    decoding reuses `pipeline.STAGES`, it does not invent its own), `status`
    is `start`/`progress`/`done`/`skip` with exactly one terminal event per
    stage that runs, `progress` never decreases inside a stage, `elapsed`
    increases across the whole stream, exactly one `result` frame arrives
    immediately before `done`, and a failure after the headers are flushed
    arrives as a single `error` frame with HTTP 200.

    `round` is always 1: guided decoding has no refinement rounds. It is still
    emitted because the frontend reads it, and a client should not have to
    branch on which pipeline produced the frame.
    """
    from ..humanize import guided as guided_mod

    try:
        source = guided_mod.stream(
            body.text,
            config=_guided_config(body),
            guide=None if body.use_guide else False,
        )
        for event in _with_keepalive(source):
            if event is None:
                yield SSE_KEEPALIVE
                continue
            payload = event.as_dict()
            result = payload.pop("result", None)
            yield sse_frame("progress", payload)
            if result is not None:
                yield sse_frame("result", result)
        yield sse_frame("done", {"ok": True})
    except (RuntimeError, ValueError) as exc:
        yield sse_frame("error", _guided_unavailable_payload(str(exc)))
    except Exception as exc:  # noqa: BLE001 - the stream must not just stop
        yield sse_frame(
            "error",
            {
                "error": "guided_failed",
                "detail": "%s: %s" % (type(exc).__name__, exc),
                "fallback": GUIDED_FALLBACK_NOTE,
                "fallback_endpoint": "/api/humanize/llm",
                "fallback_endpoint_no_gpu": "/api/humanize",
            },
        )


def _register_guided_routes(app: FastAPI) -> FastAPI:
    """Add the detector-guided decoding routes to an already-built app."""
    if not hasattr(app.state, "guided_jobs"):
        app.state.guided_jobs = {}

    @app.get("/api/humanize/guided/health")
    def guided_health() -> JSONResponse:
        """Whether guided decoding can run, and exactly what it would run.

        Cheap: no weights are touched. Publishes `stages` so the frontend
        builds its checklist from the service rather than from a hard-coded
        list, and publishes `guide` and `yardstick` separately because the
        distinction between the two is the whole design -- the guide steers
        the search, the yardstick produces the number the response reports.
        """
        from ..humanize import guided as guided_mod
        from ..humanize import llm as llm_mod

        reason = guided_mod.guided_unavailable_reason()
        guide_reason = guided_mod.guide_unavailable_reason()
        return _ok(
            {
                "available": reason is None,
                "reason": reason,
                "model": llm_mod.DEFAULT_MODEL,
                "stages": list(guided_mod.STAGES),
                "modes": list(guided_mod.GuideMode.ALL),
                "guide_contexts": list(guided_mod.GuideContext.ALL),
                "guide": {
                    "model": guided_mod.GUIDE_MODEL,
                    "engine": guided_mod.GUIDE_ENGINE,
                    "available": guide_reason is None,
                    "reason": guide_reason,
                    "role": (
                        "in-loop search heuristic. Never reported as a "
                        "verdict; it calls 8 of 14 genuine human PMC "
                        "paragraphs AI on this repo's own bench."
                    ),
                },
                "yardstick": {
                    "model": guided_mod.YARDSTICK_MODEL,
                    "role": (
                        "the reported score. Every probability in the "
                        "response body comes from here."
                    ),
                },
                "max_words": GUIDED_MAX_WORDS,
                "fallback_endpoint": "/api/humanize/llm",
                "fallback_endpoint_no_gpu": "/api/humanize",
                "fallback": GUIDED_FALLBACK_NOTE,
            }
        )

    @app.post("/api/humanize/guided")
    def humanize_guided_endpoint(body: GuidedHumanizeRequest) -> JSONResponse:
        """Rewrite a document with detector-guided decoding. Blocking.

        Generation runs sentence by sentence under a beam search whose
        ordering comes from a fast guide detector
        (`MayZhou/e5-small-lora-ai-generated-detector`, ~28 ms a document).
        The finalists are scrubbed, gated by the *same* meaning gates
        `POST /api/humanize/llm` uses, and then scored by the yardstick
        (`desklib/ai-text-detector-v1.01`).

        The body is shaped like `POST /api/humanize/llm`'s -- `original`,
        `humanized`, `edits`, `before`, `after`, `summary`, `model`, `backend`,
        `stages`, `paragraphs` -- plus a `guide` block. `edits` is always
        empty: a sentence-by-sentence regeneration has no honest span-edit
        representation, so diff `paragraphs[i].original` against
        `paragraphs[i].humanized` instead.

        **`after.ai_probability`, `summary.risk_delta` and
        `summary.verdict_flipped` are the yardstick's numbers and never the
        guide's.** The guide's opinion lives in `guide.probability_before` /
        `guide.probability_after` and is labelled as a search heuristic.
        `summary.guide_transfer` reports how far the two disagreed on this
        run, because research/08 §3.2 measures detector-to-detector transfer
        as "real but wildly uneven" and a mean improvement number against a
        proxy is actively misleading on its own.

        Returns 503 with both fallbacks named when MLX is missing, 413 over
        `GUIDED_MAX_WORDS`, 400 on an unknown mode -- never 500 for any of
        them.
        """
        refusal = guided_precheck(body)
        if refusal is not None:
            return refusal

        from ..humanize import guided as guided_mod

        try:
            result = guided_mod.humanize_guided(
                body.text,
                config=_guided_config(body),
                guide=None if body.use_guide else False,
            )
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content=json_safe(_guided_unavailable_payload(str(exc))),
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "bad_request", "detail": str(exc)},
            )
        return _ok(result.as_dict())

    @app.post("/api/humanize/guided/stream")
    def humanize_guided_stream_post(body: GuidedHumanizeRequest) -> Any:
        """Server-Sent Events over POST. See `guided_event_stream` for the format.

        Use from `fetch()` + `ReadableStream`. A browser `EventSource` issues
        GET only, so for that path POST to `/api/humanize/guided/jobs` first
        and open `GET /api/humanize/guided/stream?job=<id>`; both routes emit
        byte-identical frames from the same generator.
        """
        refusal = guided_precheck(body)
        if refusal is not None:
            return refusal
        return StreamingResponse(
            guided_event_stream(body),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.post("/api/humanize/guided/jobs")
    def humanize_guided_job(body: GuidedHumanizeRequest) -> JSONResponse:
        """Park a guided request and return a single-use `job_id`."""
        refusal = guided_precheck(body)
        if refusal is not None:
            return refusal
        jobs = app.state.guided_jobs
        _reap_jobs(jobs)
        job_id = _uuid.uuid4().hex
        jobs[job_id] = {"body": body, "created": _time.time()}
        return _ok(
            {
                "job_id": job_id,
                "stream_url": "/api/humanize/guided/stream?job=%s" % job_id,
                "expires_in_s": LLM_JOB_TTL_S,
            }
        )

    @app.get("/api/humanize/guided/stream")
    def humanize_guided_stream_get(job: str = "") -> Any:
        """`EventSource`-compatible SSE for a parked guided job.

        Emits exactly the frames documented on `guided_event_stream`. An
        unknown or already-consumed id is a 404 before the stream opens.
        """
        jobs = app.state.guided_jobs
        _reap_jobs(jobs)
        entry = jobs.pop(job, None)
        if entry is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "unknown_job",
                    "detail": (
                        "No queued guided job %r. Ids are single-use and "
                        "expire after %.0f seconds; POST "
                        "/api/humanize/guided/jobs for a new one."
                        % (job, LLM_JOB_TTL_S)
                    ),
                },
            )
        return StreamingResponse(
            guided_event_stream(entry["body"]),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    from starlette.routing import Mount

    routes = app.router.routes
    mounts = [r for r in routes if isinstance(r, Mount)]
    if mounts:
        app.router.routes = [r for r in routes if not isinstance(r, Mount)] + mounts
    return app


_create_app_without_guided = create_app


def create_app(  # noqa: F811 - deliberate: wraps the definition above
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
    default_reference: Optional[str] = None,
) -> FastAPI:
    """Build the ASGI application, including the detector-guided routes.

    Identical to the definition above plus `_register_guided_routes`. A
    wrapper rather than an edit, so this whole block stays append-only against
    the file it landed in; `_create_app_without_guided` is kept as the
    LLM-only factory and `_create_app_without_llm` as the smallest one.
    """
    app = _create_app_without_guided(
        reference_dir=reference_dir,
        web_dir=web_dir,
        default_reference=default_reference,
    )
    return _register_guided_routes(app)


# ===========================================================================
# Resident-model inventory: GET  /api/models
#                           POST /api/models/release
#
# Appended below the guided block for the reasons that block gives. Every
# model cache in the package (`detectors.local`, `humanize.pretrained`,
# `humanize.llm`, `humanize.guided`) is bounded on its own; these two routes
# let an operator see what is resident and give it all back without
# restarting the process. The modules are imported inside the handlers so the
# app still builds on a machine with neither torch nor MLX.
# ===========================================================================


def _models_payload() -> Dict[str, Any]:
    from .. import memory as memory_mod
    from ..detectors import local as local_mod
    from ..humanize import llm as llm_mod
    from ..humanize import pretrained as pretrained_mod

    return {
        "detectors": local_mod.loaded_models(),
        "pretrained": pretrained_mod.loaded_models(),
        "llm": llm_mod.loaded_backends(),
        "rss_mb": memory_mod.rss_mb(),
    }


def _release_all_models() -> None:
    from ..detectors import local as local_mod
    from ..humanize import guided as guided_mod
    from ..humanize import llm as llm_mod
    from ..humanize import pretrained as pretrained_mod

    # Wrappers first, weights second: the guide cache holds the last references
    # to detector weights, so clearing it before the detector cache is what lets
    # the memory actually go.
    guided_mod.clear_guide_cache()
    llm_mod.release_backends()
    pretrained_mod.clear_model_cache()
    local_mod.clear_model_cache()


def _register_models_routes(app: FastAPI) -> FastAPI:
    """Add the resident-model routes to an app built by `create_app`."""

    @app.get("/api/models")
    def models_endpoint() -> JSONResponse:
        """What is resident right now, and how big the process has grown."""
        return _ok(_models_payload())

    @app.post("/api/models/release")
    def models_release_endpoint() -> JSONResponse:
        """Unload every cached model and report what is left (nothing)."""
        _release_all_models()
        payload = _models_payload()
        payload["released"] = True
        return _ok(payload)

    from starlette.routing import Mount

    routes = app.router.routes
    mounts = [r for r in routes if isinstance(r, Mount)]
    if mounts:
        app.router.routes = [r for r in routes if not isinstance(r, Mount)] + mounts
    return app


_create_app_without_models = create_app


def create_app(  # noqa: F811 - deliberate: wraps the definition above
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
    default_reference: Optional[str] = None,
) -> FastAPI:
    """Build the ASGI application, including the resident-model routes.

    Identical to the definition above plus `_register_models_routes`, kept as
    a wrapper so this block is append-only like the two before it.
    """
    app = _create_app_without_models(
        reference_dir=reference_dir,
        web_dir=web_dir,
        default_reference=default_reference,
    )
    return _register_models_routes(app)


# ===========================================================================
# Accounts and sign-in gating: GET  /  /signin  /signup  /app  /index.html
#                              POST /api/auth/signup  /signin  /signout
#                              GET  /api/auth/me
#
# Appended below the models block for the reasons that block gives. The
# implementation lives in `humanizer.api.auth`; this block only decides
# whether it is on, builds the store, registers it, and moves the static
# mount back to the end of the router. `create_app` gains one parameter:
#
#     auth: bool | None   None reads HUMANIZER_AUTH (default "1", on).
#
# With auth on, the page routes here shadow the StaticFiles `html=True`
# behaviour for "/" and "/index.html". Whether /app and the product API
# need a session is `HUMANIZER_PUBLIC_APP` (default on: public, and the
# paywall limits visitors by address); see `humanizer.api.auth`. With auth
# off nothing is registered and the app serves exactly as before.
# ===========================================================================

from . import auth as _auth_mod  # noqa: E402

AUTH_ENV = "HUMANIZER_AUTH"


def auth_enabled_from_env() -> bool:
    """`HUMANIZER_AUTH` as a bool; unset or anything but 0/false/no/off is on."""
    return os.environ.get(AUTH_ENV, "1").strip().lower() not in ("0", "false", "no", "off", "")


def _register_auth(app: FastAPI, auth_db: Optional[Path], public_app: Optional[bool] = None) -> FastAPI:
    """Install accounts on an app built by `create_app`, then re-order mounts."""
    store = _auth_mod.AuthStore(auth_db or _auth_mod.default_auth_db_path())
    _auth_mod.register_auth(app, store, getattr(app.state, "web_dir", None), public_app=public_app)
    # Google and Apple sign-in, configured by environment; unconfigured
    # providers report false at /api/auth/providers and redirect with an error.
    from . import oauth as _oauth_mod

    _oauth_mod.register_oauth(app, store)

    from starlette.routing import Mount

    routes = app.router.routes
    mounts = [r for r in routes if isinstance(r, Mount)]
    if mounts:
        app.router.routes = [r for r in routes if not isinstance(r, Mount)] + mounts
    return app


_create_app_without_auth = create_app


def create_app(  # noqa: F811 - deliberate: wraps the definition above
    reference_dir: Optional[Path] = None,
    web_dir: Optional[Path] = None,
    default_reference: Optional[str] = None,
    auth: Optional[bool] = None,
    auth_db: Optional[Path] = None,
    public_app: Optional[bool] = None,
) -> FastAPI:
    """Build the ASGI application, with accounts and sign-in gating.

    `auth=None` reads `HUMANIZER_AUTH` (default on). `auth_db` overrides the
    SQLite path (`HUMANIZER_AUTH_DB`, else `data/auth.sqlite`). `public_app`
    (None reads `HUMANIZER_PUBLIC_APP`, default on) serves /app and the
    product API without a session. Everything else is identical to the
    definition above.
    """
    app = _create_app_without_auth(
        reference_dir=reference_dir,
        web_dir=web_dir,
        default_reference=default_reference,
    )
    enabled = auth_enabled_from_env() if auth is None else bool(auth)
    app.state.auth_enabled = enabled
    if enabled:
        _register_auth(app, auth_db, public_app=public_app)
    return app
