"""Document analysis report.

Combines feature extraction, human-band checks, reference distance and detector
scores into one object that can be printed or serialised.

The findings list is ordered by the research/09 conflict matrix, so what
surfaces first is what is both safe to change and worth changing. That matrix
is the reason this product can exist: the four highest-value humanizing edits
(strip formal connectives, bring sentence-length variance into band, remove AI
vocabulary, break the paragraph template) all carry zero or negative cost to an
essay grade, while the four most grade-damaging edits carry only medium
detector benefit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..clean import clean_markup, looks_like_markup
from ..features import band_report, extract_features
from ..features.punctuation import MAX_ACADEMIC_CONTRACTIONS_PER_1K
from ..features.shape import HUMAN_SENTENCE_CV
from ..reference import ReferenceDistribution

# Severity ordering for findings.
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Finding:
    """One actionable observation about a document."""

    code: str
    severity: str
    message: str
    detail: str = ""
    grade_cost: str = "none"  # from the research/09 conflict matrix

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
            "grade_cost": self.grade_cost,
        }


@dataclass
class DocumentReport:
    features: Dict[str, float]
    bands: Dict[str, str]
    findings: List[Finding] = field(default_factory=list)
    reference: Optional[Dict[str, float]] = None
    detectors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    deviations: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "bands": self.bands,
            "findings": [f.as_dict() for f in self.findings],
            "reference": self.reference,
            "detectors": self.detectors,
            "deviations": self.deviations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=_json_default)

    def render(self) -> str:
        lines: List[str] = []
        f = self.features
        lines.append(
            f"{int(f.get('n_words', 0))} words, "
            f"{int(f.get('n_sentences', 0))} sentences, "
            f"{int(f.get('n_paragraphs', 0))} paragraphs"
        )

        if f.get("n_words", 0) < 300:
            lines.append(
                "  note: under 300 words. Shape features are noisy and "
                "detectors hedge below this length."
            )

        lines.append("")
        lines.append("Shape")
        for key, label in (
            ("sent_len_mean", "sentence length mean"),
            ("sent_len_cv", "sentence length CV"),
            ("sent_lag1_autocorr", "lag-1 autocorrelation"),
            ("sent_spectral_beta", "spectral beta"),
            ("sent_short_share", "share under 10 words"),
            ("sent_long_share", "share over 30 words"),
            ("para_words_cv", "paragraph length CV"),
        ):
            lines.append(
                f"  {label:<26} {_fmt(f.get(key))}{_band_suffix(self.bands.get(key))}"
            )

        lines.append("")
        lines.append("Register and lexicon")
        for key, label in (
            ("ai_vocab_weighted_per_1k", "AI vocabulary (weighted/1k)"),
            ("formal_connective_per_1k", "formal connectives /1k"),
            ("participial_tail_per_1k", "participial tails /1k"),
            ("tricolon_per_1k", "tricolons /1k"),
            ("nominalization_per_1k", "nominalizations /1k"),
            ("passive_per_1k", "passives /1k"),
            ("hedge_per_1k", "hedges /1k"),
            ("comma_per_1k", "commas /1k"),
            ("contraction_per_1k", "contractions /1k"),
            ("mtld", "MTLD"),
        ):
            lines.append(
                f"  {label:<26} {_fmt(f.get(key))}{_band_suffix(self.bands.get(key))}"
            )

        if self.reference:
            lines.append("")
            lines.append("Reference distance")
            lines.append(
                f"  Mahalanobis {_fmt(self.reference.get('distance'))} "
                f"at the {_fmt(self.reference.get('percentile'))} percentile "
                f"of the human corpus "
                f"({int(self.reference.get('n_features', 0))} features, "
                f"coverage {_fmt(self.reference.get('coverage'))})"
            )
            if self.deviations:
                lines.append("  largest deviations:")
                for name, z in list(self.deviations.items())[:5]:
                    lines.append(f"    {name:<32} z = {z:+.1f}")

        if self.detectors:
            lines.append("")
            lines.append("Detectors")
            for name, res in self.detectors.items():
                lines.append(
                    f"  {name:<12} p(ai) = {_fmt(res.get('ai_probability'))}"
                    f"  verdict = {res.get('label', 'n/a')}"
                )

        if self.findings:
            lines.append("")
            lines.append("Findings")
            for finding in self.findings:
                lines.append(f"  [{finding.severity}] {finding.message}")
                if finding.detail:
                    lines.append(f"      {finding.detail}")
                if finding.grade_cost != "none":
                    lines.append(f"      grade cost if changed: {finding.grade_cost}")
        else:
            lines.append("")
            lines.append("No findings.")

        return "\n".join(lines)


def analyze(
    text: str,
    reference: Optional[ReferenceDistribution] = None,
    detectors: Optional[List] = None,
    auto_clean: bool = True,
) -> DocumentReport:
    """Full Stage 0 analysis of one document."""
    if auto_clean and looks_like_markup(text):
        text = clean_markup(text)

    feats = extract_features(text)
    bands = band_report(feats)
    report = DocumentReport(features=feats, bands=bands)

    if reference is not None:
        report.reference = reference.score(feats)
        zs = reference.zscores(feats)
        report.deviations = dict(
            sorted(zs.items(), key=lambda kv: -abs(kv[1]))[:10]
        )

    for detector in detectors or []:
        try:
            result = detector.score(text)
        except Exception as exc:  # a missing key or offline model is not fatal
            report.detectors[detector.name] = {"error": str(exc)}
            continue
        report.detectors[detector.name] = {
            "ai_probability": result.ai_probability,
            "label": result.label,
            "confidence": result.confidence,
        }

    report.findings = build_findings(feats, bands)
    return report


def build_findings(feats: Dict[str, float], bands: Dict[str, str]) -> List[Finding]:
    """Derive actionable findings, ordered by the research/09 conflict matrix."""
    out: List[Finding] = []

    # --- Tier 1: high detector benefit, zero or negative grade cost ---------

    ai_vocab = feats.get("ai_vocab_weighted_per_1k", 0.0) or 0.0
    if ai_vocab > 8.0:
        out.append(
            Finding(
                code="ai_vocabulary",
                severity="high" if ai_vocab > 20 else "medium",
                message=f"AI vocabulary density is {ai_vocab:.1f} per 1,000 words.",
                detail=(
                    "Highest-effect lexical signal in the catalog and it "
                    "survives paraphrasing. Removing it also removes the red "
                    "highlights a human grader sees."
                ),
                grade_cost="none, slightly positive",
            )
        )

    formal = feats.get("formal_connective_per_1k", 0.0) or 0.0
    if formal > 8.0:
        out.append(
            Finding(
                code="formal_connectives",
                severity="high" if formal > 15 else "medium",
                message=f"Formal connectives run {formal:.1f} per 1,000 words.",
                detail=(
                    "Stripping these is the single best joint move: high "
                    "detector benefit, and it raises the grade because top-band "
                    "coherence 'attracts no attention'."
                ),
                grade_cost="negative, it raises the grade",
            )
        )

    opener_share = feats.get("para_opener_formal_share", 0.0) or 0.0
    if opener_share > 0.04:
        out.append(
            Finding(
                code="paragraph_openers",
                severity="medium",
                message=(
                    f"{opener_share:.0%} of paragraphs open with a formal "
                    "connective."
                ),
                detail="Human academic prose runs at most 3-4%.",
                grade_cost="none",
            )
        )

    cv = feats.get("sent_len_cv")
    if bands.get("sent_len_cv") == "low":
        out.append(
            Finding(
                code="uniform_sentences",
                severity="high",
                message=f"Sentence-length CV is {_fmt(cv)}, below the human band.",
                detail=(
                    f"Human academic prose runs {HUMAN_SENTENCE_CV[0]}-"
                    f"{HUMAN_SENTENCE_CV[1]}. Widen the tails rather than "
                    "the middle: target 9-14% of sentences under 10 words and "
                    "18-28% over 30."
                ),
                grade_cost="none, rubrics reward sentence variety",
            )
        )
    elif bands.get("sent_len_cv") == "high":
        out.append(
            Finding(
                code="overshot_variance",
                severity="medium",
                message=f"Sentence-length CV is {_fmt(cv)}, above the human band.",
                detail=(
                    "Overshooting is as anomalous as undershooting. A CV of "
                    "0.85 sits as far outside the human distribution as 0.30."
                ),
                grade_cost="none",
            )
        )

    tricolon = feats.get("tricolon_per_1k", 0.0) or 0.0
    negpar = feats.get("negative_parallel_per_1k", 0.0) or 0.0
    if tricolon > 3.0 or negpar > 1.5:
        out.append(
            Finding(
                code="parallelism",
                severity="medium",
                message=(
                    f"Rhetorical parallelism is heavy "
                    f"(tricolons {tricolon:.1f}/1k, "
                    f"'not X but Y' {negpar:.1f}/1k)."
                ),
                detail="AI text runs about 2x human on tricolons and 3x on negative parallelism.",
                grade_cost="none",
            )
        )

    tails = feats.get("participial_tail_per_1k", 0.0) or 0.0
    if tails > 4.0:
        out.append(
            Finding(
                code="participial_tails",
                severity="medium",
                message=f"Sentence-final participial clauses run {tails:.1f} per 1,000 words.",
                detail="One of the strongest syntactic tells, about 5.3x the human rate.",
                grade_cost="none",
            )
        )

    # --- Tier 2: structural signals that need care -------------------------

    if bands.get("sent_lag1_autocorr") == "low":
        out.append(
            Finding(
                code="alternating_sentences",
                severity="medium",
                message=(
                    "Sentence lengths alternate long and short "
                    f"(lag-1 autocorrelation {_fmt(feats.get('sent_lag1_autocorr'))})."
                ),
                detail=(
                    "Humans do not alternate; their lag-1 sits near zero and "
                    "the real structure is long-range. This is the signature "
                    "of a naive burstiness rule."
                ),
                grade_cost="none",
            )
        )

    para_cv = feats.get("para_words_cv")
    if para_cv is not None and math.isfinite(para_cv) and para_cv < 0.35:
        out.append(
            Finding(
                code="uniform_paragraphs",
                severity="medium",
                message=f"Paragraph lengths are uniform (CV {_fmt(para_cv)}).",
                detail="Human paragraph CV runs 0.42-0.71, above the sentence CV in every genre.",
                grade_cost="none",
            )
        )

    # --- Tier 3: things that would COST grade if 'fixed' naively -----------

    contractions = feats.get("contraction_per_1k", 0.0) or 0.0
    if contractions > MAX_ACADEMIC_CONTRACTIONS_PER_1K:
        out.append(
            Finding(
                code="contractions_in_academic",
                severity="medium",
                message=f"Contractions run {contractions:.1f} per 1,000 words.",
                detail=(
                    "Research articles sit under 1.4. Contractions help evade "
                    "detection but cost grade under genre conventions, so they "
                    "are a 'never in body' edit for academic targets."
                ),
                grade_cost="medium if added",
            )
        )

    opinion = feats.get("opinion_marker_count", 0.0) or 0.0
    if opinion > 0:
        out.append(
            Finding(
                code="opinion_markers",
                severity="high",
                message=f"Found {int(opinion)} bare opinion marker(s) such as 'I think'.",
                detail=(
                    "First person for argumentative acts ('I argue') is fine "
                    "and endorsed by style guides. Bare opinion reads as an "
                    "unsupported position and is a 'never' edit."
                ),
                grade_cost="high",
            )
        )

    nominalization = feats.get("nominalization_per_1k")
    if (
        nominalization is not None
        and math.isfinite(nominalization)
        and nominalization < 40.0
    ):
        out.append(
            Finding(
                code="low_nominalization",
                severity="low",
                message=f"Nominalization is {nominalization:.1f} per 1,000 words.",
                detail=(
                    "Academic sub-registers run 61-72. Better essays are MORE "
                    "nominalized, not less, so do not reduce this to sound "
                    "human."
                ),
                grade_cost="high if reduced further",
            )
        )

    coord = feats.get("sent_initial_coordinator_share", 0.0) or 0.0
    if coord > 0.05:
        out.append(
            Finding(
                code="sentence_initial_coordinators",
                severity="low",
                message=f"{coord:.0%} of sentences open with And/But/So.",
                detail=(
                    "This falls from 5.1% to 2.3% between a weak and a strong "
                    "essay, so heavy use reads as a lower grade band."
                ),
                grade_cost="low to medium",
            )
        )

    out.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    return out


def _fmt(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.2f}"


def _band_suffix(band: Optional[str]) -> str:
    if not band or band == "unknown":
        return ""
    return f"   [{band}]"


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
