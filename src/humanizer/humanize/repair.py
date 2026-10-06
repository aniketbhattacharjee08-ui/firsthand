"""The self-correcting repair stage: diagnose why a paragraph failed, change
what is asked for, retry, and keep a log a person can read.

Why this exists
---------------
The normal rounds are best-of-N draws from one distribution. research/24
§6.14 measured what that leaves on the table: the supply of candidates that
are both faithful (through the gates) and human (to GPTZero) is about one in
seven, more rounds from the same distribution do not lift the pass rate, and
the failures cluster by cause: `content_drift`, `invariant:numbers` and
`length_ratio` account for most rejections, and a few paragraphs never
produce a passer at all. Those are three different problems with three
different fixes, and a pipeline that answers all of them with "sample again"
is not learning anything from its own rejections.

The owner's instruction, verbatim: "if the humanizer fails, it tries to
realize what it did wrong and edit it again. It's okay if that takes more
time. Specific case-by-case model this time."

So this stage is case by case. For every paragraph still failing after the
last round it:

1. **Diagnoses** from evidence already in hand (no model call): how many of
   its candidates each gate rejected, whether any passed and what the judge
   made of them, which particulars the rejected ones invented, the
   research/24 structure flags of the best one, and whether the author
   supplied facts. The result is a category and a plain sentence, e.g.
   "6 of 8 candidates invented figures not in your notes; the 2 faithful
   ones read as AI".
2. **Chooses one action** from the ladder for that category (`LADDER`),
   never the same action twice in a row for one paragraph.
3. **Retries** with that action, using the same generate, scrub, score and
   gate helpers the main loop uses, and **accepts** the new best only if it
   passes every gate and beats the paragraph's current best on the judge.
   The factual invariants are never relaxed by any action.
4. **Records** the attempt on the paragraph (`ParagraphOutcome.repairs`) as
   `{"attempt", "diagnosis", "action", "result": {"passed", "best",
   "accepted"}}`, so the page can show what it tried and a bench can count
   what each action rescued.

The ladder
----------
=========================  ===================================================
diagnosis                  actions, in order
=========================  ===================================================
invented_specifics         restrict_particulars, raise_fidelity, split_paragraph
content_drift              raise_fidelity, restrict_particulars, split_paragraph
length                     match_length, split_paragraph, raise_fidelity
ai_passers                 change_distribution, sentence_repair, split_paragraph
nothing_passes             split_paragraph, bridge_draft, raise_fidelity
=========================  ===================================================

* `restrict_particulars`: a NOTES line telling the model to use only the
  listed particulars (or, with no facts, no figures beyond the draft's own);
  and a deterministic rescue: a candidate that failed *only* on an invented
  number has the sentence carrying it dropped and is re-gated.
* `raise_fidelity`: NOTES "keep every claim of the draft in order; do not
  add new points", the two shortest exemplar triples, temperatures 0.6-0.85.
* `match_length`: a larger token cap and NOTES "match the draft's length".
* `split_paragraph`: the paragraph in two halves, each humanized and gated
  against its own half, the best halves rejoined and gated as a whole.
* `change_distribution`: the retry adapter when configured, temperatures
  0.9-1.2, the exemplar order reversed, plus two `register` candidates when
  facts are present.
* `sentence_repair`: the sentences of the best passer that the judge scores
  highest are rewritten one at a time through the `guided` module's
  sentence prompts and gate, and the paragraph is re-gated whole.
* `bridge_draft`: the instruct checkpoint writes a faithful bridge draft,
  the base model rewrites *that*, and the result is gated against the
  original source.

An optional critic (`PipelineConfig.repair_critic`) asks the instruct
checkpoint one question per attempt, with the draft, the best failed
candidate and the gate reasons: what did this rewrite do wrong and what one
instruction would fix it. The one-line answer goes into the next attempt's
NOTES and into the record. Off by default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from ..text import split_sentences, words
from . import llm as llm_mod
from . import pipeline as pipe
from .exemplars import FREEFORM_SHOTS

__all__ = [
    "CATEGORIES",
    "ACTIONS",
    "LADDER",
    "Diagnosis",
    "diagnose",
    "choose_action",
    "run_repairs",
]

# --------------------------------------------------------------- vocabulary

CAT_INVENTED = "invented_specifics"
CAT_DRIFT = "content_drift"
CAT_LENGTH = "length"
CAT_AI_PASSERS = "ai_passers"
CAT_NOTHING = "nothing_passes"
CATEGORIES: Tuple[str, ...] = (CAT_INVENTED, CAT_DRIFT, CAT_LENGTH, CAT_AI_PASSERS, CAT_NOTHING)

ACT_RESTRICT = "restrict_particulars"
ACT_FIDELITY = "raise_fidelity"
ACT_LENGTH = "match_length"
ACT_SPLIT = "split_paragraph"
ACT_DISTRIBUTION = "change_distribution"
ACT_SENTENCE = "sentence_repair"
ACT_BRIDGE = "bridge_draft"
ACTIONS: Tuple[str, ...] = (
    ACT_RESTRICT,
    ACT_FIDELITY,
    ACT_LENGTH,
    ACT_SPLIT,
    ACT_DISTRIBUTION,
    ACT_SENTENCE,
    ACT_BRIDGE,
)

#: Category -> ordered actions. See the module docstring.
LADDER: Dict[str, Tuple[str, ...]] = {
    CAT_INVENTED: (ACT_RESTRICT, ACT_FIDELITY, ACT_SPLIT),
    CAT_DRIFT: (ACT_FIDELITY, ACT_RESTRICT, ACT_SPLIT),
    CAT_LENGTH: (ACT_LENGTH, ACT_SPLIT, ACT_FIDELITY),
    CAT_AI_PASSERS: (ACT_DISTRIBUTION, ACT_SENTENCE, ACT_SPLIT),
    CAT_NOTHING: (ACT_SPLIT, ACT_BRIDGE, ACT_FIDELITY),
}

#: Short labels for the UI and the log.
ACTION_LABELS: Dict[str, str] = {
    ACT_RESTRICT: "told the model to use only the listed particulars",
    ACT_FIDELITY: "asked for every claim in order, shorter exemplars, cooler sampling",
    ACT_LENGTH: "asked it to match the draft's length with more room to write",
    ACT_SPLIT: "split the paragraph in two, rewrote each half, rejoined",
    ACT_DISTRIBUTION: "changed the distribution: adapter, hotter sampling, exemplars reordered",
    ACT_SENTENCE: "rewrote the sentences the judge flags highest, one at a time",
    ACT_BRIDGE: "wrote a faithful bridge draft and had the base model rewrite that",
}

_INVARIANT_GATES = (
    "invariant:numbers",
    "invariant:dates",
    "invariant:quotations",
    "invariant:citations",
    "invariant:entities",
)
_LENGTH_GATES = ("too_short", "length_ratio")
_STRUCTURE_FLAGS = (
    "repeated_openers",
    "uniform_beats",
    "paired_coordination_template",
    "unhedged_absolutes",
    "over_hedged",
    "surface_statements",
)

#: The NOTES lines, one per action that adds one.
NOTE_FIDELITY = "Keep every claim of the draft in order; do not add new points."
NOTE_LENGTH = "Match the draft's length: about {n} words."
NOTE_NO_FIGURES = "Use no figures, dates or names beyond the draft's own."
NOTE_PARTICULARS = "Use only these particulars and nothing else that is new: {facts}"


# ---------------------------------------------------------------- diagnosis


@dataclass
class Diagnosis:
    """What the evidence says about one failing paragraph."""

    category: str
    text: str
    n_candidates: int = 0
    n_passed: int = 0
    n_invariant: int = 0
    n_drift: int = 0
    n_length: int = 0
    best_passer_p: Optional[float] = None
    specifics: List[str] = field(default_factory=list)
    structure_flags: List[str] = field(default_factory=list)
    facts_supplied: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "text": self.text,
            "n_candidates": self.n_candidates,
            "n_passed": self.n_passed,
            "n_invariant": self.n_invariant,
            "n_drift": self.n_drift,
            "n_length": self.n_length,
            "best_passer_p": self.best_passer_p,
            "specifics": list(self.specifics),
            "structure_flags": list(self.structure_flags),
        }


def diagnose(outcome: pipe.ParagraphOutcome, cfg: pipe.PipelineConfig) -> Diagnosis:
    """Read the paragraph's candidates and say, in one sentence, what went wrong.

    Pure: no model, no detector. The counts are over every candidate the
    paragraph has seen so far (all rounds and earlier repair attempts), so a
    later attempt's diagnosis reflects what the earlier ones learned.
    """
    cands = list(outcome.candidates)
    facts_supplied = bool(cfg.facts and cfg.facts.strip())
    n = len(cands)
    passers = [c for c in cands if c.passed]
    rejected = [c for c in cands if not c.passed]
    n_inv = sum(1 for c in rejected if any(g in _INVARIANT_GATES for g in c.gates))
    n_drift = sum(1 for c in rejected if "content_drift" in c.gates)
    n_len = sum(
        1 for c in rejected if any(g in _LENGTH_GATES for g in c.gates) and "content_drift" not in c.gates
    )
    scored_passers = [c for c in passers if c.ai_probability is not None]
    best_p = min((c.ai_probability for c in scored_passers), default=None)

    specifics: List[str] = []
    for c in sorted(rejected, key=lambda c: c.ai_probability if c.ai_probability is not None else 1.0):
        if not any(g in _INVARIANT_GATES for g in c.gates):
            continue
        for item in pipe.new_specifics(outcome.original, c.scrubbed, cfg.facts):
            if item not in specifics:
                specifics.append(item)
        if len(specifics) >= 6:
            break
    specifics = specifics[:6]

    best_any = min(
        cands,
        key=lambda c: (0 if c.passed else 1, c.ai_probability if c.ai_probability is not None else 1.0),
        default=None,
    )
    flags = [
        f for f in (best_any.quality.get("flags", ()) if best_any is not None else ())
        if f in _STRUCTURE_FLAGS
    ]

    diag = Diagnosis(
        category=CAT_NOTHING,
        text="",
        n_candidates=n,
        n_passed=len(passers),
        n_invariant=n_inv,
        n_drift=n_drift,
        n_length=n_len,
        best_passer_p=best_p,
        specifics=specifics,
        structure_flags=flags,
        facts_supplied=facts_supplied,
    )

    if n == 0:
        diag.text = "no candidate was written for this paragraph (the time budget ran out before it)"
        return diag

    # A cause "dominates" when the rejections are at least half the
    # candidates and it accounts for at least half of those rejections. When
    # most candidates passed, the problem is the passers, not the gates.
    groups = {CAT_INVENTED: n_inv, CAT_DRIFT: n_drift, CAT_LENGTH: n_len}
    dominant = max(groups, key=lambda k: groups[k])
    dominates = (
        bool(rejected)
        and 2 * len(rejected) >= n
        and groups[dominant] >= max(1, (len(rejected) + 1) // 2)
    )

    passers_clause = (
        "; the %s read%s as AI (best p %.2f)"
        % (
            "%d faithful one%s" % (len(passers), "" if len(passers) == 1 else "s"),
            "s" if len(passers) == 1 else "",
            best_p,
        )
        if passers and best_p is not None
        else ("; the %d faithful one%s could not be scored" % (len(passers), "" if len(passers) == 1 else "s") if passers else "; none was faithful")
    )

    if dominates and dominant == CAT_INVENTED:
        diag.category = CAT_INVENTED
        what = "invented figures or names not in your notes" if facts_supplied else "added figures or names the draft does not contain"
        diag.text = "%d of %d candidates %s%s%s" % (
            n_inv,
            n,
            what,
            " (%s)" % ", ".join(specifics) if specifics else "",
            passers_clause,
        )
    elif dominates and dominant == CAT_DRIFT:
        diag.category = CAT_DRIFT
        diag.text = "%d of %d candidates drifted from what the draft says%s" % (n_drift, n, passers_clause)
    elif dominates and dominant == CAT_LENGTH:
        diag.category = CAT_LENGTH
        diag.text = "%d of %d candidates came out too short or too long%s" % (n_len, n, passers_clause)
    elif passers:
        diag.category = CAT_AI_PASSERS
        diag.text = "%d of %d candidates kept the meaning but every one reads as AI%s%s" % (
            len(passers),
            n,
            " (best p %.2f)" % best_p if best_p is not None else "",
            "; the best one has " + ", ".join(f.replace("_", " ") for f in flags) if flags else "",
        )
    else:
        diag.category = CAT_NOTHING
        reasons = sorted({g for c in rejected for g in c.gates})
        diag.text = "no candidate passed the gates in %d tries (%s)" % (n, ", ".join(reasons) or "no gate named")
    return diag


def choose_action(diagnosis: Diagnosis, last_action: Optional[str] = None) -> str:
    """The first action on the category's ladder that was not the last one tried."""
    ladder = LADDER.get(diagnosis.category, LADDER[CAT_NOTHING])
    for action in ladder:
        if action != last_action:
            return action
    for action in ACTIONS:  # pragma: no cover - every ladder has three entries
        if action != last_action:
            return action
    return ladder[0]  # pragma: no cover


# ------------------------------------------------------------------ helpers


def _temps(lo: float, hi: float, n: int) -> List[float]:
    if n <= 1:
        return [round((lo + hi) / 2, 3)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 3) for i in range(n)]


def _shortest_shots(k: int = 2) -> Tuple[Tuple[str, str, str, str], ...]:
    """The `k` shortest exemplar triples, in their original order."""
    ranked = sorted(
        range(len(FREEFORM_SHOTS)),
        key=lambda i: len(FREEFORM_SHOTS[i][1].split()) + len(FREEFORM_SHOTS[i][3].split()),
    )[:k]
    return tuple(FREEFORM_SHOTS[i] for i in sorted(ranked))


def _swapped_shots() -> Tuple[Tuple[str, str, str, str], ...]:
    return tuple(reversed(FREEFORM_SHOTS))


def _one_line(text: str, limit: int = 400) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


_WS_RE = re.compile(r"\s+")


def _chosen_gate_failures(outcome: pipe.ParagraphOutcome) -> List[str]:
    """Gates the shipped candidate failed, when `select` fell back to one.

    Since 2026-09-22 `select` ships the least-bad candidate rather than the
    original when nothing passes every gate; such a paragraph is repaired as
    if it were still failing, and a clean passer replaces it.
    """
    if not outcome.changed or outcome.chosen_index is None:
        return []
    for c in reversed(outcome.candidates):
        if c.index == outcome.chosen_index and c.scrubbed == outcome.chosen:
            return list(c.gates)
    return []


def _is_failing(outcome: pipe.ParagraphOutcome, cfg: pipe.PipelineConfig, det: Any) -> bool:
    """Whether the paragraph still needs repairing."""
    if not outcome.changed:
        # Owner's rule: every paragraph gets a rewrite, so an unchanged one
        # (only junk candidates so far) is always a target.
        return True
    if _chosen_gate_failures(outcome):
        return True
    if outcome.ai_probability_after is None:
        return False
    return outcome.ai_probability_after >= cfg.pass_threshold


def _next_index(outcome: pipe.ParagraphOutcome) -> int:
    return max((c.index for c in outcome.candidates), default=-1) + 1


# ------------------------------------------------------------- the repairer


class _Repairer:
    """Holds the backends and config; one method per action."""

    def __init__(
        self,
        cfg: pipe.PipelineConfig,
        backends: Dict[str, llm_mod.Backend],
        det: Any,
        retry_backend: Optional[llm_mod.Backend],
        caller_backend: Optional[llm_mod.Backend],
        rounds_run: int,
    ) -> None:
        self.cfg = cfg
        self.backends: Dict[str, llm_mod.Backend] = dict(backends)
        self.det = det
        self.retry_backend = retry_backend
        self.caller_backend = caller_backend
        self.rounds_run = rounds_run
        self.generation_seconds = 0.0
        self.generation_tokens = 0
        styles = cfg.styles()
        self.primary_style = "freeform" if "freeform" in styles else styles[0]
        #: Set by an action that could not run, read into the record.
        self.note: Optional[str] = None

    # -- backends -----------------------------------------------------------

    def backend(self, style: str) -> llm_mod.Backend:
        """The backend for a style, built and loaded on first use.

        A caller-supplied backend (tests, the echo backend) serves every
        style, exactly as it does in the main loop.
        """
        if self.caller_backend is not None:
            return self.caller_backend
        b = self.backends.get(style)
        if b is None:
            b = llm_mod.backend_for(style, self.cfg.model_for(style))
            self.backends[style] = b
        b.load()
        return b

    # -- generation, mirroring the main loop ---------------------------------

    def generate(
        self,
        outcome: pipe.ParagraphOutcome,
        draft: str,
        style: str,
        temperatures: Sequence[float],
        attempt: int,
        persona: str,
        constraint: str,
        extra_notes: str = "",
        shots: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        max_tokens: Optional[int] = None,
        backend: Optional[llm_mod.Backend] = None,
        gate_against: Optional[str] = None,
    ) -> List[pipe.Candidate]:
        """Generate, scrub, score and gate one batch. Same order as `stream()`."""
        if not temperatures:
            return []
        base_specs = llm_mod.specs_for(len(temperatures), style)
        start = _next_index(outcome)
        specs = [
            llm_mod.CandidateSpec(
                index=start + i,
                persona=persona,
                constraint=constraint,
                temperature=float(t),
                style=style,
            )
            for i, (t, _b) in enumerate(zip(temperatures, base_specs))
        ]
        # The base specs carry the register/faithful persona text the prompt
        # needs; the repair persona is what the record shows.
        prompt_specs = [
            llm_mod.CandidateSpec(
                index=s.index, persona=b.persona, constraint=b.constraint, temperature=s.temperature, style=style
            )
            for s, b in zip(specs, base_specs)
        ]
        requests = [
            llm_mod.GenerationRequest(
                prompt=llm_mod.build_prompt(draft, ps, facts=self.cfg.facts, extra_notes=extra_notes, shots=shots),
                temperature=ps.temperature,
                max_tokens=max_tokens or llm_mod.estimate_max_tokens(draft),
                tag=(outcome.index, ps.index),
            )
            for ps in prompt_specs
        ]
        gen = backend or self.backend(style)
        batch = gen.generate(requests)
        self.generation_seconds += batch.seconds
        self.generation_tokens += batch.generation_tokens
        stops = llm_mod.STOPS_FOR_STYLE.get(style, ())
        cands = [
            pipe.Candidate(
                paragraph_index=outcome.index,
                index=s.index,
                persona=s.persona,
                constraint=s.constraint,
                temperature=s.temperature,
                round=self.rounds_run + attempt,
                raw=llm_mod.clean_completion(raw, stops),
                style=style,
            )
            for s, raw in zip(specs, batch.texts)
        ]
        self.finish(gate_against or outcome.original, cands)
        return cands

    def finish(self, source: str, cands: Sequence[pipe.Candidate]) -> None:
        """Scrub, score, gate, measure. The main loop's verify stage, inline."""
        for c in cands:
            c.scrubbed, c.scrub_edits = pipe._scrub(c.raw, self.cfg.aggressiveness)
            c.ai_probability = pipe._score(c.scrubbed, self.det)
            c.gates = pipe._gate_candidate(source, c.scrubbed, self.cfg)
            c.quality = pipe._quality(source, c.scrubbed) if c.scrubbed else {}

    def derived(
        self,
        outcome: pipe.ParagraphOutcome,
        text: str,
        attempt: int,
        persona: str,
        constraint: str,
        style: Optional[str] = None,
    ) -> pipe.Candidate:
        """A candidate assembled from pieces rather than generated whole."""
        c = pipe.Candidate(
            paragraph_index=outcome.index,
            index=_next_index(outcome),
            persona=persona,
            constraint=constraint,
            temperature=0.0,
            round=self.rounds_run + attempt,
            raw=text,
            style=style or self.primary_style,
        )
        self.finish(outcome.original, [c])
        return c

    # -- the actions --------------------------------------------------------

    def apply(
        self,
        outcome: pipe.ParagraphOutcome,
        action: str,
        diagnosis: Diagnosis,
        attempt: int,
        extra_notes: str = "",
    ) -> List[pipe.Candidate]:
        self.note = None
        handler: Callable[..., List[pipe.Candidate]] = {
            ACT_RESTRICT: self.restrict_particulars,
            ACT_FIDELITY: self.raise_fidelity,
            ACT_LENGTH: self.match_length,
            ACT_SPLIT: self.split_paragraph,
            ACT_DISTRIBUTION: self.change_distribution,
            ACT_SENTENCE: self.sentence_repair,
            ACT_BRIDGE: self.bridge_draft,
        }[action]
        return handler(outcome, diagnosis, attempt, extra_notes)

    def _notes(self, *lines: str) -> str:
        return "\n- ".join(l.strip() for l in lines if l and l.strip())

    def restrict_particulars(
        self, outcome: pipe.ParagraphOutcome, diagnosis: Diagnosis, attempt: int, extra: str
    ) -> List[pipe.Candidate]:
        cfg = self.cfg
        line = (
            NOTE_PARTICULARS.format(facts=_one_line(cfg.facts))
            if cfg.facts and cfg.facts.strip()
            else NOTE_NO_FIGURES
        )
        n = cfg.n_candidates
        temps = [s.temperature for s in llm_mod.specs_for(n, self.primary_style)]
        fresh = self.generate(
            outcome,
            outcome.original,
            self.primary_style,
            temps,
            attempt,
            persona="the same writer, told which particulars are allowed",
            constraint=line,
            extra_notes=self._notes(line, extra),
        )
        # The deterministic rescue: a candidate that failed only on an
        # invented number loses the sentence carrying it and is re-gated.
        rescued: List[pipe.Candidate] = []
        for c in list(outcome.candidates) + fresh:
            text = _drop_invented_sentences(outcome.original, c, cfg)
            if text is None:
                continue
            rescued.append(
                self.derived(
                    outcome,
                    text,
                    attempt,
                    persona="candidate %d with the sentence carrying an invented figure removed" % (c.index + 1),
                    constraint="drop the invented sentence",
                    style=c.style,
                )
            )
        return fresh + rescued

    def raise_fidelity(
        self, outcome: pipe.ParagraphOutcome, diagnosis: Diagnosis, attempt: int, extra: str
    ) -> List[pipe.Candidate]:
        n = self.cfg.n_candidates
        return self.generate(
            outcome,
            outcome.original,
            self.primary_style,
            _temps(0.6, 0.85, n),
            attempt,
            persona="the same writer, held to the draft's claims",
            constraint=NOTE_FIDELITY,
            extra_notes=self._notes(NOTE_FIDELITY, extra),
            shots=_shortest_shots(2),
        )

    def match_length(
        self, outcome: pipe.ParagraphOutcome, diagnosis: Diagnosis, attempt: int, extra: str
    ) -> List[pipe.Candidate]:
        n = self.cfg.n_candidates
        n_words = len(words(outcome.original))
        line = NOTE_LENGTH.format(n=n_words)
        temps = [s.temperature for s in llm_mod.specs_for(n, self.primary_style)]
        return self.generate(
            outcome,
            outcome.original,
            self.primary_style,
            temps,
            attempt,
            persona="the same writer, given the draft's length and more room",
            constraint=line,
            extra_notes=self._notes(line, extra),
            max_tokens=llm_mod.estimate_max_tokens(outcome.original, headroom=2.8),
        )

    def split_paragraph(
        self, outcome: pipe.ParagraphOutcome, diagnosis: Diagnosis, attempt: int, extra: str
    ) -> List[pipe.Candidate]:
        sentences = split_sentences(outcome.original)
        if len(sentences) < 2:
            self.note = "the paragraph is one sentence; it cannot be split"
            return []
        cut = (len(sentences) + 1) // 2
        halves = [" ".join(sentences[:cut]).strip(), " ".join(sentences[cut:]).strip()]
        k = max(2, self.cfg.n_candidates // 2)
        temps = [s.temperature for s in llm_mod.specs_for(k, self.primary_style)]
        chosen: List[str] = []
        n_half_passers = 0
        for half in halves:
            cands = self.generate(
                outcome,
                half,
                self.primary_style,
                temps,
                attempt,
                persona="the same writer, on half the paragraph",
                constraint="half of the paragraph, rewritten on its own",
                extra_notes=extra,
                gate_against=half,
            )
            passers = [c for c in cands if c.passed]
            n_half_passers += len(passers)
            if passers:
                scored = [c for c in passers if c.ai_probability is not None]
                best = min(scored, key=lambda c: c.ai_probability) if scored else passers[0]
                chosen.append(best.scrubbed)
            else:
                chosen.append(half)
        if all(c == h for c, h in zip(chosen, halves)):
            self.note = "neither half produced a passer"
            return []
        joined = " ".join(chosen)
        return [
            self.derived(
                outcome,
                joined,
                attempt,
                persona="two halves rewritten separately and rejoined (%d half-passers)" % n_half_passers,
                constraint="split the paragraph",
            )
        ]

    def change_distribution(
        self, outcome: pipe.ParagraphOutcome, diagnosis: Diagnosis, attempt: int, extra: str
    ) -> List[pipe.Candidate]:
        cfg = self.cfg
        n = cfg.n_candidates
        use_register = bool(cfg.facts and cfg.facts.strip()) and self.primary_style != "register"
        n_register = 2 if use_register else 0
        n_base = max(1, n - n_register)
        backend = None
        if self.caller_backend is None and self.retry_backend is not None and self.primary_style == "freeform":
            self.retry_backend.load()
            backend = self.retry_backend
        out = self.generate(
            outcome,
            outcome.original,
            self.primary_style,
            _temps(0.9, 1.2, n_base),
            attempt,
            persona="the same writer from a different distribution",
            constraint="adapter %s, temperatures 0.9-1.2, exemplars reversed" % ("on" if backend is not None else "off"),
            extra_notes=extra,
            shots=_swapped_shots(),
            backend=backend,
        )
        if n_register:
            try:
                out += self.generate(
                    outcome,
                    outcome.original,
                    "register",
                    [0.9, 1.0],
                    attempt,
                    persona="a well-read specialist rewriting a generic draft",
                    constraint="the trained register, with the author's facts",
                    extra_notes=extra,
                )
            except RuntimeError as exc:
                self.note = "register checkpoint unavailable: %s" % exc
        return out

    def sentence_repair(
        self, outcome: pipe.ParagraphOutcome, diagnosis: Diagnosis, attempt: int, extra: str
    ) -> List[pipe.Candidate]:
        from . import guided

        passers = [c for c in outcome.candidates if c.passed and c.scrubbed.strip()]
        if not passers:
            self.note = "no passer to repair sentence by sentence"
            return []
        scored = [c for c in passers if c.ai_probability is not None]
        best = min(scored, key=lambda c: c.ai_probability) if scored else passers[0]
        sentences = split_sentences(best.scrubbed)
        if len(sentences) < 2:
            self.note = "the best passer is one sentence"
            return []
        # The judge's reading of each sentence on its own; the two highest
        # are the ones to rewrite. Without a judge, the two longest.
        if self.det is not None:
            ranked = sorted(
                range(len(sentences)),
                key=lambda i: -(pipe._score(sentences[i], self.det) or 0.0),
            )
        else:
            ranked = sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))
        targets = sorted(ranked[:2])
        backend = self.backend("faithful")
        working = list(sentences)
        replaced = 0
        for i in targets:
            prefix = " ".join(working[:i])
            following = working[i + 1] if i + 1 < len(working) else ""
            moves = guided.sentence_moves(4)
            requests = [
                llm_mod.GenerationRequest(
                    prompt=guided.build_sentence_prompt(prefix, working[i], move, following),
                    temperature=temp,
                    max_tokens=guided.sentence_max_tokens(working[i]),
                    tag=(outcome.index, "sentence", i),
                )
                for move, temp in moves
            ]
            batch = backend.generate(requests)
            self.generation_seconds += batch.seconds
            self.generation_tokens += batch.generation_tokens
            options: List[Tuple[float, str]] = []
            for raw in batch.texts:
                sentence = guided.clean_sentence(raw)
                if not sentence or guided.sentence_gate(working[i], sentence):
                    continue
                context = " ".join(working[:i] + [sentence] + working[i + 1 :])
                p = pipe._score(context, self.det)
                options.append((p if p is not None else 0.5, sentence))
            if options:
                options.sort(key=lambda o: o[0])
                working[i] = options[0][1]
                replaced += 1
        if not replaced:
            self.note = "no sentence rewrite passed the sentence gate"
            return []
        return [
            self.derived(
                outcome,
                " ".join(working),
                attempt,
                persona="candidate %d with %d sentence%s rewritten" % (best.index + 1, replaced, "" if replaced == 1 else "s"),
                constraint="sentence-level repair through the guided prompts",
                style=best.style,
            )
        ]

    def bridge_draft(
        self, outcome: pipe.ParagraphOutcome, diagnosis: Diagnosis, attempt: int, extra: str
    ) -> List[pipe.Candidate]:
        cfg = self.cfg
        # Step 1: faithful bridge drafts from the instruct checkpoint.
        bridges = self.generate(
            outcome,
            outcome.original,
            "faithful",
            [0.7, 0.85],
            attempt,
            persona="the instruct checkpoint writing a faithful bridge draft",
            constraint="bridge draft",
            extra_notes=self._notes(NOTE_FIDELITY, extra),
        )
        usable = [b for b in bridges if not any(g.startswith("invariant:") for g in b.gates) and b.scrubbed.strip()]
        if not usable:
            self.note = "no bridge draft kept every fact"
            return bridges
        # Step 2: the base model rewrites the bridge; the result is gated
        # against the original, never the bridge.
        k = max(2, cfg.n_candidates // 2)
        temps = [s.temperature for s in llm_mod.specs_for(k, "freeform")]
        out: List[pipe.Candidate] = list(bridges)
        for bridge in usable[:2]:
            out += self.generate(
                outcome,
                bridge.scrubbed,
                "freeform" if self.primary_style != "register" else self.primary_style,
                temps,
                attempt,
                persona="the base model rewriting bridge draft %d" % (bridge.index + 1),
                constraint="rewrite of a faithful bridge draft",
                extra_notes=extra,
                gate_against=outcome.original,
            )
        return out

    # -- the critic ---------------------------------------------------------

    def critic(self, outcome: pipe.ParagraphOutcome) -> Optional[str]:
        """One question to the instruct checkpoint. Returns one line or None."""
        cands = [c for c in outcome.candidates if c.scrubbed.strip()]
        if not cands:
            return None
        best = min(
            cands,
            key=lambda c: (0 if c.passed else 1, c.ai_probability if c.ai_probability is not None else 1.0),
        )
        reasons = ", ".join(best.gates) if best.gates else "passed every gate but reads as AI to the detector"
        user = (
            "A draft paragraph was rewritten to read as human writing while keeping "
            "every fact. The rewrite failed. Say in one line what the rewrite did "
            "wrong and what one instruction would fix it. No preamble.\n\n"
            "DRAFT:\n%s\n\nREWRITE:\n%s\n\nWHY IT FAILED:\n%s\n\nONE LINE:"
            % (outcome.original.strip(), best.scrubbed.strip(), reasons)
        )
        prompt = llm_mod.Prompt(
            kind="chat",
            messages=(
                ("system", "You are a blunt writing editor. You answer in one sentence."),
                ("user", user),
            ),
        )
        backend = self.backend("faithful")
        batch = backend.generate(
            [llm_mod.GenerationRequest(prompt=prompt, temperature=0.3, max_tokens=80, tag=(outcome.index, "critic"))]
        )
        self.generation_seconds += batch.seconds
        self.generation_tokens += batch.generation_tokens
        text = llm_mod.clean_completion(batch.texts[0] if batch.texts else "")
        for line in text.split("\n"):
            if line.strip():
                return _one_line(line, 240)
        return None


def _drop_invented_sentences(
    source: str, cand: pipe.Candidate, cfg: pipe.PipelineConfig
) -> Optional[str]:
    """The text of `cand` with the sentences carrying invented numbers removed,
    if that alone makes it pass every gate. None otherwise.

    Only for candidates whose sole failure is `invariant:numbers`; the gate
    re-run is what guarantees no source claim was lost with the sentence
    (content overlap and every invariant must still hold).
    """
    if list(cand.gates) != ["invariant:numbers"] or not cand.scrubbed.strip():
        return None
    extra = [
        t for t in pipe.new_specifics(source, cand.scrubbed, cfg.facts) if pipe._NUMBER_RE.fullmatch(t)
    ]
    if not extra:
        return None
    sentences = split_sentences(cand.scrubbed)
    kept = [s for s in sentences if not any(t in s for t in extra)]
    if not kept or len(kept) == len(sentences):
        return None
    text = " ".join(kept).strip()
    if pipe._gate_candidate(source, text, cfg):
        return None
    return text


# ---------------------------------------------------------------- acceptance


def _accept(
    outcome: pipe.ParagraphOutcome,
    cands: Sequence[pipe.Candidate],
    cfg: pipe.PipelineConfig,
    det: Any,
) -> Tuple[int, Optional[float], bool]:
    """Take the best gated candidate if it beats the paragraph's current best.

    Returns `(n_passed, best_probability, accepted)`. "Current best" is the
    chosen rewrite's score when the paragraph was changed, otherwise the
    original's; with no judge at all, only an unchanged paragraph is filled.
    """
    passers = [c for c in cands if c.passed]
    if not passers:
        return 0, None, False
    scored = [c for c in passers if c.ai_probability is not None]
    if scored:
        best = min(scored, key=lambda c: pipe._rank_key(c, cfg))
    else:
        best = min(passers, key=lambda c: (pipe._cv_distance(c), c.index))
    current = outcome.ai_probability_after if outcome.changed else outcome.ai_probability_before
    if _chosen_gate_failures(outcome):
        # A clean passer always replaces a candidate that failed a gate.
        accepted = True
    elif best.ai_probability is None:
        accepted = not outcome.changed and det is None
    else:
        accepted = current is None or best.ai_probability < current
    if accepted:
        outcome.chosen = best.scrubbed
        outcome.chosen_index = best.index
        outcome.ai_probability_after = best.ai_probability
        outcome.fallback_reason = None
    return len(passers), best.ai_probability, accepted


# ---------------------------------------------------------------- the stage


def run_repairs(
    outcomes: Sequence[pipe.ParagraphOutcome],
    cfg: pipe.PipelineConfig,
    backends: Dict[str, llm_mod.Backend],
    det: Any,
    clock: Callable[[], float],
    event: Callable[..., pipe.ProgressEvent],
    retry_backend: Optional[llm_mod.Backend] = None,
    caller_backend: Optional[llm_mod.Backend] = None,
    rounds_run: int = 1,
) -> Iterator[pipe.ProgressEvent]:
    """The `repair` stage. A generator of progress events; its return value
    (the `yield from` expression in `pipeline.stream`) is the summary block:
    `{"attempted", "accepted", "rescued", "log", "generation_seconds",
    "generation_tokens"}`.

    Attempts are round-robin over the failing paragraphs (every paragraph
    gets its first attempt before any gets its second), so the time budget
    is shared rather than spent on the first stubborn one.
    """
    summary: Dict[str, Any] = {
        "attempted": 0,
        "accepted": 0,
        "rescued": 0,
        "log": [],
        "generation_seconds": 0.0,
        "generation_tokens": 0,
    }
    if cfg.repair_attempts <= 0:
        yield event("repair", "skip", 1.0, "repair stage disabled (repair_attempts = 0)")
        return summary
    failing = [o for o in outcomes if _is_failing(o, cfg, det)]
    if not failing:
        yield event("repair", "skip", 1.0, "every paragraph passed; nothing to repair")
        return summary

    yield event(
        "repair",
        "start",
        0.0,
        "%s still failing; up to %d attempt%s each, %.0fs budget"
        % (
            pipe._plural(len(failing), "paragraph"),
            cfg.repair_attempts,
            "" if cfg.repair_attempts == 1 else "s",
            cfg.repair_time_budget_s,
        ),
    )
    t0 = clock()
    repairer = _Repairer(cfg, backends, det, retry_backend, caller_backend, rounds_run)
    total_slots = len(failing) * cfg.repair_attempts
    slot = 0
    stopped_for_budget = False
    for attempt in range(1, cfg.repair_attempts + 1):
        for n, outcome in enumerate(failing):
            slot += 1
            if not _is_failing(outcome, cfg, det):
                continue
            if clock() - t0 > cfg.repair_time_budget_s:
                stopped_for_budget = True
                break
            diagnosis = diagnose(outcome, cfg)
            last_action = outcome.repairs[-1]["action"] if outcome.repairs else None
            action = choose_action(diagnosis, last_action)
            carried = ""
            if outcome.repairs and outcome.repairs[-1].get("critic"):
                carried = str(outcome.repairs[-1]["critic"])
            yield event(
                "repair",
                "progress",
                (slot - 1) / max(1, total_slots),
                "paragraph %d of %d, attempt %d of %d: %s. Trying: %s"
                % (n + 1, len(failing), attempt, cfg.repair_attempts, diagnosis.text, ACTION_LABELS.get(action, action)),
            )
            record: Dict[str, Any] = {
                "attempt": attempt,
                "diagnosis": diagnosis.text,
                "category": diagnosis.category,
                "action": action,
                "action_label": ACTION_LABELS.get(action, action),
            }
            if carried:
                record["notes_from_critic"] = carried
            try:
                fresh = repairer.apply(outcome, action, diagnosis, attempt, carried)
            except RuntimeError as exc:
                fresh = []
                record["error"] = str(exc)
            outcome.candidates.extend(fresh)
            passed, best_p, accepted = _accept(outcome, fresh, cfg, det)
            record["result"] = {
                "passed": passed,
                "best": None if best_p is None else round(best_p, 4),
                "accepted": accepted,
                "n_candidates": len(fresh),
            }
            if repairer.note:
                record["note"] = repairer.note
            still_failing = _is_failing(outcome, cfg, det)
            if cfg.repair_critic and still_failing and attempt < cfg.repair_attempts:
                try:
                    critic_line = repairer.critic(outcome)
                except RuntimeError as exc:
                    critic_line = None
                    record["critic_error"] = str(exc)
                if critic_line:
                    record["critic"] = critic_line
            outcome.repairs.append(record)
            summary["log"].append(dict(record, paragraph=outcome.index))
            summary["attempted"] += 1
            if accepted:
                summary["accepted"] += 1
            yield event(
                "repair",
                "progress",
                slot / max(1, total_slots),
                "paragraph %d of %d, attempt %d: %d passed, best %s, %s"
                % (
                    n + 1,
                    len(failing),
                    attempt,
                    passed,
                    "unscored" if best_p is None else "p(AI) %.3f" % best_p,
                    ("accepted" + ("" if still_failing else ", now passing")) if accepted else "not accepted",
                ),
            )
        if stopped_for_budget:
            break

    summary["rescued"] = sum(1 for o in failing if not _is_failing(o, cfg, det))
    summary["generation_seconds"] = round(repairer.generation_seconds, 3)
    summary["generation_tokens"] = repairer.generation_tokens
    yield event(
        "repair",
        "done",
        1.0,
        "rescued %d of %s after %d attempt%s%s"
        % (
            summary["rescued"],
            pipe._plural(len(failing), "failing paragraph"),
            summary["attempted"],
            "" if summary["attempted"] == 1 else "s",
            "; stopped by the %.0fs budget" % cfg.repair_time_budget_s if stopped_for_budget else "",
        ),
    )
    return summary
