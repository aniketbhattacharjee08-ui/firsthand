"""LLM candidate generation for the humanizing pipeline, on MLX.

What this module is
-------------------
The rule-based engine in `engine.py` edits a document in place: it deletes
discourse glue, swaps hand-checked vocabulary and moves sentence boundaries.
That is safe and it is measurable, but it cannot change the *shape* of a
paragraph, so it does not change a detector's verdict. This module is the other
half: it asks a local language model for whole rewritten paragraphs, several at
a time, and hands them to `pipeline.py` to be scrubbed, scored and gated.

Nothing here scores, gates or selects. This module only knows how to turn a
paragraph into prompts and prompts into completions. That separation is what
makes the pipeline testable without a GPU: `pipeline` talks to a `Backend`,
and the tests pass it a fake one.

The measured finding this module is designed around
---------------------------------------------------
**The instruct model reproduces the AI fingerprint in its own voice.** Asked to
humanize, `Qwen2.5-3B-Instruct` returns text that still contains "rapidly
evolving digital landscape", "pivotal", "delve into", "multifaceted", "robust"
and "Consequently" - research/04's finding that the fingerprint comes from
instruction tuning rather than pretraining, reproduced here. Two consequences
are baked into the design:

1. The deterministic scrub (`transforms.py`) runs **after** the model, never
   before. `pipeline.py` enforces the ordering; this module's docstring is
   where the reason lives.
2. The prompts carry an explicit banned-vocabulary list anyway. It reduces the
   rate but does not eliminate it, which is why (1) is not optional.

Diversity, and why it is instructions rather than temperature
-------------------------------------------------------------
research/08: correlated candidates make best-of-N decay as a power law rather
than exponentially, so candidate *correlation* is the binding constraint, not
N. Sampling temperature buys very little decorrelation here - measured on this
machine, six completions of the same prompt at temperatures from 0.7 to 2.0 all
opened with the same seven words as the source and all scored 1.0000 under
desklib. So each candidate gets its own persona *and* its own structural
constraint (`CANDIDATE_SPECS`), and temperature varies underneath that. The
personas are drawn from research/00 §4's "safe operating region": every one of
them asks for an edit rated zero or negative grade cost.

Two model families
------------------
`faithful` (default) uses the **instruct** checkpoint with a style contract. It
preserves meaning, and its candidates pass the pipeline's gates.

`freeform` uses the **base** checkpoint with a few-shot draft/human pair, which
is research/05 §6's recommendation - the fingerprint is in the instruction
tuning, so the base model should not have it. Measured here, that is half true:
the base model does produce candidates desklib scores below 0.5, but it buys
every one of them with semantic drift (content-word overlap with the source of
0.01-0.15), and the pipeline's gates reject them. It is exposed because the
measurement is worth having and because a larger base model may behave
differently; it is not the default, and `pipeline.py` reports the rejections
rather than hiding them.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..memory import max_resident, release_memory

__all__ = [
    "DEFAULT_MODEL",
    "FREEFORM_MODEL",
    "REGISTER_MODEL",
    "FACTS_BLOCK",
    "REGISTER_ADAPTER",
    "REGISTER_SPECS",
    "register_model_path",
    "BANNED_VOCABULARY",
    "STYLE_EXEMPLAR",
    "CandidateSpec",
    "CANDIDATE_SPECS",
    "FREEFORM_SPECS",
    "specs_for",
    "Prompt",
    "build_prompt",
    "GenerationRequest",
    "GenerationBatch",
    "Backend",
    "MlxBackend",
    "EchoBackend",
    "mlx_available",
    "mlx_unavailable_reason",
    "clean_completion",
    "estimate_max_tokens",
    "default_backend",
    "backend_for",
    "loaded_backends",
    "release_backends",
]


#: 4-bit Qwen2.5-7B-Instruct (Apache 2.0), used by the `faithful` style and by
#: the repair stage's bridge draft. Replaced Qwen2.5-3B-Instruct on 2026-09-11
#: because the 3B checkpoints carry the non-commercial Qwen Research licence;
#: same family as the shipped base, ~4GB on disk. Override with
#: HUMANIZER_INSTRUCT_MODEL.
DEFAULT_MODEL = __import__("os").environ.get("HUMANIZER_INSTRUCT_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")

#: The same model without instruction tuning, for `style="freeform"`. This
#: is the only checkpoint on the machine whose output GPTZero has read as
#: human (research/24 §6.2: 6 of 12 raw candidates at 0.000-0.079, against
#: 0 of 24 for the two instruct checkpoints), which is research/00 finding 5
#: (base 96-99% human on GPTZero, instruct 17-30%) reproduced locally.
#: Override with HUMANIZER_BASE_MODEL. The default moved from the 3B to the
#: 7B base on 2026-09-10 (research/26): Qwen2.5-3B is under the Qwen Research
#: licence (non-commercial) and Qwen2.5-7B is Apache 2.0, and with the HIP
#: adapter below the 7B base passed 9 of 9 bench paragraphs on own-key GPTZero
#: against the plain 7B's 6 of 9 the same day and the 3B's 81% over three runs.
FREEFORM_MODEL = __import__("os").environ.get("HUMANIZER_BASE_MODEL", "mlx-community/Qwen2.5-7B-4bit")


#: `style="register"`: Qwen3-4B-Instruct with a LoRA adapter trained on
#: hand-written rewrites in the one register that has flipped the local
#: detector AND the held-out ones (academic, fakespot, RADAR) on the bench:
#: short plain sentences, a real name, year or figure in almost every
#: sentence, a limiting or sceptical judgement, no transition words, no
#: closing summary. The particulars come from `PipelineConfig.facts`: the
#: round-1 adapter invented them and the shipping gates refused every
#: candidate (research/24 §5.2). Round 4 was trained on 59 fidelity-first
#: targets (content overlap >= 0.50 with the draft, every particular traceable
#: to the draft or the prompt's FACTS block, desklib < 0.5) and, benched
#: through the shipped pipeline with author facts (research/24 §5.5): 8/9
#: desklib flips, 8/9 fakespot, 9/9 academic, 8/9 RADAR, 0/5 human paragraphs
#: harmed, mean content overlap 0.59, no invented specifics. Without facts it
#: returns the draft and names what it would have needed.
#: The checkpoint is quantised to 4-bit on first use into data/cache/models.
REGISTER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
REGISTER_MODEL_DIR = str(_REPO_ROOT / "data" / "cache" / "models" / "qwen3-4b-instruct-4bit")
#: The LoRA that ships on the base checkpoint (research/26; trained in
#: `.hip7b/`, terse-notes data, step 200). `HUMANIZER_BASE_ADAPTER` overrides
#: it; set it to an empty string for the plain base. Absent on disk means the
#: plain base, so a checkout without `data/adapters/` still starts.
DEFAULT_BASE_ADAPTER = str(_REPO_ROOT / "data" / "adapters" / "hip7b-r4-it200")
#: Override with HUMANIZER_REGISTER_ADAPTER to bench another round
#: (register-r3, 6/9, is kept alongside for comparison).
REGISTER_ADAPTER = __import__("os").environ.get(
    "HUMANIZER_REGISTER_ADAPTER",
    str(_REPO_ROOT / "data" / "adapters" / "register-r4"),
)


def register_model_path() -> str:
    """Local 4-bit MLX copy of `REGISTER_MODEL`, converted on first call.

    The conversion reads the Hugging Face checkpoint (downloaded to the hub
    cache if absent) and writes about 2.1GB. Raises `RuntimeError` with a
    user-facing message when neither the converted copy nor a conversion is
    possible.
    """
    import os

    if os.path.isfile(os.path.join(REGISTER_MODEL_DIR, "config.json")):
        return REGISTER_MODEL_DIR
    try:
        from mlx_lm import convert  # noqa: WPS433 - deliberately lazy

        os.makedirs(os.path.dirname(REGISTER_MODEL_DIR), exist_ok=True)
        convert(REGISTER_MODEL, mlx_path=REGISTER_MODEL_DIR, quantize=True, q_bits=4)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not prepare {REGISTER_MODEL!r} for mlx-lm ({exc}). The register "
            "style needs the 4-bit copy in data/cache/models; the faithful style "
            "and the rule-based engine at POST /api/humanize work without it."
        ) from exc
    return REGISTER_MODEL_DIR


# --------------------------------------------------------------- the prompts


#: Vocabulary the model is told not to use. This is the same list the
#: deterministic scrub removes (`transforms.replace_ai_vocabulary` and
#: `features.ai_lexicon`), restated in the prompt because it is cheaper to not
#: generate a word than to repair it. It does not work reliably - see the
#: module docstring - which is why the scrub still runs afterwards.
BANNED_VOCABULARY = (
    "delve, tapestry, pivotal, multifaceted, robust, landscape, realm, "
    "navigate, underscore, testament, crucial, vital, comprehensive, "
    "holistic, leverage, foster, seamless, meaningful, proactively, "
    "significant, essential, furthermore, moreover, additionally, "
    "consequently, therefore, ultimately, in conclusion, overall, "
    "it is important to note, in today's world, plays a key role"
)

#: A short passage of the register the model is being asked to write in.
#:
#: Written by hand rather than lifted from the PMC corpus on purpose: a real
#: corpus paragraph drags its own subject matter into the rewrite, and the
#: model copies content from an exemplar more readily than it copies rhythm.
#: This one has the properties research/00 §5 puts in the human band - a
#: five-word sentence next to a forty-word one, a semicolon, a hedge, first
#: person for an argumentative act, and no closing summary. Measured under
#: desklib at 0.011.
STYLE_EXEMPLAR = (
    "Most of the teams I have worked with are still guessing, and the honest "
    "answer is that nobody has this settled. Automation saves money. That much "
    "is not in dispute; the cost line moves in the first quarter you switch "
    "something on, which is usually why finance is the department pushing "
    "hardest. The harder part arrives later, when somebody asks where the "
    "training data came from. Regulation is catching up unevenly. Some of "
    "these questions are ethical and some are only legal exposure, and the two "
    "get muddled in the same meeting."
)

#: The hard rules every `faithful` prompt carries.
#:
#: Each numbered item is an edit that research/00 §4 rates as high detector
#: benefit and zero or negative grade cost. Nothing on the refusal list is
#: here: no contractions in body text, no injected errors, no vocabulary
#: simplification, no "I think", no dropped evidence.
STYLE_CONTRACT = """STYLE CONTRACT - follow every line of it:
1. Keep every fact, number, date, name, quotation and citation exactly as
   written. Add nothing that is not in the draft.
2. Vary sentence length hard. At least one sentence under eight words, at
   least one over thirty.
3. No sentence may open with a connective: Furthermore, Moreover,
   Additionally, However, Therefore, Consequently, In conclusion, Ultimately,
   Overall, In today's.
4. Do not use any of these words or phrases: {banned}.
5. Do not close on a summary, a lesson or a forward-looking sentence. Stop on
   a fact.
6. Break the topic-sentence-then-support shape. The main claim does not have
   to arrive first.
7. Stay within fifteen percent of the draft's length.
8. Output the rewritten paragraph and nothing else. No preamble, no heading,
   no quotation marks around it.
9. Sentence structure, not just words: no two consecutive sentences may begin
   with the same word. Fewer than half the sentences may begin with The,
   This, It or In. Open at least one sentence on a subordinate clause
   (Although..., When..., Because...) and at least one on a concrete noun.
10. Never three sentences in a row of similar length. Follow a long sentence
   with a short one.
11. No lists of three ("X, Y, and Z"), no "not just X but Y".
12. Read the draft for its meaning first. Then say that meaning in your own
   sentences as if you had never seen the wording. Do not swap synonyms into
   the draft's sentences; rebuild them.
13. Hedge every claim the draft states as fact without a number or a
   citation: "can play", "may affect", "appears", "is believed to", "is
   likely to". Never hedge a number, a date or a quoted claim.
14. Where the draft states a claim flat, one sentence may name a limit, a
   condition or a competing reading that the draft already implies. Never
   invent a study, a source or a citation.
15. No two consecutive sentences may share a shape: not the same opener
   kind, not within three words of the same length, not both closing on an
   "X and Y" pair. Merge or rebuild one of them.
16. Delete any clause that adds no proposition: "in various contexts",
   "plays an important role", "it is important to note", "overall
   well-being"."""


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate's instructions. The unit of diversity.

    `persona` sets the voice, `constraint` sets a structural move, and
    `temperature` varies underneath both. Two candidates never share a
    (persona, constraint) pair, which is the whole point: research/08 shows
    correlated candidates are what makes best-of-N stop paying.
    """

    index: int
    persona: str
    constraint: str
    temperature: float
    style: str = "faithful"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "persona": self.persona,
            "constraint": self.constraint,
            "temperature": self.temperature,
            "style": self.style,
        }


def _spec(index: int, persona: str, constraint: str, temperature: float) -> CandidateSpec:
    return CandidateSpec(index, persona, constraint, temperature)


#: The candidate ladder. Ordered so that a truncated run (small N, or a budget
#: cut) still spans the space rather than taking six variations of one idea:
#: index 0 is the most conservative, and the personas alternate between
#: "revise" and "re-say".
CANDIDATE_SPECS: Tuple[CandidateSpec, ...] = (
    # The manual method of research/24 leads the ladder, so that the default
    # six-candidate run (three instruct candidates under `mixed`) always
    # contains it. Three specs, one per family of moves: re-say from memory,
    # hedge and qualify, break the repeated beat.
    _spec(
        0,
        "a supervisor who read the paragraph once and is now saying it back from memory",
        "Close the draft. State its meaning in your own sentences without reusing "
        "any of its clause structures. Keep every fact, number and name.",
        0.80,
    ),
    _spec(
        1,
        "a peer reviewer who hedges what the evidence cannot carry",
        "Turn each absolute claim into a hedged one (can, may, appears, is "
        "believed to) and add one sentence naming a limit or competing reading "
        "the draft already implies.",
        0.85,
    ),
    _spec(
        2,
        "an editor who listens for repeated beats",
        "Give consecutive sentences different shapes: a different opener kind "
        "each time, never two similar lengths in a row, and merge any two "
        "sentences that both close on an 'X and Y' pair.",
        0.90,
    ),
    _spec(
        3,
        "a careful researcher revising your own draft for clarity",
        "Open on the most concrete noun in the draft, not on a general claim.",
        0.70,
    ),
    _spec(
        4,
        "a domain expert writing plainly for a colleague who already knows the field",
        "Put the main claim in the middle of the paragraph, not in the first sentence.",
        0.80,
    ),
    _spec(
        5,
        "an editor cutting filler out of someone else's prose",
        "Delete every transition. Let the sentences sit next to each other unglued.",
        0.90,
    ),
    _spec(
        6,
        "a working journalist on deadline, writing for an informed reader",
        "Split the longest sentence into one long sentence and one very short one.",
        1.00,
    ),
    _spec(
        7,
        "a practitioner writing from experience rather than from a textbook",
        "Replace each abstract noun with the concrete thing it stands for.",
        0.85,
    ),
    _spec(
        8,
        "a graduate student turning lecture notes back into prose",
        "State the consequence first and the cause second.",
        0.95,
    ),
    _spec(
        9,
        "a reviewer rewriting a passage you found badly organised",
        "Drop the draft's opening move entirely and begin on its second idea.",
        0.75,
    ),
    _spec(
        10,
        "an author revising a passage a copy-editor flagged as flat",
        "Let one clause trail into a parenthetical aside; hedge one claim explicitly.",
        1.05,
    ),
    # Sentence-structure moves. research/06 §1 and §6 list the cadence
    # heuristics GPTZero's own material says it flags: runs of similar-length
    # sentences, more than half a paragraph opening on The/This/It/In,
    # tricolons, a metronomic 18-24 word mean. These specs attack each one
    # directly rather than leaving it to the style contract's rule 9-11.
    _spec(
        11,
        "a writer who hears the rhythm of a paragraph before its argument",
        "Begin every sentence with a different kind of word: a noun, then a "
        "subordinate clause, then a verb-ish participle, then a pronoun. Never "
        "open two sentences the same way.",
        0.90,
    ),
    _spec(
        12,
        "an essayist who distrusts symmetry",
        "Make one sentence a single clause of six words or fewer and the next a "
        "three-clause sentence of thirty or more. No sentence may sit between "
        "seventeen and twenty-three words.",
        1.00,
    ),
    _spec(
        13,
        "a historian who front-loads circumstance",
        "Open at least two sentences with a fronted adverbial or subordinate "
        "clause (Because..., After..., Where..., Although...) before the "
        "subject arrives.",
        0.85,
    ),
    _spec(
        14,
        "a plain-spoken lecturer who breaks lists apart",
        "Dissolve every list of three into separate sentences or a pair. Put "
        "the least expected item first.",
        0.95,
    ),
)

#: The `freeform` ladder. Fewer knobs, because a base model does not follow
#: instructions - the only levers that reach it are the few-shot pair and the
#: sampler.
FREEFORM_SPECS: Tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(
        index=i,
        persona="the human writer in the editing log",
        constraint="continue the log",
        temperature=t,
        style="freeform",
    )
    for i, t in enumerate((0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 0.85, 1.15))
)


#: The `register` prompt. These two strings are the exact prompt the adapter
#: was trained with (.flip/anchor/build_data.py); changing a word here moves
#: the model off its training distribution.
REGISTER_SYSTEM = (
    "You are an experienced editor. You turn generic draft paragraphs into the kind of paragraph a well-read specialist "
    "writes: plain, concrete, sceptical where the evidence is thin, with real names, dates and figures, and no filler."
)
REGISTER_USER = (
    "Rewrite this draft. Keep every claim in order. Short plain sentences, a real name, year or figure in almost every "
    "sentence, a limiting or sceptical judgement where warranted, no transition words, no closing summary.\n\nDRAFT:\n{d}"
)

#: Inserted ahead of the draft (the draft stays last, where the adapter saw
#: it in training) when the author supplied `PipelineConfig.facts`. The
#: adapter invents particulars otherwise; the gates refuse invented ones, so
#: this block is the only way a register candidate gets to add a figure.
FACTS_BLOCK = (
    "FACTS FROM THE AUTHOR - the only names, dates, figures and sources you may "
    "add beyond what the draft already contains. Use them where they fit. "
    "Invent nothing else.\n{f}\n\n"
)

#: The `register` ladder: one prompt, the sampler is the only knob, so the
#: candidates differ in temperature only. 0.9 was the evaluation setting.
REGISTER_SPECS: Tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(
        index=i,
        persona="a well-read specialist rewriting a generic draft",
        constraint="the trained register",
        temperature=t,
        style="register",
    )
    for i, t in enumerate((0.90, 0.80, 1.00, 0.85, 0.95, 0.75, 1.05, 0.90))
)


def specs_for(n: int, style: str = "faithful") -> List[CandidateSpec]:
    """The first `n` candidate specs for a style, cycling if `n` exceeds the ladder.

    Cycling bumps the temperature on each repeat so that a caller asking for
    more candidates than there are personas still gets distinguishable ones
    rather than exact duplicates.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if style == "mixed":
        # Interleave: even indices instruct, odd indices base, so a truncated
        # run still holds both families. research/13 §2 measures base-model
        # output at 96-99% human on GPTZero against 17-30% for the instruct
        # version of the same model, and research/08 wants candidate
        # diversity to come from different generators, not one generator's
        # temperature.
        n_faithful = (n + 1) // 2
        n_freeform = n // 2
        faithful = specs_for(n_faithful, "faithful") if n_faithful else []
        freeform = specs_for(n_freeform, "freeform") if n_freeform else []
        out: List[CandidateSpec] = []
        for i in range(n):
            base = faithful[i // 2] if i % 2 == 0 else freeform[i // 2]
            out.append(
                CandidateSpec(
                    index=i,
                    persona=base.persona,
                    constraint=base.constraint,
                    temperature=base.temperature,
                    style=base.style,
                )
            )
        return out
    ladder = {"freeform": FREEFORM_SPECS, "register": REGISTER_SPECS}.get(style, CANDIDATE_SPECS)
    out = []
    for i in range(n):
        base = ladder[i % len(ladder)]
        lap = i // len(ladder)
        out.append(
            CandidateSpec(
                index=i,
                persona=base.persona,
                constraint=base.constraint,
                # Cap at 1.3: past that the model starts emitting non-words,
                # which the gates reject anyway, so it is pure wasted budget.
                temperature=min(1.3, round(base.temperature + 0.12 * lap, 3)),
                style=base.style,
            )
        )
    return out


@dataclass(frozen=True)
class Prompt:
    """A prompt in one of two shapes.

    `kind="chat"` carries `messages` for a tokenizer's chat template;
    `kind="text"` carries a raw `text` continuation prefix for a base model.
    Keeping both in one type means `Backend.generate` takes one list, and a
    fake backend in a test can read either without importing MLX.
    """

    kind: str
    messages: Tuple[Tuple[str, str], ...] = ()
    text: str = ""

    @property
    def message_dicts(self) -> List[Dict[str, str]]:
        return [{"role": r, "content": c} for r, c in self.messages]

    def rendered(self) -> str:
        """A plain-text rendering. Used for logging and for prompt tests."""
        if self.kind == "text":
            return self.text
        return "\n\n".join(f"[{r}]\n{c}" for r, c in self.messages)


#: The system turn. The style exemplar lives here rather than in the user turn
#: for a measured reason: with the exemplar in the user turn ahead of the
#: draft, Qwen2.5-3B-Instruct rewrote *the exemplar* in four of six candidates
#: on the benchmark corpus, producing fluent text with a content-word overlap
#: of 0.00 against the source. Framing material belongs in the system turn; the
#: user turn should contain exactly one thing to act on.
_SYSTEM_FAITHFUL = (
    "You are a human writer revising a draft. You are not an assistant, you "
    "never announce what you are about to do, and you never comment on the "
    "text. You write the way a person writes: uneven sentence lengths, plain "
    "words, no throat-clearing, no tidy sign-off.\n\n"
    "For calibration only, here is a passage in the register you write in. "
    "Copy its rhythm. Never copy its subject matter, and never rewrite it:\n\n"
    + STYLE_EXEMPLAR
    + "\n\nOutput only the revision of the draft the user gives you."
)

#: The `freeform` few-shot, three (draft, notes, rewrite) triples from
#: `exemplars.FREEFORM_SHOTS`. The header frames the task as a pattern the
#: model is continuing, which is the only framing a base model reliably picks
#: up. Measured 2026-09-07 with GPTZero scoring every candidate (research/24
#: §6.3-6.4): one draft/rewrite pair, no notes: 9 of 18 candidates human, 1
#: also through the gates; three pairs with the author's notes in the
#: pattern: 16 of 54 human, 6 of 9 paragraphs with a gated human candidate
#: at best-of-6. The notes slot carries `PipelineConfig.facts`.
_FREEFORM_HEADER = (
    "Each draft below was rewritten by its author in their own words: same "
    "claims, same order, plainer and more exact, using only the particulars "
    "in the author's notes, with the author's own judgement where the "
    "evidence is thin.\n\n"
)
_FREEFORM_NO_NOTES = "(none: use only what the draft already says)"

#: Kept for the faithful system turn's calibration passage and for tests.
_FREEFORM_DRAFT = (
    "In today's rapidly evolving digital landscape, artificial intelligence "
    "has become an increasingly important tool for organizations of all sizes. "
    "It is important to note that the adoption of these technologies requires "
    "careful consideration of both the opportunities and the challenges "
    "involved. Furthermore, the ethical implications of these systems cannot "
    "be overlooked. Ultimately, success in this area depends on a commitment "
    "to transparency, accountability, and continuous improvement."
)

#: Where a base-model completion stops. It has no end-of-turn token, so it will
#: happily invent the next entry.
FREEFORM_STOPS = ("\nDRAFT:", "\nNOTES:", "\nHUMAN:", "\nEditing log")


def _freeform_shots(shots: Optional[Sequence[Tuple[str, str, str, str]]] = None) -> str:
    """The few-shot block. `shots` overrides `exemplars.FREEFORM_SHOTS`.

    The repair stage (`repair.py`) passes a subset or a reordering: the two
    shortest triples when candidates drift, the reversed order when every
    passer reads as AI, so the base model sees a different pattern rather
    than the same one at a different temperature.
    """
    if shots is None:
        from .exemplars import FREEFORM_SHOTS

        shots = FREEFORM_SHOTS
    return "".join(
        f"DRAFT:\n{draft}\n\nNOTES:\n{notes}\n\nHUMAN:\n{rewrite}\n\n"
        for _topic, draft, notes, rewrite in shots
    )


#: Label for the repair stage's instruction inside a chat prompt.
EDITOR_NOTES_BLOCK = "NOTES FROM THE EDITOR, after a failed attempt:\n{n}\n\n"


def _with_block(user: str, marker: str, block: str) -> str:
    """Insert `block` immediately before `marker` in a user turn."""
    idx = user.rfind(marker)
    if idx < 0:  # pragma: no cover - every chat prompt here carries the marker
        return user + "\n\n" + block.rstrip()
    return user[:idx] + block + user[idx:]


def _with_facts(user: str, marker: str, facts: str) -> str:
    """Insert `FACTS_BLOCK` immediately before `marker` in a user turn."""
    if not facts or not facts.strip():
        return user
    block = FACTS_BLOCK.format(f=facts.strip())
    idx = user.rfind(marker)
    if idx < 0:  # pragma: no cover - every chat prompt here carries the marker
        return user + "\n\n" + block.rstrip()
    return user[:idx] + block + user[idx:]


def build_prompt(
    paragraph: str,
    spec: CandidateSpec,
    facts: str = "",
    extra_notes: str = "",
    shots: Optional[Sequence[Tuple[str, str, str, str]]] = None,
) -> Prompt:
    """The prompt for one candidate rewrite of one paragraph.

    `extra_notes` is the repair stage's one-line instruction for a retry
    ("use only the listed particulars", "match the draft's length"). In the
    freeform shape it is appended to the NOTES slot, which is the only place
    a base model reads an instruction from; in the chat shapes it is a block
    ahead of the draft. `shots` replaces the freeform exemplar triples.

    The paragraph is placed last in both shapes. Instruction-following degrades
    with distance from the end of the context, and the rules are what we most
    need obeyed, so the rules sit next to the draft rather than at the top of a
    long system message.

    `facts` (see `PipelineConfig.facts`) is inserted ahead of the draft in the
    two chat shapes. The freeform base-model shape ignores it: its few-shot
    continuation format has no place for an instruction, and base candidates
    are gated on exactly the same invariants regardless.
    """
    if not paragraph or not paragraph.strip():
        raise ValueError("paragraph must not be empty")

    extra = (extra_notes or "").strip()

    if spec.style == "register":
        user = _with_facts(REGISTER_USER.format(d=paragraph.strip()), "DRAFT:\n", facts)
        if extra:
            user = _with_block(user, "DRAFT:\n", EDITOR_NOTES_BLOCK.format(n=extra))
        return Prompt(
            kind="chat",
            messages=(("system", REGISTER_SYSTEM), ("user", user)),
        )

    if spec.style == "freeform":
        notes = facts.strip() if facts and facts.strip() else _FREEFORM_NO_NOTES
        if extra:
            # A NOTES line, in the pattern's own vocabulary: the exemplar
            # notes are "- " bullets, so the instruction is one more bullet.
            notes = notes + "\n- " + extra
        return Prompt(
            kind="text",
            text=(
                _FREEFORM_HEADER
                + _freeform_shots(shots)
                + "DRAFT:\n"
                + paragraph.strip()
                + "\n\nNOTES:\n"
                + notes
                + "\n\nHUMAN:\n"
            ),
        )

    user = (
        STYLE_CONTRACT.format(banned=BANNED_VOCABULARY)
        + "\n\nVoice: "
        + spec.persona
        + ".\nStructural move: "
        + spec.constraint
        + "\n\nMethod: read the draft below once for what it says. Then put it "
        "aside and write what it says in your own sentences, keeping every "
        "fact, number, name, quotation and citation. Output the rewrite and "
        "nothing else.\n\nDRAFT TO REVISE:\n"
        + paragraph.strip()
    )
    user = _with_facts(user, "DRAFT TO REVISE:\n", facts)
    if extra:
        user = _with_block(user, "DRAFT TO REVISE:\n", EDITOR_NOTES_BLOCK.format(n=extra))
    return Prompt(
        kind="chat",
        messages=(("system", _SYSTEM_FAITHFUL), ("user", user)),
    )


# ------------------------------------------------------- completion cleaning


#: Openers the instruct model prepends despite rule 8. Matched at the start of
#: the completion only, and only when a blank line or a colon follows, so a
#: rewrite that legitimately begins "Here is the problem." is not truncated.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is| are)|below is|the following is|"
    r"revised(?: paragraph)?|rewritten(?: paragraph)?|revision)\b[^\n:]{0,80}:\s*",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_DASH_RE = re.compile(r"\s*[—–]\s*|\s+--\s+")
_LABEL_RE = re.compile(r"^\s*(?:HUMAN|DRAFT|REVISION|OUTPUT)\s*:\s*", re.IGNORECASE)
_TRAILING_NOTE_RE = re.compile(
    r"\n\s*(?:\(?note[:\s]|let me know|i hope this|this revision\b|"
    r"in this (?:revision|version)\b).*$",
    re.IGNORECASE | re.DOTALL,
)


def clean_completion(raw: str, stops: Sequence[str] = ()) -> str:
    """Strip the scaffolding a chat model wraps around a rewrite.

    Removes, in order: everything at or after the first stop string, a markdown
    code fence, a leading `HUMAN:`-style label, a leading "Here is the revised
    paragraph:" preamble, a trailing "Note: I have..." commentary block, and
    quotation marks wrapping the whole thing.

    The wrapping-quote case is deliberately narrow: a paragraph is unwrapped
    only when the *entire* text is quoted and contains no other quote
    character, because the meaning gate in `engine.check_invariants` counts
    quotation spans and stripping a real one is a factual change.
    """
    text = raw or ""
    for stop in stops:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]

    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1)

    text = text.strip()
    text = _LABEL_RE.sub("", text, count=1)
    text = _PREAMBLE_RE.sub("", text, count=1)
    text = _TRAILING_NOTE_RE.sub("", text)
    # No em or en dashes in output, ever: research/23 §4 lists dash density
    # among GPTZero's published bypasser signatures and the product owner's
    # rule is absolute. A spaced dash becomes a comma; a closed one too.
    text = _DASH_RE.sub(", ", text)
    text = text.strip()

    for open_q, close_q in (('"', '"'), ("“", "”")):
        if (
            len(text) > 2
            and text.startswith(open_q)
            and text.endswith(close_q)
            and open_q not in text[1:-1]
            and close_q not in text[1:-1]
        ):
            text = text[1:-1].strip()
            break

    # Collapse the model's habit of returning several short blocks: the caller
    # asked for one paragraph and the pipeline reassembles by paragraph.
    return re.sub(r"\n{2,}", "\n", text).strip()


def estimate_max_tokens(paragraph: str, headroom: float = 1.9) -> int:
    """Generation cap for one paragraph, from its own length.

    Roughly 1.35 tokens per word for English prose under Qwen's tokenizer, times
    `headroom` so a candidate that legitimately runs long is not truncated into
    a gate failure. Floored at 96 so a two-sentence paragraph still has room,
    capped at 900 so one runaway candidate cannot eat the whole time budget.
    """
    n_words = len(paragraph.split())
    return int(max(96, min(900, n_words * 1.35 * headroom)))


# -------------------------------------------------------------- the backends


@dataclass(frozen=True)
class GenerationRequest:
    """One completion to produce."""

    prompt: Prompt
    temperature: float
    max_tokens: int
    #: Opaque to the backend; the pipeline uses it to route completions back to
    #: (paragraph, candidate).
    tag: Any = None


@dataclass
class GenerationBatch:
    """What a backend returns for a list of requests, in request order."""

    texts: List[str]
    seconds: float = 0.0
    generation_tokens: int = 0
    #: Aggregate decode rate across the whole call, not per stream.
    tokens_per_second: float = 0.0
    #: True when the backend used a real batched decode rather than a loop.
    batched: bool = False
    backend: str = ""
    model: str = ""


class Backend:
    """What `pipeline.py` needs from a text generator.

    Subclass or duck-type. `generate` receives every request for one stage at
    once so a backend that can batch does; a backend that cannot loops.
    """

    name = "backend"
    model_name = ""

    def available(self) -> bool:  # pragma: no cover - trivial
        return True

    def unavailable_reason(self) -> Optional[str]:  # pragma: no cover - trivial
        return None

    def load(self) -> None:  # pragma: no cover - trivial
        return None

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        raise NotImplementedError


def mlx_available() -> bool:
    """Whether `mlx_lm` can be imported at all. Cheap: no weights are touched."""
    try:
        import mlx_lm  # noqa: F401
    except Exception:  # noqa: BLE001 - an ImportError or a broken install both count
        return False
    return True


def mlx_unavailable_reason() -> Optional[str]:
    """A sentence a user can act on, or None when MLX is importable."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return (
            "mlx-lm is not installed. It needs Apple Silicon: "
            "pip install mlx-lm. The rule-based engine at POST /api/humanize "
            "works without it."
        )
    except Exception as exc:  # noqa: BLE001
        return f"mlx-lm failed to import: {exc}"
    return None


class MlxBackend(Backend):
    """`mlx_lm` on the local GPU.

    Batching. `mlx_lm.batch_generate` (0.29+) decodes many prompts in one pass
    and is a large win: measured on an M4 Pro with Qwen2.5-3B-Instruct-4bit,
    61.8 tok/s single-stream against 200-210 tok/s aggregate at batch 6, so
    six candidates cost about the same wall-clock as two. When the installed
    version has no `batch_generate`, `generate` falls back to a loop and
    reports `batched=False`, and the pipeline's time budget takes over.

    Temperature grouping. `batch_generate` takes one sampler for the whole
    call, so a batch can only hold requests that share a temperature. The
    default ladder has six distinct temperatures, so grouping by exact value
    gives six calls of one prompt each and throws the batching entirely away -
    measured, that is 4.3s for three candidates against 1.5s batched.

    So requests are sorted by temperature and split into `temperature_groups`
    contiguous cohorts, each decoded at its cohort's mean temperature. The
    default of 2 keeps a genuinely cool cohort and a genuinely warm one while
    halving or better the number of calls. It is a deliberate trade and the
    evidence supports it: temperature is the *weak* diversity lever here (six
    completions of one prompt at 0.7 through 2.0 all opened with the same seven
    words and all scored 1.0000), and the prompts, which are the strong lever,
    are untouched. Set `temperature_groups=1` for maximum speed or to
    `len(specs)` to honour every temperature exactly.
    """

    name = "mlx"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_p: float = 0.95,
        max_batch: int = 8,
        temperature_groups: int = 2,
        adapter_path: Optional[str] = None,
    ):
        self.model_name = model_name
        #: LoRA adapter directory passed to `mlx_lm.load`, or None.
        self.adapter_path = adapter_path
        self.top_p = top_p
        self.temperature_groups = max(1, int(temperature_groups))
        #: Cap on prompts per batched call. 24GB of unified memory holds far
        #: more than this at 3B/4-bit, but peak memory scales with batch and
        #: the detector needs ~2GB of it at the same time.
        self.max_batch = max_batch
        self._model: Any = None
        self._tokenizer: Any = None
        self._batch_fn: Any = None
        self._load_error: Optional[str] = None
        #: Held for the whole of a `generate()` and of an `unload()`, so an
        #: eviction from `backend_for` cannot drop the weights out from
        #: under a batch that another request is decoding with them.
        self._gen_lock = threading.RLock()

    # -- availability ------------------------------------------------------

    def available(self) -> bool:
        if self._load_error is not None:
            return False
        return mlx_available()

    def unavailable_reason(self) -> Optional[str]:
        if self._load_error is not None:
            return self._load_error
        return mlx_unavailable_reason()

    def load(self) -> None:
        """Load the weights, or record why they could not be loaded.

        Raises `RuntimeError` with a message meant for a user, never a bare
        import error: a missing model is a normal state for this package and
        the HTTP layer turns it into a 503 with the rule-based fallback named.
        """
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        reason = mlx_unavailable_reason()
        if reason is not None:
            self._load_error = reason
            raise RuntimeError(reason)
        try:
            from mlx_lm import load as mlx_load

            self._model, self._tokenizer = mlx_load(
                self.model_name, adapter_path=self.adapter_path
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = (
                f"Could not load {self.model_name!r} with mlx-lm: {exc}. "
                "The rule-based engine at POST /api/humanize works without it."
            )
            raise RuntimeError(self._load_error)
        try:
            from mlx_lm import batch_generate

            self._batch_fn = batch_generate
        except ImportError:
            self._batch_fn = None

    def unload(self) -> None:
        """Drop the weights and hand the MLX allocator cache back.

        The reverse of `load()`: afterwards `loaded` is False and the next
        `load()` reads the checkpoint again. A recorded load error is kept, so
        an unloaded backend that could never load still says why.
        """
        with self._gen_lock:
            model, tokenizer, batch_fn = self._model, self._tokenizer, self._batch_fn
            self._model = None
            self._tokenizer = None
            self._batch_fn = None
        release_memory(model, tokenizer, batch_fn)

    @property
    def loaded(self) -> bool:
        """True while the weights are resident."""
        return self._model is not None

    @property
    def batched(self) -> bool:
        """True when a real batched decode is available. Valid after `load()`."""
        return self._batch_fn is not None

    # -- generation --------------------------------------------------------

    def _encode(self, prompt: Prompt) -> List[int]:
        tok = self._tokenizer
        if prompt.kind == "chat" and hasattr(tok, "apply_chat_template"):
            return list(
                tok.apply_chat_template(
                    prompt.message_dicts, add_generation_prompt=True
                )
            )
        return list(tok.encode(prompt.rendered()))

    def _sampler(self, temperature: float):
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(temp=float(temperature), top_p=self.top_p)

    def _cohorts(
        self, requests: Sequence[GenerationRequest]
    ) -> List[Tuple[float, List[int]]]:
        """`(temperature, request indices)` cohorts, sorted cool to warm.

        When the requests carry no more distinct temperatures than
        `temperature_groups`, each temperature gets its own cohort and nothing
        is approximated. Past that, cohorts are contiguous in temperature and
        as near equal-sized as the count allows, so a cohort's mean is a fair
        stand-in for each of its members.

        Returns request *indices*, because completions have to go back where
        they came from.
        """
        order = sorted(range(len(requests)), key=lambda i: requests[i].temperature)
        distinct = sorted({r.temperature for r in requests})
        if len(distinct) <= self.temperature_groups:
            # Every temperature can be honoured exactly and still batch. This
            # is the common case for a caller that passes one temperature.
            exact: Dict[float, List[int]] = {}
            for i in order:
                exact.setdefault(requests[i].temperature, []).append(i)
            return [(t, exact[t]) for t in distinct]
        n_groups = max(1, min(self.temperature_groups, len(order)))
        cohorts: List[Tuple[float, List[int]]] = []
        for g in range(n_groups):
            lo = (g * len(order)) // n_groups
            hi = ((g + 1) * len(order)) // n_groups
            members = order[lo:hi]
            if not members:
                continue
            mean = sum(requests[i].temperature for i in members) / len(members)
            cohorts.append((round(mean, 4), members))
        return cohorts

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        with self._gen_lock:
            return self._generate_locked(requests)

    def _generate_locked(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        requests = list(requests)
        if not requests:
            return GenerationBatch(
                texts=[], backend=self.name, model=self.model_name, batched=False
            )
        self.load()

        texts: List[Optional[str]] = [None] * len(requests)
        started = time.perf_counter()
        total_tokens = 0
        used_batching = False

        for temperature, indices in self._cohorts(requests):
            for chunk_start in range(0, len(indices), self.max_batch):
                chunk = indices[chunk_start : chunk_start + self.max_batch]
                prompts = [self._encode(requests[i].prompt) for i in chunk]
                max_tokens = max(requests[i].max_tokens for i in chunk)
                sampler = self._sampler(temperature)

                if self._batch_fn is not None:
                    response = self._batch_fn(
                        self._model,
                        self._tokenizer,
                        prompts,
                        max_tokens=max_tokens,
                        sampler=sampler,
                    )
                    used_batching = used_batching or len(chunk) > 1
                    for i, out in zip(chunk, response.texts):
                        texts[i] = out
                    stats = getattr(response, "stats", None)
                    total_tokens += int(getattr(stats, "generation_tokens", 0) or 0)
                else:
                    from mlx_lm import generate as mlx_generate

                    for i, tokens in zip(chunk, prompts):
                        out = mlx_generate(
                            self._model,
                            self._tokenizer,
                            prompt=tokens,
                            max_tokens=requests[i].max_tokens,
                            sampler=sampler,
                        )
                        texts[i] = out
                        total_tokens += len(self._tokenizer.encode(out))

        seconds = time.perf_counter() - started
        return GenerationBatch(
            texts=[t or "" for t in texts],
            seconds=seconds,
            generation_tokens=total_tokens,
            tokens_per_second=(total_tokens / seconds) if seconds > 0 else 0.0,
            batched=used_batching,
            backend=self.name,
            model=self.model_name,
        )


class EchoBackend(Backend):
    """A dependency-free backend that returns canned completions.

    Not a mock bolted onto the tests from outside: the pipeline has eight
    stages, gates, a time budget and an event protocol, and none of that should
    need a GPU to exercise. `responses` is either a flat list consumed in
    request order or a callable taking the request.
    """

    name = "echo"

    def __init__(self, responses: Any = None, model_name: str = "echo", delay: float = 0.0):
        self.responses = responses
        self.model_name = model_name
        self.delay = delay
        self.calls: List[List[GenerationRequest]] = []

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        requests = list(requests)
        self.calls.append(requests)
        started = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)

        out: List[str] = []
        for n, req in enumerate(requests):
            if callable(self.responses):
                out.append(self.responses(req))
            elif isinstance(self.responses, (list, tuple)):
                cursor = sum(len(c) for c in self.calls[:-1]) + n
                out.append(
                    str(self.responses[cursor])
                    if cursor < len(self.responses)
                    else ""
                )
            else:
                out.append(req.prompt.rendered())
        return GenerationBatch(
            texts=out,
            seconds=time.perf_counter() - started,
            generation_tokens=sum(len(t.split()) for t in out),
            batched=True,
            backend=self.name,
            model=self.model_name,
        )


def default_backend(
    style: str = "faithful", model: Optional[str] = None, adapter: Optional[str] = None
) -> MlxBackend:
    """A fresh MLX backend for a style, without loading anything yet.

    Not memoised: two calls give two backends and, once loaded, two copies of
    the weights. Long-lived callers want `backend_for`. `adapter` is a LoRA
    directory for the `freeform` base checkpoint; None reads
    HUMANIZER_BASE_ADAPTER, and an empty string means the plain base.
    """
    if style == "register":
        path = register_model_path() if model in (None, REGISTER_MODEL) else model
        return MlxBackend(model_name=path, adapter_path=REGISTER_ADAPTER)
    if model is None:
        model = FREEFORM_MODEL if style == "freeform" else DEFAULT_MODEL
    if style == "freeform":
        # An optional LoRA on the base checkpoint (the HIP-style fine-tune of
        # research/17 §A.11 and research/24 §6.6).
        if adapter is None:
            env = __import__("os").environ.get("HUMANIZER_BASE_ADAPTER")
            if env is None:
                adapter = DEFAULT_BASE_ADAPTER if __import__("os").path.isdir(DEFAULT_BASE_ADAPTER) else None
            else:
                adapter = env or None
        return MlxBackend(model_name=model, adapter_path=adapter or None)
    return MlxBackend(model_name=model)


# ------------------------------------------------------------ backend registry

#: Environment variable naming the cap on resident MLX backends.
MAX_RESIDENT_ENV = "HUMANIZER_MAX_RESIDENT_LLM"
#: (style, resolved model path, adapter path or None) -> backend, LRU first.
_BACKENDS: "OrderedDict[Tuple[str, str, Optional[str]], MlxBackend]" = OrderedDict()
_BACKENDS_LOCK = threading.Lock()


def _max_resident_backends() -> int:
    """The cap, read from `MAX_RESIDENT_ENV` on every call; 0 = unlimited."""
    return max_resident(MAX_RESIDENT_ENV, 2)


def backend_for(
    style: str = "faithful", model: Optional[str] = None, adapter: Optional[str] = None
) -> MlxBackend:
    """The process-wide backend for `(style, model)`, built on first use.

    The same arguments return the same object, so the weights are read once
    per process rather than once per request. Each Qwen 4-bit checkpoint is
    1.7-2.2GB resident and every style has its own, so when the registry grows
    past `HUMANIZER_MAX_RESIDENT_LLM` (default 2, so the `mixed` style's two
    checkpoints coexist without reloading per paragraph) the least-recently-requested
    backends are `unload()`ed. They stay usable -- `load()` simply reads the
    checkpoint again -- so a caller still holding an evicted backend keeps
    working, at the cost of a reload.
    """
    fresh = default_backend(style, model, adapter)
    key = (style, fresh.model_name, fresh.adapter_path)
    evicted: List[MlxBackend] = []
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = fresh
            _BACKENDS[key] = backend
        _BACKENDS.move_to_end(key)
        cap = _max_resident_backends()
        while cap and len(_BACKENDS) > cap:
            _, old = _BACKENDS.popitem(last=False)
            evicted.append(old)
    for old in evicted:
        old.unload()
    return backend


def loaded_backends() -> List[str]:
    """`style:model` for each registered backend, loaded ones first-class.

    Only backends whose weights are resident are listed; a registered backend
    that has never been `load()`ed costs nothing and would be misleading here.
    """
    with _BACKENDS_LOCK:
        items = list(_BACKENDS.items())
    return [f"{style}:{model}" for (style, model, _), b in items if b.loaded]


def release_backends() -> None:
    """Unload and forget every registered backend."""
    with _BACKENDS_LOCK:
        dropped = list(_BACKENDS.values())
        _BACKENDS.clear()
    for backend in dropped:
        backend.unload()


#: Exposed for `pipeline.py`, which needs the stop list when cleaning a
#: freeform completion and should not have to know the header format.
STOPS_FOR_STYLE: Dict[str, Tuple[str, ...]] = {
    "faithful": (),
    "freeform": FREEFORM_STOPS,
    "register": (),
}
