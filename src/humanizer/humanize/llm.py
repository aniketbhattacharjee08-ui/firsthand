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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_MODEL",
    "FREEFORM_MODEL",
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
]


#: 4-bit Qwen2.5-3B-Instruct. Measured on an M4 Pro: 0.5s warm load, 61.8 tok/s
#: single-stream, 200-210 tok/s aggregate at batch 6. ~1.7GB on disk.
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

#: The same model without instruction tuning, for `style="freeform"`.
FREEFORM_MODEL = "mlx-community/Qwen2.5-3B-4bit"


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
11. No lists of three ("X, Y, and Z"), no "not just X but Y"."""


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
    _spec(
        0,
        "a careful researcher revising your own draft for clarity",
        "Open on the most concrete noun in the draft, not on a general claim.",
        0.70,
    ),
    _spec(
        1,
        "a domain expert writing plainly for a colleague who already knows the field",
        "Put the main claim in the middle of the paragraph, not in the first sentence.",
        0.80,
    ),
    _spec(
        2,
        "an editor cutting filler out of someone else's prose",
        "Delete every transition. Let the sentences sit next to each other unglued.",
        0.90,
    ),
    _spec(
        3,
        "a working journalist on deadline, writing for an informed reader",
        "Split the longest sentence into one long sentence and one very short one.",
        1.00,
    ),
    _spec(
        4,
        "a practitioner writing from experience rather than from a textbook",
        "Replace each abstract noun with the concrete thing it stands for.",
        0.85,
    ),
    _spec(
        5,
        "a graduate student turning lecture notes back into prose",
        "State the consequence first and the cause second.",
        0.95,
    ),
    _spec(
        6,
        "a reviewer rewriting a passage you found badly organised",
        "Drop the draft's opening move entirely and begin on its second idea.",
        0.75,
    ),
    _spec(
        7,
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
        8,
        "a writer who hears the rhythm of a paragraph before its argument",
        "Begin every sentence with a different kind of word: a noun, then a "
        "subordinate clause, then a verb-ish participle, then a pronoun. Never "
        "open two sentences the same way.",
        0.90,
    ),
    _spec(
        9,
        "an essayist who distrusts symmetry",
        "Make one sentence a single clause of six words or fewer and the next a "
        "three-clause sentence of thirty or more. No sentence may sit between "
        "seventeen and twenty-three words.",
        1.00,
    ),
    _spec(
        10,
        "a historian who front-loads circumstance",
        "Open at least two sentences with a fronted adverbial or subordinate "
        "clause (Because..., After..., Where..., Although...) before the "
        "subject arrives.",
        0.85,
    ),
    _spec(
        11,
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
    ladder = FREEFORM_SPECS if style == "freeform" else CANDIDATE_SPECS
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

#: The `freeform` few-shot. One pair: a stereotypically generated paragraph and
#: a human-voiced rewrite of it. The header frames the task as a log the model
#: is continuing, which is the only framing a base model reliably picks up.
_FREEFORM_DRAFT = (
    "In today's rapidly evolving digital landscape, artificial intelligence "
    "has become an increasingly important tool for organizations of all sizes. "
    "It is important to note that the adoption of these technologies requires "
    "careful consideration of both the opportunities and the challenges "
    "involved. Furthermore, the ethical implications of these systems cannot "
    "be overlooked. Ultimately, success in this area depends on a commitment "
    "to transparency, accountability, and continuous improvement."
)

_FREEFORM_HEADER = (
    "Editing log. Each entry pairs a machine-written draft with the version a "
    "human writer produced from it. The human version keeps every fact, every "
    "number and every name, and throws out the transitions, the abstractions "
    "and the tidy closing sentence.\n\n"
)

#: Where a base-model completion stops. It has no end-of-turn token, so it will
#: happily invent the next log entry.
FREEFORM_STOPS = ("\nDRAFT:", "\nEditing log", "\nHUMAN:")


def build_prompt(paragraph: str, spec: CandidateSpec) -> Prompt:
    """The prompt for one candidate rewrite of one paragraph.

    The paragraph is placed last in both shapes. Instruction-following degrades
    with distance from the end of the context, and the rules are what we most
    need obeyed, so the rules sit next to the draft rather than at the top of a
    long system message.
    """
    if not paragraph or not paragraph.strip():
        raise ValueError("paragraph must not be empty")

    if spec.style == "freeform":
        return Prompt(
            kind="text",
            text=(
                _FREEFORM_HEADER
                + "DRAFT:\n"
                + _FREEFORM_DRAFT
                + "\n\nHUMAN:\n"
                + STYLE_EXEMPLAR
                + "\n\nDRAFT:\n"
                + paragraph.strip()
                + "\n\nHUMAN:\n"
            ),
        )

    user = (
        STYLE_CONTRACT.format(banned=BANNED_VOCABULARY)
        + "\n\nVoice: "
        + spec.persona
        + ".\nStructural move: "
        + spec.constraint
        + "\n\nRewrite the draft below and nothing else.\n\nDRAFT TO REVISE:\n"
        + paragraph.strip()
    )
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
    ):
        self.model_name = model_name
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

            self._model, self._tokenizer = mlx_load(self.model_name)
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


def default_backend(style: str = "faithful", model: Optional[str] = None) -> MlxBackend:
    """The MLX backend for a style, without loading anything yet."""
    if model is None:
        model = FREEFORM_MODEL if style == "freeform" else DEFAULT_MODEL
    return MlxBackend(model_name=model)


#: Exposed for `pipeline.py`, which needs the stop list when cleaning a
#: freeform completion and should not have to know the header format.
STOPS_FOR_STYLE: Dict[str, Tuple[str, ...]] = {
    "faithful": (),
    "freeform": FREEFORM_STOPS,
}
