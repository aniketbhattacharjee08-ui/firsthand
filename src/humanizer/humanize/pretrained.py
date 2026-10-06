"""Published *evasion-trained* checkpoints, wrapped as a humanizing engine.

WHAT THIS IS
------------
Every other rewriting path in this repo asks a **general instruct model** to
sound human. `humanize.transforms` does it with hand-written rules,
`humanize.pipeline` does it with a prompted Qwen2.5. Both fail for the same
reason: the model's own output distribution is part of what the detector fires
on, so asking it nicely to leave that distribution does not work.

This module is the other experiment. Several research groups did not prompt a
model, they **trained** one, with a detector's score as the reward signal. That
is a different object, and it deserved a real measurement rather than an
assumption. `pretrained.py` is the wrapper that made that measurement possible
and the place any future checkpoint gets plugged in.

The default checkpoint is `authormist/authormist-originality` (MIT): Qwen2.5-3B
Instruct fine-tuned with GRPO against Originality.ai's detector as the reward,
from David & Gervais, *AuthorMist: Evading AI Text Detectors with Reinforcement
Learning* (arXiv 2503.08716). It is the strongest published checkpoint of its
kind that actually has downloadable weights.

READ THIS BEFORE YOU BELIEVE THE PAPER'S NUMBER
-----------------------------------------------
AuthorMist reports 92.3% attack success against GPTZero. **We do not reproduce
anything like that against our yardstick, and the gap is not a bug in this
wrapper.** Measured here on `.llmbench`'s 9 AI paragraphs, scored with
`detectors.local.ModernDetector` (desklib/ai-text-detector-v1.01):

    configuration                    flips/9  clean   mean p_ai      overlap
    -------------------------------------------------------------------------
    (no-op control)                    0/9     0/9  1.0000 -> 1.0000   1.00
    card prompt, t=0.7                 0/9     0/9  1.0000 -> 0.9972   0.52
    chat template, t=0.7               0/9     0/9  1.0000 -> 0.9994   0.58
    card prompt, best-of-6             1/9     0/9  1.0000 -> 0.8312   0.52
    -------------------------------------------------------------------------
    (`clean` = flips that also kept the content: overlap >= 0.45, no invariant
    loss, and length within 0.7-1.4x of the source.)

Zero content-clean verdict flips, from a checkpoint trained for exactly this
task. That is the same 0-of-9 that rule-based rewriting, prompted Qwen2.5-3B
and 7B, translation round trips and cross-family blending all returned.

THE ONE FLIP WAS AN ARTIFACT, AND WHY THAT MATTERS
---------------------------------------------------
Best-of-6 sampling produced a single raw flip -- `ai-gen-teaching`, 1.0000 ->
0.2088 -- and it is worth dissecting because it is exactly the result somebody
would otherwise report as a breakthrough.

The "rewrite" is 375 words long against a 155-word source. Driven in
completion mode the model ran past its own answer and paraphrased the
paragraph three times over, splicing in its own stage directions ("Paraphrase
with less formal writing and friendly approach:", "Let me refine the passage
using more human friendly language:"). The result is incoherent, and desklib
scores incoherent text as human.

That is not a one-off. Across the 9 paragraphs, length ratio and detector
score are strongly anticorrelated (Pearson r = -0.588), and splitting on it is
stark:

    the model rambled  (length ratio > 1.5, n=5)   mean p_ai after = 0.7002
    the model did not  (length ratio <= 1.5, n=4)  mean p_ai after = 0.9950

Every bit of apparent evasion in the best-of-6 run came from over-generation.
When AuthorMist produced a normal-length rewrite it scored 0.9950, which is
the no-op control to three decimal places. `_gate`'s length-ratio bound is
what keeps this out of the product, and `PretrainedConfig.prompt_style`
defaults to `"chat"` partly because the chat template terminates properly and
does not produce this failure at all.

The one place AuthorMist visibly *works* is against older detectors, and that
result is a trap worth naming. With the paper's own completion-style prompt it
took `andreas122001/roberta-academic-detector` from 0.9998 to 0.4323 and
flipped 5 of 9; best-of-6 took it to 0.3481 and 6 of 9. But those are the same
configurations that drove content overlap to 0.52 and corrupted a number, date
or citation in 4-5 of 9 paragraphs. Run the same checkpoint in a configuration
that *keeps* the content (chat template, overlap 0.58, invariant failures 2 of
9) and the academic detector goes back to 0/9 as well. The apparent evasion was
content destruction, not evasion, which is exactly the failure mode the gates
in `pipeline._gate_candidate` exist to catch.

A sibling experiment on this repo found the same split from another direction:
back-translated text flipped `roberta-academic-detector` 9 of 9 and the 2023
`Hello-SimpleAI/chatgpt-detector-roberta` 6 of 9 while flipping desklib 0 of 9.
Older detectors are simply easy. AuthorMist's published 92.3% was measured
against the GPTZero of its day. **Quote the desklib number, not the academic
one, and never the paper's.**

WHY IT STILL FAILS, WHICH IS THE USEFUL PART
--------------------------------------------
The obvious explanation is overfitting to the reward: AuthorMist's reward was
one 2024-era detector's score, so what it learned was that detector's decision
surface, and that surface is not desklib's. Desklib is a DeBERTa-v3-large
trained on RAID, which contains adversarial and paraphrased attacks by
construction.

**That explanation is not sufficient, and StealthRL is why.** StealthRL was
trained with GRPO against a multi-detector *ensemble* specifically to fix
single-detector overfitting, and against desklib it does no better -- 0 of 9 at
p_ai 0.9985, with the most aggressive rewriting of anything tested. Widening
the reward from one detector to several bought nothing.

What is left is the explanation the other experiments on this repo converged on
independently. Rule-based substitution, prompted rewriting at 3B and 7B,
translation round trips through German and Chinese, cross-family token
blending, and now two RL-tuned evasion policies have all returned 0 of 9. The
one thing they have in common is that every one of them is a **surface**
operation: they change which words carry the content, never the content plan or
the register underneath it. Desklib appears to read that plan, and a paraphrase
by construction preserves it. The corroborating control is that the same
translation round trip leaves genuine human prose human (0.1765 -> 0.0926), so
desklib is not simply flagging "text that passed through a model".

So the honest summary is that a detector-reward-trained checkpoint *is* a
genuinely different object from a prompted instruct model, and it is still not
enough -- not because the reward was too narrow, but because the whole class of
paraphrase-shaped attacks does not touch what this detector reads. Nothing
published today generalises across detector families.

IT IS ALSO NOT BETTER THAN WHAT WE ALREADY SHIP
------------------------------------------------
This is the finding that should stop anyone from wiring this in as the default.
Measured on the same 9 paragraphs and the same detector, against the existing
prompted pipeline (`humanize.pipeline`, `.llmbench/results_faithful.json`):

    engine                            flips/9  p_ai after  overlap  s/para  human
    -------------------------------------------------------------------------------
    prompted Qwen2.5-3B, faithful       0/9      0.9998     0.849    12.4    0/5
    AuthorMist, chat template           0/9      0.9994     0.577    10.6    0/5
    AuthorMist, paper's card prompt     0/9      0.9972     0.519    18.5    4/5
    -------------------------------------------------------------------------------
    (`human` = of 5 genuine PMC human paragraphs, how many the rewrite pushed
    over 0.5 and made *look* AI-written. Lower is better; this is a safety
    check, not a performance one.)

The prompted pipeline we already have wins on content fidelity by a wide margin
(0.849 against 0.577) and ties on everything else that matters. The
detector-reward-trained checkpoint buys nothing here. And the paper's own
prompt configuration is actively unsafe: it made 4 of 5 genuine human
paragraphs read as AI-written, which is the same pathology an ungated instruct
model shows.

EVERY CHECKPOINT THAT WAS TESTED
--------------------------------
Same 9 AI paragraphs, same detector, same gates. Raw per-paragraph rows are in
`.weightshunt/res_*.json`; the runners that produced them are beside them.

    checkpoint                              trained vs  flips clean p_ai after ovl
                                            a detector?
    ------------------------------------------------------------------------------
    (no-op control)                             --       0/9   0/9   1.0000  1.00
    authormist/authormist-originality  3.1B    yes       1/9   0/9   0.8312  0.52
    suraj-ranganath/StealthRL          4B      yes       0/9   0/9   0.9985  0.20
    SamSJackson/paraphrase-dipper-no-ctx 1B    no        0/9   0/9   0.9807  0.27
    humarin/chatgpt_paraphraser_on_T5_base .2B no        0/9   0/9   0.9999  0.60
    ------------------------------------------------------------------------------

Two of those rows kill specific hypotheses and are worth reading on their own.

**StealthRL is the strongest disconfirmation in the set.** Its reward was a
multi-detector *ensemble*, which is precisely the fix for the "AuthorMist
overfitted to one detector" story told above -- and it still returns 0 of 9, at
p_ai 0.9985, which is the no-op control. It rewrites more aggressively than
anything else tested (148 words in, 44 out; overlap 0.20) and desklib does not
notice. Because that compression is severe enough to be a confound, it was
measured a second time with an explicit length-matching instruction: overlap
rose to 0.29 and p_ai stayed at 0.9873, still 0/9. Training against several
detectors at once does not produce a policy that transfers to a detector
outside the set.

**The T5 row kills the model-family hypothesis.** A non-Qwen decoder might have
sat in a different output distribution; it does not. A T5 paraphraser leaves
desklib at 0.9999.

`KNOWN_CHECKPOINTS` below records the rest of the hunt, including the four
methods -- MASH, SICO, RAFT, HMGC, ToBlend -- that published papers and code
but never published weights, and the 11B DIPPER that was skipped on disk.

WHAT THIS MODULE IS GOOD FOR, THEN
----------------------------------
Mainly it is **the socket and the record**. When a checkpoint appears that was
trained against a modern detector ensemble rather than a 2024-era one,
`PretrainedConfig.model` is the only thing that has to change, and
`KNOWN_CHECKPOINTS` means nobody has to redo the search that found there is no
such checkpoint today. It is a serviceable second paraphraser at roughly the
speed of the existing one, and nothing more than that.

INTERFACE (for whoever wires this into the API)
-----------------------------------------------
Deliberately shaped like `humanize.pipeline`, so the HTTP layer can treat them
interchangeably:

    available()                 -> bool          # never raises, never loads
    unavailable_reason()        -> str | None    # a user-facing sentence
    humanize_pretrained(text, config=None) -> PretrainedResult
    PretrainedResult.as_dict()  -> the JSON body

`as_dict()` returns `original`, `humanized`, `model`, `backend`, `before`,
`after`, `summary` and `candidates`, which is a subset of what
`LlmHumanizeResult.as_dict()` returns with the same meanings, plus
`summary["notice"]` -- a plain-language warning that this engine did not flip a
verdict on our benchmark, which any UI showing this path should surface.

The model is loaded lazily on the first rewrite and cached process-wide;
`available()` only checks that torch and the weights could be reached, so a
machine with no torch degrades to `False` rather than to an exception.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..memory import max_resident, release_memory

from .pipeline import (
    AI_THRESHOLD,
    PipelineConfig,
    _invariant_failures,
    _label,
    _measure,
    _score,
    content_overlap,
)

__all__ = [
    "DEFAULT_MODEL",
    "KNOWN_CHECKPOINTS",
    "PretrainedConfig",
    "PretrainedCandidate",
    "PretrainedResult",
    "available",
    "unavailable_reason",
    "humanize_pretrained",
    "clear_model_cache",
    "loaded_models",
]


#: `authormist/authormist-originality`, MIT. Qwen2.5-3B-Instruct + GRPO against
#: Originality.ai. 12.4GB on disk (the repo ships fp32), ~6.2GB resident when
#: loaded as fp16, which is what this module does.
DEFAULT_MODEL = "authormist/authormist-originality"


#: Every published checkpoint that was hunted for and what happened. Kept as
#: data because the negative results are the point: a future reader should not
#: have to repeat the search to learn that MASH and SICO never shipped weights.
KNOWN_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "authormist/authormist-originality": {
        "trained_against_detector": True,
        "method": "GRPO, Originality.ai reward",
        "paper": "arXiv:2503.08716",
        "params": "3.1B",
        "disk_gb": 12.4,
        "license": "mit",
        "status": "tested",
        "flips_of_9": 0,
    },
    "suraj-ranganath/StealthRL": {
        "trained_against_detector": True,
        "method": "GRPO, multi-detector *ensemble* reward, LoRA r=32 all-linear",
        "paper": "arXiv:2602.08934",
        "params": "4B base + 0.28GB adapter",
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "license": "mit",
        "status": "tested",
        "flips_of_9": 0,
        "note": "The most interesting negative result of the hunt. An "
                "*ensemble* reward is exactly the fix for AuthorMist's "
                "single-detector overfitting, and it still returns 0/9 at "
                "p_ai 0.9985. It also compresses hard -- 148 words in, 44 "
                "out -- so it summarises rather than paraphrases, and it "
                "pushed 4 of 5 genuine human paragraphs over the AI "
                "threshold.",
    },
    "SamSJackson/paraphrase-dipper-no-ctx": {
        "trained_against_detector": False,
        "method": "DIPPER control-code paraphrase, 1B reproduction",
        "paper": "arXiv:2303.13408",
        "params": "0.97B",
        "disk_gb": 3.9,
        "license": "mit",
        "status": "tested",
        "flips_of_9": 0,
    },
    "kalpeshk2011/dipper-paraphraser-xxl": {
        "trained_against_detector": False,
        "method": "DIPPER, the real 11B T5",
        "params": "11B",
        "disk_gb": 45.0,
        "status": "skipped: 45GB of weights against 31GB of free disk, and "
                  "11B fp32 does not fit in 25.7GB of RAM. The 0.97B "
                  "reproduction above was tested in its place and returned "
                  "0/9 with content overlap 0.27.",
    },
    "MASH": {
        "trained_against_detector": True,
        "method": "0.1B BART, SFT -> DPO -> refine",
        "status": "no weights published. Nothing on the Hub under this name; "
                  "`catninja123/mash-authormist-essay` is an unlabelled "
                  "third-party Qwen2.5-3B with an empty model card and no "
                  "stated provenance, not the paper's checkpoint.",
    },
    "humarin/chatgpt_paraphraser_on_T5_base": {
        "trained_against_detector": False,
        "method": "T5-base paraphraser -- tested as the 'different model "
                  "family' hypothesis, since a non-Qwen decoder might sit in a "
                  "different output distribution. It does not help.",
        "params": "0.22B",
        "license": "openrail",
        "status": "tested",
        "flips_of_9": 0,
    },
    "SICO": {"status": "no weights published (code-only release)"},
    "RAFT": {"status": "no weights published (code-only release)"},
    "HMGC": {"status": "no weights published (code-only release)"},
    "ToBlend": {"status": "no weights published (code-only release)"},
}


#: The prompt AuthorMist was trained on, verbatim from its model card. Keeping
#: it exact matters: an RL-tuned policy is on-policy only for the prompt shape
#: it saw during training, and the chat-template form measurably moves the
#: detector less (see the table in the module docstring).
CARD_PROMPT = (
    "Please paraphrase the following text to make it more human-like while "
    "preserving the original meaning:\n\n%s\n\nParaphrased text:"
)

#: Scaffolding the model emits around the rewrite. `llm.clean_completion`
#: truncates at the first of these it finds.
#:
#: The last group is not cosmetic. In `card` (completion) mode this checkpoint
#: runs past its own answer and paraphrases the paragraph two or three times
#: over, narrating as it goes -- which triples the length, and desklib scores
#: the resulting incoherent text as human. That is the whole of the one
#: "verdict flip" this engine has ever produced. Truncating at the stage
#: direction is what turns that back into an honest measurement.
STOP_STRINGS: Tuple[str, ...] = (
    "Sure, here",
    "Here is a",
    "Here's a",
    "Note:",
    "Please paraphrase",
    "\nUser:",
    "\nAssistant:",
    "<|im_end|>",
    # the over-generation tells
    "Paraphrase with",
    "Paraphrased with",
    "Let me refine",
    "Let me rewrite",
    "Here is another",
    "Alternatively,",
    "\nRewritten",
)


@dataclass(frozen=True)
class PretrainedConfig:
    """Everything tunable about one run.

    `prompt_style` defaults to `"chat"` rather than to the paper's `"card"`
    even though `"card"` moves the detector more, because `"card"` moves it by
    destroying content: measured overlap 0.52 against 0.58, and invariant
    failures in 5 of 9 paragraphs against 2 of 9. A rewrite that drops a number
    is not a better rewrite. Pass `prompt_style="card"` to reproduce the
    paper's configuration.
    """

    model: str = DEFAULT_MODEL
    #: Sample this many, keep the one with the lowest detector score among
    #: those that pass the content gates. 1 disables sampling-and-selection.
    n_candidates: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    #: `"chat"` uses the tokenizer's chat template; `"card"` uses the raw
    #: completion prompt the checkpoint was trained with.
    prompt_style: str = "chat"
    #: Multiplier on the source length for the generation cap.
    headroom: float = 2.0
    #: Reject a candidate that keeps less than this share of content words.
    #: Read from the prompted pipeline's default gate so both engines are
    #: held to the same standard; see `GATE_PRESETS` in pipeline.py.
    min_content_overlap: float = PipelineConfig().min_content_overlap
    min_length_ratio: float = 0.75
    max_length_ratio: float = 1.35
    #: `None` picks mps when available, else cpu.
    device: Optional[str] = None
    #: fp16 halves the 12.4GB fp32 checkpoint to ~6.2GB resident.
    dtype: str = "float16"

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"


@dataclass
class PretrainedCandidate:
    """One sample, with the reason it was or was not usable."""

    index: int
    text: str
    temperature: float
    ai_probability: Optional[float] = None
    overlap: float = 0.0
    length_ratio: float = 0.0
    gate_failures: List[str] = field(default_factory=list)
    seconds: float = 0.0
    chosen: bool = False

    @property
    def passed(self) -> bool:
        return not self.gate_failures

    def as_dict(self, with_text: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": self.index,
            "temperature": self.temperature,
            "ai_probability": self.ai_probability,
            "label": _label(self.ai_probability),
            "overlap": self.overlap,
            "length_ratio": self.length_ratio,
            "gate_failures": list(self.gate_failures),
            "passed": self.passed,
            "seconds": self.seconds,
            "chosen": self.chosen,
        }
        if with_text:
            out["text"] = self.text
        return out


@dataclass
class PretrainedResult:
    """The full result. `as_dict()` is the HTTP response body."""

    original: str
    humanized: str
    model: str
    backend: str = "transformers"
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    candidates: List[PretrainedCandidate] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.humanized.strip() != self.original.strip()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "humanized": self.humanized,
            "model": self.model,
            "backend": self.backend,
            "before": self.before,
            "after": self.after,
            "summary": self.summary,
            "candidates": [c.as_dict() for c in self.candidates],
        }


# --------------------------------------------------------------- the notice

#: Shown to the user by any surface that runs this engine. It is not optional
#: garnish: this engine measured 0 verdict flips out of 9 and a UI that
#: presents it as a detector bypass would be lying.
NOTICE = (
    "This engine runs a published checkpoint that was reinforcement-trained "
    "against an AI-text detector (AuthorMist, arXiv:2503.08716). On this "
    "repo's benchmark it changed the verdict on 0 of 9 AI paragraphs when "
    "scored with desklib/ai-text-detector-v1.01. Use it as a faithful "
    "paraphraser, not as a way to pass a detector."
)


# ---------------------------------------------------------- model loading

#: (model, device, dtype) -> (tokenizer, model), least-recently-used first.
#: Each entry is 6-12GB resident, so the default budget is a single checkpoint:
#: loading a different key evicts and frees the one before it.
_MODEL_CACHE: "OrderedDict[Tuple[str, str, str], Any]" = OrderedDict()
#: Environment variable naming the cap on resident paraphraser checkpoints.
MAX_RESIDENT_ENV = "HUMANIZER_MAX_RESIDENT_PRETRAINED"


def loaded_models() -> List[str]:
    """Names of the checkpoints currently resident. For `/api/health`."""
    return sorted({key[0] for key in _MODEL_CACHE})


def clear_model_cache() -> None:
    """Drop every loaded checkpoint, free its memory, re-arm the reachability check.

    ~6.2GB back per resident model. Call this after downloading a checkpoint
    so `available()` stops reporting the cached "not available" verdict.
    """
    dropped = list(_MODEL_CACHE.values())
    _MODEL_CACHE.clear()
    _REACHABLE.clear()
    release_memory(*dropped)


def _max_resident() -> int:
    """The cap, read from `MAX_RESIDENT_ENV` on every call; 0 = unlimited."""
    return max_resident(MAX_RESIDENT_ENV, 1)


def _remember(key: Tuple[str, str, str], value: Any) -> Any:
    """Insert `value` under `key`, evicting and freeing the least-recently-used past the cap."""
    _MODEL_CACHE[key] = value
    _MODEL_CACHE.move_to_end(key)
    cap = _max_resident()
    evicted: List[Any] = []
    while cap and len(_MODEL_CACHE) > cap:
        _, old = _MODEL_CACHE.popitem(last=False)
        evicted.append(old)
    if evicted:
        release_memory(*evicted)
    return value


def _torch_unavailable_reason() -> Optional[str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return (
            "The pretrained-checkpoint engine needs torch and transformers "
            "(%s). The rule-based engine at POST /api/humanize still works."
            % exc
        )
    return None


#: Memoised `(model) -> reason or None`. `available()` is called from a health
#: endpoint; without this it would make an HTTP request to the Hub per call.
_REACHABLE: Dict[str, Optional[str]] = {}


def unavailable_reason(config: Optional[PretrainedConfig] = None) -> Optional[str]:
    """A user-facing sentence, or `None` when the engine can run.

    Deliberately cheap: it checks that the libraries import and that the
    weights are either in the Hub cache or reachable. It does **not** load the
    model, because loading is ~20 seconds and ~6.2GB resident and this is
    called from a health endpoint.

    The cache lookup is tried *before* the network one so that a machine which
    already has the 12GB checkpoint keeps working with the network down. The
    verdict is then memoised per checkpoint; call `clear_model_cache()` to
    re-check after a download.
    """
    reason = _torch_unavailable_reason()
    if reason:
        return reason

    cfg = config or PretrainedConfig()
    if _MODEL_CACHE.get((cfg.model, cfg.resolved_device(), cfg.dtype)) is not None:
        return None
    if cfg.model in _REACHABLE:
        return _REACHABLE[cfg.model]

    verdict: Optional[str] = None
    try:
        from huggingface_hub import snapshot_download

        try:
            snapshot_download(cfg.model, allow_patterns=["config.json"],
                              local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached, so ask the Hub
            if _offline():
                raise
            snapshot_download(cfg.model, allow_patterns=["config.json"])
    except Exception as exc:  # noqa: BLE001
        verdict = (
            "The checkpoint %s is not available (%s). It is ~12GB and is not "
            "downloaded by default; the rule-based engine at "
            "POST /api/humanize still works." % (cfg.model, type(exc).__name__)
        )
    _REACHABLE[cfg.model] = verdict
    return verdict


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip() not in ("", "0", "false")


def available(config: Optional[PretrainedConfig] = None) -> bool:
    """Whether this engine can run right now. Never raises, never loads."""
    try:
        return unavailable_reason(config) is None
    except Exception:  # noqa: BLE001
        return False


def _load(cfg: PretrainedConfig):
    """The tokenizer and model, built once per (checkpoint, device, dtype)."""
    key = (cfg.model, cfg.resolved_device(), cfg.dtype)
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype)
    model = model.to(cfg.resolved_device()).eval()
    return _remember(key, (tokenizer, model))


# ------------------------------------------------------------- generation


def _build_prompt(tokenizer: Any, text: str, cfg: PretrainedConfig) -> str:
    body = CARD_PROMPT % text
    if cfg.prompt_style == "card":
        return body
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(tokenizer: Any, model: Any, text: str, temperature: float,
              cfg: PretrainedConfig) -> str:
    """One sample. Returns "" rather than raising on a backend hiccup.

    MPS occasionally hands `torch.multinomial` a probability tensor containing
    NaN on this checkpoint -- observed once in 14 paragraphs with the card
    prompt. That is a backend bug, not a bad rewrite, and one bad sample must
    not fail the request, so it is swallowed and the sample is dropped.
    """
    import torch

    from .llm import clean_completion, estimate_max_tokens

    prompt = _build_prompt(tokenizer, text, cfg)
    encoded = tokenizer(prompt, return_tensors="pt").to(cfg.resolved_device())
    n_input = encoded.input_ids.shape[1]
    try:
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=estimate_max_tokens(text, headroom=cfg.headroom),
                temperature=max(temperature, 1e-3),
                top_p=cfg.top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    except Exception:  # noqa: BLE001
        return ""
    raw = tokenizer.decode(output[0][n_input:], skip_special_tokens=True)
    # The card prompt echoes its own trailer when the model restates it.
    raw = raw.split("Paraphrased text:")[-1]
    return clean_completion(raw, STOP_STRINGS)


def _gate(source: str, candidate: str, cfg: PretrainedConfig) -> List[str]:
    """Names of every content gate the candidate fails. Empty means usable.

    Same gates and same thresholds as `pipeline._gate_candidate`, minus the
    pipeline-only `unchanged` check, so a rewrite from this engine is held to
    exactly the standard a rewrite from the prompted pipeline is held to. This
    is the check that catches the failure the AuthorMist paper's own
    configuration walks into: a "successful" evasion that dropped the facts.
    """
    text = (candidate or "").strip()
    if not text:
        return ["empty"]
    n_source = len(source.split())
    if n_source == 0:
        return ["empty_source"]
    ratio = len(text.split()) / n_source

    failures: List[str] = []
    if ratio < cfg.min_length_ratio:
        failures.append("too_short")
    elif ratio > cfg.max_length_ratio:
        failures.append("too_long")
    if content_overlap(source, text) < cfg.min_content_overlap:
        failures.append("content_drift")
    failures.extend("invariant:" + name
                    for name in _invariant_failures(source, text))
    return failures


def _temperatures(cfg: PretrainedConfig) -> List[float]:
    """One temperature per candidate, spread so the samples actually differ."""
    if cfg.n_candidates <= 1:
        return [cfg.temperature]
    step = 0.5 / max(cfg.n_candidates - 1, 1)
    return [round(cfg.temperature - 0.1 + step * i, 3)
            for i in range(cfg.n_candidates)]


def humanize_pretrained(
    text: str,
    config: Optional[PretrainedConfig] = None,
    detector: Any = None,
) -> PretrainedResult:
    """Rewrite `text` with a published evasion-trained checkpoint.

    Samples `config.n_candidates` rewrites, scores each with the detector,
    discards any that fail the content gates, and returns the survivor with the
    lowest AI probability. When every candidate fails its gates the least-bad
    one is still returned but `summary["all_gates_failed"]` is set, because
    silently handing back a rewrite that dropped a number would be worse than
    saying so.

    `detector` is injectable so a caller can pass a warm `ModernDetector` or a
    stub; `None` uses the process-wide one from `humanize.pipeline`.
    """
    cfg = config or PretrainedConfig()
    source = (text or "").strip()
    started = time.perf_counter()

    if not source:
        return PretrainedResult(
            original=text or "", humanized=text or "", model=cfg.model,
            summary={"notice": NOTICE, "reason": "empty input",
                     "seconds": 0.0, "changed": False},
        )

    reason = unavailable_reason(cfg)
    if reason:
        return PretrainedResult(
            original=text, humanized=text, model=cfg.model,
            summary={"notice": NOTICE, "available": False, "reason": reason,
                     "seconds": 0.0, "changed": False},
        )

    if detector is None:
        from .pipeline import _detector

        detector = _detector()

    tokenizer, model = _load(cfg)

    candidates: List[PretrainedCandidate] = []
    for index, temperature in enumerate(_temperatures(cfg)):
        t0 = time.perf_counter()
        sample = _generate(tokenizer, model, source, temperature, cfg)
        elapsed = time.perf_counter() - t0
        if not sample:
            continue
        n_words = len(sample.split())
        candidates.append(PretrainedCandidate(
            index=index,
            text=sample,
            temperature=temperature,
            ai_probability=_score(sample, detector),
            overlap=content_overlap(source, sample),
            length_ratio=n_words / max(len(source.split()), 1),
            gate_failures=_gate(source, sample, cfg),
            seconds=elapsed,
        ))

    passing = [c for c in candidates if c.passed]
    pool = passing or candidates
    if pool:
        winner = min(pool, key=lambda c: (c.ai_probability
                                          if c.ai_probability is not None
                                          else 1.0))
        winner.chosen = True
        humanized = winner.text
    else:
        winner = None
        humanized = source

    before = _measure(source, detector)
    after = _measure(humanized, detector)
    p_before = before.get("ai_probability")
    p_after = after.get("ai_probability")

    summary = {
        "notice": NOTICE,
        "available": True,
        "n_candidates_total": cfg.n_candidates,
        "n_candidates_generated": len(candidates),
        "n_candidates_passed": len(passing),
        "all_gates_failed": bool(candidates) and not passing,
        "chosen_index": winner.index if winner else None,
        "content_overlap": winner.overlap if winner else 1.0,
        "gate_failures": list(winner.gate_failures) if winner else [],
        "prompt_style": cfg.prompt_style,
        "device": cfg.resolved_device(),
        "seconds": time.perf_counter() - started,
        "changed": humanized.strip() != source,
        "verdict_flipped": bool(
            p_before is not None and p_after is not None
            and p_before >= AI_THRESHOLD > p_after
        ),
    }

    return PretrainedResult(
        original=text,
        humanized=humanized,
        model=cfg.model,
        backend="transformers/" + cfg.resolved_device(),
        before=before,
        after=after,
        summary=summary,
        candidates=candidates,
    )
