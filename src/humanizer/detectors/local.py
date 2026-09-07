"""Local AI-text detectors. Every one of them is somebody else's model.

POLICY: THIS REPO WRITES NO DETECTION ARITHMETIC
------------------------------------------------
Detection comes from published, pretrained checkpoints downloaded from the
Hugging Face hub. What this module contributes is plumbing -- chunking,
batching, reading the label mapping off the config, checking that the weights
actually loaded -- and *measurement*. It contributes no weights, no thresholds
and no scoring formula. The one exception is `PerplexityDetector`, which is
kept out of every default, flagged `is_published_detector: false` on the wire,
and documented as a reimplementation rather than a published detector.

The hand-written `HeuristicDetector` that used to live alongside these is
gone. `humanizer.detectors.heuristic` now exports `AiStyleSignals`, which
produces named, cited style signals and no score at all.

WHAT IS AVAILABLE
-----------------
    engine        checkpoint                                        params
    ---------------------------------------------------------------------
    modern *      desklib/ai-text-detector-v1.01                      434M
    fakespot      fakespot-ai/roberta-base-ai-text-detection-v1       125M
    academic      andreas122001/roberta-academic-detector             125M
    radar         TrustSafeAI/RADAR-Vicuna-7B                         355M
    fast          MayZhou/e5-small-lora-ai-generated-detector          33M
    ---------------------------------------------------------------------
    perplexity    gpt2 + OUR arithmetic (research comparison only)    124M
    classifier    openai-community/roberta-base-openai-detector       125M

`* = default`. All five published engines are `ModernDetector` with a
different `model_name`; there is one wrapper, not five. `PUBLISHED_DETECTORS`
carries the full registry including five more checkpoints that were
benchmarked and rejected, each with the number that rejected it.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This module is **not** GPTZero and does not reproduce GPTZero.

GPTZero today is a proprietary supervised deep model. research/07 §3 quotes
their own arXiv paper: hierarchical multi-task heads (document-level {Human,
AI, Mixed} over a sub-head {Pure AI, Polished, AI Paraphrased}, plus a jointly
trained *binary* sentence head), trained on ~28.6M documents, with four tiers
of adversarial red teaming and a post-hoc ℝ³ calibration remap. "Architecture
and hyperparameters are proprietary." The weights have never been published.
We cannot run their model, and nothing here approximates their numbers.

What `ModernDetector` *does* match is the **architectural class**. Every
serious detector shipped after 2023 -- GPTZero, Originality, Pangram,
Turnitin -- is a supervised fine-tuned transformer classifier, not a perplexity
statistic. GPTZero support states plainly that the product "no longer uses
perplexity and burstiness for its AI detection" as a decision mechanism
(research/01 §1.1). Running a fine-tuned transformer classifier locally puts us
in the same *family* as the current products. It does not put us anywhere near
their training data, their thresholds, or their calibration, and a score out of
this module predicts a GPTZero verdict no better than any other third-party
detector does.

MEASURED ON THIS REPO'S OWN CORPUS -- READ BEFORE BELIEVING A NUMBER
--------------------------------------------------------------------
28 paragraphs: 14 human paragraphs pulled from `data/raw/pmc/*.txt` (biomedical
research articles) and 14 written AI-style paragraphs in four registers
(general explainer, blog, five-paragraph essay, and -- deliberately -- four in
the same biomedical-abstract register as the human side, so the benchmark is
not secretly measuring "general prose vs biomedical prose"). Scored on CPU,
Apple Silicon, torch 2.8. Every checkpoint loaded with 0 missing keys.

    engine       checkpoint                    AI mean  hum mean  FPR  TPR  pairwise
    --------------------------------------------------------------------------------
    modern *     desklib v1.01                   1.000     0.123  1/14 14/14   1.000
    fakespot     fakespot-ai roberta-base        1.000     0.213  3/14 14/14   1.000
    academic     andreas122001 academic          0.926     0.072  1/14 13/14   0.980
    fast         MayZhou e5-small-lora           0.870     0.497  8/14 14/14   0.944
    radar        TrustSafeAI RADAR               0.491     0.084  1/14  7/14   0.867
    --------------------------------------------------------------------------------
    (rejected)   andreas122001 mixed             0.661     0.084  1/14  9/14   0.944
    (rejected)   akshayvkt detect-ai-text        0.998     0.877 13/14 14/14   0.959
    (rejected)   Hello-SimpleAI chatgpt-roberta  0.463     0.090  1/14  6/14   0.811
    (rejected)   PirateXX AI-Content-Detector    0.086     0.001  0/14  1/14   0.832
    (rejected)   roberta-large-openai-detector   0.944     0.859 12/14 13/14   0.638
    --------------------------------------------------------------------------------
    perplexity   gpt2 + our arithmetic           0.920     0.797 13/14 14/14   0.740
    classifier   roberta-base-openai-detector    0.239     0.215  3/14  3/14   0.444

"pairwise" is P(a random AI paragraph outscores a random human paragraph); 0.5
is chance. "FPR@0.5" is how many of the fourteen genuine human PMC paragraphs
were labelled AI.

Read the last two rows together with the `akshayvkt` row. **Pairwise
separation and false-positive rate are different measurements and a detector
can pass one while failing the other.** `akshayvkt/detect-ai-text` ranks AI
above human at 0.959 and is still unusable, because it called 13 of 14 genuine
human academic paragraphs AI; ranking is invariant to a constant offset and
accusing people is not. Both columns are reported for every checkpoint for
that reason, and `fast` is shipped with an explicit warning rather than
quietly, because its 8/14 is the same failure in milder form.

**The classic perplexity method does not usefully separate templated AI prose
from human biomedical academic prose**, and the 2019 OpenAI classifier is
*below chance* on this set. That is why neither is a default. The perplexity
result is not a bug in the implementation and was not fixed by moving the
thresholds, because moving them would only relabel this one sample. It is the
documented failure mode: research/01 §5 records that "memorised or formulaic
human text (constitutions, boilerplate, rote writing) reads as AI to
perplexity-based methods", and Liang et al. measured 61.3% false positives on
TOEFL essays from exactly this signal. Formal biomedical prose is highly
templated, so GPT-2 finds it extremely predictable. research/01 §3.9 adds the
converse, "perplexity inversion", for modern generators.

Two checkpoints would have shipped **inverted** if their output index had been
assumed rather than resolved: `TrustSafeAI/RADAR-Vicuna-7B` and
`PirateXX/AI-Content-Detector` both put AI at index 0 while publishing only
`LABEL_0`/`LABEL_1`. See `_KNOWN_AI_INDEX` for the evidence used on each, and
`resolve_ai_index` for the general rule.

Caveats on the 1.000, which is the number most likely to be misread:

  * n = 28. There is no confidence interval here and one more borderline
    document would move every figure.
  * The AI side was written by one model in one session. A real detector
    benchmark spans generators, decoding settings and adversarial attacks;
    RAID does, and desklib's leaderboard position is evidence from RAID, not
    from this file.
  * One genuine human PMC paragraph scored 0.505 -- over the line by five
    thousandths. A detector that flags 1 in 14 real academic paragraphs is
    still a detector that will accuse someone. research/01 §5 on non-native
    writers applies to this model too; it is better than the old default, not
    safe.
  * Document scores are much steadier than sentence scores. On a 9-sentence
    human paragraph the document score was 0.015 while individual sentences
    ranged 0.06-0.87. Sentence rows exist for the heatmap; do not threshold
    them.

WHAT IS HERE
------------
`ModernDetector`
    The wrapper every published engine uses, default
    `desklib/ai-text-detector-v1.01`. Two head shapes are supported: the
    ordinary softmax-over-labels head (label index read from `config.id2label`
    via `resolve_ai_index`, never assumed) and desklib's mean-pooled
    single-logit sigmoid head, which is not one of transformers' `AutoModel`
    classes and so is rebuilt here. `check_load_integrity` refuses any load
    that left a head or encoder weight randomly initialised. Long inputs are
    chunked to the model's context and aggregated (both the token-weighted
    mean and the max over chunks are reported). Per-sentence probabilities
    come from scoring each sentence in a batch, segmented with
    `humanizer.text.Document.parse` so the indices line up with
    `/api/analyze`.

`PerplexityDetector`
    **Not a published detector.** gpt2 perplexity plus burstiness, mapped to a
    probability by arithmetic this repo invented. Research comparison only;
    see the class docstring.

`ClassifierDetector`
    A published but obsolete checkpoint,
    ``openai-community/roberta-base-openai-detector`` (research/01 §3.6).
    Superseded by `ModernDetector`; kept because its 0.444 is part of the
    evidence for the change.

`EnsembleDetector`
    A weighted mean over whichever members are available, with every member's
    own score preserved in `raw`. Published members only.

DEPENDENCIES AND LAZINESS
-------------------------
`torch` and `transformers` are in the optional ``detectors`` extra and are
**never imported at module import time** -- importing `humanizer.detectors`
must stay cheap enough for the CLI and the HTTP API to start without them.
Every import of them is inside a function body. Models load on the first
`score()` call and are cached module-level, so a process pays the load cost
once. Everything runs on CPU with `torch.no_grad()` and the model in eval mode;
no GPU is required or requested.

Weights are fetched from the Hugging Face hub on first use and are not small:
desklib ~1.74GB, RADAR ~1.4GB, gpt2 ~525MB, the OpenAI RoBERTa detector
~480MB, fakespot and academic ~500MB each, `fast` ~130MB. Only the engines a
caller actually names are ever downloaded. Every path
degrades to `available() -> False` with an actionable message rather than
raising, and `/api/detect` turns that into a 200 with an `error` string.
"""

from __future__ import annotations

import importlib.util
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..text import Document
from .base import Detector, DetectorResult

__all__ = [
    "ModernDetector",
    "PerplexityDetector",
    "ClassifierDetector",
    "EnsembleDetector",
    "backend_available",
    "clear_model_cache",
    "loaded_models",
    "resolve_ai_index",
    "resolve_head",
    "check_load_integrity",
    "DEFAULT_MODERN_MODEL",
    "PUBLISHED_DETECTORS",
    "SHIPPED_DETECTORS",
    "MODEL_HEADS",
    "DEFAULT_PERPLEXITY_MODEL",
    "FAST_PERPLEXITY_MODEL",
    "DEFAULT_CLASSIFIER_MODEL",
]

#: ~548 MB of weights. The model GPTZero's original method was built on.
DEFAULT_PERPLEXITY_MODEL = "gpt2"
#: ~353 MB, 6 layers instead of 12. Roughly 2x faster, noisier perplexities.
FAST_PERPLEXITY_MODEL = "distilgpt2"
#: ~499 MB. OpenAI's RoBERTa-base GPT-2 output detector (research/01 §3.6).
DEFAULT_CLASSIFIER_MODEL = "openai-community/roberta-base-openai-detector"
#: ~1.74 GB, 435M parameters. A DeBERTa-v3-large fine-tuned by Desklib on the
#: RAID corpus (liamdugan/raid) with a mean-pooled single-logit sigmoid head;
#: it topped the RAID detection leaderboard. This is the default engine, and
#: the module docstring records what it measured on this repo's own corpus.
DEFAULT_MODERN_MODEL = "desklib/ai-text-detector-v1.01"

#: Every published detector this repo will run, and what was measured about
#: each one on this repo's own 28-paragraph set (14 human PMC paragraphs, 14
#: written AI-style paragraphs; see the module docstring). All of them are
#: other people's trained checkpoints -- this repo contributes the wrapper and
#: the measurement, never the scoring.
#:
#: `head` is the classification head shape (see `resolve_head`). `ship` marks
#: the ones exposed as named engines by `/api/detect`; the rest are recorded
#: here because they were benchmarked and rejected, and a rejection with a
#: number attached is more useful than a silent omission.
#:
#: pairwise = P(a random AI paragraph outscores a random human paragraph),
#: 0.5 is chance. FPR = human PMC paragraphs scored >= 0.5, out of 14.
#: TPR = AI paragraphs scored >= 0.5, out of 14.
PUBLISHED_DETECTORS: Dict[str, Dict[str, Any]] = {
    "modern": {
        "model": "desklib/ai-text-detector-v1.01",
        "head": "pooled-sigmoid",
        "params": 434_000_000,
        "ship": True,
        "pairwise": 1.000,
        "fpr": 1,
        "tpr": 14,
        "note": (
            "DeBERTa-v3-large fine-tuned on RAID; led the RAID detection "
            "leaderboard. Best measured here on both separation and false "
            "positives, so it is the default."
        ),
    },
    "fakespot": {
        "model": "fakespot-ai/roberta-base-ai-text-detection-v1",
        "head": "softmax",
        "params": 125_000_000,
        "ship": True,
        "pairwise": 1.000,
        "fpr": 3,
        "tpr": 14,
        "note": (
            "Fakespot/Apollo DFT RoBERTa-base. Separates as well as the "
            "default and is 6x cheaper, but flagged 3 of 14 genuine human "
            "academic paragraphs against the default's 1, so it is a second "
            "opinion rather than the default."
        ),
    },
    "academic": {
        "model": "andreas122001/roberta-academic-detector",
        "head": "softmax",
        "params": 125_000_000,
        "ship": True,
        "pairwise": 0.980,
        "fpr": 1,
        "tpr": 13,
        "note": (
            "RoBERTa-base trained on academic abstracts, which is the domain "
            "this project's users write in. Ties the default on false "
            "positives at a fifth of the cost."
        ),
    },
    "radar": {
        "model": "TrustSafeAI/RADAR-Vicuna-7B",
        "head": "softmax",
        "params": 355_000_000,
        "ship": True,
        "pairwise": 0.867,
        "fpr": 1,
        "tpr": 7,
        "note": (
            "RoBERTa-large from IBM/TrustSafeAI, trained adversarially "
            "against a Vicuna-7B paraphraser (arXiv 2307.03838) -- the name "
            "is the paraphraser, not the detector, which is a 355M encoder "
            "that runs fine on CPU. Deliberately conservative: it missed 7 of "
            "14 AI paragraphs, but it is the only checkpoint here that was "
            "trained against paraphrase attacks, which is what a humanizer "
            "produces. Worth having precisely because it is the hardest to "
            "move."
        ),
    },
    "fast": {
        "model": "MayZhou/e5-small-lora-ai-generated-detector",
        "head": "softmax",
        "params": 33_000_000,
        "ship": True,
        "pairwise": 0.944,
        "fpr": 8,
        "tpr": 14,
        "note": (
            "e5-small + merged LoRA, trained on RAID-train. 33M parameters "
            "and ~19ms a document, cheap enough for an inner optimisation "
            "loop (research/05). READ THE FPR: it called 8 of 14 genuine "
            "human PMC paragraphs AI. Fine as a fast relative signal while "
            "editing, useless as a verdict."
        ),
    },
    # ---- benchmarked and NOT shipped. Kept for the record. -----------------
    "mixed": {
        "model": "andreas122001/roberta-mixed-detector",
        "head": "softmax",
        "params": 125_000_000,
        "ship": False,
        "pairwise": 0.944,
        "fpr": 1,
        "tpr": 9,
        "note": "Same family as `academic` and dominated by it here.",
    },
    "chatgpt-detector": {
        "model": "Hello-SimpleAI/chatgpt-detector-roberta",
        "head": "softmax",
        "params": 125_000_000,
        "ship": False,
        "pairwise": 0.811,
        "fpr": 1,
        "tpr": 6,
        "note": (
            "HC3, tuned to 2022-era ChatGPT output. Missed 8 of 14 AI "
            "paragraphs."
        ),
    },
    "piratexx": {
        "model": "PirateXX/AI-Content-Detector",
        "head": "softmax",
        "params": 125_000_000,
        "ship": False,
        "pairwise": 0.832,
        "fpr": 0,
        "tpr": 1,
        "note": (
            "Fires on almost nothing: AI mean 0.086, 1 of 14 AI paragraphs "
            "over 0.5. A detector that never says AI has no false positives "
            "and no use."
        ),
    },
    "distilbert": {
        "model": "akshayvkt/detect-ai-text",
        "head": "softmax",
        "params": 67_000_000,
        "ship": False,
        "pairwise": 0.959,
        "fpr": 13,
        "tpr": 14,
        "note": (
            "The cautionary one. Ranks well (0.959) and is still unusable: it "
            "called 13 of 14 genuine human academic paragraphs AI. Separation "
            "and safety are different measurements and this is why both are "
            "reported."
        ),
    },
    "roberta-large-openai": {
        "model": "openai-community/roberta-large-openai-detector",
        "head": "softmax",
        "params": 355_000_000,
        "ship": False,
        "pairwise": 0.638,
        "fpr": 12,
        "tpr": 13,
        "note": (
            "The large sibling of the legacy `classifier` engine. 0.638 "
            "pairwise, 12 of 14 human paragraphs flagged. Also the only "
            "checkpoint benchmarked with unexpected keys on load (an unused "
            "RoBERTa pooler), which is harmless but worth recording."
        ),
    },
}

#: Detector name -> model id, for the ones `/api/detect` exposes.
SHIPPED_DETECTORS: Dict[str, str] = {
    name: spec["model"]
    for name, spec in PUBLISHED_DETECTORS.items()
    if spec["ship"]
}

#: Model id -> head shape, consulted by `resolve_head` when the checkpoint's
#: own `config.architectures` does not settle it.
MODEL_HEADS: Dict[str, str] = {
    spec["model"]: spec["head"] for spec in PUBLISHED_DETECTORS.values()
}


# --------------------------------------------------------------------- tuning

# GPTZero's published rule of thumb was that a document perplexity above ~85
# leaned human (research/01 §1.2). We keep 85 as the *centre* of a logistic on
# log-perplexity rather than as a hard cut, because a hard cut throws away the
# distinction between a perplexity of 84 and one of 12.
#
# READ THIS BEFORE TRUSTING A NUMBER OUT OF THIS MODULE: the constants below
# are HEURISTIC AND UNCALIBRATED. They are not fitted to any labelled dataset,
# no ROC curve was computed, and no claim is made that a 0.7 here means
# anything is 70% likely to be AI. The ordering of two documents is more
# meaningful than the absolute value of either.
#
# 85 is kept deliberately even though the "MEASURED ON THIS REPO'S OWN CORPUS"
# block above shows it labels every human PMC paragraph as AI. It is the one
# threshold in this whole area with a citation behind it, and lowering it to
# roughly 25 -- where gpt2 actually centres on academic prose -- would produce a
# detector that looked calibrated on twelve paragraphs and was fitted to
# nothing. Callers who want to recentre on their own corpus and scoring model
# should pass `ppl_center=` to `PerplexityDetector`, which is honest about
# being a choice; silently baking it in would not be.
PPL_HUMAN_CENTER = 85.0
#: Centre for burstiness (SD of per-sentence perplexity). Even more arbitrary
#: than the perplexity centre -- GPTZero never published a burstiness
#: threshold, only the claim that low variance read as AI (research/01 §1.2).
#: 35 sits near the middle of what gpt2 yields on mixed human/AI prose.
BURSTINESS_HUMAN_CENTER = 35.0
#: Logistic slopes on the log-ratios. Perplexity carries more weight than
#: burstiness because it is the better-attested signal of the two.
PPL_WEIGHT = 1.60
BURSTINESS_WEIGHT = 0.70
#: Below this many scoreable tokens a sentence's perplexity is noise, so the
#: sentence inherits the document score instead of reporting a fake one. Same
#: reasoning as `api.server.MIN_WORDS_FOR_SENTENCE_RISK`; Winston AI similarly
#: ignores sentences under 60 characters (research/01 §2).
MIN_SENTENCE_TOKENS = 4
#: Every vendor imposes a floor because the statistics are meaningless below it
#: (research/01 §5: GPTZero 250 chars, Sapling 300, Turnitin 300 words). We do
#: not refuse, but we flag it in `raw` so a caller can.
MIN_RELIABLE_CHARS = 250

_SENTENCE_BATCH = 16


# ------------------------------------------------------------- model registry

# (kind, model_name) -> (tokenizer, model). Module-level so a long-lived
# process loads gpt2 once, not once per request.
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
# (kind, model_name) -> exception message, for loads that already failed. A
# missing download must not be retried on every request; the second call should
# be as fast and as quiet as `available()` implies.
_LOAD_FAILURES: Dict[Tuple[str, str], str] = {}
_CACHE_LOCK = threading.Lock()
#: Held for the whole of a model load. `from_pretrained` is not safe to run
#: concurrently in one process (transformers materialises weights through a
#: process-global meta-device patch, and a half-imported module in one thread
#: is visible to the other), and the API's sync endpoints run in a thread
#: pool, so /api/detect and /api/analyze fired together used to race here and
#: cache a broken model or remember a spurious failure for the process life.
_LOAD_LOCK = threading.RLock()


def backend_available() -> bool:
    """True when `torch` and `transformers` are importable.

    Uses `importlib.util.find_spec`, which resolves the module without
    executing it, so calling this does not drag torch into the process.
    """
    for module in ("torch", "transformers"):
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except (ImportError, ValueError):  # pragma: no cover - broken install
            return False
    return True


def loaded_models() -> List[str]:
    """Cache keys currently held in memory, for diagnostics."""
    with _CACHE_LOCK:
        return [f"{kind}:{name}" for kind, name in sorted(_MODEL_CACHE)]


def clear_model_cache() -> None:
    """Drop cached models and remembered load failures."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _LOAD_FAILURES.clear()


def _load(kind: str, model_name: str) -> Tuple[Any, Any]:
    """Serialised entry point; see `_LOAD_LOCK`."""
    with _LOAD_LOCK:
        return _load_unlocked(kind, model_name)


def _load_unlocked(kind: str, model_name: str) -> Tuple[Any, Any]:
    """Load and cache `(tokenizer, model)`, in eval mode, on CPU.

    Raises `RuntimeError` with the original message when the backend is absent
    or the weights cannot be fetched. A failure is remembered so that repeated
    calls fail immediately rather than re-attempting a download each time.
    """
    key = (kind, model_name)
    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        failure = _LOAD_FAILURES.get(key)
    if failure is not None:
        raise RuntimeError(failure)

    if not backend_available():
        message = (
            "Local detectors need the `detectors` extra: "
            "pip install -e '.[detectors]'  (torch, transformers)"
        )
        with _CACHE_LOCK:
            _LOAD_FAILURES[key] = message
        raise RuntimeError(message)

    try:
        from transformers import (  # noqa: WPS433 - deliberately lazy
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if kind == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif kind == "sequence":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown model kind {kind!r}")
        model.eval()
    except Exception as exc:  # noqa: BLE001 - offline, disk, hub, all the same
        message = (
            f"Could not load {model_name!r}: {type(exc).__name__}: {exc}. "
            "The weights are downloaded from the Hugging Face hub on first "
            "use; this fails offline or without disk space."
        )
        with _CACHE_LOCK:
            _LOAD_FAILURES[key] = message
        raise RuntimeError(message) from exc

    with _CACHE_LOCK:
        _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _known_failure(kind: str, model_name: str) -> Optional[str]:
    with _CACHE_LOCK:
        return _LOAD_FAILURES.get((kind, model_name))


# ------------------------------------------------- modern-detector head types

#: Architectures whose classification head is a mean-pooled single logit
#: squashed with a sigmoid, rather than transformers' standard
#: `...ForSequenceClassification` softmax over `num_labels`. These are not
#: `AutoModel` classes -- the checkpoint ships weights under `model.*` plus a
#: `classifier.*` of shape (1, hidden) and expects the caller to supply the
#: head. `_PooledSigmoidClassifier` below is that head.
#:
#: desklib's config.json still says `num_labels: 2` with `id2label`
#: {0: LABEL_0, 1: LABEL_1}, which is simply wrong for the checkpoint: loading
#: it with `AutoModelForSequenceClassification` silently discards the trained
#: classifier and gives you a randomly initialised two-way head. That failure
#: is *quiet* -- you get plausible-looking probabilities that mean nothing --
#: which is why the head is resolved explicitly instead of assumed.
_POOLED_SIGMOID_ARCHITECTURES = frozenset({"DesklibAIDetectionModel"})

#: Head kinds `ModernDetector` understands.
HEAD_SOFTMAX = "softmax"
HEAD_POOLED_SIGMOID = "pooled-sigmoid"


def resolve_head(architectures: Sequence[str], model_name: str = "") -> Tuple[str, str]:
    """Decide which classification head a checkpoint needs.

    Returns `(head, how)`. `architectures` is `config.architectures`. Anything
    not in `_POOLED_SIGMOID_ARCHITECTURES` is treated as an ordinary softmax
    sequence classifier, which is the right default: that is what every
    `...ForSequenceClassification` checkpoint is.
    """
    for arch in architectures or ():
        if str(arch) in _POOLED_SIGMOID_ARCHITECTURES:
            return HEAD_POOLED_SIGMOID, f"architecture:{arch}"
    declared = MODEL_HEADS.get(model_name)
    if declared:
        return declared, "known-model"
    return HEAD_SOFTMAX, "default"


def _pooled_sigmoid_class():
    """Build the mean-pool + single-logit `PreTrainedModel` subclass.

    Defined inside a function because the class statement needs `torch.nn` and
    `transformers` at definition time, and this module must import without
    either. Mirrors the reference implementation on the desklib model card:
    mean-pool the last hidden state under the attention mask, then one linear
    layer to a single logit. `from_pretrained` on this class reports zero
    missing and zero unexpected keys against the desklib checkpoint, which is
    the check that the head is the right one.
    """
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    class _PooledSigmoidClassifier(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None, **_kwargs):
            hidden = self.model(input_ids, attention_mask=attention_mask)[0]
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.classifier(pooled)

    return _PooledSigmoidClassifier


#: Substrings that mark a parameter as belonging to the *classification head*
#: rather than the encoder. If one of these turns up in `missing_keys` the
#: checkpoint did not supply that weight and `from_pretrained` has quietly
#: randomly initialised it -- which is the desklib trap: a load that succeeds,
#: emits confident-looking probabilities, and is measuring nothing.
_HEAD_KEY_MARKERS = ("classifier", "score", "logit", "out_proj", "class_head")


def check_load_integrity(
    model_name: str, missing_keys: Sequence[str], unexpected_keys: Sequence[str]
) -> Dict[str, Any]:
    """Turn `from_pretrained`'s loading info into a verdict, and raise on the
    dangerous case.

    Returns a `load_check` dict recording both key lists, so `raw` can carry
    "0 missing / 0 unexpected" as the evidence that the wrapper matched the
    checkpoint rather than as an assurance.

    The two directions are not symmetric:

      * A **missing** head key means the trained classifier was not in the
        checkpoint and torch filled it with noise. Every score after that is
        meaningless, so this raises. It is exactly what happens if you load
        `desklib/ai-text-detector-v1.01` with
        `AutoModelForSequenceClassification`, whose config advertises a
        two-way softmax head that the checkpoint does not contain.
      * **Unexpected** keys mean the checkpoint carried weights the model
        class does not use -- e.g. `openai-community/roberta-large-openai-detector`
        ships an unused RoBERTa pooler. Harmless, but recorded.

    Missing *encoder* keys are also refused: a half-initialised backbone is no
    more trustworthy than a random head.
    """
    missing = [str(k) for k in missing_keys or ()]
    unexpected = [str(k) for k in unexpected_keys or ()]
    head_missing = [
        k for k in missing if any(m in k.lower() for m in _HEAD_KEY_MARKERS)
    ]
    if missing:
        which = head_missing or missing
        kind = "classification head" if head_missing else "encoder"
        raise RuntimeError(
            f"Refusing to use {model_name!r}: {len(missing)} weight(s) were "
            f"not present in the checkpoint and have been randomly "
            f"initialised, including the {kind} ({which[:4]}). A load like "
            "this succeeds silently and then reports confident nonsense. The "
            "wrapper's head shape does not match this checkpoint; see "
            "`resolve_head`."
        )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "head_weights_loaded": True,
        "note": (
            "0 missing keys is the proof the published head was loaded rather "
            "than randomly initialised. Unexpected keys are weights the "
            "checkpoint carries that this model class does not use."
        ),
    }


def _modern_cache_name(model_name: str, head_override: Optional[str]) -> str:
    """Cache/failure key for a checkpoint, disambiguated by any head override.

    `available()` and `_load_modern` must agree on this string or a remembered
    load failure is invisible to the availability check and every request
    retries a download that is not going to work.
    """
    if head_override is None:
        return model_name
    return f"{model_name}#{head_override}"


def _load_modern(
    model_name: str, head_override: Optional[str] = None
) -> Tuple[Any, Any, str, str, Dict[str, Any]]:
    """Serialised entry point; see `_LOAD_LOCK`."""
    with _LOAD_LOCK:
        return _load_modern_unlocked(model_name, head_override)


def _load_modern_unlocked(
    model_name: str, head_override: Optional[str] = None
) -> Tuple[Any, Any, str, str, Dict[str, Any]]:
    """Load `(tokenizer, model, head, head_source, load_check)`.

    Shares `_MODEL_CACHE` and `_LOAD_FAILURES` with `_load` under the cache
    kind ``"modern"``, so `clear_model_cache`, `loaded_models` and
    `_known_failure` all cover it. The config is read first, because which
    model class to instantiate depends on what the config says the
    architecture is.

    `head_override` forces the head shape, and forces it at *load* time rather
    than at interpretation time -- the two must not disagree, because the head
    decides which model class is instantiated and therefore which weights are
    looked for. An override gets its own cache slot so the two shapes of the
    same checkpoint cannot alias.
    """
    key = ("modern", _modern_cache_name(model_name, head_override))
    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        failure = _LOAD_FAILURES.get(key)
    if failure is not None:
        raise RuntimeError(failure)

    if not backend_available():
        message = (
            "Local detectors need the `detectors` extra: "
            "pip install -e '.[detectors]'  (torch, transformers)"
        )
        with _CACHE_LOCK:
            _LOAD_FAILURES[key] = message
        raise RuntimeError(message)

    try:
        from transformers import (  # noqa: WPS433 - deliberately lazy
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(model_name)
        if head_override is not None:
            head, head_source = head_override, "explicit"
        else:
            head, head_source = resolve_head(
                getattr(config, "architectures", None) or (), model_name
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if head == HEAD_POOLED_SIGMOID:
            model, info = _pooled_sigmoid_class().from_pretrained(
                model_name, config=config, output_loading_info=True
            )
        else:
            model, info = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=config, output_loading_info=True
            )
        model.eval()
        # Raises when a head or encoder weight was randomly initialised.
        load_check = check_load_integrity(
            model_name,
            info.get("missing_keys", ()),
            info.get("unexpected_keys", ()),
        )
    except Exception as exc:  # noqa: BLE001 - offline, disk, hub, all the same
        message = (
            f"Could not load {model_name!r}: {type(exc).__name__}: {exc}. "
            "The weights are downloaded from the Hugging Face hub on first "
            "use (~1.74GB for the default model); this fails offline or "
            "without disk space."
        )
        with _CACHE_LOCK:
            _LOAD_FAILURES[key] = message
        raise RuntimeError(message) from exc

    loaded = (tokenizer, model, head, head_source, load_check)
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = loaded
    return loaded


# ------------------------------------------------------------------- numerics


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _stdev(values: Sequence[float]) -> float:
    """Sample standard deviation; 0.0 for fewer than two values.

    Sample (n-1) rather than population, matching the plain reading of
    "standard deviation of per-sentence perplexity" over a sampled document.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _finite(values: Sequence[float]) -> List[float]:
    return [v for v in values if v is not None and math.isfinite(v)]


# ------------------------------------------------------- perplexity detector


class PerplexityDetector(Detector):
    """GPTZero's original 2023 method, **reimplemented here from the paper**.

    NOT A PUBLISHED DETECTOR. Read this before quoting a number from it.

    Everything else in this module wraps somebody else's trained checkpoint
    and adds no arithmetic. This class is the exception and is kept only as a
    research comparison: `gpt2` is a published language model, but the way its
    perplexities are turned into a probability is **ours**. Specifically, this
    repo chose the logistic form, the `PPL_HUMAN_CENTER` and
    `BURSTINESS_HUMAN_CENTER` constants, the two slopes, and the decision to
    combine document perplexity with burstiness at all. GPTZero published a
    rule of thumb (~85) and a definition of burstiness; it never published a
    mapping from those to a probability, so the mapping cannot be a
    reimplementation of anything. It is an invention with a citation attached.

    It is therefore:

      * excluded from `DEFAULT_DETECTORS`, so no caller gets it by accident;
      * flagged on the wire as `is_published_detector: false`;
      * measured, and the measurement is bad -- 0.740 pairwise on this repo's
        28-paragraph set with 13 of 14 genuine human PMC paragraphs labelled
        AI. That finding is the reason the class survives at all: it is the
        evidence for why the default is a published transformer instead.

    Use it to reproduce the 2023 behaviour, including the reason the 2023
    method was abandoned. Do not use it to judge a document.

    THE METHOD ITSELF
    -----------------

    Perplexity of a sentence of tokens t_1..t_N under LM theta is

        PPL(s) = exp( -1/N * sum_i log p_theta(t_i | t_<i) )

    and burstiness is the standard deviation of PPL(s) over the document's
    sentences (research/01 §1.2). Both are computed here under `gpt2` by
    default, which is the same model class the 2023 product used.

    Sentence segmentation goes through `humanizer.text.Document.parse` so the
    sentence indices line up with `/api/analyze`'s rows and with every shape
    feature in the package. That matters for the frontend: the heatmap indexes
    both by the same sentence list.

    Perplexity is computed twice, deliberately:

      * `perplexity` scores the whole document in context windows, so a token
        late in a paragraph is conditioned on everything before it. This is the
        number to compare against the historical ~85 threshold.
      * `sentence_perplexities` scores each sentence *in isolation*, with no
        cross-sentence context. That is what makes burstiness meaningful --
        conditioning each sentence on the preceding ones would drag every
        sentence after the first toward the same low value and flatten exactly
        the variance the statistic exists to measure.
    """

    name = "perplexity"
    #: False: the scoring arithmetic is this repo's, not a published model's.
    #: `/api/detect` copies this onto every row so a UI can label it.
    is_published_detector = False

    def __init__(
        self,
        model_name: str = DEFAULT_PERPLEXITY_MODEL,
        threshold: float = 0.5,
        max_length: int = 512,
        ppl_center: float = PPL_HUMAN_CENTER,
        burstiness_center: float = BURSTINESS_HUMAN_CENTER,
    ):
        self.model_name = model_name
        self.threshold = threshold
        # gpt2's context is 1024; 512-token windows halve the quadratic
        # attention cost per window at a negligible cost in conditioning.
        self.max_length = max_length
        # Exposed so a caller can recentre on their own corpus and scoring
        # model instead of inheriting GPTZero's 2023 constant. See the module
        # docstring for what happens if you do not.
        self.ppl_center = float(ppl_center)
        self.burstiness_center = float(burstiness_center)

    # -- availability -------------------------------------------------------

    def available(self) -> bool:
        """Whether a `score()` call can be expected to work.

        This is deliberately cheap: it does not download anything. It reports
        False when torch/transformers are missing, or when a previous load of
        this model already failed (offline, no disk, hub error). It therefore
        reports True *optimistically* before the first load attempt, when the
        deps are installed but the weights are not yet on disk. The first
        `score()` call is what turns an optimistic True into a remembered
        False, which is why every caller in this package -- including
        `/api/detect` -- runs `score()` inside a try/except and reports the
        error rather than trusting this flag alone.
        """
        if not backend_available():
            return False
        return _known_failure("causal", self.model_name) is None

    def unavailable_reason(self) -> Optional[str]:
        """Human-readable reason `available()` is False, or None."""
        if not backend_available():
            return (
                "torch/transformers are not installed. "
                "pip install -e '.[detectors]'"
            )
        return _known_failure("causal", self.model_name)

    def load(self) -> None:
        """Force the model into the cache. Optional; `score()` does it too."""
        _load("causal", self.model_name)

    # -- scoring ------------------------------------------------------------

    def score(self, text: str) -> DetectorResult:
        tokenizer, model = _load("causal", self.model_name)
        doc = Document.parse(text)
        sentences = [s.text for s in doc.sentences]

        doc_ppl, doc_tokens = self._document_perplexity(tokenizer, model, doc.text)
        sent_ppl = self._sentence_perplexities(tokenizer, model, sentences)

        usable = _finite([p for p in sent_ppl if p is not None])
        burstiness = _stdev(usable)
        mean_ppl = sum(usable) / len(usable) if usable else float("nan")
        max_ppl = max(usable) if usable else float("nan")
        min_ppl = min(usable) if usable else float("nan")

        prob = self._probability(doc_ppl, burstiness)
        sentence_scores = [
            self._sentence_probability(p) if p is not None else prob for p in sent_ppl
        ]

        raw: Dict[str, Any] = {
            "method": (
                "per-sentence perplexity + burstiness, this repo's "
                "reimplementation of GPTZero's January-2023 published method"
            ),
            "model": self.model_name,
            "is_published_detector": False,
            "scoring_is_ours": (
                "gpt2 is published; the mapping from its perplexities to a "
                "probability is not. The logistic form, both centres and both "
                "slopes were chosen in this repo and fitted to nothing. "
                "Research comparison only -- excluded from the defaults."
            ),
            "perplexity": doc_ppl,
            "burstiness": burstiness,
            "mean_sentence_perplexity": mean_ppl,
            "max_sentence_perplexity": max_ppl,
            "min_sentence_perplexity": min_ppl,
            "sentence_perplexities": [
                p if p is not None else None for p in sent_ppl
            ],
            "n_sentences": len(sentences),
            "n_scored_sentences": len(usable),
            "n_tokens": doc_tokens,
            "human_threshold_reference": self.ppl_center,
            "burstiness_center": self.burstiness_center,
            "thresholds_are_calibrated": False,
            "calibration_note": (
                "Heuristic thresholds, fitted to nothing. On this repo's own "
                "PMC corpus this method separates templated AI from human "
                "academic prose at 0.583 pairwise (0.5 is chance) and labels "
                "all 12 human PMC paragraphs 'ai'. See the module docstring."
            ),
            "below_reliable_length": len(text.strip()) < MIN_RELIABLE_CHARS,
        }
        return DetectorResult(
            detector=self.name,
            ai_probability=prob,
            label="ai" if prob >= self.threshold else "human",
            confidence="low",  # uncalibrated by construction; never claim more
            sentence_scores=sentence_scores,
            raw=raw,
        )

    # -- probability mapping ------------------------------------------------

    def _probability(self, doc_ppl: float, burstiness: float) -> float:
        """Smooth logistic on log-perplexity and log-burstiness.

        Both terms are log-ratios against a centre, so the mapping is
        scale-free: halving perplexity moves the logit by the same amount
        wherever you start. Low perplexity and low burstiness both push toward
        AI, which is the direction research/01 §1.2 records.
        """
        logit = 0.0
        if doc_ppl is not None and math.isfinite(doc_ppl) and doc_ppl > 0:
            logit += PPL_WEIGHT * (math.log(self.ppl_center) - math.log(doc_ppl))
        # +1 keeps a perfectly uniform document (burstiness 0) finite instead of
        # sending the logit to +inf on what is usually a one-sentence input.
        if burstiness is not None and math.isfinite(burstiness):
            logit += BURSTINESS_WEIGHT * (
                math.log(self.burstiness_center + 1.0) - math.log(burstiness + 1.0)
            )
        return _sigmoid(logit)

    def _sentence_probability(self, ppl: float) -> float:
        """Per-sentence AI risk from that sentence's perplexity alone.

        Burstiness is a document property and has no sentence-level analogue,
        so only the perplexity term applies. research/07 §2 is worth repeating
        here: GPTZero's document verdict is *not* an aggregate of its sentence
        scores, and neither is ours -- `ai_probability` comes from the document
        perplexity and the document burstiness, not from these numbers.
        """
        if ppl is None or not math.isfinite(ppl) or ppl <= 0:
            return float("nan")
        return _sigmoid(PPL_WEIGHT * (math.log(self.ppl_center) - math.log(ppl)))

    # -- language-model plumbing -------------------------------------------

    def _document_perplexity(
        self, tokenizer, model, text: str
    ) -> Tuple[float, int]:
        """Perplexity of the whole document, in non-overlapping windows.

        Returns `(perplexity, n_tokens)`. Windows are disjoint rather than
        striding, which slightly overstates perplexity at each window's first
        few tokens; with a 512-token window on documents of a few hundred words
        that boundary effect touches one window in most cases.
        """
        import torch

        # verbose=False: the tokenizer warns that the sequence exceeds the
        # model's max length, which is exactly the case this method windows.
        ids = tokenizer(text, return_tensors=None, verbose=False)["input_ids"]
        if len(ids) < 2:
            return float("nan"), len(ids)

        total_nll = 0.0
        total_tokens = 0
        with torch.no_grad():
            for start in range(0, len(ids), self.max_length):
                window = ids[start : start + self.max_length]
                if len(window) < 2:
                    continue
                tensor = torch.tensor([window], dtype=torch.long)
                logits = model(input_ids=tensor).logits
                log_probs = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
                targets = tensor[0, 1:]
                token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                total_nll += float(-token_lp.sum())
                total_tokens += int(targets.numel())

        if total_tokens == 0:
            return float("nan"), len(ids)
        return math.exp(total_nll / total_tokens), total_tokens

    def _sentence_perplexities(
        self, tokenizer, model, sentences: Sequence[str]
    ) -> List[Optional[float]]:
        """Per-sentence perplexity, each sentence scored on its own.

        Batched with right padding and an attention mask. GPT-2 ships no pad
        token, so the tokenizer's `pad_token` is pointed at `eos_token` for the
        call; padded positions are excluded from the loss by the mask, so the
        choice of pad id cannot affect a result.
        """
        import torch

        out: List[Optional[float]] = [None] * len(sentences)
        if not sentences:
            return out

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Which sentences are long enough to be worth scoring at all.
        indices = []
        for i, sent in enumerate(sentences):
            n = len(tokenizer(sent, return_tensors=None)["input_ids"])
            if n >= MIN_SENTENCE_TOKENS:
                indices.append(i)

        with torch.no_grad():
            for start in range(0, len(indices), _SENTENCE_BATCH):
                batch_idx = indices[start : start + _SENTENCE_BATCH]
                batch = [sentences[i] for i in batch_idx]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = enc["input_ids"]
                attention = enc["attention_mask"]
                logits = model(input_ids=input_ids, attention_mask=attention).logits
                log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                targets = input_ids[:, 1:]
                mask = attention[:, 1:].to(log_probs.dtype)
                token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                nll = -(token_lp * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                for row, doc_i in enumerate(batch_idx):
                    n = float(counts[row])
                    if n <= 0:
                        continue
                    out[doc_i] = math.exp(float(nll[row]) / n)
        return out


# --------------------------------------------------- label-index resolution

# Label vocabularies for `config.id2label`. Checked as exact tokens after
# lowercasing and splitting on non-letters, so "Real" cannot match "ai" by
# accident -- which is precisely the bug this is here to avoid.
_AI_LABEL_TOKENS = frozenset(
    {"fake", "ai", "machine", "generated", "chatgpt", "gpt", "llm", "bot",
     "artificial", "synthetic", "aigenerated", "machinegenerated"}
)
_HUMAN_LABEL_TOKENS = frozenset(
    {"real", "human", "original", "student", "person", "humanwritten",
     "humangenerated"}
)

# Fallback for models whose id2label is the uninformative LABEL_0/LABEL_1
# default. `openai-community/roberta-base-openai-detector` is the one we ship,
# and its heads are (0: fake/machine, 1: real/human) per OpenAI's gpt-2-output
# detector -- but it publishes proper labels, so this is belt and braces.
_KNOWN_AI_INDEX: Dict[str, int] = {
    "openai-community/roberta-base-openai-detector": 0,
    "roberta-base-openai-detector": 0,
    "roberta-large-openai-detector": 0,
    # config.json says LABEL_0/LABEL_1; the model card is explicit that
    # "Label_0: human-written, Label_1: AI-generated". Recorded here so the
    # resolver reports "known-model" rather than "assumed" -- the fallback
    # would land on 1 anyway, but silently guessing right is not the same as
    # knowing.
    "MayZhou/e5-small-lora-ai-generated-detector": 1,
    # RADAR publishes no labels in config.json and none in its card prose.
    # Two independent lines of evidence put AI at index 0: the upstream demo
    # code takes `log_softmax(logits, -1)[:, 0].exp()` as "probability of
    # AI-generated text", and on this repo's 28-paragraph set index 0 ranks
    # the AI paragraphs above the human ones at 0.867 pairwise while index 1
    # gives the exact inverse, 0.133. The default assumption of 1 would have
    # shipped this detector backwards.
    "TrustSafeAI/RADAR-Vicuna-7B": 0,
    # Card: "Label_0 represents Fake, Label_1 represents Real". Same inversion
    # trap, same empirical confirmation (0.832 pairwise at index 0, 0.168 at
    # index 1). Benchmarked, not shipped -- see PUBLISHED_DETECTORS.
    "PirateXX/AI-Content-Detector": 0,
}


def _label_tokens(label: str) -> List[str]:
    """Split a label into lowercase alphabetic tokens."""
    token = ""
    parts: List[str] = []
    for ch in str(label).lower():
        if ch.isalpha():
            token += ch
        elif token:
            parts.append(token)
            token = ""
    if token:
        parts.append(token)
    return parts


def resolve_ai_index(id2label: Dict[int, str], model_name: str = "") -> Tuple[int, str]:
    """Work out which output index means "machine-generated".

    Returns `(index, how)` where `how` records the evidence used, so a caller
    can see whether the mapping was read off the config or guessed.

    Assuming index 1 is "AI" is the classic way to ship a detector that is
    exactly backwards. `openai-community/roberta-base-openai-detector` labels
    index 0 "Fake" and index 1 "Real", so the naive assumption inverts every
    score it produces. This reads `config.id2label` and only falls back when
    the config says nothing useful.
    """
    normalized = {int(k): _label_tokens(v) for k, v in id2label.items()}
    ai_hits = [i for i, toks in normalized.items() if _AI_LABEL_TOKENS & set(toks)]
    human_hits = [i for i, toks in normalized.items() if _HUMAN_LABEL_TOKENS & set(toks)]
    # A label matching both vocabularies ("human_vs_ai") is evidence of nothing.
    ai_only = [i for i in ai_hits if i not in human_hits]
    human_only = [i for i in human_hits if i not in ai_hits]

    if len(ai_only) == 1 and not (len(human_only) > 1):
        return ai_only[0], "id2label"
    if len(human_only) == 1 and len(normalized) == 2 and not ai_only:
        other = [i for i in normalized if i != human_only[0]]
        if len(other) == 1:
            return other[0], "id2label:complement"

    known = _KNOWN_AI_INDEX.get(model_name)
    if known is not None:
        return known, "known-model"
    # Nothing to go on. Say so loudly in `raw` rather than silently picking.
    return 1, "assumed"


# ------------------------------------------------- modern detector (default)

#: Sentences per forward pass when producing the heatmap rows. Eight keeps the
#: padded batch small on a 435M-parameter model on CPU; a 9-sentence paragraph
#: costs ~0.6s end to end on Apple Silicon.
_MODERN_SENTENCE_BATCH = 8

#: Hard cap on how many sentences get their own forward pass. Beyond this the
#: remaining sentences inherit the document probability and `raw` says so. A
#: 2,000-sentence paste should not turn one HTTP request into four minutes of
#: CPU; the document score, which is the one worth trusting, is unaffected.
MAX_SCORED_SENTENCES = 300


class ModernDetector(Detector):
    """A current fine-tuned transformer classifier. The default engine.

    Default model: `desklib/ai-text-detector-v1.01` -- DeBERTa-v3-large (435M
    parameters, ~1.74GB) fine-tuned on the RAID corpus, which led the RAID
    detection leaderboard. Pass `model_name=` to swap it; see
    `PUBLISHED_DETECTORS` for every checkpoint that was benchmarked here,
    including the ones that were rejected and why.

    WHY THIS IS THE DEFAULT AND WHAT IT DOES NOT MEAN
    -------------------------------------------------
    Every detector shipped after 2023 -- GPTZero, Originality, Pangram,
    Turnitin -- is a supervised fine-tuned transformer, not a perplexity
    statistic (research/01 §1.1, research/07 §3). Running one locally puts this
    module in the same architectural family as the current products. It does
    **not** make it GPTZero: GPTZero's model, its ~28.6M-document training
    corpus, its hierarchical {Human, AI, Mixed} / {Pure AI, Polished, AI
    Paraphrased} heads and its post-hoc calibration remap are all proprietary
    and unpublished. There is no relationship between the number this returns
    and the number GPTZero would return. Use it to measure what changed when
    you edit a document.

    HOW IT SCORES
    -------------
    Two head shapes are handled, chosen by `resolve_head` from
    `config.architectures` rather than assumed:

      * `pooled-sigmoid` (desklib): mean-pool the last hidden state under the
        attention mask, one linear layer to a single logit, sigmoid. The logit
        *is* P(AI), so there is no label index to get backwards.
      * `softmax`: an ordinary `...ForSequenceClassification` head. Which
        output index means "AI" is read from `config.id2label` through
        `resolve_ai_index`, never assumed -- `roberta-base-openai-detector`
        labels index 0 "Fake" and index 1 "Real", so the naive assumption
        inverts every score it produces.

    Long documents are split into disjoint windows of the model's context and
    every window is scored. `ai_probability` is the token-count-weighted mean
    over windows; `raw["max_chunk_probability"]` reports the loudest window,
    which is what you want when only part of a document is generated. GPTZero
    does the same thing and keeps which aggregation proprietary (research/07
    §2), so both are reported rather than picking one silently.

    `sentence_scores` is a real per-sentence forward pass for each sentence,
    segmented by `humanizer.text.Document.parse` so the indices line up with
    `/api/analyze` and with the frontend heatmap. **Treat those rows as much
    weaker than the document score.** They are the same document-trained model
    run on 15-word inputs, well under every vendor's stated reliability floor
    (research/01 §5: GPTZero 250 chars, Sapling 300, Turnitin 300 words), and
    on this repo's corpus a human paragraph scoring 0.015 at document level had
    individual sentences up at 0.87. They are a visualisation, not a verdict.
    """

    name = "modern"
    #: True: this is somebody else's trained checkpoint and this class adds no
    #: scoring arithmetic of its own -- only chunking, batching and the label
    #: or head resolution needed to read the model's output correctly.
    is_published_detector = True

    def __init__(
        self,
        model_name: str = DEFAULT_MODERN_MODEL,
        threshold: float = 0.5,
        max_length: Optional[int] = None,
        ai_index: Optional[int] = None,
        head: Optional[str] = None,
        score_sentences: bool = True,
    ):
        self.model_name = model_name
        self.threshold = threshold
        #: None means "ask the model's config". An explicit int/str is an
        #: escape hatch for a checkpoint whose config lies about itself.
        self.max_length = max_length
        self.ai_index = ai_index
        #: Forces the head shape at load time. Only needed for a checkpoint
        #: whose `config.architectures` does not identify it; getting it wrong
        #: is caught by `check_load_integrity`, not silently tolerated.
        self.head = head
        #: Turn off to pay the document forward pass only. Roughly a 5x saving
        #: on a fifteen-sentence paragraph.
        self.score_sentences = score_sentences

    # -- availability -------------------------------------------------------

    def available(self) -> bool:
        """See `PerplexityDetector.available` for the optimistic-True caveat.

        Cheap by construction: nothing is downloaded here. It returns False
        when torch/transformers are missing or when a previous load of this
        model already failed, and optimistically True before the first attempt.
        """
        if not backend_available():
            return False
        return _known_failure("modern", self._cache_name()) is None

    def _cache_name(self) -> str:
        return _modern_cache_name(self.model_name, self.head)

    def unavailable_reason(self) -> Optional[str]:
        if not backend_available():
            return (
                "torch/transformers are not installed. "
                "pip install -e '.[detectors]'"
            )
        return _known_failure("modern", self._cache_name())

    def load(self) -> None:
        """Force the model into the cache. Optional; `score()` does it too."""
        _load_modern(self.model_name, self.head)

    # -- scoring ------------------------------------------------------------

    def _window_size(self, tokenizer, model) -> int:
        """Content tokens per window, excluding the special tokens.

        `tokenizer.model_max_length` is a sentinel-sized int for tokenizers
        that do not declare a limit (desklib's is 1e19), so it is only trusted
        when it looks like a real context length.
        """
        if self.max_length is not None:
            limit = int(self.max_length)
        else:
            limit = getattr(tokenizer, "model_max_length", None)
            if not isinstance(limit, int) or limit <= 0 or limit > 100_000:
                limit = int(getattr(model.config, "max_position_embeddings", 512))
                limit = min(limit, 512)
        special = int(tokenizer.num_special_tokens_to_add(pair=False) or 0)
        return max(8, limit - special)

    def score(self, text: str) -> DetectorResult:
        tokenizer, model, head, head_source, load_check = _load_modern(
            self.model_name, self.head
        )

        id2label = dict(getattr(model.config, "id2label", {}) or {})
        ai_index: Optional[int] = None
        ai_index_source: Optional[str] = None
        if head == HEAD_SOFTMAX:
            if self.ai_index is not None:
                ai_index, ai_index_source = self.ai_index, "explicit"
            else:
                ai_index, ai_index_source = resolve_ai_index(id2label, self.model_name)

        window = self._window_size(tokenizer, model)
        # verbose=False: over-length is expected here; chunking is the point.
        ids = tokenizer(
            text, add_special_tokens=False, return_tensors=None, verbose=False
        )["input_ids"]
        if not ids:
            raise ValueError("Cannot score empty text.")

        chunks = [ids[i : i + window] for i in range(0, len(ids), window)]
        # A trailing sliver of a chunk is mostly noise; fold it into the
        # previous window when the model can still hold both.
        if len(chunks) > 1 and len(chunks[-1]) < window // 8:
            tail = chunks.pop()
            if len(chunks[-1]) + len(tail) <= window:
                chunks[-1] = chunks[-1] + tail
            else:
                chunks.append(tail)

        probs = self._probabilities(tokenizer, model, head, ai_index, chunks)
        weights = [float(len(c)) for c in chunks]
        total = sum(weights) or 1.0
        mean_p = sum(p * w for p, w in zip(probs, weights)) / total
        max_p = max(probs)

        doc = Document.parse(text)
        sentences = [s.text for s in doc.sentences]
        sentence_scores, n_scored = self._sentence_probabilities(
            tokenizer, model, head, ai_index, sentences, window, mean_p
        )

        raw: Dict[str, Any] = {
            "method": "published fine-tuned transformer sequence classifier",
            "model": self.model_name,
            "is_published_detector": True,
            "head": head,
            "head_source": head_source,
            "load_check": load_check,
            "id2label": {str(k): v for k, v in id2label.items()},
            "ai_index": ai_index,
            "ai_label": id2label.get(ai_index) if ai_index is not None else None,
            "ai_index_source": ai_index_source,
            "n_chunks": len(chunks),
            "chunk_tokens": [int(w) for w in weights],
            "chunk_probabilities": probs,
            "mean_chunk_probability": mean_p,
            "max_chunk_probability": max_p,
            "window_tokens": window,
            "n_tokens": len(ids),
            "n_sentences": len(sentences),
            "n_scored_sentences": n_scored,
            "sentence_scores_are_unreliable": True,
            "sentence_score_note": (
                "Per-sentence rows are the document-trained model run on "
                "inputs far below every vendor's reliability floor "
                "(research/01 §5). They exist for the heatmap. Do not "
                "threshold them; the document score is the verdict."
            ),
            "architecture_class": (
                "Same architectural class as GPTZero, Originality, Pangram "
                "and Turnitin today -- a supervised fine-tuned transformer "
                "classifier, not the 2023 perplexity/burstiness statistic."
            ),
            "is_gptzero": False,
            "gptzero_relationship": (
                "None. GPTZero's model, its ~28.6M-document training corpus, "
                "its hierarchical Human/AI/Mixed heads and its calibration "
                "remap are proprietary and unpublished (research/07 §3). This "
                "score has no relationship to theirs."
            ),
            "thresholds_are_calibrated": False,
            "calibration_note": (
                "Measured on this repo's own 28-paragraph set (14 human PMC "
                "paragraphs, 14 written AI-style paragraphs in four "
                "registers): pairwise separation 1.000 where 0.5 is chance, "
                "AI mean 1.000, human mean 0.123, and 1 of 14 genuine human "
                "academic paragraphs scored over 0.5 (at 0.505). n=28 with no "
                "confidence interval and one author on the AI side. The "
                "legacy 'perplexity' engine scored 0.740 pairwise with 13 of "
                "14 human paragraphs called AI on the same set."
            ),
            "below_reliable_length": len(text.strip()) < MIN_RELIABLE_CHARS,
        }
        return DetectorResult(
            detector=self.name,
            ai_probability=mean_p,
            label="ai" if mean_p >= self.threshold else "human",
            # Not "high" and never will be: nothing here is calibrated against
            # a labelled holdout, so the confidence field stays honest even
            # when the separation on 28 paragraphs looks perfect.
            confidence="medium" if not raw["below_reliable_length"] else "low",
            sentence_scores=sentence_scores,
            raw=raw,
        )

    # -- forward passes -----------------------------------------------------

    def _probabilities(
        self,
        tokenizer,
        model,
        head: str,
        ai_index: Optional[int],
        id_lists: Sequence[Sequence[int]],
    ) -> List[float]:
        """P(AI) for each pre-tokenised sequence, in one padded batch.

        Takes token ids rather than strings so that the document chunks are
        scored on exactly the ids they were split into -- decoding a chunk back
        to text and re-encoding it is not a round trip for every tokenizer, and
        a shifted chunk boundary is a silently wrong number.

        Both head shapes end in one probability per row: the sigmoid of the
        pooled single logit, or the softmax entry at `ai_index`. Padded
        positions are masked out, so the pad id cannot affect a result.
        """
        import torch

        if not id_lists:
            return []
        built = [
            list(tokenizer.build_inputs_with_special_tokens(list(ids)))
            for ids in id_lists
        ]
        width = max(len(b) for b in built)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0
        input_ids = [b + [pad_id] * (width - len(b)) for b in built]
        attention = [[1] * len(b) + [0] * (width - len(b)) for b in built]

        with torch.no_grad():
            out = model(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=torch.tensor(attention, dtype=torch.long),
            )
            logits = out if isinstance(out, torch.Tensor) else out.logits
            if head == HEAD_POOLED_SIGMOID:
                return torch.sigmoid(logits.reshape(-1).float()).tolist()
            index = 0 if ai_index is None else int(ai_index)
            return torch.softmax(logits.float(), dim=-1)[:, index].tolist()

    def _sentence_probabilities(
        self,
        tokenizer,
        model,
        head: str,
        ai_index: Optional[int],
        sentences: Sequence[str],
        window: int,
        document_probability: float,
    ) -> Tuple[List[float], int]:
        """Per-sentence P(AI), batched. Returns `(scores, n_actually_scored)`.

        Sentences too short to carry any signal, and anything past
        `MAX_SCORED_SENTENCES`, inherit the document probability instead of
        reporting a fabricated one. Same reasoning as
        `api.server.MIN_WORDS_FOR_SENTENCE_RISK`; Winston AI similarly ignores
        sentences under 60 characters (research/01 §2).
        """
        out = [document_probability] * len(sentences)
        if not sentences or not self.score_sentences:
            return out, 0

        eligible: List[int] = []
        encoded: List[List[int]] = []
        for i, sent in enumerate(sentences):
            if len(eligible) >= MAX_SCORED_SENTENCES:
                break
            ids = tokenizer(sent, add_special_tokens=False, return_tensors=None)[
                "input_ids"
            ]
            if len(ids) >= MIN_SENTENCE_TOKENS:
                eligible.append(i)
                encoded.append(list(ids[:window]))

        for start in range(0, len(eligible), _MODERN_SENTENCE_BATCH):
            stop = start + _MODERN_SENTENCE_BATCH
            probs = self._probabilities(
                tokenizer, model, head, ai_index, encoded[start:stop]
            )
            for i, p in zip(eligible[start:stop], probs):
                out[i] = p
        return out, len(eligible)


# ------------------------------------------- legacy classifier detector


class ClassifierDetector(Detector):
    """An open fine-tuned sequence classifier for machine-generated text.

    Default: `openai-community/roberta-base-openai-detector`, OpenAI's
    RoBERTa-base fine-tuned on 1.5B GPT-2 outputs versus WebText (research/01
    §3.6). Two things about it are worth stating up front:

      * It reached ~95% *in-distribution* accuracy, i.e. on GPT-2 output. RAID
        (research/01 §3.6, §6) measured the GPT-2 RoBERTa detectors as the
        worst-generalising family tested, 44.8% across generators, and found
        they can *improve* under paraphrase because paraphrased text drifts
        back toward their training distribution. It is a real fine-tuned
        classifier, which is the point, but it is a 2019 one.
      * Its label order is (0: Fake, 1: Real). See `resolve_ai_index`.

    Long inputs are split into non-overlapping windows of the model's context
    and every window is scored. `ai_probability` is the token-count-weighted
    mean over windows, and `raw["max_chunk_probability"]` reports the loudest
    window, which is what you want when only part of a document is generated.
    """

    name = "classifier"
    #: True: a published checkpoint with a thin wrapper. Legacy, but not ours.
    is_published_detector = True

    def __init__(
        self,
        model_name: str = DEFAULT_CLASSIFIER_MODEL,
        threshold: float = 0.5,
        max_length: Optional[int] = None,
        ai_index: Optional[int] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        # None means "ask the model's config". An explicit int is an escape
        # hatch for a model whose labels are unreadable.
        self.max_length = max_length
        self.ai_index = ai_index

    def available(self) -> bool:
        """See `PerplexityDetector.available` for the optimistic-True caveat."""
        if not backend_available():
            return False
        return _known_failure("sequence", self.model_name) is None

    def unavailable_reason(self) -> Optional[str]:
        if not backend_available():
            return (
                "torch/transformers are not installed. "
                "pip install -e '.[detectors]'"
            )
        return _known_failure("sequence", self.model_name)

    def load(self) -> None:
        _load("sequence", self.model_name)

    def _window_size(self, tokenizer, model) -> int:
        """Content tokens per window, excluding the two special tokens.

        `tokenizer.model_max_length` is a sentinel-sized int (a very large
        number) for tokenizers that do not declare a limit, so it is only
        trusted when it looks like a real context length.
        """
        if self.max_length is not None:
            limit = self.max_length
        else:
            limit = getattr(tokenizer, "model_max_length", None)
            if not isinstance(limit, int) or limit <= 0 or limit > 100_000:
                limit = int(getattr(model.config, "max_position_embeddings", 512))
                # RoBERTa reserves two position slots for padding offsets.
                limit = min(limit, 512)
        special = int(tokenizer.num_special_tokens_to_add(pair=False) or 0)
        return max(8, limit - special)

    def score(self, text: str) -> DetectorResult:
        import torch

        tokenizer, model = _load("sequence", self.model_name)
        id2label = dict(getattr(model.config, "id2label", {}) or {})
        if self.ai_index is not None:
            ai_index, source = self.ai_index, "explicit"
        else:
            ai_index, source = resolve_ai_index(id2label, self.model_name)

        window = self._window_size(tokenizer, model)
        # verbose=False: over-length is expected here; chunking is the point.
        ids = tokenizer(
            text, add_special_tokens=False, return_tensors=None, verbose=False
        )["input_ids"]
        if not ids:
            raise ValueError("Cannot score empty text.")

        chunks = [ids[i : i + window] for i in range(0, len(ids), window)]
        # A trailing sliver of a chunk is mostly noise; fold it into the
        # previous window when the model can still hold both.
        if len(chunks) > 1 and len(chunks[-1]) < window // 8:
            tail = chunks.pop()
            if len(chunks[-1]) + len(tail) <= window:
                chunks[-1] = chunks[-1] + tail
            else:
                chunks.append(tail)

        probs: List[float] = []
        weights: List[float] = []
        with torch.no_grad():
            for chunk in chunks:
                built = tokenizer.build_inputs_with_special_tokens(list(chunk))
                tensor = torch.tensor([built], dtype=torch.long)
                logits = model(input_ids=tensor).logits
                p = torch.softmax(logits[0].float(), dim=-1)
                probs.append(float(p[ai_index]))
                weights.append(float(len(chunk)))

        total = sum(weights) or 1.0
        mean_p = sum(p * w for p, w in zip(probs, weights)) / total
        max_p = max(probs)

        # Sentence scores would need a per-sentence forward pass each; that is
        # a different cost profile, so the classifier reports document level
        # only and the frontend heatmap uses the perplexity detector's rows.
        raw: Dict[str, Any] = {
            "method": "fine-tuned sequence classifier (2019, legacy)",
            "model": self.model_name,
            "is_published_detector": True,
            "superseded_by": (
                "`modern`. On this repo's 28-paragraph set this checkpoint "
                "scored 0.444 pairwise -- below the 0.5 chance line -- "
                "against 1.000 for the default. Kept for comparison only."
            ),
            "id2label": {str(k): v for k, v in id2label.items()},
            "ai_index": ai_index,
            "ai_label": id2label.get(ai_index),
            "ai_index_source": source,
            "n_chunks": len(chunks),
            "chunk_tokens": [int(w) for w in weights],
            "chunk_probabilities": probs,
            "mean_chunk_probability": mean_p,
            "max_chunk_probability": max_p,
            "window_tokens": window,
            "n_tokens": len(ids),
            "in_distribution_caveat": (
                "Trained on GPT-2 output; RAID measured this detector family "
                "at 44.8% across modern generators (research/01 §3.6)."
            ),
            "below_reliable_length": len(text.strip()) < MIN_RELIABLE_CHARS,
        }
        return DetectorResult(
            detector=self.name,
            ai_probability=mean_p,
            label="ai" if mean_p >= self.threshold else "human",
            confidence="low",
            sentence_scores=[],
            raw=raw,
        )


# --------------------------------------------------------- ensemble detector

#: Default member weights. Hand-set, not fitted. Both members are published
#: checkpoints; the ensemble contributes an average, not a score of its own.
#:
#: The hand-written `heuristic` member was removed: this repo does not write
#: AI-detection arithmetic any more, and an invented signal inside an average
#: is still an invented signal. See `detectors.heuristic`, which is now
#: explanatory style signals with no aggregate.
DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "modern": 2.0,
    "perplexity": 1.0,
    "classifier": 1.0,
}


class EnsembleDetector(Detector):
    """Weighted mean of whichever member detectors actually ran.

    Members that are unavailable or that raise are dropped from the average and
    recorded in `raw["members"]` with their error, so a partial ensemble still
    returns a number and the caller can see what it is missing. If nothing ran,
    `available()` is False and `score()` raises rather than inventing 0.5.

    `sentence_scores` comes from the members that produced one of the right
    length, weighted the same way. In the default configuration that is the
    perplexity detector alone, because the modern member is constructed with
    `score_sentences=False` to keep the ensemble affordable.
    """

    name = "ensemble"

    def __init__(
        self,
        members: Optional[Sequence[Detector]] = None,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ):
        if members is None:
            # Published checkpoints only. `ModernDetector` carries most of the
            # weight because it is the only member that measured better than
            # chance on this repo's academic corpus; the other two are here so
            # `raw["members"]` shows the comparison, not because averaging
            # them in improves anything.
            members = [
                ModernDetector(score_sentences=False),
                PerplexityDetector(),
                ClassifierDetector(),
            ]
        self.members = list(members)
        self.weights = dict(weights) if weights is not None else dict(
            DEFAULT_ENSEMBLE_WEIGHTS
        )
        self.threshold = threshold

    def weight_for(self, detector: Detector) -> float:
        return float(self.weights.get(detector.name, 1.0))

    def available(self) -> bool:
        return any(m.available() for m in self.members)

    def unavailable_reason(self) -> Optional[str]:
        if self.available():
            return None
        reasons = []
        for m in self.members:
            reason = getattr(m, "unavailable_reason", lambda: None)()
            reasons.append(f"{m.name}: {reason or 'unavailable'}")
        return "no ensemble member is available -- " + "; ".join(reasons)

    def score(self, text: str) -> DetectorResult:
        member_raw: Dict[str, Any] = {}
        scored: List[Tuple[Detector, DetectorResult]] = []

        for member in self.members:
            if not member.available():
                reason = getattr(member, "unavailable_reason", lambda: None)()
                member_raw[member.name] = {
                    "available": False,
                    "error": reason or "unavailable",
                    "weight": self.weight_for(member),
                }
                continue
            try:
                result = member.score(text)
            except Exception as exc:  # noqa: BLE001 - one member must not kill it
                member_raw[member.name] = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "weight": self.weight_for(member),
                }
                continue
            scored.append((member, result))
            member_raw[member.name] = {
                "available": True,
                "ai_probability": result.ai_probability,
                "label": result.label,
                "weight": self.weight_for(member),
                "raw": result.raw,
            }

        if not scored:
            raise RuntimeError(
                self.unavailable_reason() or "no ensemble member produced a score"
            )

        total = sum(self.weight_for(m) for m, _ in scored) or 1.0
        prob = sum(self.weight_for(m) * r.ai_probability for m, r in scored) / total

        return DetectorResult(
            detector=self.name,
            ai_probability=prob,
            label="ai" if prob >= self.threshold else "human",
            confidence="low",
            sentence_scores=self._blend_sentence_scores(scored),
            raw={
                "method": "weighted mean of member ai_probability",
                "members": member_raw,
                "n_scored": len(scored),
                "total_weight": total,
                "weights": dict(self.weights),
            },
        )

    def _blend_sentence_scores(
        self, scored: Sequence[Tuple[Detector, DetectorResult]]
    ) -> List[float]:
        """Weighted mean of member sentence scores, over the modal length.

        Members disagree on length only if they segmented differently, which
        would be a bug, but a length mismatch must not raise here. The modal
        length wins and mismatched members are ignored.
        """
        lengths = [len(r.sentence_scores) for _, r in scored if r.sentence_scores]
        if not lengths:
            return []
        target = max(set(lengths), key=lengths.count)
        contributors = [
            (self.weight_for(m), r.sentence_scores)
            for m, r in scored
            if len(r.sentence_scores) == target
        ]
        total = sum(w for w, _ in contributors) or 1.0
        out: List[float] = []
        for i in range(target):
            acc = 0.0
            for weight, scores in contributors:
                value = scores[i]
                if value is None or not math.isfinite(value):
                    value = 0.5
                acc += weight * value
            out.append(acc / total)
        return out
