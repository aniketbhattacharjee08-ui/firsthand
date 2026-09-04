"""Local, open-weight AI-text detectors: the classic perplexity/burstiness
method and an open fine-tuned classifier.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This module is **not** GPTZero and does not reproduce GPTZero.

What it *does* reproduce is GPTZero's **original, published, January-2023
method**: score every sentence for perplexity under a GPT-2-class language
model, and take the standard deviation of those per-sentence perplexities as
"burstiness". research/01 §1.2 records that definition, and GPTZero's own
historical rule of thumb that a document perplexity above roughly 85 read as
human. `PerplexityDetector` below implements exactly that, with the same GPT-2
family of scoring model, so the numbers it produces are the same *kind* of
numbers the 2023 product showed.

GPTZero **today** is something else entirely, and nothing here approximates it.
research/07 §3 quotes GPTZero's own arXiv paper: a proprietary supervised deep
model with hierarchical multi-task heads (document-level {Human, AI, Mixed}
over a sub-head {Pure AI, Polished, AI Paraphrased}, plus a separate binary
sentence head), trained on ~28.6M documents, with four tiers of adversarial red
teaming and a post-hoc calibration remap. GPTZero support states plainly that
the product "no longer uses perplexity and burstiness for its AI detection" as
a decision mechanism (research/01 §1.1). Architecture and weights are
proprietary and unpublished. We cannot run it locally and do not claim to.

Practically, that means:

  * The probabilities here are **not** calibrated against GPTZero's, or against
    anything else. See `PPL_HUMAN_CENTER` for how arbitrary the mapping is.
  * research/01 §3.6 and the RAID results in research/13 both show that
    GPT-2-era statistical detectors and the GPT-2 RoBERTa detectors generalise
    *worst* of all detector families against modern LLM output (RAID: 44.8% for
    RoBERTa-L GPT2, and note the "perplexity inversion" finding in research/01
    §3.9 that modern LLM output can have *higher* perplexity than human text).
    Treat a "human" verdict from this module as worth nothing at all against a
    2026 commercial detector.
  * Use this to *measure* what changed when you edit a document, not to certify
    that a document will pass anything.

MEASURED ON THIS REPO'S OWN CORPUS -- READ BEFORE BELIEVING A NUMBER
--------------------------------------------------------------------
Four templated LLM-style paragraphs versus twelve human paragraphs from
`data/raw/pmc/*.txt` (biomedical research articles), scored with `gpt2`:

    document perplexity   AI-templated  mean 24.5   range 11.1 - 33.2
                          PMC human     mean 30.6   range 17.3 - 63.9
    burstiness (SD)       AI-templated  mean 36.3
                          PMC human     mean 56.0

    pairwise separation (P[AI scores lower perplexity than a human doc])
                          0.583   -- 0.5 is chance

**The classic perplexity method does not usefully separate templated AI prose
from human biomedical academic prose.** It is barely above chance, and the
burstiness signal is *inverted* from the theory: the human academic paragraphs
were burstier than the AI ones, not flatter. With `PPL_HUMAN_CENTER` at the
historical 85, all twelve genuine human PMC paragraphs are labelled "ai".

This is not a bug in the implementation and it was not fixed by moving the
thresholds, because moving them would only relabel this one sample. It is the
documented failure mode: research/01 §5 records that "memorised or formulaic
human text (constitutions, boilerplate, rote writing) reads as AI to
perplexity-based methods", and Liang et al. measured 61.3% false positives on
TOEFL essays from exactly this signal (research/01 §5, §3.5 -- Ghostbuster's
perplexity-only baseline scored 13.2% there). Formal biomedical prose is highly
templated, so GPT-2 finds it extremely predictable. research/01 §3.9 adds the
converse, "perplexity inversion", for modern generators.

Where it *does* work is the textbook demo case: a human narrative paragraph in
the same run scored perplexity 52.8 and burstiness 94.5 against the AI mean of
24.5, and is the only text in the set the mapping calls human. So the method
reproduces the 2023 behaviour faithfully, including the reason the 2023 method
was abandoned.

`ClassifierDetector` did better on the same sample -- pairwise separation 0.812,
and 0.0002 on the human narrative -- but it is noisy in the direction that
matters, scoring two of the twelve genuine PMC paragraphs above 0.999.

Both numbers come from tiny samples with no confidence interval. They are here
to stop anyone reading a probability out of this module as a fact.

WHAT IS HERE
------------
`PerplexityDetector`
    Per-sentence perplexity under a causal LM (default ``gpt2``), document
    perplexity, burstiness = SD of the per-sentence values, and mean/max
    sentence perplexity. Maps to a probability with a smooth logistic on
    log-perplexity and log-burstiness rather than the historical hard cut.

`ClassifierDetector`
    An open fine-tuned sequence classifier, default
    ``openai-community/roberta-base-openai-detector`` -- the RoBERTa-base model
    OpenAI released with the GPT-2 output dataset (research/01 §3.6). Long
    inputs are chunked to the model's context and aggregated.

`EnsembleDetector`
    A weighted combination of whichever members are available, with every
    member's own score preserved in `raw`.

DEPENDENCIES AND LAZINESS
-------------------------
`torch` and `transformers` are in the optional ``detectors`` extra and are
**never imported at module import time** -- importing `humanizer.detectors`
must stay cheap enough for the CLI and the HTTP API to start without them.
Every import of them is inside a function body. Models load on the first
`score()` call and are cached module-level, so a process pays the load cost
once. Everything runs on CPU with `torch.no_grad()` and the model in eval mode;
no GPU is required or requested.
"""

from __future__ import annotations

import importlib.util
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..text import Document
from .base import Detector, DetectorResult

__all__ = [
    "PerplexityDetector",
    "ClassifierDetector",
    "EnsembleDetector",
    "backend_available",
    "clear_model_cache",
    "loaded_models",
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
    """GPTZero's original method: per-sentence perplexity plus burstiness.

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
            "method": "per-sentence perplexity + burstiness (GPTZero, Jan 2023)",
            "model": self.model_name,
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


# ------------------------------------------------------- classifier detector

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
            "method": "fine-tuned sequence classifier",
            "model": self.model_name,
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

#: Default member weights. Hand-set, not fitted -- the classifier gets the
#: larger share because research/01 §7.2 finds fine-tuned encoders are the
#: harder family to move, and the heuristic gets a small share because it is
#: the only member that sees discourse-level features at all.
DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "perplexity": 1.0,
    "classifier": 1.5,
    "heuristic": 0.5,
}


class EnsembleDetector(Detector):
    """Weighted mean of whichever member detectors actually ran.

    Members that are unavailable or that raise are dropped from the average and
    recorded in `raw["members"]` with their error, so a partial ensemble still
    returns a number and the caller can see what it is missing. If nothing ran,
    `available()` is False and `score()` raises rather than inventing 0.5.

    `sentence_scores` comes from the members that produced one of the right
    length, weighted the same way. In the default configuration that is the
    perplexity detector alone.
    """

    name = "ensemble"

    def __init__(
        self,
        members: Optional[Sequence[Detector]] = None,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ):
        if members is None:
            from .heuristic import HeuristicDetector

            members = [
                PerplexityDetector(),
                ClassifierDetector(),
                HeuristicDetector(),
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
