# Evasion Code Mining — What the Published Attacks Actually Do, and What Survives Contact With a 2026 Detector

*Compiled 2026-09-04. This report reads source, not abstracts. Every repository named below
was cloned and read; every number in Part B was measured on this machine against
`desklib/ai-text-detector-v1.01` through `humanizer.detectors.local.ModernDetector`.
Scratch, scripts and raw results are in `.codemine/`.*

`research/02` surveyed this ecosystem and `research/03` surveyed the papers. This report goes
one level down — into the loops, the constraints, the prompt strings and the accept criteria —
and then does the thing none of the papers do: **runs the published artefacts against a detector
that is three years newer than the ones they were evaluated on.**

---

## 0. The three findings, stated up front

**Finding 1 — the detector-generation gap is real and it is most of the story.**
The NeurIPS 2025 Adversarial Paraphrasing repository ships its actual generated outputs. Those
texts cut MAGE / RADAR / OpenAI-RoBERTa TPR@1%FPR by an average of 87.88% in the paper.
Scored on desklib they are **flagged 59–60 times out of 60**, mean P(AI) 0.95–0.98, and separate
from domain-matched human text at pairwise 0.976–0.991. The attack that a NeurIPS paper calls
"universal" transfers to desklib **not at all**. See §B1.

**Finding 2 — desklib's decision boundary is locally movable by surface edits, but the human
region is not reachable by them.** The one prototype built here (HMGC + RAFT reimplemented
against desklib itself) flips **9 of 9** paragraphs, 1.0000 → 0.3916 mean, in 66 s/sample —
against 0 of 9 for every method previously recorded in this repository. But the flipped texts score
**0.9998 on fakespot, 0.9984 on the academic detector and 0.9060 on the fast detector — 0 of 9
flips on every held-out detector**, and they are visibly damaged English. Putting those detectors
*inside* the objective does not help: they then move by 0.0002 and 0.0022 while desklib still falls
to 0.41. What was produced is an adversarial example against one checkpoint, not a humanised
document. See §B4.

**Finding 3 — the one lever with a plausible route out is not a decoding trick.** HIP
(arXiv 2605.19516) is widely described, including in `research/03` §B.18 and `research/13` §4, as
"generate from base models." Reading the code, that is wrong: HIP **LoRA-fine-tunes** a base
checkpoint on (AI-paraphrase → original human passage) pairs with completion-only loss and a
plain-text tag format instead of a chat template. The naive prompt-only version of it was
prototyped here and does nothing (§B3). Their own conclusion is that HIP works because it
"reintroduces a human-text training signal, not because it discovers a special decoding trick."

---

## Part A — What each project actually does, mechanically

### A.1 SICO — `ColinLu50/Evade-GPT-Detector` (TMLR 2024, arXiv 2305.10847)

SICO optimises **the in-context exemplars**, not the output. `sico/incontext_trainer.py`:

* **Step 1, feature extraction.** `prompt_extract_feature` shows the LLM a block of AI texts
  labelled `P1` and a block of human texts labelled `P2` (the labels are deliberately
  anonymous — the code comment says "we refer P1 to AI and P2 to Human to eliminate LLM's
  bias") and asks:

  ```
  Compare and give the key distinct feature (specifically vocabulary, sentence
  structure) of P2's writings (do not show specific examples):
  ```

  Five candidate feature strings are sampled at temperature 0.9, each is dropped into the final
  prompt, and the one with the highest utility score is kept. So the *style description in the
  prompt is itself hill-climbed against the detector.*

* **Step 2, exemplar construction.** The chosen feature text is used to paraphrase the AI
  outputs into initial in-context outputs `y_ic`.

* **Step 3, substitution optimisation.** `_optimize_ic_outputs` alternates `edit_type='sent'`
  and `edit_type='word'` and calls `context_text_optimization` (`sico/context_optimizer.py`),
  a greedy one-step hill-climb: enumerate every single-position swap, score them all with the
  proxy detector, sort by score delta, apply the best non-overlapping ops, keep the result only
  if the whole-text score improved.

* **The objective** (`evaluate_prompt`) is exactly SICO's Equation 1:

  ```python
  ai_score_list, label_list = self.detector(eval_task_outputs)
  U_score = 1 - np.mean(ai_score_list)
  ```

  computed on *freshly generated* eval outputs, so the prompt is scored by what it makes the
  model write, not by the exemplars themselves.

**Direct answer to the question "does SICO optimise the content of its exemplars or only their
phrasing?" — only the phrasing.** The task inputs `x_ic` are never touched. `WordNetCandGenerator`
produces POS-matched WordNet synonyms filtered by a RoBERTa MLM confidence threshold;
`ParaLLMGenerator.generate_para_dict` sentence-tokenises the exemplar, merges any sentence under
5 words into its neighbour, and asks the LLM for `para_num` paraphrases *of that sentence*. Both
edit operators are meaning-preserving substitutions on the exemplar surface.

The one place SICO does change the content plan is architectural rather than optimised: **SICO-Gen
writes the answer from scratch** given the optimised prompt, while SICO-Para paraphrases an
existing AI answer. SICO-Gen scores better (GPTZero AUC 0.184 vs 0.290 on the writing task,
`research/13` §2). That is weak but real evidence that letting the model re-plan beats letting it
restate — it is the only controlled Gen-vs-Para comparison in the literature I found.

Proxy detectors supported: `chatdetect`, `gpt2detect`, `gptzero`, `openai`, `detectgpt`, `logrank`
— **all 2023 vintage**.

### A.2 HMGC — `zhouying20/HMGC` (COLING 2024, arXiv 2404.01907)

A TextAttack recipe. `attack/recipes/rspmu_mlm_dualir.py` is the whole method in 30 lines:

```python
goal_function = functools.partial(TargetedClassification, target_class=target_cls)
constraints = [
    RepeatModification(), StopwordModification(), PartOfSpeech(),
    MaxWordsPerturbed(max_percent=0.4),
    UniversalSentenceEncoder(threshold=0.75, metric="cosine",
                             compare_against_original=True, window_size=50),
]
transformation = WordSwapMaskedLM(batch_size=128)
search_method = GreedyDualWIR(alpha=0.2, wir_method="gradient")
```

The novelty is `GreedyDualWIR` (`attack/methods/search_methods/greedy_dual_wir.py`). Two
min-max-normalised rankings are blended:

```python
index_scores = (1 - self.alpha) * victim_scores + self.alpha * llm_scores
index_order  = np.array(indices_to_order)[(-index_scores).argsort()]
```

* `victim_scores` with `wir_method="gradient"` is one backward pass through the victim,
  `np.linalg.norm(agg_grad, ord=1)` over the word's subtokens.
* `llm_scores` is the *change in Pythia perplexity when the word is deleted*, with the sign
  chosen by the Chinese comment in the source: deleting a word and perplexity going **up** means
  the word carries meaning, so it gets a **low** swap priority; perplexity going **down** means
  the word is filler and is safe to swap.

`alpha=0.2` — the detector gradient dominates 4:1. Search is plain greedy: take positions in
importance order, keep the best swap only if it raises the goal score, early-exit on
`GoalFunctionResultStatus.SUCCEEDED` and then re-pick among the successful candidates by
`similarity_score`.

One transformation in the repo *does* change the content plan, and it is worth naming because
it is the only such operator in any of these codebases:
`attack/methods/transformations/mlm_sentence_suggestion.py` deletes a random non-first sentence,
splices in one of 16 hardcoded discourse openers —

```python
self.mlm_prompts = ["We know that", "We can say that", "It is about that",
    "It is known that", "The fact is that", "The answer is that",
    "In accordance with", "As per", "In light of", "In line with",
    "On the basis of", "In agreement with", "Based on", "As evidenced by",
    "Per the data", "Per the information"]
```

— and has `facebook/bart-base` infill the rest, with a `difflib` guard that the generated span
is not a copy of the prefix or suffix, and a **fallback that deletes the sentence outright** if
no candidate passes. This is a genuine "remove a proposition and invent a different one"
operator. It is used only as a *baseline* (`attack/baseline/flint_mlm_sentence_trans.py`,
without any detector in the loop), never inside the main recipe.

### A.3 RAFT — `jameslwang/raft` (EMNLP 2024, arXiv 2410.03658)

`experiment.py::raft()`, `--mask_pct` default **0.10**. Two-stage:

1. **Which words to replace.** With a next-token proxy (`gpt2`/`opt-2.7b`/`neo-2.7b`/`gpt-j-6b`)
   it scores every token by its probability under the proxy, sorts ascending, then
   `ranks_filter.pop()`s from the end — i.e. **it replaces the *most predictable* tokens**, the
   ones that make the passage look machine-written. With a detector proxy
   (`roberta-{base,large}-detector`) it instead brackets each word with `=` markers and asks the
   detector for that span's `llm_likelihood`.
2. **What to replace them with.** GPT-3.5, one call per word:

   ```python
   query = f"""Given some input paragraph, we have highlighted a word using
   brackets. List {top_k} alternative words for it that ensure grammar
   correctness and semantic fluency. Output words only.\n{paragraph}"""
   ```

   Candidates are POS-filtered (`nltk.pos_tag` equality) and accepted only if
   `self.detector_model.crit(paragraph_temp) <= min_score` — the *target* detector, queried
   directly.

RAFT is therefore the same family as HMGC with a cheaper importance signal and an LLM instead of
an MLM for candidates. Its "realistic" claim rests on the 10% perturbation cap.

### A.4 Adversarial Paraphrasing — `chengez/Adversarial-Paraphrasing` (NeurIPS 2025, arXiv 2506.07001)

The cleanest code of the set, and the one a sibling agent is reimplementing, so here is exactly
what `utils.py::Paraphraser.paraphrase()` does that a naive version would not.

* **Guide model is not the model under attack.** `meta-llama/Meta-Llama-3-8B-Instruct`, generation
  config `top_p=0.99, top_k=50, temperature=0.6`, `use_cache=False` at load and manual
  `past_key_values` threading in the loop.
* **The system prompt is fixed and content-neutral:**

  ```
  You are a rephraser. Given any input text, you are supposed to rephrase the text
  without changing its meaning and content, while maintaining the text quality.
  Also, it is important for you to output a rephrased text that has a different
  style from the input text. You can not just make a few changes to the input text.
  The input text is given below. Print your rephrased output text between tags
  <TAG> and </TAG>.
  ```

  The prompt is then **pre-seeded with `"<TAG> "`** (`inputs = [inp + "<TAG> " for inp in inputs]`)
  so the model cannot preamble.
* **Candidate set per step**: top-p mask at `top_p=0.9` (the CLI default, overriding the 0.99 in
  the constructor), then truncated to the first `top_k=50` survivors.
* **How the detector signal enters the logits — it does not.** In the default path there is no
  mixing at all. The partial continuation is decoded and scored, and the token is chosen by
  `argmin` of the detector score, with the language model contributing *only* through the top-p
  mask:

  ```python
  adv_scores.extend(self.classifier.get_scores(next_words))  # only the partial
                                                             # paraphrased output
  if option == 2:
      adv_scores = -np.array(prob_scores[i]) + float(adversarial)*np.array(adv_scores)
  if deterministic:
      idx = np.argmin(adv_scores)
  ```

  `option=2` is the only setting that forms a weighted sum of LM probability and detector score;
  `option=1` is a probability floor; `option=None` (the default) is **pure detector argmin over
  the top-p set**. `deterministic=0` samples proportionally to the negated, shifted, normalised
  detector scores instead.
* **Scoring granularity.** `next_words = tokenizer.batch_decode(cat([generated_tokens[i], tok]))`
  — the classifier sees *only the generated prefix so far*, never the source text and never the
  prompt. Early in generation it is scoring three-word fragments.
* **Cost.** One classifier forward pass per candidate token per step, `O(top_k)` = up to 50
  detector calls per generated token.

Guidance classifiers implemented: `MAGEDetector` (`yaful/MAGE`), `OpenAIRoberta`
(`openai-community/roberta-{base,large}-openai-detector`), `RADAR`
(`TrustSafeAI/RADAR-Vicuna-7B`). **Every one is 2023.** The paper's central claim is transfer:

> *"Most, if not all, high-performing detectors tend to converge toward a common distribution
> that characterizes human-authored text."* — repo README

§B1 measures that claim against desklib. It does not hold.

**Things a reimplementation against desklib should copy and probably would not think of:**
pre-seeding the assistant turn with the open tag; scoring the bare continuation rather than
prompt+continuation; the `weirds` post-filter that strips `"Note: I rephrased"` trailers; and the
fact that `option=None` uses no LM probability whatsoever in the choice — quality is carried
entirely by the top-p mask, which is why `top_p` is the real quality knob.

### A.5 MASH — `githigher/MASH` (Findings of ACL 2026, arXiv 2601.08564)

Four stages, all cheap. The reusable parts are stages 1 and 4.

* **Stage 1** (`stage1_data_construction.py`) streams a HF dataset and keeps only rows the
  *target detector* confidently calls human (`HUMAN_LABEL_ID = 0`, `argmax == 0`). The target
  style is therefore defined by the detector, not by a corpus choice.
* **Stage 2** is a BART-base with a **style label embedding fused into the encoder output** and a
  dual loss: reconstruct the AI text under the AI label, produce the human text under the human
  label —
  `loss = lambda_recon * CE(f(src, AI), src) + lambda_style * CE(f(src, HUMAN), trg)`.
* **Stage 3** DPO with `prompt = original AI text`, `chosen = the real human text`,
  `rejected = the SFT output that failed to evade`. Note `chosen` is *genuine human prose*, not a
  better paraphrase — this is the retrieval/splice idea expressed as a training target.
* **Stage 4** (`stage4_refinement.py`) is the inference-time trick and needs no weights:
  split into sentences (merging any under 5 words), ask an LLM for improved alternatives with an
  explicit licence to restructure —

  ```
  3. You may split, merge, or remove sentences for clarity
  ```

  — rank sentences by **descending perplexity** (`order = np.argsort(ppls)[::-1]`), then greedily
  accept each replacement **only if the detector still calls the whole document human**:

  ```python
  trial = list(structure); trial[idx] = proposals[idx]
  if is_human(flatten(trial)):
      structure[idx] = proposals[idx]
  ```

  That accept-if-still-human gate is the single most portable idea in the paper: it converts any
  rewriter into a monotone-safe rewriter at the cost of one detector call per edit.

### A.6 DIPPER — `martiansideofthemoon/ai-detection-paraphrases` (NeurIPS 2023)

The control-code scheme, verbatim from `dipper_paraphrases/paraphrase_minimal.py`:

```python
lex_code   = int(100 - lex_diversity)     # both must be multiples of 20, 0..100
order_code = int(100 - order_diversity)
for sent_idx in range(0, len(sentences), sent_interval):     # sent_interval = 3
    curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
    final_input_text = f"lexical = {lex_code}, order = {order_code}"
    if prefix:
        final_input_text += f" {prefix}"
    final_input_text += f" <sent> {curr_sent_window} </sent>"
    outputs = self.model.generate(**final_input, **kwargs)
    prefix += " " + outputs[0]        # discourse context carried forward
    output_text += " " + outputs[0]
```

Three things matter and are usually lost in retellings: the codes are *inverted* (100 − diversity)
because the model was trained on similarity not diversity; the window is **three sentences**, not
one, which is what makes reordering possible at all; and the growing `prefix` means the model sees
its own earlier output as discourse context, so paragraph-level cohesion is maintained without
paragraph-level attention over the source. `order_diversity` is the only published knob in any of
these systems that explicitly permutes propositions. The checkpoint is T5-XXL 11B and needs ~40 GB;
the local `SamSJackson/paraphrase-dipper-no-ctx` distillation drops the `prefix` mechanism and
therefore drops exactly the discourse behaviour that makes DIPPER interesting.

### A.7 ToBlend — `muyuhuatang/token_ensemble` (arXiv 2402.11167)

Relevant because "escape any single model's distribution" is precisely this repository's diagnosed
failure mode. The code is notebooks, not a library, and the mechanism is cruder than the paper
suggests.

* **Model choice per step is uniform i.i.d. random.** No round-robin, no entropy gating:

  ```python
  def select_model_random_loaded(model_info_list):
      model_info = random.choice(model_info_list)
      return model_info[0], model_info[1], model_info[2]
  ```

* **Tokenizer reconciliation is a string round-trip.** Every step decodes to text, re-encodes the
  entire running prompt with the newly chosen model's tokenizer, and re-runs prefill. No KV cache
  reuse, no logit-space blending, no shared vocabulary. `word_tokenize` + `" ".join` destroys
  original whitespace, which is why the paper's own samples contain mangled URLs.
* **Blending is at generation time**, autoregressive, `k ∈ {1..5, random, whole-sentence}` tokens
  per model. Quality trick: generate `k+3` tokens, keep `k`.
* Two disjoint model sets of four: classic (`gpt2-xl`, `opt-2.7b`, `gpt-neo-2.7B`, `gpt-j-6B`) and
  advanced (`llama-2-7b`, `phi-2`, `Mistral-7B-v0.3`, `gemma-7b`).
* **Detectors: Likelihood / Rank / LogRank / Entropy (2019), GPT-2 Detector (2019), RADAR (2023),
  Fast-DetectGPT (2023). No commercial detector was ever tested, and no fine-tuned modern
  classifier.** Best case Fast-DetectGPT XSum 0.9845 → 0.3968 AUROC. Two detectors got *better*
  under the attack: Entropy 0.5300 → 0.8121 and RADAR 0.8904 → 0.9423.

Because ToBlend was only ever evaluated against likelihood-family detectors, its result is close
to uninformative about a fine-tuned classifier: mixing generators perturbs a *perplexity* profile
while leaving register and discourse untouched. A sibling agent measured token blending against
desklib directly and got 0 of 9 flips, which is what this reading predicts.

### A.8 StealthRL — `suraj-ranganath/StealthRL` (arXiv 2602.08934)

GRPO + LoRA (r=32, α=32) on `Qwen/Qwen3-4B-Instruct-2507`, 10k filtered MAGE samples, KL β=0.05,
rollout temp 1.0 / top-p 0.9. Reward `R = α·R_det + β·R_sem + γ·R_ppl` with canonical
`α=1.0, β=0.1, γ=0.0` (perplexity term disabled — "KL penalty handles fluency"):

* `R_det = 1.0 - detector_prob`, ensemble `0.6 × roberta-large-openai-detector + 0.4 ×
  Fast-DetectGPT(gpt-neo-2.7B)`, **z-scored against a running mean/std and clipped to [−3, 3]**
* `R_sem` = E5 cosine rescaled `min(1, score/(1-0.90))`
* degenerate guards: empty output → −1.0; `len < 10 or len > 3× original` → −0.5

**Two things are reusable without the weights.** First, the prompt is deliberately plain and
identical at train and test time (`stealthrl/tinker/env.py`):

```
Please paraphrase the following text while maintaining its meaning and ensuring
it reads naturally:

{ai_text}

Paraphrased text:
```

Second, the repo ships `n_candidates` + `rerank_detector="roberta"` — **best-of-N with a detector
reranker** — which is model-agnostic. Their own no-training baseline is raw Qwen3-4B-Instruct with
that prompt at temperature 0.9, top-p 0.95. Note the README states the paper's headline numbers are
*single-shot*, not best-of-N.

### A.9 AuthorMist — `authormist/authormist-originality` (arXiv 2503.08716)

No training code exists; one of the six trained models was released
(Qwen2.5-3B-Instruct full finetune, 12.4 GB fp32, already in this machine's HF cache). Reward is a
single term with **no semantic or fluency component at all**:

```
R(X, Y) = 1 − (1/k) Σ_j P_{d_j}(Y)     with k = 1 in practice
```

Advantage is mean-subtracted only, no std normalisation. lr 5e-5, KL β=0.001, 714 steps, 16 GPU-hours.

**The paper's inference recipe is missing from the model card and it matters:** 512-token chunking
plus **best-of-8 with the live detector in the loop** — *"For each chunk, the system generates 8
candidate paraphrases and selects the one with the lowest average detector score."* The published
ASRs (92.33% on GPTZero, etc.) are best-of-8-with-detector-access numbers, not single-sample
numbers. Anyone comparing AuthorMist to a single-shot method is comparing the wrong things.
(Caveat found while checking: the abstract's "78.6% to 96.2%" range matches no table in the body.)

### A.10 ESPERANTO — `navid-aub/Esperanto-Dataset` (arXiv 2409.14285)

Data only; the method is not implemented anywhere. Two corrections to how this is usually
described, including in `research/13`:

* **It is not a chain.** Ten *parallel* single round trips through Google Translate:
  PT, ES, FR, IT, ZH, NL, DA, JA, DE, KO. Japanese is the single most influential pivot; the
  ablation says six or more languages are needed before the effect appears.
* **The "combiner model" is not a model.** It is a two-stage deterministic greedy algorithm:
  align each source sentence to its most similar sentence in each of the ten back-translations,
  then compute word error rate against the source and **keep the maximum-WER candidate** — i.e.
  deliberately select the most textually divergent rewrite for every sentence. No threshold, no
  perplexity filter, no detector in the loop.

Their own numbers already contain the warning: average TPR@1%FPR falls 0.718 → 0.044 on ZeroGPT
and 0.943 → 0.708 on GPTZero, but **only 0.958 → 0.931 on Pangram**, the strongest detector they
tested. §B2 measures the two-pivot version against desklib and finds the Pangram end of that
spectrum, not the ZeroGPT end.

### A.11 HIP / "Base Models Look Human To AI Detectors" — `YixuanEvenXu/humanization-by-iterative-paraphrasing` (arXiv 2605.19516)

This is the most consequential correction in the report, because `research/03` §B.18 and
`research/13` §4 both read as though base-model *generation* is the lever.

The headline observation is real and is quoted verbatim in §1 of the paper:

> *"For Llama-3-8B with human prefixes, GPTZero and Pangram assign human probabilities of 96.7%
> and 98.8% to the base model's continuations, respectively, and 30.3% and 17.1% to the instruct
> model's continuations."*

Measured by feeding a first-sentence prefix as a continuation prompt and scoring **only the
generated continuation**.

But **HIP itself is a fine-tune, not a prompt.** There is no few-shot template. What is
"base-like" is the *format*: paraphrasing is cast as plain-text completion with structural tags
and no chat template (`src/hip/prompts.py`):

```python
def build_source_target_prompt(source_text: str) -> str:
    return ("<source_text>\n"
            f"{str(source_text or '').strip()}\n"
            "</source_text>\n\n"
            "<target_text>\n")

def build_prompt_completion(source_text: str, target_text: str) -> dict[str, str]:
    return {"prompt": build_source_target_prompt(source_text),
            "completion": f"{str(target_text or '').strip()}\n</target_text>"}
```

Training is TRL `SFTTrainer` with `completion_only_loss=True`, LoRA r=128 α=128 on all seven
attention+MLP projections, 1 epoch, lr 5e-5 cosine. **The training pair direction is the whole
idea: the input is a GPT-5-nano paraphrase of a human passage, and the target is the original
human passage.** The model is optimised to reconstruct human prose from AI prose.

Inference is a fixed `for round_index in range(1, num_rounds + 1)` with `num_rounds = 10` —
**no adaptive stopping criterion, no detector threshold check** — at temperature 1.0, top-p 0.95,
`max_new_tokens=512`, re-feeding the previous output into the source slot each round.

Two caveats worth carrying forward. First, the widely quoted "~100% human by round 10" is a single
qualitative trajectory in Figure 5, not an aggregate; the paper contains no tables, and the honest
aggregate statement is directional (human-probability rises monotonically with rounds while the
GPT-5-nano semantic score declines). Second, the authors' own interpretation is that HIP works
because it "reintroduces a human-text training signal, not because it discovers a special decoding
trick" — and they report that plain FineWeb-Edu continued pretraining of an *instruct* model also
recovers human-likeness. Their released adapters are Qwen3 and Llama-3 family; none apply to
Qwen2.5.

Their baselines are also worth knowing: **StealthRL and simple paraphrase both "fail to achieve a
meaningful trade-off" against commercial detectors.** Only DIPPER remains non-trivial.

---

## Part B — Measurements

All scored with `ModernDetector(score_sentences=False)`, default
`desklib/ai-text-detector-v1.01`, CPU, this machine. Scripts in `.codemine/`.

### B1. The published Adversarial Paraphrasing outputs, scored on desklib

`chengez/Adversarial-Paraphrasing` ships its generations as HuggingFace arrow datasets
(`outputs/guided_generations_mage/{noadv,adv/{mage,radar,robbase,roblarge}}`, 2000 rows each) and
the human MAGE texts it plotted ROC curves against
(`outputs/human_text_scores/data-mage_model-mage.json`, 2000 rows). Same domain, same source
distribution, no re-generation needed. n = 60 per set, ≥40 words, deterministic sample.
Script: `.codemine/score_advpara2.py`; raw scores `.codemine/advpara_desklib_raw.json`.

| set | mean P(AI) | median | flagged ≥0.5 | pairwise vs human |
|---|---:|---:|---:|---:|
| **human MAGE text** | **0.2271** | **0.1021** | **12/60** | — |
| plain Llama-3-8B paraphrase (`noadv`) | 0.9967 | 0.9999 | 60/60 | 0.997 |
| adversarial, guided by MAGE | 0.9530 | 0.9987 | 60/60 | 0.976 |
| adversarial, guided by RADAR | 0.9809 | 0.9994 | 60/60 | 0.991 |
| adversarial, guided by roberta-base | 0.9743 | 0.9995 | 60/60 | 0.988 |
| adversarial, guided by roberta-large | 0.9708 | 0.9986 | 59/60 | 0.986 |

The human control matters: desklib is not simply saturated on this domain. It puts genuine human
MAGE text at a median of 0.10 while putting every adversarially-guided output at a median of 0.999.
Guiding against MAGE buys **0.0437 of mean probability** and **zero verdicts** on desklib.

This is a direct falsification of the paper's stated key insight for this detector pair. Detectors
do not converge on a common human distribution; MAGE, RADAR and the OpenAI RoBERTas converge on
*each other*, and desklib is somewhere else. Since desklib's separation on this repo's own corpus
is 1.000 pairwise (`detectors/local.py` docstring), the gap is not an artefact of a weak yardstick.

**Read this as the general caveat on Part A.** SICO (2023 proxies), HMGC (2023 surrogates), RAFT
(GPT-2-era proxies), ToBlend (2019–2023 likelihood detectors) and ESPERANTO (six 2019–2024 open
detectors) were all evaluated in the same era. The one paper that tested a modern commercial
detector, ESPERANTO on Pangram, got 0.958 → 0.931.

### B2. Back-translation

Helsinki-NLP opus-mt Marian models, beam 4, sentence-wise, on the 9 `.llmbench` AI paragraphs.
Script `.codemine/screen_backtranslate.py`, results `.codemine/res_backtranslate.json`.

| chain | flips | mean P(AI) | content overlap | s/sample |
|---|---:|---|---:|---:|
| EN→DE→EN | **0/9** | 1.0000 → 0.9999 | 0.77 | 24.7 |
| EN→ZH→EN | **0/9** | 1.0000 → 0.9838 | 0.27 | 6.3 |
| EN→DE→EN→ZH→EN | **0/9** | 1.0000 → 0.9348 | 0.23 | 9.9 |

The probability moves only as far as the content is destroyed — the 0.9348 costs 77% of the
source's content words. Consistent with a sibling agent's independent sweep (11 configurations,
Marian/NLLB/M2M100/Qwen, 0/9 everywhere) and with ESPERANTO's own Pangram row. Caveat on
faithfulness: ESPERANTO is ten *parallel* pivots with WER-max sentence selection and says six or
more pivots are required, so this tests a weaker variant than the paper. Given that their strongest
detector moved 2.7 points, running the full ten-pivot version is not where the next hour should go.

### B3. Base model vs instruct, few-shot pair prompt (the naive HIP)

Before reading the HIP source I prototyped what everyone assumes HIP is: `mlx-community/Qwen2.5-3B-4bit`
(base) given few-shot `### Draft / ### Original` pairs, where the "Draft" side is a
Qwen2.5-3B-Instruct restatement of a real PMC paragraph and the "Original" side is that paragraph
— MASH's stage-1/2 parallel data construction done in-context instead of by SFT.
Scripts `.codemine/make_pairs.py`, `.codemine/basefewshot.py`.

On AI paragraphs, best-of-4: **P(AI) 0.999993 → 0.999991**. The format works — the base model
produces coherent, on-topic prose and correctly emits the next `### Draft` header afterwards — and
the output is indistinguishable to desklib from the input.

The intended controlling measurement (human text through the model) is inconclusive and I am
reporting it as such: with this prompt the model **copied 3 of 5 inputs verbatim** (content overlap
1.00), so the comparison is contaminated.

| mode | mean P(AI), 5 human PMC paragraphs | verbatim copies |
|---|---|---:|
| base | 0.1156 → 0.1114 | 3/5 |
| instruct, same raw completion prompt | 0.1156 → 0.1061 | 3/5 |

On the two that actually changed, both checkpoints *kept* human text human (0.0808 → 0.0601 base,
0.0808 → 0.0594 instruct). That is worth flagging against this repo's recorded control (human text
through the model 0.176 → 0.661): the earlier number was measured through a chat-style "rewrite
this paragraph" instruction, and a separate run here reproduces that effect — asking
Qwen2.5-3B-Instruct to "rewrite as a polished, professional summary" moved the same PMC paragraphs
from 0.0067–0.5048 to 0.0557–0.9982 (`.codemine/demo_pairs.json`). **The tentative reading is that
the damage comes from the instruction, which asks for the polished register detectors are trained
on, rather than from the checkpoint.** This needs a clean re-run with a copy-suppressing prompt
before anyone acts on it.

Post-hoc, §A.11 explains the null: HIP does not prompt a base model, it fine-tunes one, and its
authors say explicitly that the effect comes from reintroducing a human-text training signal.

### B4. The prototype — detector-guided greedy substitution (`.codemine/gsub.py`)

**Mechanism.** HMGC's recipe and RAFT's accept criterion, reimplemented against desklib itself
rather than against a 2023 surrogate, plus MASH stage-4's accept-only-if-the-detector-agrees gate.

* Word importance: HMGC `wir_method="gradient"` — one backward pass, L1 norm of
  ∂P(AI)/∂embedding per token, max-pooled onto whitespace words. desklib ships as a bespoke
  `_PooledSigmoidClassifier` whose `forward()` takes only `input_ids`, so the prototype reaches
  through to the inner `AutoModel`, injects `inputs_embeds`, and redoes the mean-pool + linear head
  by hand. Positions are re-ranked against the *current* text at the start of each pass.
* Candidates: HMGC's `WordSwapMaskedLM` — mask the word, take `roberta-base` top-k fills,
  case-matched, alphabetic only.
* Constraints: stopwords and sub-3-character words never modified, no position counted twice,
  `MaxWordsPerturbed(0.4)`, a content-overlap floor standing in for HMGC's USE 0.75, and an
  optional distilgpt2 perplexity ceiling (not in any of the source papers as an *accept* gate —
  added because of the damage described below).
* Search: greedy in importance order, keep the best candidate that lowers P(AI) and clears the
  constraints, up to `--passes` sweeps, early-exit as soon as the verdict flips.

Chosen over the alternatives because every technique this repo has tried is open-loop generation
and therefore inherits Qwen's output distribution, whereas substitution generates nothing and takes
desklib's own score as the objective. It is also mechanically disjoint from the sibling agents'
work: no decoding-time logit intervention (`humanize/guided.py`), no pretrained checkpoint
(`humanize/pretrained.py`).

**Result, 9 `.llmbench` AI paragraphs, `--passes 4 --top-k 30 --no-pos --min-overlap 0.6`**
(`.codemine/res_gsub_main.json`):

| sample | words | P(AI) before | after | flip | words changed | overlap | s |
|---|---:|---:|---:|:--:|---:|---:|---:|
| ai-explainer-1 | 148 | 0.999993 | 0.4450 | ✓ | 23 (15.5%) | 0.73 | 111 |
| ai-explainer-2 | 132 | 0.999991 | 0.4712 | ✓ | 20 (15.2%) | 0.80 | 58 |
| ai-explainer-3 | 124 | 0.999988 | 0.2541 | ✓ | 23 (18.5%) | 0.74 | 59 |
| ai-gen-microplastics | 162 | 0.999958 | 0.3486 | ✓ | 14 (8.6%) | 0.87 | 56 |
| ai-gen-antibiotic | 128 | 0.999934 | 0.3033 | ✓ | 13 (10.2%) | 0.85 | 36 |
| ai-gen-urbanheat | 161 | 0.999985 | 0.4538 | ✓ | 14 (8.7%) | 0.85 | 53 |
| ai-gen-sleep | 149 | 0.999974 | 0.4141 | ✓ | 14 (9.4%) | 0.84 | 53 |
| ai-gen-supplychain | 153 | 0.999993 | 0.4023 | ✓ | 20 (13.1%) | 0.79 | 90 |
| ai-gen-teaching | 155 | 0.999985 | 0.4316 | ✓ | 20 (12.9%) | 0.80 | 79 |

**Verdict flips 9/9. Mean P(AI) 1.0000 → 0.3916. Mean content overlap 0.808. Mean 66 s/sample,
~590 detector forward passes per sample.** Prior methods in this repo: 0/9, probability drop
≤0.00013.

**And now the two reasons not to ship it.**

**(a) Zero transfer.** The same nine outputs, scored on the other published checkpoints in
`humanizer.detectors.local` (`.codemine/transfer.py`, `.codemine/res_transfer.json`):

| detector | flips | mean P(AI) before → after |
|---|---:|---|
| `desklib/ai-text-detector-v1.01` (guided) | **9/9** | 1.0000 → 0.3916 |
| `fakespot-ai/roberta-base-ai-text-detection-v1` | **0/9** | 0.9999 → 0.9998 |
| `andreas122001/roberta-academic-detector` | **0/9** | 0.9998 → 0.9984 |
| `MayZhou/e5-small-lora-ai-generated-detector` | **0/9** | 0.9448 → 0.9060 |

Three detectors do not move at all. This is the mirror image of §B1 and it makes the same point
from the other side: what the optimiser found is a direction in desklib's input space, not a step
toward the human region. `research/12`'s obfuscation-detection line and Soto et al.
(arXiv 2505.14608, "attacks retain stylistic fingerprints") both predict exactly this.

**(b) Visible quality damage that content-overlap 0.81 does not see.** Actual output, ai-explainer-1:

> "In now rapidly evolving digital workforce, artificial intelligence has become an increasingly
> important tool for organizations of all disciplines. It is important to note that the rollout of
> these technologies requires careful consideration of both the risks and the challenges necessary.
> … organizations must ensure that appropriate safeguards are in **site** to protect user
> **performance** and maintain trust. Certainly, the ethical foundations of these systems **cannot
> be guaranteed**. … **Their, success in this approach rest on a commitment to increased,
> performance, and consistent improvement.**"

`cannot be overlooked` → `cannot be guaranteed` inverts the claim. `in place` → `in site` is a
homophone corruption. On ai-gen-antibiotic the *topic phrase* was replaced: "Antibiotic resistance"
→ "Multiple resistance". This is DAMAGE's L3 tier and `research/03`'s "tortured phrases" artifact
reproduced exactly, and it is the reason HMGC carries a Universal Sentence Encoder floor of 0.75
and RAFT caps perturbation at 10%.

**The constrained variant collapses.** Re-run with HMGC's POS constraint restored, the overlap
floor at 0.85 and a distilgpt2 perplexity ceiling of 1.05× the source
(`--passes 4 --top-k 30 --min-overlap 0.85 --ppl-guard --max-ppl-ratio 1.05`), the first paragraph
took 1,925 detector calls over 224 s and accepted **one** substitution:
`0.999993 → 0.999992`, no flip. The run was stopped there; the shape of the answer was already
clear and the remaining eight paragraphs would have cost another half hour to say the same thing.
An intermediate setting (POS off, overlap 0.75, perplexity ceiling 1.20) was also abandoned: with
the fluency gate rejecting most swaps the search stops early-exiting and grinds all four passes,
roughly half an hour per paragraph.

So the honest characterisation of the quality trade-off is **binary rather than a smooth Pareto
curve**: with a real fluency constraint the attack finds almost nothing, and without one it finds a
flip every time by breaking the English. That is consistent with what the source papers chose —
HMGC caps perturbation at 40% and leans on a USE floor, RAFT caps it at 10% — and it is why both of
them report probability and AUC deltas rather than verdict flips.

**(c) Ensembling the objective does not fix (a).** The obvious repair is to guide against several
detectors at once. `gsub.py --guides` scores a candidate by the **max** over members — a swap is
kept only if it helps whichever detector is currently loudest — with the position ranking still
taken from desklib's gradient. Guided against desklib + fakespot + academic simultaneously,
`--passes 3 --top-k 12 --min-overlap 0.7`, first four paragraphs
(`.codemine/res_gsub_ens.json`, `.codemine/res_transfer_ens.json`):

| detector | in guidance? | flips | mean P(AI) before → after |
|---|---|---:|---|
| `desklib/ai-text-detector-v1.01` | yes | **4/4** | 1.0000 → 0.4126 |
| `fakespot-ai/roberta-base-ai-text-detection-v1` | **yes** | **0/4** | 0.9999 → **0.9997** |
| `andreas122001/roberta-academic-detector` | **yes** | **0/4** | 0.9998 → **0.9976** |
| `MayZhou/e5-small-lora-ai-generated-detector` | held out | 0/4 | 0.9488 → 0.9320 |
| `TrustSafeAI/RADAR-Vicuna-7B` | held out | 0/3 | 0.8198 → 0.6913 |

Read the second and third rows carefully: those detectors were **inside the objective**, the search
spent 21–32 substitutions per paragraph explicitly trying to lower them, and they moved by 0.0002
and 0.0022. This is not a transfer failure, it is an *attackability* failure — at a 20% word budget,
`fakespot-ai/roberta-base` is simply not movable by MLM word substitution, while desklib is movable
to 0.41. Mean overlap 0.749, 119 s/sample, up to 2,235 detector forward passes per paragraph.

The practical consequence: **the 9/9 in the headline table is a property of desklib, not a property
of the method.** Any future number produced by optimising against one checkpoint should be read the
same way.

---

## Part C — So which mechanism escapes the model's own distribution?

The question as posed was: every local model's output sits inside the detector's AI distribution,
which technique escapes that, and how, mechanically? The measured answer is uncomfortable.

**Nothing in the open literature escapes it against a 2026 fine-tuned classifier.** Ranked by what
was actually measured here or measured by the papers against a strong detector:

1. **Detector-guided edits against the *deployed* detector move the verdict** — 9/9 here — but they
   move it by finding an adversarial direction, not by becoming human-like. 0/9 transfer, and
   ensembling the objective does not help: two detectors placed *inside* the objective moved by
   0.0002 and 0.0022 (§B4c).
2. **Detector-guided edits against a *different* detector do nothing** — the NeurIPS 2025 outputs
   score 0.95–0.98 on desklib. The universality claim is false across a detector generation.
3. **Surface distribution attacks (back-translation, token blending) do nothing**, because the
   thing being perturbed — token-level predictability — is not what a fine-tuned DeBERTa keys on.
   ESPERANTO's own Pangram row (0.958 → 0.931) said this in 2024.
4. **Prompt-level and base-model-prompt approaches do nothing** (0/9 here, and this repo's prior
   0/9 with 8 personas and 6 candidates).
5. **The only approaches with evidence against a strong, current detector all change what the model
   *is*, not what it emits**: HIP's LoRA on (AI → human) pairs, MASH's DPO toward genuine human
   text as `chosen`, AuthorMist's GRPO against live commercial detectors. All three are training,
   and all three define the target by real human prose rather than by a paraphrase objective.

On the sibling agent's hypothesis that the signal is "content plan and register, not the English
token distribution": the gsub result is a partial counter-example and the qualification is worth
stating precisely. **Twenty-three single-word swaps flipped desklib, so the decision boundary
is reachable by pure surface edits.** What the surface edits cannot do is reach the *human* region
— the same text stays at 0.9998 for three other detectors. So the accurate statement is not
"surface edits cannot move a modern detector" but **"surface edits can move one detector's
boundary and cannot move the underlying machine-ness signal, which is what all of them share."**
Register and content plan are a plausible name for that shared signal; ARB (arXiv 2607.29539,
recall 91.2%/93.5% on direct LLM text vs 30.8%/15.1% on LLM-rewritten *human* text) and Beemo
(expert human editing evades, LLM editing does not) are the strongest independent support for it.

A note on what the code search did *not* turn up, because it constrains the design space: **no
published attack optimises the content plan against a detector.** Every objective-driven operator
in SICO, HMGC, RAFT, MASH and Adversarial Paraphrasing is a meaning-preserving substitution or a
constrained paraphrase. The only content-plan operators anywhere in these codebases are DIPPER's
`order_diversity` control code (trained, not optimised, and open-loop), HMGC's
`MLMSentenceSuggestion` (a *baseline*, no detector in the loop, and its fallback is deleting the
sentence) and MASH stage 4's "you may split, merge, or remove sentences" licence to the polishing
LLM (detector-gated, but the LLM is asked for fluency, not for a different plan). That gap is
either an oversight in the literature or the most obvious unexplored direction, and this repo is
better placed than most to find out which.

---

## Part D — Ranked recommendation

**1. Fine-tune the paraphraser on (AI → real human) pairs, HIP-style. Highest expected value.**
This is the only mechanism with evidence against a current commercial detector, and the code is
public and simple. The recipe transfers to what is on this machine: build pairs by having
Qwen2.5-3B-Instruct paraphrase real human paragraphs (PMC and any wider corpus from
`research/14`), then LoRA-fine-tune `Qwen2.5-3B` **base** with the input being the AI paraphrase
and the target being the original human passage, in HIP's plain-tag completion format with
completion-only loss. Their hyperparameters are in §A.11. Two adaptations worth making: HIP's ten
rounds have no stopping criterion, so add MASH's accept-if-still-human gate and stop at the first
flip; and evaluate against a *held-out* detector from the start, because §B4 shows how easy it is
to fool one and only one.

Feasibility on this machine is better than it looks. `mlx_lm.lora` already implements LoRA plus
prompt masking (`mlx_lm/tuner/datasets.py`, `mask_prompt`), so the training loop is off the shelf.
The one gap is that `CompletionsDataset` masks by re-rendering the pair through
`tokenizer.apply_chat_template`, which is precisely the chat framing HIP avoids; the plain-tag
`<source_text>/<target_text>` format needs about twenty lines of custom dataset class returning
`(tokens, offset)` where `offset` is the length of the tag-prefixed prompt. Data construction is
one batch job with the instruct model. Budget an overnight run, not an afternoon.

**2. Make held-out-detector flips the reported metric immediately — but do not expect ensembling
the objective to rescue a method.** §B1 and §B4a are the same failure measured from both
directions, and §B4c shows the obvious repair does not work: with fakespot and academic *inside*
the objective they still moved 0.0002 and 0.0022 while desklib fell to 0.41. So the ensemble is a
**measuring instrument**, not a fix. Concretely: report flips on `fast` and RADAR held out, and
treat any single-detector flip rate as uninterpretable. Applies directly to the sibling's
`guided.py` — an ensemble guide there will cost 3× the decode and, on this evidence, buy a better
number for exactly one checkpoint.

**3. Steal MASH stage 4 verbatim as a safety layer.** Sentence-level LLM polish, sentences ordered
by descending perplexity, each replacement accepted only if the detector still calls the document
human. It is ~60 lines, needs no training, and it converts any rewriter into one that cannot make
things worse. It is also the natural place to put the sentence-restructuring licence
("you may split, merge, or remove sentences") that is the closest thing to a content-plan operator
in the published code.

**4. Test the one unexplored direction: detector-guided edits to the content plan.** Nobody has
done this. The operator set already exists in the code read here — HMGC's `MLMSentenceSuggestion`
(delete a proposition, infill a different one behind a discourse opener, or just delete it) and
DIPPER's `order_diversity` (permute propositions). Neither has ever been run with a detector as the
objective. Concretely: take a paragraph, enumerate single-sentence deletions, reorderings and
LLM-generated *additions* of a new specific (a number, a concession, a named case), score each with
the detector ensemble, greedily accept. This is exactly `gsub.py`'s search loop with sentences
instead of words, so it is a small delta on working code. If §C's shared-signal hypothesis is right,
this is where it should show up.

**5. Fix the base-vs-instruct control before anyone builds on it.** §B3 is contaminated by verbatim
copying, and the tentative reading — that the 0.176 → 0.661 damage comes from asking for a polished
register rather than from the checkpoint — would change how prompts are written throughout the
product if it holds. One clean run with a copy-suppressing prompt and n ≥ 14 settles it.

**Do not build:** back-translation in any form (§B2; a sibling measured it net *harmful*, pushing
3 of 5 genuine human paragraphs over the threshold), token blending (§A.7 — evaluated only against
likelihood detectors, 0/9 here), character/homoglyph tricks (`research/13` §4.1), or a shipped
version of `gsub.py` (§B4b — it produces damaged, semantically inverted English and fools exactly
one detector).

**And the reporting standard this report argues for:** a verdict flip against a single detector is
not evidence of humanisation. It is evidence of an adversarial example. From here on, quote
held-out-detector flips and a semantic check stronger than content overlap, or quote nothing.

---

## Sources

**Code read for this report** (cloned to `.codemine/repos/`):
ColinLu50/Evade-GPT-Detector (SICO) ·
zhouying20/HMGC ·
jameslwang/raft ·
chengez/Adversarial-Paraphrasing ·
githigher/MASH ·
martiansideofthemoon/ai-detection-paraphrases (DIPPER) ·
muyuhuatang/token_ensemble (ToBlend, de-anonymised from the dead `anonymous.4open.science` link) ·
suraj-ranganath/StealthRL ·
YixuanEvenXu/humanization-by-iterative-paraphrasing (HIP) ·
navid-aub/Esperanto-Dataset ·
huggingface.co/authormist/authormist-originality

**Papers:** SICO https://arxiv.org/abs/2305.10847 · HMGC https://arxiv.org/abs/2404.01907 ·
RAFT https://arxiv.org/abs/2410.03658 · Adversarial Paraphrasing https://arxiv.org/abs/2506.07001 ·
MASH https://arxiv.org/abs/2601.08564 · DIPPER https://arxiv.org/abs/2303.13408 ·
ToBlend https://arxiv.org/abs/2402.11167 · StealthRL https://arxiv.org/abs/2602.08934 ·
AuthorMist https://arxiv.org/abs/2503.08716 · ESPERANTO https://arxiv.org/abs/2409.14285 ·
Base Models Look Human / HIP https://arxiv.org/abs/2605.19516 ·
Soto et al. stylistic fingerprints https://arxiv.org/abs/2505.14608 ·
ARB https://arxiv.org/abs/2607.29539 · Sadasivan recursive paraphrase https://arxiv.org/abs/2303.11156

**Scripts and raw results in `.codemine/`:**
`score_advpara2.py` → `advpara_desklib_raw.json` (§B1) ·
`screen_backtranslate.py` → `res_backtranslate.json` (§B2) ·
`make_pairs.py`, `basefewshot.py` → `demo_pairs.json`, `res_ctrl_{base,instr}.json` (§B3) ·
`gsub.py` → `res_gsub_main.json` (§B4), `res_gsub_ens.json` (§B4c) ·
`transfer.py` → `res_transfer.json` (§B4a) ·
`transfer_ens.py` → `res_transfer_ens.json` (§B4c) ·
`score_mage_human.py`, `score_advpara.py` (earlier passes of §B1)

**Reproducing the prototype:**

```
.venv/bin/python .codemine/gsub.py --tag gsub_main \
    --passes 4 --top-k 30 --no-pos --min-overlap 0.6
.venv/bin/python .codemine/transfer.py

# ensemble-guided variant (§B4c)
.venv/bin/python .codemine/gsub.py --tag gsub_ens --passes 3 --top-k 12 \
    --no-pos --min-overlap 0.7 --limit 4 \
    --guides fakespot-ai/roberta-base-ai-text-detection-v1,andreas122001/roberta-academic-detector
```
