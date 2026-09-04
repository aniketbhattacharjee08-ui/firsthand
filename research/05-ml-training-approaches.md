# Training a Model That Learns to Humanize Text

**Research note 05 — architectures, training signals, datasets, evaluation**
Compiled 2026-09-03. All claims are sourced; URLs in the Sources section at the end.

---

## 0. Framing: what "humanize" actually optimizes

Across the literature there are three distinct — and frequently conflated — objectives:

1. **Detector evasion.** Minimise `P(AI)` under one or more classifiers. This is the objective with the most published, reproducible work, and it is the easiest to optimise (dangerously so).
2. **Style transfer toward a human target.** Move the text's stylistic representation toward a target author or toward the empirical distribution of human writing. Measured with authorship/style embeddings, not detectors.
3. **Perceived humanness to a reader.** Quality/naturalness judged by people or an LLM judge.

These come apart badly. Rivera Soto et al. (2025) show that models fine-tuned with RL to fool token-distribution detectors — driving FastDetectGPT to ~0.17–0.18 AUROC, i.e. worse than chance — still get **0.95 AUROC under a style-based detector**. Optimising (1) alone does not deliver (2). Conversely, TH-Bench (2025) documents an "impossibility triangle": across six humanizing attacks and 13 detectors, no attack simultaneously maximises evasion, text quality, and compute efficiency; HMGC drives metric-based detector AUC from 0.913 to 0.185 but under unrealistic assumptions, while prompt-based rewriting preserves quality and barely evades.

A v1 system that "learns to humanize" should therefore have a **multi-term objective from day one**, not a detector-only reward. The rest of this note surveys how each paradigm is actually built.

---

## 1. Training paradigms

### 1.1 Supervised paraphrase fine-tuning (seq2seq / LoRA on paired data)

The oldest and most robust starting point. The canonical artifact is **DIPPER** (Krishna et al., NeurIPS 2023): an 11B T5-XXL paraphraser that operates at **paragraph** granularity (not sentence) and exposes two **control codes** — lexical diversity and content reordering — as prefix tokens at training time. Passing GPT-2 XL output through DIPPER drops DetectGPT accuracy from **70.3% → 4.6%** at 1% FPR, and evades GPTZero, the OpenAI classifier, and watermarking. Semantics are largely preserved.

**How paired data is made.** Four practical recipes appear in the literature:

- **Reverse-direction synthesis (the dominant recipe).** Take human text; have a frontier LLM produce an AI version of it; then train the model on `(AI → human)`. AuthorMist does exactly this: 10,000 human abstracts from CheckGPT, paraphrased by GPT-4o, Claude 3.5 Sonnet and Gemini 1.5, across CS / humanities / physics, 100–500 words (median ~250). This gives aligned pairs with the AI side guaranteed to be "AI-shaped."
- **Continuation pairing.** The HAP-E corpus seeds an LLM with a ~500-word human chunk and asks for the next ~500 words, then pairs it against the *actual* human continuation. This yields pairs matched on topic, register and prior context — closer to what a humanizer sees at inference.
- **Round-trip / pseudo-parallel via paraphrase.** STRAP (Krishna, Wieting, Iyyer, EMNLP 2020) is the template: paraphrase input into a "style-neutral" form with a diverse paraphraser, then train an **inverse paraphraser** to reconstruct the original style from the neutral form. This manufactures parallel data for a task with no parallel data. ASTRAPOP reuses this exact trick to build its SFT reference model.
- **Author-conditioned pairs.** Rivera Soto et al. SFT on `(machine paraphrase → human original)` **with author exemplars in the prompt**, over 63,184 stratified authors from the Reddit Million User Dataset, base model Mistral-7B + LoRA.

**When SFT alone is enough:** it isn't, for evasion. AuthorMist reports the GRPO model reaching detector F1 of 0.11 where the SFT baseline stayed at 0.90 — SFT on paraphrase pairs barely moved the target detector. SFT's real role is as a **strong initialisation** that fixes format, length behaviour and register before any RL. Every good RL result in this space is preceded by an SFT or instruct-tuned starting point.

**A striking recent shortcut.** "Base Models Look Human To AI Detectors" (2026) reports GPTZero assigning **96.7% human probability to base-model continuations vs 30.3% for instruction-tuned continuations** of the same model on the same prefix. Their **HIP** method (Humanization by Iterative Paraphrasing) *minimally* fine-tunes a **base** (not instruct) model into a paraphraser — deliberately keeping it close to base continuation behaviour — then applies it iteratively so local context is progressively rewritten toward human context. This beat baselines on the semantic-preservation vs evasion frontier. Design implication: **low-distortion fine-tuning off a base checkpoint** may outperform aggressive post-training.

### 1.2 RL / DPO / GRPO against a detector reward

**Nicks et al., ICLR 2024 — "Language Model Detectors Are Easily Optimized Against."** The foundational result. Detector "human-ness" scores are used as a reward under a KL constraint. Their concrete pipeline is **DPO**: sample completions, score each with the detector, form pairwise preferences with the higher-humanness completion marked preferred, fine-tune on that preference set. It is data-efficient and works on relatively small (~7B) source models. The takeaway the whole field cites: detector evasion is cheap to learn.

**Zhou et al. (LREC-COLING 2024) — HMGC.** Adversarial *perturbation* rather than free paraphrase, in both white-box and black-box settings, using a fine-tuned proxy/surrogate detector to supply gradients for word-level substitution. Detectors are compromised "in as little as 10 seconds." Iterative adversarial training of the detector recovers only modest robustness. Code at `zhouying20/HMGC`.

**Zhou et al. (ICLR 2025) — HUMPA, proxy attacks.** The most operationally interesting variant: rather than fine-tuning the large generator, an RL-fine-tuned **small** "humanized" LM steers the large model **at decoding time**. Weights of the target LLM are never touched. Evaluated with Llama2-13B, Llama3-70B, Mixtral-8×7B, white- and black-box. **Average AUROC drop of 70.4%**, max 95.0% on a single dataset; up to 90.9% relative accuracy drop cross-discipline and 91.3% cross-language, with generation quality preserved.

**Cheng, Sadasivan et al. (NeurIPS 2025) — Adversarial Paraphrasing.** Training-free but detector-guided, and the clearest evidence for transferability. At each decoding step, top-p (0.99) / top-k (50) candidates are scored by a **guide detector** and the lowest-AI-score token is chosen — "a targeted, detector-guided beam search at depth one." Paraphraser: Llama-3-8B-Instruct. Guides: OpenAI-RoBERTa base/large, MAGE, RADAR. Result: **98.96% reduction in TPR@1%FPR on Fast-DetectGPT** (vs 15.03% for plain paraphrasing) with RoBERTa-large guidance; **87.88% average T@1%F reduction across eight detectors**, including watermark detectors (KGW, Unigram) and GLTR. Quality: SBERT cosine 0.81, perplexity comparable to human text, 87% of outputs rated 4–5/5 by GPT-4o. Their explanatory claim matters for architecture: *high-performing detectors converge toward a shared model of human text, so evading one pushes you toward that shared human distribution and thus evades others.*

**AuthorMist (David & Gervais, 2025) — the GRPO reference implementation.**

| Component | Setting |
|---|---|
| Base model | Qwen2.5-3B-Instruct |
| Algorithm | GRPO (no value critic; group mean as baseline) |
| Group size G | 8 |
| Learning rate | 5e-5 |
| KL coefficient β | 0.001 |
| Training | 1 epoch, 714 steps, 8×H100, **16 GPU-hours total** |
| Reward | `R(X,Y) = 1 − (1/k)·Σ_j P_dj(Y)` — **detector score only** |
| Inference | 8 paraphrases per 512-token chunk; pick lowest mean detector score |

Notably there is **no explicit semantic or fluency term** — meaning preservation is left entirely to the KL anchor. Results: ASR 78.6–96.2% per detector; the Originality-trained model reaches 95.17% mean ASR across all detectors; median E5 semantic similarity >0.94; target AUROC falls from ~0.95 to 0.07–0.80; perplexity moves from the AI-typical 25–40 band into a human-like 50–75 band. Failure modes admitted: semantic drift in technical domains despite high cosine scores, and poor transfer for the OpenAI-detector-trained model (32.33% mean ASR) — **detector-specific overfitting is real**.

**StealthRL (2026) — the multi-detector, multi-term reward.** The closest published thing to a well-designed v1 reward:

- Base: Qwen3-4B-Instruct, LoRA r=32, α=32, dropout 0.05
- GRPO: lr 2.8e-4, batch 16, group 8, 3 epochs on 10k MAGE AI samples, **KL λ = 0.05**
- Reward: `R = α·R_det + β·R_sem` with **α=1.0, β=0.1**
- `R_det = 1 − p_ens`, `p_ens = 0.6·p_RoBERTa + 0.4·p_FastDetectGPT` — deliberately mixing a **trained classifier** and a **zero-shot statistical** detector
- `R_sem` = E5 embedding cosine
- Train on RoBERTa + Fast-DetectGPT; **hold out Binoculars and MAGE** for transfer evaluation

Results: mean TPR@1%FPR 0.024, 97.6% ASR across four detectors; Binoculars 0.002 TPR, MAGE 0.089 — strong architectural transfer. The honest cost: **E5 similarity falls to 0.901 (vs 0.974 for simple paraphrase) and LLM-judge quality to 2.51 (vs 3.78)**. β=0.1 was not enough. This is the single most important calibration datapoint in this note.

**Constrained policy optimization (2026).** Instead of a weighted sum, formulate evasion as the objective and semantic similarity / fluency as **hard constraints** via a Lagrangian. The authors argue — and it is the theoretically right argument — that a Lagrangian prevents the model from trading meaning away for evasion at *any* exchange rate, which a fixed β cannot. Better trade-off frontier than unconstrained baselines.

**GradEscape (2025).** White-box-ish gradient attack: a **surrogate detector** provides gradients, token embeddings are perturbed and mapped back to nearby vocabulary items ("warm-start"). Transfers to unseen detectors including GPTZero, Sapling, Scribbr. Cheap in gradient evaluations. Relevant mainly as evidence that a *surrogate* detector is sufficient — you do not need query access to the deployed detector during training.

**Reward design synthesis.** Combining the above, a defensible reward is:

```
R = w_det · (1 − p_ensemble(y))              # ensemble of ≥2 detector families
  + w_sem · sim_embed(x, y)                  # E5 / SBERT cosine, or NLI entailment
  + w_flu · band(PPL(y); [lo, hi])           # band, NOT "minimise PPL"
  + w_sty · (−D(feat(y), feat(human)))       # stylometric distance
  − w_len · |len(y)/len(x) − 1|              # length control
  − β · KL(π ‖ π_ref)
```

Two design notes. First, **fluency must be a band, not a monotone term**: minimising perplexity makes text *more* AI-like, since low perplexity is exactly what Fast-DetectGPT and Binoculars key on; AuthorMist's success shows PPL *rising* from 25–40 into 50–75. Second, in TRL, `beta` now defaults to 0.0 (no reference model at all) — you must set it explicitly for this task, and StealthRL's 0.05 is a reasonable anchor, far above AuthorMist's 0.001.

**Reward hacking failure modes documented in the general RLHF literature** and directly applicable here: verbosity/length exploitation (mitigated by explicit length penalties and length-normalised losses), degenerate repetition, and exploitation of reward-model blind spots. Standard mitigations: raise the KL coefficient (raising it from 0.01 → 0.1 raises win-rate while lowering measured reward — the classic signature of hacking being suppressed), cap the maximum achievable reward, and use bounded reward signals.

### 1.3 Feature-conditioned generation and controllable style transfer

This paradigm swaps a detector for an **explicit representation of style**, which fixes the "beat one detector, fail the next" problem structurally.

**Style/authorship embeddings available off the shelf:**

- **Wegmann et al. (2022), "Same Author or Just Same Topic?"** — contrastive style representations trained with conversation-level controls so the embedding encodes style rather than topic. Controlled stylistic perturbations produce consistent linear shifts in this space, which is what makes it usable as a *reward*.
- **StyleDistance (NAACL 2025)** — RoBERTa-base + LoRA (lr 1e-4, batch 512, triplet loss margin 0.1) trained on **SynthStel**: GPT-4-generated near-exact paraphrases varying **40 style features across 7 categories** (syntactic, graphical/digital, emotional/cognitive, stylistic/aesthetic, social/interpersonal, lexical, temporal/aspectual), 100 positive/negative pairs per feature → ~320K contrastive triplets. Evaluated on STEL / STEL-or-Content and PAN 2011–2015 authorship verification. Models, data and code released.
- **LUAR** (LLNL) — the authorship representation used as the reward signal in ASTRAPOP.
- **Neurobiber** — fast neural predictor of Biber's linguistic feature inventory; useful as an interpretable feature extractor for a feature-diff critic.

**TinyStyler (EMNLP 2024 Findings).** The strongest efficiency argument in the whole area: an **800M** model conditioned on pre-trained **authorship embeddings** beats GPT-4 on authorship style transfer and beats controllable-generation methods on formal↔informal. Training uses reconstruction plus authorship-embedding conditioning, with filtering by style/meaning scores. Code at `zacharyhorvitz/TinyStyler`.

**STYLL / Patel et al. (2022), low-resource authorship transfer.** Transfers between Reddit authors given ~16 posts (~500 words) of target style, using in-context learning; authorship-representation models make automatic evaluation of this task feasible for the first time.

**ASTRAPOP (2024) — style transfer with policy optimization.** Two stages: (a) SFT on pseudo-parallel data built by paraphrasing (the STRAP trick); (b) policy optimization. Backbone LLaMA-2-7B + LoRA. The reward has exactly the shape a humanizer wants: a **toward** term (similarity to target style embedding), an **away** term (dissimilarity from source style), and a **length penalty to prevent degenerate outputs**. Preference pairs for DPO/CPO are built by shifting target style labels so source ≠ target, avoiding contradictory gradients. **DPO/CPO beat PPO.** Evaluated on Reddit MUD (5 examples/author) and ETS TOEFL (11 L1 communities), scored on toward/away/SBERT jointly.

**Feature conditioning mechanics.** The conventional implementation is **control tokens/codes prepended to the source** (DIPPER's lexical-diversity and reordering codes are the proven instance), or style-augmented attention. For a humanizer, the natural control vector is a discretised stylometric bucket: sentence-length mean and standard deviation, burstiness, type-token ratio, rare-word rate, clause depth, punctuation profile. The relevant empirical grounding: AI text shows **lower perplexity, more uniform sentence structure, higher lexical repetitiveness, and a flatter burstiness profile** than human text; sentence-length standard deviation is a reliable discriminator. Train with control codes computed from the *target* (human) side, then at inference sample control codes from the empirical human distribution for that genre.

### 1.4 Iterative edit models and critic-guided loops

**Edit-based generation.** The **Levenshtein Transformer** (NeurIPS 2019) models generation as an MDP alternating **insertion and deletion**, supporting dynamic length change and — importantly — *refinement of an existing sequence*. **EDITOR** adds a "reposition" operation that disentangles lexical choice from word position and decodes faster. Both are natively suited to "minimally edit this text," which is exactly the humanizer's job and is far more semantics-preserving than free regeneration. The practical caveat: these are trained NMT-style architectures, not LLM-compatible; in 2026 the pragmatic substitute is an LLM that emits a **diff or constrained edit list** rather than a full rewrite.

**Prompt-Based Editing for Text Style Transfer (Findings EMNLP 2023)** is the more directly reusable design: perform **discrete search with word-level editing** to maximise a composite scoring function, turning prompt-based generation into a classification problem — "more controllable than autoregressive generation of sentences." This is the template for a critic-guided loop.

**SICO** ("LLMs can be Guided to Evade AI-Generated Text Detection") builds evasion **in-context**: it iteratively constructs the prompt using word- and sentence-level substitutions selected by detector feedback, producing a reusable evasion prompt. **Self-Disguise Attack (2025)** pushes the same idea further — inducing the model to disguise its own output without external tooling.

**A practical critic-guided architecture** (no training required for v1):
```
draft → [detector critic: p(AI) per sentence]
      → [feature critic: Δ between output stylometrics and target human distribution]
      → [semantic critic: NLI entailment + embedding cosine, as a veto]
      → LLM rewriter, given the two diffs, edits only flagged spans
      → repeat ≤ k rounds, accept best point on the Pareto front
```
This is the "no-training baseline" (§4) and it is genuinely competitive: Adversarial Paraphrasing is training-free and achieves 87.88% mean T@1%F reduction.

### 1.5 Discriminator-in-the-loop / GAN-like adversarial training

**RADAR (NeurIPS 2023)** is the reference: a paraphraser and a detector trained **jointly and adversarially**, with the detector's feedback updating the paraphraser (PPO) and vice versa. Eight generators (Pythia, Dolly 1.0/2.0, Palmyra, Camel, GPT-J, LLaMA, Vicuna), four datasets, transfer validated on GPT-3.5-Turbo. RADAR significantly outperforms prior detectors *especially when paraphrasing is in play* — which is why RADAR is now a standard **held-out** detector in evasion papers.

The lesson for a humanizer builder is inverted but useful: **co-training a detector against your own humanizer produces a much harder adversary than any static detector**, and training against that moving target is the best available proxy for robustness to future detectors. The counter-evidence is DAMAGE (2025): a Mistral-NeMo-12B + LoRA detector trained with humanized text as an *invariance* (humanized data was only ~0.68% of training data, oversampled 18×, plus active learning with hard-negative mining) reached **98.26% TPR on humanized academic text at 3.47% FPR**, and caught even **detector-specific adversarial humanizers at 93.2%**. Adversarial training is a real defense; a humanizer trained against static detectors will not survive it.

---

## 2. Datasets

| Dataset | Size | Content / generators | Aligned human↔AI pairs? | License | Notes |
|---|---|---|---|---|---|
| **HC3** (Hello-SimpleAI) | ~24.3k rows main split, ~48.6k total, ~147MB | Reddit ELI5 (17.1k), finance (3.93k), medicine (1.25k), open-QA (1.19k), wiki-CSAI (842); human vs ChatGPT | **Yes** — `human_answers` and `chatgpt_answers` on the same question | CC-BY-SA-4.0 (source licenses may be stricter) | Best small paired starter set; ChatGPT-3.5 era, stylistically dated |
| **RAID** | >10M docs (train 802M / test 81M / extra 275M rows incl. attacks) | 11 generators, 8 domains, 4 decoding strategies, repetition-penalty variants, **11 adversarial attacks** (homoglyph, number, article_deletion, insert_paragraphs, perplexity_misspelling, upper_lower, whitespace, zero_width_space, synonym, paraphrase, alternative_spelling) | Partial (shared prompts/domains) | **MIT** | `pip install raid-bench`; leaderboard; the best robustness testbed |
| **MAGE / DeepfakeTextDetect** | 436,606 rows (319k/56.8k/60.7k) | Many domains × many LLMs, "in the wild" | Not prompt-matched | **Apache-2.0** | Used as the training pool by StealthRL and Adversarial Paraphrasing |
| **M4 / M4GT-Bench** | Multilingual, multi-domain, multi-generator | 3 tasks: binary detection, multi-way attribution, boundary detection | Prompt-matched within domains | **CC-BY-4.0** | ACL 2024; good for multilingual and for mixed human/AI boundary work |
| **Ghostbuster data** | 3 corpora: student essays, creative writing, news | Paired human + machine versions; ships per-token logprobs from ada/davinci | **Yes** (paired) | See repo | Authors warn training data is not representative of all styles/topics |
| **OpenAI GPT-2 output dataset** | 250k WebText test docs + 250k samples per model size (+5k valid/test each) | GPT-2 small/medium/large/XL; temp-1 and top-k 40 | Not paired | **MIT** | Historically important; obsolete stylistically |
| **TuringBench** | ~200k samples, 20 labels | 19 generators (GPT-1/2/3, GROVER, CTRL, XLM, XLNet, FAIR wmt19/20, Transformer-XL, PPLM) over ~10k news articles | Machine versions of the same news articles → effectively paired | Check site | Two tasks: Turing Test and Authorship Attribution |
| **HAP-E** (`browndw/human-ai-parallel-corpus`) | 66,320 texts, ~33.5M words; 12,000 human originals (2,000 × 6 types); 8,290 texts per model | Academic (Elsevier OA), news, fiction (Gutenberg), spoken (podcasts), blogs, TV/film scripts; GPT-4o, GPT-4o-mini, Llama-3 70B/8B base **and** instruct | **Yes — continuation-aligned**, human continuation vs LLM continuation of the same 500-word seed | **MIT** | The single best paired resource for training a humanizer; includes base vs instruct contrast |
| **Herbold et al. essays** | ~90 topics, human student essays + ChatGPT-3/4 essays | Argumentative student essays; zero-shot "write ~200 words on [topic]" | **Yes** (same topics) | Zenodo, open | Scientific Reports 2023; useful education-domain eval |
| **IDMGSP** | ~4k human + ~4k machine scientific papers | SCIgen, GPT-2, GPT-3/ChatGPT, Galactica | Partially | HF `tum-nlp/IDMGSP-*` | Scientific-writing domain |
| **Reddit Million User Dataset (MUD)** | 300M posts, 1M users (100–1000 posts each, Jul 2015–Jun 2016, Pushshift) | Human only | n/a | Pushshift terms | The standard authorship-style corpus; used by STYLL, ASTRAPOP, Rivera Soto et al. (63,184 stratified authors) |
| **Blog Authorship Corpus** | 19,320 bloggers, 681,288 posts, >140M words (2004) | Human only, with age/gender/industry metadata | n/a | Unknown/non-commercial research | Pre-LLM human text, useful as a "human" reference distribution |
| **PAN authorship verification** | PAN13/14/15/20/21/22; e.g. PAN14: 300 train + 400 test cases | Human only; cross-domain, fanfiction, cross-discourse-type | n/a | **CC-BY-4.0** | Evaluation set for style embeddings |
| Pre-2022 Wikipedia / Project Gutenberg | Large | Human only, LLM-contamination-free | n/a | CC-BY-SA / public domain | Best "uncontaminated human" reference distributions for feature matching |

**Selection guidance.** For *supervised pair training*, HAP-E (MIT, continuation-aligned, six genres, includes base-model outputs) is the strongest single choice, backed by HC3 for QA register and Ghostbuster/Herbold for essays. For *RL training pools*, MAGE (Apache-2.0) is what published GRPO work uses. For *held-out robustness evaluation*, RAID (MIT, 11 attacks) is non-negotiable. For *human reference feature distributions*, pre-2022 Wikipedia + Gutenberg + Blog Authorship + Reddit MUD.

---

## 3. Evaluation stack

| Axis | Metric / tool | How to run | Cost | Notes |
|---|---|---|---|---|
| **Evasion — zero-shot statistical** | **Fast-DetectGPT** (conditional probability curvature) | `baoguangsheng/fast-detect-gpt`, `scripts/local_infer.py`; scoring/sampling model gpt-neo-2.7B / gpt-j-6B, now Llama3-8B | 1 GPU; 340× faster than DetectGPT on A100 | AUROC 0.9887 (5-model), 0.9338 (ChatGPT/GPT-4) black-box |
| **Evasion — zero-shot paired-LM** | **Binoculars** (observer/performer cross-perplexity) | `ahans30/Binoculars`, default Falcon-7B + Falcon-7B-Instruct; `bino.compute_score(text)` | Two 7B models in memory (~15GB fp16, less quantised) | >90% TPR at **0.01% FPR**; code **BSD-3-Clause**, but "academic purposes only," not a consumer product |
| **Evasion — trained classifiers** | **RoBERTa OpenAI detector** (base/large); **MAGE** Longformer; **RADAR**; **desklib/ai-text-detector-v1.01** (DeBERTa-v3-large, trained on RAID, **MIT**, RAID-leaderboard leader); **MayZhou/e5-small-lora-ai-generated-detector** (e5-small + LoRA r=8 α=16, 33.4M params, **MIT**, 0.939 acc on RAID-test) | HF `transformers` | Trivial (e5-small is 33M params — cheap enough to sit **inside** an RL reward loop) | Use ≥2 families; StealthRL's 0.6/0.4 RoBERTa/Fast-DetectGPT mix is a reasonable default |
| **Evasion — style-based (the hard one)** | StyleDetect-type few-shot detectors over style embeddings (Wegmann, StyleDistance, LUAR) | Embed candidate + reference sets, measure separability | Cheap | Rivera Soto et al.: **0.95 AUROC on RL-evasion-optimised text** where standard detectors read 0.17–0.18. **Must be in your eval set or you will fool yourself.** |
| **Evasion — commercial APIs** | GPTZero; Originality.ai | API | GPTZero: free tier 10k words/mo; Essential $14.99, Premium $23.99, Professional $45.99/mo; **API priced separately, contact sales**. Originality.ai: **1 credit = 100 words** AI-only (2 credits/100 words with plagiarism); Pro $12.95–14.95/mo for 2,000 credits (200k words); Enterprise $136.58–179/mo for 15,000 credits — **API is Enterprise-only** | Budget: ~$0.007/100 words on Originality Pro. Do **not** put a paid API in the inner RL loop unless you have budgeted for G×steps calls (AuthorMist did and hit rate limits) |
| **Evasion metric form** | **TPR@1%FPR** (and TPR@0.01%FPR for Binoculars), AUROC, ASR | — | — | Report TPR at fixed low FPR, not accuracy; that is the operationally meaningful number |
| **Meaning preservation** | E5 / SBERT embedding cosine; BERTScore; **NLI entailment (bidirectional)**; LLM-as-judge | `sentence-transformers`, `bert-score`, DeBERTa-MNLI, GPT-4-class judge | Cheap except judge | Benchmarks: AuthorMist median E5 >0.94; StealthRL 0.901 (degraded); Adversarial Paraphrasing SBERT 0.81. **Cosine alone hides drift** — AuthorMist reports semantic drift in technical domains at 0.94+ cosine. Add NLI as a hard veto. |
| **Fluency / quality** | Perplexity under a strong LM (GPT-2 in TH-Bench; prefer Llama-3-8B); grammar error rate via **LanguageTool** (errors per 100 words); ROUGE-L for lexical overlap; Flesch Reading Ease | `evaluate`, `language_tool_python` | Cheap | **Target a band, not a minimum.** TH-Bench: recursive paraphrasing raises PPL ~4×. AuthorMist's human-like band is 50–75 vs AI's 25–40. |
| **Quality — judged** | LLM-as-judge 1–5, pairwise win-rate vs original; human eval | GPT-4-class | Moderate | DAMAGE: L1 commercial humanizers won only **26%** of fluency comparisons against originals — commercial tools do degrade quality measurably |
| **Style similarity** | StyleDistance / Wegmann / LUAR cosine to target; STEL / STEL-or-Content | HF models | Cheap | Recommended ensemble from the style-eval survey: **BLEU + ROUGE-1 + StyleDistance + GPT-4.1, performance-weighted → 0.821 mean accuracy**, beating any single metric |
| **Feature-distribution matching** | KL or Wasserstein distance between output and human feature distributions (sentence-length mean/σ, burstiness, TTR, rare-word rate, punctuation, clause depth); Biber features via **Neurobiber** | Custom; `scipy.stats.wasserstein_distance` | Cheap | Compute per-genre human reference distributions from pre-2022 corpora. Also see **AURA** (area under the robustness-accuracy curve), which summarises max distributional overlap between human and machine text as sample size grows — a single scalar for "are these distributions actually the same." |

**Evaluation protocol that avoids self-deception:**
1. Train the reward on detectors A and B; evaluate on held-out C, D, E from *different families* (StealthRL's pattern: train RoBERTa + Fast-DetectGPT, hold out Binoculars + MAGE).
2. Always include a **style-embedding detector** among held-outs.
3. Always include **RAID** for cross-domain/cross-generator robustness.
4. Report a Pareto frontier (evasion × semantic similarity × quality), never a single evasion number — TH-Bench's impossibility triangle means a single number is always cherry-picked.
5. Test on **long documents**, not just 100–300 token snippets; Rivera Soto et al. show multi-document analysis still separates human from machine even after strong attacks.

---

## 4. Practical recipe: hardware, base models, tooling

**Base models feasible on one GPU or a Mac.** Published detector-evasion work uses small models successfully: **Qwen2.5-3B-Instruct** (AuthorMist), **Qwen3-4B-Instruct** (StealthRL), **Mistral-7B + LoRA** (Rivera Soto et al.), **LLaMA-2-7B + LoRA** (ASTRAPOP), **800M** (TinyStyler). This is a task where 3–8B is genuinely sufficient; frontier scale is not required.

**Single-GPU LoRA/QLoRA.** Qwen3-8B LoRA (r=16, α=16, `target_modules="all-linear"`, 2–3 epochs, lr 2e-4) is roughly 2–4h on an A100 and 6–8h on an RTX 4090 (24GB, ~$1/hr rented). 16–24GB is the comfortable floor. Unsloth's rule of thumb for **QLoRA 4-bit is ≈1GB VRAM per 1B parameters**, with 5GB minimum for a 1.5B model; LoRA 16-bit needs ~4× more than QLoRA 4-bit. Unsloth reports Llama-3.1-8B GRPO at 20k context with 8 generations fitting in **54.3GB vs 510.8GB** for a standard implementation (~90% reduction).

**GRPO in TRL.** `GRPOTrainer` takes reward functions with signature `(prompts, completions, completion_ids, trainer_state, **kwargs) -> list[float]`, supports **multiple reward functions summed or weighted via `reward_weights`**, supports async reward functions (essential if a reward calls a detector service), and supports PEFT/LoRA and 4-bit quantization directly. Key config: `num_generations` (default 8), `max_completion_length` (512), **`beta` (KL coefficient) defaults to 0.0 — set it explicitly**, `epsilon` 0.2, `loss_type` default `dapo` (token-level normalisation; `dr_grpo` removes length bias — worth trying given length-hacking risk), `scale_rewards` ('group'), plus entropy regularisation (`entropy_coef`, adaptive entropy targeting) which is a useful anti-collapse lever. vLLM integration in colocate or server mode. Single-GPU ≤24GB recipe from the docs: `per_device_train_batch_size=1, num_generations=4, max_completion_length=256, gradient_accumulation_steps=8, use_vllm=False`.

**Unsloth's practical guidance:** expect a **minimum ~300 steps** before rewards move meaningfully, 500+ rows of data, and ~12 hours of training for decent results. Prefer **rubric-style rewards** (several small verifiable components) over one scalar — which is exactly the multi-term reward structure argued for above.

**Apple Silicon.** `mlx-lm` supports LoRA/QLoRA for Mistral, Llama, Qwen2, Gemma, Phi-2, Mixtral and others; a 7B on a 32GB M1 Max runs at ~250 tok/s with `--batch-size 1 --num-layers 4`. Quantized models automatically train as QLoRA. Stock `mlx-lm` has **no RL support**, but **`mlx-lm-lora`** (Goekdeniz-Guelmez) adds SFT, DPO, ORPO, **GRPO, Dr. GRPO, DAPO** natively on MLX, with `mlx-tune` and MLX-LoRA-Studio wrapping it. Realistic Mac plan: **SFT and DPO on Mac; rent a 24GB GPU for the GRPO phase**, where 8 generations × detector scoring per step makes MLX's lack of vLLM-class batched generation painful.

**Inference cost.** A 3–4B LoRA humanizer serves comfortably on a single 24GB GPU or a 32GB Mac. AuthorMist's inference pattern — 8 candidates per 512-token chunk, pick the lowest detector score — multiplies generation cost 8× but adds a large evasion gain for free; budget for best-of-N at serve time.

**The no-training baseline you must beat.** Prompted frontier LLM + feature-diff feedback loop:
1. Compute the target genre's human feature distribution offline (pre-2022 corpora).
2. Rewrite with an explicit style brief plus the measured feature gaps ("your sentence-length σ is 4.1; target 9.3 — merge two sentences and split one").
3. Score with a local ensemble (e5-small-lora + Fast-DetectGPT), plus NLI + cosine as vetoes.
4. Iterate ≤3 rounds, keep the best Pareto point.

This is essentially Adversarial Paraphrasing / SICO in prompt form. It requires zero training, gets 87.88% mean T@1%F reduction in the published version, and is the honest baseline for any learned model. **If a trained model does not clearly beat this on the Pareto frontier, it is not worth its training cost.**

---

## 5. Failure modes to anticipate

**Reward hacking via degeneration.** Detector-only rewards admit gibberish, homoglyph tricks, whitespace/zero-width-space insertion and other artifacts that fool classifiers without being text. RAID explicitly enumerates these as attacks (homoglyph, zero_width_space, whitespace, upper_lower, perplexity_misspelling) — meaning detectors are increasingly *trained against* them, and they are trivially strippable by normalisation. Mitigations: Unicode/whitespace normalisation before scoring; explicit length penalties (ASTRAPOP includes one specifically "preventing degenerate outputs"); bounded rewards and reward ceilings; higher KL coefficient (0.01 → 0.1 raises win-rate while lowering measured reward, the classic hacking signature); `dr_grpo` loss to remove length bias.

**Semantic drift under a high cosine score.** AuthorMist reports meaning changes in technical domains *despite* median E5 similarity >0.94. Embedding cosine is topic-sensitive and forgiving. Use **bidirectional NLI entailment as a hard gate** and an LLM judge on a sample, not cosine alone.

**Quality collapse as evasion improves.** StealthRL is the cleanest measurement: pushing to 97.6% ASR moved E5 similarity 0.974 → 0.901 and LLM-judge quality 3.78 → 2.51. TH-Bench formalises this as the impossibility triangle. DAMAGE finds commercial L1 humanizers win only 26% of fluency comparisons. **Expect to pay for evasion in quality, and decide the exchange rate deliberately** — the constrained-policy-optimization framing (hard constraints on semantics/fluency, evasion as objective) is the principled way to fix the rate.

**Detector overfitting (beat one, not others).** AuthorMist's OpenAI-detector-trained model transferred at only 32.33% mean ASR while the Originality-trained one hit 95.17%. Mitigations: train against an **ensemble spanning detector families** (trained classifier + zero-shot statistical + style-based), hold out detectors of unseen families, and randomise the detector sampled per rollout. The optimistic counterpoint is Adversarial Paraphrasing's finding that strong detectors converge toward a shared model of human text, so guidance by *one strong* detector transfers — but "strong" is load-bearing, and weak or idiosyncratic guides overfit.

**Style fingerprints survive detector evasion.** The most important negative result for a detector-only approach: RL-optimised evaders that reduce FastDetectGPT to 0.17–0.18 AUROC are still caught at **0.95 AUROC by a style-based detector**. This is a direct argument for including a **style-embedding term in the reward**, not just detector scores.

**The retrieval defense.** Krishna et al.'s countermeasure: the API provider stores its generations and does approximate matching against candidates. With a 15M-generation database it detects **80–97% of paraphrased generations at 1% FPR**. Paraphrase-based humanization *does not defeat this at all* — output remains near-duplicate to a stored generation. The only mitigations are heavy rewriting (which costs semantics) or not passing through a logging provider. It is currently deployed by no major provider at scale, but it is a structural, not statistical, defense: assume it may appear.

**Watermarking.** Sadasivan et al. show recursive paraphrasing degrades watermark detection, and Adversarial Paraphrasing evades KGW and Unigram watermarks. But the same work shows watermarks can be **spoofed** — an adversary can infer signatures without white-box access and make human text register as AI, a reputational hazard. Their theoretical result links best-achievable detector AUROC to the **total variation distance** between human and AI distributions: as models improve, reliable detection becomes information-theoretically hard. This cuts both ways — it is also why detector-evasion is a receding target and style/feature matching is the more durable objective.

**Adversarially trained detectors.** DAMAGE catches humanized academic text at 98.26% TPR / 3.47% FPR, and detector-specific adversarial humanizers at 93.2%, by treating humanization as an invariance. Anything trained only against today's static detectors will be obsolete against tomorrow's invariance-trained ones. **The durable objective is matching human feature and style distributions, not beating a specific classifier.**

**Base-model artifact risk.** "Base Models Look Human To AI Detectors" argues current detectors mostly track **instruction-tuning artifacts and local context**, not an invariant notion of machine-generation. Exploiting this (base-model paraphrasers, HIP) works *now* but is a detector bug that will be patched. Treat it as a cheap win with a short shelf life.

---

## 6. Recommended architecture for a v1 humanizer that learns

**Design principles derived from the evidence.** (a) Detector-only rewards overfit and leave style fingerprints (Rivera Soto et al.; AuthorMist transfer gap) → the reward must include a style/feature term. (b) Quality degrades measurably at high evasion (StealthRL; TH-Bench) → constrain semantics and fluency rather than weighting them lightly. (c) 3–4B models suffice (AuthorMist, StealthRL, TinyStyler) → this fits one GPU. (d) Minimal-distortion editing beats free regeneration for meaning preservation (Levenshtein/EDITOR, prompt-based editing, HIP). (e) A training-free detector-guided loop already gets ~88% T@1%F reduction → that is the bar.

### Proposed pipeline

**Stage 0 — Feature and style infrastructure (build first, it is reused everywhere).**
- Stylometric extractor: sentence-length mean/σ, burstiness, TTR, rare-word rate (frequency-tier histogram), clause depth, punctuation/contraction/discourse-marker profiles, paragraph-length distribution; optionally Biber features via Neurobiber.
- Per-genre human reference distributions from pre-2022 Wikipedia, Gutenberg, Blog Authorship Corpus, Reddit MUD, and the human halves of HAP-E.
- Style embedding: **StyleDistance** (content-independent, 40 features, released) plus **LUAR** for authorship.
- Detector bench, all local: `MayZhou/e5-small-lora-ai-generated-detector` (33M — cheap enough for the inner loop), RoBERTa-OpenAI-large, Fast-DetectGPT, Binoculars, `desklib/ai-text-detector-v1.01`, RADAR, plus a StyleDistance-based few-shot style detector.

**Stage 1 — SFT on reverse-direction pairs off a *base* checkpoint.**
- Base: **Qwen3-4B base** (or Qwen2.5-3B / Mistral-7B). Base, not instruct, per the base-model-artifact finding and HIP's low-distortion argument. LoRA r=32, α=32, dropout 0.05.
- Data: HAP-E `(LLM continuation → human continuation)` pairs across all six genres (MIT-licensed, continuation-aligned), plus HC3 `(chatgpt_answer → human_answer)`, plus Ghostbuster/Herbold essay pairs. Augment by generating AI mirrors of held-out human text with 2–3 different frontier models so the input distribution is not single-generator (AuthorMist's recipe).
- **Condition on control codes** (DIPPER-style prefix): discretised target stylometric buckets computed from the human side at training time. At inference, sample codes from the genre's human distribution. This is what makes the model *controllable* rather than a black-box paraphraser.
- Train on Mac via `mlx-lm` / `mlx-lm-lora`, or a rented 24GB GPU (~2–4h).

**Stage 2 — GRPO with a constrained, multi-family reward.**
- TRL `GRPOTrainer`, `num_generations=8`, `max_completion_length=512`, **`beta=0.05`** (StealthRL's value; explicitly set — TRL defaults to 0.0), `loss_type="dr_grpo"` to suppress length bias, modest `entropy_coef` to resist collapse, LoRA + 4-bit, vLLM colocate.
- Reward functions passed as a **list with `reward_weights`** (TRL supports this natively), so each term is individually loggable:
  1. `R_det = 1 − p_ens`, ensemble over **e5-small-lora + Fast-DetectGPT** (a trained classifier and a zero-shot statistical detector), with the specific detector **sampled per rollout** to prevent single-detector overfitting. Weight 1.0.
  2. `R_style = cos(StyleDistance(y), μ_human_genre)` — the term that addresses the style-fingerprint failure. Weight ~0.4.
  3. `R_feat = −W₁(feat(y), feat_human_genre)` — Wasserstein distance on the stylometric vector. Weight ~0.3.
  4. **Hard gates returning large negatives rather than soft weights:** bidirectional NLI entailment below threshold → reject; E5 cosine < 0.88 → reject; LanguageTool errors/100 words above the human band → reject; perplexity outside the human band `[lo, hi]` → reject; `|len(y)/len(x) − 1| > 0.25` → reject. This is a cheap approximation of the constrained-policy-optimization formulation and it is the main defense against the StealthRL quality collapse.
- Hold out **Binoculars, RADAR, desklib, and the style detector** entirely. ~300+ steps minimum, 10k MAGE/HAP-E prompts, one 24GB GPU, roughly overnight.

**Stage 3 — Inference-time critic loop (ship this even without Stage 2).**
- Best-of-N: generate 8 candidates, rank by the *full* reward (not detector alone).
- Optional detector-guided decoding (Adversarial Paraphrasing) with the cheap e5-small guide when latency budget allows.
- Iterative refinement, ≤3 rounds, **editing only flagged sentences** rather than rewriting wholesale: the sentence-level detector score and the feature diff both feed the rewrite prompt, and the semantic gate vetoes any round that fails NLI. This is the HIP iterative-rewrite insight plus the prompt-based-editing controllability argument.

**Stage 4 — Adversarial hardening (v1.5).** Co-train a discriminator on your own outputs, RADAR-style, and add it as a rotating member of the reward ensemble. DAMAGE shows this is exactly what serious detectors will do to you; doing it to yourself first is the only available proxy for future robustness.

**Evaluation gate for shipping.** Report the full Pareto frontier on RAID (held-out domains and generators) plus HAP-E test: TPR@1%FPR on each of six held-out detectors *including the style detector*; E5 cosine and bidirectional NLI entailment rate; PPL under Llama-3-8B relative to the human band; LanguageTool errors/100 words; GPT-4-class pairwise win-rate vs the unmodified input; and Wasserstein distance to the human feature distribution. **Ship only if it dominates the Stage-3-only no-training baseline on that frontier.**

**Why this rather than the obvious alternative.** The obvious v1 — LoRA + GRPO against a single detector — is AuthorMist, and it is known to reach 95%+ ASR while (a) transferring unevenly across detectors (32% in the worst case), (b) leaving a 0.95-AUROC style fingerprint, and (c) drifting semantically in technical text at 0.94 cosine. The three additions above — style/feature reward terms, hard semantic/fluency gates instead of a soft β=0.1, and control-code conditioning learned from paired human data — each target one of those documented failures, and each is supported by a published result rather than intuition.

---

## Sources

**RL / detector-evasion attacks**
- Nicks, Mitchell, Rafailov, Sharma, Manning, Finn, Ermon — *Language Model Detectors Are Easily Optimized Against*, ICLR 2024. https://openreview.net/forum?id=4eJDMjYZZG · https://proceedings.iclr.cc/paper_files/paper/2024/file/1f9f07df0992ce21698d800eaa891bd8-Paper-Conference.pdf
- Zhou, He, Sun — *Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack* (HMGC), LREC-COLING 2024. https://arxiv.org/abs/2404.01907 · https://aclanthology.org/2024.lrec-main.739/ · code https://github.com/zhouying20/HMGC
- *Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors* (HUMPA), ICLR 2025. https://arxiv.org/abs/2410.19230 · https://proceedings.iclr.cc/paper_files/paper/2025/file/ab1ee157f7804a13f980414b644a9460-Paper-Conference.pdf
- Cheng, Sadasivan, Saberi, Saha, Feizi — *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text*, NeurIPS 2025. https://arxiv.org/abs/2506.07001 · https://arxiv.org/html/2506.07001 · code https://github.com/chengez/Adversarial-Paraphrasing
- David & Gervais — *AuthorMist: Evading AI Text Detectors with Reinforcement Learning*, 2025. https://arxiv.org/abs/2503.08716 · https://arxiv.org/html/2503.08716
- *StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors*. https://arxiv.org/html/2602.08934
- *Detector-Evasive LLM Paraphrasing via Constrained Policy Optimization*. https://arxiv.org/pdf/2606.00392
- *GradEscape: A Gradient-Based Evader Against AI-Generated Text Detectors*. https://arxiv.org/pdf/2506.08188
- *MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization*. https://arxiv.org/pdf/2601.08564
- *Base Models Look Human To AI Detectors* (HIP). https://arxiv.org/html/2605.19516v1
- *Large Language Models can be Guided to Evade AI-Generated Text Detection* (SICO). https://arxiv.org/pdf/2305.10847
- *Self-Disguise Attack: Induce the LLM to disguise itself for AIGT detection evasion*. https://arxiv.org/pdf/2508.15848

**Paraphrase, style transfer, style representations**
- Krishna, Wieting, Iyyer — *Reformulating Unsupervised Style Transfer as Paraphrase Generation* (STRAP), EMNLP 2020. https://aclanthology.org/2020.emnlp-main.55/ · code https://github.com/martiansideofthemoon/style-transfer-paraphrase
- Krishna et al. — *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense* (DIPPER), NeurIPS 2023. https://arxiv.org/abs/2303.13408
- Patel et al. — *Low-Resource Authorship Style Transfer: Can Non-Famous Authors Be Imitated?* (STYLL), 2022. https://arxiv.org/abs/2212.08986
- Horvitz et al. — *TinyStyler: Efficient Few-Shot Text Style Transfer with Authorship Embeddings*, EMNLP 2024 Findings. https://arxiv.org/abs/2406.15586 · code https://github.com/zacharyhorvitz/TinyStyler
- *Authorship Style Transfer with Policy Optimization* (ASTRAPOP). https://arxiv.org/html/2403.08043v2 · code https://github.com/isi-nlp/astrapop
- Wegmann, Schraagen, Nguyen — *Same Author or Just Same Topic? Towards Content-Independent Style Representations*, 2022.
- Patel, Zhu et al. — *StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples*, NAACL 2025. https://arxiv.org/html/2410.12757v1 · https://aclanthology.org/2025.naacl-long.436.pdf
- *Neurobiber: Fast and Interpretable Stylistic Feature Extraction*. https://arxiv.org/pdf/2502.18590
- LUAR authorship representations. https://github.com/LLNL/LUAR
- *Prompt-Based Editing for Text Style Transfer*, Findings of EMNLP 2023. https://arxiv.org/abs/2301.11997
- Gu, Wang, Zhao — *Levenshtein Transformer*, NeurIPS 2019. https://papers.nips.cc/paper/9297-levenshtein-transformer
- Xu & Carpuat — *EDITOR: An Edit-Based Transformer with Repositioning*, TACL 2021. https://arxiv.org/abs/2011.06868

**Detection, defenses, benchmarks**
- Hu, Chen, Ho — *RADAR: Robust AI-Text Detection via Adversarial Learning*, NeurIPS 2023. https://arxiv.org/abs/2307.03838
- Hans et al. — *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*. https://arxiv.org/abs/2401.12070 · code https://github.com/ahans30/Binoculars
- Bao et al. — *Fast-DetectGPT*. https://github.com/baoguangsheng/fast-detect-gpt
- Verma et al. — *Ghostbuster: Detecting Text Ghostwritten by Large Language Models*. https://arxiv.org/abs/2305.15047 · code https://github.com/vivek3141/ghostbuster
- Sadasivan et al. — *Can AI-Generated Text be Reliably Detected?* https://arxiv.org/abs/2303.11156
- Rivera Soto et al. — *Attacks on Machine-Text Detectors Retain Stylistic Fingerprints* / *Language Models Optimized to Fool Detectors Still Have a Distinct Style*. https://arxiv.org/abs/2505.14608 · https://arxiv.org/html/2505.14608v1
- *DAMAGE: Detecting Adversarially Modified AI Generated Text*. https://arxiv.org/html/2501.03437v1
- *TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors*. https://arxiv.org/pdf/2503.08708 · https://arxiv.org/html/2503.08708v1
- *Evaluating Style-Personalized Text Generation: Challenges and Directions*. https://arxiv.org/html/2508.06374v2

**Datasets**
- HC3. https://huggingface.co/datasets/Hello-SimpleAI/HC3
- RAID. https://arxiv.org/abs/2405.07940 · https://github.com/liamdugan/raid · https://raid-bench.xyz/
- MAGE / DeepfakeTextDetect. https://huggingface.co/datasets/yaful/MAGE
- M4GT-Bench. https://arxiv.org/abs/2402.11175 · https://github.com/mbzuai-nlp/M4GT-Bench
- OpenAI GPT-2 output dataset. https://github.com/openai/gpt-2-output-dataset
- TuringBench. https://arxiv.org/abs/2109.13296
- HAP-E human-AI parallel corpus. https://huggingface.co/datasets/browndw/human-ai-parallel-corpus · mini: https://huggingface.co/datasets/browndw/human-ai-parallel-corpus-mini
- Herbold et al. — *A large-scale comparison of human-written versus ChatGPT-generated essays*, Scientific Reports 2023. https://www.nature.com/articles/s41598-023-45644-9 · https://arxiv.org/abs/2304.14276
- IDMGSP. https://github.com/qwenzo/-IDMGSP · https://aclanthology.org/2023.trustnlp-1.17/
- Blog Authorship Corpus. https://huggingface.co/datasets/barilan/blog_authorship_corpus · https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
- PAN authorship verification. https://pan.webis.de/clef20/pan20-web/author-identification.html · https://zenodo.org/records/6337137

**Detectors and pricing**
- desklib/ai-text-detector-v1.01. https://huggingface.co/desklib/ai-text-detector-v1.01
- MayZhou/e5-small-lora-ai-generated-detector. https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector
- GPTZero pricing. https://gptzero.me/pricing
- Originality.ai pricing. https://originality.ai/pricing

**Training infrastructure**
- TRL `GRPOTrainer`. https://huggingface.co/docs/trl/main/en/grpo_trainer
- Unsloth RL/GRPO guide. https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide.md · Qwen3: https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune
- MLX-LM LoRA. https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- mlx-lm-lora (SFT/DPO/ORPO/GRPO/Dr.GRPO/DAPO on Apple Silicon). https://github.com/Goekdeniz-Guelmez/mlx-lm-lora · https://pypi.org/project/mlx-lm-lora/

**Reward hacking**
- Weng — *Reward Hacking in Reinforcement Learning*. https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- *Reward Shaping to Mitigate Reward Hacking in RLHF*. https://arxiv.org/pdf/2502.18770
- *The Energy Loss Phenomenon in RLHF*. https://arxiv.org/pdf/2501.19358
