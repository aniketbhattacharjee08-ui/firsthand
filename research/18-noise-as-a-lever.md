# Noise as a Lever: Is Deliberate Imperfection a Viable Humanization Strategy?

Date 2026-09-07. Report 18. About 110 sources, roughly 40 read in full from PDF or HTML rather than abstract. Builds on reports 00, 04, 12 and 13; does not re-derive what they established (character-level tricks are dead against GPTZero, base models read as human, best-of-N decays as a power law).

**The question asked:** the project owner's hypothesis is that our pipeline produces text that is too clean and too high-quality, and that deliberately introducing noise would work better than polishing. Is that true, and if so which noise?

---

## 0. Findings up front

**The hypothesis is half right, and the half that is right does not license the half that is wrong.**

1. **"Too clean" is a real, quantified signal in the aggregate distribution.** A detector that does nothing but count grammatical-error-correction edits reaches **98.62% AUROC** (GECScore): human text needs 10-30 edits per sample, LLM text 0-10 and often zero. A PAN 2025 feature study measured **6.41 errors per AI text versus 14.08 per human text**. Detector scores correlate with lexical formality at Spearman **ρ = 0.30-0.35** on Pangram and GPTZero, and genuine human abstracts are flagged at rates from **2.1% (chemistry) to 26.6% (theology)** on the same detector. Clean, formal, register-perfect prose is more detectable. That part of the hypothesis is supported.

2. **Injecting noise into finished text does not exploit that signal against a supervised detector. In most measured cases it backfires.** Across every controlled study on RoBERTa/DeBERTa-family and commercial detectors, token-level noise either leaves the score flat or raises it: synonym swap at 50% moved RoBERTa-Large **+22.7 points toward AI** and Originality **+11.5**; keystroke-realistic typos left fine-tuned DeBERTa at **96-112% of baseline AUROC**; light LLM paraphrase raised AUROC on every HC3-trained detector "without exception"; a homoglyph attack dropped GPTZero's human probability from **99.9% to 18.0%**. Pangram's DAMAGE holds **98.26% TPR** on humanizer output whose worst tier is defined by injected typos and broken grammar. Structural noise (sentence reordering, paragraph splitting) leaves the supervised OpenAI classifier flat or higher and moves GPTZero by 0.3 points. **No published study has ever run an error-injection-only experiment against GPTZero, Pangram, Originality or Turnitin.** That is a gap, not a result.

3. **The one kind of noise with strong evidence is distributional noise introduced at generation time, not text noise added afterwards.** Sampling with a repetition penalty moved GPTZero from **98.8% to 82.5%** accuracy on chat-model text and from **74.7% to 4.8%** on base-model text (RAID, 2024, 5% FPR). Frequency and presence penalties of 0.3-0.6 halved detection while preserving quality (Fishchuk and Braun). Zero-shot detectors do not merely degrade under raised entropy, they **invert**: Binoculars falls from 0.9928 AUROC at T=0.5 to **0.0200 at T=1.1**. But this lever has a hard ceiling: a RoBERTa trained on a mixture of decoding settings recovers to **≥95% on every one of 37 configurations**, and temperature above 1.0 produces text with MTLD 684 and perplexity 612 against human values of 95 and 14. The durable part of the entropy lever is the base-model distribution (report 13), not the sampler.

4. **Exactly one class of noise is invisible to a grader and visible to a detector: lexical-choice perturbation that restores perplexity into the human band without introducing errors.** Simple paraphrase drives perplexity to **9.28** (human 15.02); adversarial paraphrasing restores it to **14.26-17.11** while **87% of outputs are rated 4-5/5** with no statistically significant quality gap. Every other noise type either fails to move the detector, or costs grade, or both.

5. **Mechanical error injection is the most expensive noise per unit of detector movement.** Going from 0 to 2 errors per 100 words cost **2.18 points on a 7-point quality scale** (5.79 → 3.61); the next 2 errors cost 0.66. The first visible error does the damage. LLM graders penalise informality (−1.90 of 10) three times more than grammar errors (−0.60) and ignore instructions to stop. Meanwhile the human error distribution that a detector would need to see is **not typos**: in W&I+LOCNESS, spelling is 3.7-5.1% of errors, while punctuation is 17-19%, determiners 10-11%, prepositions 8-10%. Random typo injection models the wrong 4%.

6. **Content-level noise (tangents, digressions, redundancy, self-correction) has zero supporting evidence and one piece of contrary evidence.** In the only cue-frequency data that exists, "over-explains or includes irrelevant details" is an **AI** cue at 19.5% frequency. The only content-level manipulation that has been measured — adding specific places, people, metrics and removing formulaic openers — is the *opposite* of adding noise, and it halved GPTZero (76.7 → 46.7 TPR) while costing Pangram only 10 points.

**Verdict:** replace the hypothesis "our text is too clean, add noise" with "our text is too low-entropy at the source and too register-uniform; restore entropy at generation, perturb lexical choice under a quality gate, stop over-polishing, and never inject errors." Sections 2 and 8 give the numbers; section 9 ranks the moves.

---

## 1. Splitting the hypothesis: five things "noise" can mean

The word covers five mechanisms that act at different points in the pipeline and are attacked by different detector families. Conflating them is why the folk advice ("add typos", "raise burstiness") persists after the detectors it targeted were retired.

| Noise class | Where it acts | What it changes | Detector family it plausibly moves |
|---|---|---|---|
| **Information-theoretic** | Generation (sampler, checkpoint) | Per-token surprisal mean and variance, tail-token usage, lexical diversity | Zero-shot/perplexity (directly); supervised (only via train/test mismatch) |
| **Token/word** | Post-hoc edit | Individual tokens: synonyms, deletions, case, Unicode | Zero-shot (strongly); supervised (weakly, often reversed) |
| **Structural** | Post-hoc edit | Sentence length distribution, order, paragraphing, punctuation habits | Hand-engineered feature detectors; almost nothing else |
| **Semantic/content** | Generation or heavy rewrite | What is said: specificity, stance, digressions, hedging | Content-origin detectors (HART, D2C); human readers |
| **Error** | Post-hoc edit | Grammar, spelling, punctuation correctness | GEC-count detectors; perplexity detectors; supervised only if trained on it |

The synthesis (report 00, finding 1) established that GPTZero is a supervised transformer, and report 12 established that Pangram, Turnitin and Originality are too. So the column that matters is "supervised", and the rest of this report reads every result through that filter. Results on GLTR, DetectGPT, Fast-DetectGPT and Binoculars are reported because they are what most papers measure, but they are not our target.

---

## 2. Information-theoretic noise

### 2.1 Human text is higher-entropy and less uniform than LLM text, as measured under an LM

Every study that measures surprisal under a language model finds the same direction: human text has higher mean surprisal and higher surprisal variance.

| Quantity | Human | LLM | Source |
|---|---|---|---|
| Perplexity under Llama-2-7B, news | **9.09** | GPT-2 8.33; GPT-3 3.90; ChatGPT **3.39**; GPT-4 5.01 | RAID, ACL 2024 |
| Perplexity under GPT-2-XL | **21.2** | GPT-2 8.10; ChatGPT 9.31; GPT-4 13.4 | RAID |
| Self-BLEU (repetitiveness) | **7.64** | GPT-2 23.9; ChatGPT 10.3; GPT-4 9.42 | RAID |
| Words outside GPT-2 top-100 | **2.41×** as often as generated text; odds ratio 5.32 for >top-100, **0.09** for top-1 | | GLTR, ACL 2019 |
| STTR / MTLD, news | **0.491 / 96.51** | Mistral-7B 0.452/86.34; Falcon-7B 0.424/57.37 | Muñoz-Ortiz et al. 2024 |
| Story-generation perplexity vs reference (GPT-2-L) | Reference **16.33** | nucleus 0.9 → 7.75; top-k 30 → 7.07; T=1 → 25.67; **typical τ=0.2 → 14.25** | Meister et al., TACL 2023 |
| Summarization, beam search | Reference 10.29 | beam k=5 → **1.39** | Meister et al. |
| Grammatical-error-correction edits needed | **10-30** per sample | 0-10, often ≈0 | GECScore, COLING 2025 |

The Uniform Information Density hypothesis says humans spread information evenly. Both papers that operationalise UID as surprisal variance (GPT-who, NAACL Findings 2024; Venkatraman, He and Reitter, EACL Findings 2023) find that **human text is *less* uniform than LLM text**: "the UID scores of human-written text have a higher mean and larger standard deviation than most machine-written text... machines seem to be spreading information more evenly." Greedy decoding produces "a very high and narrow peak" of near-zero variance. GPT-who is explicit that this cannot refute UID as a theory of human production, because GPT-2-XL is a poor proxy for the human generative distribution and uniformity is only defined relative to a distribution. But the model-measured quantity is exactly what detectors see, so operationally it is the right target. **Unverified:** neither paper publishes scalar mean/variance values, only histograms; the direction is verified, the magnitude is not.

Meister et al.'s core empirical result (Figure 1 of the TACL paper) is that on human text the per-token deviation ε = −log p(y_t | y_<t) − H(Y_t | Y_<t) is sharply peaked at zero across three tasks. Human text sits near the conditional entropy. Every standard decoder misses in a specific direction: nucleus, top-k and beam land far *below* human perplexity, temperature 1.0 overshoots, and locally typical sampling lands closest (−2.08 on story generation) with human quality ratings that match or beat nucleus (4.15 vs 4.09 on a 5-point scale; reference 4.12).

Zipf and Heaps behaviour follows the same pattern. Mikhaylovskiy (Findings of EMNLP 2025) reports LLM text obeys both laws only within a narrow, model-dependent temperature window, near T=1 for base models and higher for instruction-tuned ones, independent of size and prompting. **Unverified in this report:** I could not re-fetch the paper (aclanthology.org is blocked from this environment and the paper is not on arXiv); the finding is taken from report 04.

### 2.2 RLHF collapses entropy, and it is getting worse

| Finding | Number | Source |
|---|---|---|
| Direct-prompt diversity after SFT vs after DPO | **20.8% → 10.8%** | Verbalized Sampling, 2510.01171 |
| Verbalized sampling recovers base-model diversity | 66.8% vs 23.8% for direct prompting; typicality-bias α = 0.57-0.65 | same (medium confidence: HTML summariser pass, not line-verified) |
| Cross-model response similarity, 25 models at top-p 0.9 / T=1 | **79%** of pairs exceed 0.8 similarity; 61.2% even with min-p | Artificial Hivemind, NeurIPS 2025 |
| Lexical diversity of instruction-tuned models vs human | 2023 models above human; **2025 models below human** | "More Aligned, Less Diverse?" (agent-reported; not independently verified) |
| Base vs instruct on GPTZero, Llama3-8B | **96.7% vs 30.3%** human | Base Models Look Human, 2605.19516 |
| Present participial clauses, instruct vs human | **5.3×** (d=1.38); nominalizations 2.1×; base Llama at human rates | Reinhart et al., PNAS |

Kirk et al. (ICLR 2024) is the canonical citation for RLHF diversity collapse but reports only figures, no table. The convergent picture is that the entropy gap between human and LLM text is manufactured at the alignment stage, which is why report 13's base-model finding is the largest lever in the evidence base. That is the first and most important sense in which the "too clean" intuition is right: instruction tuning removes the entropy that a human writer has.

### 2.3 Does raising sampling entropy move detectors? Yes, and it depends entirely on the detector family

This is the sharpest result in the entire literature and it cuts both ways. Three sources, two of which I verified line by line from the PDF.

**RAID (Dugan et al., ACL 2024), Table 5, accuracy at 5% FPR, verified.**

| Detector | Open-source **chat** models: greedy → sampling → greedy+RP → sampling+RP | Open-source **non-chat**: same | Closed chat: greedy → sampling | Closed non-chat: greedy → sampling |
|---|---|---|---|---|
| GPTZero | **98.8 → 93.7 → 98.4 → 82.5** | **74.7 → 34.6 → 9.4 → 4.8** | 92.3 → 88.5 | 60.6 → 53.4 |
| Originality | 98.6 → 86.3 → 97.7 → 72.5 | 99.9 → 64.1 → 89.0 → 51.2 | 96.8 → 89.0 | 91.7 → 85.4 |
| RoBERTa-B (GPT2) | 84.1 → 52.3 → 77.9 → 26.2 | 98.6 → 44.1 → 60.5 → 35.4 | 70.9 → 41.7 | 65.1 → 52.5 |
| RADAR | 88.8 → 77.4 → 85.6 → 66.4 | 91.8 → 63.8 → 48.3 → 31.8 | 81.6 → 75.3 | 72.2 → 67.7 |
| Binoculars | 99.9 → 86.6 → 99.7 → 60.6 | 99.9 → 62.3 → 72.4 → **0.6** | 99.2 → 92.1 | 99.0 → 95.0 |
| Fast-DetectGPT | 98.6 → 74.5 → 96.2 → 40.5 | 97.8 → 56.1 → 79.7 → **0.6** | 96.0 → 74.1 | 93.8 → 86.3 |
| GLTR | 89.8 → 67.5 → 83.9 → 38.3 | 99.6 → 56.9 → 44.5 → **0.5** | 80.7 → 54.3 | 75.6 → 63.7 |

RP = repetition penalty θ=1.2. RAID's own summary: "adding a repetition penalty decreases accuracy by up to **32 points** regardless of decoding strategy" and "random sampling with a repetition penalty consistently makes output generations very difficult to detect, especially for open-source non-chat models." (Note: a machine summary of this paper circulating in our earlier notes said 38 points; the PDF says 32.)

Read the GPTZero row carefully. On chat models, sampling alone costs 5 points and sampling plus repetition penalty costs 16. On base models the same change takes GPTZero from 74.7% to **4.8%**. The two levers compound: base checkpoint plus entropy-restoring decoding was nearly invisible to the 2024 GPTZero. This is the strongest single quantitative support for the owner's intuition, and it is entirely about generation, not about editing.

**Dubois, Yvon and Piantanida (Oct 2025, arXiv 2510.13681), verified.** 2,000 RAID human texts, 37 decoding configurations of Llama-3.2-3B, six sampler families. Two results matter.

First, the diversity table shows where the human band is (Table 2):

| Setting | MTLD | Simpson ×100 | Zipf α | Perplexity |
|---|---|---|---|---|
| **Human** | **94.60** | **0.66** | **1.20** | **14.32** |
| T=0.7 | 38.50 | 0.87 | 1.29 | 4.10 |
| T=0.9 | 70.20 | 0.73 | 1.19 | 11.07 |
| T=1.0 | 121.41 | 0.57 | 1.12 | 38.08 |
| T=1.1 | **684.80** | 0.22 | 1.02 | **612.48** |
| T=1.3 | 15,264.92 | 0.02 | 0.91 | 14,475.99 |
| Repetition penalty 1.05 | **428.22** | 0.31 | 1.10 | 107.55 |
| Repetition penalty 1.20 | 4,373.57 | 0.04 | 1.08 | 547.10 |
| top-p 0.95 | 83.70 | 0.67 | 1.19 | 19.07 |
| top-k 100 | 81.71 | 0.68 | 1.23 | 11.12 |
| typical 0.95 | 87.52 | 0.68 | 1.19 | 19.70 |
| η = 1e-4 | 93.45 | 0.65 | 1.18 | 17.38 |

The authors' summary: "η = 10⁻⁴, T = 1.0, k = 100, p = 0.95 and τ = 0.95 yield diversity metrics closest to human text, whereas extreme repetition penalties or overly permissive settings push MTLD and perplexity far beyond — and Simpson far below — their natural (human) ranges." Two warnings for us. Temperature 1.1 is already nonsense (MTLD 7× human, perplexity 43× human). And **even a repetition penalty of 1.05 drives MTLD to 4.5× the human value** — the RAID result that repetition penalty fools detectors is real, but the text it produces is measurably abnormal on diversity metrics, which means a detector trained on it catches it trivially (next table). Repetition penalty is a 2024 exploit, not a human-matching setting.

Second, the supervised result (Table 3, RoBERTa-base accuracy, train on row family, test on column family):

| Train ↓ / Test → | T<1 | T=1 | T>1 | rep-pen | top-k | top-p | typical | η |
|---|---|---|---|---|---|---|---|---|
| T<1 | 0.95 | 0.79 | **0.63** | 0.68 | 0.91 | 0.94 | 0.91 | 0.92 |
| T=1 | 0.88 | 0.96 | **0.99** | **0.99** | 0.92 | 0.87 | 0.90 | 0.90 |
| T>1 | **0.52** | 0.61 | 0.95 | 0.88 | 0.54 | 0.52 | 0.53 | 0.53 |
| rep-pen | 0.57 | 0.70 | 0.98 | 0.96 | 0.60 | 0.57 | 0.59 | 0.59 |
| **mixture** | **0.96** | **0.95** | **0.98** | **0.98** | **0.95** | **0.96** | **0.95** | **0.95** |

A supervised detector that has seen only cool text is fooled by hot text (0.63) — but one that has seen T=1 text catches hot and repetition-penalised text at **0.99**, and one trained on the mixture is at ≥0.95 everywhere. "Training on a uniformly sampled mixture of the training data seems to solve misclassification issues almost entirely." Temperature is not a durable evasion against a vendor who samples the decoding space, and GPTZero's 2026 technical report says they augment with TempParaphraser and multi-configuration paraphrasers. What did break the supervised RoBERTa was a shift in the *human* side: replacing BBC News with CCNews as the human set dropped accuracy from 95% to **72%**. The authors conclude the classifier "is mostly overfitting its human training dataset."

The zero-shot picture is the opposite (Table 4, AUROC):

| Sampler | Binoculars | Fast-DetectGPT |
|---|---|---|
| T=0.5 / 0.9 / **1.1** / 1.3 | 0.9928 / 0.9449 / **0.0200** / 0.0019 | 0.9468 / 0.8729 / 0.3693 / 0.1672 |
| Repetition penalty 1.05 / 1.10 | **0.0711** / 0.0182 | 0.4443 / 0.3235 |
| top-p 0.9 / 0.95 | 0.9726 / 0.8625 | 0.8898 / 0.8254 |
| typical 0.8 / 0.95 | **0.9933** / 0.8703 | 0.9382 / 0.8362 |
| η = 1e-4 / 0.05 | 0.7672 / 0.9965 | 0.7799 / 0.9466 |

Binoculars does not degrade at T=1.1; its score **inverts** (AUROC 0.02 means it now calls the machine text human with high confidence), because a cross-perplexity ratio reads high entropy as humanity. Note also that typical sampling at τ=0.8, the setting Meister et al. recommend for quality, is *maximally* detectable by Binoculars (0.9933). Typical sampling is a quality tool, not an evasion tool. The correlation analysis (Table 5) pins the mechanism: the average entropy of the generator over the text correlates **−0.910 / −0.935** with Binoculars / Fast-DetectGPT AUROC, far more than perplexity (−0.458 / −0.611). Zero-shot detectors are, to first order, entropy meters.

**Fishchuk and Braun (Nov 2023, arXiv 2311.01873), verified.** GPT-3.5 via the OpenAI API against the GPT-2 output detector, the OpenAI classifier and Turnitin. "Starting from a frequency penalty of 0.3-0.4 and a presence penalty of 1.0-1.2, the detection rate fell under 50%." Quality: "for values between 0.0 and 0.6, increases in value caused an incremental increase in text complexity while preserving quality and readability"; above 0.6, "punctuation mistakes and unclear wordings quickly increased." Joint tuning of both penalties dropped detection "for smaller values... thereby minimising the potential negative effects on text quality." Temperature and top-p above default were excluded from the design "because of the strong negative effects on text quality," and lowering them raised detection. The "perplexity and burstiness" prompt lowered detection but only the GPT-2 detector fell below 0.5; the OpenAI classifier and Turnitin stayed above it. Caveat: 2023 detectors, small n, and Turnitin is the only survivor of the three.

**Ippolito et al. (ACL 2020)** established the underlying trade-off: the decoding setting that most fools humans (top-k 40, human accuracy 0.64) is the one a BERT detector catches most easily (0.88), because top-k over-generates high-likelihood tokens. Pure sampling fooled BERT most (0.79) and humans least (0.71). A BERT trained on top-k and tested on pure sampling scored **43.8%**, below chance. This is the 2020 version of Dubois's Table 3.

### 2.4 Does any supervised detector key on entropy statistics?

No published probe of GPTZero's or Pangram's internals exists. The indirect evidence says supervised transformers are *not* primarily entropy meters: pure UID features match a supervised BERT on TuringBench (F1 0.88 vs 0.88) but lose badly on GPABenchmark (0.83 vs **0.98**), so the transformer is using substantial non-information-theoretic signal (GPT-who). Bolting surprisal-variance features onto RADAR lifts its AUROC from 0.62 to **0.90** (DiVEye, TMLR 2026), which implies RADAR was not already extracting them. Pangram's technical report contains no discussion of decoding parameters or entropy features at all.

### 2.5 What this section means for the hypothesis

Restoring entropy at generation is the one form of "noise" with strong, replicated support, and it is exactly what the owner's intuition is pointing at. But three qualifications are non-negotiable:

- The human band is narrow. T≈0.9-1.0 for a base model, η=1e-4 or top-p 0.95, and *no* repetition penalty above about 1.02 if we care about diversity metrics landing in the human range. T>1.0 and repetition penalty ≥1.05 buy detector evasion by producing text that is measurably abnormal, which a mixture-trained detector catches at 0.98.
- Against a supervised detector whose vendor has sampled the decoding space, the sampler alone is not durable. The durable part is the checkpoint: base-model text at default sampling is 96.7-98.8% human on GPTZero and Pangram (report 13).
- Zero-shot detectors are broken by this trivially and will stay broken. They are not our target and their numbers should not be used to evaluate the pipeline. Report 00's recommendation to hold Binoculars out of the reward is reinforced: an optimiser will learn to raise entropy and nothing else.

---

## 3. Token- and word-level noise

### 3.1 RAID, all twelve detectors

Report 13 gave the GPTZero column. The full table (RAID Table 16, accuracy at 5% FPR, verified by the token-noise sweep from the ACL PDF) is what matters, because it shows GPTZero's flatness is the norm for supervised detectors, not an outlier.

| Detector | Type | None | ArtDel 50% | Homoglyph | InsPara | Paraphrase | Misspell 20% | **Synonym 50%** | UpperLow 5% | Whitespace |
|---|---|---|---|---|---|---|---|---|---|---|
| RoBERTa-L GPT2 | supervised | 56.7 | 33.2 | 21.3 | 55.1 | **72.9 ↑** | 39.5 | **79.4 ↑** | 19.3 | 40.1 |
| RoBERTa-B GPT2 | supervised | 59.1 | 37.1 | 7.6 | 56.9 | 68.9 ↑ | 43.8 | **71.5 ↑** | 18.8 | 45.2 |
| RADAR | adversarially trained | 70.9 | 67.9 | 59.3 | 73.7 ↑ | 67.3 | 69.5 | 67.5 | 70.4 | 66.1 |
| GPTZero | commercial | 66.5 | 61.0 | 66.2 | 66.2 | 64.0 | 65.1 | 61.0 | 56.5 | 66.2 |
| Originality | commercial | 85.0 | 71.4 | **9.3** | 85.1 | **96.7 ↑** | 78.6 | **96.5 ↑** | 75.8 | 84.9 |
| Winston | commercial | 71.0 | 66.9 | 26.3 | 69.8 | 52.6 | 67.5 | 63.6 | 56.8 | 46.8 |
| Binoculars | zero-shot | 79.6 | 74.3 | 37.7 | 71.7 | 80.3 | 78.0 | **43.5** | 73.8 | 70.1 |
| Fast-DetectGPT | zero-shot | 73.6 | 64.7 | 51.4 | 72.0 | 71.8 | 70.7 | **34.0** | 60.4 | 64.4 |
| GLTR | zero-shot | 62.6 | 52.1 | 24.3 | 61.4 | 47.2 | 59.8 | **31.2** | 48.1 | 45.8 |
| ZeroGPT | commercial | 65.5 | 59.7 | 82.4 | 64.9 | 46.7 | 64.7 | **18.8** | 54.5 | 64.2 |

Three readings. **Synonym swap is the best semantic-preserving attack in the table, and only against zero-shot detectors**: Binoculars −36, Fast-DetectGPT −40, GLTR −31, ZeroGPT −47. Against RoBERTa-Large it is **+22.7 toward AI** and against Originality **+11.5**. RAID attributes this to the substitution model (BERT) being contemporaneous with the detectors' training generators, so the attack pushes text *toward* the detector's training distribution. **Homoglyphs** are the strongest attack overall (Originality −76, RoBERTa-B −52) and GPTZero is at −0.3 because it normalises Unicode. **RADAR**, the one adversarially trained detector, is the flattest row: worst case −11.6, synonym −3.4, article deletion −3.0. Adversarial training against noise works and every commercial vendor now does it.

RAID reports **no quality metric for any attack** — no perplexity, similarity, grammaticality or human rating. The attack rates (50% article deletion, 50% synonym swap) were set by "manual review." Every RAID delta is quality-blind.

### 3.2 Follow-ups, 2024-2026

| Attack | Detector | Metric | Clean → attacked | Quality cost | Source |
|---|---|---|---|---|---|
| Keystroke-realistic typos (2-6 edits per 120 tokens; substitution 55.6%, deletion 23.0%, insertion 20.3%, transposition 1.1%) | GLTR / LogRank | relative AUROC | 100 → **2.1 / 2.6** | perplexity ×~2 | Stumbling Blocks, 2402.11638 |
| same | **fine-tuned DeBERTa / SimpleAI RoBERTa** | relative AUROC | 100 → **108.4 / 111.6** (improved) | same | same |
| PWWS synonym / DeepWordBug / Pruthi char | HC3 RoBERTa | accuracy | 100 → 69.0 / 76.5 / 78.5 | not reported | 2406.01179 |
| RAFT (10% words, LLM-guided) | Fast-DetectGPT | AUROC | 0.990 → **0.0006** | PPL 8.5 → 17.6-22.5; MTurk indistinguishable | 2410.03658 |
| HMGC (white-box gradient substitution) | HC3 RoBERTa | AUC | 99.63 → **51.06** | PPL +6.17; Flesch −7.7% | 2404.01907 |
| HMGC after 7 rounds adversarial retraining | same | AUC | 49.05 → **88.58** | attack cost 9 → 33 s/sample | same |
| HMGC | Binoculars / **RADAR** | AUC | 0.986 → 0.419 / **0.947 → 0.933** | PPL 16.8 → **62.5** | TH-Bench, 2503.08708 |
| Homoglyph (SilverSpeak) | ArguGPT RoBERTa | MCC | 0.94 → **0.00** | none measured | 2406.11239 |
| Homoglyph (SilverSpeak) on 2026 commercial detectors | GPTZero | human probability | in one example **99.9% → 18.0%**; verdicts unchanged in 4 of 5 | — | Base Models Look Human, 2605.19516 |
| Recursive DIPPER paraphrase, 5 rounds | RoBERTa-Large | AUROC (OPT-1.3B/XSum) | 0.998 → 0.945 → **0.900** | PPL 5.2 → 8.7; MTurk content 4.37 → 3.85, grammar 4.62 → 4.17 | Sadasivan et al. |
| same | DetectGPT / LogRank | AUROC | 0.900 → 0.244 / 0.943 → 0.181 | same | same |
| Light LLM paraphrase (L1) | all HC3-trained supervised detectors | AUROC | **L1 ≥ L0 "across all detectors and both domains without exception"** | — | Detecting the Machine, 2603.17522 |
| Heavy humanization (L2) | RoBERTa | AUROC | 0.9896 → 0.9621; no detector below 0.857 | — | same |

The pattern is exact. Zero-shot detectors collapse under any token noise. Supervised detectors are flat, and **light noise frequently raises their AI score**, because a small perturbation model superimposes its own fingerprint ("renders the composite text more detectable"). The only token-level attacks that move a supervised detector are white-box (HMGC needs the detector's gradients), cost 3.7× perplexity, and are closed by seven rounds of adversarial retraining.

### 3.3 Back-translation

| Pivot | Detector | TPR@1%FPR before → after | Source |
|---|---|---|---|
| 10-language combined | GPTZero | News 0.97 → **0.42**; Reddit-Llama 0.65 → **0.09** | ESPERANTO, 2409.14285 |
| same | **Pangram** | News 0.969 → **0.936**; Reddit-Llama 0.864 → 0.725 | same |
| same | ZeroGPT | 0.96 → 0.04 | same |
| m2m100 / nllb-200 | fine-tuned multilingual detectors | attack success rate **0.21 / 0.21**; AUC −5.1% | Macko et al., 2401.07867 |
| GPTZzzs synonym dictionary | same | 0.07 avg (0.41 English) | same |

Semantic cost: P-SP 0.950-0.985 single pivot, 0.872-0.960 combined. Back-translation is the one token-adjacent noise with a real GPTZero delta in the supervised era, and it costs Pangram three points. That asymmetry is the trap report 12 warned about: a lever tuned on GPTZero is invisible to Pangram, and a detector-in-the-loop optimiser will find it and stop there.

### 3.4 Noise as a learnable feature

Vendors have not made "noisy" a synonym for "AI", and the reason is instructive. DAMAGE (Pangram) humanizes *both* human and AI documents and labels humanized-human as human, "treating the model's response to humanization as an invariance rather than as a separate domain." Their ablation: training on noisy L2/L3 humanizers raises humanized recall from 98.26% to 98.92% but nearly doubles human FPR (3.47% → 6.00%). GPTZero's 2026 report describes adversarial training against multi-turn paraphrase-translation chains, DIPPER, TempParaphraser, nine named bypass services and HMGC-objective white-box substitution; its Deep Scan is built on the observation that "writers mainly replace some words with their synonyms, or remove them altogether. We mimic this behavior." PIFE (2510.02319) goes furthest: normalise the text, then feed the *magnitude of the normalisation delta* (Levenshtein, WER, Jaccard) as features, lifting TPR@1%FPR from 0.488 to **0.826**. None of these makes noise an AI signal outright, but all of them ensure noise is not a human signal either.

**Verdict on token noise:** net negative or neutral against every supervised detector measured. The only "moved the verdict" results are on zero-shot detectors, on 2022-era HC3 checkpoints, or via white-box gradients that we do not have for GPTZero.

---

## 4. Structural noise

### 4.1 What has actually been measured

| Manipulation | Detector | Before → after | Source |
|---|---|---|---|
| "Increase burstiness" prompt (full LLM rewrite varying sentence *and* paragraph length) | 7-detector mean | 39.5% → **15.9%**; GPTZero 26.4% → 16.7% | Perkins et al. 2024 |
| DIPPER order knob isolated (lex 60 → lex 60 + order 60), GPT-2 XL | **OpenAI RoBERTa** | 13.3 → **14.8 (up)**; OPT-13B 9.1 → 10.0; GPT-3.5 15.6 → 15.6 | DIPPER, 2303.13408, Table 1 |
| same | Watermark / DetectGPT / GPTZero (2023) | 68.9 → 57.2 / 8.7 → 4.6 / 7.1 → 1.2 | same |
| Order-knob semantic cost | P-SP | 96.9 → 94.3; GPT-3.5 94.2 → **88.4** | same |
| Insert paragraph breaks at 50% of sentence boundaries | GPTZero / Originality / RADAR / Binoculars | 66.5 → **66.2** / 85.0 → 85.1 / 70.9 → 73.7 / 79.6 → 71.7 | RAID Table 16 |
| Sentence-level vs word-level perturbation, equal budget | 11 detectors | "lower-level perturbations show greater attack success than higher-level ones" | Stumbling Blocks §6.3.2 |
| Sentence-shuffled perplexity ratio | RAID corpus | AI text 5.58 vs human 1.59 — AI text is 3.5× *more* order-fragile | 2604.25860 |
| Leave-one-feature-family-out, 11 families | linear SVM on MAGE | remove lexical richness **−0.131**; remove surface/structural **−0.005**; readability 0.000 | 2606.04177 |
| 14-feature stability across 8 LLMs × 5 domains | logistic regression | entropy and lexical diversity stable top-5 in 12-13/14 conditions; **burstiness 3rd globally but unstable**; mean sentence length near bottom | 2608.27855 |
| Sentence-length CV, human vs LLM | measurement | human **0.334 (SD 0.110)**; LLaMA 0.244; LLaDA 0.184-0.251 | 2507.10475 |
| Inter-sentence transition variance | CSFG detector | AI > human, r = 0.22-0.38; formulaic *human* news FPR 1.57% → **25.0%**; GPT-5 FNR **61.6%** | 2608.26694 |
| RST discourse-relation profile, LLM vs "humanized" | RACE | cosine **0.95** — discourse fingerprint survives humanizing | 2604.04932 |
| Random punctuation insertion, sentence-length Hurst | MFDFA | H = **0.81** for randomly punctuated text vs 0.65 for a real novel | 2508.19782 |

### 4.2 Reading

Perkins is the only "burstiness" number in the literature and it does not isolate structure: the prompt is a full GPT-4/Claude/Bard rewrite, lexis changes with it, baseline accuracy was already 39.5%, and the authors note outputs "sometimes resulted in overly short sentences that might not be suitable for formal or professional contexts." The clean isolation is DIPPER's order knob, and it exonerates supervised detectors completely: reordering at fixed lexical diversity cut watermarking and DetectGPT and left the OpenAI RoBERTa classifier flat or higher in all three generators, at a cost of up to 5.8 P-SP points. Paragraph insertion at 50% density moved GPTZero 0.3 points. Stumbling Blocks has no reordering attack at all (the taxonomy is character, word, span, sentence-paraphrase, prompt, co-generation), and its finding is that fine-tuned detectors "concentrate on long-form patterns... localized disturbances directly interrupt" them — which is why word-level beats sentence-level.

Two things are absent from the literature entirely: **any measured detector delta from em-dash removal or connective stripping**, and **any sentence-permutation attack benchmark**. The frequency evidence is strong (em dashes: GPT-4.1 10.62 per 1,000 words vs human 3.23; discourse markers +126% under 11 self-training generations), but no one has run the before/after. Desaire et al. found "however", "but" and "although" *more* frequent in human text than in 2023 ChatGPT, so even the direction of the connective folklore is model-dependent.

Structure is decisive for exactly one detector class: hand-engineered feature models that score layout. Desaire's 20-feature XGBoost reaches 99.5% with sentences-per-paragraph and words-per-paragraph as top features; J-Guard's rank-1 SHAP feature is mean sentences per paragraph; a PAN 2025 SVM's rank-1 feature is paragraph count (median 1 for AI true positives vs 17 for human true negatives). Those are not the detectors we face.

The fractal premise needs correcting. Alabdulmohsin et al. (2402.01825) measure the Hurst exponent of the bits-per-token series of *human* Pile text under an LLM (H = 0.70 ± 0.09); they never analyse model output and never mention detection. The one paper that measures sentence-length long-range dependence shows randomly punctuated text has *higher* Hurst than a real novel. Report 10's 1/f target (β ≈ 0.5) stands as a description of human text, and generating sentence lengths as fractional Gaussian noise remains the right way to hit the human band — but nobody should expect a detector to reward it. Note also that 2507.10475 measures human CV at 0.334 where report 10 measured 0.42-0.60; the corpora and units differ (the former is a small mixed-genre set), so treat this as a discrepancy to resolve on our own reference corpora, not as a contradiction.

**Verdict on structural noise:** moves scores on feature-based detectors, does not flip verdicts on supervised or commercial detectors, costs semantic fidelity when done by reordering, and leaves the RST discourse fingerprint intact. The synthesis's structural recommendations (CV in band, break the template, ≤3-4% connective openers) survive as *quality and target-matching* moves, not as detector levers.

---

## 5. Semantic and content noise

### 5.1 Content origin is detectable, and it is what survives humanization

| Finding | Number | Source |
|---|---|---|
| HART: content-only feature (GPT-4o neutralised outline) beats full-text detector on AI-content/human-expression text | Fast-Detect AUROC 0.711 → **0.855** | HART, 2503.00258 |
| HART: humanizers (Humbot, BypassGPT, Undetectable) shift text only along the expression axis; content axis unchanged | qualitative, Fig. 3 | same |
| Triospect on Humanize-16K after-attack | Fast-Detect 0.627 → **0.850**; TPR@1% 31% → 44% | 2606.31074 |
| Triospect fails against synonym substitution | no improvement (r ≈ 0.80 between axes) | same |
| D2C-Routing on AI-content/human-expression | content head **0.9477** vs expression head 0.6438 accuracy; features = entity chains, RST relations | 2608.27380 |
| LLM fingerprint survives paraphrase / translation / **summarization** | 5-way attribution 97.8% → 91.4% / 91.8% / **58.1%** (chance 20%); word-shuffled 88.9% | Sun et al., 2502.12150 |
| Pangram on idea-level detection | "Pangram cannot infer where an idea came from" | Pangram blog |

HART's "content" is operational, not semantic: it is what survives a GPT-4o paraphrase-to-plainness. That is exactly why humanizers cannot touch it — the neutraliser undoes their rewrite. D2C goes further with real content features (entity recurrence, discourse relations) and finds content origin far easier to read than expression origin. Sun et al. is the strongest evidence that model identity lives partly in *what is said*: 91% attribution after full paraphrase, 89% on a bag of words, 58% after summarisation destroys nearly all form.

### 5.2 Content-level manipulations that have been measured

Only one. Russell et al.'s expert "evader" prompt is genuinely content-level, and it reads like the owner's hypothesis inverted: add specific places, people, brands, facts, metrics; "always opt for specifics over broad details"; add references to darker topics; kill scene-setting introductions and tidy conclusions.

| Detector | o1-Pro baseline TPR → humanized | Source |
|---|---|---|
| GPTZero | 76.7 → **46.7** | Russell et al., 2501.15654 |
| Pangram | 100 → 90.0 | same |
| Pangram-Humanizers | 100 → 96.7 | same |
| Fast-DetectGPT / Binoculars | 86.7 → 23.3 / 73.3 → 6.67 | same |
| Expert human majority vote | 96.7 → **100** | same |

It halved GPTZero, gutted the zero-shot detectors, cost Pangram ten points, and made expert humans *more* accurate. It was not tested against HART/Triospect/D2C.

### 5.3 Tangents, digressions, hedges, redundancy, self-correction

There is no study that isolates any of these as a manipulated variable with before/after detector scores. What exists is:

- **Cue frequencies from expert annotators** (Russell, Table 17): vocabulary 53.1%, sentence structure 35.9%, grammar/punctuation 24.8%, originality 23.7%, quotes 22.3%, **clarity 19.5%** (coded as "over-explains or includes irrelevant details" — an *AI* cue), names and titles 11.7% (generic entities read as AI, unique or branded as human), tone 9.3%, topics 3.1%. Irrelevant detail is a tell for AI, not for humanity, in the only cue data that exists.
- **Hedging and epistemic markers are genuinely depleted in LLM text.** Herbold et al.: epistemic markers per sentence 0.06 human vs 0.02 GPT-3 vs **0.00 GPT-4**; discourse markers 0.57 / 0.52 / 0.36; modals 10.84 / 8.97 / 6.12. Reinhart confirms instruct models under-use epistemic constructions while base models sit at human rates. Jiang and Hyland report ChatGPT uses "significantly lower frequency of interactional metadiscourse" — **numbers unverified** (paywalled). Adding hedges toward the Hyland band (8.2-20 per 1,000 words, report 15) is therefore a target-matching move with zero grade cost (Lee and Deakin find successful essays hedge *more*; counts unverified). There is no evidence it moves Pangram or GPTZero.
- **Redundancy and self-correction** ("or rather", "to put it differently"): nothing. Not one paper.
- **Quality cost is measured and large.** Herbold's human raters scored human essays *below* both GPT versions (3.69 vs 4.36 / 4.68 out of 7), with the largest gap in language mastery (d = −1.43). Making text more "human" in the digressive sense makes it worse by the raters' own standard.

**Verdict on content noise:** the content axis is real and is the part that survives every expression-level humanizer. But the only measured content manipulation that helps is *adding specificity and removing formula*, which is what report 00's conflict matrix already rates as the best joint move. Adding noise to content — tangents, filler, redundancy — is unsupported and points the wrong way on the one cue dataset that exists.

---

## 6. The "too clean" hypothesis, directly

### 6.1 Perplexity era versus supervised era

| Finding | Detector, date | Number | Source |
|---|---|---|---|
| **Perplexity era** | | | |
| US Constitution preamble / Genesis | GPTZero, Apr 2023 | 96.2% / 88.2% AI | secondary blog write-ups of a viral screenshot; **anecdote** |
| TOEFL essays flagged | 7 detectors, 2023 | 61.2% mean FPR; 97.8% flagged by ≥1 | Liang et al., 2304.02819 |
| TOEFL essays after "enhance word choices" | same | 61.2% → **11.8%**; US 8th-grade after "simplify" 5.2% → **56.7%** | same |
| GPTZero re-run of Liang's code | GPTZero, Oct 2023 | 1 of 91 = **1.1%** | GPTZero blog |
| **Supervised era** | | | |
| TOEFL / ELLIPSE (3,907) / ICNALE (5,600) | Pangram | **0.00% / 0 / 0.09%** FPR | Pangram tech report, 2402.14873 |
| ELL vs native, ~2,000 samples | Turnitin 2024 | 1.4% vs 1.3% | Turnitin blog (search snippet; page 403'd) |
| Human peer reviews | Pangram / GPTZero / Fast-DetectGPT | 0% / 1% / 0.2% FPR | 2603.20450 |
| Pre-LLM abstracts, 2013-15 (n=306) | Pangram 3.2 / GPTZero, Aug 2026 | **0.0% / 0.0%** | Karr et al., 2608.11256 |
| Human abstracts 2023-25 (n=336) | same | **14.9% / 8.9%** flagged | same |
| same, by field (Pangram) | | Chemistry 2.1%, CS 9.6%, PoliSci 24.1%, **Theology 26.6%** (p<0.001) | same |
| Detector score vs lexical formality | Pangram + GPTZero | long-token ratio ρ **0.30-0.35**; AWL ratio 0.32-0.35; both p<0.001 | same |
| Professional native editing of non-native manuscripts, 135,389 pairs, 13 detectors | various, Aug 2026 | FPR 0.0%-100.0% across detectors; editing *raised* scores on some and *lowered* on others | 2608.26710 |
| Likely-autistic vs general Reddit writers | **OpenAI GPT-2 detector (2019)** | length-normalised OR +50%; perplexity and burstiness near-equal between groups | 2607.14729 — perplexity-era detector with a 2026 date |
| Working Educators, 247 verified essays | GPTZero, fall 2025 | 23% FPR overall, **41% ESL** vs 18% native | report 06 |

The perplexity-era story is closed: the ESL bias was real and driven by low perplexity from limited lexical range, and the supervised vendors engineered it down to ~0-1% on their own benchmarks. What re-emerged is a **register effect**, not an error effect. Karr et al. is the cleanest evidence: zero false positives on pre-LLM abstracts, but a monotone formality gradient from chemistry to theology and a ρ≈0.3 correlation with academic-word density on both commercial detectors. Two caveats: the 2023-25 human abstracts may contain undisclosed AI assistance (the authors acknowledge this), and the Working Educators 41% ESL number on GPTZero conflicts with Pangram's 0% — different detector, different population, and GPTZero's Model 3.15b (Dec 2025) explicitly traded false positives on formal prose for humanizer recall (report 06).

### 6.2 The reverse direction: polishing human text triggers detectors

| Manipulation | Detector | Flag rate | Source |
|---|---|---|---|
| "Refine abstract only" light Gemini edit of human abstract | Pangram 3.2 / GPTZero | **64.4% / 37.6%** (2013-15); 80.1% / 48.5% (2023-25) | Karr et al. |
| "Extremely minor" GPT-4o polish of human text | GLTR / RoBERTa-Large | 40.9% / 47.7% | APT-Eval, 2502.15666 |
| same, Llama2-7B polish | GPTZero / Pangram | **64.7% / 42.6%** | same |
| 1% of words modified | GLTR / Fast-DetectGPT | 26.9% / 10.1% | same |
| LLM-polished-for-grammar peer reviews | Pangram / GPTZero | 3.1% / 3.4% | 2603.20450 |
| 1,000 human files through a grammar checker (~35% of text changed) | Copyleaks | **0 / 1,000** flagged | Copyleaks blog (vendor) |
| Human IELTS essays + Grammarly grammar-only vs rephrase | Originality | grammar-only → still human; rephrase → flagged | Originality blog (vendor) |
| Expert human editing of MGT vs LLM editing, 33 detector configs | Beemo, 2411.04032 | expert editing evades; LLM editing does not | (per-detector numbers not extractable) |

This is the "Polished" class in GPTZero's hierarchy and the AI-assisted class in Pangram, and it is the mechanism behind the intuition that clean text gets flagged: *LLM-cleaned* text gets flagged, because the cleaner leaves its fingerprint. Pure grammar correction (Copyleaks, Originality grammar-only) does not. The asymmetry Karr et al. name is real and uncomfortable: honest light AI polishing is flagged at up to 80% while a commercial humanizer's output survives at <4%.

### 6.3 Does injecting errors help on current detectors? No

| Test | Detector | Result | Source |
|---|---|---|---|
| Phrase swap → word swap → full first-sentence rewrite of AI essay | Pangram, Aug 2025 | **99.9% AI at every stage** | Pangram blog |
| Typos / errors alone | Pangram, Oct 2025 | "do not evade"; humanizers aggregate >90% caught | Pangram blog |
| 19 humanizers whose worst tier is defined by injected typos, capitalisation errors, broken grammar | DAMAGE | GPTZero 99.73 → 60.04; Binoculars 94.15 → 28.23; **DAMAGE 100 → 98.26** | 2501.03437 |
| Keystroke-realistic typos | fine-tuned DeBERTa / SimpleAI RoBERTa | relative AUROC **108 / 112** | Stumbling Blocks |
| Error count as sole feature | GECScore | **98.62% AUROC**, beats RoBERTa-base (69.95) on XSum; 87.7-88.3 under T5 paraphrase | 2405.04286 |
| Mean grammatical errors per text | PAN CLEF 2025 SVM | AI **6.41** vs human **14.08** | 2603.23146 |

DAMAGE is the decisive counter-example to the naive story. The L3 humanizers *do* inject typos and broken grammar, GPTZero *does* drop 40 points on their output, and a purpose-trained detector holds at 98.26% on exactly those texts. The errors are correlated with evasion; the distributional rewrite is causing it. GECScore shows the flip side: error rate is a real aggregate separator (10-30 vs 0-10 edits), so a detector *can* use it — but a 500-word essay at Lunsford's 2.45 per 100 words carries ~12 errors, squarely in the human band, and "moving it by a few typos is far below the threshold" of any detector that is also reading lexis and entropy.

**The gap.** There is no published controlled study that injects errors — and only errors — into LLM text and measures GPTZero, Pangram, Originality or Turnitin before and after. Only vendor demos (Pangram) and confounded humanizer studies (DAMAGE) exist. Section 9 proposes running it ourselves, because the result would be new data.

---

## 7. Learned noise

### 7.1 Learning a human error distribution is a solved problem

The synthetic-data-for-GEC literature is precisely "learn a corruption distribution from human text and apply it to clean text", and it publishes the distributions.

**ERRANT error-type shares, W&I+LOCNESS (native and learner, BEA-2019):**

| Type | Train % | Dev % | Test % |
|---|---|---|---|
| **PUNCT** | **17.16** | 19.37 | 16.73 |
| OTHER (lexical substitution) | 12.76 | 12.84 | 15.69 |
| **DET** | 11.25 | 10.43 | 10.41 |
| **PREP** | 9.79 | 9.70 | 8.33 |
| VERB:TENSE | 6.07 | 6.20 | 5.43 |
| VERB | 5.86 | | |
| ORTH (casing, spacing) | 4.77 | 4.61 | 8.03 |
| NOUN | 4.36 | | |
| NOUN:NUM | 4.05 | | |
| **SPELL** | **3.74** | 5.07 | 4.63 |
| VERB:FORM | 3.56 | | |
| PRON | 2.64 | | |
| VERB:SVA | 2.23 | | |
| WO (word order) | 1.64 | | |

Edit operations: replacement 59-64%, missing 19-26%, unnecessary 10-19%. Lunsford and Lunsford 2008: 2.45 errors per 100 words, "wrong word" ≈14% of errors (rank 1), spelling 6.5% (rank 5) — **partly secondhand**, the full top-20 table could not be retrieved.

Stahlberg and Kumar (2021) built the tool: a seq2seq corrupter conditioned on an ERRANT tag, tags sampled from a target corpus's empirical distribution, applied to C4 to make 200M corrupted sentences. Matching realistic error *types* is worth **+9.2 F0.5** over untagged corruption (48.0 vs 38.8, real data 50.4), and when the target is *native* English, synthetic data with a native tag distribution **beat real parallel data** (42.9 vs 42.1). The corrupter genuinely captures native-writer noise. Its shape is punctuation, lexical choice, determiners, prepositions and casing, not typos. If you inject character-level misspellings you are modelling 4% of the distribution — and the 4% that both faculty and detectors treat as most anomalous.

### 7.2 Un-editing and first-draft simulation do not exist

IteraTeR (31,631 revisions, 196,987 edits, intentions CLARITY 39.9%, FLUENCY 23.4%, MEANING-CHANGED 22.3%, COHERENCE 9.8%, STYLE 3.2%), WikiAtomicEdits and ScholaWrite (~62K keystroke changes; Planning 9.7%, Implementation 65.0%, Revision 25.3%) all encode draft→final. **No paper trains the inverse.** ScholaWrite's finding that an instruction-tuned Llama-8B predicts human writing intentions at 0.13 weighted F1 even after fine-tuning (encoder: 0.64; GPT-4o zero-shot: 0.08) supports "LLMs cannot model the drafting process", but it is an intention-prediction result, not evidence that LLM prose is detectable for lacking draft residue. The data to build a reverse-CLARITY conditional corrupter (Stahlberg's recipe with IteraTeR intentions as tags) is sitting there, unpublished.

### 7.3 No evasion paper has ever used learned errors

AuthorMist, StealthRL, MASH, Adversarial Paraphrasing, HIP, StyleShield — all paraphrase or style transfer, none injects errors, none constrains an error distribution. An arXiv full-text search for "ERRANT error type distribution" returns exactly one paper (Stahlberg and Kumar). The closest thing to the key experiment is Stumbling Blocks' keystroke-realistic typo mix, and its answer is discouraging: metric detectors collapse to 2-10% relative AUROC, **supervised DeBERTa and RoBERTa sit at 96-112%**.

So the genuinely untested hypothesis is narrow: **do ERRANT-typed, native-distribution, non-orthographic errors (punctuation, determiner, preposition, lexical choice) at ~1-2.5 per 100 words move a Pangram- or GPTZero-class detector, and do they differ from random perturbation at matched edit budget?** Nobody has run it. Given section 3's pattern (light perturbation raises supervised scores) and section 8's quality numbers, the prior should be that it does not help, but it is cheap to test and the result is publishable either way.

---

## 8. Noise versus quality

### 8.1 What each noise type costs on a grade

| Noise | Grade evidence | Number | Source |
|---|---|---|---|
| Mechanical errors, 0 → 2 → 4 per 100 words | community raters, 7-pt writing quality, identical content | **5.79 → 3.61 → 2.95** (η² = 0.39); rating of *substance* 5.85 → 4.46 → 4.03 | Bleske-Rechek et al. 2019 |
| Informal language / non-native phrasing / grammar+spelling errors | LLaMA 3.3 70B grader, 1-10 scale, base 8.45 | **−1.90 (d=4.25) / −1.35 / −0.60**; hierarchy persists when told not to penalise style | 2603.18765 |
| Error features vs TOEFL proficiency | regression | error group alone r = 0.55; non-spelling errors weight **−1.51**, spelling −0.72 | 1612.00729 |
| Errors in e-rater's ~99k essays | prevalence | 92-97% of real essays already carry a spelling flag; missing comma 42-61%; fragment 12-21% | ETS RR-17-04 |
| Features *positively* correlated with human scores | e-rater | **missing comma, hyphen error, passive voice** | same |
| Sentence-initial And/But | e-rater | fires on only 1.4-3.8% of essays, threshold-based | same |
| Error type salience, faculty and professionals | error-gravity studies | lexical/wrong-word most serious (3.38/5); spelling among most acceptable to faculty; fragments and misspellings most bothersome to business readers | Santos 1988; Vann et al. 1984; Beason 2001; Gubala et al. 2020 |
| Mechanics weight in rubric | AP English Language | at most **1 of 6 points**, and only if errors "interfere with communication" | AP rubric |
| same | IELTS Task 2 | Grammatical Range and Accuracy = flat **25%** | IELTS |
| Lexical sophistication vs score | multiple | word range explains 17-43% of variance; content-word frequency r = −0.29 to −0.44; MTLD r = −0.35 | CJAL review |
| Hedges vs grade | Lee and Deakin 2016 | successful essays hedge *more* (**counts unverified**, paywalled) | |
| Humanizer output vs original, GPT-4o fluency win rate | DAMAGE | L1 **26.0%**, L2 14.7%, L3 2.7% — even the best tier loses 74% of the time | 2501.03437 |
| Humanizer meaning preservation | HumanizerBench, Sep 2026 | WriteHuman meaning 72.3, readability 56.9 — the same tool DAMAGE puts in L3 | humanizerbench.com |

Four conclusions. **The cost of visible errors is front-loaded**: the first two per 100 words cost 2.18 of 7 points, the next two 0.66. At any rate high enough to shift a detector's statistics we are past the cliff. **Error type matters more than count**: lexical errors are most serious to every human rater; spelling is forgiven by faculty but is the highest-weighted and most reliably measured feature in automated scoring, so its cost depends on who grades. **Some "errors" are free or positive**: missing commas, hyphen inconsistencies and passives correlate positively with human holistic scores in ETS's own data, and sentence-initial coordinators are invisible below a threshold. **Rubric structure dominates**: mechanics is ≤17% on AP and only when it impedes comprehension; 25% flat on IELTS. There is no universal answer to "what does noise cost"; the answer is rubric-specific.

No study puts humanized essays in front of rubric-trained human graders. DAMAGE, Adversarial Paraphrasing and HumanizerBench all use LLM judges, and the LLM-grader bias study shows those judges penalise informality three times harder than errors and cannot be instructed out of it. The two available quality signals on the same tool (WriteHuman: DAMAGE L3 "elementary school level, incomplete sentences" vs HumanizerBench #1 overall) directly contradict each other.

### 8.2 Noise that is invisible to a grader but visible to a detector

Exactly one class qualifies, and it is not error noise. Adversarial Paraphrasing (2506.07001), perplexity under Llama-3.1-8B:

| Text | Perplexity |
|---|---|
| Original AI | 14.94 ± 10.40 |
| **Human** | **15.02 ± 7.71** |
| Simple LLM paraphrase | **9.28 ± 3.86** |
| Adversarial paraphrase | **14.26-17.11** |

Simple paraphrase crushes perplexity to 9.28 — the "Polished" signature from section 6.2 — while adversarial (detector-guided) paraphrase restores it to the human band with **87% of outputs rated 4-5 of 5** and no statistically significant quality gap against simple paraphrase, cutting TPR@1%FPR by 64-88% across detectors. This is perplexity restored via *lexical choice*, not via mechanical error. It changes which correct word is used, not whether the word is correct. That is the operating region the owner's hypothesis is actually pointing at: the text should be less predictable, not less correct.

The conflict matrix in report 00 (section 4) needs one amendment. The row "Low-visibility punctuation slips, 0.3-0.8 per 100 words: detector benefit M" should be downgraded to **detector benefit unknown, untested on any supervised detector**. Its grade cost stays low (missing commas correlate positively with score), so it remains permissible as a floor-setting device, but it should not be counted on for detector movement until section 9's experiment runs. The row "Leave text perfectly error-free: Negative" stands, on GECScore and the PAN error counts, with the mechanism restated: the fix is to *not over-polish* — LanguageTool rate inside the human band achieved by leaving native-shaped imperfections in — rather than to inject anything.

---

## 9. Ranked recommendation for our pipeline

Ranked by strength of evidence × expected detector movement on supervised commercial detectors, net of grade cost.

**1. Restore entropy at the source, not in the text. (Strong evidence; large effect; zero grade cost if kept in band.)** Generate and paraphrase from a **base checkpoint** (report 13: 96.7-98.8% human), sampled at **T = 0.9-1.0 with η = 1e-4 or top-p 0.95**, no repetition penalty above ~1.02, and optionally a **frequency penalty of 0.3-0.6** to be validated on our own diversity metrics. Gate every output on MTLD, Simpson, Zipf α and perplexity landing inside the human bands from Dubois Table 2 (MTLD ≈ 95, Simpson ≈ 0.66, α ≈ 1.20, PPL ≈ 14 under the same scorer), because the RAID-era repetition-penalty trick produces text that is 4× too diverse and a mixture-trained detector catches it at 0.98. Do not use T > 1.0. Do not use typical sampling as an evasion device (it is maximally detectable to Binoculars at τ = 0.8 and is a quality tool). Expect the sampler's contribution to decay as vendors sample the decoding space; expect the checkpoint's contribution to last longer.

**2. Perturb lexical choice under a perplexity-band gate, never correctness. (Strong evidence; the only grader-invisible noise.)** Add a hard gate to Stage 3's reward: perplexity under a fixed scorer inside the human band (roughly 12-18 on the Adversarial Paraphrasing scorer; recalibrate on our reference corpora), with a *penalty on both sides* so the optimiser cannot flee to high-entropy nonsense. Pair with the existing NLI, cosine ≥ 0.88 and quality gates. This turns the RL loop into the adversarial-paraphrasing mechanism (PPL 9.28 → 14-17 at 87% quality) rather than a synonym-swapper, and it is the concrete form of "make it less predictable without making it worse."

**3. Set an error floor by not over-polishing; do not inject. (Moderate evidence for the floor; no evidence for injection.)** Keep the LanguageTool rate inside the genre band (report 09: 2.45 per 100 words for students; lower for published academic prose) by suppressing the final "fix everything" pass rather than adding anything. If any imperfection is left in deliberately, it must be **ERRANT-native-shaped** — a missing comma, an inconsistent hyphenation, a determiner slip, a preposition choice — never spelling, never agreement, never in a quotation, citation, thesis, first or last sentence. Expected detector benefit: unknown. Expected grade cost: near zero for missing commas and hyphens (positively correlated with score in ETS data), high for anything lexical.

**4. Match structural bands; do not randomise them. (Structural noise has no measured effect on supervised detectors.)** Report 10's targets stand as descriptions of human text and as quality guards: sentence-length CV 0.42-0.60, lag-1 autocorrelation 0.01-0.10, paragraph-length CV 0.42-0.71, ≤3-4% connective openers, 1/f realisation via fractional Gaussian noise. Expect them to keep us out of the "anomalously bursty" region and to help quality; do not expect them to flip a verdict. Do not shuffle sentences (semantic cost, zero supervised effect, and AI text is 3.5× more order-fragile). Breaking the topic-sentence/summary-sentence template is worth keeping because it targets a measured signal (inter-sentence transition variance, r = 0.22-0.38) and raises the grade — but its detector effect is unmeasured.

**5. Content: specificity, stance and hedging within the Hyland band; no tangents, no redundancy, no self-correction. (Content origin is detectable; only specificity has evidence.)** Russell's evader prompt (specific places, people, metrics; kill formulaic openers and closers) halved GPTZero at a 10-point Pangram cost and is already the matrix's best joint move. Hedges and epistemic markers are genuinely depleted in instruct-model text (GPT-4: 0.00 per sentence) and raise grades; bring them to 8-20 per 1,000 words per discipline. Irrelevant detail is an AI cue at 19.5% frequency in expert annotations; redundancy and self-correction have no evidence. Accept that HART/Triospect/D2C-class detectors read content origin and cannot be moved by any expression-level noise; this is report 12's finding and it is unchanged.

**6. Do not build any of the following.** Typos and spelling errors (supervised detectors +8-12% relative AUROC, grade −2.18/7 at 2 per 100 words). Homoglyphs and Unicode (GPTZero normalises; one example went 99.9% → 18.0% human). Random synonym substitution (RoBERTa-L +22.7, Originality +11.5 toward AI). Word dropout and article deletion (−5.5 on GPTZero at 50% deletion, unreadable). Case swapping. Sentence shuffling. Temperature above 1.0. Repetition penalty ≥ 1.05. Light LLM "humanizing" passes on already-good text (L1 ≥ L0 on every HC3 detector; Karr's 64-80% flag rate on light polish). Tangents and filler.

**Experiments to run, because nobody has and the answers are cheap.** Each is a same-document A/B against the live GPTZero API and Pangram, ≥100 documents, verdict rate at the user-visible threshold as the metric (report 13: measure the verdict, not the probability).

1. **Frequency/presence penalty sweep** (0.0-0.8) from a base checkpoint at T = 0.95, with MTLD/Simpson/PPL recorded per setting. This replicates Fishchuk and Braun on 2026 detectors and tells us where the human-band/evasion overlap is.
2. **ERRANT-typed error injection** at 0, 1, 2 per 100 words using Stahlberg and Kumar's tagged corrupter (or a rule-based approximation for PUNCT/DET/PREP), versus random typos at matched edit budget. This is the section 7.3 experiment.
3. **Em-dash-only and connective-only ablations**: remove em dashes; strip sentence-initial *Furthermore/Moreover/Additionally*; measure separately. No published number exists for either.
4. **Template-break ablation**: rewrite only paragraph-final summary sentences and paragraph-initial topic restatements, holding everything else fixed.

Each costs a few hundred API calls. Results feed directly into the calibration harness that report 00 already names as the only number that matters.

---

## 10. What could not be verified

- Exact scalar surprisal mean/variance for human vs LLM text: every paper (GPT-who, DiVEye, Venkatraman et al.) publishes histograms only.
- Mikhaylovskiy's Zipf/Heaps temperature-window numbers: ACL Anthology unreachable from this environment; taken from report 04.
- Verbalized Sampling diversity figures: HTML-summariser pass, medium confidence.
- "More Aligned, Less Diverse?" (2025 instruct models below human lexical diversity): agent-reported, not independently fetched.
- Jiang and Hyland metadiscourse frequencies; Lee and Deakin hedge counts by grade; Hairston 1981 and Gray and Heuser 2003 error rankings; Hyland and Anan severity ratings: all paywalled or 403; only directions confirmed.
- Lunsford and Lunsford 2008 full top-20 error table: encrypted PDF; aggregate rate and two ranks are secondhand.
- Turnitin's 1.4% vs 1.3% ELL figure: from a search snippet, page returned 403.
- Beemo per-detector before/after numbers; APT-Eval full grid; MixSet per-detector numbers.
- The US Constitution and Genesis GPTZero scores: secondary write-ups of an April 2023 screenshot, no primary source.
- The "GEDE benchmark" cited in report 13 via Pangram 4 Table 21: the "too clean" sweep could not locate a benchmark by that name.
- e-rater's operational weight for mechanics: Attali and Burstein's table digits corrupted in every extraction attempted.
- Any human-graded quality assessment of humanizer output; any error-injection-only study on a commercial detector; any em-dash, connective, shuffle or template-break ablation against any detector: **these do not exist in the literature as of this date**, as far as roughly 110 sources and exhausted search budgets across seven parallel sweeps could establish.

---

## 11. What to change in our pipeline, and the expected effect on the pass rate

**The metric.** A recommendation counts only if it flips verdicts: p(AI) crossing the detector's threshold to "human" on a supervised transformer detector, held out from whatever guided the edit. Report 17 measured **0 of 9 flips** for both the rule-based path (`transforms.py`) and the LLM path (`llm.py` + `pipeline.py`) on desklib; the one method that flipped 9 of 9 on desklib (`gsub`, HMGC+RAFT reimplemented against desklib itself) flipped **0 of 9** on fakespot, academic and fast, and produced damaged English. Report 17's standard applies here: quote held-out flips or quote nothing.

Most of the literature does not report flips. Two metric families are flip-equivalent and are used below: **accuracy or TPR at a calibrated FPR** (RAID's accuracy@5%FPR, Pangram's TPR@1%FPR, Russell's TPR at the vendor threshold) is the fraction of AI documents that stay above the threshold, so a drop of X points means X% of documents flipped; **AUROC deltas and mean-probability drops are not** flip evidence and are marked as such. DUPE (report 13) is the cautionary case: FNR 22-51% under a lenient reading, **1 of 400** documents actually labelled Human.

**Module map.** `transforms.py` = deterministic edits (`strip_formal_connectives`, `replace_ai_vocabulary`, `break_paragraph_template`, `break_parallelism`, `vary_sentence_length`; `HumanizeConfig`). `llm.py` = prompting and candidate specs (`CandidateSpec`, `CANDIDATE_SPECS` ×12, `FREEFORM_SPECS` on `Qwen2.5-3B-4bit` base, `MlxBackend._sampler` which currently exposes only `temp` and `top_p`). `guided.py` = detector-guided decoding (`reweight_logprobs`, `GuidedConfig`, guide `e5-small-lora`). `pipeline.py` = hard gates (`_gate_candidate`: length ratio, content overlap, invariants) and selection (`_rank_key`: p(AI) + `quality_weight` × (CV distance + drift)). `pretrained.py` = the AuthorMist checkpoint. Local detectors: desklib (yardstick), fakespot, academic, fast, RADAR; `PerplexityDetector` (gpt2); `features/lexical.mtld`.

**Local confirmation (report 19, same date).** A sibling report ran the noise experiments locally against desklib with fakespot, academic, RADAR and fast held out, and its results match this review's predictions. Instruct-model temperature from 0.7 to 1.2 left **28-29 of 30** paragraphs flagged while Flesch collapsed (−3 → −12); `min_p` 0.05-0.10 made output *more* detectable (0.991-0.995). The base checkpoint at exactly T=1.0 scored **0.199 on desklib, 0.052 fakespot, 0.001 academic (3 of 24 flagged)** — the base-model result reproduced at 3B — but with topic overlap 0.11 versus 0.28, i.e. fluent invention, and it mode-collapsed at T=0.7 (0.892) and T=1.2 (0.952). Visible noise (function-word deletion 13/30, WordNet synonyms 12/30, typos 10/30, casing 9/30, punctuation slips 7/30, hedges 6/30) flipped only *boundary* text (desklib 0.731 before) and **0 of 40 on saturated text (0.997)**; plausible masked-LM substitution flipped **1/30 at every dose**; sentence shuffling **0/30**, connective removal 2/30, CV reshaping into the 0.42-0.60 band 3/30. Light instruct-model polish of human text flipped **11 of 33** human paragraphs to AI (the Karr/APT-Eval "Polished" effect, locally). Every one of those numbers lands where sections 2-6 say it should.

### 11.1 Ranked by expected pass-rate lift per unit of grade cost

| Rank | Change | Module and change type | Does the literature show **verdict flips** on 2025-26 supervised detectors? | Grade cost | Lift ÷ cost |
|---|---|---|---|---|---|
| **1** | **Fine-tune the base paraphraser on (AI-paraphrase → real human) pairs, HIP-style, and iterate with a stop-at-first-held-out-flip rule** | `llm.py`: replace `FREEFORM_MODEL` with a LoRA'd `Qwen2.5-3B` base; new dataset job. **Training-data change.** `pipeline.py`: keep-best archive across rounds already exists (`rounds`). | **Yes.** Base Llama3-8B judged human **96.7% GPTZero / 98.8% Pangram** vs instruct 30.3% / 17.1% — verdict-level, 2026. HIP reached **~100% human on GPTZero by round 10** with judged semantics 6-10/10. Report 17 confirms prompt-only base did **0/9** locally; the fine-tune is the mechanism, not the checkpoint alone. | Medium until gated: HIP's semantic score has a wide range; our existing invariant and overlap gates apply unchanged. | **Highest.** The only lever with flip-level evidence on current commercial detectors. |
| **2** | **Content-specificity candidate spec** (Russell's evader: add specific places, people, dates, metrics; delete scene-setting opener and tidy closer; opt for specifics over generalities) | `llm.py`: **new `CandidateSpec`** (index 12) plus a `mixed`-ladder slot. `pipeline.py`: no change. | **Yes, modest.** GPTZero TPR at vendor threshold **76.7 → 46.7** (30% of documents flipped, n=30, Jan 2025); Pangram **100 → 90.0**; Pangram-Humanizers 100 → 96.7. Expert humans 96.7 → 100 (not fooled). Not tested against desklib. | **Negative (raises grade):** report 00's best joint move; AP rubric rewards specific evidence. | **Best ratio.** Moderate absolute lift, zero or negative cost. |
| **3** | **Sampler in the human diversity band, plus mild presence/frequency penalty**, replacing temperature as the diversity axis of `FREEFORM_SPECS` | `llm.py`: `MlxBackend._sampler` add `min_p` (mlx `make_sampler` supports it) and `repetition_penalty` via `make_logits_processors`; a presence-penalty logits processor is ~20 lines. **Config change + new candidate specs.** `pipeline.py`: **new gate** `diversity_band` on `mtld`, Simpson and gpt2 PPL against human bands (Dubois Table 2: MTLD ≈ 95, PPL ≈ 14 under a 3B scorer; recalibrate to gpt2). | **Yes on a 2024 supervised detector, with a decay warning.** RAID accuracy@5%FPR, GPTZero: chat models greedy → sampling+RP **98.8 → 82.5** (16% flipped); base models **74.7 → 4.8** (70% flipped). Originality 98.6 → 72.5 / 99.9 → 51.2. **But** a RoBERTa trained on a decoding mixture holds **≥0.95 on all 37 settings** (Dubois 2025), and desklib is RAID-trained, so expect the local lift to be small. **Fishchuk & Braun (penalties 0.3-0.6 → detection <50%) is 2023, against the OpenAI classifier and GPT-2 detector: perplexity-era, flagged.** | Zero inside the band (Fishchuk: quality preserved to 0.6; Dubois: T=0.9-1.0, η=1e-4, top-p 0.95 match human MTLD/Zipf). **Out of band it is catastrophic:** T=1.1 → PPL 612; RP 1.05 → MTLD 4.5× human. | **Moderate, decaying.** Free to try; must be gated. |
| **4** | **Perplexity-band gate and rank term** so the RL/guided loop restores lexical unpredictability without errors (the Adversarial Paraphrasing mechanism) | `pipeline.py`: **new gate** `perplexity_band` (two-sided, gpt2 via `PerplexityDetector`) in `_gate_candidate`; **new term** in `_rank_key` under `quality_weight`. `guided.py`: optionally add the band as a second guide term in `reweight_logprobs`. | **Yes on the detectors it was tuned against, zero transfer.** Adversarial Paraphrasing cut TPR@1%FPR by **64-88%** on RADAR / MAGE / OpenAI-RoBERTa (flip-equivalent, 2025), PPL 9.28 → 14.3-17.1, 87% rated 4-5/5. Report 17 §B1: those exact outputs are flagged **59-60 of 60 on desklib**. | ≈ 0 (no significant quality gap vs simple paraphrase). | **Low-moderate as a lever; high as a guard.** Its real value is stopping the "Polished" collapse (PPL 9.28) that light rewriting causes, which is what flags honest edits at 64-80% on Pangram. |
| **5** | **Stop over-polishing: an error floor, never injection.** If anything is left in, ERRANT-native shape only (PUNCT, DET, PREP, ORTH), never SPELL, never in protected spans | `transforms.py`: **new config flag** in `HumanizeConfig` to suppress any normalising pass; optional **new transform** `native_imperfections` (missing comma before a non-restrictive clause, hyphen inconsistency) at ≤1 per 100 words, excluded from `protected_spans`. `pipeline.py`: **new gate** `error_floor` (LanguageTool rate ≥ genre floor). | **No.** No error-injection-only study exists on any commercial detector. Stumbling Blocks (keystroke-realistic typos): fine-tuned DeBERTa **96-112% relative AUROC** — no flips, slightly *worse*. DAMAGE holds 98.26% on typo-laden L3 output. GECScore 98.62% AUROC supports the aggregate direction only. | ≈ 0 for missing commas, hyphens, passives (positively correlated with human scores in ETS data); **−2.18 of 7 points** at 2 visible errors/100 words for anything a reader notices. | **Near zero expected lift.** Keep only as a floor; run the experiment because no one has. |
| **6** | **Structural band-matching** (already built) | `transforms.py`: `vary_sentence_length`, `break_paragraph_template`, `break_parallelism`, `strip_formal_connectives` exist. **No change**; do not add shuffling or paragraph randomisation. | **No.** DIPPER order knob at fixed lexis: OpenAI RoBERTa **13.3 → 14.8 (up)**, 9.1 → 10.0, 15.6 → 15.6. RAID paragraph insertion: GPTZero **66.5 → 66.2**. **Perkins 2024 "burstiness" (39.5 → 15.9%) is a full LLM rewrite on detectors already below 40% accuracy — treat as perplexity-era, flagged.** Report 17: 0/9 locally. | 0 to positive (report 10: better essays have CV 0.41-0.53). | **≈ 0 lift on flips.** Retain for quality and to stay out of the anomalous-burstiness region. |
| — | **Do not build** | — | Typos/spelling: Perkins 2024 −27 pts on 2023-24 detectors, **perplexity-era, flagged**; supervised +8-12% relative AUROC toward AI. Homoglyphs: GPTZero −0.3; one 2026 example 99.9% → 18.0% human. Synonym swap: RoBERTa-L **+22.7**, Originality **+11.5** toward AI. Back-translation: report 17 **0/9**, pushed 3 of 5 human paragraphs over the threshold; Pangram −3. Sentence shuffle: no supervised effect, AI text 3.5× more order-fragile. T > 1.0, RP ≥ 1.05: out of human band, caught at 0.98 by mixture-trained RoBERTa. Light LLM "humanize" pass on good text: **L1 ≥ L0 on every HC3 detector**; Karr 64-80% flag rate. Tangents/redundancy: no evidence; "irrelevant detail" is a 19.5% AI cue. | High for all of them | Negative |

**Expected effect on the pass rate, stated honestly.** Nothing in this table except rank 1 has flip-level evidence against a *current, held-out* supervised detector, and rank 1's evidence is on GPTZero and Pangram, not desklib. Ranks 2-4 have flip-level evidence on at least one supervised detector but either small n (rank 2, n=30), a 2024 detector with a demonstrated countermeasure (rank 3), or demonstrated zero transfer to desklib (rank 4). Ranks 5-6 have no flip evidence anywhere. The honest forecast for the local desklib/fast/fakespot bench is: rank 1 is the only change likely to move the 0-of-9 baseline by itself; ranks 2 and 3 are worth adding because they are free and compound with rank 1 (RAID shows base-model × sampling is multiplicative: 74.7 → 4.8); rank 4 is a guard; rank 5 is an experiment.

### 11.2 The three cheapest experiments per top recommendation (each under one hour on the local bench)

Score everything on desklib (yardstick) **and** fast, fakespot, academic held out; report flips at 0.5, not mean p(AI). Use the nine paragraphs from report 17 so results are comparable.

**Rank 1 — HIP-style fine-tune.** Training itself is an overnight job (report 17 §D1), so the sub-hour tests are about whether the mechanism exists locally.
1. *Load the published adapter.* `YixuanEvenXu/Qwen3-4B-Base-HIP-adapter` is on HuggingFace; load `Qwen3-4B-Base` fp16 on MPS with PEFT, run the nine paragraphs through 3 rounds, score. If it flips 0/9 on desklib the mechanism does not transfer to our yardstick and rank 1 drops. (~40 min including download.)
2. *Clean the base-vs-instruct control.* Report 17 §B3 is contaminated by verbatim copying. Re-run `FREEFORM_SPECS` with a copy-suppressing few-shot pair (n ≥ 14 paragraphs), measure the p(AI) gap between `Qwen2.5-3B-4bit` and `-Instruct-4bit` continuations of the same prefix. If the base checkpoint's *own* continuations do not sit below 0.5 on desklib, the human-text training signal is the whole story and the checkpoint choice is secondary. (~20 min.)
3. *Check the target region.* Score 100 human PMC paragraphs (report 14 corpus) on desklib/fast. If the human side of the training pairs is not itself called human by our yardstick, the fine-tune target is wrong for this bench. (~10 min.)

**Rank 2 — Content-specificity spec.**
1. Add spec 12 with the Russell evader constraint; run `n_candidates=13`, `style="mixed"`; count flips from spec 12 alone vs the existing twelve. (~15 min.)
2. Ablate: constraint A "add one concrete named, dated or numbered specific per paragraph" vs constraint B "delete the opening move and the closing summary" as two separate specs. Tells us whether it is specificity or de-templating that moves desklib. (~20 min.)
3. Isolate content from expression: hand-write one specific sentence into each of the nine source paragraphs without touching anything else; score. If p(AI) moves, content is a lever independent of the rewriter. (~15 min.)

**Rank 3 — Sampler band + penalties.** Report 19 Experiment C has already run the temperature and `min_p` sweeps: temperature does nothing on the instruct model, `min_p` hurts, and the base model has a one-temperature-wide human-scoring window at T=1.0 with topic overlap 0.11. What remains untested locally is the penalty family and the band gate.
1. Sweep `repetition_penalty ∈ {1.00, 1.02, 1.05, 1.10}` at T ∈ {0.9, 1.0} on `FREEFORM_MODEL` via `make_logits_processors`; record p(AI), `mtld`, gpt2 PPL and topic overlap per cell. Expect the detector to move only where MTLD leaves the human band; if so, the lever is a quality violation and rank 3 drops. (~25 min.)
2. Implement a presence-penalty logits processor (subtract a constant from any token already in the context) at 0.3 and 0.6, T=0.95; this reproduces Fishchuk and Braun on a 2026 detector for the first time. (~20 min.)
3. Re-run report 19's base-model T=1.0 cell with the `diversity_band` gate and the existing content-overlap gate both on, and count survivors. If the human-scoring window and the content-preserving window do not intersect at 3B, the sampler cannot substitute for rank 1's fine-tune. (~15 min.)

**Rank 4 — Perplexity-band gate.** Zero-generation experiments first.
1. From existing candidate logs, compute gpt2 PPL for every candidate and the nine sources; plot p(AI) against PPL. If desklib's score is flat across PPL 8-20, the band is a guard only and cannot be a lever here. (~10 min.)
2. Re-select from the same logs with `quality_weight > 0` and the PPL-band term added to `_rank_key`; count how often the selected candidate changes and whether any flip is gained or lost. (~10 min.)
3. Kill test: point `guided.py`'s guide at a two-sided PPL-band scorer instead of `e5-small` and run one paragraph. Expect 0 flips on desklib; if so, the guide must stay a detector and the band stays a gate. (~15 min.)

**Rank 5 — Error floor.** The experiment nobody in the literature has run; report 19 Experiment B has now run most of it locally. Punctuation slips flipped 7/30 and typos 10/30 on *boundary* text (desklib 0.731) and **0/40 on saturated text**, with flips transferring to fakespot and academic but not RADAR; plausible masked-LM substitution flipped 1/30 at every dose. Experiment A ran the reverse direction: an instruct-model polish flipped 11/33 human paragraphs to AI, a T5 grammar corrector at 2% words changed flipped 2/33. What remains:
1. ERRANT-native-shaped injection specifically (missing comma after a short introductory element or around a non-restrictive clause, one hyphen inconsistency, one dropped article), at 0.5 and 1 per 100 words, on report 19's *saturated* corpus, protected spans excluded. Report 19's punctuation arm used generic slips; this tests the native distribution from §7.1 and report 22. Expect 0/40. (~15 min.)
2. The same injection on the *output of rank 1 or rank 2*, not on raw AI text: the floor is meant to keep an already-borderline document on the human side, and report 19 shows noise only moves boundary text. Measure whether it moves the boundary in the right direction there. (~15 min.)
3. LanguageTool auto-fix (not an LLM) on 20 human PMC paragraphs, re-scored. Report 19's T5 arm suggests ≈ 2/33 flips; if LanguageTool matches that, pure rule-based cleaning is safe and the floor gate only needs to block LLM polish passes. (~15 min.)

If ranks 3-5 all come back at 0 flips held out, that is consistent with every number in this report, and the pipeline's remaining budget should go entirely to rank 1 with rank 2 as its candidate spec.

---

## Sources

**Internal reports referenced**
- `research/00-SYNTHESIS.md`, `04-measurable-features-human-vs-ai.md`, `12-anti-humanizer-countermeasures.md`, `13-gptzero-quantified-attacks.md`, `17-evasion-code-mining.md` (0/9 flip baseline, gsub 9/9 on desklib and 0/9 held out), `19-noise-injection-measured.md` (local noise experiments A-E, same date), `22-human-imperfection-profile.md` (LOCNESS ERRANT re-analysis).

**Verified directly from PDF in this report**
- Dubois, Yvon, Piantanida, "How Sampling Affects the Detectability of Machine-written texts," Oct 2025. https://arxiv.org/abs/2510.13681 · code https://github.com/BaggerOfWords/Sampling-and-Detection
- Dugan et al., RAID, ACL 2024 (Tables 5, 15, 16; Finding 3). https://aclanthology.org/2024.acl-long.674.pdf · https://arxiv.org/abs/2405.07940
- Fishchuk and Braun, "Efficient Black-Box Adversarial Attacks on Neural Text Detectors," ICNLSP 2023. https://arxiv.org/abs/2311.01873

**Information-theoretic**
- Meister et al., "Locally Typical Sampling," TACL 2023. https://aclanthology.org/2023.tacl-1.7.pdf
- Ippolito et al., "Automatic Detection of Generated Text is Easiest when Humans are Fooled," ACL 2020. https://aclanthology.org/2020.acl-main.164.pdf
- Gehrmann et al., GLTR, ACL 2019. https://aclanthology.org/P19-3019.pdf
- Venkatraman, Uchendu, Lee, GPT-who, NAACL Findings 2024. https://aclanthology.org/2024.findings-naacl.8.pdf
- Venkatraman, He, Reitter, "How do decoding algorithms distribute information in dialogue responses?" EACL Findings 2023. https://aclanthology.org/2023.findings-eacl.70/
- DiVEye, TMLR 2026. https://arxiv.org/abs/2509.18880
- Muñoz-Ortiz et al., "Contrasting Linguistic Patterns in Human and LLM-Generated News Text," 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11422446/
- Mikhaylovskiy, "Zipf's and Heaps' Laws for Tokens and LLM-generated Texts," Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.837/
- Kirk et al., "Understanding the Effects of RLHF on LLM Generalisation and Diversity," ICLR 2024. https://arxiv.org/abs/2310.06452
- Zhang et al., "Verbalized Sampling," 2025. https://arxiv.org/abs/2510.01171
- "Artificial Hivemind," NeurIPS 2025. https://arxiv.org/abs/2510.22954
- Xu et al., "Base Models Look Human To AI Detectors," 2026. https://arxiv.org/abs/2605.19516
- Reinhart et al., "Do LLMs write like humans?" PNAS. https://arxiv.org/abs/2410.16107
- Pangram technical report. https://arxiv.org/abs/2402.14873

**Token-level noise**
- Wang et al., "Stumbling Blocks," 2024. https://arxiv.org/abs/2402.11638
- Zhou et al., HMGC, 2024. https://arxiv.org/abs/2404.01907
- Wang et al., RAFT, 2024. https://arxiv.org/abs/2410.03658
- Zheng et al., TH-Bench, 2025. https://arxiv.org/abs/2503.08708
- Creo and Pudasaini, SilverSpeak, 2024. https://arxiv.org/abs/2406.11239
- "Adversarial Attacks on AI-Generated Text Detection Models," 2024. https://arxiv.org/abs/2406.01179
- Kadhim et al., 2025. https://arxiv.org/abs/2501.18998
- Sadasivan et al., "Can AI-Generated Text be Reliably Detected?" https://arxiv.org/abs/2303.11156
- Krishna et al., DIPPER, 2023. https://arxiv.org/abs/2303.13408
- ESPERANTO, 2024. https://arxiv.org/abs/2409.14285
- Macko et al., "Authorship Obfuscation in Multilingual MGT Detection," 2024. https://arxiv.org/abs/2401.07867
- "Detecting the Machine," Mar 2026. https://arxiv.org/abs/2603.17522
- Masrour, Emi, Spero, DAMAGE, GenAIDetect 2025. https://arxiv.org/abs/2501.03437 · https://aclanthology.org/2025.genaidetect-1.9/
- GPTZero technical report, Feb 2026. https://arxiv.org/abs/2602.13042
- PIFE, Sep 2025. https://arxiv.org/abs/2510.02319
- StealthRL, 2026. https://arxiv.org/abs/2602.08934
- Adversarial Paraphrasing, 2025. https://arxiv.org/abs/2506.07001
- Hu, Chen, Ho, RADAR, NeurIPS 2023. https://arxiv.org/abs/2307.03838

**Structural**
- Perkins et al., 2024. https://arxiv.org/abs/2403.19148
- "Burstiness / sentence-length CV, human vs LLaMA vs LLaDA," 2025. https://arxiv.org/abs/2507.10475
- Feature stability across LLMs and domains, 2026. https://arxiv.org/abs/2608.27855
- Leave-one-family-out ablation on MAGE, 2026. https://arxiv.org/abs/2606.04177
- CSFG inter-sentence transition detector, 2026. https://arxiv.org/abs/2608.26694
- Sentence-shuffled perplexity ratio, 2026. https://arxiv.org/abs/2604.25860
- RACE RST discourse fingerprint, 2026. https://arxiv.org/abs/2604.04932
- Em-dash frequencies by model, 2026. https://arxiv.org/abs/2603.27006
- Discourse markers under self-training, 2026. https://arxiv.org/abs/2605.20602
- Sentence-length Hurst under random punctuation, 2025. https://arxiv.org/abs/2508.19782
- Alabdulmohsin et al., "Fractal Patterns May Illuminate the Success of Next-Token Prediction," 2024. https://arxiv.org/abs/2402.01825
- Desaire et al., 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10328544/
- PAN CLEF 2025 feature study, 2026. https://arxiv.org/abs/2603.23146

**Semantic/content**
- Bao et al., HART, 2025. https://arxiv.org/abs/2503.00258
- Bao et al., Triospect, 2026. https://arxiv.org/abs/2606.31074
- Chen et al., D2C-Routing, 2026. https://arxiv.org/abs/2608.27380
- Russell et al., "People who frequently use ChatGPT for writing tasks are accurate and robust detectors," 2025. https://arxiv.org/abs/2501.15654
- Sun et al., "Idiosyncrasies in LLMs," 2025. https://arxiv.org/abs/2502.12150
- Herbold et al., Scientific Reports 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10616290/
- Homogenization in AI-assisted essays, 2026. https://arxiv.org/abs/2603.21228
- Pangram, "Introducing AI Assistance Detection." https://www.pangram.com/blog/introducing-ai-assistance-detection

**"Too clean" and false positives**
- Karr et al., "Why AI Detection Fails for Academic Integrity," Aug 2026. https://arxiv.org/abs/2608.11256
- Liang et al., 2023. https://arxiv.org/abs/2304.02819
- GPTZero, "ESL and AI detection," Oct 2023. https://gptzero.me/news/esl-and-ai-detection/
- Turnitin ELL bias study. https://www.turnitin.com/blog/new-research-turnitin-s-ai-detector-shows-no-statistically-significant-bias-against-english-language-learners
- APT-Eval, 2025. https://arxiv.org/abs/2502.15666
- Human and polished peer reviews, 2026. https://arxiv.org/abs/2603.20450
- Professional editing of non-native manuscripts, 135k pairs, 2026. https://arxiv.org/abs/2608.26710
- Autistic writers and the GPT-2 detector, 2026. https://arxiv.org/abs/2607.14729
- GECScore, COLING 2025. https://arxiv.org/abs/2405.04286
- Artemova et al., Beemo, 2024. https://arxiv.org/abs/2411.04032
- ImBD, 2024. https://arxiv.org/abs/2412.10432
- MixSet, NAACL Findings 2024. https://aclanthology.org/2024.findings-naacl.29/
- Pangram, "Can you avoid AI detection through editing?" Aug 2025. https://www.pangram.com/blog/can-you-avoid-ai-detection-through-editing
- Pangram, "How students try to avoid AI detection," Oct 2025. https://www.pangram.com/blog/how-students-try-to-avoid-ai-detection
- GPTZero, "Detecting AI-humanized text," Jan 2026. https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/
- GPTZero, "AI paraphrasing detection." https://gptzero.me/news/ai-paraphrasing-detection/
- Copyleaks, grammar checker FPR test. https://copyleaks.com/blog/do-grammar-checkers-get-flagged-as-ai
- Originality, Grammarly test, Oct 2025. https://originality.ai/blog/grammarly-use-trigger-ai-detection
- Constitution/Genesis anecdote (secondary). https://ludwig.guru/blog/that-time-when-the-american-constitution-resulted-as-ai-generated-2/

**Learned noise**
- Bryant et al., BEA-2019 shared task (ERRANT distributions, Table 4). https://aclanthology.org/W19-4406.pdf
- Stahlberg and Kumar, "Synthetic Data Generation for GEC with Tagged Corruption Models," 2021. https://arxiv.org/abs/2105.13318
- Kiyono et al., 2019. https://aclanthology.org/D19-1119/
- Xie et al., "Noising and Denoising Natural Language," NAACL 2018. https://aclanthology.org/N18-1057/
- LLM-based error generation, 2024. https://arxiv.org/abs/2403.05493
- Du et al., IteraTeR, ACL 2022. https://aclanthology.org/2022.acl-long.250.pdf
- Wang et al., ScholaWrite, 2025. https://arxiv.org/abs/2502.02904
- Lunsford and Lunsford 2008 (secondary). https://eric.ed.gov/?id=EJ802721
- AuthorMist. https://arxiv.org/abs/2503.08716 · MASH. https://arxiv.org/abs/2601.08564

**Quality and grading**
- Bleske-Rechek et al., "Grammar Matters," 2019. https://www.bleske-rechek.com/April%20Website%20Files/Bleske-Rechek%20et%20al.,%202019.%20Grammar%20Matters.pdf
- LLM grader bias to surface features, 2026. https://arxiv.org/abs/2603.18765
- TOEFL11 error features vs proficiency. https://arxiv.org/abs/1612.00729
- Chen et al., ETS RR-17-04 (e-rater microfeatures). https://files.eric.ed.gov/fulltext/EJ1168485.pdf
- AP English Language scoring rubrics. https://apcentral.collegeboard.org/media/pdf/ap-english-language-and-composition-frqs-1-2-3-scoring-rubrics.pdf
- IELTS scoring. https://www.ielts.org/for-test-takers/how-ielts-is-scored
- Error-gravity systematic review (Santos, Vann et al., Politzer, Ensz). https://www.nepjol.info/index.php/mrj/article/download/73470/56240/213760
- Vann, Meyer, Lorenz 1984, TESOL Quarterly. https://onlinelibrary.wiley.com/doi/abs/10.2307/3586713
- Beason 2001, "Ethos and Error." https://eric.ed.gov/?id=EJ632329
- Gubala, Larson, Melonçon 2020, JBTC. https://journals.sagepub.com/doi/10.1177/1050651920910205
- Lexical sophistication and essay quality review, CJAL. https://files.eric.ed.gov/fulltext/EJ1402840.pdf
- HumanizerBench leaderboard, Sep 2026. https://humanizerbench.com/leaderboard/
