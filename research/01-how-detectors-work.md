# How GPTZero and Other AI-Text Detectors Actually Work

*Research report, September 2026. All claims are cited to the Sources list at the end; vendor-reported numbers are labelled as such.*

---

## 1. GPTZero

### 1.1 History and architecture in one paragraph

GPTZero launched in January 2023 as Edward Tian's Princeton thesis project, originally scoring text on two statistics computed with a GPT-2-class language model: **perplexity** and **burstiness** [1][2]. In autumn 2023 the company migrated to "a deep-learning based architecture" and states plainly that it "no longer uses perplexity and burstiness for its AI detection" as the decision mechanism [3][4]. The current system is described as "an end-to-end deep learning approach, trained on text datasets from the web, education, and AI-generated from a range of LLMs," with "a sentence-by-sentence classification model" that "determines the probability and confidence that a text was created by AI" [5]. In June 2026 GPTZero (19M registered users, ~$30M ARR) agreed to be acquired by Superhuman [6].

### 1.2 The original signals: perplexity and burstiness

- **Perplexity** (per sentence). GPTZero defined it as "how likely an AI model would have chosen the exact same set of words as found in the document." Formally, for a sentence of tokens $t_1..t_N$ scored by LM $\theta$:
  $$\mathrm{PPL}(s) = \exp\Big(-\tfrac{1}{N}\sum_{i=1}^{N}\log p_\theta(t_i \mid t_{<i})\Big)$$
  GPTZero's historical rule of thumb: sentence perplexity **above ~85** leaned human [3][4].
- **Burstiness**. "How much writing patterns and text perplexities vary over the entire document"; operationally the **standard deviation of per-sentence perplexity** across the passage. Low variance (uniformly medium-low perplexity sentences) was read as AI [3][7]. GPTZero described burstiness as "unique to GPTZero" [3].
- These two numbers fed a small classifier (e.g., logistic regression) to output a verdict [7]. Both remain visible in the UI as explanatory indicators but are "one element of several" and, per the support centre, obsolete as a decision rule [4].

### 1.3 The "seven-component" model

A 2024 IACIS paper documenting GPTZero (drawing on GPTZero's own product documentation) lists **seven machine-learning components**, each providing a weighted score to the document classification: **Education Module, Burstiness, Perplexity, GPTZeroX, GPTZero Shield, Internet Text Search, and Deep Learning** [8]. GPTZero's current tech page collapses this into five pipeline stages: input processing (text/docx/pdf/images, up to 50 files) → deep-learning foundation → sentence classification → "Paraphraser Shield" (protecting against "paraphrasing and homoglyph attacks") → output with premium "AI vocabulary, plagiarism, and citeable sources" features [5]. The "Internet Text Search" component checks whether spans exist verbatim on the web so that commonly quoted text is not tagged as AI [8].

### 1.4 Deep Scan / Advanced Scan and sentence highlighting

- The **Advanced Scan** (called "Deep Scan" in the product) runs the sentence-level classifier and labels each sentence as **high / medium / low impact** on the document verdict. High-impact sentences "disproportionately influence the AI/human determination" [9][10].
- The API exposes `document_classification` ∈ {`HUMAN_ONLY`, `MIXED`, `AI_ONLY`}, `class_probabilities`, `confidence_category` ∈ {high, medium, low}, and `highlight_sentence_for_ai` [11].
- Internally GPTZero now uses a **five-class taxonomy**: Human, Mixed (Concatenated), Mixed (Polished), AI (Pure AI), AI (Paraphrased); for benchmarking these are binarized [12].

### 1.5 Confidence thresholds and calibration

- GPTZero's percentages are **predictive probabilities, not proportions**: a "4% AI" score means a 4-in-100 chance the whole document is AI, with remaining mass on human/mixed [9].
- Post-hoc **calibration** is applied "to ensure that for the most part, our detector is not overconfident." Stated meaning: "a predicted probability of 79% means that on documents that have similar predictions, the detector is correct 79% of the time" [10].
- Confidence bands as published: **Highly confident ≈ <2% error** (tech page: <1% on high-confidence), **Moderately confident ≈ 10% error**, **Low confidence ≥ 14% error** [9][10]. The ML page says high confidence ⇒ "99.1% of human articles are classified as human, and 98.4% of AI articles are classified as AI" [11].
- The system "is designed to favor false negatives over false positives, erring toward 'human' designations when uncertain" [9].
- Minimum text for benchmarks: 250 characters (~50 words); GPTZero acknowledges short passages "offer insufficient linguistic patterns" [3][12].

### 1.6 Mixed text

GPTZero claims to be the only major detector outputting a three-way human/AI/mixed label and reports **96.5% accuracy on mixed documents** versus competitors' 82.5–87.5% [5][13]. Mixed documents are handled by the sentence-level model plus the "Concatenated" vs "Polished" subclasses [12].

### 1.7 Paraphrasers and humanizers

- **Two-step approach**: "first: detecting if the modified text is originally AI-written, and only then analyzing if the content has been altered using other writing tools to appear human." Output label: **"AI-paraphrased"** [14].
- Beta announcement (2024): ~95% detection of text from Phrasly, Undetectable, QuillBot with <0.1% false positives [14][15].
- 2025/26 "Paraphraser Shield": trained against **12+ humanizer tools** (including the open-source "Temp paraphraser", EMNLP 2025) using "internal systems to simulate adversarial attacks"; the model looks for "deeper semantic and structural signals beyond surface form." Reported AI recall on 1,000 humanized texts from GPT-5/4o/4.1/Claude/Gemini: **GPTZero 93.5%** (0.21% FPR in a later run: 91.8%) vs **Pangram v3 50.2%** (later 68.1%) vs **Originality lite 57.3%** [12][16].
- Pangram counters with its own numbers (97.67% on commercial humanizers for Pangram 4) [17], so treat all vendor-vs-vendor numbers as contested.

### 1.8 AI Vocabulary feature

- Launched October 2024: GPTZero "regularly scans millions of AI texts and compares them to similarly uploaded human documents (based on subject matter, length, etc.)" and lists phrases used by AI **≥2–3× more than humans**; the Top-50 (later Top-100) list spans **10× to 200×+** enrichment, updated monthly. Example: "objective study aimed" ≈ 269× more likely from AI; others: "delve," "tapestry," "multifaceted," "today's digital age," "shed light" [18][19][20].
- It is **explanatory only**: "It doesn't directly affect probability scores" [9].
- Independent check (BEA 2025, Schmalz & Tack): using GPTZero's vocabulary lists as bag-of-words features, mere *presence* of terms separated ChatGPT essays from student essays reasonably, but "performance drops to near-random when applied to Claude-generated essays," and every AI-Vocabulary classifier "significantly under-perform[s] compared to Bag-of-Words classifiers trained directly on the full dataset vocabulary" [21].

### 1.9 ESL de-biasing

Responding to Liang et al., GPTZero re-ran the TOEFL set in October 2023: 1 of 91 essays flagged as AI (**1.1%**), with a further 6.6% marked "possible AI content" (so ≈7.7% if that band counts as positive). Methods: a CNN layer with an "education tag," an ESL-writer pre-classifier, and dataset expansion (TOEFL, 180k Medium articles, 31k Persuade, 12k Hewlett) [22].

### 1.10 Published benchmark numbers (vendor)

- Headline: 99% accuracy AI-vs-human, ≤1% FPR, 96.5% on mixed [5][13].
- 2026 benchmark post (model 4.3b, 4 domains, 1,000 texts each from GPT-5.2/Gemini 3 Pro/Claude Sonnet 4.5/Grok 4 Fast): **FPR 0.08%, recall 99.60%, precision 99.93%, accuracy 99.76%**; multilingual (24 langs, model 3.7m): 0.09% FPR / 97.62% recall [12].
- Independent: RAID (2024) measured GPTZero at 0.03% FPR at default threshold and 66.5% accuracy at 5% FPR across all generators [23]; Epoch AI / Nature 2026: zero FPs [24]; Pangram 4 report: GPTZero 98.64% TPR@1%FPR on the UChicago set but 77.6% on GEDE and 0% on VUB fully-AI academic papers [17].
- Patents: no GPTZero patent surfaced in searches; Originality.ai holds a text-analysis patent [25].

---

## 2. Other commercial detectors

| Detector | Stated approach | Key published numbers / notes |
|---|---|---|
| **Originality.ai** | Supervised fine-tuning of "modified BERT and RoBERTa" transformers on millions of labelled samples; models Lite 1.0.2, Turbo 3.0.2, Academic 0.0.5; 30-language multilingual model; "AI Allowance" thresholds (0/5/15/25/40%) [25][26] | Vendor: Lite 99% acc / 0.5% FPR; Turbo 99%+ / 1.5% FPR and "97% on humanized"; 456,872-sample Benchmark V6 [26]. RAID: 85% acc @5% FPR (best commercial) but **collapses to 9.3% under homoglyphs** and 13% FPR on Wikipedia [23]. GPTZero reports 37% FPR for Turbo on o1 text and 14.8% multilingual FPR [12][13]. |
| **Turnitin** | Splits document into overlapping segments of a few hundred words, scores each sentence 0–1, aggregates to a document percentage; 300-word minimum; no score/highlights shown in the **1–19% band** (asterisk) to control FPs; detects "AI-paraphrased" text [27][28] | Document FPR <1% for docs with ≥20% AI; **sentence-level FPR ≈4%**; FPs concentrate at human/AI transitions and list-like text [27][28]. |
| **Copyleaks** | Proprietary classifier trained on human vs AI corpora; cites phrase-frequency, grammar/syntax, syllable dispersion, perplexity and burstiness; independent QA team; sentence highlighting; ~100-word minimum [29][30] | Vendor: 99.1% accuracy on >1M samples; independent 2026 reviews put it at 77–96% [30]. GPTZero measured 5.0% FPR on o1 text [13]. |
| **Winston AI** | ML models trained on millions of samples; analyses perplexity, burstiness, repetition, sentence structure; "AI Prediction Map" colour-codes sentences by predictability (ignores sentences <60 chars); Flesch-Kincaid readability shown separately | Vendor: "99.98% accuracy." Human Score is confidence, not proportion. RAID: 0.75% default FPR, 68.1% acc @5% FPR on sampled closed-source [23][31]. |
| **Sapling** | Transformer classifier trained across GPT/Gemini/Claude/Llama/Mistral; API returns document `score` 0–1, `sentence_scores`, and **`token_probs`** per token; 300-char minimum; recommends 0.9 threshold for high-stakes use [32] | Vendor admits "Small modifications to AI-generated text can cause that text to no longer be flagged." |
| **ZeroGPT** | "DeepAnalyse" multi-stage: sentence segmentation, token patterns, burstiness, entropy, ensemble features [33] | RAID: 1.71% default FPR; could not reach 5% FPR target; 16.9% FPR on some domains; 0.3% acc on sampled+penalty non-chat [23]. |
| **Pangram Labs** | Transformer classifier trained with **"synthetic mirrors"** (for each human doc, prompt an LLM to write on the same topic/length/style) plus iterative **hard-negative mining** and active learning; Pangram 4 uses an open-weight MoE backbone, "Repeat2" input duplication, per-token 3-class logits (human / AI-assisted / AI) decoded with a linear-chain CRF [17][34][35] | Pangram 4: AUROC 0.9916, **FPR 0.0041%** (1 FP in 24,586 ESL samples), FNR 0.34% on 520k outputs from 26 models; 97.67% on commercial humanizers; 99.97–100% TPR@1%FPR under homoglyph/Unicode attacks; but only 65.7% recall on heavily *edited* AI text and admits it "cannot account for humans intentionally writing in LLM style" [17]. |

---

## 3. Academic detection methods and their signals

### 3.1 Log-likelihood / perplexity and GLTR (Gehrmann, Strobelt & Rush 2019)
GLTR computes, for every token under GPT-2 small, three tests: (i) the **probability** $p(t_i\mid t_{<i})$, (ii) the **absolute rank** of the token in the predicted distribution, (iii) the **entropy** of that distribution. Tokens are colour-bucketed by rank (top-10 green, top-100 yellow, top-1000 red, else purple) and histogrammed; heavily green text is machine-like. Human detection accuracy rose from 54% to 72% with the overlay [36][37]. Baselines "Rank" and "LogRank" (mean log-rank over tokens) descend from this.

### 3.2 DetectGPT (Mitchell et al. 2023) — probability curvature
Hypothesis: LLM samples sit in **negative-curvature regions** of the model's log-probability surface. Perturb passage $x$ with T5 mask-filling to get $\tilde{x}_1..\tilde{x}_k$ and compute
$$d(x) = \log p_\theta(x) - \tfrac{1}{k}\sum_j \log p_\theta(\tilde{x}_j)$$
Large positive $d$ ⇒ machine. Improved GPT-NeoX fake-news detection from 0.81 to 0.95 AUROC but needs ~100 perturbations per document [38].

### 3.3 Fast-DetectGPT (Bao et al., ICLR 2024) — conditional probability curvature
Replaces perturbation with sampling from the scoring model's own next-token distributions, computable analytically in one forward pass:
$$d(x,p_\theta,q_\varphi)=\frac{\log p_\theta(x\mid x)-\tilde{\mu}}{\tilde{\sigma}},\quad \tilde\mu=\sum_j\sum_{\tilde x_j} q_\varphi(\tilde x_j\mid x_{<j})\log p_\theta(\tilde x_j\mid x_{<j})$$
with $\tilde\sigma^2$ the analogous variance. 340× faster than DetectGPT; AUROC 0.9887 white-box (vs 0.9554), 0.9677 black-box using GPT-J / GPT-Neo-2.7B surrogates; later Llama-3-8B pairs perform better [39][40].

### 3.4 Binoculars (Hans et al., ICML 2024)
Uses two closely related models sharing a tokenizer — observer $M_1$ = Falcon-7B, performer $M_2$ = Falcon-7B-Instruct:
$$B(s)=\frac{\log \mathrm{PPL}_{M_1}(s)}{\log \mathrm{X\text{-}PPL}_{M_1,M_2}(s)},\qquad \log\mathrm{X\text{-}PPL}=\tfrac1N\sum_i \sum_{v} p_{M_2}(v\mid s_{<i})\cdot(-\log p_{M_1}(v\mid s_{<i}))$$
Cross-perplexity estimates how surprising the text *should* be to a machine given the context (fixing the "capybara problem" where an unusual prompt inflates raw perplexity). Global threshold ≈0.901. >90% TPR at **0.01% FPR** on ChatGPT text without ChatGPT training data; 99.67% accuracy on ESL EssayForum essays; but flags memorized text (US Constitution) as AI and showed 58% FNR on GPT-4 in one table [41]. RAID found it the strongest overall detector at 5% FPR (79.6%) [23].

### 3.5 Ghostbuster (Verma et al., NAACL 2024)
Runs the document through **weak models** — a unigram model, a Kneser-Ney trigram model, and GPT-3 `ada` and `davinci` — to get per-token probability vectors. A **structured search** over vector ops (add, sub, mul, div, >, <) and scalar reducers (max, min, avg, avg-top25, len, L2, var) discovers features such as `var(unigram_probs > ada_probs - davinci_probs)`. These plus seven handcrafted features (word length, largest token probabilities, outlier counts) feed **L2-regularised logistic regression**. 99.0 F1 across essays/creative/news (+5.9 over GPTZero, +41.6 over DetectGPT); robust across prompts (99.5 F1) but only 74.7% accuracy on the 91 TOEFL essays, and perplexity-only scored 13.2% there [42].

### 3.6 Fine-tuned RoBERTa classifiers (OpenAI 2019; 2023 classifier)
OpenAI fine-tuned RoBERTa-base/large on 1.5B GPT-2 outputs vs WebText, reaching ~95% in-distribution accuracy [43]. Its January 2023 "AI Text Classifier" caught only **26% of AI text** with a **9% FPR** and was shut down in July 2023 [44]. RAID shows the GPT-2 RoBERTa detectors generalise worst (44.8%) and can even *improve* under paraphrase because paraphrased text drifts toward their training distribution [23].

### 3.7 Intrinsic dimension (Tulchinskii et al., NeurIPS 2023)
Treat RoBERTa token embeddings of a passage as a point cloud; estimate its **persistent homology dimension** (a fractal dimension from topological data analysis). Human text ≈ **9** for alphabetic languages (≈7 for Chinese); AI text ≈ **1.5 lower**. Robust to domain, generator, and (to a degree) paraphrasing; the gap shrinks for newer models [45].

### 3.8 Watermarking (Kirchenbauer et al., ICML 2023)
At each step, hash the previous token to split the vocabulary into a **green list** of size $\gamma|V|$ and a red list; add $\delta$ to green logits ("soft" watermark). Detection needs only the hash key: count green tokens $|s|_G$ over $T$ tokens and compute
$$z=\frac{|s|_G-\gamma T}{\sqrt{T\gamma(1-\gamma)}}$$
Reject $H_0$ at $z>4$ (one-sided $p\approx3\times10^{-5}$); hard watermark detectable from 16 tokens, soft watermark strength depends on text entropy [46][47]. Only works if the generator cooperates; paraphrasing degrades it (DIPPER, recursive paraphrase) [48][49].

### 3.9 2025–2026 methods
- **DivEye** (2025/26): interpretable "diversity" features — mean, variance, skew, kurtosis, autocorrelation and burstiness of per-token surprisal plus entropy features — outperform zero-shot detectors by up to 33.2% and boost others by 18.7% as an auxiliary signal; claimed paraphrase-robust [50].
- **IRON** (adversarial-training framework) and **M-RangeDetector** (multi-range attention masks) target robustness [51].
- **Pangram 4** style token-level CRF decoding for mixed authorship [17].
- 2026 benchmarks: "Detecting the Machine" finds "no method generalizes robustly across domains and LLM sources" and observes **perplexity inversion** (modern LLM output sometimes has *higher* perplexity than human text) [52]; "Paraphrasing Attack Resilience" finds Binoculars-inclusive ensembles score highest yet "suffer the most significant losses during attacks" [53].

---

## 4. Concrete features detectors consume, and how they are computed

1. **Per-token log-probability** $\ell_i=\log p_\theta(t_i\mid t_{<i})$ from a scoring LM (GPT-2, GPT-Neo, Falcon, Llama). Everything below derives from it.
2. **Document perplexity / mean NLL**: $-\frac1N\sum\ell_i$; the "Likelihood" baseline.
3. **Sentence-level perplexity vector** and its **std-dev** ("burstiness", GPTZero) or coefficient of variation; also autocorrelation and skew/kurtosis of the surprisal series (DivEye) [3][50].
4. **Token rank** $r_i=\mathrm{rank}(t_i)$ in the predicted distribution; **log-rank** mean; **rank-bucket histogram** (fractions in top-10/100/1000, GLTR) [36].
5. **Predictive entropy** $H_i=-\sum_v p(v)\log p(v)$ per position; mean entropy and entropy–surprisal relationship [36].
6. **Curvature statistics**: DetectGPT $d(x)$ from perturbations; Fast-DetectGPT normalised $(\ell - \tilde\mu)/\tilde\sigma$ [38][39].
7. **Cross-model ratios**: Binoculars $\log \mathrm{PPL}/\log\mathrm{X\text{-}PPL}$; Ghostbuster's arithmetic on weak-model probability vectors [41][42].
8. **Watermark z-score** from green-list token counts [46].
9. **Embedding-geometry**: persistent homology dimension of contextual embeddings [45].
10. **Learned representations**: fine-tuned encoder (RoBERTa/DeBERTa/MoE) producing sentence- or token-level logits; GPTZero, Originality, Pangram, Turnitin, Copyleaks, Sapling all sit here. What such models learn is opaque but empirically includes lexical choice (the same signal GPTZero's AI-Vocabulary list surfaces), discourse templates, punctuation habits, and sentence-length regularity [5][17][25].
11. **Stylometric/surface features** used by Copyleaks, Winston, and XGBoost baselines: sentence-length mean/variance, syllable dispersion, type-token ratio and other vocabulary-richness measures, function-word ratios, readability scores [29][31][52].
12. **Phrase-frequency ratios** $\frac{f_{AI}(w)}{f_{human}(w)}$ (GPTZero AI Vocabulary ≥2–3×, top list 10–200×) [18].
13. **Retrieval**: semantic-similarity search against a corpus of prior LLM generations (Krishna et al.) or the open web (GPTZero Internet Text Search) [8][48].
14. **Meta-signals**: sequence length (all vendors degrade below ~250–300 chars), language, and an ESL pre-classifier (GPTZero) [22][32].

---

## 5. Known weaknesses and GPTZero's countermeasures

- **Paraphrasing.** Krishna et al.'s **DIPPER** (11B T5-XXL, knobs for lexical diversity L and reordering O) cut DetectGPT from 70.3% to 4.6% at 1% FPR and evaded GPTZero, OpenAI's classifier and watermarking; their retrieval defence recovers 80–97% but needs the API provider's generation log [48]. **Sadasivan et al.** prove $\mathrm{AUROC}\le\frac12+TV-\frac{TV^2}{2}$ where TV is the total-variation distance between human and AI text distributions, and show **recursive paraphrasing** also breaks retrieval defences; they add watermark **spoofing** [49]. **PADBen** (2025) finds detectors handle "AI-then-paraphrased" text but fail on the "intermediate laundering region" and on iterative paraphrase [54]. GPTZero's counter: adversarial-simulation training data, an "AI-paraphrased" subclass, and the Paraphraser Shield trained on 12+ humanizers (93.5% recall claimed) [14][16].
- **Homoglyph / Unicode / whitespace.** RAID: homoglyphs at 100% substitution dropped Originality by 75.7 points and Binoculars by 41.9, while **GPTZero lost only 0.3 points**; zero-width spaces and whitespace also hurt metric-based methods [23]. **SilverSpeak** drove seven detectors' MCC from 0.64 to −0.01 [55]. Stumbling Blocks: typo insertion took GLTR's AUC to ≈0, format-character editing cut the OpenAI detector ~63% [56]. Pangram 4 and GPTZero both now normalise/handle these (99.97–100% TPR for Pangram) [5][17].
- **Prompting for "human style".** Liang et al.: "Elevate the provided text by employing literary language" cut detection of ChatGPT essays "to near zero" [57]. Stumbling Blocks: in-context-learning and persona prompts drop metric-based and fine-tuned detectors [56]. RAID: **repetition penalty** alone cut accuracy by up to 32 points, and random sampling is harder to detect than greedy [23].
- **Non-native English false positives.** Liang et al. (Patterns 2023): 7 detectors averaged **61.3% FPR** on 91 TOEFL essays (97.8% flagged by at least one), near-perfect on US 8th-grade essays; enriching vocabulary with ChatGPT dropped FPR to 11.6%. Mechanism: lower perplexity from restricted vocabulary [57]. Counter: GPTZero's ESL de-biasing (1.1%), Pangram's hard-negative mining (1 FP in 24,586 ESL samples), Binoculars' cross-perplexity normalisation [17][22][41].
- **Short text.** Every vendor imposes a floor (GPTZero 250 chars; Sapling 300 chars; Turnitin 300 words; Winston ignores sentences <60 chars) because the statistics have too few tokens [12][27][31][32].
- **Mixed / edited text.** Turnitin's sentence-level FPR is ~4% and errors cluster at authorship transitions [28]; Pangram 4 catches only 65.7% of substantially edited AI text and Nature reports "fully human" verdicts 41% of the time on heavily modified essays [17][24].
- **Memorised or formulaic human text** (constitutions, boilerplate, rote writing) reads as AI to perplexity-based methods [32][41].
- **Domain/generator shift.** RAID: neural detectors are "biased toward training data"; on code/Czech/German, GPTZero scored 39.0% vs Binoculars 69.6% [23].

---

## 6. RAID and other benchmarks

**RAID** (Dugan et al., ACL 2024): 6M+ generations, 11 generators, 8 domains, 4 decoding settings, **11 attacks** — alternative spelling (100% of eligible words), article deletion (50%), homoglyph (100%), insert paragraphs (50%), number swap (50%), paraphrase (T5, 100%), misspelling (20%), synonym (50%), upper-lower (5%), whitespace (20%), zero-width space (100%) — against 12 detectors (RoBERTa GPT-2/ChatGPT, RADAR, GLTR, Binoculars, Fast-DetectGPT, LLMDet, GPTZero, Originality, Winston, ZeroGPT) [23].

Key numbers at 5% FPR (accuracy, all generators):

| Detector | None | Paraphrase | Synonym | Misspelling | Homoglyph | Whitespace | Delete articles |
|---|---|---|---|---|---|---|---|
| Binoculars | 79.6 | 80.3 | **43.5** | 78.0 | **37.7** | 70.1 | 74.3 |
| GPTZero | 66.5 | 64.0 | 61.0 | 65.1 | 66.2 | 66.2 | 61.0 |
| Originality | 85.0 | 96.7 | 96.5 | 78.6 | **9.3** | 84.9 | 71.4 |
| GLTR | 62.6 | 47.2 | 31.2 | 59.8 | 24.3 | 45.8 | 52.1 |
| RoBERTa-L GPT2 | 56.7 | 72.9 | 79.4 | 39.5 | 21.3 | 40.1 | 33.2 |

Findings: chat-tuned models are easier to detect than base models; **sampling + repetition penalty** on open non-chat models collapses most detectors (GPTZero 4.8%, Binoculars 0.6%, GLTR 0.5%); default-threshold FPRs range from 0.03% (GPTZero) to 1.71% (ZeroGPT); adversarially trained RADAR is most uniformly robust; different detector families have different loopholes, so an attacker who does not know the detector should stack attack types [23].

**Other benchmarks**: Stumbling Blocks (average 35% performance drop across editing/paraphrasing/prompting/co-generation attacks; typo insertion worst for GLTR, inter-sentence DIPPER paraphrase worst for fine-tuned detectors, watermark most robust at ~3% loss) [56]; DetectRL (Pangram 4 F1 95.30) [17]; PADBen paraphrase taxonomy [54]; GPTZero's own quarterly benchmark [12]; Epoch AI / Nature independent test [24]; the "Detecting the Machine" 2026 cross-domain study [52].

---

## 7. Implications for building a humanizer: which signals matter most

1. **Statistical detectors (Binoculars, Fast-DetectGPT, GLTR, DivEye) key on token-level surprisal relative to an LLM.** The levers are: raise mean surprisal *unevenly* (some very unpredictable word choices amid ordinary ones), raise the variance/autocorrelation structure of sentence-level perplexity (burstiness), and push tokens out of the top-10 rank bucket. RAID shows synonym swaps of ~50% of eligible words cut Binoculars by 36 points and GLTR by 31 — lexical substitution is the cheapest attack against this family [23]. But beware perplexity inversion: over-doing rare words can itself be a learned AI tell for classifier-based systems [52].
2. **Commercial classifiers (GPTZero, Pangram, Originality, Turnitin) are fine-tuned encoders, and lexical substitution alone barely moves them** (GPTZero −5.5 under synonyms; Originality actually *improved*) [23]. They learn phrase-level and discourse-level habits: enriched AI vocabulary (GPTZero's 10–200× list), templated openers/closers, uniform paragraph and sentence lengths, hedging and summarising patterns. A humanizer must change **structure and vocabulary together**: vary sentence length distribution, break parallelism, remove list-like triads, drop the 100 flagged phrases, and introduce discourse features these models associate with humans (specific detail, asides, non-standard but grammatical constructions).
3. **Character-level tricks are dead against the major vendors.** Homoglyphs and zero-width spaces still break open-source detectors and Originality, but GPTZero (−0.3) and Pangram 4 (≈100% TPR) explicitly normalise them, and such text is trivially exposed by Unicode inspection [17][23].
4. **The single best-documented lever is deep paraphrase with lexical *and* order diversity** (DIPPER L/O knobs; recursive paraphrase) — it defeats watermarks, curvature methods and older classifiers [48][49]. The frontier vendors have responded by training on humanizer outputs: GPTZero claims 93.5% recall on 12+ tools and Pangram 97.67% [16][17]. Independent work (PADBen) still finds an exploitable "intermediate laundering region" and weakness on iterative paraphrase, and GPTZero itself ships an "AI-paraphrased" label precisely because paraphrased AI has its own signature [12][54]. So a humanizer should avoid producing *paraphraser-style* text (which now has a detectable sub-class) and instead aim for text indistinguishable from human first drafts.
5. **Generation-time settings matter as much as post-editing.** RAID's most powerful non-adversarial lever was **sampling + repetition penalty**, cutting accuracy up to 32 points and near zero on some settings [23]; Liang et al.'s "literary language" prompt drove detection to near zero [57]. Controlling decoding and prompting is upstream of any rewriting.
6. **Sentence-level and mixed-text handling is the vendors' weakest documented spot.** Turnitin's sentence FPR is 4%, Pangram catches only 65.7% of substantially edited AI text, and Nature's test saw "fully human" verdicts 41% of the time on heavily modified essays [17][24][28]. Genuine human editing interleaved at the sentence level — not global paraphrase — is where classifiers lose confidence, and GPTZero's design of "erring toward human when uncertain" amplifies this [9].
7. **Length and calibration.** Detectors refuse or hedge below ~50 words and have wide "medium/low confidence" bands (10–14%+ error) [9][12]. Anything a humanizer does should be evaluated at realistic lengths (300–1,000 words) against calibrated thresholds, not the default UI verdict.
8. **Measure what the detectors measure.** A local evaluation harness should compute: mean and per-sentence NLL under two open models (observer/performer), Binoculars score, Fast-DetectGPT curvature, GLTR rank buckets, sentence-length mean/std, type-token ratio, and hits against the GPTZero AI-Vocabulary list — then validate against live GPTZero/Pangram APIs, because the fine-tuned classifiers are the ones that will not be fooled by statistics alone.

---

## Sources

1. NPR, "A college student made an app to detect AI-written text" — https://www.npr.org/2023/01/09/1147549845/gptzero-ai-chatgpt-edward-tian-plagiarism
2. Wikipedia, GPTZero — https://en.wikipedia.org/wiki/GPTZero
3. GPTZero, "What is perplexity & burstiness for AI detection?" — https://gptzero.me/news/perplexity-and-burstiness-what-is-it/
4. GPTZero Support, "How do I interpret burstiness or perplexity?" — https://support.gptzero.me/articles/9585228410-how-do-i-interpret-burstiness-or-perplexity
5. GPTZero Technology page — https://gptzero.me/technology
6. TechCrunch, "Superhuman acquires AI detection startup GPTZero" — https://techcrunch.com/2026/06/23/superhuman-acquires-ai-detection-startup-gptzero/
7. Originality.ai, "GPTZero AI Detection Review" — https://originality.ai/blog/gptzero-ai-content-detection-review
8. IACIS 2024, "Can GPTZero detect if students are using artificial intelligence?" — https://www.iacis.org/iis/2024/3_iis_2024_165-174.pdf
9. GPTZero, "How to Understand Your AI Scan Results" — https://gptzero.me/news/understand-gptzero-ai-scan/
10. GPTZero, "How the Best AI Detector Provides Interpretable Scores" — https://gptzero.me/news/how-the-best-ai-detector-provides-interpretable-scores/
11. GPTZero for ML engineers (API fields) — https://gptzero.me/machine-learning
12. GPTZero, "AI Detection Benchmarking: The Industry Standard…" (2026) — https://gptzero.me/news/gptzero-ai-detection-benchmarking-the-industry-standard-in-accuracy-transparency-and-fairness/
13. GPTZero, "How AI Detection Benchmarking Works at GPTZero (2025)" — https://gptzero.me/news/ai-accuracy-benchmarking/
14. GPTZero, "GPTZero Detects AI Paraphrasers" — https://gptzero.me/news/ai-paraphrasing-detection/
15. GPTZero Support, "How does GPTZero detect AI paraphrasing and AI bypassers?" — https://support.gptzero.me/articles/5593633457-how-does-gptzero-detect-ai-paraphrasing-and-ai-bypassers
16. GPTZero, "Detecting AI-Humanized Text: How GPTZero Stays Ahead" — https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/
17. Pangram 4 Technical Report (arXiv 2607.27183) — https://arxiv.org/html/2607.27183
18. GPTZero AI Vocabulary tool — https://gptzero.me/ai-vocabulary
19. EdScoop, "New GPTZero feature flags AI's favorite words and phrases" — https://edscoop.com/gptzero-common-ai-words-detection-education-2024/
20. Forbes, "New List Ranks AI's 50 Most Overused Words" — https://www.forbes.com/sites/torconstantino/2024/10/07/new-list-ranks-ais-50-most-overused-words---updates-monthly/
21. Schmalz & Tack, BEA 2025, "Can GPTZero's AI Vocabulary Distinguish Between LLM-Generated and Student-Written Essays?" — https://aclanthology.org/2025.bea-1.71.pdf
22. GPTZero, "ESL Bias in AI Detection is an Outdated Narrative" — https://gptzero.me/news/esl-and-ai-detection/
23. Dugan et al., ACL 2024, "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors" — https://aclanthology.org/2024.acl-long.674.pdf (HTML: https://arxiv.org/html/2405.07940v1)
24. Nature, "AI-detection tools have made huge leaps forward — how good are they?" (2026) — https://www.nature.com/articles/d41586-026-02569-3
25. Originality.ai, "How Does AI Content Detection Work?" — https://originality.ai/blog/how-does-ai-content-detection-work
26. Originality.ai, "We Have 99% Accuracy in Detecting AI" — https://originality.ai/blog/ai-accuracy
27. Turnitin Guides, "AI writing detection model" — https://guides.turnitin.com/hc/en-us/articles/28294949544717-AI-writing-detection-model
28. Turnitin, "Understanding false positives within our AI writing detection capabilities" — https://www.turnitin.com/blog/understanding-false-positives-within-our-ai-writing-detection-capabilities ; sentence-level FPR — https://www.turnitin.com/blog/understanding-the-false-positive-rate-for-sentences-of-our-ai-writing-detection-capability
29. Copyleaks, "How does Copyleaks AI Detection work?" — https://help.copyleaks.com/hc/en-us/articles/33816916374285-How-does-Copyleaks-AI-Detection-work ; testing methodology — https://copyleaks.com/ai-content-detector/testing-methodology
30. Fastio, "Copyleaks AI Checker Review 2026" — https://fast.io/resources/copyleaks-ai-detector-review-2026/
31. Winston AI, "Interpreting our AI detection scores" — https://gowinston.ai/interpreting-our-ai-detection-scores/ ; "How Do AI Detectors Work?" — https://gowinston.ai/how-ai-detectors-work/
32. Sapling AI Detector API docs — https://sapling.ai/docs/api/detector/
33. ZeroGPT DeepAnalyse coverage — https://news.bitcoin.com/zerogpt-releases-deepanalyse-detection-upgrade-and-multi-tool-workflow-for-ai-content-verification/
34. Pangram, "How AI Detection Works" — https://www.pangram.com/research/how-it-works
35. Emi & Spero, "Technical Report on the Pangram AI-Generated Text Classifier" (arXiv 2402.14873) — https://arxiv.org/abs/2402.14873
36. Gehrmann, Strobelt & Rush, "GLTR: Statistical Detection and Visualization of Generated Text" (ACL 2019) — https://arxiv.org/abs/1906.04043
37. GLTR GitHub — https://github.com/HendrikStrobelt/detecting-fake-text
38. Mitchell et al., "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature" (ICML 2023) — https://proceedings.mlr.press/v202/mitchell23a.html
39. Bao et al., "Fast-DetectGPT" (ICLR 2024) — https://arxiv.org/html/2310.05130v3
40. Fast-DetectGPT GitHub — https://github.com/baoguangsheng/fast-detect-gpt
41. Hans et al., "Spotting LLMs With Binoculars" (ICML 2024) — https://arxiv.org/html/2401.12070v2 ; code https://github.com/ahans30/Binoculars
42. Verma et al., "Ghostbuster: Detecting Text Ghostwritten by Large Language Models" (NAACL 2024) — https://arxiv.org/abs/2305.15047
43. OpenAI GPT-2 output detector (RoBERTa) — https://github.com/openai/gpt-2-output-dataset/tree/master/detector ; https://openai.com/index/gpt-2-1-5b-release/
44. OpenAI, "New AI classifier for indicating AI-written text" (discontinued) — https://openai.com/index/new-ai-classifier-for-indicating-ai-written-text/
45. Tulchinskii et al., "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts" (NeurIPS 2023) — https://arxiv.org/abs/2306.04723
46. Kirchenbauer et al., "A Watermark for Large Language Models" (ICML 2023) — https://proceedings.mlr.press/v202/kirchenbauer23a/kirchenbauer23a.pdf
47. Same, arXiv abstract — https://arxiv.org/abs/2301.10226
48. Krishna et al., "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense" (NeurIPS 2023) — https://arxiv.org/abs/2303.13408
49. Sadasivan et al., "Can AI-Generated Text be Reliably Detected?" — https://arxiv.org/abs/2303.11156
50. "Diversity Boosts AI-Generated Text Detection" (DivEye) — https://arxiv.org/abs/2509.18880
51. "AI Generated Text Detection" survey (2026) citing IRON and M-RangeDetector — https://arxiv.org/abs/2601.03812
52. "Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text Detectors…" (2026) — https://arxiv.org/abs/2603.17522
53. "Paraphrasing Attack Resilience of Various AI-Generated Text Detection Methods" (2026) — https://arxiv.org/abs/2605.14240
54. "PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks" — https://arxiv.org/abs/2511.00416
55. Creo & Pudasaini, "SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs" — https://arxiv.org/abs/2406.11239 ; GenAIDetect 2025 version https://aclanthology.org/2025.genaidetect-1.1.pdf
56. Wang et al., "Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks" — https://arxiv.org/html/2402.11638
57. Liang et al., "GPT detectors are biased against non-native English writers" (Patterns 2023) — https://arxiv.org/abs/2304.02819 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10382961/
