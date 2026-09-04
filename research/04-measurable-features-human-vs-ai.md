# Measurable Linguistic Features That Separate AI-Generated Text From Human Text

A feature catalog for a humanizer feature-extraction pipeline. Compiled 2026-09-03 from the stylometry, corpus-linguistics and detection literature (sources listed at the end). Each entry gives: definition, how to compute it, direction of the AI/human difference, reported effect sizes, and citation.

Conventions: "AI higher" means LLM output shows more of the feature than matched human text. Effect sizes are Cohen's d unless stated. Numbers are quoted from the cited paper; where two versions of a paper differ (e.g. arXiv v1 vs journal), both are given.

---

## 0. Two framing facts that shape everything below

1. **Instruction tuning, not pretraining, creates most of the fingerprint.** Reinhart et al. (PNAS 2025) found base Llama models use Biber features "at rates similar to human texts", while instruction-tuned variants diverge sharply; Freeburg (2026) found a Llama 3.1 8B base model produces 0.49 em dashes/1,000 words versus 0.00 for the instruct version; Padmakumar & He (ICLR 2024) found co-writing with InstructGPT, but not base GPT-3, significantly reduced lexical and content diversity; Juzek & Ward (COLING 2025) and Juzek (2025) trace overused words like *delve* to learning-from-human-feedback, showing that preference annotators systematically prefer variants containing those words. Practical consequence: the fingerprint is a *register* (formal, dense, hedged-but-confident, structurally tidy), not a bug, and it is fairly stable across model families.

2. **Variance, not mean, is the most robust signal.** Almost every corpus study finds that the strongest separator is that humans are *scattered* and LLMs are *tight*. Zanotto & Aroyehun (2024) report PCA-space variability of 296.75 (human) vs 4.72 (ChatGPT) on Wikipedia, 50.67 vs 7.70 on Reddit, 4.70 vs 1.64 on arXiv. The Oxford DSH essay study (2026) describes AI essays forming "tight, compact clusters" vs dispersed human clusters. A humanizer therefore needs to widen distributions, not just shift means.

---

## 1. Lexical fingerprints

### 1.1 Excess-vocabulary words ("AI words")
- **Definition:** Words whose frequency in post-ChatGPT text exceeds a counterfactual projection from pre-LLM years. Kobak et al. adapt "excess mortality": for each word compute frequency gap δ = p_observed − p_expected and ratio r = p_observed / p_expected, where the counterfactual is a conservative linear extrapolation from 2021–22 (q = p₋₂ + 2·max{p₋₂ − p₋₃, 0}).
- **Compute:** Tokenize, lowercase, lemmatize optionally (spaCy). Build a dictionary of marker words; compute per-1,000-word rate and a weighted "AI-word density" score. For your own corpus, replicate the δ/r method against a pre-2022 reference corpus.
- **Direction:** AI higher, by large ratios for rare words.
- **Numbers (Kobak et al., Science Advances 2025 / arXiv 2406.07016):** 2024 PubMed abstracts: *delves* r = 28.0 (v1: 25.2), *underscores* r = 13.8 (v1: 9.1), *showcasing* r = 10.7 (v1: 9.2); high-gap common words *potential* δ = 0.052, *findings* δ = 0.041, *crucial* δ = 0.037. Other excess style words named: *across, additionally, comprehensive, enhancing, exhibited, insights, notably, particularly, within*. 379 excess style words in 2024 (v1: 280–319), of which 66% verbs and 14–18% adjectives, versus pre-2020 excess words that were almost all content nouns (*coronavirus, lockdown, zika*). Lower bound: ≥13.5% of 2024 abstracts LLM-processed (v1: 10–11%), rising to ~20% in computational fields, China/South Korea/Taiwan, and MDPI/Frontiers journals; 34% for South Korean papers in *Sensors*, 41% for computational papers from China; Nature/Science 7–10%.
- **Liang et al. 2024 (arXiv 2404.01268):** words with a decade of flat frequency in arXiv CS abstracts (2010–22) then a 2023 surge: *realm, intricate, showcasing, pivotal*, plus *meticulous, commendable*. Estimated fraction of LLM-modified sentences by Feb 2024: CS 17.5% (abstracts) / 15.3% (introductions), EE 14.4% / 12.4%, Math 4.9% / 3.9%, Nature portfolio 6.3% / 4.3%; pre-ChatGPT baseline 2.3–3.1%. Companion ICML 2024 paper on peer reviews: 6.5–16.9% of review text LLM-modified, concentrated in reviews submitted near deadlines and with low reviewer confidence.
- **Gray 2024 (arXiv 2403.16887):** tracked 12 adverbs and 12 adjectives favored by LLMs (adjectives include *commendable, innovative, meticulous, intricate, notable, versatile*); *intricate* and *meticulously* roughly doubled in 2023; ≥60,000 papers (~1%) in 2023 likely LLM-assisted.
- **Matsui 2024/2025 (medRxiv; PMC12679996):** 135 candidate AI-influenced terms compiled from 15 studies; 103 showed a modified Z-score ≥ 3.5 in 2024 PubMed. Highest: *delve* (highest Z of all), *underscore, primarily, meticulous, boast, commendable, showcase, surpass, intricate, tapestry, unlocking*. Mixed-model β = 0.655 (p < 0.001) vs control phrases. Important nuance: the rise began around 2020 and accelerated 2023–24, so LLMs amplified an existing formal-academic drift.
- **Reinhart et al. 2025 (PNAS):** GPT-4o uses *camaraderie, tapestry, intricate, palpable* 100+ times above human baseline (ChatGPT variants ~150x for *camaraderie* and *tapestry*), appearing in 23–27% of outputs; Llama variants use *unease* 60–100x. Humans use these words mainly in fiction; LLMs use them in every genre, creating genre misalignment.
- **Why (Juzek & Ward, COLING 2025; Juzek 2025):** 21 focal words; no evidence that architecture, decoding or pretraining data cause the overuse; Llama experiments consistent with RLHF being the source; participants emulating LHF annotators systematically prefer variants containing the focal words. Popular hypothesis: OpenAI's RLHF labelers included Nigerian English speakers, where *delve into* is far more common.

### 1.2 Multi-word "AI phrases" (GPTZero list)
- **Definition:** N-grams whose AI:human frequency ratio is extreme, estimated from ~3.3M documents.
- **Compute:** N-gram (2–5) matching against a phrase list, weighted by log ratio.
- **Numbers (GPTZero, 2024):** *objective study aimed* 269x; *play a significant role in shaping* 182x; *notable works include* 120x; *today's fast-paced world* 107x; *aims to explore* 50x; *showcasing* 20x; *remarked* 18x; *aligns* 16x; *surpassing* 12x; *tragically* 11x; *impacting* 11x. Also *research needed to understand, despite facing, today's digital age, expressed excitement*. Range described as 10x–200x+. GPTZero cautions these are frequency differences, not proof of authorship.

### 1.3 Wikipedia WP:AISIGNS vocabulary lists (era-tagged)
The WikiProject AI Cleanup page is the most detailed community catalog and is useful because it tracks drift over time:
- **2023 to mid-2024:** *Additionally, boasts, bolstered, crucial, delve, emphasizing, enduring, garner, intricate/intricacies, interplay, key, landscape, meticulous(ly), pivotal, underscore, tapestry, testament, valuable, vibrant.*
- **Mid-2024 to mid-2025:** *align with, bolstered, crucial, emphasizing, enhance, enduring, fostering, highlighting, pivotal, showcasing, underscore, vibrant.*
- **Mid-2025 onward:** *emphasizing, enhance, highlighting, showcasing.*
- **Grok-specific:** *causal, empirical, correlate, underscore.*
- **Significance/legacy inflation:** *stands/serves as, is a testament/reminder, crucial/pivotal/vital role, underscores importance, reflects broader, setting the stage for, marks a shift, key turning point, evolving landscape, indelible mark, deeply rooted.*
- **Promotional register:** *boasts a, vibrant, rich, profound, showcasing, exemplifies, commitment to, nestled, in the heart of, groundbreaking, renowned, diverse array.*
- **Copula avoidance:** *serves as / stands as / marks / functions as / represents [a]; boasts / features / offers [a]* in place of *is/are*.
- **Vague attribution:** *Industry reports, Observers have cited, Experts argue, Some critics argue, several sources.*
- **Vague connection:** *in connection with, associated with, widely associated.*
Direction: AI higher for all. No ratios given; treat as a curated lexicon to be weighted by your own corpus.

### 1.4 Lexical diversity (TTR, STTR, MTLD, hapax rate)
- **Definition:** TTR = types/tokens (length-sensitive); STTR = mean TTR over fixed windows; MTLD = mean length of word sequences maintaining TTR ≥ 0.72; hapax legomena rate = fraction of types occurring once.
- **Compute:** `lexicalrichness` (Python) gives TTR, MTLD, HD-D, Maas; `textstat` does not. Use lemmas or surface forms consistently; compare only at matched lengths or use MTLD/HD-D.
- **Direction: contested and model/domain-dependent.** Older/base models and naturalistic corpora: AI lower. Instruction-tuned chat models with repetition penalties in short essays: AI *higher* on TTR/MTLD (they cycle synonyms, "elegant variation").
- **Numbers:** Muñoz-Ortiz et al. (news): human STTR 0.491 vs 0.424–0.466 across six LLMs; MTLD 96.51 vs 57.37 (Falcon-7B) to 94.56 (LLaMa-65B). Herbold et al. (essays): MTLD human 95.72, ChatGPT-3 75.68 (d = 1.06 favoring humans), ChatGPT-4 108.91 (d = −0.60, GPT-4 higher). Frontiers 2025 (ChatGPT vs L2 students): TTR 0.69 vs 0.61, MTLD 118 vs 66.56, Voc-D 105 vs 69.73. EMNLP 2025 profiling paper: human TTR ~0.64, chat models with repetition penalties 0.72–0.92, base models without penalty 0.10–0.57. The 2026 systematic analysis (arXiv 2606.04177) found the *lexical-richness* feature group was the dominant discriminator: removing it cost 13.1% in-domain and 2.5–11.8% across nearly all domains, whereas information-theoretic features cost 1.8% and every other group (POS, readability, dependency, emotion, NER) <0.5% each.
- **Pipeline note:** Human text has a "balanced mix of unique words and repetitions"; both too-low and too-high TTR are suspicious. Model the human *distribution* per genre rather than a threshold.

### 1.5 N-gram (phrase) diversity and phrase repetition
- **Definition:** Type/token ratio at the 2-, 3-, 4-gram level; counts of high-frequency stock trigrams.
- **Compute:** NLTK `ngrams` over tokens; compute unique/total per n; also log-log slope of n-gram rank-frequency curve.
- **Direction:** AI lower diversity, steeper Zipf-like slope (mass concentrated in frequent phrases).
- **Numbers ("Literary Non-Style", 2026, fiction corpora):** 1-gram TTR 0.0057 (LLM) vs 0.0153 (human); 2-gram 0.1648 vs 0.2553; 3-gram 0.4728 vs 0.6560; 4-gram 0.6882 vs 0.8868. *one of the most* 2,787 times in the LLM corpus vs 870 in the human corpus; *as an AI* 110 times in one ChatGPT "memoir". Jiang & Hyland (2025) find ChatGPT essays have fewer lexical bundles but a higher bundle type/token ratio, i.e. "more rigid and formulaic".

### 1.6 Lexical sophistication / word length / Romance-origin vocabulary
- **Definition:** Share of words outside the 2,000 most frequent; mean characters per word; share of Latinate vocabulary.
- **Compute:** `textstat.difficult_words`, `avg_letter_per_word`; frequency lists (SUBTLEX/COCA); etymology via wordfreq + a Latinate suffix heuristic (-tion, -ity, -ize, -ment).
- **Direction:** AI higher. Frontiers 2025: lexical sophistication 0.41 vs 0.35. Ming et al. (2026, cited in Reinhart's notebook) document a "base-to-instruct shift towards Romance-origin vocabulary". Muñoz-Ortiz found LLMs use *fewer* adjectives (6.7–6.9% vs 7.58%) and *fewer* nouns (17.4–17.9% vs 19.69%) but more auxiliaries and pronouns, so "fancier words" does not mean "more content words".

---

## 2. Syntactic and structural fingerprints

### 2.1 Sentence-length variance (burstiness)
- **Definition:** Standard deviation (or coefficient of variation) of words per sentence; also median absolute difference between adjacent sentence lengths; GPTZero's original "burstiness" was variance of per-sentence perplexity.
- **Compute:** spaCy sentencizer; `np.std(lengths)`, `np.std/np.mean`, `np.median(np.abs(np.diff(lengths)))`; per-paragraph SD averaged (Desaire).
- **Direction:** AI lower variance; means similar or slightly longer.
- **Numbers:** Savoy (2024, French presidential speeches): human sentences 21 ± 16.4 words vs GPT 21.7 ± 10.9, i.e. one-third less spread. A 200-sample academic-writing comparison reports human SD 8.2 words vs GPT-4o 4.1. Muñoz-Ortiz: "human texts exhibit more scattered sentence length distributions"; LLMs concentrate in the 10–30-token band. Desaire et al. (2023) found within-paragraph SD of sentence length and the median adjacent-sentence difference to be among the top 20 discriminators (99–100% article-level, 92% paragraph-level accuracy). Caveat: larger models show shrinking burstiness gaps; do not rely on this alone.

### 2.2 Paragraph-length uniformity and paragraph count
- **Definition:** SD of sentences per paragraph and words per paragraph.
- **Compute:** split on blank lines; compute SD and CV.
- **Direction:** AI lower variance (Desaire: humans "vary in the number of sentences and total words per paragraph"; ChatGPT "predictable, uniform").

### 2.3 Present participial (-ing) clauses, especially sentence-final tails
- **Definition:** Non-finite -ing adjunct clauses ("…, highlighting its importance"; "…, reflecting broader trends").
- **Compute:** spaCy: count tokens with `tag_ == "VBG"` whose dep is `advcl` or `acl` and that are not preceded by an auxiliary; flag those after a comma in the final clause of a sentence.
- **Direction:** AI much higher. Reinhart et al.: GPT-4o 5.3x human rate (d = 1.38); 2–5x across models. WP:AISIGNS lists it as a core "superficial analysis" sign with the vocabulary *highlighting, underscoring, emphasizing, ensuring, reflecting, symbolizing, contributing to, fostering, encompassing, enhancing*.

### 2.4 Nominalizations
- **Definition:** Nouns derived from verbs/adjectives (-tion, -ment, -ness, -ity, -ance, -ence, -ism, -al).
- **Compute:** Biber-style suffix rule on tokens tagged NOUN; normalize per 1,000 words (Reinhart's `pybiber` implements the 67-feature Biber tagger).
- **Direction:** AI higher. Reinhart: GPT-4o 2.1x (d = 1.23), 1.5–2x across models. Herbold: 1.06 (human) vs 1.56 (GPT-3, d = −0.88) vs 1.73 (GPT-4, d = −1.35) per sentence.

### 2.5 Clausal complexity / dependency depth
- **Definition:** Clauses per sentence, dependent clauses per T-unit, mean and max dependency-tree depth, mean dependency distance.
- **Compute:** spaCy parse; depth via recursion on `children`; dependency distance = |i(head) − i(dep)| averaged.
- **Direction:** Mostly AI higher for clause counts and depth; humans have shorter constituents and more "optimized" dependency distances. Herbold: clauses per sentence 1.81 vs 2.31 (d = −0.93); tree depth n.s. Frontiers 2025: DC/T 0.75 (ChatGPT) vs 0.57 (students), but mean T-unit length 15.91 vs 23.61 (students longer). Zanotto & Aroyehun: humans "significantly lower syntactic depth" than all LLMs. EMNLP 2025 profiling: human dependency depth 6.83 and dependency length 2.41; GPT-2 16.59/5.15; GPT-4 6.54–6.56, i.e. new models have converged on depth, so this is a *weakening* feature.

### 2.6 Passive voice (agentless)
- **Definition:** be + past participle without a *by*-agent.
- **Compute:** spaCy `auxpass`/`nsubjpass` deps; check for `agent` child.
- **Direction: AI lower** (contrary to folk belief). Reinhart: GPT-4o agentless passives at ~half the human rate. Muñoz-Ortiz: LLMs use more auxiliaries overall (5.4–6.0% vs 3.81%), so auxiliary count is not a passive proxy. Do not "convert passive to active" as a humanizing step; if anything, add some.

### 2.7 Other Biber features (Reinhart 2025; Interpretable Stylistic Variation 2026)
Over-used by instruction-tuned LLMs: *that*-clauses as subjects (2.6x, d = 0.77), phrasal coordination "X and Y" (1.9x, d = 0.81), past-participial postnominal clauses, sentence relatives ("…, which means…"), contractions (in chat models). Under-used: downtoners (*barely, nearly, slightly*: "all LLM variants avoid"), wh-relative clauses as objects, pied-piping relatives ("the manner in which"), concessive subordinators (*although, though*), synthetic negation (*no, neither, nor*), discourse particles (*well, anyway*). Classifier: random forest 66% seven-way (14% chance), 4.2% of LLM texts mistaken for human, 9.8% of human for LLM; per-model 93–98%. Interpretable-variation paper: RF F1 0.67 (human) / 0.99 (LLM), AUC 0.9775; genre explains more variance than source; model choice matters more than decoding strategy.

### 2.8 Sentence-initial connectives and conjunctive adverbs
- **Definition:** Rate of *Additionally, Moreover, Furthermore, However, Overall, In conclusion* etc. at sentence start; also discourse markers overall.
- **Compute:** regex on sentence-initial token + comma; normalize per 100 sentences.
- **Direction: mixed.** *Additionally* is an excess word (Kobak) and an AISIGNS marker. But overall discourse-marker density is *lower* in GPT-4: Herbold 0.57 (human) vs 0.36 (GPT-4, d = 0.85); a 2026 informality study found sentence-initial *however* 11.40/10k tokens (human) vs 8.50 (ChatGPT) and *and/also/but/so/besides* rare or absent in ChatGPT; Desaire found humans prefer *however, but, although* while ChatGPT prefers *others, researchers*. Interpretation: LLMs use a narrow set of formal connectives (*Additionally, Moreover, Furthermore*) and avoid informal ones (*But, So, And, Also*). Feature to extract: share of connectives that are formal vs informal, not raw count.

### 2.9 Em dashes, colons, and punctuation variety
- **Definition:** Em dashes (— or --) per 1,000 words; punctuation entropy (Shannon entropy over the distribution of , . ; : ? ! — ( ) " ').
- **Compute:** regex counts; `scipy.stats.entropy` over punctuation histogram.
- **Direction:** Em dash AI higher for most frontier models; punctuation entropy AI lower (fewer semicolons, parentheses, question marks, exclamation points; more periods).
- **Numbers (Freeburg 2026, "The Last Fingerprint"):** human essays 3.23/1,000 words (range 0.33–17.12). Unconstrained: GPT-4.1 10.62, Claude Opus 4.6 9.09, Claude Sonnet 4 8.29, Claude Haiku 3.5 7.51, DeepSeek V3 6.95, GPT-4o mini 4.16, GPT-4o 4.12, Gemini 2.5 Pro 3.53, GPT-5.4 1.43, Gemini 2.5 Flash 1.28, Llama 3.1/3.3 Instruct 0.00. Under "prose only, no markdown": GPT-4.1 9.10 (barely moves), Claude models fall to 0.18–1.31, Gemini 2.5 Pro 0.00. Explicit em-dash prohibition: GPT-4.1 11.51 → 8.20 → 3.86. Hypothesis: the em dash is "markdown leaking into prose." Model-specific: do not treat em-dash rate as universal; Llama and GPT-5.x barely use it.
- **WP:AISIGNS** also flags: excessive boldface, inline-header bullet lists, title case headings, curly quotes, emoji bullets, and Oxford-comma consistency. Human text shows *inconsistent* serial-comma usage.

### 2.10 Rule of three (tricolon) and parallelism
- **Definition:** Coordinated lists of exactly three items; runs of sentences with matching syntactic frames.
- **Compute:** spaCy: count `conj` chains of length exactly 3 under a shared head; parallelism via POS-sequence similarity of adjacent sentences/clauses (e.g. Jaccard on first-4 POS tags).
- **Direction:** AI higher. The 2026 epistemic-rhetorical miscalibration study reports tricolon at nearly twice the expert rate (Δ = 0.95; reported means 7.13 vs 3.73 per document, p < 0.001) while humans produce rhetorical questions (erotema) at more than twice the LLM rate; "performed hesitancy markers" occur at twice the human density in LLM output. Gorrie's rhetorical analysis: LLMs deploy parallelism, tricolon and explicit antithesis "at every possible opportunity".

### 2.11 Negative parallelism ("not X, but Y"; "it's not about X, it's about Y")
- **Definition:** Contrastive frames including *not only … but also*, *not just … but*, *it's not … it's*, *less about … more about*, *rather than*.
- **Compute:** regex family with clause-boundary constraints; normalize per 1,000 words.
- **Direction:** AI higher, ~3x human (Wikipedia "Negative parallelism", citing Oremus, *The Atlantic*, 2026). WP:AISIGNS treats it as one of the most reliable tells; the giveaway is repetition within a paragraph, not any single use.

### 2.12 Readability indices
- **Definition:** Flesch Reading Ease (FRE), Flesch–Kincaid Grade (FKGL), Gunning Fog, SMOG.
- **Compute:** `textstat.flesch_reading_ease`, `flesch_kincaid_grade`, `gunning_fog`.
- **Direction:** AI harder (higher grade, lower ease) in most comparisons. Frontiers 2025: FKGL 16.61 (ChatGPT) vs 12.12 (students); Gunning Fog 20.10 vs 14.18. Summarization study: FRE 46.49 (ChatGPT) vs 55.52 (CNN/DailyMail). Patient-education study: FRE 33 vs 58.5. Exception: when explicitly asked for plain language (Cochrane summaries), ChatGPT-4o was slightly *easier*. Readability alone had <0.5% marginal discriminative value in the 2606.04177 ablation, so use it as a register check, not a detector.

---

## 3. Statistical / probabilistic fingerprints

### 3.1 Perplexity and mean token log-probability
- **Definition:** PPL = exp(−mean log p(token | context)) under a reference LM (GPT-2, Llama, Falcon).
- **Compute:** HuggingFace `transformers`: run the LM, gather `log_softmax` at gold tokens, average; also compute per-sentence PPL and its SD ("burstiness" in GPTZero's original sense).
- **Direction:** AI lower PPL (higher mean log-prob). HC3 (Guo et al. 2023) found ChatGPT has lower PPL than human text at both sentence and text level. A style-transfer study reports human reference PPL 23.69 under GPT-2 with AI outputs varying by model.
- **Caveats:** (a) Raw PPL fails on prompted text: an LLM answering "write about capybaras" yields high-PPL tokens that were forced by the prompt (Binoculars' "capybara problem"). (b) Short or technical human text is also low-PPL, driving false positives on non-native and formulaic writing. (c) GPTZero abandoned PPL/burstiness as its primary signal in autumn 2023 for a trained deep classifier.

### 3.2 Token rank buckets (GLTR)
- **Definition:** Fraction of tokens in the reference LM's top-10 / top-100 / top-1000 predictions; entropy of the predicted distribution at each position.
- **Compute:** as above but sort logits and record the rank of the gold token; histogram.
- **Direction:** AI has far more top-10 tokens, fewer tail tokens. GLTR (Gehrmann et al. 2019) raised human detection of GPT-2 text from 54% to 72% with no training.

### 3.3 Probability curvature (DetectGPT) and cross-perplexity ratio (Binoculars)
- **DetectGPT (Mitchell et al. 2023):** machine text sits in negative-curvature regions of log p; perturb with T5 and compare log p(original) to mean log p(perturbations). AUROC 0.81 → 0.95 on GPT-NeoX news. Fast-DetectGPT is the sampling-based successor.
- **Binoculars (Hans et al. 2024):** score = log PPL (observer) / log cross-PPL (observer scoring performer's next-token distribution); normalizes away prompt-induced surprise. >90% detection of ChatGPT at 0.01% FPR, zero-shot.
- **Direction:** AI lower Binoculars score; AI more negative curvature.
- **Robustness:** these are the *least* robust to paraphrase and synonym swaps (see §6).

### 3.4 Compressibility and entropy
- **Definition:** Ratio of gzip/zstd-compressed length to raw length; Shannon entropy of the word or character distribution.
- **Compute:** `len(zlib.compress(text.encode()))/len(text)`; `scipy.stats.entropy` over token counts.
- **Direction:** AI more compressible (lower ratio), lower entropy. "The Statistical Signature of LLMs" (2026) finds higher compressibility for LLM text consistently across models, tasks and domains, attenuating at small fragment sizes. Information-theoretic features were the second most useful group in the 2606.04177 ablation (−1.8%).

### 3.5 Zipf / Heaps behavior
- **Definition:** Zipf exponent (slope of log frequency vs log rank) and Heaps exponent (vocabulary growth vs text length).
- **Compute:** fit on token counts with `numpy.polyfit` on log-log; Heaps by tracking types vs tokens.
- **Direction:** LLM text obeys Zipf/Heaps only within a narrow, model-dependent temperature window (near t = 1 for base models, higher for instruct models); human text shows better goodness-of-fit and smaller Zipf exponents at the numeral level. Log-log n-gram slopes are steeper for LLM prose (Literary Non-Style 2026). Useful as a sanity check; noisy on short texts.

### 3.6 Function-word distributions
- **Definition:** Relative frequencies of ~70 function words (Mosteller–Wallace list) or of POS bigrams.
- **Compute:** count per 1,000 words; feed to Burrows' Delta, SVM or random forest.
- **Direction:** Not a simple direction; the *profile* differs and AI profiles are tightly clustered. Oxford DSH essay study (4,346 paired essays, 110 topics): RF 99.93%, SVM 99.26%, Burrows' Delta 97.98% on full essays; substantial degradation on 200-word excerpts. PLOS One Japanese study: function-word unigrams + POS bigrams + phrase patterns gave 99.8% RF accuracy while human judges were at 31.5–56.8%. Kumarage et al. (2023) showed phraseology, punctuation and lexical-diversity stylometrics improve tweet-level detection.

### 3.7 POS distribution
- **Numbers (Muñoz-Ortiz, % of tokens, human vs LLM range):** numbers 1.77 vs 1.95–2.05; symbols 0.09 vs 0.17–0.19; auxiliaries 3.81 vs 5.41–6.02; pronouns 5.32 vs 6.11–7.33; adjectives 7.58 vs 6.69–6.86; nouns 19.69 vs 17.44–17.85; adverbs 3.26 vs 2.61–3.68. The 2025 survey (arXiv 2510.05136) summarizes the literature as AI higher on nouns, determiners and adpositions and lower on adjectives and adverbs; note the noun result conflicts with Muñoz-Ortiz's news data, so treat POS shares as domain-specific. Savoy (French speeches): ChatGPT overuses nouns, possessive determiners, numbers and *nous*; underuses verbs, pronouns and adverbs.

---

## 4. Discourse and pragmatic fingerprints

### 4.1 Interactional metadiscourse: hedges, boosters, attitude markers, self-mention
- **Definition (Hyland 2005):** hedges (*might, perhaps, seem*), boosters (*clearly, certainly*), attitude markers (*unfortunately, remarkably*), self-mention (*I, we, my*), engagement markers (*you, note that*).
- **Compute:** lexicon counts per 10,000 words; first-person pronoun rate; epistemic verbs (*I think, I believe*).
- **Direction:** AI *lower* on genuine stance markers. Jiang & Hyland (2025): ChatGPT essays show significantly fewer hedges, boosters and attitude markers, more interactive transitions and endophoric markers, yielding an "impersonal and expository tone". Herbold: epistemic markers 0.06 (human) vs 0.02 (GPT-3, d = 1.01) vs 0.00 (GPT-4); modals 10.84 vs 8.97 vs 6.12. Reinhart: LLMs avoid downtoners.
- **Apparent paradox:** popular lists say AI "over-hedges" (*It's important to note*, *may potentially*). The resolution from the 2026 miscalibration paper: LLMs produce *performed* hesitancy markers at twice the human density while lacking genuine epistemic hedging; form–meaning divergence Δ = 0.68 (p < 0.001). So extract two separate features: formulaic hedge phrases (AI higher) and clause-level epistemic modality with a first-person subject (AI lower).

### 4.2 Emotional flatness and positivity skew
- **Definition:** Distribution over emotion classes; sentiment polarity variance.
- **Compute:** a transformer emotion classifier (e.g. `j-hartmann/emotion-english-distilroberta-base`) or NRC lexicon; compute per-document class shares and the SD of sentence-level polarity.
- **Direction:** AI more neutral/joyful, less fear/disgust/anger. Muñoz-Ortiz (% of articles): joy 8.30 (human) vs 8.53–9.80; fear 10.77 vs 8.34–9.25; disgust 9.35 vs 7.19–8.32; anger 8.04 vs 6.11–7.72; neutral 52.16 vs 53.65–56.55. Zanotto & Aroyehun: human text "higher emotionality, especially negative emotions", anger more prevalent. Russell et al. experts cited lack of humor and "safe", "straightforward" tone (originality cue, 23.7% of explanations).

### 4.3 Sycophantic, meta and letter-like framing
- **WP:AISIGNS items:** direct address to reader/editor ("I hope this helps", "Certainly!"), knowledge-cutoff disclaimers ("as of my last update"), didactic disclaimers, formulaic "Despite these challenges… / Future outlook" closers, "In summary / In conclusion / Overall" paragraph openers, canned "Awards and recognition" headers, restating the prompt in the first sentence.
- **Compute:** regex lexicon; position-aware (first/last sentence of document or paragraph).
- **Direction:** AI higher; near-zero in edited human prose.

### 4.4 Summary sentence at paragraph end / topic-sentence rigidity
- **Definition:** Paragraphs that open with a claim and close with a generalizing restatement containing *overall, ultimately, thus, in this way, highlighting, underscoring*.
- **Compute:** for each paragraph, test whether the last sentence contains a closer lexeme and has high cosine similarity to the first sentence (sentence-transformers).
- **Direction:** AI higher. Herbold notes ChatGPT's "rigid paragraph-based structure" replaces connectives, which is why discourse-marker density is negatively correlated with coherence in their data.

### 4.5 Vague attribution and "balanced perspective" clichés
- Lexicon from WP:AISIGNS: *Experts argue, Some critics argue, Observers have noted, Industry reports, it is widely regarded, offers a nuanced perspective, on the other hand, while X, Y*. Direction: AI higher. Also lower named-entity density in LLM text across models (multiple 2025–26 studies), so pair the lexicon with an NER-density feature (spaCy `ents` per 100 tokens).

### 4.6 Semantic redundancy / homogeneity
- **Definition:** Mean pairwise cosine similarity between sentences in a document; between documents on the same prompt.
- **Compute:** sentence-transformers embeddings; mean off-diagonal similarity.
- **Direction:** AI higher intra-document similarity (restating), and higher inter-author similarity. Zanotto & Aroyehun: humans have lower semantic similarity between sentences despite richer vocabulary. Padmakumar & He: InstructGPT co-writing raised inter-author similarity and lowered content diversity; Sun et al. (ICML 2025) classify the source model of a text with 97.1% five-way accuracy and show the signal survives rewriting, translation and summarization, i.e. it lives partly in *content choices*, not just wording.

### 4.7 Error-free surface
- **Definition:** Spelling errors, agreement errors, punctuation slips per 1,000 words.
- **Compute:** `language_tool_python` or `pyspellchecker`; count flagged issues.
- **Direction:** AI near zero. Russell et al.: annotators described AI text as "usually grammatically perfect" (grammar/punctuation cue in 24.8% of explanations). Deliberately injected errors do lower detector scores, but human experts were not fooled, and injecting errors reduces perceived quality; use sparingly if at all.

---

## 5. Human-text traits AI lacks (targets for a humanizer)

| Trait | Evidence | Measurable proxy |
|---|---|---|
| Wide sentence-length spread including very short and very long sentences | Savoy ±16.4 vs ±10.9; Desaire SD feature; Muñoz-Ortiz | SD/CV of sentence length; count of sentences <6 and >40 words |
| Fragments, run-ons, digressions | Interpretable-variation paper: LLMs suppress discourse particles and already-rare constructions | Sentences without a finite verb; parenthetical asides per 1,000 words |
| Informal connectives and register shifts | *But/So/And/Also* sentence-initial rare in ChatGPT; humans favor *however/but/although* | Ratio of informal to formal sentence-initial connectives |
| Genuine first-person stance | Herbold epistemic markers 0.06 vs 0.00; Jiang & Hyland | *I think/I'd argue/honestly* per 10k words |
| Negative and mixed emotion | Muñoz-Ortiz fear/disgust/anger deficits | Emotion class shares; polarity SD |
| Concrete specifics, named entities, numbers used as evidence | Lower NE density in LLM text; experts' "originality" cue | NER per 100 tokens; proper-noun share |
| Rhetorical questions | Humans >2x LLM erotema rate | Question marks per 1,000 words |
| Idiosyncratic punctuation (semicolons, parentheses, dashes used inconsistently) | Survey: AI has fewer commas/dashes/parentheses/semicolons/colons and more periods | Punctuation entropy; parentheses per 1,000 words |
| Inconsistent formatting (Oxford comma, quote style) | WP:AISIGNS | Serial-comma consistency ratio |
| Passive constructions where natural | Reinhart: GPT-4o half the human agentless-passive rate | Agentless passives per 100 finite clauses |
| Humor, hedged strong claims, personal anecdote | Russell experts; Gorrie | Hard to compute; use narrative-feature extraction (Russell et al. 2026) or LLM-judge |
| Typos and slips | Russell experts | LanguageTool flags per 1,000 words |

---

## 6. Which features detectors actually use, and which survive paraphrasing

### 6.1 What production detectors use
- **Zero-shot probability detectors** (GLTR, DetectGPT, Fast-DetectGPT, Binoculars) use only §3.1–3.3 features. They are model-agnostic but brittle.
- **Trained neural classifiers** (RoBERTa-based, GPTZero since autumn 2023, Pangram) learn from millions of paired samples; their features are opaque but empirically they key on vocabulary and phrasing (Pangram reports 99.66% TPR at 0.0041% FPR using hard-negative mining with "synthetic mirrors"). GPTZero also surfaces an explicit AI-vocabulary highlighter.
- **Stylometric/hand-crafted classifiers** (Desaire; Kumarage; DSH essay study; 2606.04177) rely on lexical richness, function words, sentence-length variance, punctuation, and discourse markers. These reach 99%+ *in-domain* but degrade on short texts and unseen domains.
- **Human experts** (Russell et al. 2025) reach 99.3% majority-vote accuracy using vocabulary (53.1%), sentence structure (35.9%), grammar/punctuation (24.8%), originality (23.7%), quotations (22.3%), clarity (19.5%).

### 6.2 Robustness to paraphrasing and light editing
- **Probability features collapse.** Sadasivan et al.: recursive DIPPER paraphrasing drives a retrieval defense from 100% to <60% and detection to 25% after five rounds. Adversarial Paraphrasing (2025): Fast-DetectGPT TPR@1%FPR −98.96%, RADAR −64.49%, average −87.88% across detectors. Paraphrase-resilience study (2026): Binoculars F1 0.7497 → 0.5533 after GPTinf paraphrasing, RoBERTa 0.7586 → 0.6594, simple text features (word length, lexical diversity, punctuation, sentence-length SD, stopword ratio) 0.7207 → 0.6682, "almost no degradation". RAID (ACL 2024): synonym swapping cut Binoculars-type metric detectors by 36.1%; homoglyphs cut most detectors ~40.6% but GPTZero only 0.3%.
- **Stylometric and lexical features are comparatively durable**, because a paraphraser preserves register, sentence rhythm and often the AI vocabulary. Russell's experts stayed at 100% on paraphrased and "humanized" o1-pro articles where Binoculars fell to 6.7% and Fast-DetectGPT to 23.3%.
- **Content-level idiosyncrasies are the most durable** (Sun et al.: survive rewriting, translation, summarization), which means a humanizer that only rewrites surface form leaves a residue in *what is said* (rule-of-three coverage, balanced-perspective structure, generic examples).
- **Implication for a humanizer:** fooling perplexity-based detectors is easy and largely irrelevant; the hard targets are vocabulary, variance, participial tails, tricolon/negative-parallelism density, metadiscourse profile and emotional range, because those are what both trained classifiers and expert humans use.

---

## Top 20 features to extract for a humanizer feature-extraction pipeline (ranked)

Ranking weighs (a) reported effect size, (b) consistency across studies and models, (c) robustness to paraphrase, (d) computability with spaCy/NLTK/textstat.

1. **AI-vocabulary density** (weighted lexicon: Kobak excess words, Liang, Matsui, Reinhart, WP:AISIGNS era lists, GPTZero phrases) per 1,000 words. Largest ratios in the literature (10x–270x); top cue for expert humans; survives paraphrase.
2. **Sentence-length SD / CV and adjacent-sentence length difference** (burstiness). ±16.4 vs ±10.9; 8.2 vs 4.1; top Desaire feature; robust to paraphrase.
3. **Present-participial clause rate, especially sentence-final -ing tails** (5.3x, d = 1.38).
4. **Nominalization rate** (2.1x, d = 1.23; Herbold d = −1.35).
5. **Lexical richness bundle: MTLD, HD-D, hapax rate, 2–4-gram TTR**, scored as distance from the human distribution for the genre (dominant feature group in 2606.04177; direction is genre/model dependent).
6. **Tricolon and parallel-structure density** (≈2x human).
7. **Negative-parallelism / "not X but Y" family rate** (≈3x human).
8. **Em dash rate + punctuation entropy** (GPT-4.1 10.62 vs human 3.23 per 1,000; AI has lower punctuation variety). Model-specific; weight by suspected source.
9. **Formal vs informal sentence-initial connective ratio** (*Additionally/Moreover/Furthermore* vs *But/So/And/Also*; *however* 8.50 vs 11.40 per 10k).
10. **Interactional metadiscourse profile:** hedges/boosters/attitude markers/self-mention per 10k (AI lower) versus formulaic hedge phrases (*it's important to note*; AI higher).
11. **Epistemic first-person markers and modal density** (0.06 vs 0.00; 10.84 vs 6.12).
12. **Emotion distribution and polarity variance** (fear 10.77 vs 8.3–9.3; disgust 9.35 vs 7.2–8.3; neutral share).
13. **Agentless passive rate** (AI ~half human; add, don't remove).
14. **Downtoner and concessive-subordinator rate** (*barely, nearly, although, though*; AI avoids).
15. **Intra-document semantic redundancy** (mean sentence-embedding similarity; AI higher) and paragraph-closer detection (*Overall/Ultimately/…, highlighting*).
16. **Paragraph-length SD and structural over-formatting** (bullets, bold, headers, title case; AI higher).
17. **Named-entity and numeral-as-evidence density** (AI lower NE density; AI higher raw numerals/symbols: 1.77 vs ~2.0%, 0.09 vs 0.18%).
18. **Rhetorical-question rate** (humans >2x).
19. **Compressibility ratio and top-10 token-rank share** under a small LM (cheap proxies for PPL; AI more compressible, more top-10 tokens). Useful for monitoring, weak for humanizing because paraphrase already defeats them.
20. **Surface-error rate** (LanguageTool flags per 1,000 words; AI ≈ 0). Track it; inject sparingly.

Features deliberately left out of the top 20: raw perplexity/Binoculars score (brittle and prompt-sensitive), readability grade (<0.5% marginal value), dependency depth (GPT-4-class models have converged to human levels), raw POS shares (domain-inconsistent).

---

## Sources

- Kobak, González-Márquez, Horvát, Lause. "Delving into ChatGPT usage in academic writing through excess vocabulary" / "Delving into LLM-assisted writing in biomedical publications through excess vocabulary." arXiv 2406.07016 (v1/v2); Science Advances 2025, PMC12219543. https://arxiv.org/abs/2406.07016 ; https://arxiv.org/html/2406.07016v1 ; https://arxiv.org/html/2406.07016v2 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12219543/
- Liang et al. "Mapping the Increasing Use of LLMs in Scientific Papers." arXiv 2404.01268. https://arxiv.org/abs/2404.01268 ; https://arxiv.org/html/2404.01268
- Liang et al. "Monitoring AI-Modified Content at Scale" (ICML 2024). https://arxiv.org/abs/2403.07183
- Gray, A. "ChatGPT 'contamination': estimating the prevalence of LLMs in the scholarly literature." arXiv 2403.16887. https://arxiv.org/abs/2403.16887
- Matsui, K. "Delving into PubMed Records…" medRxiv 2024; PMC12679996 (2025). https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12679996/
- Juzek & Ward. "Why Does ChatGPT 'Delve' So Much?" COLING 2025. https://arxiv.org/abs/2412.11385 ; https://aclanthology.org/2025.coling-main.426/
- Juzek. "Word Overuse and Alignment in Large Language Models: The Influence of Learning from Human Feedback." arXiv 2508.01930. https://arxiv.org/abs/2508.01930
- Wikipedia:Signs of AI writing (WikiProject AI Cleanup). https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- Wikipedia: Negative parallelism. https://en.wikipedia.org/wiki/Negative_parallelism
- GPTZero. "Top 10 Most Common Words Used by AI"; AI Vocabulary page. https://gptzero.me/news/most-common-ai-vocabulary/ ; https://gptzero.me/ai-vocabulary ; EdScoop coverage https://edscoop.com/gptzero-common-ai-words-detection-education-2024/
- GPTZero. "What is perplexity & burstiness for AI detection?" https://gptzero.me/news/perplexity-and-burstiness-what-is-it/
- Reinhart, Markey, Laudenbach, Pantusen, Yurko, Weinberg, Brown. "Do LLMs write like humans? Variation in grammatical and rhetorical styles." PNAS 2025; arXiv 2410.16107. https://www.pnas.org/doi/10.1073/pnas.2422455122 ; https://arxiv.org/html/2410.16107 ; CMU news https://www.cmu.edu/dietrich/news/news-stories/2025/large-language-models-writing-text ; Reinhart's notebook https://www.refsmmat.com/notebooks/llm-style.html
- Herbold, Hautli-Janisz, Heuer, Kikteva, Trautsch. "A large-scale comparison of human-written versus ChatGPT-generated essays." Scientific Reports 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10616290/ ; https://arxiv.org/abs/2304.14276
- Muñoz-Ortiz, Gómez-Rodríguez, Vilares. "Contrasting Linguistic Patterns in Human and LLM-Generated News Text." Artificial Intelligence Review 2024. https://arxiv.org/abs/2308.09067 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11422446/
- Zanotto & Aroyehun. "Human Variability vs. Machine Consistency: A Linguistic Analysis of Texts Generated by Humans and LLMs." arXiv 2412.03025. https://arxiv.org/html/2412.03025
- "Linguistic and Embedding-Based Profiling of Texts Generated by Humans and Large Language Models." EMNLP 2025; arXiv 2507.13614. https://arxiv.org/html/2507.13614
- "Interpretable Stylistic Variation in Human and LLM Writing Across Genres, Models, and Decoding Strategies." arXiv 2604.14111. https://arxiv.org/html/2604.14111v1
- "A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models." arXiv 2606.04177. https://arxiv.org/html/2606.04177
- "Linguistic Characteristics of AI-Generated Text: A Survey." arXiv 2510.05136. https://arxiv.org/abs/2510.05136 ; https://www.alphaxiv.org/abs/2510.05136
- Berber Sardinha. "AI-generated vs human-authored texts: A multidimensional comparison." Applied Corpus Linguistics 2024. https://www.sciencedirect.com/science/article/abs/pii/S2666799123000436
- Jiang & Hyland. "Rhetorical distinctions: Comparing metadiscourse in essays by ChatGPT and students." English for Specific Purposes 2025. https://ueaeprints.uea.ac.uk/id/eprint/99123/
- Frontiers in Education 2025. "Lexical diversity, syntactic complexity, and readability: a corpus-based analysis of ChatGPT and L2 student essays." https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1616935/full
- "Informality features in AI-generated academic writing: A corpus-based comparison between human and AI." Journal of English for Academic Purposes 2026. https://www.sciencedirect.com/science/article/pii/S1475158526000019
- Desaire et al. "Distinguishing academic science writing from humans or ChatGPT with over 99% accuracy using off-the-shelf machine learning tools." Cell Reports Physical Science 2023. https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(23)00200-X ; https://techxplore.com/news/2023-06-ai-generated-academic-science-accuracy.html
- Savoy. "ChatGPT as speechwriter for the French presidents." arXiv 2411.18382. https://arxiv.org/abs/2411.18382
- Freeburg, E. M. "The Last Fingerprint: How Markdown Training Shapes LLM Prose." arXiv 2603.27006. https://arxiv.org/html/2603.27006v1 ; McGill OSS summary https://www.mcgill.ca/oss/article/critical-thinking-student-contributors-technology/why-did-llms-steal-our-em-dashes
- "Saying More Than They Know: A Framework for Quantifying Epistemic-Rhetorical Miscalibration in Large Language Models." arXiv 2604.19768. https://arxiv.org/abs/2604.19768
- Gorrie, C. "Why ChatGPT writes like that." Dead Language Society. https://www.deadlanguagesociety.com/p/rhetorical-analysis-ai
- "Literary Non-Style in LLM-Generated Text." arXiv 2607.17228. https://arxiv.org/html/2607.17228v1
- "The Statistical Signature of LLMs." arXiv 2602.18152. https://arxiv.org/abs/2602.18152
- "Zipf's and Heaps' Laws for Tokens and LLM-generated Texts." Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.837/
- "Stylometric detection of AI-generated texts: evidence from human and machine-written essays." Digital Scholarship in the Humanities 2026. https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
- "Stylometry can reveal artificial intelligence authorship, but humans struggle: A comparison of human and seven LLMs in Japanese." PLOS One 2025. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335369
- Kumarage et al. "Stylometric Detection of AI-Generated Text in Twitter Timelines." arXiv 2303.03697. https://arxiv.org/abs/2303.03697
- Guo et al. "How Close is ChatGPT to Human Experts? (HC3)." arXiv 2301.07597. https://arxiv.org/abs/2301.07597
- Gehrmann, Strobelt, Rush. "GLTR: Statistical Detection and Visualization of Generated Text." ACL 2019. https://aclanthology.org/P19-3019/
- Mitchell et al. "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature." ICML 2023. https://arxiv.org/abs/2301.11305
- Hans et al. "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text." ICML 2024. https://arxiv.org/abs/2401.12070 ; https://github.com/ahans30/Binoculars
- Russell et al. "People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text." arXiv 2501.15654. https://arxiv.org/html/2501.15654
- Sun, Yin, Xu, Kolter, Liu. "Idiosyncrasies in Large Language Models." ICML 2025. https://arxiv.org/abs/2502.12150
- Padmakumar & He. "Does Writing with Language Models Reduce Content Diversity?" ICLR 2024. https://arxiv.org/abs/2309.05196
- Sadasivan et al. "Can AI-Generated Text be Reliably Detected?" https://www.alphaxiv.org/abs/2303.11156
- Krishna et al. "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense." https://arxiv.org/pdf/2303.13408
- "Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text." arXiv 2506.07001. https://arxiv.org/abs/2506.07001
- "Paraphrasing Attack Resilience of Various AI-Generated Text Detection Methods." arXiv 2605.14240. https://arxiv.org/html/2605.14240v1
- Dugan et al. "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors." ACL 2024. https://arxiv.org/abs/2405.07940
- Emi & Spero. "Technical Report on the Pangram AI-Generated Text Classifier"; Pangram 4 Technical Report. https://arxiv.org/abs/2402.14873 ; https://arxiv.org/html/2607.27183v1
- Readability comparisons: patient-education FRE 33 vs 58.5 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12278881/ ; summarization FRE 46.49 vs 55.52 https://arxiv.org/pdf/2306.07799 ; Cochrane plain-language RCT https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12302524/
- The Conversation. "Too many em dashes? Weird words like 'delves'? Spotting text written by ChatGPT is still more art than science." https://theconversation.com/too-many-em-dashes-weird-words-like-delves-spotting-text-written-by-chatgpt-is-still-more-art-than-science-259629
