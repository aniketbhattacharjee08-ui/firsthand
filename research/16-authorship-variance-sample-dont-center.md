# Stylometric Variance: Why You Sample a Distribution, Never Target a Mean

**Bottom line.** Every line of evidence says sample from a between-author distribution rather than target a genre mean. Two independent reasons.

1. Within a single genre, the spread across individual authors on standard features is roughly as large as the spread between wildly different genres.
2. The features a humanizer would naively target (mean sentence length, type-token ratio, Yule's K) are demonstrably the *weakest* author discriminators, while function-word and character-n-gram profiles carry almost all the individuating signal.

Also: individual authors differ enormously in how distinctive they are, from 7% to 81% identifiability across twelve authors in one register. **There is no single "human profile" to hit.**

---

## 1. The decisive number: within-genre spread versus between-genre spread

Computed from LIWC2015 published statistics (Pennebaker et al. 2015; six corpora, over 80,000 writers, 231M words). Their "Mean SDs" column is the typical within-genre, across-author standard deviation. Compare it to the standard deviation across the six genre means, where those genres are as far apart as English registers get (blogs, expressive writing, novels, natural speech, New York Times, Twitter).

| Feature | Between-genre SD | Within-genre SD | Within / Between |
|---|---|---|---|
| Total function words | 6.29 | 5.13 | **0.82** |
| Total pronouns | 4.58 | 3.61 | 0.79 |
| 1st person singular | 2.96 | 2.46 | 0.83 |
| Words over 6 letters | 4.39 | 3.76 | 0.86 |
| Articles | 1.82 | 1.79 | **0.99** |
| Auxiliary verbs | 2.24 | 2.04 | 0.91 |
| Adverbs | 1.69 | 1.61 | 0.95 |
| Prepositions | 1.65 | 2.11 | **1.28** |
| Conjunctions | 1.18 | 1.57 | **1.33** |
| Negations | 0.58 | 0.86 | **1.48** |
| Adjectives | 0.25 | 1.30 | **5.22** |
| Quantifiers | 0.23 | 0.83 | **3.62** |
| Words per sentence | 3.62 | 16.38 | 4.53 |

**For essentially every function-word feature, the spread among individual texts inside one genre is 0.8× to 1.5× the spread between six radically different genres.** A text sitting exactly at the genre mean is roughly one full genre-distance from a typical real author. It occupies an atypically empty region of feature space.

**Reference distributions to sample from** (grand mean, within-genre SD):

| Feature | Mean | SD | CV | ±1 SD range |
|---|---|---|---|---|
| Words/sentence (speech excluded) | 17.40 | 16.38 | 94% | 1.0 - 33.8 |
| Total function words % | 51.87 | 5.13 | 10% | 46.7 - 57.0 |
| Total pronouns % | 15.22 | 3.61 | 24% | 11.6 - 18.8 |
| 1st person singular % | 4.99 | 2.46 | **49%** | 2.5 - 7.5 |
| Articles % | 6.51 | 1.79 | 28% | 4.7 - 8.3 |
| Prepositions % | 12.93 | 2.11 | 16% | 10.8 - 15.0 |
| Auxiliary verbs % | 8.53 | 2.04 | 24% | 6.5 - 10.6 |
| Adverbs % | 5.27 | 1.61 | 31% | 3.7 - 6.9 |
| Conjunctions % | 5.90 | 1.57 | 27% | 4.3 - 7.5 |
| Negations % | 1.66 | 0.86 | **52%** | 0.8 - 2.5 |
| Words over 6 letters % | 15.60 | 3.76 | 24% | 11.8 - 19.4 |
| Adjectives % | 4.49 | 1.30 | 29% | 3.2 - 5.8 |
| Quantifiers % | 2.02 | 0.83 | 41% | 1.2 - 2.9 |

---

## 2. Sentence length and lexical diversity are near-worthless discriminators

**Mosteller and Wallace on the Federalist Papers.** Hamilton: mean sentence length 34.55 words, SD 19.2. Madison: mean 34.59, SD 20.3. **The two authors' means differ by 0.04 words while within-author SD is about 20 words.** They discarded sentence length entirely.

**Grieve (2005).** Telegraph Columnist Corpus: 40 columnists, 1,600 texts, 1.5M words, one register, 200 random author-set permutations. Accuracy by number of candidate authors:

| Feature | 40 authors | 10 | 5 | 2 |
|---|---|---|---|---|
| **Average sentence length (words)** | **6** | 21 | 37 | **69** |
| Average word length | 7 | 22 | 39 | 70 |
| Sentence-length profile (5-word bins) | 11 | 29 | 44 | 74 |
| Unrestricted TTR | 8 | 27 | 44 | 75 |
| **Yule's K / Simpson's D** | 6 | 18 | 33 | **65** |
| Entropy | 8 | 24 | 40 | 72 |
| 3-word collocations | 3 | 11 | 21 | 53 |
| **Word-frequency profile** | **48** | 67 | 77 | 88 |
| **Word + punctuation profile** | **63** | 80 | 87 | **95** |
| 2-gram profile | 65 | 79 | 86 | 94 |
| **Weighted combination of 16 algorithms** | **69** | 85 | 91 | **97** |

Mean sentence length cannot reliably separate even two authors (69%, below Grieve's own 75% usability threshold). Function-word and punctuation profiles reach 88-95% for two authors and 48-65% for forty, against 2.5% chance. **Punctuation frequency alone, only eight features, is one of the strongest single indicators in the entire study.**

**Design implication.** Do not spend effort tuning mean sentence length or lexical-diversity indices for authenticity. They carry almost no individuating signal. But they *are* cheap for detectors to compute, so match the **distribution including within-document variance**, not the mean.

---

## 3. Topic contaminates every style measurement

**Sari et al., COLING 2018.** Feature ablation across four datasets. When topics are homogeneous, removing style costs 4-8 accuracy points and removing content costs about 0-3. When topics differ, it inverts. **The key number: 50 journalists writing the same newswire genre at about 584 words each yields only 60.6% attributability.** That is roughly the regime a humanizer operates in.

**Wegmann et al., RepL4NLP 2022.** Authorship verification AUC by negative-sampling strategy:

| Training content control | Same-conversation negatives | Same-domain | Random negatives |
|---|---|---|---|
| None | **.58** | .63 | **.79** |
| Conversation | .69 | .70 | .71 |
| Domain | .68 | .71 | .73 |

A model trained on random negatives scores .79 but only **.58 when the negative comes from the same conversation**, a 21-point collapse against a .50 floor. Conversation-trained models are flat, meaning they are the only ones not topic-inflated.

**PAN 2020 organizers** quantified it directly: regressing system predictions on topic similarity gives R² = 0.16 on correctly solved pairs and 0.03 on incorrectly solved ones. Their conclusion was that topic dependence "is one of the main causes driving misclassifications."

---

## 4. The PAN arc: verification is far from solved

**The single most striking result in the literature.** PAN 2021 top systems scored above 0.93 overall on fanfiction. Run unchanged on the PAN 2022 cross-discourse-type test set:

| System | AUROC 2021 | AUROC 2022 | Overall 2022 |
|---|---|---|---|
| boenninghoff21 | **0.9869** | **0.513** | 0.310 |
| embarcaderoruiz21 | 0.9697 | 0.538 | 0.360 |
| weerasinghe21 | 0.9719 | 0.488 | 0.306 |

boenninghoff21 emitted **10 positives out of 10,478 pairs**. In PAN 2022 the naive character-n-gram cosine baseline beat every participant, and the best AUROC anywhere was 0.598. PAN 2023 topped out at 0.616. Organizers: "The very good results obtained by the top-performing submissions may have given the false impression that authorship verification is an almost solved problem. This is in fact not the case."

**The clean topic-versus-style experiment.** PAN multi-author style-change detection holds authors, genre and pipeline constant and varies only whether adjacent paragraphs share a topic. PAN 2024, paragraph level:

| Team | Easy (varied topics) | Hard (same topic) |
|---|---|---|
| no-999 | **0.991** | 0.832 |
| fosu-stu | 0.987 | 0.834 |
| nycu-nlp | 0.964 | **0.863** |
| karami-sh | 0.972 | 0.642 |

**Best easy 0.991 versus best hard 0.863. In error terms 0.9% to 13.7%, a 15× increase**, from topic control alone.

**Most relevant single number for this product.** PAN 2025 subtask 2 covers six degrees of human-AI collaboration including "machine-humanized" text, with over 500,000 examples. **The best system reached F1 of only 0.65**, best baseline 0.48. The detection frontier for humanized text is nowhere near saturated.

**PAN 2021 organizers, verbatim:** "Even within a single genre, textual features that work well to differentiate author A from a set of peers might fail to separate author B from the same set of peers. Modeling authorial writing style requires bespoke models that are tailored to the characteristics of a single author."

---

## 5. Individual distinctiveness varies 11-fold

Wright's Enron study, 12 authors, 300 tests each:

| Author | % identified | Author | % |
|---|---|---|---|
| Lavorato | **80.67** | Nemec | 64.00 |
| Kaminski | 77.67 | Derrick | 63.00 |
| Allen | 76.33 | Steffes | 52.00 |
| Arnold | 68.00 | Dorland | 43.67 |
| Germany | 67.00 | Haedicke | 42.00 |
| Farmer | 64.33 | Zipper | **7.33** |

Mean 58.9%, range 7.33% to 80.67%, **uncorrelated with data volume**. Zhu and Jurgens found intra-author consistency and inter-author distinctiveness are both normally distributed with no meaningful correlation between them. "Human" is not one distribution. It is a distribution over distributions.

---

## 6. How many words are needed

- **Eder (2015):** minimum reliable sample 2,500 words (Latin prose) to 5,000 words (English, German, Polish, Hungarian novels). Under 5,000 is poor; **under 3,000 is "simply disastrous."**
- **Eder (2017)** revises downward: under 2,000 words can suffice, **but only for texts with a clear authorial signal.** Some texts are never correctly attributed at any length.
- **Nini (2023):** with a large reference corpus (300,000 known tokens per author), a **50-token query already beats chance at over 40% top-1**; accuracy approaches 100% at 5,000-token queries. A large reference corpus compensates far more than a large query.
- **Evert et al. (2018):** n-gram tracing hits about 50% at 250 words against 30,000-token profiles, and over 80% at 1,000 tokens.
- **Below about 500 words nothing works well**, which cuts both ways.

**Evert's sampling experiment, directly relevant.** Drawing 25 authors from a pool of 131, accuracy ranges 80-100% for Cosine Delta. Holding the same 25 authors and only resampling texts, accuracy still fluctuates by 15 percentage points. **Which authors you happen to compare matters more than which algorithm you use.**

---

## 7. Neural style embeddings

**LUAR** outputs 512-dimensional unit-sphere vectors. Zero-shot cross-domain R@8 shows the Reddit-trained model keeps over 80% of its in-domain performance transferred to Amazon and fanfiction; the reverse does not hold. Diminishing returns are steep: Amazon 100K to 250K authors gives +1.1% R@8.

**Candidate-set degradation (STAR, 1,616 unseen Reddit authors, top-1 accuracy):**

| Model | Support docs | N=10 | 100 | 500 | 1616 |
|---|---|---|---|---|---|
| STAR | 8 | 99.30 | 95.39 | 88.06 | **80.42** |
| STAR | 1 | 76.05 | 46.05 | 30.37 | **21.39** |
| PART | 8 | 94.85 | 78.92 | 63.61 | 51.67 |
| Wegmann 2022 style embeddings | 8 | 52.05 | 20.01 | 9.09 | **5.09** |
| RoBERTa | 8 | 37.95 | 15.44 | 8.07 | 5.12 |

**Narayanan et al., IEEE S&P 2012.** 100,000 blogs, 1B words, 1,188 features. Top-1 over 20% from about three posts. **A single post of about 305 words: 7.5%.** With abstention, precision rises to over 80% at half recall. PCA: the data can be represented to 96% accuracy using **500 dimensions**.

**STEL-or-Content**, the task where a model must pick style over content: **every representation scores below the 0.5 random baseline.** RoBERTa base 0.05, best (conversation-trained) 0.42. Content dominates style in every embedding tested.

**StyleDistance** (NAACL 2025) covers 40 style features across 7 categories. STEL / STEL-or-Content: roberta-base .90/.05, LUAR .86/.03, StyleDistance .87/**.29**. A geometry note from its appendix matters: a natural-negative-trained space **fragments a single style into multiple modes**, so "informal" is not one point.

**LISA** curated 1,255,874 interpretable style attributes down to **768 explicitly named non-redundant dimensions.** This is the closest published estimate of how many nameable discriminative style features exist.

**The sharpest finding for our design.** Wang et al., TACL 2023, on scaling training authors: UAR trained on 100K authors beats UAR trained on 5M authors at identifying *broad stylistic categories*, even though the 5M model is far better at identifying *individual authors*. Verbatim: "training UAR on more authors produces representations that are more discriminative of individual authors, something which is at odds with identifying broad stylistic categories." **Idiolect-level and register-level style geometry pull in opposite directions.**

Same paper, paraphrase attack on authorship models (MRR before/after):

| Domain | Model | Original | Paraphrased | Δ |
|---|---|---|---|---|
| Reddit | UAR | 0.263 | 0.026 | **0.237** |
| Reddit | SBERT | 0.043 | 0.026 | 0.017 |
| Amazon | UAR | 0.266 | 0.025 | **0.241** |
| fanfic | UAR | 0.325 | 0.139 | 0.186 |

**UAR loses about 90% of its Reddit MRR to paraphrasing.** The degradation is not explained by content overlap (Kendall's τ around −0.02 to −0.09).

**STEB (2026)**, 96 datasets, 40 models. Clustering V-measure: best of all 40 models is STAR at 25.92; best semantic embedding 11.06. Content-independence sub-score: StyleDistance 44.23 versus LUAR-CRUD 10.57. **The representations that best separate individual authors are the ones most entangled with content.**

---

## 8. Cross-genre and cross-mode collapse

**Stamatatos (2017)**, same 13 Guardian authors throughout: single-domain 80.6% → cross-topic 66.2% → **cross-genre 39.4%**. Usable feature counts collapse from 18,859 to about 1,564, and only function words survive.

**Overdorf and Greenstadt (2016)**: in-domain 83.5% → **cross-domain 34.3%** with identical methods and training data.

**CROSSNEWS (AAAI 2025)**, 500 authors, chance 0.2%: Article→Article 56.9% versus **Article→Tweet 18.4%**, a 3× drop.

**Mode effects (Wang, Juola and Riddell, EACL 2021).** Same 18 people writing offline in a word processor versus online: **mean sentence length 116.88 → 73.84 characters, a difference of 42.36 characters, Cohen's d = −5.44** ("huge"). 17 of 18 participants used longer sentences offline. Formality was matched, so this is not a formality artifact. **The same human's sentence-length distribution shifts enormously with writing mode**, which is a strong argument that no single sentence-length target is "human."

---

## 9. Idiolectal drift over time

**Seminck et al. (2022)**, 11 prolific 19th-century French novelists, 37M words. 10 of 11 show stronger-than-chance monotone chronological drift. Predicting year of writing from lexico-morphosyntactic motifs *within a single author*:

| Author | r | R² | Features | RMSE (years) |
|---|---|---|---|---|
| Jules Verne | 0.94 | **0.89** | 57 | 3.91 |
| Émile Zola | 0.92 | 0.83 | 34 | 4.50 |
| Balzac | 0.90 | 0.78 | 42 | **2.44** |
| George Sand | 0.88 | 0.77 | 61 | 6.13 |
| Ponson du Terrail | **−0.04** | −0.55 | 10 | 5.69 |

**Style alone dates a novel to within 2.5 to 6 years within a single author, explaining up to 89% of chronological variance, from only 10 to 61 features.** Toledo et al. (2022) confirm across 11 English novelists: all showed significant style change, most classifiable early-versus-late at over 80%.

---

## 10. Design implications for the humanizer

1. **Sample, do not center.** Within-genre across-author SD is 0.8-1.5× the between-genre SD for function-word features. Sample function-word rate from about N(51.9, 5.1), articles from N(6.5, 1.8), first-person-singular from N(5.0, 2.5). Then **hold that persona fixed across the document**, because within-author consistency is what verification systems key on.

2. **Do not optimize mean sentence length or lexical-diversity indices for authenticity.** They carry almost no individuating signal. Match the distribution and its within-document variance instead.

3. **The signal that matters is function words, punctuation, and character 8-9-grams / word 2-3-grams.** Punctuation alone, eight features, reaches 95% on two authors. The persona vector should be rich, on the order of hundreds of dimensions, not a handful of readability knobs.

4. **Pick a target author, not a target genre.** Distinctiveness varies 11-fold across individuals and is uncorrelated with data volume.

5. **Add drift for long or multi-document output.** A perfectly stationary style is itself anomalous over a career-length corpus, though probably irrelevant for a single document.

6. **Length thresholds.** Under about 2,000 words attribution is weak; 2,500-5,000 is the classical reliability floor; under about 500 words nothing works well.

7. **Paraphrasing already breaks authorship embeddings** (UAR loses 90% of Reddit MRR). That is encouraging for evasion, but the style-embedding detectors used against humanizers are trained differently, so do not over-read it.

---

## Gaps

- **No ICC or variance-component estimate for "author" as a random effect on stylometric features exists in the retrievable literature.** This is a genuine hole worth flagging as an open question.
- Grieve (2023) CLLT internal numbers, McCarthy and Jarvis (2010) MTLD norming values, Rybicki and Eder (2011) MFW curves, and Barlow (2013) primary numbers are all paywalled and unobtained.
- No intrinsic-dimension measurement (TwoNN, MLE, participation ratio) of any neural style embedding space exists. Closest are Narayanan's PCA (500 dimensions for 96%) and LISA's 768-of-1.26M curation.
- No published across-author mean and SD of mean sentence length for a set of named academic authors.

**One URL correction:** Wegmann, Schraagen and Nguyen 2022 is at aclanthology.org/2022.repl4nlp-1.26/, not the Findings-of-ACL ID often cited.

---

## Addendum: additional verified figures

**LIWC Analytic score by genre** (useful as a formality target; higher = more formal/analytic):

| Genre | Analytic |
|---|---|
| New York Times | 92.57 |
| Novels | 70.33 |
| Twitter | 61.94 |
| Blogs | 49.89 |
| Expressive writing | 44.88 |
| Natural speech | 18.43 |

Grand mean 56.34, within-genre SD 17.58. Academic prose sits near the NYT end, so an Analytic score in the 80s-90s is the target, but with a wide sampled spread.

**Punctuation n-grams invert in usefulness across domains** (Sapkota et al., NAACL 2015). Mid-word punctuation n-grams are the *worst* feature in-domain (12.4% on CCAT-50) and the *best* cross-domain (46.1% on Guardian1). Word n-grams do the reverse. Punctuation habits survive topic and genre shift better than vocabulary does, which makes them both a robust detector signal and a thing we must get right.

**Content words carry slightly MORE idiolectal signal than function words** (Zhu and Jurgens, EMNLP 2021), even across topics: content-words-only F1 0.795 (Amazon) and 0.708 (Reddit) versus function-words-only 0.786 and 0.683. Word-order shuffling barely hurts. This complicates the usual "style lives in function words" assumption and reinforces the content-origin finding in report 12.

**Domain adaptation partially recovers cross-domain loss** (Overdorf and Greenstadt, PoPETs 2016): 83.5% in-domain falls to 34.3% cross-domain but recovers to about 70% with adaptation. Part-of-speech features degrade least across the shift.

**Feature counts collapse across genre shift** (Stamatatos 2017): usable features fall from 18,859 in-domain to about 1,564 cross-genre, and the optimal text-masking parameter drops from 2,000-4,000 to 150. Only function words survive a genre change.

**Methodological note on this research.** The shared web-search budget was exhausted, so discovery ran through the OpenAlex API and extraction through direct fetch plus local PDF text extraction. Every number was read from primary source text rather than search snippets. The four load-bearing arXiv IDs (STEB 2606.31741, Wang TACL 2308.11490, STAR 2310.11081, StyleDistance 2410.12757) were independently verified against arXiv metadata.
