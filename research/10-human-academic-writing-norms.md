# What Real, High-Quality Human Academic Writing Looks Like, Statistically

**The target distribution.** Report 04 catalogued how LLM text *differs* from human text. This report supplies the positive target: empirical distributions of stylometric features in genuine, good human writing, per genre and per discipline, with the spread around each mean — because a humanizer that hits a genre mean exactly is as detectable as one that misses it.

Compiled 2026-09-03. Two kinds of evidence are used and always distinguished:

- **[LIT]** — published corpus-linguistics numbers (Hyland, Biber, BAWE/MICUSP studies, COHA). Authoritative, but the literature reports metadiscourse and grammar well and reports *variance* almost not at all.
- **[OWN]** — measured for this report on four freely-obtainable, certainly-human corpora (details in §1.2). Every [OWN] number is reproducible from the pipeline described there. These fill the gaps the literature leaves: sentence-length SD/CV, punctuation rates per 1,000 words, paragraph-shape distributions, autocorrelation, and between-author variance.

---

## 1. Method

### 1.1 What the literature does and does not give you

Corpus linguistics has excellent published norms for *rates of discrete lexical/grammatical items* (hedges per 1,000 words, citations per 1,000 words, self-mention per 10,000 words, nominalizations per 1,000 words). It has almost nothing on the second-order statistics a humanizer actually needs — the standard deviation of sentence length inside a document, the coefficient of variation, the paragraph-length distribution, the punctuation profile per 1,000 words. Where a published number exists it is nearly always a corpus-level mean with no dispersion. That gap is why §1.2 exists.

### 1.2 Corpora measured for this report [OWN]

| Corpus | Genre label | n docs | Mean length | Provenance / certainty of human authorship |
|---|---|---|---|---|
| PERSUADE 2.0 (deduplicated to unique essays) | Student argumentative essay, grades 6–12, holistically scored 1–6 | 15,486 | 421 w | Crossley et al. 2022; collected 2010s, pre-LLM |
| PMC Open Access research articles, 2016–2020 | Research article (mostly biomedical/STEM) | 224 | 4,380 w | Europe PMC `fullTextXML`, body paragraphs only, tables/figures stripped; pre-ChatGPT |
| English Wikipedia **Featured Articles**, revision as of 2021-06-01 | Encyclopedic expository prose, community-vetted as best-quality | 104 | 4,020 w | MediaWiki API `rvstart=2021-06-01`; pre-ChatGPT by 18 months |
| r/AskHistorians answers ≥ 350 words, 2019–2021 | Long-form expert explanatory prose | 2,876 (816 authors) | 810 w | arctic-shift Reddit dump; heavily moderated subreddit; pre-ChatGPT |
| CNN/DailyMail articles ≥ 400 words | News journalism | 283 | 704 w | HF `abisee/cnn_dailymail`; 2007–2015 |

Pipeline: regex sentence segmentation with an abbreviation guard, `[A-Za-z][A-Za-z'-]*` tokenisation, MTLD at the standard 0.72 TTR factor threshold (bidirectional), suffix-rule nominalization count (`-tion -sion -ment -ness -ity -ance -ence -ism -ancy -ency -ship -hood`), a `be + V-ed/en` passive approximation, and a narrowed Hyland-style hedge list (60 items) and booster list (34 items). No spaCy, so clause counts and agentless-passive discrimination are approximations; treat passive and nominalization as *relative* not absolute. Paragraph metrics are unavailable for CNN/DailyMail because that dataset ships articles without paragraph breaks.

Caveat that matters: MTLD, TTR and "unique sentence-opener ratio" are length-sensitive. Never compare them across documents of very different length; compare within a length band.

---

## 2. THE TARGET-DISTRIBUTION TABLE

"SD" is the **between-document** standard deviation — the spread you should sample from — not a standard error. Where a source gives only a corpus mean, SD is marked "—".

### 2.1 Sentence-level shape (the most important block)

| Feature | Genre | Human mean | Human SD | Source |
|---|---|---|---|---|
| Words/sentence | Student argumentative essay (all scores) | 22.0 | 11.0 | [OWN] PERSUADE |
| Words/sentence | Student essay, top-scoring (5–6/6) | 22.3 | 7.0 | [OWN] PERSUADE |
| Words/sentence | Research article (PMC OA) | 23.3 | 4.1 | [OWN] |
| Words/sentence | Wikipedia Featured Article | 21.8 | 2.5 | [OWN] |
| Words/sentence | Long-form expert explanation (AskHistorians) | 24.7 | 6.4 | [OWN] |
| Words/sentence | News journalism | 18.3 | 3.0 | [OWN] CNN/DM |
| Words/sentence | Non-fiction, 2000s | 19.94 | — | Rudnicka 2018, COHA (19.8M sentences) |
| Words/sentence | Magazine, 2000s | 17.14 | — | Rudnicka 2018 |
| Words/sentence | Newspaper, 2000s | 16.70 | — | Rudnicka 2018 |
| Words/sentence | Fiction, 2000s | 12.07 | — | Rudnicka 2018 |
| Words/sentence | Drama/script (spoken proxy), 1990s | 8.62 | — | Rudnicka 2018 |
| **Within-doc SD of sentence length** | Student essay (all) | **11.2** | 9.3 | [OWN] |
| **Within-doc SD of sentence length** | Student essay, top-scoring | **9.7** | 5.3 | [OWN] |
| **Within-doc SD of sentence length** | Research article | **12.4** | 3.4 | [OWN] |
| **Within-doc SD of sentence length** | Wikipedia FA | **10.4** | 2.0 | [OWN] |
| **Within-doc SD of sentence length** | AskHistorians | **14.4** | 5.5 | [OWN] |
| **Within-doc SD of sentence length** | News | **9.9** | 1.7 | [OWN] |
| **Sentence-length CV (SD/mean)** | Student essay, top-scoring | **0.43** | 0.11 | [OWN] |
| **Sentence-length CV** | Research article | **0.54** | 0.12 | [OWN] |
| **Sentence-length CV** | Wikipedia FA | **0.48** | 0.07 | [OWN] |
| **Sentence-length CV** | AskHistorians | **0.59** | 0.17 | [OWN] |
| **Sentence-length CV** | News | **0.55** | 0.08 | [OWN] |
| Mean abs. adjacent-sentence diff ÷ mean length | Research article / Wikipedia / News / AskHistorians | 0.53 / 0.50 / 0.59 / 0.61 | 0.13 / 0.06 / 0.11 / 0.14 | [OWN] |
| Lag-1 autocorrelation of sentence length | all four genres | **0.01–0.10** | 0.09–0.19 | [OWN] |
| % sentences < 10 words | Research article / Wikipedia / News / AskHistorians / top essays | 11.6 / 8.8 / 21.8 / 14.3 / 8.0 | 8.5 / 4.4 / 8.6 / 10.4 / 8.2 | [OWN] |
| % sentences > 30 words | Research article / Wikipedia / News / AskHistorians / top essays | 23.5 / 18.1 / 13.1 / 28.3 / 18.0 | 10.3 / 8.0 / 8.4 / 16.1 / 14.7 | [OWN] |
| % sentences > 40 words | Research article / Wikipedia / News / AskHistorians | 9.1 / 5.4 / 2.8 / 13.5 | 7.2 / 4.4 / 3.5 / 12.2 | [OWN] |
| 10th percentile of sentence length | Research article / Wikipedia / AskHistorians | 9.3 / 10.3 / 9.5 | 3.8 / 2.1 / 4.9 | [OWN] |
| 90th percentile of sentence length | Research article / Wikipedia / AskHistorians | 39.0 / 34.8 / 41.8 | 7.6 / 4.8 / 10.8 | [OWN] |

### 2.2 Paragraph and document shape

| Feature | Genre | Human mean | Human SD | Source |
|---|---|---|---|---|
| Words/paragraph | Research article | 107.9 | 35.2 | [OWN] |
| Words/paragraph | Wikipedia FA | 115.2 | 23.4 | [OWN] |
| Words/paragraph | AskHistorians | 89.2 | 45.3 | [OWN] |
| Words/paragraph | Student essay, top-scoring | 125.8 | 40.8 | [OWN] |
| **Within-doc CV of paragraph length** | Research article / Wikipedia / AskHistorians / top essays | **0.71 / 0.51 / 0.69 / 0.42** | 0.24 / 0.12 / 0.25 / 0.20 | [OWN] |
| Sentences/paragraph | Research article / Wikipedia / AskHistorians / top essays | 4.7 / 5.3 / 3.7 / 5.8 | 1.4 / 1.1 / 1.7 / 2.0 | [OWN] |
| Paragraphs/document | Student essay (all) | 5.5 | 3.7 | [OWN] |
| Paragraphs/document | Student essay, top-scoring | 6.2 | 4.2 | [OWN] |
| Document length | MICUSP upper-level student paper | 3,157 w (829 papers / 2,616,529 w) | — | Wang 2022 |
| Document length | BAWE assignment | 2,357 w (2,761 assignments / 6,506,995 w) | — | Nesi & Gardner / Coventry |
| **% paragraphs opening with a formal connective** (*However, Moreover, Furthermore, Therefore, Additionally*…) | Research article / Wikipedia FA / AskHistorians | **3.4 / 0.3 / 3.3** | — | [OWN] |
| % paragraphs opening with *The* | Research article / Wikipedia / AskHistorians | 17.5 / 22.0 / 11.8 | — | [OWN] |
| % paragraphs opening with a demonstrative/pronoun | Research article / Wikipedia / AskHistorians | 8.8 / 2.0 / 13.8 | — | [OWN] |
| % paragraphs opening with a coordinator (*But/And/So/Or/Yet*) | Research article / Wikipedia / AskHistorians | 0.3 / 0.3 / 5.7 | — | [OWN] |
| % *sentences* opening with a formal connective | Research article / Wikipedia / News / AskHistorians / top essays | 7.0 / 1.8 / 0.5 / 3.0 / 3.2 | 4.5 / 1.6 / 1.3 / 4.1 / 4.5 | [OWN] |
| % *sentences* opening with a coordinator | Research article / Wikipedia / News / AskHistorians / top essays | 0.4 / 0.4 / 4.2 / 6.5 / 2.4 | 1.1 / 0.8 / 4.8 / 6.4 / 4.4 | [OWN] |

### 2.3 Lexical

| Feature | Genre | Human mean | Human SD | Source |
|---|---|---|---|---|
| MTLD | Research article (~4.4k w) | 75.5 | 17.8 | [OWN] |
| MTLD | Wikipedia FA (~4.0k w) | 80.8 | 16.8 | [OWN] |
| MTLD | AskHistorians (~810 w) | 97.8 | 20.7 | [OWN] |
| MTLD | News (~700 w) | 105.0 | 23.0 | [OWN] |
| MTLD | Student essay, top-scoring (~695 w) | 74.3 | 15.9 | [OWN] |
| MTLD | Native-speaker student essays | 95.72 | — | Herbold et al. 2023 |
| TTR (raw, length-confounded) | Research article / Wikipedia / AskHistorians / News / essays | 0.29 / 0.29 / 0.46 / 0.49 / 0.44 | 0.10 / 0.06 / 0.05 / 0.04 / 0.08 | [OWN] |
| Mean word length (chars) | Research article / Wikipedia / AskHistorians / News / top essays | 5.42 / 4.96 / 4.87 / 4.85 / 4.65 | 0.27 / 0.18 / 0.24 / 0.26 / 0.29 | [OWN] |
| % words > 6 chars | Research article / Wikipedia / AskHistorians / News / top essays | 34.6 / 27.0 / 25.6 / 25.2 / 21.8 | 3.8 / 3.3 / 4.1 / 4.8 / 4.9 | [OWN] |
| Nouns per 1,000 words | Specialist science RA (2005) | 423.0 | — | Biber & Gray 2013, Table 5 |
| Nouns per 1,000 words | History/humanities RA (2005) | 314.2 | — | Biber & Gray 2013 |

### 2.4 Punctuation (per 1,000 words)

| Feature | Genre | Human mean | Human SD | Source |
|---|---|---|---|---|
| Comma | Research article / Wikipedia / News / AskHistorians | 65.4 / 62.2 / 59.1 / 57.4 | 15.5 / 10.2 / 14.4 / 17.2 | [OWN] |
| Comma | Student essay, top-scoring | 40.7 | 16.7 | [OWN] |
| Semicolon | Research article / Wikipedia / News / AskHistorians / top essays | 5.02 / 7.67 / 0.76 / 1.40 / 0.84 | 4.9 / 8.2 / 2.1 / 2.2 / 1.8 | [OWN] |
| Colon | Research article / Wikipedia / News / AskHistorians / top essays | 2.60 / 4.36 / 2.31 / 3.72 / 0.43 | 2.4 / 10.3 / 5.4 / 4.5 / 1.2 | [OWN] |
| Parentheses (open-paren count) | Research article / Wikipedia / News / AskHistorians / top essays | 26.2 / 5.2 / 2.4 / 7.6 / 0.71 | 11.8 / 5.1 / 4.2 / 6.9 / 2.3 | [OWN] |
| Em dash | Research article / Wikipedia / AskHistorians / News | 0.38 / 0.69 / 0.23 / 0.00 | 1.2 / 1.0 / 1.1 / 0.0 | [OWN] |
| Em dash | Human-written essays | 3.23 (range 0.33–17.12) | — | Freeburg 2026 (Report 04 §2.9) |
| Question mark | Research article / Wikipedia / News / AskHistorians | 0.20 / 0.10 / 0.82 / 1.77 | 0.7 / 0.3 / 2.0 / 3.9 | [OWN] |
| Exclamation mark | Research article / Wikipedia / News / AskHistorians | 0.12 / 0.10 / 0.20 / 0.62 | 1.5 / 0.3 / 0.7 / 1.4 | [OWN] |
| Contractions | Research article / Wikipedia / News / AskHistorians / top essays | 1.57 / 9.06 / 16.8 / 11.9 / 8.58 | 2.6 / 6.5 / 7.9 / 7.4 / 8.3 | [OWN] |
| Contractions | Published research articles, 2015 | 0.13–1.35 (Bio 0.12, EE 0.01, Soc 0.51, App.Ling 1.35) | — | Hyland & Jiang 2016 (÷10 from per-10k) |
| Numerals | Research article / Wikipedia / News / AskHistorians | 45.5 / 31.3 / 15.6 / 12.7 | 23.5 / 27.0 / 12.9 / 12.8 | [OWN] |

The universal shape behind these numbers: **distances between consecutive punctuation marks, measured in words, follow a discrete Weibull distribution** across 240 literary works in seven languages, with English the least constrained (longest uninterrupted word runs) — Stanisz, Drożdż & Kwapień 2023. Sentence lengths, i.e. distances between *sentence-final* marks, are explicitly *not* Weibull-constrained and are the freer, more idiosyncratic layer (Stanisz et al. 2024). Practical reading: intra-sentential comma placement is close to a language universal you should not fiddle with; sentence length is where individual and genre style actually lives.

### 2.5 Grammar and stance (per 1,000 words unless noted)

| Feature | Genre / discipline | Human mean | Human SD | Source |
|---|---|---|---|---|
| Passives (approx.) | Research article / Wikipedia / News / AskHistorians / top essays | 17.9 / 13.8 / 9.2 / 8.5 / 6.7 | 6.8 / 4.7 / 5.5 / 4.4 / 4.8 | [OWN] |
| Passives | Academic prose (LGSWE) | 18.5 (≈25% of finite verbs) | — | Biber et al. 1999 |
| Passives | News / conversation (LGSWE) | 12.0 / ~2% of verbs | — | Biber et al. 1999 |
| Nominalizations | Research article / Wikipedia / News / AskHistorians / top essays | 59.5 / 27.2 / 27.3 / 32.6 / 26.0 | 14.8 / 10.2 / 13.3 / 14.5 / 13.8 | [OWN] |
| Nominalizations | Specialist science RA / social science RA / history RA (2005) | 61.0 / 70.5 / 62.8 | — | Biber & Gray 2013 |
| Noun + noun premodification | Specialist science / social science / history RA (2005) | 76.6 / 66.3 / 24.3 | — | Biber & Gray 2013 |
| Noun + *of*-phrase | Specialist science / social science / history RA (2005) | 30.2 / 29.8 / 36.6 | — | Biber & Gray 2013 |
| Relative clauses | Specialist science / social science / history RA (2005) | 4.5 / 6.1 / 9.3 | — | Biber & Gray 2013 |
| PP as noun postmodifier | Academic prose | 68 | — | Biber et al. 1999 |
| Hedges | Research articles, 8 disciplines (overall) | 14.5 | — | Hyland 2005 |
| Hedges | MICUSP upper-level student papers (overall) | 12.71 | — | Wang 2022 |
| Hedges | Medical research article / newspaper opinion column | 13.5 / 20.3 | — | Shen & Tao 2021 |
| Hedges (our narrower list) | Research article / Wikipedia / News / AskHistorians / top essays | 9.4 / 9.3 / 10.4 / 19.1 / 21.3 | 4.8 / 5.1 / 5.4 / 7.7 / 11.2 | [OWN] |
| Boosters | Research articles, 8 disciplines | 5.8 | — | Hyland 2005 |
| Attitude markers | Research articles, 8 disciplines | 6.4 | — | Hyland 2005 |
| Self-mention | Research articles, 8 disciplines | 4.2 | — | Hyland 2005 |
| Engagement markers | Research articles, 8 disciplines | 5.9 | — | Hyland 2005 |
| Citations | Research articles, 8 disciplines | 7.3–15.5 (see §4) | — | Hyland 1999 |
| Citations (parenthetical + bracketed regex) | PMC research article | 7.25 | 8.6 | [OWN] |
| First-person singular | Research article / Wikipedia / News / AskHistorians / top essays | 1.05 / 1.47 / 7.57 / 7.10 / 7.78 | 3.9 / 2.6 / 11.0 / 8.3 / 13.7 | [OWN] |
| First-person plural | Research article / Wikipedia / News / AskHistorians / top essays | 5.51 / 0.89 / 5.65 / 3.89 / 4.76 | 4.7 / 2.0 / 5.9 / 5.9 / 9.2 | [OWN] |
| Second person | Research article / Wikipedia / News / AskHistorians / top essays | 0.44 / 0.37 / 3.05 / 5.30 / 13.2 | 1.8 / 0.7 / 5.8 / 7.0 / 21.7 | [OWN] |
| Formal connectives (lexical count) | Research article / Wikipedia / News / AskHistorians / top essays | 4.64 / 1.90 / 0.67 / 2.70 / 2.50 | 2.5 / 1.2 / 1.2 / 2.4 / 2.8 | [OWN] |
| Sentence-initial conjunctions/conjunctive adverbs ("illicit initials") | Published RAs 2015, avg of 4 disciplines | 4.04 (App.Ling 3.79, Soc 4.90, Bio 4.01, EE 3.47) | — | Hyland & Jiang 2016 (÷10) |
| Sentence-initial *however* | Published RAs 2015 | 0.81 (Bio 0.94 highest, EE 0.48 lowest) | — | Hyland & Jiang 2016 |
| Sentence-initial *but / and / so* | Published RAs 2015 | 0.19 / 0.10 / 0.09 | — | Hyland & Jiang 2016 |

---

## 3. Variance and burstiness: the exact human target, and how not to overshoot

Report 04's headline finding was that LLM output is too uniform. The question this report has to answer is *how* varied human writing actually is, so a humanizer can widen the distribution to the right width and stop.

### 3.1 The number to hit is the coefficient of variation, and it is genre-specific

Sentence-length SD scales with sentence-length mean, so SD alone is not portable. The stable quantity is CV = SD/mean, and it sits in a narrow band across very different human genres:

| Genre | CV mean | CV SD | CV p10 | CV p25 | CV p50 | CV p75 | CV p90 |
|---|---|---|---|---|---|---|---|
| Student essay, top-scoring (5–6/6) | 0.43 | 0.11 | 0.32 | 0.36 | 0.42 | 0.49 | 0.58 |
| Student essay, all scores | 0.50 | 0.16 | 0.33 | 0.39 | 0.47 | 0.57 | 0.69 |
| Wikipedia Featured Article | 0.48 | 0.07 | 0.40 | 0.42 | 0.47 | 0.51 | 0.57 |
| PMC research article | 0.54 | 0.12 | 0.43 | 0.46 | 0.51 | 0.58 | 0.67 |
| News journalism | 0.55 | 0.08 | 0.45 | 0.49 | 0.54 | 0.59 | 0.65 |
| AskHistorians long answer | 0.59 | 0.17 | 0.42 | 0.48 | 0.56 | 0.67 | 0.79 |

[OWN]. Read this as the operational target: **human prose in any expository genre has a sentence-length CV of roughly 0.42–0.60 at the median, with a genre-dependent between-document SD of 0.07 (tightly-edited encyclopedic prose) to 0.17 (unedited expert forum prose).** Report 04 quotes Savoy's human political speech at 21 ± 16.4 words, i.e. CV ≈ 0.78, and GPT at 21.7 ± 10.9, CV ≈ 0.50 — note that the *machine's* CV of 0.50 is squarely inside the human expository band. That is the trap: burstiness alone does not separate models from humans in every genre; it separates them in the genres where humans are bursty. Pushing a research-article rewrite to CV 0.8 would put it two SDs above the PMC median and make it *more* anomalous, not less.

Corresponding absolute SDs, if you prefer to target those directly: Wikipedia FA 10.4 ± 2.0 words, research article 12.4 ± 3.4, news 9.9 ± 1.7, top-scoring student essay 9.7 ± 5.3, AskHistorians 14.4 ± 5.5.

### 3.2 Humans do not alternate long/short at lag 1

Measured lag-1 autocorrelation of the sentence-length series is **0.01–0.10** across all four corpora (SDs 0.09–0.19; medians −0.00 to 0.10; the 10th–90th percentile range is about −0.25 to +0.28). There is essentially no short-range alternation. Any humanizer rule of the form "follow a long sentence with a short one" produces a strongly negative lag-1 autocorrelation and is itself a detectable artefact.

What humans *do* have is long-range structure. Drożdż, Oświęcimka, Kwapień et al. computed the power spectrum of the sentence-length series for 113 canonical literary works and found 1/f^β scaling with **β ≈ 0.5 on average, dispersed between β ≈ 0.25 and β ≈ 0.75**, and multifractal (not merely monofractal) organisation in the stream-of-consciousness texts. In plain terms, human sentence-length series are pink-noise-like: correlated across many scales, with occasional clustered bursts of long sentences, and the correlations survive well beyond adjacent sentences. White-noise jitter around a mean reproduces neither the spectral slope nor the clustering.

The practical generator is therefore: sample a document-level (mean, CV) pair from the genre distribution, then generate the per-sentence length series as a **correlated** process (e.g. fractional Gaussian noise with H ≈ 0.75, equivalently β ≈ 0.5, or a two-state "expository/elaborating" Markov switch) rather than i.i.d. draws.

### 3.3 Paragraph-length variance

Paragraph length is *more* variable than sentence length in every genre measured. Within-document CV of words-per-paragraph: Wikipedia FA 0.51 ± 0.12, top student essays 0.42 ± 0.20, AskHistorians 0.69 ± 0.25, research article 0.71 ± 0.24. The 10th–90th percentile of paragraph CV in research articles is 0.44–1.00. A document whose paragraphs are all 4–5 sentences with CV < 0.25 is outside the human range for every genre except a highly-drilled five-paragraph school essay.

Sentences per paragraph cluster in a narrow band — 3.7 (AskHistorians) to 5.8 (top student essays) — but with meaningful spread (SD 1.1–2.0), and the tails matter: single-sentence paragraphs are common in human expository writing and near-absent in LLM output.

### 3.4 Punctuation-interval structure

Because inter-punctuation distances are Weibull-distributed with language-level parameters (Stanisz et al. 2023), comma density is the *most* stable feature across human genres measured here: 57.4–65.4 per 1,000 words in research articles, Wikipedia, news and AskHistorians alike — a spread of under 15%, despite the genres differing by 3× in nominalization rate and 25× in first-person rate. The student-essay figure (40.7 in top essays, 32.2 overall) is genuinely lower, and comma density is one of the strongest single correlates of essay quality (§6). Semicolons and parentheses, by contrast, vary by an order of magnitude across genres and by author (§5) — they are style, not grammar.

---

## 4. Disciplinary differences

A humanizer aimed at a history essay must hit different numbers than one aimed at a lab report. The disciplinary spread is large enough that a single "academic" target is wrong for most of the target space.

### 4.1 Stance and engagement (Hyland 2005, 240 research articles, 1.4M words, per 1,000 words)

| Feature | Phil | Soc | App.Ling | Marketing | Physics | Biology | Mech.Eng | Elec.Eng | Overall |
|---|---|---|---|---|---|---|---|---|---|
| Hedges | 18.5 | 14.7 | 18.0 | 20.0 | 9.6 | 13.6 | 8.2 | 9.6 | **14.5** |
| Boosters | 9.7 | 5.1 | 6.2 | 7.1 | 6.0 | 3.9 | 5.0 | 3.2 | **5.8** |
| Attitude markers | 8.9 | 7.0 | 8.6 | 6.9 | 3.9 | 2.9 | 5.6 | 5.5 | **6.4** |
| Self-mention | 5.7 | 4.3 | 4.4 | 5.5 | 5.5 | 3.4 | 1.0 | 3.3 | **4.2** |
| Stance total | 42.8 | 31.1 | 37.2 | 39.5 | 25.0 | 23.8 | 19.8 | 21.6 | **30.9** |
| Reader reference | 11.0 | 2.3 | 1.9 | 1.1 | 2.1 | 0.1 | 0.5 | 1.0 | **2.9** |
| Directives | 2.6 | 1.6 | 2.0 | 1.3 | 2.1 | 1.3 | 2.0 | 2.9 | **1.9** |
| Questions | 1.4 | 0.7 | 0.5 | 0.3 | 0.1 | 0.1 | 0.1 | 0.0 | **0.5** |
| Engagement total | 16.3 | 5.1 | 5.0 | 3.2 | 4.9 | 1.6 | 2.8 | 4.3 | **5.9** |
| **Grand total** | **59.1** | 36.2 | 42.2 | 42.7 | 29.9 | 25.4 | **22.6** | 25.9 | **36.8** |

Philosophy uses 2.6× the interactional metadiscourse of mechanical engineering. Soft-field writing hedges roughly twice as much as hard-field writing and addresses the reader roughly 10× as often.

### 4.2 Hedging in *student* writing by discipline (MICUSP, per 10,000 words)

Wang (2022) analysed all 829 MICUSP papers (2,616,529 words, 16 disciplines, A-graded senior undergraduate and graduate work) and found **127.12 hedges per 10,000 words overall** — i.e. 12.71 per 1,000, close to Hyland's 14.5 for published research articles. Disciplinary range: Philosophy 201.13, Psychology 163.39, Biology 146.88, Natural Resources & Environment 122.83, Nursing 118.78, Physics 98.09, History & Classical Studies 95.29 (per 10,000). Ten items — *would, can, may, could, seem, might, possible, suggest, likely, often* — account for ≥74% of all hedging, and the five modals *would/can/may/could/might* for ≥46% in every discipline. Concrete target: **hedging in student academic writing is overwhelmingly modal, not lexical-verb-based.** LLM hedging tends to be phrasal ("it is important to note that", "it could be argued that"), which is the wrong register even when the *rate* is right.

### 4.3 Citation density by discipline (Hyland 1999, 80 research articles; per 1,000 words)

Biology 15.5, Sociology 12.5, Philosophy 10.8, Applied Linguistics 10.8, Marketing 10.1, Electronic Eng 8.4, Physics 7.4, Mechanical Eng 7.3. Average citations per paper ranges from 24.8 (Physics) to 104.0 (Sociology). Integral-vs-non-integral split is equally diagnostic: Biology 9.8% integral, Physics 16.9%, Mechanical Eng 28.7%, Applied Linguistics 34.4%, Sociology 35.4%, **Philosophy 64.6%** — philosophy is the only discipline that prefers author-prominent citation. Direct quotation within citations: 0% in Biology and Electronic Engineering, 10% Applied Linguistics, 13% Sociology. Our own regex count on PMC OA biomedical articles gives 7.25 per 1,000 words (SD 8.6) — lower than Hyland's biology figure because numeric-bracket styles collapse multiple references into one bracket.

### 4.4 Self-mention by discipline (Hyland 2001, per 10,000 words)

Total first-person: Physics 64.6, Marketing 61.3, Biology 56.2, Philosophy 52.7, Applied Linguistics 51.8, Sociology 47.1, Electronic Eng 44.4, Mechanical Eng 17.8; overall 50.5. The *form* differs sharply: Philosophy is 35.6 *I* and 1.4 *we*; Physics is 0.0 *I* and 39.3 *we*. Getting the rate right while getting the pronoun wrong is a discipline-level error a reader spots instantly.

Students use far less: Hyland (2002), 64 Hong Kong final-year undergraduate reports (630,100 words) vs 240 research articles (1,323,000 words), gives **10.1 per 10,000 in student reports vs 41.2 in research articles** — experts use roughly 4× more first person. Function also inverts: research articles use *I/we* mostly to explain a procedure (38%) and state results (26%); student reports use it mostly to state a goal (36%).

### 4.5 Phrasal, not clausal, complexity (Biber & Gray 2013, per 1,000 words, 2005 sub-corpora)

| | Specialist science | Specialist social science | Multidisciplinary science | History/humanities |
|---|---|---|---|---|
| Nouns | 423.0 | 372.8 | 340.9 | 314.2 |
| Nominalizations | 61.0 | 70.5 | 72.1 | 62.8 |
| Noun + noun premodifier | 76.6 | 66.3 | 56.7 | 24.3 |
| Noun + *of*-phrase | 30.2 | 29.8 | 35.5 | 36.6 |
| Relative clauses | 4.5 | 6.1 | 6.6 | 9.3 |

The direction that matters for a humanizer: as writing becomes more scientific it gets **more phrasally dense and less clausally complex** — more noun-noun compounds, fewer relative clauses. History writing has 2× the relative clauses and one-third the noun-noun compounding of specialist science. Our own measurements corroborate the register split: nominalizations 59.5/1,000 in PMC research articles versus 27.2 in Wikipedia and 26.0 in top student essays.

### 4.6 Informality is discipline-specific and has been drifting (Hyland & Jiang 2016, 360 papers, 2.2M words, per 10,000 words)

Total informality features in 2015: Applied Linguistics 191.7, Sociology 198.8, Electrical Engineering 155.2, Biology 140.2 (average 171.5, up only 2% since 1965). Contractions in 2015: Applied Linguistics 13.5, Sociology 5.1, Biology 1.2, Electrical Engineering 0.1 — a 100× disciplinary spread. Sentence-initial conjunctions and conjunctive adverbs in 2015: Sociology 49.0, Biology 40.1, Applied Linguistics 37.9, Electrical Engineering 34.7 per 10,000 words, of which initial *however* is 4.8–9.4 and initial *and/but/so* together only 1.4–6.0. Published academic prose therefore *does* begin sentences with connectives roughly 4 times per 1,000 words, but almost always with *however/thus/indeed*, not *and/but/so*.

---

## 5. Individual-author variation: how tightly should we target the mean?

This determines the whole architecture. If authors within a genre were near-identical, we could target the genre mean; if they are highly individual, we must sample a persona and hold it fixed across a document.

### 5.1 A direct measurement [OWN]

Using 1,833 AskHistorians answers from 130 authors with ≥5 answers each — one genre, one register, one community style guide, topic varying freely within author — the one-way random-effects intraclass correlation (proportion of total variance attributable to author identity) is:

| Feature | Overall mean | Overall SD | Within-author SD | **ICC (author share of variance)** |
|---|---|---|---|---|
| Question marks /1k | 1.79 | 4.19 | 2.57 | **0.63** |
| Em dashes /1k | 0.32 | 1.38 | 0.96 | **0.52** |
| Parentheses /1k | 8.55 | 7.46 | 5.34 | **0.49** |
| Commas /1k | 59.1 | 17.4 | 13.0 | **0.44** |
| Colons /1k | 5.02 | 6.16 | 4.68 | **0.43** |
| Mean sentence length | 25.5 | 6.58 | 5.00 | **0.43** |
| SD of sentence length | 15.1 | 5.34 | 4.18 | 0.39 |
| Mean paragraph length (words) | 89.0 | 44.2 | 35.6 | 0.35 |
| Contractions /1k | 11.9 | 8.57 | 6.91 | 0.35 |
| Mean word length | 4.87 | 0.243 | 0.199 | 0.33 |
| % sentences starting with a coordinator | 6.74 | 7.02 | 5.74 | 0.33 |
| % words > 6 chars | 25.5 | 3.96 | 3.25 | 0.33 |
| Nominalizations /1k | 31.8 | 13.4 | 11.3 | 0.29 |
| Sentence-length CV | 0.601 | 0.160 | 0.136 | **0.28** |
| Paragraph-length CV | 0.713 | 0.292 | 0.249 | 0.28 |
| Boosters /1k | 5.99 | 3.91 | 3.48 | 0.21 |
| MTLD | 95.4 | 19.8 | 17.8 | 0.20 |
| Hedges /1k | 19.1 | 8.72 | 7.83 | 0.20 |
| Passives /1k | 8.88 | 5.14 | 4.70 | 0.16 |
| Semicolons /1k | 1.85 | 5.17 | 4.98 | 0.07 |

Three consequences:

1. **No feature exceeds ICC ≈ 0.63, and most sit at 0.2–0.45.** Within a single genre, the majority of stylometric variance is *within*-author, document to document. A humanizer should therefore never aim for a point estimate; it should sample a document-level target from the genre distribution.
2. **Punctuation habits are the most author-stable features** (question marks, em dashes, parentheses, colons, commas: ICC 0.43–0.63), followed by mean sentence length. These are the features worth freezing per-persona across a document.
3. **Hedging, passives, lexical diversity and semicolon use are almost entirely document-level noise** (ICC 0.07–0.20). Trying to reproduce a specific author's hedge rate is targeting noise; trying to reproduce their comma and parenthesis habits is targeting signal.

The right generative model is a two-level one: draw a persona offset per author/document set (weighted by ICC), then draw a document-level target around it with the residual within-author SD. Concretely, for a feature with total SD σ and ICC ρ, the persona offset has SD σ√ρ and the per-document residual has SD σ√(1−ρ).

### 5.2 What the attribution literature adds

Two robust results from authorship attribution bear on this. First, reliable attribution needs a **lot** of text: Eder's "Does size matter?" experiments on English prose put the minimum reliable sample in the low thousands of words, with attribution accuracy degrading sharply below roughly the 2,000–5,000-word range (verify the exact threshold against the paper before quoting it) — consistent with the ICCs above, since low ICC means many documents are needed to estimate an author's true mean. Second, style and topic are deeply confounded: content-controlled representations (Wegmann et al.'s conversation-based contrastive setup; the STEL benchmark) exist precisely because naive stylometric similarity mostly recovers topic. For a humanizer this is reassuring: matching the *distributional* profile of a genre is achievable, whereas matching an individual identity is not the goal and is not well-defined from a single document.

---

## 6. What high-scoring human writing actually does

We measured every feature against the PERSUADE holistic score (1–6), 15,486 essays, then re-checked inside a 450–750-word band to remove the length confound. The length-controlled means (score 1 → 6):

| Feature | 1 | 2 | 3 | 4 | 5 | 6 | Spearman ρ with score (all essays) |
|---|---|---|---|---|---|---|---|
| Mean sentence length | 30.1 | 30.3 | 24.2 | 21.9 | 21.5 | 22.4 | +0.07 |
| **SD of sentence length** | 13.6 | 18.4 | 12.7 | 10.5 | 9.4 | 9.1 | **−0.10** |
| **Sentence-length CV** | 0.53 | 0.55 | 0.51 | 0.47 | 0.44 | **0.41** | **−0.22** |
| % sentences < 10 words | 14.5 | 13.3 | 11.7 | 10.2 | 8.5 | 6.8 | −0.21 |
| Adjacent-sentence diff (normalised) | 0.55 | 0.58 | 0.54 | 0.50 | 0.47 | 0.44 | −0.20 |
| Words per paragraph | 99.8 | 97.4 | 97.8 | 100.5 | 110.4 | 121.6 | +0.50 |
| Sentences per paragraph | 4.66 | 4.08 | 4.55 | 4.85 | 5.31 | 5.56 | +0.46 |
| Commas /1k | 27.1 | 28.9 | 30.0 | 34.6 | 40.1 | 45.7 | +0.28 |
| Semicolons /1k | 0.74 | 0.46 | 0.45 | 0.55 | 0.86 | 0.97 | +0.19 |
| **Nominalizations /1k** | 20.3 | 19.6 | 17.7 | 19.6 | 25.4 | **31.2** | **+0.21** |
| **Passives /1k** | 5.99 | 5.66 | 4.59 | 5.35 | 6.71 | **7.53** | **+0.15** |
| Mean word length | 4.45 | 4.37 | 4.36 | 4.45 | 4.64 | 4.81 | +0.33 |
| MTLD | 68.4 | 71.5 | 69.0 | 68.9 | 73.2 | 80.4 | +0.15 |
| Formal connectives /1k | 0.62 | 0.34 | 0.84 | 1.38 | 2.35 | 3.75 | +0.38 |
| **% sentences starting with a coordinator** | 5.24 | 7.52 | 5.11 | 3.79 | 2.52 | **2.28** | **−0.12** |
| First-person plural /1k | 11.5 | 11.6 | 10.5 | 7.26 | 5.22 | 3.74 | −0.15 |
| Second person /1k | 9.95 | 20.3 | 20.0 | 22.8 | 13.9 | 8.28 | −0.10 |
| Hedges /1k | 15.1 | 16.0 | 19.6 | 21.4 | 21.6 | 20.7 | +0.09 |
| Contractions /1k | 9.43 | 9.62 | 10.0 | 10.2 | 8.76 | 8.23 | +0.01 |

Findings that contradict common "humanizing" folklore:

- **Better essays are LESS bursty, not more.** CV falls monotonically from 0.53 to 0.41 and the effect survives length control. Weak essays achieve high variance through run-ons and fragments, not through rhetorical control. If a humanizer's only move is "add variance", it makes text look like a score-2 essay, not a score-6 one.
- **Better essays use MORE nominalization and MORE passive voice** (+54% and +64% from score 3 to score 6). Report 04 correctly notes that LLMs over-nominalize relative to *average* human writing; but nominalization is positively, not negatively, correlated with graded quality in student writing. Removing nominalizations to "humanize" moves text toward the low-scoring end.
- **Better essays use far more formal connectives** (0.84 → 3.75 per 1,000, ρ = +0.38) and far fewer sentence-initial *And/But/So* (5.1% → 2.3%). Inserting informal openers to look human degrades apparent quality. The right target is Hyland's published-article profile: ~4 sentence-initial connectives per 1,000 words, dominated by *however/thus/indeed*, with *and/but/so* under 0.2 per 1,000.
- **Hedging is flat above score 3** (19.6 → 20.7). Hedge *rate* is not a quality signal once a floor is cleared; hedge *placement* presumably is, but that is not measurable with counts.
- **Paragraph size, not sentence complexity, is the strongest structural correlate of quality** (ρ = +0.50 for words/paragraph, +0.46 for sentences/paragraph, vs +0.07 for mean sentence length). Longer, better-developed paragraphs, not longer sentences.
- **Comma density is the strongest punctuation correlate** (ρ = +0.28), rising from 27 to 46 per 1,000 words. Higher-scoring writing converges toward the ~57–65 per 1,000 seen in professional prose (§2.4).

This maps onto the wider literature's conclusion that top-graded writing is not the most *clausally* complex writing: Biber & Gray's phrasal-complexity result (§4.5) and these data agree that maturity in academic writing shows up as denser noun phrases, longer developed paragraphs and controlled sentence rhythm, not as longer or more embedded sentences.

---

## 7. Where to get the data ourselves

Status codes: **[V]** = verified working during this research session; **[D]** = documented but not exercised here.

### 7.1 Named academic-writing corpora

| Corpus | Contents | Size | Access | Licence | Bulk raw text? |
|---|---|---|---|---|---|
| **MICUSP** (Michigan Corpus of Upper-level Student Papers) | A-graded senior UG + graduate papers, 16 disciplines, 7 paper types, 4 levels | 829 papers / 2,616,529 words | `https://micusp.elicorpora.info/` (MICUSP Simple, no login) **[V]** | © Regents of the University of Michigan. "Freely available for study, research and teaching"; commercial use requires a licence and possibly a fee (`micusp-help@umich.edu`) | **No** — browse/search interface only, one paper at a time |
| **BAWE** (British Academic Written English) | "Good-standard" UG + taught-masters assignments, 30+ disciplines, 4 levels, 13 genre families | 2,761 assignments / 6,506,995 words | Oxford Text Archive, `https://ota.bodleian.ox.ac.uk/` (handle 20.500.12024/2539); also queryable via Sketch Engine, Lextutor, CorpusMate **[D]** | Free to researchers who accept the conditions of use | **Yes**, after accepting terms |
| **PERSUADE 2.0** | US grades 6–12 argumentative essays, holistic score 1–6, discourse-element annotation, demographics | 25,996 essays (15,594 unique in the copy used here) | HuggingFace `ruudra1/PERSUADE` → `persuade_corpus_2.0_train.csv` (617 MB), no auth **[V]**; canonical release via Crossley et al. 2022 | Public research release; confirm terms for commercial use | **Yes** |
| **ELLIPSE** | ELL essays with proficiency ratings (Crossley et al.) | ~6,500 essays | Kaggle "Feedback Prize – English Language Learning" release **[D]** | Kaggle competition terms | Yes |
| **ICLE v3 / LOCNESS** | Learner argumentative essays (ICLE) and native-speaker student essays (LOCNESS), UCLouvain | ICLE v3 ~5.5M words; LOCNESS ~324k words | `https://uclouvain.be/en/research-institutes/ilc/cecl/icle.html` **[V] (page exists)** | Licensed/paid; institutional purchase | Yes, once licensed |
| **COCA academic subcorpus** | ~120M words of academic prose within COCA | ~1B words total | `https://www.english-corpora.org/coca/` — online interface free; full-text/word-frequency data sold separately | Paid for downloadable data | Online: no. Purchased data: yes |
| **Uppsala Student English (USE)** | Swedish university students' English essays | ~1.2M words | Uppsala University / Oxford Text Archive **[D]** | Free for research | Yes |
| **Hyland's RA corpora** | 240 research articles, 8 disciplines, 1.4M words | 1.4M words | Not distributed; only published statistics | — | No |

### 7.2 Free proxies (all verified working)

| Source | Genre it proxies | How to get it | Notes |
|---|---|---|---|
| **Europe PMC / PMC OA Subset** | STEM/biomedical research article | Search: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=OPEN_ACCESS:y AND PUB_YEAR:2019 AND PUB_TYPE:"research-article"&format=json&pageSize=100`; full text: `https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML` **[V]** | Cleanest option: JATS XML, so you can strip tables/figures/refs and keep only `<body>` `<p>` elements. Bulk mirrors at `ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/` (`oa_comm` = CC-BY-ish, `oa_noncomm` = NC) and on AWS S3 `s3://pmc-oa-opendata` |
| **English Wikipedia, pre-2022 revisions** | Encyclopedic expository prose | `https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=X&rvstart=2021-06-01T00:00:00Z&rvdir=older&rvlimit=1&rvprop=content&rvslots=main&format=json` **[V]**; article list from `list=categorymembers&cmtitle=Category:Featured articles` | Use **Featured Articles** specifically — community-vetted as best-quality prose. Rate-limit to ~1 req/1.5 s with a real UA or you get HTTP 429. Full dumps at `https://dumps.wikimedia.org/` (2021 snapshots still mirrored) |
| **arctic-shift Reddit archive** | Long-form expert explanatory prose | `https://arctic-shift.photon-reddit.com/api/comments/search?subreddit=AskHistorians&limit=100&sort=desc&before={epoch}` **[V]** | Pushshift's public API is gone; arctic-shift is the working replacement. Filter `author != AutoModerator`, length ≥ 350 words, and cut before 2022-11 to guarantee human authorship. Also on Academic Torrents |
| **HuggingFace datasets-server** | News, essays, misc. | `https://datasets-server.huggingface.co/rows?dataset={id}&config={cfg}&split=train&offset=0&length=100` **[V]** | No auth needed for public datasets. Verified: `abisee/cnn_dailymail` (config `3.0.0`), `ruudra1/PERSUADE`. Beware datasets that ship articles without paragraph breaks (CNN/DM does) — paragraph metrics are then meaningless |
| **arXiv bulk** | STEM preprint prose, by field | `https://info.arxiv.org/help/bulk_data/` **[V]**; requester-pays S3 `s3://arxiv/` for PDF/source tarballs; metadata via the Kaggle arXiv dataset or the arXiv API `http://export.arxiv.org/api/query` **[V]** | Filter to `v1` submissions before 2022-11 for a clean human baseline. LaTeX source needs de-macro-ing |
| **OpenAlex** | Discovery layer for OA full text across all disciplines | `https://api.openalex.org/works?search=...` **[V]** — returns `best_oa_location.pdf_url` | Free, no key, no rate-limit problems with a mailto. The practical way to assemble a *humanities and social science* reference set, which PMC does not cover |
| **CORE / OpenDOAR / OATD / DART-Europe** | Open-access theses and repository papers, all disciplines | `https://core.ac.uk/services/api` (free key) **[D]** | Best route to humanities dissertations, which are otherwise hard to obtain in bulk |
| **Project Gutenberg / SPGC** | Literary and pre-1930 non-fiction | `https://gutendex.com/books` (was intermittently down during this session), mirrors at `aleph.gutenberg.org`; Standardized Project Gutenberg Corpus on Zenodo **[D]** | Prose is dated; use for fiction/literary reference only, not for contemporary academic targets |
| **Semantic Scholar S2ORC / AI2 peS2o** | Large full-text scientific corpus | S2 API `https://api.semanticscholar.org/`; HF `allenai/peS2o` **[D]** | peS2o is already cleaned and deduplicated, and is snapshot-dated so you can take a pre-2023 cut |

### 7.3 Existing paired human/AI sets (the human halves are usable as reference)

HC3 (`Hello-SimpleAI/HC3`), Herbold et al. 2023 essay dataset, M4/M4GT-Bench, RAID (`liamdugan/raid`), the Kaggle DAIGT collections, and Ghostbuster's essay sets. All ship the human side; RAID and M4 are the most genre-diverse.

---

## 8. Sources

- Rudnicka, K. (2018). "Variation of sentence length across time and genre." *Diachronic Corpora, Genre, and Language Change* (SCL 85), 220–240. Pre-print: https://arxiv.org/pdf/2502.04321 — COHA, 19,768,290 sentences, mean sentence length by decade and genre.
- Drożdż, S., Oświęcimka, P., Kwapień, J. et al. "Quantifying origin and character of long-range correlations in narrative texts." *Information Sciences* (2016). https://arxiv.org/abs/1412.8319 — 1/f^β sentence-length spectra, β ≈ 0.25–0.75, multifractality.
- Stanisz, T., Drożdż, S., Kwapień, J. "Universal versus system-specific features of punctuation usage patterns in major Western languages." *Chaos, Solitons & Fractals* (2023). https://arxiv.org/abs/2212.11182 — discrete Weibull inter-punctuation distances, 240 works, 7 languages.
- Stanisz, T., Drożdż, S., Kwapień, J. "Statistics of punctuation in experimental literature — the remarkable case of *Finnegans Wake*." *Chaos* (2024). https://arxiv.org/abs/2409.00483 — sentence lengths are *not* Weibull-constrained.
- Hyland, K. (2005). "Stance and engagement: a model of interaction in academic discourse." *Discourse Studies* 7(2), 173–192. https://doi.org/10.1177/1461445605050365 — 240 RAs, 1.4M words, per-1,000-word stance/engagement table by discipline.
- Hyland, K. (1999). "Academic attribution: citation and the construction of disciplinary knowledge." *Applied Linguistics* 20(3). Tables reproduced in Thompson & Tribble (2001), *Language Learning & Technology* 5(3), 91–105. https://scholarspace.manoa.hawaii.edu/bitstreams/d086fabd-5df6-47be-a414-e86b985942a6/download
- Hyland, K. (2001). "Humble servants of the discipline? Self-mention in research articles." *English for Specific Purposes* 20, 207–226.
- Hyland, K. (2002). "Authority and invisibility: authorial identity in academic writing." *Journal of Pragmatics* 34, 1091–1112.
- Hyland, K. & Jiang, F. (2016). "Is academic writing becoming more informal?" *English for Specific Purposes* 45, 40–51. https://ueaeprints.uea.ac.uk/id/eprint/66139/1/Accepted_manuscript.pdf — 360 papers, 2.2M words, 1965/1985/2015, per-10,000-word informality tables.
- Hyland, K. & Jiang, F. (2018). "'In this paper we suggest': changing patterns of disciplinary metadiscourse." *ESP* 51. https://ueaeprints.uea.ac.uk/id/eprint/66279/1/Accepted_manuscript.pdf
- Wang, X. (2022). "Hedging in academic writing: cross-disciplinary comparisons in the Michigan Corpus of Upper-Level Student Papers." *JALT CALL SIG* / OSF preprint. https://doi.org/10.37546/JALTSIG.CALL.PCP2021-09 ; https://osf.io/5xj27/download — 829 MICUSP papers, 2,616,529 words, hedges per 10,000 words by discipline.
- Gardner, S., Nesi, H. & Biber, D. (2018). "Discipline, Level, Genre: integrating situational perspectives in a new MD analysis of university student writing." *Applied Linguistics* 40(4), 646–674. https://academic.oup.com/applij/article/40/4/646/4924026 — BAWE, 2,760 assignments, four dimensions and their feature loadings.
- Gardner, S. & Nesi, H. (2012). "A classification of genre families in university student writing." *Applied Linguistics* 34(1). https://pure.coventry.ac.uk/ws/files/3978905/gardnercomb.pdf
- Nesi, H. et al. BAWE corpus page, Coventry University. https://www.coventry.ac.uk/research/research-directories/current-projects/2015/british-academic-written-english-corpus-bawe/
- Biber, D. & Gray, B. (2013). "Being specific about historical change: the influence of sub-register." *Journal of English Linguistics* 41(2), 104–134. https://web.archive.org/web/20240412224228id_/https://jan.ucc.nau.edu/~biber/Biber/Biber_Gray_2013.pdf — per-1,000-word noun/nominalization/N+N/relative-clause rates by academic sub-register.
- Biber, D. & Gray, B. (2016). *Grammatical Complexity in Academic English*. Cambridge UP — phrasal not clausal complexity.
- Biber, D., Johansson, S., Leech, G., Conrad, S. & Finegan (1999). *Longman Grammar of Spoken and Written English* — passives 18,500 per million words in academic prose (~25% of finite verbs), PP postmodifiers ~68,000 pmw.
- Shen, Q. & Tao, Y. (2021). "Stance markers in English medical research articles and newspaper opinion columns." *PLOS ONE* 16(3): e0247981. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247981 — hedges/boosters/attitude/self-mention per 1,000 words in two genres.
- Crossley, S. et al. (2022). "The persuasive essays for rating, selecting, and understanding argumentative and discourse elements (PERSUADE) corpus 1.0." *Assessing Writing* 54, 100667. https://doi.org/10.1016/j.asw.2022.100667
- Römer, U. & Wulff, S. (2010). "Applying corpus methods to written academic texts: explorations of MICUSP." *Journal of Writing Research* 2(2), 99–127. https://doi.org/10.17239/jowr-2010.02.02.2
- Aull, L. (2019). "Linguistic markers of stance and genre in upper-level student writing." *Written Communication* 36(2). https://doi.org/10.1177/0741088318819472
- Herbold, S. et al. (2023). "A large-scale comparison of human-written versus ChatGPT-generated essays." *Scientific Reports*. https://www.nature.com/articles/s41598-023-45644-9 — human essay MTLD 95.72.
- Lu, X. (2010/2011). L2 Syntactic Complexity Analyzer (T-unit, clause and phrase indices). https://www.personal.psu.edu/xxl13/downloads/l2sca.html
- Kyle, K. & Crossley, S. (2018). "Measuring syntactic complexity in L2 writing using fine-grained clausal and phrasal indices." *Modern Language Journal* 102(2). https://doi.org/10.1111/modl.12468 (TAASSC)
- Shen, C., Guo, J., Shi, P., Qu, S. & Tian, J. (2023). "A corpus-based comparison of syntactic complexity in academic writing of L1 and L2 English students across years and disciplines." *PLOS ONE* 18(10): e0292688. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0292688 — BAWE, 472 texts, TAASSC fine-grained indices.
- Eder, M. (2015). "Does size matter? Authorship attribution, small samples, big problem." *Digital Scholarship in the Humanities* 30(2). https://doi.org/10.1093/llc/fqt066
- Wegmann, A., Schraagen, M. & Nguyen, D. (2022). "Same author or just same topic? Towards content-independent style representations." *ACL RepL4NLP*. https://aclanthology.org/2022.repl4nlp-1.26/
- Rivera-Soto, R. et al. (2021). "Learning universal authorship representations." *EMNLP*. https://aclanthology.org/2021.emnlp-main.70/
- MICUSP Simple interface and fair-use statement. https://micusp.elicorpora.info/
- Europe PMC RESTful web service. https://europepmc.org/RestfulWebService ; PMC Open Access Subset. https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
- MediaWiki Revisions API. https://www.mediawiki.org/wiki/API:Revisions
- arXiv bulk data. https://info.arxiv.org/help/bulk_data/
- OpenAlex API. https://docs.openalex.org/
- arctic-shift Reddit archive API. https://arctic-shift.photon-reddit.com/
- CORE API. https://core.ac.uk/services/api
- COCA. https://www.english-corpora.org/coca/ ; ICLE/LOCNESS, UCLouvain CECL. https://uclouvain.be/en/research-institutes/ilc/cecl/icle.html

---

## 9. Recommended reference corpora to build our per-genre human distributions, and how to sample from them at inference time

### 9.1 The five reference corpora to build

Build one reference set per output genre, each ≥ 300 documents, all sourced pre-November 2022, all stored as plain text with paragraph breaks preserved.

1. **`ref/student-essay`** — PERSUADE 2.0 filtered to holistic score ≥ 4, plus BAWE (once the OTA conditions of use are signed) for university-level work. PERSUADE alone skews to US school level; BAWE is the one that actually matches "university-level human writing", and getting it is the single highest-value acquisition step in this report.
2. **`ref/research-article-stem`** — 500+ PMC OA research articles, 2016–2021, body paragraphs only via `fullTextXML`. Already built and measured here.
3. **`ref/research-article-humss`** — the gap. PMC does not cover humanities. Assemble via OpenAlex (`best_oa_location.pdf_url`, filter to `concepts` = History/Philosophy/Sociology/Literature, years 2015–2021) plus CORE-sourced open-access theses. Budget a day for PDF-to-text cleanup.
4. **`ref/expository-encyclopedic`** — Wikipedia Featured Articles at their 2021-06-01 revision. Already built. This is the best available proxy for "clean, edited, impersonal expository prose".
5. **`ref/longform-explanatory`** — AskHistorians answers ≥ 350 words, 2019–2021, with author IDs retained so the two-level model in §5.1 can be re-estimated per genre. Already built (2,876 texts, 816 authors).

Store, for each corpus and each feature: mean, SD, and the 10/25/50/75/90 percentiles, plus the empirical covariance matrix. Features are correlated (comma density with sentence length, nominalization with word length), so sampling them independently produces documents that are individually plausible on every axis and jointly impossible.

### 9.2 Inference-time sampling recipe

**Step 1 — pick the genre and discipline.** Classify the input (or take it from the user) into one of the five reference corpora, plus a discipline tag from Hyland's eight if it is academic. Discipline sets hedge rate (§4.2: Philosophy 20.1/1,000, History 9.5/1,000), citation density (§4.3: Biology 15.5, Mech.Eng 7.3), self-mention form (*I* for philosophy, *we* for physics), and noun-noun compounding (§4.5).

**Step 2 — draw a document target vector, do not use the mean.** For each feature *f* with genre mean μ and between-document SD σ, draw a target *t_f* from the empirical distribution (preferably by resampling a real reference document's feature vector, which preserves the covariance for free) rather than from N(μ, σ) independently. Resampling a whole document's profile is the cheapest correct implementation.

**Step 3 — split the target into persona and document components.** For features with a measured ICC (§5.1), hold a persona offset fixed across everything one user writes: offset ~ N(0, σ√ρ), per-document residual ~ N(0, σ√(1−ρ)). Freeze the high-ICC punctuation habits (question marks, em dashes, parentheses, colons, commas, mean sentence length) per persona; let the low-ICC features (hedges, passives, MTLD, semicolons) float per document.

**Step 4 — realise sentence lengths as a correlated series, not i.i.d. draws.** Given a drawn (mean, CV), generate the length series with long-range correlation (fractional Gaussian noise, H ≈ 0.75 / spectral β ≈ 0.5, per §3.2), then quantise. Check the result has lag-1 autocorrelation in [−0.1, +0.25] and does *not* alternate. Enforce the tail shares (§2.1): about 9–14% of sentences under 10 words and 18–28% over 30 words for academic prose; do not simply widen the middle.

**Step 5 — realise paragraph lengths with the genre's paragraph CV** (0.42 top student essays, 0.51 Wikipedia, 0.69–0.71 research article / long-form), which is *higher* than the sentence CV in every genre. Allow one- and two-sentence paragraphs. Constrain paragraph openers: at most 3–4% may begin with a formal connective, and essentially none with a coordinator in academic or encyclopedic registers (§2.2).

**Step 6 — do not "informalise" academic output.** The quality gradient in §6 runs the opposite way to naive humanizing: more nominalization, more passive, more commas, more formal connectives, fewer sentence-initial *And/But/So*, lower sentence-length CV. For academic targets, the humanizer's job is to move *register* toward professional prose (comma density 57–65/1,000, contraction rate under 1.4/1,000 in research articles) while breaking the *uniformity* fingerprint through the correlated-variance machinery above, plus the vocabulary and participial-tail fixes from Report 04.

**Step 7 — score, don't clamp.** Report each generated document's Mahalanobis distance from the genre's reference feature distribution, and reject/regenerate outside roughly the 90th percentile. This automatically prevents the failure mode of overshooting into "unnaturally bursty": a CV of 0.85 in a research-article target is as far outside the human distribution as a CV of 0.30.

### 9.3 Known gaps to close next

- **BAWE is not yet acquired.** Every university-level student-writing number in §2 is either MICUSP-derived from the literature or PERSUADE-derived from school-level essays. Acquiring BAWE would let us compute the same SD/CV/percentile tables for genuine university assignments across 30 disciplines and four levels.
- **No humanities research-article reference set.** PMC is biomedical; Biber & Gray's history figures are the only humanities anchor here.
- **Clause-level syntax is unmeasured** in the [OWN] pipeline (no parser). Adding spaCy would give clauses per T-unit, dependent clauses per clause, agentless-passive discrimination and participial-tail counts on the same reference corpora, making the §2.5 block directly comparable to Report 04's AI-side numbers.
- **Em-dash rates in the [OWN] corpora (0.0–0.7 per 1,000) are far below Freeburg's 3.23 for human essays**, almost certainly because XML/wire-copy pipelines normalise dashes away. Do not use the [OWN] em-dash figures as a target; use Freeburg's.
