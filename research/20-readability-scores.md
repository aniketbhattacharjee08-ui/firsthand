# Readability Scores: What They Measure, Where Human and AI Text Land, and the Target Band

Date 2026-09-07. Report 20. Builds on 09 (§3-4), 10 (§2), 04 (§2.12) and 12 (item 16, "register whiplash"). About 70 sources.

**Bottom line.** Flesch Reading Ease (FRE) is a two-variable regression from 1948: average sentence length and average syllables per word. It measures surface load, not comprehensibility, and it cannot see coherence, word order, or meaning. On matched tasks, instruction-tuned LLM output is *harder* by FRE than human text almost everywhere it has been measured (typically 8 to 25 FRE points lower), and the difficulty is lexical, not syntactic: GPT-4 writes shorter T-units than students but at grade 16.6 versus 12.1. Human graders barely care: FRE correlates with essay score at r = −0.17 to +0.13 on ASAP, and the only indices with positive correlations are the ones that reward longer words. **Do not optimize FRE in either direction. Measure it, keep the output within about ±5 FRE and ±1 grade of the input, keep the document inside a wide per-genre band, and treat any larger shift as a humanizer artifact.**

---

## 1. The twelve findings

1. **FRE = 206.835 − 1.015 × (words/sentence) − 84.6 × (syllables/word).** One extra word per sentence costs 1.0 point; one extra tenth of a syllable per word costs 8.5 points. Moving from 1.5 to 1.7 syllables per word costs 17 points; moving from 18 to 28 words per sentence costs 10. **FRE is mostly a vocabulary meter.** (Flesch 1948; §2.1)

2. **FRE and Flesch-Kincaid Grade Level (FKGL) are not monotone transforms of each other.** FRE weights syllables 83:1 over sentence length; FKGL weights them 30:1. Two texts can swap rank between the two scores. Kincaid's 1975 recalibration used 531 Navy enlisted personnel reading 18 passages from Rate Training Manuals, scored against the Gates-MacGinitie comprehension test. Neither formula was validated on university prose. (§2.1)

3. **Implementations disagree by 5 to 8 percent on the same text, and disagree on rankings.** Across five Python libraries on 70 ground-truth passages, median syllable counts were off by −9.4% to +4.1%, FRE by −1.3% to +8.0%, FKGL by −4.6% to +0.8%, and SMOG by up to +17.2%. Only `py-readability-metrics` and a patched spaCy implementation tracked ground truth; the others ranked texts inconsistently. `textstat` (0.6.2 in that test, still true in `main`) silently drops sentences of two words or fewer. (Wagner 2022, Appendix A; §2.3)

4. **Human academic writing sits at FRE 10 to 40.** Top-100 neuroimaging papers: abstracts 15.7 ± 14.1, full text 32.1 ± 8.6. Medical journals 15 to 24; marketing 35. NeurIPS abstracts fell from about 24 in 1987 to 13 in 2024. More than a fifth of all scientific abstracts now score *below zero*. (§3.1)

5. **Everything humans write for the public sits at FRE 50 to 65.** English Wikipedia 51.2 (SD 13.8, N = 1.71M); *Time* about 52; everyday English corpus 56; *Reader's Digest* about 65; CNN/DailyMail 55.5; human lay summaries of eLife papers 52.4. Contracts score 20 at 42 words per sentence; privacy policies 31. (§3.2-3.4)

6. **On matched prompts, LLM output is lower-FRE than the human comparator in nearly every study.** Patient education: GPT-4 33 vs human 58.5. Lay summaries: ChatGPT 37.4 vs human 52.4. Detection benchmarks: AI 43.3 / 72.1 / 69.8 vs human 64.3 / 78.2 / 82.1. A three-class annotation study: human 69.4, undecided 57.4, AI 48.0. Scholarly abstracts with a strong AI signal score 25 to 50% lower FRE than those without, Cohen's d up to 0.76. (§4.1)

7. **The AI difficulty is lexical.** GPT-4 Turbo essays: FKGL 16.6 vs 12.1 for L2 students, Gunning Fog 20.1 vs 14.2, yet *mean T-unit length 15.9 vs 23.6 words*. ChatGPT is harder with shorter sentences, which is the signature of polysyllabic vocabulary and nominalization (Herbold: nominalization d = −0.88 to −1.35). (§4.1)

8. **Instruction-tuned models have a compressed readability range and a default register.** Asked for lay versus expert summaries, humans spanned 29.2 FRE points; ChatGPT spanned 7.0. Asked for CEFR A2, ChatGPT produced B1; asked for grade 2, it produced grade 8.8, or 4.6 with maximal specification. GPT-4 hit the requested grade level 50% of the time versus 79% for humans. (§4.2)

9. **"Write like a human" prompts make text harder, not easier.** TH-Bench's prompt-based rewrite lowered FRE by 5 to 14 points on every dataset and failed to evade detectors. Paraphrasers move the other way: DIPPER +3 to +9, recursive paraphrase +14 to +22 with ROUGE-L below 0.3. Commercial humanizers drift about 4 points harder (Undetectable 52 → 48; Rephrasy 52.3 → 48.7). (§4.3)

10. **GPTZero exposed readability as one of six explainability metrics until about March 2024, then removed them.** A surrogate trained on the six reached only 76 to 79% agreement with GPTZero's verdicts, with readability and perplexity carrying the weight. The production model is not a readability classifier, but readability is *correlated* with its output, and a feature-ablation study puts the readability group's marginal value below 0.5%. (§5)

11. **Readability does not predict grades, and where it does the sign is wrong for the "make it easier" theory.** ASAP human raters: FRE r = −0.17/−0.15 (argumentative), +0.13 (narrative); Coleman-Liau +0.30/+0.28; Gunning Fog −0.19 to −0.31. The strongest lexical predictor of quality in the Coh-Metrix literature is *lower* word frequency, and word frequency is the strongest discriminator between 9th-grade, 11th-grade and college essays. (§6)

12. **Modern readability research has abandoned the formulas.** Against eye-tracking on 360 readers, traditional formulas were "moderate to poor"; Coleman-Liau was the best of them and word surprisal beat all of them. GPT-4 zero-shot judgments correlate r = 0.76 with human ratings. The formulas survive because they are free and deterministic, which is exactly why we should use them only as a gate. (§2.4)

---

## 2. The formulas

### 2.1 Flesch (1948) and Kincaid (1975)

Rudolf Flesch published "A new readability yardstick" in the *Journal of Applied Psychology* 32(3), 1948. The Reading Ease formula was fitted by regression to the McCall-Crabbs Standard Test Lessons in Reading, with the criterion being the grade at which 75% of readers could answer comprehension questions correctly.

```
FRE  = 206.835 − 1.015 × (total words / total sentences) − 84.6 × (total syllables / total words)
FKGL = 0.39 × (total words / total sentences) + 11.8 × (total syllables / total words) − 15.59
```

The nominal range is 0 to 100, but the formula is unbounded: a sentence in *Moby-Dick* chapter 64 scores −146.77, and the maximum for one-syllable one-word sentences is 121.22. Flesch's own scale:

| FRE | School level | Flesch's description | Example (Flesch's measurements) |
|---|---|---|---|
| 90-100 | 5th grade | Very easy | comics |
| 80-90 | 6th | Easy | pulp fiction |
| 70-80 | 7th | Fairly easy | |
| 60-70 | 8th-9th | Plain English | *Reader's Digest* ≈ 65 |
| 50-60 | 10th-12th | Fairly difficult | *Time* ≈ 52 |
| 30-50 | College | Difficult | |
| 10-30 | College graduate | Very difficult | *Harvard Law Review* low 30s |
| < 10 | Professional | Extremely difficult | |

Kincaid, Fishburne, Rogers and Chissom recalibrated three formulas (ARI, Fog Count, Flesch) for the U.S. Navy in Research Branch Report 8-75 (1975). Participants were 531 enlisted personnel in technical training at two installations; the texts were 18 passages from Rate Training Manuals; comprehension was the Gates-MacGinitie test. The result is the grade-level rewrite above. Wikipedia's compilation of validation coefficients puts FKGL at r = 0.91 (SE 1.9 grades) and FRE at r = 0.88 (SE 2.44 grades) against comprehension criteria; Dale-Chall leads at 0.93.

Two arithmetic facts drive everything downstream. First, the syllable term dominates FRE: at typical academic values (23 words/sentence, 1.75 syllables/word) the sentence term contributes −23 points and the syllable term −148. Second, FKGL "emphasizes sentence length over word length" relative to FRE (the weight ratio is 30:1 rather than 83:1), so the two scores can rank a pair of texts differently. The privacy-policy study below found FKGL ranking texts *inversely* to the other measures on a 1,500-document corpus.

### 2.2 The rest of the family

| Formula | Year | Equation | Rewards / punishes | Known failure modes |
|---|---|---|---|---|
| Gunning Fog | 1952 | 0.4 × [(words/sentences) + 100 × (complex words/words)]; complex = 3+ syllables, *excluding* proper nouns, familiar jargon, compound words, and words made polysyllabic only by -es/-ed/-ing | Long sentences and polysyllables equally | The exclusion rules are rarely implemented (textstat does not exclude proper nouns; py-readability-metrics excludes capitalized and hyphenated words). Results differ by library more than for any other formula |
| SMOG | McLaughlin 1969 | 1.0430 × √(polysyllables × 30 / sentences) + 3.1291; original: 3 + √(polysyllables in 30 sentences) | Polysyllables only; sentence length enters only as a denominator | Calibrated to 100% comprehension, so it runs about two grades above Flesch. Statistically invalid under 30 sentences, which rules it out for paragraph-level checks |
| Coleman-Liau | 1975 | 0.0588 × L − 0.296 × S − 15.8, where L = letters per 100 words and S = sentences per 100 words | Letters per word; sentence count enters inversely | No syllable counting, so it is implementation-stable, and it was the best traditional formula against eye-tracking (§2.4). It is also the index most positively correlated with essay score (§6) |
| ARI | Senter & Smith 1967 | 4.71 × (characters/words) + 0.5 × (words/sentences) − 21.43 | Characters and sentence length | Designed for typewriter tallies; no syllable problem |
| Dale-Chall | 1948, revised 1995 | 0.1579 × (% difficult words) + 0.0496 × (words/sentences), plus 3.6365 if difficult words exceed 5%. Difficult = not on a 3,000-word list known to 80% of 4th graders. Rules: regular plurals, possessives and -d/-ing forms of listed words are not difficult; short numbers are not; proper names count once per 100-word sample | Vocabulary familiarity rather than length | All five libraries tested by Wagner (2022) use the 1948 list, none implement the rules, and difficult-word counts were off by +5% to +74%. Technical vocabulary is "difficult" by definition, so every research article is off the scale |
| Linsear Write | c. 1977 | On a 100-word sample: easy words (≤2 syllables) count 1, hard (≥3) count 3; r = total / sentences; grade = r/2 if r > 20, else (r − 2)/2 | Polysyllables, steeply | textstat uses only the first 100 words by default; the score is a function of where the sample starts |
| Spache | 1953, revised 1974 | 0.121 × (words/sentences) + 0.082 × (% unfamiliar words) + 0.659 | Primary-grade vocabulary | Built for grades 1-4. Irrelevant above grade 4; textstat still reports it |
| Lexile | MetaMetrics 1988- | Log of mean sentence length and mean log word frequency combined in the "Lexile specification equation" (Stenner et al. report the theoretical logit as 9.82247 × ln(MSL) − 2.14634 × MLWF − constant, mapped to Lexiles as 180 × logit + 200); scale 0L-2000L | Sentence length and word *frequency*, not length | Proprietary corpus and constants; the CCSS "stretch" bands are grades 6-8 925-1185L, 9-10 1050-1335L, 11-CCR 1185-1385L. *The New York Times* is reported at about 1380L, the top of the college-readiness band |

The calibration criteria differ (75% comprehension for Flesch, 100% for SMOG; cloze for Bormuth and Dale-Chall), so the formulas do not share a zero point and should never be averaged into a "consensus grade." textstat's `text_standard` does exactly that.

### 2.3 Why two tools give two numbers for one text

Wagner (2022) is the only controlled comparison I found. She built 70 passages with hand-verified counts of words, sentences, syllables, characters, difficult words and polysyllables, and ran five Python libraries.

| Library (version) | Sentence splitter | Syllables | Median error: syllables | Median error: FRE | Median error: FKGL | Median error: SMOG |
|---|---|---|---|---|---|---|
| textstat 0.6.2 | regex, **ignores short sentences** | Pyphen | −8.44% | **+6.45%** | −4.04% | −0.47% |
| readability (andreasvc) 0.3.1 | newline split | vowel count + dictionary | −7.75% | +7.99% | −4.62% | −0.70% |
| readability-score 2.1 | NLTK | Pyphen | −9.43% | +7.92% | −4.62% | −3.55% |
| spacy-readability 1.4.1 | spaCy | syllapy | +0.89% | +6.38% | −3.57% | **+17.16%** (counts syllables, not words, in polysyllables: +279.8%) |
| **py-readability-metrics 1.4.4** | NLTK | vowel count | +4.06% | **−1.32%** | **+0.80%** | +2.57% |
| patched spaCy (SR*) | NLTK | syllapy | +0.89% | −0.18% | +0.66% | +1.06% |

Difficult-word counts (Dale-Chall) were off by +8.7% (textstat) to +73.9%. On 1,500 real documents, Spearman rank correlations between implementations showed "substantial disagreement" except between the two accurate ones. Wagner's conclusion: "It is therefore important to document the choice of readability library and version." A concrete instance from the same paper: privacy policies of the top-million websites averaged FRE 39.8 in one 2018 study and 32.2 in hers, and she attributes the gap to implementation.

Specific mechanisms that matter for us:

- **Sentence filtering.** textstat's `count_sentences` in `main` still applies `if count_words(sentence) <= 2: ignore_count += 1`. A writer who uses "Not so." or "It failed." loses those sentences from the denominator, which *raises* average sentence length and *lowers* FRE. Since short-sentence tails (9-14% under 10 words, report 10) are part of the human shape we are targeting, textstat will systematically misreport our own output.
- **Syllable dictionaries.** Modern textstat tries CMUdict first, falling back to Pyphen hyphenation positions + 1. Pyphen is a hyphenation tool, not a syllabifier: it under-counts by about 8%. Any word not in CMUdict (technical terms, names, neologisms) gets the hyphenation estimate.
- **Word tokenization.** Whitespace splitting counts "(Smith et al., 2019)" as three words and "[12]" as one; digit tokens are dropped from textstat's syllable count but not necessarily from its word count. Citation-dense text therefore gets a depressed syllables-per-word ratio and an inflated FRE. Conversely, the period in "et al." or "Fig." can split a sentence, shortening average sentence length and again inflating FRE. Strip citations, numerals, URLs, equations and Greek symbols before scoring; Cunningham et al. (2025) note that "mathematical/chemical notation may artificially depress FRE scores" in the other direction.
- **Proper nouns and acronyms.** Fog's proper-noun exclusion is unimplemented in textstat; "fMRI" or "CRISPR" get arbitrary syllable counts.
- **Microsoft Word caps FKGL at 12** in several versions, so a paper that reads 15.1 on one machine reads 12.0 on another. Any user comparing our number to Word's will see a discrepancy.
- **Length.** SMOG needs 30 sentences; Linsear uses 100 words; FRE on a single paragraph swings by 10 points. Score documents, not paragraphs.

### 2.4 What replaced the formulas

- **Coh-Metrix Text Easability** (Graesser, McNamara & Kulikowich 2011) reports five principal components as percentiles rather than a grade: narrativity, syntactic simplicity, word concreteness, referential cohesion, deep (causal) cohesion, plus verb cohesion, connectivity and temporality in 3.0. It is the tool behind most of the writing-quality literature in report 09 and is web-only; `lingfeat` and `LFTK` are the open approximations.
- **CLEAR corpus / CommonLit Readability Prize** (Crossley et al. 2021, 2022). 4,724 excerpts of 140-200 words, fiction and non-fiction, 1875-2020, rated by teachers in pairwise comparisons and scaled with a Bradley-Terry model to an "easiness" score from −3.6 to 1.7. The corpus ships FRE, FKGL, ARI, SMOG, New Dale-Chall, CAREC, CARES and CML2RI columns. The 2021 Kaggle competition was won by transformer ensembles (the 2nd-place solution used RoBERTa, BART, ELECTRA, DeBERTa and XLNet; 4th place reported RMSE 0.447 on the BT scale). Note the population: grades 3-12 readers. Nothing in CLEAR calibrates university prose.
- **ReadMe++** (Naous et al., EMNLP 2024): 9,757 sentences in five languages from 112 sources, annotated on the CEFR 1-6 scale; benchmarks supervised, unsupervised and few-shot LLM readability assessment.
- **Eye-tracking validation** (Gruteke Klein et al. 2025, arXiv 2502.11150): 360 L1 readers, 30 Guardian articles in original and simplified form, four reading measures. Traditional formulas showed "moderate to poor predictive power"; Coleman-Liau was the best of them; word length and frequency matched or beat the formulas; surprisal from 32 language models (70M-13B) was the strongest single predictor. Idea density, integration cost and embedding depth were "very poor."
- **LLM-as-judge**: GPT-4 Turbo and GPT-4o-mini zero-shot readability ratings correlate r = 0.76 and 0.74 with human judgments, above all formulas (Trott & Rivière 2024). Across five datasets and 897 human judgments, four model-based metrics took the top four ranks; the best traditional metric averaged rank 8.6 (Readability Reconsidered, TSAR 2025). A 2026 ACL paper reports zero-shot open LLMs beating prior methods on 13 of 14 datasets and a hybrid (LAURAE) that fuses LLM scores with surface formulas. Si and Callan showed in 2001 that a language-model readability score beat FKGL on scientific text; Hartley asked in 2016 whether "time is up" for Flesch. The field's answer is yes, for measurement. For a cheap deterministic gate, the formulas are still the tool.

---

## 3. Where human writing lands

Every number below is FRE unless marked. Where a source gives SD or quantiles they are included; where it gives only a mean, that is stated.

### 3.1 Research articles

| Corpus | N | FRE | FKGL | Other | Source |
|---|---|---|---|---|---|
| Top-100 most-cited neuroimaging papers, **abstracts** | 100 | **15.70 ± 14.11** | 16.92 ± 3.02 | Fog 18.98, SMOG 17.19, CLI 15.89, ARI 16.77 | Yeung et al. 2018 |
| Same papers, **full text** | 100 | **32.11 ± 8.56** | 13.83 ± 1.70 | Fog 15.84, SMOG 15.04, CLI 13.24, ARI 15.04 | Yeung et al. 2018 |
| Medical journals (cited in the same paper) | | 15.4-23.8 | | | Yeung et al. 2018 |
| Marketing journals | | 35.3 | | | Yeung et al. 2018 |
| Scientific abstracts, 123 journals, 12 fields, 1881-2015 | 709,577 | trend r = −0.93; **14% below 0 in 1960, 22% below 0 in 2015** | | NDC rising r = 0.93; abstract-fulltext FRE r = 0.60 on 143,957 articles | Plavén-Sigray et al. 2017 |
| NeurIPS abstracts | 24,772 | **≈24 (1987) → ≈13 (2024)** | | acronyms in titles 0.33 → 3.21 per 100 words; more readable papers get more citations | arXiv 2605.08889 |
| Semantic Scholar abstracts, 23 fields, 2000-2024 | 16.97M | declining in every field; **≈18% mean fall 2020-2024** (CS 25%, Business >25%); Mathematics highest, Business lowest | | high-AI-signal quartile 25-50% lower than low-AI quartile; d = 0.16 (Math) to 0.76 (Medicine) | Cunningham, Smyth & Smyth 2025 |
| arXiv abstracts, 2013-2024 | 823,798 | declining, steeper after 2023 | | NDC rising; lexical complexity up, syntactic complexity down | arXiv 2505.12218 |
| PMC OA research articles (own measurement) | | | | 23.3 words/sentence (SD 4.1); 5.42 chars/word; 34.6% words > 6 chars | Report 10 |

Three things to read off this table. The abstract-to-body gap is about 16 FRE points and 3 grades in the same papers; abstracts are the hardest text scientists write. The decline is lexical: Plavén-Sigray et al. attribute it to "general scientific jargon" (their examples: *moreover, underlying, robust, suggesting*) with sentence length rising only "slightly since 1960." And the between-paper spread is large: an SD of 14 on abstracts means a normal-looking paper can sit anywhere from FRE 0 to 30.

### 3.2 Public prose, journalism, and encyclopedias

| Corpus | N | FRE | Grade | Source |
|---|---|---|---|---|
| English Wikipedia, articles with > 5 sentences | 1,710,752 | **51.18 (SD 13.84)**; 73.5% below 60 | | Lucassen et al. 2012 |
| Simple English Wikipedia | 21,366 | 61.69 (SD 13.00); 42.3% below 60, 94.7% below 80; fell from ≈80 (2003) to ≈70 (2006) | | Lucassen et al. 2012 |
| Matched titles, English vs Simple | | 49.27 vs 61.46 | | Lucassen et al. 2012 |
| English Wikipedia, disease articles with ICD-10 codes | 4,235 | **28.69 ± 11.00**; ≈97% below 50 | FKGL 14.26 ± 1.80; Fog 16.70; SMOG 16.12; CLI 15.16; ARI 13.65 | Adam et al. 2022 |
| Wikipedia 2020-2025 | | stable, "does not appear to be influenced by LLMs" | | arXiv 2503.02879 |
| Everyday English (news, magazines, blogs, web, scripts) | > 1M words | **56** | FKGL 9; 17 words/sentence | Martinez, Mollica & Gibson 2022 via Schiess |
| CNN/DailyMail news articles | | 55.52 | | Pu & Demberg 2023 (v1) |
| Human lay summaries of eLife papers | | 52.42 | CLI 12.46; DCR 8.93 | Pu & Demberg 2023 (v2) |
| Human expert digests of the same papers | | 23.20 | CLI 17.62; DCR 11.78 | Pu & Demberg 2023 (v2) |
| *Reader's Digest* / *Time* / *Harvard Law Review* | | ≈65 / ≈52 / low 30s | | Flesch, via Wikipedia |
| *The New York Times* | | | FKGL 10-12; ≈1380L | industry measurements (weak provenance) |
| Human essays, PAN CLEF 2025 (essays, news, fiction) | 9,101 train | **64.26** | | Pudasaini et al. 2025 |
| Human texts, COLING 2025 GenAI detection (peer reviews, student essays, papers, news, more) | 228,922 | 78.22 | | Pudasaini et al. 2025 |
| Human texts, Ghostbuster (student essays, creative writing, news) | 2,000 | 82.07 | | Pudasaini et al. 2025 |
| Human texts in a three-class annotation study | | 69.42 | FKGL 7.95 | Ji et al. 2025 |
| Magazine / newspaper prose, 2000s (sentence length only) | 19.8M sentences | | 17.1 / 16.7 words/sentence | Rudnicka 2018, report 10 |

Two cautions. The NYT figures come from content-marketing sites, not a study; I could not find a peer-reviewed measurement of *The New Yorker* or *The Atlantic*, and none of Orwell, Didion or Baldwin. The only literary data point with any provenance is *Moby-Dick* at FKGL 10.8 (Amazon's text stats, quoted by Wikipedia). Treat "op-ed FRE 50-65" as triangulated from Wikipedia, *Time*, the everyday-English corpus and CNN/DM, not measured directly.

### 3.3 Legal and consumer text

| Corpus | N | FRE | Grade | Words/sentence | Source |
|---|---|---|---|---|---|
| Contracts | 837,000 words | **20** | FKGL 19 | **42** | Martinez, Mollica & Gibson 2022 via Schiess |
| Privacy policies, 2001 → 2011 → 2021 | 1,500+ policies | 37 → 34 → **31**; 2021 median 31.8; 41.6% below 30; only 6.7% above 45 | SMOG 16.7 yrs; CLI 13.4 yrs | | Wagner 2022 |
| Privacy policies, top-1M sites, 2018 (other implementation) | | 39.8 vs 32.2 same year | | | Wagner 2022 |

FRE 45 is the Florida statutory minimum for insurance policies; legal text routinely fails it. Contracts are the one genre where sentence length, not vocabulary, does the damage.

### 3.4 Student essays

No study I could find reports FRE by holistic score band for BAWE, MICUSP, PERSUADE or ASAP. What exists:

| Corpus | Finding | Source |
|---|---|---|
| L2 undergraduate argumentative essays (n = 50) | **FKGL 12.12; Fog 14.18**; MLT 23.61; MTLD 66.56 | Frontiers in Education 2025 |
| PERSUADE top-scoring (5-6/6), own measurement | 22.3 words/sentence (SD 7.0); 4.65 chars/word; 21.8% words > 6 chars; CV 0.43 | Report 10 |
| PERSUADE all scores | 22.0 words/sentence (SD 11.0); CV 0.53 → 0.41 from score 1 to 6 | Report 10 |
| Human essays (Herbold 2023) | 339 words / 19 sentences ≈ 17.8 words/sentence | Herbold et al. 2023 |
| Secondary-school essays, above vs below median | 287 vs 162 words; more additive and adversative connectives in high scorers | Tate et al. 2024 |
| 9th grade → 11th grade → first-year college | **Word frequency is the strongest discriminator**: college writers use less frequent words | Crossley et al. 2011, via Crossley 2020 |
| ASAP grades 7-10 | readability indices uncorrelated with score (§6) | Kundu & Barbosa 2024 |

Report 10's own numbers let us bound the answer. Top PERSUADE essays average 4.65 characters per word against 5.42 for research articles and 4.85 for news; sentence length is research-article-like at 22 words. That places good high-school and first-year argumentative writing at roughly FRE 50-60, FKGL 10-12, and the L2 undergraduate measurement at FKGL 12.1 agrees. Upper-division and BAWE-style writing, with its heavier nominalization (report 10: 26 per 1,000 in top essays vs 60 in research articles), should sit between the two, around FRE 35-50. **This is an inference, not a measurement; measuring FRE on BAWE by grade band is the first thing to do once the corpus is acquired.**

---

## 4. Where AI text lands

### 4.1 Matched comparisons: AI is harder

| Task | Human | AI | Model | Source |
|---|---|---|---|---|
| Total knee arthroplasty patient education (5 vs 5) | **FRE 58.5, FKGL 8.52** | **FRE 33, FKGL 13.1** (p < 0.001) | GPT-4 | Report 04 source PMC12278881 |
| Stroke education for clinicians (1 vs 1) | FRE 20.6, FKGL 16.2 | FRE 20.4, FKGL 13.3 (ns) | GPT-4o vs UpToDate | PMC12787534 |
| Medical communication meta-analysis | recommended FRE 80-90 | FRE range 21.3-76.9, FKGL 3.6-15.9; Bard/Gemini FRE **+10.36** vs ChatGPT; FKGL 1.62 lower; neither reaches grade 6 | GPT-3.5/4, Bard | PMC12403948 |
| eLife lay summary | FRE 52.42 | FRE 37.38 | ChatGPT | Pu & Demberg 2023 |
| eLife expert digest | FRE 23.20 | FRE 30.38 | ChatGPT | Pu & Demberg 2023 |
| CNN/DailyMail summary | FRE 55.52 | FRE 46.49 | ChatGPT | Pu & Demberg 2023 (v1), report 04 |
| Argumentative essays (50 vs 50) | FKGL 12.12; Fog 14.18; **MLT 23.61**; DC/T 0.57; MTLD 66.56 | **FKGL 16.61; Fog 20.10; MLT 15.91**; DC/T 0.75; MTLD 118 | GPT-4 Turbo, Dec 2024 | Frontiers in Education 2025 |
| Essays, Herbold | 339 w / 19 sent; nominalizations 1.06; modals 10.84; epistemic 0.06 | 248-254 w / 12-13 sent; nominalizations 1.56 (GPT-3, d = −0.88) and 1.73 (GPT-4, d = −1.35); modals 8.97 / 6.12; epistemic 0.02 / 0.00 | ChatGPT-3 / 4 | Herbold et al. 2023 |
| PAN CLEF 2025 | FRE 64.26 | **43.26** | GPT-4o and others | Pudasaini et al. 2025 |
| COLING 2025 | FRE 78.22 | 72.07 | GPT-4/4o, Mistral, Llama 3.1, Qwen-2, Claude | Pudasaini et al. 2025 |
| Ghostbuster | FRE 82.07 | 69.81 | GPT-3.5, Claude | Pudasaini et al. 2025 |
| Three-class human annotation (human / undecided / AI) | **69.42 / 57.44 / 48.02**; FKGL 7.95 / 9.28 / 10.72; GPT-2 PPL 52.7 / 34.2 / 21.6 | | ChatGPT-3.5, Llama-13B, Gemini Pro, Qwen2-72B | Ji et al. 2025 |
| Wikipedia paragraphs, original vs LLM-revised | | "less readable" on all six formulas | | arXiv 2503.02879 |
| Scholarly abstracts, low- vs high-AI-signal quartile | | FRE 25-50% lower; d 0.16-0.76 | | Cunningham et al. 2025 |
| Hotel reviews | | lower readability | GPT-3.5 | Markowitz et al. 2023, via Terčon survey |

The direction is consistent: **instruction-tuned models write at a lower FRE than the human text they are compared with**, by 6 to 25 points, on patient information, news, essays, encyclopedia text and reviews alike. The exception is when the human comparator is itself expert prose (stroke content for clinicians, eLife expert digests), where the model is at or slightly above the human level. Report 04's note stands: when explicitly prompted for plain language, GPT-4o can match or beat Cochrane plain-language summaries, and an "extended" lay-summary prompt lifted median FRE from 29.1 to 40.9.

The Frontiers numbers are the important ones. GPT-4 Turbo produced *shorter* T-units (15.9 vs 23.6 words) with more dependent clauses per T-unit (0.75 vs 0.57) and still landed 4.5 grades higher. That is a vocabulary effect: MTLD 118 vs 67, lexical sophistication 0.41 vs 0.35, and Herbold's nominalization gap. **The AI fingerprint on readability is polysyllabic Latinate vocabulary in moderate-length sentences, not long sentences.** This matters for the humanizer because the obvious lever, sentence splitting, does not address what actually differs.

### 4.2 Instruction tuning, default register, and the compressed range

There is no controlled study of FRE for the same checkpoint before and after instruction tuning; I looked and it is a real gap, worth running ourselves on Llama-3-8B and Qwen3-8B base versus instruct (report 13 already has the detector numbers for those pairs). What exists is indirect but consistent:

- **Compressed stylistic range.** Asked to write lay and expert versions, humans spanned 29.22 FRE points (52.42 to 23.20), ChatGPT 6.99 (37.38 to 30.38). Coleman-Liau: humans 5.16 apart, ChatGPT 1.04. Dale-Chall: humans 2.85 apart, ChatGPT 0.68. On GYAFC formality transfer, humans moved 3.7 formality points between informal and formal; ChatGPT moved 1.3, but its "formal" MTLD jumped to 31.68 against a human 18.70. The model does formality by swapping in rarer words, which is exactly the readability signature in §4.1. (Pu & Demberg 2023)
- **A default level it cannot leave.** Targeting FKGL 2 (acceptable 1-3), ChatGPT produced 8.83 with no specification and 4.57 with full specification; Llama-2-7B 8.22 → 6.34; FlanT5-base got closest at 5.13. On CEFR, ChatGPT and Dolly "produced outputs one level higher than the target level, which is B1 instead of A2," with 0-13% accuracy; FlanT5 reached 85-98%. (Imperial & Tayyar Madabushi 2023)
- **Grade targeting fails half the time.** On 13.4K "why" questions across elementary, high-school and graduate levels, GPT-4 explanations matched the intended grade 50% of the time versus 79% for human-written ones, and "explanations generated across different language model families for different informational needs remain indistinguishable in their grade-level." (ELI-Why 2025)
- **Family differences exist.** Bard/Gemini output ran about 10 FRE points easier than ChatGPT across the medical meta-analysis. Different RLHF pipelines produce different default registers; the same is likely true of Claude and Llama-Instruct, but I found no clean measurement.

The practical reading: a chat model has a house register around FRE 30-50 for informational prose, resists instructions to leave it, and its version of "more formal" is "more polysyllabic." Base models, per report 13, do not carry this register, which is one more reason to generate from them.

### 4.3 What humanizing does to readability

| Intervention | Dataset | FRE before → after | Δ | Notes | Source |
|---|---|---|---|---|---|
| **Prompt-based "humanize" rewrite** | Essay / Wikipedia / Reuters | 41.15 → 28.24 / 74.97 → 69.84 / 48.88 → 34.44 | **−12.9 / −5.1 / −14.4** | preserves meaning (cosine ≈ 0.88) and **fails to evade** (AUC 0.939, report 12) | TH-Bench 2025 |
| DIPPER paraphrase | same | 41.15 → 50.31 / 74.97 → 81.35 / 48.88 → 51.84 | +9.2 / +6.4 / +3.0 | | TH-Bench |
| TOBLEND | same | → 53.65 / 82.25 / 57.66 | +12.5 / +7.3 / +8.8 | | TH-Bench |
| Recursive paraphrase | same | → 62.88 / 88.71 / 67.91 | **+21.7 / +13.7 / +19.0** | ROUGE-L < 0.3; PPL +41 to +79 | TH-Bench |
| HMGC (adversarial word substitution) | same | ±1.5 | ≈0 | PPL 16.8 → 62.5 (report 12) | TH-Bench |
| RAFT | same | ±2 | ≈0 | cosine > 0.95 | TH-Bench |
| Undetectable AI | mixed | ≈52 → ≈48 | **−4.2** | "roughly one grade level higher in reading difficulty" | fast.io 2026 |
| Rephrasy | mixed | 52.3 → 48.7 in 45% of samples | −3.6 | | Report 03 |
| 19 commercial humanizers | academic and general | not measured numerically | | "Some humanizers write exclusively in an academic, formal, and/or university level tone. Others write at the elementary school, middle school, or high school level. The better humanizers, usually the ones that are LLM-based, do not commit to a specific writing level or tone, and instead adopt the writing level and tone of the original document." Fluency win rates: L1 tools 26.0%, L2 14.7%, L3 2.7% | DAMAGE 2025 |
| SICO | | | | human-rated readability 3.84 vs 3.92 for human text; better than DIPPER | Report 13 |

Two lessons. Prompting an instruct model to "sound human" pushes FRE *down* by 5 to 14 points, the opposite of what the folk theory predicts and in the direction of the AI signature. Paraphrasers push it *up*, and the more aggressively they paraphrase the more they move, until recursive paraphrase has raised FRE by 20 points and destroyed the content. Either shift, if large, is a fingerprint. A humanizer that preserves the input's readability within a few points is the one that DAMAGE describes as "better," and that is a property we can enforce.

---

## 5. Readability and detectors

**GPTZero.** Ji et al. (2025) document that GPTZero's interface, as of March 2024, exposed six explainability metrics: readability, percent SAT words, simplicity, perplexity, burstiness and average sentence length; "these features have since been removed." Training four surrogate classifiers on those six values to predict GPTZero's own verdicts gave 75.8-78.8% accuracy, with readability and perplexity carrying most of the weight (logistic regression weights 3.09 for readability, −2.52 for perplexity; perceptron 4.11 and −4.44 with simplicity at 8.15) and burstiness near zero. Their conclusion: "GPTZero employs more complex calculations or utilizes additional sophisticated features that are not disclosed," which matches report 07's finding that the production model is a supervised transformer. Their worked examples show two ChatGPT texts flagged AI with readability 72.3 ("High") and 61.8 ("Medium"): a high FRE did not save either.

**Other detectors.** Winston shows FKGL as a separate panel; Copyleaks, Winston and the XGBoost baselines in report 01 include readability among stylometric features. Pudasaini et al. (2025) put FRE in a five-family feature set for explainable detection and recover the AI-is-harder gap on all three benchmarks, but do not report its individual importance. Report 04's ablation (arXiv 2606.04177) found the readability group's marginal contribution under 0.5% while lexical richness carried 13.1%. Originality.ai's FRE blog post says nothing connecting readability to detection. I found no documentation of readability features in Turnitin or Pangram.

**Correlation, not causation.** Ji et al.'s Table 25 is the cleanest gradient: texts humans judged human / undecided / AI had FRE 69.4 / 57.4 / 48.0 and GPT-2 perplexity 52.7 / 34.2 / 21.6. Liang et al. (2023) found the same association from the other side: TOEFL essays with low perplexity and limited vocabulary were flagged AI more than half the time, "enhancement of word choice in non-native English writing samples reduced misclassification, while simplifying native writing samples increased it." Rarer, longer words made text read *more* human to perplexity-based detectors even though they lower FRE. Readability tracks the AI signal because both are downstream of vocabulary choice; it is not itself the mechanism.

**Readability shift as a humanizer artifact.** Report 12 item 16 flagged "register whiplash" with a fix of "measure formality and readability on the input, require the output to sit within a narrow band of it." The evidence above quantifies it: paraphrasers move FRE by +3 to +22, prompt rewrites by −5 to −14, commercial tools by about −4. A detector that has access to the source (Turnitin's paraphrase class, GPTZero's AI-Paraphrased head, Pangram's humanizer head) or that models genre-conditional readability can use the mismatch. TH-Bench already reports FRE distance as a standard quality metric. **Gate on |ΔFRE| ≤ 5 and |ΔFKGL| ≤ 1 between input and output, computed with the same implementation on both.**

---

## 6. Readability and grades

**The direct evidence.** Kundu and Barbosa (2024) correlated eight readability indices with human and LLM scores on ASAP (about 13,000 essays, grades 7-10):

| Index | Task 1 (argumentative, 1-6): Rater 1 / Rater 2 | Task 7 (narrative, 0-15): Rater 1 / Rater 2 |
|---|---|---|
| Flesch-Kincaid Grade | 0.02 / 0.01 | −0.19 / −0.19 |
| **Flesch Reading Ease** | **−0.17 / −0.15** | +0.13 / +0.13 |
| SMOG | 0.16 / 0.14 | 0.16 / 0.16 |
| **Coleman-Liau** | **0.30 / 0.28** | 0.11 / 0.10 |
| Gunning Fog | −0.19 / −0.21 | −0.30 / −0.31 |
| ARI | 0.01 / −0.01 | −0.14 / −0.14 |
| Linsear Write | −0.10 / −0.12 | −0.21 / −0.21 |
| Dale-Chall | 0.16 / 0.16 | −0.06 / −0.09 |

"Both human and LLM scores seem minimally influenced by readability indices ... some correlations with all scoring methods are negative, indicating that both humans and LLMs may assign high grades to less readable texts." Pull the table apart and a pattern appears. The indices that are positively correlated with score in the argumentative task (Coleman-Liau, SMOG, Dale-Chall) are the ones that reward longer or rarer words; the indices that mix in sentence length (Fog, Linsear, FKGL) are flat or negative. Higher-scoring argumentative essays use longer words in sentences that are not longer.

**The Coh-Metrix literature says the same thing with better measures.** McNamara, Crossley and McCarthy (2010) found the three strongest predictors of expert essay ratings were words before the main verb, MTLD, and *lower* CELEX word frequency; none of 26 cohesion indices distinguished score bands. Crossley et al. (2011) found word frequency the strongest discriminator between 9th-grade, 11th-grade and college essays, with college writers using less frequent words. Crossley's 2020 overview: proficient writers "produce less frequent words ... and words with more letters or syllables." Every one of these lowers FRE.

**But not monotonically.** Report 09 §4.5 covers the ceiling: AP English withholds its Sophistication point for "complicated or complex sentences or language that is ineffective because it does not enhance the analysis," and very low word frequency reads as thesaurus abuse. Herbold's teachers rated ChatGPT essays higher than student essays (4.68 vs 3.69 out of 7) *despite* their heavier nominalization, which says polysyllables are not penalized, not that they are rewarded without limit. And Perelman's length confound (report 09 §4.4) means any naive scorer will punish a humanizer that shortens, regardless of what it does to FRE.

**So what does good college writing score?** Triangulating §3.1 and §3.4: published research articles sit at FRE 25-40 in the body; the L2 undergraduate sample at FKGL 12.1; top PERSUADE essays at about 4.65 characters per word and 22 words per sentence, which is FRE 50-60 territory. Good upper-division writing, with nominalization and academic-vocabulary coverage between the two (report 09: AVL 9-14% in published prose, under 5% in weak student prose), lands around **FRE 35-50, FKGL 12-15**. **Pushing FRE up from there, by shortening sentences or simplifying vocabulary, moves the text toward the score-2 profile of report 10 (less nominalization, more informality). Pushing it down by inflating vocabulary moves it toward the "ineffective" ceiling. Neither direction raises a grade; only the evidence-and-commentary edits in report 09 do.**

---

## 7. Recommendation

### 7.1 Target bands

Two-sided bands, wide enough that the between-author spread (SD 8-14 in every measured corpus) is inside them. These are *plausibility gates*, not objectives. The operative constraint is the input-output delta.

| Genre | FRE band | FKGL band | Center (for sampling a reference) | Basis |
|---|---|---|---|---|
| Research article, abstract (STEM/biomed) | **5-30** | 15-18 | 16 / 17 | Yeung 2018; Plavén-Sigray 2017 (22% below 0) |
| Research article, body (STEM/biomed) | **22-42** | 13-16 | 32 / 14 | Yeung 2018 full text 32.1 ± 8.6 |
| Research article, humanities and social science | **28-48** | 12-15 | 38 / 13 | marketing 35.3; Mathematics highest of 23 fields (Cunningham 2025); Business lowest |
| Upper-division essay, BAWE/MICUSP-style | **35-52** | 12-15 | 44 / 13 | inference from §3.4 and §6; **measure on BAWE first** |
| First-year argumentative essay | **45-62** | 10-13 | 53 / 11 | PERSUADE top essays (report 10); L2 sample FKGL 12.1 |
| Op-ed and long-form journalism | **48-66** | 9-12 | 56 / 10 | everyday English 56; Wikipedia 51.2 ± 13.8; *Time* 52; CNN/DM 55.5; *Reader's Digest* 65 |

Rules of use:

1. **Preserve the input's readability first.** Compute FRE, FKGL, syllables/word and words/sentence on the input and the output with the same implementation. Require |ΔFRE| ≤ 5, |ΔFKGL| ≤ 1, |Δ syllables/word| ≤ 0.05. This is the artifact gate from §5 and is stricter than the genre band.
2. **Then check the genre band.** If the input is already outside its band (a first-year essay at FRE 30 because the model wrote it), the humanizer may move toward the band, but by no more than the delta allows per pass, and only via the safe edits in report 00 §4 (kill AI vocabulary, strip connectives, add specifics), never by sentence splitting.
3. **Track the two components separately.** The human profile is moderate sentence length with vocabulary-driven difficulty, and a sentence-length CV of 0.42-0.60 (report 10). A text can hit the FRE band with 12-word sentences and 2.0 syllables per word, or 30-word sentences and 1.5; only one of those is human academic prose. Report the components, and let report 10's shape features govern sentence length.
4. **Never put FRE in a reward.** A reward term on readability produces detector-shaped or grade-shaped text in exactly the ways §7.3 lists. Use it as a hard gate returning a large negative outside the delta and the band, as report 00 §6 Stage 3 does for perplexity and error rate.
5. **Sample, do not center.** Draw the target FRE from the reference document whose whole feature vector is being resampled (report 16), not from the band center.

### 7.2 Implementation

- **Standardize on one pinned implementation and log its version with every score.** Wagner's evaluation supports `py-readability-metrics` (NLTK sentence tokenizer, FRE within 1.3% and FKGL within 0.8% of ground truth, and rank-consistent with the patched spaCy implementation). If `textstat` is preferred for its breadth, pin ≥ 0.7.x (CMUdict first, Pyphen fallback) and **override `count_sentences`** so that sentences of one or two words are not dropped; otherwise every short human sentence we generate is erased from the denominator.
- **Preprocess before scoring.** Strip parenthetical and bracketed citations, DOIs, URLs, footnote markers, equations and inline code; normalize "et al.", "Fig.", "e.g." so they do not split sentences; drop numerals from the syllable count *and* the word count consistently. Score the cleaned text, store both raw and cleaned values, and compare only cleaned-to-cleaned.
- **Use spaCy for sentence boundaries, not regex.** Then compute syllables with CMUdict and a Pyphen fallback flagged per word, so the fraction of dictionary misses is known (technical text will have 10-20% misses; that fraction is itself a useful register feature).
- **Do not report a "consensus grade."** `text_standard` averages formulas with different calibration criteria. Report FRE, FKGL and Coleman-Liau, the last because it is syllable-free, implementation-stable, the best traditional formula against eye-tracking, and the index most aligned with graders.
- **Score whole documents.** SMOG below 30 sentences and any formula below about 200 words is noise. Per-paragraph readability in the heatmap should be shown as a trend, not a number.
- **Never compare our number to Word's.** Word caps FKGL at 12 in several versions. Say so in the UI when a user pastes a Word value.
- **For actual comprehensibility**, when we need it for the quality dial, use surprisal under a mid-sized LM or an LLM judge calibrated on CLEAR-style pairwise data, not a formula. The formula is the gate; the model is the measure.

### 7.3 How a humanizer games readability while hurting quality

Each of these raises FRE or lowers it in a way that looks like progress on a dial and is a documented quality or detection cost.

| Move | Readability effect | Cost | Evidence |
|---|---|---|---|
| Split sentences at conjunctions and relative clauses | FRE up 3-8 | Unpacks phrasal complexity into finite clauses, which is a move toward conversation and a "Never" in report 00 §4; kills the > 30-word tail humans keep at 18-28% | Biber et al. 2011; report 10 |
| De-nominalize ("the implementation of" → "when we implemented") | FRE up | Nominalization rises 17.7 → 31.2 per 1,000 from score 1 to 6; this reverses it | Report 10 |
| Replace technical terms with short vague words | FRE up | AP "ineffective language"; DAMAGE's L3 tier behavior ("the findings indicate" → "we can see that") | Reports 03, 09 |
| Strip hedges (*approximately, potentially, arguably*) | FRE up | Hedges are polysyllabic and are the academic register at 8-20 per 1,000 | Hyland, report 15 |
| Delete citations to shorten sentences | FRE up (and measurement artifact) | "Reorder or drop evidence" is a very-high-cost edit | Report 00 §4 |
| Add contractions and fragments | FRE up | Never in body; ≤ 1 fragment per 1,500 words | Report 00 §4 |
| Recursive paraphrase to escape detectors | FRE up 14-22 | ROUGE-L below 0.3; content destroyed | TH-Bench |
| Insert periods in abbreviations or after list items | FRE up with no text change | Pure measurement gaming; a splitter-aware detector sees nothing, a formula-based gate is fooled | §2.3 |
| **The reverse: inflate vocabulary to hit an "academic" FRE of 30** | FRE down | Tortured synonyms, high-MI/low-T collocations, the thesaurus profile that report 12's masked-LM check exists to catch; also exactly what ChatGPT does when asked to be formal (MTLD 31.7 vs 18.7) | Pu & Demberg 2023; report 12 |
| Prompt an instruct model to "sound more human" | FRE down 5-14 | Does not evade (AUC 0.939) and moves toward the AI signature | TH-Bench; report 12 |
| Convert every passive to active | FRE up slightly | Passives are 25% of finite verbs in academic prose; halving them is register whiplash | Biber et al. 1999; report 10 |

The safe region is the same one report 00 identified: kill AI vocabulary (*delve, pivotal, underscore* are polysyllabic, so FRE drifts up 1-2 points as a side effect and that is fine), strip sentence-initial connectives (*Furthermore, Moreover*: same), add concrete named and dated specifics (numerals and proper nouns; FRE roughly neutral once they are stripped for scoring), and realize the sentence-length distribution from report 10. None of those requires touching FRE on purpose, and all of them leave it inside the delta.

### 7.4 Open items

1. Measure FRE, FKGL, Coleman-Liau, syllables/word and words/sentence on BAWE by grade band and discipline, and on MICUSP by level, with the pinned implementation. This fills the largest hole in §3.4 and replaces the inferred upper-division band with a measured one.
2. Run the same panel on Llama-3-8B and Qwen3-8B base versus instruct on identical prompts, to quantify the instruction-tuning shift that §4.2 can only infer.
3. Add the input-output readability delta to the report 12 artifact suite as item 16's concrete test, and log it on every GPTZero calibration call so we learn whether the delta predicts verdict flips.
4. Add Coleman-Liau and surprisal to the quality dial in place of any single Flesch number.

---

## Sources

Formulas and history
- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology* 32(3), 221-233.
- Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). Derivation of new readability formulas (ARI, Fog Count and Flesch Reading Ease) for Navy enlisted personnel. Research Branch Report 8-75. https://stars.library.ucf.edu/istlibrary/56/
- Wikipedia, Flesch-Kincaid readability tests (scale table, range, example scores). https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
- Wikipedia, Readability (formula table with validation coefficients; Fog, SMOG, Dale-Chall, New Dale-Chall, Lexile). https://en.wikipedia.org/wiki/Readability
- Wagner, I. (2022). Privacy Policies Across the Ages: Content and Readability of Privacy Policies 1996-2021. arXiv 2201.08739, §3.5.4 and Appendix A (five-library evaluation). https://arxiv.org/abs/2201.08739
- textstat source, `count_sentences` (≤2-word filter) and `count_syllables` (CMUdict then Pyphen). https://raw.githubusercontent.com/textstat/textstat/main/textstat/backend/counts/_count_sentences.py ; https://raw.githubusercontent.com/textstat/textstat/main/textstat/backend/counts/_count_syllables.py ; https://raw.githubusercontent.com/textstat/textstat/main/textstat/textstat.py
- py-readability-metrics documentation and source. https://py-readability-metrics.readthedocs.io/en/latest/ ; https://raw.githubusercontent.com/cdimascio/py-readability-metrics/master/readability/text/analyzer.py
- Microsoft Q&A, Word's FKGL ceiling of 12. https://learn.microsoft.com/en-us/answers/questions/4804270/why-does-my-flesch-kincaid-grade-level-only-go-up
- MetaMetrics, The Lexile Framework for Reading. https://metametricsinc.com/wp-content/uploads/2017/07/The-Lexile-Framework-for-Reading.pdf ; Stenner, Burdick, Sanford & Burdick, Lexile Framework technical report. http://cdn.lexile.com/m/resources/materials/Stenner_Burdick_Sanford__Burdick-_The_LFR_Technical_Report.pdf
- California Dept. of Education, New Research on Text Complexity (CCSS Appendix A supplement, Lexile grade bands). https://www.cde.ca.gov/be/cc/cd/documents/sept2012item2aatt4.doc
- Graesser, McNamara & Kulikowich (2011). Coh-Metrix: Providing multilevel analyses of text characteristics. *Educational Researcher* 40(5). https://journals.sagepub.com/doi/abs/10.3102/0013189x11413260
- Crossley, S. et al. (2021). The CommonLit Ease of Readability (CLEAR) Corpus. EDM 2021. https://educationaldatamining.org/EDM2021/virtual/static/pdf/EDM21_paper_35.pdf ; (2022) A large-scaled corpus for assessing text readability. *Behavior Research Methods*. https://link.springer.com/article/10.3758/s13428-022-01802-x ; corpus README. https://github.com/scrosseye/CLEAR-Corpus ; CommonLit blog. https://www.commonlit.org/blog/introducing-the-clear-corpus-an-open-dataset-to-advance-research-28ff8cfea84a/
- Kaggle CommonLit Readability Prize, 1st/2nd/4th place solutions. https://github.com/mathislucka/kaggle_clrp_1st_place_solution ; https://github.com/TakoiHirokazu/kaggle_commonLit_readability_prize ; https://github.com/Anjum48/commonlitreadabilityprize
- Naous, T. et al. (2024). ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.682/
- Gruteke Klein, K. et al. (2025). Readability Formulas, Systems and LLMs are Poor Predictors of Reading Ease (OneStop eye tracking). arXiv 2502.11150. https://arxiv.org/abs/2502.11150
- Trott, S. & Rivière, P. (2024). Measuring and Modifying the Readability of English Texts with GPT-4. TSAR 2024. https://arxiv.org/abs/2410.14028
- Readability Reconsidered: A Cross-Dataset Analysis of Reference-Free Metrics. TSAR @ EMNLP 2025. https://arxiv.org/abs/2510.15345
- Zero-shot Large Language Models for Automatic Readability Assessment (LAURAE). ACL 2026. https://arxiv.org/abs/2604.24470
- Imperial, J. M. & Tayyar Madabushi, H. (2023). Flesch or Fumble? Evaluating Readability Standard Alignment of Instruction-Tuned Language Models. GEM 2023. https://arxiv.org/abs/2309.05454
- Si, L. & Callan, J. (2001). A statistical model for scientific readability. CIKM. Hartley, J. (2016). Is time up for the Flesch measure of reading ease? *Scientometrics* 107(3). (Both cited via Cunningham et al. 2025.)

Human text
- Plavén-Sigray, P., Matheson, G. J., Schiffler, B. C., & Thompson, W. H. (2017). The readability of scientific texts is decreasing over time. *eLife* 6:e27725. https://elifesciences.org/articles/27725
- Yeung, A. W. K. et al. (2018). Readability of the 100 Most-Cited Neuroimaging Papers Assessed by Common Readability Formulae. *Frontiers in Human Neuroscience*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6104455/
- Cunningham, P., Smyth, P., & Smyth, B. (2025). Shifting norms in scholarly publications: trends in readability, objectivity, authorship, and AI use. arXiv 2510.21725. https://arxiv.org/abs/2510.21725
- Examining Linguistic Shifts in Academic Writing Before and After the Launch of ChatGPT: A Study on Preprint Papers. arXiv 2505.12218. https://arxiv.org/abs/2505.12218
- Machine Learning Research Has Outpaced Its Communication Norms and NeurIPS Should Act. arXiv 2605.08889. https://arxiv.org/abs/2605.08889
- Lucassen, T., Dijkstra, R., & Schraagen, J. M. (2012). Readability of Wikipedia. *First Monday* 17(9). https://firstmonday.org/ojs/index.php/fm/article/view/3916
- Adam, M. et al. (2022). Readability of English, German, and Russian Disease-Related Wikipedia Pages. *JMIR*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9152717/
- Wikipedia in the Era of LLMs: Evolution and Risks. arXiv 2503.02879. https://arxiv.org/abs/2503.02879
- An Open Multilingual System for Scoring Readability of Wikipedia. arXiv 2406.01835. https://arxiv.org/abs/2406.01835
- Martinez, E., Mollica, F., & Gibson, E. (2022). Poor writing, not specialized concepts, drives processing difficulty in legal language. *Cognition* 224; numbers via Schiess, W., Readable Contracts Parts 1 and 3, UT Austin LEGIBLE. https://sites.utexas.edu/legalwriting/2022/06/13/readable-contracts-part-1/ ; https://sites.utexas.edu/legalwriting/2022/09/07/readable-contracts-part-3/
- Eye And Pen, Analyzing The New York Times' Reading Level (industry measurement). https://www.eyeandpen.com/new-york-times-reading-level/
- Lexical diversity, syntactic complexity, and readability: a corpus-based analysis of ChatGPT and L2 student essays. *Frontiers in Education* 2025. https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1616935/full
- Tate, T. P. et al. (2024). Linguistic Features of Secondary School Writing. *Written Communication* 41(3). https://files.eric.ed.gov/fulltext/EJ1426628.pdf
- Morris, W., Crossley, S., Holmes, L., & Suh Choi, J. (2025). Distinguishing Effective Writing Styles in the PERSUADE Corpus. *Journal of Writing Research* 17(2).
- Schumacher, E. & Eskenazi, M. (2016). A Readability Analysis of Campaign Speeches from the 2016 US Presidential Campaign. arXiv 1603.05739. https://arxiv.org/abs/1603.05739

AI text and humanizers
- Herbold, S., Hautli-Janisz, A., Heuer, U., Kikteva, Z., & Trautsch, A. (2023). A large-scale comparison of human-written versus ChatGPT-generated essays. *Scientific Reports* 13:18617. https://pmc.ncbi.nlm.nih.gov/articles/PMC10616290/
- Pu, D. & Demberg, V. (2023). ChatGPT vs Human-authored Text: Insights into Controllable Text Summarization and Sentence Style Transfer. arXiv 2306.07799 (v1 CNN/DM; v2 eLife and GYAFC). https://arxiv.org/abs/2306.07799 ; https://arxiv.org/html/2306.07799v2
- Readability of ChatGPT-4 vs human patient education materials for total knee arthroplasty (FRE 33 vs 58.5). https://pmc.ncbi.nlm.nih.gov/articles/PMC12278881/
- Readability Comparison of AI-Generated Versus UpToDate Educational Content on Stroke Management. https://pmc.ncbi.nlm.nih.gov/articles/PMC12787534/
- Comparison of the readability of ChatGPT and Bard in medical communication: a meta-analysis. https://pmc.ncbi.nlm.nih.gov/articles/PMC12403948/
- Leveraging LLMs for High-Quality Lay Summaries: ChatGPT-4 with custom prompts (median FKRE 40.9 vs 29.1). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11854015/
- Improving the readability of trauma patient education materials: a ChatGPT solution. https://pubmed.ncbi.nlm.nih.gov/40969547/
- ELI-Why: Evaluating the Pedagogical Utility of Language Model Explanations. arXiv 2506.14200. https://arxiv.org/abs/2506.14200
- Pudasaini, S., Miralles-Pechuán, L. et al. (2025). Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy (Table 3, FRE by dataset).
- Terčon, L. Linguistic Characteristics of AI-Generated Text: A Survey (cites Markowitz, Hancock & Bailenson 2023, *Journal of Language and Social Psychology*, on lower readability of AI reviews).
- Zheng et al. (2025). TH-Bench: Evaluating Evading Attacks via Humanizing AI Text. arXiv 2503.08708 (FRE before/after each attack). https://arxiv.org/html/2503.08708v1
- Masrour, E., Emi, B., & Spero, M. (2025). DAMAGE: Detecting Adversarially Modified AI Generated Text. GenAIDetect @ COLING 2025. https://arxiv.org/abs/2501.03437
- fast.io, Undetectable AI Review 2026 (−4.2 Flesch points). https://fast.io/resources/undetectable-ai-review-2026/

Detectors and grading
- Ji, J., Li, R., Li, S. et al. (2025). Detecting Machine-Generated Texts: Not Just "AI vs Humans" and Explainability is Complicated. arXiv 2406.18259 (§5.1, Appendix B Table 8, Appendix G Table 25). https://arxiv.org/abs/2406.18259
- Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors are biased against non-native English writers. *Patterns* 4(7). https://www.cell.com/patterns/fulltext/S2666-3899(23)00130-7
- Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods. arXiv 2406.15583. https://arxiv.org/abs/2406.15583
- Originality.ai, Finding Out About the Flesch Kincaid Reading Ease Formula. https://originality.ai/blog/finding-out-about-the-flesch-kincaid-reading-ease-formula
- Kundu, A. & Barbosa, D. (2024). Are Large Language Models Good Essay Graders? arXiv 2409.13120 (Table 10, Table 33). https://arxiv.org/abs/2409.13120
- McNamara, D. S., Crossley, S. A., & McCarthy, P. M. (2010). Linguistic Features of Writing Quality. *Written Communication* 27(1), 57-86. https://journals.sagepub.com/doi/10.1177/0741088309351547
- Crossley, S. A. (2020). Linguistic features in writing quality and development: An overview. *Journal of Writing Research* 11(3), 415-443. https://www.jowr.org/jowr/article/download/582/469/441
- Vajjala, S. (2016). Automated assessment of non-native learner essays: Investigating the role of linguistic features. arXiv 1612.00729.
- Internal: reports 00, 03, 04 (§2.12), 09 (§3-4), 10 (§2), 12 (item 16), 13, 15, 16.
