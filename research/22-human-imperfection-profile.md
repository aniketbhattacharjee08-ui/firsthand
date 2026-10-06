# The Measurable Imperfection Profile of Skilled Human Writers

Date 2026-09-07. Builds on `00-SYNTHESIS.md` finding 11 ("zero errors is itself a tell"), `09` §6 (2.45 errors per 100 words, the 0.3-0.8 per 100 budget) and `16` (sample a distribution, never a mean). Those are taken as given and not re-derived. This report answers a narrower question: **what kind of noise do good human writers actually produce, at what rates, how visible is each kind to a grader, and which kinds does a supervised detector demonstrably respond to.**

Two original measurements were added because the literature does not isolate native writers. (1) The native-speaker (LOCNESS) portion of the BEA-2019 W&I+LOCNESS corpus was re-analysed from the public m2 files: 50 essays, 20,759 words, 950 ERRANT-annotated edits. (2) The project's 1,500 paired human/LLM-rewrite passages (`.hiplora/pairs.jsonl`, PMC and Wikipedia sources) were measured for repetition, hedging, punctuation and habit-consistency on both sides.

---

## Findings up front

1. **Human "imperfection" in native academic prose is mostly punctuation and orthography, not grammar.** In LOCNESS, 37% of all edits are punctuation and a further 19% are spelling or orthography (case, hyphen, spacing). Missing commas alone are 23% of every edit a native undergraduate makes. Subject-verb agreement is 0.01 per 100 words. Learner corpora invert this: C1-C2 learners run 13% determiners and 13% prepositions.
2. **Rates.** Lunsford & Lunsford: 2.45 errors per 100 words (24.5 per 1,000) in a 40-type taxonomy. ERRANT on native essays, which counts every minimal correction including all commas, gives 4.58 per 100 words; excluding punctuation it is 2.9; excluding punctuation and spelling it is 2.3. The three studies agree once the taxonomy is matched. **Just under half (47.8%) of native sentences carry no edit at all**; the rest average two.
3. **The single most common native error is a missing comma (1.06 per 100 words), and it is one of the least noticed.** Teachers marked only 28% of missing-introductory-element commas and 27% of missing nonrestrictive commas, against 54% of spelling errors and 54% of apostrophe errors. Williams's readers did not see 100 planted errors; Staub's readers detected 1 of 9 planted function-word errors in natural reading.
4. **Spelling errors in skilled writers are 84% nonword typos** ("occurence", "catagorized") that any spellchecker removes, so in word-processed text the residue is homonyms and proper nouns (Lunsford: spelling fell from the #1 error in 1986 to #5 in 2006, and "wrong word" rose to #1 at 13.7%, partly from accepted spellchecker suggestions). Nonword misspellings are therefore the wrong kind of noise for a polished document, independent of the detector question.
5. **Errors do not cluster late in a document.** Native error density by essay quintile is 4.9, 4.8, 4.1, 4.8, 4.7 per 100 words: flat. The opening sentence is the cleanest place (4.5) and the closing sentence the dirtiest (5.5, small n). **Errors cluster in long sentences**: 3.5 per 100 words in sentences under 10 words, 5.3 in sentences of 30-39 words, and missing-comma insertions per sentence rise from 0.04 to 0.65 across that range.
6. **Published academic prose is not clean either.** Hyland & Jiang's 2.2M-word research-article corpus has 40.4 "illicit" sentence-initial conjunctions and conjunctive adverbs per 10,000 words in 2015 (up 50% since 1985), contractions at 13.5 per 10,000 in applied linguistics, and split infinitives at 2.1-2.5. Published AI papers average 3.8-5.9 objective mistakes each (Bianchi et al. 2025).
7. **Detectors have stopped rewarding typos.** RAID: misspelling moves GPTZero −1.4 points, British spelling −1.6, homoglyph and whitespace −0.3. Perkins's 2024 spelling-error technique still cut seven mostly older detectors by 27 points but produced text the authors judged "very unlikely to be submitted by a student." Pangram and DAMAGE now treat inserted errors, odd spacing and hyphen removal as humanizer artefacts and detect humanized text at 90-100%.
8. **What a supervised detector does key on is regularity.** GPTZero's own definition of burstiness is "how much writing patterns and text perplexities vary over the entire document." A 2026 SHAP analysis of trained detectors found the AI texts that evaded detection had a gzip compression ratio of 0.724 against 0.535 for caught AI text, i.e. **more variability, not more errors**. RAID's repetition penalty alone cost detectors up to 32 points.
9. **In the paired human/LLM comparison, lexical repetition does not separate the two** (29.7% vs 27.5% of sentences contain a repeated content word; 61.6% vs 60.1% of adjacent sentences share a content word). What separates them is punctuation habit and metadiscourse: semicolons 3.3 vs 0.6 per 1,000 words, colons 1.2 vs 4.0, dashes 2.7 vs 4.8, "notably/importantly/it is important to note" 0.12 vs 0.52, sentence-initial And/But 0.13% vs 0.03% of sentences, and serial-comma inconsistency within a document 10% of human texts vs 5.6% of LLM texts.
10. **Grader penalty is ordinal and well documented.** Hairston's status-marking errors (nonstandard verb forms, "we was", double negatives, objective pronoun as subject) and very serious errors (fragments, run-ons, uncapitalized proper nouns, comma between verb and complement, faulty parallelism) are the ones professionals condemn. Comma placement, split infinitives, sentence-initial conjunctions and which/that are the ones they ignore. Williams's four categories formalize this: rules whose violation is noticed but whose observance is not; rules noticed neither way (most comma and usage folklore); rules whose observance is noticed (hyper-formal shall/whom).
11. **The free moves are distribution restorations, not error injection.** Restoring semicolons, cutting colons and dashes to the human rate, deleting "notably"-class metadiscourse, permitting one serial-comma or hyphenation inconsistency per couple of thousand words, and allowing 0.5-2% of sentences to start with But/And carry zero grade cost, are documented human behaviours, and target the punctuation and metadiscourse features where the paired data shows the largest human/LLM gaps.
12. **The one error class worth a small budget is the low-visibility comma** (missing after a short introductory element or around a nonrestrictive element; an occasional unnecessary comma in a compound predicate), placed in sentences of 25+ words in body paragraphs, at about a third of the human rate. Everything in Hairston's status-marking and very-serious tiers, plus wrong word, homophones and anything in a quotation, citation, thesis or first/last sentence, stays at zero.

---

## 1. Error taxonomy of skilled writers

### 1.1 The two national samples, side by side

Connors & Lunsford (1988) hand-coded 3,000 first-year papers; Lunsford & Lunsford (2008) replicated with 877 papers from a stratified national sample. Both report the twenty most frequent formal errors, the share of all errors, and the share teachers actually marked. Spelling was excluded from the 1986 list and included in 2006.

| Rank | 1986 error (Connors & Lunsford) | % of errors | % marked | 2006 error (Lunsford & Lunsford) | % of errors | % marked |
|---|---|---|---|---|---|---|
| 1 | No comma after introductory element | 11.5 | 30 | Wrong word | 13.7 | 48 |
| 2 | Vague pronoun reference | 9.8 | 32 | Missing comma after introductory element | 9.6 | 28 |
| 3 | No comma in compound sentence | 8.6 | 29 | Incomplete or missing documentation | 7.1 | 46 |
| 4 | Wrong word | 7.8 | 50 | Vague pronoun reference | 6.7 | 27 |
| 5 | No comma in nonrestrictive element | 6.5 | 31 | Spelling (incl. homonyms) | 6.5 | 54 |
| 6 | Wrong/missing inflected endings | 5.9 | 51 | Mechanical error with a quotation | 6.4 | 47 |
| 7 | Wrong or missing preposition | 5.5 | 43 | Unnecessary comma | 5.2 | 29 |
| 8 | Comma splice | 5.5 | 54 | Unnecessary or missing capitalization | 5.2 | 42 |
| 9 | Possessive apostrophe error | 5.1 | 62 | Missing word | 4.6 | 48 |
| 10 | Tense shift | 5.1 | 33 | Faulty sentence structure | 4.4 | 30 |
| 11 | Unnecessary shift in person | 4.7 | 30 | Missing comma with nonrestrictive element | 3.8 | 27 |
| 12 | Sentence fragment | 4.2 | 55 | Unnecessary shift in verb tense | 3.8 | 36 |
| 13 | Wrong tense or verb form | 3.3 | 49 | Missing comma in compound sentence | 3.6 | 37 |
| 14 | Subject-verb agreement | 3.2 | 58 | Unnecessary or missing apostrophe (incl. its/it's) | 3.1 | 54 |
| 15 | Lack of comma in series | 2.7 | 24 | Fused (run-on) sentence | 3.0 | 28 |
| 16 | Pronoun agreement error | 2.6 | 48 | Comma splice | 2.9 | 40 |
| 17 | Unnecessary comma with restrictive element | 2.4 | 34 | Lack of pronoun-antecedent agreement | 2.7 | 41 |
| 18 | Run-on or fused sentence | 2.4 | 45 | Poorly integrated quotation | 2.7 | 25 |
| 19 | Dangling or misplaced modifier | 2.0 | 29 | Unnecessary or missing hyphen | 2.5 | 27 |
| 20 | Its/it's error | 1.0 | 64 | Sentence fragment | 2.4 | 42 |

Sources: Lunsford & Lunsford 2008, Tables 2 and 7. Teachers marked 43% of coder-found errors in 1986 and 38% in 2006, concentrating on "the highly visible and easy-to-circle mistakes, such as apostrophe and spelling errors" and on "errors that confused a sentence's meaning, such as wrong words."

Three things in this table matter for us. First, **comma errors of four subtypes (introductory, compound, nonrestrictive, unnecessary) sum to 22.2% of all errors in 2006 and 29.0% in 1986**, and every one of them is marked less than a third of the time. Second, **the errors that rose between 1986 and 2006 are source-handling errors** (documentation, quotation mechanics, poorly integrated quotation: ranks 3, 6 and 18, together 16.2% of all errors), which Lunsford attributes to the shift toward research-based assignments; these cluster wherever sources are used and are the ones we must never introduce. Third, the errors that fell are exactly the ones a word processor catches: spelling (from three times any other error to 6.5%) and, plausibly, agreement and inflection (Word's grammar checker). The "wrong word" rise is partly machine-caused: "a student trying to spell 'frantic' ... accepted the spell-checker's suggestion of 'fanatic'."

Converting the 2006 shares to **per-1,000-word rates** at 24.5 total errors per 1,000 words: wrong word 3.4; missing introductory comma 2.4; documentation 1.7; vague pronoun 1.6; spelling 1.6; quotation mechanics 1.6; unnecessary comma 1.3; capitalization 1.3; missing word 1.1; faulty structure 1.1; nonrestrictive comma 0.9; tense shift 0.9; compound-sentence comma 0.9; apostrophe 0.8; fused sentence 0.7; comma splice 0.7; pronoun agreement 0.7; poorly integrated quotation 0.7; hyphen 0.6; fragment 0.6. Subject-verb agreement, dangling modifier and series comma fell off the 2006 top twenty; in 1986 they were 3.2%, 2.0% and 2.7%. Faulty parallelism is on neither list, though Hairston rates it "very serious."

### 1.2 Native undergraduates under ERRANT: an original measurement

The BEA-2019 shared-task paper reports ERRANT error-type distributions for the whole W&I+LOCNESS corpus but not for the 100 native LOCNESS essays alone (Bryant et al. 2019, Table 4). The public dev split contains 50 native essays with gold edits, so the native distribution was computed directly, alongside the C1-C2 learner split for contrast.

| Measure | LOCNESS native (N) | W&I C1-C2 learners |
|---|---|---|
| Essays / words / sentences | 50 / 20,759 / 988 | 70 / 19,220 / 1,069 |
| Edits per 100 words (all ERRANT types) | **4.58** | 5.72 |
| Excluding punctuation | 2.9 | 4.6 |
| Excluding punctuation and spelling | 2.3 | 4.4 |
| Sentences with zero edits | **47.8%** | 46.2% |
| Missing / replacement / unnecessary | 33 / 58 / 9% | 30 / 58 / 12% |

Per-100-word rates by type, native essays: **missing comma 1.06**; spelling 0.57; OTHER (multi-token rewordings) 0.43; punctuation replacement 0.33 (mostly a comma or run-on repaired to a full stop or semicolon); orthography 0.30 (half capitalization, half hyphen/spacing); preposition 0.31 (all operations); apostrophe/possessive 0.18; determiner 0.19; unnecessary comma 0.09; hyphen added 0.13; noun number 0.08; verb tense 0.06; word order 0.02; **subject-verb agreement 0.01**; comma-splice repair (comma replaced by period or semicolon) 0.03.

The learner profile is the mirror image: prepositions 0.73, determiners 0.75, verb form and tense 0.35, agreement 0.13, spelling 0.18 (they use spellcheck; natives on the LOCNESS exam-style prompts evidently did not). Native writers' errors are **overwhelmingly a punctuation-and-orthography phenomenon**: PUNCT + SPELL + ORTH = 56% of native edits versus 25% of C-level learner edits. This is also visible in Bryant's own Table 4, where W&I+LOCNESS (the only corpus with native text) has 17-19% PUNCT against 9.7% in FCE and 5.2% in NUCLE.

Two details are load-bearing for the recommendation. **84% of native spelling edits are nonwords** ("underound", "tallent", "particullary", "beaurocrats", "intergrated"), which is the signature of unspellchecked drafting, not of skilled writers' finished prose; only 16% are real-word or homophone slips. And of 354 native punctuation edits, 221 are comma insertions, 29 are deletions and 27 are hyphen insertions; the comma is the whole story.

### 1.3 What persists in polished and published prose

- **Sloan (1990)**, cited in Lunsford & Lunsford: 9.60 errors per essay and 2.04 per 100 words for freshmen, and "professional writers were prone to making errors, though the errors they made often differed significantly from those of the first-year writers."
- **Williams (1981)** planted about 100 formal errors in "The Phenomenology of Error" and catalogued rule violations in the very handbooks that prohibit them: E. B. White violating Strunk & White's rules, Barzun violating a rule "almost immediately" after stating it, Orwell writing in the passive "in the very act of criticising" it, none of which was noticed by author, editor or readers.
- **Hyland & Jiang (2017)**, 360 research articles, 2.2M words, 1965-2015: informal features run 140-199 per 10,000 words in 2015 depending on discipline. Sentence-initial conjunctions and conjunctive adverbs ("illicit initials") are 40.4 per 10,000 words overall, up 50% since 1985: initial *however* 4.8-9.4, *thus* 3.4-4.7, *but* 1.3-2.8, *and* 0.0-1.9, *so* 0.0-1.3, *yet* 0.0-2.7 per 10,000 by discipline. Contractions reach 13.5 per 10,000 in applied linguistics (1.2 in biology, 0.1 in engineering). Split infinitives are 2.1-2.5 per 10,000 in every field. Unattended *this/these* remains 44-72 per 10,000 despite a 34-45% decline.
- **Bianchi et al. (2025)** ran a GPT-5 correctness checker over top-venue AI papers: mean objective mistakes per paper rose from 3.8 (NeurIPS 2021) to 5.9 (NeurIPS 2025), 4.1 to 5.2 at ICLR, 5.0 to 5.5 at TMLR, with 83.2% precision on 316 expert-checked flags. These are formula, figure and table errors, not typos, but they establish that peer-reviewed prose is not error-free.
- **O'Neill (2018)**, 113 sets of business-student documents graded against the Hairston/Gray-Heuser list: the "vital few" errors were wordiness and missing words, comma errors excluding splices, and passive voice; comma splices sat in the next tier with apostrophes, capitalization and number style.

### 1.4 Consolidated per-type table for skilled writers

Rates are per 1,000 words. "Lunsford" is the 2006 share × 24.5; "LOCNESS" is the native ERRANT measurement above; "Published" is Hyland & Jiang or noted source.

| Error type | Lunsford 2006 | LOCNESS native | Published prose | Note |
|---|---|---|---|---|
| Spelling, nonword | (in 1.6) | 4.8 | ≈0 after copy-editing | Unspellchecked drafting artefact |
| Spelling, homophone / real word | (in 1.6) | 0.9 | rare | Survives spellcheck |
| Wrong word | 3.4 | (in OTHER 4.3) | rare | Rose with thesaurus use |
| Subject-verb agreement | <0.6 (off list) | 0.1 | ≈0 | Hairston status-marking |
| Missing comma after introductory element | 2.4 | — | not coded | Least marked (28%) |
| Missing comma, compound sentence | 0.9 | — | not coded | 37% marked |
| Missing comma, nonrestrictive | 0.9 | — | not coded | 27% marked |
| All missing commas | 4.2 | **10.6** | not coded | ERRANT is maximal |
| Unnecessary comma | 1.3 | 0.9 | not coded | 29% marked |
| Comma splice | 0.7 | 0.3 | rare | 40% marked |
| Apostrophe (incl. its/it's) | 0.8 | 1.8 | ≈0 | 54-64% marked |
| Hyphenation | 0.6 | 1.3 | copy-editor variance | 27% marked |
| Capitalization | 1.3 | 1.5 | discipline-term variance | 42% marked |
| Vague pronoun reference | 1.6 | (in OTHER) | unattended *this* 44-72/10k | 27% marked |
| Dangling / misplaced modifier | <0.6 | — | present (Williams) | 29% marked |
| Tense shift | 0.9 | 0.6 | rare | 36% marked |
| Sentence fragment | 0.6 | — | rhetorical only | 42-55% marked |
| Fused sentence / run-on | 0.7 | (in 0.33 PUNCT-R) | rare | 28-45% marked |
| Faulty parallelism | not in top 20 | — | present (Williams) | Hairston very serious |
| Sentence-initial And/But/So/Yet/Or | not an error | 1.9% of sentences | 0.4-0.9/1k (And+But) | Legitimized |
| Contractions | not coded | 1.3 | 0.01-1.35 | Discipline-dependent |

---

## 2. Non-error imperfections: what human prose has that LLM prose lacks

### 2.1 The paired measurement

The 1,500 project pairs hold content constant (each AI text is a rewrite of the human passage), so any residual gap is expressive habit rather than topic. Human passages are PMC and Wikipedia prose; the LLM side is the project's rewrite output.

| Feature | Human | LLM rewrite | Ratio | Comment |
|---|---|---|---|---|
| Sentences with a repeated content word (≥4 letters, non-stop) | 29.7% | 27.5% | 0.93 | Not a lever in paraphrase |
| Near-repeat within 3 content words | 12.1% | 12.1% | 1.00 | |
| Adjacent sentences sharing ≥1 content word | 61.6% | 60.1% | 0.98 | Cohesion preserved |
| Median within-document sentence-length CV | 0.357 | 0.347 | 0.97 | Both below the 0.42-0.60 academic band (short passages) |
| Sentence-initial And/But/So/Yet/Or | 0.13% | 0.03% | **0.23** | LLM suppresses |
| Hedges per 1,000 words (lexical list) | 4.8 | 3.9 | 0.81 | |
| Sentences with ≥2 hedges | 0.9% | 1.0% | 1.1 | No "hedge stacking" gap |
| Code glosses per 1,000 (i.e., e.g., for example, such as, namely) | 1.5 | 1.6 | 1.07 | |
| Self-correction markers (or rather, more precisely, in other words) | 0.01 | 0.00 | — | Essentially absent in **both**; edited prose does not self-correct |
| Parentheses per 1,000 | 14.8 | 15.3 | 1.03 | Preserved in paraphrase |
| Dashes per 1,000 | 2.7 | 4.8 | **1.78** | LLM adds |
| **Semicolons per 1,000** | **3.3** | **0.6** | **0.18** | Largest single gap |
| Colons per 1,000 | 1.2 | 4.0 | **3.3** | LLM converts semicolons to colons |
| Contractions per 1,000 | 0.16 | 0.08 | 0.5 | Both near zero in academic prose |
| Not only X but Y per 1,000 | 0.13 | 0.09 | 0.7 | |
| notably / importantly / it is important to note per 1,000 | 0.12 | **0.52** | **4.3** | Performed emphasis |
| Documents mixing serial and non-serial comma (of those with lists) | 48/484 = **9.9%** | 26/466 = 5.6% | 0.56 | Human inconsistency |
| Documents mixing numerals and number-words for 2-9 | 267/940 = 28% | 293/917 = 32% | 1.13 | Not discriminative |
| Documents mixing "e.g.," and "e.g. " | 0/26 | 0/4 | — | Too rare to use |

Three conclusions. **Lexical repetition is not where the gap is** once content is controlled; the popular claim that "AI avoids repeating words" describes free generation with repetition penalties, not rewriting. **Punctuation habit is where the gap is**: the semicolon is an 82% deficit, colons and dashes are surpluses, and this is exactly the feature family report 16 found most author-stable (ICC 0.44-0.63) and Sapkota et al. found most robust across genre shift. **Metadiscourse of emphasis** ("notably", "it is important to note") is the one lexical marker that is over four times denser on the LLM side even under paraphrase.

### 2.2 What the literature adds

**Repetition.** The Terčon et al. (2025) survey of 100+ studies concludes AI text is "more repetitive overall" and that "human authors tend to avoid repeating the same words and expressions" (Simon et al. 2023; Yanagita et al. 2024), with the important qualification that the *diversity of function words* is lower in AI text even where content-word diversity is higher (Zindela 2023; André et al. 2023). Shaib et al. (2024) find LLMs repeat longer part-of-speech templates than humans, that 76% of templates in model text are present in pre-training data against 35% for human text, and that RLHF does not overwrite them. McCoy et al. (2023): for n > 6 LLMs produce *more* novel n-grams than humans, for small n fewer. A 2025 JMIR study of 35 experts judging German medical-student essays against ChatGPT versions found the features that predicted correct identification were **redundancy (OR 6.90), repetition (OR 8.05) and thread/coherence (OR 6.62)**, and experts were right in 70% of rounds. So the operative human trait is not "repeats words" but "repeats a content word when it is the right word, and does not repeat structures."

**Hedging.** Jiang & Hyland (2025) report significantly fewer hedges, boosters and attitude markers in ChatGPT argumentative essays than in British student essays. A 2026 research-article comparison (IJAL 15(3); 100 human RAs, 729,535 words, vs 100 ChatGPT RAs, 233,561 words) finds the reverse for hedges, 20.55 vs 34.29 per 10,000 words, but boosters 8.99 vs 9.55, attitude markers 2.81 vs 2.35, **self-mentions 2.14 vs 0.00**, transitions 21.5 vs 43.73, frame markers 14.0 vs 26.67, code glosses 13.6 vs 26.20 and endophorics 8.3 vs 17.26. A 2026 *Modern Language Journal* paper shows baseline GPT output has lower frequencies of transitions, hedges, self-mention and attitude markers than LOCNESS students but that a metadiscourse-inducing prompt "substantially increased" them, so these frequencies are prompt-elastic. "Uncanny Semantics" (2026, 325 linguistics introductions, six models) characterizes the AI pattern as *pseudo-commitment*: syntactically assertive prose with weak epistemic modulation, while humans use "possibly, inconsistent, by no means." This is consistent with report 04's resolution: LLMs over-produce formulaic hedge phrases and under-produce first-person epistemic stance. Salager-Meyer's "compound hedges" exist in human medical prose at roughly one hedge per 100 words overall, and the paired data shows no stacking gap (0.9% vs 1.0% of sentences). **Hedge stacking is not a humanizing move.**

**Sentence-initial conjunctions.** Published rates (Hyland & Jiang 2017): *And* 0.0-1.9, *But* 1.3-2.8, *So* 0.0-1.3 per 10,000 words. At 25 words per sentence that is roughly 0.5-1.5% of sentences. Native undergraduates in LOCNESS: 1.9% of sentences begin with And/But/So/Yet/Or; C1-C2 learners 4.5%; report 10 found PERSUADE scores 1 through 6 fall from 5.1% to 2.3%. The paired PMC data shows the LLM at a quarter of the human rate. A band of 0.5-2% is documented human behaviour at every quality level.

**Inconsistency within a document.** WP:AISIGNS flags curly quotes and notes that "ChatGPT uses the Oxford comma," which is why some Wikipedia editors now avoid it; the guide is otherwise silent on typo frequency. The measured human rate of serial-comma inconsistency is about 10% of documents that contain a list (LOCNESS: 2 of 25 essays with lists mixed conventions; PMC: 48 of 484). British/American spelling mixing within a document occurred in 2 of 50 native essays (an "-ize" alongside a British "-ise"). Numeral-versus-word inconsistency is common in both human and LLM text (28% vs 32%) and is therefore useless. Lunsford documents hyphenation drift ("put-up", "log-in", "sign-up sheet" vs "sign up here") and capitalization of a paper's subject terms ("Basketball ... after Baseball and Football") as the characteristic native inconsistencies.

**Vocabulary.** Kobak et al. (2025, 15M PubMed abstracts) find 454 excess words in 2024, of which 379 are style words, 66% verbs and 14% adjectives: *delves* r = 28.0, *underscores* 13.8, *showcasing* 10.7; among common words *potential* δ = 0.052, *findings* 0.041, *crucial* 0.037; at least 13.5% of 2024 abstracts were LLM-processed, from under 5% to over 40% by field, country and journal. Liang et al. (2024) estimate 6.5-16.9% of peer-review text at ICLR/NeurIPS/CoRL/EMNLP was substantially LLM-modified. Geng & Trotta (2025) show *delve* frequency fell "soon after [it was] pointed out in early 2024" while *significant* kept rising, so the vocabulary tell is a moving target and, per Russell et al. (2025), still the cue human experts rely on most.

**Structural imperfections with no corpus measurement.** Anacoluthon, topic drift, a claim overstated and then walked back, and "a very long sentence followed by a very short one" have no published human-vs-LLM rates. Report 10's finding that lag-1 sentence-length autocorrelation in human prose is 0.01-0.10 means the long-then-short pattern is *not* a human regularity; humans do not alternate. Paragraph-length CV of 0.42-0.71 (report 10) is the only measured "unbalanced paragraphs" number. These remain gaps.

---

## 3. Visibility: which imperfections readers notice and graders penalize

### 3.1 Professional readers

Hairston (1981) mailed 65 sentences to 101 professionals in non-academic occupations (84 usable responses; three-quarters male, aged 50-60) and had them rate each as bothering them "a lot," "a little" or not at all. She grouped the results into categories that Gray & Heuser (2003) reused with a new sample, finding respondents "less bothersome than twenty years earlier." Beason (2001) interviewed 14 business people; Gubala, Larson & Melonçon (2020) surveyed about 100 and found bothersome levels *rising* again. O'Neill (2018, Table 1) reproduces the Hairston/Gray-Heuser list as a teaching handout.

| Hairston tier | Errors | Our verdict |
|---|---|---|
| **Status marking** ("outrageous") | Nonstandard verb forms (*brung*, *had went*); lack of subject-verb agreement (*we was*, *Jones don't*); double negatives; objective pronoun as subject (*Him and Richard were*) | Never |
| **Very serious** | Sentence fragments; run-on sentences; non-capitalization of proper nouns; *would of*; non-dialect agreement errors; **comma between verb and complement**; faulty parallelism; faulty adverb forms; *set/sit* | Never (fragments: ≤1 per 1,500 words, rhetorical, body only) |
| **Serious** | Verb-form errors; dangling modifiers; *I* as object (*between you and I*); lack of commas to set off interrupters; lack of commas in a series; tense switching; plural modifier with singular noun (*these kind*) | Never, except series/interrupter comma at low rate |
| **Moderately serious / minor** | Comma splices in some samples; unnecessary commas; *whose/who's*; possessive before gerund | Tiny budget |
| **Not bothered** | Split infinitives; sentence-initial *And/But*; *different than*; *which* for *that*; prepositions at end | Free |

Beason's 14 professionals ranked, most to least bothersome: **fragments, misspellings, word-ending errors, fused sentences, quotation-mark errors**. Every panel agrees on the top: agreement, nonstandard forms, fragments, run-ons and misspellings damage the writer's ethos; comma placement and usage folklore do not.

### 3.2 Teachers

Lunsford's marking percentages (Table 1 above) are the best direct measure of grader noticing. Ordered: its/it's 64% (1986), apostrophe 62/54%, subject-verb agreement 58%, fragment 55/42%, spelling 54%, comma splice 54/40%, inflected endings 51%, wrong word 50/48%, missing word 48%, quotation mechanics 47%, documentation 46%, run-on 45/28%, capitalization 42%, pronoun agreement 48/41%, compound comma 29/37%, tense shift 33/36%, unnecessary comma 34/29%, faulty structure 30%, introductory comma 30/28%, nonrestrictive comma 31/27%, hyphen 27%, vague pronoun 32/27%, poorly integrated quotation 25%, series comma 24%. **The noticing rate for comma placement is about half that for spelling and apostrophes**, across two decades and two samples.

### 3.3 Readers in natural reading

- **Staub, Chen, Peck & Taylor (2025)**: readers read full newspaper articles and clicked on any error; each article carried nine planted function-word errors (three repetitions, three omissions, three transpositions). "The median subject made seven clicks" but "detected only one of the nine inserted errors" (about 11%). Their earlier eye-tracking work (2018) put the miss rate above 30% even in proofreading mode; content-word repetitions, by contrast, are noticed regardless of eye movements.
- **Frontiers in Psychology (2023)**, 211 readers underlining naturally occurring errors in 700-800-word texts: syntactic word-order errors detected 71.0%, verb morphology 59.1%, noun-phrase morphology 55.1%, **orthographic errors 32.7%**, overall 54.5%; errors phonologically similar to the correct form and errors common in peer writing were detected least.
- **Williams (1981)**: about 100 planted errors, unnoticed by most readers "until the final sentence—which dramatically announced their presence." Lunsford: "if the piece of writing is professional prose, and if it is cognitively challenging and interesting, then readers do not notice error. The rate of error in our study, then, should also be seen as rate of attention to error."
- Proofreading baselines cited in the eye-tracking literature: nonword misspellings 85-95% detected in proofreading mode; real-word errors (*love* for *live*) about 70%.

Visibility is therefore a function of *mode*: a grader reading for argument notices roughly a third to a half of formal errors and skews to spelling, apostrophes and agreement; a copy-editor hunting errors finds most of them.

### 3.4 The cross-table

| Imperfection | Human rate (per 1,000 words unless noted) | Grader noticing / penalty | Detector sensitivity evidence |
|---|---|---|---|
| Missing comma after introductory element | 2.4 (Lunsford); all missing commas 10.6 (LOCNESS) | 28% marked; Williams "folklore" tier | None published per type; punctuation usage is a SHAP-important feature family (2026 XAI) |
| Missing nonrestrictive comma | 0.9 | 27% marked | As above |
| Unnecessary comma | 1.3 / 0.9 | 29% marked; Hairston verb-complement comma "very serious" | As above |
| Comma splice | 0.7 / 0.3 | 40-54% marked; moderately serious | None |
| Hyphenation inconsistency | 0.6 / 1.3 | 27% marked | Pangram flagged a humanizer that produced zero hyphens in 164 opportunities |
| Capitalization inconsistency | 1.3 / 1.5 | 42% marked; proper-noun failure "very serious" | RAID case swap −10 (5% of words; unreadable) |
| Serial-comma inconsistency | 10% of documents | not marked | WP:AISIGNS lists Oxford-comma constancy as a tell |
| Spelling, nonword | 4.8 (LOCNESS, unspellchecked) | 54% marked; Beason #2; 33-95% noticed by mode | RAID −1.4 GPTZero; Perkins −27 on 2024 detectors; Pangram/DAMAGE flag as humanizer |
| Homophone / wrong word | 3.4 | 48-50% marked; confuses meaning | RAID synonym −5.5 (quality-destroying) |
| Agreement | 0.1 | 58% marked; status-marking | None; unsubmittable |
| Fragment | 0.6 | 42-55% marked; very serious | None |
| Sentence-initial And/But | 0.4-0.9 published; 1.9% of student sentences | not marked | LLM at 0.23× human in paired data |
| Semicolon use | 3.3 (PMC) | positive if correct | LLM at 0.18× human in paired data |
| Colon / dash overuse | 1.2 / 2.7 | neutral | LLM 3.3× / 1.8× human |
| "Notably / it is important to note" | 0.12 | mildly negative (AP: "ineffective") | LLM 4.3× human; Russell experts' top cue is AI vocabulary |
| Lexical repetition | 30% of sentences | neutral | No gap under paraphrase; gap only in free generation |
| Hedge stacking | 0.9% of sentences | negative | No gap |
| Self-correction markers | 0.01 | neutral | Absent in both; not a human signature of edited prose |
| Contractions | 0.01-1.35 by discipline | medium in body | Reinhart: LLM 60% of human rate |

---

## 4. Detector sensitivity: what moved a 2023 perplexity detector versus what moves a 2026 supervised transformer

### 4.1 The RAID numbers, with attack parameters

Dugan et al. (2024) applied eleven attacks to 672,000 texts. Attack intensity θ was hand-tuned "to minimize suspicion from human readers": misspelling 20% of candidate words, applied only to the top-θ most likely words by GPT-2 log-likelihood and only to words in a common-misspellings dictionary; article deletion 50% of *a/an/the*; alternative (British) spelling 100%; synonym 50% via a BERT-masked procedure; upper-lower case swap 5%; whitespace 20%; homoglyph and zero-width 100%. Accuracy at 5% FPR (Table 6):

| Detector | None | Paraphrase | Synonym | Misspelling | Homoglyph | Whitespace | Delete articles |
|---|---|---|---|---|---|---|---|
| RoBERTa-L GPT-2 | 56.7 | +16.2 | +22.7 | −17.2 | −35.4 | −16.6 | −23.5 |
| RADAR (adv. trained) | 70.9 | −3.6 | −3.4 | −1.4 | −11.6 | −4.8 | −3.0 |
| GLTR | 62.6 | −15.4 | −31.4 | −2.8 | −38.3 | −16.8 | −10.5 |
| Binoculars | 79.6 | +0.7 | −36.1 | −1.6 | −41.9 | −9.5 | −5.3 |
| **GPTZero** | 66.5 | **−2.5** | **−5.5** | **−1.4** | **−0.3** | **−0.3** | **−5.5** |
| Originality | 85.0 | +11.7 | +11.5 | −6.4 | −75.7 | −0.1 | −13.6 |

Report 13 adds GPTZero's remaining deltas: alternative spelling −1.6, number swap −0.7, zero-width −0.3, upper-lower −10.0. Two further RAID findings matter more than any attack: a **repetition penalty on the generator cut detector accuracy by up to 32 points**, and metric detectors that key on token likelihood were hurt most by exactly the edits (synonyms, homoglyphs) that raise perplexity without changing style.

### 4.2 The 2023-2024 picture

- **Liang et al. (2023)**: seven detectors misclassified 61.22% of 91 human TOEFL essays; 18 of 91 were flagged by all seven; US eighth-grade essays 5.19%. Prompting ChatGPT to "enhance the word choices" cut the TOEFL false-positive rate to 11.77%; prompting it to "simplify word choices as if written by a non-native speaker" raised the eighth-grade rate to 56.65%. A second-round "elevate the provided text by employing literary language" prompt cut detection of ChatGPT essays from 100% to 13% and of abstracts from 68% to 28%. Every one of these effects ran through perplexity.
- **Perkins et al. (2024)**, 805 tests across Turnitin, GPTZero, ZeroGPT, Copyleaks, Crossplag, GPT-2 Output Detector and GPTKit: baseline accuracy 39.5% (GPTZero 26%); adversarial techniques ranked by accuracy drop, **add spelling errors −27 (to 12.9%)**, increase burstiness −24, paraphrase −21, decrease complexity −19, write as non-native −12, increase complexity −2. The authors: the spelling-error output "would be very unlikely to be submitted by a student ... they would very likely receive poor marks in a real-world setting because of the high number of errors."
- **Sadasivan et al. (2023)** recursive paraphrasing with DIPPER, Krishna et al. (2023) retrieval defence dropping to 25% after five rounds, and Chakraborty et al.'s detectability ranking are all likelihood-based results; none tests supervised transformers on error injection specifically.

### 4.3 The 2025-2026 picture

- **DAMAGE (Masrour et al. 2025)** audited 19 humanizers. Transformation modes range from close synonym replacement preserving sentence structure to full LLM rewrites; low-tier tools add "nonsensical phrases, words, and characters," hallucinated citations and in-line junk. Fluency win-rate against the original: best tier 26.0%, middle 14.7%, worst 2.7%. On academic text at 5% FPR, GPTZero fell 99.73% → 60.04% under humanization and Binoculars 94.15% → 28.23%, while a detector trained with humanized augmentation held 98.26%; on RAID's paraphrase and synonym subsets it scored 93.0% and 97.0%. The gap between GPTZero's 60% and DAMAGE's 98% is what a supervised detector learns when shown humanizer artefacts.
- **Pangram** (2025): humanizers add "extra or missing spaces and non-standard characters," "one-to-one synonym replacement," thin-space U+2009 insertions and "grammar and punctuation errors"; one humanizer produced zero hyphens across 194 texts with 164 hyphen opportunities. Detection of humanized text: 90.3% (Undetectable AI) to 100% (Quillbot, Grammarly, Ahrefs and seven others), and Russell et al. report Pangram at 97% on humanized text against GPTZero 46%, Fast-DetectGPT 23%, Binoculars 7%. Pangram's stated position: "These flags cannot be completely removed through editing ... it detects tiny markers in structure, organization, and tone, not just words."
- **Explainable-AI audit (2026, arXiv 2603.23146)**: SHAP analysis of trained detectors on 10,000 test samples found the top features were paragraph count (a formatting artefact; true positives had a median of one paragraph, true negatives seventeen) and **gzip compression ratio**, where AI texts that *evaded* detection had a mean ratio of 0.724 against 0.535 for detected AI text, "approximately 35% higher," indicating "unusually high lexical diversity or structural variability." Individual predictions "depend on combinations of stylistic features such as sentence variation, punctuation usage, and ..." The authors explicitly reject the assumption that "grammatical errors or typos serve as reliable distinguishing cues."
- **RLHF and detectability (2025, arXiv 2503.17965)**: RLHF "tends to produce more detectable, lengthy, and repetitive outputs"; training-based detectors are weakest on short and code-containing text.
- **Russell et al. (ACL 2025)**: five frequent-LLM-user annotators misclassified 1 of 300 articles by majority vote, outperforming detectors even under paraphrase and humanization; their explanations lean on AI vocabulary, then formality, originality and clarity, and in the earlier report-04 analysis 24.8% mentioned grammar or punctuation being "usually grammatically perfect."

### 4.4 Synthesis

What moved 2023 detectors was **anything that raised token surprisal**: typos, homoglyphs, rare synonyms, literary elevation. What moves 2026 supervised transformers is **distributional**: repetition-penalty-like diversity, real sentence-length variance inside the human band, punctuation habit, the absence of emphasis metadiscourse, base-model rather than RLHF fingerprint (report 13: Llama3-8B base 96.7% human vs instruct 30.3%). Injected surface errors now do three bad things at once: they barely move GPTZero (−1.4), they are precisely the artefacts Pangram and DAMAGE were trained to catch, and they are the errors graders notice most. There is no published per-type sensitivity for comma placement, hyphenation or serial-comma inconsistency on any supervised detector; the case for those rests on the punctuation-family SHAP importance and on the size of the measured human/LLM gaps, not on a direct attack study.

---

## 5. Position and provenance

### 5.1 Where human imperfections sit

Native LOCNESS essays, edits per 100 words:

| Location | Rate | Location | Rate |
|---|---|---|---|
| Quintile 1 | 4.90 | First paragraph | 4.74 |
| Quintile 2 | 4.82 | Middle paragraphs | 4.65 |
| Quintile 3 | 4.09 | Last paragraph | 4.67 |
| Quintile 4 | 4.79 | **First sentence of essay** | **4.49** |
| Quintile 5 | 4.70 | **Last sentence of essay** | **5.53** |

| Sentence length (words) | Edits / 100 words | Missing-comma insertions per sentence | n sentences |
|---|---|---|---|
| under 10 | 3.48 | 0.04 | 108 |
| 10-19 | 4.46 | 0.15 | 408 |
| 20-29 | 4.52 | 0.26 | 295 |
| 30-39 | 5.28 | 0.35 | 123 |
| 40+ | 4.95 | 0.65 | 52 |

C1-C2 learners show the same shape (first sentence 3.23, last 6.29; under-10-word sentences 4.42, 40+ words 7.03; missing commas 0.01 → 0.59 per sentence). So the folk theory that errors pile up in late paragraphs from fatigue is **not supported** for essay-length texts; density is flat across the document. Errors track **sentence length**, and the first sentence, which writers rewrite most, is the cleanest. Lunsford adds two positional regularities: source-handling errors (16.2% of all errors) occur where quotations and citations are integrated, and a class of capitalization errors arises mechanically after abbreviations ("Word automatically capitalizing a word that follows a period").

### 5.2 Uniform quality as a signal in itself

GPTZero defines burstiness as "a measure of how much writing patterns and text perplexities vary over the entire document," retained as one of "seven indicators" after the autumn-2023 move to a deep-learning verdict; its explanatory copy says humans "display inconsistent writing patterns throughout documents" while models "operate more uniformly." The 2026 SHAP audit makes the same point from the outside: compression ratio, a direct measure of within-document regularity, is the second most important feature, and the AI texts that got through were the irregular ones. Terčon et al. cite Zindela (2023) and Desaire et al. (2023) that "sentence lengths in AIGT tend to vary much less compared to HWT," and Simon et al. (2023) that LLM output "very consistently features the canonical Subject-Verb-Object ordering." Kumarage et al. (2023) built a detector around stylometric *change points* in Twitter timelines. No paper isolates "variance of per-sentence quality" as a named feature, but every component of it (sentence-length variance, punctuation variety, compressibility) is a documented discriminator. The implication is that imperfections should be distributed the way humans distribute them, flat by position and increasing with sentence length, and never back-loaded into a final paragraph, which would create a within-document quality gradient that no human corpus shows.

---

## 6. Ranked recommendation

Score = detector benefit (0-3, from the measured human/LLM gap and detector-feature evidence) × invisibility to grader (0-3, from marking rates and Hairston tiers) ÷ risk to meaning (1-3). Rates are per 1,000 words of body prose. Placement rules apply to every row: never in the thesis sentence, in or adjacent to a quotation, in a citation or reference, or in the first or last sentence of the document or of a paragraph; distribute flat across the document; prefer sentences of 25 words or more.

| Rank | Imperfection | Benefit | Invisibility | Risk | Score | Rate band / 1,000 words | Rule |
|---|---|---|---|---|---|---|---|
| 1 | **Restore semicolons; cut colons and dashes to human rate** | 3 | 3 | 1 | 9.0 | Semicolons 2-4; colons ≤1.5; dashes ≤3 | Join two related independent clauses; never a semicolon before a fragment. **Free.** |
| 2 | **Delete emphasis metadiscourse** (notably, importantly, it is important/worth noting, crucially) | 3 | 3 | 1 | 9.0 | ≤0.15 | Cut or replace with the claim itself. **Free; raises grade.** |
| 3 | **Sentence-initial But / And / So / Yet** | 2 | 3 | 1 | 6.0 | 0.5-2% of sentences (≈0.3-1.0 per 1,000 words) | Body paragraphs; not two in a row; never to open a paragraph more than once per document. **Free.** |
| 4 | **Serial-comma inconsistency** | 2 | 3 | 1 | 6.0 | 1 mixed list per 1,500-3,000 words, only in ~10% of documents | Only when the non-serial reading is unambiguous. **Free.** |
| 5 | **Hyphenation inconsistency** (open vs hyphenated compound; missing hyphen in a compound modifier) | 2 | 3 | 1 | 6.0 | 0.5-1.3 | Choose one compound that appears 3+ times; vary it once. Never in a term of art from the sources. **Free.** |
| 6 | **Keep, do not suppress, lexical repetition** (same content word in adjacent sentences when it is the right word) | 2 | 3 | 1 | 6.0 | Target 25-30% of sentences containing a repeated content word; 55-65% adjacent-sentence overlap | Do not synonym-cycle a key term; humans repeat it. **Free; raises clarity.** |
| 7 | **Missing comma after a short introductory element** ("In 2019 the court held ...") | 2 | 2 | 1 | 4.0 | 0.7-1.2 (human 2.4) | Introductory element ≤4 words; sentence ≥25 words; never where a misreading is possible. |
| 8 | **Missing comma before/after a nonrestrictive element, or a dropped comma in a compound sentence** | 2 | 2 | 1 | 4.0 | 0.3-0.6 (human 0.9 each) | Long sentences only; keep the pair symmetrical (drop both or neither). |
| 9 | **Unnecessary comma** in a compound predicate or before a short *because/that* clause | 2 | 2 | 1 | 4.0 | 0.3-0.6 (human 1.3) | Never between subject and verb or verb and complement (Hairston very serious). |
| 10 | **Capitalization inconsistency of a discipline term** (Romanticism/romanticism; the Court/the court) | 1 | 2 | 1 | 2.0 | ≤0.5 | Common noun that some style guides capitalize; never a proper noun. |
| 11 | **First-person epistemic hedge replacing a formulaic one** ("we suspect" for "it is possible that") | 2 | 2 | 2 | 2.0 | Within discipline band (report 15) | Not an imperfection; a stance shift. |
| 12 | **Comma splice** between two short, closely related independent clauses | 1 | 1 | 1 | 1.0 | ≤0.3 (human 0.7) | Both clauses under 8 words; never in an argumentative sentence. Marginal; skip if in doubt. |
| 13 | Rhetorical fragment | 1 | 1 | 2 | 0.5 | ≤0.7 (1 per 1,500 words) | Body only; a deliberate emphatic fragment, never an accidental one. |
| — | Nonword misspelling | 1 | 0 | 2 | 0 | **0** | Graders mark 54%; detectors −1.4; Pangram/DAMAGE flag. |
| — | Homophone / wrong word | 1 | 0 | 3 | 0 | **0** | Top-marked, meaning-changing. |
| — | Agreement, tense shift, pronoun-antecedent, dangling modifier, faulty parallelism, its/it's, apostrophe | 1 | 0 | 3 | 0 | **0** | Hairston status-marking / very serious; 54-64% marked. |
| — | Hedge stacking, self-correction markers ("or rather"), redundancy | 0 | 1 | 2 | 0 | **0** | No human/LLM gap; redundancy and repetition are the cues experts use *against* AI (OR 6.9-8.1). |
| — | Any edit inside a quotation, citation, thesis, or first/last sentence | — | — | — | — | **0** | Source-handling errors are 16% of student errors and 46-47% marked. |

**Total injected error budget from rows 7-12: roughly 1.6-3.3 per 1,000 words** (0.16-0.33 per 100), at the low end of report 09's 0.3-0.8 per 100 band, which is deliberate: the free rows 1-6 already move the distribution, and the paired data says punctuation habit rather than error count is where the LLM is anomalous. A document that has semicolons, one *But*-initial sentence, one open/hyphenated compound variant and two dropped low-visibility commas per 1,500 words sits inside every measured human distribution while remaining, to a grader, a carefully edited paper.

**What is essentially free** (zero grade cost, documented human behaviour, largest measured gaps): rows 1-6. **What is cheap but not free**: rows 7-10, low-visibility comma and orthography slips at a third of the human rate, in long body sentences. **What is not worth it**: everything below row 12, and every spelling or agreement error, because the detector benefit has collapsed to a point or two while grader noticing and humanizer-artefact detection have not.

---

## Gaps

- No published per-error-type sensitivity for any supervised detector (GPTZero, Pangram, Turnitin, DAMAGE) to comma placement, hyphenation, serial-comma inconsistency or sentence-initial conjunctions. The case for rows 3-9 rests on measured human/LLM gaps and SHAP feature-family importance. This is the highest-value experiment to run against the live GPTZero API in Stage 0.
- The LOCNESS measurement is 50 exam-style essays (20,759 words, one annotator), British and American undergraduates; it should be replicated on BAWE and MICUSP once acquired (report 14), which are the true "university-level" corpora and are word-processed.
- The paired human/LLM comparison uses the project's own rewrite pairs; the AI side is a paraphrase, which is the conservative case. Free-generation comparisons in the literature show larger repetition and function-word-diversity gaps than measured here.
- Hairston 1981 and Gray & Heuser 2003 full response tables are paywalled or 404 at every archive tried; tiers were reconstructed from O'Neill 2018 Table 1, Williams 1981 and secondary summaries. Sloan 1990 was read only through Lunsford's quotation.
- No corpus measurement exists for anacoluthon, topic drift, overstate-then-retract, or within-document variance of "quality" as a named detector feature. Paragraph-length CV (0.42-0.71) is the only measured "unbalanced paragraphs" number.
- Jiang & Hyland (2025) essay-level metadiscourse frequencies and Bell (2007) sentence-initial *And/But* rates were not retrievable in full; the research-article numbers from IJAL (2026) and Hyland & Jiang (2017) were used instead.

---

## 7. What to implement, at what rate, and how to verify it

Conventions. Every transform below is a pure `(Document, HumanizeConfig) -> List[Edit]` function in `src/humanizer/humanize/transforms.py`, emitting `Edit` spans that the engine resolves and gates; none mutates text. **Universal placement exclusions, enforced once in the engine, not per rule:** no edit inside or within one token of a quotation, a citation or reference entry, the thesis sentence (first paragraph's claim sentence as identified by the paragraph-template pass), the first or last sentence of the document, the first or last sentence of any paragraph, or any span containing a numeral, date, percentage, unit, or equation (this also protects commas inside numbers and hyphens in numeric ranges). Rates are per 1,000 words of eligible body prose after exclusions. Rate bands are sampled per document from the band, not set to its midpoint (report 16). Detector-evidence labels: **S** = 2025-2026 evidence involving a supervised transformer detector; **P** = 2023-2024 evidence on perplexity or likelihood detectors only; **N** = none, labelled *quality-realism only, no pass-rate claim*.

### 7.1 Per-type implementation table

| # | Type | Rate band / 1,000 w | Extra exclusions | Deterministic transform (one line) | Detector evidence | Grader visibility |
|---|---|---|---|---|---|---|
| 1 | Semicolon restoration; colon and dash normalization | Semicolons 2-4; colons ≤1.5; dashes ≤3; net punctuation count unchanged ±10% | Not before a fragment; not where either clause has an internal comma list | `restore_clause_punctuation`: for adjacent independent clauses currently split by `. ` or joined by a colon/em dash, where both sides parse as finite clauses and the second begins lowercase-able, emit an `Edit` replacing the boundary with `; ` (and for surplus colons/dashes above band, the reverse), sampling targets to hit a per-document rate drawn from the band | **S (indirect)**: punctuation usage is a top SHAP feature family in the 2026 detector audit; Desaire 2023 lists semicolons/dashes/parentheses among top human features; paired gap 0.18× (semicolons) and 3.3× (colons). No verdict-flip study | Not an error; correct semicolons carry zero penalty and are read as control (Williams category III) |
| 2 | Delete emphasis metadiscourse | ≤0.15 residual; delete 100% above that | Keep when the word is quoted or is part of a source's title | `strip_emphasis_metadiscourse`: regex-and-parse match of *notably, importantly, crucially, significantly (sentence-adverbial only), it is (important\|worth\|crucial\|essential) to note (that)?, it should be noted that*, emit a deletion `Edit` with capitalization repair of the following token | **S**: Russell et al. (ACL 2025) experts' primary cue is AI vocabulary; Kobak 2025 excess-word list; paired gap 4.3×; consistent with report 04 "strip connectives" (High benefit). No per-phrase GPTZero delta published | Zero penalty; AP withholds sophistication points for "ineffective" language, so this is grade-positive |
| 3 | Sentence-initial *But / And / So / Yet* | 0.3-1.0 (≈0.5-2% of sentences) | Never two in a row; ≤1 paragraph-opener per document; never replacing a connective that carries a logical relation the rubric scorer needs (*however*→*But* is fine; *therefore*→*So* only in body) | `informalize_connective_opener`: for sentences opening with *However,* / *Moreover,* / *Therefore,* / *In addition,* that survive `strip_formal_connectives`, replace with *But* / *And* / *So* / *And* (no comma) up to the sampled per-document count | **N — quality-realism only, no pass-rate claim.** Matrix rating L-M rests on register reasoning; paired gap 0.23× is descriptive, not a detector result | Not marked (Hairston "not bothered"; Hyland & Jiang 2017: 40.4 illicit initials per 10,000 words in published articles) |
| 4 | Serial-comma inconsistency | 1 mixed list per 1,500-3,000 words, fired in only ~10% of documents (Bernoulli p = 0.10 per document) | Only lists of ≥3 nominal items where the non-serial reading is unambiguous (no item contains *and/or*) | `vary_serial_comma`: detect the document's dominant list convention, pick one eligible list, emit an `Edit` toggling the final comma | **N — quality-realism only.** WP:AISIGNS lists Oxford-comma constancy as a tell (editorial, not a detector) | Not marked (series comma 24% marked even when *wrong*; toggling to non-serial is a valid convention, not an error) |
| 5 | Hyphenation inconsistency | 0.5-1.3 | Never in a term of art or a compound copied from a source; never in numeric ranges | `vary_compound_hyphenation`: find a compound modifier occurring ≥3 times in a consistent form (*long-term* / *long term*), toggle one non-first occurrence; skip if the compound appears in any quotation | **S (adverse direction only)**: Pangram 2025 flagged a humanizer producing 0 hyphens in 164 opportunities, so hyphen *distribution* is monitored; no evidence that mild inconsistency flips a verdict | 27% marked (Lunsford hyphen row); Lunsford documents this as a native habit |
| 6 | Retain lexical repetition | Hold 25-30% of sentences with a repeated content word; 55-65% adjacent-sentence overlap; do not add | — | Not a transform; a **guard** in `replace_ai_vocabulary` and the LLM rewrite prompt: refuse a synonym substitution when the replaced word is the document's key term (top-5 by TF-IDF) or when the substitution lowers adjacent-sentence overlap below the band | **P/S (generator-side)**: RAID 2024 repetition penalty cost up to 32 points across metric *and* supervised detector classes; JMIR 2025 experts used repetition/redundancy to *catch* AI (OR 6.9-8.1). No post-hoc-edit evidence | Zero penalty; synonym-cycling a key term is the grade risk, not repetition |
| 7 | Missing comma after a short introductory element | 0.7-1.2 (human 2.4) | Introductory element ≤4 words and not a participial phrase; host sentence ≥25 words; never where the noun after the gap could be misread as the element's object | `drop_introductory_comma`: match `^(In\|By\|At\|After\|Since\|For\|Under\|Within) [^,]{1,25}, ` at sentence start in eligible sentences, emit a one-character deletion, sampling sentences weighted by length | **N — quality-realism only, no pass-rate claim.** No supervised- or perplexity-detector study isolates comma placement | 28% marked (Lunsford 2006); Williams category II ("folklore"); Staub: function-word-level errors noticed ≈11% in natural reading |
| 8 | Missing comma around a nonrestrictive element / in a compound sentence | 0.3-0.6 each (human 0.9 each) | Nonrestrictive: drop both commas or neither; compound: only when both clauses are ≥6 words and the second has an explicit subject | `drop_nonrestrictive_or_compound_comma`: dependency-parse the eligible sentence; for `, which …,` appositive spans or `, and <subj> <verb>` coordinations emit paired/single deletions up to the sampled count | **N — quality-realism only, no pass-rate claim** | 27% (nonrestrictive) and 37% (compound) marked |
| 9 | Unnecessary comma in a compound predicate or before a short *because/that* clause | 0.3-0.6 (human 1.3) | Never between subject and verb or verb and complement (Hairston very serious); never before a restrictive *that* | `insert_predicate_comma`: in eligible sentences with `<subj> <verb> … and <verb>` (shared subject, no comma) or `… because <clause ≤6 words>`, emit a comma insertion | **N — quality-realism only, no pass-rate claim** | 29% marked (Lunsford); Hairston "moderately serious" |
| 10 | Capitalization inconsistency of a discipline common noun | ≤0.5 | Only nouns some style guides capitalize (*Romanticism/romanticism*, *the Court/the court*, *Internet/internet*); never proper nouns, never in citations or titles | `vary_term_capitalization`: from a curated allowlist, find a term used ≥3 times consistently, toggle one mid-document occurrence | **P/S (gross perturbation only)**: RAID case-swap at 5% of words moved GPTZero −10 but is unreadable; no evidence at realistic rates | 42% marked (Lunsford capitalization row): the most visible item kept on the list; lowest priority |
| 12 | Comma splice between two short clauses | ≤0.3 (human 0.7); default **off** | Both clauses <8 words; declarative; body only | `splice_short_clauses`: replace `. ` with `, ` between two adjacent eligible short sentences | **N — quality-realism only** | 40-54% marked; Hairston moderately serious. Ship disabled until the §7.4 test says otherwise |

Types 11 (first-person epistemic stance) and 13 (rhetorical fragment) from §6 are not surface transforms and are left to the LLM rewrite stage with the rate caps already in the matrix.

### 7.2 Detector-evidence summary, stated plainly

- **Evidence involving a supervised detector (2025-2026):** emphasis-metadiscourse deletion (expert-cue studies, vocabulary lists); punctuation-habit restoration (SHAP feature importance, paired gap; indirect); hyphen distribution (Pangram, adverse direction); retention of repetition (RAID repetition penalty on supervised classes; generator-side).
- **Evidence only on 2023-2024 perplexity/likelihood detectors:** none of the recommended types rest on this. The perplexity-era wins (typos, synonyms, literary elevation) are the ones this report *excludes*.
- **No detector evidence at all — quality-realism only, no pass-rate claim:** sentence-initial *But/And*, serial-comma inconsistency, every comma-slip type (7-9), comma splice. These are justified solely by matching the measured human distribution; the pipeline must not report them as contributing to pass rate until §7.4 says otherwise.

### 7.3 Per-document imperfection budget

Grader visibility is quantified by Lunsford's marking rates (probability a teacher circles a given error instance): introductory comma 0.28, nonrestrictive comma 0.27, compound comma 0.37, unnecessary comma 0.29, hyphen 0.27, capitalization 0.42, comma splice 0.40. Natural-reading noticing is lower still (Staub 2025: ≈0.11 for function-word errors). Using the marking rates as the conservative bound:

- **Injected formal slips (types 7-10, 12): expected teacher-noticed count ≤ 1 per 1,000 words.** With a mean marking rate of about 0.30 across the permitted types, that caps injected slips at **3 per 1,000 words** (0.3 per 100), and at most **4 in any single document under 1,500 words**. Combined with whatever slips the source text already carries, the LanguageTool rate (grammar and typography categories only) must land inside the report-09 band of **0.3-0.8 per 100 words** including pre-existing errors; if the input is already at 0.6, the injection budget shrinks to fill, never to exceed.
- **Distribution:** flat across paragraphs (human quintile densities are flat); at least 60% of injected slips in sentences of ≥25 words (human error density rises with length); never more than one slip per sentence; no two slips in consecutive sentences.
- **Types 1-6 are not counted against this budget.** They are not errors and carry no marking rate; they are bounded only by their own bands and by the punctuation-count and repetition guards.
- **Hard reject** if any injected edit is later flagged by GECToR with high confidence in the spelling / wrong-word / agreement / apostrophe band, or lands in an excluded span (report 09 §6.3 gate).

### 7.4 Three types to test first on the local detector bench, with expected outcomes

Bench: the Stage-0 detectors (e5-small-lora, RoBERTa-OpenAI, desklib, Fast-DetectGPT, Binoculars) plus the held-out style detector, on 300 held-out RAID academic documents, one transform at a time, reporting mean verdict-probability shift and TPR@1%FPR change, then a 100-document confirmation on the live GPTZero API.

1. **Emphasis-metadiscourse deletion (type 2).** Expected: the largest single-transform shift among the three on every supervised detector (several points of TPR at 1% FPR), near zero on Fast-DetectGPT and Binoculars, because it removes lexical tokens the supervised models weight rather than changing token surprisal. If it does *not* move the supervised detectors, the AI-vocabulary theory of detection is weaker than Russell et al. imply and the vocabulary map should be re-prioritized.
2. **Punctuation-habit restoration (type 1).** Expected: a measurable shift on the style-embedding detector and a small but consistent shift on the supervised transformers (punctuation tokens are in-vocabulary features), with essentially no change on likelihood detectors. Also expected: no change in the LanguageTool rate and no change in the rubric-derived quality score. A null result here would demote the semicolon finding from "S (indirect)" to "quality-realism only."
3. **Low-visibility comma drops (types 7-8) at 1.5 per 1,000 words.** Expected: **a null result** on every detector (|Δ| under one point), which is the point of running it first: it settles whether any error injection at grader-tolerable rates buys anything against a 2026 supervised detector. A null confirms the "quality-realism only" label and fixes the budget in §7.3 as a realism floor, not a lever. A positive result at this rate would be the first published-quality evidence that supervised detectors read punctuation slips, and would justify testing the compound-predicate comma (type 9) next. Under no outcome does spelling or agreement injection enter the test plan.

---

## Sources

- Lunsford, A. A. & Lunsford, K. J. (2008). "Mistakes Are a Fact of Life": A National Comparative Study. *CCC* 59(4): 781-806. http://www2.csudh.edu/ccauthen/575S12/Lunsford.pdf ; https://publicationsncte.org/content/journals/10.58680/ccc20086677
- Connors, R. & Lunsford, A. (1988). Frequency of Formal Errors in Current College Writing. *CCC* 39(4). Reproduced as Table 2 in Lunsford & Lunsford 2008; Dartmouth/Gocsik summary http://drwilliamdoverspike.com/files/apa_style_-_top_20_grammar_errors.pdf
- Sloan, G. (1990). Frequency of Errors in Essays by College Freshmen and by Professional Writers. *CCC* 41(3): 299-308 (via Lunsford & Lunsford 2008).
- Williams, J. M. (1981). The Phenomenology of Error. *CCC* 32(2): 152-168. https://www.cs.tufts.edu/~nr/cs257/archive/joseph-williams/phenomenology-of-error.pdf
- Bryant, C., Felice, M., Andersen, Ø. E. & Briscoe, T. (2019). The BEA-2019 Shared Task on Grammatical Error Correction. https://aclanthology.org/W19-4406.pdf ; data https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
- Hairston, M. (1981). Not All Errors Are Created Equal. *College English* 43(8): 794-806. https://publicationsncte.org/content/journals/10.58680/ce198113755 ; https://eric.ed.gov/?id=EJ254967
- Gray, L. S. & Heuser, P. (2003). Nonacademic Professionals' Perception of Usage Errors. *Journal of Basic Writing* 22(1).
- Beason, L. (2001). Ethos and Error: How Business People React to Errors. *CCC* 53(1): 33-64.
- Gubala, C., Larson, K. & Melonçon, L. (2020). Do Writing Errors Bother Professionals? *JBTC* 34(3). https://journals.sagepub.com/doi/10.1177/1050651920910205
- O'Neill, K. S. (2018). Applying the Pareto Principle to the analysis of students' errors in grammar, mechanics and style. *Research in Higher Education Journal* 34. https://files.eric.ed.gov/fulltext/EJ1178476.pdf
- Staub, A., Chen, A., Peck, E. & Taylor, N. (2025). Estimating the rate of failure to notice function word errors in natural reading. *Psychonomic Bulletin & Review*. https://link.springer.com/article/10.3758/s13423-024-02586-1
- Staub, A. et al. (2018). Failure to detect function word repetitions and omissions in reading. *PBR*. https://link.springer.com/article/10.3758/s13423-018-1492-z
- Not all grammar errors are equally noticed (2023). *Frontiers in Psychology* 14:1124227. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1124227/full
- Hyland, K. & Jiang, F. (2017). Is academic writing becoming more informal? *English for Specific Purposes* 45: 40-51. https://ueaeprints.uea.ac.uk/id/eprint/66139/1/Accepted_manuscript.pdf
- Jiang, F. & Hyland, K. (2025). Rhetorical distinctions: Comparing metadiscourse in essays by ChatGPT and students. *ESP* 79: 17-29. https://ueaeprints.uea.ac.uk/id/eprint/99123/
- Metadiscourse in ChatGPT-generated and human-written research articles (2026). *Indonesian Journal of Applied Linguistics* 15(3). https://vm113.upi.edu/index.php/ijal/article/download/101/55
- Prompt Sensitivity in AI Writing: A Methodological Caution for Corpus-Based Analysis (2026). *Modern Language Journal*. https://doi.org/10.1111/modl.70089
- Uncanny Semantics: How AI and Human Authors Use Language Differently in Academic Writing (2026). https://doi.org/10.62408/ai-ling.v5i1.32
- Reinhart, A. et al. (2024/2025). Do LLMs write like humans? Variation in grammatical and rhetorical styles. https://arxiv.org/html/2410.16107v1
- Kobak, D. et al. (2025). Delving into LLM-assisted writing in biomedical publications through excess vocabulary. *Science Advances*. https://www.science.org/doi/10.1126/sciadv.adt3813 ; https://arxiv.org/abs/2406.07016
- Liang, W. et al. (2024). Monitoring AI-Modified Content at Scale. https://arxiv.org/abs/2403.07183
- Geng, M. & Trotta, R. (2025). Human-LLM Coevolution: Evidence from Academic Writing. https://arxiv.org/abs/2502.09606
- Liang, W. et al. (2023). GPT detectors are biased against non-native English writers. https://arxiv.org/abs/2304.02819
- Terčon, L. et al. (2025). Linguistic Characteristics of AI-Generated Text: A Survey. https://arxiv.org/pdf/2510.05136
- Shaib, C. et al. (2024). Detection and Measurement of Syntactic Templates in Generated Text. https://arxiv.org/abs/2407.00211
- Shaib, C. et al. (2024). Standardizing the Measurement of Text Diversity. https://arxiv.org/abs/2403.00553
- Padmakumar, V. & He, H. (2024). Does Writing with Language Models Reduce Content Diversity? ICLR. https://arxiv.org/abs/2309.05196
- Detecting AI-Generated Versus Human-Written Medical Student Essays (2025). *JMIR*. https://doi.org/10.2196/62779
- Dugan, L. et al. (2024). RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. ACL. https://arxiv.org/pdf/2405.07940
- Perkins, M. et al. (2024). GenAI Detection Tools, Adversarial Techniques and Implications for Inclusivity in Higher Education. https://arxiv.org/abs/2403.19148
- Masrour, E. et al. (2025). DAMAGE: Detecting Adversarially Modified AI Generated Text. https://arxiv.org/abs/2501.03437
- Pangram Labs. How Students Try to Avoid AI Detection. https://www.pangram.com/blog/how-students-try-to-avoid-ai-detection ; How do AI Humanizers work? https://www.pangram.com/blog/ai-humanizers-the-slop-2-problem ; Humanizer performance, August 2025. https://www.pangram.com/blog/humanizers-aug-25
- GPTZero. What is perplexity & burstiness for AI detection? https://gptzero.me/news/perplexity-and-burstiness-what-is-it/
- Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy (2026). https://arxiv.org/pdf/2603.23146
- Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts (2025). https://arxiv.org/abs/2503.17965
- Russell, J. et al. (2025). People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text. ACL. https://aclanthology.org/2025.acl-long.267/
- Kumarage, T. et al. (2023). Stylometric Detection of AI-Generated Text in Twitter Timelines. https://arxiv.org/abs/2303.03697
- Sadasivan, V. S. et al. (2023). Can AI-Generated Text be Reliably Detected? https://arxiv.org/abs/2303.11156 ; Krishna, K. et al. (2023). Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. https://arxiv.org/abs/2303.13408
- Bianchi, F. et al. (2025). To Err Is Human: Systematic Quantification of Errors in Published AI Papers via LLM Analysis. https://arxiv.org/abs/2512.05925
- Measuring AI "Slop" in Text (2025). https://arxiv.org/html/2509.19163v1
- Wikipedia: Signs of AI writing. https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- Original measurements in this report: LOCNESS N.dev (50 essays) and W&I C.dev (70 essays) from the BEA-2019 release above; project pairs `.hiplora/pairs.jsonl` (1,500 human/LLM-rewrite pairs, PMC and Wikipedia sources). Analysis scripts were run from `/tmp` and not committed.
