# A Practitioner's Manual Humanising Method, Transcribed and Tested

Date 2026-09-07. Report 24. Source: "AI Detector Bypass - Learn to Manually Humanise AI content with me!", Dr Kriukow (qualitative-research channel), uploaded 2025-04-25, 16:53, about 534,000 views. Captions, description and metadata were pulled with yt-dlp into `.hiplora/yt/` (`transcript.txt`, 2,834 words). The before-and-after paragraph from the demonstration was reconstructed from the transcript and scored on the local detector bench.

Why this is worth a report: the video is the most-viewed manual (non-tool) humanising tutorial aimed specifically at academic writing, and it makes the same claim as report 23, that humanizer tools are useless and damage quality. Its method is a compact statement of what a skilled human editor actually does. The question is whether that method survives a supervised detector.

---

## 0. Findings up front

1. **The method is a rewrite for meaning, not a noise recipe.** The presenter's instruction is: read the paragraph, understand it, forget the exact wording, and say it in your own words, deleting anything that is "surface-level meaning". Every concrete edit in the demonstration is a re-expression or a deletion. Nothing is misspelled, no punctuation is broken, no register is dropped. That is consistent with reports 18, 19, 22 and 23 and inconsistent with the "add noise" folk theory.

2. **The theory of detection he teaches is the 2023 one.** Perplexity and burstiness are presented as "the two most commonly cited statistical metrics," followed by grammatical polish, favourite words, overused transitions, safe vocabulary, surface-level statements and corpus comparison. Report 07 established that GPTZero has been a supervised transformer since late 2023, so the mechanism he describes is not the one running. His edits, however, target features that reports 04, 10, 21 and 22 measured as real human/LLM gaps, so the practice is better than the theory.

3. **On our bench, the method does not flip a supervised detector.** His before paragraph (98 words) scores 0.9999 on desklib; his after paragraph (92 words) scores 0.9829. Fakespot 0.9999 → 0.9997; RADAR 0.944 → 0.934; fast 0.940 → 0.928. The one detector that flipped, `academic` (0.9999 → 0.0126), flipped **only because of the added citation "(Jackson, 2020)"**: with the citation removed and every other edit kept, it scores 0.9988. The demonstration's 100% → 0% result was on a free detector (detectai / aidetector.com) with an 80-word paragraph, below the length at which any detector is reliable (report 20 §7.2).

4. **The citation result is a shortcut, and a warning.** A parenthetical author-year token alone moves a RoBERTa-based academic detector from certain-AI to certain-human. That is a detector defect, not a humanising effect. Our pipeline must never add a citation that is not in the source (fabrication; `_invariant_failures` and the new-entity check from report 23 §7), and a detector that can be flipped by one token is one to hold out, not to optimise against.

5. **Two of his edits are not yet first-class in our pipeline and are cheap to add.** (a) *Sentence-opening variation*: he treats repeated opener structure ("This study…", "It is important…") and repeated "beats" (similar length and syllable count across adjacent sentences) as the single most important tell, and fixes it with introductory clauses, dependent clauses and inversion ("Low self-esteem, on the other hand, …"; "Although widely cited, this study…"; "Central to this argument is…"). Report 21 §10.1 checks 8 and 9 (share of sentences opening on The/This/It/In/There; consecutive sentences with the same first token) measure this but nothing in `transforms.py` produces it. (b) *Breaking the recurring two-item coordination* ("X and Y" as the closing move of consecutive sentences): `break_parallelism` targets tricolons and "not X but Y"; the paired coordination across sentence ends is a different template.

---

## 1. What the video teaches, in order

### 1.1 The detector model (00:00-04:06)

Eight factors, quoted or closely paraphrased from the transcript:

| # | Factor as taught | Status against our evidence |
|---|---|---|
| 1 | Perplexity: "how surprised a language model is by our text"; human is higher | Real for zero-shot detectors; GPTZero and Pangram are supervised (07, 12). Perplexity must land *in* the human band, not above it (23 §9) |
| 2 | Burstiness: variation in sentence length, structure, word choice; AI is "uniform" | Real gap, but a band not a maximum (10); at paragraph scale the human CV median is 0.38 (19 §2); reshaping CV alone flipped 3 of 30 (19 §4) |
| 3 | Grammar and polish: "too consistent", "grammatically flawless" | Real aggregate signal (GECScore 98.6% AUROC, 18 §6) but injecting errors does not exploit it (19 §4); the fix is to stop over-polishing (19 §3) |
| 4 | Favourite words, repeated phrases | Real (04, 22); `replace_ai_vocabulary` exists |
| 5 | Overused transitions: therefore, however, in conclusion | Real; strip for the grade, expect no flip by itself (2 of 30, 19 §4) |
| 6 | Safe vocabulary, "most predictable path" | Real; matches the perplexity-collapse finding (paraphrase drives PPL to 9.3 vs human 15.0, 23 §9) |
| 7 | Surface-level statements that "seem meaningful but are not" ("it is important to consider all perspectives") | Matches report 21 finding 3 (announced importance) and 22 §2 (emphasis metadiscourse at 0.52 vs 0.12 per 1,000 words) |
| 8 | Comparison against corpora of known AI and human output | This is what a supervised detector is; he is describing the actual mechanism without naming it |

His summary: "If I had to pick one, it is really about the structure. You have to introduce variation in the structure." That is the correct emphasis and the wrong metric: the local measurement says structure edits alone do not flip verdicts, while a whole-paragraph re-expression does move them somewhat (§3 below).

### 1.2 The humanising principles (04:06-08:05)

1. **Intellectual hesitation.** Replace factual absolutes with hedged claims: "can play a critical role", "it appears that", "it is suspected", "it is believed", "it is likely". Justified as normal academic practice regardless of detection. Our evidence: hedges are depleted in instruct-model text (GPT-4 at 0.00 per sentence, 18 §5) and are inside Hyland's 8-20 per 1,000 band; report 23 §7 rates hedges "copy, budgeted" and cites Van Vlasselaer's hedging prompt beating GPTZero on 39 of 40 theses.
2. **Subtle critique.** Point out an inconsistency or a competing view rather than reporting a claim flat. Not applicable to every sentence; should be present somewhere. Our evidence: the matrix's "add a genuine counterclaim and rebuttal" row, strongly positive for the grade.
3. **Variation in sentence openings and syntax.** The one he calls "very important" and applies at every sentence: introductory clauses, dependent clauses, inverted structures. The observation behind it is that AI sentences share "the same number of beats", similar length, similar syllable count and identical opener structure even when the words differ.

He explicitly excludes the blog-humanising advice (humour, anecdotes, deliberate inconsistency, open-endedness) as unsuitable for academic writing, which is the same conclusion as the matrix's "Never" rows.

### 1.3 The demonstration (08:05-16:53)

ChatGPT paragraph on self-esteem and migrants' communication, five sentences, 98 words, flagged 100% by the free tool. His edits, sentence by sentence:

| Source sentence | Edit | Edit types (report 23 taxonomy) |
|---|---|---|
| "Self-esteem plays a critical role in shaping the communicative experiences of migrants using English as a second language." | "Self-esteem **can** play a critical role … (Jackson, 2020)." | HDG (hedge); **fabricated citation** |
| "High self-esteem fosters confidence, which is essential for engaging in conversations, expressing needs, and participating in social, educational, and professional contexts." | "It has also been shown that high self-esteem can increase confidence and, as a result, migrants' willingness to participate in various conversational exchanges." | HDG; DEL of the tricolon and the second list ("too much information", "surface-level"); opener changed |
| "Conversely, low self-esteem may hinder communication by increasing anxiety, fear of judgment, and reluctance to speak, ultimately affecting language development and social integration." | "Low self-esteem, on the other hand, may affect social integration, mainly due to increased anxiety and fear of judgment." | CNI (connective moved from opener to mid-sentence); ORD (inversion); DEL of one list item and one clause |
| "Migrants who perceive their English skills as inadequate may internalize feelings of inferiority, reinforcing exclusion and marginalization." + "Therefore, supporting self-esteem is vital not only for linguistic competence but also for promoting migrants' overall well-being and inclusion." | Merged: "As low self-assessed English skills are believed to lead to an inferiority complex and exclusion, it is crucial to support self-esteem building to promote migrants' linguistic and social inclusion." | MRG (two sentences of matching shape into one longer, "maybe slightly clumsy" one); HDG ("are believed to"); ORD (fronted causal clause); CNI ("Therefore" removed); DEL of the "not only … but also" frame |

Explicit reasoning worth keeping: "what is really more important are not individual sentences, just how they are glued together"; a sentence he likes can stay "as long as the following sentence does not follow the exact same pattern"; "almost every time it lists things it is two things divided by *and*", so consecutive sentences ending in a paired coordination are merged or reshaped; "I try to literally forget what it said exactly. I just want to remember the meaning and then say it in my own words."

Net: 98 → 92 words, content overlap 0.43 with the source (below our faithful gate of 0.45 and at the evade preset's 0.30; he deleted about a third of the propositional content as "not needed"), FRE 6.9 → 33.5 (a 27-point rise, far outside report 20's ±5 gate, driven by dropping the tricolons and Latinate lists).

---

## 2. Measurement

Scored with `humanizer.detectors.local.ModernDetector` on the reconstructed paragraphs; `after_nocite` is his edited paragraph with "(Jackson, 2020)" removed and nothing else changed.

| Detector | before | after | after, citation removed |
|---|---|---|---|
| desklib (DeBERTa-v3-large, RAID-trained) | 0.9999 | 0.9829 | 0.9900 |
| fakespot (RoBERTa) | 0.9999 | 0.9997 | 0.9999 |
| academic (RoBERTa, academic domain) | 0.9999 | **0.0126** | 0.9988 |
| RADAR (Vicuna-7B) | 0.9440 | 0.9337 | 0.9729 |
| fast (e5-small LoRA) | 0.9398 | 0.9281 | 0.9308 |

Verdict flips at 0.5: **0 of 5 detectors** on the honest edit; 1 of 5 when the fabricated citation is included, and that one flips on the citation token alone. The paragraph is 92 words, which is below the reliable length for every detector, so the probabilities should be read as direction, not magnitude. Direction: the meaning-preserving re-expression moved desklib by −0.017, which is more than any single noise operator moved saturated text in report 19 (−0.001 to −0.02 at 4 events per 100 words) and far less than the ~−0.5 needed.

Caveats. The paragraphs are reconstructed from auto-captions and his on-screen typing as narrated; wording may differ slightly from what he scored. His tool is not GPTZero and the demo used one paragraph. None of that changes the shape of the result: a careful human editor's whole-paragraph rewrite lands where report 17's LLM path landed, a small probability drop and no flip, on text that scores 0.9999 to begin with.

---

## 3. What to take from it

**Keep, already in the plan.** Hedging within the Hyland band (llm.py spec 7 and the hedge budget from 23 §7); connective stripping; killing the tricolon and the "not only … but also" frame (`break_parallelism`); whole-paragraph rewrite rather than sentence patching (research 00 finding 2 and 18); deleting emphasis metadiscourse and announced importance (22 §7, 21 §10.1 check 14).

**Add, cheap.**

1. *Opener-structure variation as a rank term and a candidate spec.* Compute, per paragraph, the share of sentences whose first constituent is a bare subject NP (versus an introductory adverbial, a fronted dependent clause, a prepositional phrase or an inversion), and the number of adjacent sentence pairs sharing the same first token or the same opener class. Human targets from report 21 §10.1 checks 8-9: at most 45-60% bare-subject openers depending on band, zero consecutive same-first-token pairs in bands A-C. Add a `CandidateSpec` line: "Begin at least two sentences with a dependent clause or an inversion; never begin two consecutive sentences with the same word or the same grammatical shape." Test on the bench; expected effect on flips near zero by itself (19 §4 pattern), value is quality and de-templating.
2. *Paired-coordination template check.* Count sentences that end in an "A and B" or "A, B, and C" coordination; flag when two consecutive sentences do. The presenter's diagnosis matches report 10's template finding; no transform currently targets it.
3. *"Beats" check.* Adjacent-sentence similarity in word count and syllable count (report 10's lag-1 autocorrelation is the document-level version). Rank down candidates where three consecutive sentences are within ±3 words and ±0.1 syllables per word of each other.

**Do not copy.**

1. *The citation.* Adding "(Jackson, 2020)" is fabrication and is what produced his only real flip on our bench. Real citations from the source are already protected by the invariants gate and are the legitimate version of this move (research 00 §4, "concrete named, dated specifics").
2. *Deleting a third of the content as "surface-level".* Content overlap fell to 0.43. Some of what he deleted was empty ("participating in social, educational, and professional contexts"), some was propositional ("reluctance to speak, ultimately affecting language development"). The matrix rates "trim commentary" and "drop evidence" as very-high grade cost; the pipeline's faithful preset (0.45 overlap, ±25% length) would reject his output, correctly.
3. *The 27-point readability jump.* Report 20's gate is ±5 FRE. His rewrite went from 6.9 to 33.5 by stripping Latinate lists, which is the register drop report 23 identifies as the engine of the commercial passers and the thing a grader marks down.

## 4. Implemented (2026-09-07, same day)

The method is now first-class in the humanizer, in the form the evidence supports:

| Move from the video | Where it lives | How |
|---|---|---|
| Intellectual hesitation (hedge absolutes) | `transforms.hedge_absolutes`, in `balanced` and `strong` | Third-person claim verbs ("plays", "fosters", "leads") and "is/are + essential/vital/crucial" frames get a modal or "appears"; skips sentences with a hedge, digit or quotation; at most one sentence in three; never past Hyland's 20 per 1,000. Folds the AI-vocabulary map in ("fosters" becomes "can encourage"). |
| Vary sentence openings, "reverse the order" | `transforms.relocate_contrastive_openers`, in `balanced` and `strong` | "Conversely, X may Y" becomes "X, on the other hand, may Y"; However, In contrast, Nevertheless, Similarly are moved, never deleted. Formal connectives still go to `strip_formal_connectives`. |
| Introductory and dependent clauses where consecutive sentences open the same way ("Although widely cited, this study ...") | `transforms.front_trailing_clauses`, in `balanced` and `strong`, as its own fourth pass | Only on a sentence that repeats the opener pattern of the one before it (same first word, or the same bare-subject/pronoun class across three sentences or at the same beat). Its trailing clause headed by because / although / while / when / if / unless / until / after / before / once / since / whereas moves to the front: "The second cohort answered in April because the funding arrived late" becomes "Because the funding arrived late, the second cohort answered in April." No word is added or removed, so the module's refusal of *added* subordination (research/15: academic complexity is phrasal) is untouched. Guards: no complement heads ("is unclear if"), no pronoun-led clause (cataphora), no proper-noun subject, no internal punctuation, one sentence in four at most, never two in a row. Added 2026-09-07, second pass. |
| Same beats, repeated openers, paired "X and Y" closers, surface statements, unhedged absolutes | `kriukow.structure_report` and `structure_flags` | Measured on every candidate; reported in `quality["structure"]` and as flags; a `penalty` in [0, 1] is the third term of `pipeline._rank_key` under `quality_weight`. |
| Forget the wording, say the meaning; subtle critique; merge same-shaped sentences; delete empty clauses | `llm.STYLE_CONTRACT` rules 12-16, the "Method:" line of the user turn, and `CANDIDATE_SPECS` 0-2 | The three method specs lead the ladder so the default six-candidate `mixed` run always contains them. Rule 14 permits one limit or competing reading *the draft already implies* and forbids inventing a study or citation. |
| The fabricated citation | refused | `test_method_never_adds_a_citation_or_a_number`; the invariants gate; research/23's new-entity check. |
| Deleting a third of the content; the 27-point FRE jump | refused by the existing gates | Content overlap 0.43 is below the faithful preset and is penalised by the drift term; `test_rank_still_penalises_the_presenters_content_loss` pins that the ranker cannot learn this from his good habits. |

Also changed on the same evidence: `pipeline.HUMAN_SENTENCE_CV` moved from the document band 0.42-0.60 to the measured paragraph band 0.30-0.49 (research/19 §2). Tests: `tests/test_kriukow.py`, 34 cases, plus the existing 56 in `test_humanize.py`; the closed-vocabulary test now reads `transforms.CLOSED_VOCABULARY`, the complete set of tokens any transform may introduce. Bench results for the new rule-based and LLM paths are in `.kriukow/` and summarised in §5 below.

## 5. Bench results for the implemented method (2026-09-07, `.kriukow/`)

### 5.1 Rule path (`engine.humanize`, strong)

Nine report-17 AI paragraphs and five human PMC paragraphs, scored on desklib with fakespot, academic and RADAR held out (`.kriukow/results_rule.json`, `log_rule.txt`).

| | AI paragraphs (9) | human paragraphs (5) |
|---|---|---|
| verdict flips at 0.5 | **0 of 9** | pushed over 0.5: **0 of 5** |
| mean p(AI) | 1.0000 → 1.0000 | 0.116 → 0.136 |
| mean content overlap | 0.966 | 1.00 (two paragraphs edited) |
| structure penalty (`kriukow.structure_report`) | 0.140 → 0.122 | 0.058 → 0.056 |
| held-out flips | fakespot 0/9, academic 0/9, RADAR 0/9 (mean 0.874 → 0.857) | |

Edits landed on 7 of 9 AI paragraphs (0-5 each: connective strips, vocabulary, negative-parallel and tricolon breaks, one hedge, one opener relocation, one merge). The structure penalty fell on 5, rose on 3 (a stripped connective can expose two same-length bare-subject sentences, which the beat check then counts). The new `front_trailing_clauses` transform did not fire on this corpus: only 3 of the 61 AI sentences end in a frontable clause and none of the three sits in a repeated-opener run. On the presenter's own paragraph and on the test fixtures it fires as designed. One human paragraph (PMC8730340) moved from 0.244 to 0.336 after a split and a vocabulary swap, which is the report-19 §3 warning in miniature: editing text that already scores human can only make it look more edited.

Verdict: exactly what §0 predicted. The method as deterministic edits is a quality and de-templating layer with no measurable effect on a supervised detector.

### 5.2 The shipped LLM path, first bench through `pipeline.humanize_llm`

The pipeline's default style since the register work is `register` (Qwen3-4B-Instruct plus the round-1 LoRA), and this was the first time it was benched through the shipped code rather than the scratch harness. Result: **9 of 9 paragraphs returned unchanged** (`results_llm.json`). Every candidate, 54 of 54, failed `invariant:numbers`; most also failed `content_drift`. The adapter does what it was trained to do: anchor each sentence in a name, year or figure, and the figures are its own. Its 79 training targets carry a median of 9 numbers absent from their drafts (none carries zero) and a median content-word overlap of 0.26 with the draft, below the evade gate of 0.30. The scratch harness had reported 7 of 9 flips because it gated only on overlap and length; the shipped gates refuse fabricated evidence, and they are right to (§3 above, research/23 §7).

### 5.3 Can the adapter be told not to invent? (`results_nonum.json`)

| variant | candidates | changed | desklib flips | what failed |
|---|---|---|---|---|
| shipped prompt | 6 | 0 of 9 | 0 | numbers 54/54, drift 31/54 |
| + "use only the draft's figures; add no number, year or named study" | 6 | 0 of 9 | 0 | numbers 52/54 |
| same, 12 candidates | 12 | 1 of 9 | 0 (best 0.816) | numbers 105/108 |
| **measurement only:** numbers invariant made directional (new numbers allowed) | 6 | 7 of 9 | **5 of 9** | drift 28/54; the passers sit at overlap 0.30-0.41 |

The instruction is ignored; the LoRA's prior wins. The last row is the ceiling of the register path and it shows what the flips are made of: fabricated statistics plus a rewrite that keeps a third of the draft's content words. That is the commercial passers' recipe from research/23 and it is not shipping. The numbers, dates, quotation and citation invariants stay exact.

### 5.4 Author-supplied facts through the shipped pipeline (`results_facts.json`, `results_facts_r3.json`)

The fix that follows from §5.2 and §5.3 is architectural: the particulars must come from the author. `PipelineConfig.facts` (API `facts`, web "Facts you can vouch for") is now the only channel through which a candidate may add a number, date, quotation or citation; anything else new is refused and named in `unverified_specifics`. To bench the product scenario, each AI paragraph's facts were the number- or name-bearing sentences of its hand-written oracle rewrite, listed as notes. Eight candidates, one round, evade gates, five human PMC paragraphs with no facts as the control.

| adapter | trained on | desklib flips | fakespot / academic / RADAR | human pushed over 0.5 | mean overlap of changed paragraphs | dominant rejection |
|---|---|---|---|---|---|---|
| round 1 (shipped until now) | 79 pairs, prompt without facts | **2 of 9** | 2 / 2 / 2 | 0 of 5 | 0.31 | invented numbers (52 of 72 candidates), then drift |
| round 3 | same 79 pairs, target's particulars placed in the prompt's FACTS block | **6 of 9** | 6 / 6 / 5 | 0 of 5 | 0.34 (0.30-0.44) | content drift (32 of 72); invented numbers 2 of 72 |

Round 3 learned the lesson it was taught: with the facts in front of it, it stopped inventing (unverified specifics fell from 3-5 per paragraph to 0 on 7 of 9). What it did not learn is fidelity, because its targets have a median content overlap of 0.26 with their drafts: the surviving candidates sit at 0.30-0.44, just above the evade floor, and the three paragraphs that stayed unchanged failed on drift alone. Validation loss bottomed at step 40 of 120 (0.83) and rose to 1.27, so 72 pairs overfit quickly.

Two gate defects surfaced in the process and are fixed: the modal "may" was counted as the month, so any rewrite that added or dropped a hedge failed `invariant:dates` (8 of 8 candidates on the antibiotic paragraph); and a digit run inside a token ("CO2") counted as a number.

### 5.5 Round 4: fidelity-first targets

The remaining rejection is drift, and drift is in the training targets. Round 4 rewrites the 79 targets under the shipping gates themselves (content overlap at least 0.50, length 0.80-1.25, every particular traceable to the draft or the FACTS block, no connective openers, no AI-marked vocabulary, one limiting judgement, end on a fact), checked mechanically by `.flip/anchor/r4/check_r4.py` before training. The oracle-era targets scored 0.001-0.083 on desklib at every overlap band up to 0.41 (n = 79), so fidelity in that range costs nothing on the detector. Above it, it does: of the 79 new targets (overlap 0.51-0.84), 57 read human on desklib, 74 on fakespot and 79 on academic, and the 22 desklib failures are concentrated above overlap 0.70. Those 20 (after the gate check) were dropped and the adapter trained on the 59 passers for 100 steps (validation loss 1.77 → 0.68 at step 50 → 0.97 at 100).

Writing the targets against the gates also exposed three defects in `engine.invariants`, all fixed with tests: the modal "may" counted as the month, so a rewrite that added or dropped a hedge failed `invariant:dates`; "Uber, DoorDash and Upwork" was one entity because punctuation did not end a capitalised run; and "V2G", "R&D" and "CO2" were split at the digit or ampersand. An identity rewrite of four drafts failed its own entity gate before the fixes.

| adapter, benched through the shipped pipeline with author facts | desklib flips | fakespot / academic / RADAR | human pushed over 0.5 | mean overlap (changed) | invented specifics |
|---|---|---|---|---|---|
| round 1 | 2 of 9 | 2 / 2 / 2 | 0 of 5 | 0.31 | 3-5 per paragraph |
| round 3 | 6 of 9 | 6 / 6 / 5 | 0 of 5 | 0.34 | 2 candidates of 72 |
| **round 4 (shipped)** | **8 of 9** | **8 / 9 / 8** | 0 of 5 | **0.59** | none |

Round 4 changed all nine paragraphs (72 of 72 candidates passed the invariants on 3 paragraphs; 50 of 72 overall), and the one miss is instructive: the sleep paragraph's best surviving candidate kept 82% of the draft's content words and scored 0.958. Fidelity above about 0.7 leaves the draft's shape intact, and the shape is what the detector reads. The working band for this register is content overlap 0.45-0.70: below it the faithful gate rejects, above it the detector does.

This is the local bench. Section 6 below records what GPTZero itself said about the same outputs once a key was available.

## 6. GPTZero itself (2026-09-07, evening; key obtained, `.kriukow/gptzero_rows.json`, `data/cache/gptzero/`)

The first live GPTZero measurements in this project. Every call is cached; the whole section cost about $6.

### 6.1 The shipped pipeline (round-4 register, author facts, 8 candidates)

| | GPTZero before → after | verdict |
|---|---|---|
| 9 AI paragraphs | 1.000 → 1.000 on 7; 0.787 (microplastics); 0.427 "mixed" (antibiotic) | **0 of 9 human** |
| 5 human PMC paragraphs (no facts; 4 were rewritten) | 0.000 → 0.000-0.147 | 5 of 5 still human |

The same outputs flipped 8 of 9 on desklib, 8 on fakespot, 9 on academic and 8 on RADAR. research/08 §2.2 said a local flip carries no information about GPTZero's verdict; this is that sentence with numbers on it.

### 6.2 What GPTZero reads as human, probed directly

| text | GPTZero P(AI) |
|---|---|
| the 9 hand-written oracle rewrites of research/00 §1a (Claude-written; 0.003-0.030 on all four local detectors) | 1.000, all nine |
| 5 round-4 training targets (Claude-written, high fidelity) | 1.000, all five |
| the presenter's AI paragraph (§1.3) | 1.000 |
| **the presenter's own manual rewrite, no citation** | **0.000, human** |
| Qwen2.5-3B-**Instruct** rewrites under the style contract (12 raw candidates, 3 paragraphs) | 1.000, 12 of 12 |
| Qwen3-4B-Instruct + round-4 LoRA, raw candidates (12) | 0.885-1.000, 0 of 12 human |
| Qwen2.5-3B **base**, the shelved `freeform` style, raw candidates (12) | **6 of 12 at 0.000-0.079**, the rest 1.000 |

Three conclusions. (1) GPTZero detects LLM authorship, not register: terse, specific, sceptical prose written by a frontier model scores exactly like the draft it replaced, so the "register" lever of §1a and §5 is a local-detector artefact. (2) A human's re-expression of an AI paragraph passes at 0.000 with no citation, so the presenter's method does work on GPTZero when a human performs it; what does not work is having a model perform it. (3) research/00 finding 5 holds on our own hardware: the base checkpoint's text reads as human on GPTZero half the time; the instruction-tuned checkpoints' text never does. The local proxy is anti-correlated on base text (desklib 0.77-1.00 on candidates GPTZero scored 0.000), so for this path the inner loop cannot use desklib.

### 6.3 The base model, steered

The base candidates in 6.2 drifted (content overlap 0.01-0.33). Given three (draft → faithful rewrite) pairs in context, Qwen2.5-3B base produced, over 18 candidates on 3 paragraphs: 9 GPTZero-human, 5 gate passers, and **one candidate that did both** (explainer-2: GPTZero 0.012, overlap 0.49, 134 words). The GPTZero-human candidates that failed did so mostly on `invariant:numbers`, figures the draft lacked, which is exactly what `PipelineConfig.facts` exists to license. The next probe (§6.4) puts the author's notes into the few-shot pattern.

### 6.4 The base model with the author's notes, shipped (`gptzero_rows_v2.json`)

The `freeform` style was rebuilt around what §6.3 showed: three (draft, notes, rewrite) triples in context (`humanize/exemplars.py`), the author's facts in the final NOTES slot, stops at any label, and GPTZero itself as the candidate scorer (`HUMANIZER_PROXY=gptzero`; about half a dollar per paragraph at eight candidates). It is the pipeline default. Benched through `scripts/gptzero_bench.py`, eight candidates, two rounds, author notes on the AI paragraphs, none on the human controls:

| paragraph | GPTZero before → after | verdict |
|---|---|---|
| explainer-1 | 1.000 → 1.000 | ai |
| explainer-2 | 1.000 → 0.870 | ai |
| explainer-3 | 1.000 → 0.010 | **human** |
| microplastics | 1.000 → 0.138 | **human** |
| antibiotic | 1.000 → 0.003 | **human** |
| urban heat | 1.000 → 0.000 | **human** |
| sleep | 1.000 → 0.004 | **human** |
| supply chain | 1.000 → 0.326 | **human** |
| teaching | 1.000 → 0.001 | **human** |
| 5 human PMC paragraphs | 0.000-0.001 → 0.000 | human, 5 of 5 |

**7 of 9 AI paragraphs pass GPTZero through every meaning gate**, from 0 of 9 for the register adapter three hours earlier. Of the 54 candidates GPTZero scored in this run, 28 were human, 2 mixed, 24 AI. The two holdouts are the two hand-written stereotype paragraphs, whose content is generic enough that the base model regresses to generic prose (1 of 6 and 0 of 6 human candidates in the probe).

What separates the human-rated candidates from the AI-rated ones, over the 54 probe candidates: nothing on the surface. Sentence-length CV 0.25 against 0.29, readability 33 against 31, hedge density 11 against 12 per 1,000, AI vocabulary 0 on both, the local proxy 0.65 against 0.67. The human-rated ones are shorter (108 against 126 words) and keep less of the draft (overlap 0.41 against 0.50). GPTZero is reading something the 68 features and the four local detectors do not measure, which is consistent with §6.2: it is an authorship signal carried by the base checkpoint's sampling distribution, and the way to get more of it is more samples chosen by GPTZero, a larger base checkpoint, or a base checkpoint fine-tuned toward human originals (the HIP recipe of research/17 §A.11).

### 6.5 A larger base checkpoint, and what "mixed" means for ranking

The two holdouts and the weakest pass were re-run with `HUMANIZER_BASE_MODEL=mlx-community/Qwen2.5-7B-4bit` (same prompt, eight candidates, two rounds, GPTZero scoring):

| paragraph | 3B base | 7B base |
|---|---|---|
| explainer-1 | 1.000 (ai) | 0.005 ai, but verdict **mixed** |
| explainer-2 | 0.870 (ai) | 0.656 (ai) |
| supply chain | 0.326 (human) | 0.094 (human) |

Two lessons. The larger base checkpoint moves every paragraph, so it is now the default `freeform` model. And GPTZero's verdict is a three-way argmax: explainer-1's winner had P(ai) 0.005 and P(mixed) about 0.6, which is the AI-paraphrased class and not a pass. The pipeline had been ranking candidates on P(ai) alone and would pick exactly that candidate. `GPTZeroClient.parse` now reports `ai_probability` as 1 − P(human), so the mixed class counts against a candidate at ranking and at the threshold. The sentence-level payload carries the same signal as a `paraphrased` probability per sentence (0.22-0.26 on the supply-chain winner), which is worth a rank term later.

On the full bench, though, the 7B base did worse: 4 of 9 (microplastics 0.022, antibiotic 0.078, urban heat 0.000, sleep 0.036; explainer-3 and teaching fell back to 1.000, supply chain 0.577), with 17 of 72 candidates human against the 3B's 28 of 54. One run each, sixteen samples a paragraph, so the difference is within sampling noise; but the 3B is twice as fast and is not worse, so it stays the default. The lesson is that the pass rate per paragraph is a best-of-N over a roughly 30-50% per-candidate human rate, and the way to lift it is to lift that rate, which is what the fine-tune below is for.

Queued behind the 7B full bench: the HIP recipe (research/17 §A.11), a LoRA on the 7B base trained on (Qwen-instruct paraphrase → human PMC original) pairs built from 240 corpus paragraphs, benched through the same script (`.flip/anchor/hip/chain_hip.sh`).

### 6.6 The HIP-style fine-tune of the base checkpoint

240 human PMC paragraphs (bench paragraphs excluded) were paraphrased by Qwen2.5-3B-Instruct into polished generic prose; the pair (paraphrase → human original) in the DRAFT/HUMAN text format trained a LoRA on the 3B base (`.flip/anchor/hip/`, 216 train / 24 valid, 300 steps, lr 1e-4). Validation loss: 1.66 → 1.64 at step 200 → 2.13 at step 300, so the adapter memorises past step 200. Benched through the shipped pipeline (`HUMANIZER_BASE_ADAPTER`), same settings as §6.4:

| adapter | GPTZero-human paragraphs | notes |
|---|---|---|
| none (3B base, §6.4) | 7 of 9 | holdouts explainer-1, explainer-2 |
| HIP step 300 | 5 of 9 | explainer-2 passed for the first time (0.000); microplastics had no gate passer; sleep and supply chain 1.000 |
| HIP step 200 | 5 of 9 | explainer-2 and microplastics had no gate passer; urban heat 0.607 |

Humans 5 of 5 unharmed in every run. The per-candidate view, from the cached GPTZero responses of each run, is the more informative one:

| run | candidates GPTZero scored | human | mixed | ai |
|---|---|---|---|---|
| 3B base, shipped prompt | 217 | 97 (45%) | 6 | 114 |
| 7B base | 111 | 55 (50%) | 5 | 51 |
| 3B base + HIP step 300 | 112 | **71 (63%)** | 3 | 38 |

The fine-tune does what the HIP recipe promises: the share of candidates GPTZero reads as human rose from 45% to 63%. The paragraph count fell anyway because the adapter also learned to write like a PMC original rather than like the draft, so more of its human-rated candidates fail the content-overlap and length gates (microplastics: none of sixteen passed). The two rates pull against each other, and the product needs their product to be high. The levers left are the checkpoint (200 rather than 300), more candidates per paragraph, and a third round for paragraphs still failing.

### 6.7 Where the human candidates die: the gates, at the boundary

Three rounds of the plain base (24 candidates a paragraph) gave 5 of 9, which sampling cannot explain if a quarter of candidates were usable. Per-candidate inspection of two failing paragraphs (`.kriukow/diag_candidates.json`) shows why:

| urban heat, 8 candidates | GPTZero | length ratio | content overlap | gates |
|---|---|---|---|---|
| 0 | 1.000 | 1.02 | 0.75 | pass |
| 2 | 0.0002 | 0.60 | 0.23 | too short, drift |
| 3 | **0.0000** | 0.63 | **0.29** | drift |
| 5 | 1.000 | 1.11 | 0.36 | pass |
| 6 | 1.000 | 1.13 | 0.85 | pass |
| 7 | 0.0000 | 0.77 | 0.15 | drift, numbers ("1", "3" used twice; both in the notes) |

Six of eight were GPTZero-human; the three that passed the gates were the three GPTZero called AI. The same pattern on supply chain: human at overlap 0.21-0.36, AI at 0.47-0.81. Fidelity and GPTZero's human verdict pull against each other, and the evade floor of 0.30 sat exactly where the human-rated candidates cluster. Two changes follow. The evade preset moves to overlap 0.25 and length 0.55-1.45 (off-topic candidates still sit at 0.01-0.15, so the gate still separates them; `faithful` is unchanged). And a particular the author supplied may now be used as often as the rewrite needs; the old multiset check failed "1 to 3 degrees ... 3 degrees" against notes that said it once, which was a bug, not fidelity. Real fabrications remain refused (candidate 4 above added "8500" and "400" and stays out).

### 6.8 Consistency runs with the corrected gates (`gptzero_rows_c1.json`, `gptzero_rows_c2.json`)

Same configuration as §6.4 (3B base, three exemplar triples, author notes, eight candidates, two rounds, GPTZero scoring) with the §6.7 gate values and fact-count fix, run twice in succession:

| paragraph | run 1 | run 2 |
|---|---|---|
| explainer-1 | **0.002 human** | 1.000 ai |
| explainer-2 | **0.111 human** | 0.026 human |
| explainer-3 | 0.000 human | 0.000 human |
| microplastics | 1.000 mixed | 0.012 human |
| antibiotic | 0.000 human | 0.003 human |
| urban heat | 0.002 human | 0.000 human |
| sleep | 0.000 human | 0.001 human |
| supply chain | 0.342 human | 0.000 human |
| teaching | 0.000 human | 1.000 ai |
| **AI paragraphs human** | **8 of 9** | **7 of 9** |
| human controls unharmed | 5 of 5 | 5 of 5 |

Every AI paragraph has passed in at least one run; which one or two miss on a given run is sampling. Across the three full runs of this configuration (7, 8 and 7 of 9) the rate is 22 of 27, **81%**, against the 75-80% target, with 15 of 15 human controls still human. That is the shipped state at the end of 2026-09-07:

* `style=freeform`: Qwen2.5-3B base, three (draft, notes, rewrite) exemplar triples, the author's facts as NOTES, eight candidates, two rounds;
* GPTZero as yardstick and candidate scorer (`GPTZERO_API_KEY`, `HUMANIZER_PROXY=gptzero`), ranked on 1 − P(human);
* evade gates at overlap 0.25, length 0.55-1.45, every number, date, quotation, citation and name from the source preserved, additions only from the author's facts;
* the deterministic scrub (connectives, vocabulary, parallelism, hedging, opener relocation, clause fronting) on the winner, no dashes in any output.

What it costs: about 16 GPTZero calls a paragraph at roughly 150 words each, about $1 per paragraph, and 20-40 seconds of generation on an M4 Pro. What it does not do: make text human without the author's particulars (without notes the base model invents them and the gates refuse), or pass a paragraph whose content is so generic that every faithful candidate reads as generic (explainer-1 passes about half the time).

### 6.9 The product judges with GPTZero and nothing else (2026-09-08)

The first real use of the page exposed the last inconsistency: the headline number came from the local detector (`/api/analyze` ran desklib) while the result fold reported GPTZero, so a rewrite GPTZero called 100% human was shown under a "100% AI" banner. Given §6.2, the local number was not a second opinion but a wrong one. Since 2026-09-08 the default detector everywhere in the API is `gptzero`; `/api/analyze` takes `detect: false` so the editor can re-measure style on every pause without a paid call, and sends `detect: true` only on Humanize and on an explicit Re-measure; the candidate scorer defaults to GPTZero whenever a key is present; and without a key the page says so rather than showing a stand-in. The local checkpoints stay reachable by name for benches. The front end was rebuilt around that single number, with measurements, edits and run detail behind disclosures.

### 6.10 A free surrogate for GPTZero's verdict (2026-09-08)

The owner's rule after using the page: the in-app scan must not cost GPTZero credits, and the pipeline must work without them, because the end user will re-scan with GPTZero on their own account. The day's benches had left 1,019 texts GPTZero had scored (bench drafts, rewrites, probes; 553 human, 430 ai, 36 mixed), each cached with its `inputText`. That is a labelled set, and it was used to train the first local scorer in this project that is fitted to GPTZero rather than to RAID: RoBERTa-base sequence classifiers, label human (1) against ai-or-mixed (0), 816 train / 203 held out, four epochs at 2e-5 on MPS (`.surrogate/train.py`).

| initialisation | zero-shot agreement with GPTZero | after fine-tuning (best epoch) |
|---|---|---|
| fakespot RoBERTa detector | AUROC **0.29**, acc 0.37 (anti-correlated, as §6.2 found) | **AUROC 0.91, acc 0.82** |
| roberta-base | AUROC 0.72, acc 0.49 | AUROC 0.90, acc 0.78 |

The fakespot-initialised model is shipped as `detectors.local` "surrogate", label index 1 = human written into its config, and is now the default judge (`HUMANIZER_YARDSTICK` auto) and the default candidate scorer (`HUMANIZER_PROXY` default) whether or not a key is present; GPTZero is opt-in for benches. Caveats stated on the card: the training texts are this project's bench domain and its own rewrites, so agreement elsewhere is unmeasured; 82% is not GPTZero; and the set should be enlarged from every future bench (each one adds cached labels) and the model retrained. The pipeline with the surrogate scoring candidates was then benched once more against real GPTZero (below).

### 6.11 The surrogate as candidate scorer, and the pass threshold

With the free surrogate choosing candidates and real GPTZero judging afterwards, two runs gave 5 of 9 and 4 of 9, against 7 to 8 of 9 with GPTZero in the loop. The surrogate's own held-out numbers did not predict that: restricted to rewrites (the hard case), its "human" calls were 92% right at P(ai) < 0.5, 96% at < 0.10 and 97% at < 0.05, with 89% accuracy overall. The loss came from the selection rule, not the classifier. The pipeline took the lowest-scoring gate passer and stopped re-running a paragraph once that score fell under 0.5, so a paragraph whose best candidate sat at 0.45 was reported changed and passed, and GPTZero called it AI. Fix: `PipelineConfig.pass_threshold` = 0.15 governs both the re-run decision and the surrogate's label (the registry entry carries the same threshold, so the page's word and the pipeline's rule cannot disagree), and the default is three rounds because the scorer is free. Labels are also being collected on purpose (`.surrogate/label_run.py` scores base-model candidates and PMC paragraphs with GPTZero) and the surrogate retrained; the third training run's held-out numbers and the re-validation follow.

### 6.12 What the free scanner can and cannot do in the loop

Re-validation with the retrained surrogate (1,136 labels, held-out AUROC 0.92), pass threshold 0.15, three rounds: **3 of 9** on GPTZero, humans 5 of 5. Scoring 24 live candidates from three paragraphs with both judges explains the gap between the held-out numbers and the bench: on the pipeline's own fresh candidates the surrogate agreed with GPTZero's verdict 75% of the time, and its confident human calls (P(ai) < 0.15) were right 7 of 9 times, against 99% on the held-out split. The held-out split is optimistic because it is drawn from earlier candidates for the same nine paragraphs, many of them near-duplicates of training texts. Worked examples: antibiotic candidate 6, surrogate 0.007, GPTZero 1.000; teaching candidate 6, surrogate 0.297 (the round-one winner at the old 0.5 threshold), GPTZero 1.000.

So the free scanner is a fair preview (about three in four verdicts right on new text) and a weak selector: as best-of-N picker it lands the pipeline at roughly 3 to 5 of 9 on GPTZero, against 7 to 8 of 9 with GPTZero choosing. Two things follow. The labelled set must grow on the pipeline's own candidate distribution (`.surrogate/label_run.py`, about $7 per hundred labels) and the surrogate retrained; that is running. And the product should let a user who has their own GPTZero key put it in, so candidates are ranked by GPTZero at the user's expense and the owner's key is never spent (§6.13).

### 6.13 Two modes, priced honestly

The product now has two judging modes and says which one is running. **Free mode** (default): the local surrogate judges the page's number and ranks candidates; no GPTZero credit is spent by anyone; measured pass rate on the bench 3 to 5 of 9, and the page's verdict word is right about three times in four on new text. **Own-key mode**: the user pastes their GPTZero key under Options (`gptzero_api_key` on the LLM endpoints; kept in the browser's local storage, sent only with rewrite requests, never stored or logged by the server); GPTZero itself judges and ranks that request's candidates at roughly a dollar per paragraph on the user's credits; measured pass rate 7 to 8 of 9 (§6.8). The owner's key is used by nothing in the product path; it exists for benches and for growing the surrogate's training set.

### 6.14 Free mode after the fourth surrogate (2026-09-09)

Two labelling runs added 180 GPTZero-scored live candidates and PMC paragraphs (1,265 labels; the fourth training run had to move to CPU after the MPS allocator wired 17 GB on this set and swapped). Held-out: AUROC 0.925, accuracy 0.842. Free mode (surrogate judging and ranking, pass threshold 0.15, three rounds), judged by real GPTZero afterwards:

| paragraph | GPTZero after |
|---|---|
| explainer-1 | 0.001 human |
| explainer-2 | 1.000 ai |
| explainer-3 | 1.000 ai |
| microplastics | 0.294 human |
| antibiotic | 0.004 human |
| urban heat | 0.017 human |
| sleep | 0.004 human |
| supply chain | 0.500 human (at the boundary) |
| teaching | 0.010 human |
| **AI paragraphs human** | **7 of 9** |
| human controls | 5 of 5 unharmed |

From 3 to 5 of 9 with the earlier surrogates to 7 of 9 with this one, at zero GPTZero cost per use. The lever was labels on the pipeline's own candidate distribution, exactly as §6.12 predicted, and each further bench keeps adding them.

The consistency run gave **5 of 9** (explainer-3, microplastics, antibiotic, urban heat, sleep human; explainer-1, explainer-2, supply chain, teaching at 1.000), humans 5 of 5. So free mode with the fourth surrogate sits at 5 to 7 of 9 across two runs, a 67% average against the 75-80% target, while own-key mode sits at 7 to 8 of 9. Nine paragraphs make a noisy bench; what is not noise is the ordering: more labels on live candidates moved free mode from 3-5 to 5-7, and the next lever is the same one, plus a bounded way to retrain (the CPU run took five hours; MPS on this set swaps).

A third labelling run (103 new labels, 1,368 total) and a fifth training run (CPU, 2.5 minutes an epoch once the machine was not swapping; the MPS failures were the running web server holding about 6 GB of unified memory alongside training) gave held-out AUROC 0.933, accuracy 0.839. Free mode with it: **6 of 9** (explainer-2, explainer-3, urban heat at 0.485, sleep, supply chain, teaching human; explainer-1, microplastics, antibiotic at 1.000), humans 5 of 5. Three free-mode runs with the fourth and fifth surrogates: 7, 5, 6 of 9, a 67% average. Each labelling round adds about a hundred labels for about $9 and has lifted the floor (3 → 5 → 5-7); the next round runs with twelve candidates a paragraph in free mode, since the scorer is free and more draws raise the chance of a confidently human passer.

Round four: 72 more labels (1,452), a sixth training run whose held-out set (a fresh random split) came out lower, AUROC 0.887 and accuracy 0.793, and a free-mode bench at twelve candidates: **5 of 9** (explainer-1, microplastics, urban heat, sleep, supply chain human), humans 5 of 5. Four free-mode runs with the fourth to sixth surrogates: 7, 5, 6, 5 of 9, a 64% average. Two corrections to the training procedure follow from the drop: the validation split is now fixed by text hash so runs are comparable as the cache grows, and a new model is promoted only if it scores at least as well as the shipped one on that same split, with the replaced weights kept in a dated copy. Twelve candidates did not beat eight, which fits §6.12: more draws through a 75%-agreement selector add false positives as fast as true ones.

A caveat on the promotion rule's first use: the shipped sixth model scored 0.963 on the new fixed split only because it had been trained on a random split that included most of those texts, so the first comparison was leakage, not quality. The seventh model, trained on the fixed split, scored 0.924 cleanly and was promoted by hand; every comparison after this one is fair. Also learned: `POST /api/models/release` does not return the MLX allocator's memory to the GPU pool (about 6 GB stays "other allocations" in MPS), so the server has to be stopped, not just released, before an MPS training run.

Round five: the seventh/eighth model (clean fixed split, AUROC 0.928, accuracy 0.835, promoted by hand) gave free mode **5 of 9** (explainer-3, antibiotic, urban heat, sleep, teaching), humans 5 of 5. Five free-mode runs: 7, 5, 6, 5, 5 of 9, a 62% average. Growing the labelled set from about 1,000 to 1,450 did not move the free-mode rate, so the binding constraint is now the selection rule, not the classifier's headline accuracy. The next measurement is a candidate-level study: every candidate for the nine paragraphs scored by both the surrogate and GPTZero, so selection rules (lowest surrogate score, a confidence floor with a fidelity tiebreak, a two-model ensemble) can be compared offline before any of them ships.

**The study (`.surrogate/selection_study.json`, 71 candidates, one round of eight per paragraph).** GPTZero read 30 of 71 candidates as human (42%); the surrogate agreed with GPTZero's verdict on 77% and its live AUROC was 0.82. After the meaning gates, only 10 of the 71 were both passers and GPTZero-human, and three paragraphs (explainer-1, explainer-3, teaching) had none in eight draws; the oracle bound for that round was therefore 6 of 9. The shipped rule, lowest surrogate score among passers, picked a GPTZero-human candidate in 6 of 9 paragraphs, which is the oracle. Every stricter rule did worse because it declined to pick (a 0.05 floor with an overlap tiebreak: 1 of 9; with a shortest tiebreak: 3 of 9; random among the confident: 3 of 9). So the selector is not the bottleneck in a given round; the supply of candidates that are both faithful and GPTZero-human is, at roughly one in seven. What that leaves as levers: more rounds for paragraphs still failing (free with the surrogate, about 30 seconds a round), the HIP adapter for retries (63% human candidates but more drift), and a better base checkpoint. The surrogate's remaining job is to stop rounds at the right time, and its false "pass" calls (a candidate under 0.15 that GPTZero rates AI) are what turn a would-be retry into a miss.

Five rounds instead of three: **5 of 9** (explainer-2, explainer-3, sleep at 0.486, supply chain, teaching), humans 5 of 5; urban heat produced no gate passer in forty candidates. Six free-mode runs across the fourth to eighth surrogates: 7, 5, 6, 5, 5, 5 of 9, a 61% average. More rounds do not lift it because each extra round is stopped by the same false "pass" calls and feeds from the same one-in-seven supply. The free mode is therefore a 55-65% product on this bench, and the own-key mode a 78-89% one; the honest way to lift the free mode further is a stronger surrogate (a larger encoder on the same labels, or an ensemble of the dated copies) and a generation change that raises the faithful-and-human supply, neither of which is a threshold tweak.

### 6.15 State at the end of 2026-09-09

* Free mode: base checkpoint, three exemplar triples, author notes, eight candidates, three rounds, surrogate judge and scorer at pass threshold 0.15. GPTZero-human 5-7 of 9, no human paragraph harmed in any run, no GPTZero credit spent per use. The page shows this as "GPTZero estimate".
* Own-key mode: the same pipeline with the user's GPTZero key judging and ranking. GPTZero-human 7-8 of 9 (§6.8). About a dollar a paragraph, on the user's credits.
* Surrogate: RoBERTa-base from the fakespot detector, 1,452 GPTZero labels, fixed hash split, AUROC 0.93 held-out and 0.82 on live candidates, agreement 77%. Retrain with the server stopped or on CPU; promotion only if better on the fixed split.
* Front end: ReadsHuman (readshuman.com unregistered on 2026-09-09), graphite and amber, two always-visible panes (Before | After) with a resizable divider, ASCII-glyph interactive field, construction reveal, Fira Sans and Fira Mono, motion pinned from a CDN. Rules in `web/DESIGN.md`.

### 6.16 A self-correcting repair stage (2026-09-10, `humanize/repair.py`)

The owner's instruction, verbatim: "if the humanizer doesn't work it doesn't give any option to make it stronger or make it better. There should be an agent that's coded or a language model that's coded such that if the humanizer fails, it tries to realize what it did wrong and edit it again. It's okay if that takes more time. Specific case-by-case model this time."

§6.14 said why another round from the same distribution cannot do that: the supply of candidates that are both faithful and human is about one in seven, the failures cluster by cause (`content_drift`, `invariant:numbers`, `length_ratio`), and a few paragraphs never produce a passer. Those are different problems, so the new stage treats them differently. `repair` runs once, after the last round, between `select` and `finalize`, for every paragraph still at or above the pass threshold or kept unchanged because nothing passed the gates (a paragraph the judge already reads as human is left alone). Per attempt, per paragraph:

1. **Diagnose**, with no model call, from what the run already measured: the gate rejection counts across the paragraph's candidates, the judge scores of passers against non-passers, the particulars the rejected candidates invented (`new_specifics`), the research/24 structure flags of the best candidate, and whether facts were supplied. A cause dominates when the rejections are at least half the candidates and it accounts for at least half of them. The output is one plain sentence, e.g. "4 of 8 candidates invented figures or names not in your notes (12.5, US, 5, 9,000, 2019, 20); the 3 faithful ones read as AI (best p 0.85)".
2. **Choose one action** from the ladder for that diagnosis, never the same action twice in a row for one paragraph:

| diagnosis | actions, in order |
|---|---|
| invented specifics | restrict particulars (NOTES line naming the allowed particulars, or "no figures beyond the draft's own" with no facts; plus a deterministic rescue: a candidate that failed only on an invented number loses the sentence carrying it and is re-gated), raise fidelity, split paragraph |
| content drift | raise fidelity (NOTES "keep every claim of the draft in order; do not add new points", the two shortest exemplar triples, temperatures 0.6 to 0.85), restrict particulars, split paragraph |
| too short or too long | match length (larger token cap, NOTES "match the draft's length: about N words"), split paragraph, raise fidelity |
| passers exist but all read as AI | change distribution (the retry adapter when configured, temperatures 0.9 to 1.2, exemplar order reversed, two `register` candidates when facts are present), sentence repair (the two sentences of the best passer the judge scores highest, rewritten through the `guided` module's sentence prompts and gate, whole paragraph re-gated), split paragraph |
| nothing passes | split paragraph (halves humanized and gated against their own half, rejoined, gated whole), bridge draft (a faithful instruct draft that the base model then rewrites, gated against the original), raise fidelity |

3. **Retry** with exactly the main loop's generate, scrub, score, gate order, and **accept** the new best only if it passes every gate and beats the paragraph's current best on the judge. No action relaxes an invariant.
4. **Record** `{"attempt", "diagnosis", "action", "result": {"passed", "best", "accepted"}}` on the paragraph (`paragraphs[i].repairs`; `summary.repair_log`, `repairs_attempted`, `repairs_accepted`, `repairs_rescued`). The page shows the log under Details as "What it tried".

Attempts are round-robin over the failing paragraphs and the stage has its own soft budget (`repair_time_budget_s`, 240 s). An optional critic (`repair_critic`, off) asks the instruct checkpoint once per attempt what the best failed rewrite did wrong and what one instruction would fix it, and feeds the line into the next attempt's NOTES. In the dry run its advice was "add specific terminology and maintain a formal tone", which is the opposite of what GPTZero rewards, so it stays off and the gate-derived diagnosis does the work.

**Dry run with the stage forced (pass threshold 0, explainer-2, eight candidates, one round).** Attempt 1 diagnosed invented specifics (4 of 8) and restricted particulars: 7 passers, best 0.313 (from 0.85), accepted, three of them rescued deterministically by dropping the invented sentence. Attempt 2 diagnosed "passers all read as AI" and changed the distribution: best 0.009, accepted. Attempt 3 repaired two sentences: best 0.006, accepted. 118 s in all, including the register and instruct loads.

**Bench (free mode, surrogate judge, eight candidates, three rounds, three repair attempts, GPTZero scoring afterwards; `.kriukow/gptzero_rows_repair.json`).**

| paragraph | GPTZero after | repair attempts | what the stage did |
|---|---|---|---|
| explainer-1 | 0.323 human | 3, none accepted | drift dominated (5 of 8); raise fidelity, restrict particulars, raise fidelity all produced passers the surrogate scored 0.99; the round winner (surrogate 0.96) stood, and GPTZero read it as human anyway |
| explainer-2 | 0.003 human | 1, accepted, **rescued** | 4 of 8 invented figures not in the notes (100, 2020, IPCC Assessment, 80, 77, GHG); restrict particulars gave 6 passers, best 0.089 |
| explainer-3 | 1.000 ai | 3, one accepted | restrict particulars (no gain), raise fidelity (accepted at surrogate 0.25), change distribution (no gain); GPTZero disagreed with the surrogate |
| microplastics | 0.000 human | 0 | passed in the rounds |
| antibiotic | 0.022 human | 1, accepted, **rescued** | 3 of 8 invented figures (20, 100,000, 2020, 10, US, 9,000); restrict particulars gave 5 passers, best 0.002 |
| urban heat | 0.018 human | 0 | passed in the rounds |
| sleep | 1.000 ai | 0 | the surrogate passed a round-one winner GPTZero calls AI; the stage never saw it |
| supply chain | 0.000 human | 0 | passed in the rounds |
| teaching | 0.000 human | 0 | passed in the rounds |
| **AI paragraphs human** | **7 of 9** | 8 attempts, 3 accepted, 2 rescued | |
| human controls | 5 of 5 unharmed | 0 | two were kept unchanged and the stage correctly skipped them: they already read as human |

Against the free-mode baseline of 5 to 7 of 9 over six runs (§6.14, 61% average) this run is at the top of the range, and the two paragraphs the stage rescued are exactly the two that would otherwise have been misses: both had spent three rounds inventing particulars the notes did not contain, and one NOTES line naming the allowed particulars, plus the deterministic drop of the sentence carrying an invented figure, produced a passer GPTZero confirmed at 0.003 and 0.022. Wall clock: 35 s for a paragraph that passes in the rounds, 67 to 131 s for one that reaches the repair stage (each attempt is one eight-candidate batch, about 40 s; the register and instruct loads are paid once). No GPTZero credit is spent by the stage itself.

What the stage cannot fix is also visible. Two misses remain and both are the surrogate's, not the generator's: sleep passed the surrogate in round one so the stage never ran, and explainer-3's accepted repair scored 0.25 on the surrogate and 1.000 on GPTZero. The stage sees only the judge it is given; with the user's own GPTZero key as judge (§6.13) the same diagnoses would be scored by the yardstick itself. The third observation is that explainer-1 spent three attempts producing surrogate-0.99 passers while its standing winner, surrogate 0.96, was GPTZero-human at 0.323: the surrogate is miscalibrated on that paragraph in the safe direction, and the stage's acceptance rule (beat the current best on the judge) kept it from doing harm. The next lever, as before, is the judge: more labels on the pipeline's own candidates, now including the repair stage's, which this bench added to the cache.

**One-line summary.** The best manual method on YouTube is a fidelity-losing whole-paragraph re-expression plus hedging plus opener variation. On a supervised detector it does what our LLM path does: it moves the score and does not flip it. Its two useful contributions are the opener-variation and paired-coordination checks, which are quality features, and the reminder that the only single token that flipped a detector was a fabricated one.

**Postscript, same day, with a GPTZero key.** The presenter's own rewrite passes GPTZero at 0.000 and every LLM-written imitation of his method scores 1.000: GPTZero detects who wrote the text, not how it reads. The pipeline that finally passed 81% of the bench does so by sampling a base checkpoint, anchoring it in the author's own particulars, and letting GPTZero choose, with the meaning gates deciding what is allowed to win.

---

## Sources

- Video: https://www.youtube.com/watch?v=LDEBs9Qw1aU (Dr Kriukow, 2025-04-25). Captions and metadata in `.hiplora/yt/`.
- Recommended detector in the description: https://aidetector.com (not used here).
- Bench scores: this report §2, scored live with `src/humanizer/detectors/local.py` `ModernDetector` and the `PUBLISHED_DETECTORS` checkpoints.
- Cross-references: research/00 §1a, §4; research/17 §B; research/19 §4; research/20 §7; research/21 §10.1; research/22 §7; research/23 §7.
