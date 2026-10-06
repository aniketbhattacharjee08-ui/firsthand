# Humanizer Research Synthesis

Date 2026-09-03, updated 2026-09-07 with reports 17-23. Twenty-three reports, about 150,000 words, roughly 900 sources. Read this file first. Section 1a holds the September 7 update; the rest of the file was revised in place where the new measurements overturned it.

Product goal: **near-100% "human" verdict on GPTZero, with output that reads as excellent university-level academic writing**, in a product that does not look vibe-coded.

| # | Report | Answers |
|---|---|---|
| 01 | how-detectors-work | How GPTZero, Turnitin, Originality, Pangram and academic detectors score text |
| 02 | open-source-humanizers | What the code in the top GitHub/HuggingFace humanizer projects actually does |
| 03 | commercial-humanizers-and-evasion-papers | The 19 commercial tools and the 19 key evasion papers |
| 04 | measurable-features-human-vs-ai | Feature catalog: the signals that separate AI from human text |
| 05 | ml-training-approaches | How to train a model that learns to humanize |
| 06 | gptzero-trends-and-what-flips-it | What edits empirically flip GPTZero, and its 2023-2026 model history |
| 07 | gptzero-api-and-architecture | GPTZero's API schema, model timeline, and internal architecture |
| 08 | reliability-near-100-percent | The math of getting a 99%+ pass rate |
| 09 | college-level-writing-quality | Rubrics, automated essay scoring, and what actually earns an A |
| 10 | human-academic-writing-norms | Original measurements of five human corpora: the target distributions |
| 11 | product-and-frontend-teardown | UX teardown of 15 competitors and the front-end stack |
| 12 | anti-humanizer-countermeasures | What betrays humanized text as a third detectable class |
| 13 | gptzero-quantified-attacks | Per-attack GPTZero deltas from every published study |
| 14 | corpora-acquisition | Where to get human text, with licenses and working download paths |
| 15 | academic-register-norms-hyland-biber | Hyland and Biber discipline-level target numbers |
| 16 | authorship-variance-sample-dont-center | Why you sample a distribution instead of hitting a mean |
| 17 | evasion-code-mining | What the published attack repositories actually do, run against a 2026 detector |
| 18 | noise-as-a-lever | Whether deliberate imperfection is a viable lever, split by noise class and detector era |
| 19 | noise-injection-measured | Ten noise operators at four doses, polishing of human text, and spliced text, measured on the local detector bench |
| 20 | readability-scores | What Flesch and its relatives measure, where human and AI text land, and the target band |
| 21 | what-good-writing-sounds-like-by-readability-band | Exemplar human prose at five readability bands, close-read, with a texture spec per band |
| 22 | human-imperfection-profile | What imperfections skilled writers actually produce, at what rates, and which are grader-invisible |
| 23 | commercial-humanizer-edit-forensics | Edit-by-edit diffs of the commercial tools, and whether noise is what makes them pass |
| 24 | practitioner-manual-humanising-video | The most-viewed manual humanising tutorial for academic writing, transcribed, and its demo paragraph scored on our bench |

---

## 1. The twelve findings that should shape the design

**1. GPTZero has not been a perplexity detector since autumn 2023.** It is a proprietary supervised transformer with a hierarchical multi-task head: a document softmax over {Human, AI, Mixed} with nested {Pure AI, Polished, AI Paraphrased}, plus a jointly trained binary sentence head. Trained on about 28.6M documents. Every "raise perplexity and burstiness" prompt targets a model that no longer exists. (07, 06)

**2. The document verdict is the argmax of a calibrated three-way head, not an aggregate of sentence scores.** GPTZero's own ablation shows thresholding mean sentence probability performs *worse*. Their support docs say whole-document rewrites move the score more than sentence edits. **Sentence-by-sentence surgical repair is the wrong architecture**, and it actively destroys the sentence-score variance that burstiness measures. (07, 08, 06)

**3. Best-of-N does not follow `1-(1-p)^N`.** Two things break it. Heterogeneity: the fleet pass rate saturates at `1 − P(p(x)=0)`, so if 1.5% of inputs can never pass, no N reaches 98.5%. Correlation: at mean p=0.6 and N=8, the pass rate is 99.93% at zero intra-candidate correlation but **87% at correlation 0.5**, and decays as a *power law* in N. RLHF-tuned checkpoints are the worst possible sampling base because of documented diversity collapse. (08)

**4. Verifier quality is a hard ceiling, and "my local detector says 0.99" carries zero information.** Resampling cannot reduce verifier false positives. Only the measured joint probability that GPTZero says human given your local score exceeds a threshold matters, calibrated on your own outputs. Conformal Language Modeling is exactly this architecture with a proof. (08)

**5. Base models already read as human. This is the single largest lever in the evidence base.** GPTZero human-probability: Llama3-8B base **96.7%** versus instruct **30.3%**; Qwen3-8B base **98.8%** versus instruct **17.1%**. Iterative paraphrasing with a base model reached about **100% human on GPTZero by round 10** with a judged semantic score of 6-10 out of 10. The AI fingerprint is created by instruction tuning and RLHF, not pretraining. (13, 04)

**6. Human sentence-length variation is a band, not a maximum.** Measured coefficient of variation in human prose is **0.42 to 0.60**. A measured GPT baseline sits at about 0.50, *inside* the human band. Pushing to 0.85 is as anomalous as 0.30. Humans also do not alternate long and short: lag-1 autocorrelation is 0.01 to 0.10, and the real structure is long-range 1/f noise with exponent near 0.5. (10)

**7. Better essays are LESS bursty and MORE formal.** Length-controlled across PERSUADE scores 1 to 6: sentence-length CV falls 0.53 → 0.41, nominalization rises 17.7 → 31.2 per thousand, passives rise 4.6 → 7.5, formal connectives rise 0.84 → 3.75, sentence-initial *And/But/So* falls 5.1% → 2.3%. **Naive informalizing moves text toward a score of 2.** The standard humanizer playbook is incompatible with the quality goal. (10, 09)

**8. Academic complexity is phrasal, not clausal.** Most clausal subordination is *more* common in conversation than in research articles. There are about twice as many dependent clauses in conversation as in academic writing. Academic prose gets its density from noun phrases: attributive adjectives, noun-noun premodification, and prepositional postmodifiers at about 68 per 1,000 words. **A humanizer that adds subordinate clauses to sound scholarly is moving toward conversation.** (15, 09)

**9. Detector-only rewards produce detector-shaped text.** A model RL-trained to push Fast-DetectGPT to 0.17 AUROC is still caught at **0.95 AUROC by a style-embedding detector**. Any learned humanizer needs a style and feature-distribution term in the reward. (05)

**10. "Humanized" is a third detectable class, and content origin is separable from expression origin.** GPTZero ships an AI-Paraphrased class; Pangram 4 has a dedicated humanizer head at 96.82% accuracy. Worse, content origin is detectable independently of expression origin, so **expression-only rewriting is asymptotically detectable** no matter how good the rewriter. A humanizer is a fixed transform, so its tics repeat across every document, which is easier to learn than an LLM fingerprint. (12)

**11. Zero errors is itself a tell.** Real student papers average **2.45 errors per 100 words**, a rate stable since 1917. Rubrics weight mechanics at only 5-25%, and AP English gates on whether errors "interfere with communication," not on error count. (09)

**12. Sample a distribution, never target a mean.** Within one genre, the across-author spread on function-word features is 0.8 to 1.5 times the spread between six radically different genres. A text at the exact genre mean sits in an atypically empty region. Measured intraclass correlations across 1,833 texts from 130 authors: no feature exceeds 0.63, punctuation habits are most author-stable (question marks 0.63, em dashes 0.52, commas 0.44), while hedges, passives and lexical diversity sit at 0.07-0.20. (16, 10)

---

## 1a. The September 7 round: is our output too clean, and would noise fix it?

The question asked was whether the pipeline fails because it maintains too high a writing quality, so that deliberately introducing noise would work better than polishing; and, separately, what the Flesch Reading Ease score measures and what good human writing sounds like at a given score. Reports 18-23 answer both. The one-line answer: **"too clean" is a real signal, but it is caused by the instruct model and cannot be undone afterwards by roughing up the surface.** Post-hoc noise flips verdicts only on text the detector was already unsure about, and the passing commercial tools do not pass because of noise. The findings, continuing the numbering of section 1:

**13. Clean text is more detectable, and the local bench confirms it in the reverse direction.** A detector that only counts grammar-correction edits reaches 98.6% AUROC; human text needs 10-30 edits per sample, LLM text 0-10. Detector scores correlate with lexical formality at ρ ≈ 0.30-0.35 on GPTZero and Pangram. Locally, a maximally conservative "fix only what is wrong" polish of genuine human paragraphs through Qwen2.5-7B-Instruct changed 11% of words and flipped **11 of 33** human paragraphs to AI on desklib (13 on fakespot); a 3B model at 25% changed flipped 16 of 33. The edits that did it were a copy editor's: an inserted article, a regularised compound, a topic-sentence opener. Pure grammar correction at 2% of words flipped 2 of 33. **Safe cleaning budget: about 2% of words, and never an instruct-model pass on text that already scores human.** (18 §6, 19 §3)

**14. Injecting noise into finished AI text does not move a supervised detector on the text we actually produce.** Fourteen operators at 0.5-4 events per 100 words on a saturated corpus (Llama-3-8B paraphrases, desklib 0.997, 40 of 40 flagged): 0-2 flips of 40 on desklib, 0-1 on fakespot, for typos, casing, function-word deletion, WordNet synonyms, punctuation slips, hedges, conjunction insertion, redundancy, connective removal, sentence shuffle and CV reshaping. Only the RAID casing attack at 20 swaps per 100 words, which is unreadable, reached 10 of 40. The literature agrees: synonym swap moved RoBERTa-Large **+22.7 toward AI**, keystroke-realistic typos left DeBERTa at 96-112% of baseline AUROC, and Pangram's DAMAGE holds 98.3% on typo-laden humanizer output. Noise flips *boundary* text (a restatement corpus at desklib 0.731: 6-13 of 30 for the visible operators) exactly to the degree that it puts tokens the detector has never seen in AI text; the one operator that substitutes *plausible* words, masked-LM neighbours, flipped 1 of 30 at every dose. Our pipeline's outputs score 0.9996-1.0000 and behave like the saturated corpus. (19 §4, 18 §3)

**15. What the passing commercial humanizers do is distribution shift, not noise, and they buy it with quality we cannot spend.** The noisiest tool (StealthWriter, 4.2 injected errors per 1,000 words) is 100% AI on GPTZero in four 2026 tests. The best passer (Undetectable AI, 1.8 errors per 1,000) passes at over 94% by moving the lexical distribution away from instruct-model academic English: long-token ratio −25%, Academic Word List ratio −23%, type-token ratio −13%, words per sentence +20%, length +27%, register scored 3 of 5, plus 1-2 meaning changes per 100 words. Fixing the noisy tools' grammar moves ZeroGPT from 58% to 100% AI while GPTZero stays at 100% both ways. Noise clears perplexity detectors; it never cleared GPTZero 3.15b or later, and GPTZero's paraphrase class and Pangram's artifact list are trained on precisely these tics. Under the hood the passers are fine-tuned or prompted generative rewriters multiplied by detector-in-the-loop selection. That is the architecture in section 6, minus the fidelity and register constraints they refuse to carry. (23 §4-7)

**16. The one noise with strong evidence is distributional noise at generation time, and it has a ceiling.** Sampling with a repetition penalty took GPTZero from 98.8% to 82.5% accuracy on chat models and 74.7% to 4.8% on base models (RAID, verdict-level). But a RoBERTa trained on a mixture of decoding settings recovers to ≥0.95 on all 37 configurations, temperature above 1.0 produces text with lexical diversity 4-7× human, and locally seven sampler settings on an instruct model left 28-30 of 30 paragraphs flagged, with `min_p` making it worse. The durable part of the entropy lever is the base checkpoint's distribution, not the sampler. (18 §2, 19 §5)

**17. Skilled human imperfection is punctuation habit, not error, and the free moves are distribution restorations.** In native-speaker undergraduate essays (LOCNESS, 20,759 words re-analysed), 56% of all edits are punctuation and orthography; the single largest item is the missing comma at 1.06 per 100 words, which teachers mark only 27-28% of the time against 54% for spelling. Spelling errors in skilled writers are 84% nonword typos that a spellchecker removes, so they are the wrong noise for polished prose independent of the detector question. Errors are flat across a document and cluster in long sentences; the first sentence is the cleanest. In 1,500 paired human/LLM passages, lexical repetition does *not* separate the two, but punctuation and metadiscourse do: semicolons 3.3 vs 0.6 per 1,000 words, colons 1.2 vs 4.0, "notably / it is important to note" 0.12 vs 0.52, sentence-initial But/And 0.13% vs 0.03% of sentences. Six moves cost nothing at the grade and target the largest measured gaps: restore semicolons, cut colons and dashes, delete emphasis metadiscourse, allow sentence-initial But, permit one serial-comma or hyphenation inconsistency, keep key-term repetition. Only metadiscourse deletion and punctuation restoration have any 2025-26 supervised-detector evidence; every comma-slip type is quality-realism only, with a total budget of 1.6-3.3 per 1,000 words and an expected null on the bench. (22)

**18. The mixing asymmetry decides the unit of work.** One matched AI sentence in a human paragraph (17% of words) flipped 10 of 33 human paragraphs to AI. Half a paragraph of human sentences in an AI paragraph flipped 5 of 30; three human sentences in saturated text flipped 1 of 40. AI signal dominates at every ratio. This reinforces finding 2 from the other direction: the whole paragraph has to be human-shaped, and any pipeline that edits, replaces or regenerates sentences inside a paragraph is on the losing side. (19 §6)

**19. The sentence-length band in section 5 was measured at the wrong scale.** On 1,264 human PMC paragraphs the coefficient of variation has a median of 0.38 and an interquartile range of 0.30-0.49; a quarter of genuine published paragraphs sit below 0.30. The 0.42-0.60 band is a document-level number. `HUMAN_SENTENCE_CV` in the pipeline scores paragraphs, so it currently asks for something 60% of human paragraphs do not have, and moving text into it flipped 3 of 30. CV is also a feature of *generation*, not of AI-ness: faithful restatements inherit the source's sentence plan (0.37 vs 0.38), while free paraphrases do not (0.29 vs 0.44). (19 §2)

**20. Readability runs the wrong way for the too-clean story, and graders do not care about it.** Flesch Reading Ease is 83:1 a syllables-per-word meter. Instruct-model output is *harder* than matched human text by 6-25 FRE points everywhere it has been measured, and the difficulty is lexical: GPT-4 writes shorter T-units at grade 16.6 against students' 12.1. Locally, Qwen restatements sit 6 FRE points below their PMC sources and Llama paraphrases 14 below. Human academic prose sits at FRE 10-40, public prose at 50-65. FRE correlates with essay score at r = −0.17 to +0.13, carries under 0.5% marginal value in detector ablations, and "write like a human" prompts lower it by 5-14 points while failing to evade. Implementations disagree by up to 8%, and `textstat` silently drops sentences under three words, which erases exactly the short-sentence tail. Gate the input-to-output delta at |ΔFRE| ≤ 5 and |ΔFKGL| ≤ 1 inside a wide per-genre band; never put readability in a reward. (20)

**21. Part of the local human baseline is extraction residue.** Removing curly quotes, en dashes and stray pre-punctuation spaces from PMC paragraphs raised desklib by +0.092 on the 21 of 40 paragraphs that had them; one went 0.237 → 0.674 on three curly apostrophes. Desklib has learned that typographic punctuation is human. This is a caveat on report 17's PMC baseline, a free rule for the pipeline (never normalise typographic punctuation out; emit it in generated text), and a warning that some of the human/AI separation on this corpus is typographic rather than authorial. (19 §3)

**22. At matched readability, what separates good human prose from AI is content first and shape second, and neither is on the FRE axis.** Forty-six skilled human passages (Darwin, Hume, Holmes, Woolf, Turing, Orwell, Einstein, Watson and Crick, MICUSP papers) and twelve Qwen2.5-7B-Instruct passages on matched topics sit within a few FRE points of each other in every band. Ranked as a detector would weight them, the differences are: (1) first-hand specifics that the writer alone could supply (human passages run a median near 30 proper nouns per 1,000 words, AI passages 0-29 except when hallucinating), which no rewrite can add; (2) an inferential turn versus a claim restated; (3) announced importance and narrated feeling ("opens the door to a deeper understanding", "my heart pounded"); (4) uniform sentence length with no verdict sentence after the long one; (5) closing on summary or resolution (11 of 12 AI passages; 0 of 46 human); (6) connective openers where a human uses a demonstrative or repeats the subject (1.3 per 1,000 words in skilled prose, 1.0 per 200 words in AI); (7) register lexis. Only items 4-7 are what humanizers fix. Two structural facts matter for the pipeline. Human burstiness is *placed*: one or two very long sentences carry the argument and the short sentence that follows is a verdict on it ("But this is absurd."). And CV rises with FRE in good prose: the dense bands have CV 0.18-0.45 built from coordinated noun-phrase lists, the essay bands 0.40-0.90 built from alternation, so a single "human shape" inferred from FRE is wrong for one of them. The instruct model cannot hold a band at all: asked for a "serious literary essayist" it produced FRE 11, asked for "plain words" FRE 83, against Orwell's actual 50. Six moves the model reliably fails at under prompting (closing discipline, short-after-long placement, withholding importance, withholding feeling, holding a band, repeating a key word instead of varying it) have to be deterministic gates or transforms. Report 21 §10 gives 22 per-band checks with thresholds and 5-8 prompt lines per band. (21)

**23. The best manual method in circulation lands where our LLM path lands.** The most-viewed manual humanising tutorial for academic writing (Dr Kriukow, 534,000 views) teaches hedging, subtle critique, sentence-opening variation, merging same-shaped sentences and deleting "surface-level" content, then restating the meaning in one's own words. Its demonstration paragraph, reconstructed and scored here, moved desklib from 0.9999 to 0.983 with 0 of 5 detectors flipped. The one detector that flipped did so on a fabricated citation "(Jackson, 2020)" alone: remove it and that detector returns to 0.999. The rewrite also cut content overlap to 0.43 and raised FRE by 27 points, both outside our gates. Two of its checks are worth adding as quality features: opener-structure variation and the recurring paired-coordination template. (24)

**24. The register adapter flips the local detectors by inventing particulars, and the shipping gates correctly refuse every one of them.** The `register` style (Qwen3-4B-Instruct plus the round-1 LoRA, finding 1a's only cross-detector mover) was benched through the *shipped* pipeline for the first time on 2026-09-07: 9 of 9 paragraphs came back unchanged. Every candidate (54 of 54) failed `invariant:numbers`, most also `content_drift`, because the adapter anchors each sentence in a figure the draft never contained. That is what it was trained on: its 79 hand-written targets carry a median of 9 numbers absent from their drafts and a median content overlap of 0.26, below the evade gate. Telling the model to use only the draft's figures changed nothing (0 of 9 at 6 and at 12 candidates). Letting new numbers through, as a measurement only, gave 5 of 9 desklib flips at overlap 0.30-0.41: the flips are bought with fabricated content, which report 23 §7 and report 24 §2 already refused. The consequence for the architecture is that the specifics have to come from the author. `PipelineConfig.facts` (API `facts`, the web "Facts you can vouch for" box) is now the only channel through which a new number, date, quotation or citation may enter a candidate; anything else new is refused and named in `unverified_specifics`. A facts-conditioned adapter (round 3) is trained on the same pairs with the target's particulars placed in the prompt. Results for both are in report 24 §5. GPTZero itself remains unmeasured: no API key is on the machine, and `scripts/gptzero_bench.py` is the calibration step waiting for one. (24 §5; `.kriukow/results_nonum.json`)

**25. Live GPTZero: it detects authorship, not register, and only base-checkpoint text passes.** The first paid GPTZero scores (2026-09-07 evening, report 24 §6) settled the question every earlier report hedged on. Every rewrite written by an instruction-tuned model scored 1.000 whatever its register: the Qwen instruct candidates, the round-4 register adapter that flips desklib, fakespot, academic and RADAR on 8 of 9 paragraphs, and Claude's own hand-written oracle rewrites, all of them. A human's manual re-expression of the same paragraph scored 0.000 with no citation. Raw candidates from the Qwen2.5-3B base checkpoint scored human about half the time, and the local proxies were anti-correlated with GPTZero on that text. The shipped pipeline is therefore the base checkpoint continuing a three-pair (draft, notes, rewrite) pattern with the author's facts in the notes slot, GPTZero ranking eight candidates over two rounds on 1 − P(human) so the mixed class cannot win, and the unchanged meaning gates: **7 of 9 AI paragraphs human on GPTZero, 0 of 5 human paragraphs harmed** (3B; the 7B base did 4 of 9 in one run). The pass rate is a best-of-N over a 30-50% per-candidate human rate, so the levers are the candidate count, GPTZero in the loop, and lifting the rate with a fine-tune of the base checkpoint toward human originals (the HIP recipe, in progress). Finding 5 stands confirmed; findings 1a and 24's register lever is downgraded to a local-detector artefact. (24 §6)

**What this round changes about the plan.** Nothing in the noise family enters the product. The budget goes to the HIP-style fine-tune (the only lever with verdict-flip evidence on current commercial detectors), the content-specificity candidate spec (the only prompt-level move with GPTZero verdict evidence and a negative grade cost), and two guards against the ways honest rewriting makes text *more* detectable: a polish gate and a register gate. The free distribution restorations from report 22 go in as quality realism, labelled as such until the bench shows flips. Section 4's matrix and section 11's next steps are revised accordingly.

---

## 2. The competitive bar, quantified

Rigorous multi-sample studies of commercial humanizers against GPTZero:

| Study | GPTZero result on humanized text |
|---|---|
| Pangram 4 re-run of the Chicago benchmark | TPR@1%FPR 98.64% → **44.32%**; under 50 words → 0.00% |
| DAMAGE, 19 humanizers, academic text | 99.73% → **60.04%** (TPR@5%FPR) |
| Russell et al., expert-guidebook prompt | **46.7%** detected |
| Van Vlasselaer, 4,000-word papers | GPTZero **2.5%** correct on humanized |
| Undetectable AI v11 on abstracts | GPTZero FNR **over 96%** |
| GPTZero's own report on 9 bypassers | **93.5% recall** (contradicts all of the above) |

Roughly **40-56% of humanized documents currently read as human** on rigorous tests, with wide variance by length and genre. Vendor benchmarks are irreconcilable and each favors its author.

**Two findings that make our specific goal the hard version of the problem.** QuillBot's Academic and Formal modes had a **0% pass rate** against GPTZero while its casual modes passed 30-46%. One reviewer's summary: "the higher the writing quality, the higher the detection risk." Formal academic register is the register LLMs imitate best, so it is intrinsically easiest for a detector.

---

## 3. What is dead, what still works

**Dead against GPTZero** (RAID deltas in parentheses): homoglyphs (−0.3), zero-width spaces (−0.3), whitespace injection (−0.3), number swaps (−0.7), misspellings (−1.4), British spelling (−1.6). The 2023 single-space exploit is patched. Also dead: "elevate with literary language" prompts, "high perplexity and burstiness" prompts, deliberate typos (now a paraphraser signal), QuillBot-style paraphrase, GPT-Store humanizer GPTs (97-100% AI), header and URL padding (masked August 2026), short-chunk splitting.

The largest RAID delta for GPTZero was upper/lower case swapping at −10.0, which wrecks readability. Synonym swap and article deletion were −5.5. Paraphrase was −2.5.

**Still works, ranked by evidence:** base-model generation or iterative base-model paraphrase; detector-in-the-loop RL (AuthorMist 92.3% attack success at 0.945 similarity); training-free detector-guided decoding (87.9% mean TPR@1%FPR reduction across eight detectors); style transfer toward a specific human author; whole-document rewriting rather than sentence patching; real sentence-length variance within the human band; first-person stance; concrete named and dated specifics.

**Revised 2026-09-07 (reports 17-19, 23).** Two entries above are downgraded. *Detector-guided decoding* flipped 9 of 9 on the detector in the loop and 0 of 9 on every held-out detector, with visibly damaged English (17 §B4); it is an adversarial example generator, not a humanizer. *Sentence-length variance* moved 3 of 30 verdicts when pushed into the band and 0 of 40 on saturated text (19 §4); keep it for quality, do not count it as a lever. Added to the dead list, now with local measurements: every post-hoc noise operator at readable doses on confident AI text (typos, casing, function-word deletion, WordNet synonyms, punctuation slips, hedges, sentence shuffle, redundancy: 0-2 flips of 40 on desklib, 0-1 on fakespot); plausible masked-LM substitution at any dose (1 of 30); temperature and `min_p` on an instruct model (28-30 of 30 still flagged); and back-translation (0 of 9). The commercial tool with the most injected noise, StealthWriter, is 100% AI on GPTZero in four 2026 tests (23). Added to the "still works" list at the top: **the HIP-style fine-tune of a base checkpoint on (AI paraphrase → human original) pairs**, which is the mechanism behind the base-model result and the only lever with verdict-flip evidence on current commercial detectors (17 §A.11, 18 §11).

---

## 4. The humanize / quality conflict matrix

The central product question, resolved. Detector benefit High/Medium/Low; grade cost from rubric analysis.

| Edit | Detector benefit | Grade cost | Verdict |
|---|---|---|---|
| Strip sentence-initial connectives (Furthermore, Moreover) | **H** | **Raises the grade** | Do aggressively |
| Raise sentence-length variance within the human band | **H** | 0 to positive | Do |
| Break the topic-sentence-then-summary template | **H** | 0 to positive | Do, preserve claim order |
| Kill AI vocabulary (delve, pivotal, tapestry, underscore) | **H** | 0 to positive | Do |
| Add concrete named, dated, numbered specifics | M | **Strongly positive** | Do — best joint move |
| Add a genuine counterclaim and rebuttal | M | **Strongly positive** | Do |
| Replace tricolons and "not X but Y" | M-H | 0 | Do |
| First person for argumentative acts ("I argue") | M | 0 (APA endorses) | Do, within discipline band |
| Sentence-initial But/And | L-M | 0 | Allow sparingly |
| Low-visibility punctuation slips | M | Low | **Yes, 0.3-0.8 per 100 words** |
| Sentence fragments | M | Medium-high | ≤1 per 1,500 words, never in intro/conclusion |
| Contractions | M-H | Medium | Never in body |
| "I think / I feel / in my opinion" | M | **High** | Never |
| Simplify vocabulary broadly | M | **High** | Never; only de-Latinize verbs |
| Add subordination to sound scholarly | L | **Medium** | No — moves toward conversation |
| Unpack phrasal syntax into finite clauses | L-M | **High** | Never |
| Spelling, agreement, wrong-word errors | M | **High** | Never |
| Any error in a quote, citation, thesis, or first/last sentence | M | **Very high** | Never |
| Leave text perfectly error-free | **Negative** | 0 | Avoid; enforce a floor |
| Reorder or drop evidence | M | **Very high** | Never |
| Trim commentary to shorten | L | **Very high** | Never |

**The headline asymmetry, and the reason this product can exist.** The four highest-value humanizing edits (strip connectives, vary sentence length, kill AI vocabulary, break the paragraph template) all have **zero or negative grade cost**. The four most damaging edits (simplify vocabulary, inject visible errors, unpack phrasal syntax, trim commentary) have only **medium** detector benefit. There is a large safe operating region. The product's job is to stay in it and refuse the rest. Every incumbent leaves it.

**Matrix corrections from the September 7 measurements (reports 19, 22, 23).** The matrix above was built from published deltas; the local bench now gives verdict counts, and several rows move.

| Edit | Was | Now | Evidence |
|---|---|---|---|
| Strip sentence-initial connectives | H | **L for detection, still do for the grade** | 2 of 30 flips, mean −0.013; the operator found only 0.37 events per 100 words to remove (19 §4) |
| Raise sentence-length variance into the band | H | **≈0 for detection, keep for quality** | 3 of 30 flips on boundary text, 0 of 40 on saturated (19 §4); the band itself was wrong at paragraph scale (§5) |
| Low-visibility punctuation slips, 0.3-0.8 per 100 words | M, Yes | **No flip evidence; floor only** | 1-2 of 30 at 0.5-1 per 100 words; 6 of 30 only at 2 per 100 words, which exceeds the human total error rate; 0 of 40 on saturated text (19 §4). Budget 1.6-3.3 per 1,000 words, ERRANT-native shape only: missing comma after a short introductory element, around a nonrestrictive element, hyphenation inconsistency (22 §7) |
| Kill AI vocabulary | H | **H, but only the curated map; never toward rarer or plainer words** | Plainer synonyms are the engine of Undetectable AI's pass rate and its register collapse (Academic Word List −23%); rarer synonyms are Pangram's headline artifact (23 §7); plausible masked-LM neighbours flip 1 of 30 (19 §4) |
| **New: delete emphasis metadiscourse** (notably, importantly, it is worth noting) | — | **Do; free; raises grade** | Human 0.12 vs LLM 0.52 per 1,000 words in 1,500 paired passages; experts' primary AI cue (22 §2, §7) |
| **New: restore semicolons, cut colons and dashes to the human rate** | — | **Do; free** | Semicolons 3.3 vs 0.6, colons 1.2 vs 4.0, dashes 2.7 vs 4.8 per 1,000 words, human vs LLM (22 §2); punctuation is a top SHAP feature family in a 2026 detector audit |
| **New: preserve typographic punctuation** (curly quotes, en and em dashes) | — | **Do; free** | Normalising them out of human text raised desklib +0.092 on affected paragraphs and flipped 2 of 33 human paragraphs to AI (19 §3); a tidy-up, not a strategy (0 of 40 on saturated) |
| **New: keep lexical repetition of the key term** | — | **Do not synonym-cycle** | Repetition does not separate human from LLM (29.7% vs 27.5% of sentences); synonym-cycling is the grade risk (22 §2) |
| **New: LLM polish pass on text that already scores human** | — | **Never above ~2% of words changed** | A maximally conservative "fix only what is wrong" pass through Qwen2.5-7B-Instruct changed 11% of words and flipped 11 of 33 human paragraphs to AI; 3B flipped 16 of 33 (19 §3) |
| **New: register drop, expansion, reader address, rhetorical questions, contractions** | — | **Refuse; gate** | These are how the passing commercial tools buy their pass rate (words per sentence +20%, length +27%, register scored 3 of 5), and Turnitin and DAMAGE now name them as humanizer signatures (23 §3, §5) |
| Typos, spelling, casing, function-word deletion, WordNet synonyms | Never | **Never, confirmed** | On confident AI text 0-2 of 40 flips at readable doses; visible damage; humanizer artefact for Pangram and GPTZero's paraphrase class (19 §4, 18 §3, 23 §4) |

---

## 5. Target distributions

From original measurement of five pre-ChatGPT human corpora plus Hyland and Biber.

**Shape**

| Feature | Academic target | Note |
|---|---|---|
| Sentence-length CV, document scale | **0.42-0.60** | Not "as high as possible" |
| Sentence-length CV, paragraph scale | **0.30-0.49** (median 0.38) | Measured on 1,264 human PMC paragraphs (19 §2). A quarter of genuine paragraphs sit below 0.30. The pipeline scores paragraphs, so `HUMAN_SENTENCE_CV` must use this band, not the document band |
| Lag-1 sentence autocorrelation | **0.01-0.10** | Do not alternate long/short |
| Long-range structure | 1/f, β ≈ 0.5 | Generate as fractional Gaussian noise |
| Sentences under 10 words | 9-14% | Enforce tails, not a wider middle |
| Sentences over 30 words | 18-28% | |
| Paragraph-length CV | 0.42-0.71 | Higher than sentence CV in every genre |
| Paragraph openers with a formal connective | ≤3-4% | |

**Register (per 1,000 words unless noted)**

| Feature | Value | Source |
|---|---|---|
| Passives | 18.5 (about 25% of finite verbs) | Biber et al. |
| Prepositional postmodifiers | about 68 | Biber et al. |
| Nominalizations | 61.0-72.1 | Biber & Gray, by sub-register |
| Noun-noun premodification | 76.6 science vs 24.3 history | 3× disciplinary spread |
| Hedges | 8.2 (mech eng) to 20.0 (marketing) | Hyland |
| Citations | 7.3 (mech eng) to 15.5 (biology) | Hyland |
| Self-mention (per 10,000) | 17.8 (mech eng) to 64.6 (physics) | Hyland |
| Comma density | 57-65 | Own measurement |
| Contractions | under 1.4 | Own measurement |
| Error rate | **2.45 per 100 words** | Lunsford, stable since 1917 |
| Type-token ratio | **Lower than fiction and news** | Counterintuitive; do not raise it |
| Semicolons / colons / dashes | 3.3 / 1.2 / 2.7 | Human side of 1,500 paired passages; LLM side 0.6 / 4.0 / 4.8 (22) |
| Emphasis metadiscourse (notably, importantly, it is worth noting) | ≤0.15 | LLM side 0.52 (22) |
| Sentence-initial And/But/So | 0.5-2% of sentences | Published articles run 40 illicit initials per 10,000 words (22) |

**Readability (report 20). Gate, never optimise.** Flesch Reading Ease is 83:1 a vocabulary meter (syllables per word) over sentence length. Instruct-model output is *harder* than matched human text by 6-25 FRE points, and the difficulty is lexical: GPT-4 writes shorter T-units at a higher grade level. Locally, Qwen restatements sit 6 FRE points below their PMC sources and Llama paraphrases 14 below (19 §2). Readability correlates with essay score at r = −0.17 to +0.13; it carries under 0.5% marginal value in detector ablations. Rules: |ΔFRE| ≤ 5 and |ΔFKGL| ≤ 1 input to output; then a wide per-genre plausibility band (research body 22-42, upper-division essay 35-52, first-year 45-62, op-ed 48-66); report Coleman-Liau alongside; pin one implementation and override `textstat`'s silent dropping of sentences under three words; strip citations and numerals before scoring. Inside a band, the target is a texture, not a number: report 21 §9 gives 9-12 checkable properties and 5 prohibitions per band, and §10.1 turns them into 22 gate and rank checks. Skilled expository prose almost never exceeds FRE 70; anything above it is narrative, so refuse band E for academic content.

**Two calibration warnings.** Hyland's headline corpus is 1997-98 text and the soft/hard field gap has since closed (by 2015 biology out-hedges applied linguistics), so calibrate on the 2018 numbers. And first person is normal in published academic writing: physics uses *we* at 39.3 per 10,000, philosophy uses *I* at 35.6. Professionals use first person about **four times more** than students do.

---

## 6. Recommended architecture

**Stage 0 — infrastructure, build first.**
- Feature extractor implementing the top-20 catalog from report 04 plus the shape features above.
- Five per-genre reference corpora, each 300+ documents, all pre-November 2022, storing mean, SD, 10/25/50/75/90 percentiles **and the empirical covariance matrix**. Features are correlated, so sampling independently produces documents that are plausible on every axis and jointly impossible.
- Style embeddings: StyleDistance (content-independent) and LUAR.
- Local detector bench: e5-small-lora (33M, cheap enough for an RL inner loop), RoBERTa-OpenAI, Fast-DetectGPT, Binoculars, desklib, RADAR, plus a style-based detector held out.
- **Calibration harness against the live GPTZero API**, measuring the joint probability of a human verdict given a local score. This is the only number that matters, and there is no substitute for buying it.

**Stage 1 — generation, not just rewriting.** Given finding 5, the primary lever is a **base checkpoint**, not an instruct model. Control-code supervised fine-tuning on reverse-direction pairs (HAP-E, HC3, M4, Ghostbuster, Herbold), conditioning on discretized target stylometric buckets drawn from the human side. LoRA r=32 on Qwen3-4B base or similar.

*Sharpened 2026-09-07.* The checkpoint alone is not the mechanism. Prompt-only base-model paraphrase flipped 0 of 9 locally (17 §B3), and the raw base model at T=1.0 scores human (desklib 0.199, 3 of 24 flagged) only by inventing content (topic overlap 0.11); at T=0.7 it collapses to 0.892 and at T=1.2 to gibberish (19 §5). HIP's own reading of its code is that it works because the fine-tune "reintroduces a human-text training signal." So Stage 1 is specifically: LoRA a base checkpoint on (AI paraphrase → original human passage) pairs with completion-only loss and a plain-text tag format, then iterate with a stop-at-first-held-out-flip rule. The published `Qwen3-4B-Base-HIP-adapter` is the 40-minute test of whether the mechanism transfers to our bench (18 §11.2). **Two gates the passing commercial tools refuse to carry and we must:** a polish gate (no instruct-model pass may change more than about 5% of words in a candidate that already scores human; 2% is the measured safe budget, 19 §3) and the Karr five (long-token ratio, Academic Word List ratio, type-token ratio, words per sentence, length ratio each within about 10% of the source; that gate alone would reject every Undetectable, Humbot and HIX output collected in 23).

**Stage 2 — sampling, not clamping.** Resample a real reference document's whole feature vector to preserve covariance for free. Split each target into a persona offset and a document residual using the measured intraclass correlations: freeze high-ICC punctuation habits per user, let low-ICC features float per document. Realize sentence lengths as correlated noise, then verify lag-1 autocorrelation lands in [−0.1, +0.25].

**Stage 3 — GRPO with a constrained multi-family reward.** Detector ensemble sampled per rollout (weight 1.0), StyleDistance cosine to the genre centroid (0.4), negative Wasserstein distance on the feature vector (0.3). **Hard gates returning large negatives, not soft weights:** bidirectional NLI entailment, embedding cosine ≥ 0.88, LanguageTool error rate inside the human band, perplexity inside the human band, length ratio within ±25%, and a **quality gate** on the rubric-derived stack. Set TRL's `beta` explicitly; it defaults to 0.0. Hold out Binoculars, RADAR, desklib and the style detector entirely.

**Stage 4 — reliability layer.** Best-of-N with **diversity engineering**, because correlated candidates are the binding constraint: per-candidate plan diversity, two base model families, temperature scaled with N. Cap refinement at 2-3 rounds with a keep-best archive and gates anchored to the *original*, never the previous round, since 60-80% of the gain lands in round one. Reject documents that fall outside roughly the 90th percentile of Mahalanobis distance from the reference distribution, which automatically prevents overshooting into unnaturally bursty.

**Stage 5 — artifact self-check before returning output.** Run the 28-item suite from report 12: collocation plausibility via logDice and PMI against a reference corpus, masked-LM token plausibility to catch tortured synonyms, Unicode normalization check, tense and pronoun consistency, citation and quotation integrity, and factual-drift detection. This is what stops us from producing detectable *humanized* text rather than human text. *Added 2026-09-07 from 23 §7:* a punctuation-class diff against the source (apostrophe count, hyphen count, comma-before-preposition, space after terminal punctuation; the 2024 Undetectable signature GPTZero now publishes as its bypasser example), a new-entity check (reject any capitalised run NER labels as a person or organisation absent from the source; the StealthGPT fabrication case), a contraction and reader-address check, a question-mark count no higher than the source, a two-sided perplexity band (fluent paraphrasers fail by landing *below* human perplexity, 9.3 against 15.0), and the readability delta from report 20.

**Selection rule that the mixing experiment forces (19 §6).** One AI sentence in a human paragraph flipped 10 of 33 human paragraphs to AI; half a paragraph of human sentences in an AI paragraph flipped 5 of 30. AI signal dominates at every mixture ratio. Rank and select whole paragraphs; never assemble a paragraph from best-scoring sentences, and never use the detector's sentence scores for selection.

**Shipping gate.** Report the full Pareto frontier on RAID held-out domains plus a live GPTZero sample: verdict rate at the user-visible threshold (not probability delta), TPR@1%FPR on six held-out detectors including the style detector, semantic similarity and NLI entailment, error rate against the human band, an essay-score estimate, and Wasserstein distance to the reference distribution. **Ship only if it dominates the no-training baseline**, which already achieves about 88% TPR reduction training-free.

---

## 7. Measuring quality

Automated essay scoring, in quadratic weighted kappa:

| System | QWK |
|---|---|
| Fine-tuned RoBERTa on PERSUADE 2.0 | 0.841 |
| Kaggle ASAP 2.0 gold band | 0.836-0.840 |
| NPCR on ASAP | 0.817 |
| **Expert human graders** | **0.712** |
| **Novice human graders** | **0.526** |
| Open 7-13B with trait-specialization prompting | 0.55-0.59 |
| **GPT-4 zero-shot on ASAP** | **0.042-0.081 on some prompts** |

**Do not use a frontier LLM zero-shot as the quality judge.** It is near-random on some prompts. Train a scorer on ELLIPSE plus PERSUADE instead; there is no credible open essay-scoring model on HuggingFace.

Two further cautions. Cohesion indices do not predict quality, because raters judge coherence by the *absence* of visible cues, matching the IELTS Band 9 descriptor "attracts no attention." And length is a massive confound in every automated scoring result.

The A-versus-B discriminator across every rubric examined is the AP English 3→4 boundary: **consistently explaining how evidence supports a line of reasoning.** Mechanics is 5-25% of the weight. No rubric rewards syntactic complexity as an end, and AP explicitly withholds its Sophistication point for "complicated or complex sentences or language that is ineffective."

---

## 8. Product and front end

Fifteen competitors ship the same page: a centered textarea, a greyscale detector-logo carousel, a vanity stat row, and three-column pricing with a savings toggle. Only four show a genuinely useful in-product detector score.

**The textarea is the confession.** No document model means no per-sentence anything. Controls are named but never explained ("Aggressive", "Samurai", "V7-N2"). Dark patterns are load-bearing: Humbot sits at 2.2/5 on Trustpilot, and reviewers describe cancellation as "intentionally designed to be confusing." **Frictionless cancellation is a differentiator.**

Recommended stack: ProseMirror with the Tiptap MIT core. Heatmap and diff live entirely in a DecorationSet, so the document only mutates on accept. Lexical is ruled out because its decorator nodes are document nodes, so a 200-sentence heatmap pollutes undo history. Do not buy Tiptap Track Changes; use the open suggest-changes plugin. Base UI over shadcn defaults.

**Five decisions that stop it looking vibe-coded:** kill the textarea; show two dials, risk *and* quality, never one; name the measurement every time ("your sentence-length variance is 0.41, human academic writing runs 0.42 to 0.60"); Word-style simple markup by default with green and red reserved for diffs and a separate ramp for risk; a paper-and-ink palette with no gradients and no purple, 13px sans chrome against a 19px serif canvas at 65 characters.

On positioning, five competitors including Undetectable, Humbot and HIX say nothing at all about academic integrity in their terms. Four explicitly condemn cheating. Twixify's terms admit it is a wrapper. There is room to be the credible, transparent option, and the same feature pipeline powers a legitimate "why does this read as AI" explainer, which matters because detector false positives hit non-native writers hardest at 37-61% in published studies.

---

## 9. Data

**Free and clean:** PMC Open Access via `s3://pmc-oa-opendata` (note the FTP paths every tutorial cites were deleted the week of 2026-08-24), arXiv sliced to `≤2112` for a hard pre-LLM cut, OpenWebText, 2021 Wikipedia dumps from archive.org, common-pile Gutenberg, RAID (MIT), Ghostbuster (CC BY 3.0), M4 (paired human and machine in one row, includes arXiv and PeerRead).

**Student writing, all NonCommercial:** BAWE (6.5M words, 30 disciplines) is the highest-value unacquired asset and the only corpus that truly matches "university-level human writing." Plus USE, PERSUADE 2.0, ELLIPSE. **These are fine for measuring reference distributions but need legal review before training a shipped commercial model.**

**Time-sensitive:** ICLE v3 becomes free on 2026-09-15. **Avoid COCA full text** at $395-$1,395; 5% of its words are deliberately removed, which corrupts sentence-level stylometry. Use the free 8.9M-word sample.

---

## 10. Open risks

- **The 40-56% baseline is a moving target.** GPTZero shipped 30+ releases in 20 months. Any static evaluation is stale in months.
- **Content origin remains detectable.** No amount of rewriting fixes it. Honest positioning matters here.
- **Provenance signals sit outside the text.** GPTZero Writing Reports and typing replay, and document version history, cannot be addressed by any text tool. Tell users this plainly.
- **Live API calibration costs money** and GPTZero's terms forbid competing use and reverse engineering, though they name neither adversarial testing nor training on outputs. Get legal review before building a distillation loop.
- **Verifier-generator gap.** Best-of-N actively seeks the surrogate's blind spots.
- **Non-commercial corpus licensing** constrains what can ship.

---

## 11. Next steps

*Status 2026-09-07.* Steps 1, 2 and 5 exist in some form (`features/`, `detectors/`, `humanize/`), and step 5's measurement is in: the training-free rule-based and LLM paths flip 0 of 9 paragraphs on the local bench (17, `pipeline.py` docstring). The next steps, in order of expected effect on the verdict-flip rate per hour of work (18 §11, 19 §9, 22 §7, 23 §7):

- **a.** Load the published HIP adapter on Qwen3-4B base, run the nine report-17 paragraphs for three rounds, score on desklib and held-out. About 40 minutes. This decides whether Stage 1's mechanism transfers to our bench before any training is spent.
- **b.** Add the content-specificity candidate spec (named, dated, numbered specifics; delete the scene-setting opener and the tidy closer). The one prompt-level move with verdict-level evidence on GPTZero (76.7 → 46.7 TPR) and a negative grade cost.
- **c.** Add the polish gate (word-change rate ≤ 5% on any candidate that already scores human) and the Karr-five register gate to `pipeline.py`. Both are guards against the two ways honest rewriting makes text *more* detectable.
- **d.** Lower `HUMAN_SENTENCE_CV` to the paragraph-level band 0.30-0.49, then condition it and the shape rules on the source paragraph's FRE band per report 21 §10.1 (dense bands 0.18-0.45, essay bands 0.40-0.90); tighten `readability_shifted` from 20 points to 8; add the closing-sentence gate (no summary, exhortation or forward-looking abstraction as the final sentence), the importance-adjective gate and the short-after-long rank term. These target the four expression-level differences in finding 22 that prompting does not fix.
- **e.** Add the free distribution restorations from report 22 to `transforms.py`: emphasis-metadiscourse deletion, semicolon restoration, typography preservation, sentence-initial But at 0.5-2% of sentences, and a repetition guard in the vocabulary pass. Label them quality-realism until the bench shows flips.
- **f.** Run the three report-22 tests and the ERRANT-native error-floor test on the saturated corpus. Expected result for the error floor: null. Its purpose is to close the question.
- **g.** Do not build: noise operators, sampler diversity on an instruct model, more candidates from the same instruct model, or extensions to `guided.py` (its e5 guide flags 24 of 40 genuine human paragraphs).

The original list follows for the record.

1. Scaffold the repo: `features/`, `detectors/`, `data/`, `eval/`.
2. Build Stage 0 and reproduce the report 10 measurements on our own machine, including a spaCy parser to close the clause-level gap.
3. Acquire BAWE and build the five reference corpora with covariance matrices.
4. Buy a GPTZero API budget and measure the local-to-GPTZero calibration curve. Nothing downstream is meaningful without it. `scripts/gptzero_bench.py` runs the shipped pipeline on a paragraph set with optional author facts, scores before and after with the cached GPTZero client, and writes rows for `humanizer calibrate`; it prices the run and exits when no key is set.
5. Build the training-free baseline: base-model rewriting plus the sampling recipe plus the artifact self-check. Measure it. This is the number to beat.
6. Only then train.
7. Front end last.
