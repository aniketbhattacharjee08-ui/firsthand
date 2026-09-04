# Humanizer Research Synthesis

Date 2026-09-03. Seventeen reports, about 89,000 words, roughly 600 sources. Read this file first.

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

---

## 5. Target distributions

From original measurement of five pre-ChatGPT human corpora plus Hyland and Biber.

**Shape**

| Feature | Academic target | Note |
|---|---|---|
| Sentence-length CV | **0.42-0.60** | Not "as high as possible" |
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

**Stage 2 — sampling, not clamping.** Resample a real reference document's whole feature vector to preserve covariance for free. Split each target into a persona offset and a document residual using the measured intraclass correlations: freeze high-ICC punctuation habits per user, let low-ICC features float per document. Realize sentence lengths as correlated noise, then verify lag-1 autocorrelation lands in [−0.1, +0.25].

**Stage 3 — GRPO with a constrained multi-family reward.** Detector ensemble sampled per rollout (weight 1.0), StyleDistance cosine to the genre centroid (0.4), negative Wasserstein distance on the feature vector (0.3). **Hard gates returning large negatives, not soft weights:** bidirectional NLI entailment, embedding cosine ≥ 0.88, LanguageTool error rate inside the human band, perplexity inside the human band, length ratio within ±25%, and a **quality gate** on the rubric-derived stack. Set TRL's `beta` explicitly; it defaults to 0.0. Hold out Binoculars, RADAR, desklib and the style detector entirely.

**Stage 4 — reliability layer.** Best-of-N with **diversity engineering**, because correlated candidates are the binding constraint: per-candidate plan diversity, two base model families, temperature scaled with N. Cap refinement at 2-3 rounds with a keep-best archive and gates anchored to the *original*, never the previous round, since 60-80% of the gain lands in round one. Reject documents that fall outside roughly the 90th percentile of Mahalanobis distance from the reference distribution, which automatically prevents overshooting into unnaturally bursty.

**Stage 5 — artifact self-check before returning output.** Run the 28-item suite from report 12: collocation plausibility via logDice and PMI against a reference corpus, masked-LM token plausibility to catch tortured synonyms, Unicode normalization check, tense and pronoun consistency, citation and quotation integrity, and factual-drift detection. This is what stops us from producing detectable *humanized* text rather than human text.

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

1. Scaffold the repo: `features/`, `detectors/`, `data/`, `eval/`.
2. Build Stage 0 and reproduce the report 10 measurements on our own machine, including a spaCy parser to close the clause-level gap.
3. Acquire BAWE and build the five reference corpora with covariance matrices.
4. Buy a GPTZero API budget and measure the local-to-GPTZero calibration curve. Nothing downstream is meaningful without it.
5. Build the training-free baseline: base-model rewriting plus the sampling recipe plus the artifact self-check. Measure it. This is the number to beat.
6. Only then train.
7. Front end last.
