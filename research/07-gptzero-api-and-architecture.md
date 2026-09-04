# 07 — GPTZero: API, Architecture, and Scoring Mechanics in Full Detail

*Research date: 2026-09-03. Builds on report 06 (trends, Model 3.7b/3.15b/4.8b). Everything below is sourced from GPTZero's own pages (support center, FAQ, news posts, Notion release notes, the Feb 2026 arXiv paper), third-party API wrappers and benchmark code, and independent tests. Where a claim is inferred rather than documented, it is flagged.*

**Access notes.** GPTZero's interactive API reference (`gptzero.me/docs` → `gptzero.stoplight.io`) is a client-rendered Stoplight app; neither the live page, its Wayback snapshots, nor Stoplight's node API (returns HTTP 500 without a session) expose the OpenAPI text to a non-browser client. The schema table below is therefore reconstructed from (a) GPTZero's support articles that name fields, (b) GPTZero's own benchmark posts, (c) four independent wrappers/skills (R `gptzeror`, terminalskills, fast.io, jentic), and (d) the field-parsing code of a May 2026 public benchmark repo that hits `v2/predict/text`. The release-notes Notion page was read via Notion's public `loadPageChunk` endpoint (three chunks, 2023→Aug 2026).

---

## 1. The public API

### 1.1 Endpoints, auth, limits

| Item | Value | Source |
|---|---|---|
| Base | `https://api.gptzero.me/v2` | fast.io guide; mattc95 benchmark script (`DEFAULT_ENDPOINT = ".../v2/predict/text"`) |
| Text scan | `POST /v2/predict/text` — body `{"document": "<text>"}` | same |
| File scan | `POST /v2/predict/files` — multipart PDF/DOCX/TXT ("AI-detection on an array of files") | Stoplight node title; jentic |
| Batch (async) | `POST /batch`, `GET /batch/{batchId}`, `GET /batch/{batchId}/results` | jentic OpenAPI listing |
| Reports | `GET /reports?limit=50`, `GET /reports/{reportId}` | jentic |
| Auth | header `x-api-key` (key from `https://app.gptzero.me/app/api`) | support "How can I get the API" |
| Optional request params seen in the wild | `multilingual: true`; `modelVersion` (e.g. `"2025-12-18-base"`); `apiVersion`; a `version` field per fast.io | mattc95 script lines 294–304; fast.io |
| Rate limit | "The rate limit for all subscription plans is **30,000 requests an hour**"; higher on request to api@gptzero.me | support "What is the rate limit" |
| Server capacity (paper) | one ECS instance on Nvidia Ampere GPUs handles ~10 req/s, median 480 ms, p95 < 860 ms | arXiv 2602.13042 App. I |
| Minimum length | GPTZero's own benchmarks enforce "a minimum of 250 characters (~50 words)"; third-party skills say "~250 characters for reliable results"; support says accuracy "increases as more text is submitted" (documents > paragraphs > sentences) | benchmarking page; terminalskills; support "Limitations" |
| Maximum length | not published. Paid plans "allow more characters in a single request" than free. EmpirioLabs' reseller truncates each document at 50,000 characters | support "free vs paid"; EmpirioLabs |
| Free API trial | "try out the API right here in the docs without an API-key for a limited number of uses" | support "Can I try the API for free" |
| Data retention | "We do not store or collect the documents passed into any calls to our API"; privacy policy (Apr 2 2026): API content "not stored by default nor used for product improvement"; dashboard inputs retained "in aggregate" | support; privacy-policy.html |
| On-prem | Docker-container deployment via sales | support "on-premise" |
| GitHub | none ("we aren't open-source") | support |
| Errors | 429 rate limit; 402 quota exceeded; 403 seen intermittently in a 1,000-doc run (2 of 1,000) | terminalskills; mattc95 README |

### 1.2 Full response schema (reconstructed)

Top-level envelope: `{ "version": "<model version string>", "neatVersion": "<string>", "documents": [ {...} ] }`. The mattc95 script reads `response["version"]`, `response["neatVersion"]`, and `response["documents"][0]`; the R wrapper flattens `documents[0]` into `doc_*`/`par_*`/sentence columns. Field-level detail:

| Field (path) | Type / values | Meaning and status | Source |
|---|---|---|---|
| `version` | string, e.g. `2025-12-18-base`, `2026-08-01-base`, `2026-06-10-multilingual` | Model version served. Format `YYYY-MM-DD-base` (English) or `-multilingual`. Release notes map these to "Model 3.15b", "4.8b", "4.1m" etc. Also echoed inside each document (`documents[0].version`) | Notion release notes; Booth post ("2025-12-18-base"); mattc95 |
| `neatVersion` | string | Undocumented; appears to version the post-processing/"neat" pipeline separately from the model. Logged but unexplained by any wrapper | mattc95 script |
| `documents[].document_id` | string | Server-side ID (used by `/reports`) | mattc95 |
| `documents[].document_classification` | `HUMAN_ONLY` \| `MIXED` \| `AI_ONLY` | "the most likely classification of the document" — the field GPTZero tells integrators to display | FAQ; support "interpret results from your API" |
| `documents[].predicted_class` | `human` \| `ai` \| `mixed` | argmax of `class_probabilities`. GPTZero used this (not `average_generated_prob`) in the Booth benchmark; Mixed counted as AI for binary metrics | support; Booth post |
| `documents[].class_probabilities` | `{human, ai, mixed}` summing to 1 | Added Dec 11 2023. Post-remapping calibrated so that a 90% AI score ≈ 90% chance the doc is entirely AI (Jan 9 2024 note) | Notion; support |
| `documents[].confidence_score` | float 0–1 | Added Dec 11 2023 ("0 least confident to 1 most confident") | Notion |
| `documents[].confidence_category` | `high` \| `medium` \| `low` | Banded confidence; see §4 | support; FAQ |
| `documents[].confidence_thresholds` | object of raw thresholds | "Raw confidence score thresholds used internally"; devs told to use `confidence_category` instead | web-search snippet of Stoplight docs |
| `documents[].result_message` | string | "the message GPTZero shows to the user on their AI classification" (e.g. "We are highly confident this text is entirely AI generated") | Stoplight snippet; mattc95 |
| `documents[].result_sub_message` | string | Secondary UI message (e.g. paraphrase notice) | mattc95 |
| `documents[].subclass` | `{predicted_class, confidence_score, confidence_category, class_probabilities}` | Sub-classification under AI. Values documented Nov 2024 (`2024-11-04-base`): `pure_ai`, `ai_paraphrased`. `confidence_category` here can be `high/medium/low/reject`. Thresholds published for `ai_paraphrased`: reject < 0.85, low 0.9, medium 0.95. Since Model 3.5b (Jun 2025) a "polished"/lightly-edited class exists; Booth post lists five UI categories: Human, AI-generated, Mixed, AI-paraphrased, Lightly edited by AI | support "What do AI subclass categories mean"; Notion; Booth post |
| `documents[].average_generated_prob` | float 0–1 | Mean of sentence `generated_prob`. Legacy; "should not be used for binary classification" | FAQ snippet; R pkg |
| `documents[].completely_generated_prob` | float 0–1 | Legacy pre-Dec-2023 headline score = P(entire doc AI). Still returned | R pkg; terminalskills |
| `documents[].overall_burstiness` | float (e.g. 101) | Legacy variance-of-perplexity statistic. Support: "GPTZero no longer uses perplexity and burstiness for its AI detection" | R pkg; support |
| `documents[].paragraphs[]` | `{completely_generated_prob, num_sentences, start_sentence_index}` | Paragraph roll-ups | R pkg (`par_*`) |
| `documents[].sentences[]` | `{sentence, generated_prob, perplexity, highlight_sentence_for_ai}` | Sentence text; sentence-head P(AI); legacy perplexity; boolean highlight flag. "API users can access sentence-level highlights through the highlight_sentence_for_ai field" | FAQ; terminalskills; R pkg |
| `documents[].writing_stats` | object | Referenced in prior research; not confirmed in any 2026 source. Treat as possibly dashboard-only | — |
| Interpretability (Deep Scan δ-scores, AI Vocabulary, AI Patterns) | not in `/predict/text` | Deep Scan attribution scores, AI Vocabulary and AI Patterns are dashboard "Advanced Scan" features; no public endpoint found. Writing Report/plagiarism/bibliography checks are Origin/Docs features, and EmpirioLabs exposes `scan_type: ai_detection | bibliography | sources | fact_check` via its own wrapper, implying GPTZero has such endpoints for partners | AI Patterns post; EmpirioLabs |

**Naming inconsistencies to expect:** camelCase at the envelope (`neatVersion`, `modelVersion`) vs snake_case inside documents; `predicted_class` values lower-case (`ai`) while `document_classification` is upper-snake (`AI_ONLY`).

### 1.3 Pricing

| Plan | Price | Words/month | Notes | Source |
|---|---|---|---|---|
| Free | $0 | conflicting reports: 10,000 words/mo + 5 advanced scans; or 5 scans/day at 1,200 words; ISCAP 2024 measured 7 submissions/day × 5,000 chars (IP-tracked); Chrome page: scans under 10,000 characters need no account | stack-junkie; ISCAP 2024; gptzero.me/chrome |
| Essential | $14.99/mo or $8.33/mo annual | 150,000 | basic scan, Chrome ext | OMR; CASRAI |
| Premium | $23.99 / $12.99 annual | 300,000 | + advanced scan, writing feedback, plagiarism | same |
| Professional | $45.99 / $24.99 annual | 500,000 | + **API**, batch 250 files, "up to 10 million excess words" | same; ISCAP |
| Overage | **$0.00046 per word** on all plans; up to 1M words over before forced upgrade | support "What happens if I go over" |
| Reseller | EmpirioLabs: $0.39 per 1,000 words pay-as-you-go | EmpirioLabs |
| Researchers | free detector + API access on application | gptzero.me/resources/researchers |

### 1.4 Determinism

No GPTZero document states that scores are deterministic. Evidence of non-determinism and its causes:

- Release notes repeatedly target it: Feb 6 2024 "less variation in predictions when cutting out a few sentences from long AI-generated documents"; **May 2 2025 (Model 3.4b) "Increased output stability when rescanning multiple times"**; Jul 23 2025 (3.6b) "More consistent results on documents with special characters"; Jul 10 2026 (4.7b) "Improved advanced scan consistency"; Aug 1 2026 (4.8b) post admits 4.7b "occasionally showed significant score variations when headers were added or removed".
- compareaitools (Apr–May 2026): same 50 documents resubmitted on different days → "noticeably different scores on 6 out of 50 tests — sometimes swinging by 15–20 percentage points"; "a known complaint on G2 and Reddit". Note the interval spans model releases (4.4b→4.6b), so version drift is the likeliest cause.
- WriteBros (2026): "small score changes appeared even without editing the writing… Short paragraphs tended to trigger more volatility."
- Architecture reasons variance can exist: long documents are split into windows and aggregated (paper §3.5); output remapping is applied post-hoc; the model is redeployed every 1–4 weeks (a `modelVersion` request parameter exists in third-party code, so pinning is possible but undocumented).

**Practical reading:** within a single model version, repeated identical calls appear to be near-deterministic (the 3.4b note implies they fixed residual jitter); across days you are usually measuring a different model.

---

## 2. From sentence scores to the document verdict

**Two heads, one forward pass.** The paper (§3.2) describes a multi-task loss ℒ = ℒ_d + α·ℒ_s: a document-level cross-entropy over {Human, AI, Mixed} (hierarchical, with {Pure AI, Polished, AI Paraphrased} under AI) plus a sentence-level *binary* BCE head. "We frame sentence-level predictions as a binary classification problem due to the lack of a well-defined criteria for what mixed sentences are." The document class is the **argmax of the document head after remapping**, *not* an aggregate of sentence scores.

**Why that matters (Mixed).** §5.5 ablation: forcing binary training and deriving Mixed by thresholding mean sentence probability (`Human if p_avg < 1−τ; Mixed if 1−τ ≤ p_avg ≤ τ; AI if p_avg > τ`) is inferior on Human/AI/Mixed sets; the ternary head needs no τ search. The Dec 11 2023 release note defines Mixed as "a combination of significant human passages and AI passages", and explicitly separates the two meanings of "50% AI": (1) "We are unsure if the entire document is AI or human", (2) "We are confident 50% is AI-like, and 50% is human-like". The paper: "the Mixed class allows us to decouple model confidence and the proportion of the text generated by an AI." No numeric "% AI text → Mixed" cutoff is published; Model 4.5b (May 6 2026) added "More granular mixed text predictions and identification of interleaving mixed texts", i.e. Mixed is meant to fire on alternating spans, not just a contiguous appended block.

**Long documents.** Preprocessing: "basic cleaning and reformatting… such as removing extraneous whitespace"; documents longer than context length T are split into disjoint windows; `f_doc(d) = agg(f_doc(W_1)…f_doc(W_m))` with agg ∈ {average, median, maximum} (which one is proprietary); sentence predictions are concatenated; then a remapping r: ℝ³→ℝ³ is applied "to improve calibration and reduce false-positive predictions" (App. K: ECE 0.074 → 0.070 on a 27,000-sample internal set, "with a focus on penalizing overly confident predictions… transforming low confidence AI/mixed predictions to human").

**Highlighting.** Support: highlights mark sentences that "disproportionately affect your overall AI or human score"; the model "holistically analyzes the whole text — adjusting an entire document changes your score more than just adjusting the sentences". The FAQ on "moderate likelihood but no yellow highlights": "there isn't one specific section of the document that is especially likely to have AI content" — the whole document shows "signs of being AI generated, but with lower certainty". So `highlight_sentence_for_ai` is a *thresholded sentence-head output*, and the dashboard's Deep Scan importance is a *separate saliency+occlusion attribution* (paper §4: "assigns a score δ_k indicating how the presence of s_k affects P(y=AI|d)", designed to mimic users who "replace some words with their synonyms, or remove them"). Removing the top-k% Deep-Scan sentences drops P(AI) faster than removing top-k% by sentence head (Fig. 4, 100 docs).

**Alignment history.** Oct 16 2023 and Oct 31 2023 notes: "upgrades to our sentence-highlighting model to align more closely with our overall document probabilities"; May 23 2023: sentence-labeling model moved to "more powerful deep learning architecture… alignment with the document-labeling model". Older (2023) highlighting used an HMM over sentence scores (technology page still mentions HMM).

**Known quirks (documented):**
- *Numbered lists*: since Nov 21 2023 "list numbers and corresponding content are treated as a contiguous span".
- *Lists and bold*: internal "Formatting Benchmark" of ~2,000 docs with lists (items merged into sentences as the control) and ~2,000 with bold (App. E) — used to verify formatting invariance; Model 3.2b, 3.5b, 3.6b, 4.2b notes: "robustness to text formatting changes", "special characters", "Decreased reliance on special characters".
- *Headers* (Model 4.8b, Aug 1 2026): "headers are now excluded from our model's input", shown gray in dashboard and extension. Masked: Markdown header markers, common words like "Conclusion", short emoji-led lines, numeric prefixes like "1. Introduction". Explicitly *not* masked: list prefixes, code-block descriptions, hashtags, email intros, author attributions (planned). Frontmatter and bibliographies "planned for future masking". Rationale: headers are "easy to edit/replace" and were "attack vectors for bypassers".
- *Dates and named entities*: 3.13b "Reduced FPR on documents with dates"; 4.1b "Improved robustness to named entities".
- *Quotes/citations/code*: no official handling statement found. The Internet Text Search component (2023–24) matched "direct quotes from existing websites through May 2023". The Limitations FAQ warns "highly procedural text" may be flagged and advises scanning "descriptive text portions rather than technical or structured content".
- *Short sentences*: not documented as unhighlighted; support only says sentence-level is least reliable.

---

## 3. Architecture clues

**What GPTZero says explicitly (arXiv 2602.13042, Feb 13 2026; authors Adam, Cui, Thomas, Napier, Shmatko, Schnell, J.J. Tian, Dronavalli, E. Tian, Dongwon Lee/Penn State):**
- "The GPTZero detector uses a deep learning architecture trained in a supervised fashion"; "Architecture and hyperparameters are proprietary."
- Hierarchical heads: L0 {Human, AI, Mixed}; L1 under AI {Pure AI, Polished, AI Paraphrased}; "falling back to the more reliable parent class for low confidence subclass predictions" (this is what `subclass.confidence_category = reject` encodes).
- Training corpus (Table 4): Academic 1.25M, News 16M, Web Articles 8M, Encyclopedia 2.8M, Essay 234K, Reviews 173K, Q&A 65K, Conversation 32K (AI-only) — **≈28.6M documents**. AI side from "a proprietary generation pipeline using various prompting strategies… OpenAI, Anthropic, Google, and open-source LLMs".
- Data QA against shortcut features (Table 5: lowercase/spacing artefacts in human scientific text, emojis, missing spaces in AI text) via "statistical methods and manual inspection datasets".
- User-feedback loop: disputed predictions are clustered to find "underrepresented domains and common failure cases".
- Polished class: human text LLM-edited for grammar/clarity, kept only if τ_min ≤ LevenshteinRatio(human, polished) ≤ τ_max (cosine similarity rejected because it "fail[s] to capture stylistic differences").
- Four-tier red teaming (data augmentation of AI texts): (1) paraphrasing prompts incl. multi-turn paraphrase↔translation chains through up to 4 languages; (2) paraphraser models Dipper and TempParaphraser at proprietary ratios α, β of N_AI that "vary between model releases"; (3) fine-tuning on limited data from nine black-box services — GPTinf, Grubby AI, HIX, Quillbot, StealthGPT, StealthWriter, TwainGPT, Undetectable, WriteHuman (Table 8); (4) white-box gradient-guided token substitution using a RoBERTa MLM under a perplexity constraint (Zhou et al. 2024).
- Inference: whitespace cleaning → windowing → aggregation → ℝ³ remap. Deployment: AWS ECS, Nvidia Ampere.
- Interpretability side-model (App. I.1): an XGBoost classifier over ~200 → ~20 mined stylistic features (2,000 balanced docs; ~91% accuracy) — the machinery behind AI Vocabulary (Oct 2024: phrases 2×–182× over-represented, e.g. "play a significant role in shaping" 182×, "today's fast-paced world" 107×) and AI Patterns (Aug 19 2026: 10 rhetorical patterns such as "Not just X, but Y", "Everything in threes", "Phantom experts", mined from the internal human/LLM database across GPT-5.x/4.x, Gemini, Claude, Grok). Both are explicitly *not* inputs to the detector score.

**Multilingual.** A separate model line (`-multilingual`, versions "3.1m"…"4.1m"): dedicated FR/ES model Apr 10 2024; "Upgraded to larger multilingual model" Sep 29 2025; +AR/IT/KO/ZH/JA Oct 24 2025; +TR/HI/NL/VI/ID Nov 9 2025; support lists 23 languages. ~15% of scans are non-English. Whether the API auto-routes or requires `multilingual: true` is not documented; third-party code sends the flag.

**Legacy "seven components".** The 2024 ISCAP paper (Issues in Information Systems 25(3)) transcribes GPTZero's then-technology page: Education Module (compares to student-written corpora), Burstiness, Perplexity, GPTZeroX (sentence-by-sentence classifier), GPTZero Shield (homoglyph/spacing attack database), Internet Text Search (quote lookup, index through May 2023), Deep Learning; "each component provides a weighted score to the Document Classification". The Nov 20 2023 note "Consolidated ensemble predictions" and the support statement that perplexity/burstiness are no longer used indicate this ensemble has been folded into the single multi-task model; "Paraphraser Shield" survives as marketing for the red-teamed paraphrase robustness and the `ai_paraphrased` subclass.

**Team/tooling.** Job posts: "5+ YOE in PyTorch/Transformers", "design, train, and fine-tune state-of-the-art language models", hallucination detection, RAG, "writing stylometry"; Toronto ML team (Cui: Caltech → U of T MSc). Company: 19M users, ~$30M ARR, acquired by Superhuman (Grammarly) June 2026 at an $88M valuation (Originality.ai review; Wikipedia).

**Patents.** No US/EP patent or application with GPTZero Inc. as assignee or Tian/Cui as inventors surfaced in Google Patents searches; the only AI-detection patents found belong to Originality.ai's founder.

---

## 4. Confidence calibration and benchmark methodology

**Bands.** Three UI/API bands; GPTZero's published error rates for them are inconsistent across pages:

| Band | API value | Published meaning |
|---|---|---|
| Highly confident | `high` | "99.1% of human articles are classified as human, and 98.4% of AI articles are classified as AI" (support, API article); technology page: "error rate less than 1%"; scan-interpretation post: "<2% error rate" |
| Moderately confident | `medium` | "~10% error rate" |
| Low confidence / uncertain | `low` | "≥14% error rate"; support UI note: low certainty means the text is "dissimilar to the training data" |
| Subclass bands | `high/medium/low/reject` | `ai_paraphrased`: reject < 0.85, low ≥ 0.9, medium ≥ 0.95 (2024-11-04-base) |

Design bias: "if GPTZero isn't confident something is AI, it generally airs on the side of 'human' with a low confidence score" (FAQ); App. K remaps low-confidence AI/Mixed toward Human. Integrators are given override recipes: require `predicted_class ∈ {human, mixed}` probability > 65% else call it `ai` (more sensitive), or override to `human` when `ai`/`mixed` < 80% (fewer FPs); "false negatives… are preferred to false positives" for academic use.

**Benchmark methodology (benchmarking page, Feb 2026; paper §5):**
- Per-domain sets: 1,000 human + 1,000 AI, AI split 250 per LLM; current LLM slate GPT-5.2, Gemini 3 Pro, Claude Sonnet 4.5, Grok 4 Fast; refreshed quarterly. Domains: ICLR reviews (pre-2022 human), abstracts, creative writing, essays, product reviews; min 250 chars.
- Metrics: AUC, plus recall and accuracy **at a 1% FPR threshold**; for multi-class competitors "sum the probabilities of the non-human classes".
- Results, GPTZero 4.1b: AUC 99.9 everywhere; accuracy 99.4–99.9; recall 97.4–99.7; vs Pangram 3.1 recall 87.2–99.8, Originality lite-102 82.6–99.2.
- Bypasser set: 1,000 AI texts (GPT-5, GPT-4o, GPT-4.1 sources) through 9 services + prompt paraphrasing → GPTZero **93.5% recall** vs Originality 57.3%, Pangram 49.7% (Jan 2026 post says 12+ tools, Pangram v3 50.2%).
- Multilingual (3.7m): 1,100/1,100 across 24 languages from CulturaX + MULTITuDE v3 → AUC 99.9, acc 98.8, recall 97.6.
- Polished: 4,631 texts → 4,175 correct, 247 → AI, 205 → Human, 4 → Mixed.
- Internal eval: 40k docs, 40% human / 40% AI / 20% mixed, incl. OOD sets.
- Mixed accuracy claim on the technology page: 96.5% with 0.9% FPR.
- ESL: TOEFL-essay FPR 1.1% (technology page). Independent (PMC 12453642, Dec 2024, 72 abstracts): FPR 0%, but for AI-*assisted* text non-native authors' misclassification rate 25% vs 11% (p = 0.036).
- Booth/BFI (Oct 2025, pre-3.15b): GPTZero "false negative rate… around 50% and above" under StealthGPT. GPTZero's own re-run on the same 1,992-doc Booth corpus with 2025-12-18-base (Jan 12 2026): FPR 0.05%, recall 99.3%, using `predicted_class` with Mixed/paraphrased counted as AI.
- Independent 2026: mattc95 (May 14 2026, 1,000 Pile-small/12-LLM texts): FPR 2.2%, recall 99.6%; MPG ONE (Jul 2026): 79–85% overall, misses on newest models, worst on <100 and >1,500 words; Verva (Dec 2025, 3.15b): 6/7 humanizers caught but a 2013 human text scored "100% AI… paraphrased".

---

## 5. Model release timeline

Version strings are `YYYY-MM-DD-base` / `-multilingual`; "Model N.Nb/m" labels began Jan 2025. Source: Notion release notes unless marked.

| Date | Version | Change (GPTZero wording, abridged) | Notes / regressions |
|---|---|---|---|
| 2023-01-03 | — | Launch (perplexity + burstiness, GPT-2 based) | Wikipedia |
| 2023-05-23 | — | "much more powerful deep learning model"; sentence-labeling model moved to DL, aligned with doc model | |
| 2023-06-30 | 2023-06-30 | "novel deep learning approach" combining prior architecture, statistical discriminators, DL; GPT-4 AUC 0.85→0.95; bi-weekly updates promised | blog Jul 2 2023 |
| 2023-08-22 | — | New DL model "convincingly surpasses" competitors; public benchmark sheet | blog Aug 24 2023 |
| 2023-10-16 | — | Fewer FPs on ESL student docs; sentence-highlighting aligned to doc probabilities | |
| 2023-10-31 | — | Highlighting consistency; faster long docs | |
| 2023-11-20/21 | — | "Consolidated ensemble predictions"; probabilities re-tuned; numbered-list span fix | ensemble → single model |
| 2023-12-11 | — | **Mixed class** (human/mixed/ai sum to 1) and **confidence score 0–1** added | UI switch Dec 29 2023; "AI probability looks lower" FAQ |
| 2024-01-09 | 2024-01-09 | Calibration: "AI score of 90% → 90% chance entirely AI" | new UI Jan 17 2024 |
| 2024-02-06 | 2024-02-06 | Long-doc consistency when sentences removed | |
| 2024-03-28 / 04-04 | 2024-03-28, base-2024-04-04 | Academic writing; better-tuned confidence; essay recall | |
| 2024-04-10 | 2024-04-10-multilingual | Dedicated FR/ES model | |
| 2024-07-12 / 08-02 | base | Open-source LLM recall; essay detection; formatting robustness | |
| 2024-10-19 / 11-11 | base | FPR tweaks | |
| 2024-11-04 | 2024-11-04-base | "Increased robustness to paraphrasing"; **subclass `pure_ai`/`ai_paraphrased`** | blog Nov 7 2024: ~95% paraphrase acc, <0.1% FPR (beta) |
| 2024-12-17 | base | Claude/Gemini detection | |
| 2025-01-10 | 3.1b | o1 + paraphrased texts | o1 bench Jan 13: 97.2% recall, 0% FPR (250/250) |
| 2025-03-04 / 03-13 | 3.2b / 3.3b | Formatting robustness; paraphrase recall | |
| 2025-04-01 / 05-13 | 3.1m / 3.2m | FR/ES/DE/PT; <1% FPR all four (blog May 27) | |
| 2025-05-02 | 3.4b | "Increased output stability when rescanning multiple times" | first determinism fix |
| 2025-06-03 | 3.5b | **"Polished" (lightly AI-edited) class**; bypasser + formatting robustness | |
| 2025-07-23 | 3.6b | Lower FPR; special-character consistency; bypassers | |
| 2025-08-11 | 3.7b | Latest LLMs; "openai deep research data"; "adversarial prompts" (RL-found bypass prompts; grammar-tool-edited human text) | blog Aug 14: recall@1%FPR 89.9% (o3)–99.1% (Sonnet 4); GPT-5 95.0% untrained |
| 2025-08-22 | 3.8b | GPT-5 | |
| 2025-09-04 / 09-20 | 3.9b / 3.10b | Broad FPR reductions | |
| 2025-09-29 | 3.3m | "larger multilingual model" | |
| 2025-10-17 / 10-30 | 3.11b / 3.12b | Scientific/web text; bypassers; creative writing | |
| 2025-10-24 / 11-09 | 3.4m / 3.5m | +AR/IT/KO/ZH/JA; +TR/HI/NL/VI/ID | |
| 2025-11-13 / 11-28 | 3.13b / 3.14b | FPR on docs with dates; "robustness against word swapping" | |
| 2025-11-20 / 12-04 | 3.6m / 3.7m | FPR on formal multilingual docs | paper's multilingual eval |
| 2025-12-18 | 3.15b | "Improved robustness to AI paraphrasers" | Verva: 6/7 humanizers caught; 2013 human text → 100% AI paraphrased. Used for Booth re-run |
| 2026-01-20 | 4.1b | Named entities; classic texts, Q&A, reviews, legal, government | paper's English eval |
| 2026-02-16 | 4.2b | "Decreased reliance on special characters" | |
| 2026-03-11 | 4.3b | AI-polished class | benchmarking page |
| 2026-04-02 | 4.4b | Frontier + open-source recall; lower FPR | |
| 2026-05-06 / 05-11 | 4.5b / 4.6b | "More granular mixed… interleaving mixed texts"; domain rebalancing | |
| 2026-06-10 | 4.1m | News-domain balance | |
| 2026-07-10 | 4.7b | Academic benchmarks; "Reduced FPR on customer-aligned data"; **"Improved advanced scan consistency"** | 4.8b post: 4.7b score swings with headers |
| 2026-08-01 | 4.8b | **Headers excluded from model input** (gray) | |
| 2026-08-12 | 4.9b | Recall on Claude 5, GPT 5.6, Gemini 3.6, Grok 4.5 | |
| 2026-08-19 | — | AI Patterns feature (dashboard) | |

Cadence: 30+ base releases in 20 months (≈ every 2–4 weeks) plus a parallel multilingual line.

---

## 6. Testing against GPTZero legitimately and at scale

**Cost math, 10,000 evaluations × 500 words = 5,000,000 words:**
- Professional plan (500k words included) + 4.5M overage × $0.00046 = $2,070 + $24.99–45.99 → **≈ $2,100 (≈ $0.21 per evaluation)**. Overage cap is 10M excess words, so this fits in one billing month; 1M+ overage triggers an upgrade prompt.
- EmpirioLabs reseller at $0.39/1k words → $1,950 (≈ $0.20/eval), no subscription.
- Time: 30,000 req/h limit → 10,000 calls in ~20 minutes at the limit; the mattc95 script ran 5 req/s with exponential backoff and hit two 403s in 1,000 calls.
- Research program: free API access "for academic researchers" on application (name, institution, publications, intended use, volume).

**Free tier** is not viable for scale: 7 scans/day × 5,000 chars (IP-tracked) or ~10k words/month depending on the report; no batch, fewer characters per request, no API key (only the in-docs "limited number of uses").

**Chrome extension / Google Docs.** "GPTZero: AI Detection & Writing Replay" (formerly Origin), v2026.8.3 (Sep 2 2026), ~400k users, 4.7★. Behaviours: scan any web text (select → right-click, or paste) via GPTZero's API; in Google Docs, live AI detection as you type, Writing Replay (time-lapse of revision history), copy-paste detection ("largest copy and pastes"), autotyper detection (Jun 26 2026: 98.5% accuracy, 99.9% precision, 97% recall across 7 autotypers — signals are typing pace, bursts, correction behaviour), rubric-driven comments, bibliography/hallucination checks; Writing Report shareable as link or PDF. Docs behaviour is process-based evidence and is orthogonal to the text classifier — a humanizer does not touch it. Microsoft Word add-in exists (support collection).

**Terms of Use (last updated Aug 21 2026), relevant clauses verbatim:**
- "you will not access the Site through automated or non-human means, whether through a bot, script or otherwise"
- "Systematically retrieve data or other content from the Site to create or compile, directly or indirectly, a collection, compilation, database"
- "use, launch, develop, or distribute any automated system, including without limitation, any spider, robot, cheat utility, scraper"
- "using any data mining, robots, or similar data gathering and extraction tools"
- "decipher, decompile, disassemble, or reverse engineer any of the software comprising or in any way making up a part of the Site"
- "use the Site as part of any effort to compete with us or otherwise use the Site and/or the Content for any revenue-generating endeavor"

There is no clause that names "training a model on our outputs" or "adversarial testing" as such; the operative restrictions are the anti-automation clause (which the paid API licence carves out for API calls), the compilation/database clause (which a labelled dataset of GPTZero scores arguably is), and the compete/revenue clause. Scraping the web app, the free landing-page scanner, or the extension's endpoints is squarely prohibited; using the paid API within the 30,000 req/h limit is the sanctioned path. GPTZero's own FAQ also says API-submitted text is not stored or used for product improvement — so the API path does not leak your evaluation corpus back into their training set, unlike dashboard scans, which are retained in aggregate.

---

## 7. What this means for building a GPTZero-targeted evaluation harness and a local surrogate detector

**Harness design.**
1. *Score on the right field.* Use `documents[0].class_probabilities` (all three) and `predicted_class`, plus `subclass.predicted_class` and its `confidence_category`. Ignore `completely_generated_prob`, `average_generated_prob`, `overall_burstiness`, and per-sentence `perplexity` except as legacy diagnostics. Reproduce GPTZero's own binary metric: Mixed, AI-paraphrased and Polished all count as "detected".
2. *Log `version` and `neatVersion` on every call and stratify results by them.* A model ships every 2–4 weeks; most reported "inconsistency" is version drift. Try `modelVersion: "<date>-base"` in the body to pin a version for a campaign, and verify the echo.
3. *Measure your own determinism* cheaply: resubmit a 200-doc subset 3× within one hour under one version before trusting any single-shot score.
4. *Respect the input contract*: ≥ 250 characters (aim for 300–800 words, where independent tests show the most stable readings); strip headers/frontmatter yourself (4.8b masks them anyway, so header padding is dead); keep prose-only test sets separate from list/code-heavy ones since formatting robustness is an explicit training target.
5. *Read the calibration.* `high` ≈ <1–2% error, `medium` ≈ 10%, `low` ≥ 14%; the remap pushes uncertain AI toward Human, so "passing" as `human`/`low` is much weaker evidence than `human`/`high`. Report the joint (class, confidence) distribution, not a pass rate.
6. *Budget* ≈ $0.21 per 500-word evaluation via API overage; 10k evaluations ≈ $2.1k and ~20 minutes at the rate limit. Keep live GPTZero as a periodic validator, not an inner-loop reward.

**Surrogate detector design (what to imitate).**
- Architecture target: a transformer encoder with a *hierarchical* head — 3-way document softmax {human, ai, mixed} with a nested {pure_ai, polished, ai_paraphrased} head under AI, plus a per-sentence binary head trained jointly (ℒ_d + α·ℒ_s). Do not derive Mixed by thresholding sentence means; GPTZero showed that is worse. Window long inputs and aggregate; apply a post-hoc ℝ³ remap fitted to minimise ECE with an FP penalty so your surrogate's "low confidence" behaves like theirs.
- Data recipe: mirror the domain mix (news and web articles dominate; academic, encyclopedia, essays, reviews, Q&A) and the augmentation tiers — LLM paraphrase prompts, translation chains (≤4 languages), Dipper/TempParaphraser outputs, outputs from the nine named bypassers, a polished class built with a Levenshtein-ratio band, gradient-guided synonym substitution as hard negatives, and human text passed through grammar tools (3.7b) as hard *positives*-for-human. De-bias formatting shortcuts (case, spacing, emoji, bold, lists, headers, dates, named entities) exactly as their release notes enumerate, or your surrogate will learn exploits GPTZero has already closed.
- Evaluate at the same operating point (recall at 1% FPR, plus AUC) on 1,000/1,000 per-domain sets with ≥250-char items, and track a separate 1,000-item bypasser set — GPTZero's public numbers (93.5% bypasser recall; 99.9 AUC) are the bar the surrogate must approximate to be a useful proxy.
- Expect the surrogate to lag: GPTZero retrains on disputed-prediction clusters and re-red-teams monthly, so agreement should be re-measured against the live API on every new `version`.
- Legal posture: generate labels only through the paid API (or the research programme), never by scraping the dashboard/extension; keep the resulting score dataset internal (the ToS compilation clause) and avoid commercialising anything positioned as a GPTZero competitor.

---

## Sources

GPTZero first-party
- Technology page — https://gptzero.me/technology
- Developers page — https://gptzero.me/developers
- ML engineers page — https://gptzero.me/machine-learning
- FAQ — https://gptzero.me/faq
- Pricing — https://gptzero.me/pricing
- Terms of Use (Aug 21 2026) — https://gptzero.me/terms-of-use.html
- Privacy Policy (Apr 2 2026) — https://gptzero.me/privacy-policy.html
- Chrome/Docs extension page — https://gptzero.me/chrome
- Researchers programme — https://gptzero.me/resources/researchers
- API docs (Stoplight, JS-only) — https://gptzero.stoplight.io/docs/gptzero-api/5bf295g49gwxp-gpt-zero-api ; https://gptzero.stoplight.io/docs/gptzero-api/0a8e7efa751a6-ai-detection-on-an-array-of-files
- Release Notes (Model and API), Notion — https://gptzero.notion.site/GPTZero-Release-Notes-Model-and-API-6f58686f6381498baef35212463b7da6
- Support: interpret API results — https://support.gptzero.me/articles/8947054519-how-do-i-use-and-interpret-the-results-from-your-api
- Support: probabilities → outcomes — https://support.gptzero.me/articles/4083713560-how-do-i-turn-the-probabilities-from-your-api-into-outcomes
- Support: AI subclass categories — https://support.gptzero.me/articles/7409387316-what-do-ai-subclass-categories-mean
- Support: rate limit — https://support.gptzero.me/articles/7371584464-what-is-the-rate-limit
- Support: overage — https://support.gptzero.me/articles/7102773231-what-happens-if-i-go-over-the-number-of-words-limit-of-my-plan
- Support: API data storage — https://support.gptzero.me/articles/3964919691-are-you-storing-data-from-api-calls
- Support: free API trial — https://support.gptzero.me/articles/7472477101-can-i-try-the-api-for-free
- Support: on-prem — https://support.gptzero.me/articles/8669075815-how-do-i-use-your-on-premise-api-deployment
- Support: GitHub — https://support.gptzero.me/articles/9145835224-what-is-your-github-address
- Support: advanced sentence scanning — https://support.gptzero.me/articles/7549392421-how-do-i-interpret-results-from-gptzero-s-advanced-sentence-scanning
- Support: moderate likelihood, no highlights — https://support.gptzero.me/articles/8694798756-my-text-has-a-a-moderate-likelihood-of-being-written-by-ai-but-i-do-not-see-any-yellow-highlights-on-the-text
- Support: Dec 29 2023 probability change — https://support.gptzero.me/articles/5087987409-why-does-the-ai-probability-look-different-on-gptzero-starting-december-29-2023
- Support: Jan 17 2024 UI — https://support.gptzero.me/articles/7719430400-how-do-i-interpret-gptzero-s-new-ui-since-january-17-2024-with-confidence-scores-and-mixed-results
- Support: paraphrasing/bypassers — https://support.gptzero.me/articles/5593633457-how-does-gptzero-detect-ai-paraphrasing-and-ai-bypassers
- Support: perplexity/burstiness — https://support.gptzero.me/articles/9585228410-how-do-i-interpret-burstiness-or-perplexity
- Support: limitations — https://support.gptzero.me/articles/1787085022-what-are-the-limitations-of-gptzero-s-ai-classifier
- Support: languages — https://support.gptzero.me/articles/1682612063-what-languages-does-gptzero-support
- Support: training data — https://support.gptzero.me/articles/2094904551-what-data-did-you-train-your-model-on
- Support: free vs paid — https://support.gptzero.me/articles/1272562776-what-is-the-difference-between-the-free-and-paid-for-plans
- Support: Origin features / Docs report / sharing — https://support.gptzero.me/articles/3030640853-what-features-does-origin-by-gptzero-have ; https://support.gptzero.me/articles/7001890416-what-is-the-google-docs-writing-report-for-origin ; https://support.gptzero.me/articles/8932684451-can-i-share-the-google-docs-writing-report
- News: Summer 2025 update (3.7b) — https://gptzero.me/news/gpt5/
- News: Meaningful AI text detection (4.8b headers) — https://gptzero.me/news/meaningful-ai-text-detection/
- News: Benchmarking standard (Feb 2026) — https://gptzero.me/news/gptzero-ai-detection-benchmarking-the-industry-standard-in-accuracy-transparency-and-fairness/
- News: How benchmarking works (2025) — https://gptzero.me/news/ai-accuracy-benchmarking/
- News: Chicago Booth 2026 — https://gptzero.me/news/chicago-booth-2026/
- News: Detecting humanized text (Jan 2026) — https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/
- News: AI paraphrasing detection (Nov 2024) — https://gptzero.me/news/ai-paraphrasing-detection/
- News: o1 benchmarking (Jan 2025) — https://gptzero.me/news/gptzero-o1-benchmarking/
- News: Deep learning model updates (Jul 2023) — https://gptzero.me/news/deep-learning-model-updates/
- News: Surpasses competitors (Aug 2023) — https://gptzero.me/news/gptzero-surpasses-competitors-in-accuracies/
- News: Multilingual update (May 2025) — https://gptzero.me/news/multilingual-detection-update-german-portuguese/
- News: Product updates Jan 29 2024 — https://gptzero.me/news/product-updates-january-29-2024/
- News: Understand your scan — https://gptzero.me/news/understand-gptzero-ai-scan/
- News: What AI range is acceptable — https://gptzero.me/news/what-ai-range-is-acceptable/
- News: AI Patterns (Aug 2026) — https://gptzero.me/news/ai-patterns/
- News: AI Vocabulary (Oct 2024) — https://gptzero.me/news/most-common-ai-vocabulary/
- News: Autotyper detection (Jun 2026) — https://gptzero.me/news/autotyper-detection-software/
- News: Quora benchmark (Jul 2026) — https://gptzero.me/news/gptzero-detects-ai-content-in-quora-posts-with-99-accuracy/
- News: Undetectable review (Jan 2025) — https://gptzero.me/news/undetectable-ai-review/
- News: EMNLP partnership (Jun 2026) — https://gptzero.me/news/emnlp-partners-with-gptzero/
- Job posting (Reach Capital) — https://jobs.reachcapital.com/companies/gptzero/jobs/43487597-machine-learning-engineer-ai-detection-toronto

Paper
- Adam, Cui, Thomas, Napier, Shmatko, Schnell, Tian, Dronavalli, Tian, Lee, "GPTZero: Robust Detection of LLM-Generated Texts", arXiv:2602.13042 (Feb 13 2026) — https://arxiv.org/abs/2602.13042 ; HTML https://arxiv.org/html/2602.13042

Third-party API documentation / code
- gptzeror R package — https://christophertkenny.com/gptzeror/ ; https://github.com/christopherkenny/gptzeror
- terminalskills GPTZero skill — https://terminalskills.io/skills/gptzero
- fast.io Hermes ↔ GPTZero guide — https://fast.io/resources/how-to-connect-hermes-agent-to-gptzero-api/
- jentic OpenAPI listing — https://jentic.com/apis/gptzero.com/gptzero
- EmpirioLabs reseller — https://empiriolabs.ai/models/gptzero
- mattc95 2026 AI Detector Benchmark (script + README) — https://github.com/mattc95/2026-AI-DETECTOR-BENCHMARK
- Chrome Web Store listing — https://chromewebstore.google.com/detail/kgobeoibakoahbfnlficpmibdbkdchap

Independent tests, pricing, and background
- Booth/BFI, "Artificial Writing and Automated Detection" (Oct 2025) — https://bfi.uchicago.edu/insights/artificial-writing-and-automated-detection/
- Verva, 3.15b humanizer test — https://verva.com/blog/does-gptzero-detect-ai-humanizers/
- compareaitools, repeated-scan variance (2026) — https://compareaitools.org/is-gptzero-accurate/
- MPG ONE 2026 retest — https://mpgone.com/is-gptzero-accurate-our-2025-test-results-here/
- WriteBros 2026 review — https://writebros.ai/blog/gptzero-detection-review
- Undetectable.ai accuracy test (Jun 2026) — https://undetectable.ai/blog/gptzero-accuracy-rate/
- PMC 12453642, accuracy–bias trade-offs (2025) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12453642/
- Stanford SCALE, assessing GPTZero — https://scale.stanford.edu/ai/repository/assessing-gptzeros-accuracy-identifying-ai-vs-human-written-essays
- ISCAP 2024, "Can GPTZero detect…" (seven components, free-tier limits) — https://www.iacis.org/iis/2024/3_iis_2024_165-174.pdf
- OMR pricing — https://omr.com/en/reviews/product/gptzero/pricing ; CASRAI — https://casrai.org/guides/gptzero-pricing
- stack-junkie free-tier comparison — https://www.stack-junkie.com/blog/zerogpt-vs-gptzero-free-ai-detector-comparison
- Originality.ai review (Superhuman acquisition) — https://originality.ai/blog/gptzero-ai-content-detection-review
- Wikipedia, GPTZero — https://en.wikipedia.org/wiki/GPTZero
- U of T, Alex Cui profile — https://web.cs.toronto.edu/news-events/news/gptzero-enters-the-chat-alum-alex-cui-and-his-breakthrough-in-ai-detection
