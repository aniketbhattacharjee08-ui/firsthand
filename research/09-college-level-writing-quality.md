# What Makes Writing "College Level" — And How To Measure And Protect It Automatically

Report 09. Companion to `15` (Hyland/Biber register norms), `16` (stylometric variance), `04` (AI-vs-human features), `14` (corpora). Runs long (~9.8k words) because the brief required verbatim descriptors from seven rubrics plus four reference tables; §7 and §8 are the operational payload if you read nothing else.

**The thesis of this report in one paragraph.** Every rubric that matters — AAC&U VALUE, WPA, GRE, IELTS, TOEFL, AP, and departmental rubrics — allocates the large majority of its weight to *argument, evidence, source use, organization and audience awareness*, and only a small residual to *sentence-level correctness*. Not one of them rewards syntactic complexity, long sentences, or rare vocabulary as ends in themselves; several explicitly *penalize* complexity that does not do work. Meanwhile, the empirical literature shows that the surface features a naive "improver" would push — longer T-units, more subordination, more explicit connectives, higher type-token ratio — either do not predict quality or predict it *negatively* in academic registers. This is a direct hazard for a humanizer: the edits that most reliably lower AI-detector scores (contractions, fragments, hedged informality, first person, deliberate imperfection) intersect the small but non-negotiable band of the rubric that *is* surface-level. The design conclusion is that quality gating must be **rubric-shaped and asymmetric**: measure argument/organization/source integration and refuse to degrade them; measure mechanics and enforce a *floor* (not zero errors); and treat lexical/syntactic "sophistication" as a *band to stay inside*, never a quantity to maximize.

---

## 1. Rubrics: what graders actually reward

### 1.1 The dimension table

Rubric dimensions across the seven instruments most relevant to US/UK university writing. Weighting is the share of the total score under that heading where the instrument makes it explicit; "—" means the instrument folds it into a holistic judgment.

| Construct | AAC&U VALUE (Written Comm.) | WPA Outcomes v3.0 | GRE Analytical Writing | IELTS Writing Task 2 | TOEFL iBT Writing | AP Eng. Lang. FRQ | Typical dept. essay rubric |
|---|---|---|---|---|---|---|---|
| Purpose / task response / audience | **Context of and Purpose for Writing** | Rhetorical Knowledge | "addresses the task", "sustained/insightful position" | **Task Response** (25%) | "addresses the task" | Thesis row (1/6) | Thesis & focus (15–25%) |
| Idea development / reasoning | **Content Development** | Critical Thinking, Reading, Composing | "cogent, well-articulated ... compelling reasons/examples" | folded into Task Response | "well-developed, well-organized explanation/argument" | Evidence **and** Commentary (4/6) | Argument & analysis (30–40%) |
| Organization / cohesion | folded into Content Development | Rhetorical Knowledge + Processes | "well-focused, well-organized" / "clearly connects ideas" | **Coherence and Cohesion** (25%) | "well-organized", "connection of ideas" | Line of reasoning (inside Evidence/Commentary) | Organization (15–20%) |
| Source use / evidence / citation | **Sources and Evidence** | Critical Thinking + Knowledge of Conventions (citation) | n/a (closed-book) | n/a | integrated task only | 2–3 of the provided sources (Q1) | Use of sources / citation (10–20%) |
| Genre & disciplinary convention | **Genre and Disciplinary Conventions** | Knowledge of Conventions | n/a | register/tone under Lexical Resource | n/a | n/a | Format & style (5–10%) |
| Lexis | inside Genre/Control rows | Knowledge of Conventions | "skillful use of language", "effective vocabulary" | **Lexical Resource** (25%) | "appropriate word choice" | inside Sophistication row | Style/word choice (5–15%) |
| Syntax, grammar, mechanics | **Control of Syntax and Mechanics** | Knowledge of Conventions | "facility with the conventions ... minor errors" | **Grammatical Range and Accuracy** (25%) | "syntactic variety ... minor lexical or grammatical errors" | gate on the 4th Evidence point | Mechanics (5–15%) |
| "Sophistication" / insight | Capstone-level language in every row | — | score-6 language | — | — | **Sophistication row (1/6)** | "exceeds expectations" band |

Two structural facts jump out. First, **on exam rubrics mechanics is 25% at most and usually much less; on classroom rubrics it is usually 5–15%.** Second, **the top band on every instrument is defined by thought, not by surface** — GRE 6 is "insightful", AP's sixth point is "sophistication of thought", VALUE's Capstone is "a thorough understanding of context, audience and purpose".

### 1.2 AAC&U VALUE — Written Communication

Five dimensions, four performance levels (4 Capstone, 3–2 Milestones, 1 Benchmark):

1. **Context of and Purpose for Writing** — including considerations of audience, purpose, and the circumstances surrounding the writing task.
2. **Content Development**
3. **Genre and Disciplinary Conventions** — formal and informal rules inherent in the expectations for writing in particular forms and/or academic fields.
4. **Sources and Evidence**
5. **Control of Syntax and Mechanics**

The rubric's own framing matters for us: AAC&U states the rubrics are "intended for institutional-level use in evaluating and discussing student learning, **not for grading**" ([AAC&U VALUE rubrics](https://www.aacu.org/initiatives/value-initiative/value-rubrics/value-rubrics-written-communication)). The progression across levels within each row is consistently *from "attempts to use" → "demonstrates consistent use of" → "demonstrates detailed attention to and successful execution of"*. The Control of Syntax and Mechanics row tops out not at "error-free" but at language that "skillfully communicates meaning to readers with clarity and fluency, and is virtually error-free." **"Virtually" is doing load-bearing work: the top band of the most widely adopted US written-communication rubric does not require zero errors.**

### 1.3 WPA Outcomes Statement for First-Year Composition, v3.0 (2014)

Verbatim bullets, from the [Council of Writing Program Administrators](https://wpacouncil.org/aws/CWPA/pt/sd/news_article/243055/_PARENT/layout_details/false). This is the closest thing to a national definition of "college-level writing" in the US, and it is the outcomes document most first-year composition rubrics are derived from.

**Rhetorical Knowledge** — "Learn and use key rhetorical concepts through analyzing and composing a variety of texts"; "Gain experience reading and composing in several genres to understand how genre conventions shape and are shaped by readers' and writers' practices and purposes"; "Develop facility in responding to a variety of situations and contexts calling for purposeful shifts in voice, tone, level of formality, design, medium, and/or structure"; plus technology/audience matching.

**Critical Thinking, Reading, and Composing** — "Use composing and reading for inquiry, learning, critical thinking, and communicating in various rhetorical contexts"; read "attending especially to relationships between assertion and evidence, to patterns of organization"; "Locate and evaluate (for credibility, sufficiency, accuracy, timeliness, bias and so on) primary and secondary research materials"; "Use strategies—such as interpretation, synthesis, response, critique, and design/redesign—to compose texts that integrate the writer's ideas with those from appropriate sources".

**Processes** — multiple drafts; flexible strategies for "reading, drafting, reviewing, collaborating, revising, rewriting, rereading, and editing"; acting on feedback; reflection.

**Knowledge of Conventions** — grammar/punctuation/spelling "through practice in composing and revising"; "Understand why genre conventions for structure, paragraphing, tone, and mechanics **vary**"; "Gain experience negotiating variations in genre conventions"; "Practice applying citation conventions systematically in their own work".

Note what is absent: no sentence-length target, no vocabulary-level target, no prohibition on first person, no requirement of error-free prose. The one explicit statement about surface form is that conventions **vary by genre** — which is the WPA's own argument against a fixed style checker.

### 1.4 GRE Analytical Writing

Six-point holistic scale, half-point increments after averaging two raters (one of which is ETS's e-rater). The band logic, paraphrased from the ETS scoring guide:

- **6 (Outstanding)** — "articulates a clear and insightful position", develops it with "compelling reasons and/or persuasive examples", is "well-focused and well-organized", "conveys ideas fluently and precisely, using effective vocabulary and sentence variety", and "demonstrates superior facility with the conventions of standard written English … but may have minor errors."
- **5 (Strong)** — clear, well-considered position; logically sound development; clear control of language; "may have minor errors".
- **4 (Adequate)** — clear position, "competent" development, adequate control; "may have some errors".
- **3 (Limited)** — "limited development"; "problems in the use of language and sentence structure that result in a lack of clarity."
- **2 / 1** — errors that "frequently interfere with meaning" / "result in incoherence."

The A-vs-B boundary here is **6 vs 5 = insight and precision, not correctness**; every band from 4 to 6 tolerates errors. The C boundary (3) is where errors start "interfering with meaning."

### 1.5 IELTS Writing Task 2 — four equally weighted criteria

Task Response, Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy, each 25% ([IELTS scoring in detail](https://ielts.org/organisations/ielts-for-organisations/ielts-scoring-in-detail)). Two details that matter directly to a humanizer:

- **Coherence and Cohesion at Band 9** requires cohesion "used in such a way that it attracts no attention". Band 6 allows cohesion that is "faulty or mechanical"; Band 5 explicitly penalizes "the overuse or underuse of cohesive devices". **Over-signposting is a mid-band symptom, not a high-band one** — the same finding as Crossley & McNamara (§4.3). This is the single most important rubric fact for anyone tempted to bolt "Furthermore," onto every third sentence.
- **Grammatical Range and Accuracy at Band 8** allows "the majority of sentences are error-free" with "occasional errors"; Band 7 allows "frequent error-free sentences". The top of the scale is not the absence of errors, it is the *dominance* of error-free sentences.

### 1.6 TOEFL iBT Writing

Five-point rubrics. The top band (5) requires "consistent facility in the use of language", "syntactic variety", "precise word choice" and **idiomaticity** — while still allowing "minor lexical or grammatical errors." Band 3 is where errors become "noticeable". TOEFL is the instrument that most explicitly names idiomaticity as a top-band property: one of the few places where "sounding human" and "scoring high" are the same axis.

### 1.7 AP English Language and Composition (Effective Fall 2019) — verbatim

Six points, three rows. This is the most operationally precise public rubric available, and it is worth quoting because its decision rules read almost like a spec ([AP Central scoring rubrics PDF](https://apcentral.collegeboard.org/media/pdf/ap-english-language-and-composition-frqs-1-2-3-scoring-rubrics.pdf)).

**Row A — Thesis (0–1).** One point for "Responds to the prompt with a thesis that presents a defensible position." Zero for responses that "Only restate the prompt", "Do not take a position, or the position is vague or must be inferred", "Equivocate or summarize other's arguments but not the student's (e.g., some people say it's good, some people say it's bad)", or "State an obvious fact rather than making a claim that requires a defense."

> That "equivocate … some people say it's good, some people say it's bad" bullet is a direct description of default LLM prose. A balanced-perspectives essay scores **zero** on the thesis row.

**Row B — Evidence AND Commentary (0–4).** The ladder is: 1 = evidence present, commentary merely "Summarizes the evidence but does not explain how the evidence supports the student's argument"; 2 = "Explains how some of the evidence relates to the student's argument, but no line of reasoning is established, or the line of reasoning is faulty"; 3 = "Provides specific evidence to support all claims in a line of reasoning" AND "Explains how some of the evidence supports a line of reasoning"; 4 = same evidence standard AND "**Consistently** explains how the evidence supports a line of reasoning."

Typical 2-point responses "Consist of a mix of specific evidence and broad generalities." Typical 4-point responses "Organize and support an argument as a line of reasoning composed of multiple supporting claims, each with adequate evidence that is clearly explained."

**The mechanics gate, verbatim:** "Writing that suffers from grammatical and/or mechanical errors that interfere with communication cannot earn the fourth point in this row." This is the exact shape a quality gate should take — not "no errors" but "**no errors that interfere with communication**", and the penalty is one point out of six.

**Row C — Sophistication (0–1).** Earned by "Explaining the significance or relevance of the writer's rhetorical choices", "Explaining a purpose or function of the passage's complexities or tensions", or "Employing a style that is consistently vivid and persuasive." **Explicitly not earned** by responses that consist of "sweeping generalizations", "Only hint at or suggest other arguments", "Oversimplify complexities", or — critically — "**Use complicated or complex sentences or language that is ineffective because it does not enhance the analysis.**"

That last bullet is the single best rubric-level statement of the counterintuitive finding in §5: **complexity for its own sake is affirmatively listed as a reason to withhold the top point.**

### 1.8 Departmental rubrics: the recurring shape

Real university essay rubrics converge on a five-to-six-criterion analytic grid with A/B/C/D columns:

| Criterion | Typical weight | "A" descriptor | "B/C" descriptor |
|---|---|---|---|
| Thesis / argument | 20–25% | Arguable, precise, non-obvious; sustained and complicated across the essay | Present but general, descriptive, or restated rather than developed |
| Evidence & analysis | 30–40% | Specific, well-chosen evidence; analysis exceeds summary; addresses counter-evidence | Evidence present but under-analyzed; "quote drop"; summary substitutes for analysis |
| Structure | 15–20% | Paragraphs make claims; order is motivated by the argument, not the sources | Formulaic five-paragraph or source-by-source order; transitions announce rather than connect |
| Sources / citation | 10–15% | Sources are put in conversation; integration is syntactic, not appended; citation clean | Sources cited but stacked; citation format inconsistent |
| Style / mechanics | 5–15% | Clear, economical, discipline-appropriate; errors negligible | Errors distract; wordiness; register slips |

The reliable A/B discriminator is **whether analysis exceeds summary and whether the claim is arguable**; the B/C discriminator is **whether evidence is specific at all**. Mechanics rarely moves a paper by a full letter.

### 1.9 What concretely distinguishes an A from a B/C

Synthesizing across all seven instruments, in descending order of leverage:

1. **The claim is contestable and specific** (AP "requires a defense"; GRE "insightful"). A summary-shaped or both-sides thesis caps the paper.
2. **Commentary is consistent, not spot-checked.** AP moves 3→4 purely on "consistently explains" — a *coverage* property over every piece of evidence.
3. **A line of reasoning exists across paragraphs** (paragraph N's claim depends on N−1's). Missing this is AP's 2-point band.
4. **Evidence is specific rather than "a mix of specific evidence and broad generalities"** (AP's literal 2-point description).
5. **Sources are integrated syntactically and put in dialogue** (VALUE Sources and Evidence).
6. **Register fits genre and audience** (VALUE Genre Conventions; WPA "purposeful shifts in … level of formality").
7. **Cohesion is invisible** (IELTS Band 9), not signposted.
8. **Errors do not interfere with communication.** Last, and explicitly capped in effect.

### 1.10 A ready-made 6-point holistic rubric you can use directly

The rubric used for the ~24,000 essays in ASAP 2.0 / PERSUADE is public and is a serviceable scorer spec ([Holistic Rating for Source-Based Writing](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf)):

> **SCORE OF 6:** "demonstrates clear and consistent mastery, although it may have a few minor errors … effectively and insightfully develops a point of view … outstanding critical thinking; … clearly appropriate examples, reasons, and other evidence taken from the source text(s) …; well organized and clearly focused, demonstrating clear coherence and smooth progression of ideas; … skillful use of language, using a varied, accurate, and apt vocabulary and … meaningful variety in sentence structure; the essay is **free of most errors** in grammar, usage, and mechanics."

> **SCORE OF 2:** "…errors in grammar, usage, and mechanics **so serious that meaning is somewhat obscured**."

Top band = "free of most errors," not error-free. Errors only become operative at 2 and 1, where they "obscure" or "persistently interfere with" meaning. **A rubric-faithful gate is a communication-interference gate, not an error-count gate.**

---

## 2. Automated Essay Scoring: where the state of the art actually is

### 2.1 The datasets

| Dataset | Size | Scores | Notes |
|---|---|---|---|
| **ASAP** (Kaggle 2012, Hewlett) | 12,978 essays, 8 prompts, grades 7–10 | prompt-specific ranges (2–12, 1–6, 0–3, 0–60 …) | Still the standard benchmark. Three genres: narrative/descriptive, persuasive, source-dependent. Named entities anonymized (`@CAPS1`, `@PERSON1`). |
| **ASAP++** (Mathias & Bhattacharyya 2018) | same essays | multi-trait | Adds Content, Organization, Word Choice, Sentence Fluency, Conventions where applicable |
| **ASAP-SAS** | short answers, 10 questions | 0–3 | Short-answer sibling |
| **PERSUADE 1.0 / 2.0** (Crossley et al. 2022) | 25k+ argumentative essays, grades 6–12; 2.0 = 15,593 with demographics | discourse-element spans + holistic 1–6 | Spans: Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement — each also rated Ineffective/Adequate/Effective |
| **ELLIPSE** (Crossley et al. 2023) | ~6,500 ELL essays, grades 8–12 | overall + analytic: cohesion, syntax, vocabulary, phraseology, grammar, conventions | Double-rated, 27 raters. Best public *analytic* corpus. |
| **ASAP 2.0 / Learning Agency Lab (Kaggle 2024)** | ~24,000 argumentative essays; ~8k hidden test | holistic 1–6 | PERSUADE-derived. Metric: QWK. |
| **TOEFL11** | 12,100 essays, 8 prompts, 11 L1s | Low/Medium/High | Also the NLI shared-task corpus |
| **CLC-FCE**, **ICLE**, **AAE** | learner corpora | holistic + error types / organization / persuasiveness | |

Feedback Prize was the three-competition series (Evaluating Student Writing → span segmentation; Predicting Effective Arguments → span quality; English Language Learning → ELLIPSE analytic scores) that produced PERSUADE and ELLIPSE. For a humanizer, **PERSUADE's span annotations are the most directly useful artifact in this literature**: they support an argument-structure detector reporting "Position, 4 Claims, 1 Counterclaim, 1 Rebuttal, 6 Evidence spans" — a defensible rubric dimension, not a proxy.

### 2.2 QWK state of the art

QWK = quadratic weighted kappa: a weighted kappa penalizing each disagreement by the *squared* score distance, so near-misses cost little.

| System | Year | Setting | ASAP avg QWK | Notes |
|---|---|---|---|---|
| EASE (SVR) | 2012 | supervised, handcrafted features | **0.699** | The open-source ASAP-competition engine |
| EASE (BLRR) | 2012 | supervised | **0.705** | |
| ALL-MTL-cTAP (Cummins et al.) | 2016 | multi-task pairwise preference | 0.747 | |
| CNN+LSTM (Taghipour & Ng) | 2016 | first neural AES | 0.761 | |
| LSTM-CNN-attention (Dong et al.) | 2017 | hierarchical + attention | 0.764 | |
| SkipFlow (Tay et al.) | 2018 | neural coherence features | 0.764 | |
| HISK+BOSWE (Cozma et al.) | 2018 | string kernels + embeddings | 0.785 | |
| R²BERT (Yang et al.) | 2020 | BERT, regression+ranking multi-loss | 0.794 | |
| Tran-BERT-MS-ML-R (Wang et al.) | 2022 | multi-scale BERT | ~0.791 | |
| **NPCR (Xie et al.)** | 2022 | pairwise contrastive regression on BERT-base | **0.817** | Per-prompt 0.750–0.858 |
| **Kaggle ASAP 2.0 top private LB** | 2024 | DeBERTa-v3 ensembles + GBDT on ~24k essays | **≈0.838–0.840** | 5th place 0.83743; gold-medal band 0.8359–0.8374 |
| Fine-tuned RoBERTa on PERSUADE 2.0 | 2026 | supervised | **0.841** | reported in a hybrid-scoring fairness audit |

**LLM-as-judge, 2023–2026:**

| Approach | Dataset | QWK | Source |
|---|---|---|---|
| GPT-4 zero-shot, rubric in prompt | ASAP | **near-random on several prompts** (Set 1 = 0.042, Set 7 = 0.081) | Xiao et al. 2024 |
| GPT-4 few-shot | ASAP | improved but "significantly lagged behind SOTA" | Xiao et al. 2024 |
| Fine-tuned GPT-3.5 / LLaMA3-8B | ASAP | "generally exceeding 0.7" but below traditional SOTA | Xiao et al. 2024 |
| Vanilla prompting (ChatGPT) | ASAP / TOEFL11 | 0.455 / 0.463 | Lee et al. 2024 (MTS) |
| Vanilla prompting (Llama2-13b-chat) | ASAP / TOEFL11 | 0.205 / 0.238 | Lee et al. 2024 |
| **Multi Trait Specialization (MTS)** — decompose into traits, score each in its own conversation with quote retrieval, average, min-max scale with outlier clipping | ASAP / TOEFL11 | **0.560 / 0.567** (Llama2-13b); **0.550 / 0.587** (Mistral-7B) | Lee et al. 2024 |
| GPT-4 with calibration examples, CEFR rating of short L2 essays | proprietary | "almost as well as modern AWE methods", varies by L1 | Yancey et al. 2023 |
| GPT-4 short-answer scoring | ASAP-SAS | 0.677 | Ormerod et al. 2024 |
| Zero-shot LLMs on PERSUADE 2.0 | PERSUADE 2.0 | 0.565 and 0.216 | 2026 fairness audit |
| GPT-4 discourse-coherence rating | expert-rated corpus | "comparable to human ratings", outperforms traditional NLP coherence metrics | Naismith et al. 2023 |

**Human-rater-equivalent QWK.** The cleanest anchor comes from Xiao et al. (2024), who measured graders on the same essays: **expert graders averaged QWK 0.712; novice graders 0.526; novices assisted by LLM feedback rose to 0.661, statistically indistinguishable from experts (p = 0.27).** So:

- **QWK ≈ 0.70–0.75 is human-expert-equivalent** for holistic essay scoring.
- **QWK ≈ 0.80+** (supervised, in-prompt) is *above* typical human double-scoring — which mainly tells you the model has learned prompt-specific regularities, not that it "understands" writing.
- **QWK < 0.55** is novice-grader territory and is roughly where naive zero-shot LLM prompting lands.

Caveats. QWK is sensitive to rating scale, prevalence, and the kappa paradox (a 2023 critique argues against using it alone). Li & Ng (EMNLP 2024) add that the field is over-fitted to ASAP: within-prompt scorers deteriorate considerably on new prompts, and neural models are uninterpretable, so a rising QWK says little about *what* was learned. **For a humanizer's gate, an interpretable rubric-trait scorer at QWK ~0.65 beats an uninterpretable holistic scorer at 0.84** — you need to know which dimension you broke. Mayfield & Black (BEA 2020) also found fine-tuning BERT "produces similar performance to classical models at significant additional cost": feature-based scorers remain competitive and are debuggable.

### 2.3 Open models on HuggingFace

There is no strong, well-maintained open AES model. What exists: (a) **Kaggle ASAP 2.0 solution artifacts** — DeBERTa-v3 regression heads plus LightGBM/XGBoost stacks over handcrafted features (spelling counts, paragraph counts, length quantiles), ~0.84 QWK but competition code, not packaged models; (b) small IELTS band classifiers such as `KevSun/IELTS_essay_scoring` and `JacobLinCool/IELTS_essay_scoring_safetensors` (~0.1B, low downloads — demos); (c) assorted DistilBERT/RoBERTa/Long-T5/SmolLM fine-tunes (~32 Hub models match "essay scoring", mostly hobby-scale); (d) **Feedback Prize / PERSUADE discourse-element taggers** — public DeBERTa token classifiers for the 7 span types. Category (d) is the most reusable, because argument structure carries the most rubric weight.

**Recommendation:** do not adopt a Hub AES model as a gate. Train a trait scorer on **ELLIPSE** (analytic: cohesion, syntax, vocabulary, phraseology, grammar, conventions — exactly the six axes a humanizer can damage) and **PERSUADE** (holistic + spans), and validate with QWK against held-out human scores. Require QWK ≥ 0.65 per trait before it gates anything.

---

## 3. Measurable correlates of writing quality: a feature catalog

### 3.1 The tool landscape

| Tool | What it measures | Indices | Install / access |
|---|---|---|---|
| **Coh-Metrix 3.0** | 106+ indices across 11 categories: descriptive, text easability principal components, referential cohesion, LSA, lexical diversity, connectives, situation model, syntactic complexity, syntactic pattern density, word information, readability | 106 | Web tool (cohmetrix.com); no pip. Closest open substitutes: `lingfeat`, `LFTK` |
| **TAALES** (Kyle & Crossley) | Lexical sophistication: word frequency, word range, n-gram frequency, n-gram range, **n-gram strength of association (MI, T-score, delta-P)**, contextual distinctiveness, word recognition norms, semantic network, word neighbours, academic-word-list coverage | **400+** | Desktop app, [linguisticanalysistools.org/taales.html](https://www.linguisticanalysistools.org/taales.html) |
| **TAACO** (Crossley, Kyle, Dascalu) | Cohesion: local (adjacent-sentence) and global (paragraph, whole-text) lexical overlap, TTR variants for POS/lemma/bigram/trigram, **connective indices**, LSA/LDA/word2vec semantic similarity | **150** | Desktop app |
| **TAASSC** (Kyle) | Syntactic complexity: full L2SCA suite + **fine-grained phrasal** (adjectives per NP, prepositional phrases per NP…) and **fine-grained clausal** (adverbials per clause, complements per clause…) indices + **verb argument construction (VAC) frequency/association** | ~400 | Desktop app |
| **TAALED** (Kyle, Crossley, Jarvis) | Lexical diversity, POS-disambiguated, lemma-based, split by all/content/function lemmas | dozens | Desktop app + `pip install taaled` |
| **L2SCA** (Lu 2010) | The 14 classic syntactic complexity measures | 14 | Original Python; modern reimplementation `pip install neosca` |
| **LCA** (Lu 2012) | 25 lexical complexity measures | 25 | Also in NeoSCA |
| **GAMET** (Crossley et al. 2019) | Grammar, spelling, punctuation, whitespace, repetition error incidence, with line-level output | ~30 | Desktop app; wraps LanguageTool |
| **SEANCE** | Sentiment, social cognition, affect (250+ from 8 lexicons) | 250+ | Desktop app |
| **CRAT / ARTE / SiNLP / TAADA / TAMMI** | cohesion+lexical hybrid; automated readability; simple NLP counts; corpus download; multimodal | varies | Desktop apps |

### 3.2 Pip-installable stack (verified on PyPI, Sept 2026)

| Package | Version | Use |
|---|---|---|
| `neosca` | 0.0.55 | L2SCA (14 syntactic complexity indices) + LCA in Python; wraps Stanford Parser/Tregex |
| `lftk` | 1.0.9 | 220+ handcrafted linguistic features over spaCy docs (lexical, syntactic, discourse, readability, word-family) |
| `lingfeat` | 1.0.0b19 | 255 features across Advanced Semantic (LDA), Discourse (entity grid/density), Syntactic, Lexico-Semantic, Shallow Traditional — the closest open analogue of Coh-Metrix |
| `lexical-diversity` | 0.1.1 | Kyle's MTLD, HD-D, MATTR, MSTTR, Maas, vocd-D |
| `lexicalrichness` | 0.5.1 | Same family, plus Herdan/Summer/Dugast/Yule's K |
| `taaled` | 0.32 | Kyle's full TAALED as a library |
| `pylats` | 0.64 | Kyle's preprocessing layer (lemmatization/POS normalization) that the SALAT tools assume |
| `language_tool_python` | 3.4.0 | LanguageTool: 5,000+ rules, self-hostable server, LGPL |
| `errant` | 3.0.2 | ERRANT — aligns original/corrected text and assigns 25 error *types* (M/U/R × POS). This is how you get an error *taxonomy*, not just a count |
| `wordfreq` | 3.1.1 | Zipf-scale word frequencies for lexical sophistication |
| `textstat` | 0.7.13 | Flesch–Kincaid, Gunning fog, SMOG, Dale–Chall, Coleman–Liau |
| `spacy` + `benepar` / `stanza` / `supar` | — | Constituency and dependency parses; needed for T-units and NP modification |
| `sentence-transformers` | — | Embedding-based local/global cohesion |

### 3.3 Computation recipes

**(a) Syntactic complexity — the Lu (2010) L2SCA 14.** Length of production (MLS, MLT, MLC = mean length of sentence / T-unit / clause); sentence complexity (C/S); subordination (C/T, CT/T, DC/C, DC/T); coordination (CP/C, CP/T, T/S); particular structures (CN/C, CN/T, VP/T).

```python
# pip install neosca; requires Java for the Stanford parser
from neosca.scaio import SCAIO           # or: nsca --text "..." --output out.csv
# CLI:  nsca essay.txt -o out.csv
# columns: W S VP C T DC CT CP CN MLS MLT MLC C/S VP/T C/T DC/C DC/T T/S CT/T CP/T CP/C CN/T CN/C
```

A **T-unit** = one main clause plus all attached subordinate clauses. Hunt (1965) designed it to measure *child* language development in narrative writing — keep that provenance in mind (§4).

**(b) Fine-grained phrasal complexity — the measures that actually track academic register.** NP pre- and post-modification: attributive adjectives/NP, noun premodifiers/NP, PP postmodifiers/NP, appositives, relative clauses/NP, *of*-phrases/NP. TAASSC computes these; spaCy approximates:

```python
import spacy; nlp = spacy.load("en_core_web_trf")
def np_modification(doc):
    nps = list(doc.noun_chunks)
    adj  = sum(1 for np in nps for t in np if t.pos_=="ADJ" and t.dep_=="amod")
    nn   = sum(1 for np in nps for t in np if t.pos_=="NOUN" and t.dep_=="compound")
    pp   = sum(1 for np in nps for t in np.root.children if t.dep_=="prep")
    rel  = sum(1 for np in nps for t in np.root.children if t.dep_ in ("relcl","acl"))
    n = max(len(nps),1)
    return dict(adj_per_np=adj/n, nn_per_np=nn/n, pp_per_np=pp/n, relcl_per_np=rel/n)
```

**(c) Lexical sophistication.** Four families with different behaviour:

1. **Frequency.** Mean log frequency of content words vs COCA/SUBTLEX (`wordfreq.zipf_frequency`). Predicts quality *weakly and non-monotonically*.
2. **Range.** Number of corpus documents/registers a word occurs in — usually a *better* predictor than raw frequency (TAALES supplies both).
3. **Academic word coverage.** % tokens in Coxhead's AWL (570 families) or, better, the **Academic Vocabulary List** (Gardner & Davies 2014, ~3,000 lemmas from COCA's 120M-word academic sub-corpus). Lemmatize, drop the GSL/BNC top-2000, compute `% tokens in AVL`. Published academic prose ≈ 9–14%; weak student prose < 5%.
4. **N-gram association strength** — the sophistication measure that best separates fluent from stilted prose, and one AI text is *unusually good* at, so a quality-safe direction. For each bigram/trigram, look up COCA and compute:
   - **MI** = log₂( f(xy) · N / (f(x)·f(y)) ) — rewards *exclusive* collocations ("tantamount to", "wreak havoc"). MI ≥ 3 is conventionally "collocation".
   - **T-score** = ( f(xy) − f(x)f(y)/N ) / √f(xy) — rewards *frequent* collocations ("in the", "of the"). T ≥ 2 conventional.
   - **Delta-P** — directional, asymmetric association; TAALES reports both directions.
   High mean MI with low mean T-score = showy but unidiomatic. High mean T-score with moderate MI = fluent, native-like. **Track both; the target is native-academic joint distribution, not a maximum.**

**(d) Lexical diversity.** Never raw TTR (a function of length). Use **MTLD** (mean length of sequential strings maintaining TTR ≥ 0.72, forward+backward averaged), **HD-D** (summed hypergeometric probability of each type appearing in a random 42-token sample), and **MATTR** (moving-average TTR, window 50).

```python
from lexical_diversity import lex_div as ld
toks = ld.flemmatize(text)
mtld, hdd, mattr = ld.mtld(toks), ld.hdd(toks), ld.mattr(toks, window_length=50)
```
Zenker & Kyle (2021) is the reference for minimum text lengths per index — most indices are unstable below ~200 tokens, which matters if you gate per-paragraph.

**(e) Cohesion.** Three constructs a naive implementation conflates: **referential overlap** (share of adjacent sentence pairs sharing ≥1 content lemma — TAACO `adjacent_overlap_*`); **semantic cohesion** (mean cosine between consecutive sentence embeddings = local; mean cosine to document centroid = global); **connective density** (incidence/1,000 words by causal / logical / adversative / temporal / additive — TAACO ships the lists; Coh-Metrix `CNCAll`, `CNCCaus`, `CNCLogic`, `CNCADC`, `CNCTemp`, `CNCAdd`).

```python
from sentence_transformers import SentenceTransformer
import numpy as np
m = SentenceTransformer("all-mpnet-base-v2")
E = m.encode(sentences, normalize_embeddings=True)
local  = float(np.mean([E[i] @ E[i+1] for i in range(len(E)-1)]))
c = E.mean(0); c /= np.linalg.norm(c)
glob   = float(np.mean(E @ c))
```
Read §4 before you use any of these as an objective to maximize.

**(f) Argumentation structure.** Train or reuse a PERSUADE-style token classifier over the 7 span labels, then compute structural features that map onto rubric language:
- has_position (AP Row A gate), n_claims, n_evidence, **evidence_per_claim**, has_counterclaim, has_rebuttal, counterclaim_answered (a Rebuttal follows within k spans of a Counterclaim)
- **commentary_ratio** = tokens in Claim/Rebuttal spans ÷ tokens in Evidence spans. AP's 3→4 boundary is "consistently explains", i.e. coverage of Evidence by nearby Claim/commentary. Compute `pct_evidence_spans_with_adjacent_commentary`.
- Order sanity: Lead → Position → (Claim, Evidence)+ → Counterclaim → Rebuttal → Concluding Statement.

**(g) Source integration.** Citations per 1,000 words; *integral* ("Hyland (2005) argues…") vs *non-integral* ("(Hyland, 2005)"); quotation ratio; **quote-drop rate** (quotations forming a whole sentence with no framing clause — rubrics penalize this explicitly); distinct-source count; and whether ≥2 sources co-occur within a sentence or adjacent pair (synthesis vs stacking).

---

## 4. The counterintuitive findings: why "improving" writing can lower the grade

The most important section for the product. Every result says the same thing: **the naive direction of "more sophisticated" is often the wrong direction.**

### 4.1 T-units and subordination measure *conversation*, not academic writing

Biber, Gray & Poonpon (2011), *TESOL Quarterly* 45(1), compared 28 grammatical complexity features across 429 research articles (2.94M words; biology, education, history, medicine, psychology) and 723 conversations (4.18M words). Verbatim from the abstract:

> "The results are surprising, showing that most clausal subordination measures are actually **more common in conversation than academic writing**. In contrast, fundamentally different kinds of grammatical complexity are common in academic writing: complex **noun phrase constituents** (rather than clause constituents) and complex **phrases** (rather than clauses)."

And on the T-unit itself: "measures of subordination capture only one kind of grammatical complexity, while the T-unit **confounds a wide range of different devices** that can be used to create complexity."

Their **hypothesized developmental sequence** (Table 7) is directly usable as a target ladder. Abbreviated:

| Stage | Structures | Register they belong to |
|---|---|---|
| 1 | Finite *that*/WH complement clauses controlled by very common verbs (*think, know, say*) | conversation |
| 2 | Finite complement clauses with a wider verb set; **finite adverbial clauses** (*if, because, although*); nonfinite complements after *want*; adverbs as adverbials; attributive adjectives | conversation |
| 3 | PPs as adverbials; complement clauses controlled by adjectives; *that*-relatives with animate heads; nouns as premodifiers; possessive premodifiers; *of*-phrases as postmodifiers; concrete PPs as postmodifiers | fiction |
| 4 | Nonfinite complements controlled by adjectives (*easy to obtain*); **extraposed complement clauses** (*It is clear that…*); **nonfinite relative clauses** (*the method used here*, *studies employing…*); denser attributive adjectives and noun premodifiers; **abstract** PPs as postmodifiers | academic |
| 5 | Preposition + nonfinite complement (*the idea of using…*); **noun-controlled complement clauses** (*the hypothesis that…*); **appositive NPs**; **multiple embedded PP postmodifiers** | academic |

Read the ladder backwards and you get a list of edits that *lower* register: *It is clear that* → *Clearly, we can see*; *studies employing X* → *studies that employ X*; *the claim that* → *they claim that*; appositives split into separate sentences; abstract *of*-postmodification unpacked into *because*-clauses. **Every one is a plausible "humanizing" edit, and every one moves the text from Stage 4–5 down to Stage 2–3.**

### 4.2 Fine-grained phrasal indices beat MLTU at predicting quality

Kyle & Crossley (2018), *MLJ* 102(2): across four studies pitting MLTU against fine-grained clausal and phrasal indices as predictors of holistic quality, the fine-grained **phrasal** indices (NP modification) were the stronger predictors; MLTU was weaker and harder to interpret. Kyle's TAASSC dissertation (2016) agrees: the construct is "under-developed" and classic indices are "overly broad."

Practical consequence: **if you must move a syntactic dial, move NP modification density, not sentence or T-unit length.** A text can have short sentences and high academic register (dense NPs, nonfinite postmodifiers) — which is also compatible with the sentence-length variance a detector-evasion pass wants.

### 4.3 Cohesion indices do not predict quality; sometimes the reverse

Crossley & McNamara (2010), "Cohesion, coherence, and expert evaluations of writing proficiency": "recent studies of essay writing have demonstrated that computational indices of cohesion are **not** predictive of evaluations of writing quality." Their expert-rating study found that raters judge coherence "based on the **absence** of cohesive cues in the essays rather than their presence."

This matches IELTS Band 9 ("attracts no attention") and Band 5 (penalizing "overuse … of cohesive devices"), and report `04`, where sentence-initial connectives are an AI tell. **Connective density is a rare case where detector-evasion and quality point the same way: fewer explicit connectives is both more human and higher-scoring.** Never "improve cohesion" by inserting transitions.

### 4.4 Length is the strongest single correlate — and it is a confound

Perelman (2014), "When 'the state of the art' is counting words," *Assessing Writing*: essay length alone explains a large share of AES score variance, and several published systems are substantially length detectors. So (a) regress out word count before trusting any quality score, and (b) a humanizer that shortens text will look like it degraded quality on a naive scorer even when it did not.

### 4.5 Lexical diversity and sophistication are non-monotonic

Report `16` establishes that lexical diversity is a weak authorship discriminator. For quality it is curvilinear: very high MTLD in a short essay means topic-hopping or thesaurus abuse, not range; very low means repetition. Mean word frequency below the academic band reads as thesaurus-driven — the failure AP names as "complicated or complex sentences or language that is **ineffective because it does not enhance the analysis**." Treat both as **two-sided bands**.

### 4.6 The registered summary

| Naive "improvement" | Empirical verdict |
|---|---|
| Longer sentences / longer T-units | Neutral to negative in academic register; measures conversation-like subordination (Biber et al. 2011) |
| More subordinate clauses | **Negative** — more frequent in conversation than in research articles |
| More explicit connectives | **Negative** — overuse is a mid-band IELTS symptom; raters judge coherence by absence of cues |
| Higher TTR / rarer words | Non-monotonic; over-shooting reads as ineffective ornament (AP Row C exclusion) |
| More words | Raises naive AES scores but is a confound, not a quality gain (Perelman 2014) |
| Denser NP modification (nonfinite postmodifiers, noun premodifiers, appositives, abstract *of*-phrases) | **Positive** — Biber Stage 4–5; Kyle & Crossley's best predictors |
| Consistent commentary on every piece of evidence | **Positive** — the AP 3→4 boundary; the highest-leverage single edit |
| Zero grammatical errors | Neutral at best; §6 shows real human academic writing is not error-free |

---

## 5. The humanize / quality conflict

### 5.1 First person: reconciling report 15 with student-writing expectations

Report `15` found first person is common in published research articles — self-mention at 4.2 per 1,000 words overall in Hyland (2005), and physics using *we* at 39.3 per 10,000 words. This is real and it is not in conflict with student expectations once you separate three things:

1. **Style guides now permit and often require it.** APA: "Use first-person pronouns in APA Style to describe your work … If you are writing a paper by yourself, use the pronoun 'I' … with coauthors, use … 'we'." APA explicitly forbids the third-person dodge — "**Do not use the third person to refer to yourself.** … *Incorrect:* The author explored treatments for social anxiety" (*Publication Manual* 7th ed. §4.16) — and forbids the **editorial "we"** meaning people-in-general ("*Incorrect:* We often worry about what other people think of us"). Chicago is more permissive still.
2. **Function differs by rank.** Hyland's authorial-identity work: experts use *I/we* for **high-risk acts** (claiming, arguing, evaluating others); novices and L2 students restrict it to **low-risk acts** (purpose, structuring, procedure). The physics *we* in report `15` is overwhelmingly procedural-collective ("we measure", "we find"), not opinion.
3. **The failure mode is not "I" but "I think".** "I think Orwell is wrong" is a C sentence; "I argue that Orwell's account of X fails on its own terms, because…" is an A sentence. Both are first person.

**Design rule:** first person is permitted and mildly humanizing. Allow *I argue / I contend / this essay argues / we can see* in argumentative acts. Block *I think / I feel / I believe / in my opinion / personally* used to substitute for an argument, and block the editorial *we* meaning "people in general." Cap self-mention density at roughly the published-register band from report `15` (a few per 1,000 words in soft fields, lower in hard fields) rather than treating it as unlimited.

### 5.2 The other edits

**Contractions.** APA-style academic prose avoids them; Chicago tolerates them in less formal registers; every corpus study puts contraction rate in academic research articles near zero. Contractions are a strong humanizing signal (they raise perplexity and match human informal distributions) and a *cheap register violation*. Verdict: **do not use in the body**; the exception is inside quoted material, where they are already licensed.

**Sentence fragments.** A recognized rhetorical device, but #20 on Lunsford & Lunsford's error list and marked by teachers 42% of the time. A grader cannot tell "deliberate" from "accidental". Verdict: **≤1 per ~1,500 words, emphatic, after a full sentence, never in intro or conclusion.**

**Informal connectives.** Sentence-initial *But/And/Yet* is Chicago-defensible and genuinely human; *plus, anyway, basically, a lot of, kind of* are register violations. Permit the former sparingly, block the latter.

**Simpler vocabulary.** The worst quality trade of any humanizing edit — it collides with Lexical Resource, Genre Conventions, and AVL coverage simultaneously. The safe version is **de-Latinizing verbs while keeping nominal density**: "utilize"→"use" is free; "demonstrate a positive correlation"→"go together" is not.

**Rhetorical questions, second person, exclamation.** Strong humanizing signals, register violations in most disciplines — except philosophy, which report `15` puts at 16.3 engagement markers per 1,000 words, three times the mean. Gate by discipline.

---

## 6. Grammar and mechanics gating: what error rate is actually normal

### 6.1 The empirical numbers

The best dataset is Lunsford & Lunsford (2008), "'Mistakes Are a Fact of Life': A National Comparative Study," *CCC* 59(4) — a stratified national sample of **877 first-year composition papers** replicating Connors & Lunsford (1986). Their Table 8:

| Study | Year | **Errors per 100 words** |
|---|---|---|
| Johnson | 1917 | 2.11 |
| Witty & Green | 1930 | 2.24 |
| Connors & Lunsford | 1986 | **2.26** (spelling excluded) |
| Lunsford & Lunsford | 2006 | **2.45** (spelling included); **2.299** excluding spelling |

The rate "remains almost exactly the same as it has been during the last century, though types of error vary considerably." Average paper length rose 162 → 422 → **1,038 words (2006)**, so a typical first-year paper of any grade contains **roughly 20–25 flaggable formal errors**. Sloan (1990) found 2.04 per 100 words and — importantly — that **professional writers also made errors**, differing in type rather than in occurrence. Teachers marked only **38%** of the errors coders found (43% in 1986), concentrating on "the highly visible and easy-to-circle mistakes" plus "errors that confused a sentence's meaning."

2006 top error patterns (% of all errors): wrong word 13.7, missing comma after an introductory element 9.6, incomplete/missing documentation 7.1, vague pronoun reference 6.7, spelling incl. homonyms 6.5, mechanical error with a quotation 6.4. Sentence fragment is 20th at 2.4. Lunsford & Lunsford also invoke Joseph Williams's "The Phenomenology of Error," which contained **100 deliberate formal errors** most readers never noticed.

### 6.2 What this means for a gate

- **A zero-error text is anomalous.** Human first-year academic prose has run ~2.0–2.5 errors per 100 words for a century, and professional prose is not error-free either. Zero LanguageTool hits in 1,000 words is outside the human distribution and is itself a detector signal (report `04` §4.7).
- **Errors are not fungible.** Rubrics gate on *interference with communication* (AP), *obscured meaning* (PERSUADE 2), *lack of clarity* (GRE 3), and teachers mark the visible ones. Never inject: spelling, homonyms, subject-verb disagreement, its/it's, wrong word, or anything inside a quotation or citation.
- **The safe budget** sits in the low-visibility, teacher-tolerated band: occasional missing comma after a short introductory element or with a nonrestrictive element, an occasional short comma splice, serial-comma or hyphenation inconsistency. Target **0.3–0.8 per 100 words** — well below the human mean, so the text reads as carefully edited rather than sloppy.

### 6.3 The tooling

| Tool | What it is | Numbers | Notes |
|---|---|---|---|
| **LanguageTool** (`language_tool_python` 3.4.0) | Rule + n-gram grammar/style checker, 5,000+ English rules, self-hostable, LGPL | — | The realistic gate. Categories map onto Lunsford's taxonomy well. Disable the STYLE and REDUNDANCY categories or you will be told to remove exactly the hedges academic prose needs. |
| **GECToR** (Grammarly, Apache-2.0) | Sequence *tagging* GEC — predicts token-level edit transformations rather than rewriting | F0.5: BERT 61.0 / RoBERTa 64.0 / XLNet 63.2 on CoNLL-2014; 68.0 / 71.8 / 71.2 on BEA-2019 | Fast, non-generative, so it cannot hallucinate content. Ideal as a *detector* of injected errors. |
| **CoEdIT** (Grammarly; `grammarly/coedit-large` 770M, `-xl` 3B, `-xxl` 11B) | Instruction-tuned FLAN-T5 text editing: grammar, fluency, coherence, style, simplification, paraphrase | SOTA on several editing benchmarks; "competitive with … largest-sized LLMs … while being ~60x smaller" | The 770M model is a practical local corrector/paraphraser |
| **ERRANT** (`errant` 3.0.2) | Aligns original vs corrected text and labels each edit with one of ~25 M/U/R × POS types | — | Use to *classify* the errors your humanizer introduces, so you can enforce a type-level allowlist rather than a count |
| **GAMET** (Crossley et al. 2019, *JoWR* 11(2)) | Desktop tool giving incidence counts for grammar, spelling, punctuation, whitespace, repetition, validated against human judgments | — | The academic-side equivalent, useful for calibrating against corpora |

**Gate design:** run LanguageTool (grammar + typography categories only), run GECToR, and diff. Any error the humanizer introduced that (a) GECToR corrects with high confidence, (b) falls in the "wrong word / spelling / agreement / apostrophe" band, or (c) occurs inside a quotation, citation, thesis sentence, or the first or last sentence of the document — reject the edit.

---

## 7. A quality scoring stack for the humanizer: what to compute, what thresholds, what to gate on

Four layers, evaluated on the *pair* (source, output). **Almost every gate is a delta gate, not an absolute gate** — you cannot know whether an essay deserves a B, but you can know with confidence whether you made it worse.

### Layer 0 — Invariants (hard fail, no exceptions)

Cheap, deterministic, no model needed.

| Check | Rule |
|---|---|
| Citation integrity | Every in-text citation, author name, year, page number, DOI, and quoted span byte-identical to source. Quotation marks balanced and content inside them unmodified. |
| Number/entity integrity | All numerals, units, percentages, dates, statistics, named entities preserved. Extract with spaCy NER + regex, set-compare. |
| Claim polarity | No negation flips, no hedge→booster or booster→hedge conversion on any sentence containing a citation. |
| Structure | Paragraph count preserved ±0; no heading text altered; reference list untouched. |
| Length | |Δ words| ≤ 8%. |
| No new content | NLI entailment (e.g. DeBERTa-MNLI) source-sentence → output-sentence ≥ 0.9 and no output sentence unentailed by any source sentence. |

### Layer 1 — Rubric-trait scorer (the primary gate)

Fine-tune a small encoder (DeBERTa-v3-base is enough) multi-head on **ELLIPSE**'s six analytic traits (cohesion, syntax, vocabulary, phraseology, grammar, conventions) plus **PERSUADE**'s holistic 1–6. Validate to **QWK ≥ 0.65 per trait** on held-out data before it gates anything — that is the novice-to-expert band and it is achievable; QWK ≥ 0.75 is expert-equivalent and is the stretch target.

| Gate | Threshold |
|---|---|
| Holistic 1–6 | Δ ≥ −0.15 score points; hard fail at Δ ≤ −0.35 |
| Each analytic trait | Δ ≥ −0.20 on the 1–5 ELLIPSE scale |
| Sign consistency | At most one trait may decrease at all |

Add an **LLM-as-judge second opinion** using the Multi Trait Specialization recipe (decompose into traits; one conversation per trait; require the model to quote evidence *before* scoring; average trait scores; min-max scale with Q1/Q3 outlier clipping). MTS lifts open 7–13B models from QWK ~0.20–0.46 to ~0.55–0.59 — usable as a *tiebreaker*, not as the primary gate. Do **not** ask a model for a single holistic score in one turn: that is the "vanilla" condition and it scores near chance on several ASAP prompts.

### Layer 2 — Rubric-dimension structural checks (interpretable, cheap, high value)

| Dimension | Metric | Gate |
|---|---|---|
| Argument structure | PERSUADE-style span tagger: has_position, n_claims, n_evidence, has_counterclaim, has_rebuttal | Any element present in source must be present in output. Position span must survive verbatim in meaning. |
| Commentary coverage | % of Evidence spans with adjacent Claim/commentary tokens | Δ ≥ 0 (this is the AP 3→4 boundary; never let it fall) |
| Source integration | citations/1k words; integral vs non-integral ratio; quote-drop count | citations/1k unchanged; quote-drops must not increase |
| Register | contraction count; colloquial-connective count; 2nd-person count; rhetorical-question count; *I think/I feel/in my opinion* count | contractions = 0 outside quotes; colloquial set = 0; `I think`-class = 0; fragments ≤ 1 per 1,500 words |
| Self-mention | self-mention per 1,000 words | inside the discipline band from report `15` (roughly 1–6/1,000 depending on field) |
| Cohesion | connective incidence/1,000 words; local embedding cohesion; adjacent lexical overlap | connective incidence must **not increase**; local cohesion within ±1 SD of source |

### Layer 3 — Two-sided distributional bands (never maximize)

Compute on the output; fail if outside the band derived from a discipline- and level-matched reference corpus (BAWE for student writing, discipline journals for expert writing — see report `14`). Bands are the 10th–90th percentile of the reference corpus, **not** the mean.

| Feature | Tool | Direction |
|---|---|---|
| Mean length of clause; nonfinite postmodifiers/NP; nouns-as-premodifiers/NP; appositives/1k; abstract *of*-postmodifiers/NP | TAASSC or spaCy recipe (§3.3b) | Band. These are Biber Stage 4–5 — the only complexity features you may *raise*. |
| Clauses per T-unit; dependent clauses per clause; finite adverbial clauses/1k | NeoSCA / L2SCA | Band, biased low. Raising these moves toward conversation. |
| Mean length of T-unit; mean length of sentence | NeoSCA | Band only. Never an objective. |
| MTLD, HD-D, MATTR | `lexical_diversity` | Band. |
| AVL/AWL coverage %; mean Zipf frequency of content words | `wordfreq` + AVL list / TAALES | Band. |
| Mean bigram/trigram MI **and** T-score against COCA | TAALES | Band on both jointly. High MI with low T = showy and unidiomatic. |
| Readability (FKGL, Coleman–Liau) | `textstat` | Band only, as a sanity check. Never optimize. |

### Layer 4 — Mechanics gate

LanguageTool (grammar + typography only, STYLE/REDUNDANCY off) plus GECToR plus ERRANT typing.

- **Floor:** total flagged errors ≥ 1 per 1,000 words. A perfectly clean 1,500-word essay is outside the human distribution.
- **Ceiling:** ≤ 0.8 per 100 words (one third of the human mean of 2.45; enough to look human, far below the "interferes with communication" threshold).
- **Type allowlist for *introduced* errors:** missing comma after a short introductory element; missing comma with a nonrestrictive element; comma splice with a short second clause; serial-comma inconsistency; hyphenation inconsistency; *that/which* alternation.
- **Type blocklist, always:** spelling, homonyms, wrong word, subject-verb agreement, its/it's, pronoun-antecedent agreement, anything inside quotations, citations, headings, the thesis sentence, or the first/last sentence of the document.
- **Position rule:** never in the first 100 or last 100 words.

### Layer 5 — Composition

```
Ship if:  Layer0 == PASS
      AND Layer1 holistic Δ ≥ -0.15 AND ≤1 trait declined AND no trait Δ ≤ -0.20
      AND Layer2 all structural invariants hold
      AND Layer3 all bands satisfied
      AND Layer4 floor ≤ errors ≤ ceiling AND introduced-error types ⊆ allowlist
Otherwise: revert to the highest-scoring candidate that passes, or return the source unchanged.
```

Two operating notes. First, **regress out length** before comparing any Layer-1 or Layer-3 number, per Perelman (2014). Second, run the whole stack over **candidate rewrites** (generate k, gate, rank) rather than as a post-hoc veto on a single output; the gate is far more useful as a selector than as a rejector.

---

## 8. The humanize / quality conflict matrix

Detector benefit is the expected reduction in AI-detector confidence (H = high, M = medium, L = low, 0 = none). Academic-grade cost is the expected effect on a rubric-faithful grade. **Verdict** is the product policy.

| Edit type | Detector benefit | Academic-grade cost | Which rubric row it hits | Verdict |
|---|---|---|---|---|
| Remove sentence-initial connectives ("Furthermore," "Moreover,") | **H** | **Negative cost — it *raises* the grade** | IELTS Coherence & Cohesion (Band 9 = "attracts no attention"); Crossley & McNamara 2010 | **Do aggressively** |
| Vary sentence length (raise burstiness) | **H** | 0 to slightly positive | GRE 6 "sentence variety"; PERSUADE 6 "meaningful variety in sentence structure" | **Do** — rubrics explicitly reward variety |
| Break the topic-sentence-then-summary paragraph template | **H** | 0 to positive | AP "line of reasoning" is about dependency between claims, not template | **Do**, but preserve claim order |
| Replace tricolons / "not X, but Y" parallelism | **M–H** | 0 | none | **Do** |
| Replace excess-vocabulary "AI words" (*delve, pivotal, tapestry, underscore, crucial*) with plainer synonyms | **H** | 0 to slightly positive | none; AP Row C penalizes ineffective ornament | **Do** |
| Add specific, concrete detail and named particulars | **M** | **Strongly positive** | AP Row B: 2 pts = "mix of specific evidence and broad generalities" | **Do** — the single best joint move |
| Add a genuine counterclaim + rebuttal | **M** | **Strongly positive** | PERSUADE elements; VALUE Content Development | **Do** |
| Use first person for argumentative acts (*I argue*, *this essay contends*) | **M** | 0 (APA §4.16 endorses it) | none, if it's an argumentative act | **Do**, within the discipline density band |
| Use *I think / I feel / in my opinion* | M | **High** | AP Row A ("obvious fact"/vague position); GRE "limited" | **Never** |
| Editorial *we* meaning people-in-general | L | Medium | APA explicitly incorrect | **Never** |
| Contractions | **M–H** | Medium | Genre & Disciplinary Conventions; near-zero in academic corpora | **Never in body**; allowed inside quotes |
| Sentence fragments | **M** | Medium–high | #20 error in Lunsford; marked 42% of the time | **≤1 per 1,500 words**, emphatic only, never in intro/conclusion |
| Colloquial connectives (*plus, anyway, basically, a lot of, so*) | M | Medium | Lexical Resource; Genre Conventions | **Never** |
| Sentence-initial *But* / *And* / *Yet* | L–M | 0 (Chicago-licensed) | none | **Allow sparingly** |
| Rhetorical questions, 2nd person, exclamation | M | Medium (discipline-dependent) | Genre Conventions; but philosophy runs 16.3 engagement/1k | **Discipline-gated only** |
| Simplify vocabulary broadly | **M** | **High** | Lexical Resource; AVL coverage; GRE "effective vocabulary" | **No.** Only de-Latinize verbs (*utilize*→*use*) |
| Shorten sentences globally | M | Low–medium | none directly, but drags NP density down with it | **Only via the variance dial**, never as a global target |
| Lengthen sentences / add subordination | L | **Medium — moves text toward conversation register** | Biber et al. 2011 | **No** |
| Unpack nonfinite relatives / appositives / abstract *of*-phrases into finite clauses | L–M | **High** | Biber Stage 4–5 → Stage 2–3 | **Never** |
| Nominalization → verb ("the implementation of X" → "implementing X") | M | Low (mildly negative for register) | Biber phrasal density | **Sparingly**, cap the total shift |
| Introduce low-visibility punctuation slips (intro comma, nonrestrictive comma, comma splice) | **M** | Low | none of the rubrics; below the "interferes" threshold | **Yes**, 0.3–0.8/100 words, outside protected spans |
| Introduce spelling / homonym / agreement / wrong-word errors | M | **High** | Lunsford's most-marked categories; AP mechanics gate | **Never** |
| Any error inside a quotation, citation, thesis, or first/last sentence | M | **Very high** | Sources & Evidence; AP Row A | **Never** |
| Leave text perfectly error-free | 0 (**negative** — a tell) | 0 | none | **Avoid** — enforce the ≥1/1,000-word floor |
| Reorder / drop evidence to break n-gram overlap | M | **Very high** | AP Row B; VALUE Sources & Evidence | **Never** |
| Reduce commentary to shorten text | L | **Very high** | AP 4→3 boundary is exactly "consistently explains" | **Never** |

**The headline asymmetry:** the four highest-value humanizing edits (strip connectives, raise sentence-length variance, kill AI vocabulary, break the paragraph template) all have **zero or negative** grade cost. The four most damaging edits (simplify vocabulary, inject visible errors, unpack academic phrasal syntax, trim commentary) have only **medium** detector benefit. There is a large, safe operating region; the product's job is to stay in it and to refuse the rest.

---

## Sources

**Rubrics and standards**
- Council of Writing Program Administrators, *WPA Outcomes Statement for First-Year Composition (3.0)*, 2014 — https://wpacouncil.org/aws/CWPA/pt/sd/news_article/243055/_PARENT/layout_details/false
- AAC&U, *Written Communication VALUE Rubric* — https://www.aacu.org/initiatives/value-initiative/value-rubrics/value-rubrics-written-communication
- College Board, *AP English Language and Composition Scoring Rubrics, Free-Response Questions 1–3*, effective Fall 2019 — https://apcentral.collegeboard.org/media/pdf/ap-english-language-and-composition-frqs-1-2-3-scoring-rubrics.pdf
- IELTS, *IELTS scoring in detail* (Task Response / Coherence and Cohesion / Lexical Resource / Grammatical Range and Accuracy band descriptors) — https://ielts.org/organisations/ielts-for-organisations/ielts-scoring-in-detail
- ETS, GRE Analytical Writing score level descriptions — https://www.ets.org/gre/test-takers/general-test/scores.html
- *Holistic Rating for Source-Based Writing / Independent Writing* (the rubric used for PERSUADE and Kaggle ASAP 2.0) — https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf
- APA Style, *First-Person Pronouns* (Publication Manual 7th ed. §4.16) — https://apastyle.apa.org/style-grammar-guidelines/grammar/first-person-pronouns

**Automated essay scoring**
- Kaggle / The Learning Agency Lab, *Automated Essay Scoring 2.0* — https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2 (data: ~24,000 argumentative essays, 1–6 holistic; leaderboard: gold band ≈0.836–0.840 QWK)
- Kaggle, *ASAP-AES* (2012) — https://www.kaggle.com/competitions/asap-aes
- Xie, Cai, Kong, Zhou & Qu, "Automated Essay Scoring via Pairwise Contrastive Regression," COLING 2022 — https://aclanthology.org/2022.coling-1.240/ (NPCR 0.817; full baseline table)
- Taghipour & Ng, "A Neural Approach to Automated Essay Scoring," EMNLP 2016 — https://aclanthology.org/D16-1193/
- Li & Ng, "Automated Essay Scoring: A Reflection on the State of the Art," EMNLP 2024 — https://aclanthology.org/2024.emnlp-main.991/
- Lee et al., "Unleashing Large Language Models' Proficiency in Zero-shot Essay Scoring," Findings of EMNLP 2024 — https://aclanthology.org/2024.findings-emnlp.10/ · https://arxiv.org/abs/2404.04941
- Xiao et al., "Human-AI Collaborative Essay Scoring: A Dual-Process Framework with LLMs," 2024 — https://arxiv.org/abs/2401.06431 (expert QWK 0.712, novice 0.526, novice+LLM 0.661)
- Mayfield & Black, "Should You Fine-Tune BERT for Automated Essay Scoring?", BEA 2020 — https://aclanthology.org/2020.bea-1.15/
- Yancey et al., "Rating Short L2 Essays on the CEFR Scale with GPT-4," BEA 2023 — https://aclanthology.org/2023.bea-1.49/
- Naismith et al., "Automated evaluation of written discourse coherence using GPT-4," BEA 2023 — https://aclanthology.org/2023.bea-1.32/
- Crossley et al., "The PERSUADE corpus 1.0," *Assessing Writing* 2022 — https://doi.org/10.1016/j.asw.2022.100667
- Crossley et al., "The ELLIPSE Corpus," *IJLCR* 2023 — https://doi.org/10.1075/ijlcr.22026.cro
- Perelman, "When 'the state of the art' is counting words," *Assessing Writing* 2014 — https://doi.org/10.1016/j.asw.2014.05.001
- "Evaluating Quadratic Weighted Kappa as the Standard Performance Metric for AES," EDM 2023 — https://doi.org/10.5281/zenodo.8115784

**Measurable features and tools**
- Coh-Metrix 3.0 — http://cohmetrix.com/
- SALAT tool suite (TAALES, TAACO, TAASSC, TAALED, GAMET, SEANCE, CRAT, ARTE) — https://www.linguisticanalysistools.org/
- Kyle & Crossley, "Automatically assessing lexical sophistication," *TESOL Quarterly* 49(4), 2015 — https://doi.org/10.1002/tesq.194
- Kyle, Crossley & Berger, "TAALES 2.0," *Behavior Research Methods* 50(3), 2018 — https://doi.org/10.3758/s13428-017-0924-4
- Crossley, Kyle & Dascalu, "TAACO 2.0," *Behavior Research Methods* 51(1), 2019 — https://doi.org/10.3758/s13428-018-1142-4
- Kyle, Crossley & Jarvis, "Assessing the validity of lexical diversity using direct judgements," *LAQ* 18(2), 2021 — https://doi.org/10.1080/15434303.2020.1844205
- Zenker & Kyle, "Investigating minimum text lengths for lexical diversity indices," *Assessing Writing* 47, 2021 — https://doi.org/10.1016/j.asw.2020.100505
- Lu, "Automatic analysis of syntactic complexity in second language writing," *IJCL* 15(4), 2010 — L2SCA
- NeoSCA (pip `neosca`) — https://github.com/tanloong/neosca · LFTK (pip `lftk`) · LingFeat (pip `lingfeat`) · `lexical-diversity` · `taaled` · `pylats` · `errant` · `language_tool_python` · `wordfreq` · `textstat`

**Counterintuitive findings**
- Biber, Gray & Poonpon, "Should We Use Characteristics of Conversation to Measure Grammatical Complexity in L2 Writing Development?", *TESOL Quarterly* 45(1), 2011 — https://doi.org/10.5054/tq.2011.244483 · PDF: https://jan.ucc.nau.edu/biber/Biber/Biber_Gray_Poonpon_2011.pdf
- Kyle & Crossley, "Measuring Syntactic Complexity in L2 Writing Using Fine-Grained Clausal and Phrasal Indices," *Modern Language Journal* 102(2), 2018 — https://doi.org/10.1111/modl.12468
- Kyle, *Measuring Syntactic Development in L2 Writing* (TAASSC dissertation), 2016 — https://doi.org/10.57709/8501051
- Crossley & McNamara, "Cohesion, coherence, and expert evaluations of writing proficiency," CogSci 2010
- Kyle & Crossley, "Measuring longitudinal writing development using indices of syntactic complexity and sophistication," *SSLA*, 2020 — https://doi.org/10.1017/s0272263120000546

**Grammar, mechanics, error rates**
- Lunsford & Lunsford, "'Mistakes Are a Fact of Life': A National Comparative Study," *CCC* 59(4), 2008, pp. 781–806 — https://doi.org/10.58680/ccc20086677 · PDF: http://staff.kellogg.edu/westdorpp/files/2019/08/lunsf20.pdf
- Connors & Lunsford, "Frequency of Formal Errors in Current College Writing, or Ma and Pa Kettle Do Research," *CCC* 39(4), 1988
- Sloan, "Frequency of Errors in Essays by College Freshmen and by Professional Writers," *CCC* 41(3), 1990
- Crossley, Bradfield & Bustamante, "Using human judgments to examine the validity of automated grammar, syntax, and mechanical errors in writing," *Journal of Writing Research* 11(2), 2019 (GAMET validation) — http://www.jowr.org/articles/vol11_2/JoWR_2019_vol11_nr2_Crossley_et_al.pdf
- GECToR — https://github.com/grammarly/gector (Apache-2.0; RoBERTa F0.5 64.0 CoNLL-2014 / 71.8 BEA-2019)
- CoEdIT — https://github.com/vipulraheja/coedit · https://huggingface.co/grammarly/coedit-large

**Companion reports:** `04` (measurable AI-vs-human features), `14` (corpora acquisition), `15` (Hyland/Biber register norms and target distributions), `16` (stylometric variance — sample the distribution, don't target the mean).
