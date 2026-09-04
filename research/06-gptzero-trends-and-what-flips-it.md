# What Actually Flips GPTZero (and Turnitin / Originality) from "AI" to "Human"

**Scope:** empirical, community-tested trends 2023–2026; what moves scores, what detectors now look for, and what stopped working.
**Compiled:** September 2026. Evidence grades used throughout: **A** = peer-reviewed or large-N controlled; **B** = vendor benchmark or multi-sample independent test; **C** = single anecdote, small-N blog, or vendor-adjacent marketing.

---

## 0. The short version

1. **The "perplexity + burstiness" mental model is three years stale.** GPTZero dropped those metrics as the decision mechanism in autumn 2023 and moved to an end-to-end deep-learning classifier; perplexity/burstiness survive only as one of "seven indicators" used for explanation, not the verdict ([GPTZero](https://gptzero.me/news/perplexity-and-burstiness-what-is-it/), [support](https://support.gptzero.me/articles/9585228410-how-do-i-interpret-burstiness-or-perplexity)). Edits that only raise surface perplexity (odd synonyms, typos) worked against the 2023 model and are now actively penalized as "paraphraser" signals.
2. **What still reliably flips a verdict is register-level change, not token-level change:** first-person stance, concrete specifics (names, dates, numbers), genuine opinion, uneven paragraph and sentence rhythm, and lower formality. GPTZero's own April 2026 advice lists exactly these ([how-to-lower-your-ai-score](https://gptzero.me/news/how-to-lower-your-ai-score/)).
3. **Whole-document edits beat sentence-level patching.** GPTZero's support docs: "adjusting an entire document changes your score more than just adjusting the sentences" ([support](https://support.gptzero.me/articles/7549392421-how-do-i-interpret-results-from-gptzero-s-advanced-sentence-scanning)).
4. **Since Dec 2025 (Model 3.15b) GPTZero is tuned to catch humanizers**, at the cost of more false positives on formal human prose. Pangram and Originality Turbo made the same trade. Turnitin merged its "AI-paraphrased" (purple) category into plain "AI" (blue) in Aug 2026.
5. **The one bypass every vendor concedes:** actually rewriting in your own words. GPTZero said this in March 2023 and has not retracted it ([GPTZero](https://gptzero.me/news/gptzero-by-passers/)).

---

## 1. How GPTZero decides today (and what "burstiness" meant numerically)

**2023 mechanics (deprecated).** GPTZero computed per-sentence perplexity with a GPT-2-class model and defined burstiness as the variance of that perplexity across the document. Its published rule of thumb: *"A perplexity above 85 is more likely than not from a human source"*; no numeric burstiness threshold was ever published ([GPTZero](https://gptzero.me/news/perplexity-and-burstiness-what-is-it/)). A January 2023 Hacker News thread on a from-scratch reimplementation (BurhanUlTayyab/GPTZero) already showed how brittle this was; commenter camjw: "most sentences I tried gave different results" from the real thing ([HN](https://news.ycombinator.com/item?id=34557382)).

**2023-Q4 onward.** Deep-learning classifier trained on millions of documents; described in the Feb 2026 paper as a "hierarchical, multi-task architecture" with "multi-tiered automated red teaming" against paraphrasers ([arXiv 2602.13042](https://arxiv.org/abs/2602.13042)). Outputs: document class (AI / human / mixed), paragraph and sentence probabilities, confidence bands ("highly confident" <2% error, "moderately" ~10%, "low" ≥14%), an optional "possible AI paraphrasing detected" flag, and a separate AI-Vocabulary highlighter that "doesn't directly affect probability scores" ([GPTZero](https://gptzero.me/news/understand-gptzero-ai-scan/)).

**Sentence highlighting.** Advanced Scan marks "sentences and phrases that disproportionately affect" the verdict; some contributing sentences are not highlighted at all, and the model "analyzes patterns across the entire document" ([support](https://support.gptzero.me/articles/7549392421-how-do-i-interpret-results-from-gptzero-s-advanced-sentence-scanning)). Practical consequence: rewriting only the highlighted sentences is the least efficient fix.

**What "burstiness" means in practice now.** Third-party analyses converge on cadence heuristics rather than a variance number: three or more consecutive sentences of 17–23 words, average sentence length parked at 18–24 words, and more than half of a paragraph's sentences opening with "The/This/It/In" are the patterns most associated with flags ([Duey](https://www.duey.ai/post/em-dash-ai-writing)). GPTZero's own rule-of-three post (Mar 2024) frames the same thing rhythmically: "Embrace tempo changes and emotional swells and contractions, rather than following a metronome's unchanging beat" ([GPTZero](https://gptzero.me/news/the-rule-of-three/)).

---

## 2. Timeline: detectors vs. humanizers, 2023–2026

| Date | Event |
|---|---|
| Jan 2023 | GPTZero launches (perplexity + burstiness). 30k uses in week one ([Wikipedia](https://en.wikipedia.org/wiki/GPTZero)). |
| Feb 2 2023 | GPTZero patches Cyrillic-homoglyph and random-space "injection" bypasses; says paraphraser (T5) detection is coming; concedes "rewriting the text in one's own words, as a human" is the bypass it can't counter ([GPTZero](https://gptzero.me/news/gptzero-by-passers/)). |
| Apr 2023 | Turnitin AI indicator launches; Liang et al. (Stanford) show 7 detectors flag 61.22% of human TOEFL essays; simple prompts drop detection from 100% to ~13% ([arXiv 2304.02819](https://arxiv.org/abs/2304.02819)). |
| Jun 30 2023 | GPTZero "2023-06-30" model: AUC 0.95 vs 0.85 on a GPT-4 set ([GPTZero](https://gptzero.me/news/deep-learning-model-updates/)). |
| Autumn 2023 | GPTZero abandons perplexity/burstiness for a deep-learning classifier; TOEFL false positives reportedly cut to 1.1% via de-biasing ([GPTZero](https://gptzero.me/news/why-writing-flagged-ai/)). |
| Dec 2023–Jul 2024 | Turnitin adds "AI-paraphrased" detection (purple highlights); Jul 2024 asterisks scores of 1–19% as unreliable ([The AI Rankings](https://theairankings.com/guides/turnitin-ai-detection/)). |
| Feb 21 2024 | Originality.ai v3.0: 98.8% accuracy, 2.8% FPR ([EIN Presswire](https://www.einpresswire.com/article/690270874/originality-ai-launches-version-3-0-a-breakthrough-in-ai-content-detection-accuracy)). |
| Mar 2024 | Perkins & Roe: six detectors average 39.5% on raw AI text; adversarial prompts cut a further 17.4 points; GPTZero worst at 26% ([arXiv 2403.19148](https://arxiv.org/abs/2403.19148)). |
| May 2024 | RAID benchmark: GPTZero is the only closed detector robust to homoglyphs (−0.3%); Originality loses 75.7% ([arXiv 2405.07940](https://arxiv.org/html/2405.07940v1)). |
| Jul 8 2024 | Originality Standard 2.0.1 + Lite 1.0.0 (Lite tolerates Grammarly-style edits) ([EIN Presswire](https://www.einpresswire.com/article/725606895/originality-ai-announces-new-ai-detection-models-2-0-1-standard-beta-and-1-0-0-lite)). |
| Sep 18 2024 | GPTZero "Authorship"/Writing Report: Google Docs typing replay, paste history, multi-editor attribution ([PR Newswire](https://www.prnewswire.com/news-releases/ai-detection-platform-gptzero-releases-authorship-tools-to-preserve-original-writing-302247978.html)). |
| Oct 8 2024 | GPTZero AI Vocabulary: monthly top-50 list; "objective study aimed" is 269x more likely in AI text ([EdScoop](https://edscoop.com/gptzero-common-ai-words-detection-education-2024/)). |
| Nov 7 2024 | GPTZero paraphraser detection (Phrasly, Undetectable, QuillBot named): ~95% recall, <0.1% FPR claimed ([GPTZero](https://gptzero.me/news/ai-paraphrasing-detection/)). |
| Jan 23 2025 | Pangram audits 19 humanizers: new model 93.66% vs GPTZero 34.53%, Binoculars 29.73% ([Pangram](https://www.pangram.com/blog/humanizers-announcement)). |
| Summer 2025 | GPTZero Model 3.7b: trained on RL-discovered adversarial prompts and on human text run through grammar tools; 95–99% recall at 1% FPR on GPT-4.1/Claude Sonnet 4/Gemini 2.5 ([GPTZero](https://gptzero.me/news/gpt5/)). |
| Aug 2025 | Turnitin extends detection to "bypasser" tools; Pangram update: 90.3–100% on all 19 humanizers, Undetectable lowest ([Pangram](https://www.pangram.com/blog/humanizers-aug-25)); Jabarian & Imas (Chicago Booth): Pangram ~0 FPR/FNR, GPTZero FNR ≥50% on StealthGPT output ([BFI](https://bfi.uchicago.edu/insights/artificial-writing-and-automated-detection/)). |
| Sep 2025 | Originality Turbo 3.0.2 ("up to 97% on AI humanizers", 1.5% FPR), Lite 1.0.2 (0.5% FPR), Academic 0.0.5 ([Originality](https://originality.ai/blog/year-in-review-2025)). |
| Oct 2025 | GPTZero multilingual model, 9 languages (later 18) ([GPTZero](https://gptzero.me/news/multilingualdetection/)). |
| Dec 18 2025 | GPTZero Model 3.15b ("2025-12-18-base"): "Improved robustness against AI paraphrasers." Independent test: 6 of 7 humanizers now caught, but a 2013 human text flipped to "100% AI, paraphrased" ([Verva](https://verva.com/blog/does-gptzero-detect-ai-humanizers/)). |
| Jan 2026 | GPTZero claims 93.5% recall on a 12-paraphraser benchmark vs Pangram 50.2%, Originality 57.3% ([GPTZero](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)); Chicago Booth raw-text numbers: GPTZero 99.5%, Pangram 99.1%, Originality 85.0% ([GPTZero](https://gptzero.me/news/chicago-booth-2026/)). |
| Feb 2026 | GPTZero Model 4.3b benchmark page: 0.08% FPR, 99.6% recall; 1,000-text bypasser set: 91.8% recall ([GPTZero](https://gptzero.me/news/gptzero-ai-detection-benchmarking-the-industry-standard-in-accuracy-transparency-and-fairness/)). |
| Mar 2026 | StealthRL: RL-trained paraphraser hits 97.6% attack success against open detectors, but output quality falls from 3.78 to 2.51/5 ([arXiv 2602.08934](https://arxiv.org/html/2602.08934)). |
| Jun 23 2026 | Superhuman (ex-Grammarly) acquires GPTZero: 19M users, ~$30M ARR ([TechCrunch](https://techcrunch.com/2026/06/23/superhuman-acquires-ai-detection-startup-gptzero/)). |
| Jul–Aug 2026 | Turnitin merges purple "AI-paraphrased" into blue "AI" (Aug 4); GPTZero Model 4.8b masks headers/URLs/bibliographies so they can't be used as "human" padding ([GPTZero](https://gptzero.me/news/meaningful-ai-text-detection/)). |

---

## 3. Edit type → observed effect on GPTZero → evidence quality

| Edit | Observed effect on GPTZero (current models unless noted) | Evidence | Sources |
|---|---|---|---|
| **Increase sentence-length variance** (mix <8-word and >28-word sentences; uneven paragraphs) | Strongest single structural lever. Perkins' "Increase Burstiness" prompt cut mean detector accuracy from 39.5% to 15.9% (2024). GPTZero itself still recommends it (Apr 2026). Effect is smaller on 2025+ models but still directionally real. | **A** (2024) / **B** (2026) | [Perkins](https://arxiv.org/abs/2403.19148), [GPTZero](https://gptzero.me/news/how-to-lower-your-ai-score/), [Duey](https://www.duey.ai/post/em-dash-ai-writing) |
| **Add first-person stance / opinion** ("I think", "honestly, I'm not sure") | Consistently reported to move mixed→human; GPTZero lists "adopt personal perspective" as a fix; Pangram says tone shifts are detectable if text is otherwise AI-structured. | **B** | [GPTZero](https://gptzero.me/news/how-to-lower-your-ai-score/), [Pangram](https://www.pangram.com/blog/how-students-try-to-avoid-ai-detection) |
| **Inject concrete specifics** (names, dates, numbers, a real incident) | Replaces "generic" sentences that Advanced Scan highlights; GPTZero: "Substitute vague language like 'many benefits' with specific examples." | **B** | [GPTZero](https://gptzero.me/news/how-to-lower-your-ai-score/) |
| **Contractions + conversational register** | Widely reported to reduce AI probability; GPTZero endorses. Alone, rarely flips a fully-AI document. | **B/C** | [GPTZero](https://gptzero.me/news/how-to-lower-your-ai-score/), [MPG ONE](https://mpgone.com/is-gptzero-accurate-our-2025-test-results-here/) |
| **Remove stacked transitions** ("Furthermore/Moreover/Additionally" openers) | Repeated transitions in every section are a flagged pattern; replacing with a date/name/number opener is a common tip. No controlled numbers. | **C** | [TextSight](https://www.textsight.ai/how-to-pass-gptzero/), [Buzzcube](https://www.buzzcube.io/chat-gpt-zero-explained/) |
| **Break rule-of-three / "not just X but Y" parallelism** | GPTZero blog identifies tricolons as an AI tell; Duey lists "not just X, but Y" at ~1/paragraph in AI output. Effect unquantified. | **C** | [GPTZero](https://gptzero.me/news/the-rule-of-three/), [Duey](https://www.duey.ai/post/em-dash-ai-writing) |
| **Swap AI vocabulary** (delve, tapestry, leverage, underscore, navigate, robust, myriad, showcase) | Changes the AI-Vocabulary highlight but GPTZero says that feature "doesn't directly affect probability scores." Community claims of "+5–10 perplexity points" are untested. | **C** | [GPTZero](https://gptzero.me/news/understand-gptzero-ai-scan/), [TextSight](https://www.textsight.ai/how-to-pass-gptzero/) |
| **Em-dashes → commas** | "A few points at most." Punctuation is not what the classifier keys on. Humanizers that mechanically strip hyphens create *new* artifacts ("1861, 1867"). | **B** | [Duey](https://www.duey.ai/post/em-dash-ai-writing), [Pangram](https://www.pangram.com/blog/ai-humanizers-the-slop-2-problem) |
| **Deliberate typos / spelling errors** | Most effective evasion in 2024 (Perkins: accuracy 39.5%→12.9%) but authors note output "would very unlikely be submitted by a student." 2025+ models treat error injection as a paraphraser signal; Pangram: still triggered. | **A** (2024) / **B** (now ineffective) | [Perkins](https://arxiv.org/abs/2403.19148), [Pangram](https://www.pangram.com/blog/how-students-try-to-avoid-ai-detection) |
| **Sentence fragments / rhetorical questions** | Anecdotally reduce uniform cadence; no controlled test. Fragments that read as "unnatural" are a listed humanizer artifact. | **C** | [Pangram](https://www.pangram.com/blog/humanizers-aug-25) |
| **Lower vocabulary formality / "decrease complexity"** | Perkins: −19 points (2024). But Liang shows simple vocabulary is exactly what gets ESL humans flagged (8th-grade essays 5.19%→56.65% FP after simplification). Register matters more than word length. | **A** | [Perkins](https://arxiv.org/abs/2403.19148), [Liang](https://arxiv.org/abs/2304.02819) |
| **Raise vocabulary / "elevate with literary language"** | 2023: detection 100%→13% (Liang). 2024: "Increase Complexity" least effective (−2 points, Perkins). Now dead. | **A** | [Liang](https://arxiv.org/abs/2304.02819), [Perkins](https://arxiv.org/abs/2403.19148) |
| **Split long paragraphs / vary paragraph length** | Included in Perkins' burstiness prompt ("Adjust paragraphs so that these are also of a different length"). Part of the strongest lever above. | **A** (bundled) | [Perkins](https://arxiv.org/abs/2403.19148) |
| **QuillBot-style synonym paraphrase** | 2024: −21 points. 2025+: GPTZero flags "possible AI paraphrasing"; Originality's paraphrase accuracy *improves* (+11.7% in RAID); Turnitin catches 71–88% of QuillBot output. | **A/B** | [Perkins](https://arxiv.org/abs/2403.19148), [RAID](https://arxiv.org/html/2405.07940v1), [HumanizeThisAI](https://humanizethisai.com/blog/can-turnitin-detect-humanized-ai-text) |
| **Homoglyphs / zero-width / Unicode spaces** | Patched by GPTZero Feb 2023; RAID: GPTZero −0.3%. Still devastates Originality (−75.7%) and Winston (−38.3%) in 2024, but vendors now flag Unicode anomalies as obfuscation. | **A** | [GPTZero](https://gptzero.me/news/gptzero-by-passers/), [RAID](https://arxiv.org/html/2405.07940v1), [Pangram](https://www.pangram.com/blog/humanizers-aug-25) |
| **Padding with headers, URLs, bibliographies** | Killed Aug 2026: Model 4.8b masks them (shown gray). | **B** | [GPTZero](https://gptzero.me/news/meaningful-ai-text-detection/) |
| **Commercial humanizer pass (Undetectable, StealthGPT, etc.)** | Contradictory. Vendor-adjacent tests: Undetectable 2%→96% human on GPTZero (Hastewire, Jan 2026). Vendor tests: still "likely AI" (GPTZero Jan 2025; Originality Aug 2026). Independent: GPTZero FNR ≥50% on StealthGPT (Booth 2025) but 6/7 humanizers caught after 3.15b (Verva). Net: works some of the time, decays within months. | **B (conflicting)** | [Hastewire](https://hastewire.com/blog/accuracy-test-top-ai-humanizers-vs-gptzero), [GPTZero](https://gptzero.me/news/undetectable-ai-review/), [BFI](https://bfi.uchicago.edu/insights/artificial-writing-and-automated-detection/), [Verva](https://verva.com/blog/does-gptzero-detect-ai-humanizers/) |
| **Grammarly / LLM "polish" of human text** | Goes the wrong way: heavy Grammarly rewrites flip human IELTS essays to AI (Originality); one freelancer went "0% to 100% AI" (GPTZero); APT-Eval: detectors "flag even minimally polished text as AI." | **A/B** | [Originality](https://originality.ai/blog/grammarly-use-trigger-ai-detection), [GPTZero](https://gptzero.me/news/why-writing-flagged-ai/), [arXiv 2502.15666](https://arxiv.org/abs/2502.15666) |
| **Shortening the sample** | Short texts (<100–300 words) are noisy in both directions; Booth found GPTZero ~2.4% FPR on short passages vs ≤1% on long. Turnitin refuses <300 words. | **A** | [GradPilot](https://gradpilot.com/news/ai-detector-false-positive-rates-compared), [The AI Rankings](https://theairankings.com/guides/turnitin-ai-detection/) |

---

## 4. Community and systematic tests: what people actually observed

**Large-N independent tests (B-grade).**
- *Working Educators (fall 2025, 247 verified-authorship essays, 8 Philadelphia schools):* GPTZero flagged 23% of human essays; 41% for ESL students vs 18% native; caught only 67% of AI essays. An 11th-grade AP Literature essay scored "95% likely AI" ([Working Educators](https://workingeducators.org/gptzero)).
- *MPG ONE retest (Jul 2026):* accuracy "79 to 85%"; frontier-model output "often missed, read as human"; paraphrased/heavily edited AI "frequently missed"; false positives clustered on technical documentation, template-driven text, ESL writing, strict-style-guide prose ([MPG ONE](https://mpgone.com/is-gptzero-accurate-our-2025-test-results-here/)).
- *Stanford SCALE (Jun 2025, 78 essays):* AI essays detected at 91–100%; "a handful" of human false positives; "reliability in distinguishing human-authored texts is limited" ([arXiv 2506.23517](https://arxiv.org/abs/2506.23517)).
- *Hastewire (Nov 2025–Jan 2026):* 50 Undetectable samples, 95% pass on GPTZero; texts under 500 words showed "15% higher bypass rates" ([Hastewire](https://hastewire.com/blog/accuracy-test-top-ai-humanizers-vs-gptzero)). Treat as vendor-adjacent.
- *Undetectable.ai's own GPTZero test (Jun 2026):* 8 samples; raw AI 4/4 caught; humanized samples scored "92% AI, 8% Mixed" ([Undetectable](https://undetectable.ai/blog/gptzero-accuracy-rate/)). Even the humanizer vendor's own test shows GPTZero mostly catching it.

**Freddie deBoer's Pangram probe (C, but instructive).** A 5,000-word essay: 100% human. A 300-word excerpt of the same essay: "100% AI, high confidence." Sub-fragments of that excerpt: human again. 239 words of his 2017 prose plus 71 ChatGPT words: 100% AI. Conclusion: "brittle," context-dependent, and polar rather than proportional ([deBoer](https://freddiedeboer.substack.com/p/i-wouldnt-say-pangram-is-broken-but)). Lesson that generalizes to GPTZero: the same sentences score differently depending on their neighbors, so document-level rhythm is the unit of action.

**Recurring community anecdotes (C).** Human LinkedIn post by a non-native author: 100% AI ([Imtiaj](https://imtiajwrites.com/blog/gptzero-flagged-human-writing/)). Student essays flagged after using only a thesaurus and Grammarly. Texts from 2013 flagged as "paraphrased AI" after the 3.15b update ([Verva](https://verva.com/blog/does-gptzero-detect-ai-humanizers/)). The pattern in every anecdote: clean, formal, well-structured, low-personal-voice prose.

**GPTZero's own Writing Report changes the game for students.** The Google Docs replay records paste events; "if a large block of text appears in a document in a single paste, even if humanized first, GPTZero notes the behavior" ([allaboutai](https://www.allaboutai.com/resources/tested-gptzero-on-ai-written-vs-human-essays/)). Pangram notes students respond with auto-typers and fake edit histories, and warns replay "should not be used in isolation" ([Pangram](https://www.pangram.com/blog/how-students-try-to-avoid-ai-detection)).

---

## 5. Prompt-engineering lore: the prompts, verbatim, and whether they still work

**The academically validated ones (A, but 2023–2024 detectors):**

> "Elevate the provided text by employing literary language" — Liang et al.; detection of ChatGPT college essays fell from up to 100% to up to 13% (Hechinger: ~3% average) across seven detectors including GPTZero ([arXiv](https://arxiv.org/abs/2304.02819), [Hechinger](https://hechingerreport.org/proof-points-its-easy-to-fool-chatgpt-detectors/)).

> "Elevate the provided text by employing advanced technical language" — same paper, scientific abstracts: 68% → 28%.

> "Enhance the word choices to sound more like that of a native speaker" — applied to *human* TOEFL essays, cut false positives from 61.22% to 11.77%.

> "Simplify word choices as if written by a non-native speaker" — applied to human 8th-grade essays, raised false positives from 5.19% to 56.65%.

> Perkins & Roe "Increase Burstiness" prompt: "Rewrite the following passage with the aim of varying sentence length to create a more dynamic and engaging text. Use a mix of short, medium, and long sentences to achieve this effect. Adjust paragraphs so that these are also of a different length... ensure that the text remains coherent and academically appropriate. The goal is to make the writing more engaging and human sounding without sacrificing its core meaning." — mean detector accuracy 39.5% → 15.9% ([arXiv 2403.19148](https://arxiv.org/abs/2403.19148)).

> Perkins & Roe "Add Spelling Errors" prompt: "Rewrite the following text passage with spelling errors. Style: someone who is not proficient in English spelling. However, the errors should not be so extreme that the text becomes incomprehensible." — 39.5% → 12.9%, but outputs had 20+ errors and were judged unsubmittable.

**The folk prompts (C):**

> Twixify #2: "Perplexity measures how predictable the content is, with higher levels indicating that it was written by a human author. Burstiness checks how different sentences are, with human writing showing greater diversity. AI detectors are useful, but they are not perfect. That said, Use natural dialogue and write with a high level of perplexity and burstiness." ([Twixify](https://www.twixify.com/post/chatgpt-prompt-to-avoid-ai-detection))

> Twixify #1: "Engagement is the highest priority. Be conversational & empathetic. Ensure heterogeneous paragraphs. Ensure heterogeneous sentence lengths. And stick to primarily short, straightforward sentences."

> Twixify #3 (banned-word list): "Strictly follow this requirement: your response should not include any of the following words and phrases: meticulous, meticulously, navigating, complexities, realm, understanding, dive, shall, tailored, towards, underpins, everchanging, ever-evolving, the world of, not only, alright, embark, Journey, In today's digital age, hey, game changer, designed to enhance, it is advisable, daunting, when it comes to, in the realm of, amongst, unlock the secrets, unveil the secrets, and robust, diving, elevate, unleash, power, cutting-edge, rapidly, expanding, mastering, excels, harness."

> Persona prompt circulating in 2026 roundups: "Write as if you're a tired 28-year-old copywriter on their third espresso. Use casual language, contractions, and occasional self-deprecating humor. Avoid lists unless absolutely necessary." Companion rules: "never start a sentence with 'Additionally' or 'Furthermore', vary sentence length between 4 and 25 words, include at least one one-word sentence, and use at least two metaphors" ([UndetectedGPT](https://www.undetectedgpt.ai/blog/best-chatgpt-prompts-for-essays), [TheHumanizeAI](https://thehumanizeai.pro/articles/chatgpt-prompts-human-like-writing)).

**Do they work against 2025–2026 GPTZero? Mostly no.**
- StealthZero (May 2026) tested five such prompts ("Rewrite this text to sound like a college student wrote it. Use varied sentence lengths, occasional informal language, and avoid perfect grammar"; "make it sound more natural. Add personal opinions, use contractions, and break up long sentences"; "Make it imperfect, use colloquialisms, and vary your vocabulary"; plus a long multi-rule Reddit prompt) against GPTZero, Originality and Copyleaks: "None of the five rewrites achieved a clean pass across all three detectors." The long Reddit rule-list did best (only "mixed" on GPTZero) ([StealthZero](https://blog.stealthzero.ai/blog/ai-humanizer/chatgpt-humanizer-prompt/)).
- Originality (Aug 2025) ran "Humanize AI" and "Humanizer Pro" custom GPTs on five samples: 97–100% AI on every one ([Originality](https://originality.ai/blog/do-humanize-ai-gpts-work)).
- Why: GPTZero's Model 3.7b was explicitly trained on prompts found by "reinforcement learning algorithms to identify which prompting techniques generate text that looks the most human-written to our detector" ([GPTZero](https://gptzero.me/news/gpt5/)). The prompt lore is now training data.

---

## 6. Which sentences get flagged

No one has published a controlled probe of GPTZero's sentence highlighter, but the consistent reports (C-grade, converging) are:

- **Topic sentences and conclusions** in the same subject-verb-object template; "predictable phrasing especially in introductions and conclusions" ([JustDone](https://justdone.com/blog/ai/how-to-pass-ai-detection), [Buzzcube](https://www.buzzcube.io/chat-gpt-zero-explained/)).
- **Transition-led sentences** ("Furthermore… Moreover… In conclusion…") repeated section after section.
- **Generic, hedged, "balanced" claims** with no named entity, number, or date; "It's important to note that" preambles ([Duey](https://www.duey.ai/post/em-dash-ai-writing)).
- **Runs of medium-length sentences** (17–23 words) with no short or long outlier.
- **Tricolons** ("improve performance, expand applications, and address bias").
- Since 4.8b, **headers and bibliographies are ignored (gray)**, so intros/conclusions carry proportionally more weight.

Professors on Reddit use the heatmap as "introduction and conclusion look human, but these three paragraphs in the middle are clearly generated" — i.e., the *body* exposition paragraphs, where students paste, are where flags cluster ([Oreate](https://discover.oreateai.com/discover/the-only-ai-detectors-reddit-actually-trusts-in-2025)).

---

## 7. False-positive lore: what "human" means to the model

- **Founding documents.** Preamble of the Declaration of Independence: ZeroGPT 97.93% AI; GPTZero 11% AI (89% human); Grammarly and QuillBot: human ([Decrypt](https://decrypt.co/286121/ai-detectors-fail-reliability-risks)). ZeroGPT also flagged the 1836 Texas Declaration at ~90% ([Dallas Express](https://dallasexpress.com/state/zerogpt-flags-1836-texas-declaration-of-independence-as-nearly-90-ai-generated/)); the Constitution and Genesis have been flagged by perplexity-era tools. Note GPTZero itself did *not* flag the Declaration; the meme mostly comes from ZeroGPT.
- **ESL writing.** Liang: 61.22% of human TOEFL essays flagged; 97.8% flagged by at least one of seven detectors; native 8th-grade essays near 0% ([arXiv](https://arxiv.org/abs/2304.02819)). Working Educators 2025: GPTZero 41% FPR on ESL students ([Working Educators](https://workingeducators.org/gptzero)). Turnitin's Oct 2023 internal study claims no significant ELL bias ([Turnitin](https://www.turnitin.com/blog/new-research-turnitin-s-ai-detector-shows-no-statistically-significant-bias-against-english-language-learners)).
- **Formal, templated, technical prose.** GPTZero (Apr 2026): "if your work follows a certain structure too heavily, it may start to resemble machine-generated writing"; overly formal register, short responses, and Grammarly-heavy text are the other named causes ([GPTZero](https://gptzero.me/news/why-writing-flagged-ai/)).
- **Short texts.** Turnitin requires 300 words and hides 1–19% scores; Originality recommends 100+ words; Booth found GPTZero FPR ~2.4% on short vs ≤1% on long passages ([GradPilot](https://gradpilot.com/news/ai-detector-false-positive-rates-compared)).
- **Sampling settings matter.** RAID: a repetition penalty on the *generator* "decreases accuracy by up to 38 points" — AI text generated with high temperature or repetition penalties is already closer to the human distribution ([RAID](https://arxiv.org/html/2405.07940v1)).

So "human," to these models, means: uneven, specific, opinionated, imperfectly structured, and native-idiomatic. Polished, neutral, balanced, formulaic prose is "AI" regardless of who wrote it.

---

## 8. Detecting the humanizer: artifacts that now betray humanized text

Pangram, GPTZero and Originality have all published what they train on:

- **Tortured phrases / broken collocations:** "my culinary preconceptions shattered" → "my gastronomical misbeliefs were broken"; "artificial intelligence" → "counterfeit consciousness"; "I need to get my car fixed" → "I require to obtain my vehicle repaired" ([Pangram](https://www.pangram.com/blog/ai-humanizers-the-slop-2-problem), [Pangram](https://www.pangram.com/blog/what-is-a-humanizer)).
- **Hard-coded punctuation rewrites:** one humanizer turned 164 hyphens in 194 texts into zero — "multi, ethnic inner, city street", "1861, 1867"; missing spaces after periods ([Pangram](https://www.pangram.com/blog/ai-humanizers-the-slop-2-problem)).
- **Non-standard Unicode:** U+2009 thin spaces, Cyrillic homoglyphs, zero-width joiners; detectors NFKC-normalize and flag anomalies as obfuscation rather than as "human" ([Pangram](https://www.pangram.com/blog/humanizers-aug-25), [Stack Junkie](https://www.stack-junkie.com/blog/how-twaingpt-humanizer-actually-works-unicode-substitution-analysis)).
- **Nonsense insertions:** e.g., "CGSizeMake pp 18-23" dropped into prose to spike perplexity ([Pangram](https://www.pangram.com/blog/what-is-a-humanizer)).
- **Tense mixing, tone-deaf synonyms ("youngster" for "youth"), altered quotations, hallucinated facts** ("two-time world champion") ([Pangram](https://www.pangram.com/blog/ai-humanizers-the-slop-2-problem)).
- **GPTZero's two-stage logic:** first "is this AI-originated," then "has the content been altered using other writing tools to appear human"; paraphrasers "introduce grammar and spelling errors or awkward synonyms" that are themselves the signal ([GPTZero](https://gptzero.me/news/ai-paraphrasing-detection/)). Its Jan 2026 post says it uses "deeper semantic and structural signals beyond surface form" ([GPTZero](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)).
- **Originality Turbo** is trained on QuillBot/Undetectable output and "looks for specific synonym-swapping patterns" ([StealthGPT blog](https://www.stealthgpt.ai/blog/originality-ai-turbo-update-bypass-guide)).
- **The quality tax is measurable:** StealthRL's evasion-optimized paraphrases dropped LLM-judge quality from 3.78 to 2.51/5 and semantic similarity from 0.974 to 0.901 ([arXiv](https://arxiv.org/html/2602.08934)). The humanizer's fingerprint *is* its damage.

---

## 9. The 15 highest-leverage transformations, ranked by evidence

1. **Rewrite the whole document, not the highlighted sentences** — GPTZero support: whole-document adjustment moves the score more; deBoer's context-dependence probe confirms neighbors matter. (B)
2. **Impose real sentence-length variance** — sub-8-word and 28+-word sentences in every paragraph, uneven paragraph lengths. Only structural edit with peer-reviewed effect (Perkins −24 pts) and still endorsed by GPTZero in 2026. (A/B)
3. **Add first-person stance and actual opinion** — take a side, express uncertainty in your own voice. Endorsed by GPTZero; consistent in anecdotes. (B)
4. **Replace generic claims with named, dated, numbered specifics** — targets exactly the "generic exposition" sentences that get highlighted. (B)
5. **Drop the register: contractions, plain verbs, conversational connectives** — reliable directional effect; insufficient alone. (B)
6. **Kill repeated transition openers** and start paragraphs with a fact, name, or claim. (C, convergent)
7. **Break tricolons and "not just X but Y"** into unequal lists or single points. (C, convergent)
8. **Vary paragraph *function*, not just length** — mix narrative, a question, an aside, a data point, so paragraphs stop following the same template GPTZero names as a false-positive cause. (B, inferred from GPTZero's own list)
9. **Write it before you generate it** — GPTZero's Writing Report and Turnitin both weight authorship evidence; pasted blocks get logged. Draft in the doc, keep revision history. (B)
10. **Generate with higher temperature / repetition penalty if generation is unavoidable** — RAID: repetition penalty alone costs detectors up to 38 points. (A)
11. **Avoid Grammarly/LLM "polish" passes on human text** — they push scores toward AI on Originality Turbo and GPTZero. (A/B)
12. **Keep samples long and prose-only** — remove headers/bullets/bibliography padding (now masked anyway) and submit >300 words so noise doesn't decide. (A)
13. **Remove AI-vocabulary words** — cosmetic for the probability score but removes the visible red highlights a grader sees. (C)
14. **Use a humanizer only as a first pass, then hand-edit for collocations, punctuation, tense, and quotes** — unfixed humanizer output is caught at 90–100% by Pangram and 6/7 by GPTZero 3.15b. (B)
15. **Ignore em-dash hygiene as a detection lever** — a few points at most; do it for style, not score. (B)

---

## 10. Things that used to work but no longer do

| Technique | Worked against | Killed by |
|---|---|---|
| Cyrillic/homoglyph substitution, random space injection | GPTZero (Jan 2023), most detectors through 2024 | GPTZero patch Feb 2 2023; RAID shows GPTZero −0.3%; Unicode anomalies now flagged as obfuscation. Still hurts Originality/Winston but is treated as tampering. |
| Zero-width spaces / Unicode space swaps (Oct4Pie tool) | ZeroGPT, early GPTZero | Normalization; Originality says removing invisible chars doesn't change its score. |
| "Elevate the text with literary/technical language" | 7 detectors, 2023 (100%→13%) | Perkins 2024: increase-complexity least effective (−2 pts); GPTZero 3.7b trained on adversarial prompts. |
| "Write with high perplexity and burstiness" prompt | perplexity-era GPTZero (through autumn 2023) | Architecture change; GPTZero: those prompts "would no longer bypass." |
| Deliberate typos / spelling errors | 2024 detectors (−27 pts) | Treated as paraphraser signal; Pangram/GPTZero still trigger; unsubmittable quality. |
| QuillBot standard-mode paraphrase | 2023–early 2024 | GPTZero paraphraser flag (Nov 2024); Turnitin purple then merged blue; Originality gets *better* on paraphrased text (RAID +11.7%). |
| One-pass commercial humanizer, unedited | GPTZero through mid-2025 (Booth: FNR ≥50%) | Model 3.15b (Dec 2025): 6/7 humanizers caught; Pangram 90–100%; Originality Turbo 3.0.2 "up to 97%". |
| Custom "Humanizer" GPTs from the GPT Store | Nothing measurable | Originality Aug 2025: 97–100% AI on every sample. |
| Padding with headers, URLs, reference lists | GPTZero through Jul 2026 | Model 4.8b header masking (Aug 1 2026). |
| Splitting into short chunks to dodge the scan | Turnitin (<300 words = no score) | Turnitin hides 1–19% and refuses short prose; GPTZero FPR/FNR both rise on short text, so it's a coin flip, not a bypass. |
| Faking the Google Docs edit history with auto-typers | GPTZero Writing Report | Typing-pattern analysis; Pangram documents the tactic. Circumventable but treated as evidence of intent. |

---

## Sources

- GPTZero, "What is perplexity & burstiness for AI detection?" https://gptzero.me/news/perplexity-and-burstiness-what-is-it/
- GPTZero Support, "How do I interpret burstiness or perplexity?" https://support.gptzero.me/articles/9585228410-how-do-i-interpret-burstiness-or-perplexity
- GPTZero Support, "How do I interpret results from Advanced Sentence Scanning?" https://support.gptzero.me/articles/7549392421-how-do-i-interpret-results-from-gptzero-s-advanced-sentence-scanning
- GPTZero Support, "How does GPTZero detect AI paraphrasing and AI bypassers?" https://support.gptzero.me/articles/5593633457-how-does-gptzero-detect-ai-paraphrasing-and-ai-bypassers
- GPTZero, "How do I bypass AI detection?" (Mar 2023) https://gptzero.me/news/gptzero-by-passers/
- GPTZero, "Deep Learning Model Updates" (Jun 2023) https://gptzero.me/news/deep-learning-model-updates/
- GPTZero, "How to Break Free from GPT's Rule of Three" (Mar 2024) https://gptzero.me/news/the-rule-of-three/
- GPTZero, "GPTZero Detects AI Paraphrasers" (Nov 2024) https://gptzero.me/news/ai-paraphrasing-detection/
- GPTZero, "How to Understand Your AI Scan Results" (Dec 2024) https://gptzero.me/news/understand-gptzero-ai-scan/
- GPTZero, "Undetectable AI Review" (Jan 2025) https://gptzero.me/news/undetectable-ai-review/
- GPTZero, "Massive AI Detector Update for Summer 2025" (Model 3.7b) https://gptzero.me/news/gpt5/
- GPTZero, "How AI Detection Benchmarking Works at GPTZero" https://gptzero.me/news/ai-accuracy-benchmarking/
- GPTZero, "Introducing GPTZero's Multilingual AI Detection" https://gptzero.me/news/multilingualdetection/
- GPTZero, "GPTZero Tops Accuracy on Chicago Booth Benchmark" (Jan 2026) https://gptzero.me/news/chicago-booth-2026/
- GPTZero, "Detecting AI-Humanized Text: How GPTZero Stays Ahead" (Jan 2026) https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/
- GPTZero, "AI Detection Benchmarking: The Industry Standard" (Feb 2026, Model 4.3b) https://gptzero.me/news/gptzero-ai-detection-benchmarking-the-industry-standard-in-accuracy-transparency-and-fairness/
- GPTZero, "How to Lower Your AI Score" (Apr 2026) https://gptzero.me/news/how-to-lower-your-ai-score/
- GPTZero, "Why Does My Writing Get Flagged as AI?" (Apr 2026) https://gptzero.me/news/why-writing-flagged-ai/
- GPTZero, "A New Dawn of More Meaningful AI Text Detection" (Model 4.8b, Aug 2026) https://gptzero.me/news/meaningful-ai-text-detection/
- Cui, Tian et al., "GPTZero: Robust Detection of LLM-Generated Texts" (arXiv, Feb 2026) https://arxiv.org/abs/2602.13042
- EdScoop, "New GPTZero feature flags AI's favorite words and phrases" (Oct 2024) https://edscoop.com/gptzero-common-ai-words-detection-education-2024/
- PR Newswire, "GPTZero releases authorship tools" (Sep 2024) https://www.prnewswire.com/news-releases/ai-detection-platform-gptzero-releases-authorship-tools-to-preserve-original-writing-302247978.html
- TechCrunch, "Superhuman acquires AI detection startup GPTZero" (Jun 2026) https://techcrunch.com/2026/06/23/superhuman-acquires-ai-detection-startup-gptzero/
- Wikipedia, "GPTZero" https://en.wikipedia.org/wiki/GPTZero
- Hacker News, "Implementing GPTZero from scratch" (Jan 2023) https://news.ycombinator.com/item?id=34557382
- Liang et al., "GPT detectors are biased against non-native English writers" (2023) https://arxiv.org/abs/2304.02819
- Hechinger Report, "It's easy to fool ChatGPT detectors" https://hechingerreport.org/proof-points-its-easy-to-fool-chatgpt-detectors/
- Perkins, Roe et al., "GenAI Detection Tools, Adversarial Techniques and Implications for Inclusivity" (2024) https://arxiv.org/abs/2403.19148
- The Cheat Sheet, "Research Edition: AI Detectors Are Inaccurate, Can be Beaten" https://thecheatsheet.substack.com/p/long-research-edition-ai-detectors
- Dugan et al., "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors" (2024) https://arxiv.org/html/2405.07940v1
- Jabarian & Imas, "Artificial Writing and Automated Detection" (Chicago Booth / BFI, 2025) https://bfi.uchicago.edu/insights/artificial-writing-and-automated-detection/
- Russell, Karpinska, Iyyer, "People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text" (2025) https://arxiv.org/abs/2501.15654
- "Assessing GPTZero's Accuracy in Identifying AI vs. Human-Written Essays" (Stanford SCALE, 2025) https://arxiv.org/abs/2506.23517
- "Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing" (APT-Eval, 2025) https://arxiv.org/abs/2502.15666
- "StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion" (2026) https://arxiv.org/html/2602.08934
- Pangram, "Pangram can now detect AI humanizers" (Jan 2025) https://www.pangram.com/blog/humanizers-announcement
- Pangram, "How well does Pangram perform on humanizers? (Aug 2025)" https://www.pangram.com/blog/humanizers-aug-25
- Pangram, "How do AI Humanizers work? The Slop 2 problem" https://www.pangram.com/blog/ai-humanizers-the-slop-2-problem
- Pangram, "What is a humanizer?" https://www.pangram.com/blog/what-is-a-humanizer
- Pangram, "How Students Try to Avoid AI Detection" https://www.pangram.com/blog/how-students-try-to-avoid-ai-detection
- Freddie deBoer, "I Wouldn't Say Pangram is Broken, But I Would Say That It's Brittle" https://freddiedeboer.substack.com/p/i-wouldnt-say-pangram-is-broken-but
- Originality.ai, "Which AI Detection Model Should I Use?" https://originality.ai/blog/which-ai-detection-model-to-use
- Originality.ai, "2025: Year in Review" https://originality.ai/blog/year-in-review-2025
- Originality.ai, "Do Humanize AI GPTs Work?" (Aug 2025) https://originality.ai/blog/do-humanize-ai-gpts-work
- Originality.ai, "Undetectable.ai Review" (Aug 2026) https://originality.ai/blog/undetectable-ai-review
- Originality.ai, "StealthGPT AI Review" (Aug 2026) https://originality.ai/blog/stealthgpt-ai-review
- Originality.ai, "Does Using Grammarly Make My Content Get Detected as AI?" https://originality.ai/blog/grammarly-use-trigger-ai-detection
- Originality.ai, "AI Content Detector False Positives" https://originality.ai/blog/ai-content-detector-false-positives
- Originality.ai, "Most Accurate AI Detector According to RAID" https://originality.ai/blog/robust-ai-detection-study-raid
- EIN Presswire, "Originality.AI Launches Version 3.0" (Feb 2024) https://www.einpresswire.com/article/690270874/originality-ai-launches-version-3-0-a-breakthrough-in-ai-content-detection-accuracy
- EIN Presswire, "Originality.ai Announces 2.0.1 Standard and 1.0.0 Lite" (Jul 2024) https://www.einpresswire.com/article/725606895/originality-ai-announces-new-ai-detection-models-2-0-1-standard-beta-and-1-0-0-lite
- Turnitin, "AI writing detection model" guide https://guides.turnitin.com/hc/en-us/articles/28294949544717-AI-writing-detection-model
- Turnitin, "Turnitin release notes" https://guides.turnitin.com/hc/en-us/articles/27251688507533-Turnitin-release-notes
- Turnitin, "AI paraphrasing detection" blog https://www.turnitin.com/blog/ai-paraphrasing-detection-strengthening-the-integrity-of-academic-writing
- Turnitin, "AI detector shows no bias against ELLs" (Oct 2023) https://www.turnitin.com/blog/new-research-turnitin-s-ai-detector-shows-no-statistically-significant-bias-against-english-language-learners
- The AI Rankings, "Turnitin AI Detection in 2026" https://theairankings.com/guides/turnitin-ai-detection/
- HumanizeThisAI, "Can Turnitin Detect Humanized AI Text?" https://humanizethisai.com/blog/can-turnitin-detect-humanized-ai-text
- Verva/TwainGPT, "Does GPTZero Detect AI Humanizers?" (Model 3.15b test) https://verva.com/blog/does-gptzero-detect-ai-humanizers/
- Working Educators, "GPTZero Review 2026" (247-essay study) https://workingeducators.org/gptzero
- MPG ONE, "Is GPTZero Accurate? 2026 Retest" https://mpgone.com/is-gptzero-accurate-our-2025-test-results-here/
- Ryne AI, "Why GPTZero is not reliable anymore" https://ryne.ai/blog/why-gptzero-is-not-reliable-anymore-we-ran-100000-texts-to-prove-it
- Undetectable.ai, "GPTZero Accuracy Rate: Our 2026 Test Results" https://undetectable.ai/blog/gptzero-accuracy-rate/
- Hastewire, "Accuracy Test: Top AI Humanizers vs GPTZero" https://hastewire.com/blog/accuracy-test-top-ai-humanizers-vs-gptzero
- GradPilot, "AI Detector False Positive Rates Compared (2026)" https://gradpilot.com/news/ai-detector-false-positive-rates-compared
- Decrypt, "AI Detectors Claim the Declaration of Independence Was 98% AI-Generated" https://decrypt.co/286121/ai-detectors-fail-reliability-risks
- Dallas Express, "ZeroGPT Flags 1836 Texas Declaration" https://dallasexpress.com/state/zerogpt-flags-1836-texas-declaration-of-independence-as-nearly-90-ai-generated/
- Duey, "The Em-Dash Myth: What Actually Gives Away AI Writing" https://www.duey.ai/post/em-dash-ai-writing
- TextSight, "How to pass GPTZero" https://www.textsight.ai/how-to-pass-gptzero/
- Twixify, "Use These 3 ChatGPT Prompts To Avoid AI Detection" https://www.twixify.com/post/chatgpt-prompt-to-avoid-ai-detection
- StealthZero, "ChatGPT Humanizer Prompt (2026)" https://blog.stealthzero.ai/blog/ai-humanizer/chatgpt-humanizer-prompt/
- UndetectedGPT, "Best ChatGPT Prompts for Essays That Sound Human" https://www.undetectedgpt.ai/blog/best-chatgpt-prompts-for-essays
- Oct4Pie, "zero-zerogpt: Bypassing AI detectors with Unicode spacing" https://github.com/Oct4Pie/zero-zerogpt
- Stack Junkie, "How TwainGPT Humanizer Actually Works: Unicode Analysis" https://www.stack-junkie.com/blog/how-twaingpt-humanizer-actually-works-unicode-substitution-analysis
- StealthGPT, "Originality.AI new detection model tested" https://www.stealthgpt.ai/blog/originality-ai-turbo-update-bypass-guide
- Imtiaj Writes, "GPTZero Flagged Human Writing: My 100% AI Result" https://imtiajwrites.com/blog/gptzero-flagged-human-writing/
- AllAboutAI, "I Tested GPTZero On 5 AI Written Vs Human Essays" https://www.allaboutai.com/resources/tested-gptzero-on-ai-written-vs-human-essays/
- Oreate, "The Only AI Detectors Reddit Actually Trusts in 2025" https://discover.oreateai.com/discover/the-only-ai-detectors-reddit-actually-trusts-in-2025
- Buzzcube, "Chat GPT Zero Explained" https://www.buzzcube.io/chat-gpt-zero-explained/
- JustDone, "Top Ways To Pass GPTZero, Originality.ai, and Copyleaks" https://justdone.com/blog/ai/how-to-pass-ai-detection
