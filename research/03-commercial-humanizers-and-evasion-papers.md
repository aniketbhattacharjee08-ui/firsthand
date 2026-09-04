# Commercial AI Humanizers and the Detector-Evasion Literature

*Research memo, compiled 2026-09-03. Web sources fetched directly where possible; where a page was blocked (Medium, OpenReview, Turnitin press) the claim is attributed to the search snippet or a secondary source and marked as such.*

## 0. How to read the evidence

Three things distort almost every public number about humanizers:

1. **The review ecosystem is overwhelmingly competitor-authored.** Nearly every "honest review" of tool X is published on the blog of tool Y (UndetectedGPT reviewing StealthGPT, Walter Writes reviewing Ryne, Phrasly reviewing StealthGPT, HumanizeMyAI reviewing HIX, and so on). The few genuinely independent sources are detector vendors (Originality.ai, GPTZero, Pangram -- themselves conflicted in the opposite direction), academic audits (Jabarian & Imas 2025; Karr 2025; Masrour/DAMAGE 2025), the ProofreaderPro academic test (itself a vendor), Plagiarism Today, and Reddit.
2. **Single-passage tests dominate.** Most reviews run one 300-500-word ChatGPT essay through the tool and three detectors. Detector scores on the same tool swing from 4% to 65% AI across runs (StealthWriter, per fast.io), so single-shot numbers are close to noise.
3. **Detectors moved under the tools' feet.** Turnitin shipped AI-paraphrase detection (AIR-1, July 2024), "AI bypasser detection" (27 Aug 2025), and a Feb 2026 recall update; Pangram retrained on 19 humanizers (Jan 2025, 73%→94% recall on humanized text); Originality's Turbo 3.0.2 reports 90-97% on five named humanizers (Sept 2025); GPTZero reports 93.5% recall on a 12-paraphraser benchmark. Any review dated before mid-2025 measures a different world.

With that caveat, the picture is consistent: **no commercial humanizer reliably beats all of Turnitin, GPTZero, Originality and Pangram at once; most beat one or two weaker detectors (ZeroGPT, Writer, Copyleaks) while degrading the text.**

---

## Part A. Commercial products

### A.1 Market structure

- **HIX Bypass, Humbot and BypassGPT are the same company** (COCOSOFT Technology Pte. Ltd., Singapore; confirmed by trademark filings). Their mode names differ (HIX: Fast/Balanced/Aggressive/Latest; Humbot: Light/Balanced/Aggressive; BypassGPT: Fast/Creative/Enhanced) but they share infrastructure, pricing ladders and output character. Treat third-party results for one as informative about the others.
- **StealthGPT and StealthWriter are different companies** despite the names. StealthGPT (stealthgpt.ai) sells "Ghost" and "Samurai" engines plus a Turnitin-vs-GPTZero target toggle; StealthWriter (stealthwriter.ai) sells "Ghost 5.2 Mini/Pro" models with Light/Medium/Aggressive intensities and a "Generator" mode.
- **Grammarly and QuillBot** shipped humanizers in 2025 but position them as polish tools. Grammarly's page states outright: "This tool is not intended to bypass AI detectors."
- Scale: NBC News (2025) reported 43 humanizer sites logging 33.9M visits in one month; Turnitin says it tracks ~150 humanizer tools.

### A.2 Comparison table

| Tool | Pricing (2026) | Modes / controls | Claimed technique | Independent test evidence | Quality / meaning |
|---|---|---|---|---|---|
| **Undetectable.ai** | $9.99/mo for 10K words ($5 annual); $19/20K; $31/35K; $42.50/50K; API at same per-word rate; 250-word free trial | Readability: High School / University / Doctorate / Journalist / Marketing. Purpose: General, Essay, Article, Marketing, Story, Cover Letter, Report, Business, Legal. Strength: More Readable / Balanced / More Human | "Multi-layered" rewrite that varies sentence structure, vocabulary and tone; internal 8-detector consensus check | Originality.ai (2023-24): 100% AI before and after; GPTZero 100%→91%. Pangram Aug 2025: 90.3% still caught (the *lowest* of 19, i.e. Undetectable is the hardest for Pangram). Originality Turbo 3.0.2: 80-87% accuracy on Undetectable output (their weakest humanizer). Karr 2025 (academic abstracts): <4% of Undetectable rewrites still flagged (FNR >96%) by the commercial detectors tested. ProofreaderPro academic test: 94% raw bypass but 17.5/25 overall. DAMAGE places it in **L3 (lowest quality)** tier. | Best bypass of the mainstream tools, at a cost: register drops ("the findings indicate" → "we can see that"), citations occasionally reformatted, conversational tone forced on academic text. Grammar relatively clean (1.8 errors/1K words vs 4.2 for StealthWriter, per fast.io). |
| **StealthGPT** | Essential ~$24.99/mo, Pro $34.99, Business $49.99, Enterprise $249.99; no free tier or trial | Engines: Ghost, Samurai (premium). Target toggle: optimize for GPTZero *or* Turnitin, not both. Tone/readability options. REST API | "Custom models trained on hundreds of hours of human text," rewriting "for linguistic unpredictability"; marketing repeatedly says "guaranteed"/"undetectable" | aixradar: internal scanner 1%→88% human, but GPTZero 100% AI; Turnitin ~98%→~22% (third-party). Phrasly test: Turnitin 86% AI, Originality 100%, GPTZero 48%. UndetectedGPT: GPTZero 14%, ZeroGPT 7%, Turnitin 36%, Originality 47%. Pangram Aug 2025: 95.6% caught. Jabarian & Imas (UChicago BFI 2025-116): Pangram's FNR "remains low even when passages are modified using StealthGPT," while GPTZero's FNR rose to ~50%+. DAMAGE: **L1 (high quality)** tier | Grammatically correct but "mechanical"; quality and evasion both degrade past ~1,000 words. Meaning largely preserved. |
| **StealthWriter** | Free (10/day, Ghost Mini); Starter $20, Plus $50, Pro $100, Scale $400/mo | Ghost 5.2 Mini (free) / Pro (paid) / Legacy; Light / Medium / Aggressive; "Generator" mode | "Linguistic Randomization": varies sentence length, injects "Human Noise," restructures | fast.io: GPTZero 100% AI; Turnitin 16% (narrow pass); Originality 12-28% AI. Other runs: Originality 12-88% human, GPTZero 9-97% AI, Turnitin 4-65% -- extreme run-to-run variance. Originality Turbo: 87.1% (Lite) accuracy on its output. DAMAGE: **L2** | 4.2 grammar errors and 4.2 meaning-altering changes per 1,000 words; meaning preservation 6.8/10; "eccentric phrasing requires manual editing." Trustpilot billing complaints. |
| **HIX Bypass** | Standard $14.99/mo (5K words), Premium $29.99 (50K), Unlimited $59.99; annual ≈ $9.99/$14.99/$15 | Fast / Balanced / Aggressive / Latest ("targets latest Originality and Turnitin models") | Marketing: "100% undetectable"; keyword-preserving rewrite | HumanizeMyAI (competitor): GPTZero 0-80%+ variance; Turnitin 25-76% AI in same mode; Originality 100% AI to 73% bypass across testers; Copyleaks/ZeroGPT still flagged. DAMAGE: **L3** | Aggressive mode yields "awkward or hard-to-read sentences, with grammar slips." |
| **Humbot** | Basic $9.99/mo 50K words; Pro $19.99/150K; Ultra $49.99/600K (~$0.00008/word); API $30-$1,999/mo | Light / Balanced / Aggressive | Same stack as HIX/BypassGPT | ProofreaderPro: worst of 12 on academic text (11.5/25) -- citations mangled in 3/10 samples, grammar errors introduced in 4/10. Originality Turbo: 90% accuracy on its output. Trustpilot 2.4-2.7★. DAMAGE: **L3** | "Gibberish," misspellings, stray mid-sentence capitals; Aggressive mode alters dates, statistics and technical terms (factual drift). |
| **BypassGPT** | Basic $8/mo (5K words, annual), Pro $12 (30K), Unlimited $30 | Fast / Creative / Enhanced | Same stack as HIX/Humbot | Pangram: 99.7% caught. Androidheadlines Oct 2025 test: 100% AI on GPTZero. DAMAGE: **L2** | Similar to siblings; Enhanced mode restructures paragraphs. |
| **WriteHuman** | Free 3 req/mo; Basic $12/mo annual (80 req, 600 words); Pro $18 (200 req, 1,200 words); Ultra $36 (unlimited, 3,000 words) | Basic vs **Enhanced Model** (paid only; 2-5 output variants) | Rewrite plus built-in detector; Enhanced does "deeper structural changes" | fast.io 53-sample test: Writer 87% pass, GPTZero 82%, Copyleaks 51%, ZeroGPT 48%, **Originality 19%**. Androidheadlines: 100% AI on GPTZero. ProofreaderPro: 16/25, "parenthetical citations frequently restructured or relocated." DAMAGE: **L3** | Quality collapses with length: marketing <300 words 8.5/10; technical docs 4/10; academic 4.5/10 with grammar errors. Basic mode described as "filler-prone." |
| **Phrasly** | ~$10.99/mo annual ($131.88/yr) or $16-20 monthly; $2 three-day trial; 300-word free humanize | Easy / Medium / Aggressive / Pro Engine; tone (formal, casual, persuasive, informative) | Bundled humanizer + detector + writer + plagiarism check | fast.io 2026: Turnitin 51.4% avg AI; Originality Turbo 98% AI; ZeroGPT 95% human; Winston failed. Originality Turbo: in 90-97% band. DAMAGE: **L3** | Best on technical/factual text, worst on creative; 2,500-word cap per run. Heavy auto-renew/refund complaints. |
| **Rephrasy** | Opaque; ~$29/mo standard, enterprise "hundreds"; API ~$0.003/word | Undetectable Model v2; SEO Model; **Writing Style Cloning** from user samples | Style cloning + evasion model | detectiondrama (competitor): GPTZero 96→23%, ZeroGPT 94→31%, Copyleaks 91→28%, Turnitin 89→35%. Another test: "all detectors flagged the output as fully AI." Trustpilot critical. | 78% of samples kept factual accuracy; readability fell in 45% (Flesch 52.3→48.7). |
| **Walter Writes** | Starter $12/mo ($8 annual, 30K words, 750/request); Pro $23 ($13, 70K); Unlimited $47 ($26); 300-word free trial | Simple / Standard / Enhanced strength; tone presets | "Structure-level rewriting" plus built-in detector | Claims 99-100% human on its own checker. Independent: 79.7% Turnitin bypass ("1-in-5 caught"); StealthGPT's competitor test says it failed Turnitin and Originality; a Medium review cites 70.31% avg human score. No neutral test found; most reviews are undisclosed affiliate content. | Reviews praise rhythm and meaning preservation but give no per-1K-word error data. |
| **Twixify** | Basic $8/mo ($5 annual, 400 words/run, 75 req); Standard $13 ($9, 600 words); Premium $27 (900 words); $150 lifetime | "Echowriting": paste 3 samples of your writing; V3 Basic vs Custom; tone presets (personable, empathetic, direct, friendly, analytical, reflective) | Style mimicry from exemplars (in practice tone presets) | ZeroGPT 85.55→18.83%; passed Writer and Copyleaks; **Originality ~100% AI**. Pangram audit: in DAMAGE's **L1** (quality) tier yet caught. Trustpilot ~2.8★ with cancellation complaints. | "Awkward or nonsensical passages" reported; inconsistent. |
| **Smodin** | Freemium; credit-based (not pinned down) | Single humanizer inside a suite (detector, writer, plagiarism) | Paraphrase-based | Verva: **100% AI on all five external detectors** (GPTZero, ZeroGPT, Turnitin, Copyleaks, QuillBot), 62% on Smodin's own. Pangram: 100% caught. Other freelance tests: GPTZero 83%→12%. | Mostly a paraphraser; readable but ineffective. |
| **Netus AI** | Credits, $19-$99/mo; 50 free credits | Models C7-N1, G9-R1 (detection avoidance), V7-N2 (plagiarism), A12-AD; "29 bypasser versions V1-V29" | Fine-tuned paraphrase models | Gold Penguin: Copyleaks 67.6-75% AI across versions; ZeroGPT 7.6-65%; own detector passes own output (circular). Originality Turbo: 97% accuracy on Netus (its *easiest* humanizer). | "Stilted, awkward sentences"; passes Copyscape plagiarism but not AI detection. |
| **GPTinf** | Free trial then monthly/annual (~$9.99/mo) | Single mode | Paraphrase | Originality.ai: Originality 100%→100%, GPTZero 100%→100%, Copyleaks 100%→100%, Writer 18%→19% -- "did very little." Pangram: 99.2% caught. DAMAGE: **L1** quality tier | Fluent, but no evasion; "just swapping words and rearranging sentences." |
| **Humanize AI Pro** | 1,500 free credits; paid credits | Standard + premium modes | Paraphrase | Originality.ai: Originality 100%→100%; ZeroGPT 98.9%→69.4%; Writer 22%→5% AI. Androidheadlines: 100% AI on GPTZero. Pangram: 100% caught. DAMAGE: **L1** quality | Readable; ineffective against strong detectors. |
| **Ryne AI** | Coin tiers: free Amethyst → $99.99/mo Ruby; "Pro Algorithm" on Emerald/Ruby | Light / Medium / Max; late-2025 dual mode (readability-tuned vs detector-tuned) | Detection-focused model with "aggressive structural variation" | Essaydone: ~70% Turnitin, ~75% GPTZero bypass; creative text 45% detection drop, academic only 6%. Site claims 99.9%. | Fine for casual rewriting; unreliable on factual/academic text. |
| **Surfer AI Humanizer** | ~500 words free; paid Surfer plans unlock up to 50K humanized words | Single mode inside SEO suite | Paraphrase | Originality.ai: 100% AI before and after (two runs); ZeroGPT 98→75→39.8%. Jan 2026 test: GPTZero 98→39%. Pangram: 100% caught. DAMAGE: **L2** | Behaves like a basic paraphraser. |
| **QuillBot Humanizer** | Free with daily limits; Premium ~$30/mo (bundle) | Styles (Simple/Formal/Creative), paraphrase strength slider | Paraphraser repurposed; "advanced models, rules, and training data" | Pangram: 100% caught. Reddit consensus: free QuillBot "gets flagged" because it swaps synonyms and keeps structure. DAMAGE: **L1** quality | Cleanest output of the group; over-simplifies; neutral tone. |
| **Grammarly Humanizer** | Free basic; Grammarly Pro for six languages; four preset voices + custom voice from samples (Sept 2025 launch) | Preset styles / custom voice | Explicitly "not intended to bypass AI detectors" | WriteHuman test: 88.2% AI after. Pangram: 100% caught. Plagiarism Today (Nov 2025) showed Grammarly *Authorship* labelled Gemini-text-then-Humanized as "75% typed by human"; Grammarly fixed the labelling in Dec 2025 so rephrased AI text stays "AI-generated or rephrased." DAMAGE: **L1** quality | Best fluency; no evasion by design. |

### A.3 Cross-cutting independent evidence

**Pangram audit (Jan 2025, updated Aug 2025).** Pangram sampled 19 public humanizers, ranked them into three quality tiers, retrained, and reported per-tool recall on humanized text: 100% on Ahrefs, aihumanizer.com, Ghost AI, Grammarly, humanizeai.pro, QuillBot, Semihuman, Smodin, Surfer, SurgeGraph; 99.7% BypassGPT; 99.2% GPTinf; 99.0% Scribbr; 98.1% Writesonic; 97.6% DIPPER; 95.6% StealthGPT; 93.8% humanizeai.io; 93.5% Just Done; 92.7% TwainGPT; **90.3% Undetectable AI**. Aggregate: 73.07% → 93.66% after retraining; GPTZero 34.53% and Binoculars 29.73% on the same set. Their striking observation: **the more fluent the humanizer, the easier it is to detect**; the tools that evade best do so by damaging the text.

**DAMAGE (Masrour, Emi & Spero, arXiv 2501.03437, COLING 2025 workshop).** Same 19 tools formally tiered: **L1** (preserves tone/vocabulary/complexity): DIPPER, Grammarly, QuillBot, HumanizeAI.pro, StealthGPT, Semihuman, Twixify, GPTinf. **L2** (degrades quality, preserves intent): AIHumanizer.com, BypassGPT, StealthWriter, Surfer. **L3** (adds nonsense, distorts meaning): Ghost AI, HIX Bypass, HumanizeAI.io, Humbot, Phrasly, Undetectable AI, WriteHuman. Failure modes documented: hallucinated citations, inline junk like "(??????)" and "CGSizeMake pp 18-23," forced elementary or university register regardless of source. Detector recall on humanized text: GPTZero 99.73%→60.04%; Binoculars 94.15%→28.23%; DAMAGE 100%→98.26%. No code released.

**GPTZero benchmark (2025).** 1,000 paraphrased examples across 12+ open and closed paraphrasers and 9 domains: GPTZero 93.5% AI recall vs Originality 57.3% vs Pangram 50.2% (vendor-run; conflicts with Pangram's own numbers -- both are self-reported).

**Originality.ai (Sept 2025).** Turbo 3.0.2 accuracy on five humanizers: Undetectable 80-90%, Phrasly, StealthWriter 87-97%, Netus 97%, Humbot 90%; Lite 1.0.2 80.3-87.1%; FPR 0.5-1.5%.

**Turnitin.** AIR-1 (July 2024) targeted synonym-swap paraphrasing; "AI bypasser detection" (27 Aug 2025) is a separate model "trained on the signals and patterns of leading humanizers," English only, surfaced as a distinct "AI-paraphrased" category; Feb 2026 update raised recall. Turnitin claims <1% FPR (unverified). Vendor blogs report post-update raw AI text is caught nearly always, QuillBot-style paraphrasing barely helps, "standard humanizers" are still flagged more than half the time, and only aggressive humanizers or manual rewrites consistently land under 20%.

**Jabarian & Imas (UChicago BFI WP 2025-116; Chicago Booth Review, Dec 2025).** 1,992 human + 1,992 AI texts from four frontier models. FPR: Pangram ~0, GPTZero and Originality <1%, RoBERTa unusable. FNR: GPTZero 0-2%, Pangram 2-4%, Originality 10-40%. After StealthGPT humanization Pangram's FNR stayed low; GPTZero's rose to roughly 50%+.

**Karr (arXiv 2608.11256, 2025).** Published abstracts, four fields. Light guideline-compliant AI editing flagged 38-80%; unmodified 2023-25 abstracts flagged 9-15%; after Undetectable AI humanization <4% still flagged (FNR >96%). Conclusion: honest editing is riskier than cheating with a humanizer.

**ProofreaderPro academic test (vendor, 2026).** 12 tools, 10 academic samples, Turnitin/GPTZero/Originality, 5 criteria. Undetectable 94% bypass but 17.5/25; WriteHuman 16; Humbot 11.5. "Multicollinearity" → "multiple connections" cited as the archetypal meaning failure.

**HumanizerAI prompt-skill test (2026).** Open-source "humanizer" agent skills (vocabulary blocklists, specificity rules) run on 30 texts against GPTZero's API: baseline 66.7% bypass; adding a 20-word AI-vocab ban *lowered* bypass to 30%; both constraints 23.3%. Banning "delve" makes the model reach for equally stiff substitutes; GPTZero tracks sentence-length variance and token predictability, not word lists.

**Human raters (PMC12752165, 2025).** Humans identified AI text with 19% accuracy (chance = 20%); free detectors dropped from ~92% on pure AI to ~43-45% on heavily AI-edited text.

**Reddit consensus** (r/college, r/ChatGPT, r/gradadmissions, r/WritingWithAI via HumanGPT's summary; direct Reddit fetches are blocked): Undetectable and StealthWriter get the most mentions and "work reasonably well"; QuillBot free "gets flagged"; students rotate free tools because "a detector update can knock out whichever one they used last"; the only durable advice is "humanize, paste into a detector, read the score."

### A.4 Observed output artifacts

Compiled from Pangram, DAMAGE, ProofreaderPro, fast.io, WriteHybrid, Gold Penguin and Trustpilot excerpts:

- **Tortured synonym swaps**: "artificial intelligence" → "counterfeit consciousness"; "I need to get my car fixed" → "I require to obtain my vehicle repaired"; "multicollinearity" → "multiple connections." Characteristic of L3 tools and of QuillBot-style paraphrasers in aggressive settings.
- **Nonsense insertions**: "(??????)", "CGSizeMake pp 18-23", hallucinated citations, random clauses -- apparently deliberate perplexity injection.
- **Character-level tricks**: thin spaces (U+2009) replacing spaces, removed inter-sentence spaces to confuse tokenizers. RAID shows why: homoglyphs and zero-width spaces are the most damaging cheap attacks against most detectors (except GPTZero, which normalizes).
- **Deliberate misspellings and grammar breaks**: "recieves," "easly," "Now a days technology let us convenience," stray mid-sentence capitals (Humbot).
- **Register flattening**: academic phrasing rewritten conversationally; formal hedges replaced with "we can see that"; forced grade-level vocabulary.
- **Citation corruption**: parenthetical citations relocated or reformatted (WriteHuman, Undetectable), mangled outright in 30% of samples (Humbot).
- **Factual drift**: dates, statistics and technical terms altered in Aggressive modes (Humbot, Rephrasy, Ryne on academic text).
- **Length sensitivity**: evasion and quality both drop past ~1,000 words (StealthGPT, WriteHuman); most tools cap runs at 400-3,000 words, forcing chunked processing with tone discontinuities.
- **Em-dash and transition purges**: open-source skills (harshaneel/humanize: max one em dash per 300 words, no semicolons, strip "Furthermore/Moreover," inject <6-word sentences every 150 words) and some tools (WriteBros) explicitly remove em-dashes. These change surface stylometry but, per the harshaneel authors themselves, learned classifiers like GPTZero still flag the output because they detect "RLHF fingerprints encoded in model weights," not punctuation.
- **Circular validation**: several tools (StealthGPT, Netus, Smodin, WriteHuman) ship a built-in detector that scores their own output as human while external detectors disagree.

### A.5 Reverse engineering and leaked prompts

No proprietary humanizer's model or prompt has leaked. What exists:

- The "Humanizer Pro" custom GPT (system prompt archived in linexjlin/GPTs, Nymbo/Leaked-GPTs-Prompts, DevIsper/Prompts-ChatGPT) is mostly prompt-protection boilerplate; substantively it instructs mimicry of the user's prior style, a middle-register tone, and cites Narrato's "tips to avoid AI detection," a ScienceDirect piece and arXiv 2301.10416 as knowledge files. It confirms the wrapper-around-GPT-4 pattern for the long tail of "humanizer GPTs."
- Behavioral inference: tools whose output is fluent and register-preserving (StealthGPT, GPTinf, HumanizeAI.pro, QuillBot, Grammarly -- DAMAGE's L1) behave like LLM paraphrasers with a style prompt, and are easiest for Pangram to catch. Tools whose output shows tokenizer-level damage and nonsense (Undetectable, Humbot, HIX, WriteHuman -- L3) behave like perturbation attacks (word-importance ranking + masked substitution, à la HMGC/Shi et al.) layered on paraphrase, and are hardest for Pangram (90.3% for Undetectable). This is the same fluency-vs-evasion tradeoff the academic literature measures.
- StealthGPT's "optimize for Turnitin or GPTZero, not both" toggle implies a detector-in-the-loop training or selection step against specific proxies, consistent with Nicks-style DPO or best-of-n filtering.
- The "Pro/Enhanced/Latest" tiers (WriteHuman Enhanced, HIX Latest, StealthWriter Ghost Pro, Phrasly Pro Engine, Ryne Pro Algorithm) are marketed as newer models targeting the latest Originality/Turnitin releases -- evidence that vendors retrain against detector updates on a months-long cadence.

---

## Part B. Academic evasion literature

### B.1 Krishna et al. 2023 -- DIPPER (arXiv 2303.13408, NeurIPS 2023)
**Method.** DIPPER is an 11B T5-XXL paraphraser trained on aligned PAR3 novel translations, conditioned on surrounding context, with two scalar controls: lexical diversity L ∈ {20..100} and order diversity O (Kendall-tau reordering). It paraphrases whole passages, not sentences.
**Detectors beaten.** At 1% FPR on GPT2-XL: watermarking 100%→57.2%; DetectGPT 70.3%→4.6%; OpenAI classifier 21.6%→14.8%; GPTZero 13.9%→1.2%. OPT-13B watermark 100%→52.8%. GPT-3.5: DetectGPT 26.5%→4.5%.
**Meaning.** P-SP semantic-similarity model; 88-99% of DIPPER outputs exceed the median P-SP of human paraphrases; human evaluation confirmed.
**Defense.** Retrieval over a 15M-generation store recovers 80.4% (Wikipedia) to 97.3% (PG19) at 1% FPR; BM25 beats dense retrievers.
**Code.** github.com/martiansideofthemoon/ai-detection-paraphrases (models, data). DIPPER is now itself a baseline "humanizer" in Pangram/DAMAGE audits (97.6% caught by Pangram).

### B.2 Sadasivan et al. 2023 -- "Can AI-Generated Text be Reliably Detected?" (arXiv 2303.11156)
**Method.** Recursive paraphrasing (T5 222M, DIPPER, LLaMA-2-7B-chat) applied 1-5 times, plus a theoretical bound tying best-case AUROC to total-variation distance between human and AI text distributions, and spoofing attacks that make human text look AI.
**Detectors beaten.** Soft watermark on OPT-13B: TPR@1%FPR 99.8%→80.7% (DIPPER, one round), 54.6% (LLaMA-2), <20% after five rounds, 4.0% with best-of selection. OpenAI RoBERTa-Large 100%→92%→80% with multiple queries. DetectGPT AUROC 96.5%→59.8%. Retrieval defense 100%→<60% under recursion.
**Meaning.** MTurk: 77% of recursively paraphrased passages rated high content-preservation, 89% high grammar; perplexity rose 5.2→8.7-10.5; SQuAD-v2 QA accuracy on paraphrased passages held at 97%.
**Spoofing.** One million queries suffice to estimate a watermark green list and drive AUROC 99.8%→1.3%.
**Code.** github.com/vinusankars/Reliability-of-AI-text-detectors.

### B.3 Lu et al. 2023 -- SICO (arXiv 2305.10847, ICLR 2024)
**Method.** Substitution-based In-Context example Optimization: starting from 40 human examples, iteratively edit in-context exemplars via WordNet word substitution and feature-guided sentence paraphrase, scored by a proxy detector, so that the LLM (GPT-3.5) *generates* evasive text directly with no post-hoc paraphraser. Tasks: essay completion, open QA, fake reviews.
**Detectors beaten.** Six: GPT-3.5 detector, GPT-2 detector, DetectGPT, Log-Rank, GPTZero, OpenAI detector. Average AUC drop 0.5. Examples (essay task): GPT-3.5 detector 0.908→0.239; GPT-2 detector 0.848→0.332; DetectGPT 0.834→0.149. Prompts optimized against one detector transfer to the others.
**Meaning.** Human readability 3.84 vs 3.92 for human text; task completion 93.6% vs 98.2%; humans could not distinguish SICO text.
**Code.** github.com/ColinLu50/Evade-GPT-Detector.

### B.4 Shi et al. 2023 -- "Red Teaming Language Model Detectors with Language Models" (arXiv 2305.19713, TACL 2024)
**Method.** Two black-box attacks using an auxiliary LLM: (1) contextual word substitution -- the LLM proposes synonyms, selected either query-free (random) or by evolutionary search against detector scores; (2) instructional prompt search -- append a style instruction to the generation prompt so ChatGPT writes evasively from the start.
**Detectors beaten.** DetectGPT AUROC on XSum 84.4%→25.9% (query-free) →3.9% (query-based); on ChatGPT text 56.0%→25.6%→4.5%. OpenAI classifier −45.4 points AUROC (query-based, XSum). Watermarking degraded but most resilient. RoBERTa-Large also attacked.
**Meaning.** Human fluency/plausibility on a 1-3 scale: query-free substitution kept fluency best; query-based showed "small degradation."
**Code.** github.com/shizhouxing/LLM-Detector-Robustness.

### B.5 Nicks et al. 2023/2024 -- "Language Model Detectors Are Easily Optimized Against" (ICLR 2024; OpenReview 4eJDMjYZZG)
**Method.** Use a detector's "humanness" score to build pairwise preferences between sampled completions and fine-tune the generator with DPO under a KL constraint -- RLHF with a detector as the reward model. Data-efficient: Llama-2-7B, under one day of training.
**Detectors beaten.** OpenAI RoBERTa-Large AUROC 0.84→0.63 on the 7B model; the TL;DR reports that evasion transfers to detectors not used for training, including commercial ones. (OpenReview blocked direct fetch; numbers from the paper's public summary and DAMAGE's citation.)
**Meaning.** KL-regularized; the paper reports generation quality held within the constraint budget -- I could not verify the specific metric.
**Code.** Not located. The recipe is reproduced by HUMPA, AuthorMist, MASH and StealthRL, and Soto et al. use it as their canonical "optimization attack."

### B.6 Wang/Zhou et al. 2024 -- HMGC "Humanizing Machine-Generated Content" (arXiv 2404.01907)
**Method.** Train a RoBERTa surrogate on the target detector's predictions (black-box) or use the detector directly (white-box); rank words by gradient norm plus perplexity change on deletion; mask-and-replace with an encoder LM's top candidates via greedy search; constraints: POS agreement, ≤40% of tokens perturbed, Universal Sentence Encoder similarity floor. A "dynamic adversarial learning" loop retrains the detector on adversarial samples across rounds.
**Detectors beaten.** HC3 RoBERTa (white-box): 97.29% attack success (TNR reduction); CheckGPT (black-box): 46.35%. Claims detectors "can be compromised in as little as 10 seconds."
**Meaning.** USE similarity, Flesch reading ease change, Pythia-3B perplexity -- not human evaluation. Adversarially retrained detectors recover only partially.
**Code.** github.com/zhouying20/HMGC.

### B.7 Zhou/Wang et al. 2024 -- HUMPA "Humanizing the Machine: Proxy Attacks" (arXiv 2410.19230, ICLR 2025)
**Method.** RL-fine-tune a *small* proxy LM (Llama2-7B, Llama3-8B, Mistral-7B) with DPO where preferences come from a scoring detector's humanness; then at decoding time blend/override the large source model's (Llama2-13B, Llama3-70B, Mixtral-8x7B) next-token distribution with the humanized SLM. β controls the KL/quality tradeoff.
**Detectors beaten.** White-box: Likelihood, LogRank, LRR, NPR, DNA-GPT, DetectGPT, Fast-DetectGPT. Black-box: RoBERTa-base/large, GPT-Neo-2.7B surrogates, Binoculars. Average AUROC drop 70.4%, max 90-95% on a single dataset; e.g., Fast-DetectGPT on Mixtral 0.876→0.259; Binoculars 0.935→0.305. Cross-discipline and cross-language relative drops up to 90.9% and 91.3%.
**Meaning.** BERTScore and ROUGE-1/2/L within a "utility budget" (ΔBERTScore ≤0.02, ΔROUGE-1 ≤0.03) -- automatic only.
**Code.** None linked in paper or on the ICLR page.

### B.8 Cheng, Sadasivan et al. 2025 -- "Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text" (arXiv 2506.07001, NeurIPS 2025)
**Method.** Training-free. LLaMA-3-8B-Instruct paraphrases under a system prompt; at each decoding step, top-p/top-k candidate tokens are appended to the partial output and scored by a guidance detector (OpenAI-RoBERTa-Large); the token with the lowest AI-score is emitted. Insight: strong detectors converge on a shared "human distribution," so evading one transfers.
**Detectors beaten.** Eight: OpenAI-RoBERTa-Base/Large, MAGE, RADAR (neural); KGW and Unigram watermarks; Fast-DetectGPT and GLTR (zero-shot). Average T@1%F reduction 87.88%; RADAR −64.49%, Fast-DetectGPT −98.96%. Simple paraphrasing *increased* detection on RADAR (+8.57%) and Fast-DetectGPT (+15.03%).
**Meaning.** SBERT cosine 0.81 (vs 0.86 for plain paraphrase); perplexity 14.3-17.1, close to human 15.0 (plain paraphrase 9.3 -- i.e., too smooth); GPT-4o rated 87% of outputs 4-5/5 with no significant difference from plain paraphrase.
**Code.** github.com/chengez/Adversarial-Paraphrasing.

### B.9 Dugan et al. 2024 -- RAID (arXiv 2405.07940, ACL 2024)
**Method.** 6.2M generations, 11 models, 8 domains, 4 decoding strategies, 11 adversarial attacks: alternative spelling, article deletion, homoglyph, paragraph insertion, number swap, T5-11B paraphrase, misspelling, BERT synonym replacement, case swap, whitespace insertion, zero-width spaces. 12 detectors (8 open, 4 commercial) at 5% FPR.
**Findings.** Homoglyphs cut Binoculars accuracy 41.9 points, RoBERTa-GPT2 35.4, GLTR 38.3 -- but GPTZero lost only 0.3 (it normalizes Unicode). Synonym replacement cut Binoculars 36.1. Zero-width spaces broke most open detectors. Originality *improved* under paraphrase (+11.7) and synonym (+11.5) attacks, suggesting it was trained on such data. Paraphrase alone was mid-pack -- weaker than character attacks against open detectors.
**Meaning.** Not measured (attacks are fixed perturbations).
**Code.** github.com/liamdugan/raid; pip install raid-bench; leaderboard raid-bench.xyz.

### B.10 Artemova et al. 2024 -- Beemo (arXiv 2411.04032, NAACL 2025)
**Method.** 6.5k texts by humans, ten instruction-tuned LLMs, and expert human editors, plus 13.1k LLM-edited variants; 33 detector configurations benchmarked.
**Findings.** Expert human editing of machine text evades detectors; LLM-edited machine text remains detectable as machine. Mirrors the commercial finding that fluent LLM paraphrase does not erase the machine signal but human intervention does.
**Code/data.** github.com/Toloka/beemo; huggingface.co/datasets/toloka/beemo.

### B.11 Masrour et al. 2025 -- DAMAGE (arXiv 2501.03437) -- see A.3. Defensive paper; the only academic audit of the actual commercial tools. No code.

### B.12 David & Gervais 2025 -- AuthorMist (arXiv 2503.08716)
**Method.** 3B LM fine-tuned with GRPO using *external commercial detector APIs* (GPTZero, WinstonAI, Originality.ai, others) as reward.
**Results.** Attack success 78.6-96.2% per detector; semantic similarity >0.94. StealthRL later reports AuthorMist at mean AUROC 0.454 on its benchmark.
**Code.** Not confirmed.

### B.13 Soto, Chen & Andrews 2025 -- "Attacks on Machine-Text Detectors Retain Stylistic Fingerprints" (arXiv 2505.14608)
**Method.** Evaluate Nicks-style DPO, DIPPER/GPT-4o-mini paraphrase, OUTFOX adversarial prompting, style-aware prompting and TinyStyler against standard detectors and against **StyleDetect**, an authorship-embedding detector seeded with 100 examples of the unattacked model.
**Findings.** DPO against Fast-DetectGPT drives its AUROC to 35 while StyleDetect stays at 96; TinyStyler drops StyleDetect only to 87 (Binoculars 92). Evasion attacks target "machine-ness" but leave an author fingerprint. Their new attack -- SFT on human paraphrase pairs conditioned on target-author exemplars, DPO against a detector trained on its own outputs, iterative inference-time refinement -- evades everything on single documents but is caught when N>1 documents are pooled.
**Code.** github.com/rrivera1849/style-aware-paraphrasing.

### B.14 Self-Disguise Attack 2025 (arXiv 2508.15848)
Prompt-only: an adversarial feature extractor produces "disguise features" and a retrieval-based context optimizer selects in-context exemplars; reduces average detector accuracy while preserving quality (numbers not in abstract; code not indicated).

### B.15 PADBen 2025 (arXiv 2511.00416)
Benchmark with a five-type taxonomy from original to "deeply laundered" text across 11 detectors and five tasks. Detectors hold up on plagiarism-style paraphrase of AI text but fail on authorship obfuscation; iterative paraphrasing creates a "laundering region" of semantic displacement with preserved generation patterns. Code on GitHub.

### B.16 MASH 2026 -- "Evading Black-Box Detectors via Style Humanization" (arXiv 2601.08564)
**Method.** Three stages: style-injection SFT on paired (AI, human-style) texts; DPO using detector confidence with hard negatives; LLM sentence-level refinement at inference that preserves the detector verdict.
**Detectors beaten.** RoBERTa, Binoculars, SCRN, and commercial Writer and Scribbr across six datasets: ~92% average attack success, ~24 points over the best baselines (DIPPER, PromptAttack, Charmer, DPO-Evader, CoPA, GradEscape).
**Meaning.** Perplexity, BERTScore, GRUEN.
**Code.** github.com/githigher/MASH.

### B.17 StealthRL 2026 (arXiv 2602.08934)
**Method.** GRPO with LoRA on Qwen3-4B; reward R = 1.0·R_det + 0.1·R_sem, where R_det is a 60/40 ensemble of RoBERTa and Fast-DetectGPT and R_sem is E5 cosine similarity; implicit KL 0.05. Binoculars and MAGE held out.
**Detectors beaten.** On MAGE test: RoBERTa TPR@1%FPR 0.225→0.002; Fast-DetectGPT AUROC 0.671→0.071; held-out Binoculars AUROC 0.716→0.041; mean AUROC 0.74→0.27; 97.6-99.9% attack success. Baselines: plain paraphrase 0.589 AUROC, adversarial paraphrasing 0.595, AuthorMist 0.454, homoglyph 0.437.
**Meaning.** E5 similarity 0.896; but gpt-5-nano Likert quality **2.59/5 vs 4.01** for plain paraphrase -- the clearest quantified quality cost in the literature.
**Code.** github.com/suraj-ranganath/StealthRL.

### B.18 "Base Models Look Human To AI Detectors" 2026 (arXiv 2605.19516)
Base (non-instruction-tuned) Llama-3/Qwen-3 outputs 0.6B-70B are judged overwhelmingly human by GPTZero and Pangram while instruct-tuned siblings are caught; HIP (Humanization by Iterative Paraphrasing) minimally fine-tunes a base model as a paraphraser and iterates. Implication: detectors largely track RLHF/instruction-tuning artifacts -- the same conclusion the harshaneel skill authors reached empirically.

### B.19 ARB 2026 (arXiv 2607.29539)
1,800 human sources × four open LLMs; FastDetectGPT/Binoculars recall at 1% FPR falls from 91.2%/93.5% on direct LLM text to 30.8%/15.1% on human text *rewritten* by an LLM, yet stays 78-83% on LLM-rewritten LLM text -- the human seed matters more than the paraphrase.

---

## Techniques ranked by evidence of effectiveness

1. **Detector-in-the-loop RL/DPO of the generator or a paraphraser** (Nicks 2024; HUMPA; AuthorMist; MASH; StealthRL). Strongest and most reproducible: AUROC of open detectors driven to 0.04-0.30, 70-99% attack success, transfer to held-out detectors demonstrated three times. Cost: training compute, and StealthRL shows a large fluency penalty unless a semantic/KL term is tuned. Commercial "Pro/Latest engine" tiers most plausibly use this.
2. **Detector-guided decoding without training** (Cheng et al. adversarial paraphrasing). 87.9% average T@1%F reduction across eight detectors including watermarks, human-level perplexity, GPT-4o quality parity. Cheapest strong method; likely what a well-engineered commercial humanizer with a "consensus detector" does at inference.
3. **Style transfer toward a specific human author** (Soto et al.; MASH stage 1; Rephrasy/Twixify "style cloning" commercially). Only method shown to beat authorship-embedding detectors on single documents; caught again with multi-document pooling.
4. **Whole-passage LLM paraphrase, especially recursive** (DIPPER; Sadasivan). Reliably beats zero-shot and watermark detectors (DetectGPT 70→5%, watermark 100→52-57%), but in 2025-26 it is the *easiest* class for retrained commercial detectors (Pangram 97.6% on DIPPER; Turnitin AIR-1/bypasser models; Originality improves under RAID paraphrase). Plain paraphrase can even raise detection (Cheng et al.).
5. **Word-importance perturbation / synonym substitution** (Shi et al.; HMGC; RAID synonym attack). Very effective against likelihood detectors (DetectGPT 84→4%) and fine-tuned RoBERTas; produces the "tortured phrase" artifacts seen in L3 commercial tools; weak against Pangram/Turnitin once trained on it.
6. **Prompt-only evasion** (SICO; Shi et al. instructional prompts; Self-Disguise; open-source "humanizer" skills). SICO's proxy-optimized exemplars cut AUC by 0.5 in 2023, but naive rule prompts (ban "delve," vary sentence length, kill em-dashes) tested in 2026 *reduce* bypass against GPTZero (66.7%→23.3%).
7. **Character-level obfuscation** (RAID homoglyph, zero-width, thin-space; seen in Pangram's audit). Devastating to open detectors (−35 to −42 points) and trivially cheap, but neutralized by Unicode normalization (GPTZero −0.3) and produces visibly broken text.
8. **Generating from base models** (HIP). Emerging; suggests the strongest long-run humanizer is not a rewriter but a differently-tuned generator.

## Quality/meaning-preservation tradeoffs observed

- **Fluency and evasion are anti-correlated in practice.** Pangram's audit, DAMAGE's L1/L2/L3 tiering and StealthRL's 2.59-vs-4.01 Likert gap all show the same curve: tools that keep register and grammar (Grammarly, QuillBot, GPTinf, StealthGPT) are detected at 95-100%; tools that evade best (Undetectable at 90.3% Pangram recall, 94% raw bypass in academic tests) do so by adding nonsense, flattening register and corrupting citations.
- **Semantic metrics hide the damage.** Papers report SBERT 0.81-0.90, BERTScore deltas ≤0.02, USE floors, ROUGE budgets -- yet Cheng et al. show perplexity rising from 9 to 14-17 and StealthRL's LLM judge scores collapse. Embedding similarity does not capture "multicollinearity → multiple connections," relocated citations, or altered statistics. Only Sadasivan (MTurk + QA), SICO (human readability/task completion) and Cheng (GPT-4o rating) ran meaningful quality evaluations.
- **Domain matters.** Every commercial test that split by genre found the same ordering: marketing/casual copy survives humanization (8.5/10), blog posts degrade with length (6/10 past 500 words), academic and technical text fares worst (4-4.5/10, grammar errors, term substitution, factual drift). Ryne's own data: 45% detection drop on creative text, 6% on academic.
- **Length is a hard limit.** Quality and evasion both fall past ~1,000 words; run caps of 400-3,000 words force chunking, which introduces tone discontinuities detectable by the multi-document/stylometric methods Soto et al. describe.
- **Recursive or aggressive settings trade meaning for score.** Sadasivan's five rounds keep 77% content fidelity; HIX/Humbot/StealthWriter "Aggressive" modes are where reviewers report gibberish, mid-sentence capitals and changed dates.
- **The cheapest artifacts are the most fragile.** Unicode tricks, typos and em-dash purges are the first thing detector vendors normalize or train on; they buy weeks, not years, and permanently damage the document.
- **Human editing is the one humanization that keeps quality and evades** (Beemo; ARB's human-seed result), which is also why Karr's finding -- honest light editing flagged 38-80% while Undetectable output flagged <4% -- indicts detector-based enforcement more than it validates humanizers.

---

## Sources

### Commercial tools, tests and industry
- Undetectable.ai pricing: https://undetectable.ai/pricing ; homepage: https://undetectable.ai/ ; mode list via https://diyai.io/ai-tools/productivity/reviews/undetectable-ai-review/
- Originality.ai review of Undetectable.ai: https://originality.ai/blog/undetectable-ai-review
- Originality.ai accuracy/humanizer study (Sept 2025): https://originality.ai/blog/ai-accuracy
- Originality.ai reviews: GPTinf https://originality.ai/blog/gptinf-review ; Humanize AI Pro https://originality.ai/blog/humanizeai-pro-review ; Surfer https://originality.ai/blog/surfer-humanizer-review ; Netus https://originality.ai/blog/netus-ai-review
- GPTZero, "Detecting AI-Humanized Text": https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/
- Pangram humanizer audit (Jan 2025): https://www.pangram.com/blog/humanizers-announcement ; Aug 2025 update: https://www.pangram.com/blog/humanizers-aug-25 ; "What is a humanizer?": https://www.pangram.com/blog/what-is-a-humanizer
- Jabarian & Imas, "Artificial Writing and Automated Detection," BFI WP 2025-116: https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf ; Chicago Booth Review summary: https://www.chicagobooth.edu/review/2025/december/do-ai-detectors-work-well-enough-trust
- Turnitin AI bypasser detection press release (27 Aug 2025): https://www.turnitin.com/press/turnitin-expands-capabilities-amid-rising-threats-posed-by-ai-bypassers ; Plagiarism Today coverage: https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/
- Plagiarism Today on Grammarly humanizer/Authorship: https://www.plagiarismtoday.com/2025/11/06/how-grammarly-launders-ai-generated-content/ and https://www.plagiarismtoday.com/2025/12/11/grammarly-updates-authorship-improves-labeling/
- Grammarly AI Humanizer page: https://www.grammarly.com/ai-humanizer
- NBC News on students and humanizers: https://nbcnews.com/tech/internet/college-students-ai-cheating-detectors-humanizers-rcna253878
- ProofreaderPro academic humanizer test (vendor): https://proofreaderpro.ai/blog/best-ai-humanizers-2026
- HumanizerAI prompt-skill vs GPTZero test (vendor): https://humanizerai.com/blog/gptzero-bypass-test-2026
- harshaneel/humanize open-source skill: https://github.com/harshaneel/humanize
- Leaked "Humanizer Pro" GPT prompt: https://github.com/linexjlin/GPTs/blob/main/prompts/Humanizer%20Pro.md ; https://github.com/DevIsper/Prompts-ChatGPT/blob/main/prompts/Humanizer%20Pro.md
- COCOSOFT ownership of HIX/Humbot/BypassGPT: https://trademarks.justia.com/owners/cocosoft-technology-pte-ltd-5864608
- Reddit consensus summary: https://humangpt.io/blog/best-ai-humanizer-reddit-2026
- StealthGPT reviews: https://aixradar.com/stealthgpt-review/ ; https://www.undetectedgpt.ai/blog/stealthgpt-review (competitor) ; https://phrasly.ai/blog/stealthgpt-review-does-it-work (competitor)
- StealthWriter review: https://fast.io/resources/stealthwriter-ai-review-2026/
- HIX Bypass review (competitor, disclosed): https://humanizemy.ai/vs/hix-ai
- Humbot review: https://www.writehybrid.com/humanizers/humbot
- WriteHuman review: https://fast.io/resources/writehuman-ai-review-2026/
- Phrasly review: https://fast.io/resources/phrasly-ai-review-2026/
- Rephrasy review (competitor): https://detectiondrama.com/rephrasy-review/
- Walter Writes review: https://mohababdelkarim.substack.com/p/walter-writes-ai-review-2026-features ; competitor test: https://www.stealthgpt.ai/blog/can-walter-writes-ai-pass-ai-checkers
- Twixify: https://www.twixify.com/ ; competitor review https://writehuman.ai/blog/twixify-review
- Smodin review: https://verva.com/blog/smodin-review
- Netus AI review (Gold Penguin): https://goldpenguin.org/blog/netus-ai-review-and-ai-bypasser/
- QuillBot humanizer review (competitor): https://aitohuman.ai/quillbot-ai-humanizer-review
- Ryne AI review: https://www.essaydone.ai/tools-review/ryne-ai-humanizer-review.html
- Androidheadlines multi-tool GPTZero test: https://www.androidheadlines.com/2025/10/which-ai-humanizer-passes-gptzero-turnitin-and-copyleaks.html
- Pangram vs humanizers (detectiondrama): https://detectiondrama.com/bypass-pangram-ai-detector/
- Human-rater/free-detector study (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12752165/

### Academic papers
- Krishna et al. 2023, DIPPER: https://arxiv.org/abs/2303.13408 ; code https://github.com/martiansideofthemoon/ai-detection-paraphrases
- Sadasivan et al. 2023: https://arxiv.org/abs/2303.11156 ; code https://github.com/vinusankars/Reliability-of-AI-text-detectors
- Lu et al. 2023, SICO: https://arxiv.org/abs/2305.10847 ; code https://github.com/ColinLu50/Evade-GPT-Detector
- Shi et al. 2023, Red Teaming LM Detectors: https://arxiv.org/abs/2305.19713 ; code https://github.com/shizhouxing/LLM-Detector-Robustness
- Nicks et al. 2024, ICLR: https://openreview.net/forum?id=4eJDMjYZZG
- Wang/Zhou et al. 2024, HMGC: https://arxiv.org/abs/2404.01907 ; code https://github.com/zhouying20/HMGC
- Zhou et al. 2024, HUMPA (ICLR 2025): https://arxiv.org/abs/2410.19230
- Dugan et al. 2024, RAID: https://arxiv.org/abs/2405.07940 ; code https://github.com/liamdugan/raid
- Artemova et al. 2024, Beemo: https://arxiv.org/abs/2411.04032 ; https://github.com/Toloka/beemo
- Masrour et al. 2025, DAMAGE: https://arxiv.org/abs/2501.03437
- David & Gervais 2025, AuthorMist: https://arxiv.org/abs/2503.08716
- Soto, Chen & Andrews 2025, Stylistic Fingerprints: https://arxiv.org/abs/2505.14608 ; code https://github.com/rrivera1849/style-aware-paraphrasing
- Cheng, Sadasivan et al. 2025, Adversarial Paraphrasing (NeurIPS 2025): https://arxiv.org/abs/2506.07001 ; code https://github.com/chengez/Adversarial-Paraphrasing
- Self-Disguise Attack 2025: https://arxiv.org/abs/2508.15848
- Karr 2025, "Why AI Detection Fails for Academic Integrity": https://arxiv.org/abs/2608.11256
- PADBen 2025: https://arxiv.org/abs/2511.00416
- MASH 2026: https://arxiv.org/abs/2601.08564 ; code https://github.com/githigher/MASH
- StealthRL 2026: https://arxiv.org/abs/2602.08934 ; code https://github.com/suraj-ranganath/StealthRL
- Base Models Look Human 2026: https://arxiv.org/abs/2605.19516
- ARB 2026: https://arxiv.org/abs/2607.29539
