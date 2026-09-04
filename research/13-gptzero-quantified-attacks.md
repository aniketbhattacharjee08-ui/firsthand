# Quantified Attacks Against GPTZero Specifically

Numbers extracted from ~25 arXiv PDFs plus vendor benchmarks. This is the single most decision-relevant evidence file: it is the only place with per-attack, GPTZero-specific deltas rather than detector-averaged claims.

**Critical caveat that applies to every table below:** GPTZero versions differ radically across 2023-2026. Its baseline TPR@1%FPR was 7-14% in the 2023 DIPPER paper and 98.6% in 2026 evaluations. Never compare an attack number from 2023 against one from 2026. Vendor-run benchmarks are also irreconcilable and each favors its author (GPTZero claims 93.5% recall on bypassers; Pangram measures GPTZero at 44-50% on the same problem).

---

## 1. RAID (Dugan et al., ACL 2024): GPTZero accuracy at 5% FPR, per attack

Thresholds calibrated per domain to 5% FPR. GPTZero's baseline is low (66.5%) because RAID includes GPT-2/GPT-3-era generators. The signal to read is the **delta**, not the level.

| Attack | GPTZero | Δ | Originality | ZeroGPT | Winston | Binoculars | RADAR |
|---|---|---|---|---|---|---|---|
| None | 66.5 | — | 85.0 | 65.5 | 71.0 | 79.6 | 70.9 |
| Alternative spelling (British) | 64.9 | −1.6 | 83.6 | 65.4 | 68.9 | 78.2 | 70.8 |
| Article deletion | 61.0 | −5.5 | 71.4 | 59.7 | 66.9 | 74.3 | 67.9 |
| Homoglyph | 66.2 | −0.3 | 9.3 | 82.4 | 26.3 | 37.7 | 59.3 |
| Insert paragraphs | 66.2 | −0.3 | 85.1 | 64.9 | 69.8 | 71.7 | 73.7 |
| Number swap | 65.8 | −0.7 | 86.0 | 64.7 | 69.0 | 77.1 | 71.0 |
| Paraphrase (DIPPER) | 64.0 | −2.5 | 96.7 | 46.7 | 52.6 | 80.3 | 67.3 |
| Misspelling | 65.1 | −1.4 | 78.6 | 64.7 | 67.5 | 78.0 | 69.5 |
| Synonym swap (BERT) | 61.0 | −5.5 | 96.5 | 18.8 | 63.6 | 43.5 | 67.5 |
| Upper-lower swap | 56.5 | −10.0 | 75.8 | 54.5 | 56.8 | 73.8 | 70.4 |
| Whitespace addition | 66.2 | −0.3 | 84.9 | 64.2 | 46.8 | 70.1 | 66.1 |
| Zero-width space | 66.2 | −0.3 | 4.9 | 48.0 | 25.0 | 99.1 | 82.2 |

**GPTZero was the most attack-robust detector in RAID.** No attack cost more than 10 points. Paper quote: "All detectors were sensitive to homoglyph attacks except for GPTZero which sustained only a 0.3% loss ... while five others dropped an average of 40.6%."

Ranked GPTZero damage: upper/lower swap (−10.0) > article deletion and synonym swap (−5.5) > paraphrase (−2.5) > misspelling (−1.4) > everything else (≤−0.7).

---

## 2. Academic studies with GPTZero-specific numbers, chronological

**Krishna et al. 2023, DIPPER** (arXiv 2303.13408). Detection accuracy at 1% FPR, early-2023 GPTZero. GPT-2 XL 13.9 → 9.1 (20% lexical) → 7.3 (40%) → 7.1 (60%) → 1.2 (60% lex + 60% reorder). GPT-3.5 davinci-003 7.1 → 1.8 at max diversity. Authors note GPTZero detected under 15% of *non*-paraphrased AI text at 1% FPR, so the baseline was near-useless. 88-99% of DIPPER paraphrases exceeded the median semantic similarity of human paraphrases.

**Cai & Cui 2023, SpaceInfi** (2307.02599). Inserting a single space before a random comma dropped GPTZero detection "from roughly 60-80% to nearly 0%". Quality cost: one character. **Patched** — RAID's 2024 whitespace attack showed −0.3.

**Liang et al. 2023** (2304.02819). GPT-3 college essays 100% flagged → 12.9% after the prompt "Elevate the provided text by employing literary language". CS224N abstracts 13.8% → 0.7%. Real TOEFL essays: 51.6% falsely flagged. Real 8th-grade essays 0% → 38.6% flagged after "simplify word choices" — simplification makes human text look AI.

**Ibrahim et al. 2023, Scientific Reports.** 280 questions, 32 courses. GPTZero FPR 18%, FNR 32% raw. After a QuillBot paraphrase, FNR rose 32% → 95%.

**Weber-Wulff et al. 2023** (2306.15666). GPTZero overall 54%, rank ~7 of 14 tools. AI plus manual synonym edits and AI plus QuillBot Standard both dropped it to 3/9.

**Lu et al. 2023, SICO** (2305.10847). AUC of GPTZero with GPT-3.5 as generator, Writing task: 0.779 baseline → DIPPER 0.689 → GPT-paraphrase 0.631 → the folk "perplexity/burstiness" prompt 0.491 → **SICO-Para 0.290 → SICO-Gen 0.184**. Review task 0.939 → 0.399. Using GPTZero itself as the proxy detector gave 0.299. **SICO beat DIPPER on both task completion and readability in human evaluation** — a rare case of evasion improving rather than degrading quality.

**DUPE 2024** (2404.11408). 208 ChatGPT essays. GPTZero FPR 0.00%, FNR 2.5% baseline. Paraphrase prompts: "increase perplexity" → FNR 22.5%; **"rewrite like a college student" → 51.25%**; recursive → 1.25%. Critical caveat: under the stricter "must receive Human label" regime only **1 of 400** attacked texts passed. Lowering the AI probability is not the same as earning a human verdict.

**Perkins et al. 2024** (2403.19148). n=805. GPTZero accuracy 26.4% → 16.7% (−9.7). Technique means across all detectors: spelling errors 12.9%, burstiness 15.9%, paraphrase 18.4%, decrease complexity 21%, non-native-speaker style 27.7%, **increase complexity 37% (least effective)**.

**ESPERANTO 2024, back-translation** (2409.14285). TPR at 1% FPR, GPTZero: News 0.97 → 0.42; Abstracts 1.00 → 0.82; Reddit-GPT 1.00 → 0.43; Reddit-Llama 0.65 → 0.09.

**HUMPA, ICLR 2025** (2410.19230). Llama2-13B with a 7B proxy fine-tuned against GPTZero: AUROC 0.9951 → 0.8295 (α=1.5) → 0.7987 (α=2.0). SBERT similarity 0.819 → 0.794. Roughly one point of similarity per 0.2 AUROC.

**DAMAGE (Pangram Labs) 2025** (2501.03437). 19 humanizers, academic text, TPR@5%FPR: **GPTZero 99.73% raw → 60.04% humanized**. Binoculars 94.15 → 28.23. GPT-4o fluency win-rate versus the original by humanizer tier: 26% / 14.7% / 2.7%. Best-fluency tier: DIPPER, GPTInf, Grammarly, HumanizeAI.pro, QuillBot, Semihuman, StealthGPT, Twixify. Worst: Ghost AI, HIX Bypass, HumanizeAI.io, Humbot, Phrasly, Undetectable AI, WriteHuman.

**AuthorMist 2025, GRPO RL** (2503.08716). Trained with the GPTZero API as the reward signal: **92.33% attack success on GPTZero**, AUROC 1.00 → 0.79, F1 → 0.14, **semantic similarity median ≈ 0.945**. DIPPER baseline ASR was ~40%. GPTZero was among the two most resilient detectors to models trained against *other* detectors.

**Russell et al. 2025** (2501.15654). TPR(FPR), 30 articles per condition. GPTZero: GPT-4o 100(0), Claude 96.7(0), GPT-4o paraphrased 100(0), o1-Pro 76.7(0), **o1-Pro humanized via an expert-guidebook prompt 46.7(3.3)**. Pangram scored 90.0 on that same humanized set; Binoculars 6.7.

**Jabarian & Imas (UChicago/NBER) 2025.** StealthGPT default rewrite. GPTZero FNR before ≤0.05; after, by genre, 0.15-0.95, mean ≈ 0.53. News was worst for GPTZero (0.94-0.95), résumés best (0.15-0.28). Originality 0.02-0.21; Pangram ≤0.04. The Pangram 4 report re-ran it: GPTZero TPR@1%FPR 98.64% → **44.32% humanized full-length, 0.00% humanized under 50 words**.

**DeepSeek detector study 2025** (2507.17944). GPTZero 94.1% raw → 82.3% QuillBot standard → **52% QuillBot "humanize" mode**.

**Van Vlasselaer et al., IJEI 2026.** 40 papers per category, ≥4,000 words, humanization by a GPT-4o prompt to raise perplexity and burstiness. **GPTZero: 0% correct on fully-AI papers, 2.5% on humanized, 0% on hybrid, 0% false positives.** Pangram 97.5/95/95. Turnitin 0/50/60. On long-form academic text GPTZero was close to blind in this study, which conflicts sharply with its own reported numbers.

**"Why AI Detection Fails for Academic Integrity" 2026** (2608.11256). 642 abstracts, Undetectable AI v11: **GPTZero FNR above 96%**. Quality cost quantified: type-token ratio 0.70 → 0.61, academic word density 0.157 → 0.121.

**"Base Models Look Human To AI Detectors" 2026** (2605.19516). **GPTZero human-probability: Llama3-8B base 96.7% vs instruct 30.3%; Qwen3-8B base 98.8% vs instruct 17.1%.** HIP (iterative paraphrasing with a base model) reached **~100% "human" on GPTZero by round 10 with a GPT-judged semantic score of 6-10/10**. Homoglyphs had minimal effect; StealthRL negligible; DIPPER modest.

**GEDE benchmark** (via Pangram 4 Table 21). GPTZero TPR@1%FPR: fully generated 93.2%, AI-improved human 54.7%, adversarially humanized 92.6%.

**GPTZero's own technical report, Feb 2026** (2602.13042). 1,000 texts through paraphrasing plus 9 bypass services. Recall: GPTZero 93.5%, Originality 57.3%, Pangram 49.7%. They train on DIPPER and TempParaphraser augmentation plus gradient-based word substitution. No homoglyph or Unicode section. Directly contradicts the Pangram, UChicago and VUB data above.

Papers that did **not** test GPTZero at all: Sadasivan 2023, SilverSpeak, ToBlend, Adversarial Paraphrasing (2506.07001), TH-Bench, HMGC, Shi et al., RAFT, Beemo.

---

## 3. Industry and vendor tests (lower rigor, conflicts of interest flagged)

- **Pangram, Aug 2025**: 19 humanizers, Pangram 90.3-100% per tool, cites GPTZero at 46%.
- **HumanizerAI blog, Apr 2026** (vendor ranks itself first): GPTZero bypass, pass = under 50% AI. HumanizerAI 83%, Undetectable 76%, WriteHuman 72%, StealthGPT 69%, Humbot 66%, HIX Bypass 61%, StealthWriter 59%, Phrasly 57%, QuillBot 51%, Grammarly 46%, BypassGPT 36%.
- **Hastewire, Nov 2025**: ~50 GPT-4 samples, success = over 90% human. Undetectable 95%, BypassGPT 92%, Humanize AI 88%, HIX 85%, WriteHuman 80%.
- **TechLifeTV** (single 102-word paragraph, weak): explicit finding that "the higher the writing quality, the higher the detection risk."
- **deceptioner.site, Jun 2025**: 100 QuillBot passages, 81% flagged. Pass rate by mode: Humanize 46%, Shorten 36%, Creative 30%, Simple 27%, Fluency 18%, Standard 17%, **Academic/Expand/Formal 0%**. Academic register is the hardest mode to hide.
- **compareaitools, May 2026**: GPTZero correct on 90% raw AI, ~65% QuillBot, ~48% Undetectable.AI; 37.5% false positives on non-native writers.
- **HumanizerBench, Sep 2026** (median across 5 detectors): WriteHuman 90.1% bypass / meaning 72.3; Undetectable 82.1 / 82.1; StealthGPT 79.6 / **meaning 59.4**; Grammarly 41.1 / meaning 86.4. The bypass/meaning inverse relation is visible in a single table.
- **GPTZero's own humanizer reviews** (single sample each, all "failed to bypass"): Uncheck AI 96% AI, WriteHuman 84% AI-paraphrasing, StealthWriter 100% AI, Grubby 100% AI, HyperWrite 100% AI.
- **Undetectable.ai's own GPTZero test, Jun 2026** concedes GPTZero flagged its humanized Claude and ChatGPT samples at 92% and 100% AI.

---

## 4. What this means for our product

1. **Character-level tricks are dead against GPTZero.** Homoglyph, zero-width, whitespace, misspelling and number swap all cost ≤1.4 points. The 2023 single-space exploit is patched. Do not build any of it.

2. **Off-the-shelf paraphrasers give partial, inconsistent evasion.** DIPPER −2.5 in RAID; QuillBot 17-46% pass by mode and **0% in Academic/Formal modes**, which is exactly our target register. Plain paraphrase is not a viable core.

3. **Commercial humanizers land at roughly 40-56% human on rigorous multi-sample tests** (UChicago 44.3%, DAMAGE 60% detected, Russell 46.7%), with outliers to 96%+ human on short abstracts and long papers. That is the bar to beat, and it is far from the near-100% goal.

4. **The methods that reach high human rates while preserving quality are all detector-in-the-loop or model-level, not text tricks:**
   - **HIP with base models: ~100% human on GPTZero by round 10, semantic 6-10/10.** The strongest single result for our stated goal.
   - **Base model generation alone: 96.7-98.8% human** versus 17-30% for the instruct sibling. This is nearly free and is the largest single lever in the entire evidence base.
   - AuthorMist RL: 92.3% attack success at 0.945 similarity.
   - SICO: AUC 0.184-0.290 with *better* readability than DIPPER.
   - HUMPA: AUROC 0.995 → 0.80 at ~1 point of SBERT cost.
   - Prompt-level: "elevate literary language" → 12.9%, expert-guidebook prompt → 46.7%. Cheap, but stale and version-fragile.

5. **Two findings directly threaten the "college-level academic writing" differentiator.** QuillBot's Academic and Formal modes had a 0% pass rate. TechLifeTV observed that higher writing quality correlates with higher detection risk. Formal academic register is intrinsically the *easiest* register for a detector, because it is the register LLMs imitate best. Our product is attempting the hardest cell in the matrix, and the plan must treat that as the central engineering problem rather than an afterthought.

6. **Measure the verdict, not the probability.** DUPE is the cautionary tale: attacks that cut false-negative rates to 22-51% still yielded only 1 of 400 documents actually labeled "Human". Our success metric must be the returned label at the user-visible threshold, not a probability delta.

---

## Sources

RAID https://arxiv.org/abs/2405.07940 · DIPPER https://arxiv.org/abs/2303.13408 · SpaceInfi https://arxiv.org/abs/2307.02599 · Liang et al. https://arxiv.org/abs/2304.02819 and https://github.com/Weixin-Liang/ChatGPT-Detector-Bias · Ibrahim et al. PMC10449897 · Weber-Wulff https://arxiv.org/abs/2306.15666 · SICO https://arxiv.org/abs/2305.10847 · DUPE https://arxiv.org/abs/2404.11408 · Perkins et al. https://arxiv.org/abs/2403.19148 · ESPERANTO https://arxiv.org/abs/2409.14285 · HUMPA https://arxiv.org/abs/2410.19230 · DAMAGE https://arxiv.org/abs/2501.03437 · AuthorMist https://arxiv.org/abs/2503.08716 · Russell et al. https://arxiv.org/abs/2501.15654 · Jabarian & Imas https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf · DeepSeek detectors https://arxiv.org/abs/2507.17944 · Van Vlasselaer et al. https://doi.org/10.1007/s40979-026-00226-w · Why AI Detection Fails https://arxiv.org/abs/2608.11256 · Base Models Look Human https://arxiv.org/abs/2605.19516 · Pangram 4 report https://arxiv.org/abs/2607.27183 · GPTZero tech report https://arxiv.org/abs/2602.13042 · GPTZero humanized-text post https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/ · GPTZero paraphraser detection https://gptzero.me/news/ai-paraphrasing-detection/ · Pangram humanizers https://www.pangram.com/blog/humanizers-aug-25 · HumanizerBench https://humanizerbench.com/ · HumanizerAI test https://humanizerai.com/blog/best-ai-humanizer-2026 · Hastewire https://hastewire.com/blog/accuracy-test-top-ai-humanizers-vs-gptzero · TechLifeTV https://techlifetv.com/best-ai-humanizer/ · deceptioner QuillBot test https://deceptioner.site/blog/can-gptzero-detect-quillbot · compareaitools https://compareaitools.org/is-gptzero-accurate/ · Undetectable.ai GPTZero test https://undetectable.ai/blog/gptzero-accuracy-rate/ · TH-Bench https://arxiv.org/abs/2503.08708
