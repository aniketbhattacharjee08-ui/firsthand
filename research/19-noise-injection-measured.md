# Noise Injection, Measured — Does "Too Clean" Explain Our Detector Scores?

*2026-09-07. Empirical report. Every number here was measured on this machine (Apple M4 Pro, CPU
inference) against `desklib/ai-text-detector-v1.01` through `humanizer.detectors.local.ModernDetector`,
and re-scored on the four other published checkpoints in `PUBLISHED_DETECTORS` as held-out detectors.
Scripts, raw scores and every generated text are in `.noise/`. Total compute: 44 minutes of detector
and model time inside a 90-minute budget; wall clock including contention about 75 minutes.*

**The hypothesis under test.** The project owner believes the humanizer output is "too clean" and
that injecting noise would move detectors more than polishing does. This report tests that in both
directions: human text made cleaner (Experiment A) and AI text made noisier by type and dose
(Experiment B), plus sampling noise (C), spliced human/AI text (D) and a readability baseline (E).

---

## 0. The three findings, stated up front

**Finding 1 — "clean reads as AI" is true, and it is mostly about the LLM, not about grammar.**
Passing genuine human PMC paragraphs through Qwen2.5-7B-Instruct with the instruction *"lightly
polish for grammar and clarity, do not change content, structure or length"* changed 11% of the words
and moved desklib from 0.200 to 0.399, flipping **11 of 33** previously-human paragraphs to AI; on the
mixed-genre MAGE human texts, 0.244 → 0.489 and **12 of 33** flips. The flips transfer: fakespot
flipped 13 and 15, radar and academic 1-5. A 3B polish is worse (16 of 33 on MAGE at 25% words
changed). By contrast, a pure grammar corrector (T5) at 2% words changed flipped 2 of 33 and pure
typographic normalisation (curly quotes → straight, en dash → hyphen, 1% of characters) flipped 2 of 33
with a mean rise of +0.05. So there is a cleaning budget: **about 2% of words is safe; 11% through an
instruct model costs a third of the human verdicts.** (§3)

**Finding 2 — noise moves boundary text, not saturated text, and in-distribution noise moves
nothing.** On the boundary corpus (Qwen restatements, desklib 0.731, 30/40 flagged), every visible
noise type flipped verdicts at 2-4 events per 100 words: function-word deletion 13/30, WordNet
synonyms 12/30, typos 10/30, casing 9/30, punctuation slips 7/30, hedges 6/30; most flips transferred
to fakespot and academic (4-11 each) but almost never to radar (0-2). On the saturated corpus
(Llama-3-8B paraphrases, 0.997, 40/40 flagged) **the same operators at the same doses produced 0-2
flips of 40 on desklib and 0-1 of 40 on fakespot**; only the RAID-scale casing attack at 20 swaps per
100 words, which is unreadable, reached 10 of 40 — see §4.2. The one operator that is *not* visible damage,
HMGC-style masked-LM substitution with `roberta-base` (the plausible neighbour, not the odd synonym),
produced **1/30 flips at every dose from 0.5 to 4 per 100 words and moved the mean by 0.007**. Noise
works exactly to the degree that it puts tokens the detector has never seen in AI text; the moment
the substitute is plausible, the effect vanishes. Sentence shuffling (0/30, even fully shuffled),
connective removal (2/30) and sentence-length reshaping to the 0.42-0.60 band (3/30) did not move
verdicts either. (§4)

**Finding 3 — the asymmetry in mixed text is the real constraint on our architecture.** Replacing
one of roughly six sentences in a *human* paragraph with the matched AI sentence (17% of words) flipped
**10 of 33** human paragraphs to AI on desklib, 6 on fakespot, 6 on radar. Replacing *half* of the
sentences in an AI paragraph with the matched human sentences (47% of words) flipped **5 of 30** on the
boundary corpus and inserting three human sentences into saturated Llama paragraphs flipped **1 of
40**. AI signal dominates human signal at every mixture ratio. Any pipeline that edits, replaces or
regenerates sentences inside a paragraph is on the losing side of this asymmetry; the whole paragraph
has to be human-shaped. (§6)

A fourth result that is not a finding because it does not survive the quality gate: the *base*
Qwen2.5-3B checkpoint continuing a real opening sentence at temperature 1.0 scored **0.199 on desklib
(3 of 24 flagged), 0.052 on fakespot, 0.001 on academic** — by far the largest "human" score of any
condition — but its content-word overlap with the source paper was 0.11 against 0.28 for the instruct
model, and the text is fluent invented science ("125 patients showed no extensor tension at the
proximal forearm"). At temperature 0.7 the same model scores 0.892 and at 1.2 it produces gibberish
that scores 0.952. research/17 §B3 and the pipeline docstring already recorded this shape; it is
reproduced, not overturned. (§5)

---

## 1. Method

**Detectors.** Every number below comes from `ModernDetector(score_sentences=False)` on CPU (8 torch
threads). The primary detector is the project default, `desklib/ai-text-detector-v1.01`
(DeBERTa-v3-large, RAID-trained). Every condition is *also* scored on the four other
`PUBLISHED_DETECTORS` checkpoints, none of which was used to choose any edit:
`fakespot-ai/roberta-base-ai-text-detection-v1`, `andreas122001/roberta-academic-detector`,
`TrustSafeAI/RADAR-Vicuna-7B` (the RoBERTa-large detector, not the paraphraser) and
`MayZhou/e5-small-lora-ai-generated-detector`. In the tables they appear as *fakespot / academic /
radar / fast*, each as mean P(AI) after the edit with the verdict-flip count in parentheses. Verdict
threshold 0.5 everywhere. Nothing here is GPTZero; see the `ModernDetector` docstring for why a local
number measures *change*, not a vendor verdict. `fast` is reported for completeness but flagged 24 of
40 genuine human paragraphs at baseline, so its flip counts are not meaningful.

**Corpora** (`.noise/common.py::load_corpora`, `random.Random(19)`, 80-260 words, at least three
sentence boundaries):

| name | n | what it is |
|---|---:|---|
| `human_pmc` | 40 | genuine PMC open-access paragraphs, human side of `.hiplora/pairs.jsonl` |
| `human_mage` | 40 | genuine human MAGE texts shipped by the Adversarial-Paraphrasing repo (mixed genres) |
| `ai_qwen` | 40 | Qwen2.5-3B-Instruct restatements of exactly the `human_pmc` paragraphs (content-matched pairs) |
| `ai_llama` | 40 | Llama-3-8B-Instruct plain paraphrases of MAGE texts (the `noadv` set research/17 §B1 scored at 0.997) |
| `ai_misc` | 17 | the 9 `.llmbench` paragraphs research/17 used plus the 8 `.codemine/demo_pairs.json` restatements |

Baseline scores:

| corpus | n | desklib (flagged) | fakespot | academic | radar | fast |
|---|---:|---:|---:|---:|---:|---:|
| human_pmc | 40 | 0.200 (7) | 0.216 (6) | 0.078 (3) | 0.130 (2) | 0.522 (24) |
| human_mage | 40 | 0.244 (7) | 0.348 (12) | 0.106 (4) | 0.290 (11) | 0.563 (24) |
| ai_qwen | 40 | 0.731 (30) | 0.806 (33) | 0.311 (13) | 0.234 (8) | 0.694 (32) |
| ai_llama | 40 | 0.997 (40) | 0.999 (40) | 0.794 (32) | 0.780 (31) | 0.913 (40) |
| ai_misc | 17 | 0.816 (14) | 0.924 (16) | 0.697 (12) | 0.505 (9) | 0.787 (14) |

Two AI corpora were used deliberately. `ai_qwen` sits near the boundary because a 3B restatement of
a real paper keeps most of the paper's vocabulary and numbers; `ai_llama` is saturated. A noise
operator that only moves the boundary corpus is not doing much, and §4.2 shows that is what happens.

**Noise dosing.** Dose is events per 100 words; the number of events for a paragraph is
`dose × words / 100` with stochastic rounding and a floor of one, seeded per
`(operator, dose, paragraph index)` so every run is reproducible. Realised events/100w is reported in
each table. Readability is `textstat` Flesch Reading Ease (FRE) and Flesch-Kincaid grade;
sentence-length CV uses one regex sentence splitter throughout (`common.split_sentences`, with
abbreviation gluing).

**Tools available and not.** No Java runtime is installed, so LanguageTool was not run.
`vennify/t5-base-grammar-correction` (850 MB) was downloaded and run on MPS. LLM polishing and
generation used MLX checkpoints already in the HF cache: `mlx-community/Qwen2.5-7B-Instruct-4bit`,
`mlx-community/Qwen2.5-3B-Instruct-4bit` and the base `mlx-community/Qwen2.5-3B-4bit`. WordNet was
installed via `nltk` (10 MB). `roberta-base` (cached) provides the MLM-neighbour substitution, which is
HMGC's `WordSwapMaskedLM` operator from research/17 §A.2 without the detector in the loop.

---

## 2. Experiment E — the readability baseline

Computed on the full corpora, not the 40-paragraph samples. Quantiles, as asked.

| corpus | n | FRE p10/p25/p50/p75/p90 | FK grade p10/p50/p90 | mean sentence length p10/p50/p90 | sentence-length CV p10/p25/p50/p75/p90 |
|---|---:|---|---|---|---|
| human PMC | 1264 | 10.4/21.4/31.5/42.3/52.5 | 9.7/13.8/18.0 | 17.0/22.8/31.3 | 0.23/0.30/**0.38**/0.49/0.61 |
| human Wikipedia FA | 236 | 32.5/39.0/47.0/56.7/63.5 | 8.6/12.2/15.3 | 16.7/22.4/30.3 | 0.28/0.34/0.42/0.50/0.66 |
| Qwen-3B restatement of PMC | 1264 | 5.6/15.5/25.1/36.0/46.1 | 10.6/14.5/17.9 | 16.7/21.7/30.9 | 0.22/0.29/0.37/0.47/0.61 |
| Qwen-3B restatement of Wikipedia | 236 | 29.0/35.1/40.8/51.0/57.7 | 9.4/12.2/14.6 | 16.1/20.3/26.7 | 0.25/0.30/0.40/0.50/0.64 |
| human MAGE (mixed genre) | 2000 | 16.8/30.4/51.0/68.5/79.3 | 5.5/11.3/17.6 | 12.1/20.4/29.8 | 0.25/0.33/0.44/0.57/0.72 |
| Llama-3-8B paraphrase of MAGE | 1998 | 11.9/23.4/37.1/49.2/58.8 | 9.4/13.7/18.1 | 16.4/22.5/29.0 | 0.16/0.22/**0.29**/0.38/0.48 |
| `ai_misc` (free generation, 3B) | 17 | 2.9/7.1/13.7/17.7/20.6 | 15.1/17.4/18.1 | 17.1/21.3/25.8 | 0.16/0.21/0.29/0.36/0.41 |

Three things to take from this.

* **There is a readability gap and it runs the "wrong" way for the too-clean story.** AI text is
  *harder* to read than the human text it paraphrases: Qwen's PMC restatements sit 6 FRE points below
  the originals at every quantile, and Llama's MAGE paraphrases sit 14 points below. Free generation
  (`ai_misc`) is denser still (median FRE 13.7, grade 17.4). The LLM fingerprint in this data is
  Latinate density, not simplicity.
* **Sentence-length CV separates free generation and paraphrase from human text, but not
  restatement.** Llama paraphrases have a median CV of 0.29 against human MAGE 0.44 — a real gap and
  the reason the 0.42-0.60 band came to be a target. But Qwen restatements of PMC match their human
  sources almost exactly (0.37 vs 0.38), because a faithful restatement inherits the source's sentence
  plan. CV is a feature of *generation*, not of AI-ness.
* **This contradicts research/00 §5 on the human band.** research/00 gives 0.42-0.60 as the academic
  target and calls 0.30 "as anomalous as 0.85". Measured here on 1,264 human PMC paragraphs, the
  median is **0.38**, the interquartile range 0.30-0.49, and a quarter of genuine published
  paragraphs sit *below* 0.30. The band in research/00 was measured on documents; at paragraph scale
  (the unit our pipeline edits and the detector scores), the human distribution is lower and wider.
  `pipeline.HUMAN_SENTENCE_CV = (0.42, 0.60)` is therefore a paragraph-level target that most human
  paragraphs would fail.

---

## 3. Experiment A — human text made cleaner

Scripts: `.noise/run_A.py`. Corpora `human_pmc` and `human_mage`, 40 each. "Words changed" is a
`difflib` word-level change rate against the source. "H→AI flips" counts paragraphs that were under
0.5 before and at or above 0.5 after, out of the 33 that were human at baseline.

| condition | corpus | words changed | desklib before → after | H→AI flips | held-out after: fakespot / academic / radar / fast (flips) | FRE before → after | gen s |
|---|---|---:|---|---:|---|---|---:|
| (i) typographic normalisation | human_pmc | 1% | 0.200 → 0.249 | 2/33 | 0.26 (1) / 0.08 (0) / 0.13 (0) / 0.53 (1) | 29.3 → 29.4 | 0 |
| (i) typographic normalisation | human_mage | 0% | 0.244 → 0.261 | 1/33 | 0.35 (1) / 0.11 (0) / 0.29 (0) / 0.56 (0) | 48.7 → 48.7 | 0 |
| (ii) grammar, T5 | human_pmc | 2% | 0.200 → 0.229 | 2/33 | 0.26 (3) / 0.14 (3) / 0.13 (0) / 0.53 (0) | 29.3 → 29.1 | 85 |
| (ii) grammar, T5 | human_mage | 1% | 0.244 → 0.246 | 0/33 | 0.38 (2) / 0.15 (2) / 0.30 (0) / 0.58 (0) | 48.7 → 48.7 | 62 |
| (iii) light polish, Qwen2.5-7B-Instruct | human_pmc | 11% | 0.200 → **0.399** | **11/33** | 0.47 (13) / 0.20 (5) / 0.15 (2) / 0.59 (5) | 29.3 → 28.7 | 85 |
| (iii) light polish, Qwen2.5-7B-Instruct | human_mage | 12% | 0.244 → **0.489** | **12/33** | 0.64 (15) / 0.20 (4) / 0.29 (1) / 0.66 (7) | 48.7 → 47.9 | 90 |
| (iv) light polish, Qwen2.5-3B-Instruct | human_pmc | 18% | 0.200 → 0.429 | 10/33 | 0.56 (17) / 0.18 (4) / 0.19 (4) / 0.63 (7) | 29.3 → 27.9 | 73 |
| (iv) light polish, Qwen2.5-3B-Instruct | human_mage | 25% | 0.244 → 0.535 | 16/33 | 0.77 (19) / 0.29 (8) / 0.34 (3) / 0.72 (10) | 48.7 → 45.2 | 32 |

**Does p(AI) rise when human text is cleaned? Yes, at every level, and roughly in proportion to how
much an instruct model touched.** Plotting the eight rows, the desklib rise is about +0.02 per 1% of
words changed: 1% → +0.05, 2% → +0.03, 11% → +0.20, 18% → +0.23, 25% → +0.29. The polish prompt was
as conservative as a prompt can be ("keep every sentence in place; fix only what is wrong or
awkward"), and the 7B model obeyed it — length ratio 1.00, FRE moved 0.6 points — and still a third
of the human verdicts were lost.

What the polish actually did is instructive. The paragraph that moved most (0.157 → 0.843) had
eleven word-level edits: a comma removed, "(~30–100 nm of" reordered to "(with a ... of ~30–100 nm",
"its" → "the", "that" → "those", "processes for phosphate cathode materials" → "process", "also"
deleted. Nothing a copy editor would object to; every edit removes a small irregularity. The second
(0.100 → 0.779) replaced "Here, ... to get better visual performance. With" with "To optimize visual
performance, ... As", regularised "signal to noise" to "signal-to-noise ratio", and inserted five
definite articles. This is what "clean" means to the detector: fewer dangling deictics, more
articles, tidier noun phrases, a topic-sentence opener.

**The typographic result deserves its own line.** Condition (i) changed nothing but characters:
curly to straight quotes, en dash to hyphen in "1–3", removing the stray space before a period left
by the PMC figure-reference stripper ("lesions ." → "lesions."). Twenty-one of forty PMC paragraphs
had any such character, and on those twenty-one the mean rise was **+0.092**; one went 0.237 → 0.674
on the strength of three curly apostrophes. Desklib has learned that typographic punctuation and
extraction debris are human. Two consequences follow: part of the "human" score of this corpus is
preprocessing residue rather than authorship (a caveat on research/17's PMC baseline too), and the
reverse edit is a free noise operator, tested in §4 as `typography` and `debris`.

**A caveat on the grammar row.** `vennify/t5-base-grammar-correction` made real corrections
("progresses ," → "progresses,", "cycle" → "cycle,") but its sentencepiece vocabulary also silently
destroyed symbols: "4±1.0" → "41.0", "μM" → "M", "g−1" → "g1". So the 2% change rate includes
corruption, and the row is best read as "a small, mixed edit at 2% moves desklib by +0.03 and flips
2 of 33". LanguageTool would have been cleaner; there is no Java on the machine.

---

## 4. Experiment B — AI text made noisier, by type and dose

Scripts: `.noise/noise_ops.py`, `.noise/run_B.py`, `.noise/run_B2.py`. Each row is 40 paragraphs;
"flips" are paragraphs at or above 0.5 before and below 0.5 after, out of those flagged at baseline
(30 of 40 for `ai_qwen`, 40 of 40 for `ai_llama`). Held-out columns show mean P(AI) after, with flips
in parentheses (fakespot 33 / academic 13 / radar 8 / fast 32 eligible on `ai_qwen`; 40 / 32 / 31 / 40
on `ai_llama`).

### 4.1 The boundary corpus (`ai_qwen`, desklib 0.731 before, 30/40 flagged)

| noise | dose | events/100w | desklib before → after | flips | fakespot / academic / radar / fast (flips) | FRE | CV |
|---|---:|---:|---|---:|---|---|---|
| (1) typos | 0.5 | 0.65 | 0.731 → 0.639 | 4/30 | 0.77 (1) / 0.29 (1) / 0.21 (1) / 0.65 (1) | 23.6 → 23.7 | 0.39 |
| (1) typos | 1 | 0.96 | 0.731 → 0.631 | 6/30 | 0.77 (2) / 0.26 (3) / 0.23 (1) / 0.65 (1) | 23.6 → 23.8 | 0.39 |
| (1) typos | 2 | 2.01 | 0.731 → 0.580 | 8/30 | 0.68 (5) / 0.23 (4) / 0.22 (1) / 0.60 (8) | 23.6 → 24.2 | 0.39 |
| (1) typos | 4 | 3.99 | 0.731 → 0.537 | **10/30** | 0.65 (8) / 0.16 (7) / 0.21 (2) / 0.52 (12) | 23.6 → 24.8 | 0.39 |
| (2) punctuation slips | 0.5 | 0.65 | 0.731 → 0.705 | 2/30 | 0.79 (0) / 0.32 (1) / 0.24 (0) / 0.69 (0) | 23.6 → 23.7 | 0.39 |
| (2) punctuation slips | 1 | 1.02 | 0.731 → 0.706 | 1/30 | 0.77 (2) / 0.25 (3) / 0.23 (0) / 0.68 (0) | 23.6 → 23.8 | 0.39 |
| (2) punctuation slips | 2 | 2.05 | 0.731 → 0.617 | 6/30 | 0.70 (5) / 0.20 (5) / 0.23 (1) / 0.66 (2) | 23.6 → 23.9 | 0.39 |
| (2) punctuation slips | 4 | 3.97 | 0.731 → 0.585 | 7/30 | 0.63 (10) / 0.18 (6) / 0.23 (0) / 0.65 (2) | 23.6 → 24.3 | 0.39 |
| (3) function words | 0.5 | 0.63 | 0.731 → 0.694 | 2/30 | 0.75 (2) / 0.27 (2) / 0.24 (0) / 0.68 (0) | 23.6 → 23.3 | 0.39 |
| (3) function words | 1 | 1.05 | 0.731 → 0.639 | 4/30 | 0.75 (2) / 0.24 (4) / 0.25 (0) / 0.68 (0) | 23.6 → 23.3 | 0.39 |
| (3) function words | 2 | 2.05 | 0.731 → 0.534 | 9/30 | 0.70 (6) / 0.21 (5) / 0.27 (0) / 0.65 (2) | 23.6 → 23.1 | 0.39 |
| (3) function words | 4 | 4.03 | 0.731 → **0.396** | **13/30** | 0.59 (11) / 0.23 (4) / 0.30 (0) / 0.62 (3) | 23.6 → 22.7 | 0.41 |
| (4a) WordNet synonym | 0.5 | 0.65 | 0.731 → 0.657 | 5/30 | 0.79 (0) / 0.31 (1) / 0.25 (0) / 0.66 (0) | 23.6 → 23.6 | 0.39 |
| (4a) WordNet synonym | 1 | 1.07 | 0.731 → 0.662 | 3/30 | 0.77 (1) / 0.29 (1) / 0.22 (1) / 0.66 (2) | 23.6 → 24.0 | 0.39 |
| (4a) WordNet synonym | 2 | 2.04 | 0.731 → 0.498 | 11/30 | 0.71 (4) / 0.27 (3) / 0.22 (1) / 0.62 (4) | 23.6 → 24.4 | 0.39 |
| (4a) WordNet synonym | 4 | 4.06 | 0.731 → 0.406 | **12/30** | 0.62 (11) / 0.20 (5) / 0.21 (1) / 0.56 (7) | 23.6 → 24.9 | 0.39 |
| (4b) MLM neighbour (roberta-base) | 0.5 | 0.62 | 0.731 → 0.732 | 1/30 | 0.81 (1) / 0.31 (1) / 0.24 (0) / 0.69 (0) | 23.6 → 23.9 | 0.39 |
| (4b) MLM neighbour | 1 | 0.97 | 0.731 → 0.731 | 1/30 | 0.81 (1) / 0.30 (1) / 0.23 (0) / 0.69 (0) | 23.6 → 23.9 | 0.39 |
| (4b) MLM neighbour | 2 | 1.98 | 0.731 → 0.734 | 1/30 | 0.79 (1) / 0.32 (1) / 0.24 (0) / 0.69 (0) | 23.6 → 23.7 | 0.39 |
| (4b) MLM neighbour | 4 | 4.03 | 0.731 → **0.724** | **1/30** | 0.80 (2) / 0.29 (2) / 0.23 (2) / 0.69 (0) | 23.6 → 24.6 | 0.39 |
| (5) reshape CV → 0.25 | target | 0.85 | 0.731 → 0.689 | 3/30 | 0.78 (0) / 0.30 (2) / 0.26 (2) / 0.68 (0) | 23.6 → 21.1 | 0.39 → 0.27 |
| (5) reshape CV → 0.50 (band) | target | 0.68 | 0.731 → 0.698 | 3/30 | 0.75 (2) / 0.27 (2) / 0.27 (0) / 0.68 (1) | 23.6 → 21.1 | 0.39 → 0.47 |
| (5) reshape CV → 0.75 | target | 1.13 | 0.731 → 0.700 | 2/30 | 0.67 (8) / 0.20 (5) / 0.30 (0) / 0.66 (2) | 23.6 → 17.7 | 0.39 → 0.74 |
| (5) reshape CV → 0.90 | target | 1.47 | 0.731 → 0.696 | 2/30 | 0.63 (9) / 0.12 (9) / 0.34 (0) / 0.66 (3) | 23.6 → 15.4 | 0.39 → 0.89 |
| (6) shuffle, 1 adjacent swap | 1 | 0.62 | 0.731 → 0.718 | 2/30 | 0.80 (1) / 0.33 (0) / 0.26 (0) / 0.68 (2) | 23.6 | 0.39 |
| (6) shuffle, 2 swaps | 2 | 1.24 | 0.731 → 0.723 | 2/30 | 0.80 (1) / 0.30 (2) / 0.26 (1) / 0.68 (1) | 23.6 | 0.38 |
| (6) shuffle, 4 swaps | 4 | 2.47 | 0.731 → 0.747 | 1/30 | 0.80 (0) / 0.35 (1) / 0.28 (0) / 0.69 (1) | 23.6 | 0.38 |
| (6) shuffle, full | all | 4.57 | 0.731 → 0.733 | **0/30** | 0.79 (2) / 0.29 (2) / 0.34 (0) / 0.70 (2) | 23.6 | 0.38 |
| (7a) remove AI connectives | all | 0.37 | 0.731 → 0.718 | 2/30 | 0.79 (1) / 0.28 (2) / 0.24 (0) / 0.68 (0) | 23.6 → 24.1 | 0.39 |
| (7b) insert But/So/Still/Yet | 0.5 | 0.65 | 0.731 → 0.701 | 2/30 | 0.69 (5) / 0.21 (5) / 0.25 (0) / 0.67 (2) | 23.6 → 24.0 | 0.38 |
| (7b) insert But/So/Still/Yet | 1 | 1.00 | 0.731 → 0.646 | 4/30 | 0.62 (10) / 0.20 (5) / 0.25 (0) / 0.66 (1) | 23.6 → 24.3 | 0.38 |
| (7b) insert But/So/Still/Yet | 2 | 1.98 | 0.731 → 0.643 | 4/30 | 0.53 (14) / 0.18 (6) / 0.26 (0) / 0.65 (4) | 23.6 → 25.0 | 0.38 |
| (7b) insert But/So/Still/Yet | 4 | 3.54 | 0.731 → 0.550 | 9/30 | 0.44 (17) / 0.13 (8) / 0.30 (0) / 0.62 (3) | 23.6 → 26.1 | 0.37 |
| (8) redundancy (pronoun → NP) | 0.5-4 | 0.42-0.85 | 0.731 → 0.72-0.73 | 2-3/30 | 0.76 (1) / 0.25 (4) / 0.22 (2) / 0.67 (1) | 23.6 → 22.6 | 0.39 |
| (9) hedges | 0.5 | 0.65 | 0.731 → 0.693 | 2/30 | 0.70 (4) / 0.24 (4) / 0.21 (1) / 0.67 (0) | 23.6 → 24.1 | 0.38 |
| (9) hedges | 1 | 0.94 | 0.731 → 0.685 | 2/30 | 0.71 (3) / 0.23 (4) / 0.22 (3) / 0.67 (2) | 23.6 → 24.2 | 0.38 |
| (9) hedges | 2 | 1.92 | 0.731 → 0.648 | 6/30 | 0.61 (11) / 0.15 (7) / 0.20 (1) / 0.65 (4) | 23.6 → 24.9 | 0.37 |
| (9) hedges | 4 | 3.80 | 0.731 → 0.564 | 6/30 | 0.45 (15) / 0.13 (8) / 0.20 (1) / 0.62 (7) | 23.6 → 25.0 | 0.35 |
| (10) casing swap | 0.5 | 0.63 | 0.731 → 0.693 | 2/30 | 0.77 (2) / 0.23 (4) / 0.25 (0) / 0.69 (0) | 23.6 | 0.39 |
| (10) casing swap | 1 | 1.02 | 0.731 → 0.630 | 6/30 | 0.77 (1) / 0.23 (4) / 0.23 (1) / 0.69 (0) | 23.6 | 0.39 |
| (10) casing swap | 2 | 2.05 | 0.731 → 0.567 | 9/30 | 0.70 (4) / 0.19 (6) / 0.26 (2) / 0.69 (0) | 23.6 | 0.40 |
| (10) casing swap | 4 | 4.03 | 0.731 → 0.538 | 9/30 | 0.66 (7) / 0.18 (6) / 0.28 (0) / 0.69 (0) | 23.6 | 0.39 |
| (10) casing swap, RAID ceiling | 20 | 19.96 | 0.731 → **0.358** | **18/30** | 0.38 (17) / 0.00 (13) / 0.12 (7) / 0.69 (0) | 23.6 | 0.41 |
| (11) typography (ASCII → curly/en dash) | all | 0.80 | 0.731 → 0.650 | 4/30 | 0.75 (4) / 0.33 (1) / 0.23 (0) / 0.68 (0) | 23.6 → 23.2 | 0.38 |
| (12) PMC extraction debris | 1 | 0.79 | 0.731 → 0.626 | 4/30 | 0.72 (3) / 0.23 (4) / 0.27 (0) / 0.69 (0) | 23.6 | 0.38 |
| (12) PMC extraction debris | 3 | 2.69 | 0.731 → 0.548 | 10/30 | 0.76 (1) / 0.15 (7) / 0.23 (1) / 0.69 (0) | 23.6 | 0.38 |
| combo: typography + punct 1 + hedges 0.5 | — | 2.44 | 0.731 → 0.572 | 7/30 | 0.61 (9) / 0.24 (5) / 0.24 (1) / 0.64 (2) | 23.6 → 24.0 | 0.38 |
| combo: typography + typos 1 | — | 1.78 | 0.731 → 0.566 | 9/30 | 0.71 (4) / 0.24 (3) / 0.21 (3) / 0.63 (1) | 23.6 → 23.5 | 0.38 |

Readability cost is small everywhere (FRE moves under 1.5 points except for the CV reshaping, which
lowers FRE 2.5 to 8 points because merging sentences lengthens them). The visible cost is not in FRE:
it is in what the text says. Sample edits at dose 2 on one paragraph: typos "should → shoild,
highlights → highligths"; WordNet "delayed → detain, necessitate → demand"; function words insert
a stray "a", "is", "the"; hedges "Patients → In practice, patients". These are the DAMAGE L2/L3 tiers
from research/03 and they are what buys the flips.

**Where the flips come from.** Splitting the 30 flagged `ai_qwen` paragraphs into a saturated bucket
(22 at P ≥ 0.9) and a boundary bucket (8 at 0.5-0.9):

| operator at dose 4 (or as stated) | mean Δ, saturated | flips, saturated | mean Δ, boundary | flips, boundary |
|---|---:|---:|---:|---:|
| function words | −0.312 | 5/22 | −0.666 | 8/8 |
| WordNet synonym | −0.303 | 4/22 | −0.664 | 8/8 |
| typos | −0.150 | 3/22 | −0.436 | 7/8 |
| casing | −0.140 | 2/22 | −0.452 | 7/8 |
| hedges | −0.127 | 1/22 | −0.377 | 5/8 |
| punctuation slips | −0.081 | 1/22 | −0.389 | 6/8 |
| MLM neighbour | −0.009 | 0/22 | −0.024 | 1/8 |
| remove connectives | −0.013 | 0/22 | −0.022 | 2/8 |

Every operator empties the boundary bucket and leaves the saturated bucket mostly alone. That is
the pattern of a classifier being nudged across a threshold it was already near, not of text being
made human.

### 4.2 The saturated corpus (`ai_llama`, desklib 0.997 before, 40/40 flagged)

The same operators on the Llama-3-8B paraphrases. Held-out eligible counts: fakespot 40, academic
32, radar 31, fast 40.

| noise | dose | events/100w | desklib before → after | flips | fakespot / academic / radar / fast (flips) | FRE | CV |
|---|---:|---:|---|---:|---|---|---|
| typography | 0 | 1.86 | 0.997 -> 0.988 | 0/40 | 1.00 (0) / 0.78 (1) / 0.77 (1) / 0.90 (0) | 35.0 -> 35.0 | 0.29 -> 0.30 |
| debris | 1 | 0.75 | 0.997 -> 0.995 | 0/40 | 1.00 (0) / 0.75 (2) / 0.82 (0) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.28 |
| debris | 3 | 2.62 | 0.997 -> 0.983 | 0/40 | 1.00 (0) / 0.61 (7) / 0.77 (1) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.28 |
| combo_low | 0 | 3.60 | 0.997 -> 0.974 | 0/40 | 0.99 (0) / 0.66 (6) / 0.76 (2) / 0.89 (0) | 35.0 -> 35.4 | 0.29 -> 0.29 |
| combo_typo | 0 | 2.89 | 0.997 -> 0.979 | 0/40 | 1.00 (0) / 0.73 (3) / 0.74 (1) / 0.88 (0) | 35.0 -> 35.2 | 0.29 -> 0.30 |
| typos | 0.5 | 0.72 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.77 (1) / 0.76 (0) / 0.90 (0) | 35.0 -> 35.1 | 0.29 -> 0.29 |
| typos | 1 | 1.02 | 0.997 -> 0.990 | 0/40 | 1.00 (0) / 0.78 (1) / 0.75 (0) / 0.90 (0) | 35.0 -> 35.4 | 0.29 -> 0.29 |
| typos | 2 | 1.95 | 0.997 -> 0.967 | 0/40 | 0.99 (0) / 0.69 (4) / 0.74 (2) / 0.88 (0) | 35.0 -> 35.5 | 0.29 -> 0.29 |
| typos | 4 | 4.00 | 0.997 -> 0.950 | 1/40 | 0.99 (0) / 0.61 (7) / 0.72 (2) / 0.82 (1) | 35.0 -> 36.3 | 0.29 -> 0.29 |
| punct | 0.5 | 0.72 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.78 (1) / 0.79 (0) / 0.91 (0) | 35.0 -> 35.2 | 0.29 -> 0.29 |
| punct | 1 | 0.99 | 0.997 -> 0.987 | 0/40 | 1.00 (0) / 0.70 (4) / 0.79 (1) / 0.91 (0) | 35.0 -> 35.2 | 0.29 -> 0.29 |
| punct | 2 | 1.97 | 0.997 -> 0.991 | 0/40 | 1.00 (0) / 0.69 (4) / 0.78 (0) / 0.91 (0) | 35.0 -> 35.3 | 0.29 -> 0.29 |
| punct | 4 | 4.00 | 0.997 -> 0.990 | 0/40 | 1.00 (0) / 0.60 (8) / 0.79 (0) / 0.90 (0) | 35.0 -> 35.7 | 0.29 -> 0.29 |
| funcword | 0.5 | 0.72 | 0.997 -> 0.992 | 0/40 | 1.00 (0) / 0.78 (0) / 0.79 (0) / 0.91 (0) | 35.0 -> 34.7 | 0.29 -> 0.30 |
| funcword | 1 | 0.99 | 0.997 -> 0.982 | 1/40 | 1.00 (0) / 0.78 (1) / 0.81 (0) / 0.91 (0) | 35.0 -> 34.8 | 0.29 -> 0.29 |
| funcword | 2 | 2.04 | 0.997 -> 0.955 | 1/40 | 1.00 (0) / 0.76 (2) / 0.81 (1) / 0.91 (0) | 35.0 -> 34.5 | 0.29 -> 0.29 |
| funcword | 4 | 3.96 | 0.997 -> 0.923 | 1/40 | 0.99 (0) / 0.70 (4) / 0.83 (0) / 0.89 (0) | 35.0 -> 34.2 | 0.29 -> 0.30 |
| syn_wordnet | 0.5 | 0.72 | 0.997 -> 0.989 | 0/40 | 1.00 (0) / 0.78 (1) / 0.77 (0) / 0.91 (0) | 35.0 -> 35.2 | 0.29 -> 0.29 |
| syn_wordnet | 1 | 0.97 | 0.997 -> 0.985 | 0/40 | 1.00 (0) / 0.74 (2) / 0.76 (1) / 0.90 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| syn_wordnet | 2 | 2.12 | 0.997 -> 0.967 | 1/40 | 0.99 (0) / 0.73 (3) / 0.75 (1) / 0.89 (1) | 35.0 -> 35.1 | 0.29 -> 0.29 |
| syn_wordnet | 4 | 3.94 | 0.997 -> 0.922 | 2/40 | 0.99 (0) / 0.71 (4) / 0.70 (3) / 0.87 (1) | 35.0 -> 35.7 | 0.29 -> 0.29 |
| hedges | 0.5 | 0.72 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.77 (1) / 0.76 (0) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.28 |
| hedges | 1 | 0.99 | 0.997 -> 0.996 | 0/40 | 0.99 (0) / 0.75 (2) / 0.74 (0) / 0.91 (0) | 35.0 -> 35.1 | 0.29 -> 0.29 |
| hedges | 2 | 2.03 | 0.997 -> 0.982 | 0/40 | 0.98 (1) / 0.67 (5) / 0.71 (4) / 0.90 (0) | 35.0 -> 35.1 | 0.29 -> 0.28 |
| hedges | 4 | 3.87 | 0.997 -> 0.972 | 1/40 | 0.97 (1) / 0.45 (14) / 0.72 (4) / 0.89 (0) | 35.0 -> 35.2 | 0.29 -> 0.26 |
| redundancy | 0.5 | 0.59 | 0.997 -> 0.997 | 0/40 | 1.00 (0) / 0.80 (0) / 0.77 (0) / 0.91 (0) | 35.0 -> 34.9 | 0.29 -> 0.29 |
| redundancy | 1 | 0.72 | 0.997 -> 0.997 | 0/40 | 1.00 (0) / 0.80 (0) / 0.76 (0) / 0.91 (0) | 35.0 -> 34.4 | 0.29 -> 0.29 |
| redundancy | 2 | 1.20 | 0.997 -> 0.997 | 0/40 | 1.00 (0) / 0.75 (2) / 0.76 (0) / 0.91 (0) | 35.0 -> 34.1 | 0.29 -> 0.29 |
| redundancy | 4 | 1.43 | 0.997 -> 0.994 | 0/40 | 1.00 (0) / 0.77 (1) / 0.75 (0) / 0.91 (0) | 35.0 -> 33.6 | 0.29 -> 0.29 |
| markers_human | 0.5 | 0.72 | 0.997 -> 0.993 | 0/40 | 1.00 (0) / 0.75 (2) / 0.79 (0) / 0.91 (0) | 35.0 -> 35.4 | 0.29 -> 0.29 |
| markers_human | 1 | 0.97 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.72 (3) / 0.78 (0) / 0.90 (0) | 35.0 -> 35.6 | 0.29 -> 0.29 |
| markers_human | 2 | 2.06 | 0.997 -> 0.990 | 0/40 | 0.99 (0) / 0.63 (7) / 0.80 (0) / 0.90 (0) | 35.0 -> 36.2 | 0.29 -> 0.29 |
| markers_human | 4 | 3.59 | 0.997 -> 0.988 | 0/40 | 0.98 (1) / 0.45 (14) / 0.85 (0) / 0.89 (0) | 35.0 -> 37.1 | 0.29 -> 0.28 |
| casing | 0.5 | 0.72 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.73 (3) / 0.77 (1) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| casing | 1 | 0.97 | 0.997 -> 0.993 | 0/40 | 1.00 (0) / 0.78 (1) / 0.76 (1) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| casing | 2 | 2.08 | 0.997 -> 0.987 | 0/40 | 0.99 (0) / 0.71 (3) / 0.76 (1) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| casing | 4 | 3.98 | 0.997 -> 0.976 | 1/40 | 0.98 (1) / 0.45 (14) / 0.77 (3) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| syn_mlm | 0.5 | 0.72 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.79 (0) / 0.78 (0) / 0.91 (0) | 35.0 -> 35.3 | 0.29 -> 0.29 |
| syn_mlm | 1 | 1.00 | 0.997 -> 0.997 | 0/40 | 1.00 (0) / 0.79 (0) / 0.78 (0) / 0.91 (0) | 35.0 -> 35.2 | 0.29 -> 0.29 |
| syn_mlm | 2 | 2.01 | 0.997 -> 0.998 | 0/40 | 1.00 (0) / 0.79 (0) / 0.77 (1) / 0.91 (0) | 35.0 -> 35.4 | 0.29 -> 0.29 |
| syn_mlm | 4 | 3.93 | 0.997 -> 0.995 | 0/40 | 1.00 (0) / 0.79 (1) / 0.76 (0) / 0.91 (0) | 35.0 -> 36.1 | 0.29 -> 0.29 |
| casing | 20 | 19.99 | 0.997 -> 0.774 | 10/40 | 0.93 (1) / 0.01 (32) / 0.41 (16) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.31 |
| markers_remove | 0 | 0.39 | 0.997 -> 0.997 | 0/40 | 1.00 (0) / 0.79 (0) / 0.78 (0) / 0.91 (0) | 35.0 -> 35.6 | 0.29 -> 0.29 |
| shuffle | 1 | 0.72 | 0.997 -> 0.995 | 0/40 | 1.00 (0) / 0.76 (2) / 0.81 (0) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| shuffle | 2 | 1.43 | 0.997 -> 0.998 | 0/40 | 1.00 (0) / 0.77 (1) / 0.81 (0) / 0.91 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| shuffle | 4 | 2.87 | 0.997 -> 0.998 | 0/40 | 1.00 (0) / 0.75 (2) / 0.82 (1) / 0.90 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| shuffle | 99 | 4.82 | 0.997 -> 0.998 | 0/40 | 1.00 (0) / 0.73 (3) / 0.86 (1) / 0.89 (0) | 35.0 -> 35.0 | 0.29 -> 0.29 |
| reshape_cv | 0.25 | 0.52 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.75 (2) / 0.78 (0) / 0.91 (0) | 35.0 -> 33.6 | 0.29 -> 0.23 |
| reshape_cv | 0.5 | 0.93 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.74 (2) / 0.80 (0) / 0.91 (0) | 35.0 -> 30.6 | 0.29 -> 0.50 |
| reshape_cv | 0.75 | 1.42 | 0.997 -> 0.998 | 0/40 | 1.00 (0) / 0.69 (4) / 0.86 (0) / 0.90 (1) | 35.0 -> 23.1 | 0.29 -> 0.74 |
| reshape_cv | 0.9 | 1.74 | 0.997 -> 0.996 | 0/40 | 1.00 (0) / 0.57 (9) / 0.86 (0) / 0.89 (0) | 35.0 -> 19.1 | 0.29 -> 0.87 |

**At every dose up to 4 per 100 words, no operator flipped more than 2 of 40 verdicts on desklib
or more than 1 of 40 on fakespot on this corpus** (typos@4 1, function words 1 at each dose, WordNet@4
2, hedges@4 1, casing@4 1; everything else 0), where research/17 §B4's detector-guided substitution
got 9/9. The mean probability moves — typos at 2/100w take desklib from 0.997 to 0.967, hedges at
4/100w to 0.972 — but a drop with zero or one flips is, per the brief's own rule, *no effect on the
pass rate*. The one exception is the RAID ceiling: casing swaps at 20 per 100 words flipped 10 of 40
on desklib, 32 of 32 on `academic` and 16 of 31 on `radar`, and 1 of 40 on fakespot, in text that
looks like "tHe cLinIcal oUtcomes". The held-out detector that does move at readable doses is
`academic`, which was already the least confident (0.794) and loses 14 of 32 verdicts to hedges,
conjunction insertion or casing at 4/100w; that is the same boundary effect as §4.1 seen on a
different detector, and fakespot (0.999) does not share it.

This is the single most important number in the report for the owner's hypothesis. On text the
detector is confident about — which is what our pipeline's candidates are (research/17 recorded
0.9996-1.0000) — **random noise of every type tested, including the RAID casing attack that research/00
§3 lists as the strongest surface edit, does not produce human verdicts at doses a reader would
tolerate.** The casing attack does produce them at 20 per 100 words — one capital letter flipped in
every five — which is the RAID result reproduced and is not a usable edit.

### 4.3 What did not move anything, anywhere

* **Sentence shuffle: 0/30 even when fully shuffled.** The detector is not reading discourse order.
* **Connective removal: 2/30, both boundary cases.** research/00 §4 rates "strip sentence-initial
  connectives" as *High* detector benefit. Measured: mean −0.013, and only 0.37 events per 100 words
  were available to remove on this corpus in the first place. It remains a good writing edit; it is
  not a detector edit.
* **CV reshaping into the human band: 3/30.** Pushing sentence-length CV from 0.39 to 0.47 (inside
  research/00's 0.42-0.60 band) cost 2.5 FRE points and moved desklib by −0.033. Pushing to 0.89 moved
  it by −0.035 but flipped 9 of 13 on `academic` and 9 of 33 on fakespot — those detectors *do* read
  sentence shape; desklib mostly does not.
* **MLM-neighbour substitution: 1/30 at every dose.** This is the operator inside HMGC, RAFT and
  `guided.py`'s spirit (change words to ones a language model finds plausible). Without a detector in
  the loop it does nothing at all. research/17 §B4 found the same operator *with* the detector in the
  loop flips 9/9 and transfers 0/9. Together: plausible substitution has no intrinsic humanising effect;
  what research/17 found was pure adversarial direction-finding.
* **Redundancy (pronoun → repeated noun phrase): 2-3/30**, and the operator only found 0.4-0.85
  events per 100 words to make, so this is under-dosed rather than null.

---

## 5. Experiment C — sampling noise

Script: `.noise/run_C.py`. Thirty-two real PMC opening sentences; the instruct model was asked to
write the next paragraph, the base model simply continued the sentence. Only the generated
continuation was scored; outputs under 60 words were dropped (n shows survivors). `mx.random.seed(19)`.

| model | sampler | n | words | desklib (flagged) | fakespot | academic | radar | fast | FRE | CV | topic overlap |
|---|---|---:|---:|---|---|---|---|---|---:|---:|---:|
| instruct 3B | T=0.7, top_p 0.95 | 28 | 87 | 0.988 (28) | 0.998 (28) | 0.964 (27) | 0.817 (23) | 0.898 (27) | −2.9 | 0.20 | 0.28 |
| instruct 3B | T=1.0 | 30 | 89 | 0.956 (29) | 0.999 (30) | 0.920 (28) | 0.710 (23) | 0.881 (30) | −4.7 | 0.25 | |
| instruct 3B | T=1.2 | 30 | 92 | 0.910 (28) | 0.998 (30) | 0.837 (25) | 0.621 (20) | 0.789 (26) | −11.7 | 0.23 | |
| instruct 3B | T=1.5 | 19 | 84 | 0.919 (18) | 0.811 (15) | 0.211 (4) | 0.742 (14) | 0.547 (9) | −32.7 | 0.21 | 0.12 |
| instruct 3B | T=1.0, min_p 0.05 | 28 | 83 | 0.991 (28) | 0.999 (28) | 0.944 (26) | 0.806 (24) | 0.884 (27) | −4.2 | 0.22 | |
| instruct 3B | T=1.0, min_p 0.10 | 28 | 88 | 0.995 (28) | 0.999 (28) | 0.967 (27) | 0.835 (23) | 0.903 (28) | −1.1 | 0.19 | |
| instruct 3B | T=1.5, min_p 0.10 | 28 | 86 | 0.983 (28) | 0.998 (28) | 0.893 (25) | 0.795 (24) | 0.903 (28) | −4.7 | 0.25 | |
| **base 3B** | T=0.7, top_p 0.95 | 31 | 147 | 0.892 (30) | 0.866 (29) | 0.187 (6) | 0.579 (18) | 0.797 (29) | 43.6 | 0.44 | 0.23 |
| **base 3B** | **T=1.0** | 24 | 143 | **0.199 (3)** | **0.052 (1)** | **0.001 (0)** | 0.414 (10) | 0.389 (8) | 29.9 | 0.54 | **0.11** |
| **base 3B** | T=1.2 | 14 | 97 | 0.952 (13) | 0.393 (6) | 0.000 (0) | 0.948 (14) | 0.369 (5) | −9.8 | 0.50 | 0.03 |
| **base 3B** | T=1.5, min_p 0.10 | 27 | 158 | 0.926 (26) | 0.698 (18) | 0.226 (6) | 0.525 (14) | 0.707 (23) | 40.2 | 0.50 | |

**Temperature on the instruct model does nothing useful.** From 0.7 to 1.2 desklib moves 0.988 →
0.910 with 28-29 of 30 still flagged, while FRE collapses from −3 to −12 (the model reaches for rarer
Latinate words as temperature rises, exactly as research/00 §3 says "elevate the language" prompts
do). At 1.5 the text is broken ("obese prone to inserting code here ser-upperannotationsGovernment
operations") and *still* scores 0.919. `min_p` at 0.05-0.10 makes the output *more* detectable
(0.991-0.995) because it prunes the tail that temperature was opening. This reproduces the pipeline
docstring's "six completions at 0.7-2.0 every one scored 1.0000" with a larger sample.

**The base model is a different animal, and the honest reading is a narrow, unfaithful window.** At
T=1.0 with no truncation, Qwen2.5-3B *base* produced 143-word continuations that desklib scores at
0.199, fakespot at 0.052 and academic at 0.001 — 3 of 24 flagged. Sentence-length CV 0.54 lands in
the research/00 band without being asked. This is the research/13 "base models read as human" result
reproduced locally with a 3B model. But: topic overlap with the seed paper is 0.11 against 0.28 for
the instruct model; the outputs are fluent, on-genre and *invented* ("The study predicts that the
perimedullary form of the syndrome is the leading cause of c.s. in a female patient population").
At T=0.7 the same model mode-collapses into list-like text (0.892), at T=1.2 into token salad
(0.952), and min_p 0.1 at T=1.5 gives 0.926. The human-scoring regime is one temperature wide and
contains no content the user supplied. That is the pipeline docstring's "candidates scoring 0.02-0.43
with content overlap 0.01 to 0.15" again, and research/17 §A.11's point that HIP works by
fine-tuning, not by prompting a base model.

---

## 6. Experiment D — mixed text

Script: `.noise/run_D.py`. "Matched" uses the content-aligned `human_pmc[i]` / `ai_qwen[i]` pairs and
replaces *k* sentences of one with the positionally aligned sentences of the other, so the paragraph
stays about one thing. "Unmatched" inserts *k* random sentences from `human_mage[i]` into
`ai_llama[i]` (and the reverse) at random positions. "Foreign fraction" is the share of words that
came from the other side.

| splice | foreign words | desklib before → after (flips) | fakespot | academic | radar | fast |
|---|---:|---|---|---|---|---|
| human → AI, matched, replace 1 | 0.14 | 0.731 → 0.689 (2/30) | 0.806 → 0.730 (3) | 0.311 → 0.258 (4) | 0.234 → 0.265 (2) | 0.694 → 0.691 (0) |
| human → AI, matched, replace 2 | 0.30 | 0.731 → 0.679 (6/30) | 0.806 → 0.670 (7) | 0.311 → 0.222 (6) | 0.234 → 0.295 (1) | 0.694 → 0.665 (2) |
| human → AI, matched, replace 3 | 0.47 | 0.731 → 0.733 (4/30) | 0.806 → 0.649 (6) | 0.311 → 0.229 (6) | 0.234 → 0.351 (1) | 0.694 → 0.670 (2) |
| human → AI, matched, replace half | 0.47 | 0.731 → 0.671 (5/30) | 0.806 → 0.582 (10) | 0.311 → 0.237 (5) | 0.234 → 0.351 (1) | 0.694 → 0.663 (2) |
| **AI → human, matched, replace 1** | 0.17 | 0.200 → 0.406 (**10/33**) | 0.216 → 0.324 (6) | 0.078 → 0.125 (2) | 0.130 → 0.256 (6) | 0.522 → 0.577 (3) |
| AI → human, matched, replace 2 | 0.34 | 0.200 → 0.442 (10/33) | 0.216 → 0.384 (9) | 0.078 → 0.131 (3) | 0.130 → 0.337 (9) | 0.522 → 0.588 (4) |
| AI → human, matched, replace 3 | 0.53 | 0.200 → 0.551 (14/33) | 0.216 → 0.515 (14) | 0.078 → 0.197 (6) | 0.130 → 0.362 (12) | 0.522 → 0.633 (5) |
| AI → human, matched, replace half | 0.46 | 0.200 → 0.558 (15/33) | 0.216 → 0.488 (12) | 0.078 → 0.183 (5) | 0.130 → 0.344 (11) | 0.522 → 0.630 (4) |
| human → AI, unmatched, insert 1 | 0.14 | 0.997 → 0.994 (0/40) | 0.999 → 0.992 (0) | 0.794 → 0.673 (5) | 0.780 → 0.902 (0) | 0.913 → 0.875 (0) |
| human → AI, unmatched, insert 2 | 0.22 | 0.997 → 0.977 (0/40) | 0.999 → 0.972 (1) | 0.794 → 0.502 (12) | 0.780 → 0.835 (1) | 0.913 → 0.832 (1) |
| human → AI, unmatched, insert 3 | 0.32 | 0.997 → 0.953 (**1/40**) | 0.999 → 0.945 (2) | 0.794 → 0.368 (17) | 0.780 → 0.807 (3) | 0.913 → 0.773 (5) |
| AI → human, unmatched, insert 1 | 0.14 | 0.244 → 0.316 (4/33) | 0.348 → 0.390 (3) | 0.106 → 0.087 (0) | 0.290 → 0.566 (11) | 0.563 → 0.520 (3) |
| AI → human, unmatched, insert 2 | 0.24 | 0.244 → 0.439 (7/33) | 0.348 → 0.521 (10) | 0.106 → 0.175 (3) | 0.290 → 0.582 (11) | 0.563 → 0.538 (3) |
| AI → human, unmatched, insert 3 | 0.31 | 0.244 → 0.507 (**13/33**) | 0.348 → 0.611 (13) | 0.106 → 0.102 (1) | 0.290 → 0.584 (12) | 0.563 → 0.593 (6) |

**Where does the verdict flip?** For human text going AI: at *one* sentence. One matched AI sentence
(17% of words) flipped 10 of 33 human paragraphs on desklib; three unmatched AI sentences (31%) flipped
13 of 33 and took radar from 0.29 to 0.58. For AI text going human: essentially nowhere. Half the
sentences of a boundary paragraph replaced with the real human sentences they paraphrase flipped 5 of
30; three human sentences inside a saturated paragraph flipped 1 of 40 and left desklib at 0.953.
Radar even goes *up* when human sentences are inserted into AI text (0.780 → 0.902), presumably
because juxtaposition looks like paraphrase, which is what RADAR was trained against.

The amount of "human signal" needed per paragraph is therefore not a fraction. Human text is fragile
(one AI sentence contaminates it) and AI text is robust (half human does not rescue it). The
detector is closer to a max-over-spans than a mean, which is also what its `max_chunk_probability`
plumbing and GPTZero's mixed-class head would suggest.

---

## 7. Where this contradicts research/00 and research/17

* **research/00 §3 says misspellings are dead (RAID −1.4) and casing swap was the strongest surface
  attack (−10.0).** Measured here on the boundary corpus, typos at 4/100w flipped 10/30 and casing at
  4/100w flipped 9/30 — comparable, not a 7× gap; and casing at RAID-scale (20/100w) flipped 18/30
  and took `academic` to 0.000. On the saturated corpus typos are 0-1/40 and casing 0-1/40 at ≤4/100w;
  only casing at 20/100w reaches 10/40 there, so at RAID's own dose the ranking holds and at readable
  doses it does not. The RAID deltas were aggregate-probability numbers on GPTZero; the direction
  agrees, the magnitude ranking at low dose does not.
* **research/00 §4 rates "strip connectives" High and "low-visibility punctuation slips" Medium at
  0.3-0.8 per 100 words.** Measured: connective removal 2/30 flips (mean −0.013); punctuation slips at
  0.5-1/100w 1-2/30 flips (mean −0.025). Neither is a detector lever on desklib at those doses.
  Punctuation slips only start flipping at 2/100w (6/30), which is past the research/09 human error
  rate of 2.45 total errors per 100 words when combined with anything else.
* **research/00 §5 gives sentence-length CV 0.42-0.60 as the human band.** 1,264 human PMC
  paragraphs measured here have a median of 0.38 and an IQR of 0.30-0.49 (§2). At paragraph scale the
  band is too high, and moving into it did not flip verdicts (3/30).
* **research/17 §B4 concluded that desklib's boundary is "locally movable by surface edits."**
  Confirmed only for text that is already near the boundary. On the corpus research/17 §B1 scored at
  0.997, no undirected surface edit at ≤4/100w flipped anything; what research/17 achieved with 590
  detector calls per paragraph, blind noise does not achieve at all.
* **research/17 §B3's tentative reading** that "the damage comes from the instruction, not the
  checkpoint" when human text goes through a model is **confirmed and quantified**: a maximally
  conservative polish instruction on a 7B model costs a third of the human verdicts (§3), and the base
  checkpoint with no instruction at all is the only configuration that produces human-scoring text
  (§5) — at the price of the content.
* **research/17's PMC human baseline (and this repo's) is partly extraction residue.** Removing curly
  quotes, en dashes and stray pre-punctuation spaces raised desklib by +0.092 on the 21 paragraphs that
  had them (§3). Some of the human/AI separation on this corpus is typographic.

---

## 8. What this does not show

* **Nothing here is GPTZero.** Five local checkpoints, one of them RAID-trained like GPTZero's
  training regime is not. The asymmetries and the boundary-vs-saturation pattern are structural
  enough to expect elsewhere; the specific flip counts are not.
* **The noise operators are crude.** Regex sentence splitting, WordNet's first sense, spaCy's small
  model for antecedents (the redundancy operator once produced "bET inhibitors' anti-tumor efficacy").
  A careful human editor introducing 2 errors per 100 words would produce more natural noise; whether
  the detector would treat it differently is untested.
* **Dose × type interactions were only sampled, not swept.** Two combinations were tried (§4.1
  bottom rows); neither beats its strongest component.
* **`ai_qwen` is a restatement corpus, not a generation corpus.** Its boundary position (0.731) is
  because it inherits the human source's numbers, entities and sentence plan. Our pipeline's outputs
  are generated, score 0.9996-1.0000, and behave like `ai_llama`, not `ai_qwen`. The §4.1 flip counts
  are the optimistic case.
* **The base-model result is n=24 at one temperature with no faithfulness gate.** It is reported
  because it is the largest effect observed, not because it is usable as measured.
* **No LanguageTool, no human error-rate calibration.** The T5 grammar row is contaminated by symbol
  corruption; the cleanest "grammar only" condition was not run.
* **Paragraph scale only.** Every text is 80-260 words. Document-level verdicts (research/00 §1.2)
  may aggregate differently, especially for Experiment D.
* **The `fast` detector's flip counts are noise.** It flagged 24 of 40 human paragraphs at baseline.

---

## 9. What to change in our pipeline, and the measured effect on the pass rate

The success metric throughout is verdict flips (P(AI) crossing 0.5 to "human"), N of M, on desklib
and on every held-out checkpoint. A probability drop with zero flips is written as **no effect on
the pass rate**.

### 9.1 Per noise type: dose at first flip, cost at that dose, held-out survival

Flips on `ai_qwen` (30 flagged; boundary bucket 8) and on `ai_llama` (40 flagged). "First flips"
means the lowest dose with at least 4 of 30 desklib flips (2 of 30 is within run-to-run noise: the
`shuffle@1` control gets 2/30).

| noise type | first flips (dose) | desklib flips at that dose | held-out flips at that dose (fakespot / academic / radar) | FRE cost | visible damage | on `ai_llama` |
|---|---|---:|---|---:|---|---|
| typos | 0.5/100w | 4/30 | 1 / 1 / 1 | +0.1 | misspelled words, one per two sentences at 1/100w | 0/40 at 0.5-2, 1/40 at 4 |
| function-word drop/dup | 1/100w | 4/30 | 2 / 4 / 0 | −0.3 | missing articles ("consist of BET inhibitor") | 1/40 at 1-4 |
| WordNet synonym | 0.5/100w | 5/30 | 0 / 1 / 0 | 0 | wrong-register words ("delayed → detain") | 0/40 at ≤1, 1-2/40 at 2-4 |
| casing swap | 1/100w | 6/30 | 1 / 4 / 1 | 0 | random capitals mid-word | 0/40 at ≤2, 1/40 at 4, 10/40 at 20 |
| punctuation slips | 2/100w | 6/30 | 5 / 5 / 1 | +0.3 | dropped commas/hyphens/apostrophes | 0/40 |
| hedges | 2/100w | 6/30 | 11 / 7 / 1 | +1.3 | "arguably", "or so it appears" every other sentence | 0/40 at ≤2, 1/40 at 4 (academic 14/32) |
| insert But/So/Still/Yet | 1/100w | 4/30 | 10 / 5 / 0 | +0.7 | sentence-initial conjunctions | 0/40 (academic 14/32 at 4) |
| typography (ASCII → curly) | n/a (all) | 4/30 | 4 / 1 / 0 | −0.3 | none | 0/40 |
| extraction debris (" .") | 1/100w | 4/30 | 3 / 4 / 0 | 0 | visible stray spaces | 0/40 |
| MLM neighbour | never | 1/30 at every dose | 1-2 / 1-2 / 0-2 | +0.3 to +1.1 | — | 0/40 |
| CV reshape into band | never | 3/30 | 2 / 2 / 0 | −2.5 | merged run-ons | 0/40 at every target |
| shuffle | never | 0-2/30 | 0-2 / 0-2 / 0-1 | 0 | incoherent order | 0/40 |
| remove connectives | never | 2/30 | 1 / 2 / 0 | +0.5 | none | 0/40 |
| redundancy | never | 2-3/30 | 1 / 2-4 / 1-2 | −1.0 | repeated NPs | 0/40 |

Every flip on `ai_qwen` came from the 8 boundary paragraphs plus at most 5 of 22 saturated ones (at
4/100w of the two most damaging operators). **On the saturated corpus no noise type at ≤4/100w
produced more than 2 of 40 flips on desklib or 1 of 40 on fakespot**; the 20/100w casing attack
produced 10 of 40 and is unreadable. Held-out survival among the flips that did happen: fakespot and academic
follow desklib at half to full strength for typos, punctuation, hedges and conjunction insertion;
radar follows for nothing (0-3 flips in every row).

### 9.2 Module-by-module

**`src/humanizer/humanize/transforms.py` (deterministic edits).**

* `strip_formal_connectives`: keep, for the grade. **Do not expect a pass-rate effect**: measured
  2/30 flips, mean −0.013, and the operator only found 0.37 events per 100 words to remove. Its value
  is research/00 §4's "raises the grade", not detection.
* `vary_sentence_length` and `HUMAN_SENTENCE_CV = (0.42, 0.60)` (`pipeline.py:207`, used by
  `_shape_cost`): **lower the band to the measured paragraph-level human IQR, 0.30-0.49**, or key it
  to paragraph length. Moving CV into 0.42-0.60 flipped 3/30 and cost 2.5 FRE points; moving to 0.89
  flipped 2/30 on desklib. The shape transform is a quality/naturalness edit, not a pass-rate edit,
  and the current band asks for something 60% of human PMC paragraphs do not have.
* `replace_ai_vocabulary`: the nearest measurements are the two substitution operators. Plausible
  (MLM) substitution: 1/30, no effect on the pass rate. Rare-synonym (WordNet) substitution: 11/30 at
  2/100w but through wrong-register words that a reader sees at once. **Do not extend the lexicon
  pass toward "unusual" replacements**; the number that justifies this is 0.731 → 0.724 (1/30 flips)
  for every plausible substitution at 4/100w.
* **Do add a typography-preservation rule, and a typography pass**: never normalise curly quotes,
  en/em dashes or non-breaking spaces out of source text (Experiment A: +0.092 mean on affected
  paragraphs, 2/33 flips from normalisation alone), and convert straight quotes / numeric-range
  hyphens / spaced hyphens in generated text to typographic forms (4/30 flips on the boundary corpus,
  4 on fakespot, zero readability cost). It is worth exactly what it costs — nothing — and it is 0/40
  on saturated text, so it is a tidy-up, not a strategy.
* **Do not implement** noise operators for typos, casing, function-word deletion, debris or WordNet
  substitution. Each needs 2-4/100w to flip a third of *boundary* paragraphs, is 0-2/40 on saturated
  text, and puts visible errors in the body that research/09 says gate the grade.
* A low-visibility recipe (typography + 1 punctuation slip/100w + 0.5 hedge/100w, `combo_low`)
  flipped 7/30 boundary and **0/40 saturated** (desklib 0.997 → 0.974), at +0.4 FRE and no spelling errors. If a "human error
  floor" is ever added per research/00 §4's 0.3-0.8/100w row, this is the ceiling of what it buys:
  about a quarter of already-marginal paragraphs and nothing else.

**`src/humanizer/humanize/pipeline.py` (gates, candidate selection).**

* **Add a polish gate (Experiment A sets the number).** Any LLM step that rewrites text that is
  currently human-scoring — `_scrub` on a human-ish candidate, a "polish" round, or `rounds=2`
  refinement of a candidate that already passed — must be bounded by word-change rate. Measured on 80
  human paragraphs: ≤2% words changed → +0.03 to +0.05 mean, 0-2 of 33 flips; 11% → +0.20, 11-12 of
  33 flips; 18-25% → +0.23 to +0.29, 10-16 of 33 flips. Concretely: compute
  `difflib`-style word change rate between the pre-polish and post-polish text; reject the polished
  candidate if it changed more than 5% of words *and* the pre-polish text scored under the threshold.
  The cleanest gate is to skip instruct-model polishing of any candidate that already scores human;
  the safe cleaning budget for a human-scoring text is about 2% of words, typographic characters
  excluded.
* **Rank with the whole paragraph, never by sentence.** Experiment D: one AI sentence in a human
  paragraph flips 10/33; half a paragraph of human sentences in an AI paragraph flips 5/30. Any
  selection logic that assembles a paragraph from best-scoring sentences (or keeps "good" source
  sentences) is on the wrong side of the asymmetry. The current `_rank_key` argmin over the
  paragraph proxy score is right; the sentence-level `sentence_scores` in `ModernDetector` must not
  be used for selection, which its docstring already says.
* **Do not add temperature or `min_p` diversity as a lever** (`llm.CANDIDATE_SPECS`, temperature
  scaling): Experiment C, 3B instruct, seven sampler settings, 28-30 of 30 flagged in every one.
  `min_p` made it worse (0.991-0.995).
* **Do not spend budget on more candidates from the same instruct model** to fix a saturated
  paragraph: no readable undirected perturbation of a 0.997 paragraph produced more than 2 flips in 40
  (§4.2), and the pipeline docstring's own 0 of 9 stands.
* The `"mixed"` style (half the candidates from the base checkpoint) is the only configuration that
  produced human-scoring text in this report (0.199 desklib, 3/24 flagged at T=1.0), and the content
  gate is what stops it (topic overlap 0.11). That is a correct gate. The number that would change
  this is a faithful base-model output, which needs the HIP-style fine-tune of research/17 §A.11, not
  a sampler setting; T=0.7 collapses to 0.892 and T=1.2 to gibberish at 0.952.

**`src/humanizer/humanize/guided.py` (detector-guided decoding).**

* `guided.py` steers a 3B instruct model token by token with the `fast` e5 guide, `top_k=8`,
  `guidance_strength=4.0`, scoring per sentence. Three measurements bear on it. (a) The `fast` guide
  flagged 24 of 40 genuine human paragraphs; guidance toward "what e5 calls human" is guidance
  toward a coin flip. (b) Plausible in-distribution substitution — which is all a `top_k=8` reweight
  can choose among — measured 1/30 flips without a detector and, in research/17 §B4, 9/9 *with*
  desklib in the loop and 0/9 transfer. (c) Sentence-level assembly is the losing side of Experiment
  D. **Do not extend guided decoding**; the number that justifies it is the 0.731 → 0.724 MLM row, and
  the 24/40 human false-positive rate of its guide. If it is kept, switch the guide to a detector with a
  usable FPR (`academic`: 3/40) and score whole paragraphs, not sentences.

### 9.3 The one-line answer to the owner's hypothesis

Noise moves the detector more than polishing *on paragraphs the detector was already unsure about*,
and polishing moves it more than noise *on paragraphs it was sure about* — because polishing by an
instruct model reliably makes human text AI (11-16 of 33), while readable noise almost never makes confident AI text
human (0-2 of 40 across 14 operators at up to 4 events per 100 words; 10 of 40 only for the
unreadable 20-per-100-word casing attack). Our outputs are the confident kind. Cleanliness is a symptom of
having been through the instruct model, not a cause the pipeline can undo afterwards by roughing up
the surface.

---

## 10. Scripts, data and cost

All in `.noise/`:

| file | what |
|---|---|
| `common.py` | corpora, detector wrappers, readability, sentence splitter, seed 19 |
| `baseline_and_E.py` | baseline scores for all corpora on all five detectors; Experiment E quantiles → `res_baseline.json`, `res_E_readability.json` |
| `noise_ops.py` | the ten noise operators plus MLM substitution and CV reshaping |
| `run_A.py` | Experiment A: normalisation, T5 grammar, Qwen 7B/3B polish → `res_A.json` |
| `run_B.py`, `run_B2.py` | Experiment B by type and dose, plus typography / debris / combos → `res_B_ai_qwen.json`, `res_B_ai_llama.json` |
| `run_C.py` | Experiment C: instruct and base sampling → `res_C.json` |
| `run_D.py` | Experiment D: matched and unmatched splicing → `res_D.json` |
| `tables.py` | renders every table above from the JSON |
| `log_*.txt` | per-condition timing and progress |

Every result JSON contains the full output text of every condition, so any row can be read, not
just its number.

**Wall clock.** Detector and model time by experiment: baseline 2.4 min, A 10.0 min (of which model
generation 6.6), B on `ai_qwen` 18.6 min, B on `ai_llama` 12.0 min, C 5.5 min
(generation 3.1), D 5.0 min — **about 55 minutes of compute**, run partly in parallel over about
75 minutes of wall clock, on 12 CPU cores and the M4 Pro GPU (MLX). One desklib call on a 150-word
paragraph is 0.25-0.45 s on CPU; the four held-out detectors together cost about the same again.
Downloads: T5 grammar model 850 MB, WordNet 10 MB, `textstat` and `nltk` from PyPI; nothing else
was fetched.
