# 26. A HIP adapter on the Apache-licensed Qwen2.5-7B base (2026-09-10)

All numbers below come from files under `.hip7b/` written on 2026-09-10, or from
research/24 §6 for the earlier baselines. Nothing here was measured on the free
surrogate; every bench in this note is own-key mode, GPTZero judging and ranking.

## 0. Findings up front

The shipped base checkpoint, Qwen2.5-3B, is under the Qwen Research licence and
cannot be sold; the 7B is Apache 2.0 but measured worse on 2026-09-07 (plain 7B
base 4 of 9 bench paragraphs GPTZero-human, against the 3B base's 22 of 27, 81%,
over three runs). A LoRA on the 7B base trained on 344 audited (instruct draft,
author notes, human original) rows closes that gap and passes it: adapter r3 at
step 150 scored **8 of 9 and 8 of 9 on two consecutive runs (16 of 18, 89%)**,
with the 5 human PMC controls unharmed in both (10 of 10) and 0 junk paragraphs.
The same eight paragraphs passed both times and the same one (explainer-2)
failed both times, so the variance sits in one paragraph. At candidate level, 26
of 128 and 24 of 120 candidates were both gate-passing and GPTZero-human, about
1 in 5, against the roughly 1 in 7 supply measured for the 3B base in research/24
§6.14. Each bench cost $8 to $10 of GPTZero credit. The cost is readability:
output Flesch reading ease rises 22 to 25 points on average, which is plainer
prose than the input, and research/23 §3.2 counts that as damage.

## 1. Why a 7B adapter

`deploy/merge-day.md` §4 (decision D9) states the licence problem: `FREEFORM_MODEL`
defaults to `mlx-community/Qwen2.5-3B-4bit`, Qwen2.5-3B is released under the
Qwen Research Licence, which does not permit commercial use, and the Apache-2.0
sizes are 0.5B, 1.5B, 7B, 14B and 32B. The options were to use 7B and accept a
lower pass rate, to bench 1.5B, or to ask Alibaba for a 3B licence.

The measured gap (research/24 §6.5, `.hip7b/results/baseline.md`): the 3B base
with three exemplar triples, author notes, 8 candidates x 2 rounds and GPTZero
ranking passed 7, 8 and 7 of 9 across three runs (22 of 27, 81%), with 45 to 65%
of scored candidates read as human. The 7B base under the same prompt passed 4
of 9 (55 of 111 candidates human, 50%). That 4-of-9 figure predates the §6.7 gate
correction (overlap floor 0.25, length 0.55-1.45), so it is not a fair control;
the same-day plain-7B run under the corrected gates gave 6 of 9 (§8).

The 3B HIP fine-tune of research/24 §6.6 showed the lever: candidate-level human
share rose from 45% to 63%, but paragraph passes fell from 7 to 5 of 9 because
the adapter learned to write a PMC original rather than a rewrite of the draft
and its human candidates died at the fidelity gates. This round applies that
recipe to the 7B base with the data problems fixed.

## 2. What changed against the 3B HIP round

From `.hip7b/NOTES.md`:

| | 3B round (research/24 §6.6) | this round |
|---|---|---|
| base | Qwen2.5-3B-4bit (Qwen Research licence) | Qwen2.5-7B-4bit (Apache 2.0) |
| prompt format | old header, DRAFT/HUMAN, no NOTES | the shipped format: header, one exemplar triple, DRAFT, NOTES, HUMAN |
| NOTES | none | particulars in the human original missing from the draft, as clauses; "(none...)" when the draft has them all |
| loss | whole sequence | masked to the HUMAN completion |
| pairs | 240, PMC only | 628 of 1,500 (PMC, PMC2, pre-2022 Wikipedia) in v1; 344 train rows after the audit |
| filters | none | length ratio, overlap, word count, FRE, connective openers, summary closer, residue, gate consistency |
| schedule | 300 iters, batch 2x2, lr 1e-4; memorised past 200 | batch 1x4, lr 5e-5, LoRA rank 8 on 16 layers (11.5M parameters, 0.151%) |

Masking the prompt stops a third of the gradient teaching the instruct paraphrase
register. NOTES in training is the mechanism the gates enforce at inference: add
a particular only when the author licensed it (`build_data.py` docstring).

### 2.1 The audit that killed r1 (`.hip7b/audit/REPORT.md`)

r1 trained on `build_data.py` output (593 train / 35 valid). A CPU-only audit of
those rows, run while r1 was training, found:

1. **63% of kept targets fail the product's own evade gate against their draft**
   with the row's NOTES as facts (274 entity failures, 174 numbers, 92
   quotations). Half are the Qwen-3B paraphraser's fault (expanded acronyms,
   invented dates, added parentheticals), but either way the pair teaches
   deletion of a source particular, the research/24 §6.6 drift mechanism.
2. **27% of train rows (an estimated 161 of 593) exceed the 1,536-token cap**, and
   the trainer truncates the tail, which is the HUMAN completion, the only part
   carrying loss. The exemplar block alone is 650 to 750 tokens.
3. **22.5% of rows have a number or name in the target that neither the draft nor
   the NOTES license**; the product refuses exactly these completions.
4. **29% of rows carry PMC scrape residue** ("shown in.", "( and, and)", "&gt;",
   bracket citations, glued section headings on 98 targets); 6% of drafts are
   markdown or meta text.
5. **The validation split leaks documents**: 30 of 31 validation documents also
   have paragraphs in train, so the 3B round's "memorises past step 200" was
   read against leaked paragraphs.
6. Wikipedia is the better source on every quality axis (3% residue against 34
   to 39%, FRE 41 against 31, 7 of 7 hand-reviewed targets worth imitating) and
   was 20% of rows.
7. Contamination: clean. No 60-character shingle of any training text appears in
   the 14 bench texts or 9 facts blocks.

The fix list, as implemented in `build_data_v2.py`: keep only pairs whose target
passes the evade gate against its draft with NOTES as facts (entities relaxed to
"some token of the name survives"); rebuild NOTES to cover every particular the
target adds and drop the row if `new_specifics(draft, human, facts=notes)` is
still non-empty, so unlicensed specifics fall to 0%; measure each row with the
tokenizer, fall back to the shortest exemplar, drop what still does not fit;
strip residue, unescape HTML entities, drop markdown and meta drafts, strip glued
headings by looking the paragraph up in the raw file; split train/valid by
document; ratio 0.60-1.25, overlap 0.35-0.85, FRE at or above 10 and not more
than 8 points denser than the draft, sentence-length CV at or above 0.15; cap 6
pairs per document, wiki floor 25%, wiki rows written twice. Result: 296 unique
pairs, 344 train rows, 22 valid, longest row 1,532 tokens, 0 validation documents
in train, gate pass 68.6% (from 36.9%), unlicensed specifics 0% (from 22.5%).

## 3. Runs r1 to r4

Validation losses from `train_r1.log` to `train_r4.log`; row counts and stop
reasons from the logs and `NOTES.md`.

| run | data | rows train / valid | val loss by step | stopped because |
|---|---|---|---|---|
| r1 | data (v1, `build_data.py`) | 593 / 35 | 1.106 (0), 0.909 (100) | at iter 180: the audit found 27% truncated completions and 63% gate-failing targets |
| r2 | data_v2 (`build_data_v2.py`) | 344 / 22 | 1.088 (0), 0.909 (100), 0.904 (200), 0.938 (300) | at 300, val rising; step 200 benched |
| r3 | data_v3 (v2 with the DRAFT stop tail) | 344 / 22 | 1.076 (0), 0.941 (50), 0.914 (100), **0.884 (150)**, 0.901 (200) | at 200, val rising; step 150 benched |
| r4 | data_v4 (`build_data_v4.py`, terse notes) | 339 / 22 | 1.107 (0), 0.966 (50), 0.929 (100), 0.927 (150), 0.923 (200), 0.932 (250) | log ends at 250; step 200 checkpoint saved, bench started (§8) |

Throughput on the M4 Pro was 0.15 to 0.21 iterations a second at batch 1 with
4-step accumulation, peak memory 14.6 to 16.4 GB, about 66 to 78 s per validation
pass. Note that `NOTES.md` records r1 as "initial val loss 0.865, peak 13.5 GB,
~0.5 it/s"; the training log says 1.106, 14.6 to 16.3 GB and 0.15 to 0.20 it/s,
and this note uses the log.

## 4. The r2 garbage tail (`results/r2-it200.json`, `r2-it200.log`)

Bench of r2 step 200, own-key, 8 x 2: AI paragraphs human **9 of 9**, humans
unharmed 5 of 5, mean overlap 0.51, mean length ratio 1.07, mean FRE change +25,
candidates passed 68 of 128, candidate-level human 41 of 88 (47%),
passed-and-human 23 of 128, $9.76.

On inspection every chosen text carried a tail of multilingual token noise after
the paragraph. The training completion ended with EOS, but the pipeline stops a
base-model completion at the next label (`FREEFORM_STOPS` in `humanize/llm.py`:
"\nDRAFT:", "\nNOTES:", "\nHUMAN:", "\nEditing log"), and the adapter had
unlearned the few-shot habit of writing the next DRAFT block, so generation ran
past the paragraph into noise that the gates did not catch (the word regex
counts Latin letters only). The sentence-length CV column in the r2 table shows
it: 0.90 to 2.11 after, against a source band of 0.16 to 0.32.

Re-scoring the nine paragraphs with the tails cut at the first newline gave
**still 9 of 9 human**, scores mostly lower (explainer-1 0.354 to 0.002,
explainer-2 0.016 to 0.249, the others within 0.03), so the prose itself passes
and the garbage was not the cause. That re-score is recorded in `NOTES.md` only;
there is no results file for it.

Two fixes:

1. Training side: data_v3 is data_v2 with each completion ending in
   `"\n\nDRAFT:\n"` instead of `"\n\n"`, so the model emits the stop the pipeline
   already handles. (There is no `build_data_v3.py`; the transform was a one-off,
   verifiable in `data_v3/train.jsonl`, and `build_data_v4.py` reads data_v3.)
2. Product side (`deploy/merge-day.md` §10): in `clean_completion`, after the
   stop-string cut, cut at the first blank line; in `pipeline._gate_candidate`,
   reject a candidate when more than 2% of its characters are letters outside
   Latin scripts, or when it contains three or more `!` in a row, reason string
   `"junk_text"`. The bench added a `junk` column at the same time.

A second r2 bench (`r2-it200-run2.log`) was started and abandoned at its header
once the tail defect was found.

## 5. r3 step 150, two runs (`results/r3-it150.json`, `results/r3-it150-run2.json`)

Own-key mode, `mlx-community/Qwen2.5-7B-4bit` with
`.hip7b/adapters/hip7b-r3-it150`, 8 candidates x 2 rounds, author facts on the AI
paragraphs, none on the human controls. "Before" is GPTZero P(ai) on the source;
"after" is 1 minus P(human) on the chosen text. Overlap, length ratio, FRE and CV
are run 1 / run 2.

| paragraph | before | after run 1 | after run 2 | overlap | length ratio | FRE source > after | CV after |
|---|---|---|---|---|---|---|---|
| explainer-1 | 1.000 | 0.006 human | 0.014 human | 0.36 / 0.27 | 1.09 / 0.91 | 10 > 23 / 40 | 0.35 / 0.42 |
| explainer-2 | 1.000 | **1.000 ai** | **1.000 ai** | 0.44 / 0.45 | 1.27 / 1.38 | 16 > 41 / 42 | 0.33 / 0.33 |
| explainer-3 | 1.000 | 0.060 human | 0.000 human | 0.41 / 0.46 | 1.21 / 1.00 | 12 > 44 / 42 | 0.38 / 0.41 |
| microplastics | 1.000 | 0.001 human | 0.004 human | 0.45 / 0.49 | 0.83 / 0.85 | 14 > 34 / 19 | 0.33 / 0.34 |
| antibiotic | 1.000 | 0.000 human | 0.002 human | 0.43 / 0.54 | 0.92 / 1.10 | 7 > 22 / 35 | 0.27 / 0.32 |
| urban heat | 1.000 | 0.000 human | 0.005 human | 0.31 / 0.44 | 0.96 / 0.67 | 18 > 59 / 38 | 0.46 / 0.12 |
| sleep | 1.000 | 0.000 human | 0.000 human | 0.27 / 0.51 | 0.86 / 0.83 | 19 > 54 / 26 | 0.39 / 0.36 |
| supply chain | 1.000 | 0.001 human | 0.003 human | 0.46 / 0.35 | 1.07 / 0.97 | -1 > 36 / 40 | 0.30 / 0.42 |
| teaching | 1.000 | 0.000 human | 0.018 human | 0.36 / 0.52 | 1.30 / 1.45 | 14 > 23 / 27 | 0.23 / 0.13 |
| **AI paragraphs human** | | **8 of 9** | **8 of 9** | | | | |
| 5 human PMC controls | 0.000-0.001 | 0.000-0.002, 5 of 5 | 0.000, 5 of 5 | | | | |

Source sentence-length CV for the AI paragraphs is 0.16 to 0.32 (mean 0.23).

Human controls: PMC8730340 had no gate passer in either run (0 of 8; the
best-scoring rewrites added Dexamethasone, COVID, 19 and similar, refused as
unverified) and was kept as the original; PMC8752981 had 1 of 8 both times;
PMC8762392 1 of 8 then 3 of 8. All five stayed at 0.000 to 0.002.

Summary lines, as printed:

- Run 1: AI paragraphs human 8/9; humans unharmed 5/5; mean overlap 0.39; mean
  length ratio 1.06; mean FRE delta +25.0; new specifics 1; junk paragraphs 0;
  candidates passed 81/128; candidate-level human 38/88 (43%), passed+human 26;
  GPTZero est. $8.94 (154 calls, 126 cache misses, 19,424 words); 1,383 s.
- Run 2: AI paragraphs human 8/9; humans unharmed 5/5; mean overlap 0.45; mean
  length ratio 1.02; mean FRE delta +22.2; new specifics 3; junk paragraphs 0;
  candidates passed 76/120; candidate-level human 33/80 (41%), passed+human 24;
  GPTZero est. $7.83 (144 calls, 116 cache misses, 17,027 words); 1,425 s.

Gate rejections, run 1 (47 of 128): numbers 28, entities 22, length ratio 9,
content drift 6, too short 2, quotations 1, dates 1. Run 2 (44 of 120): numbers
30, entities 24, content drift 4, length ratio 4, too short 3, dates 1. Against
r2's 60 of 128 (length ratio 29, numbers 25, entities 18, quotations 9), the
DRAFT stop removed most of the length failures; what remains is invented
numbers and names, the same failure class research/24 §6.16 found on the 3B
base.

Two runs of r3 give **16 of 18 (89%)** against the 3B base's 22 of 27 (81%);
the same eight paragraphs passed both times and the ninth failed both times.
`deploy/merge-day.md` §4 records the licence question as resolved in favour of
7B plus this adapter on that result.

## 6. The explainer-2 diagnosis

explainer-2 failed in both r3 runs at 1.000 with 12 of 16 candidates passing the
gates. Its author facts are full sentences. Every gate-passing candidate
reproduced them nearly verbatim (identical text at temperatures 0.8 to 0.95),
and GPTZero read the stitched result as AI at 1.000. The candidates GPTZero read
as human paraphrased the notes and then invented particulars ($300 billion,
Hurricane Maria) that the gates rightly refused (run 2 records 2 new specifics
on this paragraph).

Cause on the training side: v2 NOTES were clause windows lifted from the target,
up to 12 words each, and 17% of items were 80% or more of a target sentence (23%
in v1, per the audit). That taught copying the note into the rewrite. In r2,
before the stop fix, the same paragraph passed at 0.016 with overlap 0.66, so the
adapter can pass it; the r3 checkpoint happens to copy.

r4's change (`build_data_v4.py`): each note is the particular itself with at
most two words of context on either side, up to 8 items per row, `new_specifics`
still empty on every row, so the model has to compose the sentence around the
fact. 339 train / 22 valid; same schedule as r3. Its bench: 9 of 9, humans 5 of 5, junk 0 (§8).

## 7. Quality observations

From the two r3 summary blocks and per-paragraph tables:

- **Readability rises 22 to 25 points on average.** Sources sit at FRE -1 to 19
  (band A of research/21, very dense); outputs land at 19 to 59, mostly band B
  and C. That is plainer prose than the input. research/23 §3.2 identifies
  exactly this shift (Undetectable AI's long-token and Academic Word List loss)
  as the main damage of the best-evading commercial tool, and research/21 finding
  1 says FRE measures sentence length, not quality, so the number itself is not
  the harm; the loss of the source register is. The data builder allows a
  target up to 8 points denser than its draft and does not bound how much
  plainer, so the adapter learned the direction the training pairs had (human
  originals are FRE 34 against drafts at 27) and the candidate ranking by
  GPTZero amplifies it. This is a quality cost to weigh, not a pass.
- **Length ratio 1.02 to 1.06 mean**, range 0.67 to 1.45 per paragraph. No
  systematic expansion, which research/23 §3.2 says separates human revisers
  (who compress) from commercial humanizers (who inflate 22 to 60%). Teaching at
  1.30 and 1.45 and explainer-2 at 1.27 and 1.38 are the exceptions.
- **Sentence-length CV after: 0.23 to 0.46 in run 1, 0.12 to 0.42 in run 2**,
  means 0.34 and 0.32 from a source mean of 0.23. The paragraph band is
  0.30-0.49 (research/00 §19, research/19 §2). Run 1 sits inside it or just
  below; run 2 has urban heat at 0.12 and teaching at 0.13, well below, and
  both still passed GPTZero, which fits research/24 §6.4: GPTZero is not
  reading burstiness.
- **Formal connective openers: 9 in the sources, 0 after in run 1, 1 after in
  run 2** (explainer-1 kept one of its two). **Contractions: 0 before, 0 after in
  run 1, 1 in run 2** (antibiotic). The adapter does not lean on either lever.
- **Structure flags that remain** (research/24 §4 checks): run 1 repeated
  openers on 4 paragraphs, surface statements 3, uniform beats 2; run 2
  repeated openers 2, surface 1, uniform beats 2. Teaching carries all three in
  both runs. r2 had 3 over-hedged flags; r3 has none.
- **New specifics** (numbers or names in the output licensed by neither the
  draft nor the notes): 1 in run 1 (urban heat), 3 in run 2 (explainer-1 1,
  explainer-2 2), against 6 in r2. These are on chosen texts that passed the
  gates, so they are counted by a looser measure than the gate; the gate's own
  rejections (numbers 28 to 30, entities 22 to 24 a run) show the adapter still
  invents at roughly a quarter of candidates.
- **Content overlap 0.39 and 0.45 mean**, from r2's 0.51. The evade floor is
  0.25; sleep at 0.27 and explainer-1 at 0.27 are at the boundary research/24
  §6.7 described, where GPTZero-human candidates cluster.

## 8. r4 and the same-day control (filled 2026-09-11)

| item | AI human | humans | candidates passed | passed-and-human | cost | file |
|---|---|---|---|---|---|---|
| r4 step 200 (terse notes), own-key 8 x 2 | **9 of 9** (explainer-2 0.000, overlap 0.65) | 5 of 5 | 68 of 120 | 26 (candidate-level human 45 of 80, 56%) | $15.67 | `.hip7b/results/r4-it200.json` (the run crashed on the third control with a Metal out-of-memory while the control bench ran alongside it; resumed with `--resume` on 2026-09-11 for the last three controls) |
| plain 7B base, same day, corrected gates, own-key 8 x 2 | **6 of 9** (explainer-1 1.000, explainer-2 1.000, supply chain 0.654; microplastics passed at 0.458) | 5 of 5 | 82 of 144 | 10 (candidate-level human 19 of 104, 18%) | $15.40 | `.hip7b/results/base7b-control.json` |

Read together: under identical conditions the plain 7B base passes 6 of 9
with 18% of its candidates GPTZero-human and 10 faithful-and-human candidates
in 144; adapter r4 passes 9 of 9 with 56% human and 26 faithful-and-human in
120. Terse notes fixed explainer-2 (0.000, composed around the particulars,
not copied) without costing any of the other eight. r4 runs shorter than r3
(mean length ratio 0.88 against 1.02 to 1.06) and its readability rise is
smaller (+15 against +22 to +25). r4 is the shipped adapter
(`data/adapters/hip7b-r4-it200`, `llm.DEFAULT_BASE_ADAPTER`).

## 9. How to reproduce

Data (CPU only, no model):

```
.venv/bin/python .hip7b/build_data_v2.py         # .hip7b/data_v2; defaults per-doc 6, wiki share 0.25, wiki repeat 2, max-seq 1536
# data_v3: data_v2 with each completion's trailing "\n\n" replaced by "\n\nDRAFT:\n" (one-off; no script)
.venv/bin/python .hip7b/build_data_v4.py         # .hip7b/data_v4 from data_v3, terse notes
.venv/bin/python .hip7b/audit/profile.py         # per-row metrics behind audit/REPORT.md
.venv/bin/python .hip7b/audit/compare.py .hip7b/data .hip7b/data_v2
```

Training (`train.py`; defaults batch 1, grad accumulation 4, lr 5e-5, rank 8,
16 layers, max seq 1536; r2 used a 1,600 cap per NOTES.md):

```
.venv/bin/python .hip7b/train.py --data .hip7b/data_v3 --tag hip7b-r3 --iters 300 --steps-per-eval 50 --save-every 50
.venv/bin/python .hip7b/train.py --data .hip7b/data_v4 --tag hip7b-r4 --iters 300 --steps-per-eval 50 --save-every 50
```

Checkpoints land in `.hip7b/adapters/<tag>/` as `adapter_config.json` plus
`adapters.safetensors` and dated `0000150_adapters.safetensors` files; a
benchable step is a copy with that step's weights renamed to
`adapters.safetensors` (`.hip7b/adapters/hip7b-r3-it150/`). The staged copy for
the product is `data/adapters/hip7b-r3-it150/`.

Bench (needs `GPTZERO_API_KEY`; about $1 a paragraph):

```
.venv/bin/python .hip7b/bench.py --model mlx-community/Qwen2.5-7B-4bit --adapter .hip7b/adapters/hip7b-r3-it150 --mode ownkey --tag r3-it150
.venv/bin/python .hip7b/bench.py --model mlx-community/Qwen2.5-7B-4bit --mode ownkey --tag base7b-control
.venv/bin/python .hip7b/baseline.py --out .hip7b/results/baseline.md   # the table the adapter has to beat
```

`bench.py` sets `HUMANIZER_BASE_MODEL`, `HUMANIZER_BASE_ADAPTER`,
`HUMANIZER_PROXY=gptzero`, `HUMANIZER_YARDSTICK=gptzero` and the pass threshold
(0.5 own-key) before importing any humanizer module, because `humanize/llm.py`
reads `HUMANIZER_BASE_MODEL` at import. An empty `HUMANIZER_BASE_ADAPTER` means
the plain base. Results are rewritten after every paragraph; `--resume` skips
paragraphs already in the file; `--selftest` exercises the table code with no
model or network.

To run the product on this adapter today, without a code change:

```
HUMANIZER_BASE_MODEL=mlx-community/Qwen2.5-7B-4bit HUMANIZER_BASE_ADAPTER=data/adapters/hip7b-r3-it150
```

Merge-day edits (`deploy/merge-day.md` §4 and §10, not yet applied in
`src/humanizer/humanize/llm.py`, whose `FREEFORM_MODEL` default is still the 3B):
set the `FREEFORM_MODEL` default to `"mlx-community/Qwen2.5-7B-4bit"`; give
`HUMANIZER_BASE_ADAPTER` a default of `data/adapters/hip7b-r3-it150` in
`backend_for`; add the adapter directory to `deploy/sync-weights.sh`; in
`clean_completion` cut at the first blank line after the stop-string cut; in
`pipeline._gate_candidate` reject `junk_text` (over 2% non-Latin letters, or
three or more `!` in a row).

## Sources

- `.hip7b/NOTES.md`
- `.hip7b/audit/REPORT.md`, `.hip7b/audit/profile.py`, `.hip7b/audit/compare.py`, `.hip7b/audit/hand_review_sample.md`
- `.hip7b/results/baseline.md`, `.hip7b/baseline.py`
- `.hip7b/results/r2-it200.json`, `.hip7b/results/r2-it200.log`, `.hip7b/results/r2-it200-run2.log`
- `.hip7b/results/r3-it150.json`, `.hip7b/results/r3-it150.log`
- `.hip7b/results/r3-it150-run2.json`, `.hip7b/results/r3-it150-run2.log`
- `.hip7b/results/r4-it200.log`, `.hip7b/results/base7b-control.log` (headers only at time of writing)
- `.hip7b/train_r1.log`, `.hip7b/train_r2.log`, `.hip7b/train_r3.log`, `.hip7b/train_r4.log`
- `.hip7b/build_data.py`, `.hip7b/build_data_v2.py`, `.hip7b/build_data_v4.py`, `.hip7b/train.py`, `.hip7b/bench.py`
- `.hip7b/data_v3/train.jsonl` (the DRAFT stop tail)
- `.hip7b/adapters/hip7b-r3-it150/`, `data/adapters/hip7b-r3-it150/`
- `deploy/merge-day.md` §4, §10
- `src/humanizer/humanize/llm.py` (`FREEFORM_MODEL`, `FREEFORM_STOPS`, `clean_completion`, `backend_for`)
- research/24 §6.4 to §6.8, §6.14, §6.16; research/23 §3.2, §7; research/21 findings 1 to 5, §10.1; research/00 §19; research/19 §2

## 10. The last non-commercial reference (2026-09-11)

`llm.DEFAULT_MODEL`, used by the `faithful` style and the repair stage's
bridge draft, pointed at Qwen2.5-3B-Instruct, which carries the same Qwen
Research licence as the 3B base. It now defaults to
`mlx-community/Qwen2.5-7B-Instruct-4bit` (Apache 2.0, 4 GB), overridable with
`HUMANIZER_INSTRUCT_MODEL`. Smoke-tested through `humanize_llm`: freeform (7B
base plus adapter r4) rewrote a three-sentence draft in 20 s, faithful (7B
Instruct) in 10 s. Every checkpoint the product loads by default is now
Apache 2.0 or MIT; the licence table is in the 2026-09-11 session notes and
`deploy/merge-day.md` §4.
