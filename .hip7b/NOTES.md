# HIP adapter on the Apache-licensed Qwen2.5-7B base (2026-09-10)

Goal: an adapter on the 7B base that beats the shipped 3B base (own-key
GPTZero 7-8 of 9, 81% over three runs; free mode 5-7 of 9) on the same
bench, so the product can run on a commercially licensed checkpoint.

## What changed versus the 3B HIP round (research/24 §6.6)

| | 3B round | this round |
|---|---|---|
| base | Qwen2.5-3B-4bit (Qwen Research licence) | Qwen2.5-7B-4bit (Apache 2.0) |
| prompt format | old header, DRAFT/HUMAN, no NOTES | the shipped format: header, one exemplar triple, DRAFT, NOTES, HUMAN |
| NOTES | none | particulars in the human original missing from the draft, as clauses; "(none...)" when the draft has them all (230 of 628 rows) |
| loss | whole sequence | masked to the HUMAN completion |
| pairs | 240, PMC only | 628 of 1,500 (PMC, PMC2, pre-2022 Wikipedia), filtered for fidelity and shape |
| filters | none | human/draft length 0.70-1.35, overlap >= 0.40, 50-260 words, FRE 15-80, <= 1 connective opener, no summary closer, no citation residue or tokenised spacing |
| schedule | 300 iters, batch 2x2, lr 1e-4; memorised past 200 | 1,200 iters, batch 1x4, lr 5e-5, eval every 100, checkpoint every 100 |

Why each: the 3B adapter raised GPTZero-human candidates from 45% to 63% but
lost paragraphs to the fidelity gates because it learned to write a PMC
original rather than a rewrite of the draft (§6.6, §6.7). Masking the prompt
stops a third of the gradient teaching the instruct paraphrase register.
NOTES in training is the mechanism the gates enforce at inference: add a
particular only when the author licensed it. Shape filters follow research/23
§7 (the passing rewriters shift distribution without expansion or register
drop) and research/21 §10 (connective openers and summary closers are the AI
tells; specifics are the human ones).

## Runs

- r1: 593 train / 35 valid, tokens median 1,029, peak 13.5 GB, ~0.5 it/s.
  Initial val loss 0.865.
  Stopped at iter 180 (val 1.106 -> 0.909 at 100) after the data audit
  (`.hip7b/audit/REPORT.md`) found 27% of rows over the 1,536-token cap, so
  the trainer was truncating the HUMAN completion, and 63% of targets failing
  the product's fidelity gate against their draft (the §6.6 drift mechanism).
- r2: data_v2 from `build_data_v2.py` (gate-passing pairs only, NOTES that
  license every added particular so `new_specifics` is empty on 100% of
  rows, scrape residue and markdown drafts removed, document-level split,
  wiki pairs written twice): 344 train / 22 valid, longest row 1,532 tokens,
  seq cap 1,600. 1,000 iters (about 3 epochs at batch 1x4), lr 5e-5.
  Val: 1.088 -> 0.909 (100) -> 0.904 (200) -> 0.938 (300); stopped at 300,
  step 200 benched.

## Bench r2 step 200, own-key GPTZero, 8 candidates x 2 rounds (`results/r2-it200.json`)

AI paragraphs human **9 of 9**, humans unharmed 5 of 5, mean overlap 0.51,
mean length ratio 1.07, mean FRE +25, candidates passed 68/128, candidate-
level human 41/88 (47%), passed-and-human 23 of 128 (the 3B base supply was
about 1 in 7). $9.76.

**Defect found on inspection:** every chosen text carried a tail of
multilingual token noise after the paragraph. The training completion ended
with EOS, but the pipeline stops a base-model completion at the next label
(`FREEFORM_STOPS`: "\nDRAFT:" etc.), and the adapter had unlearned the
few-shot habit of writing the next DRAFT block, so generation ran past the
paragraph into noise that the gates did not catch (the word regex counts
Latin letters only). Re-scoring the nine paragraphs with the tails cut at
the first newline: **still 9 of 9 human**, scores mostly lower
(explainer-1 0.354 -> 0.002, explainer-2 0.016 -> 0.249, others within
0.03), so the prose itself passes and the garbage was not the cause. Two
fixes: (1) r3 trains on completions ending "\n\nDRAFT:\n" so the model
emits the stop the pipeline already handles; (2) the product should cut a
freeform completion at its first blank line and reject any candidate with a
run of non-Latin script (merge-day item).

- r3: data_v3 = data_v2 with the DRAFT stop tail; 300 iters, eval/save
  every 50, lr 5e-5.
  Val: 1.076 -> 0.941 (50) -> 0.914 (100) -> **0.884 (150)** -> 0.901 (200);
  stopped, step 150 benched.

## Bench r3 step 150, own-key GPTZero, 8 x 2 (`results/r3-it150.json`)

AI paragraphs human **8 of 9** (explainer-2 at 1.000: 12 of 16 candidates
passed the gates, all read as AI), humans unharmed 5 of 5, **junk 0**, every
chosen text one line ending on a full sentence. Mean overlap 0.39, length
ratio 1.06, FRE +25, sentence-length CV after 0.23-0.46 (paragraph band
0.30-0.49; r2's 2.x values were the garbage tails), candidates passed
81/128, candidate-level human 38/88 (43%), passed-and-human 26 of 128.
$8.94. Run 2 of the same checkpoint queued for the consistency check; a
same-day plain-7B-base run with the corrected gates is queued after it, since
the 4-of-9 baseline for the 7B base predates the §6.7 gate correction.

## explainer-2 diagnosis (both r3 runs)

Its facts are full sentences. Every gate-passing candidate reproduced them
nearly verbatim (identical text at temperatures 0.8 to 0.95) and GPTZero
read the stitched result as AI at 1.000; the candidates GPTZero read as
human paraphrased the notes and then invented particulars ($300 billion,
Hurricane Maria) that the gates rightly refused. Cause on the training side:
NOTES were clause windows lifted from the target (23% of items were 80%+ of a
target sentence per the audit), which taught copying.

- r4: data_v4 = data_v3 with terse notes (the particular plus at most two
  words of context each side, up to 8 items, `new_specifics` still empty on
  every row): 339 train / 22 valid. Same schedule as r3.

## Bench r3 step 150, run 2 (`results/r3-it150-run2.json`)

AI paragraphs human **8 of 9** (explainer-2 again), humans 5 of 5, junk 0.
Two runs of r3: **16 of 18 (89%)** against the 3B base's 22 of 27 (81%);
the same eight paragraphs passed both times and the ninth failed both
times, so the variance is in one paragraph, not in the adapter.

  Val: 1.107 -> 0.966 (50) -> 0.929 (100) -> 0.927 (150) -> **0.923 (200)**
  -> 0.932 (250). Step 200 benched.

## Bench r4 step 200, own-key GPTZero, 8 x 2 (`results/r4-it200.json`)

AI paragraphs human **9 of 9**, including explainer-2 at 0.000 (overlap
0.65, composed around the particulars rather than copied). Junk 0. Length
ratios run short: 0.61 to 1.30, most under 0.85, where r3 sat near 1.0;
overlap 0.29 to 0.65. Humans unharmed 5 of 5 (run crashed on the third control, Metal OOM with the control bench alongside; resumed 2026-09-11). Candidate-level human 45/80 (56%), passed-and-human 26 of 120. $15.67.

## Same-day control: plain Qwen2.5-7B base, corrected gates (`results/base7b-control.json`)

AI paragraphs human **6 of 9** (explainer-1 1.000, explainer-2 1.000,
supply chain 0.654; microplastics passed at 0.458). So under identical
conditions the adapter adds three paragraphs over the plain base, and the
7B base itself improved from 4 of 9 (2026-09-07, older gates) to 6 of 9.

## Shipped

`data/adapters/hip7b-r4-it200`; `llm.FREEFORM_MODEL` default is now the 7B
base and `llm.DEFAULT_BASE_ADAPTER` points at the adapter (2026-09-10).
