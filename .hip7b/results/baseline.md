# Baseline: existing GPTZero bench rows (.kriukow/gptzero_rows*.json)

Generated 2026-09-10T16:24:17 by .hip7b/baseline.py. 9 AI paragraphs with author facts, 5 human PMC controls.

| file | when | config (research/24) | AI human | humans unharmed | changed | candidates GPTZero-human (cache window) |
|---|---|---|---|---|---|---|
| gptzero_rows.json | 09-07 17:47 | register adapter r4 (Qwen3-4B-Instruct+LoRA), facts, 8 cand x 1 round, GPTZero judge [6.1] | 0/9 | 5/5 | 13/14 | 5/30 human (17%), 1 mixed |
| gptzero_rows_v2.json | 09-07 18:11 | 3B base freeform, exemplar triples + notes, 8x2, GPTZero ranks [6.4] | 7/9 | 5/5 | 12/14 | 97/217 human (45%), 6 mixed |
| gptzero_rows_7b_hard.json | 09-07 18:15 | 7B base freeform, 3 holdout paragraphs only, 8x2, GPTZero ranks [6.5] | 1/3 | 0/0 | 3/3 | 2/24 human (8%), 3 mixed |
| gptzero_rows_7b.json | 09-07 18:34 | 7B base freeform, 8x2, GPTZero ranks [6.5] | 4/9 | 5/5 | 12/14 | 55/111 human (50%), 5 mixed |
| gptzero_rows_hip3b.json | 09-07 19:07 | 3B base + HIP LoRA step 300, 8x2, GPTZero ranks [6.6] | 5/9 | 5/5 | 11/14 | 71/112 human (63%), 3 mixed |
| gptzero_rows_hip3b_it200.json | 09-07 19:19 | 3B base + HIP LoRA step 200, 8x2, GPTZero ranks [6.6] | 5/9 | 5/5 | 12/14 | 71/112 human (63%), 1 mixed |
| gptzero_rows_base_r3.json | 09-07 19:39 | 3B base, 8 cand x 3 rounds, GPTZero ranks, old gates [6.7] | 5/9 | 5/5 | 13/14 | 69/109 human (63%), 4 mixed |
| gptzero_rows_c1.json | 09-07 22:01 | 3B base, 8x2, GPTZero ranks, corrected gates (overlap 0.25, len 0.55-1.45), run 1 [6.8] | 8/9 | 5/5 | 12/14 | 88/145 human (61%), 9 mixed |
| gptzero_rows_c2.json | 09-07 23:21 | 3B base, 8x2, GPTZero ranks, corrected gates, run 2 [6.8] | 7/9 | 5/5 | 13/14 | 68/105 human (65%), 4 mixed |
| gptzero_rows_surrogate.json | 09-08 16:07 | free mode: surrogate v1 ranks, threshold 0.5, 8x2 [6.11] | 5/9 | 5/5 | 13/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_surrogate2.json | 09-08 16:23 | free mode: surrogate v1 ranks, run 2 [6.11] | 4/9 | 5/5 | 13/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_surrogate3.json | 09-08 20:00 | free mode: surrogate v3 (1,136 labels), threshold 0.15, 8x3 [6.12] | 3/9 | 5/5 | 11/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_surrogate4.json | 09-09 13:54 | free mode: surrogate v4 (1,265 labels), 0.15, 8x3 [6.14] | 7/9 | 5/5 | 12/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_surrogate5.json | 09-09 16:11 | free mode: surrogate v4, consistency run [6.14] | 5/9 | 5/5 | 13/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_surrogate6.json | 09-09 16:39 | free mode: surrogate v5 (1,368 labels), 8x3 [6.14] | 6/9 | 5/5 | 12/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_surrogate7.json | 09-09 17:07 | free mode: surrogate v6 (1,452 labels), 12 cand x 3 [6.14] | 5/9 | 5/5 | 11/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_surrogate8.json | 09-09 17:39 | free mode: surrogate v7/v8 (fixed split), 8x3 [6.14] | 5/9 | 5/5 | 13/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_r5.json | 09-09 17:56 | free mode: surrogate v8, 8 cand x 5 rounds [6.14] | 5/9 | 5/5 | 12/14 | n/a (surrogate ranked; no GPTZero on candidates) |
| gptzero_rows_repair.json | 09-10 11:18 | free mode: surrogate, 8x3 + repair stage (3 attempts) [6.16] | 7/9 | 5/5 | 12/14 | n/a (surrogate ranked; no GPTZero on candidates) |

- Baseline to beat (from research/24 6.8 and 6.14): own-key mode 7-8 of 9 (22/27 = 81% over c-runs), free mode 5-7 of 9 (61-67%), humans 5/5 in every run.
- Best own-key row here: 8 of 9 (c1). Best free-mode row here: 7 of 9 (surrogate4).
- Candidate-level human share to beat (6.6): 3B base 45%, 7B base 50%, 3B+HIP-300 63% (but more gate failures). The product of candidate-level human rate and gate pass rate is what lifts paragraphs.
- The 'cache window' column counts GPTZero cache files written between consecutive row files, minus the 14 source texts. It reproduces research/24 6.6 exactly for v2 (217, 97 human), 7b (111, 55) and hip3b (112, 71); the c1 window also contains the 6.7 diagnostic run (.kriukow/diag_candidates.json), so its N is an upper bound. Own-key runs only: in free mode the surrogate ranked candidates and GPTZero never saw them.
