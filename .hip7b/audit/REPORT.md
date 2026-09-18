# Audit of the 7B HIP adapter training data (2026-09-10)

Scope: `.hiplora/pairs.jsonl` (1,500 pairs) as consumed by `.hip7b/build_data.py`
(628 kept: 593 train / 35 valid). CPU only; no model loaded, no network.
Scripts and raw numbers in this directory: `profile.py` (per-row metrics,
`rows.jsonl`, `summary.json`, `contamination.json`, `hand_review_sample.md`),
`compare.py` (metrics of any built data directory). The proposed builder is
`.hip7b/build_data_v2.py`, output in `.hip7b/data_v2/`.

## Findings up front

1. **63% of the kept training targets fail the product's own evade gate against
   their draft** (`_invariant_failures` with the row's NOTES as facts, ratio
   0.55-1.45, overlap 0.25): 274 entity failures, 174 numbers, 92 quotations. The
   adapter is being taught the exact behaviour the gate rejects, which is the
   mechanism behind the 3B drift in research/24 §6.6 ("more human candidates,
   fewer gate passers"). Half of those are the Qwen-3B draft's fault, not the
   target's: the paraphraser expands acronyms ("GSEA" becomes "Gene Set
   Enrichment Analysis"), invents dates ("May 15, 1962" for "May 1962",
   "October 35"), adds parentheticals ("equivalent to approximately $228 million
   in 2024") and repeats years. Whatever the cause, the pair teaches deletion of
   a source particular.
2. **An estimated 161 of 593 train rows (27%) exceed train.py's 1536-token cap.**
   The exemplar block alone is 650-750 tokens. `MaskedCompletions.process`
   truncates the tail, which is the HUMAN completion, the only part carrying
   loss. Those rows train on a cut-off paragraph or on nothing.
3. **22.5% of kept rows have a number or name in the target that neither the
   draft nor the NOTES license** (`new_specifics(draft, human, facts=notes)` non-empty).
   The NOTES builder covers numbers and mid-sentence capitalised runs only, caps
   at five items, locates clauses by substring, and misses repeated uses. The
   product will refuse exactly these completions.
4. **The PMC scrape carries citation residue that the current filter misses:**
   29% of kept rows (34% pmc2, 39% pmc, 3% wiki) have at least one of "shown
   in.", "( and, and)", "( a)", "(Min et al. 2000;)", "&gt;", bracket
   citations, a URL, a lowercase start or no terminal punctuation. 98 targets
   begin with a glued section heading ("Purification High thorium
   concentrations...", "Soft parts Females of L. canarium..."). 6% of drafts are
   markdown or meta text ("### Void Fraction...", "The passage outlines...").
5. **The validation split leaks documents.** 30 of 31 validation documents also
   have paragraphs in train (split is a hash of the paragraph text). The
   3B round's "memorises past step 200" reading was made on the same kind of
   split.
6. **Wiki is the better source on every quality axis and is under-used.**
   Residue 3% vs 34-39%, FRE 41 vs 31, 2.0 NOTES items vs 1.0, and 7 of 7 wiki
   targets in the hand review were prose a product should imitate against about
   12 of 23 PMC targets. It is 20% of kept rows because 87 of 236 wiki
   paragraphs were dropped for length alone.
7. **Contamination: clean.** No 60-character shingle of any training target or
   draft (normalised whitespace and case) appears in any of the 14 bench texts or
   the 9 facts blocks; no bench PMC id appears among the 208 source documents.

## 1. Quantitative profile

All numbers from `profile.py` over the 1,500 pairs after whitespace
normalisation, using `pipeline.content_overlap`, `pipeline.flesch_reading_ease`,
`pipeline.new_specifics`, `pipeline._invariant_failures` and
`kriukow.structure_report`. "Evade gate ok" means the human target would pass
`_gate_candidate` against its own draft with the row's NOTES as `facts`.

| metric (mean, median in brackets) | kept 628 | dropped 872 | kept pmc 141 | kept pmc2 360 | kept wiki 127 |
|---|---|---|---|---|---|
| human words | 143 (131) | 170 (148) | 130 (121) | 132 (123) | 188 (188) |
| draft words | 151 (140) | 197 (179) | 137 | 140 | 197 |
| ratio human/draft | 0.95 (0.95) | 0.90 (0.95) | 0.96 | 0.95 | 0.96 |
| share human shorter than draft | 66% | 60% | 62% | 69% | 64% |
| content overlap | 0.65 (0.65) | 0.63 (0.64) | 0.65 | 0.64 | 0.66 |
| FRE human | 33.2 (31.7) | 23.3 (23.6) | 30.9 | 31.2 | 41.4 |
| FRE draft | 26.6 (25.7) | 18.4 (17.6) | 24.4 | 24.1 | 36.2 |
| share FRE human below 30 (band A) | 44% | 65% | 53% | 49% | 21% |
| sentence-length CV human | 0.38 (0.36) | 0.37 (0.35) | 0.36 | 0.38 | 0.39 |
| sentence-length CV draft | 0.35 (0.34) | 0.37 (0.34) | 0.33 | 0.35 | 0.35 |
| sentences per target | 6.4 (6) | 7.2 (6) | 5.9 | 5.9 | 8.3 |
| shortest / longest sentence (words) | 12 / 37 | 13 / 40 | 13 / 35 | 12 / 37 | 11 / 39 |
| formal connective openers per target | 0.27 | 0.76 | 0.33 | 0.31 | 0.11 |
| kriukow structure penalty human / draft | 0.070 / 0.073 | 0.067 / 0.070 | 0.069 / 0.073 | 0.073 / 0.075 | 0.063 / 0.067 |
| hedges per 1k human / draft | 9.2 / 10.2 | 9.9 / 10.3 | 8.1 / 10.2 | 9.6 / 10.3 | 9.4 / 9.9 |
| share opening on The/This/It | 0.38 | 0.36 | 0.37 | 0.41 | 0.33 |
| NOTES items per row (mean) | 1.19 | 1.27 | 0.92 | 1.02 | 1.96 |
| share "(none: use only what the draft already says)" | 36.6% | 38.1% | 44.0% | 41.1% | 15.7% |
| share with unlicensed specifics despite NOTES | 22.5% | 17.0% | 19.1% | 20.8% | 30.7% |
| share passing the evade gate vs own draft | **36.9%** | 28.7% | 41.1% | 38.3% | 28.3% |
| share with any residue flag | **29%** | 53% | 39% | 34% | **3%** |

Drop reasons in `build_data.py`: fre 271 (270 of them FRE below 15), ratio 178,
human_length 152 (87 of them wiki over 260 words), tokenised_spacing 138,
connectives 110, citation_marks 10, overlap 8, closer 5.

Invariant failures among the 628 kept, by direction:

| kind | draft item missing from target | target item not licensed by NOTES |
|---|---|---|
| entities | 274 (strict verbatim; 187 when a token of the name must survive) | n/a (gate allows new names) |
| numbers | 106 | 68 |
| quotations | 69 | 23 |
| citations | 11 | 10 |
| dates | 7 | 2 |

Other structural numbers: 192 documents, up to 8 pairs from one document (22
documents contribute 6 or more). 168 of 745 NOTES items (23%) are 80% or more
of a whole target sentence, i.e. the NOTES hand the model the sentence. Mean
NOTES item 11.7 words. Bench inputs sit at FRE -1 to 19 (mean 13), band A; the
kept targets sit at 33, the drafts at 27, and the draft-to-target move is +6.6
FRE points (plainer), which matches research/24 §6.4 ("shorter, plainer").
Targets are shorter than their draft in 66% of rows (mean ratio 0.95); the
GPTZero-human candidates in §6.4 ran at ratio about 0.85 and overlap 0.41
against 0.65 here.

## 2. Hand review (30 kept, 15 dropped; `hand_review_sample.md`, seed 7)

Kept, as writing a product should imitate: K2 (Sylvania 300), K3
(Segnosaurus), K5 (Amazing Stories), K7 (New York Dolls), K8 (SVASD repair,
tighter than its draft), K11 (slow loris), K18 (tufted jay), K19, K22, K26,
K30: 11 of 30. All seven wiki targets are in this group (K1 would be but for
its glued heading).

Flags, with counts over the 30:

* Citation or figure residue, 8: K1 "Soft parts Females of" (glued heading);
  K10 "( A)", "( B)"; K12 "( a)", "( b and)"; K13 "As depicted in,"; K14 "is
  provided in." and "&gt;50%"; K16 "are shown in." twice and "(Min et al.
  2000;)"; K20 "( and, and)"; K27 "In, representative images". K15 has
  "&amp;".
* Non-native phrasing or the draft is the better paragraph, 8: K4 "have been
  found to play as biomarkers"; K9 "diagnosis biomarkers", "Increasing studies
  have revealed"; K23 "Prior to a line of loss-of-function assays", "we also
  reached a conclusion that"; K24 "at the end of the study period that is at
  the end of the 6 months" (the draft is cleaner); K25 "Total 506 participants
  were concerned/worried", "feared of being put up in quarantine", "(54.25)",
  "worried that when the lockdown would end" (draft better); K27 "On the
  opposite,"; K29 "it might be worth to continue"; K12 "naked eyes
  observation". All are pmc/pmc2.
* Bad drafts (markdown, meta text, hallucinated expansions), 8: K4 and K17
  "The passage outlines/discusses" plus invented expansions ("Yankee Baseball
  Thrower-Left Quadrant" for YBT-LQ, "Pancarinacell Adenocarcinoma" for
  PAAD in K23); K5 bullet dashes; K6 glued heading and invented
  "Progerin-Genome Chaos Complexes"; K12 "---"; K13 "### ... - " list; K15
  LaTeX "\( S_i \)".
* NOTES that give away a whole sentence, 6: K3 (first clause verbatim), K10,
  K13 (three near-complete sentences), K16 (the residue sentence itself), K23,
  K28 (four items, most of the methods paragraph).
* NOTES missing a particular the target uses, 5 of 30 (K2 "300-lap", K15
  "12", "Miller", K19 "19", K25 "19", K26 "Omicron's", K28 "2", "Wonatech",
  "Republic of Korea"). Across all 628 kept rows it is 141 (22.5%).
* Methods boilerplate rather than argued prose, 4: K12, K13, K14, K28. Reads
  as a table in sentences; teaches nothing about pacing.
* Truncated paragraph: none in the sample (10 of 628 kept end without terminal
  punctuation).
* Tokenised spacing: none in the kept sample (the current filter catches " ."
  and " ,"); D1, D6, D14 among the dropped show it, and D5 shows " ." that the
  filter did not catch because the row was dropped for FRE first.

Dropped: 9 of 15 were correct drops (D1 spacing plus a meta sentence in the
draft; D6, D14 spacing; D10 markdown list draft and 283-word target; D11, D12,
D13 non-native dense prose at FRE 1-14, D12 with sentence CV 0.04; D15
"Supplementary Table 2"; D2 two connective openers). 6 were losses: D3 (ratio
0.61) and D7 (ratio 0.60) are exactly the compression GPTZero rewards and are
inside the product's evade floor of 0.55; D4 (El Greco, 314 words) is an
excellent target dropped for length only; D5, D8, D9 are sound dense
paragraphs at FRE 7-12 dropped by the absolute FRE floor while the bench
inputs sit at FRE 13.

## 3. Contamination

`contamination.json`: 60-character shingles of every training target and
draft (lowercased, whitespace collapsed; 3,000 texts) against the 14 bench
texts in `.kriukow/gptzero_texts.json` and the 9 facts blocks in
`gptzero_facts.json`: **0 hits**. Bench PMC ids (8719469, 8730340, 8752981,
8760182, 8762392) do not occur among the 208 source documents (they come from
`data/raw/pmc`, whose first 14 files `build_pairs.py` excluded; pmc2 is a
separate download). The same check on `.hip7b/data_v2`: 0 hits.

## 4. Recommendations, ranked by expected effect

Effect (a) is GPTZero-human candidates that also pass the fidelity gates;
effect (b) is prose quality.

1. **Keep only pairs whose target passes the product's gate against its draft
   with the NOTES as facts** (a, large). Evidence: 63% of kept rows fail it;
   research/24 §6.6 shows the fine-tune raising the human rate (45% to 63%)
   while paragraph passes fell because human candidates died at the gates. Use
   the gate's rules for numbers, citations and dates; for entities require
   that some token of each draft name survive (the strict verbatim rule fails
   on the paraphraser's acronym expansions, 274 vs 187); for quotations accept
   the draft's quote surviving inside a longer target quote. This drops the
   pairs where Qwen-3B hallucinated a particular into the draft, which is the
   right outcome: at inference the source is the user's text.
2. **Fit the token budget** (a and b, large, and free). Either pass
   `--max-seq-length 2048` to train.py (raises 296 to about 310 unique pairs
   in v2 and removes the truncation entirely) or, as v2 does, measure each row
   with the cached Qwen tokenizer, fall back to the shortest exemplar, and drop
   what still does not fit. At 1536 the current build truncates about 27% of
   completions.
3. **Rebuild NOTES to cover every particular the target adds, then reject the
   row if `new_specifics(draft, human, facts=notes)` is still non-empty** (a,
   large). Cover quotations, citations and repeated uses, not only numbers and
   entities; locate the needle by exact token (the substring match put "2" in
   "2019" and "Louis" in "Louisiana"); cap items at six and drop rows needing
   more (users write 3-6 bullets); keep windows at 12 words so NOTES do not
   hand over sentences (23% of v1 items were 80%+ of a sentence; 17% in v2,
   still worth tightening). Keep the no-notes rows: they are 37% of the data,
   they pass the gate at 58% against 25% for rows with notes, and they are the
   only rows teaching "add nothing" for the product's most common case.
4. **Clean the scrape and the drafts** (b, medium; a, small). Unescape HTML
   entities; repair " ." and marker-only parentheses; drop targets with
   "shown in.", bracket citations, URLs, lowercase starts, unterminated ends;
   strip the glued section headings by looking the paragraph up in the raw
   file (98 rows); strip markdown and meta sentences from drafts and drop
   LaTeX drafts. This is the 29% residue rate and the "draft is not the
   product's input distribution" problem. research/22 says human imperfection
   is punctuation habit, not scrape damage; nothing here should be learned.
5. **Shift the source mix toward wiki and weight it** (b, medium; a, medium).
   Wiki has 3% residue, FRE 41, twice the NOTES density, and 7 of 7 good
   targets in review; research/21 §10 wants specifics, demonstrative
   cohesion and closings on facts, which the wiki paragraphs have and the PMC
   methods paragraphs do not. Only about 70 wiki pairs survive any sane filter
   (the paraphraser hallucinates most on narrative wiki text), so weight them
   (write twice) rather than discard PMC to reach a 40% share. Longer term,
   the cheapest quality gain is more wiki paragraphs paraphrased with the
   markdown-free prompt.
6. **Loosen ratio downward, tighten upward, and drop the absolute FRE floor
   for a relative one** (a, medium). Ratio 0.60-1.25: research/23 §0 and §7 say
   the passers do not expand, and research/24 §6.4 says the human-rated
   candidates were shorter; the v1 floor of 0.70 discarded D3 and D7. FRE: keep
   targets at or above 10 and not more than 8 points denser than their draft
   (the bench inputs are FRE -1 to 19, so an absolute floor of 15 removes the
   band the product runs in; the direction that matters is "plainer than the
   draft"). Add a sentence-length CV floor of 0.15 (research/21 band A floor
   0.18; D12 at 0.04 is a list).
7. **Split train/valid by document** (measurement, not effect). 30 of 31
   validation documents in v1 are also in train. The 3B round's "memorises
   past step 200" and the r1 run's validation curve are read against leaked
   paragraphs.
8. **Do not trim targets.** A target cut at a sentence boundary no longer pairs
   with its draft and the row would fail ratio and overlap; the length problem
   is a token-budget problem (item 2), not a target problem.
9. Smaller: cap pairs per document (6; the gate filter already spreads them,
   4 to 8 makes little difference, 333 vs 345); keep the one-connective and
   no-summary-closer rules (research/21 §10.1 checks 6 and 15); consider
   excluding methods-section boilerplate by a "reagent/instrument" lexicon
   (K12, K13, K14, K28), which the current data has at roughly 15% of PMC rows
   and which teaches nothing about pacing.

What the current builder ignores from the research and v2 does not add
either, because it needs new data rather than a filter: the "verdict sentence
after the long one" shape and closings on a fact or image (research/21
findings 2 and 8) are properties of essayistic prose; PMC has them rarely,
wiki sometimes. A third source in band C/D (quality long-form) would be the
next data lever.

## 5. build_data_v2.py

`/Users/aniket.bhattacharjee/humanizer/.hip7b/build_data_v2.py`, same CLI shape as
`build_data.py` plus `--per-doc`, `--max-notes`, `--max-seq-length`,
`--min/max-ratio`, `--min/max-overlap`, `--min-fre`, `--max-fre-drop`,
`--min-cv`, `--wiki-repeat`. Output `.hip7b/data_v2/`. The prompt is built by
the same concatenation (header, one exemplar triple, DRAFT, NOTES, HUMAN) and
was checked byte-identical to build_data.py's for all three exemplars; the
completion is the target plus "\n\n" as before. Nothing outside `.hip7b/audit/`,
`.hip7b/build_data_v2.py` and `.hip7b/data_v2/` was written.

Run (defaults: per-doc 6, wiki share 0.25, wiki repeat 2, max-seq 1536):

```
kept 399 of 1500; dropped: citation_residue 165, closer 5, connectives 123,
draft_entity_lost 191, draft_markdown_or_meta 36, draft_particular_lost 150,
flat_cv 26, fre 191, lowercase_start 15, overlap 32, ratio 118,
too_long_for_seq 7, unlicensed_specific 23, unterminated 19
(98 glued headings stripped and kept)
after per-doc cap 6 and balancing: 296 unique (wiki 74, pmc 222); 163 docs
train 344 rows (70 wiki pairs x2 + 204 pmc), valid 22, no-notes 133,
longest row 1532 tokens, 0 validation documents in train
```

Comparison (`compare.py .hip7b/data .hip7b/data_v2`):

| | v1 (628) | v2 (363 rows, 296 unique) |
|---|---|---|
| passes evade gate vs own draft (strict gate, notes as facts) | 36.9% | **68.6%** (rest: 32 strict-entity, 22 quotation-multiset, both relaxed by design) |
| unlicensed specifics after NOTES | 22.5% | **0%** |
| rows over 1536 tokens | ~27% | **0** |
| residue / HTML entities / markdown drafts | 29% / 7.5% / 6.2% | 0 / 0 / 0 |
| wiki share of rows | 20% | 39% |
| valid docs also in train | 30 of 31 | 0 of 18 |
| FRE human / draft | 33.2 / 26.6 | 34.0 / 27.2 |
| ratio, overlap, CV | 0.95, 0.65, 0.38 | 0.98, 0.66, 0.37 |
| NOTES items 80%+ of a sentence | 23% | 17% |
| "(none...)" rows | 36.6% | 42.1% |

Other configurations measured (unique pairs): wiki share 0 gives 345, 0.30
gives 213, 0.40 gives 159; per-doc 4 to 8 moves the count by 12;
`--max-seq-length 2048` adds about 14 pairs and lifts the wiki mean length.

The cost is size: 296 unique pairs against 628. The 3B HIP round trained on
216 and moved the per-candidate human rate from 45% to 63%; the loss here is
mostly rows that taught gate failure or residue. If more rows are wanted,
the first things to relax are `--max-wiki-share 0` (345) and the strict
`draft_entity_lost` rule (191 rows, of which perhaps a third are paraphraser
formatting rather than real name loss), not the residue or token filters.
