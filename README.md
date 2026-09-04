# humanizer

Feature extraction, human reference distributions and a detector-calibration
harness for AI-text humanization research.

This is **Stage 0** of the pipeline in `research/00-SYNTHESIS.md`. It measures
text; it does not yet rewrite it. Everything here is grounded in the 17 research
reports in `research/`, and the code cites them where a threshold or a design
choice comes from a specific finding.

## Why this exists before any model

The research round produced three results that invalidate the standard
humanizer playbook, and Stage 0 exists to make them measurable:

1. **Burstiness is a band, not a maximum.** Human sentence-length coefficient of
   variation runs 0.42-0.60, and a measured GPT baseline sits at about 0.50,
   inside it. Overshooting to 0.85 is as anomalous as undershooting to 0.30, so
   the scorer penalises both tails.
2. **Better essays are less bursty and more formal.** Across scored student
   essays, variation falls while nominalization, passives and formal
   connectives rise. Naive informalizing lowers the grade, so the findings
   engine reports a grade cost alongside every detector benefit.
3. **A local detector score carries no information about GPTZero on its own.**
   Only the measured joint probability does, which is what the calibration
   module estimates.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest        # 112 tests
```

Core dependencies are numpy, scipy and requests. Heavier pieces are optional
extras: `syntax` (spaCy, for clause-level parsing), `detectors` (torch and
transformers, for local model detectors) and `quality` (LanguageTool).

## Use

```bash
# Full report on one document
humanizer analyze essay.txt --reference data/reference/research-article-stem.json

# Raw feature vector
humanizer features essay.txt --json

# Build a reference distribution from a corpus
humanizer build-reference research-article-stem 'data/raw/pmc/*.txt' \
    --out data/reference/research-article-stem.json

# Best-of-N planning under candidate correlation
humanizer plan --mu 0.6 --target 0.99

# Map local scores onto GPTZero verdicts
humanizer calibrate --scores results.json --target 0.95
```

Fetch pre-LLM human corpora (no credentials needed):

```bash
python scripts/fetch_reference.py pmc --count 300 --out data/raw/pmc
python scripts/fetch_reference.py wikipedia --count 300 --out data/raw/wikipedia
```

## What is implemented

| Module | Purpose |
|---|---|
| `humanizer.text` | Sentence and paragraph segmentation tuned for academic prose (abbreviations, initials, decimals, parenthetical and numeric citations), plus protected spans that must never be edited |
| `humanizer.clean` | Markdown and HTML stripping. Not optional: table rows parse as pseudo-sentences and pushed a test document's CV to 1.95 against a human band of 0.42-0.60 |
| `humanizer.features` | 68 features across four modules: shape, lexical, punctuation, register |
| `humanizer.reference` | Per-genre distributions storing the full feature matrix, with Mahalanobis distance, percentile scoring, covariance-preserving sampling and the two-level persona model |
| `humanizer.detectors` | Detector interface, a transparent heuristic baseline, a cached GPTZero client, and the calibration and best-of-N math |
| `humanizer.eval` | Document report with findings ordered by the conflict matrix |

### Features

**Shape** carries the findings that overturned the folk wisdom: sentence-length
mean, SD and CV; lag-1 and lag-2 autocorrelation; the 1/f spectral exponent;
short and long tail shares; paragraph variation.

**Lexical** covers MTLD, HD-D, MATTR, hapax rate, word length, and a weighted
AI-vocabulary lexicon whose weights are published excess-frequency ratios
(*delves* at 28x, *underscores* at 13.8x, *tapestry* and *camaraderie* around
12x, phrases up to 18x).

**Punctuation** is weighted heavily on purpose. An eight-feature punctuation
profile reached 95% accuracy separating two same-register authors where mean
sentence length managed 69%, and punctuation habits are the most author-stable
family measured (question marks at ICC 0.63, commas 0.44).

**Register** covers passives, nominalization, participial tails, hedges,
boosters, self-mention, tricolons, negative parallelism and paragraph openers,
targeting the Hyland and Biber numbers per discipline.

## Validation

The extractor was checked against independently published figures by building a
reference distribution from 60 Europe PMC open-access articles (2016-2021) and
comparing the measured means to the research targets:

| Feature | Measured | Published target |
|---|---|---|
| Sentence-length CV | 0.46 | 0.42-0.60 |
| Lag-1 autocorrelation | 0.08 | 0.01-0.10 |
| Commas per 1,000 words | 57.4 | 57-65 |
| Passives per 1,000 words | 20.2 | ~18.5 |
| Contractions per 1,000 words | 0.05 | under 1.4 |
| Long-sentence share | 0.22 | 0.18-0.28 |
| Paragraph-length CV | 0.56 | 0.42-0.71 |
| Nominalizations per 1,000 words | 57.6 | 61-72 |
| Short-sentence share | 0.07 | 0.09-0.14 |

Seven of nine land inside the published band.

### The spaCy backend

Enabling the optional parser (`pip install -e ".[syntax]"` then
`python -m spacy download en_core_web_sm`) adds 32 clause-level and phrasal
features and materially improves the two that were out of band. Measured over
25 of the same articles:

| Feature | Parsed | Published target |
|---|---|---|
| Prepositional postmodifiers per 1,000 | 67.0 | ~68 |
| Passives per 1,000 | 23.8 | ~18.5 (band 12-26) |
| Passive share of finite verbs | 0.30 | ~0.25 |
| Nominalizations per 1,000 | 57.9 | 61-72 |
| Noun-noun premodification per 1,000 | 71.1 | 24-77 by discipline |

The parser wins on precision, not just recall. On "He was tired and she was
excited", the regex backend reports 200 passives per 1,000 words; the parser
correctly reports zero, because those are copulas with adjective complements.

It also reports the clausal and phrasal families separately with a
`syn_phrasal_to_clausal_ratio`, so the central register finding stays visible:
academic complexity is phrasal, not clausal, and adding subordination to sound
scholarly moves text toward conversation.

The best-of-N math reproduces the published figures exactly: at a per-candidate
success rate of 0.6 with N=8, the pass rate is 99.93% at zero correlation,
97.5% at 0.2 and 87.3% at 0.5, and a correlation of 0.5 makes a 99% target
unreachable at any N.

## Not yet implemented

- **Local model detectors.** The interface is ready; the transformers-backed
  implementations are not written.
- **Live GPTZero calibration.** The client and the math are done, but nobody
  has bought the labels yet. This is the highest-value next step, because
  nothing downstream is meaningful without that curve.
- **Quality scoring.** No essay scorer yet. Do not use a frontier model
  zero-shot for this: it scores near-random on some prompts, well below novice
  human graders.
- **Any rewriting.** Stage 0 measures only.

## Layout

```
src/humanizer/
  text.py            segmentation and protected spans
  clean.py           markup stripping
  features/          shape, lexical, punctuation, register, ai_lexicon
  reference/         per-genre distributions, distance, sampling
  detectors/         base, heuristic, gptzero, calibration
  eval/              document report and findings
  cli.py
scripts/fetch_reference.py
research/            17 reports; start with 00-SYNTHESIS.md
tests/               112 tests
```

## A note on scope

Detector false positives fall hardest on non-native English writers, at 37-61%
in published studies. The same feature pipeline that supports humanizing also
supports a transparent "why does this read as AI" explainer, and that is worth
keeping in scope. Some things no text tool can address: typing replay,
document version history and other provenance signals sit outside the text
entirely, and users should be told so plainly.
