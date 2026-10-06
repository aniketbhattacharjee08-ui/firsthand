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

### Memory

The API server keeps model weights resident between requests and bounds each
cache with an environment variable, evicting least-recently-used entries and
returning the memory to the OS: `HUMANIZER_MAX_RESIDENT_MODELS` caps detector
checkpoints (default 3, each 0.5-1.7GB), `HUMANIZER_MAX_RESIDENT_PRETRAINED`
caps paraphraser checkpoints (default 1, each 6-12GB) and
`HUMANIZER_MAX_RESIDENT_LLM` caps MLX language models (default 2, each
1.7-2.2GB). Zero or a negative value means unlimited. `GET /api/models` lists
what is resident with the process RSS, and `POST /api/models/release` unloads
everything without a restart.

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

### Accounts and the sign-in gate

`humanizer serve` puts the humanizer behind a sign-in. The site is three pages:
`/` is the landing page (`web/home.html`), `/signin` and `/signup` are one
account page (`web/auth.html`), and `/app` is the humanizer itself
(`web/index.html`), which redirects to `/signin?next=/app` until the visitor
has an account and is signed in. `/index.html` follows the same rule, so the
app cannot be reached by file name. Every product route (`/api/humanize*`,
`/api/analyze`, `/api/detect`, `/api/plan`, `/api/features`, `/api/models*`)
answers `401 {"error": "sign_in_required"}` without a session; `/api/health`,
`/api/references` and `/api/auth/*` stay public.

```bash
humanizer serve                      # accounts on (the default)
humanizer serve --no-auth            # no sign-in; / serves the humanizer directly
HUMANIZER_AUTH=0 humanizer serve     # same as --no-auth
HUMANIZER_AUTH_DB=/srv/auth.sqlite humanizer serve   # move the account database
```

The JSON API, all same-origin with the `rh_session` cookie (HttpOnly,
SameSite=Lax, Secure over https, 30 days):

    POST /api/auth/signup   {email, password, name?}  -> 201 {ok, user: {email, name}}
                            409 email_exists, 422 invalid_email / invalid_password (< 8 chars)
    POST /api/auth/signin   {email, password}         -> 200 {ok, user}; 401 invalid_credentials
    POST /api/auth/signout                            -> 204, clears the cookie
    GET  /api/auth/me                                 -> {signed_in: true, user} or {signed_in: false}

Accounts live in `data/auth.sqlite` (tables `users` and `sessions`). Passwords
are stored as salted `hashlib.scrypt` hashes (PBKDF2-HMAC-SHA256 on a Python
whose OpenSSL lacks scrypt); the cookie value is a random token that exists
only in the `sessions` table, so signing out or deleting the row revokes it.
There are no new dependencies. The implementation is `humanizer.api.auth`;
the paid tier in `humanizer.billing` keeps its own magic-link login and
builds the server with `auth=False`.

### Passing GPTZero

The rewriting engine that moves GPTZero's verdict is the base-checkpoint path
(`style=freeform`, the default): a 4-bit Qwen2.5-7B base model continues a
few-shot pattern of (draft, author's notes, rewrite), and GPTZero itself picks
among the candidates. Everything else measured here (instruct rewrites, the
register LoRA, deterministic edits) leaves GPTZero at 1.000 however natural it
reads; see research/24 §6. Three settings matter:

```bash
export GPTZERO_API_KEY=...        # the only judge: verdicts, candidate ranking, the page's number
export HUMANIZER_BASE_MODEL=mlx-community/Qwen2.5-3B-4bit   # the default; 7B measured worse on the bench
export HUMANIZER_BASE_ADAPTER_RETRY=/path/to/hip-adapter   # optional LoRA used from round 3 for stubborn paragraphs
```

With `GPTZERO_API_KEY` set, GPTZero is the judge and the candidate scorer
(since 2026-09-22): eight candidates a paragraph, two rounds, each candidate
scored by GPTZero, about $1 of the key owner's credits per paragraph. Every
paragraph ships a rewrite; when no candidate passes every gate the least-bad
one is taken and the reason is reported, and a draft the scanner already calls
human is rewritten all the same. `HUMANIZER_PROXY=surrogate` restores the
free mode: a RoBERTa classifier fine-tuned in this project on GPTZero's own
verdicts (`.surrogate/train.py`, weights in
`data/cache/models/gptzero-surrogate`), which agrees with GPTZero on 82% of
held-out texts (AUROC 0.91) but only about 75% of live candidates, and which
ranked the 4 x 1 build that shipped AI-rated rewrites. Without a key the
surrogate is the default. End users can also paste their own key under
Options in the web app: then GPTZero ranks their request on their credits.
Benches:

```bash
python scripts/gptzero_bench.py paragraphs.json --facts facts.json   # GPTZero judges and ranks when the key is set
.venv/bin/python .hip7b/bench.py --mode ownkey --adapter data/adapters/hip7b-r4-it200
```

Every GPTZero response is cached under `data/cache/gptzero`, and each bench
enlarges the labelled set the surrogate is trained on; retrain with
`python .surrogate/train.py 4 both`. The older local checkpoints (desklib,
fakespot, academic, RADAR) remain reachable by name through `/api/detect` for
research; they were measured rating GPTZero-human rewrites as AI and are not
shown to a writer.

Give the rewrite the author's facts (the `facts` field of `/api/humanize/llm`,
the "Facts you can vouch for" box in the web UI). A rewrite may add a number,
date, quotation or citation only if it appears there; anything else new is
refused and named in `summary.unverified_specifics`. Bench the whole loop with:

```bash
python scripts/gptzero_bench.py paragraphs.json --facts facts.json
```

Measured 2026-09-07 on the nine-paragraph bench: 7 of 9 AI paragraphs read as
human on GPTZero after rewriting, 0 of 5 human paragraphs harmed.

When a paragraph still fails after the rounds, the pipeline does not just
return it: a `repair` stage (`humanize/repair.py`) reads why its candidates
were rejected (invented figures, content drift, length, or passers that all
read as AI), says so in a sentence, picks one corrective action from a
ladder (restrict the particulars, raise fidelity, match length, change the
sampling distribution, rewrite the worst sentences, split the paragraph, or
write a faithful bridge draft first) and retries, up to `repair_attempts`
times (default 3, `HUMANIZER_REPAIR_ATTEMPTS`; 0 disables it) inside its own
time budget. Each attempt is logged on the paragraph as `repairs` and shown
in the web app under Details as "What it tried"; a repair is accepted only if
it passes every gate and beats the current best on the judge. Measured
2026-09-10 in free mode: 7 of 9 AI paragraphs human on
GPTZero, 5 of 5 human controls unharmed, and 2 of the 7 were paragraphs the
repair stage rescued after three rounds had failed them. See research/24 §6.16.

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
