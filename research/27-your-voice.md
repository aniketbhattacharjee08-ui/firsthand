# Your voice: per-user samples as exemplars and a fingerprint among the gate passers

**Bottom line.** Per-user samples as few-shot exemplars plus a stylometric
fingerprint among the gate passers cost nothing on the detector (7 of 9
human, voiced and plain alike, surrogate judged) but, measured against the
paragraphs the authors actually wrote, do not yet bring the rewrite closer
to the author than the plain pipeline does. The gate is never relaxed, a
voiced round that produces nothing clean falls back to the stock
exemplars, and the feature ships opt-in with that sentence on its page.
The way to a real voice is a per-user adapter trained on the pairs the
store already collects (section 4). The GPTZero key had expired on bench
day, so every detector number here is the free surrogate's.

---

## 1. What is built

Shipped on branch `humanize-engines-and-load-lock` (commits 6327a21,
19dbae4, ea2dde6, a7d5573). Wire contract: `web/VOICE.md`.

**Samples.** A signed-in user adds pieces of their own writing at `/voice`
(80 to 2,500 words; a file upload goes through the document extract
route). Each sample is scored by the free detector and refused if it reads
as AI (threshold 0.5): a model-written sample would teach the model its own
habits, which are the ones detectors learn. Uploads over 450 words are
split at paragraph boundaries into parts, because exemplars have to fit
the served model's 6,144-token window three at a time and because the
per-paragraph picker matches paragraph-sized samples. Guests have no voice;
the data goes with the account on deletion.

**Twins.** The "draft" half of each pair is made once, in the background,
by the *plain* base checkpoint continuing the stock exemplar pattern
backwards: three (HUMAN, DRAFT) pairs from `exemplars.FREEFORM_SHOTS`, then
the sample as HUMAN and an open DRAFT slot. No instruct model is involved,
because production (Modal vLLM) serves only the base and its adapters.
Measured on Qwen2.5-7B (4-bit, local): the plain base at temperature 0.8
keeps 0.60 to 0.73 of the sample's words and reads as a generic assistant
("The library was closed on Tuesday, prompting a walk. The rain did not
deter me."); the HIP-adapter checkpoint echoes the sample (0.80 to 0.95
retained), so the adapter is left off for this one call. A twin keeping
more than 0.85 of the words is refused and retried once at 1.0.

**Fingerprint.** `build_profile` reduces the samples to per-feature mean
and spread over 28 features chosen from research/16: function-word and
punctuation rates, contraction rate, hedges and boosters, formal versus
informal connectives, sentence-initial coordinators and other openers,
sentence-type shares, first-person rate, long-word share, MATTR, and the
sentence-length *spread* (its mean carries weight 0.3; Mosteller and
Wallace's Hamilton and Madison differ by 0.04 words). Plus a 600-entry
character-trigram profile. `profile_distance` is the mean of a weighted
clipped z-score term and the trigram cosine distance, in [0, 1].

**Ranking.** `_select_scored`: without a voice, the argmin of the proxy
probability (the `evade` objective, unchanged). With one, the pool is the
candidates at or under `pass_threshold` (0.15 under the surrogate, 0.5
under GPTZero); among them `p + weight x distance` is minimised, weight
0.25 / 0.5 / 1.0 for light / balanced / strong. With no candidate under the
threshold the voice is ignored and the plain argmin applies. So a voice can
only choose among human-reading rewrites.

**Feedback.** Two chips under the verdict. A thumbs up stores the changed
paragraphs as (draft, rewrite) pairs in a separate liked pile and raises
the weight of the samples the prompts actually leaned on (+0.25, clipped to
[0.2, 2.0]); a thumbs down lowers it (-0.35) and removes the pile entry. A
second vote on the same run undoes exactly the earlier move. Liked
rewrites never join the samples: they fill at most one exemplar slot, and
only while fewer than three real samples are ready. A run in the voice
that still reads as AI offers "Run again with less voice" (strength light,
this run only), then "Run again without my voice".

## 2. The bench

`.voice/bench.py`. Four PMC articles stand in for four authors (one
article, one author): the first four usable paragraphs are the samples,
the fifth is held out. The 9 AI bench paragraphs from `.kriukow` are each
rewritten twice through the shipped pipeline in the product's own mode
(8 candidates x 2 rounds, the surrogate ranking, facts supplied): once
plain, once in the voice of author `i mod 4`. Each output is placed by
fingerprint distance against all four profiles.

Three numbers:

- **Detector.** Human verdicts on the output, voice off versus on. Pass
  condition: on is no worse than off.
- **Authorship.** The share of outputs whose nearest profile is the
  author whose voice was used, against chance (25%) and against the
  voice-off outputs placed the same way.
- **Ceiling.** The held-out real paragraphs placed on their own author:
  what the fingerprint can do on real writing at all.

### 2.1 Cross-register bench (2026-10-11, surrogate judge, 8x2, free mode)

Four PMC authors, four samples each (11 of 16 twins accepted under the
first, word-based echo check; the bigram check that replaced it accepts
more). Nine AI bench paragraphs, each rewritten plain and in the voice of
author `i mod 4`.

| condition | human after (surrogate) | output nearest its author | mean distance to own author | gate fallbacks |
|---|---|---|---|---|
| voice off | 7 of 9 | 4 of 9 | 0.409 | 0 |
| voice on | 7 of 9 | 3 of 9 | 0.439 | 1 |

Chance for the authorship column is 1 in 4. Held-out real paragraphs placed
on their own author: 3 of 4, so the fingerprint itself carries signal.

Per paragraph:

| paragraph | author | off: p after | off: nearest | on: p after | on: nearest | on: gate passers |
|---|---|---|---|---|---|---|
| ai-explainer-1 | PMC8786337 | 0.918 | other | 0.938 | own | 4 of 16 |
| ai-explainer-2 | PMC8802128 | 0.169 | own | 0.093 | own | 6 of 16 |
| ai-explainer-3 | PMC8809424 | 0.038 | other | 0.011 | other | 3 of 8 |
| ai-gen-microplastics | PMC8762392 | 0.002 | other | 0.002 | other | 7 of 8 |
| ai-gen-antibiotic | PMC8786337 | 0.002 | other | 0.011 | own | 5 of 8 |
| ai-gen-urbanheat | PMC8802128 | 0.006 | own | 0.004 | other, every candidate failed a gate | 0 of 8 |
| ai-gen-sleep | PMC8809424 | 0.003 | other | 0.003 | other | 9 of 16 |
| ai-gen-supplychain | PMC8762392 | 0.037 | own | 0.384 | other | 8 of 16 |
| ai-gen-teaching | PMC8786337 | 0.028 | own | 0.019 | other | 6 of 8 |

Two things the stored candidates show (`.voice/results/voice-8x2-a4-s4-base.json`):

1. **The ranking tier had nothing to choose from.** Under the surrogate's
   0.15 line a paragraph has one or two passers; in 11 of 13 scored rows
   the shipped candidate was already the closest passer, and re-ranking the
   same candidates at weights 0.5, 1, 2 and 5 picks the same one every
   time. Tier 2 cannot act until there is a pool.
2. **The exemplars pull content, not only style.** With the samples in the
   prompt, fewer candidates pass the meaning gates (4 and 6 of 16 on the
   explainers against 11 off), and on the urban-heat paragraph every one of
   eight drifted or invented a number; the ship-anyway rule sent a bad
   rewrite. That is now fixed (commit a7cb808): a voiced round with no
   clean candidate marks the paragraph and the next round rewrites it with
   the stock exemplars; a clean candidate from that round replaces the
   fallback. The page says so in one sentence.

This bench is deliberately hard and partly unfair: academic article
paragraphs steering explainer and essay drafts on other subjects, judged by
a fingerprint on a single paragraph. The own-text bench below is the use
case the feature exists for.

### 2.2 Own-text bench

Four authors, two held-out paragraphs each, every draft the plain base's
generic twin of the paragraph (15 of 16 sample twins accepted under the
bigram echo check). The surrogate already reads every twin as human, so
this bench answers the authorship question only. "Similarity to original"
is trigram cosine similarity between the rewrite and the paragraph the
author actually wrote; the twin itself scores 0.887, so a rewrite can only
lose ground, and the question is how much.

| condition | nearest its author | mean distance to own author | similarity to the original | content overlap with the original | gate fallbacks |
|---|---|---|---|---|---|
| voice off | 6 of 8 | 0.330 | 0.856 | 0.632 | 3 |
| voice on | 5 of 8 | 0.354 | 0.820 | 0.545 | 2 |

| held-out paragraph | off: similarity to original | on: similarity | off: overlap | on: overlap | off: nearest | on: nearest | on: clean candidates |
|---|---|---|---|---|---|---|---|
| PMC8786337-1 | 0.817 | 0.852 | 0.59 | 0.61 | other | other | 4 of 8 |
| PMC8786337-2 | 0.919 | 0.861 | 0.76 | 0.62 | other | other | 1 of 8 |
| PMC8802128-1 | 0.858 | 0.815 | 0.64 | 0.47 | own | own | 0 of 16 |
| PMC8802128-2 | 0.832 | 0.786 | 0.65 | 0.56 | own | own | 0 of 16 |
| PMC8809424-1 | 0.809 | 0.798 | 0.57 | 0.47 | own | other | 1 of 8 |
| PMC8809424-2 | 0.845 | 0.817 | 0.59 | 0.55 | own | own | 7 of 8 |
| PMC8762392-1 | 0.876 | 0.798 | 0.59 | 0.55 | own | own | 1 of 8 |
| PMC8762392-2 | 0.894 | 0.832 | 0.66 | 0.54 | own | own | 3 of 8 |

Per paragraph the voiced rewrite sits farther from the author's own text in
six of eight cases, ties in one and is closer in one; it keeps fewer of the
original's content words every time. Three of the eight drafts (dense
methods paragraphs with names, numbers and citations) produced no clean
candidate in either condition, so the gates, not the voice, decided them.

### 2.3 Two follow-ups

**The fallback fix, voiced cross-register rows rerun** (commit a7cb808):
7 of 9 human, 4 of 9 on their author, mean distance to own author 0.388,
no gate fallbacks. That is parity with the plain pipeline on both counts
and a little closer on distance; the run before the fix was 7 of 9, 3 of
9, 0.439 with one bad fallback shipped. The difference between those two
voiced runs, identical but for the fix, is also a fair measure of
run-to-run noise at nine paragraphs.

**The plain base for voiced runs, own-text drafts** (`--plain-base`,
7 of the 8 drafts; paired against the first two conditions on the same
paragraphs):

| condition | mean distance to own author | similarity to the original | content overlap | nearest its author | gate fallbacks |
|---|---|---|---|---|---|
| voice off, shipped checkpoint | 0.319 | 0.859 | 0.629 | 5 of 7 | 2 |
| voice on, shipped checkpoint | 0.337 | 0.825 | 0.542 | 4 of 7 | 1 |
| voice on, plain base | 0.311 | 0.854 | 0.664 | 5 of 7 | 1 |

With the plain base the voice reaches parity with the plain pipeline,
no better. The shipped checkpoint (base plus the HIP adapter) copies its
input, which the twin experiment also showed, so with it the exemplars
pull content rather than style and the gates then reject the result. The
plain base is unmeasured on the detector for these runs (every own-text
draft already reads human to the surrogate); the cross-register plain-base
voiced run is PLAINBASE_CROSS_PLACEHOLDER.

## 3. What the numbers say, and what they do not

1. **No detector cost.** On the only bench where the drafts read as AI,
   the voice matches the plain pipeline: 7 of 9 human in every voiced
   run, 7 of 9 plain. The hard gate did its job, and after the fallback
   fix nothing drifted or invented a number was shipped.
2. **No measurable pull toward the author yet.** On the use case the
   feature exists for, the voiced rewrite is not closer to what the
   author actually wrote than the plain rewrite: farther with the shipped
   checkpoint, level with the plain base. The fingerprint does carry
   authorship signal on real text (3 of 4 held-out paragraphs placed
   right), so the shortfall is in generation, not in the measure.
3. **Why the two tiers did not add up.** The ranking tier needs a pool of
   gate-passing candidates to choose among and, under the surrogate's 0.15
   line at 8 x 2, a paragraph has one or two; the shipped candidate was
   already the closest passer in 11 of 13 rows, and the weight is moot.
   The exemplar tier moves a base model, but what it moves is the content
   as much as the style: with the samples in the prompt fewer candidates
   pass the meaning gates, and the survivors are not closer to the author.
4. **The honest product posture, shipped:** the switch starts off, the
   page says what was measured in one sentence, and everything else stays
   (samples, the fingerprint, the votes, the fallback protection) because
   it costs nothing on the detector and because the samples, twins and
   votes are exactly the training set the next tier needs.
5. **What is noise.** Nine and seven paragraphs, one surrogate judge,
   PMC articles as authors. The two voiced cross-register runs, identical
   but for the fallback fix, differ by one authorship hit and 0.05 of
   distance; treat anything smaller than that as nothing.

## 4. What is still owed, and the way forward

1. The same benches with `--judge gptzero` once the key is reset (it
   expired on 2026-10-11). The surrogate agrees with GPTZero on about 92%
   of texts at 0.5 (research/24 §6.11), but the owner's rule is the real
   judge.
2. **Tier 3 is the lever.** The benches say prompting cannot move this
   model's style while the gates hold its content. A rank-8 LoRA per user
   on a few thousand of their words, trained on exactly the (twin, sample)
   pairs the store already keeps, is the next experiment; the serving
   cost (one adapter per user, hot-swapped on Modal) is the real question,
   not the training. Run it on three real users' essays before building
   the serving.
3. A wider pool for the ranking tier: 12 candidates when the voice is
   on, measured on the detector first. Cheap to test, likely small.
4. Real users' samples in the product's registers (essays, emails, cover
   letters); PMC articles are one register and a hard one for the gates.
5. The match percentage on the page is the fingerprint on one paragraph;
   per-author distances in the benches run 0.25 to 0.47 between
   paragraphs of the same author, so differences under about 0.05 mean
   nothing. The page keeps the number because a user can watch it move
   with their thumbs, but the early sentence is the claim.
