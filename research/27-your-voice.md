# Your voice: per-user samples as exemplars and a fingerprint among the gate passers

**Bottom line.** A user's own writing can steer the humanizer without
training anything per user, and without weakening the detector gate. The
mechanism is two-fold: the base model's few-shot block is filled with the
user's own (generic twin, real paragraph) pairs, so the pattern it
continues is that person's; and among the candidates that already pass the
detector gate, the one closest to the user's stylometric fingerprint wins.
The gate is never relaxed. The measurements below are from the first bench
(2026-10-11); the detector half of that bench was judged by the free
surrogate because the GPTZero key had expired, so it is not yet the owner's
required GPTZero number.

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

RESULTS_PLACEHOLDER

## 3. What the numbers say, and what they do not

RESULTS_DISCUSSION_PLACEHOLDER

## 4. What is still owed

1. The same bench with `--judge gptzero` once the key is reset. The
   surrogate agrees with GPTZero on about 92% of texts at 0.5
   (research/24 §6.11), but the owner's rule is the real judge.
2. More authors and real users' samples. PMC articles are one register;
   the product's users write essays, emails and cover letters.
3. The match percentage on the page is the fingerprint's own number on a
   single paragraph, which is noisy; the bench's per-author distances say
   how much to trust it.
4. Tier 3, a per-user LoRA, is not built. Nothing here trains anything.
