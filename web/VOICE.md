# Your voice: the API contract the front end builds against

Feature: a signed-in user uploads a few pieces of their own writing. The
humanizer then (1) shows the base model the user's own (AI draft, their real
paragraph) pairs instead of the stock exemplars, and (2) among candidates
that already pass the detector gate, ranks the one closest to the user's
style fingerprint first. Thumbs up/down after a run teach the ranker and
reweight which samples get used. Liked outputs are kept in a separate pile
and never mixed with the user's real samples.

Guests cannot use it: samples are personal data, so every write needs an
account. Every route below answers JSON. 401 `{error:"login_required"}` on a
write without a session.

## Routes

GET /api/voice
  Always 200. Signed out:
    {"signed_in": false, "enabled": false, "ready": 0, "samples": [], "liked": [],
     "profile": null, "limits": {...}}
  Signed in:
    {"signed_in": true,
     "enabled": true|false,          # the user's switch, stored server side
     "strength": "light"|"balanced"|"strong",
     "ready": 3,                     # samples with status "ready"
     "samples": [Sample, ...],       # newest first
     "liked": [Liked, ...],          # thumbs-up outputs, newest first
     "profile": Profile|null,        # null until one sample is ready
     "llm_available": true|false,    # false means new samples stay "preparing"
     "limits": {"min_words": 80, "max_words": 2500, "max_samples": 12, "part_words": 450, "min_ready": 1}}

Sample = {"id": "s_...", "title": "History essay", "n_words": 412,
          "queue_position": 0|N|null,   # while "preparing": 0 = being prepared now, N = N ahead
          "excerpt": "first 160 chars...",
          "status": "ready"|"preparing"|"failed",
          "detail": "one sentence on why it failed, or ''",
          "ai_probability": 0.08|null,   # the free surrogate's reading at upload
          "weight": 1.0,                 # drifts up/down with thumbs
          "uses": 4, "created": 1760000000.0}
Liked = {"id": "l_...", "excerpt": "...", "n_words": 90, "created": ...}
Profile = {"n_samples": 3, "n_words": 1300,
           "traits": [{"key": "sent_len_mean", "label": "Sentence length",
                       "value": "17 words on average, with a wide spread",
                       "note": "Short and long sentences side by side."}, ...]}

POST /api/voice/samples   {"title": "optional", "text": "..."}
  201 {"sample": Sample, "samples": [Sample, ...], "n_parts": 1}
  An upload over limits.part_words (450) is split at paragraph boundaries
  into parts of about that size, each its own sample titled "Title (part k)";
  `samples` lists them all and `sample` is the first.
  422 {"error": "too_short"|"too_long"|"too_many"|"duplicate"|"reads_as_ai",
       "detail": "a sentence the page can show as is"}
  Uploading runs the free detector on the text; if it reads as AI the sample
  is refused ("reads_as_ai"). Then a background job asks the model to write a
  generic AI version of the sample (the "draft" half of the pair); while it
  runs status is "preparing". Poll GET /api/voice every few seconds while any
  sample is "preparing".

POST /api/voice/samples/{id}/prepare     retry a failed preparation -> {"sample": Sample}
DELETE /api/voice/samples/{id}           -> {"deleted": true}
DELETE /api/voice/liked/{id}             -> {"deleted": true}

POST /api/voice/settings  {"enabled": bool, "strength": "light"|"balanced"|"strong"}
  (either field optional) -> {"enabled": ..., "strength": ...}

POST /api/voice/feedback
  {"vote": "up"|"down", "original": "<the draft paragraph(s)>", "text": "<the rewrite>",
   "pairs": [{"original": "<one draft paragraph>", "text": "<its rewrite>"}, ...],
   "run_id": "...", "sample_ids": ["s_..", ...]}
  `pairs` (optional) is the per-paragraph list from the result's `paragraphs`
  (original, humanized), changed paragraphs only; a thumbs up keeps each pair
  as its own liked rewrite. Without it the whole-document pair is kept.
  -> {"recorded": true, "liked_count": 2}
  "up" stores (original, text) in the liked pile and raises the weight of the
  samples used; "down" lowers them. A vote on the same run_id twice replaces
  the earlier vote.

## The humanize request

POST /api/humanize/stream, /api/humanize/llm, /api/humanize/jobs accept
  "voice": true
  "voice_strength": "light"|"balanced"|"strong"   # optional, this run only
Server side the user's ready samples and profile are attached at POST time
(the job resume needs no session). With voice on, the result's `summary`
carries:
  "voice": {"used": true, "n_samples": 3, "sample_ids": ["s_.."], "strength": "balanced",
            "match": 0.72,           # 0..1, how close the shipped rewrite sits to the profile
            "run_id": "r_..."}       # pass back with the feedback
With voice requested but nothing usable (signed out, no ready samples, the
server-side switch off):
  "voice": {"used": false, "reason": "no_samples"|"login_required"|"disabled"}
`sample_ids` lists the samples the prompts actually leaned on for this run
(the per-paragraph picker chooses up to three per paragraph within a word
budget), which is what the feedback reweights.
Each paragraph candidate's `quality` gains "voice_distance" (0..1, lower is closer).
