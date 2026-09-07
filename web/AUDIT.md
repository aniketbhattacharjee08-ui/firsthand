# Frontend audit — `web/`

Audited against the live service (reference corpus `research-article-stem`,
60 PMC documents). Every claim below was checked by driving a real headless
Chrome over the DevTools Protocol, not by reading code alone. **209 assertions
across eleven suites, all passing**, plus live WCAG text and non-text sweeps in
both themes. Sixteen real bugs found and fixed, four of them only visible once the real
LLM endpoint landed mid task and was driven end to end.

---

## 1. Genuinely broken, now fixed

### 1.1 Findings were sorted worst-last
```js
findings.sort((a, b) => (SEVERITY_ORDER[a.severity] || 9) - (SEVERITY_ORDER[b.severity] || 9));
```
`SEVERITY_ORDER.high` is `0`, which is falsy, so `0 || 9` evaluated to `9`.
Every high-severity finding sorted to the **bottom** of the list, below `low`
and `info`. The service returns findings already ordered high-first, and the
client actively un-ordered them. Fixed with an explicit `=== undefined` check.
Now verified: the AI sample renders `high, high, high, medium, medium, medium`.

### 1.2 Both dials could permanently read "—"
`setDial` wrote the number *only* from inside a `requestAnimationFrame`
callback. `requestAnimationFrame` does not fire in a background tab. Reproduced
in headless Chrome: the status line read `detection risk 100 of 100 · writing
quality 70 of 100` while both dials showed `—`. The two-dials rule is
non-negotiable, so a dial that silently shows nothing is a serious defect.
Fixed: the tween still runs, but a settle timer writes the exact value
unconditionally shortly after, so the reading is correct whether or not any
frame is ever produced.

### 1.3 Selecting text was destroyed on every re-analysis
The caret save/restore did `range.collapse(true)`, so only a collapsed caret
survived a canvas rebuild. Any *selection* the user had made was silently
thrown away roughly a second after they stopped typing. Fixed: anchor and focus
offsets are both captured and restored. Verified — a 17-character selection
survives a full re-render intact.

### 1.4 Re-entrancy between `selectionchange` and the canvas rebuild
Restoring the caret programmatically fires `selectionchange`, which re-entered
`selectSentence` → `renderInspector` mid-rebuild. Fixed with a `rendering`
guard, released on the next tick because `selectionchange` is async in some
engines.

### 1.5 Exiting mock mode left the reference selector permanently empty
`loadReferences()` returned early when `state.mock` was true and had no reload
path, so the "Try the live API again" button restored the API but never
repopulated the corpus list. Fixed, and covered by a regression test.

### 1.6 The reference selector lied about what was being measured
The service is started with `--reference data/reference/research-article-stem.json`,
so it applies a default corpus even when the request sends `reference: null`.
The old UI offered "none (bands only)", sent `null`, received a Mahalanobis
distance anyway, and captioned it with whatever was in the local dropdown —
`"unnamed"`. The response field that says what was actually used,
`reference_name`, was dropped entirely by `normalizeAnalysis`. Fixed: the
option is now "Service default", `reference_name` is read, and the panel
reports the corpus the service actually used. The raw value (sometimes a file
path) is prettified for display and kept verbatim in the element's `title`.

### 1.7 No offline handling at all
`navigator.onLine` and the `online`/`offline` events were unused. Pulling the
network produced a bare `network error`. Added a dedicated offline banner, an
offline-aware health pill, an offline-specific error message, a one-click route
into mock mode, and automatic recovery when the connection returns.

### 1.8 Pasting rich text corrupted the document model
The contenteditable had no `paste` handler, so pasting from a word processor
injected `<b>`, `<span style>` and friends straight into the canvas that the
sentence-span renderer owns. Fixed: paste is intercepted and inserted as plain
text.

### 1.9 Hidden banners were not actually hidden
`.banner { display: flex }` outranks the UA stylesheet's
`[hidden] { display: none }`, so the offline banner and the mock-mode banner
rendered permanently — the page claimed to be both offline *and* in mock mode
while the service was plainly connected. Caught by screenshotting the running
page, not by any DOM assertion: `element.hidden` was `true` the whole time, so
attribute-level tests passed while the user saw two false alarms. Fixed with a
global `[hidden] { display: none !important; }` and re-verified visually.

### 1.10 Measurement captions repeated themselves
The caption template prepended `human academic prose runs {band}` to a `source`
string that already restated the same band, producing "human academic prose
runs 0.42 to 0.60 — human academic prose 0.42 to 0.60, across five pre-2023
corpora". The `source` fields are now provenance only, and the assembly
tolerates an empty one without leaving a dangling em dash.

### 1.11 The action bar was pushed off the bottom of the screen
`.canvas-col` is a grid item, and grid items default to `min-height: auto`, so
the column grew to its content height instead of its 898px track. The whole
column overflowed and the Humanize button, the single most important control on
the page, sat below the viewport with no scrollbar to reach it. Fixed with
`min-height: 0` on both columns. Found by screenshot, like §1.9; every DOM
assertion passed because the element existed and was "visible", just not on
screen.

### 1.12 Loading a draft left Humanize disabled
`setText()` fires no `input` event, so anything keyed off the text through the
input handler never ran. Loading an example enabled nothing. Fixed by
refreshing the button from `setText()` and after every analysis.

### 1.13 The `?sample=` parameter threw
It still called `markSample()`, deleted along with the toolbar, so
`?sample=ai` raised a `ReferenceError` part way through `init()` and left the
page half wired.

### 1.14 `sentence_index` arrives as a string
`/api/humanize` returns `"sentence_index": "1"`, not `1`. The strict numeric
guard rejected it and every edit silently lost its sentence reference. Fixed
with a lenient coercion used for all values read off an API payload.

### 1.15 HTTP error bodies were shown as raw JSON
A 400 surfaced to the user as `HTTP 400 — {"detail":"Field 'text' is required…"}`.
Now the `detail` field is unwrapped.

### 1.16 The overlay pushed the action bar off the screen, again

The stage panel was first written as an ordinary grid item sharing the canvas
row. Grid items default to `min-height: auto`, so at 1024px wide the panel's
621px card grew the track, the column overflowed and the action bar landed 318px
below the fold. The exact failure as §1.11, in new code, three passes later.
Fixed by taking the panel out of flow: `position: absolute` with an explicit
`grid-row: 2 / 3`, so an abspos grid child is sized by its grid area and
contributes nothing to track sizing.

**`grid-row: 2` alone was not enough.** For an absolutely positioned grid child
an `auto` end line means the padding edge of the whole container, not "span 1",
so the panel spilled down over the action bar: 756px tall against a 653px
canvas. Both end lines are now explicit. Caught by measuring both rectangles at
five viewports, not by looking.

### 1.17 The ring was correct and unreadable

`setRing` handed the tween to Motion One and let the library own the value. Web
Animations does not write to `element.style`, so the inline value stayed at the
initial full-circle offset for the whole run: correct on screen, wrong to every
snapshot, and stale for good if a tween was ever cancelled. Now app.js **always**
writes the final `strokeDashoffset` itself and the tween is decoration on top;
a running animation outranks inline style in the cascade, so the glide is still
seen and the number is always right. The rings therefore fill with the CDN
blocked, which is now a test.

### 1.18 A teal arc on a teal tint

The active row was given the accent wash, and the ring track was darkened to
`--ink-faint` to clear 3:1 against it. Both changes passed every assertion. On
screen the result was a single solid ring: a teal fill against a teal
background with a track of nearly equal weight, and **the fill level could not
be read at all**. Caught by screenshot, exactly like §1.9 and §1.11. The active
row is now a raised card with an accent border, the track went back to
`--rule-strong`, and the fill went from 3.2 to 4.4 stroke width. A 50% ring now
reads as half at 34px in both themes.

### 1.19 The engine panel lost a race it had always been losing

`renderDetectors()` was only ever called after `/api/detect` returned. It reads
`state.analysis`. Whenever `/api/detect` won the race against `/api/analyze`,
which the new published checkpoint made routine, the panel rendered "No analysis
yet." and was never redrawn. It is now also called from `applyAnalysis`.

### 1.20 `event: done` deleted the rewrite

The real stream sends `event: result` and then, immediately after,
`event: done` with `{"ok":true}`. The handler accepted any payload on either
name, so the acknowledgement overwrote the result and the run reported "the
service returned an empty rewrite". A frame is now only taken as the result if
it actually carries a rewrite.

### 1.21 The status vocabulary was half wrong

The pipeline sends `start`, `progress`, `done`, `skip`. `progress` was not in
the map, so a `progress` frame fell through to a rule that infers the status
from the value, which would have marked a stage done the instant it reported
`progress: 1.0` rather than waiting for its terminal frame.

### 1.22 A ring that sat at zero through the longest stage

Measured on a real run: `generate` runs for eight to sixteen seconds and emits
only a start and a done. No intra-stage progress exists to draw. The ring
therefore sat empty at `0%` through the single longest part of the run, which
is the exact failure the whole feature was built to prevent.

Stages with no reported progress are now **indeterminate**: a short arc that
sweeps, reading `working` rather than a fabricated percentage, becoming a real
number the moment a real value above zero arrives. Under reduced motion the
sweep is off and the word plus the elapsed timer carry it. Estimating a
percentage here was the tempting fix and the wrong one: the panel would have
been lying in the one place its whole job is not to.

### 1.23 A client side reason attributed to the service

The word-limit fallback said *"The service said: This draft is 1500 words..."*.
The service said no such thing; the check is made here, from the limit the
service published. The service is credited only when it actually spoke.

Two more real states surfaced only by running the real pipeline, and both had
no handling at all:

- **The rewrite can come back unchanged.** All six candidates failed a gate, so
  the pipeline honestly kept the original. The interface showed a `Humanized`
  heading, `0 edits`, and an undo button for a no-op. It now says *"Your draft
  came back exactly as you wrote it"*, the heading reads `Left unchanged`, no
  undo is offered, and the rejected candidates are reported: *"6 candidates
  were rejected for failing a gate, usually for drifting away from what you
  actually wrote. The lowest scoring text is not automatically the best one."*
- **A run can have no detector.** Both probabilities came back null, so the
  verdict block hid itself and the user saw an empty space where the honest
  answer was a sentence. It now says *"No verdict either way. This run reported
  no detector score before or after, so there is nothing to compare and no
  claim to make about whether it would now read as human."*

---

## 2. Things that were already fine

Reported honestly — a previous agent got these right and they were left working:

- Debounced auto-analyze on typing, with request-token cancellation so a slow
  response cannot overwrite a newer one.
- `Ctrl`/`Cmd`+`Enter`, wired both inside and outside the editor without
  double-firing.
- Sample loaders (AI-ish / human-ish / blank) and the `?sample=` parameter.
- Theme persistence to `localStorage`, including the `auto` state.
- Mock mode computing genuinely real numbers from the real text using matching
  feature definitions, rather than fabricating them.
- The findings renderer already showed `code`, `severity`, `message`, `detail`
  and `grade_cost`, with the raw `grade_cost` string preserved beside its
  classification. This is the product's best idea and it was intact.
- Measurements already showed value, human band and in/low/high state, with
  server-supplied bands taking priority over locally computed ones.
- Sentence location by whitespace-tolerant forward scan, with a regex fallback.
- `Intl.Segmenter` sentence splitting with an abbreviation-merge pass.
- The 20-second `AbortController` timeout.
- The decision to give uncertainty its own visual state (dotted underline)
  rather than a mid-ramp fill.

---

## 2b. Restructure: one number, one engine, one button

The interface was showing everything it knew at once. It now leads with what
the product is for.

**Removed.** The reference corpus selector (the request still sends
`reference: null` and the service applies its default). The sample loader from
the chrome; the two onboarding examples survive inside the empty state only,
where they vanish the moment there is any text. The three row detector
ensemble.

**The rail is now three things.** A hero percentage, the largest number on the
page at 67px. One line naming the engine and what it is not. The handful of
sentences actually reading as AI, ranked, capped at five, filtered to those
scoring 0.50 or above, each clickable to scroll to and flash the sentence in
the canvas. When nothing clears 0.50 it says so and names the highest score
rather than listing filler.

**Everything else folded away.** Measurements, findings, the engine detail, the
reference distance, the quality breakdown, the sentence inspector and the edit
list all live inside one closed `<details>`. Clicking a sentence in the canvas
opens the fold, so the explanation is never a dead end.

**The quality signal survives**, deliberately, as a secondary readout beside the
hero number rather than a second dial. Risk shown alone pushes writers toward
worse writing, and that trade is the product's whole differentiator.

**One judgement call worth flagging:** the per-sentence perplexity overlay,
built for the previous direction and fully tested, was moved inside the fold
rather than deleted. It is invisible by default so it costs nothing visually.
Say the word and it goes.

## 2c. Humanize

The point of the product, and it was missing. A primary action sits directly
under the canvas with a light / balanced / strong control beside it.

Click it and the canvas is replaced by the rewrite, the before and after risk
appear side by side with the delta, and one click puts the original back byte
for byte. The edit list lands inside the fold: every edit names what changed,
why, and what it costs the grade, with the raw `grade_cost` string preserved.

**This is where the reserved diff colours are finally spent.** Deletions get
`--diff-del #a8442e` with a strike through, insertions get `--diff-ins
#2f7d51`, and a key explains both. The strike through matters: the diff must
not be colour alone.

The endpoint did not exist when this was built, so its absence is a first class
state. A schema probe at startup disables the button with an explanation, and a
404 or 405 at click time disables it calmly without an error box and without
touching the draft. Both paths are tested by forcing them with a stub. The real
endpoint shipped mid task and is now verified end to end: 22 edits at the
strong setting, risk 100 to 94, undo exact.

## 2e. The live progress experience

A rule based rewrite returns in under a second. The LLM pipeline takes about
**30 seconds**, which is long enough that a silent button is a bug. Clicking
Humanize now hands the middle of the page to a stage checklist.

**It is the middle of the site, not a corner spinner.** The panel is absolutely
positioned onto the canvas grid track, so it covers the writing canvas exactly,
edge to edge, at every viewport. Measured at 1440x900, 1280x800, 1024x700,
1024x620 and 1280x560: the panel's height equals the canvas's in all five, the
action bar does not move by a single pixel, and there is no horizontal overflow.

**Eight rings that fill.** One SVG ring per backend stage, `stroke-dasharray`
set to the circumference and `stroke-dashoffset` driven from the streamed
`progress` value:

| stage | label |
|---|---|
| `analyze` | Reading your draft |
| `plan` | Planning the rewrite |
| `generate` | Writing candidate versions |
| `scrub` | Removing AI vocabulary |
| `score` | Scoring each version |
| `verify` | Checking your facts survived |
| `select` | Choosing the best one |
| `finalize` | Polishing the final draft |

**No state is carried by colour alone.** Pending is an empty ring reading
`waiting`, active is a partly swept ring reading `43%` on a raised card with an
accent border, done is a full faded ring with a check glyph reading `done`.
Geometry, a glyph and words, three times over.

**The streamed `detail` sits beside the active stage** in mono, so the user
reads `candidate 3 of 6, paragraph 2 of 4` rather than guessing, and an elapsed
timer ticks every 100 ms. At 45 seconds a note appears saying it is slower than
usual and is not stuck. A screen reader gets one polite announcement per stage
change, `Step 5 of 8, scoring each version, scoring candidate 3 of 6`, and not
one per tick.

**Cancel is a real control**, focused on open, plus `Escape` as a second route.
It aborts through `AbortController`, verified by asserting the stub's own
`signal.aborted`. The draft is byte identical afterwards, the editor becomes
editable again, focus goes back to the Humanize button, and a second run starts
cleanly.

**The editor is `contenteditable="false"` for the duration**, so nothing can be
typed under a running rewrite and the canvas cannot be tabbed into behind the
panel. It is restored on every exit path, including cancel and error.

## 2f. Three transports, because the pipeline was built in parallel

The LLM endpoint did not exist while this was written; it landed mid task and
is now verified end to end against the real thing, streaming real frames from
`mlx-community/Qwen2.5-3B-Instruct-4bit`. Rather than guess and hope, the
client tries three routes and never leaves a dead UI:

1. **SSE.** If the schema advertises a streaming path, that path is used;
   otherwise the request goes to `/api/humanize/llm` with
   `Accept: text/event-stream`. A service that answers `text/event-stream` gets
   parsed as a stream; a service that answers JSON on the same path **is** the
   blocking case, handled on the same response, at no extra round trip.
2. **The blocking POST**, with the checklist on an **estimated** timeline. The
   panel says `estimated` in a chip and says out loud that the service is not
   reporting progress. The estimate is armed 900 ms after the click, so a
   service that streams immediately never shows the word.
3. **The rule based `POST /api/humanize`.** A 404, 405, 501, 422 or 400 falls
   through. The panel switches to it *in place*: it does not fabricate eight
   stages for a one pass engine, it collapses to one row and explains why.

The SSE reader is a `fetch` reader, not `EventSource`, so a POST body and an
`AbortController` both work. It is deliberately liberal: CRLF or LF frames,
comment lines, multi-line `data:`, `progress` on a 0 to 1 **or** 0 to 100
scale, `stage`/`name`/`step` for the key, a dozen spellings of each status, a
result under `event: result` or `done` or `complete` or simply any frame
carrying `humanized`, and a payload nested under `result`. **A stage name this
build has never heard of is appended with a generated label rather than
dropped** (verified with a stage called `quarantine`). A stream that dies
before a result is reported as an error, not left hanging.

Only the last route's error is ever shown, so nobody sees a stack of failures
for endpoints they never asked about.

### What the real endpoint changed

`GET /api/humanize/llm/health` is cheap, touches no weights, and answers four
questions this front end was otherwise guessing at. It is now probed once on
load, and its answers are used rather than assumed:

- **`stages`** is the authoritative stage order, so the checklist is built from
  the service's list and only falls back to the one hardcoded here. A pipeline
  that adds or drops a stage needs no change in this file.
- **`available` and `reason`.** When the LLM cannot run, the panel switches to
  the rule engine on the *first* request instead of discovering it over two
  wasted round trips, and it prints the service's reason.
- **`model`**, so the model can be named before a single token is generated.
- **`max_words`.** A draft over the limit goes straight to the rule engine with
  the real number in the copy: *"This draft is 1500 words, past the 1200 word
  limit the LLM path publishes for itself."* Verified live at 1500 words.

The refusals are real status codes with a body that names where to go instead:
503 when MLX or the weights are missing, 413 over the word limit, 400 for a bad
style. **The `fallback_endpoint` field in the body is trusted ahead of the
status list**, so a refusal shape this build has never seen still falls through
correctly. An `error` frame arriving mid-stream, which is HTTP 200 because the
headers are long gone, is handled the same way.

The result payload is read on its own terms: `summary.verdict_flipped` (which
the endpoint's docstring says to read *instead of* `risk_delta`, for exactly
the reason this interface was already built around), `summary.label_before` and
`label_after`, `stages[].seconds`, and `summary.n_candidates_rejected`.

## 2g. Honest results

Everything from 2c survives: canvas replaced, before and after risk with the
delta, one click byte-for-byte undo, edits in the fold with grade costs, diff
green and red with a strike through so colour is never the only signal. Added:

- **Which engine ran and which model**, stated plainly in mono:
  `LLM rewrite pipeline · model qwen2.5-14b-instruct · 23.4s · progress
  streamed live`. A payload with no model says `model not reported` rather than
  inventing one, and the rule engine claims no model at all.
- **Whether the verdict actually flipped**, which is the measure that matters.
  A reported `label` on either side is used; failing that a verdict is derived
  at the 0.50 mark and **said to be derived**. Three outcomes, all in words:
  flipped, did not flip, or there was nothing to flip because it already read
  as human.
- When it did not flip, the panel says so first: *"The verdict did not flip. It
  still reads as AI. Cutting the probability is not the same as earning a human
  verdict, and the verdict is the only thing a detector actually reports."*
  This is the measured reality of the rule engine, 0.999 to 0.881 with zero
  flips, and the interface refuses to dress it up.
- **The green delta is now conditional.** It is painted as a win only when the
  verdict moved. A falling probability that leaves the draft labelled AI is
  still reported, in neutral ink, next to the sentence explaining why it is not
  a win.
- **Total elapsed time**, from the response when the service reports it,
  otherwise measured from the click.

## 2d. No em dashes

Swept out of every string in the HTML and JS, including tooltips, aria labels,
disclaimers, empty states and the measurement captions, which used them
heavily. A test walks every text node and every `title`, `aria-label` and
`placeholder` in the rendered page and asserts none remain.

Strings the **service** returns are left verbatim, because paraphrasing a
disclaimer to fix its punctuation would be a worse trade. The test separates
the two so the distinction stays visible.

## 3. Redesign

Asha Berkeley was used **only** for its qualities — generous whitespace,
unhurried pacing, numbered section rhythm, confident display type against quiet
UI type, warm tone, calm motion. None of its colours or fonts are present.

### Palette (every value contrast-checked; see §5)

| Role | Light | Dark |
|---|---|---|
| page / surface / sunken | `#fbfaf7` `#ffffff` `#f1f0ec` | `#0f1216` `#171b20` `#0a0d10` |
| ink / secondary / muted | `#16191d` `#4a5159` `#5f6975` | `#e8eaec` `#a9b2bb` `#8a949e` |
| hairline / border | `#e3e2dd` `#7c8590` | `#242a31` `#666f7a` |
| accent (teal-green) | `#1f6f5c`, hover `#17594a`, tint `#e6f1ee` | `#3fa48a`, hover `#5cb79f`, tint `#12302a` |

**Risk ramp** — calm teal → caution ochre → alert clay, desaturated on purpose
so it never reads as a traffic light and never collides with the diff colours:

- light backgrounds `#e8eee9 #e0e3d8 #e1d8c0 #dfc6a7 #d3a98e #c48272`
- dark backgrounds `#141d1f #222822 #3c3521 #563e21 #683d23 #7a3527`

The brief's three ramp anchors fail as foreground text at UI sizes (`#b8862f`
is 3.10:1 on paper), exactly as predicted, so there are **separate readable
text tokens**: `--risk-calm-txt #2f5d51`, `--risk-caution-txt #7a5a1c`,
`--risk-alert-txt #8c3826` (7.2 / 6.1 / 7.4:1). The risk arc is recoloured from
these per reading, so a calm document is not drawn in the alert hue.

**Diff colours are declared and deliberately unused**: `--diff-ins #2f7d51`,
`--diff-del #a8442e`. A test asserts that no element in the rendered page
paints itself in either — green and red stay reserved for the rewrite layer.
Grade-cost chips, which would be the obvious place to reach for green and red,
are typographic and borrow the accent instead.

### Type
| Role | Family |
|---|---|
| display, headings, wordmark | **Fraunces** (variable; `SOFT`/`WONK`/`opsz` axes used) |
| writing canvas, quoted sentences | **Newsreader** |
| UI chrome and labels | **IBM Plex Sans** |
| every number, metric and measurement | **IBM Plex Mono** |

Loaded non-blocking from Google Fonts. Each family also has a `local()`
`@font-face` so an installed copy wins, and each CSS custom property carries a
full system fallback stack. Verified with the CDN blocked: the page falls
through to Iowan Old Style / Charter / system sans and still looks deliberate.

### Animation
`anim.js` is a separate ES module that **only enhances already-rendered DOM**.
`app.js` paints the entire interface by itself; nothing depends on the module.

- **Motion One** (`esm.sh/motion@10`) — panel entrances with `stagger`,
  `inView` reveals for the lower rail sections, dial count-ups, arc settle.
- **AutoAnimate** (`esm.sh/@formkit/auto-animate@0.8`) — attached to the
  findings, measurements, detector and flagged lists so they reflow smoothly on
  re-analysis.
- Sentence highlights fade in progressively, budgeted to the first 60 spans.
- Durations 150–450 ms, `cubic-bezier(0.22, 1, 0.36, 1)`, no bounce or overshoot.

Three independent safety mechanisms, because an animation library must never be
able to hide the product:
1. The CSS that sets `opacity: 0` is scoped to `html[data-anim="on"]`, an
   attribute set **only after both imports resolve**. A blocked CDN therefore
   cannot hide anything.
2. An 8-second import ceiling and a 1.6-second post-entrance failsafe, the
   latter armed only once `data-anim` is on.
3. `standDown()` clears the attribute and force-reveals everything on any
   error, on reduced-motion, and on a mid-session OS setting change.

All three paths are tested: CDN reachable, CDN unreachable, and
`prefers-reduced-motion: reduce`. In every case all seven panels end at
`opacity: 1` and the dials read correct numbers.

---

## 4. The detector panel, and the contract change under it

Mid task the service deleted its hand-written detector outright: detection now
comes only from published pretrained checkpoints, and the repo writes no
detection arithmetic at all. Five things in this front end were wrong the
moment that landed, and all five are fixed.

- **`detectors` is keyed `modern`, not `heuristic`.** `pickDetector()` already
  read whatever key arrived, so it survived, but it now also reads `model` and
  `is_published_detector`.
- **The engine caption was a hardcoded lie.** It said "a local heuristic
  baseline that ships with this project". It now names what the service
  actually ran, read off the response: *desklib/ai-text-detector-v1.01, a
  published pretrained checkpoint the service runs.*
- **`is_published_detector` is read as three states, not two.** True, an
  explicit false, and "the service did not say". An older build that sends
  neither `model` nor the flag gets its name printed and no publication claim
  attached to it. Verified against both builds side by side.
- **The invented fallback score is gone.** `renderHero` used to fall back to a
  locally computed `compositeRisk()` when no detector answered. That is exactly
  the arithmetic the service just deleted, in the same field a model's score
  would have occupied. On the live path there is now **no number** when no
  model answers. Mock mode still computes its own, because mock mode is
  labelled as mock everywhere it appears.
- **`detector_error` is a first class state.** When the checkpoint cannot load
  the `detectors` block is empty, every `sentences[].risk` is null, and the
  reason lands in `detector_error`. The hero reads `n/a` rather than `0%`, the
  rail prints the reason verbatim, the engine panel shows a card with the error
  and `n/a` instead of an empty box, and the flagged list says why it is empty.
  Measurements, findings and style signals all keep working. Tested by forcing
  a meta-tensor load failure: 11 assertions.

### The per-detector notes

Every published engine now carries the service's own measurement, and the
**false positive count sits beside the separation score in every row**:
`modern` 1.000 pairwise with 1 of 14 human academic paragraphs called AI,
`fakespot` 1.000 and 3 of 14, `academic` 0.980 and 1 of 14, `fast` 0.944 and
**8 of 14**, `radar` 0.867 and 1 of 14 but half the AI missed. `fast`,
`classifier` and `perplexity` each get a standing "do not act on this number"
block above the general disclaimer.

`perplexity` is demoted rather than deleted: GPT-2 is published but the
mapping from its perplexities to a probability was written in this repo and
fitted to nothing, so it reports `is_published_detector: false` and is out of
every default. The per-sentence overlay that depended on it therefore offers
itself only when the service actually returns it, which by default it no
longer does. That path degrades to a hidden toggle, which was already tested.

## 4b. New: the style signals panel

The old heuristic's per-feature signals survive on the wire as
`ai_style_signals`, carrying `is_a_detector: false`, a `research/NN` citation
per signal, and **deliberately no aggregate**. They are now a panel inside the
closed fold, "Which AI tells are present", kept visually and verbally apart
from the score.

- Each signal is a named row with its 0 to 1 value, a bar, and its citation
  printed underneath: `research/04: AI-vocabulary density, top-ranked tell`.
- Sorted strongest first, filtered to 0.25 and above, with the count shown so
  the filtering is not silent. When nothing clears the floor it names the
  strongest one and its value rather than showing an empty box.
- The service's own `note` is printed **verbatim**.
- **Nothing is summed.** The panel says so in as many words: "There is no total
  on this panel on purpose." A test asserts no total, sum or combined score
  appears in the rendered text. Summing them would rebuild the detector that
  was just deleted.
- Mock mode says it does not compute them rather than inventing them, and a
  service that does not send the block says so rather than breaking.

## 5. Accessibility

- **WCAG AA verified live**, not by eye: a sweep over every rendered text node
  in the running page — with all explainers expanded, a measurement row open
  and a sentence selected — computes each element's contrast against its true
  painted background and its own size/weight threshold. **0 failures in light,
  0 in dark.**
- Ink on every one of the six risk-ramp backgrounds stays above 5.7:1.
- Visible 2px focus rings on all interactive elements; a skip link first in the
  tab order.
- Measurement rows and the three `?` explainers are real `<button>`s with
  `aria-expanded` / `aria-controls`; the flagged list is the keyboard route into
  the canvas.
- Caret movement inside the contenteditable drives the inspector, so no
  `tabindex` is needed on sentence spans.
- `prefers-reduced-motion: reduce` disables both the CSS transitions and the
  whole animation layer, jumping to the final state.
- No horizontal overflow at 1440, 1280 or 1024 px.
- No emoji anywhere (asserted by test).

---

## 6. Verified working

| Feature | How |
|---|---|
| `/api/health` → status pill | live; version, syntax backend, corpus count |
| `/api/references` → selector | live; 2 options, `research-article-stem (60 docs)` |
| Reference switch changes the result | live; panel updates to the named corpus + real Mahalanobis distance |
| `/api/analyze` | live; 22 spans, 14 measurements, 6 findings on the AI sample |
| `/api/detect` | live; 3 detectors, verbatim disclaimer, availability handled |
| Analyze button / `Ctrl+Enter` | both, inside and outside the editor; no stray newline |
| Debounced auto-analyze | fires ~0.9 s after the last keystroke, no click |
| **Caret survives re-render** | offset 152 → 152 |
| **Scroll survives re-render** | 45 → 45 |
| **Selection extent survives** | `" relationship bet"` intact |
| Sample loaders | AI / human / clear; risk 100 vs 0 |
| Sentence click → why | risk, confidence, length vs mean, named patterns |
| Findings | `code`, `severity`, `message`, `detail`, `grade_cost` (raw string kept) |
| Measurements | value + human band + in/low/high + plain-language expansion |
| Two dials | both numeric, arcs sweep, quality labelled a placeholder |
| Theme | auto → light → dark, persists across reload |
| Mock mode | `?mock=1`, from the error state, and from offline |
| Error state | cause, reassurance, Try again, switch to mock; draft preserved |
| Offline | banner, pill, mock route, clean recovery |
| Animation | CDN up, CDN down, reduced motion |
| Humanize, real endpoint | 22 edits, risk 100 to 94, canvas replaced, undo exact |
| Humanize, endpoint absent | button disabled with an explanation, no error |
| Humanize, 404 at click | disabled calmly, draft untouched, no error box |
| Before / after + delta | populated from the response, direction marked |
| Edit list | diff green and red, strike through, grade cost each |
| Removals | reference selector, sample chrome, ensemble rows all gone |
| Fold closed by default | `checkVisibility()` false, rail collapses to 89px |
| Top sentences | ranked, capped at five, click scrolls and flashes |
| No em dashes | every text node and copy attribute in the live page |
| Hidden elements genuinely hidden | banners, error box, empty state, explainers |
| No horizontal overflow | 1440 / 1280 / 1024 px |
| WCAG AA | live sweep, every text node, both themes, 0 failures |
| **Stage checklist renders** | 8 rings, backend stage keys, labels, SVG dasharray = circumference |
| **Rings fill from streamed progress** | active ring at 47.124 of 94.248 for `progress: 0.5` |
| **Checklist covers the canvas** | identical rectangles at 5 viewports; action bar never moves |
| **Streamed detail shown** | `candidate 3 of 6, paragraph 2 of 4` beside the active stage |
| **Elapsed timer** | ticks; 1.3s to 2.1s over a 0.8s wait |
| **Live region** | one polite announcement per stage change, not per tick |
| **SSE path** | adopted, chip reads `measured`, one request only |
| **SSE on the LLM path itself** | same request serves streaming and blocking |
| **Blocking fallback** | chip reads `estimated`, rings advance with zero events |
| **Estimate never completes a run** | last ring held below 100% until the result lands |
| **Rule based fallback** | switches in place, 1 row not 8, says why |
| **Cancel** | `signal.aborted` true, draft identical, focus returned, rerunnable |
| **Escape cancels** | second escape hatch |
| **Reduced motion** | 0 running animations, every ring exactly on a final value |
| **Rings fill with the CDN blocked** | no animation library at all |
| **Unknown stage name** | appended with a generated label, not dropped |
| **Percent-scale progress** | `progress: 100` read as complete |
| **CRLF, comments, multi-line data** | all parse |
| **Broken stream** | reported, overlay closed, button restored |
| **Two minute ceiling** | aborts and says so |
| **Engine and model in the result** | named, or `model not reported` |
| **Verdict flip** | flipped / did not flip / nothing to flip, and derived-vs-reported |
| **Delta not painted as a win** | neutral ink when the verdict did not move |
| **`detector_error`** | `n/a` not `0%`, reason verbatim, everything else still works |
| **Engine caption from the response** | `desklib/ai-text-detector-v1.01`, no hardcoded claim |
| **Older contract** | no publication claim when the service does not say |
| **Style signals** | cited per signal, no aggregate, service note verbatim |
| **WCAG 1.4.11 non-text** | 25 ring strokes and bars per theme, worst 4.60:1 |
| **No em dashes, checklist running** | every text node and copy attribute, both themes |
| **Real SSE, real model** | live against `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| **Real stage details** | `scored the original: p(AI) 0.998, labelled ai`, and 10 more |
| **Indeterminate stage** | `generate` reads `working` and sweeps, never a fake percentage |
| **Real per stage seconds** | `6 candidates in 8.4s, batched decode` |
| **Unchanged rewrite** | `Left unchanged`, no undo offered, rejections reported |
| **No detector in the run** | "No verdict either way", not an empty slot |
| **Real 1200 word limit** | 1500 words goes straight to the rule engine, with the number |
| **Live `detector_error`** | hit for real when the weights failed to materialise |

---

## 7. Known limitations, stated rather than hidden

- The **writing-quality dial is a placeholder formula** computed in the browser.
  It exists so risk is never shown alone. It is not a rubric score and the UI
  says so, with its arithmetic printed in full. It is the one number on the page
  this project still computes itself, and it is labelled a placeholder in three
  separate places.
- The **detection number is a published checkpoint's output**, currently
  `desklib/ai-text-detector-v1.01`, not GPTZero, Turnitin or Pangram, and
  uncalibrated against any of them. When that checkpoint cannot load there is
  no number at all, which is the honest answer and not a regression.
- **The estimated stage timeline is a guess.** When the service cannot stream,
  the eight rings advance on weights chosen in this file, not on anything the
  service reported. The panel is labelled `estimated` for exactly that reason,
  and the final ring never completes on a guess.
- **The 30 second budget is the pipeline's, not a measurement.** The "slower
  than usual" note fires at 45 seconds and the hard ceiling is two minutes.
- **Sentence risk is advisory.** The document verdict is not an aggregate of
  sentence scores, and the two are allowed to disagree.
- `/api/detect` returns `sentence_scores: []` for every detector, so there is no
  per-detector sentence overlay to draw. Nothing is invented in its place.
- **Mock mode has no reference corpus**, so it reports no distance rather than a
  fabricated one — a change from the previous version, which invented one.
- The animation layer needs a reachable CDN. Without one the page is simply
  static, which is a supported state, not a failure.

---

## 8. How this was verified

`node` is not installed in this environment, so verification used:

- **`jsc`** (JavaScriptCore, shipped with macOS) for syntax validation of
  `app.js` and, in module mode, `anim.js`.
- **Headless Chrome driven over the DevTools Protocol** through a small
  dependency-free websocket client, for everything else: clicking real buttons,
  dispatching real keystrokes through the contenteditable, emulating
  `prefers-color-scheme` and `prefers-reduced-motion`, forcing the browser
  offline with `Network.emulateNetworkConditions`, stubbing `fetch` to exercise
  the unavailable-detector and API-failure paths, resizing the viewport, and
  capturing screenshots.
- `Runtime.exceptionThrown` and `Log.entryAdded` were recorded on every run.
  **No uncaught page errors in any suite.**
- Blocking all non-localhost DNS to prove the Google Fonts and animation-CDN
  fallbacks, then unblocking to prove the enhanced path.

**209 assertions across eleven suites**, each a separate headless Chrome:

| suite | what it covers | n |
|---|---|---|
| 1 | SSE end to end, rings, detail, timer, result, undo | 34 |
| 2 | blocking fallback, rule fallback, cancel, reduced motion | 46 |
| 3 | SSE on the LLM path, no-flip verdicts, odd wire shapes, mock, CDN down | 29 |
| 4 | WCAG AA text sweep and em dash sweep, both themes, checklist running | 18 |
| 5 | WCAG 1.4.11 non-text contrast on every ring stroke and bar | 2 |
| 6 | five viewports, overlay geometry, action bar, overflow | 21 |
| 7 | the new detector contract, live | 18 |
| 8 | `detector_error`, the empty-detector state | 11 |
| 9 | the previous contract, for backwards compatibility | 6 |
| 10 | the **real** LLM SSE endpoint, end to end | 16 |
| 11 | the **real** word limit fallback, end to end | 8 |

The transports were exercised by stubbing `window.fetch` in the page to return
a real `ReadableStream` of real SSE frames, so the client's own parser, its
`AbortController` wiring and its content-type branching are all under test
without a line of Python being written or touched. A second service instance
was started on another port to test against the new detector contract while
the original stayed up.

Screenshots were part of the loop, not decoration. Both §1.9 and §1.11 were
invisible to DOM assertions and only showed up on screen: in the first the
element reported `hidden === true` while rendering, in the second it reported
sensible dimensions while sitting below the fold. Neither would have survived a
glance at the page; neither would have been caught by any amount of
`element.hidden` or `offsetParent` checking. §1.18 joins them: every assertion
about the active ring passed while the thing it measured was invisible to a
human eye.
