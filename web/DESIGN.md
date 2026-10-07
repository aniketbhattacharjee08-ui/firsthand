# Vervly front end: design rules

Read this before editing anything in `web/`. The files are `index.html`,
`styles.css`, `app.js` (ES5, no framework, no build step), `anim.js` (an ES
module that animates what `app.js` has already rendered) and `field.js` (an
ES module that draws the ASCII field behind the page and the two ASCII tiles
in the After pane). Two more pages share the tokens in `styles.css` and the
field: the landing (`home.html`, `home.css`, `home.js`) and the sign-in and
create-account form (`auth.html`, `auth.css`, `auth.js`). See "Pages".

The product is called Vervly on the page. The ASCII field behind the app page
is still since 2026-09-13: the `#field` canvas carries `data-still`, so `field.js`
draws it once and ignores the pointer; the landing and sign-in pages have no
field. The two ASCII tiles in the After pane and the progress card still animate. Code identifiers, event names
(`humanizer:*`) and the API are unchanged.

## Pages

Since 2026-10-07 the tool is the landing page. The server serves `GET /`
and `GET /app` as `web/index.html` (the sign-in wall, `HUMANIZER_PUBLIC_APP=0`,
restores the old `web/home.html` landing and the 303 to `/signin?next=/app`),
`GET /how` as `web/how.html` (the three steps, what stays and what goes), `GET /pricing` as `web/pricing.html`, and `GET /signin`
and `GET /signup` as `web/auth.html`. How it works and Pricing are tabs in
the top row, not sections on the landing (owner, 2026-10-07). Every asset URL is absolute (`/styles.css?v=...`) so the
same file works from any of those paths. Bump `?v=` on everything you touch.
`styles.css` carries the tokens and every component; `site.css` is the page
around them (top row, hero, tool card, trust row, steps, pricing, Berkeley);
`pricing.css` and `auth.css` are their pages' own; `home.css` remains only
for the pricing page's older section classes.

A visitor who has not signed in is a guest: the paywall gives each client
address one free allowance (`LONGHAND_GUEST_CREDITS`, 1,000 words),
`GET /api/me` reports it with `guest: true`, the top row shows a gold pill
"N free words left" and a Sign in link instead of a name and Sign out, and
when the allowance is gone a charged route answers 402 with `guest: true`
and `signup_url`, which sends the page to `/signup?next=/pricing&error=free_used`.
An account created from that address inherits what the guest had left, so
one address gets one free allowance however it is used.

The page (`index.html`), top to bottom: the top row (wordmark; How it
works and Pricing links; the service pill; the account row). A centred hero
of the gold Barlow Condensed eyebrow "Built at UC Berkeley" between two
short rules, one Source Serif headline, one sentence, and a trust row of
three facts with gold dots. The tool, one card on `--g2` with a 12px radius:
the two panes flush inside it (heads on `--blue-wash`, titled "Your draft"
and "Rewrite"), the divider, then the action bar (`.console`) on `--g1`
with Strength and the judge on the left and the status and the 44px gold
Humanize on the right, then the Facts and Details folds on the same deeper
ground. Under the card: only the footer, one centred stack (lockup, the
links with the construction switch and the year, the disclaimer); a left
lockup with a right-hand link block read as off-centre once the disclaimer
wrapped (owner, 2026-10-07). The service pill in the top row is a gold dot
alone while the service is fine (the words are its title); it speaks only
while connecting or when something is wrong. The disabled primary is gold
at low alpha, the primary action not yet available, never grey. Pane
bodies have pixel minimums (320 / 220 on a phone), not viewport fractions,
so a tall screen does not stretch an empty card. `/pricing` opens with the
same `.doc-head` as `/how` (eyebrow, serif title, one lede) and its notes
section is "The fine print", so the nav's "How it works" means one thing. The steps live on `/how`; the plans on `/pricing`. The Made in
Berkeley section is gone (owner, 2026-10-07); the eyebrow and the footer
disclaimer remain. The page field is gone;
the ASCII tiles in the After pane and the progress card remain. No
testimonials, logos, slogans.

Pricing exists, once, as the Pricing section after the specimen
(`#pricing`): the heading "What it costs", the free allowance line ("1,000
words free when you create an account"), then three `.price-plan` columns
on the facts row's hairlines (Monthly $9.99 a month, Yearly $79.99 a year,
Lifetime $299 once; each with its words a month and one sentence saying
whether it renews), and one note that unused words carry over only up to a
month's amount. The hero fine print says "1,000 words free, then from $9.99
a month." Every numeral and price is in `.num`. The prices are static copy
and must match what the server charges; the plans sheet in the app says the
same things in the same words. No comparison table, no "most popular" mark,
no strikethrough prices, no per-word arithmetic.

The pricing page (`pricing.html`, `pricing.css`, `pricing.js`, served
public at `/pricing`): the landing's top row with a Pricing link marked
`aria-current`, `#nav-balance` ("N words left", signed in with the paywall
on), the sign-in and create-account links or "Open the humanizer"; the
heading "What it costs", one sentence, `#free-line`; three `.price-card`s
in `#price-grid` (name, price, words a month, one renewal sentence, a
`.btn-primary` "Choose Monthly" and so on); `#manage-btn` and
`#plans-status` under them; the top-up rows in `#topup-rows`; the How it
works list and the two legal links; the site footer. The cards in the HTML
are static fallback copy; `pricing.js` rebuilds them from
`GET /api/billing/plans` when it answers. Signed out, every button is a link
to `/signup?next=/pricing`. Signed in, Choose posts `/api/billing/checkout`
and follows the URL; the active plan's card carries `data-current`, "Your
plan" and the renewal date; a 409 offers Manage subscription, a 503 says
payments are not set up. With the paywall off, absent, or without Stripe
the buttons are disabled and `#plans-status` says why in one line. The
landing's nav, its `#pricing` section ("See all plans") and both site
footers link to it; the app's plans sheet note has a "Full details" link.

The form (`auth.html`): one column, marked `data-field-quiet` so `field.js`
holds the field to about half behind it, with the wordmark on the ground,
an opaque `--g2` card, and a "Back to the front page" link. In the card:
two mode links (Sign in, Create account) that are real links to `/signin`
and `/signup` and switch in place with `pushState` (on a server without
those routes the mode is `?mode=signup`); a Source Serif heading that names the
mode; an alert line (`role=alert`); Name (create account only, optional),
Email, Password with a Show/Hide button (`aria-pressed`) and, when creating
an account, the hint "At least 8 characters."; a full-width primary button
whose label is the mode's verb. Inputs are 16px and 44px tall. `auth.js`
validates before sending (empty fields, email shape, 8 characters on
signup), then POSTs JSON to `/api/auth/signin` or `/api/auth/signup` with
cookies. Answers in plain words: 401 "That email and password do not
match."; 409 switches to Sign in and says "There is already an account for
that email. Sign in instead."; 422 marks the field the server named; 404
"Signing in is not available on this server yet."; 5xx and network errors
say so and to try again. Success goes to `?next=` if it is a path on this
origin, else `/app`.

The app (`index.html`): the wordmark is a link to `/`. The top row's
`.account` group (`#who`, a `.btn-quiet` Sign out) is hidden until
`GET /api/auth/me` answers signed in; `signed_in: false` sends the page to
`/signin?next=/app`; a failed call leaves the row alone so a development
server without auth still runs the app at `/index.html`. Sign out POSTs
`/api/auth/signout` and goes to `/`. Any product API response with status
401 goes through `signInRequired()` in `app.js` (hooked in `request()` and
`httpFail()`) to `/signin?next=/app`, once. The same `.account` group holds
the balance and the Plans button when the paywall is on (see "Billing").

## Billing

Everything here appears only when `GET /api/me` answers `paywall: true`
with a user; `paywall: false`, a 404 or no answer leaves the page exactly
as it was, with no balance, no hint, no sheet and no gating.

- The top row's `.account` group gains `#balance` ("N words left", the
  numeral in `.num`), `#plan-name` ("Monthly plan", "Yearly plan",
  "Lifetime", or nothing) and `#plans-btn`, a `.btn-quiet` "Plans". The
  balance comes from `/api/me` on load and is refreshed from the
  `X-Longhand-Credits-Balance` header on every charged answer
  (`noteBalance()`, credits times `words_per_credit`).
- `#cost-hint` in `.console-act`, third-tier text beside the shortcut:
  "Uses about N words of your M left." When the draft is bigger than the
  balance it turns `--mixed` with `data-short` and reads "Not enough words
  left; choose a plan.", the link a `.link-btn` (a button that reads as a
  link inside a sentence). The Humanize button stays live; its click then
  opens the plans sheet instead of a run.
- The plans sheet (`#plans-backdrop`, `#plans-sheet`, `role=dialog`): a
  fixed ground at 72 percent, one opaque `--g2` card at most 560px wide,
  hairline, no shadow; the ground fades and the card rises 8px in the fast
  duration. Focus moves to the card, Tab stays inside it, Escape, Close and
  a click on the ground close it and return focus to what opened it. Inside:
  the Source Serif title "Plans"; `#plans-why` ("This draft needs X words and
  you have Y left.") when a 402 or the hint opened it; `#plans-free`, the
  free allowance line; `.plan-row`s from `GET /api/billing/plans` (interval
  name, words a month and whether it renews, the price label, a
  `.btn-primary` "Choose"; the active plan shows "Your plan" instead); the
  one-time top-ups under "One-time top-ups" with "Top up"; the note on
  renewal and carry-over; a quiet "Manage subscription" (`#manage-btn`,
  `POST /api/billing/portal`, shown only with a plan) and `#plans-status`,
  which says what is happening or what went wrong and what to do.
  Choosing posts `{plan}` or `{pack}` to `/api/billing/checkout` and goes to
  the URL that comes back.
- A 402 from a charged route is a paywall event, never an error: `httpFail()`
  marks `e.paywall` with the body, `readSse()` does the same for an error
  frame carrying `insufficient_credits`, and the `humanize()` reject branch
  updates the balance, sets the status line to "Not enough words left for
  this draft" and opens the sheet. The error box is not shown and the rule
  engine is never tried.
- Back from Stripe, `/app?purchase=success` polls `/api/me` once a second
  for up to ten seconds until the balance rises above the one remembered
  before checkout, then the status line says "Payment received: N words
  available."; `?purchase=cancelled` says "Payment cancelled, nothing was
  charged." Either way the query is cleaned with `history.replaceState`.

## What the page is

An instrument on a deep graphite ground. Top to bottom: a top row
(wordmark, one sentence, the Construction switch, the service pill) closed by
a horizon hairline; the console (the one primary button, the shortcut, the
judge's name, the status line); two panes with a draggable divider between
them, Before on the left with the draft and After on the right with the
rewrite, each with its own reading in its head and its own actions in its
foot; then the folds (Facts you can vouch for, How it works, Details, How it
works) and one line of help. Behind all of it a slow orbital system rendered
as ASCII characters that the pointer can move.

Both panes are in view once the page is wide enough. The rewrite never
replaces the draft on its own; Use this does that, Undo reverses it.

## Where the ideas came from

Six references the owner pointed at, and what each one became here.

1. Motion Panels (@letstri): resizable, collapsible panels with eased motion.
   Became the divider between Before and After (see "The split").
2. ASCII imagery (@kail_designs, @insporadesign): images and surfaces made
   of characters. Became the field (`field.js`) and the two ASCII tiles in the
   After pane (see "The field" and "The tiles").
3. Spell UI and opensourceui.in (@ErfanEbrahimnia, @BidyutKundu12): finish.
   Every control has a designed rest, hover, active, focus and disabled state;
   one radius, hairline borders, a 4px spacing scale, a real switch, a
   tooltip on the two numbers (see "Controls").
4. The wireframe reveal switch (@wherescz): see how a component is built.
   Became the construction layer for the whole page (see "The construction
   layer").

Earlier rules still hold: never pure black; surfaces step lighter as they
come forward; one accent, one hue; near-white text in three AA tiers;
hairlines at low alpha; two faces from one superfamily; every element
carries information; motion answers an action and stays under 300 ms;
every state is designed; text never sits on the canvas without an opaque
surface behind it, except the top row, console and folds, where the field
is held to a quarter of its brightness.

## Colour

Since 2026-10-06 the two UC Berkeley colours (brand.berkeley.edu) sit on
the same near-black ground as before, used sparingly: California Gold is
the one accent and Berkeley Blue is a quiet second tone for hairlines and
the secondary hover. The ground is never blue. Vervly is a student's
project and borrows the colours as a tribute; it uses no seal, no mascot
and no university mark, and every footer says it is not affiliated with or
endorsed by the University of California. Every pair below passes WCAG AA
on `--g2`; check any new pair before adding.

- Ground and surfaces, a ladder with Berkeley Blue in it, dark enough to
  read as black at a glance: `--g0 #06081a` (labels on the construction
  layer), `--g1 #0a0e20` (page and recessed inputs), `--g2 #10172f` (panes),
  `--g3 #171f3e` (hover), `--g4 #1f2a4f` (pressed, tooltips). Pane heads
  carry `--blue-wash` (.06) so the panes read as cards on the ground.
- Hairlines: Berkeley's Blue Light at low alpha, `--line
  rgba(159,209,255,.14)`, `--line-2 .24`.
- The quiet gold, `--gold-line rgba(253,181,21,.22)`: the example tag's
  border and a price column's hairline on hover. The landing's section
  headings each carry a 24px gold rule above, the eyebrow's mark repeated;
  the footer's hairline runs gold to blue like the horizon.
- Text: `--fg #ece8df`, `--fg-2 #b5b0a5`, `--fg-3 #8e897e`.
- The accent, California Gold in four strengths: `--accent #FDB515`,
  `--accent-deep #FC9313` (Gold Dark, pressed), `--accent-line
  rgba(253,181,21,.42)`, `--accent-wash .12`, text on it `--on-accent
  #010133`. The primary button's hover is Gold Medium `#FFC31B`. Used on:
  the primary button, focus rings, switches when on, the pill dot when
  connected, the start of the horizon hairline, the divider when hovered or
  dragged, the caret, the "in use" word on the Facts fold, the field and the
  tiles, the construction layer, the landing eyebrow and its rule, the
  cursor in the mark.
- Berkeley Blue, kept quiet: `--blue #002676` is ink on light grounds (the
  light lockup) only; `--blue-line rgba(159,209,255,.36)` is the middle of
  the horizon gradient, the secondary button's hover border and the Made in
  Berkeley hairline; `--blue-wash .07` is reserved for a selected row.
  Nothing else is blue.
- The verdict family, one lightness: `--human #9ad46a`, `--mixed #e8c25a`,
  `--ai #f08466`, as text and as the After pane's border after a run. Never
  a fill. Their washes at .14 mark inserted and deleted text.
- Sentence risk: `--risk-1` to `--risk-5`, the ai tone at .07 to .34 alpha.

No gradients except the risk legend swatch and the horizon hairline. No
glow, no shadow (the switch track has one inset hairline shadow so it reads
as a recess), no purple, no glass, no blobs. Dark only.

## Type

Berkeley's own web typefaces, all on Google Fonts: Inter, Source Serif 4
and, in a secondary role only, Barlow Condensed.

- Source Serif 4 for what people read: the draft and rewrite at 17.5px /
  1.7, 66ch; the wordmark, the lede, pane titles, the verdict line, the
  landing headline and the section and form headings. Weight 400 (500 for
  the wordmark). Optical size 8..60 is loaded so the headline and the body
  come from the right masters.
- Inter for controls, labels and hints: 14px controls, 13px hints, the
  progress title at 18px. Weights 400, 500 and 600.
- Barlow Condensed, `--cond`, for the landing eyebrow ("Built at UC
  Berkeley") only: 600, 18px, sentence case, 0.02em. Nowhere else yet; if
  it spreads, it stays secondary and never sets a headline or body text.
- Fira Mono, class `.num`, for every numeral and for the field, the tiles,
  the shortcut hint, the example tag and the construction labels.
- Scale: 12 / 13 / 14 / 16 / 18 / 22 px (`--t-0` to `--t-5`).
- No all caps, no letterspaced labels beyond the eyebrow's 0.02em, no
  accented word in a headline.

## Spacing and shape

- A 4px scale: `--s-1` 4, `--s-2` 8, `--s-3` 12, `--s-4` 16, `--s-5` 20,
  `--s-6` 24, `--s-8` 32, `--s-10` 40, `--s-12` 48. Use the variables; no
  literal 14px, 22px, 7px.
- One radius, `--r: 4px`, on panes, controls, chips, inputs, the error
  block, tooltips and construction boxes. The only round things are the
  switch (track and thumb), the pill dot and the stage rings, because their
  shape is their meaning.
- Hairlines separate rows and close panes. They do not frame things.

## Controls

Two button styles, `.btn` and `.btn .btn-primary`, both 36px, plus
`.btn-quiet` (no border at rest, second-tier text, for Sign out and the
landing's Sign in link). Every state is a different picture:

- rest: hairline `--line-2`, transparent;
- hover: border `--accent-line`, surface `--g3`;
- active: surface `--g4`, moved down 1px;
- focus-visible: 2px accent ring, 2px offset;
- disabled: text and border drop to `--fg-3` and `--line`, cursor
  not-allowed, no hover response.

The primary is a flat accent fill; hover lightens it, active uses
`--accent-deep`, disabled becomes `--g3` with `--fg-3` text. Chips are 32px
`.btn`-like buttons at 13px. Selects and text inputs are recessed (`--g1`)
with the same hover and focus borders.

The switch (`.switch`) is a `button[role=switch]` with `aria-checked`. Track
32 by 18, thumb 14, one inset hairline shadow off, accent track and
on-accent thumb on, thumb stretches 12 percent while pressed. Wrapped in
`label.switch-row` with a `.switch-label`. `wireSwitch()` in `app.js` is the
one place that toggles them. Two on the page: Construction in the top row
and "Rank with my own GPTZero key" under How it works, which reveals
`#own-key-field`. `userKey()` returns nothing while that switch is off, so
the key is never sent unless the switch is on. The switch state is
`humanizer.ownKey` in localStorage; a stored key with no stored switch state
counts as on.

The tooltip (`.tip[data-tip]`) is a CSS `::after` on the two readings
(`#v-before`, `#v-after`), which are focusable. It shows on hover and on
focus-visible after 80 ms, `--g4` surface, hairline, 13px, at most 34ch. The
text is set by `paintVerdict()` through `tipFor(judge, what)`: what the
number is and who judged it, naming the judge with `judgeName()`.

## The split

`.instrument` is a grid of three columns: `minmax(0, var(--col-before))`,
16px, `minmax(0, 1fr)`. `app.js` sets `--col-before` as
`calc((100% - 16px) * f)` where `f` is the Before fraction, 0.2 to 0.8.

- The divider `#divider` is `role=separator`, focusable, with
  `aria-valuenow` 0 to 100 and `aria-valuetext`. It is 16px wide: a
  hairline, a 4 by 32 grip, and three snap ticks that appear only while
  dragging.
- Drag with any pointer. Within 2 percent of 30, 50 or 70 the position
  snaps and that tick lights.
- Keyboard: Left and Right move 5 percent (Shift: 10), Home and End go to
  the limits, Enter or Space folds. Double click folds too.
- Folding: the smaller pane (Before at an even split) collapses to a 44px
  rail showing its title on its side; Enter, double click or a click on the
  rail brings the previous position back. `data-collapsed` on `.instrument`
  is `before`, `after` or empty.
- Motion: a fold, unfold or keyboard move glides in 280 ms through Motion
  One (`humanizer:split` with `{from, to}`; `anim.js` tweens the fraction
  and calls `window.readshuman.splitApply`). Without Motion the grid columns
  transition in CSS (`.is-gliding`). A drag paints directly. Reduced motion
  jumps.
- Under 960px the panes stack, the divider is a horizontal handle, and
  dragging or Up and Down set `--before-h`, the height of the draft body
  (120px to 80vh, snapping at 30, 50 and 70 vh). Enter folds Before to a
  44px bar.
- Stored as `readshuman.split` in localStorage: `{f, h, collapsed}`.

## The field (field.js)

- A 2D canvas, no library. 2600 points (1200 on screens under 700px, 800
  under 430) on three rings of a tilted disc, plus 340 (160, 100) far stars,
  projected each frame onto a grid of 10 by 16px cells. Each cell prints one
  glyph from the ramp `" .:-=+*#%@"` by how many points landed in it, at
  one of eight alphas from .16 to .67, all in the accent; a cell reaches the
  heavy glyphs at a count of 4.5. Glyphs come from an atlas drawn once, so a
  frame is one `drawImage` per occupied cell. The focal length is the short
  side of the viewport, or 72 percent of the width on a wide screen, so the
  disc reaches into the margins beside the shell.
- Behind the page column (the `.shell` rect, or an element marked
  `data-field-quiet`) the threshold rises and the level drops to 55 percent
  (the brightest glyph there is alpha .38), so text on the ground stays
  legible: third-tier text over a cell's average ink is 4.7:1, second-tier
  7.6:1. The field is at full strength only in the margins.
- Interaction: pointer parallax (camera offset), drag with a mouse or pen on
  the empty ground to turn the disc (inertia decays in about a second),
  click or tap the ground to send a pulse ring outward through the points
  (up to three at once; a pulse brightens and thickens the glyphs it
  crosses). Pointer events on a pane, the divider, a control, a fold body
  or the error block are left alone.
- Budget: 30 frames a second at most, device pixel ratio capped at 1.5, the
  loop stops while the tab is hidden, one still frame under reduced motion.
  It fades in over 600ms on its first frame. If the canvas fails the ground
  is plain.

## The tiles

Two small canvases with class `.ascii-tile`, drawn by `field.js` on the
same loop, in the accent at alphas .22 to .95 on the pane surface:

- `data-ascii="idle"`, 264 by 112, in the After pane's empty state: an
  ellipse of cells with a dither wave running round it.
- `data-ascii="run"`, 120 by 112, in the progress card: the same orbit,
  filled clockwise from the top by the run's progress. Done cells alternate
  `#` and `+`, the front is `@`, the rest `.`. When the run completes the
  centre prints `@`.
- `app.js` emits `humanizer:progress` with `{fraction, open}` on open (0),
  after every stage update (`progFraction()`, the mean over stages) and on
  finish (1). A tile only draws while it is visible. Reduced motion draws a
  frame per event.

## The construction layer

`#construction` is an absolutely positioned layer over the whole document,
pointer-events none, shown while the Construction switch is on or the
backtick key is held (not while typing in a field). `buildConstruction()` in
`app.js` fills it:

- an 8px grid in the accent at .07, and dashed lines at the shell's content
  edges;
- for every visible element with `data-c`, a hairline box, a mono tag with
  the region's name and its data path (`data-src`, drawn with a left arrow
  when the field fills the element and a right arrow, `data-flow="out"`,
  when the region sends), and the box's size when it is 100px or taller.
  Narrow boxes (under 160px, the divider) carry their tag outside to the
  right.
- Marked regions: top row, console, Before, After, divider, verdict,
  progress, both foots, the four folds. Add `data-c` and `data-src` to any
  new region; the layer documents itself.
- It rebuilds on resize, on fold toggles, and after `humanizer:verdict`,
  `humanizer:canvas`, `humanizer:analysis` and the progress events.
- Motion: the layer fades up in 160 ms and the boxes follow with a stagger
  capped so the whole reveal is inside 300 ms (`humanizer:construction`);
  hiding is one 160 ms fade. Without Motion it appears at once.

## Layout

- `.shell` is 1440px wide with a 32px gutter (16px under 600px), above the
  field on `z-index: 1`. It fills a 1440 screen edge to edge; on a wider
  screen the field runs at full strength in the margins.
- The top row's `.top-tools` hold the Construction switch, the service pill
  and the `.account` group; under 960px they wrap to the right, under 600px
  the switch sits left and the account right on the third line.
- Each `.pane` is a column flexbox: head (48px, title left, reading right),
  body (24px 32px padding, min 46vh), foot (count left, actions right).
- The before reading is `#v-before-p` (mono percent) and `#v-before-word`
  (`data-label` human, mixed, ai, none). The after reading is `#v-after`.
  `#verdict` is the instrument wrapper and carries `data-outcome`.
- The verdict line `#verdict-line` sits at the top of the after pane, one
  sentence, coloured by outcome. `#verdict-kept` and `#verdict-refused`
  follow it when they apply.
- The run checklist `#progress` fills the after pane body during a run.
- `#changed`, one line about the last run, is the after pane's foot count.
- Folded, in this order: Facts you can vouch for, How it works (rewrite
  strength, the own-key switch and its field), Details, How it works.

## Data flow (app.js)

- A finished run puts the rewrite in `state.humanize` and renders it into
  `#after` with `renderAfter()`. The draft in `#editor` is untouched. The
  run's before reading becomes `state.reading` for the draft.
- `analyzeAfter()` scores the rewrite's sentences once so the after pane can
  shade them; the verdict itself came with the run.
- Use this (`#use-btn`, `useRewrite()`) saves the draft to
  `state.preHumanize`, sets the editor to the rewrite, shows Undo and
  re-measures. Undo (`#undo-btn`, `undoHumanize()`) restores the saved draft;
  the rewrite stays on the right. Copy (`#copy-btn`) writes the rewrite to
  the clipboard.
- A reading is stale (`data-stale` on `.reading`) when the text it scored is
  not the text in the pane.
- Events for the modules: `humanizer:ready`, `humanizer:canvas`,
  `humanizer:verdict`, `humanizer:analysis`, `humanizer:progress-open`,
  `humanizer:progress`, `humanizer:progress-close`, `humanizer:split`,
  `humanizer:construction`.

## Motion

- Motion One 13.2.0 and AutoAnimate 0.10.0 from jsdelivr, pinned, 8s
  timeout. If they fail the page is static and complete.
- Two durations in CSS, 160ms and 260ms, one ease-out curve. In `anim.js`
  every element enters in 280ms or less.
- One page-load sequence: top row, console, the two panes together, the
  folds. Over in 0.6s. After that, motion only answers an action: a reading
  changing, the checklist opening, a list reflowing, a fold opening, the
  split gliding, the construction layer revealing.
- Nothing in the interface pulses or loops except the ring of a stage that
  is running with no reported progress, and the two tiles, which are the
  field's texture brought forward. `prefers-reduced-motion` turns interface
  motion off and freezes the field and the tiles.

## Copy

- Plain sentences you could say aloud. Headings are short declaratives.
- Name the judge with `judgeName()`: "GPTZero estimate" for the free local
  judge, "GPTZero" only when the service says `yardstick === 'gptzero'`.
- No em dashes or en dashes anywhere, including comments and AUDIT.md.
- No slogans, badges, eyebrow labels, sparkle icons, or the words seamless,
  unlock, transform, AI-powered. No claims the page cannot back.
- Errors say what happened and what to do next. Empty states say what to do.
- Actions keep their names through the flow: Humanize, Measure, Use this,
  Undo, Copy.

## Icons

None, apart from the CSS chevrons on folds and the SVG rings in the
checklist. If one is ever needed: one family, one stroke weight, inline.
