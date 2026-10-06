# Mobile: changes that need index.html, app.js or brand/footer.css

The mobile pass edited only styles.css, field.js and anim.js. The items
below need a file that pass could not touch. Each one is a copy-paste.

## 1. index.html: bump the cache keys

`?v=202609101109` on styles.css, anim.js and field.js (lines 21, 312, 313
or wherever they now sit) needs a new value so phones pick up the mobile
CSS and the touch changes in field.js and anim.js.

## 2. index.html, line 5: viewport with safe areas

styles.css now pads the top row, the gutters and the footer by
`env(safe-area-inset-*)`. Those values stay 0 until the viewport opts in.
Replace

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

with

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
```

## 3. index.html: footer fine print for fingers

`.site-foot-fine` tells a phone to click and hold the backtick key. styles.css
defines `.kb-only` (hidden when the device cannot hover) and `.touch-only`
(shown only then). Replace

```html
<p class="site-foot-fine">Vervly, 2026. Drag the field to turn it, click to send a pulse, hold the backtick key to see how the page is built.</p>
```

with

```html
<p class="site-foot-fine">Vervly, 2026. <span class="kb-only">Drag the field to turn it, click to send a pulse, hold the backtick key to see how the page is built.</span><span class="touch-only">Tap the field to send a pulse. The Construction switch in the top row shows how the page is built.</span></p>
```

The same applies to the Construction switch's `title` on line 40
("Holding the backtick key does the same."): harmless on touch, since
titles never show there, so no change is required.

## 4. brand/footer.css: the contact column overflows between 760 and about 1100px

At 1024px wide the three footer columns share one row with 112px gaps. The
contact column gets about 200px and the GitHub link
(`github.com/aniketbhattacharjee08-ui`, 259px at 16px) cannot break, so the
page grows to 1033px and scrolls sideways (this is what widened the
viewport in the 1024 x 768 screenshot). styles.css carries a stopgap,
`.shell .site-foot-col a { overflow-wrap: anywhere; }`, which lets the
address break mid-word rather than widen the page. The proper fix is in
footer.css; either

```css
.site-foot-col--contact { flex-basis: 280px; min-width: 280px; }
```

or stack the columns from 960px rather than 760px:

```css
@media (max-width: 960px) { /* was 760px */
```

Also in footer.css: the band's negative margins use `var(--gutter)` while
`.shell` now pads by `max(var(--gutter), env(safe-area-inset-left))`. On a
phone with a wide safe area the band stops short of the edge by the
difference. If that matters:

```css
.site-foot {
  margin-left: calc(-1 * max(var(--gutter), env(safe-area-inset-left, 0px)));
  margin-right: calc(-1 * max(var(--gutter), env(safe-area-inset-right, 0px)));
  padding-left: max(var(--gutter), env(safe-area-inset-left, 0px));
  padding-right: max(var(--gutter), env(safe-area-inset-right, 0px));
}
```

## 5. index.html, line 9: theme colour

`<meta name="theme-color" content="#0b1120">` is a blue that no longer
exists in the palette; the browser chrome on a phone shows it above the
page. Use the ground:

```html
<meta name="theme-color" content="#0e1013">
```

## 6. app.js, optional: keep the caret in view when the keyboard opens

Not required for the layout (the console sits above the draft in flow, and
`dvh` plus `scroll-margin` on `.canvas` handle the rest). If a device still
scrolls the caret under the keyboard, add after `wireSplit();` in `init`:

```js
    /* on a phone the keyboard covers the lower half; bring the draft up */
    if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) {
      editor.addEventListener('focus', function () {
        setTimeout(function () {
          if (window.visualViewport && window.visualViewport.height < window.innerHeight - 100) {
            $('pane-before').scrollIntoView({ block: 'start', behavior: reduceMotion.matches ? 'auto' : 'smooth' });
          }
        }, 300);
      });
    }
```

## Done without markup changes (for the record)

- Strength select was invisible at every width: `.field` (the canvas) and
  `label.field` shared a class, so the label was `position: fixed; opacity: 0`.
  styles.css now scopes the canvas rule to `#field`.
- Divider: 24px row under 960px with an invisible 44px hit band on coarse
  pointers (and a 44px wide band in the side-by-side layout).
- All buttons, chips, selects, inputs, fold summaries, Details rows and both
  switches reach 44px on coarse pointers. Inputs and selects are 16px so iOS
  does not zoom on focus.
- Phone top row: wordmark and pill, then the sentence, then the Construction
  switch with its label showing (the backtick key does not exist on a phone).
  Console stacks: Humanize grows beside Strength, judge and status below.
  Shortcut hint hidden wherever the device cannot hover.
- Landscape phones (height under 480px) drop the sentence and tighten.
- `dvh` for pane heights and `--before-h`, `scroll-margin` on the draft.
- field.js: 700 points on phones, 1000 on tablets, 24 fps cap, passive
  listeners, a moving finger never turns the disc or pans the camera, a
  still finger taps a pulse, loop stops on `pagehide`.
- anim.js: entrances and the construction reveal at 0.7x on screens under
  600px; reduced motion unchanged (everything off).
