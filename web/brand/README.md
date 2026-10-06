# Vervly brand files

## The idea

The mark is the product's own before and after, drawn in the cells the page already uses: above, a draft chopped into four even pieces; below, one continuous line with a cursor at its end. The even pieces stand for machine cadence and the continuous line for a sentence a person wrote, so the mark says what the product does without a brain, a bubble or a swoosh. Everything is rectangles on a 16 unit grid, which is why it stays crisp at 16 pixels and why it matches the ASCII field, the hairlines and the construction layer rather than sitting on top of them. The name is set once, in Fira Sans Medium at the page's wordmark size, tracked -0.01em, with no colour or weight change between Reads and Human; the only accent in the whole lockup is the cursor. The three other sketches (a coarse pixel R, the pane divider with its grip, a ragged three line paragraph) were drawn and rejected: the R read as retro, the divider turned to noise at 16 pixels, the paragraph was a generic text icon.

## Files

- `mark.svg`: the mark alone, 16 unit viewBox, `fill="currentColor"` so it takes the text colour of wherever it is placed; the cursor is always California Gold `#FDB515`. Use inline in HTML at 16, 20 or 24 pixels, or as an `img` at 32 and 64.
- `wordmark.svg`: the horizontal lockup, mark at 24 pixels and the name as paths (166 by 24 units). `currentColor` ink. This is the file for the page, documents and anything that can pick a colour.
- `wordmark-text.svg`: the same lockup with the name as live text. It needs Fira Sans loaded, so it is only an editable source; ship `wordmark.svg`.
- `logo-dark.svg`: the lockup with fixed white ink (`#f3f4f8`) for Berkeley Blue and other dark grounds, transparent background.
- `logo-light.svg`: the lockup with fixed Berkeley Blue ink (`#002676`) for white and pale grounds, transparent background.
- `favicon.svg`: the mark at 32 units on a `#071d4f` tile with a 4 unit radius. Link it with `<link rel="icon" type="image/svg+xml" href="brand/favicon.svg">`; export 32 and 180 pixel PNGs from it if `.ico` or Apple touch icons are needed.
- `footer.html` and `footer.css`: the site footer snippet and its styles, on the tokens in `styles.css`.
- `preview.png`: the mark at 16, 32 and 64, the lockups on dark and light, and the footer wide and narrow.

## Rules

- Minimum size: the mark 16 pixels square; the lockup 24 pixels tall (the name is then 22 pixels, matching the top row). Below that use the mark alone.
- Clear space: leave at least the mark's own width (one unit of 16 at 16 pixels, 24 at 24) on every side of the lockup and the mark. Nothing else may sit inside that space.
- Grid: place the mark on whole pixels at 16, 32 or 64 pixels, or at multiples of 8, so the cells stay sharp. Do not scale to sizes like 20 or 28 for the favicon.
- Colour: ink is the surrounding text colour (`--fg` on the blue, `#002676` on light); the cursor is `--accent`, California Gold, and never changes. No other colour, no gradient, no outline, no shadow.
- Do not rotate the mark, place it in a circle or rounded square other than the favicon tile, separate the cursor from the line, or set the name in another face or with the two words in different weights.
