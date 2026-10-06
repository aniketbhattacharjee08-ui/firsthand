/* Vervly animation layer.

   Progressive enhancement only. app.js renders the whole interface; this
   module animates what is already there. It switches itself off when either
   CDN module fails to load, when the user asks for reduced motion, or when
   anything in here throws. In every case the page is static, never broken.

   Motion       https://cdn.jsdelivr.net/npm/motion@13.2.0/+esm
   AutoAnimate  https://cdn.jsdelivr.net/npm/@formkit/auto-animate@0.10.0/+esm
   Versions are pinned; bump them on purpose, not by accident. */

const root = document.documentElement;
const reduce = window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)')
  : { matches: false, addEventListener() {} };

/* ease-out only, no overshoot; every entrance finishes within 300 ms. On a
   phone everything is a step quicker: less travels, so less should move. */
const EASE = [0.22, 1, 0.36, 1];
const smallScreen = !!(window.matchMedia && window.matchMedia('(max-width: 600px)').matches);
const K = smallScreen ? 0.7 : 1;
const D = { fast: 0.16 * K, base: 0.22 * K, slow: 0.28 * K };
const RISE = smallScreen ? 6 : 10;   /* the panes' entrance travel, in px */

const STAGED = '.top, .console, .pane, .fold';

/* Mark a node as shown. Any animation still running on it is jumped to its
   end first, so a throttled tab or a stalled timeline can never leave a node
   stuck at its first keyframe. */
function settle(nodes) {
  nodes.forEach((n) => {
    if (typeof n.getAnimations === 'function') {
      n.getAnimations().forEach((a) => { try { a.finish(); } catch (e) { try { a.cancel(); } catch (e2) { /* gone */ } } });
    }
    n.setAttribute('data-anim-done', '1');
    n.style.opacity = '';
    n.style.transform = '';
  });
}

function standDown(why) {
  try {
    root.removeAttribute('data-anim');
    settle(document.querySelectorAll(STAGED + ', #stages .stage, .reading-word, #construction, .c-box'));
    if (why) console.info('[humanizer] animation layer off:', why);
  } catch (e) { /* nothing left to do */ }
}

/* Motion 10 spelled the easing option `easing`; 11 and later spell it `ease`.
   Passing both is harmless. */
function opts(extra) {
  return Object.assign({ ease: EASE, easing: EASE }, extra || {});
}

let failsafe = null;
function armFailsafe() { failsafe = setTimeout(() => standDown('entrance did not complete in time'), 1800); }
function clearFailsafe() { if (failsafe) { clearTimeout(failsafe); failsafe = null; } }

if (reduce.matches) standDown('prefers-reduced-motion: reduce');
else boot();

if (reduce.addEventListener) {
  reduce.addEventListener('change', (e) => { if (e.matches) standDown('reduced motion switched on'); });
}
window.addEventListener('error', (e) => {
  if (e && e.filename && e.filename.indexOf('anim.js') >= 0) standDown('runtime error in the animation layer');
});

async function boot() {
  let animate, stagger, autoAnimate;
  try {
    const [motion, aa] = await Promise.race([
      Promise.all([
        import('https://cdn.jsdelivr.net/npm/motion@13.2.0/+esm'),
        import('https://cdn.jsdelivr.net/npm/@formkit/auto-animate@0.10.0/+esm')
      ]),
      new Promise((_, reject) => setTimeout(() => reject(new Error('CDN import timed out after 8s')), 8000))
    ]);
    animate = motion.animate;
    stagger = motion.stagger;
    autoAnimate = aa.default || aa.autoAnimate || aa;
    if (typeof animate !== 'function' || typeof autoAnimate !== 'function') {
      throw new Error('unexpected module shape');
    }
    if (typeof stagger !== 'function') stagger = (s) => (i) => i * s;
  } catch (err) {
    standDown('CDN modules unavailable (' + (err && err.message) + ')');
    return;
  }

  root.setAttribute('data-anim', 'on');
  armFailsafe();

  try {
    entrance(animate, stagger);
    lists(autoAnimate);
    verdict(animate);
    checklist(animate, stagger);
    sentences(animate, stagger);
    folds(animate);
    split(animate);
    construction(animate, stagger);
  } catch (err) {
    standDown('setup threw: ' + (err && err.message));
    return;
  }
  clearFailsafe();
}

/* 1. one page-load sequence, top to bottom: the top row, the console, the
   two panes together, then the folds. No single element takes longer than
   300 ms; the whole thing is over in 0.6 s. */
function entrance(animate, stagger) {
  const top = document.querySelector('.top');
  const console_ = document.querySelector('.console');
  const panes = Array.from(document.querySelectorAll('.pane'));
  const folds = Array.from(document.querySelectorAll('.fold'));
  const all = [top, console_].filter(Boolean).concat(panes, folds);
  settle(all);

  if (top) animate(top, { opacity: [0, 1] }, opts({ duration: D.base }));
  if (console_) animate(console_, { opacity: [0, 1], y: [6 * K, 0] }, opts({ duration: D.base, delay: 0.06 * K }));
  if (panes.length) {
    animate(panes, { opacity: [0, 1], y: [RISE, 0] }, opts({ duration: D.slow, delay: stagger(0.06 * K, { startDelay: 0.12 * K }) }));
  }
  if (folds.length) {
    animate(folds, { opacity: [0, 1] }, opts({ duration: D.base, delay: stagger(0.04 * K, { startDelay: 0.3 * K }) }));
  }
  setTimeout(() => settle(all), 1400);
}

/* 2. lists that reflow on every re-analysis */
function lists(autoAnimate) {
  ['findings', 'measurements', 'flagged', 'edits', 'run-summary', 'style-signals', 'refused-list'].forEach((id) => {
    const node = document.getElementById(id);
    if (node) autoAnimate(node, { duration: 220, easing: 'ease-out' });
  });
}

/* 3. a reading changes: the new word rises in; the verdict line refreshes */
function verdict(animate) {
  const words = Array.from(document.querySelectorAll('.reading-word'));
  const last = new Map(words.map((w) => [w, w.textContent]));
  document.addEventListener('humanizer:verdict', () => {
    words.forEach((w) => {
      if (w.hidden || (w.parentElement && w.parentElement.hidden)) return;
      if (last.get(w) === w.textContent) return;
      last.set(w, w.textContent);
      animate(w, { opacity: [0, 1], y: [6, 0] }, opts({ duration: D.base }));
    });
    const line = document.getElementById('verdict-line');
    if (line && line.textContent) animate(line, { opacity: [0.4, 1] }, opts({ duration: D.base }));
  });
}

/* 4. the run checklist rises in when a run starts. Ring fill is a CSS
   transition, so it glides with or without this module. */
function checklist(animate, stagger) {
  document.addEventListener('humanizer:progress-open', () => {
    const card = document.querySelector('.progress-card');
    const rows = Array.from(document.querySelectorAll('#stages .stage'));
    if (card) animate(card, { opacity: [0, 1] }, opts({ duration: D.fast }));
    if (rows.length) {
      settle(rows);
      animate(rows, { opacity: [0, 1], y: [6, 0] }, opts({ duration: D.base, delay: stagger(0.035, { startDelay: 0.1 }) }));
      setTimeout(() => settle(rows), 1200);
    }
  });
}

/* 5. per sentence shading fades in after a measurement, in either pane */
function sentences(animate, stagger) {
  const paint = () => {
    const spans = Array.from(document.querySelectorAll('.canvas .sent'));
    if (!spans.length) return;
    animate(spans.slice(0, 80), { opacity: [0.4, 1] }, opts({ duration: D.base, delay: stagger(0.006) }));
    setTimeout(() => spans.forEach((s) => { s.style.opacity = ''; }), 900);
  };
  document.addEventListener('humanizer:canvas', paint);
}

/* 6. disclosures: the body fades in when opened */
function folds(animate) {
  document.querySelectorAll('details.fold').forEach((d) => {
    d.addEventListener('toggle', () => {
      if (!d.open) return;
      const body = d.querySelector('.fold-body');
      if (body) animate(body, { opacity: [0, 1], y: [-4, 0] }, opts({ duration: D.fast }));
    });
  });
}

/* 7. the split glides when a pane folds or unfolds, or the keyboard moves the
   divider. app.js exposes the painter; this tweens the fraction. A drag never
   comes through here: it paints directly. */
function split(animate) {
  let current = null;
  document.addEventListener('humanizer:split', (e) => {
    const d = e.detail || {};
    const paint = window.readshuman && window.readshuman.splitApply;
    if (typeof paint !== 'function' || typeof d.from !== 'number' || typeof d.to !== 'number') return;
    if (current) { try { current.stop(); } catch (err) { /* already done */ } }
    try {
      current = animate(d.from, d.to, opts({ duration: D.slow, onUpdate: (v) => paint(v) }));
      if (current && current.finished) current.finished.then(() => paint(d.to)).catch(() => paint(d.to));
    } catch (err) {
      paint(d.to);
    }
  });
}

/* 8. the construction layer: the sheet fades up and the boxes follow in
   order, the whole reveal inside 300 ms. Hiding is one fade. */
function construction(animate, stagger) {
  document.addEventListener('humanizer:construction', (e) => {
    const layer = document.getElementById('construction');
    if (!layer) return;
    const on = !!(e.detail && e.detail.on);
    if (on) {
      /* Slower than the rest of the page on purpose (the owner asked for it):
         the sheet settles in over 0.7 s and the boxes fall into place from
         above, top of the page first, so the reveal reads as a drawing being
         laid down rather than a flash. Still ease-out, still one direction. */
      const boxes = Array.from(layer.querySelectorAll('.c-box'))
        .sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
      animate(layer, { opacity: [0, 1] }, opts({ duration: 0.7 * K }));
      if (boxes.length) {
        const step = Math.min(0.06 * K, 0.9 * K / boxes.length);
        animate(boxes, { opacity: [0, 1], y: [-14 * K, 0] }, opts({ duration: 0.6 * K, delay: stagger(step, { startDelay: 0.1 * K }) }));
      }
    } else {
      const boxes = Array.from(layer.querySelectorAll('.c-box'));
      if (boxes.length) animate(boxes, { opacity: [1, 0], y: [0, 10 * K] }, opts({ duration: 0.4 * K }));
      animate(layer, { opacity: [1, 0] }, opts({ duration: 0.5 * K, delay: 0.1 * K }));
    }
  });
}
