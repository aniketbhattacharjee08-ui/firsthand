/* ══════════════════════════════════════════════════════════════════════════
   humanizer — animation layer.

   PROGRESSIVE ENHANCEMENT ONLY. app.js renders the entire interface on its
   own; this module does nothing but animate what is already there. Every
   import is wrapped, every entrance has a failsafe that forces the final
   state, and the whole layer switches itself off when:

     · either CDN module fails to load (offline, blocked, CSP, 404),
     · the user asks for reduced motion,
     · anything in here throws.

   In all of those cases the page is simply static, never broken and never
   invisible. That is the contract.

   Libraries, both ESM from a CDN because this project has no build step:
     Motion One  — https://esm.sh/motion@10
     AutoAnimate — https://esm.sh/@formkit/auto-animate@0.8
   ══════════════════════════════════════════════════════════════════════════ */

const root = document.documentElement;
const reduce = window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)')
  : { matches: false, addEventListener() {} };

/* calm, no overshoot: a decelerating ease, nothing springy */
const EASE = [0.22, 1, 0.36, 1];
const D = { quick: 0.15, base: 0.28, slow: 0.45 };

/* ── the escape hatch, callable from anywhere in here ───────────────────── */
function standDown(why) {
  try {
    root.removeAttribute('data-anim');
    revealAll();
    if (why) console.info('[humanizer] animation layer off:', why);
  } catch (e) { /* nothing left to do */ }
}

/* force every animated element to its final, visible state */
function revealAll() {
  document.querySelectorAll('.panel, .empty-state > *').forEach((n) => {
    n.setAttribute('data-anim-done', '1');
    n.style.opacity = '';
    n.style.transform = '';
  });
  document.querySelectorAll('#editor .sent').forEach((n) => { n.style.opacity = ''; });
}

/* A last-resort timer, ARMED ONLY once data-anim is on. Before that nothing is
   hidden — the CSS that sets opacity:0 is scoped to html[data-anim="on"] — so a
   slow CDN cannot leave the page blank, and the timer must not race the import. */
let failsafe = null;
function armFailsafe() { failsafe = setTimeout(() => standDown('entrance did not complete in time'), 1600); }
function clearFailsafe() { if (failsafe) { clearTimeout(failsafe); failsafe = null; } }

if (reduce.matches) {
  standDown('prefers-reduced-motion: reduce');
} else {
  boot();
}

/* re-evaluate if the user flips the OS setting mid-session */
if (reduce.addEventListener) {
  reduce.addEventListener('change', (e) => { if (e.matches) standDown('reduced motion switched on'); });
}
window.addEventListener('error', (e) => {
  if (e && e.filename && e.filename.indexOf('anim.js') >= 0) standDown('runtime error in the animation layer');
});

async function boot() {
  let animate, stagger, inView, autoAnimate;
  try {
    /* two independent imports: a failure in either stands the layer down
       rather than half-animating the page */
    /* a generous ceiling: a cold CDN fetch can take seconds, and nothing is
       hidden while we wait, so patience here costs the user nothing */
    const [motion, aa] = await Promise.race([
      Promise.all([
        import('https://esm.sh/motion@10'),
        import('https://esm.sh/@formkit/auto-animate@0.8')
      ]),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('CDN import timed out after 8s')), 8000))
    ]);
    ({ animate, stagger, inView } = motion);
    autoAnimate = aa.default || aa;
    if (typeof animate !== 'function' || typeof autoAnimate !== 'function') {
      throw new Error('unexpected module shape');
    }
  } catch (err) {
    standDown('CDN modules unavailable (' + (err && err.message) + ')');
    return;
  }

  /* from here on the entrance CSS applies, so the failsafe becomes relevant */
  root.setAttribute('data-anim', 'on');
  armFailsafe();

  try {
    entrances(animate, stagger, inView);
    lists(autoAnimate);
    dials(animate);
    sentences(animate, stagger);
    rings(animate, stagger);
  } catch (err) {
    standDown('setup threw: ' + (err && err.message));
    return;
  }
  clearFailsafe();
}

/* ── 1. panels and the empty state rise in, gently staggered ────────────── */
function entrances(animate, stagger, inView) {
  const settle = (nodes) => nodes.forEach((n) => n.setAttribute('data-anim-done', '1'));

  const intro = Array.from(document.querySelectorAll('.empty-state > *'));
  if (intro.length) {
    settle(intro);
    animate(intro, { opacity: [0, 1], transform: ['translateY(12px)', 'none'] },
      { duration: D.slow, delay: stagger(0.055), easing: EASE });
  }

  /* the rail is taller than the viewport, so its sections reveal on scroll
     rather than all at once off screen */
  const panels = Array.from(document.querySelectorAll('.rail .panel'));
  panels.forEach((panel, i) => {
    const run = () => {
      if (panel.getAttribute('data-anim-done') === '1') return;
      panel.setAttribute('data-anim-done', '1');
      animate(panel, { opacity: [0, 1], transform: ['translateY(10px)', 'none'] },
        { duration: D.base, delay: Math.min(i, 3) * 0.05, easing: EASE });
    };
    /* the first few are above the fold; the rest wait for inView */
    if (i < 3 || typeof inView !== 'function') run();
    else inView(panel, run, { margin: '0px 0px -12% 0px' });
  });

  /* whichever ones inView never fires for still become visible */
  setTimeout(() => settle(panels), 2500);
}

/* ── 2. AutoAnimate the two lists that reflow on every re-analysis ──────── */
function lists(autoAnimate) {
  ['findings', 'measurements', 'detector-list', 'flagged'].forEach((id) => {
    const node = document.getElementById(id);
    if (node) autoAnimate(node, { duration: 220, easing: 'ease-out' });
  });
}

/* ── 3. dial readouts count up; the arc sweep is a CSS transition ───────── */
function dials(animate) {
  const tweens = new WeakMap();

  window.__hzAnim = {
    countTo(node, from, to) {
      const prev = tweens.get(node);
      if (prev && prev.cancel) { try { prev.cancel(); } catch (e) { /* done already */ } }
      const a = animate(
        (progress) => { node.textContent = String(Math.round(from + (to - from) * progress)); },
        { duration: 0.4, easing: EASE }
      );
      tweens.set(node, a);
      if (a && a.finished && a.finished.catch) a.finished.catch(() => {});
      return a;
    }
  };

  /* a small settle on the arc each time a fresh reading lands */
  document.addEventListener('humanizer:analysis', () => {
    document.querySelectorAll('.gauge-fill').forEach((arc) => {
      animate(arc, { opacity: [0.55, 1] }, { duration: D.base, easing: EASE });
    });
  });
}

/* ── 4. per-sentence highlights fade in progressively ───────────────────── */
function sentences(animate, stagger) {
  const paint = () => {
    const spans = Array.from(document.querySelectorAll('#editor .sent'));
    if (!spans.length) return;
    /* only the visible ones: staggering 400 spans would be slow and pointless */
    const budget = spans.slice(0, 60);
    animate(budget, { opacity: [0.35, 1] },
      { duration: D.base, delay: stagger(0.008), easing: EASE });
    /* whatever is left is simply not animated, and therefore already correct */
    setTimeout(() => spans.forEach((s) => { s.style.opacity = ''; }), 900);
  };
  document.addEventListener('humanizer:canvas', paint);
  if (window.__hzReady) paint();
}

/* ── 5. the humanize stage checklist ────────────────────────────────────
   app.js draws and drives the rings entirely on its own by writing
   strokeDashoffset directly. This only makes the fill glide between two
   values instead of jumping, and stands the whole thing down the same way
   everything else here does. If this module never loads, the rings still
   fill; they just snap.

   Reduced motion never reaches here: app.js checks the media query itself
   before asking for a tween, and standDown() has already run. */
function rings(animate, stagger) {
  const tweens = new WeakMap();

  const prev = window.__hzAnim || {};
  window.__hzAnim = Object.assign(prev, {
    ringTo(node, from, to) {
      const running = tweens.get(node);
      if (running && running.cancel) { try { running.cancel(); } catch (e) { /* finished */ } }
      const a = animate(node, { strokeDashoffset: [from, to] },
        { duration: 0.42, easing: EASE });
      tweens.set(node, a);
      if (a && a.finished && a.finished.catch) a.finished.catch(() => {});
      /* app.js has already written `to` to the inline style, so a cancelled
         or failed tween lands on the right number rather than a stale one */
      return a;
    }
  });

  /* the eight rows arrive together, so they are staggered in once */
  document.addEventListener('humanizer:progress-open', () => {
    const rows = Array.from(document.querySelectorAll('#stages .stage'));
    if (!rows.length) return;
    animate(rows, { opacity: [0, 1], transform: ['translateY(6px)', 'none'] },
      { duration: D.base, delay: stagger(0.03), easing: EASE });
    /* a failsafe of its own: nothing in this panel may stay invisible */
    setTimeout(() => rows.forEach((r) => { r.style.opacity = ''; r.style.transform = ''; }), 900);
  });
}
