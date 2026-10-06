/* Vervly field.

   The background behind the instrument, drawn as ASCII: a tilted orbital
   system of a few thousand points is projected onto a grid of character
   cells and each cell prints the glyph for how many points landed in it,
   from " .:-=+*#%@". Everything is one hue, the accent. Full strength in
   the margins, about half behind the page column.

   The same module serves the landing and the sign-in pages: it looks for
   #field and, optionally, .shell and .ascii-tile, and does nothing it
   cannot find. Pointer events on anything marked data-solid are left alone.

   Move the pointer for parallax, drag with a mouse or pen on the empty
   ground to turn the system, click or tap the ground to send a pulse
   through it. A finger never turns the disc or moves the camera: on a
   phone a moving touch is a scroll and is left to the page. The loop runs
   at 30 frames a second at most (24 on a phone or tablet, with fewer
   points), stops while the tab is hidden, and draws one still frame under
   reduced motion.

   The same module drives the two small ASCII tiles in the After pane: the
   idle tile is a slowly turning orbit; the run tile fills that orbit with
   the progress app.js reports on "humanizer:progress".

   Nothing here is required. app.js does not know this file exists. */

const canvas = document.getElementById('field');
/* data-still on the canvas: draw the field once and leave it. The disc does
   not turn, the pointer does not tilt it, clicks send no pulse; the two
   ASCII tiles keep animating because they show real state. */
const STILL = !!(canvas && canvas.hasAttribute('data-still'));
const reduce = window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)')
  : { matches: false, addEventListener() {} };

/* the accent from styles.css, as rgb */
const ACCENT = '253, 181, 21';
const RAMP = ' .:-=+*#%@';

/* a phone or a tablet gets fewer points and a lower frame cap: the field is
   texture there, and the battery matters more than glyph density */
const coarse = !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
const shortSide = Math.min(window.innerWidth, window.innerHeight);
const small = shortSide < 700 || (coarse && shortSide < 900);
const tiny = shortSide < 430;
const COUNT_RING = tiny ? 800 : small ? 1200 : 2600;
const COUNT_HALO = tiny ? 100 : small ? 160 : 340;
const DPR = Math.min(window.devicePixelRatio || 1, 1.5);
const CELL_W = 10, CELL_H = 16, FONT = '12px "Fira Mono", ui-monospace, Menlo, monospace';
const FPS_CAP = small ? 24 : 30;

/* pointer events that land on a control belong to the control */
const INTERACTIVE = '.pane, .divider, button, input, select, textarea, summary, a, [contenteditable], .fold-body, .error, .switch-row, [data-solid]';

const tiles = [];
let fieldFrame = null;       /* the field's per-frame function, once built */
let running = false, raf = 0, lastT = 0;

boot();

function boot() {
  try { fieldFrame = asciiField(); } catch (err) {
    console.info('[humanizer] field: could not draw (' + (err && err.message) + ')');
    fieldFrame = null;
  }
  if (STILL && fieldFrame) fieldFrame(performance.now(), 0);
  try { document.querySelectorAll('.ascii-tile').forEach((c) => tiles.push(tile(c))); } catch (e) { /* tiles are optional */ }
  document.addEventListener('humanizer:progress', (e) => {
    const f = e && e.detail && typeof e.detail.fraction === 'number' ? e.detail.fraction : 0;
    tiles.forEach((t) => t.progress(f));
  });
  if (reduce.matches) { still(); return; }
  start();
  /* the loop stops while the tab is hidden or the page is put away (a phone
     switching apps fires pagehide before visibilitychange in some engines) */
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else start(); });
  window.addEventListener('pagehide', stop);
  window.addEventListener('pageshow', () => { if (!document.hidden) start(); });
  if (reduce.addEventListener) reduce.addEventListener('change', (e) => { if (e.matches) { stop(); still(); } else start(); });
}

function still() {
  const now = performance.now();
  if (fieldFrame) fieldFrame(now, 0.016);
  tiles.forEach((t) => t.frame(now, 0.016));
}
function start() {
  if (running || reduce.matches) return;
  running = true; lastT = performance.now();
  raf = requestAnimationFrame(loop);
}
function stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
function loop(now) {
  raf = requestAnimationFrame(loop);
  let dt = (now - lastT) / 1000;
  if (dt < 1 / (FPS_CAP + 2)) return;
  lastT = now;
  if (dt > 0.1) dt = 0.1;
  if (fieldFrame && !STILL) fieldFrame(now, dt);
  tiles.forEach((t) => t.frame(now, dt));
}

/* ── the glyph atlas: every ramp glyph at eight alphas, drawn once ─────── */

function atlas(cw, ch, levels, color, lo, hi) {
  const a = document.createElement('canvas');
  a.width = Math.ceil(cw * DPR) * RAMP.length;
  a.height = Math.ceil(ch * DPR) * levels;
  const g = a.getContext('2d');
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
  g.font = FONT;
  g.textBaseline = 'middle';
  g.textAlign = 'center';
  for (let l = 0; l < levels; l++) {
    const alpha = lo + (l / (levels - 1)) * (hi - lo);
    g.fillStyle = 'rgba(' + color + ', ' + alpha.toFixed(3) + ')';
    for (let i = 1; i < RAMP.length; i++) g.fillText(RAMP[i], i * cw + cw / 2, l * ch + ch / 2);
  }
  return { img: a, sw: Math.ceil(cw * DPR), sh: Math.ceil(ch * DPR), levels };
}

/* ── the orbital system, as characters ─────────────────────────────────── */

function asciiField() {
  if (!canvas) return null;
  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) return null;

  const LEVELS = 8;
  /* the ladder: faint stars at .16, the brightest glyph at .67. In the margins
     nothing reads over it; behind the page column it is held down (see frame) */
  const glyphs = atlas(CELL_W, CELL_H, LEVELS, ACCENT, 0.16, 0.67);

  /* points on three dense rings, in system space (z near 0) */
  const bands = [
    { r: 1.15, w: 0.12, share: 0.30 },
    { r: 1.95, w: 0.16, share: 0.42 },
    { r: 2.75, w: 0.10, share: 0.28 }
  ];
  const px = new Float32Array(COUNT_RING), py = new Float32Array(COUNT_RING), pz = new Float32Array(COUNT_RING);
  const phase = new Float32Array(COUNT_RING), tone = new Float32Array(COUNT_RING);
  let i = 0;
  bands.forEach((b, bi) => {
    const n = bi === bands.length - 1 ? COUNT_RING - i : Math.floor(COUNT_RING * b.share);
    for (let k = 0; k < n; k++) {
      const a = Math.random() * Math.PI * 2;
      const r = b.r + gauss() * b.w;
      px[i] = Math.cos(a) * r; py[i] = Math.sin(a) * r; pz[i] = gauss() * 0.03;
      phase[i] = Math.random() * Math.PI * 2;
      tone[i] = 0.6 + Math.random() * 0.6;
      i++;
    }
  });
  /* far stars: fixed on the screen, shifted a little by the pointer */
  const halo = [];
  for (let k = 0; k < COUNT_HALO; k++) halo.push({ x: Math.random(), y: Math.random(), d: 0.3 + Math.random() * 0.7, ph: Math.random() * 6.28 });

  const sys = { rx: 1.02, rz: -0.22, y: 0.55 };
  const cam = { x: 0, y: 0.35, z: 4.6 };
  const look = { x: 0, y: 0, tx: 0, ty: 0 };
  const spin = { vz: 0, vx: 0, dragging: false, lastX: 0, lastY: 0, downX: 0, downY: 0, downAt: 0, moved: false, id: null };
  const pulses = [];
  let pulseSlot = 0;
  let t = 0;

  let W = 0, H = 0, cols = 0, rows = 0, density = null, boost = null;
  let colL = 0, colR = 0;                 /* the page column, in cells: quieter there */
  function resize() {
    W = window.innerWidth; H = window.innerHeight;
    /* the page column: .shell on the app and the landing; a page without a
       shell (the sign-in card) marks its own column with data-field-quiet */
    const shell = document.querySelector('.shell, [data-field-quiet]');
    if (shell) {
      const r = shell.getBoundingClientRect();
      colL = Math.floor(r.left / CELL_W); colR = Math.ceil(r.right / CELL_W);
    } else { colL = 0; colR = 0; }
    canvas.width = Math.floor(W * DPR); canvas.height = Math.floor(H * DPR);
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    cols = Math.ceil(W / CELL_W); rows = Math.ceil(H / CELL_H);
    density = new Float32Array(cols * rows);
    boost = new Float32Array(cols * rows);
  }
  resize();
  /* a phone fires resize as its address bar comes and goes and as the
     keyboard opens; only rebuild the grid when the size really changed,
     and at most once a frame */
  let resizeRaf = 0;
  window.addEventListener('resize', () => {
    if (resizeRaf) return;
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = 0;
      if (window.innerWidth === W && window.innerHeight === H) return;
      resize();
      if (!running || STILL) frame(performance.now(), 0);
    });
  }, { passive: true });

  /* the projection's focal length: the short side, or on a wide screen a
     share of the width, so the disc reaches into the margins beside the shell */
  function focal() { return Math.max(Math.min(W, H) * 1.05, W * 0.72); }

  function frame(now, dt) {
    t += dt;
    if (!reduce.matches) {
      if (!spin.dragging) {
        sys.rz -= 0.018 * dt + spin.vz * dt;
        sys.rx = clamp(sys.rx + spin.vx * dt, 0.55, 1.45);
        spin.vz *= Math.pow(0.08, dt);
        spin.vx *= Math.pow(0.08, dt);
      }
      look.x += (look.tx - look.x) * Math.min(1, dt * 4);
      look.y += (look.ty - look.y) * Math.min(1, dt * 4);
      cam.x = look.x * 0.32;
      cam.y = 0.35 + look.y * 0.22;
    }

    density.fill(0); boost.fill(0);
    const F = focal();
    const cx = W / 2, cy = H / 2;
    const cz = Math.cos(sys.rz), sz = Math.sin(sys.rz);
    const cxr = Math.cos(sys.rx), sxr = Math.sin(sys.rx);

    for (let k = 0; k < COUNT_RING; k++) {
      /* spin in the disc plane, then tilt, then raise */
      const x0 = px[k] * cz - py[k] * sz;
      const y0 = px[k] * sz + py[k] * cz;
      const z0 = pz[k];
      const y1 = y0 * cxr - z0 * sxr;
      const z1 = y0 * sxr + z0 * cxr;
      const X = x0 - cam.x, Y = y1 + sys.y - cam.y, Z = cam.z - z1;
      if (Z < 0.5) continue;
      const s = F / Z;
      const sx = cx + X * s, sy = cy - Y * s;
      if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
      const c = ((sy / CELL_H) | 0) * cols + ((sx / CELL_W) | 0);
      let w = tone[k] * (0.82 + 0.18 * Math.sin(t * 0.6 + phase[k]));
      let p = 0;
      for (let q = 0; q < pulses.length; q++) {
        const age = t - pulses[q].t;
        if (age <= 0 || age > 3.2) continue;
        const dx = px[k] - pulses[q].x, dy = py[k] - pulses[q].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        const band = 1 - Math.min(1, Math.abs(d - age * 1.5) / 0.4);
        p += band * (1 - age / 3.2);
      }
      density[c] += w + p * 2.2;
      if (p > boost[c]) boost[c] = p;
    }

    ctx.clearRect(0, 0, W, H);
    /* far stars, always the faintest glyph */
    const hx = look.x * 6, hy = -look.y * 6;
    for (let k = 0; k < halo.length; k++) {
      const h = halo[k];
      const sx = ((h.x * W + hx * h.d) % W + W) % W, sy = ((h.y * H + hy * h.d) % H + H) % H;
      const tw = 0.5 + 0.5 * Math.sin(t * 0.4 + h.ph);
      const lvl = tw > 0.7 ? 1 : 0;
      ctx.drawImage(glyphs.img, glyphs.sw, lvl * glyphs.sh, glyphs.sw, glyphs.sh,
        ((sx / CELL_W) | 0) * CELL_W, ((sy / CELL_H) | 0) * CELL_H, CELL_W, CELL_H);
    }
    /* the disc: one glyph per occupied cell */
    for (let c = 0; c < density.length; c++) {
      const d = density[c];
      const col = c % cols, row = (c / cols) | 0;
      const under = col >= colL && col < colR;   /* behind the page column */
      if (d < (under ? 0.7 : 0.25)) continue;
      /* a denser ramp: a cell reaches the heavy glyphs at a lower count */
      const u = Math.min(1, d / 4.5);
      const gi = 1 + Math.min(RAMP.length - 2, Math.floor(u * (RAMP.length - 1)));
      let lvl = Math.floor((0.25 + u * 0.55 + boost[c] * 0.4) * LEVELS);
      /* behind the page column the glyphs drop to about half so the text that
         sits on the ground (top row, console, folds) stays legible */
      if (under) lvl = Math.floor(lvl * 0.55);
      lvl = Math.min(LEVELS - 1, lvl);
      ctx.drawImage(glyphs.img, gi * glyphs.sw, lvl * glyphs.sh, glyphs.sw, glyphs.sh, col * CELL_W, row * CELL_H, CELL_W, CELL_H);
    }
    if (!canvas.hasAttribute('data-on')) canvas.setAttribute('data-on', '1');
  }

  /* ── the pointer ───────────────────────────────────────────────────────
     A mouse or pen turns the disc by dragging the empty ground and moves the
     camera by parallax. A finger never turns or pans anything: a touch that
     moves is a scroll and belongs to the page, so every listener here is
     passive and nothing calls preventDefault. A finger that lands on the
     ground and lifts without moving sends a pulse. */
  if (!reduce.matches) {
    window.addEventListener('pointermove', (e) => {
      if (STILL) return;
      if (spin.id !== null && e.pointerId === spin.id) {
        if (Math.abs(e.clientX - spin.downX) + Math.abs(e.clientY - spin.downY) > 6) spin.moved = true;
      }
      if (e.pointerType === 'touch') return;
      look.tx = (e.clientX / window.innerWidth) * 2 - 1;
      look.ty = -((e.clientY / window.innerHeight) * 2 - 1);
      if (!spin.dragging || e.pointerId !== spin.id) return;
      const dx = e.clientX - spin.lastX, dy = e.clientY - spin.lastY;
      spin.lastX = e.clientX; spin.lastY = e.clientY;
      sys.rz += dx * 0.004;
      sys.rx = clamp(sys.rx + dy * 0.003, 0.55, 1.45);
      spin.vz = -dx * 0.25;
      spin.vx = dy * 0.18;
    }, { passive: true });

    window.addEventListener('pointerdown', (e) => {
      if (STILL) return;
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      if (spin.id !== null) return;                       /* a second finger changes nothing */
      if (e.target && e.target.closest && e.target.closest(INTERACTIVE)) return;
      spin.downX = spin.lastX = e.clientX;
      spin.downY = spin.lastY = e.clientY;
      spin.downAt = performance.now();
      spin.moved = false;
      spin.id = e.pointerId;
      if (e.pointerType !== 'touch') {
        spin.dragging = true;
        spin.vz = 0; spin.vx = 0;
        document.body.style.cursor = 'grabbing';
      }
    }, { passive: true });
    const release = (e) => {
      if (spin.id !== null && e.pointerId !== spin.id) return;
      const wasDown = spin.id !== null;
      const quick = performance.now() - spin.downAt < 450;
      spin.dragging = false;
      spin.id = null;
      document.body.style.cursor = '';
      if (wasDown && !spin.moved && quick && e.type === 'pointerup') pulseAt(e.clientX, e.clientY);
    };
    window.addEventListener('pointerup', release, { passive: true });
    window.addEventListener('pointercancel', release, { passive: true });
    window.addEventListener('blur', () => { spin.dragging = false; spin.id = null; document.body.style.cursor = ''; });
  }

  /* a click lands on the disc's plane; the pulse spreads from there */
  function pulseAt(sxp, syp) {
    const F = focal();
    /* ray from the camera through the screen point, in world space */
    const dx = (sxp - W / 2) / F, dy = -(syp - H / 2) / F, dz = -1;
    /* plane: through (0, sys.y, 0) with normal = tilt applied to (0,0,1) */
    const nx = 0, ny = -Math.sin(sys.rx), nz = Math.cos(sys.rx);
    const ox = cam.x, oy = cam.y - sys.y, oz = cam.z;
    const denom = nx * dx + ny * dy + nz * dz;
    if (Math.abs(denom) < 1e-4) return;
    const s = -(nx * ox + ny * oy + nz * oz) / denom;
    if (s <= 0) return;
    const wx = ox + dx * s, wy = oy + dy * s, wz = oz + dz * s;
    /* back to system space: undo the tilt, then the spin */
    const y0 = wy * Math.cos(sys.rx) + wz * Math.sin(sys.rx);
    const lx = wx * Math.cos(sys.rz) + y0 * Math.sin(sys.rz);
    const ly = -wx * Math.sin(sys.rz) + y0 * Math.cos(sys.rz);
    pulses[pulseSlot] = { x: lx, y: ly, t: t };
    pulseSlot = (pulseSlot + 1) % 3;
  }

  return frame;
}

/* ── the tiles in the After pane ──────────────────────────────────────── */

function tile(c) {
  const kind = c.getAttribute('data-ascii') || 'idle';
  const ctx = c.getContext('2d');
  const cw = 10, ch = 16;
  const W = c.width, H = c.height;          /* css size; the canvas is scaled by DPR */
  c.width = Math.floor(W * DPR); c.height = Math.floor(H * DPR);
  c.style.width = W + 'px'; c.style.height = H + 'px';
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  const cols = Math.floor(W / cw), rows = Math.floor(H / ch);
  const LEVELS = 8;
  const glyphs = atlas(cw, ch, LEVELS, ACCENT, 0.22, 0.95);
  /* the orbit: an ellipse of cells, one point per cell around it */
  const a = (cols - 1) / 2, b = (rows - 1) / 2;
  const N = Math.max(12, Math.round(Math.PI * (a + b) * 1.15));
  let t = Math.random() * 10, fraction = 0, shown = 0, lastDrawn = -1;

  function progress(f) {
    fraction = Math.max(0, Math.min(1, f));
    if (reduce.matches) draw(0.016);
  }
  function visible() { return c.isConnected && c.offsetParent !== null; }

  function put(col, row, gi, lvl) {
    if (col < 0 || row < 0 || col >= cols || row >= rows) return;
    ctx.drawImage(glyphs.img, gi * glyphs.sw, lvl * glyphs.sh, glyphs.sw, glyphs.sh, col * cw, row * ch, cw, ch);
  }

  function draw(dt) {
    t += dt;
    shown += (fraction - shown) * Math.min(1, dt * 5);
    ctx.clearRect(0, 0, W, H);
    const cx = (cols - 1) / 2, cy = (rows - 1) / 2;
    const tilt = 0.55 + 0.12 * Math.sin(t * 0.17);      /* the ellipse breathes a little */
    const ry = b * tilt, rx = a;
    for (let k = 0; k < N; k++) {
      const u = k / N;                                     /* 0 at the top, clockwise */
      const ang = u * Math.PI * 2 - Math.PI / 2;
      const col = Math.round(cx + Math.cos(ang) * rx), row = Math.round(cy + Math.sin(ang) * ry);
      if (kind === 'run') {
        const done = u <= shown;
        const front = shown > 0.005 && shown < 0.995 && Math.abs(u - shown) < 0.75 / N;
        if (front) put(col, row, 9, LEVELS - 1);
        else if (done) put(col, row, (k % 3 === 0) ? 7 : 5, LEVELS - 2);   /* # and + alternate: a dithered fill */
        else put(col, row, 1, 1);
      } else {
        /* idle: a dither wave runs around the ring */
        const w = 0.5 + 0.5 * Math.sin(u * Math.PI * 6 - t * 1.4);
        const gi = 1 + Math.round(w * 5);                  /* . : - = + * */
        put(col, row, gi, 1 + Math.round(w * 3));
      }
    }
    /* the centre: one glyph for the whole, faint */
    put(Math.round(cx), Math.round(cy), kind === 'run' ? (shown >= 0.995 ? 9 : 2) : 2, kind === 'run' && shown >= 0.995 ? LEVELS - 1 : 1);
  }

  function frame(now, dt) {
    if (!visible()) return;
    if (reduce.matches && lastDrawn >= 0) return;
    lastDrawn = now;
    draw(dt);
  }
  return { frame, progress };
}

/* ── small maths ───────────────────────────────────────────────────────── */

function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * 0.5;
}
function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
