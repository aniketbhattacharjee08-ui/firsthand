/* Vervly: the Your voice page.

   Three questions on load, all at once: GET /api/auth/me (who is here),
   GET /api/me (the balance for the top row) and GET /api/voice (the switch,
   the samples, the profile, the liked pile). Then:

     signed out            one notice with a sign-in link; the sections stay,
                           read-only and empty
     route missing (404)   "Your voice is not available on this server yet."
     unreachable           one plain sentence and a Try again
     signed in             the switch and strength post /api/voice/settings;
                           the form posts /api/voice/samples; Remove and Try
                           again act on one sample; the liked pile has its own
                           Remove. While any sample is preparing the page
                           asks /api/voice again every four seconds, and
                           stops while the tab is hidden.

   ES5, no build step, no framework. Every string from the server is set
   with textContent. The contract is web/VOICE.md. */
(function () {
  'use strict';

  var AUTH_PATH = '/api/auth/me';
  var ME_PATH = '/api/me';
  var VOICE_PATH = '/api/voice';
  var SAMPLES_PATH = '/api/voice/samples';
  var SETTINGS_PATH = '/api/voice/settings';
  var LIKED_PATH = '/api/voice/liked';
  var SIGNIN_URL = '/signin?next=' + encodeURIComponent('/voice');
  var POLL_MS = 4000;
  var CONFIRM_MS = 4000;

  /* the limits the page assumes until GET /api/voice says otherwise */
  var LIMITS = { min_words: 80, max_words: 2500, max_samples: 12, min_ready: 1 };

  /* what the page says for each 422 when the server sends no sentence */
  var REFUSALS = {
    too_short: 'That sample is too short. It needs at least 80 words.',
    too_long: 'That sample is too long. Keep it under 2,500 words.',
    too_many: 'You already have 12 samples. Remove one to add another.',
    duplicate: 'You already added that one.',
    reads_as_ai: 'That sample reads as AI-written, so it was not added. A sample the model wrote would teach the humanizer the wrong habits and make the results worse.'
  };

  var state = {
    signedIn: null,       /* true, false, or null when /api/auth/me did not answer */
    user: null,
    paywall: null,
    wordsPerCredit: null,
    wordsLeft: null,
    voice: null,          /* the /api/voice answer, or null */
    available: null,      /* false after a 404 on /api/voice, true once it answered */
    reachError: '',       /* one sentence when /api/voice could not be reached */
    busy: false,          /* a settings post or a sample post in flight */
    pollTimer: null,
    confirmTimer: null,
    confirmBtn: null
  };

  function $(id) { return document.getElementById(id); }
  function each(sel, fn) { var list = document.querySelectorAll(sel); for (var i = 0; i < list.length; i++) fn(list[i]); }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function loose(v) {
    if (v === null || v === undefined || v === '') return null;
    var n = Number(v);
    return isFinite(n) ? n : null;
  }
  function plural(n, one) { return n === 1 ? one : one + 's'; }
  function fmtInt(n) {
    var v = Math.max(0, Math.round(Number(n) || 0));
    return String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }
  /* fills a node with text, every numeral in the mono face */
  function numify(node, text) {
    clear(node);
    var parts = String(text).split(/(\d[\d,]*(?:\.\d+)?)/);
    for (var i = 0; i < parts.length; i++) {
      if (!parts[i]) continue;
      if (i % 2) node.appendChild(el('span', 'num', parts[i]));
      else node.appendChild(document.createTextNode(parts[i]));
    }
    return node;
  }
  function str(v) { return typeof v === 'string' ? v : ''; }
  function words(text) {
    var t = String(text || '').replace(/^\s+|\s+$/g, '');
    return t ? t.split(/\s+/).length : 0;
  }

  /* ── network ───────────────────────────────────────────────────────── */

  /* resolves with {ok, status, body} and never rejects: a missing route is
     an answer here, not an error */
  function call(path, body, timeoutMs, method) {
    if (typeof fetch !== 'function') return Promise.resolve({ ok: false, status: 0, body: null, error: 'this browser cannot reach the server' });
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var opts = { method: method || (body ? 'POST' : 'GET'), headers: { Accept: 'application/json' }, credentials: 'same-origin', cache: 'no-store' };
    if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    if (controller) opts.signal = controller.signal;
    var timer = controller ? setTimeout(function () { controller.abort(); }, timeoutMs || 15000) : null;
    return fetch(path, opts).then(function (res) {
      if (timer) clearTimeout(timer);
      return res.text().then(function (t) {
        var j = null;
        try { j = t ? JSON.parse(t) : null; } catch (e) { j = null; }
        return { ok: res.ok, status: res.status, body: j, text: t };
      }, function () {
        return { ok: res.ok, status: res.status, body: null, text: '' };
      });
    }, function (err) {
      if (timer) clearTimeout(timer);
      var why = err && err.name === 'AbortError' ? 'the request timed out'
        : (typeof navigator !== 'undefined' && navigator.onLine === false) ? 'the browser is offline'
        : ((err && err.message) || 'network error');
      return { ok: false, status: 0, body: null, error: why };
    });
  }

  function detailOf(r) {
    var b = r && r.body;
    if (b && typeof b === 'object') {
      if (typeof b.detail === 'string') return b.detail;
      if (b.detail && typeof b.detail === 'object' && typeof b.detail.message === 'string') return b.detail.message;
      if (typeof b.message === 'string') return b.message;
    }
    return '';
  }
  function codeOf(r) {
    var b = r && r.body;
    return b && typeof b === 'object' && typeof b.error === 'string' ? b.error : '';
  }
  /* one plain sentence for an answer the page did not expect */
  function failed(r, what) {
    if (r.status === 0) return what + ' did not reach the server: ' + (r.error || 'network error') + '. Check the connection and try again.';
    var why = detailOf(r);
    return what + ' failed' + (why ? ': ' + why : ' (HTTP ' + r.status + ')') + '. Try again.';
  }

  /* ── state ─────────────────────────────────────────────────────────── */

  function applyAuth(r) {
    if (!r.ok || !r.body || typeof r.body !== 'object') { state.signedIn = null; return; }
    state.signedIn = r.body.signed_in === true;
    if (state.signedIn && r.body.user && typeof r.body.user === 'object') state.user = state.user || r.body.user;
  }

  function applyMe(r) {
    if (!r.ok || !r.body || typeof r.body !== 'object') { state.paywall = null; return; }
    var obj = r.body;
    var u = obj.user && typeof obj.user === 'object' ? obj.user : null;
    state.paywall = obj.paywall === true;
    if (u && obj.guest !== true) {
      state.user = u;
      if (state.signedIn === null) state.signedIn = true;
      var wpc = loose(u.words_per_credit);
      if (wpc !== null && wpc > 0) state.wordsPerCredit = wpc;
      state.wordsLeft = loose(u.words_left);
      if (state.wordsLeft === null && state.wordsPerCredit && loose(u.credits) !== null) {
        state.wordsLeft = Math.round(loose(u.credits) * state.wordsPerCredit);
      }
    }
  }

  function applyVoice(r) {
    state.reachError = '';
    if (r.status === 404) { state.available = false; state.voice = null; return; }
    if (r.status === 401) { state.available = true; state.voice = null; state.signedIn = false; return; }
    if (r.status === 0) { state.reachError = r.error || 'network error'; return; }
    if (!r.ok || !r.body || typeof r.body !== 'object') { state.reachError = 'HTTP ' + r.status; return; }
    state.available = true;
    state.voice = r.body;
    if (r.body.limits && typeof r.body.limits === 'object') {
      var L = r.body.limits, k;
      for (k in LIMITS) if (Object.prototype.hasOwnProperty.call(LIMITS, k) && loose(L[k]) !== null) LIMITS[k] = loose(L[k]);
    }
    if (r.body.signed_in === false && state.signedIn === null) state.signedIn = false;
  }

  /* the page is live when the voice route answered for a signed-in user */
  function live() { return !!(state.voice && state.voice.signed_in === true); }
  function samples() { return state.voice && Array.isArray(state.voice.samples) ? state.voice.samples : []; }
  function liked() { return state.voice && Array.isArray(state.voice.liked) ? state.voice.liked : []; }
  function readyCount() {
    var n = state.voice ? loose(state.voice.ready) : null;
    if (n !== null) return n;
    var list = samples(), c = 0;
    for (var i = 0; i < list.length; i++) if (list[i] && list[i].status === 'ready') c++;
    return c;
  }
  function anyPreparing() {
    var list = samples();
    for (var i = 0; i < list.length; i++) if (list[i] && list[i].status === 'preparing') return true;
    return false;
  }

  /* ── render ────────────────────────────────────────────────────────── */

  function renderNav() {
    var signedIn = state.signedIn === true || !!state.user || live();
    each('[data-when="out"]', function (n) { n.hidden = signedIn; });
    each('[data-when="in"]', function (n) { n.hidden = !signedIn; });
    var who = $('nav-who');
    if (who) {
      var name = signedIn && state.user ? (state.user.name || state.user.email || '') : '';
      who.textContent = name;
      who.title = state.user && state.user.email ? state.user.email : '';
      who.hidden = !name;
    }
    var bal = $('nav-balance');
    if (bal) {
      if (signedIn && state.paywall === true && state.wordsLeft !== null) {
        clear(bal);
        bal.appendChild(el('span', 'num', fmtInt(state.wordsLeft)));
        bal.appendChild(document.createTextNode(' ' + plural(state.wordsLeft, 'word') + ' left'));
        bal.hidden = false;
      } else {
        bal.hidden = true;
      }
    }
  }

  function renderNotice() {
    var node = $('voice-notice');
    if (!node) return;
    clear(node);
    if (state.available === false) {
      node.textContent = 'Your voice is not available on this server yet.';
    } else if (state.reachError && !state.voice) {
      node.appendChild(document.createTextNode('The page could not reach the server: ' + state.reachError + '. '));
      var b = el('button', 'link-btn', 'Try again');
      b.type = 'button';
      b.addEventListener('click', function () { refresh(); });
      node.appendChild(b);
    } else if (!live()) {
      node.appendChild(document.createTextNode('Sign in to add your writing. '));
      var a = el('a', '', 'Sign in');
      a.href = SIGNIN_URL;
      node.appendChild(a);
    } else {
      node.hidden = true;
      return;
    }
    node.hidden = false;
  }

  function setStatus(id, kind, message) {
    var node = $(id);
    if (!node) return;
    node.setAttribute('data-state', kind || 'idle');
    numify(node, message || '');
  }

  function renderUse() {
    var sw = $('voice-switch');
    var sel = $('voice-strength');
    var hint = $('use-hint');
    var on = live() && state.voice.enabled === true;
    var ready = live() ? readyCount() : 0;
    var usable = live() && ready >= LIMITS.min_ready;
    if (sw) {
      sw.setAttribute('aria-checked', on ? 'true' : 'false');
      sw.disabled = !usable;
    }
    if (sel) {
      var s = live() ? str(state.voice.strength) : '';
      if (s === 'light' || s === 'balanced' || s === 'strong') sel.value = s;
      sel.disabled = !usable;
    }
    if (hint) {
      var text = '';
      if (live() && ready < LIMITS.min_ready) {
        text = anyPreparing() ? 'Your first sample is being prepared. The switch turns on once it is ready.' : 'Add one sample first.';
      } else if (live() && !on) {
        text = 'Off. Rewrites use the stock examples until you turn this on.';
      } else if (live() && on) {
        text = 'On. Rewrites in the humanizer lean on your ' + fmtInt(ready) + ' ready ' + plural(ready, 'sample') + '.';
      }
      numify(hint, text);
    }
  }

  function renderForm() {
    var form = $('sample-form');
    var title = $('sample-title');
    var text = $('sample-text');
    var add = $('sample-add');
    var count = $('sample-count');
    var limits = $('sample-limits');
    var isLive = live();
    var n = text ? words(text.value) : 0;
    var full = isLive && samples().length >= LIMITS.max_samples;
    if (title) title.disabled = !isLive;
    if (text) text.disabled = !isLive;
    if (add) add.disabled = !isLive || full || n < LIMITS.min_words || n > LIMITS.max_words;
    if (count) {
      var off = n > 0 && (n < LIMITS.min_words || n > LIMITS.max_words);
      count.setAttribute('data-short', off ? '1' : '0');
      var tail = '';
      if (n > 0 && n < LIMITS.min_words) tail = ', ' + fmtInt(LIMITS.min_words - n) + ' more needed';
      else if (n > LIMITS.max_words) tail = ', ' + fmtInt(n - LIMITS.max_words) + ' over the limit';
      numify(count, fmtInt(n) + ' ' + plural(n, 'word') + tail);
    }
    if (limits) numify(limits, 'Up to ' + fmtInt(LIMITS.max_samples) + ' samples, ' + fmtInt(LIMITS.min_words) + ' to ' + fmtInt(LIMITS.max_words) + ' words each.');
    if (form) form.setAttribute('aria-disabled', isLive ? 'false' : 'true');
  }

  function statusWord(s) {
    if (s === 'ready') return 'Ready';
    if (s === 'preparing') return 'Preparing';
    if (s === 'failed') return 'Needs attention';
    return s ? String(s) : '';
  }

  /* the two-click Remove: the first click asks, the second removes, and four
     seconds of silence puts the button back */
  function armConfirm(btn, fn) {
    btn.addEventListener('click', function () {
      if (btn.getAttribute('data-confirm') === '1') {
        disarmConfirm();
        fn();
        return;
      }
      disarmConfirm();
      btn.setAttribute('data-confirm', '1');
      btn.textContent = 'Remove?';
      btn.setAttribute('aria-label', 'Click again to remove');
      state.confirmBtn = btn;
      state.confirmTimer = setTimeout(disarmConfirm, CONFIRM_MS);
    });
  }
  function disarmConfirm() {
    if (state.confirmTimer) clearTimeout(state.confirmTimer);
    state.confirmTimer = null;
    var b = state.confirmBtn;
    state.confirmBtn = null;
    if (b && b.parentNode) {
      b.removeAttribute('data-confirm');
      b.textContent = 'Remove';
      b.removeAttribute('aria-label');
    }
  }

  function sampleRow(s) {
    var row = el('article', 'sample');
    row.setAttribute('data-id', str(s.id));
    var main = el('div', 'sample-main');
    var head = el('div', 'sample-head');
    var title = str(s.title).replace(/^\s+|\s+$/g, '') || 'Untitled';
    head.appendChild(el('span', 'sample-title', title));
    var n = loose(s.n_words);
    if (n !== null) {
      var w = el('span', 'sample-words');
      w.appendChild(el('span', 'num', fmtInt(n)));
      w.appendChild(document.createTextNode(' ' + plural(n, 'word')));
      head.appendChild(w);
    }
    var st = el('span', 'sample-state', statusWord(s.status));
    st.setAttribute('data-status', str(s.status));
    head.appendChild(st);
    main.appendChild(head);
    if (s.status === 'failed' && str(s.detail)) main.appendChild(el('p', 'sample-detail', str(s.detail)));
    else if (s.status === 'preparing' && state.voice && state.voice.llm_available === false) {
      main.appendChild(el('p', 'sample-detail', 'The model is offline on this server, so this sample waits until it is back.'));
    }
    if (str(s.excerpt)) main.appendChild(el('p', 'sample-excerpt', str(s.excerpt)));
    row.appendChild(main);

    var acts = el('div', 'sample-acts');
    if (s.status === 'failed') {
      var retry = el('button', 'btn btn-quiet', 'Try again');
      retry.type = 'button';
      retry.addEventListener('click', function () { prepare(s, retry); });
      acts.appendChild(retry);
    }
    var rm = el('button', 'btn btn-quiet', 'Remove');
    rm.type = 'button';
    armConfirm(rm, function () { removeSample(s, rm); });
    acts.appendChild(rm);
    row.appendChild(acts);
    return row;
  }

  function renderSamples() {
    var list = $('sample-list');
    var empty = $('sample-empty');
    if (!list) return;
    disarmConfirm();
    clear(list);
    var items = live() ? samples() : [];
    for (var i = 0; i < items.length; i++) if (items[i] && typeof items[i] === 'object') list.appendChild(sampleRow(items[i]));
    if (empty) {
      empty.hidden = !live() || items.length > 0;
    }
  }

  function renderTraits() {
    var rows = $('traits');
    var empty = $('traits-empty');
    if (!rows) return;
    clear(rows);
    var p = live() && state.voice.profile && typeof state.voice.profile === 'object' ? state.voice.profile : null;
    var traits = p && Array.isArray(p.traits) ? p.traits : [];
    for (var i = 0; i < traits.length; i++) {
      var t = traits[i];
      if (!t || typeof t !== 'object') continue;
      var row = el('div', 'row');
      row.appendChild(numify(el('span', 'row-k'), str(t.label) || str(t.key)));
      var v = numify(el('span', 'row-v'), str(t.value));
      if (str(t.note)) v.appendChild(numify(el('span', 'row-note'), str(t.note)));
      row.appendChild(v);
      rows.appendChild(row);
    }
    if (empty) empty.hidden = traits.length > 0;
  }

  function likedRow(l) {
    var row = el('article', 'sample');
    row.setAttribute('data-id', str(l.id));
    var main = el('div', 'sample-main');
    var n = loose(l.n_words);
    if (n !== null) {
      var head = el('div', 'sample-head');
      var w = el('span', 'sample-words');
      w.appendChild(el('span', 'num', fmtInt(n)));
      w.appendChild(document.createTextNode(' ' + plural(n, 'word')));
      head.appendChild(w);
      main.appendChild(head);
    }
    if (str(l.excerpt)) main.appendChild(el('p', 'sample-excerpt', str(l.excerpt)));
    row.appendChild(main);
    var acts = el('div', 'sample-acts');
    var rm = el('button', 'btn btn-quiet', 'Remove');
    rm.type = 'button';
    armConfirm(rm, function () { removeLiked(l, rm); });
    acts.appendChild(rm);
    row.appendChild(acts);
    return row;
  }

  function renderLiked() {
    var list = $('liked-list');
    var count = $('liked-count');
    if (!list) return;
    clear(list);
    var items = live() ? liked() : [];
    for (var i = 0; i < items.length; i++) if (items[i] && typeof items[i] === 'object') list.appendChild(likedRow(items[i]));
    if (count) {
      if (!live()) count.textContent = 'Rewrites you give a thumbs up in the humanizer are kept here.';
      else if (!items.length) count.textContent = 'Nothing yet. Give a rewrite a thumbs up in the humanizer and it appears here.';
      else numify(count, fmtInt(items.length) + ' ' + plural(items.length, 'rewrite') + ' you marked as sounding like you.');
    }
  }

  function render() {
    renderNav();
    renderNotice();
    renderUse();
    renderForm();
    renderSamples();
    renderTraits();
    renderLiked();
    schedulePoll();
  }

  /* ── polling while a sample is preparing ───────────────────────────── */

  function stopPoll() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  function schedulePoll() {
    stopPoll();
    if (!live() || !anyPreparing()) return;
    if (typeof document.hidden === 'boolean' && document.hidden) return;
    state.pollTimer = setTimeout(function () {
      state.pollTimer = null;
      refresh();
    }, POLL_MS);
  }

  /* asks /api/voice again and repaints; a 401 here turns the page signed out */
  function refresh() {
    return call(VOICE_PATH, null, 8000).then(function (r) {
      applyVoice(r);
      if (r.status === 401 || (r.ok && r.body && r.body.signed_in === false)) { state.signedIn = false; state.user = null; }
      render();
      return r;
    });
  }

  /* an answer that means the session is gone: show the signed-out page */
  function signedOut() {
    state.signedIn = false;
    state.user = null;
    state.voice = state.voice ? { signed_in: false, enabled: false, ready: 0, samples: [], liked: [], profile: null, limits: state.voice.limits } : null;
    state.busy = false;
    render();
  }

  /* ── actions ───────────────────────────────────────────────────────── */

  function postSettings(body, statusText) {
    if (state.busy || !live()) return;
    state.busy = true;
    renderUse();
    setStatus('use-status', 'wait', statusText);
    call(SETTINGS_PATH, body).then(function (r) {
      state.busy = false;
      if (r.status === 401) { setStatus('use-status', 'idle', ''); signedOut(); return; }
      if (r.status === 404) { state.available = false; state.voice = null; render(); return; }
      if (r.ok && r.body && typeof r.body === 'object') {
        if (typeof r.body.enabled === 'boolean') state.voice.enabled = r.body.enabled;
        if (typeof r.body.strength === 'string') state.voice.strength = r.body.strength;
        renderUse();
        var on = state.voice.enabled === true;
        var how = str(state.voice.strength);
        var strengthWord = how === 'light' ? 'Light' : how === 'strong' ? 'Strong' : 'Balanced';
        setStatus('use-status', 'ok', on ? 'Saved. Rewrites use your voice at ' + strengthWord.toLowerCase() + ' strength.' : 'Saved. Your voice is off.');
        return;
      }
      renderUse();
      setStatus('use-status', 'error', failed(r, 'Saving the setting'));
    });
  }

  function addSample() {
    if (state.busy || !live()) return;
    var title = $('sample-title');
    var text = $('sample-text');
    var add = $('sample-add');
    if (!text) return;
    var n = words(text.value);
    if (n < LIMITS.min_words) { setStatus('sample-status', 'error', REFUSALS.too_short); return; }
    if (n > LIMITS.max_words) { setStatus('sample-status', 'error', REFUSALS.too_long); return; }
    state.busy = true;
    renderForm();
    renderUse();
    if (add) add.setAttribute('data-busy', '1');
    setStatus('sample-status', 'wait', 'Checking the sample and adding it.');
    var body = { text: text.value };
    var t = title ? title.value.replace(/^\s+|\s+$/g, '') : '';
    if (t) body.title = t;
    call(SAMPLES_PATH, body, 30000).then(function (r) {
      state.busy = false;
      if (add) add.removeAttribute('data-busy');
      if (r.status === 401) { setStatus('sample-status', 'idle', ''); signedOut(); return; }
      if (r.status === 404) { state.available = false; state.voice = null; render(); return; }
      if (r.status === 422) {
        var code = codeOf(r);
        var why = detailOf(r) || REFUSALS[code] || 'That sample was not accepted.';
        setStatus('sample-status', 'error', why);
        renderForm();
        renderUse();
        return;
      }
      if ((r.status === 201 || r.ok) && r.body && r.body.sample && typeof r.body.sample === 'object') {
        var s = r.body.sample;
        var parts = Array.isArray(r.body.samples) && r.body.samples.length ? r.body.samples : [s];
        if (text) text.value = '';
        if (title) title.value = '';
        state.voice.samples = parts.concat(samples());
        if (s.status === 'ready') state.voice.ready = readyCount() + parts.length;
        var lead = parts.length > 1 ? 'Added in ' + parts.length + ' parts.' : 'Added.';
        setStatus('sample-status', 'ok', s.status === 'preparing'
          ? lead + (parts.length > 1 ? ' They are' : ' The sample is') + ' being prepared, which takes a minute or so.'
          : s.status === 'failed' ? lead + ' It could not be prepared. Try again from the list below.'
          : lead + ' Ready.');
        render();
        refresh();
        return;
      }
      setStatus('sample-status', 'error', failed(r, 'Adding the sample'));
      renderForm();
      renderUse();
    });
  }

  function prepare(s, btn) {
    if (!live()) return;
    btn.disabled = true;
    setStatus('sample-status', 'wait', 'Preparing the sample again.');
    call(SAMPLES_PATH + '/' + encodeURIComponent(str(s.id)) + '/prepare', {}).then(function (r) {
      if (r.status === 401) { setStatus('sample-status', 'idle', ''); signedOut(); return; }
      if (r.status === 404) {
        btn.disabled = false;
        setStatus('sample-status', 'error', 'That sample is no longer on your account.');
        refresh();
        return;
      }
      if (r.ok && r.body && r.body.sample && typeof r.body.sample === 'object') {
        var list = samples();
        for (var i = 0; i < list.length; i++) if (list[i] && list[i].id === s.id) list[i] = r.body.sample;
        setStatus('sample-status', 'ok', r.body.sample.status === 'ready' ? 'Ready.' : 'Preparing again. This takes a minute or so.');
        render();
        return;
      }
      btn.disabled = false;
      setStatus('sample-status', 'error', failed(r, 'Preparing the sample'));
    });
  }

  function removeSample(s, btn) {
    if (!live()) return;
    btn.disabled = true;
    setStatus('sample-status', 'wait', 'Removing the sample.');
    call(SAMPLES_PATH + '/' + encodeURIComponent(str(s.id)), null, 15000, 'DELETE').then(function (r) {
      if (r.status === 401) { setStatus('sample-status', 'idle', ''); signedOut(); return; }
      if (r.ok || r.status === 404) {
        var list = samples(), kept = [];
        for (var i = 0; i < list.length; i++) if (!list[i] || list[i].id !== s.id) kept.push(list[i]);
        state.voice.samples = kept;
        if (s.status === 'ready') state.voice.ready = Math.max(0, readyCount() - 1);
        setStatus('sample-status', 'ok', 'Removed.');
        render();
        refresh();
        return;
      }
      btn.disabled = false;
      setStatus('sample-status', 'error', failed(r, 'Removing the sample'));
    });
  }

  function removeLiked(l, btn) {
    if (!live()) return;
    btn.disabled = true;
    setStatus('liked-status', 'wait', 'Removing it.');
    call(LIKED_PATH + '/' + encodeURIComponent(str(l.id)), null, 15000, 'DELETE').then(function (r) {
      if (r.status === 401) { setStatus('liked-status', 'idle', ''); signedOut(); return; }
      if (r.ok || r.status === 404) {
        var list = liked(), kept = [];
        for (var i = 0; i < list.length; i++) if (!list[i] || list[i].id !== l.id) kept.push(list[i]);
        state.voice.liked = kept;
        setStatus('liked-status', 'ok', 'Removed.');
        renderLiked();
        return;
      }
      btn.disabled = false;
      setStatus('liked-status', 'error', failed(r, 'Removing it'));
    });
  }

  /* ── wiring ────────────────────────────────────────────────────────── */

  /* a switch is a button with role=switch; aria-checked is its state */
  function switchOn(btn) { return !!btn && btn.getAttribute('aria-checked') === 'true'; }
  function setSwitch(btn, on) { if (btn) btn.setAttribute('aria-checked', on ? 'true' : 'false'); }
  function wireSwitch(btn, onChange) {
    if (!btn) return;
    btn.addEventListener('click', function () {
      if (btn.disabled) return;
      setSwitch(btn, !switchOn(btn));
      onChange(switchOn(btn));
    });
  }

  function wire() {
    wireSwitch($('voice-switch'), function (on) {
      if (!live()) { setSwitch($('voice-switch'), false); return; }
      /* a click while a save is in flight is put back, not queued */
      if (state.busy) { setSwitch($('voice-switch'), !on); return; }
      postSettings({ enabled: on }, on ? 'Turning your voice on.' : 'Turning your voice off.');
    });
    var sel = $('voice-strength');
    if (sel) sel.addEventListener('change', function () {
      var v = sel.value;
      if (v !== 'light' && v !== 'balanced' && v !== 'strong') return;
      if (state.busy) { renderUse(); return; }
      postSettings({ strength: v }, 'Saving the strength.');
    });
    var form = $('sample-form');
    if (form) form.addEventListener('submit', function (e) { e.preventDefault(); addSample(); });
    var text = $('sample-text');
    if (text) text.addEventListener('input', function () { renderForm(); });
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) stopPoll();
      else if (live() && anyPreparing()) refresh();
    });
  }

  function start() {
    wire();
    if (typeof fetch !== 'function' || typeof Promise !== 'function') {
      state.reachError = 'this browser cannot reach the server';
      render();
      return;
    }
    Promise.all([call(AUTH_PATH, null, 6000), call(ME_PATH, null, 6000), call(VOICE_PATH, null, 8000)]).then(function (rs) {
      applyAuth(rs[0]);
      applyMe(rs[1]);
      applyVoice(rs[2]);
      render();
    }, function () { render(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
