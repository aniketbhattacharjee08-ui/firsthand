/* Vervly pricing page.

   Three questions on load, all at once: GET /api/auth/me (who is here),
   GET /api/me (paywall on, balance, active plan) and GET /api/billing/plans
   (the prices and word counts Stripe charges). The cards in pricing.html
   are the static fallback; when the plans route answers they are rebuilt
   from it so the page never drifts from the server. Then:

     signed out            every button is a link to /signup?next=/pricing
     paywall off / absent  buttons disabled, one note
     stripe not set up     buttons disabled, one note
     signed in             Choose posts /api/billing/checkout and follows the URL;
                           the active plan says "Your plan" with its renewal date
                           and the quiet Manage subscription button appears.

   ES5, no build step. The same words as the plans sheet in app.js. */
(function () {
  'use strict';

  var ME_PATH = '/api/me';
  var AUTH_PATH = '/api/auth/me';
  var PLANS_PATH = '/api/billing/plans';
  var CHECKOUT_PATH = '/api/billing/checkout';
  var PORTAL_PATH = '/api/billing/portal';
  var SIGNUP_URL = '/signup?next=' + encodeURIComponent('/pricing');
  var SIGNIN_URL = '/signin?next=' + encodeURIComponent('/pricing');
  var BALANCE_STORE = 'humanizer.balanceBefore';

  /* the static copy, used when the plans route does not answer */
  var FALLBACK = {
    free_words: 1000,
    plans: [
      { name: 'monthly', label: '$9.99 a month', interval: 'month', words_per_month: 30000 },
      { name: 'yearly', label: '$79.99 a year', interval: 'year', words_per_month: 30000 },
      { name: 'lifetime', label: '$299 once', interval: 'lifetime', words_per_month: 20000 }
    ],
    packs: [
      { name: 'topup', label: '$4.99', words: 10000 }
    ]
  };

  var state = {
    signedIn: null,       /* true, false, or null when /api/auth/me did not answer */
    user: null,           /* the /api/me user, when the paywall package is mounted */
    paywall: null,        /* true, false, or null when /api/me did not answer */
    plans: null,          /* the /api/billing/plans answer, or null */
    wordsPerCredit: null,
    wordsLeft: null,
    plan: '',
    busy: false
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
  /* fills a node with text, every numeral and price in the mono face */
  function numify(node, text) {
    clear(node);
    var parts = String(text).split(/(\$?\d[\d,]*(?:\.\d+)?)/);
    for (var i = 0; i < parts.length; i++) {
      if (!parts[i]) continue;
      if (i % 2) node.appendChild(el('span', 'num', parts[i]));
      else node.appendChild(document.createTextNode(parts[i]));
    }
    return node;
  }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }

  function intervalName(interval, name) {
    if (interval === 'month') return 'Monthly';
    if (interval === 'year') return 'Yearly';
    if (interval === 'lifetime') return 'Lifetime';
    return name ? cap(name) : 'Plan';
  }
  function renewal(interval) {
    if (interval === 'lifetime') return 'One payment. It never renews.';
    if (interval === 'year') return 'Renews every year until you cancel.';
    return 'Renews every month until you cancel.';
  }
  function priceText(item, staticItem) {
    var label = typeof item.label === 'string' ? item.label : '';
    if (/\d/.test(label)) return label;
    if (staticItem && staticItem.label) return staticItem.label;
    return label;
  }
  function staticPlan(name, interval) {
    var i;
    for (i = 0; i < FALLBACK.plans.length; i++) if (FALLBACK.plans[i].name === name) return FALLBACK.plans[i];
    for (i = 0; i < FALLBACK.plans.length; i++) if (FALLBACK.plans[i].interval === interval) return FALLBACK.plans[i];
    return null;
  }

  /* plan_until is seconds since the epoch from the server; a string date
     is accepted too. Null means it never ends. */
  function untilDate(v) {
    if (v === null || v === undefined || v === '') return null;
    var n = loose(v);
    var d = n !== null ? new Date(n < 1e12 ? n * 1000 : n) : new Date(String(v));
    if (isNaN(d.getTime())) return null;
    try {
      return d.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
    } catch (e) {
      return d.toDateString();
    }
  }

  /* ── network ───────────────────────────────────────────────────────── */

  /* resolves with {ok, status, body} and never rejects: a missing route is
     an answer here, not an error */
  function call(path, body, timeoutMs) {
    if (typeof fetch !== 'function') return Promise.resolve({ ok: false, status: 0, body: null, error: 'no fetch' });
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var opts = { method: body ? 'POST' : 'GET', headers: { Accept: 'application/json' }, credentials: 'same-origin', cache: 'no-store' };
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
      if (typeof b.error === 'string') return b.error;
    }
    return r && r.error ? r.error : '';
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
    if (u) {
      state.user = u;
      if (state.signedIn === null) state.signedIn = true;
      var wpc = loose(u.words_per_credit);
      if (wpc !== null && wpc > 0) state.wordsPerCredit = wpc;
      state.wordsLeft = loose(u.words_left);
      if (state.wordsLeft === null && state.wordsPerCredit && loose(u.credits) !== null) {
        state.wordsLeft = Math.round(loose(u.credits) * state.wordsPerCredit);
      }
      state.plan = typeof u.plan === 'string' && u.plan_active !== false ? u.plan : '';
    }
  }

  function applyPlans(r) {
    if (!r.ok || !r.body || typeof r.body !== 'object') { state.plans = null; return; }
    state.plans = r.body;
    var wpc = loose(r.body.words_per_credit);
    if (state.wordsPerCredit === null && wpc !== null && wpc > 0) state.wordsPerCredit = wpc;
    if (state.paywall === null && typeof r.body.paywall === 'boolean') state.paywall = r.body.paywall;
  }

  /* what the buttons can do right now */
  function mode() {
    if (state.paywall === false) return 'off';
    if (state.paywall === null && state.plans === null) return 'absent';
    if (state.plans && state.plans.stripe === false) return 'nostripe';
    if (state.signedIn === false || (state.signedIn === null && !state.user)) return 'out';
    return 'in';
  }

  /* ── render ────────────────────────────────────────────────────────── */

  function renderNav() {
    var signedIn = state.signedIn === true || !!state.user;
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

  function renderFree() {
    var line = $('free-line');
    if (!line) return;
    var free = state.plans ? loose(state.plans.free_words) : null;
    if (free === null) free = FALLBACK.free_words;
    var text = fmtInt(free) + ' words free when you create an account.';
    if (state.user && state.paywall === true && !state.plan && state.wordsLeft !== null) {
      text = fmtInt(free) + ' words free when you create an account; you have ' + fmtInt(state.wordsLeft) + ' left.';
    }
    numify(line, text);
  }

  function setStatus(kind, message, retry) {
    var node = $('plans-status');
    if (!node) return;
    clear(node);
    node.setAttribute('data-state', kind || 'idle');
    if (message) node.appendChild(document.createTextNode(message));
    if (retry) {
      node.appendChild(document.createTextNode(' '));
      var b = el('button', 'link-btn', retry.label || 'Try again');
      b.type = 'button';
      b.addEventListener('click', retry.fn);
      node.appendChild(b);
    }
  }

  /* the button for a card or a top-up row, by the current mode */
  function actionFor(item, isPack, title) {
    var m = mode();
    var label = isPack ? 'Top up' : 'Choose ' + title;
    if (m === 'out') {
      var a = el('a', isPack ? 'btn' : 'btn btn-primary', isPack ? 'Create account to top up' : 'Create account to choose');
      a.href = SIGNUP_URL;
      a.setAttribute('aria-label', (isPack ? 'Create an account to top up with ' : 'Create an account to choose ') + title);
      return a;
    }
    var btn = el('button', isPack ? 'btn' : 'btn btn-primary', label);
    btn.type = 'button';
    btn.setAttribute('data-choose', '1');
    btn.setAttribute('aria-label', (isPack ? 'Top up with ' : 'Choose ') + title);
    if (m !== 'in') { btn.disabled = true; return btn; }
    btn.addEventListener('click', function () {
      checkout(isPack ? { pack: item.name } : { plan: item.name }, btn, title);
    });
    return btn;
  }

  function planCard(item) {
    var name = typeof item.name === 'string' ? item.name : '';
    var interval = typeof item.interval === 'string' ? item.interval : '';
    var fallback = staticPlan(name, interval);
    var card = el('article', 'price-card');
    card.setAttribute('data-plan', name);
    card.setAttribute('data-interval', interval);
    var title = intervalName(interval, name);
    card.appendChild(el('h2', 'price-card-h', title));
    card.appendChild(numify(el('p', 'price-card-amount'), priceText(item, fallback)));
    var wpm = loose(item.words_per_month);
    if (wpm === null && loose(item.allowance_credits) !== null && state.wordsPerCredit) {
      wpm = Math.round(loose(item.allowance_credits) * state.wordsPerCredit);
    }
    if (wpm === null && fallback) wpm = fallback.words_per_month;
    if (wpm !== null) card.appendChild(numify(el('p', 'price-card-words'), fmtInt(wpm) + ' words a month'));
    card.appendChild(el('p', 'price-card-p', renewal(interval)));
    var act = el('div', 'price-card-act');
    if (state.plan && state.plan === name) {
      card.setAttribute('data-current', '1');
      act.appendChild(el('p', 'price-current', 'Your plan'));
      var when = interval === 'lifetime' ? null : (state.user ? untilDate(state.user.plan_until) : null);
      if (interval === 'lifetime') act.appendChild(el('p', 'price-until', 'Paid once. It never renews.'));
      else if (when) act.appendChild(el('p', 'price-until', 'Renews on ' + when + '.'));
    } else {
      act.appendChild(actionFor(item, false, title));
    }
    card.appendChild(act);
    return card;
  }

  function packRow(item) {
    var row = el('div', 'topup-row');
    row.setAttribute('data-pack', item.name || '');
    var main = el('div', 'topup-main');
    var words = loose(item.words);
    if (words === null && loose(item.credits) !== null && state.wordsPerCredit) {
      words = Math.round(loose(item.credits) * state.wordsPerCredit);
    }
    var title = words !== null ? fmtInt(words) + ' words' : (item.name ? cap(item.name) : 'Top-up');
    main.appendChild(numify(el('p', 'topup-h'), title));
    main.appendChild(el('p', 'topup-meta', 'Paid once, added to whatever you have. No renewal.'));
    row.appendChild(main);
    row.appendChild(numify(el('p', 'topup-price'), priceText(item, FALLBACK.packs[0])));
    row.appendChild(actionFor(item, true, title));
    return row;
  }

  function renderPlans() {
    var grid = $('price-grid');
    var rows = $('topup-rows');
    if (!grid || !rows) return;
    var p = state.plans;
    var plans = p && Array.isArray(p.plans) && p.plans.length ? p.plans : FALLBACK.plans;
    var packs = p && Array.isArray(p.packs) ? p.packs : FALLBACK.packs;
    clear(grid);
    plans.forEach(function (pl) { if (pl && typeof pl === 'object') grid.appendChild(planCard(pl)); });
    clear(rows);
    packs.forEach(function (pk) { if (pk && typeof pk === 'object') rows.appendChild(packRow(pk)); });
    var topups = rows.parentNode;
    if (topups) topups.hidden = !packs.length;

    var manage = $('manage-btn');
    if (manage) manage.hidden = !(mode() === 'in' && (state.plan || (state.user && state.user.has_billing_portal === true)));

    var m = mode();
    if (m === 'off') setStatus('note', 'Payments are off on this server, so there is nothing to buy; the humanizer runs without a plan.');
    else if (m === 'absent') setStatus('note', 'Plans are not available on this server yet. The prices shown are the ones Vervly charges.');
    else if (m === 'nostripe') setStatus('note', 'Payments are not set up yet, so nothing can be bought here.');
    else setStatus('idle', '');
  }

  function render() {
    renderNav();
    renderFree();
    renderPlans();
  }

  /* ── actions ───────────────────────────────────────────────────────── */

  function showPortalOffer() {
    var manage = $('manage-btn');
    if (manage) manage.hidden = false;
  }

  function checkout(body, btn, title) {
    if (state.busy) return;
    state.busy = true;
    var was = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Opening checkout';
    setStatus('wait', 'Opening the checkout page for ' + title + '.');
    try { if (state.wordsLeft !== null) sessionStorage.setItem(BALANCE_STORE, String(state.wordsLeft)); } catch (e) { /* private mode */ }
    call(CHECKOUT_PATH, body).then(function (r) {
      var url = r.ok && r.body && typeof r.body.url === 'string' ? r.body.url : '';
      if (url) { location.href = url; return; }
      state.busy = false;
      btn.disabled = false;
      btn.textContent = was;
      if (r.status === 401) { location.href = SIGNIN_URL; return; }
      if (r.status === 409) {
        showPortalOffer();
        setStatus('error', 'You already have a subscription. Change or cancel it from Manage subscription first.');
        return;
      }
      if (r.status === 503) { setStatus('error', 'Payments are not set up yet, so nothing can be bought here.'); return; }
      if (r.status === 404) { setStatus('error', 'Plans are not available on this server yet.'); return; }
      var why = detailOf(r) || (r.status ? 'HTTP ' + r.status : 'no answer');
      setStatus('error', 'The checkout page did not open: ' + why + '.', { fn: function () { checkout(body, btn, title); } });
    });
  }

  function openPortal(btn) {
    if (state.busy) return;
    state.busy = true;
    btn.disabled = true;
    setStatus('wait', 'Opening your subscription page.');
    call(PORTAL_PATH, {}).then(function (r) {
      var url = r.ok && r.body && typeof r.body.url === 'string' ? r.body.url : '';
      if (url) { location.href = url; return; }
      state.busy = false;
      btn.disabled = false;
      if (r.status === 401) { location.href = SIGNIN_URL; return; }
      if (r.status === 404) { setStatus('error', 'There is no subscription on this account to manage yet. Choose a plan first.'); return; }
      if (r.status === 503) { setStatus('error', 'Payments are not set up yet.'); return; }
      var why = detailOf(r) || (r.status ? 'HTTP ' + r.status : 'no answer');
      setStatus('error', 'The subscription page did not open: ' + why + '.', { fn: function () { openPortal(btn); } });
    });
  }

  function wire() {
    var manage = $('manage-btn');
    if (manage) manage.addEventListener('click', function () { openPortal(manage); });
  }

  function start() {
    wire();
    if (typeof fetch !== 'function' || typeof Promise !== 'function') { render(); return; }
    Promise.all([call(AUTH_PATH, null, 6000), call(ME_PATH, null, 6000), call(PLANS_PATH, null, 8000)]).then(function (rs) {
      applyAuth(rs[0]);
      applyMe(rs[1]);
      applyPlans(rs[2]);
      render();
    }, function () { render(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
