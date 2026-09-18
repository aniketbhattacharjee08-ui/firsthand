/* Firsthand account module: sign-in, credits and purchases.

   Drop-in. Links account.css, mounts a pill into .top-tools (or a fixed
   corner), wraps window.fetch so app.js needs no edits, and exposes
   window.readshumanAccount. ES5, no framework, no build step, no styles
   injected from here.

   Rules for the copy and the controls: web/DESIGN.md. Backend contract:
   deploy/README.md and src/humanizer/billing/routes.py. */
(function () {
  'use strict';

  var rawFetch = window.fetch;
  if (typeof rawFetch !== 'function') return;
  rawFetch = rawFetch.bind(window);

  var PARAMS = new URLSearchParams(location.search);
  var API_BASE = String(window.API_BASE || PARAMS.get('api') || '');
  var POLL_MS = 1000;
  var POLL_LIMIT_MS = 10000;
  var TOAST_MS = 4200;
  var MAX_RETRY_S = 120;

  var state = {
    health: null,      /* GET /api/billing/health */
    user: null,        /* GET /api/me .user */
    paywall: false,
    packs: null,       /* GET /api/billing/packs */
    linkMinutes: 15,
    blocked: null      /* {needed, balance} from the last 402 */
  };

  var ui = { mount: null, pill: null, backdrop: null, sheet: null, toast: null, toastTimer: null, restoreFocus: null, sheetName: '' };

  /* ---- small helpers ---------------------------------------------------- */

  function h(tag, attrs, kids) {
    var n = document.createElement(tag);
    var k, i;
    if (attrs) {
      for (k in attrs) {
        if (!attrs.hasOwnProperty(k)) continue;
        if (k === 'text') n.textContent = attrs[k];
        else if (k === 'on') { for (i in attrs.on) if (attrs.on.hasOwnProperty(i)) n.addEventListener(i, attrs.on[i]); }
        else if (attrs[k] === false || attrs[k] === null || attrs[k] === undefined) { /* skip */ }
        else n.setAttribute(k, attrs[k] === true ? '' : String(attrs[k]));
      }
    }
    if (kids) {
      for (i = 0; i < kids.length; i++) {
        if (kids[i] === null || kids[i] === undefined) continue;
        n.appendChild(typeof kids[i] === 'string' ? document.createTextNode(kids[i]) : kids[i]);
      }
    }
    return n;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function emit(name, detail) {
    try { document.dispatchEvent(new CustomEvent(name, { detail: detail || null })); } catch (e) { /* old browser */ }
  }

  function plural(n, word) { return n + ' ' + word + (n === 1 ? '' : 's'); }

  function dateOf(seconds) {
    if (!seconds) return '';
    try {
      return new Date(Number(seconds) * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    } catch (e) { return ''; }
  }

  function setBusy(btn, busy) {
    if (!btn) return;
    if (busy) { btn.setAttribute('disabled', ''); btn.setAttribute('data-busy', '1'); }
    else { btn.removeAttribute('disabled'); btn.removeAttribute('data-busy'); }
  }

  /* One JSON call to our own API, never through the interceptor. Resolves
     with {status, ok, body, headers}; body is null when there was none. */
  function api(method, path, body) {
    var opts = { method: method, headers: { Accept: 'application/json' }, credentials: 'same-origin', cache: 'no-store' };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return rawFetch(API_BASE + path, opts).then(function (res) {
      return res.text().then(function (t) {
        var j = null;
        try { j = t ? JSON.parse(t) : null; } catch (e) { j = null; }
        return { status: res.status, ok: res.ok, body: j, headers: res.headers };
      });
    }, function (err) {
      return { status: 0, ok: false, body: { error: 'network', detail: (err && err.message) || 'network error' }, headers: null };
    });
  }

  function detailOf(r, fallback) {
    var d = r && r.body && r.body.detail;
    return d ? String(d) : fallback;
  }

  /* ---- account state ---------------------------------------------------- */

  function setUser(user) {
    var before = state.user ? state.user.id : null;
    var beforeCredits = state.user ? state.user.credits : null;
    state.user = user || null;
    renderPill();
    var after = state.user ? state.user.id : null;
    if (before !== after || (state.user && state.user.credits !== beforeCredits)) {
      emit('readshuman:account', { user: state.user });
    }
  }

  function setBalance(n) {
    if (!state.user || isNaN(n)) return;
    if (state.user.credits === n) return;
    state.user.credits = n;
    renderPill();
    emit('readshuman:account', { user: state.user });
  }

  function refresh() {
    return api('GET', '/api/me').then(function (r) {
      if (r.ok && r.body) {
        if (typeof r.body.paywall === 'boolean') state.paywall = r.body.paywall;
        setUser(r.body.user || null);
      }
      return state.user;
    });
  }

  function loadPacks() {
    if (state.packs) return Promise.resolve(state.packs);
    return api('GET', '/api/billing/packs').then(function (r) {
      if (r.ok && r.body && r.body.packs) state.packs = r.body;
      else if (state.health) state.packs = { stripe: !!state.health.stripe, words_per_credit: state.health.words_per_credit, packs: state.health.packs || [] };
      else state.packs = { stripe: false, words_per_credit: 0, packs: [] };
      return state.packs;
    });
  }

  function wordsPerCredit() {
    var w = (state.health && state.health.words_per_credit) || (state.packs && state.packs.words_per_credit) || 0;
    return Number(w) || 0;
  }

  function costFor(wordCount) {
    var w = wordsPerCredit();
    var n = Math.max(0, Number(wordCount) || 0);
    if (!w) return 1;
    return Math.max(1, Math.ceil(n / w));
  }

  /* ---- the pill --------------------------------------------------------- */

  function mountPill() {
    if (ui.mount) return;
    var tools = document.querySelector('.top-tools');
    ui.mount = h('div', { 'class': 'rha-pill' + (tools ? '' : ' rha-pill-fixed') });
    if (tools) tools.appendChild(ui.mount);
    else document.body.appendChild(ui.mount);
    renderPill();
  }

  function renderPill() {
    if (!ui.mount) return;
    clear(ui.mount);
    if (!state.user) {
      ui.mount.appendChild(h('button', { 'class': 'chip', type: 'button', on: { click: function () { openSheet('signin'); } } }, ['Sign in']));
      return;
    }
    var c = state.user.credits;
    ui.mount.appendChild(h('span', { 'class': 'rha-pill-email', title: state.user.email, text: String(state.user.email || '') }));
    ui.mount.appendChild(h('span', { 'class': 'rha-pill-credits', title: plural(c, 'credit') }, [
      h('span', { 'class': 'num', text: String(c) }), ' ' + (c === 1 ? 'credit' : 'credits')
    ]));
    ui.mount.appendChild(h('button', { 'class': 'chip', type: 'button', on: { click: function () { openSheet('account'); } } }, ['Account']));
  }

  /* ---- toasts ----------------------------------------------------------- */

  function toast(message) {
    if (!ui.toast) {
      ui.toast = h('p', { 'class': 'rha-toast', role: 'status', 'aria-live': 'polite', hidden: true });
      document.body.appendChild(ui.toast);
    }
    ui.toast.textContent = message;
    ui.toast.removeAttribute('hidden');
    ui.toast.removeAttribute('data-in');
    void ui.toast.offsetWidth;
    ui.toast.setAttribute('data-in', '1');
    if (ui.toastTimer) clearTimeout(ui.toastTimer);
    ui.toastTimer = setTimeout(function () {
      ui.toast.removeAttribute('data-in');
      ui.toastTimer = setTimeout(function () { ui.toast.setAttribute('hidden', ''); }, 200);
    }, TOAST_MS);
  }

  /* ---- the sheet: one dialog, three contents ---------------------------- */

  var FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  function focusables() {
    var list = ui.sheet.querySelectorAll(FOCUSABLE);
    var out = [], i;
    for (i = 0; i < list.length; i++) if (list[i].offsetParent !== null || list[i] === document.activeElement) out.push(list[i]);
    return out;
  }

  function onSheetKey(e) {
    if (e.key === 'Escape' || e.key === 'Esc') { e.preventDefault(); closeSheet(); return; }
    if (e.key !== 'Tab') return;
    var f = focusables();
    if (!f.length) { e.preventDefault(); ui.sheet.focus(); return; }
    var first = f[0], last = f[f.length - 1];
    if (e.shiftKey && (document.activeElement === first || document.activeElement === ui.sheet)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function ensureSheet() {
    if (ui.backdrop) return;
    ui.sheet = h('div', { 'class': 'rha-sheet', role: 'dialog', 'aria-modal': 'true', tabindex: '-1', 'aria-labelledby': 'rha-title' });
    ui.backdrop = h('div', { 'class': 'rha-backdrop', hidden: true }, [ui.sheet]);
    ui.backdrop.addEventListener('mousedown', function (e) { if (e.target === ui.backdrop) closeSheet(); });
    ui.sheet.addEventListener('keydown', onSheetKey);
    document.body.appendChild(ui.backdrop);
  }

  function openSheet(name, extra) {
    ensureSheet();
    var wasOpen = !ui.backdrop.hasAttribute('hidden');
    if (!wasOpen) ui.restoreFocus = document.activeElement;
    ui.sheetName = name;
    clear(ui.sheet);
    ui.sheet.className = 'rha-sheet' + (name === 'account' ? ' rha-sheet-wide' : '');
    if (name === 'signin') buildSignin(extra || {});
    else if (name === 'packs') buildPacks(extra || {});
    else if (name === 'account') buildAccount();
    else return;
    ui.backdrop.removeAttribute('hidden');
    if (!wasOpen) {
      ui.backdrop.removeAttribute('data-in');
      void ui.backdrop.offsetWidth;
      ui.backdrop.setAttribute('data-in', '1');
    }
    var first = ui.sheet.querySelector('[data-autofocus]') || focusables()[0] || ui.sheet;
    setTimeout(function () { first.focus(); }, 0);
  }

  function closeSheet() {
    if (!ui.backdrop || ui.backdrop.hasAttribute('hidden')) return;
    ui.backdrop.removeAttribute('data-in');
    ui.sheetName = '';
    var hide = function () { if (ui.sheetName) return; ui.backdrop.setAttribute('hidden', ''); clear(ui.sheet); };
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) hide();
    else setTimeout(hide, 180);
    var back = ui.restoreFocus;
    ui.restoreFocus = null;
    if (back && typeof back.focus === 'function' && document.body.contains(back)) back.focus();
  }

  function sheetHead(title, lede) {
    var head = h('div', { 'class': 'rha-head' }, [
      h('h2', { 'class': 'rha-h', id: 'rha-title', text: title }),
      h('button', { 'class': 'chip', type: 'button', on: { click: closeSheet } }, ['Close'])
    ]);
    var frag = document.createDocumentFragment();
    frag.appendChild(head);
    if (lede) frag.appendChild(h('p', { 'class': 'rha-p', text: lede }));
    return frag;
  }

  function errLine() { return h('p', { 'class': 'rha-err', role: 'alert', hidden: true }); }

  function showErr(p, text) {
    if (!text) { p.setAttribute('hidden', ''); p.textContent = ''; return; }
    p.textContent = text;
    p.removeAttribute('hidden');
  }

  /* ---- sign in ---------------------------------------------------------- */

  function buildSignin(opts) {
    var err = errLine();
    var input = h('input', { 'class': 'text-input rha-input', id: 'rha-email', type: 'email', name: 'email', autocomplete: 'email', spellcheck: 'false', required: true, placeholder: 'you@example.com', 'data-autofocus': true });
    var submit = h('button', { 'class': 'btn btn-primary', type: 'submit' }, ['Email me a link']);
    var form = h('form', { 'class': 'rha-form', novalidate: true }, [
      h('label', { 'class': 'rha-label', 'for': 'rha-email', text: 'Email address' }),
      input,
      err,
      h('div', { 'class': 'rha-acts' }, [submit])
    ]);
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var email = input.value.trim();
      if (!email || email.indexOf('@') < 1) { showErr(err, 'Enter the email address you want the link sent to.'); input.focus(); return; }
      showErr(err, '');
      setBusy(submit, true);
      api('POST', '/api/auth/request-link', { email: email }).then(function (r) {
        setBusy(submit, false);
        if (r.ok && r.body && r.body.sent) {
          if (r.body.expires_in_minutes) state.linkMinutes = Number(r.body.expires_in_minutes) || state.linkMinutes;
          renderSent(r.body.email || email, r.body.dev_link || '');
          return;
        }
        var code = r.body && r.body.error;
        if (r.status === 429 && code === 'signup_limit') showErr(err, 'Too many new accounts from this connection today. Sign in with an existing address, or try again tomorrow.');
        else if (r.status === 429) showErr(err, 'Too many attempts. Wait a minute and try again.');
        else if (code === 'bad_email') showErr(err, 'That does not look like an email address. Check it and try again.');
        else if (code === 'blocked_email') showErr(err, 'Disposable addresses cannot be used. Use a regular address.');
        else if (r.status === 502) showErr(err, 'The sign-in email could not be sent. Try again in a moment.');
        else if (r.status === 0) showErr(err, 'The service could not be reached. Check your connection and try again.');
        else showErr(err, detailOf(r, 'Something went wrong. Try again.'));
        input.focus();
      });
    });

    ui.sheet.appendChild(sheetHead('Sign in', 'We email you a link that signs you in. There is no password.'));
    if (opts.message) {
      var note = h('p', { 'class': 'rha-err', role: 'alert', text: opts.message });
      ui.sheet.appendChild(note);
    }
    ui.sheet.appendChild(form);
    ui.sheet.appendChild(legalLinks());
  }

  function renderSent(email, devLink) {
    clear(ui.sheet);
    ui.sheet.appendChild(sheetHead('Check your email', 'We sent a sign-in link to ' + email + '. It works for ' + plural(state.linkMinutes, 'minute') + '. This tab can stay open.'));
    if (devLink) {
      ui.sheet.appendChild(h('p', { 'class': 'rha-note' }, [
        'This server is in development mode, so here is the link: ',
        h('a', { href: devLink, 'class': 'rha-link', text: 'open the sign-in link' })
      ]));
    }
    ui.sheet.appendChild(h('div', { 'class': 'rha-acts' }, [
      h('button', { 'class': 'btn', type: 'button', 'data-autofocus': true, on: { click: function () { openSheet('signin'); } } }, ['Use a different address']),
      h('button', { 'class': 'btn btn-primary', type: 'button', on: { click: closeSheet } }, ['Done'])
    ]));
    (ui.sheet.querySelector('[data-autofocus]') || ui.sheet).focus();
  }

  function legalLinks() {
    return h('p', { 'class': 'rha-links' }, [
      h('a', { href: '/legal/terms.html', 'class': 'rha-link', text: 'Terms' }),
      ' and ',
      h('a', { href: '/legal/privacy.html', 'class': 'rha-link', text: 'privacy' }),
      '.'
    ]);
  }

  /* ---- packs ------------------------------------------------------------ */

  function buildPacks(opts) {
    var needed = opts.needed !== undefined ? Number(opts.needed) : (state.blocked ? state.blocked.needed : null);
    var balance = opts.balance !== undefined ? Number(opts.balance) : (state.user ? state.user.credits : null);
    var lede = '';
    if (needed !== null && !isNaN(needed) && balance !== null) lede = 'This rewrite needs ' + plural(needed, 'credit') + '; you have ' + balance + '.';
    else if (state.user) lede = 'You have ' + plural(balance, 'credit') + '.';
    else lede = 'Credits pay for rewrites by the language model. Measuring is free.';

    ui.sheet.appendChild(sheetHead('Credits', lede));
    var list = h('div', { 'class': 'rha-rows', 'aria-busy': 'true' }, [h('p', { 'class': 'rha-note', text: 'Loading packs.' })]);
    var err = errLine();
    ui.sheet.appendChild(list);
    ui.sheet.appendChild(err);

    loadPacks().then(function (packs) {
      clear(list);
      list.removeAttribute('aria-busy');
      var w = wordsPerCredit();
      if (w) list.appendChild(h('p', { 'class': 'rha-note', text: 'One credit covers a rewrite of up to ' + w + ' words. Nothing is charged for a rewrite that comes back unchanged.' }));
      if (!state.user) {
        list.appendChild(h('p', { 'class': 'rha-p', text: 'Sign in to buy credits. New accounts start with ' + plural(Number((state.health && state.health.free_credits) || 0), 'free credit') + '.' }));
        list.appendChild(h('div', { 'class': 'rha-acts' }, [
          h('button', { 'class': 'btn btn-primary', type: 'button', on: { click: function () { openSheet('signin'); } } }, ['Sign in'])
        ]));
        return;
      }
      if (!packs.stripe) {
        list.appendChild(h('p', { 'class': 'rha-p', text: 'Purchases are not available yet on this server. Your free credits still work.' }));
        return;
      }
      if (!packs.packs.length) {
        list.appendChild(h('p', { 'class': 'rha-p', text: 'No packs are on sale right now.' }));
        return;
      }
      var i;
      for (i = 0; i < packs.packs.length; i++) list.appendChild(packRow(packs.packs[i], err));
    });
    ui.sheet.appendChild(legalLinks());
  }

  function packRow(pack, err) {
    var buy = h('button', { 'class': 'btn', type: 'button' }, ['Buy']);
    var row = h('div', { 'class': 'rha-row rha-pack' }, [
      h('div', { 'class': 'rha-pack-name' }, [
        h('span', { 'class': 'rha-strong', text: pack.label || pack.name }),
        h('span', { 'class': 'rha-note', text: plural(Number(pack.credits) || 0, 'credit') })
      ]),
      h('span', { 'class': 'num rha-pack-n', text: String(pack.credits) }),
      buy
    ]);
    buy.addEventListener('click', function () {
      showErr(err, '');
      setBusy(buy, true);
      api('POST', '/api/billing/checkout', { pack: pack.name }).then(function (r) {
        if (r.ok && r.body && r.body.url) { location.assign(r.body.url); return; }
        setBusy(buy, false);
        if (r.status === 401) { openSheet('signin', { message: 'Sign in before buying credits.' }); return; }
        if (r.status === 503) showErr(err, 'Purchases are not available yet on this server.');
        else if (r.status === 502) showErr(err, 'The payment provider could not be reached. Try again in a moment.');
        else showErr(err, detailOf(r, 'The purchase could not be started. Try again.'));
      });
    });
    return row;
  }

  /* ---- account panel ---------------------------------------------------- */

  function buildAccount() {
    if (!state.user) { buildSignin({}); return; }
    var u = state.user;
    ui.sheet.appendChild(sheetHead('Account'));

    ui.sheet.appendChild(h('div', { 'class': 'rha-rows' }, [
      h('div', { 'class': 'rha-row' }, [h('span', { 'class': 'rha-k', text: 'Email' }), h('span', { 'class': 'rha-v', text: u.email })]),
      h('div', { 'class': 'rha-row' }, [
        h('span', { 'class': 'rha-k', text: 'Credits' }),
        h('span', { 'class': 'rha-v' }, [h('span', { 'class': 'num', text: String(u.credits) })]),
        h('button', { 'class': 'btn', type: 'button', on: { click: function () { state.blocked = null; openSheet('packs'); } } }, ['Buy credits'])
      ])
    ]));

    var ledger = h('div', { 'class': 'rha-rows rha-ledger', 'aria-busy': 'true' }, [h('p', { 'class': 'rha-note', text: 'Loading.' })]);
    ui.sheet.appendChild(section('History', ledger));
    api('GET', '/api/billing/ledger').then(function (r) {
      clear(ledger);
      ledger.removeAttribute('aria-busy');
      var entries = (r.ok && r.body && r.body.entries) || [];
      if (r.ok && r.body && typeof r.body.balance === 'number') setBalance(r.body.balance);
      if (!entries.length) { ledger.appendChild(h('p', { 'class': 'rha-note', text: 'Nothing yet. Your first rewrite will appear here.' })); return; }
      var i, e, d;
      for (i = 0; i < Math.min(entries.length, 12); i++) {
        e = entries[i];
        d = Number(e.delta) || 0;
        ledger.appendChild(h('div', { 'class': 'rha-row rha-ledger-row' }, [
          h('span', { 'class': 'rha-note', text: dateOf(e.created) }),
          h('span', { 'class': 'num rha-delta', 'data-sign': d < 0 ? 'neg' : 'pos', text: (d > 0 ? '+' : '') + d }),
          h('span', { 'class': 'rha-kind', text: String(e.kind || '') }),
          h('span', { 'class': 'rha-ref num', title: String(e.ref || ''), text: String(e.ref || '') })
        ]));
      }
    });

    var keys = h('div', { 'class': 'rha-rows rha-keys', 'aria-busy': 'true' }, [h('p', { 'class': 'rha-note', text: 'Loading.' })]);
    var keyErr = errLine();
    var keyLabel = h('input', { 'class': 'text-input rha-input', id: 'rha-key-label', type: 'text', maxlength: '80', placeholder: 'What this key is for', autocomplete: 'off' });
    var keyBtn = h('button', { 'class': 'btn', type: 'submit' }, ['Create key']);
    var keyForm = h('form', { 'class': 'rha-inline', novalidate: true }, [
      h('label', { 'class': 'rha-label sr-only', 'for': 'rha-key-label', text: 'Key label' }),
      keyLabel, keyBtn
    ]);
    var once = h('div', { 'class': 'rha-once', hidden: true });
    keyForm.addEventListener('submit', function (e) {
      e.preventDefault();
      showErr(keyErr, '');
      setBusy(keyBtn, true);
      api('POST', '/api/auth/keys', { label: keyLabel.value.trim() }).then(function (r) {
        setBusy(keyBtn, false);
        if (!r.ok || !r.body || !r.body.key) { showErr(keyErr, detailOf(r, 'The key could not be created.')); return; }
        keyLabel.value = '';
        showOnce(once, r.body.key);
        loadKeys(keys);
      });
    });
    var keySection = section('API keys', h('div', {}, [
      h('p', { 'class': 'rha-note', text: 'For scripts. Send a key as "Authorization: Bearer <key>". Each key is shown once, when it is created.' }),
      keys, once, keyForm, keyErr
    ]));
    ui.sheet.appendChild(keySection);
    loadKeys(keys);

    var del = h('button', { 'class': 'btn rha-danger', type: 'button' }, ['Delete account']);
    var delBox = h('div', { 'class': 'rha-delete', hidden: true });
    del.addEventListener('click', function () { buildDelete(delBox); del.setAttribute('hidden', ''); delBox.removeAttribute('hidden'); (delBox.querySelector('input') || delBox).focus(); });
    ui.sheet.appendChild(h('div', { 'class': 'rha-foot' }, [
      legalLinks(),
      h('div', { 'class': 'rha-acts' }, [
        del,
        h('button', { 'class': 'btn', type: 'button', on: { click: signOut } }, ['Sign out'])
      ])
    ]));
    ui.sheet.appendChild(delBox);
  }

  function section(title, body) {
    return h('section', { 'class': 'rha-section' }, [h('h3', { 'class': 'rha-h3', text: title }), body]);
  }

  function loadKeys(box) {
    box.setAttribute('aria-busy', 'true');
    api('GET', '/api/auth/keys').then(function (r) {
      clear(box);
      box.removeAttribute('aria-busy');
      var list = (r.ok && r.body && r.body.keys) || [];
      if (!list.length) { box.appendChild(h('p', { 'class': 'rha-note', text: 'No keys yet.' })); return; }
      var i;
      for (i = 0; i < list.length; i++) box.appendChild(keyRow(list[i], box));
    });
  }

  function keyRow(k, box) {
    var revoke = h('button', { 'class': 'btn', type: 'button' }, ['Revoke']);
    revoke.addEventListener('click', function () {
      setBusy(revoke, true);
      api('DELETE', '/api/auth/keys/' + encodeURIComponent(k.id)).then(function (r) {
        if (r.ok) { toast('Key revoked.'); loadKeys(box); }
        else { setBusy(revoke, false); toast(detailOf(r, 'The key could not be revoked.')); }
      });
    });
    return h('div', { 'class': 'rha-row rha-key-row' }, [
      h('span', { 'class': 'num rha-prefix', text: String(k.prefix || '') + '…' }),
      h('span', { 'class': 'rha-v', text: k.label || 'No label' }),
      h('span', { 'class': 'rha-note', text: k.last_used ? 'Used ' + dateOf(k.last_used) : 'Created ' + dateOf(k.created) }),
      revoke
    ]);
  }

  function showOnce(box, key) {
    clear(box);
    var code = h('code', { 'class': 'num rha-key-value', text: key });
    var copy = h('button', { 'class': 'btn', type: 'button' }, ['Copy']);
    copy.addEventListener('click', function () {
      var done = function () { copy.textContent = 'Copied'; setTimeout(function () { copy.textContent = 'Copy'; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(key).then(done, function () { selectText(code); });
      else selectText(code);
    });
    box.appendChild(h('p', { 'class': 'rha-p', text: 'Your new key. Copy it now; it is not shown again.' }));
    box.appendChild(h('div', { 'class': 'rha-inline' }, [code, copy]));
    box.removeAttribute('hidden');
    copy.focus();
  }

  function selectText(node) {
    try {
      var range = document.createRange();
      range.selectNodeContents(node);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    } catch (e) { /* nothing to do */ }
  }

  function buildDelete(box) {
    clear(box);
    var err = errLine();
    var input = h('input', { 'class': 'text-input rha-input', id: 'rha-delete-word', type: 'text', autocomplete: 'off', spellcheck: 'false', placeholder: 'delete' });
    var confirm = h('button', { 'class': 'btn rha-danger', type: 'submit', disabled: true }, ['Delete my account']);
    input.addEventListener('input', function () { if (input.value.trim().toLowerCase() === 'delete') confirm.removeAttribute('disabled'); else confirm.setAttribute('disabled', ''); });
    var form = h('form', { 'class': 'rha-form', novalidate: true }, [
      h('p', { 'class': 'rha-p', text: 'This signs you out everywhere, revokes your keys and removes your email. Credits you have not used are lost. Type delete to confirm.' }),
      h('label', { 'class': 'rha-label sr-only', 'for': 'rha-delete-word', text: 'Type delete to confirm' }),
      input, err,
      h('div', { 'class': 'rha-acts' }, [
        h('button', { 'class': 'btn', type: 'button', on: { click: function () { openSheet('account'); } } }, ['Keep my account']),
        confirm
      ])
    ]);
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      if (input.value.trim().toLowerCase() !== 'delete') return;
      setBusy(confirm, true);
      api('DELETE', '/api/me').then(function (r) {
        if (!r.ok) { setBusy(confirm, false); showErr(err, detailOf(r, 'The account could not be deleted.')); return; }
        setUser(null);
        closeSheet();
        toast('Your account was deleted.');
      });
    });
    box.appendChild(form);
  }

  function signOut() {
    api('POST', '/api/auth/logout').then(function () {
      setUser(null);
      closeSheet();
      toast('Signed out.');
    });
  }

  /* ---- fetch interception ----------------------------------------------- */

  function urlOf(input) {
    try {
      var s = (input && typeof input === 'object' && typeof input.url === 'string') ? input.url : String(input);
      return new URL(s, location.href);
    } catch (e) { return null; }
  }

  function isApiRequest(input) {
    var u = urlOf(input);
    if (!u) return false;
    if (API_BASE) {
      var b = urlOf(API_BASE);
      if (!b) return false;
      return u.origin === b.origin && u.pathname.indexOf((b.pathname.replace(/\/$/, '')) + '/api/') === 0;
    }
    return u.origin === location.origin && u.pathname.indexOf('/api/') === 0;
  }

  function isOurs(input) {
    var u = urlOf(input);
    var p = u ? u.pathname : '';
    return /\/api\/(me|auth\/|billing\/)/.test(p);
  }

  function retryAfter(res, body) {
    var s = Number(body && body.retry_after_s);
    if (!s) s = Number(res.headers.get('Retry-After'));
    if (!s || isNaN(s)) s = 10;
    return Math.min(MAX_RETRY_S, Math.max(1, Math.round(s)));
  }

  function inspect(res, input, init, retryCopy, retried) {
    var bal = res.headers.get('X-Longhand-Credits-Balance');
    if (bal !== null && bal !== '') setBalance(Number(bal));
    var charged = Number(res.headers.get('X-Longhand-Credits-Charged'));
    if (charged > 0) setTimeout(refresh, 2500); /* an unchanged rewrite is refunded after the response */

    if (res.ok) return res;
    var ct = String(res.headers.get('Content-Type') || '').toLowerCase();
    if (ct.indexOf('text/event-stream') >= 0) return res;
    if (res.status !== 401 && res.status !== 402 && res.status !== 503 && res.status !== 429) return res;
    if (ct.indexOf('json') < 0) return res;

    return res.clone().json().then(function (body) {
      var code = body && body.error;
      if (res.status === 401 && code === 'login_required') {
        openSheet('signin', { message: 'Sign in to run this rewrite. Measuring stays free.' });
        return res;
      }
      if (res.status === 402 && code === 'insufficient_credits') {
        state.blocked = { needed: Number(body.needed), balance: Number(body.balance) };
        if (state.user && !isNaN(state.blocked.balance)) setBalance(state.blocked.balance);
        openSheet('packs', state.blocked);
        return res;
      }
      if (res.status === 503 && code === 'busy') {
        if (retried) { toast('The rewriter is still busy. Try again in a minute.'); return res; }
        var wait = retryAfter(res, body);
        toast('The rewriter is busy, retrying in ' + wait + ' s');
        return new Promise(function (resolve) {
          setTimeout(function () {
            if (init && init.signal && init.signal.aborted) { resolve(res); return; }
            var again = retryCopy || input;
            rawFetch(again, init).then(function (r2) {
              resolve(inspect(r2, input, init, null, true));
            }, function () { resolve(res); });
          }, wait * 1000);
        });
      }
      if (res.status === 429) {
        toast(code === 'signup_limit'
          ? 'Too many new accounts from this connection today.'
          : 'Too many requests. Wait ' + retryAfter(res, body) + ' s and try again.');
        return res;
      }
      return res;
    }, function () { return res; });
  }

  window.fetch = function (input, init) {
    if (!isApiRequest(input) || isOurs(input)) return rawFetch(input, init);
    var copy = null;
    try { if (typeof Request !== 'undefined' && input instanceof Request) copy = input.clone(); } catch (e) { copy = null; }
    return rawFetch(input, init).then(function (res) { return inspect(res, input, init, copy, false); });
  };

  /* ---- load-time query parameters --------------------------------------- */

  function cleanUrl(keys) {
    try {
      var u = new URL(location.href);
      var i, had = false;
      for (i = 0; i < keys.length; i++) if (u.searchParams.has(keys[i])) { u.searchParams['delete'](keys[i]); had = true; }
      if (had) history.replaceState(history.state, '', u.pathname + (u.search || '') + u.hash);
    } catch (e) { /* leave the URL */ }
  }

  function handleQuery(startCredits) {
    var loginError = PARAMS.get('login_error');
    var purchase = PARAMS.get('purchase');
    if (loginError) {
      var msg = loginError === 'used_token'
        ? 'That sign-in link was already used. Request a new one.'
        : 'That sign-in link is invalid or has expired. Request a new one.';
      openSheet('signin', { message: msg });
    }
    if (purchase === 'cancelled') toast('Purchase cancelled. Nothing was charged.');
    if (purchase === 'success') pollForCredits(startCredits);
    cleanUrl(['login_error', 'purchase']);
  }

  function pollForCredits(startCredits) {
    var began = Date.now();
    var start = typeof startCredits === 'number' ? startCredits : (state.user ? state.user.credits : 0);
    function tick() {
      refresh().then(function (u) {
        if (u && u.credits > start) { toast('Credits added. You have ' + plural(u.credits, 'credit') + '.'); return; }
        if (Date.now() - began >= POLL_LIMIT_MS) { toast('Payment received. Your credits arrive in a moment; reload to see them.'); return; }
        setTimeout(tick, POLL_MS);
      });
    }
    toast('Payment received. Adding your credits.');
    tick();
  }

  /* ---- bootstrap -------------------------------------------------------- */

  function boot() {
    var healthP = api('GET', '/api/billing/health');
    var meP = api('GET', '/api/me');
    Promise.all([healthP, meP]).then(function (rs) {
      var health = rs[0].ok ? rs[0].body : null;
      var me = rs[1].ok ? rs[1].body : null;
      state.health = health;
      if (health && health.magic_link_minutes) state.linkMinutes = Number(health.magic_link_minutes) || state.linkMinutes;
      state.paywall = !!((health && health.paywall) || (me && me.paywall));
      if (me && me.user) state.user = me.user;
      if (!state.paywall) { emit('readshuman:account', { user: state.user }); return; }
      mountPill();
      emit('readshuman:account', { user: state.user });
      handleQuery(state.user ? state.user.credits : 0);
    });
  }

  window.readshumanAccount = {
    costFor: costFor,
    refresh: refresh,
    open: function (name) { openSheet(name || 'account'); },
    close: closeSheet,
    user: function () { return state.user; },
    health: function () { return state.health; }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
