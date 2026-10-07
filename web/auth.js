/* Vervly sign-in and create-account form.

   One file for both modes. The mode comes from the path (/signin or
   /signup; on a development server without those routes, ?mode=signup) and
   the two links at the top of the card switch it in place with pushState.
   On success the page goes to ?next= (a same-origin path, default /app).

   Calls, same origin, cookies included:
     POST /api/auth/signin  {email, password}         200 {ok, user} | 401 {error: "invalid_credentials"}
     POST /api/auth/signup  {email, password, name?}  201 {ok, user} | 409 (exists) | 422 (bad email or short password)

   ES5, no build step. */
(function () {
  'use strict';

  function $(id) { return document.getElementById(id); }

  var API = { signin: '/api/auth/signin', signup: '/api/auth/signup' };
  var COPY = {
    signin: { title: 'Sign in to Vervly', submit: 'Sign in', busy: 'Signing in', autocomplete: 'current-password' },
    signup: { title: 'Create your Vervly account', submit: 'Create account', busy: 'Creating your account', autocomplete: 'new-password' }
  };
  var EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  var MIN_PASSWORD = 8;

  var mode = 'signin';
  var busy = false;

  var form, alertBox, emailIn, passIn, nameIn, submitBtn, toggleBtn;

  /* ── the mode ───────────────────────────────────────────────────────── */

  function pathMode() {
    var p = location.pathname.replace(/\/+$/, '');
    if (p === '/signup') return 'signup';
    if (p === '/signin') return 'signin';
    return null;
  }
  function modeFromLocation() {
    var m = pathMode();
    if (m) return m;
    return new URLSearchParams(location.search).get('mode') === 'signup' ? 'signup' : 'signin';
  }
  /* the address for a mode: the real path when the server has it, a query
     parameter when this file is being served straight from web/ */
  function urlFor(m) {
    var search = new URLSearchParams(location.search);
    if (pathMode()) {
      search.delete('mode');
      var q = search.toString();
      return '/' + m + (q ? '?' + q : '');
    }
    search.set('mode', m);
    return location.pathname + '?' + search.toString();
  }
  function nextUrl() {
    var n = new URLSearchParams(location.search).get('next') || '/app';
    /* a path on this origin only: one leading slash, no scheme, no host */
    if (!/^\/(?![\/\\])/.test(n)) n = '/app';
    return n;
  }

  function setMode(m, push) {
    mode = m === 'signup' ? 'signup' : 'signin';
    var c = COPY[mode];
    document.title = c.title;
    $('auth-h').textContent = c.title;
    submitBtn.textContent = busy ? c.busy : c.submit;
    form.setAttribute('action', API[mode]);
    $('f-name').hidden = mode !== 'signup';
    $('password-hint').hidden = mode !== 'signup';
    passIn.setAttribute('autocomplete', c.autocomplete);
    ['signin', 'signup'].forEach(function (k) {
      var a = $('mode-' + k);
      a.setAttribute('href', urlFor(k));
      if (k === mode) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    clearErrors();
    if (push && history.pushState) history.pushState({ mode: mode }, '', urlFor(mode));
  }

  /* ── errors ─────────────────────────────────────────────────────────── */

  function fieldError(input, id, msg) {
    var err = $(id);
    if (msg) {
      err.textContent = msg; err.hidden = false;
      input.setAttribute('aria-invalid', 'true');
    } else {
      err.textContent = ''; err.hidden = true;
      input.removeAttribute('aria-invalid');
    }
  }
  function say(msg, kind) {
    if (!msg) { alertBox.hidden = true; alertBox.textContent = ''; alertBox.removeAttribute('data-kind'); return; }
    alertBox.textContent = msg;
    if (kind) alertBox.setAttribute('data-kind', kind); else alertBox.removeAttribute('data-kind');
    alertBox.hidden = false;
  }
  function clearErrors() {
    say('');
    fieldError(emailIn, 'email-err', '');
    fieldError(passIn, 'password-err', '');
  }

  /* what the person can fix before anything is sent */
  function validate() {
    var ok = true, first = null;
    var email = emailIn.value.trim();
    if (!email) { fieldError(emailIn, 'email-err', 'Enter your email address.'); ok = false; first = first || emailIn; }
    else if (!EMAIL.test(email)) { fieldError(emailIn, 'email-err', 'That does not look like an email address.'); ok = false; first = first || emailIn; }
    else fieldError(emailIn, 'email-err', '');
    var pw = passIn.value;
    if (!pw) { fieldError(passIn, 'password-err', 'Enter your password.'); ok = false; first = first || passIn; }
    else if (mode === 'signup' && pw.length < MIN_PASSWORD) { fieldError(passIn, 'password-err', 'Use at least 8 characters.'); ok = false; first = first || passIn; }
    else fieldError(passIn, 'password-err', '');
    if (first) first.focus();
    return ok;
  }

  /* the server's answer, in plain words */
  function serverError(status, body) {
    var text = '';
    if (body && typeof body === 'object') {
      text = String(body.error || body.detail || body.message || '');
      if (Array.isArray(body.detail)) text = body.detail.map(function (d) { return (d && (d.msg || d.message)) || ''; }).join(' ');
    }
    if (status === 401) {
      say('That email and password do not match.');
      passIn.setAttribute('aria-invalid', 'true');
      passIn.focus(); passIn.select();
      return;
    }
    if (status === 409) {
      setMode('signin', true);
      say('There is already an account for that email. Sign in instead.');
      passIn.value = '';
      passIn.focus();
      return;
    }
    if (status === 422 || status === 400) {
      var lower = text.toLowerCase();
      var hit = false;
      if (lower.indexOf('email') >= 0) { fieldError(emailIn, 'email-err', 'Enter a valid email address.'); hit = true; }
      if (lower.indexOf('password') >= 0) { fieldError(passIn, 'password-err', 'Use at least 8 characters.'); hit = true; }
      if (!hit) say('Check the email address, and use a password of at least 8 characters.');
      (lower.indexOf('email') >= 0 ? emailIn : passIn).focus();
      return;
    }
    if (status === 429) { say('Too many attempts. Wait a minute and try again.'); return; }
    if (status === 404 || status === 405) { say('Signing in is not available on this server yet.'); return; }
    if (status >= 500) { say('The service hit a problem. Try again in a moment.'); return; }
    say('The request did not go through (HTTP ' + status + '). Try again.');
  }

  /* ── the request ────────────────────────────────────────────────────── */

  function setBusy(on) {
    busy = on;
    form.setAttribute('aria-busy', on ? 'true' : 'false');
    submitBtn.disabled = on;
    submitBtn.textContent = on ? COPY[mode].busy : COPY[mode].submit;
  }

  function submit(e) {
    e.preventDefault();
    if (busy) return;
    say('');
    if (!validate()) return;
    var body = { email: emailIn.value.trim(), password: passIn.value };
    if (mode === 'signup' && nameIn.value.trim()) body.name = nameIn.value.trim();
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var opts = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
      credentials: 'same-origin',
      cache: 'no-store'
    };
    if (controller) opts.signal = controller.signal;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 15000) : null;
    setBusy(true);
    fetch(API[mode], opts).then(function (res) {
      if (timer) clearTimeout(timer);
      return res.text().then(function (t) {
        var json = null;
        try { json = t ? JSON.parse(t) : null; } catch (err) { json = null; }
        return { status: res.status, ok: res.ok, body: json };
      });
    }).then(function (r) {
      if (r.ok) {
        submitBtn.textContent = mode === 'signup' ? 'Account created' : 'Signed in';
        location.assign(nextUrl());
        return;
      }
      setBusy(false);
      serverError(r.status, r.body);
    }, function (err) {
      if (timer) clearTimeout(timer);
      setBusy(false);
      if (err && err.name === 'AbortError') say('The service took too long to answer. Try again.');
      else if (typeof navigator !== 'undefined' && navigator.onLine === false) say('You are offline. Connect and try again.');
      else say('The service could not be reached. Try again in a moment.');
    });
  }

  /* ── show or hide the password ──────────────────────────────────────── */

  function togglePassword() {
    var show = passIn.type === 'password';
    passIn.type = show ? 'text' : 'password';
    toggleBtn.setAttribute('aria-pressed', show ? 'true' : 'false');
    toggleBtn.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
    toggleBtn.textContent = show ? 'Hide' : 'Show';
    passIn.focus();
  }

  /* ── wiring ─────────────────────────────────────────────────────────── */

  function init() {
    form = $('auth-form'); alertBox = $('auth-alert');
    emailIn = $('email'); passIn = $('password'); nameIn = $('name');
    submitBtn = $('auth-submit'); toggleBtn = $('pw-toggle');

    setMode(modeFromLocation(), false);

    /* provider buttons: carry ?next= through, and explain a failed return */
    var nxt = nextUrl();
    ['google', 'apple'].forEach(function (p) {
      var a = $('oauth-' + p);
      if (a) a.setAttribute('href', '/api/auth/oauth/' + p + '/start?next=' + encodeURIComponent(nxt));
    });
    /* only show the providers this server has keys for. GET /api/auth/providers
       answers {google: bool, apple: bool}; a button for a provider that is off
       would only dead-end in an error, so it goes, and so does the "or" when
       none are left. If the call fails the page stays as written. */
    (function showProviders() {
      if (!window.fetch) return;
      fetch('/api/auth/providers', { credentials: 'same-origin' }).then(function (r) {
        return r.ok ? r.json() : null;
      }).then(function (on) {
        if (!on) return;
        var left = 0;
        ['google', 'apple'].forEach(function (p) {
          var a = $('oauth-' + p);
          if (!a) return;
          if (on[p]) { left++; } else { a.hidden = true; }
        });
        var wrap = $('auth-oauth'), or = document.querySelector('.auth-or');
        if (!left) { if (wrap) wrap.hidden = true; if (or) or.hidden = true; }
      }).catch(function () {});
    })();
    var oerr = new URLSearchParams(location.search).get('error');
    if (oerr) {
      var OERR = {
        google_unavailable: 'Google sign-in is not set up on this server yet. Use your email and password.',
        apple_unavailable: 'Apple sign-in is not set up on this server yet. Use your email and password.',
        oauth_denied: 'The sign-in was cancelled before it finished.',
        oauth_state: 'That sign-in link had expired. Try again.',
        oauth_email_missing: 'The provider did not share a verified email address, so no account could be matched. Use your email and password.',
        oauth_failed: 'The provider sign-in did not complete. Try again, or use your email and password.'
      };
      say(OERR[oerr] || OERR.oauth_failed);
      if (history.replaceState) {
        var clean = new URL(location.href); clean.searchParams.delete('error');
        history.replaceState(null, '', clean.pathname + (clean.search || '') + clean.hash);
      }
    }

    ['signin', 'signup'].forEach(function (k) {
      $('mode-' + k).addEventListener('click', function (e) {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
        e.preventDefault();
        if (k !== mode) { setMode(k, true); emailIn.focus(); }
      });
    });
    window.addEventListener('popstate', function () { setMode(modeFromLocation(), false); });

    form.addEventListener('submit', submit);
    toggleBtn.addEventListener('click', togglePassword);
    /* a field that was wrong clears its message as soon as it is edited */
    emailIn.addEventListener('input', function () { fieldError(emailIn, 'email-err', ''); });
    passIn.addEventListener('input', function () { fieldError(passIn, 'password-err', ''); });

    /* a hint for the console and tests */
    window.__authReady = true;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
