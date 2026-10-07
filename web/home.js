/* Vervly landing page.

   One job: ask GET /api/auth/me who is here. Signed in, the primary action
   becomes "Open the humanizer" and the top row shows the name. Anything
   else (not signed in, no such route yet, no network) leaves the page as
   written: Try it free (the app, no account needed) and Sign in. ES5, no
   build step. */
(function () {
  'use strict';

  function $(id) { return document.getElementById(id); }
  function each(sel, fn) { var list = document.querySelectorAll(sel); for (var i = 0; i < list.length; i++) fn(list[i]); }

  function signedIn(user) {
    var primary = $('primary-act');
    if (primary) { primary.textContent = 'Open the humanizer'; primary.setAttribute('href', '/app'); }
    var secondary = $('secondary-act');
    if (secondary) secondary.hidden = true;
    each('[data-when="out"]', function (n) { n.hidden = true; });
    each('[data-when="in"]', function (n) { n.hidden = false; });
    var who = $('nav-who');
    if (who) {
      var name = (user && (user.name || user.email)) || '';
      who.textContent = name;
      who.title = (user && user.email) || '';
      who.hidden = !name;
    }
  }

  /* reveal the specimen as it scrolls into view, once. Without
     IntersectionObserver everything is simply shown. */
  function reveal() {
    var marked = document.querySelectorAll('[data-reveal]');
    if (!marked.length) return;
    if (typeof IntersectionObserver !== 'function') {
      each('[data-reveal]', function (n) { n.classList.add('is-in'); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add('is-in');
        io.unobserve(e.target);
      });
    }, { rootMargin: '0px 0px -12% 0px', threshold: 0.15 });
    each('[data-reveal]', function (n) { io.observe(n); });
  }

  function whoAmI() {
    if (typeof fetch !== 'function') return;
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var opts = { method: 'GET', headers: { Accept: 'application/json' }, credentials: 'same-origin', cache: 'no-store' };
    if (controller) opts.signal = controller.signal;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 4000) : null;
    fetch('/api/auth/me', opts).then(function (res) {
      if (timer) clearTimeout(timer);
      if (!res.ok) return null;
      return res.json();
    }).then(function (me) {
      if (me && me.signed_in) signedIn(me.user || {});
    }).catch(function () { /* the page already shows both ways in */ });
  }

  function start() { reveal(); whoAmI(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
