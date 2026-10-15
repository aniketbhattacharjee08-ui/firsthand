/* Vervly's own page statistics. No cookie, no third party, nothing stored
   in the browser: one POST per page view and one per named event, to the
   same server that serves the page (POST /api/stats/event, see
   src/humanizer/api/stats.py). What is sent: the path, the referrer's host,
   the utm_source / utm_medium / utm_campaign (or ?ref=) from the address,
   the event name and a small bag of numbers. Never the text. The server
   counts a visitor by a hash that changes every day. Browsers that send
   DNT or Global Privacy Control are counted without the hash.

   Pages call window.vervlyTrack(name, props) for their own events; app.js
   does for a rewrite's start and finish. ES5, loaded with defer. */
(function () {
  'use strict';
  if (typeof window === 'undefined' || !window.location) return;
  /* a page served from a file or a dev harness without the API has no one to tell */
  var PARAMS;
  try { PARAMS = new URLSearchParams(location.search); } catch (e) { PARAMS = null; }
  var API_BASE = (PARAMS && PARAMS.get('api')) || '';
  var ENDPOINT = API_BASE + '/api/stats/event';

  function get(k) { return PARAMS ? (PARAMS.get(k) || '') : ''; }

  var refHost = '';
  try {
    if (document.referrer) {
      var h = new URL(document.referrer).host;
      if (h && h !== location.host) refHost = h;
    }
  } catch (e) { refHost = ''; }

  var source = get('utm_source') || get('ref') || get('source');
  var medium = get('utm_medium');
  var campaign = get('utm_campaign');

  function send(name, props) {
    var body;
    try {
      body = JSON.stringify({
        name: String(name || ''),
        path: location.pathname,
        ref: refHost,
        source: source,
        medium: medium,
        campaign: campaign,
        props: props && typeof props === 'object' ? props : {}
      });
    } catch (e) { return; }
    try {
      if (navigator.sendBeacon) {
        var blob = new Blob([body], { type: 'application/json' });
        if (navigator.sendBeacon(ENDPOINT, blob)) return;
      }
    } catch (e) { /* fall through to fetch */ }
    try {
      if (window.fetch) {
        fetch(ENDPOINT, { method: 'POST', body: body, headers: { 'Content-Type': 'application/json' }, keepalive: true, credentials: 'same-origin' })
          .catch(function () { /* statistics are never worth an error */ });
      }
    } catch (e) { /* nothing to do */ }
  }

  window.vervlyTrack = function (name, props) { send(name, props); };
  send('pageview', {});
})();
