/* The free check (/check). POST /api/detect with the pasted draft, which is
   free in the billing gate, and show the reading: the whole-text probability
   from the default engine (the local GPTZero surrogate) and each sentence
   shaded by its own score. "Rewrite it" leaves the draft in sessionStorage
   under 'vervly:handoff' and opens /app, where app.js picks it up and scores
   it. ES5, no framework, like app.js. */
(function () {
  'use strict';
  var PARAMS = new URLSearchParams(location.search);
  var API_BASE = PARAMS.get('api') || '';
  var HANDOFF_STORE = 'vervly:handoff';
  var MIN_WORDS = 40;
  var MAX_CHARS = 60000;

  var SAMPLE = 'In today’s rapidly evolving digital landscape, it is important to note that effective communication plays a crucial role in fostering collaboration. By leveraging innovative tools and embracing a growth mindset, teams can navigate complex challenges and unlock their full potential. Furthermore, cultivating a culture of transparency ensures that every stakeholder feels valued and heard. Ultimately, organizations that prioritize these principles are well positioned to thrive in an increasingly competitive environment.';

  function $(id) { return document.getElementById(id); }
  function words(t) { var m = String(t || '').trim().match(/\S+/g); return m ? m.length : 0; }
  function track(name, props) { try { if (typeof window.vervlyTrack === 'function') window.vervlyTrack(name, props || {}); } catch (e) { /* never */ } }

  var text = $('check-text'), btn = $('check-btn'), status = $('check-status'), counts = $('check-counts');
  var result = $('check-result'), verdict = $('check-verdict'), pct = $('check-pct'), label = $('check-label'), note = $('check-note');
  var list = $('check-sentences'), cta = $('check-cta');
  var busy = false, lastText = '';

  function setStatus(kind, msg) { status.setAttribute('data-kind', kind); status.textContent = msg; }

  function renderCounts() {
    var n = words(text.value);
    counts.textContent = n ? (n + (n === 1 ? ' word' : ' words')) : '';
    btn.disabled = busy || n < MIN_WORDS;
    if (!busy) {
      if (!n) setStatus('idle', 'Paste a draft to begin.');
      else if (n < MIN_WORDS) setStatus('idle', 'A few more words: ' + (MIN_WORDS - n) + ' to go. Detectors are not reliable under ' + MIN_WORDS + '.');
      else if (text.value !== lastText) setStatus('idle', 'Ready to check.');
    }
  }

  function bandOf(p, reported) {
    var s = String(reported || '').toLowerCase();
    if (s === 'human' || s === 'human-written') return 'human';
    if (s === 'ai' || s === 'ai-written' || s === 'ai_written') return 'ai';
    if (s === 'mixed') return 'mixed';
    if (p === null) return '';
    if (p >= 0.65) return 'ai';
    if (p < 0.35) return 'human';
    return 'mixed';
  }

  var LABELS = {
    human: ['likely written by a person', 'A detector would most likely read the whole text as human. The shaded sentences are the ones that pull the other way.'],
    mixed: ['could go either way', 'A detector would call this mixed: some sentences read as generated, some as written. The darker the sentence, the more it reads as generated.'],
    ai: ['likely to be flagged as AI', 'A detector would most likely call the whole text generated. The darker the sentence, the more it reads that way.']
  };

  function riskLevel(p) {
    if (p === null || isNaN(p)) return 0;
    if (p < 0.2) return 0;
    if (p < 0.4) return 1;
    if (p < 0.55) return 2;
    if (p < 0.7) return 3;
    if (p < 0.85) return 4;
    return 5;
  }

  function render(data, draft) {
    var rows = Array.isArray(data.detectors) ? data.detectors : [];
    var row = rows.length ? rows[0] : null;
    if (!row || row.available === false || typeof row.ai_probability !== 'number') {
      var why = row && row.error ? String(row.error) : 'the detector did not answer';
      setStatus('error', 'Could not score this: ' + why.slice(0, 160) + '.');
      result.hidden = true;
      return;
    }
    var p = row.ai_probability;
    var band = bandOf(p, row.label);
    verdict.setAttribute('data-band', band);
    pct.textContent = String(Math.round(p * 100));
    var copy = LABELS[band] || LABELS.mixed;
    label.textContent = copy[0];
    note.textContent = copy[1];

    list.textContent = '';
    var sentences = Array.isArray(data.sentences) ? data.sentences : [];
    var scores = Array.isArray(row.sentence_scores) ? row.sentence_scores : [];
    for (var i = 0; i < sentences.length; i++) {
      var li = document.createElement('li');
      var sp = typeof scores[i] === 'number' ? scores[i] : null;
      li.setAttribute('data-risk', String(riskLevel(sp)));
      var s = document.createElement('span'); s.textContent = sentences[i]; li.appendChild(s);
      var n = document.createElement('span'); n.className = 'num'; n.textContent = sp === null ? '' : Math.round(sp * 100) + '%'; li.appendChild(n);
      list.appendChild(li);
    }
    list.hidden = sentences.length === 0;

    cta.hidden = band === 'human';
    result.hidden = false;
    setStatus('ok', 'Scored ' + words(draft) + ' words.');
    track('check', { words: words(draft), pct: Math.round(p * 100), band: band });
    try { result.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } catch (e) { /* older browsers */ }
  }

  function check() {
    var draft = text.value;
    if (busy || words(draft) < MIN_WORDS) return;
    if (draft.length > MAX_CHARS) { setStatus('error', 'That is longer than the check reads at once. Try the first ' + MAX_CHARS + ' characters, or open the humanizer for a whole document.'); return; }
    busy = true; btn.disabled = true;
    setStatus('loading', 'Scoring ' + words(draft) + ' words.');
    var ctl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, 90000) : null;
    fetch(API_BASE + '/api/detect', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ text: draft }), signal: ctl ? ctl.signal : undefined, cache: 'no-store'
    }).then(function (res) {
      if (!res.ok) return res.text().then(function (t) { throw new Error('HTTP ' + res.status + (t ? ': ' + t.slice(0, 120) : '')); });
      return res.json();
    }).then(function (data) {
      lastText = draft;
      render(data && typeof data === 'object' ? data : {}, draft);
    }, function (err) {
      var msg = err && err.name === 'AbortError' ? 'the service took too long' : ((err && err.message) || 'unknown error');
      setStatus('error', 'That did not work: ' + msg + '. Try again in a moment.');
    }).then(function () {
      if (timer) clearTimeout(timer);
      busy = false; renderCounts();
    });
  }

  function rewrite() {
    var draft = text.value;
    if (!draft.trim()) return;
    try { sessionStorage.setItem(HANDOFF_STORE, draft); } catch (e) { /* private mode: the app opens empty */ }
    track('handoff', { words: words(draft) });
    location.assign('/app?from=check');
  }

  text.addEventListener('input', renderCounts);
  btn.addEventListener('click', check);
  $('check-sample').addEventListener('click', function () { text.value = SAMPLE; renderCounts(); text.focus(); });
  $('check-rewrite').addEventListener('click', rewrite);
  text.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); check(); }
  });
  renderCounts();
})();
