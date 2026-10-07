/* Humanizer front end.

   No build step, no bundler, no runtime dependency. One classic script.

   The pipeline rewrites a draft with a base language model, a judge ranks
   the candidates, and meaning gates decide what is allowed to win. This file
   shows the draft on the left, the rewrite on the right, and the judge's
   reading of each. The rewrite never touches the draft until Use this.
   Everything else sits behind Details.

   The judge is a free local estimate of GPTZero unless a bench configured
   GPTZero itself (see judgeName). Typing pauses measure the draft once it is
   long enough, Measure does it on demand, and a finished Humanize run
   measures the rewrite. anim.js listens for the events emitted here.

   Also here: the divider between the panes (drag, arrow keys, Enter to
   fold), the two switches, and the construction layer that outlines every
   region with its label and the API field that fills it. */
(function () {
  'use strict';

  var PARAMS = new URLSearchParams(location.search);
  var API_BASE = PARAMS.get('api') || '';
  var DEBOUNCE_MS = 900;
  var AUTO_MEASURE_CHARS = 250;

  var LLM_STREAM_PATH = '/api/humanize/stream';
  var LLM_BLOCKING_PATH = '/api/humanize/llm';
  var RULE_PATH = '/api/humanize';
  /* the server plans 55 s per paragraph plus a 120 s repair stage (capped at
     480 + 120 s); the ceiling sits above that so a long document is stopped
     by the server's own budget, not by the page. */
  var LLM_CEILING_MS = 660000;

  /* ── small utilities ─────────────────────────────────────────────────── */

  function $(id) { return document.getElementById(id); }

  function num(v) { return typeof v === 'number' && isFinite(v) ? v : null; }

  /* some endpoints send numbers as strings */
  function loose(v) {
    if (typeof v === 'number') return isFinite(v) ? v : null;
    if (typeof v === 'string' && v.trim() !== '') {
      var n = Number(v);
      return isFinite(n) ? n : null;
    }
    return null;
  }

  function fmt(v, digits) {
    var n = num(v);
    if (n === null) return 'n/a';
    return n.toFixed(digits === undefined ? 2 : digits);
  }

  function pct(p) {
    var n = num(p);
    return n === null ? 'n/a' : Math.round(n * 100) + '%';
  }

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

  function plural(n, one, many) { return n === 1 ? one : (many || one + 's'); }

  function pickString() {
    for (var i = 0; i < arguments.length; i++) {
      if (typeof arguments[i] === 'string' && arguments[i].trim()) return arguments[i];
    }
    return '';
  }

  function words(text) {
    var m = text.toLowerCase().match(/[a-z][a-z'\u2019-]*/g);
    return m || [];
  }

  var reduceMotion = window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)')
    : { matches: false };

  /* anim.js listens for these; nothing depends on it */
  function emit(name, detail) {
    try { document.dispatchEvent(new CustomEvent(name, { detail: detail || null })); }
    catch (e) { /* animation is optional */ }
  }

  /* ── the measurement table (Details) ─────────────────────────────────── */

  var FEATURES = [
    { key: 'sent_len_cv', label: 'sentence length variation', band: [0.42, 0.60], digits: 2,
      plain: 'The spread of your sentence lengths divided by their average. Human academic writing lands between 0.42 and 0.60. Below it every sentence is about the same length; above it you have overcorrected.' },
    { key: 'sent_lag1_autocorr', label: 'long after long', band: [-0.10, 0.25], digits: 2,
      plain: 'Whether a long sentence tends to follow another long one. Humans sit near zero. A clearly negative value means long, short, long, short, which is what a naive burstiness rule produces.' },
    { key: 'sent_short_share', label: 'sentences under 10 words', band: [0.09, 0.14], pct: true,
      plain: 'Human academic prose keeps 9 to 14 percent of sentences this short. A few genuinely short sentences buy more variation than nudging every sentence a word or two.' },
    { key: 'sent_long_share', label: 'sentences over 30 words', band: [0.18, 0.28], pct: true,
      plain: 'Humans sit between 18 and 28 percent. If your variation is low, the usual cause is that nothing is long, not that nothing is short.' },
    { key: 'para_words_cv', label: 'paragraph length variation', band: [0.42, 0.71], digits: 2,
      plain: 'Human paragraphs vary more than human sentences do, in every genre measured. Evenly sized blocks read as generated even when the sentences inside them do not.' },
    { key: 'ai_vocab_weighted_per_1k', label: 'AI vocabulary per 1k words', band: [0, 8], digits: 1,
      plain: 'Words that turn up far more often in model output than in human writing, weighted by how lopsided the gap is. The strongest lexical signal there is, and removing it costs nothing.' },
    { key: 'formal_connective_per_1k', label: 'formal connectives per 1k', band: [1, 8], digits: 1,
      plain: '"Moreover", "Furthermore", "Additionally". Models open sentences with these far more often than people do. Cutting them lowers detection risk and raises the grade at the same time.' },
    { key: 'nominalization_per_1k', label: 'nominalizations per 1k', band: [55, 80], digits: 1,
      plain: 'Verbs turned into nouns. Academic writing runs 61 to 72 per thousand words, and stronger essays are more nominalized, not less, so do not strip these to sound human.' },
    { key: 'passive_per_1k', label: 'passives per 1k', band: [12, 25], digits: 1,
      plain: 'Academic prose sits around 18.5 passives per thousand words, roughly a quarter of finite verbs. The advice to avoid the passive comes from journalism, not from anyone who counted academic writing.' },
    { key: 'comma_per_1k', label: 'commas per 1k', band: [57, 65], digits: 1,
      plain: 'Punctuation habits are the most stable thing about a writer and barely move under paraphrase, which makes them a strong identity signal.' },
    { key: 'contraction_per_1k', label: 'contractions per 1k', band: [0, 1.4], digits: 2,
      plain: 'Research articles keep these under 1.4 per thousand words. Adding contractions lowers detection risk and lowers the grade, so it is a never in the body edit for academic work.' },
    { key: 'mtld', label: 'lexical diversity (MTLD)', band: null, digits: 1,
      plain: 'How varied your vocabulary is. There is no published academic band, and academic writing is less varied than fiction on purpose, so a higher number is not automatically better.' },
    { key: 'n_words', label: 'words', band: null, digits: 0,
      plain: 'Below roughly 300 words every shape measure above is noisy, and detectors hedge too.' },
    { key: 'n_sentences', label: 'sentences', band: null, digits: 0,
      plain: 'How many sentences the service found. An odd count usually means an unusual abbreviation or a stray full stop.' }
  ];

  function formatFeature(spec, value) {
    var n = num(value);
    if (n === null) return 'n/a';
    if (spec.pct) return (n * 100).toFixed(1) + '%';
    return n.toFixed(spec.digits === undefined ? 2 : spec.digits);
  }

  function formatBand(spec) {
    if (!spec.band) return null;
    var lo = spec.band[0], hi = spec.band[1];
    if (spec.pct) return (lo * 100).toFixed(0) + '% to ' + (hi * 100).toFixed(0) + '%';
    var d = spec.digits === undefined ? 2 : spec.digits;
    return lo.toFixed(d) + ' to ' + hi.toFixed(d);
  }

  function bandState(spec, value, serverBands) {
    var fromServer = serverBands && serverBands[spec.key];
    if (fromServer && fromServer !== 'unknown') return fromServer;
    var n = num(value);
    if (n === null) return 'unknown';
    if (!spec.band) return 'no band';
    if (n < spec.band[0]) return 'low';
    if (n > spec.band[1]) return 'high';
    return 'in band';
  }

  var SEVERITY_ORDER = { high: 0, medium: 1, low: 2, info: 3 };

  /* ── the example drafts ──────────────────────────────────────────────── */

  var SAMPLE = [
    'The relationship between urban green space and public health has become a pivotal area of academic inquiry. Researchers across a range of disciplines have begun to delve into the mechanisms that connect vegetation cover to wellbeing. Moreover, this work underscores a broader shift in how modern cities are planned, managed and understood. It is important to note that this shift is multifaceted, ongoing and highly context dependent.',
    'Furthermore, a comprehensive review of the existing literature reveals several recurring themes worth considering. First, access to green space is consistently associated with improved mental health indicators. Second, the observed effect appears robust across a wide range of demographic groups. Third, the underlying causal mechanisms remain the subject of considerable scholarly debate. These findings collectively demonstrate the significance of the topic for policy makers.',
    'Additionally, it is worth noting that methodological challenges persist throughout this body of research. Many studies rely on cross sectional designs that cannot establish causal direction with confidence. Consequently, the strength of the evidence base remains somewhat limited in scope. Nevertheless, the overall pattern of reported results is compelling and deserves attention. Therefore, further investigation is not merely desirable but genuinely necessary.',
    'In conclusion, urban green space represents an intricate intersection of environmental, social and psychological factors. Future research should explore the causal pathways at play in far greater depth. Ultimately, such work will pave the way for more effective and more equitable urban policy. This is not simply an academic exercise, but a pressing societal concern.'
  ].join('\n\n');

  /* the chips under the empty sheet; each is long enough to be measured on its own */
  var SAMPLES = {
    ai: SAMPLE,
    letter: [
      'I am writing to express my strong interest in the Research Assistant position at the Institute for Urban Studies. With a robust academic background in geography and a proven track record of collaborative fieldwork, I am confident that I would be a valuable addition to your dynamic team.',
      'Throughout my undergraduate studies, I honed my analytical skills and developed a deep passion for evidence based policy. Moreover, my role as a teaching assistant allowed me to cultivate strong communication skills and a commitment to fostering inclusive learning environments. Additionally, I have leveraged geographic information systems to deliver actionable insights on housing access.',
      'I am particularly drawn to the Institute because of its commitment to innovative, community centred research. I would welcome the opportunity to contribute to your ongoing projects and to further develop my expertise. Thank you for considering my application. I look forward to the possibility of discussing how my skills align with your needs.'
    ].join('\n\n'),
    report: [
      'The purpose of this experiment was to investigate the effect of temperature on the rate of an enzyme catalysed reaction. Catalase was selected as a model enzyme because it is readily available and its activity can be quantified through the volume of oxygen released. It is important to note that enzyme kinetics are highly sensitive to environmental conditions.',
      'Samples were incubated at five distinct temperatures ranging from 10 to 50 degrees Celsius. Subsequently, the volume of oxygen produced over a two minute interval was recorded for each condition. The results clearly demonstrate that reaction rate increased with temperature up to approximately 37 degrees, after which a marked decline was observed. This pattern is consistent with the denaturation of the enzyme at elevated temperatures.',
      'In conclusion, the findings underscore the crucial role of temperature in regulating enzymatic activity. Furthermore, they highlight the importance of maintaining optimal conditions in biological systems. Future experiments could explore a broader range of temperatures and incorporate additional replicates to enhance the reliability of the data.'
    ].join('\n\n'),
    history: [
      'The causes of the First World War have long been the subject of intense historical debate. Scholars have delved into a wide array of factors, ranging from the intricate web of alliances to the pervasive influence of nationalism. Moreover, the role of individual decision makers in the summer of 1914 continues to be scrutinised. It is essential to recognise that no single cause can fully account for the outbreak of hostilities.',
      'Firstly, the alliance system created a landscape in which a regional dispute could rapidly escalate into a continental conflict. Secondly, the arms race between the great powers fostered a climate of mutual suspicion and heightened tensions. Additionally, imperial rivalries in Africa and Asia exacerbated existing frictions between the European states. These factors collectively contributed to a volatile international environment.',
      'Ultimately, the assassination of Archduke Franz Ferdinand served as the catalyst that transformed underlying tensions into open war. Nevertheless, it would be overly simplistic to attribute the conflict to this single event. In conclusion, the war emerged from a complex interplay of structural pressures and contingent decisions, a combination that continues to shape how historians approach the period.'
    ].join('\n\n')
  };

  /* ── document model ──────────────────────────────────────────────────── */

  function paragraphRanges(text) {
    var out = [];
    var re = /\n{2,}/g;
    var start = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > start) out.push({ from: start, to: m.index });
      start = m.index + m[0].length;
    }
    if (start < text.length) out.push({ from: start, to: text.length });
    if (!out.length) out.push({ from: 0, to: text.length });
    return out;
  }

  /* the API returns sentence text, not offsets, so each is located in the
     document with a whitespace tolerant forward scan */
  function locateSentences(text, sentences) {
    var cursor = 0, out = [];
    (sentences || []).forEach(function (s) {
      var raw = s && typeof s.text === 'string' ? s.text.trim() : '';
      if (!raw) return;
      var from = text.indexOf(raw, cursor);
      var to;
      if (from >= 0) {
        to = from + raw.length;
      } else {
        var tokens = raw.split(/\s+/).slice(0, 200).map(escapeRe);
        var hit = null;
        try { hit = new RegExp(tokens.join('\\s+')).exec(text.slice(cursor)); }
        catch (e) { hit = null; }
        if (!hit) return;
        from = cursor + hit.index;
        to = from + hit[0].length;
      }
      cursor = to;
      out.push({
        index: num(s.index) === null ? out.length : s.index,
        text: text.slice(from, to),
        length: num(s.length),
        risk: num(s.risk),
        from: from,
        to: to
      });
    });
    return out;
  }

  /* ── HTTP ────────────────────────────────────────────────────────────── */

  /* The product API answers 401 {error: "sign_in_required"} when nobody is
     signed in. Every HTTP path here funnels through this: the page goes to
     the sign-in form and comes back to /app afterwards. Called at most once. */
  var SIGN_IN_URL = '/signin?next=' + encodeURIComponent('/app');
  var leaving = false;
  function signInRequired() {
    if (leaving) return;
    leaving = true;
    try { setStatus('idle', 'Sign in to continue.'); } catch (e) { /* before the DOM */ }
    location.replace(SIGN_IN_URL);
  }

  function request(path, body, timeoutMs) {
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var opts = { method: body ? 'POST' : 'GET', headers: { Accept: 'application/json' } };
    if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    if (controller) opts.signal = controller.signal;
    var limit = timeoutMs || 30000;
    var timer = controller ? setTimeout(function () { controller.abort(); }, limit) : null;

    return fetch(API_BASE + path, opts).then(function (res) {
      if (timer) clearTimeout(timer);
      if (res.status === 401) signInRequired();
      if (!res.ok) {
        return res.text().catch(function () { return ''; }).then(function (t) {
          var detail = t;
          try { var j = JSON.parse(t); if (j && j.detail) detail = String(j.detail); } catch (e) { /* text */ }
          throw new Error('HTTP ' + res.status + (detail ? ': ' + detail.slice(0, 200) : ''));
        });
      }
      return res.json();
    }, function (err) {
      if (timer) clearTimeout(timer);
      if (err && err.name === 'AbortError') throw new Error('the request timed out');
      if (!navigator.onLine) throw new Error('the browser is offline');
      throw new Error((err && err.message) || 'network error');
    });
  }

  function normalizeAnalysis(raw) {
    var r = raw && typeof raw === 'object' ? raw : {};
    var findings = Array.isArray(r.findings) ? r.findings.filter(function (f) {
      return f && typeof f === 'object' && typeof f.message === 'string';
    }).map(function (f) {
      return {
        code: typeof f.code === 'string' ? f.code : '',
        severity: SEVERITY_ORDER.hasOwnProperty(f.severity) ? f.severity : 'info',
        message: f.message,
        detail: typeof f.detail === 'string' ? f.detail : ''
      };
    }) : [];
    findings.sort(function (a, b) { return SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]; });
    var ss = r.ai_style_signals && typeof r.ai_style_signals === 'object' ? r.ai_style_signals : null;
    return {
      features: r.features && typeof r.features === 'object' ? r.features : {},
      bands: r.bands && typeof r.bands === 'object' ? r.bands : {},
      findings: findings,
      detectors: r.detectors && typeof r.detectors === 'object' ? r.detectors : {},
      detectorRan: r.detector_ran === true,
      detectorError: typeof r.detector_error === 'string' && r.detector_error ? r.detector_error : null,
      sentences: Array.isArray(r.sentences) ? r.sentences : [],
      signals: ss && ss.signals && typeof ss.signals === 'object' ? ss.signals : null
    };
  }

  /* ── state ───────────────────────────────────────────────────────────── */

  var state = {
    text: '',
    analysis: null,
    located: [],
    reading: null,        /* GPTZero's last reading of the text: {p, label, derived, text} */
    readingError: null,
    view: 'reading',      /* 'reading' | 'run': what the verdict block shows */
    humanize: null,       /* the last rewrite result; shown in the after pane */
    afterLocated: [],     /* the rewrite's sentences with risk, for shading */
    afterToken: 0,
    preHumanize: null,    /* the draft as it was before Use this, for undo */
    yardstick: null,      /* null = the service has not said; 'gptzero' or a local model name */
    requestToken: 0,
    openMeasure: null
  };

  var editor, scroller, statusLine, errorBox;
  var rendering = false;

  /* ── editor plumbing ─────────────────────────────────────────────────── */

  function getText() {
    var t = editor.innerText || '';
    return t.replace(/\r\n?/g, '\n').replace(/\u00a0/g, ' ').replace(/\n{3,}/g, '\n\n');
  }

  function nodeOffset(container, offset) {
    var walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
    var acc = 0, node;
    if (container === editor) {
      var i = 0;
      while ((node = walker.nextNode())) {
        if (i >= offset) break;
        acc += node.nodeValue.length;
        i++;
      }
      return acc;
    }
    while ((node = walker.nextNode())) {
      if (node === container) return acc + offset;
      acc += node.nodeValue.length;
    }
    return null;
  }

  function captureSelection() {
    if (document.activeElement !== editor) return null;
    var sel = window.getSelection();
    if (!sel || !sel.rangeCount) return null;
    var range = sel.getRangeAt(0);
    if (!editor.contains(range.startContainer) || !editor.contains(range.endContainer)) return null;
    var start = nodeOffset(range.startContainer, range.startOffset);
    var end = nodeOffset(range.endContainer, range.endOffset);
    if (start === null) return null;
    return { start: start, end: end === null ? start : end };
  }

  function pointAt(offset) {
    var walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
    var acc = 0, node, last = null;
    while ((node = walker.nextNode())) {
      var len = node.nodeValue.length;
      if (acc + len >= offset) return { node: node, offset: clamp(offset - acc, 0, len) };
      acc += len;
      last = node;
    }
    if (last) return { node: last, offset: last.nodeValue.length };
    return null;
  }

  function restoreSelection(saved) {
    if (!saved) return;
    var a = pointAt(saved.start);
    var b = pointAt(saved.end);
    if (!a) return;
    try {
      var r = document.createRange();
      r.setStart(a.node, a.offset);
      if (b) r.setEnd(b.node, b.offset); else r.collapse(true);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
    } catch (e) { /* best effort */ }
  }

  function riskTier(risk) {
    var r = num(risk);
    if (r === null) return '';
    return 'risk-' + Math.min(5, Math.floor(r * 6));
  }

  function hasRisk() {
    return state.located.some(function (s) { return num(s.risk) !== null; });
  }

  /* paragraphs with each located sentence wrapped and shaded by risk; used
     for the draft on the left and the rewrite on the right */
  function paintParagraphs(host, text, located) {
    clear(host);
    paragraphRanges(text).forEach(function (p) {
      var para = el('p');
      var pos = p.from;
      located.forEach(function (s) {
        if (s.from < p.from || s.from >= p.to) return;
        if (s.from > pos) para.appendChild(document.createTextNode(text.slice(pos, s.from)));
        var end = Math.min(s.to, p.to);
        var tier = riskTier(s.risk);
        var span = el('span', 'sent' + (tier ? ' ' + tier : ''), text.slice(s.from, end));
        span.setAttribute('data-index', String(s.index));
        if (tier) span.title = 'Estimated sentence risk ' + fmt(s.risk);
        para.appendChild(span);
        pos = end;
      });
      if (pos < p.to) para.appendChild(document.createTextNode(text.slice(pos, p.to)));
      if (!para.childNodes.length) para.appendChild(el('br'));
      host.appendChild(para);
    });
    if (!host.childNodes.length) host.appendChild(el('p', null, ''));
  }

  /* the after pane: the rewrite, read only, shaded once its sentences are scored */
  function renderAfter() {
    var host = $('after');
    var h = state.humanize;
    clear(host);
    $('after-empty').hidden = !!h;
    if (h) paintParagraphs(host, h.text, state.afterLocated);
    var afterRisk = state.afterLocated.some(function (s) { return num(s.risk) !== null; });
    if (afterRisk) $('legend').hidden = false;
    refreshAfterActions();
    emit('humanizer:canvas');
  }

  function rewriteInUse() {
    var h = state.humanize;
    return !!h && state.text.trim() === h.text.trim();
  }

  function refreshAfterActions() {
    var h = state.humanize;
    $('copy-btn').disabled = !h;
    $('use-btn').disabled = !h || h.unchanged || rewriteInUse() || !!activeRun;
  }

  function renderCanvas() {
    var text = state.text;
    var located = state.located;
    var saved = captureSelection();
    var scroll = scroller.scrollTop;
    var wasFocused = document.activeElement === editor;

    rendering = true;
    paintParagraphs(editor, text, located);
    editor.setAttribute('data-empty', text.trim() ? '0' : '1');

    scroller.scrollTop = scroll;
    if (wasFocused) restoreSelection(saved);
    setTimeout(function () { rendering = false; }, 0);

    $('empty-state').hidden = !!text.trim();
    $('legend').hidden = !hasRisk();
    emit('humanizer:canvas');
  }

  function setText(text) {
    state.text = text;
    state.located = [];
    renderCanvas();
    renderCounts();
    refreshHumanizeButton();
  }

  function renderCounts() {
    var node = $('doc-counts');
    if (!state.text.trim()) { node.textContent = ''; return; }
    var w = words(state.text).length;
    var s = state.located.length;
    var t = w + ' ' + plural(w, 'word');
    if (s) t += ', ' + s + ' ' + plural(s, 'sentence');
    node.textContent = t;
  }

  function jumpToSentence(index) {
    Array.prototype.forEach.call(editor.querySelectorAll('.sent'), function (node) {
      if (Number(node.getAttribute('data-index')) !== index) return;
      if (node.scrollIntoView) {
        node.scrollIntoView({ block: 'center', behavior: reduceMotion.matches ? 'auto' : 'smooth' });
      }
      node.classList.remove('flash');
      void node.offsetWidth;
      node.classList.add('flash');
      setTimeout(function () { node.classList.remove('flash'); }, 1200);
    });
  }

  /* ── GPTZero's reading ───────────────────────────────────────────────── */

  function labelOf(raw, p) {
    var s = typeof raw === 'string' ? raw.trim().toLowerCase() : '';
    if (s === 'ai' || s === 'ai-generated' || s === 'ai_generated' || s === 'machine') return { label: 'ai', derived: false };
    if (s === 'human' || s === 'human-written' || s === 'human_written') return { label: 'human', derived: false };
    if (s === 'mixed') return { label: 'mixed', derived: false };
    var n = num(p);
    if (n === null) return { label: null, derived: false };
    return { label: n >= 0.5 ? 'ai' : 'human', derived: true };
  }

  /* The judge's row: the free surrogate by default, GPTZero when a bench
     configures it. Whichever the service ran is the only row it sends. */
  function readGptzero(a) {
    var rows = a && a.detectors && typeof a.detectors === 'object' ? a.detectors : {};
    var key = rows.gptzero ? 'gptzero' : Object.keys(rows)[0];
    var d = key ? rows[key] : null;
    if (!d || typeof d !== 'object' || d.error) return null;
    var p = loose(d.ai_probability);
    var lab = labelOf(d.label, p);
    if (p === null && !lab.label) return null;
    return { p: p, label: lab.label, derived: lab.derived };
  }

  function tipFor(judge, what) {
    return 'The chance ' + what + ' is AI written, as ' + judge + ' reads it. From 50% up it reads as AI.';
  }

  function setWord(id, label, placeholder) {
    var node = $(id);
    node.textContent = label || placeholder || 'unscored';
    node.setAttribute('data-label', label || 'none');
  }

  function renderVerdict() {
    paintVerdict();
    emit('humanizer:verdict');
  }

  function paintVerdict() {
    var panel = $('verdict');
    var line = $('verdict-line');
    var strip = $('verdict-strip');
    var kept = $('verdict-kept');
    var refused = $('verdict-refused');
    var beforeCell = $('v-before');
    var afterCell = $('v-after');
    var h = state.humanize;
    var text = state.text.trim();

    kept.hidden = true;
    refused.hidden = true;
    clear(refused);
    beforeCell.removeAttribute('data-stale');
    afterCell.removeAttribute('data-stale');

    /* what each number is, and who judged it */
    var judgeNow = h && state.view === 'run' ? h.judge : judgeName();
    var hasBefore = !!(state.reading || (h && state.view === 'run' && h.before.label));
    var assumedBefore = !!(h && state.view === 'run' && h.before.assumed);
    beforeCell.setAttribute('data-tip', assumedBefore
      ? 'Your draft is treated as AI written and every paragraph is rewritten. Add your own GPTZero key for a measured reading.'
      : hasBefore
        ? tipFor(judgeNow, 'your draft')
        : 'No reading yet. Measure scores the draft with ' + judgeNow + '.');
    afterCell.setAttribute('data-tip', h ? tipFor(h.judge, 'the rewrite') : '');

    /* the before pane: the run's own reading of the draft while the run is
       current, otherwise the live reading */
    var r = state.reading;
    if (h && state.view === 'run') {
      $('verdict-h').textContent = h.judge;
      setWord('v-before-word', h.before.label);
      $('v-before-p').textContent = h.before.p === null ? '' : pct(h.before.p);
      if (text !== h.source.trim() && !rewriteInUse()) beforeCell.setAttribute('data-stale', '1');
    } else {
      $('verdict-h').textContent = judgeName();
      if (r) {
        setWord('v-before-word', r.label);
        $('v-before-p').textContent = r.p === null ? '' : pct(r.p);
        if (text !== r.text.trim()) beforeCell.setAttribute('data-stale', '1');
      } else {
        setWord('v-before-word', null, state.readingError ? 'unscored' : 'not measured');
        $('v-before-p').textContent = '';
      }
    }

    /* the after pane: quiet until there is a rewrite */
    if (!h) {
      afterCell.hidden = true;
      strip.hidden = true;
      line.textContent = '';
      panel.setAttribute('data-outcome', 'none');
      return;
    }
    afterCell.hidden = false;
    setWord('v-after-word', h.after.label);
    $('v-after-p').textContent = h.after.p === null ? '' : pct(h.after.p);
    strip.hidden = false;

    var t;
    if (!h.before.label && !h.after.label) {
      panel.setAttribute('data-outcome', 'none');
      t = 'No verdict came back on either side, so there is nothing to compare.';
    } else if (h.flipped) {
      panel.setAttribute('data-outcome', 'flipped');
      t = 'Read as AI. Now reads as ' + h.after.label + '.';
    } else if (h.before.label !== 'ai') {
      panel.setAttribute('data-outcome', 'none');
      t = 'Nothing to flip. ' + h.judge + ' already read your draft as ' + h.before.label + '.';
    } else if (h.after.label === 'mixed') {
      panel.setAttribute('data-outcome', 'mixed');
      t = 'Read as AI. Now reads as mixed: closer, not there.';
    } else {
      panel.setAttribute('data-outcome', 'held');
      t = 'Still reads as ' + (h.after.label || 'unscored') + '. The probability moved; the verdict did not.';
    }
    if (h.unchanged) {
      kept.hidden = false;
      kept.textContent = 'Your draft was kept. ' + (h.reason || 'No rewrite passed the meaning gates.');
    }
    if (h.invented.length) {
      refused.hidden = false;
      refused.appendChild(el('p', 'refused-h', 'Refused because they are not in your draft:'));
      var ul = el('ul', 'refused-list');
      h.invented.forEach(function (x) { ul.appendChild(el('li', null, x)); });
      refused.appendChild(ul);
      refused.appendChild(el('p', 'refused-hint',
        'If any of these are true, add them under Facts you can vouch for and run again.'));
    }
    if (rewriteInUse()) {
      t += ' In use as your draft.';
    } else if (text !== h.source.trim()) {
      afterCell.setAttribute('data-stale', '1');
      t += ' Your draft has changed since this run.';
    }
    line.textContent = t;
  }

  /* ── Details: measurements, findings, signals, flagged ───────────────── */

  function renderMeasurements() {
    var host = $('measurements');
    clear(host);
    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }

    FEATURES.forEach(function (spec) {
      var value = a.features[spec.key];
      var stateName = bandState(spec, value, a.bands);
      var plainId = 'plain-' + spec.key;
      var open = state.openMeasure === spec.key;

      var row = el('button', 'measure');
      row.type = 'button';
      row.setAttribute('aria-expanded', open ? 'true' : 'false');
      row.setAttribute('aria-controls', plainId);

      var top = el('div', 'measure-top');
      top.appendChild(el('span', 'measure-name', spec.label));
      var right = el('span', 'measure-right');
      right.appendChild(el('span', 'measure-value num', formatFeature(spec, value)));
      if (spec.band) {
        var chip = el('span', 'state', stateName);
        chip.setAttribute('data-state', stateName);
        right.appendChild(chip);
      }
      top.appendChild(right);
      row.appendChild(top);

      var band = formatBand(spec);
      if (band) row.appendChild(el('div', 'measure-band', 'human band ' + band));

      var plain = el('div', 'measure-plain');
      plain.id = plainId;
      plain.hidden = !open;
      plain.appendChild(el('p', null, spec.plain));
      row.appendChild(plain);

      row.addEventListener('click', function () {
        var nowOpen = state.openMeasure !== spec.key;
        state.openMeasure = nowOpen ? spec.key : null;
        renderMeasurements();
        if (nowOpen) {
          var again = host.querySelector('[aria-controls="' + plainId + '"]');
          if (again) again.focus();
        }
      });

      host.appendChild(row);
    });
  }

  function renderFindings() {
    var host = $('findings');
    clear(host);
    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }
    if (!a.findings.length) {
      host.appendChild(el('p', 'muted', 'Nothing to fix. Every measured feature sits inside its human band.'));
      return;
    }
    a.findings.forEach(function (f) {
      var card = el('article', 'finding');
      card.setAttribute('data-severity', f.severity);
      var top = el('div', 'finding-top');
      top.appendChild(el('span', 'sev', f.severity));
      if (f.code) top.appendChild(el('span', 'finding-code', f.code));
      card.appendChild(top);
      card.appendChild(el('p', 'finding-msg', f.message));
      if (f.detail) card.appendChild(el('p', 'finding-detail', f.detail));
      host.appendChild(card);
    });
  }

  var SIGNAL_LABELS = {
    ai_vocab_weighted: 'AI vocabulary',
    cv_band_distance: 'Uniform sentence lengths',
    formal_connective: 'Formal connectives',
    participial_tail: 'Participial tail clauses',
    tricolon: 'Three part lists',
    negative_parallel: '"Not X, but Y"',
    para_opener_formal: 'Paragraphs opening on a connective',
    uniform_paragraphs: 'Paragraphs all the same length',
    no_contractions: 'No contractions anywhere'
  };
  var SIGNAL_FLOOR = 0.25;

  function signalLabel(key) {
    return SIGNAL_LABELS[key] ||
      String(key).replace(/_+/g, ' ').replace(/^./, function (c) { return c.toUpperCase(); });
  }

  function renderStyleSignals() {
    var host = $('style-signals');
    clear(host);
    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }
    if (!a.signals) { host.appendChild(el('p', 'muted', 'The service returned no style signals.')); return; }

    var rows = [];
    Object.keys(a.signals).forEach(function (k) {
      var v = num(a.signals[k]);
      if (v !== null) rows.push({ key: k, v: clamp(v, 0, 1) });
    });
    rows.sort(function (x, y) { return y.v - x.v; });
    var present = rows.filter(function (r) { return r.v >= SIGNAL_FLOOR; });

    if (!rows.length) { host.appendChild(el('p', 'muted', 'No style signals came back for this draft.')); return; }
    if (!present.length) {
      host.appendChild(el('p', 'muted',
        'None of the ' + rows.length + ' known tells is strongly present. The strongest, ' +
        signalLabel(rows[0].key).toLowerCase() + ', reads ' + fmt(rows[0].v) + ' of 1.00.'));
      return;
    }
    present.forEach(function (r) {
      var row = el('div', 'signal');
      var head = el('div', 'signal-head');
      head.appendChild(el('span', 'signal-name', signalLabel(r.key)));
      head.appendChild(el('span', 'signal-v num', fmt(r.v)));
      row.appendChild(head);
      var bar = el('div', 'signal-bar');
      var fill = el('i');
      fill.style.width = Math.round(r.v * 100) + '%';
      bar.appendChild(fill);
      bar.setAttribute('role', 'img');
      bar.setAttribute('aria-label', signalLabel(r.key) + ', ' + fmt(r.v) + ' of 1.00');
      row.appendChild(bar);
      host.appendChild(row);
    });
  }

  function renderFlagged() {
    var host = $('flagged');
    clear(host);
    if (!state.located.length || !hasRisk()) {
      host.appendChild(el('p', 'muted', state.readingError
        ? 'No sentence scores. ' + judgeName() + ' could not run: ' + state.readingError
        : 'No reading yet. Add a few more sentences, or press Measure.'));
      return;
    }
    var ranked = state.located.filter(function (s) { return num(s.risk) !== null; })
      .sort(function (a, b) { return b.risk - a.risk; });
    var top = ranked.filter(function (s) { return s.risk >= 0.5; }).slice(0, 5);
    if (!top.length) {
      host.appendChild(el('p', 'muted',
        'No sentence reads as AI. The highest is ' + fmt(ranked[0].risk) + ' of 1.00.'));
      return;
    }
    top.forEach(function (s) {
      var btn = el('button', 'flag-btn');
      btn.type = 'button';
      btn.title = 'Jump to this sentence';
      btn.appendChild(el('span', 'flag-risk num', fmt(s.risk)));
      btn.appendChild(el('span', 'flag-text', s.text));
      btn.addEventListener('click', function () { jumpToSentence(s.index); });
      host.appendChild(btn);
    });
    if (ranked.length > top.length) {
      host.appendChild(el('p', 'muted', 'The other ' + (ranked.length - top.length) + ' scored below 0.50.'));
    }
  }

  /* ── Details: the run and its edits ──────────────────────────────────── */

  function addRow(host, label, value) {
    if (value === null || value === undefined || value === '') return;
    var r = el('div', 'row');
    r.appendChild(el('span', 'row-k', label));
    r.appendChild(el('span', 'row-v', String(value)));
    host.appendChild(r);
  }

  /* one sentence under the buttons about the last run; Details has the rest */
  function renderChanged(h) {
    var strip = $('changed');
    if (!strip) return;
    if (!h) { strip.hidden = true; strip.textContent = ''; return; }
    var sm = h.summary || {};
    var bits = [];
    var pr = loose(sm.paragraphs_rewritten), pu = loose(sm.paragraphs_unchanged);
    if (h.unchanged) bits.push('Your draft was kept as it was');
    else if (pr !== null) bits.push('Rewrote ' + pr + ' ' + plural(pr, 'paragraph') + (pu ? ' and kept ' + pu : ''));
    else bits.push('Rewrote your draft');
    if (h.edits.length) bits.push(h.edits.length + ' ' + plural(h.edits.length, 'edit'));
    var total = loose(sm.n_candidates_total), passed = loose(sm.n_candidates_passed);
    if (total !== null && passed !== null) bits.push(passed + ' of ' + total + ' candidates passed the meaning gates');
    else if (total !== null) bits.push(total + ' ' + plural(total, 'candidate') + ' written');
    var secs = loose(sm.seconds) !== null ? loose(sm.seconds) : h.elapsed;
    if (secs !== null && secs !== undefined) bits.push(secs.toFixed(0) + 's');
    strip.textContent = bits.join(', ');
    strip.title = 'Every edit is listed under Details.';
    strip.hidden = false;
  }

  /* the repair stage's log: one line per attempt, diagnosis, action, result */
  function renderRepairs(h) {
    var section = $('repairs-section');
    var host = $('repairs');
    if (!section || !host) return;
    clear(host);
    var sm = h && h.summary && typeof h.summary === 'object' ? h.summary : null;
    var log = sm && Array.isArray(sm.repair_log) ? sm.repair_log : [];
    if (!log.length) { section.hidden = true; return; }
    section.hidden = false;
    log.forEach(function (rec) {
      if (!rec || typeof rec !== 'object') return;
      var para = loose(rec.paragraph);
      var attempt = loose(rec.attempt);
      var label = (para === null ? 'paragraph' : 'paragraph ' + (para + 1)) +
        (attempt === null ? '' : ', try ' + attempt);
      var res = rec.result && typeof rec.result === 'object' ? rec.result : {};
      var passed = loose(res.passed);
      var best = loose(res.best);
      var outcome = [
        passed === null ? null : passed + ' passed the gates',
        best === null ? null : 'best ' + best.toFixed(3),
        res.accepted === true ? 'accepted' : (res.accepted === false ? 'not accepted' : null)
      ].filter(Boolean).join(', ');
      var parts = [];
      if (typeof rec.diagnosis === 'string' && rec.diagnosis) parts.push(rec.diagnosis + '.');
      var action = pickString(rec.action_label, rec.action);
      if (action) parts.push('Tried: ' + action + '.');
      if (typeof rec.critic === 'string' && rec.critic) parts.push('Critic: ' + rec.critic);
      if (typeof rec.note === 'string' && rec.note) parts.push('(' + rec.note + ')');
      if (typeof rec.error === 'string' && rec.error) parts.push('Error: ' + rec.error);
      if (outcome) parts.push('Result: ' + outcome + '.');
      addRow(host, label, parts.join(' '));
    });
    var tried = loose(sm.repairs_attempted), acc = loose(sm.repairs_accepted), resc = loose(sm.repairs_rescued);
    if (tried !== null) {
      addRow(host, 'in all', [
        tried + ' ' + plural(tried, 'attempt'),
        acc === null ? null : acc + ' accepted',
        resc === null ? null : resc + ' ' + plural(resc, 'paragraph') + ' rescued'
      ].filter(Boolean).join(', '));
    }
  }

  function renderRun() {
    var section = $('run-section');
    var host = $('run-summary');
    var h = state.humanize;
    clear(host);
    renderChanged(h);
    renderFlow();
    renderRepairs(h);
    if (!h) { section.hidden = true; return; }
    section.hidden = false;
    var sm = h.summary || {};
    var yard = sm.yardstick && typeof sm.yardstick === 'object' ? sm.yardstick : null;

    addRow(host, 'judge', yard
      ? (yard.kind === 'gptzero' ? 'GPTZero' : pickString(yard.name, yard.kind)) +
        (pickString(yard.model) ? ', ' + yard.model : '')
      : h.judge);
    addRow(host, 'rewriter', h.engine === 'llm' ? (h.model || 'model not reported') : 'rule based engine');
    addRow(host, 'setting', h.level);
    addRow(host, 'facts supplied', typeof sm.facts_supplied === 'boolean' ? (sm.facts_supplied ? 'yes' : 'no') : null);
    var total = loose(sm.n_candidates_total), passed = loose(sm.n_candidates_passed), rej = loose(sm.n_candidates_rejected);
    if (total !== null || passed !== null || rej !== null) {
      addRow(host, 'candidates', [
        total === null ? null : total + ' written',
        passed === null ? null : passed + ' passed',
        rej === null ? null : rej + ' rejected'
      ].filter(Boolean).join(', '));
    }
    if (sm.gate_rejections && typeof sm.gate_rejections === 'object') {
      var gates = Object.keys(sm.gate_rejections).filter(function (k) { return loose(sm.gate_rejections[k]) > 0; })
        .map(function (k) { return k + ' ' + sm.gate_rejections[k]; });
      if (gates.length) addRow(host, 'gate rejections', gates.join(', '));
    }
    var pr = loose(sm.paragraphs_rewritten), pu = loose(sm.paragraphs_unchanged);
    if (pr !== null || pu !== null) {
      addRow(host, 'paragraphs', [
        pr === null ? null : pr + ' rewritten',
        pu === null ? null : pu + ' kept'
      ].filter(Boolean).join(', '));
    }
    addRow(host, 'edits', h.edits.length);
    var secs = loose(sm.seconds) !== null ? loose(sm.seconds) : h.elapsed;
    if (secs !== null && secs !== undefined) addRow(host, 'time', secs.toFixed(1) + 's');
    var tps = loose(sm.tokens_per_second);
    if (tps !== null) addRow(host, 'generation', tps.toFixed(0) + ' tokens per second');
    addRow(host, 'progress', h.streamed ? 'streamed live' : 'estimated');
    if (typeof sm.backend_error === 'string' && sm.backend_error) addRow(host, 'backend error', sm.backend_error);
  }

  function renderEdits() {
    var section = $('edits-section');
    var host = $('edits');
    var h = state.humanize;
    clear(host);
    if (!h || !h.edits.length) { section.hidden = true; return; }
    section.hidden = false;

    h.edits.forEach(function (e) {
      if (!e || typeof e !== 'object') return;
      var card = el('article', 'edit');
      var si = loose(e.sentence_index);
      card.appendChild(el('p', 'edit-kind',
        (typeof e.kind === 'string' && e.kind ? e.kind : 'edit') + (si === null ? '' : ', sentence ' + (si + 1))));
      var diff = el('p', 'edit-diff');
      if (typeof e.before === 'string' && e.before) diff.appendChild(el('span', 'edit-del', e.before));
      if (e.before && e.after) diff.appendChild(el('span', 'edit-arrow', ' → '));
      if (typeof e.after === 'string' && e.after) diff.appendChild(el('span', 'edit-ins', e.after));
      card.appendChild(diff);
      if (typeof e.rationale === 'string' && e.rationale) card.appendChild(el('p', 'edit-why', e.rationale));
      host.appendChild(card);
    });
  }

  /* ── the stage checklist ─────────────────────────────────────────────
     A 30 second rewrite must never look stuck. Each stage is a ring driven
     by the streamed progress value. State is carried by geometry, a glyph
     and words, so colour is never the only signal. */

  var RING_R = 15;
  var RING_C = 2 * Math.PI * RING_R;

  /* weights drive the estimated timeline only, used when the service does
     not stream */
  var STAGES = [
    { key: 'queue',    label: 'Waiting for another rewrite to finish', weight: 0.5 },
    { key: 'analyze',  label: 'Reading your draft',           weight: 1.5 },
    { key: 'plan',     label: 'Planning the rewrite',         weight: 3.0 },
    { key: 'generate', label: 'Writing candidate versions',   weight: 10.0 },
    { key: 'scrub',    label: 'Removing AI vocabulary',       weight: 2.5 },
    { key: 'score',    label: 'Scoring each version',         weight: 5.0 },
    { key: 'verify',   label: 'Checking your facts survived', weight: 3.5 },
    { key: 'select',   label: 'Choosing the best one',        weight: 1.5 },
    { key: 'finalize', label: 'Polishing the final draft',    weight: 3.0 }
  ];
  var RULE_STAGE = [{ key: 'rewrite', label: 'Rewriting with the rule based engine', weight: 1.0 }];

  function stageSpec(key) {
    for (var i = 0; i < STAGES.length; i++) if (STAGES[i].key === key) return STAGES[i];
    return null;
  }

  function stageLabel(key) {
    var spec = stageSpec(key);
    if (spec) return spec.label;
    return String(key).replace(/[_\-]+/g, ' ').replace(/^./, function (c) { return c.toUpperCase(); });
  }

  function svgEl(name, attrs) {
    var n = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) n.setAttribute(k, String(attrs[k]));
    }
    return n;
  }

  var prog = {
    open: false, rows: {}, order: [], weights: {},
    mode: 'unknown',      /* unknown | estimated | live | rule */
    started: 0, ticker: null, estimator: null, armTimer: null,
    said: '', warned: false, wasEditable: 'true'
  };

  function buildStageRow(key, label) {
    var li = el('li', 'stage');
    li.setAttribute('data-stage', key);
    li.setAttribute('data-status', 'pending');

    var ringWrap = el('span', 'stage-ring');
    var svg = svgEl('svg', { viewBox: '0 0 40 40', focusable: 'false' });
    svg.setAttribute('aria-hidden', 'true');
    svg.appendChild(svgEl('circle', { 'class': 'ring-track', cx: 20, cy: 20, r: RING_R }));
    var fill = svgEl('circle', { 'class': 'ring-fill', cx: 20, cy: 20, r: RING_R, 'stroke-dasharray': RING_C.toFixed(3) });
    fill.style.strokeDashoffset = RING_C.toFixed(3);
    svg.appendChild(fill);
    svg.appendChild(svgEl('path', { 'class': 'ring-check', d: 'M13.4 20.4 L18.1 25.1 L26.9 15.4' }));
    ringWrap.appendChild(svg);

    var body = el('span', 'stage-body');
    body.appendChild(el('span', 'stage-label', label));
    var detail = el('span', 'stage-detail', '');
    body.appendChild(detail);
    var pctNode = el('span', 'stage-pct num', 'waiting');

    li.appendChild(ringWrap);
    li.appendChild(body);
    li.appendChild(pctNode);
    return { li: li, fill: fill, detail: detail, pct: pctNode, value: 0, status: 'pending', determinate: false };
  }

  /* the glide between values is a CSS transition on stroke-dashoffset */
  function setRing(row, value) {
    var v = clamp(num(value) === null ? row.value : value, 0, 1);
    row.value = v;
    row.fill.style.strokeDashoffset = (RING_C * (1 - v)).toFixed(3);
  }

  function progSetStages(list) {
    var host = $('stages');
    clear(host);
    prog.rows = {}; prog.order = []; prog.weights = {};
    list.forEach(function (s) {
      var row = buildStageRow(s.key, s.label);
      prog.rows[s.key] = row;
      prog.order.push(s.key);
      prog.weights[s.key] = s.weight;
      host.appendChild(row.li);
    });
  }

  function progSetNote(kind, text) {
    var n = $('progress-note');
    n.textContent = text;
    if (kind) n.setAttribute('data-kind', kind); else n.removeAttribute('data-kind');
  }

  function progSetEngine(text, tag) {
    var host = $('progress-engine');
    clear(host);
    host.appendChild(document.createTextNode(text));
    if (tag) {
      var chip = el('span', 'progress-tag', tag.label);
      chip.setAttribute('data-kind', tag.kind);
      host.appendChild(document.createTextNode(' '));
      host.appendChild(chip);
    }
  }

  /* one number for the whole run, for the ASCII tile in the checklist */
  function progFraction() {
    if (!prog.order.length) return 0;
    var sum = 0;
    prog.order.forEach(function (k) {
      var row = prog.rows[k];
      if (!row) return;
      if (row.status === 'done' || row.status === 'skipped') sum += 1;
      else if (row.status === 'active') sum += clamp(row.value, 0, 0.98);
    });
    return sum / prog.order.length;
  }
  function progEmit(fraction) {
    emit('humanizer:progress', { fraction: fraction === undefined ? progFraction() : fraction, open: prog.open });
  }

  function progSay(text) {
    if (text === prog.said) return;
    prog.said = text;
    $('progress-say').textContent = text;
  }

  function progOpen() {
    prog.open = true;
    prog.mode = 'unknown';
    prog.started = Date.now();
    prog.said = '';
    prog.warned = false;
    prog.eta = null;

    progSetStages(stageList());
    $('progress-title').textContent = 'Rewriting your draft';
    progSetEngine('Contacting the service', null);
    progSetNote(null, 'Your draft is not touched until this finishes.');
    $('progress-elapsed').textContent = '0.0s';
    $('progress').hidden = false;

    prog.wasEditable = editor.getAttribute('contenteditable') || 'true';
    editor.setAttribute('contenteditable', 'false');

    if (prog.ticker) clearInterval(prog.ticker);
    prog.ticker = setInterval(progTick, 100);
    progTick();

    if (prog.armTimer) clearTimeout(prog.armTimer);
    prog.armTimer = setTimeout(function () {
      prog.armTimer = null;
      if (prog.mode === 'unknown') progStartEstimate('estimated');
    }, 900);

    emit('humanizer:progress-open');
    progEmit(0);
    try { $('progress-cancel').focus(); } catch (e) { /* detached */ }
  }

  function progTick() {
    if (!prog.open) return;
    var t = (Date.now() - prog.started) / 1000;
    $('progress-elapsed').textContent = t.toFixed(1) + 's';
    var slowAfter = prog.eta ? Infinity : 45;
    if (!prog.warned && t > slowAfter && prog.mode !== 'rule') {
      prog.warned = true;
      progSetNote('warn', 'Still running. Long drafts take a few minutes on this machine. You can stop at any time and keep your draft.');
    }
  }

  function progStopEstimate() {
    if (prog.estimator) { clearInterval(prog.estimator); prog.estimator = null; }
    if (prog.armTimer) { clearTimeout(prog.armTimer); prog.armTimer = null; }
  }

  function progStartEstimate(mode) {
    progStopEstimate();
    prog.mode = mode;
    if (mode === 'estimated') {
      progSetEngine('Rewrite pipeline', { kind: 'est', label: 'estimated' });
      progSetNote(null, 'The service is not reporting progress, so these steps run on an estimated timeline. The result is real.');
    }
    var slow = reduceMotion.matches;
    prog.estimator = setInterval(function () {
      if (!prog.open || prog.mode === 'live') { progStopEstimate(); return; }
      var t = (Date.now() - prog.started) / 1000;
      var acc = 0;
      for (var i = 0; i < prog.order.length; i++) {
        var k = prog.order[i];
        var w = prog.weights[k] || 1;
        var local = (t - acc) / w;
        acc += w;
        var last = i === prog.order.length - 1;
        if (local <= 0) continue;
        if (local >= 1 && !last) {
          progStage(k, { status: 'done', progress: 1 });
        } else {
          prog.rows[k].determinate = true;
          progStage(k, { status: 'active', progress: Math.min(local, last ? 0.95 : 0.99) });
        }
      }
    }, slow ? 900 : 110);
  }

  function progAdoptLive() {
    if (prog.mode === 'live') return;
    progStopEstimate();
    prog.mode = 'live';
    progSetEngine('Rewrite pipeline', { kind: 'live', label: 'measured' });
    progSetNote(null, 'Each step is reported by the service as it happens. Your draft is not touched until this finishes.');
  }

  function progSwitchToRule(why) {
    progStopEstimate();
    prog.mode = 'rule';
    progSetStages(RULE_STAGE);
    progSetEngine('Rule based engine', { kind: 'est', label: 'fallback' });
    progSetNote(null, (why || 'The rewrite pipeline is not available.') + ' The rule based engine is running instead.');
    progStartEstimate('rule');
  }

  function normalizeStatus(raw) {
    var s = String(raw === undefined || raw === null ? '' : raw).toLowerCase().trim();
    if (!s) return null;
    if (s === 'active' || s === 'running' || s === 'started' || s === 'start' ||
        s === 'progress' || s === 'in_progress' || s === 'in-progress' || s === 'working') return 'active';
    if (s === 'done' || s === 'complete' || s === 'completed' || s === 'finished' || s === 'ok' || s === 'end') return 'done';
    if (s === 'skip' || s === 'skipped') return 'skipped';
    if (s === 'error' || s === 'failed' || s === 'failure') return 'error';
    if (s === 'pending' || s === 'queued' || s === 'waiting') return 'pending';
    return null;
  }

  function normalizeProgress(raw) {
    var p = loose(raw);
    if (p === null) return null;
    if (p > 1 && p <= 100) p = p / 100;
    return clamp(p, 0, 1);
  }

  function progStage(key, ev) {
    if (!prog.open) return;
    var row = prog.rows[key];
    if (!row) {
      row = buildStageRow(key, stageLabel(key));
      prog.rows[key] = row;
      var spec = stageSpec(key);
      prog.weights[key] = spec ? spec.weight : 2;
      /* the queue stage arrives before analyze when the model is busy; it
         belongs above the pipeline's own rows */
      if (key === 'queue') {
        prog.order.unshift(key);
        $('stages').insertBefore(row.li, $('stages').firstChild);
      } else {
        prog.order.push(key);
        $('stages').appendChild(row.li);
      }
    }

    var status = normalizeStatus(ev && ev.status);
    var p = normalizeProgress(ev && ev.progress);
    if (status === 'done') p = 1;
    if (p === null) p = row.value;
    if (status === null) status = p >= 1 ? 'done' : (p > 0 ? 'active' : row.status);
    if (status === row.status && p < row.value && row.status !== 'pending') p = row.value;

    row.status = status;
    row.li.setAttribute('data-status', status);

    /* some stages report no intra-stage progress: the ring sweeps and reads
       "working" rather than claiming 0% */
    if (normalizeProgress(ev && ev.progress) > 0) row.determinate = true;
    var unknown = status === 'active' && !row.determinate;
    if (unknown) row.li.setAttribute('data-progress', 'unknown');
    else row.li.removeAttribute('data-progress');

    setRing(row, unknown ? 0.28 : p);

    row.pct.textContent =
      status === 'done'    ? 'done' :
      status === 'skipped' ? 'skipped' :
      status === 'error'   ? 'failed' :
      unknown              ? 'working' :
      status === 'active'  ? Math.round(p * 100) + '%' : 'waiting';

    if (ev && typeof ev.detail === 'string') row.detail.textContent = ev.detail;

    if (status === 'active' || status === 'error') {
      var at = prog.order.indexOf(key);
      for (var i = 0; i < at; i++) {
        var earlier = prog.rows[prog.order[i]];
        if (earlier && earlier.status !== 'done' && earlier.status !== 'skipped') {
          earlier.status = 'done';
          earlier.li.setAttribute('data-status', 'done');
          earlier.li.removeAttribute('data-progress');
          earlier.pct.textContent = 'done';
          setRing(earlier, 1);
        }
      }
      var d = row.detail.textContent;
      progSay('Step ' + (at + 1) + ' of ' + prog.order.length + ', ' +
        row.li.querySelector('.stage-label').textContent.toLowerCase() + (d ? ', ' + d : '') + '.');
    }
    progEmit();
  }

  function progApplyReported(stages) {
    if (!Array.isArray(stages)) return;
    stages.forEach(function (s) {
      if (!s || typeof s !== 'object') return;
      var key = String(s.stage || s.name || s.step || '');
      var row = prog.rows[key];
      if (!key || !row) return;
      var bits = [];
      if (typeof s.detail === 'string' && s.detail) bits.push(s.detail);
      var e = loose(s.seconds !== undefined && s.seconds !== null ? s.seconds
        : (s.elapsed !== undefined && s.elapsed !== null ? s.elapsed : s.duration));
      if (e !== null && e >= 0.05) bits.push(e.toFixed(1) + 's');
      if (bits.length) row.detail.textContent = bits.join(', ');
    });
  }

  function progFinish(stages, done) {
    if (!prog.open) { if (done) done(); return; }
    progStopEstimate();
    prog.order.forEach(function (k) {
      var row = prog.rows[k];
      if (!row || row.status === 'skipped' || row.status === 'error') return;
      row.status = 'done';
      row.li.setAttribute('data-status', 'done');
      row.li.removeAttribute('data-progress');
      row.pct.textContent = 'done';
      setRing(row, 1);
    });
    progApplyReported(stages);
    $('progress-title').textContent = 'Done';
    progSay('All steps finished.');
    progEmit(1);
    setTimeout(function () { progClose(); if (done) done(); }, reduceMotion.matches ? 0 : 340);
  }

  function progClose() {
    if (!prog.open) return;
    prog.open = false;
    progStopEstimate();
    if (prog.ticker) { clearInterval(prog.ticker); prog.ticker = null; }
    $('progress').hidden = true;
    $('progress-say').textContent = '';
    if (editor) editor.setAttribute('contenteditable', prog.wasEditable || 'true');
    emit('humanizer:progress-close');
    setTimeout(function () {
      try {
        var btn = $('humanize-btn');
        if (btn && !btn.disabled) btn.focus();
        else if (editor) editor.focus();
      } catch (e) { /* nothing focusable */ }
    }, 0);
  }

  /* ── humanize: the transport ─────────────────────────────────────────
     Three routes, tried in order: the SSE stream so the checklist is driven
     by measured progress, the blocking LLM route on an estimated timeline,
     and the rule based engine as a last resort. A refusal on one route
     falls through to the next; only the last error is ever shown. */

  var AGGRESSIVENESS = { light: 'light', balanced: 'balanced', strong: 'strong' };

  var caps = {
    llmReady: null,     /* true | false | null = never answered */
    llmReason: '',
    model: '',
    stageNames: null,
    maxWords: null
  };

  var activeRun = null;

  /* The judge is the free local estimate of GPTZero unless a bench set
     GPTZero itself. Either way the tool is usable; no key is required. */
  function judgeName() {
    return state.yardstick === 'gptzero' ? 'GPTZero' : 'GPTZero estimate';
  }

  function setHumanizeState(kind, message) {
    var btn = $('humanize-btn');
    var label = $('humanize-label');
    btn.removeAttribute('data-busy');
    if (kind === 'busy') {
      btn.disabled = true;
      btn.setAttribute('data-busy', '1');
      label.textContent = 'Humanizing';
      return;
    }
    label.textContent = 'Humanize';
    if (kind === 'absent' || kind === 'empty') {
      btn.disabled = true;
      btn.title = message || '';
      return;
    }
    btn.disabled = false;
    btn.title = 'Rewrite the draft. Ctrl+Enter or Cmd+Enter.';
  }

  function refreshHumanizeButton() {
    var measure = $('analyze-btn');
    renderCost();
    if (activeRun) { setHumanizeState('busy'); measure.disabled = true; refreshAfterActions(); return; }
    measure.disabled = !state.text.trim();
    renderFlow();
    if (!state.text.trim()) { setHumanizeState('empty', 'Paste or type a draft first.'); refreshAfterActions(); return; }
    setHumanizeState('ready');
    refreshAfterActions();
  }

  /* cheap: touches no weights. Stage names, the word limit, and whether the
     LLM path can run at all. */
  function probeLlmHealth() {
    request('/api/humanize/llm/health').then(function (h) {
      if (!h || typeof h !== 'object') return;
      caps.llmReady = h.available === true;
      caps.llmReason = typeof h.reason === 'string' ? h.reason : '';
      caps.model = pickString(h.model, h.freeform_model);
      caps.maxWords = loose(h.max_words);
      if (Array.isArray(h.stages) && h.stages.length) {
        caps.stageNames = h.stages.filter(function (x) { return typeof x === 'string' && x; });
      }
      /* the older health route does not name the yardstick; this one does */
      if (state.yardstick === null && typeof h.yardstick === 'string' && h.yardstick) applyYardstick(h.yardstick);
    }, function () { /* no route: every field stays null */ });
  }

  function stageList() {
    if (!caps.stageNames || !caps.stageNames.length) return STAGES.filter(function (s) { return s.key !== 'queue'; });
    return caps.stageNames.map(function (key) {
      return stageSpec(key) || { key: key, label: stageLabel(key), weight: 2 };
    });
  }

  function httpFail(res) {
    if (res.status === 401) signInRequired();
    return res.text().catch(function () { return ''; }).then(function (t) {
      var body = null;
      try { body = JSON.parse(t); } catch (e) { body = null; }
      var detail = body && body.detail ? String(body.detail) : t;
      var e2 = new Error('HTTP ' + res.status + (detail ? ': ' + detail.slice(0, 200) : ''));
      e2.status = res.status;
      /* 402: the balance is short. A paywall event, not a failure, and never
         a reason to fall through to the rule engine. The body carries
         needed, balance, words_needed and words_left. */
      if (res.status === 402) {
        e2.paywall = body && typeof body === 'object' ? body : { error: 'insufficient_credits' };
        throw e2;
      }
      /* the LLM routes refuse with a body naming where to go instead */
      if (body && typeof body.fallback_endpoint === 'string') {
        e2.missing = true;
        e2.fallbackNote = typeof body.detail === 'string' ? body.detail : '';
      } else if (res.status === 404 || res.status === 405 || res.status === 501 ||
                 res.status === 503 || res.status === 413 || res.status === 422 || res.status === 400) {
        e2.missing = true;
      }
      throw e2;
    });
  }

  function netFail(err) {
    if (err && (err.name === 'AbortError' || err.missing)) return err;
    if (!navigator.onLine) return new Error('the browser is offline');
    return new Error((err && err.message) || 'network error');
  }

  function looksLikeResult(o) {
    return !!(o && typeof o === 'object' && (typeof o.humanized === 'string' || (o.edits && o.summary)));
  }

  /* an SSE reader on fetch rather than EventSource, so a POST body and an
     AbortController both work */
  function readSse(reader, sink) {
    var dec = typeof TextDecoder !== 'undefined' ? new TextDecoder() : null;
    var buf = '';
    var result = null;
    var seen = [];
    var failure = null;

    function dispatch(frame) {
      if (!frame || !frame.trim()) return;
      var name = 'message';
      var data = [];
      frame.split('\n').forEach(function (line) {
        if (!line || line.charAt(0) === ':') return;
        var c = line.indexOf(':');
        var field = c < 0 ? line : line.slice(0, c);
        var value = c < 0 ? '' : line.slice(c + 1).replace(/^ /, '');
        if (field === 'event') name = value;
        else if (field === 'data') data.push(value);
      });
      handle(name, data.join('\n'));
    }

    function handle(name, dataStr) {
      var d = null;
      if (dataStr) { try { d = JSON.parse(dataStr); } catch (e) { d = null; } }
      var lower = String(name || '').toLowerCase();

      if (lower === 'error' || (d && d.error && !d.stage)) {
        var msg = (d && (d.detail || d.error || d.message)) || dataStr || 'the rewrite failed';
        failure = new Error(String(msg).slice(0, 300));
        if (d && typeof d.fallback_endpoint === 'string') failure.missing = true;
        /* a 402 can arrive as an error frame once the stream is open */
        if (d && (d.error === 'insufficient_credits' || loose(d.status) === 402)) {
          failure.paywall = d;
          failure.missing = false;
        }
        return;
      }
      if (lower === 'result' || lower === 'done' || lower === 'complete' || lower === 'final' ||
          (d && (typeof d.humanized === 'string' || (d.result && typeof d.result === 'object')))) {
        var payload = d && d.result && typeof d.result === 'object' ? d.result : d;
        /* `event: done` carries {"ok":true} right after `event: result`, and
           must not overwrite the rewrite */
        if (looksLikeResult(payload)) result = payload;
        return;
      }
      if (d && typeof d === 'object') {
        var key = d.stage || d.name || d.step;
        if (!key) return;
        seen.push(d);
        if (sink) sink(String(key), d);
      }
    }

    function drain(s) {
      s = s.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
      var parts = s.split('\n\n');
      var rest = parts.pop();
      parts.forEach(dispatch);
      return rest;
    }

    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.value) buf += dec ? dec.decode(chunk.value, { stream: true }) : String(chunk.value);
        buf = drain(buf);
        if (!chunk.done) return pump();
        if (dec) buf += dec.decode();
        drain(buf + '\n\n');
        if (failure) throw failure;
        if (!result) throw new Error('the stream ended before a result arrived');
        return { raw: result, streamed: true, stages: seen };
      });
    }
    return pump();
  }

  function streamAttempt(path, body, signal, sink) {
    var opts = {
      method: 'POST',
      headers: { Accept: 'text/event-stream, application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store'
    };
    if (signal) opts.signal = signal;
    return fetch(API_BASE + path, opts).then(function (res) {
      if (!res.ok) return httpFail(res);
      noteBalance(res);
      var ct = String(res.headers.get('Content-Type') || '').toLowerCase();
      if (ct.indexOf('text/event-stream') < 0) {
        return res.json().then(function (j) { return { raw: j, streamed: false, stages: [] }; });
      }
      if (!res.body || typeof res.body.getReader !== 'function') {
        return res.text().then(function (t) {
          var fake = { read: function () {
            if (fake.spent) return Promise.resolve({ done: true });
            fake.spent = true;
            return Promise.resolve({ done: false, value: new TextEncoder().encode(t) });
          } };
          return readSse(fake, sink);
        });
      }
      return readSse(res.body.getReader(), sink);
    }, function (err) { throw netFail(err); });
  }

  function plainAttempt(path, body, signal) {
    var opts = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store'
    };
    if (signal) opts.signal = signal;
    return fetch(API_BASE + path, opts).then(function (res) {
      if (!res.ok) return httpFail(res);
      noteBalance(res);
      return res.json().then(function (j) { return { raw: j, streamed: false, stages: [] }; });
    }, function (err) { throw netFail(err); });
  }

  function runAttempts(attempts, i, body, signal, sink) {
    if (i >= attempts.length) return Promise.reject(new Error('no rewriting endpoint answered'));
    var a = attempts[i];
    var work = a.stream ? streamAttempt(a.path, body, signal, sink) : plainAttempt(a.path, body, signal);
    return work.then(function (res) {
      res.engine = a.engine;
      return res;
    }, function (err) {
      if (err && err.name === 'AbortError') throw err;
      if (err && err.missing && i + 1 < attempts.length) {
        if (attempts[i + 1].engine === 'rule' && a.engine === 'llm') {
          progSwitchToRule(err.fallbackNote ? 'The service refused the rewrite pipeline: ' + err.fallbackNote : '');
        }
        return runAttempts(attempts, i + 1, body, signal, sink);
      }
      throw err;
    });
  }

  /* absent from the body when empty so `facts_supplied` reads false */
  function factsText() {
    var box = $('facts');
    return box && typeof box.value === 'string' ? box.value.trim() : '';
  }

  /* The user's own GPTZero key, kept in this browser only. With it, GPTZero
     judges and ranks the candidates for their request on their credits. */
  var KEY_STORE = 'humanizer.gptzeroKey';
  var OWN_KEY_STORE = 'humanizer.ownKey';

  /* a switch is a button with role=switch; aria-checked is its state */
  function switchOn(btn) { return !!btn && btn.getAttribute('aria-checked') === 'true'; }
  function setSwitch(btn, on) { if (btn) btn.setAttribute('aria-checked', on ? 'true' : 'false'); }
  function wireSwitch(btn, onChange) {
    if (!btn) return;
    btn.addEventListener('click', function () {
      setSwitch(btn, !switchOn(btn));
      onChange(switchOn(btn));
    });
  }

  function ownKeyOn() { return switchOn($('own-key-switch')); }

  /* the key is only ever read, and only ever sent, while the switch is on */
  function userKey() {
    if (!ownKeyOn()) return '';
    var box = $('gptzero-key');
    var v = box && typeof box.value === 'string' ? box.value.trim() : '';
    return v;
  }
  (function wireKey() {
    var box = $('gptzero-key');
    var sw = $('own-key-switch');
    var field = $('own-key-field');
    if (!box) return;
    var stored = '';
    try { stored = localStorage.getItem(KEY_STORE) || ''; } catch (e) { /* private mode */ }
    box.value = stored;
    var on = false;
    try {
      var saved = localStorage.getItem(OWN_KEY_STORE);
      on = saved === null ? !!stored : saved === '1';
    } catch (e) { on = !!stored; }
    function apply(v, save) {
      setSwitch(sw, v);
      if (field) field.hidden = !v;
      if (save) { try { localStorage.setItem(OWN_KEY_STORE, v ? '1' : '0'); } catch (e) { /* private mode */ } }
    }
    apply(on, false);
    wireSwitch(sw, function (v) {
      apply(v, true);
      if (v && !box.value) { try { box.focus(); } catch (e) { /* detached */ } }
    });
    box.addEventListener('input', function () {
      try {
        if (box.value.trim()) localStorage.setItem(KEY_STORE, box.value.trim());
        else localStorage.removeItem(KEY_STORE);
      } catch (e) { /* private mode */ }
    });
  })();

  function runRewrite(text, level, signal) {
    var body = { text: text, aggressiveness: level };
    var facts = factsText();
    if (facts) body.facts = facts;
    var key = userKey();
    if (key) body.gptzero_api_key = key;

    var sink = function (key, d) {
      progAdoptLive();
      var detail = typeof d.detail === 'string' ? d.detail : '';
      if (key === 'plan' && d.status === 'done') {
        var eta = /about ([^,]+)$/.exec(detail);
        if (eta) {
          prog.eta = eta[1].trim();
          progSetNote(null, 'Expected to take about ' + prog.eta + '. Your draft is not touched until this finishes.');
        }
      }
      if (key === 'queue') {
        progSetNote(d.status === 'done' ? null : 'warn', d.status === 'done'
          ? 'The model is free. Your rewrite is starting.'
          : 'Someone else\'s rewrite is using the model. Yours starts the moment it finishes. You can stop at any time and keep your draft.');
      }
      var round = loose(d.round);
      if (round !== null && round > 1) detail = detail ? detail + ', pass ' + round : 'pass ' + round;
      progStage(key, { status: d.status, progress: d.progress, detail: detail });
    };

    var attempts = [];
    var nWords = words(text).length;
    var over = caps.maxWords !== null && nWords > caps.maxWords;
    if (caps.llmReady === false || over) {
      progSwitchToRule(over
        ? 'This draft is ' + nWords + ' words, past the ' + caps.maxWords + ' word limit of the rewrite pipeline.'
        : (caps.llmReason ? 'The rewrite pipeline cannot run: ' + caps.llmReason : ''));
      attempts.push({ path: RULE_PATH, engine: 'rule', stream: false });
      return runAttempts(attempts, 0, body, signal, sink);
    }
    attempts.push({ path: LLM_STREAM_PATH, engine: 'llm', stream: true });
    attempts.push({ path: LLM_BLOCKING_PATH, engine: 'llm', stream: false });
    attempts.push({ path: RULE_PATH, engine: 'rule', stream: false });
    return runAttempts(attempts, 0, body, signal, sink);
  }

  function humanize() {
    var text = getText();
    if (!text.trim() || activeRun) return;

    /* the balance is known to be short: the button stays live, but the
       click opens the plans instead of a run the service would refuse */
    if (billingShort(text)) {
      setStatus('idle', 'Not enough words left for this draft. Choose a plan to continue.');
      openPlans({ words_needed: words(text).length, words_left: billing.wordsLeft }, $('humanize-btn'));
      return;
    }

    var level = $('aggressiveness').value;
    if (!AGGRESSIVENESS[level]) level = 'balanced';

    var ctl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var run = { ctl: ctl, cancelled: false, timedOut: false, t0: Date.now() };
    activeRun = run;

    run.ceiling = setTimeout(function () {
      run.timedOut = true;
      if (ctl) { try { ctl.abort(); } catch (e) { /* gone */ } }
    }, LLM_CEILING_MS);

    refreshHumanizeButton();
    setStatus('loading', 'Rewriting. Nothing is kept until it finishes.');
    progOpen();

    runRewrite(text, level, ctl ? ctl.signal : null).then(function (res) {
      if (run.cancelled) return;
      clearTimeout(run.ceiling);
      var elapsed = (Date.now() - run.t0) / 1000;
      var reported = Array.isArray(res.raw && res.raw.stages) ? res.raw.stages : res.stages;
      progFinish(reported, function () {
        activeRun = null;
        finishHumanize(res, text, level, elapsed);
      });
    }, function (err) {
      if (run.cancelled) return;
      clearTimeout(run.ceiling);
      activeRun = null;
      progClose();
      refreshHumanizeButton();
      var msg = (err && err.message) || 'unknown error';
      /* the service said the balance is short: the plans, not the error box.
         A guest has no plans to choose from yet: the sign-up page instead. */
      if (err && err.paywall) {
        billingFrom402(err.paywall);
        if (err.paywall.guest === true || billing.guest) { guestOut(err.paywall); return; }
        setStatus('idle', 'Not enough words left for this draft. Choose a plan to continue.');
        openPlans(err.paywall, $('humanize-btn'));
        return;
      }
      if (run.timedOut || (err && err.name === 'AbortError')) {
        setStatus('error', 'The rewrite did not finish within eleven minutes, so it was stopped. Your draft is untouched.');
        return;
      }
      setStatus('error', 'The rewrite did not work: ' + msg + '. Your draft is untouched.');
      showError(msg);
    });
  }

  function cancelHumanize() {
    if (!activeRun) return;
    var run = activeRun;
    run.cancelled = true;
    clearTimeout(run.ceiling);
    activeRun = null;
    if (run.ctl) { try { run.ctl.abort(); } catch (e) { /* gone */ } }
    refreshHumanizeButton();
    progClose();
    setStatus('idle', 'Stopped. Your draft is exactly as you left it.');
  }

  /* ── the result ──────────────────────────────────────────────────────── */

  function side(obj, reportedLabel) {
    var o = obj && typeof obj === 'object' ? obj : {};
    var p = loose(o.ai_probability);
    if (p === null) p = loose(o.probability);
    var lab = labelOf(pickString(reportedLabel, o.label), p);
    return { p: p, label: lab.label, derived: lab.derived, assumed: o.assumed === true };
  }

  function keptReason(r) {
    var paras = Array.isArray(r.paragraphs) ? r.paragraphs : [];
    for (var i = 0; i < paras.length; i++) {
      var p = paras[i];
      if (p && typeof p.fallback_reason === 'string' && p.fallback_reason) return p.fallback_reason;
    }
    var sm = r.summary && typeof r.summary === 'object' ? r.summary : {};
    var rej = loose(sm.n_candidates_rejected);
    if (rej !== null && rej > 0) {
      return rej + ' ' + plural(rej, 'candidate') + ' ' + (rej === 1 ? 'was' : 'were') +
        ' rejected for drifting from what you wrote.';
    }
    return '';
  }

  function finishHumanize(res, text, level, elapsed) {
    var r = res.raw && typeof res.raw === 'object' ? res.raw : {};
    var out = pickString(r.humanized, r.rewritten, r.output, r.text);
    if (!out.trim()) {
      setStatus('error', 'The service returned an empty rewrite, so your draft was left alone.');
      refreshHumanizeButton();
      return;
    }

    var sm = r.summary && typeof r.summary === 'object' ? r.summary : {};
    var unchanged = out.trim() === text.trim();
    var before = side(r.before, sm.label_before);
    var after = side(r.after, sm.label_after);
    var yard = sm.yardstick && typeof sm.yardstick === 'object' ? sm.yardstick : null;
    var flipped = typeof sm.verdict_flipped === 'boolean'
      ? sm.verdict_flipped
      : (before.label === 'ai' && after.label === 'human');
    var invented = Array.isArray(sm.unverified_specifics)
      ? sm.unverified_specifics.filter(function (x) { return typeof x === 'string' && x; })
      : [];

    state.humanize = {
      text: out,
      source: text,
      unchanged: unchanged,
      edits: Array.isArray(r.edits) ? r.edits : [],
      before: before,
      after: after,
      flipped: flipped,
      summary: sm,
      judge: yard && yard.kind === 'gptzero' ? 'GPTZero' : judgeName(),
      reason: unchanged ? keptReason(r) : '',
      invented: invented,
      level: pickString(r.aggressiveness, level),
      engine: res.engine,
      streamed: res.streamed === true,
      model: pickString(r.model, sm.model, caps.model),
      elapsed: loose(r.elapsed) !== null ? loose(r.elapsed) : elapsed
    };
    state.afterLocated = [];
    state.view = 'run';

    /* the draft stays on the left, untouched; the rewrite goes on the right.
       The run's before reading is the judge's reading of this exact draft. */
    if (before.label) {
      state.reading = { p: before.p, label: before.label, derived: before.derived, text: text };
      state.readingError = null;
    }
    state.preHumanize = null;
    $('undo-btn').hidden = true;
    renderVerdict();
    renderRun();
    renderEdits();
    renderAfter();
    refreshHumanizeButton();

    if (unchanged) {
      if (invented.length) {
        var fold = $('facts-fold');
        if (fold) fold.open = true;
        setStatus('idle', 'Kept your draft. The best rewrites added material that is not in it; the list is under the verdict.');
      } else {
        setStatus('idle', 'Kept your draft. ' + (state.humanize.reason || 'No rewrite passed the meaning gates.'));
      }
      return;
    }

    setStatus('ok', flipped
      ? 'Done. ' + state.humanize.judge + ' reads the rewrite as ' + after.label + '. Use this makes it your draft.'
      : 'Done. ' + state.humanize.judge + ' reads the rewrite as ' + (after.label || 'unscored') + '. Use this makes it your draft.');

    /* the one judge call after a run: per sentence risk for the rewrite */
    analyzeAfter(state.humanize);
  }

  /* scores the rewrite's sentences so the after pane can shade them. The
     verdict came with the run; this only adds the shading, and fills in a
     reading if the run did not report one. */
  function analyzeAfter(h) {
    var token = ++state.afterToken;
    request('/api/analyze', { text: h.text, detect: true }, 90000).then(function (raw) {
      if (token !== state.afterToken || state.humanize !== h) return;
      var a = normalizeAnalysis(raw);
      state.afterLocated = locateSentences(h.text, a.sentences);
      if (!h.after.label) {
        var g = readGptzero(a);
        if (g) {
          h.after = { p: g.p, label: g.label, derived: g.derived };
          h.flipped = h.before.label === 'ai' && g.label === 'human';
          renderVerdict();
        }
      }
      renderAfter();
    }, function () { /* shading is optional; the verdict is already shown */ });
  }

  /* Use this: the rewrite becomes the draft. Undo brings the old draft back. */
  function useRewrite() {
    var h = state.humanize;
    if (!h || !h.text.trim() || activeRun || rewriteInUse()) return;
    state.preHumanize = state.text;
    state.view = 'reading';
    state.reading = h.after.label ? { p: h.after.p, label: h.after.label, derived: h.after.derived, text: h.text } : null;
    state.readingError = null;
    setText(h.text);
    state.located = state.afterLocated.slice();
    if (state.located.length) renderCanvas();
    $('undo-btn').hidden = false;
    renderVerdict();
    renderAfter();
    setStatus('ok', 'The rewrite is now your draft. Undo brings the old one back.');
    analyze({ detect: true });
    try { editor.focus(); } catch (e) { /* nothing to focus */ }
  }

  function undoHumanize() {
    if (state.preHumanize === null) return;
    var original = state.preHumanize;
    var h = state.humanize;
    state.preHumanize = null;
    state.view = h ? 'run' : 'reading';
    /* the before reading is the judge's reading of this exact text */
    state.reading = h && h.before.label ? { p: h.before.p, label: h.before.label, derived: h.before.derived, text: original } : null;
    $('undo-btn').hidden = true;
    setText(original);
    renderVerdict();
    renderAfter();
    refreshHumanizeButton();
    setStatus('idle', 'Your original draft is back. The rewrite is still on the right.');
    analyze({ detect: true });
  }

  function copyRewrite() {
    var h = state.humanize;
    if (!h) return;
    var done = function () { setStatus('ok', 'Copied the rewrite.'); };
    var fail = function () { setStatus('error', 'Could not copy. Select the text on the right and copy it yourself.'); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(h.text).then(done, fail);
      return;
    }
    try {
      var ta = document.createElement('textarea');
      ta.value = h.text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed'; ta.style.left = '-200vw';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(ta);
      if (ok) done(); else fail();
    } catch (e) { fail(); }
  }

  function refreshFactsState() {
    var summary = document.querySelector('#facts-fold .facts-summary');
    if (summary) {
      if (factsText()) summary.setAttribute('data-state', 'filled');
      else summary.removeAttribute('data-state');
    }
    renderFlow();
  }

  /* the three steps in the band show where you are */
  function renderFlow() {
    var flow = $('flow');
    if (!flow) return;
    var done = {
      draft: !!(state.text && state.text.trim()),
      facts: !!factsText(),
      run: !!state.humanize
    };
    Array.prototype.forEach.call(flow.querySelectorAll('.step'), function (li) {
      var key = li.getAttribute('data-step');
      if (done[key]) li.setAttribute('data-done', '1');
      else li.removeAttribute('data-done');
    });
  }

  function goTo(where) {
    if (where === 'editor') { editor.focus(); return; }
    if (where === 'facts') {
      var fold = $('facts-fold');
      if (fold) fold.open = true;
      var box = $('facts');
      if (box) { box.focus(); box.scrollIntoView({ block: 'center' }); }
      return;
    }
    if (where === 'humanize') {
      var btn = $('humanize-btn');
      if (btn.disabled) { editor.focus(); setStatus('idle', btn.title || 'Paste or type a draft first.'); return; }
      btn.scrollIntoView({ block: 'center' });
      if (!activeRun) humanize();
    }
  }

  /* ── status, errors, the analyze cycle ───────────────────────────────── */

  function setStatus(kind, message) {
    statusLine.setAttribute('data-state', kind);
    statusLine.textContent = message;
  }

  function hideError() { errorBox.hidden = true; clear(errorBox); }

  function showError(message) {
    clear(errorBox);
    errorBox.hidden = false;
    errorBox.appendChild(el('h2', 'error-title', 'The service did not answer'));
    errorBox.appendChild(el('code', 'error-code', message));
    errorBox.appendChild(el('p', 'error-help', navigator.onLine
      ? 'Start the service with "humanizer serve --port 8000", or point this page elsewhere with ?api=http://host:port. Your draft is untouched.'
      : 'Your browser reports no network connection. Your draft is untouched.'));
    var retry = el('button', 'btn', 'Try again');
    retry.type = 'button';
    retry.addEventListener('click', function () { checkHealth(); analyze({ detect: false }); });
    errorBox.appendChild(retry);
  }

  function setPill(kind, text) {
    var pill = $('health');
    pill.setAttribute('data-kind', kind);
    pill.textContent = text;
    pill.title = text;
  }

  function applyYardstick(y) {
    state.yardstick = typeof y === 'string' && y ? y : null;
    var vh = $('verdict-h');
    if (vh && !(state.humanize && state.view === 'run')) vh.textContent = judgeName();
    var note = $('verdict-note');
    if (note) {
      note.textContent = state.yardstick === 'gptzero'
        ? 'Scored by GPTZero itself.'
        : 'A free local estimate of what GPTZero would say. Check the final draft with GPTZero itself.';
    }
    refreshHumanizeButton();
    setPillHealthy();
  }

  function applyAnalysis(text, a, detect, keepView) {
    state.analysis = a;
    state.located = locateSentences(text, a.sentences);

    if (detect) {
      var g = readGptzero(a);
      state.reading = g ? { p: g.p, label: g.label, derived: g.derived, text: text } : null;
      state.readingError = g ? null : (a.detectorError || (a.detectorRan ? 'no reading came back' : null));
      if (!keepView) state.view = 'reading';
    }

    var fresh = getText() === text;
    if (fresh) renderCanvas();
    renderCounts();
    refreshHumanizeButton();
    renderVerdict();
    renderMeasurements();
    renderFindings();
    renderStyleSignals();
    renderFlagged();

    if (detect && state.reading && !keepView) {
      setStatus('ok', judgeName() + ' reads this as ' + (state.reading.label || 'unscored') +
        (state.reading.p === null ? '' : ', ' + pct(state.reading.p) + ' AI') + '.');
    } else if (detect && !state.reading && !keepView) {
      setStatus('error', judgeName() + ' could not run' + (state.readingError ? ': ' + state.readingError : '') + '.');
    } else if (keepView && state.humanize) {
      var h = state.humanize;
      setStatus('ok', h.flipped
        ? 'Done. ' + h.judge + ' reads the rewrite as ' + h.after.label + '.'
        : 'Done. ' + h.judge + ' still reads it as ' + (h.after.label || 'unscored') + '.');
    } else if (!keepView) {
      var n = a.findings.length;
      setStatus(fresh ? 'ok' : 'stale', (fresh ? 'Measured. ' : 'Edited since this ran. ') +
        n + ' ' + plural(n, 'finding') + ' in Details.');
    }
    emit('humanizer:analysis', { fresh: fresh });
  }

  var debounceTimer = null;

  function scheduleAnalyze() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () {
      debounceTimer = null;
      /* the judge is the free local estimate, so once there is enough text
         (the scorer needs about 250 characters) every pause measures it */
      analyze({ detect: getText().trim().length >= AUTO_MEASURE_CHARS });
    }, DEBOUNCE_MS);
  }

  /* detect: true runs the judge. With the default free surrogate it costs
     nothing, so typing pauses run it once the draft is long enough; Measure
     and a finished Humanize run always do. With GPTZero configured for a
     bench it would be a paid call, which is why that is opt-in. */
  function analyze(opts) {
    opts = opts || {};
    var detect = opts.detect === true;
    if (debounceTimer) { clearTimeout(debounceTimer); debounceTimer = null; }
    if (activeRun) return;
    var text = getText();
    state.text = text;

    if (!text.trim()) {
      state.analysis = null;
      state.located = [];
      state.reading = null;
      state.readingError = null;
      state.view = 'reading';
      $('empty-state').hidden = false;
      $('legend').hidden = true;
      refreshHumanizeButton();
      renderCounts(); renderVerdict(); renderMeasurements();
      renderFindings(); renderStyleSignals(); renderFlagged();
      setStatus('idle', 'Ready.');
      return;
    }

    var token = ++state.requestToken;
    var wordCount = words(text).length;
    var measure = $('analyze-btn');
    measure.disabled = true;
    setStatus('loading', detect
      ? 'Scoring ' + wordCount + ' ' + plural(wordCount, 'word')
      : 'Measuring ' + wordCount + ' ' + plural(wordCount, 'word'));

    request('/api/analyze', { text: text, detect: detect }, detect ? 90000 : 30000).then(function (raw) {
      if (token !== state.requestToken) return;
      measure.disabled = false;
      hideError();
      setPillHealthy();
      applyAnalysis(text, normalizeAnalysis(raw), detect, opts.keepView === true);
    }, function (err) {
      if (token !== state.requestToken) return;
      measure.disabled = false;
      var msg = (err && err.message) || 'unknown error';
      setStatus('error', 'That did not work: ' + msg + '. Your draft is untouched.');
      setPill('bad', 'Service unreachable');
      showError(msg);
    });
  }

  /* ── health ──────────────────────────────────────────────────────────── */

  function setPillHealthy() {
    /* the judge is named in the console row; the pill carries the service state */
    if (state.yardstick === 'gptzero') setPill('ok', 'Connected, GPTZero live (paid)');
    else setPill('ok', 'Connected');
  }

  function checkHealth() {
    setPill('wait', 'Checking the service');
    request('/api/health').then(function (h) {
      var obj = h && typeof h === 'object' ? h : {};
      hideError();
      /* the contract names the judge here; an older service leaves it out
         and the LLM health route fills it in */
      if (typeof obj.yardstick === 'string' && obj.yardstick) applyYardstick(obj.yardstick);
      else setPillHealthy();
      probeLlmHealth();
    }, function (err) {
      setPill('bad', 'Service unreachable');
      showError((err && err.message) || 'the health check failed');
    });
  }

  /* ── billing ─────────────────────────────────────────────────────────
     GET /api/me says whether the paywall is on, how many words are left and
     which plan is active. The top row shows the balance, the console says
     what a run will use, and a 402 from a charged route opens the plans
     sheet instead of the error box. With the paywall off (paywall: false,
     or no /api/me route at all) none of this appears and nothing is gated.
     Every charged answer carries the new balance, in credits, in the
     X-Longhand-Credits-Balance header; words_per_credit turns it into words. */
  var ME_PATH = '/api/me';
  var PLANS_PATH = '/api/billing/plans';
  var CHECKOUT_PATH = '/api/billing/checkout';
  var PORTAL_PATH = '/api/billing/portal';
  var BALANCE_STORE = 'humanizer.balanceBefore';
  var BALANCE_HEADER = 'X-Longhand-Credits-Balance';

  var billing = {
    on: false,            /* paywall true and someone to charge: a user or a guest */
    guest: false,         /* nobody signed in; the free words belong to this address */
    user: null,
    wordsPerCredit: null,
    wordsLeft: null,
    plan: '',             /* "" | "monthly" | "yearly" | "lifetime" */
    plans: null,          /* the /api/billing/plans answer, once loaded */
    loading: null         /* the pending plans request */
  };

  function fmtInt(n) {
    var v = Math.max(0, Math.round(Number(n) || 0));
    return String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /* fills a node with text, every numeral (and a price) in the mono face */
  function numify(node, text) {
    var parts = String(text).split(/(\$?\d[\d,]*(?:\.\d+)?)/);
    for (var i = 0; i < parts.length; i++) {
      if (!parts[i]) continue;
      if (i % 2) node.appendChild(el('span', 'num', parts[i]));
      else node.appendChild(document.createTextNode(parts[i]));
    }
    return node;
  }

  function planLabel(plan) {
    if (plan === 'monthly') return 'Monthly plan';
    if (plan === 'yearly') return 'Yearly plan';
    if (plan === 'lifetime') return 'Lifetime';
    return plan ? plan.charAt(0).toUpperCase() + plan.slice(1) + ' plan' : '';
  }

  function intervalName(interval, name) {
    if (interval === 'month') return 'Monthly';
    if (interval === 'year') return 'Yearly';
    if (interval === 'lifetime') return 'Lifetime';
    return name ? name.charAt(0).toUpperCase() + name.slice(1) : 'Plan';
  }

  function applyMe(me) {
    var obj = me && typeof me === 'object' ? me : {};
    var u = obj.user && typeof obj.user === 'object' ? obj.user : null;
    billing.on = obj.paywall === true && !!u;
    billing.guest = !!(u && u.guest === true);
    billing.user = u;
    var wpc = loose(u && u.words_per_credit);
    if (wpc === null) wpc = loose(obj.words_per_credit);
    if (wpc !== null && wpc > 0) billing.wordsPerCredit = wpc;
    billing.wordsLeft = u ? loose(u.words_left) : null;
    if (billing.wordsLeft === null && u && billing.wordsPerCredit && loose(u.credits) !== null) {
      billing.wordsLeft = Math.round(loose(u.credits) * billing.wordsPerCredit);
    }
    billing.plan = u && typeof u.plan === 'string' ? u.plan : '';
    renderBalance();
    renderCost();
  }

  function checkBilling() {
    request(ME_PATH).then(applyMe, function () { applyMe(null); });
  }

  function refreshMe() {
    return request(ME_PATH).then(function (me) { applyMe(me); return me; });
  }

  function noteBalance(res) {
    if (!billing.on || !res || !res.headers) return;
    var v = loose(res.headers.get(BALANCE_HEADER));
    if (v === null || !billing.wordsPerCredit) return;
    billing.wordsLeft = Math.round(v * billing.wordsPerCredit);
    renderBalance();
    renderCost();
  }

  function billingFrom402(body) {
    if (!body || typeof body !== 'object') return;
    var wl = loose(body.words_left);
    if (wl === null && loose(body.balance) !== null && billing.wordsPerCredit) {
      wl = Math.round(loose(body.balance) * billing.wordsPerCredit);
    }
    if (wl !== null) billing.wordsLeft = wl;
    billing.on = true;
    renderBalance();
    renderCost();
  }

  /* the account row for a visitor who has not signed in: the free words for
     this address and a way to sign in; no name, no plan, no Sign out */
  function guestRow() {
    var box = $('account');
    var signin = $('signin-link');
    var signout = $('signout-btn');
    var who = $('who');
    if (who) { who.textContent = ''; who.title = ''; }
    if (signin) signin.hidden = false;
    if (signout) signout.hidden = true;
    if (box) box.hidden = false;
  }

  function renderBalance() {
    var bal = $('balance');
    var plan = $('plan-name');
    var btn = $('plans-btn');
    if (!bal || !plan || !btn) return;
    if (!billing.on) { bal.hidden = true; plan.hidden = true; btn.hidden = true; return; }
    clear(bal);
    if (billing.wordsLeft !== null) {
      bal.appendChild(el('span', 'num', fmtInt(billing.wordsLeft)));
      bal.appendChild(document.createTextNode(billing.guest
        ? ' free ' + plural(billing.wordsLeft, 'word') + ' left'
        : ' ' + plural(billing.wordsLeft, 'word') + ' left'));
      bal.hidden = false;
    } else {
      bal.hidden = true;
    }
    if (billing.guest) {
      plan.hidden = true;
      btn.hidden = true;
      guestRow();
      return;
    }
    var name = planLabel(billing.plan);
    plan.textContent = name;
    plan.hidden = !name;
    btn.hidden = false;
    /* /api/me answered with a user, so someone is signed in: show the row
       even if /api/auth/me has not answered yet */
    var box = $('account');
    var who = $('who');
    var signin = $('signin-link');
    if (signin) signin.hidden = true;
    if (who && !who.textContent && billing.user) {
      who.textContent = billing.user.name || billing.user.email || '';
      who.title = billing.user.email || '';
    }
    if (box) box.hidden = false;
  }

  /* where a guest goes when the free words run out: the sign-up page, then
     the plans. The server names the URL on its 402; this is the fallback. */
  var SIGNUP_URL = '/signup?next=' + encodeURIComponent('/pricing') + '&error=free_used';
  function guestOut(body) {
    var url = body && typeof body.signup_url === 'string' && body.signup_url.charAt(0) === '/' ? body.signup_url : SIGNUP_URL;
    try { setStatus('idle', 'Your free words are used up. Create an account to continue.'); } catch (e) { /* before the DOM */ }
    location.assign(url);
  }

  function billingShort(text) {
    if (!billing.on || billing.wordsLeft === null) return false;
    if (billing.user && billing.user.is_admin === true) return false;
    return words(text).length > billing.wordsLeft;
  }

  /* the hint beside Humanize: what this draft uses, out of what is left */
  function renderCost() {
    var node = $('cost-hint');
    if (!node) return;
    var n = words(state.text).length;
    if (!billing.on || billing.wordsLeft === null || !n) {
      node.hidden = true;
      node.removeAttribute('data-short');
      clear(node);
      return;
    }
    clear(node);
    node.hidden = false;
    if (billingShort(state.text)) {
      node.setAttribute('data-short', '1');
      node.appendChild(document.createTextNode(billing.guest ? 'Not enough free words left; ' : 'Not enough words left; '));
      var link = el('button', 'link-btn', billing.guest ? 'create an account' : 'choose a plan');
      link.type = 'button';
      link.id = 'cost-plans';
      link.addEventListener('click', function () {
        if (billing.guest) { guestOut(null); return; }
        openPlans({ words_needed: n, words_left: billing.wordsLeft }, link);
      });
      node.appendChild(link);
      node.appendChild(document.createTextNode('.'));
      return;
    }
    node.removeAttribute('data-short');
    node.appendChild(document.createTextNode('Uses about '));
    node.appendChild(el('span', 'num', fmtInt(n)));
    node.appendChild(document.createTextNode(' ' + plural(n, 'word') + ' of your '));
    node.appendChild(el('span', 'num', fmtInt(billing.wordsLeft)));
    node.appendChild(document.createTextNode(' left.'));
  }

  /* ── the plans sheet ──────────────────────────────────────────────────
     One dialog: the free allowance, the plans, the one-time top-ups, and a
     way to the subscription page. Opened by a 402, the cost hint and the
     Plans button. Focus stays inside it; Escape, the Close button and a
     click on the ground close it and return focus to what opened it. */
  var sheet = { open: false, opener: null, busy: false };

  function focusablesIn(root) {
    var list = root.querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])');
    return Array.prototype.filter.call(list, function (n) { return !n.hidden && n.offsetParent !== null; });
  }

  function onSheetKey(e) {
    if (e.key === 'Escape' || e.key === 'Esc') { e.preventDefault(); e.stopPropagation(); closePlans(); return; }
    if (e.key !== 'Tab') return;
    var box = $('plans-sheet');
    var f = focusablesIn(box);
    if (!f.length) { e.preventDefault(); box.focus(); return; }
    var first = f[0];
    var last = f[f.length - 1];
    if (e.shiftKey && (document.activeElement === first || document.activeElement === box)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function setPlansStatus(kind, message, retry) {
    var node = $('plans-status');
    if (!node) return;
    clear(node);
    node.setAttribute('data-state', kind || 'idle');
    if (message) node.appendChild(document.createTextNode(message));
    if (retry) {
      node.appendChild(document.createTextNode(' '));
      var b = el('button', 'link-btn', 'Try again');
      b.type = 'button';
      b.addEventListener('click', retry);
      node.appendChild(b);
    }
  }

  /* the line that says why the sheet opened: the draft is bigger than the balance */
  function renderWhy(why) {
    var p = $('plans-why');
    if (!p) return;
    clear(p);
    var need = why ? loose(why.words_needed) : null;
    if (need === null && why && loose(why.needed) !== null && billing.wordsPerCredit) {
      need = Math.round(loose(why.needed) * billing.wordsPerCredit);
    }
    var have = why ? loose(why.words_left) : null;
    if (have === null) have = billing.wordsLeft;
    if (need === null) { p.hidden = true; return; }
    numify(p, 'This draft needs ' + fmtInt(need) + ' ' + plural(need, 'word') + ' and you have ' + fmtInt(have || 0) + ' left.');
    p.hidden = false;
  }

  function openPlans(why, opener) {
    var back = $('plans-backdrop');
    var box = $('plans-sheet');
    if (!back || !box) return;
    sheet.opener = opener || document.activeElement;
    sheet.open = true;
    renderWhy(why);
    back.hidden = false;
    document.documentElement.setAttribute('data-sheet', '1');
    void back.offsetWidth; /* so the fade runs from the hidden state */
    back.setAttribute('data-in', '1');
    setTimeout(function () { try { box.focus(); } catch (e) { /* detached */ } }, 0);
    loadPlans(false);
  }

  function closePlans() {
    var back = $('plans-backdrop');
    if (!back || !sheet.open) return;
    sheet.open = false;
    back.removeAttribute('data-in');
    document.documentElement.removeAttribute('data-sheet');
    var hide = function () { if (!sheet.open) back.hidden = true; };
    if (reduceMotion.matches) hide(); else setTimeout(hide, 160);
    var opener = sheet.opener;
    sheet.opener = null;
    if (opener && typeof opener.focus === 'function' && document.body.contains(opener) && !opener.hidden) {
      try { opener.focus(); } catch (e) { /* gone */ }
    }
  }

  function loadPlans(force) {
    if (billing.plans && !force) { renderPlans(); return; }
    if (billing.loading) return;
    setPlansStatus('wait', 'Loading the plans.');
    billing.loading = request(PLANS_PATH).then(function (p) {
      billing.loading = null;
      billing.plans = p && typeof p === 'object' ? p : {};
      var wpc = loose(billing.plans.words_per_credit);
      if (billing.wordsPerCredit === null && wpc !== null && wpc > 0) billing.wordsPerCredit = wpc;
      setPlansStatus('idle', '');
      renderPlans();
    }, function (err) {
      billing.loading = null;
      setPlansStatus('error', 'The plans could not be loaded: ' + ((err && err.message) || 'no answer') + '.', function () { loadPlans(true); });
    });
  }

  function planRow(item, isPack) {
    var row = el('div', 'plan-row');
    var main = el('div', 'plan-main');
    var title, meta;
    if (isPack) {
      var credits = loose(item.credits);
      var wordsIn = credits !== null && billing.wordsPerCredit ? Math.round(credits * billing.wordsPerCredit) : null;
      title = wordsIn !== null ? fmtInt(wordsIn) + ' words' : (item.name || 'Top-up');
      meta = 'One-time top-up, paid once.';
    } else {
      title = intervalName(item.interval, item.name);
      var wpm = loose(item.words_per_month);
      if (wpm === null && loose(item.allowance_credits) !== null && billing.wordsPerCredit) {
        wpm = Math.round(loose(item.allowance_credits) * billing.wordsPerCredit);
      }
      meta = (wpm !== null ? fmtInt(wpm) + ' words a month, ' : '') +
        (item.interval === 'lifetime' ? 'one payment, no renewal.'
          : item.interval === 'year' ? 'renews every year until you cancel.'
          : 'renews every month until you cancel.');
    }
    main.appendChild(numify(el('p', 'plan-h'), title));
    main.appendChild(numify(el('p', 'plan-meta'), meta));
    row.appendChild(main);
    row.appendChild(numify(el('p', 'plan-price'), typeof item.label === 'string' ? item.label : ''));
    var current = !isPack && !!billing.plan && billing.plan === item.name;
    if (current) {
      row.setAttribute('data-current', '1');
      row.appendChild(el('p', 'plan-current', 'Your plan'));
      return row;
    }
    var btn = el('button', 'btn btn-primary', isPack ? 'Top up' : 'Choose');
    btn.type = 'button';
    btn.disabled = !!(billing.plans && billing.plans.stripe === false);
    btn.setAttribute('aria-label', (isPack ? 'Top up with ' : 'Choose ') + title);
    btn.addEventListener('click', function () {
      checkout(isPack ? { pack: item.name } : { plan: item.name }, btn);
    });
    row.appendChild(btn);
    return row;
  }

  function renderPlans() {
    var p = billing.plans || {};
    var free = $('plans-free');
    clear(free);
    var freeWords = loose(p.free_words);
    if (freeWords !== null) {
      var line = fmtInt(freeWords) + ' words free when you sign up';
      if (billing.user && !billing.plan) {
        line += billing.wordsLeft !== null && billing.wordsLeft > 0
          ? ', ' + fmtInt(billing.wordsLeft) + ' of them still here.'
          : ', already used.';
      } else {
        line += '.';
      }
      numify(free, line);
      free.hidden = false;
    } else {
      free.hidden = true;
    }

    var rows = $('plan-rows');
    clear(rows);
    var plans = Array.isArray(p.plans) ? p.plans : [];
    plans.forEach(function (pl) { if (pl && typeof pl === 'object') rows.appendChild(planRow(pl, false)); });
    if (!plans.length) rows.appendChild(el('p', 'muted', 'No plans are on offer on this server yet.'));

    var packs = Array.isArray(p.packs) ? p.packs : [];
    var packsH = $('packs-h');
    var packRows = $('pack-rows');
    clear(packRows);
    packsH.hidden = !packs.length;
    packs.forEach(function (pk) { if (pk && typeof pk === 'object') packRows.appendChild(planRow(pk, true)); });

    $('manage-btn').hidden = !billing.plan;
    if (p.stripe === false) setPlansStatus('error', 'Payments are not set up on this server yet, so nothing can be bought here.');
  }

  function checkout(body, btn) {
    if (sheet.busy) return;
    sheet.busy = true;
    var was = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Opening checkout';
    setPlansStatus('wait', 'Opening the checkout page.');
    /* remembered so the return can tell a new balance from the old one */
    try { if (billing.wordsLeft !== null) sessionStorage.setItem(BALANCE_STORE, String(billing.wordsLeft)); } catch (e) { /* private mode */ }
    request(CHECKOUT_PATH, body).then(function (r) {
      var url = r && typeof r.url === 'string' ? r.url : '';
      if (!url) throw new Error('no checkout address came back');
      location.href = url;
    }, function (err) {
      sheet.busy = false;
      btn.disabled = false;
      btn.textContent = was;
      setPlansStatus('error', 'The checkout page did not open: ' + ((err && err.message) || 'no answer') + '. Try again.');
    });
  }

  function openPortal(btn) {
    if (sheet.busy) return;
    sheet.busy = true;
    btn.disabled = true;
    setPlansStatus('wait', 'Opening your subscription page.');
    request(PORTAL_PATH, {}).then(function (r) {
      var url = r && typeof r.url === 'string' ? r.url : '';
      if (!url) throw new Error('no address came back');
      location.href = url;
    }, function (err) {
      sheet.busy = false;
      btn.disabled = false;
      var msg = (err && err.message) || 'no answer';
      setPlansStatus('error', /HTTP 404/.test(msg)
        ? 'There is no subscription on this account to manage.'
        : 'The subscription page did not open: ' + msg + '. Try again.');
    });
  }

  /* back from Stripe: /app?purchase=success or /app?purchase=cancelled */
  function handleReturn() {
    var p = PARAMS.get('purchase');
    if (!p) return;
    try {
      var u = new URL(location.href);
      u.searchParams.delete('purchase');
      history.replaceState(null, '', u.pathname + u.search + u.hash);
    } catch (e) { /* an old browser keeps the query */ }
    if (p === 'cancelled') { setStatus('idle', 'Payment cancelled, nothing was charged.'); return; }
    if (p !== 'success') return;
    var before = null;
    try {
      before = loose(sessionStorage.getItem(BALANCE_STORE));
      sessionStorage.removeItem(BALANCE_STORE);
    } catch (e) { /* private mode */ }
    setStatus('loading', 'Payment received. Updating your balance.');
    var tries = 0;
    var finish = function (rose) {
      var n = billing.wordsLeft;
      if (n === null) { setStatus('idle', 'Payment received. Reload in a minute to see your new balance.'); return; }
      var line = 'Payment received: ' + fmtInt(n) + ' ' + plural(n, 'word') + ' available.';
      setStatus('ok', rose ? line : line + ' If that does not include your purchase yet, reload in a minute.');
    };
    var poll = function () {
      tries++;
      refreshMe().then(function () {
        var now = billing.wordsLeft;
        if (before === null) before = now;
        if (now !== null && before !== null && now > before) { finish(true); return; }
        if (tries >= 10) { finish(false); return; }
        setTimeout(poll, 1000);
      }, function () {
        if (tries >= 10) { finish(false); return; }
        setTimeout(poll, 1000);
      });
    };
    poll();
  }

  function wireBilling() {
    var plansBtn = $('plans-btn');
    if (plansBtn) plansBtn.addEventListener('click', function () { openPlans(null, plansBtn); });
    var close = $('plans-close');
    if (close) close.addEventListener('click', closePlans);
    var back = $('plans-backdrop');
    if (back) back.addEventListener('mousedown', function (e) { if (e.target === back) closePlans(); });
    var box = $('plans-sheet');
    if (box) box.addEventListener('keydown', onSheetKey);
    var manage = $('manage-btn');
    if (manage) manage.addEventListener('click', function () { openPortal(manage); });
  }

  /* ── the account ─────────────────────────────────────────────────────
     GET /api/auth/me says who is signed in; the top row shows the name and a
     Sign out button. If the route is missing (a development server with no
     auth) the row stays as it is and the app runs. If it answers that nobody
     is signed in, the visitor is a guest: the row shows a Sign in link and
     GET /api/me (billing) fills in the free words for this address. The
     page only leaves for the sign-in form when a product route answers 401. */
  function checkAccount() {
    var box = $('account');
    if (!box) return;
    request('/api/auth/me').then(function (me) {
      if (!me || typeof me !== 'object') return;
      if (me.signed_in === false) { guestRow(); return; }
      if (!me.signed_in) return;
      var u = me.user || {};
      var who = $('who');
      who.textContent = u.name || u.email || '';
      who.title = u.email || '';
      var signin = $('signin-link');
      var signout = $('signout-btn');
      if (signin) signin.hidden = true;
      if (signout) signout.hidden = false;
      box.hidden = false;
    }, function () { /* no auth routes: nothing to show */ });
  }

  function signOut() {
    var btn = $('signout-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Signing out'; }
    var done = function () { location.href = '/'; };
    fetch(API_BASE + '/api/auth/signout', { method: 'POST', headers: { Accept: 'application/json' } }).then(done, done);
  }

  function renderOnline() {
    var offline = !navigator.onLine;
    $('offline-banner').hidden = !offline;
    if (offline) setPill('bad', 'Offline');
  }

  /* ── wiring ──────────────────────────────────────────────────────────── */

  /* ── the divider ─────────────────────────────────────────────────────────
     The Before column is the CSS variable --col-before, a fraction of the
     usable width (the instrument minus the 16px divider). Drag it, or focus
     it and use the arrow keys (Home and End for the limits). Enter, Space or
     a double click folds the smaller pane to a 44px rail and the same again
     brings it back; a folded pane also opens when clicked. Dragging snaps
     at 30, 50 and 70 percent. Under 960px the panes stack and the same
     handle sets the height of the draft. Everything is remembered. */
  var SPLIT_STORE = 'readshuman.split';
  var RAIL = 44;
  var SNAPS = [0.3, 0.5, 0.7];
  var split = { f: 0.5, last: 0.5, h: null, collapsed: '', shown: 0.5 };
  var narrowMQ = window.matchMedia ? window.matchMedia('(max-width: 960px)') : { matches: false };

  function splitNarrow() { return !!narrowMQ.matches; }
  function splitLoad() {
    try {
      var s = JSON.parse(localStorage.getItem(SPLIT_STORE) || 'null');
      if (!s || typeof s !== 'object') return;
      if (typeof s.f === 'number' && isFinite(s.f)) split.f = clamp(s.f, 0.2, 0.8);
      if (typeof s.h === 'number' && isFinite(s.h)) split.h = s.h;
      if (s.collapsed === 'before' || s.collapsed === 'after') split.collapsed = s.collapsed;
      split.last = split.f;
    } catch (e) { /* private mode */ }
  }
  function splitSave() {
    try { localStorage.setItem(SPLIT_STORE, JSON.stringify({ f: split.f, h: split.h, collapsed: split.collapsed })); }
    catch (e) { /* private mode */ }
  }
  function splitUsable() { return Math.max(1, $('verdict').getBoundingClientRect().width - 16); }
  function splitTarget() {
    if (split.collapsed === 'before') return RAIL / splitUsable();
    if (split.collapsed === 'after') return 1 - RAIL / splitUsable();
    return split.f;
  }
  /* paint one fraction; anim.js calls this every frame while the split glides */
  function splitApply(f) {
    split.shown = f;
    $('verdict').style.setProperty('--col-before', 'calc((100% - 16px) * ' + f.toFixed(4) + ')');
  }
  function splitAria() {
    var d = $('divider');
    var inst = $('verdict');
    var now = split.collapsed === 'before' ? 0 : split.collapsed === 'after' ? 100 : Math.round(split.f * 100);
    d.setAttribute('aria-valuenow', String(now));
    d.setAttribute('aria-valuetext', split.collapsed
      ? (split.collapsed === 'before' ? 'Before is folded away' : 'After is folded away')
      : 'Before ' + now + ' percent, After ' + (100 - now) + ' percent');
    inst.setAttribute('data-collapsed', split.collapsed);
  }
  function splitGo(animate) {
    var inst = $('verdict');
    var from = split.shown;
    var to = splitTarget();
    splitAria();
    splitSave();
    if (splitNarrow()) { inst.classList.remove('is-gliding'); splitApply(to); return; }
    if (!animate || reduceMotion.matches) { inst.classList.remove('is-gliding'); splitApply(to); return; }
    if (document.documentElement.getAttribute('data-anim') === 'on') {
      inst.classList.remove('is-gliding');
      emit('humanizer:split', { from: from, to: to });
    } else {
      inst.classList.add('is-gliding');
      splitApply(to);
      setTimeout(function () { inst.classList.remove('is-gliding'); }, 320);
    }
  }
  function splitSet(f) { split.f = clamp(f, 0.2, 0.8); split.last = split.f; split.collapsed = ''; splitGo(true); }
  function splitToggle() {
    if (split.collapsed) { split.collapsed = ''; split.f = split.last; }
    else { split.last = split.f; split.collapsed = split.f <= 0.5 ? 'before' : 'after'; }
    splitGo(true);
  }
  function snapF(f, ticks) {
    var out = f;
    for (var i = 0; i < SNAPS.length; i++) {
      var near = Math.abs(f - SNAPS[i]) < 0.02;
      if (ticks && ticks[i]) ticks[i].setAttribute('data-near', near ? '1' : '0');
      if (near) out = SNAPS[i];
    }
    return out;
  }
  function snapH(h) {
    var vh = window.innerHeight;
    for (var i = 0; i < SNAPS.length; i++) if (Math.abs(h - SNAPS[i] * vh) < 12) return SNAPS[i] * vh;
    return h;
  }
  function beforeHeight() {
    var body = $('canvas-scroll');
    return split.h || (body ? body.getBoundingClientRect().height : 300);
  }
  function setBeforeHeight(h) {
    split.h = Math.round(clamp(h, 120, window.innerHeight * 0.8));
    split.collapsed = '';
    $('verdict').style.setProperty('--before-h', split.h + 'px');
    splitAria();
    splitSave();
  }

  function wireSplit() {
    var d = $('divider');
    var inst = $('verdict');
    if (!d || !inst) return;
    var ticks = Array.prototype.slice.call(d.querySelectorAll('.divider-snaps i'));
    splitLoad();
    splitApply(splitTarget());
    splitAria();
    if (split.h) inst.style.setProperty('--before-h', split.h + 'px');

    var drag = null;
    d.addEventListener('pointerdown', function (e) {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      e.preventDefault();
      try { d.setPointerCapture(e.pointerId); } catch (x) { /* older engines */ }
      var r = inst.getBoundingClientRect();
      var dr = d.getBoundingClientRect();
      drag = { id: e.pointerId, x0: e.clientX, y0: e.clientY, rect: r, h0: beforeHeight() };
      for (var i = 0; i < ticks.length; i++) {
        ticks[i].style.left = Math.round(r.left + (r.width - 16) * SNAPS[i] + 8 - dr.left) + 'px';
        ticks[i].setAttribute('data-near', '0');
      }
      inst.classList.add('is-dragging');
      inst.classList.remove('is-gliding');
    });
    d.addEventListener('pointermove', function (e) {
      if (!drag || e.pointerId !== drag.id) return;
      if (splitNarrow()) {
        setBeforeHeight(snapH(drag.h0 + (e.clientY - drag.y0)));
        return;
      }
      var f = clamp((e.clientX - drag.rect.left - 8) / Math.max(1, drag.rect.width - 16), 0.2, 0.8);
      f = snapF(f, ticks);
      split.f = f; split.last = f; split.collapsed = '';
      splitApply(f);
      splitAria();
    });
    function end(e) {
      if (!drag || e.pointerId !== drag.id) return;
      drag = null;
      inst.classList.remove('is-dragging');
      try { d.releasePointerCapture(e.pointerId); } catch (x) { /* already released */ }
      splitSave();
    }
    d.addEventListener('pointerup', end);
    d.addEventListener('pointercancel', end);
    d.addEventListener('dblclick', function () { splitToggle(); });
    d.addEventListener('keydown', function (e) {
      var k = e.key;
      var step = e.shiftKey ? 0.1 : 0.05;
      if (k === 'Enter' || k === ' ') { e.preventDefault(); splitToggle(); return; }
      if (splitNarrow()) {
        if (k === 'ArrowUp' || k === 'ArrowDown') { e.preventDefault(); setBeforeHeight(beforeHeight() + (k === 'ArrowDown' ? 40 : -40)); }
        return;
      }
      if (k === 'Home') { e.preventDefault(); splitSet(0.2); return; }
      if (k === 'End') { e.preventDefault(); splitSet(0.8); return; }
      if (k === 'ArrowLeft' || k === 'ArrowRight') {
        e.preventDefault();
        var base = split.collapsed === 'before' ? 0.2 : split.collapsed === 'after' ? 0.8 : split.f;
        splitSet(base + (k === 'ArrowRight' ? step : -step));
      }
    });
    /* a folded pane opens when clicked */
    ['before', 'after'].forEach(function (side) {
      $('pane-' + side).addEventListener('click', function () { if (split.collapsed === side) splitToggle(); });
    });
    window.addEventListener('resize', function () { splitApply(splitTarget()); });
  }

  /* ── the construction layer ─────────────────────────────────────────────
     Every region marked data-c gets a hairline box, its name, the API field
     that fills it (data-src; data-flow="out" when the region sends instead)
     and its size. Shown while the Construction switch is on or the backtick
     key is held. Rebuilt on resize and whenever the interface changes. */
  var construction = { on: false, held: false, timer: null };
  function constructionShown() { return construction.on || construction.held; }

  function buildConstruction() {
    var layer = $('construction');
    if (!layer) return;
    clear(layer);
    layer.style.height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight) + 'px';
    var sx = window.pageXOffset, sy = window.pageYOffset;
    var shell = document.querySelector('.shell');
    if (shell) {
      var sr = shell.getBoundingClientRect();
      var pad = parseFloat(getComputedStyle(shell).paddingLeft) || 0;
      var edge = el('div', 'construction-shell');
      edge.style.left = (sr.left + sx + pad) + 'px';
      edge.style.width = Math.max(0, sr.width - pad * 2) + 'px';
      layer.appendChild(edge);
    }
    Array.prototype.forEach.call(document.querySelectorAll('[data-c]'), function (node) {
      if (node.hidden) return;
      var r = node.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return;
      var box = el('div', 'c-box');
      box.style.left = (r.left + sx) + 'px';
      box.style.top = (r.top + sy) + 'px';
      box.style.width = r.width + 'px';
      box.style.height = r.height + 'px';
      if (node.getAttribute('data-flow') === 'out') box.setAttribute('data-flow', 'out');
      var tag = el('span', 'c-tag');
      if (r.width < 160) tag.className += ' c-tag-out';
      tag.appendChild(el('span', 'c-name', node.getAttribute('data-c')));
      var src = node.getAttribute('data-src');
      if (src) tag.appendChild(el('span', 'c-src', src));
      box.appendChild(tag);
      if (r.height >= 100) box.appendChild(el('span', 'c-dim', Math.round(r.width) + 'x' + Math.round(r.height)));
      layer.appendChild(box);
    });
  }

  function setConstruction(show) {
    var layer = $('construction');
    if (!layer) return;
    var root = document.documentElement;
    if (show) {
      buildConstruction();
      layer.hidden = false;
      root.setAttribute('data-construction', 'on');
      emit('humanizer:construction', { on: true });
      return;
    }
    root.removeAttribute('data-construction');
    emit('humanizer:construction', { on: false });
    var wait = root.getAttribute('data-anim') === 'on' && !reduceMotion.matches ? 200 : 0;
    setTimeout(function () { if (!constructionShown()) { layer.hidden = true; clear(layer); } }, wait);
  }
  function refreshConstruction() { if (constructionShown()) buildConstruction(); }
  function queueConstruction(ms) {
    if (!constructionShown()) return;
    if (construction.timer) clearTimeout(construction.timer);
    construction.timer = setTimeout(refreshConstruction, ms || 60);
  }

  function wireConstruction() {
    var sw = $('construction-switch');
    wireSwitch(sw, function (on) { construction.on = on; setConstruction(constructionShown()); });
    document.addEventListener('keydown', function (e) {
      if (e.key !== '`' || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      e.preventDefault();
      construction.held = true;
      if (!construction.on) setConstruction(true);
    });
    document.addEventListener('keyup', function (e) {
      if (e.key !== '`' || !construction.held) return;
      construction.held = false;
      if (!construction.on) setConstruction(false);
    });
    window.addEventListener('blur', function () {
      if (!construction.held) return;
      construction.held = false;
      if (!construction.on) setConstruction(false);
    });
    window.addEventListener('resize', function () { queueConstruction(80); });
    ['humanizer:verdict', 'humanizer:canvas', 'humanizer:progress-open', 'humanizer:progress-close', 'humanizer:analysis']
      .forEach(function (n) { document.addEventListener(n, function () { queueConstruction(60); }); });
    Array.prototype.forEach.call(document.querySelectorAll('details.fold'), function (d) {
      d.addEventListener('toggle', function () { queueConstruction(220); });
    });
  }

  function init() {
    editor = $('editor');
    scroller = $('canvas-scroll');
    statusLine = $('status-line');
    errorBox = $('error-box');

    /* the whole before pane is the place to write: a click anywhere on it focuses the draft */
    scroller.addEventListener('mousedown', function (e) {
      if (e.target !== scroller) return;
      e.preventDefault();
      editor.focus();
    });

    editor.addEventListener('input', function () {
      state.text = getText();
      $('empty-state').hidden = !!state.text.trim();
      editor.setAttribute('data-empty', state.text.trim() ? '0' : '1');
      refreshHumanizeButton();
      renderCounts();
      renderVerdict();
      setStatus('stale', 'Edited.');
      scheduleAnalyze();
    });

    /* plain text only, so pasted markup cannot break the document model */
    editor.addEventListener('paste', function (e) {
      if (!e.clipboardData) return;
      e.preventDefault();
      document.execCommand('insertText', false, e.clipboardData.getData('text/plain') || '');
    });

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && activeRun) { e.preventDefault(); cancelHumanize(); return; }
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!activeRun && !$('humanize-btn').disabled) humanize();
      }
    });

    $('analyze-btn').addEventListener('click', function () { analyze({ detect: true }); });
    $('humanize-btn').addEventListener('click', function () { humanize(); });
    $('undo-btn').addEventListener('click', function () { undoHumanize(); });
    $('use-btn').addEventListener('click', function () { useRewrite(); });
    $('copy-btn').addEventListener('click', function () { copyRewrite(); });
    $('progress-cancel').addEventListener('click', function () { cancelHumanize(); });

    Array.prototype.forEach.call(document.querySelectorAll('[data-go]'), function (b) {
      b.addEventListener('click', function () { goTo(b.getAttribute('data-go')); });
    });

    var facts = $('facts');
    if (facts) { facts.addEventListener('input', refreshFactsState); refreshFactsState(); }

    wireSplit();
    /* on a phone the keyboard covers the lower half; bring the draft up */
    if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) {
      $('editor').addEventListener('focus', function () {
        setTimeout(function () {
          if (window.visualViewport && window.visualViewport.height < window.innerHeight - 100) {
            var pane = $('pane-before') || $('editor');
            pane.scrollIntoView({ block: 'start', behavior: 'smooth' });
          }
        }, 300);
      });
    }
    wireConstruction();
    /* anim.js drives the split's glide through this */
    window.readshuman = { splitApply: splitApply };

    Array.prototype.forEach.call(document.querySelectorAll('[data-sample]'), function (b) {
      b.addEventListener('click', function () {
        state.humanize = null; state.afterLocated = []; state.preHumanize = null; state.reading = null; state.view = 'reading';
        $('undo-btn').hidden = true;
        renderRun(); renderEdits(); renderAfter();
        setText(SAMPLES[b.getAttribute('data-sample')] || SAMPLE);
        editor.focus();
        analyze({ detect: false });
      });
    });

    window.addEventListener('online', function () {
      renderOnline();
      checkHealth();
      if (state.text.trim()) analyze({ detect: false });
    });
    window.addEventListener('offline', renderOnline);

    renderOnline();
    checkHealth();
    checkAccount();
    var signoutBtn = $('signout-btn');
    if (signoutBtn) signoutBtn.addEventListener('click', signOut);
    wireBilling();
    checkBilling();

    setText(PARAMS.get('sample') === 'ai' ? SAMPLE : '');
    refreshHumanizeButton();
    renderVerdict();
    renderRun();
    renderEdits();
    renderAfter();
    renderMeasurements();
    renderFindings();
    renderStyleSignals();
    renderFlagged();

    if (state.text.trim()) analyze({ detect: false });
    else setStatus('idle', 'Ready.');
    handleReturn();

    window.__hzReady = true;
    emit('humanizer:ready');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
