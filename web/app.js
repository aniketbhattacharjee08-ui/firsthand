/* ══════════════════════════════════════════════════════════════════════════
   humanizer — Stage 0 front end.

   No build step, no bundler, no runtime dependency. One classic script so the
   page also works from file:// (ES-module imports are blocked there).

   Design constraints come from research/11 §7:
     1. no textarea — a real document model with one span per sentence;
     2. two dials, risk and quality, never one;
     3. name the measurement every time — value, human band, consequence;
     4. Word-style simple markup; green/red reserved for the rewrite diff,
        risk gets its own amber->oxblood ramp, uncertainty its own state;
     5. paper and ink, no gradients, no purple.
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var PARAMS = new URLSearchParams(location.search);
  var API_BASE = PARAMS.get('api') || '';
  var DEBOUNCE_MS = 800;

  /* ── small utilities ───────────────────────────────────────────────────── */

  function $(id) { return document.getElementById(id); }

  function num(v) {
    return typeof v === 'number' && isFinite(v) ? v : null;
  }

  /* Every numeric in the contract may be null (NaN server-side). */
  function fmt(v, digits) {
    var n = num(v);
    if (n === null) return 'n/a';
    return n.toFixed(digits === undefined ? 2 : digits);
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

  var reduceMotion = window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)')
    : { matches: false };

  /* ── the measurement table ─────────────────────────────────────────────
     Bands are the ones the Python extractor checks (features/shape.py,
     punctuation.py, register.py) plus the report-10 / Biber targets. Nothing
     is shown without the band and the source beside it: rule 3. */

  var FEATURES = [
    { key: 'sent_len_cv', label: 'sentence-length variation (CV)', band: [0.42, 0.60], digits: 2,
      source: 'human academic prose 0.42 to 0.60 (report 10, five pre-2023 corpora)',
      note: 'a band, not a maximum — a measured GPT baseline sits at 0.50, inside it, and 0.85 is as anomalous as 0.30' },
    { key: 'sent_lag1_autocorr', label: 'lag-1 sentence autocorrelation', band: [-0.10, 0.25], digits: 2,
      source: 'human 0.01 to 0.10, tolerated to -0.10/+0.25',
      note: 'humans do not alternate long and short; a strong negative value is the signature of a naive burstiness rule' },
    { key: 'sent_short_share', label: 'sentences under 10 words', band: [0.09, 0.14], pct: true,
      source: 'human 9% to 14%',
      note: 'widen the tails, not the middle' },
    { key: 'sent_long_share', label: 'sentences over 30 words', band: [0.18, 0.28], pct: true,
      source: 'human 18% to 28%' },
    { key: 'para_words_cv', label: 'paragraph-length variation (CV)', band: [0.42, 0.71], digits: 2,
      source: 'human 0.42 to 0.71',
      note: 'higher than the sentence CV in every genre measured' },
    { key: 'ai_vocab_weighted_per_1k', label: 'AI vocabulary, weighted /1k', band: [0, 8], digits: 1,
      source: 'flagged above 8.0 per 1,000 words',
      note: 'weights are published excess-frequency ratios (delve 25x, tapestry 18x in phrase, underscores 13.8x)' },
    { key: 'formal_connective_per_1k', label: 'formal connectives /1k', band: [1, 8], digits: 1,
      source: 'flagged above 8.0; better essays rise 0.84 to 3.75 across PERSUADE scores 1 to 6',
      note: 'stripping these is the one edit that cuts detection risk and raises the grade' },
    { key: 'nominalization_per_1k', label: 'nominalizations /1k', band: [55, 80], digits: 1,
      source: 'academic sub-registers 61.0 to 72.1 (Biber & Gray)',
      note: 'better essays are MORE nominalized — do not cut this to sound human' },
    { key: 'passive_per_1k', label: 'passives /1k', band: [12, 25], digits: 1,
      source: 'academic prose 18.5, about 25% of finite verbs (Biber et al.)' },
    { key: 'comma_per_1k', label: 'commas /1k', band: [57, 65], digits: 1,
      source: 'own measurement 57 to 65',
      note: 'punctuation is the most author-stable family measured (commas ICC 0.44)' },
    { key: 'contraction_per_1k', label: 'contractions /1k', band: [0, 1.4], digits: 2,
      source: 'research articles under 1.4',
      note: 'contractions help evasion and cost grade — a never-in-body edit for academic targets' },
    { key: 'mtld', label: 'lexical diversity (MTLD)', band: null, digits: 1,
      source: 'no published academic band — reported for context',
      note: 'academic type-token ratio is LOWER than fiction and news; raising it is not an improvement' },
    { key: 'n_words', label: 'words', band: null, digits: 0,
      source: 'shape features are noisy and detectors hedge below 300 words' },
    { key: 'n_sentences', label: 'sentences', band: null, digits: 0, source: '' }
  ];

  var FEATURE_BY_KEY = {};
  FEATURES.forEach(function (f) { FEATURE_BY_KEY[f.key] = f; });

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

  /* server bands win; we only compute one when the server did not send it */
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

  /* ── grade-cost classification ─────────────────────────────────────────
     The API sends grade_cost as free text from the research/09 conflict
     matrix ("none, slightly positive", "negative, it raises the grade",
     "medium if added", "high if reduced further"). The product's whole point
     is that some edits pay twice and some trade, so the string is classified
     rather than printed bare — and the raw string is still shown. */

  function classifyGradeCost(raw) {
    var s = String(raw === undefined || raw === null ? '' : raw).toLowerCase().trim();
    if (!s) return { kind: 'neutral', label: 'grade cost not stated', raw: '' };
    if (s.indexOf('negative') === 0 || s.indexOf('raises the grade') >= 0) {
      return { kind: 'gain', label: 'fixing this raises the grade', raw: raw };
    }
    if (s.indexOf('positive') >= 0 && s.indexOf('none') === 0) {
      return { kind: 'gain', label: 'free to fix, slightly positive', raw: raw };
    }
    if (s === 'none' || s.indexOf('none') === 0) {
      return { kind: 'neutral', label: 'no grade cost to fix', raw: raw };
    }
    if (/\bhigh\b/.test(s)) return { kind: 'cost', label: 'high grade cost', raw: raw };
    if (/\bmedium\b/.test(s)) return { kind: 'cost', label: 'medium grade cost', raw: raw };
    if (/\blow\b/.test(s)) return { kind: 'cost', label: 'low grade cost', raw: raw };
    return { kind: 'neutral', label: 'grade cost: ' + s, raw: raw };
  }

  var SEVERITY_WEIGHT = { high: 6, medium: 3, low: 1, info: 0 };
  var SEVERITY_ORDER = { high: 0, medium: 1, low: 2, info: 3 };

  /* ── samples ───────────────────────────────────────────────────────────── */

  var SAMPLES = {
    ai: [
      'In today\'s rapidly evolving academic landscape, the study of urban green space has become a pivotal area of inquiry. Researchers have begun to delve into the intricate relationship between vegetation cover and public health outcomes. Moreover, the topic underscores a broader shift in how cities are planned, managed, and understood. It is important to note that this shift is multifaceted, complex, and ongoing.',
      'Furthermore, a comprehensive review of the literature reveals several key themes. First, green space is consistently associated with improved mental health indicators. Second, the effect appears robust across a range of demographic groups. Third, the mechanisms remain the subject of considerable debate. These findings collectively showcase the significance of the topic.',
      'Additionally, it is worth noting that methodological challenges persist throughout this body of work. Many studies rely on cross-sectional designs that cannot establish causality. Consequently, the evidence base remains somewhat limited in scope. Nevertheless, the overall pattern of results is compelling and noteworthy.',
      'In conclusion, urban green space represents a rich tapestry of environmental, social, and psychological factors. Future research should delve deeper into the causal mechanisms at play. Ultimately, such work will pave the way for more effective and equitable urban policy. This is not merely an academic exercise, but a pressing societal need.'
    ].join('\n\n'),

    human: [
      'Between 1993 and 2011 the city of Leipzig lost roughly a fifth of its population, and the vacant lots left behind were converted, piecemeal and without much of a plan, into small parks. The conversions were documented well enough to support a natural experiment. Kabisch and Haase used them for exactly that.',
      'Their result is modest. Residents living within 300 metres of a converted lot reported better general health on the SF-12 than residents 800 metres away, with an effect size of about 0.14 standard deviations after adjustment for income, age and prior health status. The gap widened slightly over the study period, though the confidence interval at the final wave includes zero, and the authors say so.',
      'What makes the Leipzig data unusual is that the assignment of green space was determined by demolition schedules rather than by neighbourhood affluence, which breaks the confounding that damages most cross-sectional work in this literature. It does not break all of it. Demolition was concentrated in the northeast, and the northeast differed from the rest of the city in ways that were not measured at baseline.',
      'A second objection is harder to dismiss. Self-reported health was collected by postal survey, and response rates near the converted lots ran eleven points higher than elsewhere. If people who felt healthier were also more inclined to return a survey about their neighbourhood park, the effect is inflated by an unknown amount. The authors acknowledge the problem in a footnote and do not attempt to correct for it.',
      'I take the Leipzig study as suggestive rather than decisive. The design is better than what preceded it, the measured effect is small, and the direction of the likely bias is toward the finding. That combination argues for replication in a city where the demolition pattern was not spatially clustered.'
    ].join('\n\n')
  };

  /* ── client-side lexicons, used only for the per-sentence explanation and
     for mock mode. Weights mirror src/humanizer/features/ai_lexicon.py. ──── */

  var AI_WORDS = {
    delve: 25, delves: 28, delving: 20, underscore: 12, underscores: 13.8,
    underscoring: 11, showcase: 9, showcases: 9, showcasing: 10.7, pivotal: 8.5,
    intricate: 8, intricacies: 8, intricately: 5.5, realm: 7.5, realms: 7,
    garnered: 7, burgeoning: 6.5, noteworthy: 6, meticulous: 6, meticulously: 6,
    commendable: 6, encompassing: 5.5, multifaceted: 5.5, nuanced: 5, leverage: 5,
    leveraging: 5, harness: 4.5, harnessing: 4.5, unveiling: 4.5, unravel: 4.5,
    elucidate: 4.5, endeavors: 4, paramount: 4, profound: 4, seamless: 4,
    seamlessly: 4, robust: 3.5, crucial: 3.5, vital: 3, significant: 2,
    comprehensive: 3, holistic: 4, innovative: 3.5, transformative: 4.5,
    groundbreaking: 4.5, landscape: 4, tapestry: 12, testament: 8, camaraderie: 12
  };

  var AI_PHRASES = {
    'it is important to note': 15, 'it is worth noting': 12, 'it is crucial to': 10,
    'plays a crucial role': 12, 'plays a vital role': 11, 'plays a pivotal role': 14,
    'in the realm of': 14, 'in the ever-evolving': 16, "in today's fast-paced": 14,
    "in today's digital age": 14, 'navigating the complexities': 16,
    'a testament to': 10, 'rich tapestry': 18, 'delve into': 20, 'delves into': 22,
    'shed light on': 8, 'sheds light on': 8, 'at the forefront': 8,
    'pave the way': 8, 'paves the way': 8, 'paving the way': 8,
    'when it comes to': 5, 'in conclusion': 8, 'in summary': 6, 'to sum up': 6
  };

  var FORMAL_CONNECTIVES = [
    'additionally', 'moreover', 'furthermore', 'consequently', 'therefore',
    'thus', 'hence', 'nevertheless', 'nonetheless', 'accordingly',
    'subsequently', 'similarly', 'likewise', 'conversely', 'notably',
    'importantly', 'specifically', 'ultimately', 'overall', 'indeed',
    'in addition', 'in conclusion', 'in summary', 'in contrast',
    'on the other hand', 'as a result', 'for instance', 'for example'
  ];

  function words(text) {
    var m = text.toLowerCase().match(/[a-z][a-z'’-]*/g);
    return m || [];
  }

  /* ── document model ────────────────────────────────────────────────────
     Paragraph and sentence ranges over the plain text. This is what replaces
     the textarea: the canvas is rendered from these ranges, and every span in
     the DOM is addressable by sentence index. */

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

  var ABBREV = /\b(?:Dr|Mr|Mrs|Ms|Prof|St|Fig|No|vs|etc|e\.g|i\.e|cf|al|Jr|Sr|Ph\.D)\.$/i;

  function splitSentences(chunk) {
    /* Intl.Segmenter when present (jsdiff's own README recommends it over a
       .!? regex, which breaks on "Dr." and "e.g."); regex fallback otherwise. */
    if (typeof Intl !== 'undefined' && Intl.Segmenter) {
      try {
        var seg = new Intl.Segmenter('en', { granularity: 'sentence' });
        var out = [];
        var iter = seg.segment(chunk)[Symbol.iterator]();
        var step = iter.next();
        while (!step.done) {
          var s = step.value;
          if (s.segment.trim()) out.push({ from: s.index, to: s.index + s.segment.length });
          step = iter.next();
        }
        if (out.length) return out;
      } catch (e) { /* fall through */ }
    }
    var res = [], start = 0, i;
    for (i = 0; i < chunk.length; i++) {
      var c = chunk[i];
      if (c === '.' || c === '!' || c === '?') {
        var head = chunk.slice(start, i + 1);
        if (ABBREV.test(head)) continue;
        var next = chunk.slice(i + 1, i + 3);
        if (/^\s/.test(next) || i === chunk.length - 1) {
          if (head.trim()) res.push({ from: start, to: i + 1 });
          start = i + 1;
          while (start < chunk.length && /\s/.test(chunk[start])) start++;
          i = start - 1;
        }
      }
    }
    if (start < chunk.length && chunk.slice(start).trim()) res.push({ from: start, to: chunk.length });
    return res;
  }

  /* The API returns sentence text, not offsets, so each sentence is located
     in the document text with a whitespace-tolerant forward scan. */
  function locateSentences(text, sentences) {
    var cursor = 0, out = [];
    (sentences || []).forEach(function (s) {
      var raw = s && typeof s.text === 'string' ? s.text.trim() : '';
      if (!raw) { out.push(null); return; }
      var from = text.indexOf(raw, cursor);
      var to;
      if (from >= 0) {
        to = from + raw.length;
      } else {
        var tokens = raw.split(/\s+/).slice(0, 200).map(escapeRe);
        var hit = null;
        try {
          var re = new RegExp(tokens.join('\\s+'));
          hit = re.exec(text.slice(cursor));
        } catch (e) { hit = null; }
        if (!hit) { out.push(null); return; }
        from = cursor + hit.index;
        to = from + hit[0].length;
      }
      cursor = to;
      out.push({
        index: num(s.index) === null ? out.length : s.index,
        text: text.slice(from, to),
        paragraph_index: num(s.paragraph_index),
        length: num(s.length),
        risk: num(s.risk),
        from: from,
        to: to
      });
    });
    return out.filter(Boolean);
  }

  /* ── risk arithmetic shared by the fallback dial and by mock mode ─────── */

  function outOfBand(value, lo, hi) {
    var n = num(value);
    if (n === null) return null;
    var half = Math.abs(hi - lo) / 2 || 1;
    if (n < lo) return clamp((lo - n) / (half * 2), 0, 1);
    if (n > hi) return clamp((n - hi) / (half * 2), 0, 1);
    return 0;
  }

  /* Composite used only when the API returns no detector. It is a distance
     from the human bands, not a detector score, and is labelled as such. */
  function compositeRisk(f) {
    var terms = [
      [outOfBand(f.sent_len_cv, 0.42, 0.60), 1.4],
      [outOfBand(f.sent_lag1_autocorr, -0.10, 0.25), 0.8],
      [outOfBand(f.sent_short_share, 0.09, 0.14), 0.6],
      [outOfBand(f.sent_long_share, 0.18, 0.28), 0.6],
      [outOfBand(f.para_words_cv, 0.42, 0.71), 0.8],
      [num(f.ai_vocab_weighted_per_1k) === null ? null : clamp(f.ai_vocab_weighted_per_1k / 20, 0, 1), 1.2],
      [num(f.formal_connective_per_1k) === null ? null : clamp(f.formal_connective_per_1k / 16, 0, 1), 1.0]
    ];
    var sum = 0, wsum = 0;
    terms.forEach(function (t) {
      if (t[0] === null) return;
      sum += t[0] * t[1]; wsum += t[1];
    });
    if (!wsum) return null;
    return clamp(0.08 + 0.90 * (sum / wsum), 0, 0.99);
  }

  /* ══════════════════════════════════════════════════════════════════════
     WRITING-QUALITY DIAL — client-side heuristic placeholder.

     Labelled as a placeholder in the UI (the "?" beside the dial). It exists
     because research/11 §7 decision 2 says risk must never be shown without
     quality beside it, and the API does not return a quality score.

       start          78            a mid-band essay
       register term  +/- 18        five features that research/10 and
                                    research/09 show RISE with essay grade:
                                    nominalization 55-80, passives 12-25,
                                    formal connectives 1-8, commas 57-65,
                                    paragraph CV 0.42-0.71. Each is worth
                                    +3.6 in band and up to -3.6 outside,
                                    scaled by how far outside it sits.
       findings term  0 to -30      severity weight (high 6, medium 3, low 1)
                                    multiplied by what the grade_cost string
                                    says about fixing it:
                                      x1.00  fixing costs grade, so the
                                             current state is a real defect
                                             (opinion markers, contractions)
                                      x0.35  fixing is free — a detector
                                             problem, not a quality problem
                                      x0.15  fixing raises the grade — an
                                             opportunity, barely a defect
       short-document  pull to 70 by half if under 300 words, because shape
                       features are noisy there and detectors hedge.

     Deliberately NOT a rubric score. research/09: a fine-tuned RoBERTa on
     PERSUADE 2.0 reaches QWK 0.841 and a frontier LLM zero-shot 0.042-0.081,
     so nothing derived from seven summary statistics deserves more than the
     word "placeholder".
     ══════════════════════════════════════════════════════════════════════ */

  var QUALITY_REGISTER = [
    { key: 'nominalization_per_1k', band: [55, 80] },
    { key: 'passive_per_1k', band: [12, 25] },
    { key: 'formal_connective_per_1k', band: [1, 8] },
    { key: 'comma_per_1k', band: [57, 65] },
    { key: 'para_words_cv', band: [0.42, 0.71] }
  ];

  function qualityScore(features, findings) {
    var f = features || {};
    var lines = [];
    var score = 78;
    lines.push('base                       78.0');

    var registerDelta = 0, counted = 0;
    QUALITY_REGISTER.forEach(function (spec) {
      var d = outOfBand(f[spec.key], spec.band[0], spec.band[1]);
      if (d === null) return;
      counted++;
      registerDelta += d === 0 ? 3.6 : -3.6 * d;
    });
    score += registerDelta;
    lines.push('register (' + counted + '/5 measured)   ' + (registerDelta >= 0 ? '+' : '') + registerDelta.toFixed(1));

    var penalty = 0;
    (findings || []).forEach(function (fd) {
      var w = SEVERITY_WEIGHT[fd.severity];
      if (!w) return;
      var kind = classifyGradeCost(fd.grade_cost).kind;
      var mult = kind === 'cost' ? 1.0 : kind === 'gain' ? 0.15 : 0.35;
      penalty += w * mult;
    });
    penalty = Math.min(penalty, 30);
    score -= penalty;
    lines.push('findings (' + (findings || []).length + ')            -' + penalty.toFixed(1));

    var nWords = num(f.n_words);
    var lowConfidence = nWords !== null && nWords < 300;
    if (lowConfidence) {
      score = 70 + (score - 70) * 0.5;
      lines.push('under 300 words: halved toward 70');
    }
    score = clamp(score, 0, 100);
    lines.push('---------------------------------');
    lines.push('quality                    ' + score.toFixed(0));

    return { score: score, lowConfidence: lowConfidence, breakdown: lines.join('\n') };
  }

  /* ── HTTP ───────────────────────────────────────────────────────────────
     The API is being written by another agent right now, so nothing here
     assumes a field exists. */

  function request(path, body) {
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var opts = { method: body ? 'POST' : 'GET', headers: { Accept: 'application/json' } };
    if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    if (controller) opts.signal = controller.signal;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 20000) : null;

    return fetch(API_BASE + path, opts).then(function (res) {
      if (timer) clearTimeout(timer);
      if (!res.ok) {
        return res.text().catch(function () { return ''; }).then(function (t) {
          throw new Error('HTTP ' + res.status + (t ? ' — ' + t.slice(0, 160) : ''));
        });
      }
      return res.json();
    }, function (err) {
      if (timer) clearTimeout(timer);
      if (err && err.name === 'AbortError') throw new Error('request timed out after 20s');
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
        severity: SEVERITY_WEIGHT.hasOwnProperty(f.severity) ? f.severity : 'info',
        message: f.message,
        detail: typeof f.detail === 'string' ? f.detail : '',
        grade_cost: f.grade_cost
      };
    }) : [];
    findings.sort(function (a, b) {
      return (SEVERITY_ORDER[a.severity] || 9) - (SEVERITY_ORDER[b.severity] || 9);
    });
    return {
      features: r.features && typeof r.features === 'object' ? r.features : {},
      bands: r.bands && typeof r.bands === 'object' ? r.bands : {},
      findings: findings,
      reference: r.reference && typeof r.reference === 'object' ? r.reference : null,
      detectors: r.detectors && typeof r.detectors === 'object' ? r.detectors : {},
      deviations: r.deviations && typeof r.deviations === 'object' ? r.deviations : {},
      sentences: Array.isArray(r.sentences) ? r.sentences : [],
      advisory: r.sentence_risk_is_advisory !== false
    };
  }

  /* ══════════════════════════════════════════════════════════════════════
     MOCK MODE (?mock=1, or the "use mock data" button on the error state).

     A reduced in-browser reimplementation of the Stage 0 extractor, so the
     whole interface — heatmap, dials, measurements, findings — can be shown
     with the Python service down or absent. It computes real numbers from the
     real text using the same feature definitions; it is abridged, it has no
     reference distribution and no trained detector, and the UI says so in a
     persistent banner.
     ══════════════════════════════════════════════════════════════════════ */

  function mtldPass(tokens, threshold) {
    var factors = 0, types = Object.create(null), tokenCount = 0, typeCount = 0, i;
    for (i = 0; i < tokens.length; i++) {
      tokenCount++;
      if (!types[tokens[i]]) { types[tokens[i]] = 1; typeCount++; }
      var ttr = typeCount / tokenCount;
      if (ttr <= threshold) {
        factors++; types = Object.create(null); tokenCount = 0; typeCount = 0;
      }
    }
    if (tokenCount > 0) {
      var last = typeCount / tokenCount;
      factors += (1 - last) / (1 - threshold);
    }
    return factors > 0 ? tokens.length / factors : tokens.length;
  }

  function countPhrases(lower, table) {
    var total = 0, hits = [];
    Object.keys(table).forEach(function (p) {
      var re;
      try { re = new RegExp('\\b' + escapeRe(p).replace(/\\ /g, '\\s+').replace(/ /g, '\\s+') + '\\b', 'g'); }
      catch (e) { return; }
      var m = lower.match(re);
      if (m) { total += table[p] * m.length; hits.push({ surface: p, weight: table[p], count: m.length }); }
    });
    return { weight: total, hits: hits };
  }

  function aiVocabHits(fragment) {
    var lower = fragment.toLowerCase();
    var hits = [];
    var weight = 0;
    words(fragment).forEach(function (w) {
      if (AI_WORDS[w]) { hits.push({ surface: w, weight: AI_WORDS[w] }); weight += AI_WORDS[w]; }
    });
    var ph = countPhrases(lower, AI_PHRASES);
    weight += ph.weight;
    ph.hits.forEach(function (h) { hits.push({ surface: h.surface, weight: h.weight }); });
    return { weight: weight, hits: hits };
  }

  function connectiveOpener(sentence) {
    var head = sentence.toLowerCase().replace(/^[^a-z]+/, '');
    for (var i = 0; i < FORMAL_CONNECTIVES.length; i++) {
      var c = FORMAL_CONNECTIVES[i];
      if (head.indexOf(c) === 0) return c;
    }
    return null;
  }

  var PASSIVE_RE = /\b(?:is|are|was|were|be|been|being|am)\s+(?:\w+ly\s+)?(?:\w+ed|born|known|shown|given|taken|seen|made|found|used|held|done|built|written|drawn|brought|thought)\b/g;
  var NOMINAL_RE = /\b\w{5,}(?:tion|tions|sion|sions|ment|ments|ness|ity|ities|ance|ence|ancy|ency|ism|isms)\b/g;
  var CONTRACTION_RE = /\b\w+['’](?:s|t|re|ve|ll|d|m)\b/g;

  function mockAnalyze(text, referenceName) {
    var paras = paragraphRanges(text);
    var sentences = [];
    paras.forEach(function (p, pi) {
      var chunk = text.slice(p.from, p.to);
      splitSentences(chunk).forEach(function (s) {
        var rawSlice = chunk.slice(s.from, s.to);
        var lead = rawSlice.length - rawSlice.replace(/^\s+/, '').length;
        var body = rawSlice.trim();
        if (!body) return;
        sentences.push({
          index: sentences.length,
          text: body,
          paragraph_index: pi,
          from: p.from + s.from + lead,
          to: p.from + s.from + lead + body.length,
          length: words(body).length
        });
      });
    });

    var allWords = words(text);
    var n = allWords.length;
    var per1k = function (count) { return n ? (count / n) * 1000 : null; };
    var lengths = sentences.map(function (s) { return s.length; });

    var mean = lengths.length ? lengths.reduce(function (a, b) { return a + b; }, 0) / lengths.length : null;
    var sd = null, cv = null, lag1 = null;
    if (lengths.length > 1 && mean) {
      var variance = lengths.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / (lengths.length - 1);
      sd = Math.sqrt(variance);
      cv = sd / mean;
      var numr = 0, den = 0;
      for (var i = 0; i < lengths.length; i++) {
        den += (lengths[i] - mean) * (lengths[i] - mean);
        if (i < lengths.length - 1) numr += (lengths[i] - mean) * (lengths[i + 1] - mean);
      }
      lag1 = den ? numr / den : null;
    }

    var paraWordCounts = paras.map(function (p) { return words(text.slice(p.from, p.to)).length; })
      .filter(function (c) { return c > 0; });
    var paraCv = null;
    if (paraWordCounts.length > 1) {
      var pm = paraWordCounts.reduce(function (a, b) { return a + b; }, 0) / paraWordCounts.length;
      var pv = paraWordCounts.reduce(function (a, b) { return a + (b - pm) * (b - pm); }, 0) / (paraWordCounts.length - 1);
      paraCv = pm ? Math.sqrt(pv) / pm : null;
    }

    var ai = aiVocabHits(text);
    var lower = text.toLowerCase();
    var connectiveCount = 0;
    FORMAL_CONNECTIVES.forEach(function (c) {
      var re;
      try { re = new RegExp('\\b' + escapeRe(c).replace(/ /g, '\\s+') + '\\b', 'g'); } catch (e) { return; }
      var m = lower.match(re);
      if (m) connectiveCount += m.length;
    });

    var passives = (text.match(PASSIVE_RE) || []).length;
    var nominals = (lower.match(NOMINAL_RE) || []).length;
    var contractions = (text.match(CONTRACTION_RE) || []).length;
    var commas = (text.match(/,/g) || []).length;
    var openerFormal = 0, seenPara = {};
    sentences.forEach(function (s) {
      if (seenPara[s.paragraph_index]) return;
      seenPara[s.paragraph_index] = 1;
      if (connectiveOpener(s.text)) openerFormal++;
    });
    var coordinatorOpeners = sentences.filter(function (s) {
      return /^(and|but|so)\b/i.test(s.text.replace(/^[^A-Za-z]+/, ''));
    }).length;

    var features = {
      n_words: n,
      n_sentences: sentences.length,
      n_paragraphs: paras.length,
      sent_len_mean: mean,
      sent_len_sd: sd,
      sent_len_cv: cv,
      sent_lag1_autocorr: lag1,
      sent_short_share: lengths.length ? lengths.filter(function (l) { return l < 10; }).length / lengths.length : null,
      sent_long_share: lengths.length ? lengths.filter(function (l) { return l > 30; }).length / lengths.length : null,
      para_words_cv: paraCv,
      ai_vocab_weighted_per_1k: per1k(ai.weight),
      formal_connective_per_1k: per1k(connectiveCount),
      nominalization_per_1k: per1k(nominals),
      passive_per_1k: per1k(passives),
      comma_per_1k: per1k(commas),
      contraction_per_1k: per1k(contractions),
      mtld: allWords.length >= 20
        ? (mtldPass(allWords, 0.72) + mtldPass(allWords.slice().reverse(), 0.72)) / 2
        : null,
      para_opener_formal_share: paras.length ? openerFormal / paras.length : null,
      sent_initial_coordinator_share: sentences.length ? coordinatorOpeners / sentences.length : null
    };

    var bands = {};
    FEATURES.forEach(function (spec) {
      if (!spec.band) return;
      bands[spec.key] = bandState(spec, features[spec.key], null);
    });

    /* findings mirror src/humanizer/eval/report.py: same codes, same
       grade_cost strings, thresholds abridged. */
    var findings = [];
    var aiV = features.ai_vocab_weighted_per_1k || 0;
    if (aiV > 8) findings.push({
      code: 'ai_vocabulary', severity: aiV > 20 ? 'high' : 'medium',
      message: 'AI vocabulary density is ' + aiV.toFixed(1) + ' per 1,000 words.',
      detail: 'Highest-effect lexical signal in the catalog and it survives paraphrasing. Removing it also removes the red highlights a human grader sees.',
      grade_cost: 'none, slightly positive'
    });
    var fc = features.formal_connective_per_1k || 0;
    if (fc > 8) findings.push({
      code: 'formal_connectives', severity: fc > 15 ? 'high' : 'medium',
      message: 'Formal connectives run ' + fc.toFixed(1) + ' per 1,000 words.',
      detail: 'Stripping these is the single best joint move: high detector benefit, and it raises the grade because top-band coherence "attracts no attention".',
      grade_cost: 'negative, it raises the grade'
    });
    if (num(features.para_opener_formal_share) !== null && features.para_opener_formal_share > 0.04) findings.push({
      code: 'paragraph_openers', severity: 'medium',
      message: Math.round(features.para_opener_formal_share * 100) + '% of paragraphs open with a formal connective.',
      detail: 'Human academic prose runs at most 3-4%.',
      grade_cost: 'none'
    });
    if (bands.sent_len_cv === 'low') findings.push({
      code: 'uniform_sentences', severity: 'high',
      message: 'Sentence-length CV is ' + fmt(cv) + ', below the human band.',
      detail: 'Human academic prose runs 0.42-0.60. Widen the tails rather than the middle: target 9-14% of sentences under 10 words and 18-28% over 30.',
      grade_cost: 'none, rubrics reward sentence variety'
    });
    else if (bands.sent_len_cv === 'high') findings.push({
      code: 'overshot_variance', severity: 'medium',
      message: 'Sentence-length CV is ' + fmt(cv) + ', above the human band.',
      detail: 'Overshooting is as anomalous as undershooting. A CV of 0.85 sits as far outside the human distribution as 0.30.',
      grade_cost: 'none'
    });
    if (bands.sent_lag1_autocorr === 'low') findings.push({
      code: 'alternating_sentences', severity: 'medium',
      message: 'Sentence lengths alternate long and short (lag-1 autocorrelation ' + fmt(lag1) + ').',
      detail: 'Humans do not alternate; their lag-1 sits near zero and the real structure is long-range. This is the signature of a naive burstiness rule.',
      grade_cost: 'none'
    });
    if (num(paraCv) !== null && paraCv < 0.35) findings.push({
      code: 'uniform_paragraphs', severity: 'medium',
      message: 'Paragraph lengths are uniform (CV ' + fmt(paraCv) + ').',
      detail: 'Human paragraph CV runs 0.42-0.71, above the sentence CV in every genre.',
      grade_cost: 'none'
    });
    if ((features.contraction_per_1k || 0) > 1.4) findings.push({
      code: 'contractions_in_academic', severity: 'medium',
      message: 'Contractions run ' + fmt(features.contraction_per_1k) + ' per 1,000 words.',
      detail: 'Research articles sit under 1.4. Contractions help evade detection but cost grade under genre conventions, so they are a "never in body" edit for academic targets.',
      grade_cost: 'medium if added'
    });
    var opinion = (lower.match(/\bi (?:think|feel|believe)\b|\bin my opinion\b/g) || []).length;
    if (opinion) findings.push({
      code: 'opinion_markers', severity: 'high',
      message: 'Found ' + opinion + ' bare opinion marker(s) such as "I think".',
      detail: 'First person for argumentative acts ("I argue") is fine and endorsed by style guides. Bare opinion reads as an unsupported position and is a "never" edit.',
      grade_cost: 'high'
    });
    if (num(features.nominalization_per_1k) !== null && features.nominalization_per_1k < 40) findings.push({
      code: 'low_nominalization', severity: 'low',
      message: 'Nominalization is ' + fmt(features.nominalization_per_1k, 1) + ' per 1,000 words.',
      detail: 'Academic sub-registers run 61-72. Better essays are MORE nominalized, not less, so do not reduce this to sound human.',
      grade_cost: 'high if reduced further'
    });
    if (num(features.sent_initial_coordinator_share) !== null && features.sent_initial_coordinator_share > 0.05) findings.push({
      code: 'sentence_initial_coordinators', severity: 'low',
      message: Math.round(features.sent_initial_coordinator_share * 100) + '% of sentences open with And/But/So.',
      detail: 'This falls from 5.1% to 2.3% between a weak and a strong essay, so heavy use reads as a lower grade band.',
      grade_cost: 'low to medium'
    });

    /* per-sentence advisory risk: document composite, nudged by the named
       patterns actually present in that sentence */
    var docRisk = compositeRisk(features);
    var base = docRisk === null ? 0.4 : docRisk;
    var outSentences = sentences.map(function (s) {
      var local = 0;
      var hits = aiVocabHits(s.text);
      local += clamp(hits.weight / 30, 0, 0.22);
      if (connectiveOpener(s.text)) local += 0.12;
      if ((s.text.match(/,/g) || []).length >= 3 && /\band\b/.test(s.text.toLowerCase())) local += 0.06;
      if (mean !== null && Math.abs(s.length - mean) < 2.5) local += 0.06;
      if (/\d/.test(s.text)) local -= 0.10;
      if (/\b[A-Z][a-z]+\s+(?:and|&)?\s*[A-Z][a-z]+\b/.test(s.text)) local -= 0.04;
      return {
        index: s.index, text: s.text, paragraph_index: s.paragraph_index,
        length: s.length, risk: clamp(base * 0.75 + local + 0.05, 0.02, 0.98)
      };
    });

    var reference = null;
    if (referenceName) {
      var d = docRisk === null ? null : 1.5 + docRisk * 6.5;
      reference = {
        distance: d,
        percentile: d === null ? null : clamp(d * 12, 0, 99),
        n_features: 48,
        coverage: 0.0,
        mock: true
      };
    }

    return {
      features: features,
      bands: bands,
      findings: findings,
      reference: reference,
      detectors: docRisk === null ? {} : {
        'mock-heuristic': {
          ai_probability: docRisk,
          label: docRisk > 0.65 ? 'ai' : docRisk < 0.35 ? 'human' : 'mixed',
          confidence: n < 300 ? 'uncertain' : 'moderate'
        }
      },
      deviations: {},
      sentences: outSentences,
      sentence_risk_is_advisory: true
    };
  }

  /* ══════════════════════════════════════════════════════════════════════
     STATE
     ══════════════════════════════════════════════════════════════════════ */

  var state = {
    text: '',
    analyzedText: '',
    analysis: null,      // normalized response
    located: [],         // sentences with document offsets
    selected: null,      // sentence index
    reference: '',
    mock: PARAMS.get('mock') === '1',
    status: 'idle',
    error: null,
    requestToken: 0
  };

  var editor, statusLine, errorBox;

  /* ── editor plumbing ───────────────────────────────────────────────────
     The document model is the plain text plus the sentence ranges. The DOM is
     a rendering of it, rebuilt only when an analysis lands for the text that
     is currently on screen, with the caret restored by character offset. */

  function getText() {
    var t = editor.innerText || '';
    return t.replace(/\r\n?/g, '\n').replace(/ /g, ' ').replace(/\n{3,}/g, '\n\n');
  }

  function caretOffset() {
    var sel = window.getSelection();
    if (!sel || !sel.rangeCount) return null;
    var range = sel.getRangeAt(0);
    if (!editor.contains(range.startContainer)) return null;
    var walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
    var offset = 0, node;
    while ((node = walker.nextNode())) {
      if (node === range.startContainer) return offset + range.startOffset;
      offset += node.nodeValue.length;
    }
    return null;
  }

  function setCaret(offset) {
    if (offset === null || offset === undefined) return;
    var walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
    var acc = 0, node, last = null;
    while ((node = walker.nextNode())) {
      var len = node.nodeValue.length;
      if (acc + len >= offset) {
        try {
          var r = document.createRange();
          r.setStart(node, clamp(offset - acc, 0, len));
          r.collapse(true);
          var sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(r);
        } catch (e) { /* selection is best-effort */ }
        return;
      }
      acc += len;
      last = node;
    }
    if (last) {
      try {
        var r2 = document.createRange();
        r2.setStart(last, last.nodeValue.length);
        r2.collapse(true);
        var s2 = window.getSelection();
        s2.removeAllRanges();
        s2.addRange(r2);
      } catch (e) { /* ignore */ }
    }
  }

  function riskTier(risk) {
    var r = num(risk);
    if (r === null) return { cls: 'risk-0', confidence: 'none' };
    var tier = Math.min(5, Math.floor(r * 6));
    /* uncertainty is its own state, not a mid-ramp fill (research/11 §3.1) */
    var lowConfidence = Math.abs(r - 0.5) < 0.15;
    return { cls: 'risk-' + tier, confidence: lowConfidence ? 'low' : 'high', tier: tier };
  }

  function renderCanvas() {
    var text = state.text;
    var located = state.located;
    var caret = document.activeElement === editor ? caretOffset() : null;
    var scroll = editor.parentNode.scrollTop;

    clear(editor);
    editor.setAttribute('data-empty', text.trim() ? 'false' : 'true');

    paragraphRanges(text).forEach(function (p) {
      var para = el('p');
      var pos = p.from;
      located.forEach(function (s) {
        if (!s || s.from < p.from || s.from >= p.to) return;
        if (s.from > pos) para.appendChild(document.createTextNode(text.slice(pos, s.from)));
        var end = Math.min(s.to, p.to);
        var tier = riskTier(s.risk);
        var span = el('span', 'sent ' + tier.cls, text.slice(s.from, end));
        span.setAttribute('data-index', String(s.index));
        span.setAttribute('data-confidence', tier.confidence);
        if (num(s.risk) !== null) span.style.setProperty('--risk', num(s.risk).toFixed(3));
        if (state.selected === s.index) span.setAttribute('aria-current', 'true');
        para.appendChild(span);
        pos = end;
      });
      if (pos < p.to) para.appendChild(document.createTextNode(text.slice(pos, p.to)));
      if (!para.childNodes.length) para.appendChild(el('br'));
      editor.appendChild(para);
    });
    if (!editor.childNodes.length) editor.appendChild(el('p'));

    editor.parentNode.scrollTop = scroll;
    if (caret !== null) setCaret(caret);
  }

  function setText(text) {
    state.text = text;
    state.located = [];
    state.selected = null;
    renderCanvas();
    renderCounts();
  }

  function renderCounts() {
    var wordCount = words(state.text).length;
    var sentCount = state.located.length;
    $('doc-counts').textContent = state.text.trim()
      ? wordCount + ' words · ' + (sentCount ? sentCount + ' sentences · ' : '') +
        (wordCount < 300 ? 'under 300 words, shape features are noisy here' : 'length adequate for shape features')
      : 'no text yet';
  }

  /* ── dials ─────────────────────────────────────────────────────────────
     Decision 2: risk and quality side by side, always. The value animates,
     the geometry does not (research/11 §3.3). */

  var ARC = 157.08; /* pi * r, r = 50, half circle */
  var tweens = {};

  function setDial(idValue, idFill, target, suffix) {
    var valueNode = $(idValue);
    var fill = document.querySelector('#' + idFill + ' .gauge-fill');
    if (target === null) {
      valueNode.textContent = '—';
      if (fill) fill.style.strokeDashoffset = ARC;
      return;
    }
    if (fill) fill.style.strokeDashoffset = String(ARC * (1 - clamp(target / 100, 0, 1)));
    if (tweens[idValue]) cancelAnimationFrame(tweens[idValue]);
    if (reduceMotion.matches) {
      valueNode.textContent = Math.round(target) + (suffix || '');
      return;
    }
    var startValue = parseFloat(valueNode.textContent);
    if (!isFinite(startValue)) startValue = 0;
    var t0 = performance.now();
    var step = function (now) {
      var k = clamp((now - t0) / 320, 0, 1);
      var eased = 1 - Math.pow(1 - k, 3);
      valueNode.textContent = Math.round(startValue + (target - startValue) * eased) + (suffix || '');
      if (k < 1) tweens[idValue] = requestAnimationFrame(step);
    };
    tweens[idValue] = requestAnimationFrame(step);
  }

  function pickDetector(detectors) {
    var best = null;
    Object.keys(detectors || {}).forEach(function (name) {
      var d = detectors[name];
      if (!d || typeof d !== 'object' || d.error) return;
      var p = num(d.ai_probability);
      if (p === null) return;
      if (!best || p > best.p) best = { name: name, p: p, label: d.label, confidence: d.confidence };
    });
    return best;
  }

  function renderDials() {
    var a = state.analysis;
    if (!a) {
      setDial('risk-value', 'dial-risk', null);
      setDial('quality-value', 'dial-quality', null);
      $('risk-source').textContent = 'no measurement yet';
      $('quality-source').textContent = 'heuristic placeholder';
      $('quality-breakdown').textContent = '';
      return;
    }

    var det = pickDetector(a.detectors);
    var p = det ? det.p : compositeRisk(a.features);
    setDial('risk-value', 'dial-risk', p === null ? null : p * 100);
    $('risk-source').textContent = det
      ? 'detector "' + det.name + '", p(AI) ' + fmt(det.p) +
        (det.confidence ? ' · ' + det.confidence : '') + (det.label ? ' · verdict ' + det.label : '')
      : p === null
        ? 'not measurable from this text'
        : 'no detector returned — composite distance from the human bands, not a detector score';

    var q = qualityScore(a.features, a.findings);
    setDial('quality-value', 'dial-quality', q.score);
    $('quality-source').textContent = 'heuristic placeholder' +
      (q.lowConfidence ? ' · under 300 words, pulled toward 70' : '');
    $('quality-breakdown').textContent = q.breakdown;

    var refLine = $('reference-line');
    if (a.reference) {
      refLine.textContent =
        'Reference "' + (state.reference || 'unnamed') + '": Mahalanobis ' + fmt(a.reference.distance) +
        ', ' + fmt(a.reference.percentile, 1) + 'th percentile of the human corpus (' +
        (num(a.reference.n_features) === null ? 'n/a' : a.reference.n_features) + ' features, coverage ' +
        fmt(a.reference.coverage) + ')' + (a.reference.mock ? ' — fabricated in mock mode' : '');
    } else {
      refLine.textContent = state.reference
        ? 'Reference "' + state.reference + '" selected but the API returned no distance.'
        : 'No reference distribution selected. Bands only.';
    }
  }

  /* ── measurements: value, human band, state. Decision 3. ──────────────── */

  function renderMeasurements() {
    var host = $('measurements');
    clear(host);
    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }

    FEATURES.forEach(function (spec) {
      var value = a.features[spec.key];
      var stateName = bandState(spec, value, a.bands);
      var row = el('div', 'measure');

      row.appendChild(el('div', 'measure-name', spec.label));

      var right = el('div', 'measure-value', formatFeature(spec, value));
      if (spec.band) {
        var chip = el('span', 'state', stateName);
        chip.setAttribute('data-state', stateName);
        right.appendChild(chip);
      }
      row.appendChild(right);

      var band = formatBand(spec);
      var caption = band
        ? 'human ' + band + ' — ' + spec.source
        : spec.source;
      if (spec.note) caption += '. ' + spec.note;
      row.appendChild(el('div', 'measure-band', caption));
      host.appendChild(row);
    });
  }

  /* ── findings, each with its grade cost. The differentiator. ──────────── */

  function renderFindings() {
    var host = $('findings');
    clear(host);
    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }
    if (!a.findings.length) {
      host.appendChild(el('p', 'muted', 'No findings. Every checked feature sits inside its human band.'));
      return;
    }

    a.findings.forEach(function (f) {
      var card = el('article', 'finding');
      card.setAttribute('data-severity', f.severity);

      var top = el('div', 'finding-top');
      top.appendChild(el('span', 'sev', f.severity));
      top.appendChild(el('span', 'finding-code', f.code || '—'));
      card.appendChild(top);

      card.appendChild(el('p', 'finding-msg', f.message));
      if (f.detail) card.appendChild(el('p', 'finding-detail', f.detail));

      var g = classifyGradeCost(f.grade_cost);
      var chip = el('span', 'grade');
      chip.setAttribute('data-kind', g.kind);
      chip.appendChild(el('b', null, g.label));
      if (g.raw) chip.appendChild(el('span', 'grade-raw', '“' + g.raw + '”'));
      card.appendChild(chip);

      host.appendChild(card);
    });
  }

  /* ── highest-risk sentences: the keyboard route into the canvas ───────── */

  function renderFlagged() {
    var host = $('flagged');
    clear(host);
    if (!state.located.length) {
      host.appendChild(el('p', 'muted', 'No analysis yet.'));
      return;
    }
    var ranked = state.located.filter(function (s) { return num(s.risk) !== null; })
      .slice().sort(function (a, b) { return b.risk - a.risk; }).slice(0, 6);
    if (!ranked.length) {
      host.appendChild(el('p', 'muted', 'The API returned no per-sentence risk.'));
      return;
    }
    ranked.forEach(function (s) {
      var btn = el('button', 'flag-btn');
      btn.type = 'button';
      btn.appendChild(el('span', 'flag-risk', fmt(s.risk)));
      btn.appendChild(el('span', 'flag-text', s.text));
      btn.addEventListener('click', function () { selectSentence(s.index, true); });
      host.appendChild(btn);
    });
  }

  /* ── sentence inspector: why this sentence looks the way it does ─────── */

  function selectSentence(index, scrollTo) {
    state.selected = index;
    Array.prototype.forEach.call(editor.querySelectorAll('.sent'), function (node) {
      if (Number(node.getAttribute('data-index')) === index) {
        node.setAttribute('aria-current', 'true');
        if (scrollTo && node.scrollIntoView) {
          node.scrollIntoView({ block: 'center', behavior: reduceMotion.matches ? 'auto' : 'smooth' });
        }
      } else {
        node.removeAttribute('aria-current');
      }
    });
    renderInspector();
  }

  function renderInspector() {
    var host = $('sentence-inspector');
    clear(host);
    var s = null;
    state.located.forEach(function (item) { if (item.index === state.selected) s = item; });
    if (!s) {
      host.appendChild(el('p', 'muted', 'Click a sentence, or move the caret into one, to see what was measured.'));
      return;
    }

    host.appendChild(el('blockquote', 'inspect-quote', s.text));

    var rows = el('div', 'inspect-rows');
    var tier = riskTier(s.risk);
    function addRow(label, value) {
      var r = el('div', 'inspect-row');
      r.appendChild(el('span', null, label));
      r.appendChild(el('span', null, value));
      rows.appendChild(r);
    }
    addRow('risk (0 to 1)', fmt(s.risk));
    addRow('confidence', tier.confidence === 'none' ? 'no score returned'
      : tier.confidence === 'low' ? 'low — within 0.15 of the 0.50 midpoint' : 'outside the ambiguous middle');
    addRow('length', (num(s.length) === null ? words(s.text).length : s.length) + ' words');
    var docMean = state.analysis ? num(state.analysis.features.sent_len_mean) : null;
    if (docMean !== null) {
      addRow('vs document mean', (s.length - docMean >= 0 ? '+' : '') + fmt(s.length - docMean, 1) + ' words');
    }
    addRow('paragraph', num(s.paragraph_index) === null ? 'n/a' : String(s.paragraph_index + 1));
    host.appendChild(rows);

    var reasons = [];
    var hits = aiVocabHits(s.text);
    hits.hits.slice(0, 5).forEach(function (h) {
      reasons.push('AI-vocabulary hit "' + h.surface + '", published excess frequency about ' + h.weight + 'x');
    });
    var opener = connectiveOpener(s.text);
    if (opener) reasons.push('opens with the formal connective "' + opener + '" — human academic paragraphs open this way at most 3 to 4% of the time');
    if (docMean !== null && Math.abs(s.length - docMean) < 2.5) {
      reasons.push('length sits within 2.5 words of the document mean, which is what drives sentence-length CV below the human 0.42 to 0.60 band');
    }
    if ((s.text.match(/,/g) || []).length >= 3 && /\band\b/i.test(s.text)) {
      reasons.push('three-part coordinated list — AI text runs about 2x human on tricolons');
    }
    if (/\d/.test(s.text)) {
      reasons.push('contains a number: concrete dated or numbered specifics are the best joint move in the conflict matrix, strongly positive for the grade');
    }
    if (!reasons.length) reasons.push('no named pattern from the report 04 catalog fires on this sentence.');

    var list = el('ul', 'inspect-list');
    reasons.forEach(function (r) { list.appendChild(el('li', null, r)); });
    host.appendChild(list);

    var caveat = 'These are named patterns present in the sentence. They do not explain the score: ' +
      'GPTZero says the same of its own AI Vocabulary tool, and per report 07 the document verdict is not an aggregate of sentence scores.';
    host.appendChild(el('p', 'inspect-caveat', caveat));
  }

  /* ── status, errors and the analyze cycle ─────────────────────────────── */

  function setStatus(kind, message) {
    state.status = kind;
    statusLine.setAttribute('data-state', kind);
    statusLine.textContent = message;
  }

  function hideError() { errorBox.hidden = true; }

  function showError(message) {
    clear(errorBox);
    errorBox.hidden = false;
    var block = el('div', 'error-block');
    block.appendChild(el('p', null, 'The analysis API did not answer.'));
    var code = el('p');
    code.appendChild(el('code', null, message));
    block.appendChild(code);
    block.appendChild(el('p', 'muted',
      'The service is POST /api/analyze on the same origin. Start it, or pass ?api=http://localhost:8000 to point elsewhere.'));

    var retry = el('button', 'btn', 'Retry');
    retry.type = 'button';
    retry.addEventListener('click', function () { checkHealth(); analyze(); });

    var toMock = el('button', 'btn', 'Use mock data instead');
    toMock.type = 'button';
    toMock.addEventListener('click', function () { setMock(true); analyze(); });

    var actions = el('div', 'inspect-rows');
    actions.appendChild(retry);
    actions.appendChild(toMock);
    block.appendChild(actions);
    errorBox.appendChild(block);
  }

  function setMock(on) {
    state.mock = !!on;
    $('mock-banner').hidden = !state.mock;
    if (state.mock) {
      hideError();
      setPill('wait', 'mock mode — API not contacted');
    }
  }

  function setPill(kind, text) {
    var pill = $('health');
    pill.className = 'pill pill-' + kind;
    pill.textContent = text;
  }

  function applyAnalysis(text, a) {
    state.analysis = a;
    state.analyzedText = text;
    state.located = locateSentences(text, a.sentences);
    if (state.selected !== null) {
      var stillThere = state.located.some(function (s) { return s.index === state.selected; });
      if (!stillThere) state.selected = null;
    }

    var current = getText();
    var fresh = current === text;
    if (fresh) renderCanvas();

    renderCounts();
    renderDials();
    renderMeasurements();
    renderFindings();
    renderFlagged();
    renderInspector();
    $('advisory-note').hidden = !a.advisory;

    var unlocated = a.sentences.length - state.located.length;
    var parts = [];
    parts.push(state.located.length + ' of ' + a.sentences.length + ' sentences located');
    parts.push(a.findings.length + ' finding' + (a.findings.length === 1 ? '' : 's'));
    if (unlocated > 0) parts.push(unlocated + ' could not be matched to the text');
    if (state.mock) parts.push('mock data');
    setStatus(fresh ? 'ok' : 'stale',
      (fresh ? 'Analyzed. ' : 'Text edited since this analysis. ') + parts.join(' · '));
  }

  var debounceTimer = null;

  function scheduleAnalyze() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () { debounceTimer = null; analyze(); }, DEBOUNCE_MS);
  }

  function analyze() {
    if (debounceTimer) { clearTimeout(debounceTimer); debounceTimer = null; }
    var text = getText();
    state.text = text;

    if (!text.trim()) {
      state.analysis = null;
      state.located = [];
      state.selected = null;
      renderCounts(); renderDials(); renderMeasurements(); renderFindings(); renderFlagged(); renderInspector();
      setStatus('idle', 'Idle. Paste an essay or load a sample.');
      return;
    }

    var token = ++state.requestToken;
    var btn = $('analyze-btn');
    btn.disabled = true;
    setStatus('loading', 'Analyzing ' + words(text).length + ' words…');

    var work;
    if (state.mock) {
      work = new Promise(function (resolve) {
        setTimeout(function () { resolve(mockAnalyze(text, state.reference)); }, 60);
      });
    } else {
      work = request('/api/analyze', { text: text, reference: state.reference || null });
    }

    work.then(function (raw) {
      if (token !== state.requestToken) return;
      btn.disabled = false;
      hideError();
      applyAnalysis(text, normalizeAnalysis(raw));
    }, function (err) {
      if (token !== state.requestToken) return;
      btn.disabled = false;
      setStatus('error', 'Analysis failed — ' + ((err && err.message) || 'unknown error'));
      setPill('bad', 'API unreachable');
      showError((err && err.message) || String(err));
    });
  }

  /* ── health and reference list ─────────────────────────────────────────── */

  function checkHealth() {
    if (state.mock) { setPill('wait', 'mock mode — API not contacted'); return; }
    setPill('wait', 'checking API…');
    request('/api/health').then(function (h) {
      var refs = h && h.references;
      var refCount = Array.isArray(refs) ? refs.length : num(refs);
      setPill('ok', [
        'API ' + ((h && h.version) || 'unknown'),
        (h && h.status) ? String(h.status) : null,
        'syntax: ' + ((h && h.syntax_backend) || 'none'),
        refCount === null || refCount === undefined ? null : refCount + ' references'
      ].filter(Boolean).join(' · '));
      hideError();
    }, function (err) {
      setPill('bad', 'API unreachable');
      showError((err && err.message) || 'health check failed');
    });
  }

  function loadReferences() {
    if (state.mock) return;
    request('/api/references').then(function (list) {
      if (!Array.isArray(list)) return;
      var select = $('reference-select');
      list.forEach(function (r) {
        if (!r || !r.name) return;
        var opt = el('option', null,
          r.name + (r.genre ? ' — ' + r.genre : '') +
          (num(r.n_documents) !== null ? ' (' + r.n_documents + ' docs)' : ''));
        opt.value = r.name;
        if (r.source) opt.title = String(r.source);
        select.appendChild(opt);
      });
      var wanted = PARAMS.get('reference');
      if (wanted && list.some(function (r) { return r && r.name === wanted; })) {
        select.value = wanted;
        state.reference = wanted;
      }
    }, function () { /* the selector simply stays at "none" */ });
  }

  /* ── theme ─────────────────────────────────────────────────────────────── */

  var THEMES = ['auto', 'light', 'dark'];

  function applyTheme(theme) {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    $('theme-label').textContent = theme.charAt(0).toUpperCase() + theme.slice(1);
    try { localStorage.setItem('humanizer.theme', theme); } catch (e) { /* private mode */ }
  }

  /* ── wiring ────────────────────────────────────────────────────────────── */

  function init() {
    editor = $('editor');
    statusLine = $('status-line');

    errorBox = el('section', 'panel');
    errorBox.hidden = true;
    var rail = document.querySelector('.rail');
    rail.insertBefore(errorBox, rail.firstChild);

    var savedTheme = 'auto';
    try { savedTheme = localStorage.getItem('humanizer.theme') || 'auto'; } catch (e) { /* ignore */ }
    if (THEMES.indexOf(savedTheme) < 0) savedTheme = 'auto';
    applyTheme(savedTheme);
    $('theme-btn').addEventListener('click', function () {
      var current = document.documentElement.getAttribute('data-theme') || 'auto';
      applyTheme(THEMES[(THEMES.indexOf(current) + 1) % THEMES.length]);
    });

    $('quality-info').addEventListener('click', function () {
      var panel = $('quality-formula');
      panel.hidden = !panel.hidden;
      this.setAttribute('aria-expanded', panel.hidden ? 'false' : 'true');
    });

    editor.addEventListener('input', function () {
      state.text = getText();
      renderCounts();
      setStatus('stale', 'Edited — re-analyzing in ' + (DEBOUNCE_MS / 1000).toFixed(1) + 's');
      scheduleAnalyze();
    });

    editor.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); analyze(); }
    });

    editor.addEventListener('click', function (e) {
      var node = e.target;
      while (node && node !== editor && !(node.classList && node.classList.contains('sent'))) node = node.parentNode;
      if (node && node !== editor) selectSentence(Number(node.getAttribute('data-index')), false);
    });

    /* caret movement is the keyboard route to the inspector inside a
       contenteditable, where tabindex on every span would be hostile */
    document.addEventListener('selectionchange', function () {
      if (document.activeElement !== editor) return;
      var sel = window.getSelection();
      if (!sel || !sel.rangeCount) return;
      var node = sel.getRangeAt(0).startContainer;
      while (node && node !== editor && !(node.classList && node.classList.contains('sent'))) node = node.parentNode;
      if (node && node !== editor) {
        var idx = Number(node.getAttribute('data-index'));
        if (idx !== state.selected) selectSentence(idx, false);
      }
    });

    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && document.activeElement !== editor) {
        e.preventDefault(); analyze();
      }
    });

    $('analyze-btn').addEventListener('click', function () { analyze(); });

    $('reference-select').addEventListener('change', function () {
      state.reference = this.value;
      if (state.text.trim()) analyze();
    });

    $('sample-select').addEventListener('change', function () {
      var key = this.value;
      if (!key) return;
      setText(key === 'blank' ? '' : SAMPLES[key] || '');
      this.value = '';
      editor.focus();
      analyze();
    });

    $('mock-exit').addEventListener('click', function () {
      setMock(false);
      checkHealth();
      analyze();
    });

    setMock(state.mock);
    checkHealth();
    loadReferences();

    var wantSample = PARAMS.get('sample');
    if (wantSample === 'ai' || wantSample === 'human') setText(SAMPLES[wantSample]);
    else if (state.mock) setText(SAMPLES.ai);
    else setText('');

    if (state.text.trim()) analyze();
    else setStatus('idle', 'Idle. Paste an essay or load a sample.');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
