/* ══════════════════════════════════════════════════════════════════════════
   humanizer — Stage 0 front end.

   No build step, no bundler, no runtime dependency. One classic script so the
   page also works from file:// (ES-module imports are blocked there).

   Non-negotiable design constraints:
     1. no textarea — a real document model with one span per sentence;
     2. two dials, risk and quality, never one;
     3. name the measurement every time — value, human band, consequence, and
        a plain-language explanation one click away;
     4. green/red are reserved for the future rewrite diff; risk gets its own
        warm sand -> clay ramp; uncertainty gets its own visual state;
     5. the disclaimers stay: the score comes from a published pretrained
        checkpoint and not from a commercial detector, the quality dial is a
        placeholder formula, and mock mode is labelled everywhere it appears;
     6. this project writes no detection arithmetic. When the published model
        cannot run there is NO number, not a substitute one. The style
        signals are explanation and are never summed into a score.
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var PARAMS = new URLSearchParams(location.search);
  var API_BASE = PARAMS.get('api') || '';
  var DEBOUNCE_MS = 900;
  var ARC = 163.363;            /* pi * r, r = 52 — matches the SVG path */

  /* ── small utilities ───────────────────────────────────────────────────── */

  function $(id) { return document.getElementById(id); }

  function num(v) { return typeof v === 'number' && isFinite(v) ? v : null; }

  /* the humanize endpoint returns sentence_index as a string, so anything that
     reads a number off an API payload goes through this instead */
  function loose(v) {
    if (typeof v === 'number') return isFinite(v) ? v : null;
    if (typeof v === 'string' && v.trim() !== '') {
      var n = Number(v);
      return isFinite(n) ? n : null;
    }
    return null;
  }

  /* every numeric in the contract may be null (NaN server-side) */
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

  function plural(n, one, many) { return n === 1 ? one : (many || one + 's'); }

  var reduceMotion = window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)')
    : { matches: false };

  /* the animation layer in anim.js listens for these; nothing depends on it */
  function emit(name, detail) {
    try { document.dispatchEvent(new CustomEvent(name, { detail: detail || null })); }
    catch (e) { /* older engines: animation is optional anyway */ }
  }

  /* ── the measurement table ───────────────────────────────────────────────
     Bands are the ones the Python extractor checks, plus the published
     academic-register targets. Nothing is shown without its band, its source
     and a plain-language reading of what the number means. */

  var FEATURES = [
    {
      key: 'sent_len_cv', label: 'sentence-length variation (CV)', band: [0.42, 0.60], digits: 2,
      source: 'measured across five pre-2023 corpora',
      note: 'a band, not a maximum. A measured GPT baseline sits at 0.50, inside it, and 0.85 is as anomalous as 0.30',
      plain: 'How much your sentence lengths vary: the spread of your sentence lengths divided by their average. ' +
             'Human academic writing lands between 0.42 and 0.60. Below that, every sentence is roughly the same ' +
             'length, which is the most reliable tell there is. Above it you have overcorrected, and 0.85 is as ' +
             'strange to a detector as 0.30.'
    },
    {
      key: 'sent_lag1_autocorr', label: 'lag-1 sentence autocorrelation', band: [-0.10, 0.25], digits: 2,
      source: 'the typical human value is 0.01 to 0.10; the band shown is the tolerated range',
      note: 'humans do not alternate long and short; a strong negative value is the signature of a naive burstiness rule',
      plain: 'Whether a long sentence tends to be followed by another long one. Humans sit near zero: no pattern ' +
             'either way. A clearly negative number means you are alternating long, short, long, short. That is ' +
             'what a naive "add burstiness" instruction produces, and it is easy to spot.'
    },
    {
      key: 'sent_short_share', label: 'sentences under 10 words', band: [0.09, 0.14], pct: true,
      source: '',
      note: 'widen the tails, not the middle',
      plain: 'The share of your sentences that are under ten words. Human academic prose keeps 9 to 14 per cent ' +
             'this short. A few genuinely short sentences buy you more variation than nudging every sentence a ' +
             'word or two.'
    },
    {
      key: 'sent_long_share', label: 'sentences over 30 words', band: [0.18, 0.28], pct: true,
      source: '',
      plain: 'The share of your sentences that run past thirty words. Humans sit between 18 and 28 per cent. ' +
             'If your variation is low, the usual cause is that nothing here is long, not that nothing is short.'
    },
    {
      key: 'para_words_cv', label: 'paragraph-length variation (CV)', band: [0.42, 0.71], digits: 2,
      source: '',
      note: 'higher than the sentence CV in every genre measured',
      plain: 'The same spread measure, applied to paragraphs instead of sentences. Human paragraphs vary more ' +
             'than human sentences do, in every genre anyone has measured. Evenly sized blocks of text read as ' +
             'generated even when the sentences inside them do not.'
    },
    {
      key: 'ai_vocab_weighted_per_1k', label: 'AI vocabulary, weighted /1k', band: [0, 8], digits: 1,
      source: 'anything past the top of the band is flagged',
      note: 'weights are published excess-frequency ratios (delve 25x, tapestry 18x in phrase, underscores 13.8x)',
      plain: 'Words and phrases that turn up far more often in model output than in human writing, weighted by ' +
             'how lopsided the gap is: "delve" about 25 times more often, "tapestry" about 18, "underscores" ' +
             'about 13.8. This is the strongest lexical signal in the catalogue and it survives paraphrasing. ' +
             'Removing it costs you nothing at all.'
    },
    {
      key: 'formal_connective_per_1k', label: 'formal connectives /1k', band: [1, 8], digits: 1,
      source: 'better essays rise from 0.84 to 3.75 across PERSUADE scores 1 to 6',
      note: 'stripping these is the one edit that cuts detection risk and raises the grade',
      plain: '"Moreover", "Furthermore", "Additionally", "Consequently". Models open sentences with these far ' +
             'more often than people do. This is the single best edit available to you: it lowers detection ' +
             'risk and raises your grade at the same time, because top-band coherence is the kind that never ' +
             'announces itself.'
    },
    {
      key: 'nominalization_per_1k', label: 'nominalizations /1k', band: [55, 80], digits: 1,
      source: 'academic sub-registers 61.0 to 72.1 (Biber & Gray)',
      note: 'better essays are MORE nominalized, so do not cut this to sound human',
      plain: 'Verbs and adjectives turned into nouns: "investigate" becomes "investigation". Academic writing ' +
             'runs 61 to 72 per thousand words. Counter-intuitively, stronger essays are more nominalized, not ' +
             'less, so do not strip these in an attempt to sound human. You would trade a small risk gain for ' +
             'a real grade loss.'
    },
    {
      key: 'passive_per_1k', label: 'passives /1k', band: [12, 25], digits: 1,
      source: 'academic prose averages 18.5, about a quarter of all finite verbs (Biber et al.)',
      plain: 'Passive constructions per thousand words. Academic prose sits around 18.5, roughly a quarter of ' +
             'all finite verbs. The familiar advice to avoid the passive comes from journalism style guides, ' +
             'not from anyone who counted academic writing.'
    },
    {
      key: 'comma_per_1k', label: 'commas /1k', band: [57, 65], digits: 1,
      source: 'this project\u2019s own measurement',
      note: 'punctuation is the most author-stable family measured (commas ICC 0.44)',
      plain: 'Commas per thousand words. Punctuation habits are the most stable thing about an individual ' +
             'writer, which makes them a strong identity signal, and they barely move when text is ' +
             'paraphrased, so they are hard to launder.'
    },
    {
      key: 'contraction_per_1k', label: 'contractions /1k', band: [0, 1.4], digits: 2,
      source: 'research articles sit at the very bottom of this band',
      note: 'contractions help evasion and cost grade, so they are a never in body edit for academic targets',
      plain: '"Don\u2019t", "it\u2019s", "we\u2019ve". Research articles keep these under 1.4 per thousand words. ' +
             'Adding contractions genuinely does lower detection risk, and it also lowers your grade under ' +
             'academic genre conventions. That makes it a "never in the body" edit for academic work.'
    },
    {
      key: 'mtld', label: 'lexical diversity (MTLD)', band: null, digits: 1,
      source: 'no published academic band, so this is reported for context only',
      note: 'academic type-token ratio is LOWER than fiction and news; raising it is not an improvement',
      plain: 'How varied your vocabulary is, measured in a way that does not punish long documents. There is no ' +
             'published academic band, so this is here for context only. Academic writing is less lexically ' +
             'varied than fiction or journalism, on purpose: terms of art get repeated rather than elegantly ' +
             'varied. A higher number here is not automatically better writing.'
    },
    {
      key: 'n_words', label: 'words', band: null, digits: 0,
      source: 'shape features are noisy and detectors hedge below 300 words',
      plain: 'How long your draft is. Below roughly 300 words every shape measure above is noisy, and detectors ' +
             'themselves start hedging, so treat the whole page as provisional until you are past that.'
    },
    {
      key: 'n_sentences', label: 'sentences', band: null, digits: 0,
      source: 'as split by the service',
      plain: 'How many sentences the service found. The splitter knows about "Dr.", "e.g." and similar, but no ' +
             'splitter is perfect, so an odd count usually means an unusual abbreviation or a stray full stop.'
    }
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

  /* server bands win; one is computed only when the server did not send it */
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

  /* Rule 3: never a bare label. This is the one-line form used in the status
     bar and the sentence inspector. */
  function namedMeasurement(spec, value, serverBands) {
    var band = formatBand(spec);
    var s = spec.label + ' ' + formatFeature(spec, value);
    if (band) s += ', where human academic prose runs ' + band;
    var st = bandState(spec, value, serverBands);
    if (st === 'low') s += ', you are below it';
    else if (st === 'high') s += ', you are above it';
    else if (st === 'in band') s += ', you are inside it';
    return s;
  }

  /* ── grade-cost classification ───────────────────────────────────────────
     The API sends grade_cost as free text from the conflict matrix ("none,
     slightly positive", "negative, it raises the grade", "medium if added",
     "high if reduced further"). The whole point of the product is that some
     edits pay twice and some trade, so the string is classified rather than
     printed bare — and the raw string is still shown beside it. */

  function classifyGradeCost(raw) {
    var s = String(raw === undefined || raw === null ? '' : raw).toLowerCase().trim();
    if (!s) return { kind: 'neutral', label: 'grade cost not stated', raw: '' };
    if (s.indexOf('negative') === 0 || s.indexOf('raises the grade') >= 0) {
      return { kind: 'gain', label: 'fixing this also raises the grade', raw: raw };
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

  /* ── samples ─────────────────────────────────────────────────────────────
     Two drafts chosen so the difference is visible in the dials immediately:
     the first is uniform, connective-heavy and vocabulary-flagged; the second
     varies its sentence and paragraph lengths, carries dated and numbered
     specifics, and states a counterargument. */

  var SAMPLES = {
    ai: [
      'The relationship between urban green space and public health has become a pivotal area of academic inquiry. Researchers across a range of disciplines have begun to delve into the mechanisms that connect vegetation cover to wellbeing. Moreover, this work underscores a broader shift in how modern cities are planned, managed and understood. It is important to note that this shift is multifaceted, ongoing and highly context dependent.',
      'Furthermore, a comprehensive review of the existing literature reveals several recurring themes worth considering. First, access to green space is consistently associated with improved mental health indicators. Second, the observed effect appears robust across a wide range of demographic groups. Third, the underlying causal mechanisms remain the subject of considerable scholarly debate. These findings collectively demonstrate the significance of the topic for policy makers.',
      'Additionally, it is worth noting that methodological challenges persist throughout this body of research. Many studies rely on cross sectional designs that cannot establish causal direction with confidence. Consequently, the strength of the evidence base remains somewhat limited in scope. Nevertheless, the overall pattern of reported results is compelling and deserves attention. Therefore, further investigation is not merely desirable but genuinely necessary.',
      'The implications of these findings extend well beyond the boundaries of a single discipline. Planners, clinicians and community organisations all have a crucial stake in the eventual outcome. Similarly, the instruments used to measure exposure to green space deserve renewed scrutiny. Overall, the field now stands at an important juncture in its development.',
      'In conclusion, urban green space represents an intricate intersection of environmental, social and psychological factors. Future research should explore the causal pathways at play in far greater depth. Ultimately, such work will pave the way for more effective and more equitable urban policy. This is not simply an academic exercise, but a pressing societal concern.'
    ].join('\n\n'),

    human: [
      'Between 1993 and 2011 Leipzig lost close to a fifth of its population, and the vacant lots that demolition left behind were converted, piecemeal and without any coordinating plan, into small neighbourhood parks. The sequence of demolitions was determined by structural condition and by the availability of federal reconstruction money, not by the affluence of the surrounding blocks, which makes the allocation of new green space close to exogenous. Kabisch and Haase treated that sequence as a natural experiment, and the municipal documentation was complete enough to support the design. Their analysis covers 4,100 households across three survey waves. Attrition ran to eleven per cent, which is high, though not unusual for a postal instrument of this length.',
      'The measured effect is small, and the authors are careful not to oversell it. Residents within 300 metres of a converted lot reported better general health on the SF-12 than residents at 800 metres, by roughly 0.14 standard deviations, after adjustment for income, age and prior diagnosis. The gap widened slightly across the study period, though not by enough to carry the argument. At the final wave the confidence interval includes zero, and the authors say so in the text.',
      'Two objections deserve more attention than the published discussion gives them. The first is spatial: demolition was concentrated in the northeast, and the northeast differed from the remaining districts in ways that were measured only imperfectly at baseline, among them the density of ground-floor commercial space and the proportion of households in receipt of housing assistance. In contrast, the second objection is procedural, and it is harder to dismiss. Self-reported health was collected by postal questionnaire, and the response rate near the converted lots ran eleven points above the rate elsewhere. If residents in better health were also more willing to return a survey about their own neighbourhood park, then the estimated benefit is inflated by an amount the data cannot recover. The authors acknowledge the problem in a footnote and make no attempt at correction.',
      'None of this makes the study worthless. It makes the interpretation conditional.',
      'On balance the Leipzig evidence is suggestive rather than decisive, and the reasons are worth stating precisely. The identification strategy is stronger than anything that preceded it in the German literature. The estimated benefit is small enough to be produced by residual confounding alone, and the direction of the likely selection bias runs toward the reported finding. A replication in a city where demolition was dispersed rather than clustered would settle most of the remaining doubt. Halle offers a plausible setting, since its own demolition programme was administered building by building between 1999 and 2009, and the case records survive, complete, in the municipal archive.'
    ].join('\n\n')
  };

  /* ── client-side lexicons, used for the per-sentence explanation and for
     mock mode. Weights mirror the Python ai_lexicon module. ─────────────── */

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
    var m = text.toLowerCase().match(/[a-z][a-z'\u2019-]*/g);
    return m || [];
  }

  /* ── document model ──────────────────────────────────────────────────────
     Paragraph and sentence ranges over the plain text. This is what replaces
     the textarea: the canvas is rendered from these ranges and every span in
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
        /* UAX#29 has no abbreviation list, so it breaks "Dr. Smith". Merge a
           segment ending in a known abbreviation into the one after it. */
        var merged = [];
        for (var k = 0; k < out.length; k++) {
          var cur = out[k];
          while (k + 1 < out.length && ABBREV.test(chunk.slice(cur.from, cur.to).trim())) {
            k++;
            cur = { from: cur.from, to: out[k].to };
          }
          merged.push(cur);
        }
        if (merged.length) return merged;
      } catch (e) { /* fall through to the regex splitter */ }
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

  /* The API returns sentence text, not offsets, so each sentence is located in
     the document text with a whitespace-tolerant forward scan. */
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

  /* ── risk arithmetic shared by the fallback dial and by mock mode ──────── */

  function outOfBand(value, lo, hi) {
    var n = num(value);
    if (n === null) return null;
    var half = Math.abs(hi - lo) / 2 || 1;
    if (n < lo) return clamp((lo - n) / (half * 2), 0, 1);
    if (n > hi) return clamp((n - hi) / (half * 2), 0, 1);
    return 0;
  }

  /* Used only when the API returns no detector. It is a distance from the
     human bands, not a detector score, and it is labelled as such. */
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

     Labelled a placeholder everywhere it appears. It exists because risk must
     never be shown without quality beside it, and the API returns no quality
     score.

       base            78        a mid-band essay
       register term   +/- 18    five features that RISE with essay grade:
                                 nominalization 55-80, passives 12-25, formal
                                 connectives 1-8, commas 57-65, paragraph CV
                                 0.42-0.71. Each worth +3.6 in band and down to
                                 -3.6 outside, scaled by how far outside.
       findings term   0 to -30  severity weight (high 6, medium 3, low 1)
                                 times what the grade_cost string says:
                                   x1.00 fixing costs grade, so the current
                                         state is a genuine defect
                                   x0.35 fixing is free — a detector problem,
                                         not a quality problem
                                   x0.15 fixing raises the grade — an
                                         opportunity, barely a defect
       short document  pulled halfway to 70 under 300 words.

     Deliberately NOT a rubric score.
     ══════════════════════════════════════════════════════════════════════ */

  var QUALITY_REGISTER = [
    { key: 'nominalization_per_1k', band: [55, 80] },
    { key: 'passive_per_1k', band: [12, 25] },
    { key: 'formal_connective_per_1k', band: [1, 8] },
    { key: 'comma_per_1k', band: [57, 65] },
    { key: 'para_words_cv', band: [0.42, 0.71] }
  ];

  function padRight(s, n) {
    s = String(s);
    while (s.length < n) s += ' ';
    return s;
  }
  function padLeft(s, n) {
    s = String(s);
    while (s.length < n) s = ' ' + s;
    return s;
  }

  function qualityScore(features, findings) {
    var f = features || {};
    var lines = [];
    var score = 78;
    var W = 28;
    lines.push(padRight('base', W) + padLeft('78.0', 7));

    var registerDelta = 0, counted = 0;
    QUALITY_REGISTER.forEach(function (spec) {
      var d = outOfBand(f[spec.key], spec.band[0], spec.band[1]);
      if (d === null) return;
      counted++;
      registerDelta += d === 0 ? 3.6 : -3.6 * d;
    });
    score += registerDelta;
    lines.push(padRight('academic register (' + counted + '/5)', W) +
               padLeft((registerDelta >= 0 ? '+' : '') + registerDelta.toFixed(1), 7));

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
    lines.push(padRight('findings (' + (findings || []).length + ')', W) +
               padLeft('-' + penalty.toFixed(1), 7));

    var nWords = num(f.n_words);
    var lowConfidence = nWords !== null && nWords < 300;
    if (lowConfidence) {
      score = 70 + (score - 70) * 0.5;
      lines.push(padRight('under 300 words, halved to 70', W) + padLeft(score.toFixed(1), 7));
    }
    score = clamp(score, 0, 100);
    lines.push(new Array(W + 8).join('-'));
    lines.push(padRight('writing quality', W) + padLeft(score.toFixed(0), 7));

    return { score: score, lowConfidence: lowConfidence, breakdown: lines.join('\n') };
  }

  /* ── HTTP ────────────────────────────────────────────────────────────────
     Nothing here assumes a field exists. */

  function request(path, body) {
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var opts = { method: body ? 'POST' : 'GET', headers: { Accept: 'application/json' } };
    if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    if (controller) opts.signal = controller.signal;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 30000) : null;

    return fetch(API_BASE + path, opts).then(function (res) {
      if (timer) clearTimeout(timer);
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
      if (err && err.name === 'AbortError') throw new Error('the request timed out after 30 seconds');
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
        severity: SEVERITY_WEIGHT.hasOwnProperty(f.severity) ? f.severity : 'info',
        message: f.message,
        detail: typeof f.detail === 'string' ? f.detail : '',
        grade_cost: f.grade_cost
      };
    }) : [];
    findings.sort(function (a, b) {
      return (SEVERITY_ORDER[a.severity] === undefined ? 9 : SEVERITY_ORDER[a.severity]) -
             (SEVERITY_ORDER[b.severity] === undefined ? 9 : SEVERITY_ORDER[b.severity]);
    });
    return {
      features: r.features && typeof r.features === 'object' ? r.features : {},
      bands: r.bands && typeof r.bands === 'object' ? r.bands : {},
      findings: findings,
      reference: r.reference && typeof r.reference === 'object' ? r.reference : null,
      referenceName: typeof r.reference_name === 'string' ? r.reference_name : '',
      detectors: r.detectors && typeof r.detectors === 'object' ? r.detectors : {},
      deviations: r.deviations && typeof r.deviations === 'object' ? r.deviations : {},
      sentences: Array.isArray(r.sentences) ? r.sentences : [],
      advisory: r.sentence_risk_is_advisory !== false,
      /* the published model could not run: the reason, and no score anywhere */
      detectorError: typeof r.detector_error === 'string' && r.detector_error
        ? r.detector_error : null,
      /* explanation, never evidence: is_a_detector is false on the wire and
         there is deliberately no aggregate to read off these */
      styleSignals: r.ai_style_signals && typeof r.ai_style_signals === 'object'
        ? {
            signals: r.ai_style_signals.signals && typeof r.ai_style_signals.signals === 'object'
              ? r.ai_style_signals.signals : {},
            sources: r.ai_style_signals.sources && typeof r.ai_style_signals.sources === 'object'
              ? r.ai_style_signals.sources : {},
            isDetector: r.ai_style_signals.is_a_detector === true,
            note: typeof r.ai_style_signals.note === 'string' ? r.ai_style_signals.note : ''
          }
        : null,
      mock: r.mock === true
    };
  }

  /* ══════════════════════════════════════════════════════════════════════
     MOCK MODE (?mock=1, or the button on the error / offline state).

     A reduced in-browser reimplementation of the Stage 0 extractor so the
     whole interface can be exercised with the Python service down. It computes
     real numbers from the real text using the same feature definitions; it is
     abridged, it has no reference distribution and no trained detector, and
     the UI says so in a persistent banner.
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
  /* possessive "today's" is not a contraction; only unambiguous suffixes,
     plus pronoun + 's/'d, are counted */
  var CONTRACTION_RE = /\b(?:\w+['\u2019](?:re|ve|ll|m|t)|(?:it|he|she|that|there|here|what|who|where|when|how|let|one|nobody|somebody)['\u2019](?:s|d))\b/gi;

  function mockAnalyze(text) {
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

    /* findings mirror the Python report module: same codes, same grade_cost
       strings, abridged thresholds. */
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
      message: 'Found ' + opinion + ' bare opinion ' + plural(opinion, 'marker') + ' such as "I think".',
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

    /* per-sentence advisory risk: the document composite, nudged by the named
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
        length: s.length, risk: clamp(0.12 + 0.60 * base + local, 0.02, 0.98)
      };
    });

    return {
      features: features,
      bands: bands,
      findings: findings,
      /* no reference distribution exists in the browser, so none is invented */
      reference: null,
      reference_name: '',
      detectors: docRisk === null ? {} : {
        'mock-heuristic': {
          ai_probability: docRisk,
          label: docRisk > 0.65 ? 'ai' : docRisk < 0.35 ? 'human' : 'mixed',
          confidence: n < 300 ? 'uncertain' : 'moderate'
        }
      },
      deviations: {},
      sentences: outSentences,
      sentence_risk_is_advisory: true,
      mock: true
    };
  }

  /* ══════════════════════════════════════════════════════════════════════
     STATE
     ══════════════════════════════════════════════════════════════════════ */

  var state = {
    text: '',
    analyzedText: '',
    analysis: null,      /* normalized response */
    detectors: null,     /* POST /api/detect result */
    heatSource: 'advisory',  /* 'advisory' | 'perplexity' */
    humanize: null,          /* the last /api/humanize response */
    preHumanize: null,       /* the draft as it was before, for undo */
    humanizeAvailable: null, /* null = unknown, false = endpoint absent */
    located: [],         /* sentences with document offsets */
    selected: null,      /* sentence index */
    reference: '',
    referencesLoaded: false,
    mock: PARAMS.get('mock') === '1',
    status: 'idle',
    requestToken: 0,
    openMeasure: null    /* which measurement row has its plain reading open */
  };

  var editor, scroller, statusLine, errorBox;
  /* set while the canvas is being rebuilt, so the programmatic selection
     restore does not re-enter selectSentence through selectionchange */
  var rendering = false;

  /* ── editor plumbing ─────────────────────────────────────────────────────
     The document model is the plain text plus the sentence ranges. The DOM is
     a rendering of it, rebuilt only when an analysis lands, with the full
     selection (anchor AND focus, not just a collapsed caret) restored by
     character offset and the scroll position preserved. */

  function getText() {
    var t = editor.innerText || '';
    return t.replace(/\r\n?/g, '\n').replace(/\u00a0/g, ' ').replace(/\n{3,}/g, '\n\n');
  }

  /* offsets are measured over the concatenated text nodes of the editor, and
     read back the same way, so the round trip is self-consistent even though
     paragraph breaks are element boundaries rather than characters */
  function nodeOffset(container, offset) {
    var walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
    var acc = 0, node;
    if (container === editor) {
      /* selection anchored on the element itself: count the text in the
         children that precede the given child index */
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
    } catch (e) { /* selection restore is best-effort */ }
  }

  function riskTier(risk) {
    var r = num(risk);
    if (r === null) return { cls: 'risk-0', confidence: 'none', tier: null };
    var tier = Math.min(5, Math.floor(r * 6));
    /* uncertainty is its own state, never a mid-ramp fill */
    var lowConfidence = Math.abs(r - 0.5) < 0.15;
    return { cls: 'risk-' + tier, confidence: lowConfidence ? 'low' : 'high', tier: tier };
  }

  function renderCanvas() {
    var text = state.text;
    var located = state.located;
    var saved = captureSelection();
    var scroll = scroller.scrollTop;
    var wasFocused = document.activeElement === editor;

    rendering = true;
    clear(editor);

    paragraphRanges(text).forEach(function (p) {
      var para = el('p');
      var pos = p.from;
      located.forEach(function (s) {
        if (!s || s.from < p.from || s.from >= p.to) return;
        if (s.from > pos) para.appendChild(document.createTextNode(text.slice(pos, s.from)));
        var end = Math.min(s.to, p.to);
        var heat = heatValue(s);
        var perplexityMode = state.heatSource === 'perplexity' && state.overlay && state.overlay.ok;
        var tier = riskTier(heat);
        var span = el('span', 'sent ' + tier.cls, text.slice(s.from, end));
        span.setAttribute('data-index', String(s.index));
        span.setAttribute('data-confidence', tier.confidence);
        span.title = heat === null
          ? 'no score returned for this sentence'
          : perplexityMode
            ? 'GPT-2 perplexity signal ' + fmt(heat) + ' of 1.00, the classic signal, not evidence'
            : 'sentence risk ' + fmt(heat) + ' of 1.00. Click for the named patterns behind it';
        if (state.selected === s.index) span.setAttribute('aria-current', 'true');
        para.appendChild(span);
        pos = end;
      });
      if (pos < p.to) para.appendChild(document.createTextNode(text.slice(pos, p.to)));
      if (!para.childNodes.length) para.appendChild(el('br'));
      editor.appendChild(para);
    });
    if (!editor.childNodes.length) editor.appendChild(el('p', null, ''));

    scroller.scrollTop = scroll;
    if (wasFocused) restoreSelection(saved);
    /* selectionchange is async in some engines, so the guard is lifted on the
       next tick rather than immediately */
    setTimeout(function () { rendering = false; }, 0);

    $('empty-state').hidden = !!text.trim();
    emit('humanizer:canvas');
  }

  function setText(text) {
    state.text = text;
    state.located = [];
    state.selected = null;
    renderCanvas();
    renderCounts();
    /* setText fires no input event, so anything keyed off the text has to be
       refreshed here as well as in the input handler */
    refreshHumanizeButton();
  }

  function renderCounts() {
    var wordCount = words(state.text).length;
    var sentCount = state.located.length;
    var node = $('doc-counts');
    if (!state.text.trim()) { node.textContent = 'Nothing yet.'; return; }
    var parts = [wordCount + ' ' + plural(wordCount, 'word')];
    if (sentCount) parts.push(sentCount + ' ' + plural(sentCount, 'sentence'));
    parts.push(wordCount < 300
      ? 'under 300 words, so every shape measure below is noisy'
      : 'long enough for the shape measures to settle');
    node.textContent = parts.join(' \u00b7 ');
  }

  /* ══════════════════════════════════════════════════════════════════════
     THE HERO READOUT

     One number leads: how likely this reads as AI written. The quality score
     sits beside it as a secondary readout rather than a second dial, because
     risk shown alone pushes people toward worse writing, and that trade is
     the whole point of the product.
     ══════════════════════════════════════════════════════════════════════ */

  function pickDetector(detectors) {
    var best = null;
    Object.keys(detectors || {}).forEach(function (name) {
      var d = detectors[name];
      if (!d || typeof d !== 'object' || d.error) return;
      var p = num(d.ai_probability);
      if (p === null) return;
      if (!best || p > best.p) {
        best = {
          name: name, p: p, label: d.label, confidence: d.confidence,
          /* the checkpoint the service actually ran, so the caption names it
             instead of asserting something this file made up */
          model: typeof d.model === 'string' && d.model ? d.model : '',
          /* three states, not two: true, explicitly false, and "the service
             did not say", which must not be reported as either */
          published: d.is_published_detector === true ? true
                   : d.is_published_detector === false ? false : null
        };
      }
    });
    return best;
  }

  function bandOf(p) {
    if (p === null) return 'none';
    return p > 0.66 ? 'alert' : p > 0.33 ? 'caution' : 'calm';
  }

  var heroTween = null, heroSettle = null;

  function setHeroNumber(node, target, suffix) {
    if (heroTween) { cancelAnimationFrame(heroTween); heroTween = null; }
    if (heroSettle) { clearTimeout(heroSettle); heroSettle = null; }
    if (target === null) { node.textContent = 'n/a'; return; }
    var settled = Math.round(target) + (suffix || '');
    if (reduceMotion.matches) { node.textContent = settled; return; }

    var from = parseFloat(node.textContent);
    if (!isFinite(from)) from = 0;

    var hook = window.__hzAnim && window.__hzAnim.countTo;
    if (hook) {
      try {
        hook(node, from, target, suffix || '');
        heroSettle = setTimeout(function () { node.textContent = settled; }, 500);
        return;
      } catch (e) { /* fall through */ }
    }
    var t0 = performance.now();
    var step = function (now) {
      var k = clamp((now - t0) / 400, 0, 1);
      var eased = 1 - Math.pow(1 - k, 3);
      node.textContent = Math.round(from + (target - from) * eased) + (suffix || '');
      if (k < 1) heroTween = requestAnimationFrame(step); else heroTween = null;
    };
    heroTween = requestAnimationFrame(step);
    heroSettle = setTimeout(function () {
      if (heroTween) { cancelAnimationFrame(heroTween); heroTween = null; }
      node.textContent = settled;
    }, 460);
  }

  function renderHero() {
    var a = state.analysis;
    var value = $('hero-value');
    var engine = $('hero-engine');
    var qual = $('quality-value');

    if (!a) {
      setHeroNumber(value, null);
      value.setAttribute('data-band', 'none');
      value.removeAttribute('aria-label');
      engine.textContent = 'No measurement yet.';
      qual.textContent = 'n/a';
      $('quality-breakdown').textContent = '';
      renderReferenceLine(null);
      return { risk: null, quality: null };
    }

    var det = pickDetector(a.detectors);
    /* Mock mode computes its own labelled number. The live service does not:
       if no published detector answered there is no score, and the old
       fall back to a locally computed composite would be exactly the invented
       arithmetic this project removed. */
    var p = det ? det.p : (state.mock ? compositeRisk(a.features) : null);

    setHeroNumber(value, p === null ? null : p * 100, '%');
    value.setAttribute('data-band', bandOf(p));
    value.setAttribute('aria-label', p === null
      ? 'No AI likelihood could be measured'
      : Math.round(p * 100) + ' per cent likely to read as AI written');

    /* name the engine, and say what it is not */
    if (state.mock) {
      engine.textContent = 'Engine: a reduced in browser reimplementation. Mock mode, not the service.';
    } else if (det) {
      engine.textContent = 'Engine: ' + (det.model || det.name) +
        (det.published === true ? ', a published pretrained checkpoint the service runs'
         : det.published === false ? ', whose scoring was written in this project and fitted to nothing'
         : '') +
        '. Not GPTZero, Turnitin or Pangram, and not calibrated against them.' +
        (det.label ? ' It calls this draft "' + det.label + '".' : '');
    } else if (a.detectorError) {
      engine.textContent = 'No score. The published detector could not run: ' + a.detectorError +
        ' This project writes no detection arithmetic of its own, so there is no substitute ' +
        'number to show. Everything else on this page still measured.';
    } else {
      engine.textContent = 'No detector answered this draft, so there is no likelihood to report. ' +
        'The measurements and the style signals below still hold.';
    }

    var q = qualityScore(a.features, a.findings);
    qual.textContent = Math.round(q.score);
    $('quality-breakdown').textContent = q.breakdown;

    renderReferenceLine(a);
    return { risk: p === null ? null : p * 100, quality: q.score };
  }

  /* the service reports reference_name as whatever it was given — sometimes a
     bare name, sometimes the path of the JSON it was started with */
  function prettyReference(name) {
    if (!name) return '';
    return String(name).replace(/^.*\//, '').replace(/\.json$/i, '');
  }

  function renderReferenceLine(a) {
    var line = $('reference-line');
    var text;
    if (state.mock) {
      text = 'Mock mode holds no reference corpus, so there is no distance to report. ' +
             'Only the bands below are in play. ';
    } else if (a && a.reference) {
      var used = prettyReference(a.referenceName) || state.reference || 'the service default';
      line.title = a.referenceName ? 'reference_name: ' + a.referenceName : '';
      text = 'Measured against "' + used + '": Mahalanobis distance ' + fmt(a.reference.distance, 1) +
        ', which puts this draft at the ' + fmt(a.reference.percentile, 0) +
        'th percentile of that human corpus, meaning ' + fmt(a.reference.percentile, 0) +
        '% of the human documents in it look less unusual than yours. ' +
        (num(a.reference.n_features) === null ? '' :
          'Computed over ' + a.reference.n_features + ' features at ' +
          Math.round((num(a.reference.coverage) || 0) * 100) + '% coverage. ');
    } else if (a) {
      text = state.reference
        ? 'Reference "' + state.reference + '" was selected, but the service returned no distance. '
        : 'No reference distribution came back with this analysis, so the bands below are all there is. ';
    } else {
      text = 'No analysis yet. ';
    }
    /* the "?" button is the last child and must survive the update */
    if (line.firstChild && line.firstChild.nodeType === 3) line.firstChild.nodeValue = text;
    else line.insertBefore(document.createTextNode(text), line.firstChild);
  }

  /* ══════════════════════════════════════════════════════════════════════
     DETECTOR ENSEMBLE — POST /api/detect

     The dial shows one number. This panel shows every detector the service
     ran, including the ones that failed to load and the one that has been
     measured and does not work on this genre. The service's own disclaimer is
     printed verbatim above the list, because it is more honest than anything
     this front end could write.
     ══════════════════════════════════════════════════════════════════════ */

  /* What the service reports about each published checkpoint, restated
     plainly. Every figure here is the service’s own measurement on its
     28-paragraph set, and the false positive count sits beside the separation
     score in every single row on purpose: ranking well and accusing people are
     different things, and a checkpoint was rejected from this project for
     scoring 0.959 pairwise while calling 13 of 14 real human academic
     paragraphs AI. */
  var DETECTOR_NOTES = {
    modern: 'desklib/ai-text-detector-v1.01, an open fine-tuned DeBERTa-v3-large with 435M ' +
            'parameters, trained on the RAID corpus, which it led. Same architectural class ' +
            'GPTZero uses today, and that is the whole of the resemblance. On this repo’s own ' +
            '28 paragraph set: 1.000 pairwise separation, 1 of 14 genuine human academic ' +
            'paragraphs called AI.',
    fakespot: 'A published checkpoint, same wrapper as the default with different weights. ' +
              '1.000 pairwise separation here, but 3 of 14 genuine human academic paragraphs ' +
              'called AI, three times the default’s false positive rate.',
    academic: 'roberta-academic-detector, a published checkpoint aimed at academic prose. ' +
              '0.980 pairwise separation here with 1 of 14 human paragraphs called AI.',
    fast: 'e5-small-lora, 33M parameters, published. It is the fastest engine here and the ' +
          'least safe to act on: 0.944 pairwise separation but 8 of 14 genuine human academic ' +
          'paragraphs called AI. Ranking well and accusing people are different things.',
    radar: 'The only engine here trained adversarially against a paraphraser, which is the ' +
           'attack this product is closest to. 0.867 pairwise separation with 1 of 14 human ' +
           'paragraphs called AI, but it missed half the AI paragraphs.',
    perplexity: 'GPTZero’s ORIGINAL January-2023 published method: per-sentence GPT-2 perplexity ' +
                'plus burstiness. GPT-2 is published; the mapping from its perplexities to a ' +
                'probability was written in this project and fitted to nothing, so it reports ' +
                'is_published_detector false and is excluded from every default. 0.740 pairwise ' +
                'here with 13 of 14 genuine human paragraphs called AI. Do not act on this number.',
    classifier: 'OpenAI’s 2019 RoBERTa GPT-2 output detector, published but measured below ' +
                'chance here at 0.444 pairwise. It was trained to recognise GPT-2, a model ' +
                'generation that no longer resembles what students use, and OpenAI withdrew its ' +
                'own detector in 2023 for low accuracy.',
    ensemble: 'A combination of the members above. It inherits every weakness they have; ' +
              'combining uncalibrated scores does not calibrate them.'
  };

  /* members that must never be presented as authoritative, each for its own
     measured reason */
  var UNRELIABLE = {
    perplexity: 'Not a published detector. The probability mapping was written in this project ' +
                'and fitted to nothing, and it called 13 of 14 genuine human academic paragraphs ' +
                'AI. Treat this number as decoration, not evidence.',
    fast: 'It called 8 of 14 genuine human academic paragraphs AI. A false positive here is a ' +
          'student accused of cheating, so this row is reported and not used.',
    classifier: 'Measured at 0.444 pairwise separation on this repo’s own corpus, where 0.5 is ' +
                'a coin flip. That is below chance. Treat this number as decoration.'
  };

  /* ── the per-sentence overlay ────────────────────────────────────────────
     Only the perplexity detector returns real per-sentence values; the other
     two return an empty array. The detector agent routed segmentation through
     the same Document.parse the analyze endpoint uses, so index i lines up
     with sentence i — but an overlay that silently mis-highlights is worse
     than no overlay, so the alignment is checked rather than trusted:
       · the score array must be the same length as the analyzer's sentences,
       · and the detector's own sentence texts must match the analyzer's.
     If either check fails the canvas falls back to advisory risk and says so. */

  function perplexityDetector() {
    var d = state.detectors;
    if (!d || d.error || !d.detectors) return null;
    for (var i = 0; i < d.detectors.length; i++) {
      if (d.detectors[i] && d.detectors[i].name === 'perplexity') return d.detectors[i];
    }
    return null;
  }

  function normText(t) { return String(t || '').trim().replace(/\s+/g, ' '); }

  /* → { ok, scores, reason } */
  function perplexityOverlay() {
    var a = state.analysis;
    if (!a) return { ok: false, reason: 'nothing has been analyzed yet' };
    if (state.mock) return { ok: false, reason: 'mock mode does not run the detector ensemble' };

    var d = state.detectors;
    if (!d) return { ok: false, reason: 'the detector ensemble has not answered yet' };
    if (d.error) return { ok: false, reason: 'the detector request failed' };

    var det = perplexityDetector();
    if (!det) return { ok: false, reason: 'the service ran no perplexity detector' };
    if (!det.available) return { ok: false, reason: 'the perplexity detector did not load' };

    var scores = det.sentence_scores;
    if (!Array.isArray(scores) || !scores.length) {
      return { ok: false, reason: 'the perplexity detector returned no per-sentence scores' };
    }

    var mine = a.sentences || [];
    if (scores.length !== mine.length) {
      return { ok: false, reason: 'the two endpoints disagree on the sentence count (' +
        scores.length + ' scores against ' + mine.length + ' sentences), so nothing is shaded' };
    }

    /* the texts must line up too, not just the counts */
    var theirs = d.sentences || [];
    if (theirs.length === mine.length) {
      for (var i = 0; i < mine.length; i++) {
        if (normText(theirs[i]) !== normText(mine[i] && mine[i].text)) {
          return { ok: false, reason: 'the two endpoints segmented sentence ' + (i + 1) +
            ' differently, so nothing is shaded' };
        }
      }
    }
    return { ok: true, scores: scores, detector: det };
  }

  /* the score that actually shades a sentence, honouring the picker */
  function heatValue(sentence) {
    if (state.heatSource === 'perplexity') {
      var ov = state.overlay;
      if (ov && ov.ok) {
        var v = num(ov.scores[sentence.index]);
        if (v !== null) return v;
      }
      return null;
    }
    return num(sentence.risk);
  }

  function refreshOverlay() {
    state.overlay = perplexityOverlay();
    var note = $('heat-note');
    var pbtn = $('heat-perplexity');
    var legend = $('legend-label');

    var have = state.overlay && state.overlay.ok;
    pbtn.disabled = !have;
    pbtn.title = have
      ? 'Shade the canvas with the per-sentence GPT-2 perplexity signal'
      : 'Unavailable: ' + (state.overlay ? state.overlay.reason : 'no data');

    if (state.heatSource === 'perplexity' && !have) {
      /* never mis-highlight: drop back rather than draw the wrong thing */
      note.setAttribute('data-state', 'fallback');
      note.textContent = 'Falling back to advisory risk, because ' +
        (state.overlay ? state.overlay.reason : 'no per-sentence data') + '.';
      legend.textContent = 'Sentence risk';
    } else if (state.heatSource === 'perplexity') {
      note.setAttribute('data-state', 'ok');
      note.textContent = 'Per sentence GPT-2 perplexity, the classic signal, not evidence. ' +
        'It carries the same calibration problem as the document score above it.';
      legend.textContent = 'Perplexity signal';
    } else {
      note.setAttribute('data-state', 'ok');
      note.textContent = 'Advisory per-sentence risk from the analyzer.';
      legend.textContent = 'Sentence risk';
    }

    $('heat-advisory').setAttribute('aria-pressed', state.heatSource === 'advisory' ? 'true' : 'false');
    pbtn.setAttribute('aria-pressed', state.heatSource === 'perplexity' ? 'true' : 'false');
  }

  function setHeatSource(which) {
    state.heatSource = which;
    refreshOverlay();
    renderCanvas();
    renderFlagged();
    renderInspector();
  }

  /* ══════════════════════════════════════════════════════════════════════
     THE ENGINE PANEL, one engine only.

     The three row ensemble is gone on purpose. The heuristic and classifier
     rows told the reader nothing they could act on, and the perplexity row
     invited trust in a number the service itself says is barely better than a
     coin flip on academic prose. What survives is the engine behind the hero
     number, plus the perplexity overlay offer, which is the only part of the
     ensemble that produces genuine per sentence data.
     ══════════════════════════════════════════════════════════════════════ */

  function renderDetectors() {
    var host = $('engine-detail');
    if (!host) return;
    clear(host);

    var box = $('overlay-toggle');
    var ov = state.overlay;
    var can = !!(ov && ov.ok);
    if (box) {
      box.hidden = !can && state.heatSource !== 'perplexity';
      $('heat-perplexity').hidden = state.heatSource === 'perplexity';
      $('heat-advisory').hidden = state.heatSource !== 'perplexity';
    }

    if (state.mock) {
      host.appendChild(el('p', 'muted',
        'Mock mode runs no detectors. The number above is computed in your browser from the ' +
        'features on this page, using matching feature definitions and abridged thresholds.'));
      return;
    }

    var a = state.analysis;
    var det = a ? pickDetector(a.detectors) : null;
    if (!det) {
      if (a && a.detectorError) {
        var bad = el('article', 'detector');
        bad.setAttribute('data-available', 'false');
        var badTop = el('div', 'detector-top');
        badTop.appendChild(el('span', 'detector-name', 'no detector ran'));
        badTop.appendChild(el('span', 'detector-p', 'n/a'));
        bad.appendChild(badTop);
        bad.appendChild(el('code', 'detector-err', a.detectorError));
        bad.appendChild(el('p', 'detector-note',
          'The published checkpoint could not load, so there is no likelihood on this page. ' +
          'This project writes no detection arithmetic of its own, so the honest answer is no ' +
          'number rather than a substitute one. The measurements, the findings and the style ' +
          'signals below were all computed without it.'));
        host.appendChild(bad);
        return;
      }
      host.appendChild(el('p', 'muted', 'No analysis yet.'));
      return;
    }

    var card = el('article', 'detector');
    card.setAttribute('data-available', 'true');

    var top = el('div', 'detector-top');
    var nameWrap = el('span');
    nameWrap.appendChild(el('span', 'detector-name', det.name));
    if (det.label) {
      var tag = el('span', 'detector-tag', det.label);
      tag.setAttribute('data-label', String(det.label));
      nameWrap.appendChild(tag);
    }
    top.appendChild(nameWrap);
    top.appendChild(el('span', 'detector-p', Math.round(det.p * 100) + '%'));
    card.appendChild(top);

    var bar = el('div', 'detector-bar');
    var fill = el('i');
    fill.style.width = clamp(det.p * 100, 0, 100) + '%';
    fill.style.background = det.p > 0.66 ? 'var(--risk-alert-txt)'
      : det.p > 0.33 ? 'var(--risk-caution-txt)' : 'var(--risk-calm-txt)';
    bar.appendChild(fill);
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', det.name + ' reports p(AI) ' + fmt(det.p));
    card.appendChild(bar);

    card.appendChild(el('p', 'detector-note',
      DETECTOR_NOTES[det.name] ||
      'A published pretrained checkpoint the service runs. This project wrote none of the ' +
      'detection arithmetic behind it, and has not calibrated it against anything.'));

    if (UNRELIABLE[det.name]) {
      var unrel = el('p', 'detector-warn');
      unrel.appendChild(el('strong', null, 'Do not act on this number. '));
      unrel.appendChild(document.createTextNode(UNRELIABLE[det.name]));
      card.appendChild(unrel);
    }

    var warn = el('p', 'detector-warn');
    warn.appendChild(el('strong', null, 'Not a commercial detector. '));
    warn.appendChild(document.createTextNode(
      'It has never been calibrated against GPTZero, Turnitin or Pangram, so a low number here ' +
      'guarantees nothing. Use it to find patterns worth editing, not to predict a marker.'));
    card.appendChild(warn);

    host.appendChild(card);
  }

  function fetchDetectors(text, token) {
    if (state.mock) { state.detectors = null; refreshOverlay(); renderDetectors(); return; }
    request('/api/detect', { text: text }).then(function (raw) {
      if (token !== state.requestToken) return;
      var r = raw && typeof raw === 'object' ? raw : {};
      state.detectors = {
        detectors: Array.isArray(r.detectors) ? r.detectors : [],
        sentences: Array.isArray(r.sentences) ? r.sentences : [],
        disclaimer: typeof r.disclaimer === 'string' ? r.disclaimer : '',
        isGptzero: r.is_gptzero === true,
        backend_available: r.backend_available,
        error: null
      };
      afterDetectors();
    }, function (err) {
      if (token !== state.requestToken) return;
      state.detectors = { detectors: [], sentences: [], disclaimer: '',
                          error: (err && err.message) || 'request failed' };
      afterDetectors();
    });
  }

  /* the ensemble usually lands after the analysis, so the overlay is
     recomputed here and the canvas repainted only if it is actually in use */
  function afterDetectors() {
    refreshOverlay();
    renderDetectors();
    if (state.heatSource === 'perplexity') {
      renderCanvas();
      renderFlagged();
      renderInspector();
    }
  }

  function renderMeasurements() {
    var host = $('measurements');
    clear(host);
    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }

    FEATURES.forEach(function (spec) {
      var value = a.features[spec.key];
      var stateName = bandState(spec, value, a.bands);

      var row = el('button', 'measure');
      row.type = 'button';
      var plainId = 'plain-' + spec.key;
      var open = state.openMeasure === spec.key;
      row.setAttribute('aria-expanded', open ? 'true' : 'false');
      row.setAttribute('aria-controls', plainId);

      var top = el('div', 'measure-top');
      top.appendChild(el('div', 'measure-name', spec.label));
      var right = el('div', 'measure-right');
      right.appendChild(el('span', 'measure-value', formatFeature(spec, value)));
      if (spec.band) {
        var chip = el('span', 'state', stateName);
        chip.setAttribute('data-state', stateName);
        right.appendChild(chip);
      }
      top.appendChild(right);
      row.appendChild(top);

      var band = formatBand(spec);
      var bits = [];
      if (band) bits.push('human academic prose runs ' + band);
      if (spec.source) bits.push(spec.source);
      var caption = bits.join(', ');
      if (spec.note) caption += (caption ? '. ' : '') + spec.note;
      row.appendChild(el('div', 'measure-band', caption));

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

  /* ── findings, each with its grade cost. The differentiator. ─────────────── */

  function renderFindings() {
    var host = $('findings');
    clear(host);
    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }
    if (!a.findings.length) {
      host.appendChild(el('p', 'muted',
        'Nothing to fix. Every feature the service checks is sitting inside its human band.'));
      return;
    }

    a.findings.forEach(function (f) {
      var card = el('article', 'finding');
      card.setAttribute('data-severity', f.severity);

      var top = el('div', 'finding-top');
      top.appendChild(el('span', 'sev', f.severity));
      top.appendChild(el('span', 'finding-code', f.code || 'unnamed'));
      card.appendChild(top);

      card.appendChild(el('p', 'finding-msg', f.message));
      if (f.detail) card.appendChild(el('p', 'finding-detail', f.detail));

      var g = classifyGradeCost(f.grade_cost);
      var chip = el('span', 'grade');
      chip.setAttribute('data-kind', g.kind);
      chip.appendChild(el('b', null, g.label));
      if (g.raw) chip.appendChild(el('span', 'grade-raw', 'grade_cost: \u201c' + g.raw + '\u201d'));
      card.appendChild(chip);

      host.appendChild(card);
    });
  }

  /* ── AI style signals: explanation, and never a score ────────────────────
     The service sends these with `is_a_detector: false`, a research citation
     per signal and, on purpose, no aggregate. Summing them would rebuild the
     hand-written detector that was deleted, so nothing here totals anything.
     ─────────────────────────────────────────────────────────────────────── */

  var SIGNAL_LABELS = {
    ai_vocab_weighted: 'AI vocabulary, weighted by how lopsided each word is',
    cv_band_distance: 'Sentence length variety, distance outside the human band',
    formal_connective: 'Formal connectives such as "moreover" and "furthermore"',
    participial_tail: 'Participial tail clauses',
    tricolon: 'Three part lists',
    negative_parallel: 'The "not X, but Y" construction',
    para_opener_formal: 'Paragraphs opening on a formal connective',
    uniform_paragraphs: 'Paragraphs all the same length',
    no_contractions: 'No contractions anywhere in the draft'
  };

  var SIGNAL_FLOOR = 0.25;

  function signalLabel(key) {
    return SIGNAL_LABELS[key] ||
      String(key).replace(/_+/g, ' ').replace(/^./, function (c) { return c.toUpperCase(); });
  }

  function renderStyleSignals() {
    var host = $('style-signals');
    if (!host) return;
    clear(host);

    var a = state.analysis;
    if (!a) { host.appendChild(el('p', 'muted', 'No analysis yet.')); return; }
    if (state.mock) {
      host.appendChild(el('p', 'muted',
        'Mock mode does not compute the style signals. They come from the service.'));
      return;
    }
    var ss = a.styleSignals;
    if (!ss) {
      host.appendChild(el('p', 'muted',
        'This service returned no style signals with the analysis.'));
      return;
    }

    var rows = [];
    Object.keys(ss.signals).forEach(function (k) {
      var v = num(ss.signals[k]);
      if (v === null) return;
      rows.push({ key: k, v: clamp(v, 0, 1) });
    });
    rows.sort(function (x, y) { return y.v - x.v; });

    var present = rows.filter(function (r) { return r.v >= SIGNAL_FLOOR; });

    if (!rows.length) {
      host.appendChild(el('p', 'muted', 'No style signals came back for this draft.'));
      return;
    }
    if (!present.length) {
      host.appendChild(el('p', 'muted',
        'None of the ' + rows.length + ' known tells is strongly present in this draft. The ' +
        'strongest, ' + signalLabel(rows[0].key).toLowerCase() + ', reads ' + fmt(rows[0].v) +
        ' out of 1.00.'));
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
      bar.setAttribute('aria-label', signalLabel(r.key) + ', ' + fmt(r.v) + ' out of 1.00');
      row.appendChild(bar);

      var src = ss.sources[r.key];
      if (typeof src === 'string' && src) row.appendChild(el('p', 'signal-src', src));

      host.appendChild(row);
    });

    /* the service's own words, verbatim, because paraphrasing this one would
       be the exact mistake it exists to prevent */
    if (ss.note) host.appendChild(el('p', 'signal-note', ss.note));
    host.appendChild(el('p', 'signal-note',
      'There is no total on this panel on purpose. Each value is one feature scaled to a 0 to 1 ' +
      'reading aid, nothing here was trained, and nothing here votes on the percentage above. ' +
      (present.length ? 'Showing the ' + present.length + ' of ' + rows.length +
        ' signals reading 0.25 or higher.' : '')));

    if (ss.isDetector) {
      host.appendChild(el('p', 'signal-note',
        'The service marked these as a detector, which contradicts the contract this panel was ' +
        'built against. They are still shown as explanation only.'));
    }
  }

  /* ── highest-risk sentences: the keyboard route into the canvas ─────────── */

  function renderFlagged() {
    var host = $('flagged');
    clear(host);
    if (!state.located.length) {
      host.appendChild(el('p', 'muted', 'No analysis yet.'));
      return;
    }
    var scored = state.located.map(function (s) { return { s: s, v: heatValue(s) }; })
      .filter(function (x) { return x.v !== null; });
    if (!scored.length) {
      var why = state.analysis && state.analysis.detectorError;
      host.appendChild(el('p', 'muted', why
        ? 'No per sentence scores, because the published detector could not run: ' + why
        : 'The service returned no per sentence score for this draft.'));
      return;
    }
    var ranked = scored.sort(function (a, b) { return b.v - a.v; });
    /* only the ones actually worth looking at, and never more than five */
    var top = ranked.filter(function (x) { return x.v >= 0.5; }).slice(0, 5);
    if (!top.length) {
      var best = ranked[0];
      host.appendChild(el('p', 'muted',
        'Nothing here reads as AI written. The highest scoring sentence is ' + fmt(best.v) +
        ' out of 1.00, which is well inside the calm end of the range.'));
      return;
    }

    top.forEach(function (item, i) {
      var s = item.s;
      var btn = el('button', 'flag-btn');
      btn.type = 'button';
      btn.title = 'Jump to this sentence in your draft';
      btn.appendChild(el('span', 'flag-rank', String(i + 1)));
      btn.appendChild(el('span', 'flag-risk num', fmt(item.v)));
      btn.appendChild(el('span', 'flag-text', s.text));
      btn.addEventListener('click', function () { selectSentence(s.index, true); });
      host.appendChild(btn);
    });

    if (ranked.length > top.length) {
      host.appendChild(el('p', 'muted',
        'Showing the ' + top.length + ' highest of ' + ranked.length + ' sentences. ' +
        'The rest scored below 0.50.'));
    }
  }

  /* ── sentence inspector: why this sentence looks the way it does ────────── */

  function selectSentence(index, scrollTo) {
    state.selected = index;
    Array.prototype.forEach.call(editor.querySelectorAll('.sent'), function (node) {
      if (Number(node.getAttribute('data-index')) === index) {
        node.setAttribute('aria-current', 'true');
        if (scrollTo && node.scrollIntoView) {
          node.scrollIntoView({ block: 'center', behavior: reduceMotion.matches ? 'auto' : 'smooth' });
          node.classList.remove('flash');
          void node.offsetWidth;          /* restart the animation */
          node.classList.add('flash');
          setTimeout(function () { node.classList.remove('flash'); }, 1200);
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
      host.appendChild(el('p', 'muted',
        'Click any sentence in your draft, or just move the caret into one, and the named ' +
        'patterns behind its shading appear here.'));
      return;
    }

    host.appendChild(el('blockquote', 'inspect-quote', s.text));

    var rows = el('div', 'inspect-rows');
    var tier = riskTier(heatValue(s));
    function addRow(label, value) {
      var r = el('div', 'inspect-row');
      r.appendChild(el('span', null, label));
      r.appendChild(el('span', null, value));
      rows.appendChild(r);
    }
    addRow('advisory risk, 0 to 1', fmt(s.risk));
    var ov = state.overlay;
    if (ov && ov.ok) {
      addRow('perplexity signal, 0 to 1', fmt(ov.scores[s.index]));
      var det = ov.detector && ov.detector.raw;
      var sp = det && Array.isArray(det.sentence_perplexities) ? num(det.sentence_perplexities[s.index]) : null;
      if (sp !== null) addRow('raw GPT-2 perplexity', sp.toFixed(0));
    }
    addRow('how sure', tier.confidence === 'none' ? 'no score returned'
      : tier.confidence === 'low' ? 'not sure, within 0.15 of the 0.50 midpoint'
      : 'clear of the ambiguous middle');
    var len = num(s.length) === null ? words(s.text).length : s.length;
    addRow('length', len + ' ' + plural(len, 'word'));
    var docMean = state.analysis ? num(state.analysis.features.sent_len_mean) : null;
    if (docMean !== null) {
      addRow('vs your average sentence',
        (len - docMean >= 0 ? '+' : '') + fmt(len - docMean, 1) + ' words');
    }
    addRow('paragraph', num(s.paragraph_index) === null ? 'n/a' : String(s.paragraph_index + 1));
    host.appendChild(rows);

    host.appendChild(el('p', 'inspect-why-h', 'Named patterns in this sentence'));

    var reasons = [];
    var hits = aiVocabHits(s.text);
    hits.hits.slice(0, 5).forEach(function (h) {
      reasons.push('AI-vocabulary hit \u201c' + h.surface + '\u201d, which appears about ' + h.weight +
        ' times more often in model output than in human writing.');
    });
    var opener = connectiveOpener(s.text);
    if (opener) {
      reasons.push('Opens with the formal connective \u201c' + opener + '\u201d. Human academic paragraphs ' +
        'open this way at most 3 to 4 per cent of the time.');
    }
    if (docMean !== null && Math.abs(len - docMean) < 2.5) {
      var spec = FEATURE_BY_KEY.sent_len_cv;
      var cv = state.analysis ? state.analysis.features.sent_len_cv : null;
      reasons.push('Its length sits within 2.5 words of your document average, which is what drags ' +
        namedMeasurement(spec, cv, state.analysis ? state.analysis.bands : null) + '.');
    }
    if ((s.text.match(/,/g) || []).length >= 3 && /\band\b/i.test(s.text)) {
      reasons.push('Three-part coordinated list. Model output runs about twice the human rate on tricolons.');
    }
    if (/\d/.test(s.text)) {
      reasons.push('Contains a number. Concrete dated or numbered specifics are the best joint move ' +
        'available: they lower detection risk and raise the grade at the same time.');
    }
    if (!reasons.length) {
      reasons.push('No named pattern from the catalogue fires on this sentence. Its shading comes from the ' +
        'document-level score alone.');
    }

    var list = el('ul', 'inspect-list');
    reasons.forEach(function (r) { list.appendChild(el('li', null, r)); });
    host.appendChild(list);

    host.appendChild(el('p', 'inspect-caveat',
      'These are patterns present in the sentence, not an account of how the score was produced. ' +
      'GPTZero says the same of its own AI Vocabulary tool. The document verdict is not an aggregate ' +
      'of sentence scores, so these two numbers do not have to agree.'));
  }

  /* ══════════════════════════════════════════════════════════════════════
     THE STAGE CHECKLIST

     A 30 second rewrite must never look stuck, so the run takes over the
     middle of the page. The checklist sits in the SAME grid cell as the
     writing canvas: while it is up it is the active thing, not a spinner in
     a corner.

     Each stage is a ring that fills, driven by the streamed progress value
     from 0 to 1. Pending is an empty ring reading "waiting", active fills
     and is highlighted, done draws a check. State is carried by geometry, by
     a glyph and by words, so colour is never the only signal.

     Stage keys are the stable names the pipeline reports. Anything it sends
     that this build has never heard of is appended with a generated label
     rather than dropped.
     ══════════════════════════════════════════════════════════════════════ */

  var RING_R = 15;
  var RING_C = 2 * Math.PI * RING_R;

  /* The weights drive the ESTIMATED timeline only, used when the service
     cannot stream. They are a guess at where a 30 second run spends itself,
     and the panel says "estimated" out loud rather than passing a guess off
     as a measurement. */
  var STAGES = [
    { key: 'analyze',  label: 'Reading your draft',           weight: 1.5 },
    { key: 'plan',     label: 'Planning the rewrite',         weight: 3.0 },
    { key: 'generate', label: 'Writing candidate versions',   weight: 10.0 },
    { key: 'scrub',    label: 'Removing AI vocabulary',       weight: 2.5 },
    { key: 'score',    label: 'Scoring each version',         weight: 5.0 },
    { key: 'verify',   label: 'Checking your facts survived', weight: 3.5 },
    { key: 'select',   label: 'Choosing the best one',        weight: 1.5 },
    { key: 'finalize', label: 'Polishing the final draft',    weight: 3.0 }
  ];

  var RULE_STAGE = [
    { key: 'rewrite', label: 'Rewriting with the rule based engine', weight: 1.0 }
  ];

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
    open: false,
    rows: {},
    order: [],
    weights: {},
    mode: 'unknown',     /* unknown | estimated | live | rule */
    started: 0,
    ticker: null,
    estimator: null,
    armTimer: null,
    said: '',
    warned: false,
    wasEditable: 'true'
  };

  function buildStageRow(key, label) {
    var li = el('li', 'stage');
    li.setAttribute('data-stage', key);
    li.setAttribute('data-status', 'pending');

    var ringWrap = el('span', 'stage-ring');
    var svg = svgEl('svg', { viewBox: '0 0 40 40', focusable: 'false' });
    svg.setAttribute('aria-hidden', 'true');
    svg.appendChild(svgEl('circle', { 'class': 'ring-track', cx: 20, cy: 20, r: RING_R }));
    var fill = svgEl('circle', {
      'class': 'ring-fill', cx: 20, cy: 20, r: RING_R,
      'stroke-dasharray': RING_C.toFixed(3)
    });
    fill.style.strokeDashoffset = RING_C.toFixed(3);
    svg.appendChild(fill);
    svg.appendChild(svgEl('path', { 'class': 'ring-check', d: 'M13.4 20.4 L18.1 25.1 L26.9 15.4' }));
    ringWrap.appendChild(svg);

    var body = el('span', 'stage-body');
    body.appendChild(el('span', 'stage-label', label));
    var detail = el('span', 'stage-detail', '');
    body.appendChild(detail);

    var pct = el('span', 'stage-pct num', 'waiting');

    li.appendChild(ringWrap);
    li.appendChild(body);
    li.appendChild(pct);
    return { li: li, fill: fill, detail: detail, pct: pct,
             value: 0, status: 'pending', determinate: false };
  }

  /* the one place a ring moves. Reduced motion snaps straight to the value:
     no tween, ever. */
  function setRing(row, value, animated) {
    var v = clamp(num(value) === null ? row.value : value, 0, 1);
    var from = RING_C * (1 - row.value);
    row.value = v;
    var to = RING_C * (1 - v);

    /* The inline style is ALWAYS written, so the ring is correct the instant
       the value is known even if the animation layer is absent, blocked or
       broken. Motion One, when it is there, plays a Web Animations tween
       from the old value to the new one; a running animation outranks the
       inline style in the cascade, so the tween is what is seen and this
       value is what it lands on. */
    row.fill.style.strokeDashoffset = to.toFixed(3);

    var anim = window.__hzAnim;
    if (animated && !reduceMotion.matches && anim && typeof anim.ringTo === 'function' &&
        Math.abs(to - from) > 0.5) {
      anim.ringTo(row.fill, from, to);
    }
  }

  function progSetStages(list) {
    var host = $('stages');
    clear(host);
    prog.rows = {};
    prog.order = [];
    prog.weights = {};
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

  function progSay(text) {
    if (text === prog.said) return;
    prog.said = text;
    $('progress-say').textContent = text;
  }

  function progOpen(level) {
    prog.open = true;
    prog.mode = 'unknown';
    prog.started = Date.now();
    prog.said = '';
    prog.warned = false;

    progSetStages(stageList());
    $('progress-title').textContent = 'Rewriting your draft';
    $('progress-kicker').textContent = 'Working';
    progSetEngine('Contacting the service', null);
    progSetNote(null,
      'Your draft is not touched until this finishes. The "' + level + '" setting is being used.');
    $('progress-elapsed').textContent = '0.0s';
    $('progress').hidden = false;

    /* nothing may edit the document underneath a run */
    prog.wasEditable = editor.getAttribute('contenteditable') || 'true';
    editor.setAttribute('contenteditable', 'false');

    if (prog.ticker) clearInterval(prog.ticker);
    prog.ticker = setInterval(progTick, 100);
    progTick();

    /* If real events arrive first the estimate never starts. If nothing has
       arrived by then, an estimated timeline begins, clearly labelled. */
    if (prog.armTimer) clearTimeout(prog.armTimer);
    prog.armTimer = setTimeout(function () {
      prog.armTimer = null;
      if (prog.mode === 'unknown') progStartEstimate('estimated');
    }, 900);

    emit('humanizer:progress-open');
    try { $('progress-cancel').focus(); } catch (e) { /* headless or detached */ }
  }

  function progTick() {
    if (!prog.open) return;
    var t = (Date.now() - prog.started) / 1000;
    $('progress-elapsed').textContent = t.toFixed(1) + 's';
    if (!prog.warned && t > 45 && prog.mode !== 'rule') {
      prog.warned = true;
      progSetNote('warn',
        'This is slower than the 30 seconds the pipeline usually takes. It is still running. ' +
        'You can stop it at any time and your draft stays exactly as it is.');
    }
  }

  function progStopEstimate() {
    if (prog.estimator) { clearInterval(prog.estimator); prog.estimator = null; }
    if (prog.armTimer) { clearTimeout(prog.armTimer); prog.armTimer = null; }
  }

  /* the fallback timeline: honest about being a guess */
  function progStartEstimate(mode) {
    progStopEstimate();
    prog.mode = mode;
    if (mode === 'estimated') {
      progSetEngine('LLM rewrite pipeline', { kind: 'est', label: 'estimated' });
      progSetNote(null,
        'The service is not reporting its progress, so these steps are running on an estimated ' +
        'timeline rather than a measured one. The result itself is real.');
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
          /* the final ring is never allowed to complete on a guess: only the
             arrival of the result completes a run */
          prog.rows[k].determinate = true;
          progStage(k, { status: 'active', progress: Math.min(local, last ? 0.95 : 0.99) });
        }
      }
    }, slow ? 900 : 110);
  }

  /* real events have started: the estimate stands down */
  function progAdoptLive() {
    if (prog.mode === 'live') return;
    progStopEstimate();
    prog.mode = 'live';
    progSetEngine('LLM rewrite pipeline', { kind: 'live', label: 'measured' });
    progSetNote(null,
      'Each step above is reported by the service as it happens. Your draft is not touched ' +
      'until this finishes.');
  }

  /* the LLM pipeline is not there: fall back and say so in place */
  /* `why` is the reason this run is not using the LLM path, already written
     as a sentence. It is attributed to the service ONLY when the service
     actually said it: the word-limit check is made in this file from the
     limit the service published, and claiming otherwise would be a small lie
     in the one panel whose whole job is to tell the truth about what is
     happening. */
  function progSwitchToRule(why) {
    progStopEstimate();
    prog.mode = 'rule';
    progSetStages(RULE_STAGE);
    progSetEngine('Rule based engine', { kind: 'est', label: 'fallback' });
    progSetNote(null,
      (why || 'The LLM pipeline is not available on this service.') +
      ' The rule based engine is running instead. It works in a single pass, so there are no ' +
      'stages to report.');
    progStartEstimate('rule');
  }

  function normalizeStatus(raw) {
    var s = String(raw === undefined || raw === null ? '' : raw).toLowerCase().trim();
    if (!s) return null;
    if (s === 'active' || s === 'running' || s === 'started' || s === 'start' ||
        s === 'progress' || s === 'in_progress' || s === 'in-progress' ||
        s === 'working') return 'active';
    if (s === 'done' || s === 'complete' || s === 'completed' || s === 'finished' ||
        s === 'ok' || s === 'end') return 'done';
    if (s === 'skip' || s === 'skipped') return 'skipped';
    if (s === 'error' || s === 'failed' || s === 'failure') return 'error';
    if (s === 'pending' || s === 'queued' || s === 'waiting') return 'pending';
    return null;
  }

  /* 0..1, but a service that speaks in percent is not a reason to break */
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
      prog.order.push(key);
      prog.weights[key] = 2;
      $('stages').appendChild(row.li);
    }

    var status = normalizeStatus(ev && ev.status);
    var p = normalizeProgress(ev && ev.progress);
    if (status === 'done') p = 1;
    if (p === null) p = row.value;
    if (status === null) status = p >= 1 ? 'done' : (p > 0 ? 'active' : row.status);

    /* rings only go backwards when a stage genuinely restarts */
    if (status === row.status && p < row.value && row.status !== 'pending') p = row.value;

    row.status = status;
    row.li.setAttribute('data-status', status);

    /* The service reports intra-stage progress for some stages and not for
       others: `generate` runs for seconds emitting only a start and a done.
       Claiming 0% for that is worse than admitting we do not know, so the
       ring goes indeterminate, sweeps, and reads "working". It becomes a
       real percentage the moment a real value above zero arrives. */
    if (normalizeProgress(ev && ev.progress) > 0) row.determinate = true;
    var unknown = status === 'active' && !row.determinate;
    if (unknown) row.li.setAttribute('data-progress', 'unknown');
    else row.li.removeAttribute('data-progress');

    setRing(row, unknown ? 0.28 : p, true);

    row.pct.textContent =
      status === 'done'    ? 'done' :
      status === 'skipped' ? 'skipped' :
      status === 'error'   ? 'failed' :
      unknown              ? 'working' :
      status === 'active'  ? Math.round(p * 100) + '%' : 'waiting';

    if (ev && typeof ev.detail === 'string') row.detail.textContent = ev.detail;

    if (status === 'active' || status === 'error') {
      var at = prog.order.indexOf(key);
      /* a service reporting stage 4 has finished 1 to 3, whether it said so
         or not */
      for (var i = 0; i < at; i++) {
        var earlier = prog.rows[prog.order[i]];
        if (earlier && earlier.status !== 'done' && earlier.status !== 'skipped') {
          earlier.status = 'done';
          earlier.li.setAttribute('data-status', 'done');
          earlier.li.removeAttribute('data-progress');
          earlier.pct.textContent = 'done';
          setRing(earlier, 1, true);
        }
      }
      var d = row.detail.textContent;
      progSay('Step ' + (at + 1) + ' of ' + prog.order.length + ', ' +
              row.li.querySelector('.stage-label').textContent.toLowerCase() +
              (d ? ', ' + d : '') + '.');
    }
  }

  /* durations and details the service reports once it is finished */
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
      /* a stage that finished in under 50ms is noise as "0.0s" */
      if (e !== null && e >= 0.05) bits.push(e.toFixed(1) + 's');
      if (bits.length) row.detail.textContent = bits.join('  ·  ');
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
      setRing(row, 1, true);
    });
    progApplyReported(stages);
    $('progress-kicker').textContent = 'Finished';
    progSay('All steps finished.');
    /* a short beat so the last ring is seen to close, none under reduced motion */
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

    /* Focus came from the Humanize button and goes back to it. Deferred by a
       turn because the button is still in its busy state at this point: the
       caller re-enables it immediately after, and focusing a disabled button
       silently drops focus onto the body. */
    setTimeout(function () {
      try {
        var btn = $('humanize-btn');
        if (btn && !btn.disabled) btn.focus();
        else if (editor) editor.focus();
      } catch (e) { /* nothing focusable, nothing to do */ }
    }, 0);
  }

  /* ══════════════════════════════════════════════════════════════════════
     HUMANIZE — the transport

     Three routes, tried in order, because the LLM pipeline is being built in
     parallel with this interface and may not be there yet:

       1. Server-Sent Events, so the checklist is driven by measured
          progress. Tried on whatever streaming path the schema advertises,
          otherwise on /api/humanize/llm with an event-stream Accept header,
          which costs nothing: a service that answers JSON on that path is
          simply the blocking case, handled on the same response.
       2. The blocking POST /api/humanize/llm, with the checklist on an
          ESTIMATED timeline, labelled estimated rather than measured.
       3. POST /api/humanize, the rule based engine that already exists. The
          panel switches to it in place and the result says which engine ran.

     A 404, 405, 501, 422 or 400 on one route falls through to the next. Only
     the last route's error is ever shown, so the user never sees a stack of
     failures for endpoints they did not ask for.
     ══════════════════════════════════════════════════════════════════════ */

  var AGGRESSIVENESS = { light: 'light', balanced: 'balanced', strong: 'strong' };
  var LLM_CEILING_MS = 120000;

  var caps = {
    probed: false,
    rule: null,        /* /api/humanize is in the schema */
    llm: null,         /* /api/humanize/llm is in the schema */
    stream: null,      /* a separate streaming path, if one is advertised */
    streamMethods: [],
    /* from GET /api/humanize/llm/health, which is cheap and touches no
       weights. Null everywhere means the endpoint is not there. */
    llmReady: null,    /* true | false | null = never answered */
    llmReason: '',
    model: '',
    stageNames: null,  /* the service's own stage order, authoritative */
    maxWords: null
  };

  var activeRun = null;

  var STREAM_HINT = /(stream|sse|events?|progress|watch)$/i;

  function declaresEventStream(pathItem) {
    if (!pathItem || typeof pathItem !== 'object') return false;
    var found = false;
    Object.keys(pathItem).forEach(function (method) {
      var op = pathItem[method];
      if (!op || typeof op !== 'object' || !op.responses) return;
      Object.keys(op.responses).forEach(function (code) {
        var r = op.responses[code];
        if (r && r.content && typeof r.content === 'object' &&
            Object.keys(r.content).some(function (ct) { return ct.indexOf('text/event-stream') >= 0; })) {
          found = true;
        }
      });
    });
    return found;
  }

  function setHumanizeState(kind, message) {
    var btn = $('humanize-btn');
    var label = $('humanize-label');
    if (kind === 'busy') {
      btn.disabled = true;
      btn.setAttribute('data-busy', '1');
      label.textContent = 'Humanizing your draft';
      return;
    }
    btn.removeAttribute('data-busy');
    if (kind === 'absent') {
      btn.disabled = true;
      label.textContent = 'Humanize this draft';
      btn.title = message || 'The rewriting service is not available yet.';
      return;
    }
    if (kind === 'empty') {
      btn.disabled = true;
      label.textContent = 'Humanize this draft';
      btn.title = 'Paste or type a draft first.';
      return;
    }
    btn.disabled = false;
    label.textContent = 'Humanize this draft';
    btn.title = 'Rewrite the flagged sentences, then show you exactly what changed and what it costs.';
  }

  function refreshHumanizeButton() {
    if (activeRun) { setHumanizeState('busy'); return; }
    if (state.humanizeAvailable === false) {
      setHumanizeState('absent',
        'The rewriting endpoint is not built yet. Measurement still works; this button will ' +
        'switch on by itself once POST /api/humanize exists.');
      return;
    }
    if (state.mock) {
      setHumanizeState('absent', 'Mock mode cannot rewrite. It has no service to ask.');
      return;
    }
    if (!state.text.trim()) { setHumanizeState('empty'); return; }
    setHumanizeState('ready');
  }

  /* one capability probe on load, so the button and the panel are honest
     before anything is clicked */
  function probeHumanize() {
    if (state.mock) { refreshHumanizeButton(); return; }
    request('/openapi.json').then(function (doc) {
      var paths = doc && doc.paths && typeof doc.paths === 'object' ? doc.paths : null;
      if (!paths) { state.humanizeAvailable = null; refreshHumanizeButton(); return; }
      caps.probed = true;
      var keys = Object.keys(paths);
      caps.rule = keys.indexOf('/api/humanize') >= 0;
      caps.llm = keys.indexOf('/api/humanize/llm') >= 0;
      caps.stream = null;
      caps.streamMethods = [];
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        if (k.indexOf('humanize') < 0) continue;
        if (k === '/api/humanize' || k === '/api/humanize/llm') continue;
        if (STREAM_HINT.test(k) || declaresEventStream(paths[k])) {
          caps.stream = k;
          caps.streamMethods = Object.keys(paths[k] || {}).map(function (m) { return m.toLowerCase(); });
          break;
        }
      }
      state.humanizeAvailable = !!(caps.rule || caps.llm);
      refreshHumanizeButton();
      /* a probe is a nicety; it must never take the page down with it, and a
         fetch shim that throws synchronously would otherwise surface as an
         unhandled rejection */
      if (caps.llm) { try { probeLlmHealth(); } catch (e) { /* no health route */ } }
    }, function () {
      /* no schema to read: leave it enabled and find out on click */
      state.humanizeAvailable = null;
      refreshHumanizeButton();
    });
  }

  /* The LLM path advertises its own readiness, the model it would run and the
     stage names it will report, without touching a GPU. Asking is strictly
     better than guessing, and a service that does not have this route simply
     leaves every field null and nothing changes. */
  function probeLlmHealth() {
    var p;
    try { p = request('/api/humanize/llm/health'); }
    catch (e) { return; }
    if (!p || typeof p.then !== 'function') return;
    p.then(function (h) {
      if (!h || typeof h !== 'object') return;
      caps.llmReady = h.available === true;
      caps.llmReason = typeof h.reason === 'string' ? h.reason : '';
      caps.model = pickString(h.model, h.freeform_model);
      caps.maxWords = loose(h.max_words);
      if (Array.isArray(h.stages) && h.stages.length) {
        caps.stageNames = h.stages.filter(function (x) { return typeof x === 'string' && x; });
      }
    }, function () { /* no health route: every field stays null */ });
  }

  /* the checklist is built from the service's own stage order when it gives
     one, so a pipeline that adds or drops a stage does not need this file
     changed */
  function stageList() {
    if (!caps.stageNames || !caps.stageNames.length) return STAGES;
    return caps.stageNames.map(function (key) {
      var spec = stageSpec(key);
      return spec || { key: key, label: stageLabel(key), weight: 2 };
    });
  }

  function httpFail(res) {
    return res.text().catch(function () { return ''; }).then(function (t) {
      var detail = t;
      try { var j = JSON.parse(t); if (j && j.detail) detail = String(j.detail); } catch (e) { /* text */ }
      var body = null;
      try { body = JSON.parse(t); } catch (e) { body = null; }
      var e2 = new Error('HTTP ' + res.status + (detail ? ': ' + String(detail).slice(0, 200) : ''));
      e2.status = res.status;
      /* The LLM routes refuse with a real status code and a body that names
         where to go instead: 503 when MLX or the weights are missing, 413
         over the word limit. The named endpoint is authoritative, so it is
         checked before the status list. */
      if (body && typeof body.fallback_endpoint === 'string') {
        e2.missing = true;
        e2.fallbackNote = typeof body.detail === 'string' ? body.detail : '';
      } else if (res.status === 404 || res.status === 405 || res.status === 501 ||
                 res.status === 503 || res.status === 413 ||
                 res.status === 422 || res.status === 400) {
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

  /* a Server-Sent Events reader built on fetch rather than EventSource, so a
     POST body and an AbortController both work */
  /* a frame is the result only if it actually carries a rewrite */
  function looksLikeResult(o) {
    return !!(o && typeof o === 'object' && (
      typeof o.humanized === 'string' || typeof o.rewritten === 'string' ||
      typeof o.output === 'string' || (o.edits && o.summary)));
  }

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
        /* the service names where to go instead; that is a fall-through to
           the rule based engine, not an error to show the user */
        if (d && typeof d.fallback_endpoint === 'string') failure.missing = true;
        return;
      }
      if (lower === 'result' || lower === 'done' || lower === 'complete' || lower === 'final' ||
          (d && (typeof d.humanized === 'string' || (d.result && typeof d.result === 'object')))) {
        var payload = d && d.result && typeof d.result === 'object' ? d.result : d;
        /* `event: done` carries {"ok":true} and arrives immediately AFTER
           `event: result`. Accepting any payload here overwrote the rewrite
           with the acknowledgement and reported an empty result. */
        if (payload && typeof payload === 'object' && looksLikeResult(payload)) result = payload;
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

  function streamAttempt(path, method, body, signal, sink) {
    var url = API_BASE + path;
    var opts = {
      method: method,
      headers: { Accept: 'text/event-stream, application/json' },
      cache: 'no-store'
    };
    if (signal) opts.signal = signal;
    if (method === 'POST') {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    } else {
      url += (url.indexOf('?') >= 0 ? '&' : '?') +
        'text=' + encodeURIComponent(body.text) +
        '&aggressiveness=' + encodeURIComponent(body.aggressiveness);
    }

    return fetch(url, opts).then(function (res) {
      if (!res.ok) return httpFail(res);
      var ct = String(res.headers.get('Content-Type') || '').toLowerCase();
      if (ct.indexOf('text/event-stream') < 0) {
        /* the service answered in one shot on the same path: that IS the
           blocking fallback, and it cost no extra round trip */
        return res.json().then(function (j) { return { raw: j, streamed: false, stages: [] }; });
      }
      if (!res.body || typeof res.body.getReader !== 'function') {
        /* no streaming reader in this engine: take the whole transcript and
           parse it in one go. Late, but correct. */
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
      return res.json().then(function (j) { return { raw: j, streamed: false, stages: [] }; });
    }, function (err) { throw netFail(err); });
  }

  /* the known streaming route, tried when the schema could not be read at all */
  var LLM_STREAM_PATH = '/api/humanize/stream';
  var LLM_BLOCKING_PATH = '/api/humanize/llm';
  var RULE_PATH = '/api/humanize';

  function runAttempts(attempts, i, body, signal, sink) {
    if (i >= attempts.length) return Promise.reject(new Error('no rewriting endpoint answered'));
    var a = attempts[i];
    var work = a.stream
      ? streamAttempt(a.path, a.method || 'POST', body, signal, sink)
      : plainAttempt(a.path, body, signal);
    return work.then(function (res) {
      res.engine = a.engine;
      res.path = a.path;
      return res;
    }, function (err) {
      if (err && err.name === 'AbortError') throw err;
      if (err && err.missing && i + 1 < attempts.length) {
        /* switching engine in front of the user, with the service's own
           reason when it gave one */
        if (attempts[i + 1].engine === 'rule' && a.engine === 'llm') {
          progSwitchToRule(err.fallbackNote
            ? 'The service refused the LLM path: ' + err.fallbackNote
            : '');
        }
        return runAttempts(attempts, i + 1, body, signal, sink);
      }
      throw err;
    });
  }

  function runRewrite(text, level, signal) {
    var body = { text: text, aggressiveness: level };
    var sink = function (key, d) {
      progAdoptLive();
      /* a run with rounds > 1 repeats from `generate` with `round`
         incremented, so the detail says which pass this is */
      var detail = typeof d.detail === 'string' ? d.detail : '';
      var round = loose(d.round);
      if (round !== null && round > 1) detail = detail ? detail + '  \u00b7  pass ' + round : 'pass ' + round;
      progStage(key, { status: d.status, progress: d.progress, detail: detail });
    };

    var attempts = [];

    /* The service already told us the LLM cannot run, or that this draft is
       past its word limit. Say so up front and spend one request, not three. */
    var nWords = words(text).length;
    var over = caps.maxWords !== null && nWords > caps.maxWords;
    if (caps.llmReady === false || over) {
      progSwitchToRule(over
        ? 'This draft is ' + nWords + ' words, past the ' + caps.maxWords +
          ' word limit the LLM path publishes for itself.'
        : (caps.llmReason
            ? 'The LLM pipeline reported that it cannot run: ' + caps.llmReason
            : 'The LLM pipeline is not available on this service.'));
      attempts.push({ path: RULE_PATH, engine: 'rule', stream: false });
      return runAttempts(attempts, 0, body, signal, sink);
    }

    if (caps.stream) {
      /* whatever streaming path the schema advertises, on whichever method */
      var m = caps.streamMethods;
      var getOnly = m.length && m.indexOf('post') < 0 && m.indexOf('get') >= 0;
      attempts.push({ path: caps.stream, engine: 'llm', stream: true, method: getOnly ? 'GET' : 'POST' });
      attempts.push({ path: LLM_BLOCKING_PATH, engine: 'llm', stream: false });
    } else if (caps.probed) {
      /* the schema was readable and named no streaming path, so one request
         covers both the streaming and the blocking LLM case: a service that
         answers JSON on this path IS the blocking fallback, at no extra cost */
      attempts.push({ path: LLM_BLOCKING_PATH, engine: 'llm', stream: true, method: 'POST' });
    } else {
      /* no schema at all: try the documented streaming route, then blocking */
      attempts.push({ path: LLM_STREAM_PATH, engine: 'llm', stream: true, method: 'POST' });
      attempts.push({ path: LLM_BLOCKING_PATH, engine: 'llm', stream: true, method: 'POST' });
    }
    attempts.push({ path: RULE_PATH, engine: 'rule', stream: false });

    return runAttempts(attempts, 0, body, signal, sink);
  }

  function humanize() {
    var text = getText();
    if (!text.trim() || state.mock || state.humanizeAvailable === false) return;
    if (activeRun) return;

    var level = $('aggressiveness').value;
    if (!AGGRESSIVENESS[level]) level = 'balanced';

    var ctl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var run = { ctl: ctl, cancelled: false, timedOut: false, t0: Date.now() };
    activeRun = run;

    run.ceiling = setTimeout(function () {
      run.timedOut = true;
      if (ctl) { try { ctl.abort(); } catch (e) { /* already gone */ } }
    }, LLM_CEILING_MS);

    setHumanizeState('busy');
    setStatus('loading',
      'Rewriting your draft at the "' + level + '" setting. Nothing is kept until it finishes.');
    progOpen(level);

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
      var msg = (err && err.message) || 'unknown error';
      if (run.timedOut || (err && err.name === 'AbortError')) {
        setStatus('error',
          'The rewrite did not finish within two minutes, so it was stopped. Your draft is untouched.');
        refreshHumanizeButton();
        return;
      }
      if (/HTTP 40[45]/.test(msg) || /no rewriting endpoint answered/.test(msg)) {
        state.humanizeAvailable = false;
        refreshHumanizeButton();
        setStatus('idle',
          'Rewriting is not available on this service yet. Measurement is unaffected, and your ' +
          'draft is untouched.');
        return;
      }
      refreshHumanizeButton();
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
    if (run.ctl) { try { run.ctl.abort(); } catch (e) { /* already gone */ } }
    refreshHumanizeButton();
    progClose();
    setStatus('idle', 'Stopped. Your draft is exactly as you left it, and nothing was changed.');
  }

  /* ── the result ─────────────────────────────────────────────────────────
     Which engine ran, which model, how long it took, and the only measure
     that actually matters: whether the verdict moved, not just the number. */

  function pickString() {
    for (var i = 0; i < arguments.length; i++) {
      if (typeof arguments[i] === 'string' && arguments[i].trim()) return arguments[i];
    }
    return '';
  }

  function readModel(r) {
    var m = pickString(r.model, r.model_name);
    if (m) return m;
    if (r.engine && typeof r.engine === 'object') {
      var e = pickString(r.engine.model, r.engine.name);
      if (e) return e;
    }
    if (r.meta && typeof r.meta === 'object') {
      var mm = pickString(r.meta.model, r.meta.model_name);
      if (mm) return mm;
    }
    return '';
  }

  function readFlipped(r) {
    var sm = r.summary && typeof r.summary === 'object' ? r.summary : {};
    if (typeof sm.verdict_flipped === 'boolean') return sm.verdict_flipped;
    if (typeof sm.flipped === 'boolean') return sm.flipped;
    if (typeof r.verdict_flipped === 'boolean') return r.verdict_flipped;
    if (typeof r.flipped === 'boolean') return r.flipped;
    return null;
  }

  /* a verdict, reported if the service reports one, otherwise read off the
     probability at 0.50 and labelled as derived */
  function verdictOf(side, reportedLabel) {
    if (!side || typeof side !== 'object') side = {};
    var raw = pickString(reportedLabel, side.label, side.verdict, side.decision, side.classification);
    if (raw) {
      var s = raw.trim().toLowerCase();
      if (s === 'ai' || s === 'ai-generated' || s === 'ai_generated' || s === 'machine') {
        return { label: 'ai', reported: true };
      }
      if (s === 'human' || s === 'human-written' || s === 'human_written') {
        return { label: 'human', reported: true };
      }
      if (s === 'mixed') return { label: 'mixed', reported: true };
    }
    if (typeof side.is_ai === 'boolean') return { label: side.is_ai ? 'ai' : 'human', reported: true };
    var p = loose(side.ai_probability);
    if (p === null) p = loose(side.probability);
    if (p === null) return null;
    return { label: p >= 0.5 ? 'ai' : 'human', reported: false };
  }

  function finishHumanize(res, text, level, elapsed) {
    var r = res.raw && typeof res.raw === 'object' ? res.raw : {};
    var out = pickString(r.humanized, r.rewritten, r.output, r.text);
    if (!out.trim()) {
      setStatus('error', 'The service returned an empty rewrite, so your draft was left alone.');
      refreshHumanizeButton();
      return;
    }

    var unchanged = out.trim() === text.trim();
    /* Nothing to undo when nothing changed, and offering an undo button for a
       no-op is a lie about what happened. */
    state.preHumanize = unchanged
      ? null
      : (typeof r.original === 'string' && r.original.trim() ? r.original : text);
    state.humanize = {
      unchanged: unchanged,
      edits: Array.isArray(r.edits) ? r.edits : [],
      before: r.before && typeof r.before === 'object' ? r.before : null,
      after: r.after && typeof r.after === 'object' ? r.after : null,
      summary: r.summary && typeof r.summary === 'object' ? r.summary : null,
      level: pickString(r.aggressiveness, level),
      engine: res.engine,
      streamed: res.streamed === true,
      model: readModel(r),
      elapsed: loose(r.elapsed) !== null ? loose(r.elapsed) : elapsed,
      /* The endpoint's own docstring says to read verdict_flipped and NOT
         risk_delta, because research/13's DUPE result is that attacks which
         halve the false-negative rate still earn almost no human verdicts.
         That is the whole point of this line, so it is read first. */
      flipped: readFlipped(r)
    };

    if (!unchanged) setText(out);
    $('undo-btn').hidden = unchanged;
    renderHumanizeResult();
    refreshHumanizeButton();
    if (unchanged) {
      setStatus('idle',
        'Your draft came back unchanged. Every rewrite the pipeline wrote failed one of its ' +
        'own checks, so it kept yours. The detail is in the fold.');
    }
    analyze();
  }

  function undoHumanize() {
    if (!state.preHumanize) return;
    var original = state.preHumanize;
    state.preHumanize = null;
    state.humanize = null;
    $('undo-btn').hidden = true;
    renderHumanizeResult();
    setText(original);
    refreshHumanizeButton();
    setStatus('idle', 'Your original draft is back. Re-measuring it now.');
    analyze();
  }

  function renderHumanizeResult() {
    var block = $('ba-block');
    var section = $('edits-section');
    var host = $('edits');
    var h = state.humanize;

    if (!h) {
      block.hidden = true;
      section.hidden = true;
      $('ba-verdict').hidden = true;
      $('ba-engine').hidden = true;
      clear(host);
      return;
    }

    /* before and after risk, side by side */
    var before = h.before ? loose(h.before.ai_probability) : null;
    var after = h.after ? loose(h.after.ai_probability) : null;
    block.hidden = false;
    $('ba-title').textContent = h.unchanged ? 'Left unchanged' : 'Humanized';
    $('ba-before').textContent = before === null ? 'n/a' : Math.round(before * 100) + '%';
    $('ba-after').textContent = after === null ? 'n/a' : Math.round(after * 100) + '%';

    /* ── the honest measure ── */
    var vn = $('ba-verdict');
    /* the LLM endpoint puts the two labels on `summary`, the rule endpoint
       puts nothing anywhere, so both places are read before falling back to
       the probability */
    var sm = h.summary || {};
    var vb = verdictOf(h.before, sm.label_before);
    var va = verdictOf(h.after, sm.label_after);
    var flipped = null;
    clear(vn);
    if (!vb || !va) {
      /* Both verdicts missing is itself worth a sentence. Measured live: the
         pipeline ran with no detector installed and reported no probability
         on either side, and an empty slot would have read as "no news". */
      vn.hidden = false;
      vn.setAttribute('data-flip', 'na');
      vn.appendChild(el('b', null, 'No verdict either way. '));
      vn.appendChild(document.createTextNode(
        'This run reported no detector score before or after, so there is nothing to compare ' +
        'and no claim to make about whether it would now read as human.'));
    } else {
      vn.hidden = false;
      flipped = h.flipped !== null && h.flipped !== undefined
        ? h.flipped
        : (vb.label === 'ai' && va.label !== 'ai');
      if (vb.label !== 'ai') {
        vn.setAttribute('data-flip', 'na');
        vn.appendChild(el('b', null, 'There was no verdict to flip. '));
        vn.appendChild(document.createTextNode(
          'The detector already read this draft as ' + va.label + ' before the rewrite ran.'));
      } else if (flipped) {
        vn.setAttribute('data-flip', 'yes');
        vn.appendChild(el('b', null, 'The verdict flipped. '));
        vn.appendChild(document.createTextNode(
          'It read as AI before and reads as ' + va.label + ' now.'));
      } else {
        vn.setAttribute('data-flip', 'no');
        vn.appendChild(el('b', null, 'The verdict did not flip. '));
        vn.appendChild(document.createTextNode(
          'It still reads as AI. Cutting the probability is not the same as earning a human ' +
          'verdict, and the verdict is the only thing a detector actually reports.'));
      }
      vn.appendChild(document.createTextNode(
        vb.reported && va.reported
          ? ' Both verdicts come from the service.'
          : ' Read from the probability at the 0.50 mark, because the service reported a number ' +
            'and not a verdict.'));
    }

    /* The delta is drawn as a win only when the verdict actually moved. A
       falling probability that leaves the draft still labelled AI is a real
       measurement and is still shown, but it is not painted as a success. */
    var deltaNode = $('ba-delta');
    if (before !== null && after !== null) {
      var d = (after - before) * 100;
      deltaNode.textContent = (d > 0 ? '+' : '') + d.toFixed(0) + ' points';
      deltaNode.setAttribute('data-dir', d > 0 ? 'up' : (flipped === true ? 'down' : 'flat'));
    } else {
      deltaNode.textContent = '';
      deltaNode.removeAttribute('data-dir');
    }

    /* ── which engine, which model, how long ── */
    var eng = $('ba-engine');
    var bits = [];
    bits.push(h.engine === 'llm' ? 'LLM rewrite pipeline' : 'rule based engine');
    if (h.model) bits.push('model ' + h.model);
    else if (h.engine === 'llm') bits.push('model not reported');
    if (h.elapsed !== null && h.elapsed !== undefined) {
      bits.push(h.elapsed < 0.05 ? 'under 0.1s' : h.elapsed.toFixed(1) + 's');
    }
    bits.push(h.streamed ? 'progress streamed live' : 'progress estimated');
    eng.hidden = false;
    eng.textContent = bits.join('  ·  ');

    var n = h.summary && loose(h.summary.n_edits) !== null
      ? loose(h.summary.n_edits) : h.edits.length;
    var note = '';
    if (h.unchanged) {
      note += 'Your draft came back exactly as you wrote it. ';
    }
    note += n + ' ' + plural(n, 'edit') + ' at the "' + h.level + '" setting. ';

    /* The pipeline throws away candidates that scored well by drifting off
       the source. Reporting the rejections is the difference between a
       rewrite and a paraphraser, so they are stated rather than hidden. */
    var rejected = h.summary ? loose(h.summary.n_candidates_rejected) : null;
    if (rejected !== null && rejected > 0) {
      note += rejected + ' ' + plural(rejected, 'candidate') + ' ' +
        (rejected === 1 ? 'was' : 'were') + ' rejected for failing a gate, usually for ' +
        'drifting away from what you actually wrote. The lowest scoring text is not ' +
        'automatically the best one. ';
    }
    note += 'The number on the right comes from the same local baseline as the one above it, so it ' +
      'inherits the same caveats. A lower score is not proof of anything.';
    $('ba-note').textContent = note;

    /* the edit list, with the diff colours finally in use */
    section.hidden = !h.edits.length;
    clear(host);
    if (!h.edits.length) return;

    var key = el('div', 'diff-key');
    var kd = el('span'); kd.appendChild(el('i', 'k-del')); kd.appendChild(document.createTextNode('removed'));
    var ki = el('span'); ki.appendChild(el('i', 'k-ins')); ki.appendChild(document.createTextNode('added'));
    key.appendChild(kd); key.appendChild(ki);
    host.appendChild(key);

    h.edits.forEach(function (e) {
      if (!e || typeof e !== 'object') return;
      var card = el('article', 'edit');

      var kind = el('p', 'edit-kind');
      var si = loose(e.sentence_index);
      kind.textContent = (typeof e.kind === 'string' && e.kind ? e.kind : 'edit') +
        (si === null ? '' : '  ·  sentence ' + (si + 1));
      card.appendChild(kind);

      var diff = el('p', 'edit-diff');
      if (typeof e.before === 'string' && e.before) {
        diff.appendChild(el('span', 'edit-del', e.before));
      }
      if (e.before && e.after) diff.appendChild(el('span', 'edit-arrow', ' → '));
      if (typeof e.after === 'string' && e.after) {
        diff.appendChild(el('span', 'edit-ins', e.after));
      }
      card.appendChild(diff);

      if (typeof e.rationale === 'string' && e.rationale) {
        card.appendChild(el('p', 'edit-why', e.rationale));
      }

      var g = classifyGradeCost(e.grade_cost);
      var chip = el('span', 'grade');
      chip.setAttribute('data-kind', g.kind);
      chip.appendChild(el('b', null, g.label));
      if (g.raw) chip.appendChild(el('span', 'grade-raw', 'grade_cost: “' + g.raw + '”'));
      card.appendChild(chip);

      host.appendChild(card);
    });
  }

  /* ── status, errors and the analyze cycle ───────────────────────────────── */

  function setStatus(kind, message) {
    state.status = kind;
    statusLine.setAttribute('data-state', kind);
    statusLine.textContent = message;
  }

  function hideError() { errorBox.hidden = true; clear(errorBox); }

  function showError(message) {
    clear(errorBox);
    errorBox.hidden = false;

    errorBox.appendChild(el('h2', 'error-title', 'The service did not answer'));
    errorBox.appendChild(el('code', 'error-code', message));
    errorBox.appendChild(el('p', 'error-help',
      navigator.onLine
        ? 'Analysis is a POST to /api/analyze on this origin. Start the Python service with ' +
          '"humanizer serve --port 8000", or point this page somewhere else with ?api=http://localhost:8000. ' +
          'Your draft is untouched. Nothing was lost.'
        : 'Your browser reports no network connection. Your draft is untouched. Nothing was lost.'));

    var actions = el('div', 'error-actions');
    var retry = el('button', 'btn btn-primary', 'Try again');
    retry.type = 'button';
    retry.addEventListener('click', function () { checkHealth(); analyze(); });
    actions.appendChild(retry);

    if (!state.mock) {
      var toMock = el('button', 'btn', 'Use mock data instead');
      toMock.type = 'button';
      toMock.addEventListener('click', function () { setMock(true); analyze(); });
      actions.appendChild(toMock);
    }
    errorBox.appendChild(actions);
  }

  function setMock(on) {
    state.mock = !!on;
    $('mock-banner').hidden = !state.mock;
    if (state.mock) {
      hideError();
      setPill('wait', 'Mock mode. The service is not being contacted.');
    }
    state.detectors = null;
    if ($('engine-detail')) { refreshOverlay(); renderDetectors(); }
    refreshHumanizeButton();
  }

  function setPill(kind, text) {
    var pill = $('health');
    pill.setAttribute('data-kind', kind);
    pill.textContent = text;
    pill.title = text;
  }

  function applyAnalysis(text, a) {
    state.analysis = a;
    state.analyzedText = text;
    state.located = locateSentences(text, a.sentences);
    if (state.selected !== null) {
      var stillThere = state.located.some(function (s) { return s.index === state.selected; });
      if (!stillThere) state.selected = null;
    }

    refreshOverlay();

    var current = getText();
    var fresh = current === text;
    if (fresh) renderCanvas();

    renderCounts();
    refreshHumanizeButton();
    var dials = renderHero();
    renderMeasurements();
    renderFindings();
    renderStyleSignals();
    renderFlagged();
    renderInspector();
    /* the engine card reads state.analysis, and /api/detect can land BEFORE
       /api/analyze does. Without this the panel stayed on "No analysis yet."
       whenever it won the race. */
    renderDetectors();

    var unlocated = a.sentences.length - state.located.length;
    var parts = [];
    if (dials.risk !== null) parts.push('detection risk ' + Math.round(dials.risk) + ' of 100');
    if (dials.quality !== null) parts.push('writing quality ' + Math.round(dials.quality) + ' of 100');
    parts.push(a.findings.length + ' ' + plural(a.findings.length, 'finding'));
    if (unlocated > 0) parts.push(unlocated + ' ' + plural(unlocated, 'sentence') + ' could not be matched to your text');
    if (state.mock) parts.push('mock data');
    setStatus(fresh ? 'ok' : 'stale',
      (fresh ? 'Done. ' : 'You have edited since this ran. ') + parts.join(' \u00b7 ') + '.');
    emit('humanizer:analysis', { fresh: fresh });
  }

  var debounceTimer = null;

  function scheduleAnalyze() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () { debounceTimer = null; analyze(); }, DEBOUNCE_MS);
  }

  function analyze() {
    if (debounceTimer) { clearTimeout(debounceTimer); debounceTimer = null; }
    /* a rewrite owns the document while it runs; a stale debounce must not
       re-render the canvas underneath it or overwrite the status line */
    if (activeRun) return;
    var text = getText();
    state.text = text;

    if (!text.trim()) {
      state.analysis = null;
      state.detectors = null;
      state.located = [];
      state.selected = null;
      $('empty-state').hidden = false;
      refreshOverlay();
      refreshHumanizeButton();
      renderCounts(); renderHero(); renderMeasurements();
      renderFindings(); renderStyleSignals(); renderFlagged();
      renderInspector(); renderDetectors();
      setStatus('idle', 'Ready when you are. Paste a draft, or load one of the samples above.');
      return;
    }

    var token = ++state.requestToken;
    var btn = $('analyze-btn');
    btn.disabled = true;
    var wordCount = words(text).length;
    setStatus('loading', 'Measuring ' + wordCount + ' ' + plural(wordCount, 'word') + '...');

    var work;
    if (state.mock) {
      work = new Promise(function (resolve) {
        setTimeout(function () { resolve(mockAnalyze(text)); }, 60);
      });
    } else {
      work = request('/api/analyze', { text: text, reference: state.reference || null });
    }

    /* the ensemble runs in parallel; it must never block the main reading */
    fetchDetectors(text, token);

    work.then(function (raw) {
      if (token !== state.requestToken) return;
      btn.disabled = false;
      hideError();
      if (!state.mock) setPillHealthy();
      applyAnalysis(text, normalizeAnalysis(raw));
    }, function (err) {
      if (token !== state.requestToken) return;
      btn.disabled = false;
      var msg = (err && err.message) || 'unknown error';
      setStatus('error', 'That did not work: ' + msg + '. Your draft is untouched.');
      if (!state.mock) setPill('bad', 'Service unreachable');
      showError(msg);
    });
  }

  /* ── health and reference list ──────────────────────────────────────────── */

  var lastHealth = null;

  function setPillHealthy() {
    if (!lastHealth) { setPill('ok', 'Service connected'); return; }
    var h = lastHealth;
    var refs = h.references;
    var refCount = Array.isArray(refs) ? refs.length : num(refs);
    setPill('ok', [
      'Service ' + (h.version || 'connected'),
      h.syntax_backend ? 'syntax parser on' : 'no syntax parser',
      refCount === null || refCount === undefined ? null : refCount + ' reference ' + plural(refCount, 'corpus', 'corpora')
    ].filter(Boolean).join(' \u00b7 '));
  }

  function checkHealth() {
    if (state.mock) { setPill('wait', 'Mock mode. The service is not being contacted.'); return; }
    setPill('wait', 'Checking the service...');
    request('/api/health').then(function (h) {
      lastHealth = h && typeof h === 'object' ? h : {};
      setPillHealthy();
      hideError();
    }, function (err) {
      lastHealth = null;
      setPill('bad', 'Service unreachable');
      showError((err && err.message) || 'the health check failed');
    });
  }

  /* ── theme ──────────────────────────────────────────────────────────────── */

  var THEMES = ['auto', 'light', 'dark'];
  var THEME_HINT = {
    auto: 'follow the system setting',
    light: 'light',
    dark: 'dark'
  };

  function applyTheme(theme) {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    $('theme-label').textContent = theme.charAt(0).toUpperCase() + theme.slice(1);
    var btn = $('theme-btn');
    var next = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    btn.setAttribute('aria-label', 'Colour scheme: ' + THEME_HINT[theme] + '. Switch to ' + next + '.');
    btn.title = 'Colour scheme: ' + THEME_HINT[theme] + '. Click for ' + next + '.';
    try { localStorage.setItem('humanizer.theme', theme); } catch (e) { /* private mode */ }
  }

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') || 'auto';
  }

  /* ── an expandable explanation, shared by the three "?" buttons ─────────── */

  function wireExplain(buttonId, panelId) {
    var btn = $(buttonId), panel = $(panelId);
    if (!btn || !panel) return;
    btn.addEventListener('click', function () {
      var open = panel.hidden;
      panel.hidden = !open;
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  /* ── network state ──────────────────────────────────────────────────────── */

  function renderOnline() {
    var offline = !navigator.onLine;
    $('offline-banner').hidden = !offline || state.mock;
    if (offline && !state.mock) setPill('bad', 'Offline. No connection to the service.');
  }

  /* ══════════════════════════════════════════════════════════════════════
     WIRING
     ══════════════════════════════════════════════════════════════════════ */

  function init() {
    editor = $('editor');
    scroller = $('canvas-scroll');
    statusLine = $('status-line');
    errorBox = $('error-box');

    /* theme */
    var savedTheme = 'auto';
    try { savedTheme = localStorage.getItem('humanizer.theme') || 'auto'; } catch (e) { /* ignore */ }
    if (THEMES.indexOf(savedTheme) < 0) savedTheme = 'auto';
    applyTheme(savedTheme);
    $('theme-btn').addEventListener('click', function () {
      applyTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]);
    });

    /* the three "?" explainers */
    wireExplain('quality-why', 'quality-explain');
    wireExplain('reference-why', 'reference-explain');

    /* editing */
    editor.addEventListener('input', function () {
      state.text = getText();
      $('empty-state').hidden = !!state.text.trim();
      refreshHumanizeButton();
      renderCounts();
      setStatus('stale', 'Edited. Measuring again in ' + (DEBOUNCE_MS / 1000).toFixed(1) + 's.');
      scheduleAnalyze();
    });

    /* plain text only, so pasted markup cannot break the document model */
    editor.addEventListener('paste', function (e) {
      if (!e.clipboardData) return;
      e.preventDefault();
      var t = e.clipboardData.getData('text/plain') || '';
      document.execCommand('insertText', false, t);
    });

    editor.addEventListener('click', function (e) {
      var node = e.target;
      while (node && node !== editor && !(node.classList && node.classList.contains('sent'))) node = node.parentNode;
      if (node && node !== editor) selectSentence(Number(node.getAttribute('data-index')), false);
    });

    /* caret movement is the keyboard route into the inspector; putting a
       tabindex on every sentence span would be hostile inside a contenteditable */
    document.addEventListener('selectionchange', function () {
      if (rendering || document.activeElement !== editor) return;
      var sel = window.getSelection();
      if (!sel || !sel.rangeCount) return;
      var node = sel.getRangeAt(0).startContainer;
      while (node && node !== editor && !(node.classList && node.classList.contains('sent'))) node = node.parentNode;
      if (node && node !== editor) {
        var idx = Number(node.getAttribute('data-index'));
        if (idx !== state.selected) selectSentence(idx, false);
      }
    });

    /* Ctrl/Cmd+Enter analyses from anywhere on the page */
    document.addEventListener('keydown', function (e) {
      /* Escape is the second escape hatch out of a running rewrite */
      if (e.key === 'Escape' && activeRun) { e.preventDefault(); cancelHumanize(); return; }
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!activeRun) analyze();
      }
    });

    $('analyze-btn').addEventListener('click', function () { analyze(); });

    $('heat-advisory').addEventListener('click', function () {
      setHeatSource('advisory'); renderDetectors();
    });
    $('heat-perplexity').addEventListener('click', function () {
      if (this.disabled) return;
      setHeatSource('perplexity'); renderDetectors();
    });

    /* clicking a sentence should not lead to a dead end, so the fold opens */
    editor.addEventListener('click', function () {
      if (state.selected !== null) $('more').open = true;
    });

    /* the two onboarding examples in the empty state: the only sample
       affordance left, and it disappears the moment there is any text */
    function loadSample(key) {
      state.humanize = null; state.preHumanize = null;
      $('undo-btn').hidden = true;
      renderHumanizeResult();
      setText(SAMPLES[key] || '');
      editor.focus();
      analyze();
    }
    Array.prototype.forEach.call(document.querySelectorAll('[data-sample]'), function (b) {
      b.addEventListener('click', function () { loadSample(b.getAttribute('data-sample')); });
    });

    /* the point of the product */
    $('humanize-btn').addEventListener('click', function () { humanize(); });
    $('undo-btn').addEventListener('click', function () { undoHumanize(); });
    $('progress-cancel').addEventListener('click', function () { cancelHumanize(); });
    $('aggressiveness').addEventListener('change', function () { refreshHumanizeButton(); });

    /* mock mode */
    $('mock-exit').addEventListener('click', function () {
      setMock(false);
      renderOnline();
      checkHealth();
      probeHumanize();
      analyze();
    });
    $('offline-mock').addEventListener('click', function () {
      setMock(true);
      $('offline-banner').hidden = true;
      analyze();
    });

    /* network */
    window.addEventListener('online', function () {
      renderOnline();
      if (!state.mock) { checkHealth(); probeHumanize(); if (state.text.trim()) analyze(); }
    });
    window.addEventListener('offline', renderOnline);

    setMock(state.mock);
    renderOnline();
    checkHealth();
    probeHumanize();

    var wantSample = PARAMS.get('sample');
    if (wantSample === 'ai' || wantSample === 'human') setText(SAMPLES[wantSample]);
    else setText('');

    refreshOverlay();
    refreshHumanizeButton();
    renderHumanizeResult();
    renderHero();
    renderMeasurements();
    renderFindings();
    renderStyleSignals();
    renderFlagged();
    renderInspector();
    renderDetectors();

    if (state.text.trim()) analyze();
    else setStatus('idle', 'Ready when you are. Paste a draft, or load one of the samples above.');

    window.__hzReady = true;
    emit('humanizer:ready');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
