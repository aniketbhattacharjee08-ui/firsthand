# 11 — Product and Front-End Teardown of the Humanizer Market

Date: 2026-09-03. Companion to `03-commercial-humanizers-and-evasion-papers.md`, which covers what these tools do to *text*. This report covers what they look and feel like, what a premium alternative should look like instead, and which front-end machinery builds it.

**Method.** Landing pages, app/product pages, pricing pages, terms of service and XML sitemaps were fetched directly for fifteen incumbents; Trustpilot for churn and dark-pattern signal; GPTZero's own product surfaces as the reference implementation of AI-risk visualisation. Editor and diff libraries were evaluated from primary docs, npm/GitHub health data and developer post-mortems. Design direction was sourced from first-party design writing (Linear, iA, Klim, Radix, Butterick) and from two empirical catalogues of "AI design slop." Public web search was substantially degraded during this session (Bing, DuckDuckGo, Mojeek and SearXNG all captcha'd or returned scrambled SERPs), so nearly everything below is a direct primary-source fetch rather than a search snippet. Sources with URLs are in §8.

---

## 1. The incumbent UX is one page, rendered fifteen times

Every tool in this market ships the same page. A hero headline containing *AI*, *Human*, and either *Bypass* or *Undetectable*. A single centred textarea with a word counter (`0/1000`, `0/250`, `0/10,000`) and one primary button. A horizontally scrolling greyscale carousel of detector logos — GPTZero, Turnitin, Originality.ai, Copyleaks, ZeroGPT, Winston AI, Writer, Crossplag, Content at Scale, Sapling — almost always in that order. A row of round numbers (*10M+ users*, *99.5% success rate*, *1.82bn words/month*). A press-logo bar (Forbes, Business Insider, TechRadar, Yahoo Finance). A TikTok testimonial carousel. A three-column pricing table with a monthly/yearly toggle advertising "SAVE 45%." An accordion FAQ. A footer with two hundred SEO links.

The only real variation between them is the mode names.

### 1.1 Competitor UX table

| Tool | Core interaction | Controls exposed | Detector score in-product? | Trust signals | Pricing shape | What reads as templated |
|---|---|---|---|---|---|---|
| **Undetectable.ai** | Dual-button textarea (5,000 words): "Check for AI" and "Humanize." Paste, upload, or load a sample labelled ChatGPT/Claude/DeepSeek | Readability: High School / University / Doctorate / Journalist / Marketing. Purpose: General / Essay / Article / Marketing / Story / Cover Letter / Report / Business / Legal. Strength: More Readable / Balanced / More Human | Yes — before/after percentages grouped as "Free Detectors / Standard Detectors / Advanced Detectors"; claims sentence-level flagging | "24M+ users," "40,919 new users this week," Forbes "#1 Best AI Detector," logo wall incl. *Nature* | $9.99–$42.50/mo by word bucket (10K/20K/35K/50K); 50% annual; 250-word free trial; refund guarantee if flagged | Gradient overlays on feature sections; dark-hero/bright-CTA contrast; the vanity counter that ticks |
| **StealthGPT** | Hero *is* the editor. Before/after panes with "99% AI" → "97% human" | Model dropdown (Ghost / Samurai; Heavy / Lite), 1,000-word cap, settings menu, GPTZero-vs-Turnitin target toggle | Yes — hard-coded marketing numbers ("Turnitin: 98% human average (0/30 detections)") | "1.5M+ active users," "930 satisfied companies," money-back guarantee, competitor comparison table naming Undetectable/HIX/Humbot | ~$24.99–$249.99/mo, no free tier | Icon-topped feature cards on a uniform grid; light/dark toggle as a feature; arrow graphics as "gradient" |
| **HIX Bypass** | Paste box with a Bypass-AI / AI-detector segmented toggle; Upload File / Try A Sample | Fast, Aggressive, Latest (Balanced on siblings) | **Yes, and the strongest in the market** — a post-run badge row of ZeroGPT, Crossplag, Content at Scale, Copyleaks, OpenAI, GPTZero, Sapling, Writer | "1,280,000+ Students," "99.5% Success Rate," "1.82bn+ Words/Month"; Harvard, Columbia, Shopify, Etsy logos | $9.99 / $14.99 / $15.00 monthly-equivalent annual; $14.99/$29.99/$59.99 monthly | Navy + accent gradient; university logos with no relationship behind them |
| **Humbot** | "Start Humanizing Free" CTA; paste box behind signup | Light / Balanced / Aggressive | Claims verification against Originality.ai, ZeroGPT, Turnitin, GPTZero; no live score on the marketing page | "30M+ users," "6300+ schools/teams," 4.9/5, ISO 27001 / SOC 2 / GDPR badges | "86% OFF" banner; tiers behind `/pricing`; 30-day money-back | Card-based grid, subtle background gradients, avatar testimonial wall; a *study-assistant* repositioning bolted onto a bypass tool |
| **BypassGPT** | Three-step paste → click → receive | Fast / Creative / Enhanced | No live score; asserts "99%+ Human Score" | "10M+ Users," "200M+ Data," TechRadar/BI/PCWorld, TikTok proof | Free 150 words/mo; $6.80–$33.15 by tier and discount | Numbered 1-2-3 steps; user-category imagery (students, bloggers, webmasters, agencies) |
| **WriteHuman** | Paste box (0/250 free), "Write Human" button | Standard vs Enhanced; 2–5 output variations by tier | **Yes, and the best-designed in the market** — per-sentence probability with explicit bands: <40% human, 40–60% mixed, >60% AI; overall score plus a Human/Mixed/AI class breakdown; re-scan without leaving the page | 7.5M+ documents, 4.6/5 (6,583 Play reviews), 4.7/5 App Store, "#1 on HumanizerBench," and a *detector benchmark table* (Copyleaks 98, Turnitin 97, GPTZero 96, Pangram 94) | Free 3 req/mo; $12/$19/$39 annual, $20/$29/$59 monthly; request-count + word-cap + variation-count matrix | White/navy with subtle gradients; TikTok embeds; the pricing matrix is genuinely legible, which is rare here |
| **Phrasly** | Suite hub: Humanizer, Detector, Writer, "Phrasly Pages" | Easy / Medium / Aggressive / Pro Engine; tone (formal, casual, persuasive, informative) | Yes — highlights "robotic sentences" with a score, but **highlighting is gated behind sign-in** | "3,000,000+ users, 180+ countries, 300+ institutions," 4.7/5 Trustpilot (3,200+) | $2 three-day access; $10.99/mo annual "Unlimited"; SAVE 45% toggle | Generic hero ("Supercharge Your Writing With AI"), avatar row, gradient CTAs |
| **Rephrasy** | Paste box with source tabs (ChatGPT / Claude / DeepSeek / Human) | Humanizer model *or* **your own cloned writing style** (1–10 custom styles by tier) | Yes — "AI Detection likelihood" with free re-humanize | "124,610+ Users," "651,589+ Papers"; Harvard/Stanford/MIT logos; 4.8/5 | $12.99 24-hour pass; $8.25–$18.33/mo annual; $199 lifetime | Light bg, blue accent, icon feature cards; the lifetime deal and 24-hour pass are churn tells |
| **Walter Writes** | Paste → "seasoning your words with human nuances…" loading copy → detector runs automatically | Simple / Standard / Enhanced; tone presets | Yes — an "Authenticity Score" after every rewrite | Reuters, Inc., The Guardian, Zapier, Shopify, HubSpot, Hootsuite, SEMrush | 300-word free trial; $8–$26/mo annual; API $49–$1,699/mo | **Purple checkmarks**; fingerprint iconography; copy like "detection-proof" and "AI-amazing" |
| **Twixify** | Three steps: input → pick style → tune tone | Style chips with emoji: Personable, Empathetic, Direct, Friendly, Analytical, Reflective; "Expand & Elaborate" slider; "Echowriting" from 3 pasted samples | No | AriseHealth/Forbes/Fiverr/Envato logos, creator GIFs, "25% off for the first 7M users" | $5–$27/mo; $150 lifetime | Emoji as iconography; gradient styling; a fake-scarcity banner. Its ToS says usage "must comply with the permissions and restrictions of ChatGPT" — i.e. it is a thin OpenAI wrapper and says so |
| **QuillBot** | Paste → humanize → refine; browser extension into ChatGPT/Gemini | Advanced Mode, synonym slider, Tone setting, History | Yes — a human-sounding score you click through to the full AI Detector | Part of a mature suite; no bypass claims | Free ≤125 words; Premium ~$30/mo bundle | Least templated of the group, because it is a real product with a humanizer bolted on |
| **GPTinf** | Single dashboard, "Check. Fix. Submit." | 8 writing modes; **"Freeze" to lock terms**; selective re-humanization of only flagged segments | **Yes — "8 systems cross-check" with per-system visibility and segment-level colours** | "0% AI — trained on real human writing," "15B+ words," proprietary "GPTinf Steel" | $4.99 / $12.49 / $29.99 | Percentage-superiority stat banner ("37% Better"); named-engine branding on a paraphraser |
| **Netus AI** | Two product cards: Writer and Chat | 29 bypasser versions (V1–V29), model codes C7-N1, G9-R1, V7-N2, A12-AD; 36 languages | **Yes — red for AI, green for human, yellow for unsure, per sentence** | 4.4/5 Trustpilot; explicit "your text is not used to train any model"; Chrome extension + Google Docs add-on | Free 50 credits; $97–$710/yr; 1 credit ≈ 10 words | Minimal, but the model-code naming is cargo-cult ML branding |
| **Ryne AI** | Before/after cards: "100% AI" → "1% AI" | Light / Medium / Max; dual mode (readability-tuned vs detector-tuned) | Yes — a multi-detector card (Turnitin 0%, GPTZero 0%, ZeroGPT 0%) plus plagiarism %, exportable as a **downloadable PDF report** | "2.3M+ students," "99.9% success rate," Discord | Free → $19.99 → $29.99 → $99.99/mo, tiers named Amethyst/Sapphire/Emerald/Ruby | Gradient hero with glow SVGs, gemstone tier names, testimonial carousel with photo avatars, a "Bionic Reading" gimmick |
| **Grammarly** | Paste or upload → Humanize, inside the full Docs suite | Four preset voices + custom voice trained on your samples; six languages; document domain (General / **Academic** / Business / Email / Creative) | Separate AI Detector + Authorship; humanizer explicitly not a bypass tool | The suite itself; AI Grader, Citation Finder, Proofreader agents | Free basic; Pro | Not templated. This is the competitor to actually worry about |

### 1.2 Who shows a live detector score inside the product

Ten of fifteen show something. Only four show something *useful*:

- **WriteHuman** — per-sentence probability with published bands (<40 / 40–60 / >60) plus a Human/Mixed/AI class breakdown and in-place re-scan.
- **Netus AI** — three-colour per-sentence classification (red / green / yellow-for-unsure). Surfacing *uncertainty* as its own colour is the most honest thing anyone in this market does.
- **GPTinf** — eight-system cross-check with per-system visibility, segment colours, and re-humanize-only-the-flagged-segments. This is the right *workflow*, badly drawn.
- **HIX Bypass** — an eight-detector badge row after each run.

The rest either show a single number, show a number only after sign-in (Phrasly), or show nothing and print a marketing claim instead (BypassGPT, Twixify). Note the circularity everyone hand-waves: these scores come from the vendor's *own* detector, which is tuned to pass the vendor's own output. Netus's own detector passing its own output is documented in report 03. Any score we display has to be defensible or it is worth less than no score at all.

### 1.3 What specifically looks vibe-coded

Cross-referencing the incumbents against the two empirical catalogues of AI-design fingerprints — Adrian Krebs scored 1,590 Show HN submissions and found 22% exhibited four or more AI-design patterns — the market hits nearly every marker:

**"VibeCode Purple."** Tailwind `indigo-500` #6366F1 → `violet-500` #8B5CF6. Walter Writes uses literal purple checkmarks. Ryne uses a gradient hero with glow SVGs.

**Inter for everything**, especially the centred hero headline, with a serif italic on one accent word.

**Uniform icon-topped feature cards on a bento grid**, 1-2-3 numbered steps, an all-caps section header above each, a stat banner row, and generated-avatar testimonial walls. BypassGPT, Humbot, Phrasly and Ryne are all four-for-four.

**Gradient orbs and glow shadows** behind the hero. **Glassmorphism** on the pricing cards. **Uniform `opacity:0, y:20` fade-up** on every section on scroll.

The key insight from that research is that human teams use these patterns too — *the fingerprint is unstated defaults applied uniformly*. That is precisely the diagnosis here: nobody made a decision, so every tool inherited the same one.

Three deeper tells, beyond aesthetics:

1. **The textarea is the whole product.** Ten of fifteen expose a plain `<textarea>` with a character counter. There is no document model, so there can be no per-sentence anything, no diff, no undo of a single edit, no track-changes. The UI is a shape constraint that reveals there is no engine behind it.
2. **Controls are named, not explained.** "Aggressive," "Enhanced," "Max," "Latest," "Ghost," "Samurai," "V7-N2," "GPTinf Steel." None of these tell you what changes or what it costs you in quality. Report 03 shows what they actually cost: Humbot's Aggressive mode alters dates and statistics; HIX Aggressive yields "awkward or hard-to-read sentences."
3. **The dark patterns are load-bearing.** Undetectable.ai sits at 3.5/5 on 967 Trustpilot reviews with a dominant complaint of post-trial charges; Humbot at 2.2/5 ("Poor") with unauthorised recurring charges and no in-app cancellation; StealthGPT at 3.9/5 with reviewers calling the cancellation flow "intentionally designed in a way that is confusing and misleading"; WriteHuman at 3.9/5 on 296 reviews with "$432 charge" complaints. Frictionless cancellation is not a nicety in this market — it is a differentiator you can put on the pricing page.

---

## 2. The good parts worth stealing

### 2.1 Craft references

**Linear** published the most useful case study in its two-parter on the 2024–25 redesign. Three stealable decisions: (a) *two optical cuts of one family* — Inter Display for headings "to add more expression… while maintaining their readability," plain Inter elsewhere; (b) *colour as three variables, not ninety-eight* — they migrated HSL → LCH ("perceptually uniform") and collapsed per-theme variables into **base colour, accent colour, contrast**, where the contrast variable auto-generates accessible high-contrast themes; (c) *timelessness by restraint* — "limiting how much chrome (blue in our case) was used." They describe the bulk of the work as aligning labels, icons and buttons vertically and horizontally. The *architecture* is worth copying; the *skin* (dark violet-tinged greys, glow shadows) is now itself a slop signal.

**iA Writer** is the strongest precedent for a serious writing tool. They built their own faces (Duospace, Quattro) rather than picking one, and their governing principle is the posture we want toward a student's prose: **"The Notebook is for you, not about us."** Their guidelines are designed to optically disappear once ink is on them.

**Rauno Freiberg's Web Interface Guidelines** is the most citable rulebook: interactions under 200ms; font weight must not change on hover (layout shift); box-shadow not outline for focus rings so it respects border-radius; inputs never below 16px on mobile; toggles take effect immediately without confirmation. His *Invisible Details of Interaction Design* contributes the rule that matters most for a diff UI: **lightweight actions may trigger mid-gesture; destructive actions only at gesture end** — Accept can be optimistic, Reject cannot.

**Emil Kowalski on animation**: springs or <300ms ease-out; animate only `transform` and `opacity`; animations must be interruptible; do not animate keyboard-driven repeated actions, because they feel sluggish; respect `prefers-reduced-motion`. Directly relevant: if a user holds `J` to walk twenty suggestions, every transition must be instant.

**Lex.page** shows the right social proof for an academic product — Berkeley, Harvard, Yale, NYT — and the right feature vocabulary: AI Feedback, keyboard-invoked AI Commands, and **Versions** ("draft variations without losing the original").

**Sudowrite** contributes one move worth stealing: the named house model. "Muse 1.5, our AI model built just for fiction" converts "we call an LLM" into a product with a version number.

**Grammarly** is the competitor to watch. It rebuilt its editor on Coda with a block-first document model and a right-hand AI sidebar, and now ships AI Grader (rubric-aligned feedback with an estimated grade), Citation Finder, Proofreader, Paraphraser, **AI Humanizer**, and an AI Detector + Plagiarism Checker in one surface. Its document domains (General / Academic / Business / Email / Creative / Casual) change suggestion strictness — the correct model for an academic-mode toggle. Its framing line is the one to beat: agents "don't overwrite your voice—they amplify it."

### 2.2 Diff and inline-suggestion conventions

The field has converged on a small vocabulary. Do not invent a new one.

| System | Insertion | Deletion | Controls live | Distinctive idea |
|---|---|---|---|---|
| Google Docs Suggesting | coloured text, underlined, per-author hue | struck through, same hue | margin card, right | *Review suggested edits* box with a **preview toggle**: "with changes" vs "without" |
| Word Track Changes | underline, per-reviewer colour | strikethrough | balloons + vertical change bars | **Four display modes**: Simple Markup (a red margin line only), All Markup, No Markup, Original; Reviewing Pane with an exact count; Next/Previous traversal |
| CKEditor 5 | **green** highlight + border | **red** highlight + border + strikethrough | responsive: wide sidebar → narrow sidebar → inline balloon | **Blue** as a third semantic for *format* changes; all driven by CSS variables |
| GitHub | green line, `+` | red line, `−` | comment thread block | **Batching** — "Add suggestion to batch" collapses N suggestions into one commit; unified/split toggle persisted; "Viewed" checkbox with a progress bar |
| Cursor / Copilot | green inline | red inline | inline widget at the hunk | Change stays in flow; no context switch |
| Notion AI | replacement text | — | toolbar under the generated block | The minimum viable AI control set: **keep / discard / try again** |

The synthesis: green/red is near-universal and you should not deviate; strikethrough is what keeps a deletion legible when colour fails (colour-blindness, print, dark mode); margin cards scale to long documents better than inline popovers, because popovers occlude the text you are being asked to judge; every system that handles real volume ships both per-change and accept-all, plus keyboard traversal.

---

## 3. The specific UI this product needs

### 3.1 Sentence-level AI-risk heatmap

The reference implementation is GPTZero's own **Advanced Scan**, and it is worth copying closely because it is what your buyer already knows. Its layout: highlighted sentences in the document, a **right-side panel of pattern categories with occurrence counts**, **filterable chips** to isolate one pattern type, **expandable cards** with natural-language explanations per instance, and click-to-focus linking a definition to its example. Its ten named patterns are: Overblown Importance, Name Dropping, Empty Commentary, Sales-pitch Tone, Phantom Experts, Dressed-up Verbs, "Not just X, but Y", Everything in Threes, Unnecessary Caveats, Chatbot Language. Patterns are validated as frequency multiples between human and AI corpora across LLM families.

Three design constraints follow from report 01 and report 06:

- **Do not use red/green for risk.** Those two colours must be reserved exclusively for diff insert/delete, or the two layers become unreadable when shown together. Use a single-hue sequential ramp (a warm amber-to-oxblood, or a neutral-to-ink density ramp) for risk, and keep green/red semantic.
- **Show uncertainty as its own state,** as Netus does. GPTZero exposes confidence categories — uncertain, moderately confident, highly confident — with a reported <1% error rate only in the "high" band. A heatmap that renders a 55% sentence with the same visual weight as a 95% sentence is lying.
- **Never claim the highlight explains the score.** GPTZero says this explicitly of its own AI Vocabulary tool: it is "NOT [tied] to our AI probability score," a document can classify human while using AI-favoured phrases, and results "should not be used to punish." That honesty is a positioning asset, not a weakness.

### 3.2 Original vs humanized diff, with per-sentence accept/reject

The right default is **Word's Simple Markup**, not All Markup: change bars in the gutter, prose still reading as prose, with escalation to full inline diff on demand and a held-key toggle to Original (Google Docs' preview toggle). Full green/red inline everywhere turns a 3,000-word essay into confetti.

Batch like GitHub: review N sentence rewrites, then apply as **one undoable transaction**. Per-edit controls follow Notion: keep / discard / **try again** — "try again" is essential here, because rejection in a humanizer usually means "this rewrite is bad," not "leave it as AI."

Keyboard: `J`/`K` to traverse, `⌘↵` accept, `⌘⌫` reject, `⌘⇧↵` accept all. Per Rauno's rule, Accept applies optimistically mid-gesture; Reject waits for gesture end.

### 3.3 Live quality and detector dials

Two numbers, side by side, always: **risk** and **quality**. The entire finding of report 03 is that these are anti-correlated — fluent tools are caught 95–100%, and the best evaders win by damaging the text. A product that shows only the risk score is structurally the same product as the incumbents. Showing both, and refusing to let one move without the other being visible, *is* the differentiator.

Use tabular numerals everywhere a number can change, and animate the dial's value, not its geometry.

### 3.4 Feature-level explanation

This is the highest-value screen nobody in the market has. The form: a named metric, this document's value, the human academic reference band, and a one-line consequence. *"Sentence-length variance: 4.1. Human academic writing runs 9–16. Your paragraphs are metronomic — that's the single strongest separator in the stylometric literature."* Report 04 supplies twenty such features ranked by effect size; report 06 supplies which edits actually move GPTZero.

Grammarly's AI Grader does the rubric-aligned version of this and it is the most defensible thing in their suite. Elicit's analogue is sentence-level citations plus published benchmarks (95% search recall, 97% abstract screening, PRISMA 2020 alignment). Consensus does it with a "Consensus Meter" and Study Snapshots naming methodology. In all three cases the credibility comes from *naming the measurement*, not from a bigger number.

---

## 4. Editor library comparison

| Library | Non-mutating decorations | Track-changes / suggestions | Bundle (gzip) | React | SSR | Yjs | Health (2026-09) |
|---|---|---|---|---|---|---|---|
| **ProseMirror** | **Best in class.** `Decoration.inline/widget/node`, `DecorationSet` in plugin state, `.map()`ed through transactions | `prosemirror-changeset` 2.4.2 (free, 3.5 kB); OSS suggest-changes plugins; CKEditor is paid | view 53 + model 14 + state 18 ≈ **86 kB** | via wrappers | manual, easy | `y-prosemirror` 1.3.7 | v1.42.3, 19.8M/wk. GitHub repos **archived Apr 2026**, moved to self-hosted Forgejo |
| **Tiptap v3** | Inherits PM decorations via `addProseMirrorPlugins` | Tracked Changes + AI Toolkit are **custom-priced add-ons, not in any plan**; `@tiptap-pro/compare` is a Business pilot | starter-kit **103 kB**, 24 deps | excellent | `immediatelyRender: false` | official extension | v3.31.2, 18.6M/wk, MIT core. 833 open issues; v3.20.3 shipped to npm without `dist/` |
| **Lexical** | **No decoration layer.** `DecoratorNode` is a node *in the document*; overlays need `$createMarkNode`, which mutates the doc, history and collab state | `@lexical/mark` only | 55 kB core | first-class | **No SSR story** (issue #4960 open since 2023) | `@lexical/yjs` | v0.50.0, **still no 1.0**; 21 of last 35 releases had breaking changes |
| **Slate** | `decorate` prop, recomputed at render; known perf cliff | none built in | 27 + 21 kB | native | ok | `@slate-yjs/core` **stale (2023)** | v0.126.4, still 0.x after ten years, 654 issues |
| **Plate** | Slate `decorate` | **`@platejs/suggestion` + `/comment` + `/diff` are MIT and free** | heavy | excellent | ok | via Slate | v53.3.9, still on Slate |
| **CodeMirror 6** | Excellent (`Decoration.mark/widget/replace/line`, `RangeSet`) | `@codemirror/merge` has real `acceptChunk`/`rejectChunk` | view 77 kB, merge 85 kB | none | poor | `y-codemirror.next` | v6.43.11. **Plain text only** |
| **Monaco** | Model decorations | DiffEditor only | ~5 MB | poor | poor | no | Wrong tool — it is a code editor |
| **Quill** | Limited; Parchment formats mutate the doc | none | 56 kB | wrapper | poor | `y-quill` | **Dormant** — last release Nov 2024 |

**Recommendation: ProseMirror, wrapped by Tiptap v3, with `@handlewithcare/react-prosemirror` for React-rendered widget decorations.** Keep the heatmap and the diff entirely in a `DecorationSet`; mutate the document only when a user accepts.

The three requirements — many overlapping coloured spans keyed to a 0–1 score, per-sentence accept/reject buttons anchored to ranges, hover cards — map exactly onto ProseMirror's three decoration primitives (`Decoration.inline`, `Decoration.widget`, `DecorationSet.find()`). The guide is explicit about the performance model: keep the set in plugin state and `map()` it, which "exploits the tree shape of the decoration set — only the parts of the tree actually touched by the changes need to be rebuilt." That is the property that lets a heatmap survive live typing across a 3,000-word document.

Lexical cannot do this without mutating the document. A 200-sentence heatmap means 200 `MarkNode`s in the EditorState, in the undo history, and in the Yjs doc. Combined with no SSR and no 1.0 after four years, it is out.

**The one real ProseMirror risk:** Marijn Haverbeke archived the GitHub repos on 2026-04-01 and shipped *Wordgard 0.1* in July 2026 — a from-scratch successor with no upgrade path. The HN thread panicked; he replied that he "went out of my way in all the announcements to stress that ProseMirror maintenance is continuing," and releases back him up (prosemirror-view 1.42.3, Aug 2026). Treat it as stable and low-velocity, and keep suggestion logic in plain ProseMirror plugins rather than Tiptap `Extension` sugar so the escape hatch stays ~200 lines wide.

**Do not buy Tiptap Track Changes.** Tiptap is Start $59 / Team $179 / Business $1,199 per month, with Tracked Changes and the AI Toolkit as *custom-priced add-ons on top of that*. Our case is single-user AI redlining, not multi-user Word-style redlining. Free alternatives that fit better:

- `@handlewithcare/prosemirror-suggest-changes` (MIT) — Google-Docs-style suggestions via four marks; `withSuggestChanges()` wraps `dispatchTransaction` and transforms transactions into suggestions; `applySuggestion(id)` / `revertSuggestion(id)` are accept/reject. Forked by BlockNote and actively maintained.
- `prosemirror-suggestion-mode` (MIT) — closer to our shape: `applySuggestion({textBefore, textAfter, textToReplace})` is designed for AI-generated replacements, with `acceptSuggestionsInRange` / `rejectSuggestionsInRange` and a pluggable hover-menu renderer.
- `prosemirror-changeset` (3.5 kB, already in the tree as `@tiptap/pm/changeset`) for a persistent revision trail. Caveat from its author: it "works with steps. If you don't have the steps that made the changes, it won't help you."

### 4.1 Diff stack

Use two layers, not one library.

**Sentence alignment:** `Intl.Segmenter(locale, {granularity:'sentence'})` → **jsdiff `diffArrays`** over sentence arrays. `diff@9.0.0` (Apr 2026, 144M/wk, 7.8 kB) is the healthiest diff package on npm. Do *not* use its `diffSentences` — the `.!?` regex breaks on "Dr." and "e.g."; jsdiff's own README recommends the `Intl.Segmenter` route.

**Intra-sentence word diff:** `@sanity/diff-match-patch` v3.2.0 (6.8 kB, TypeScript, tree-shakeable, fixes surrogate-pair index bugs). Run `cleanupSemantic()`, which "rewrites the diff, expanding it into a more intelligible format" for humans. Never `cleanupEfficiency`, which optimises op count for transmission and produces uglier diffs. Google's own `google/diff-match-patch` is archived (last push May 2024) and npm `diff-match-patch@1.0.5` dates to 2020.

**Reject:** `react-diff-viewer` (dead since 2020), `react-diff-viewer-continued` (alive but line-oriented, 50 kB), `@git-diff-view/react` (322 kB), Monaco DiffEditor, and `htmldiff-js` (emits `<ins>`/`<del>` HTML strings with **no offsets** — a dead end for per-sentence accept/reject).

**Worth stealing:** `@codemirror/merge` exports a headless `presentableDiff(a,b)` returning changes already aligned to word boundaries.

**Mapping offsets into ProseMirror.** `doc.textBetween()` is not offset-isomorphic — block joins cost two positions but one `\n`. Build an index table once by walking `doc.descendants()`, recording `{off, pos, len}` per text node and adding a synthetic `\n` at block boundaries, then binary-search it.

### 4.2 Implementation sketch

**Heatmap.** Segment with `Intl.Segmenter`, store `[{from, to, score}]` in a plugin `StateField`, emit one `Decoration.inline(from, to, {class:'risk', style:'--risk: 0.82'})` per sentence, and drive colour in CSS from the custom property. This keeps the DecorationSet cheap and lets you retheme without rebuilding it. Use *separate* DecorationSets for heatmap and diff so each toggles independently. On each transaction, `set.map(tr.mapping, tr.doc)` and recompute scores only for touched blocks, debounced off the hot path.

**Diff widgets.** For each changed sentence, `Decoration.inline` over the deletion (strikethrough) plus `Decoration.widget(pos, () => <SentenceActions/>, {side: 1, key: id, stopEvent: () => true, ignoreSelection: true})`. The `key` matters — without it ProseMirror compares widget DOM by identity and redraws constantly. Accept dispatches a `ReplaceStep` at mapped positions; reject drops the decoration.

**Hover card.** One `mouseover` handler on the editor root → `view.posAtCoords()` → `decorationSet.find(pos, pos)` → a Floating UI *virtual element* built from the DOM Range (`getBoundingClientRect` + `getClientRects`) with the `inline` middleware so multi-line sentences anchor correctly. Render the card in a portal, outside `contenteditable`.

**Supporting stack.** Vite + TanStack Start (per-route `ssr: false`) is the cleanest fit for a one-client-island product; Next.js works but requires `useEditor({immediatelyRender: false})`. **Base UI**, not Radix, for primitives — it hit 1.0 in Dec 2025, is at 1.7.0, is built by the Radix + MUI + Floating UI authors, and composes with Range-anchored hover cards because it is built on Floating UI. `motion@13` for the little motion there is; prefer CSS `@starting-style` + `transition-behavior: allow-discrete` for accept/reject enter/exit, and never animate inside `contenteditable` — animate the overlay layer.

---

## 5. Design direction: serious instrument, not AI SaaS

**The canvas is a document; the chrome is an instrument.** Serif on the writing surface, sans in the UI. The canvas should look like the artifact the student is producing.

**Canvas type.** Free options that are genuinely good: **Source Serif 4** (transitional, simplified shapes for screen; Butterick — hostile to free fonts generally — specifically endorses it), **Literata** (TypeTogether, built as the Google Play Books brand face, variable, for continuous reading), **Newsreader** (Production Type, "primarily intended for continuous on-screen reading," with an optical-size axis), **Charis SIL** (SIL's linguistics-grade family with near-complete diacritics — unbeatable if non-English quotations will ever render), **IBM Plex Serif** (a superfamily, which solves pairing by fiat). Paid, if a distinct voice is worth a licence: **Tiempos Text** (Klim), **Signifier** (Klim — Sowersby argues its details "work at all sizes, it doesn't need text and display variants"), **Lyon** (Commercial Type), **GT Sectra** (Grilli Type, drawn for the long-form magazine *Reportagen*). Avoid **Editorial New** — currently over-used in exactly the AI-adjacent branding we are differentiating from.

**Chrome type.** Inter is safe but is itself a slop signal *used alone*; pairing it against a real serif canvas defuses that. **Söhne** (Klim, Stripe's face) if budget allows. For diff gutters, version tags and model IDs: **JetBrains Mono** or Source Code Pro — mono as function, never as decoration.

**Metrics.** Butterick: body text determines everything; 15–25px on web; line spacing 120–145%; measure 45–90 characters. Target a 65ch measure at ~19px/1.6 on an off-white ground, with `font-optical-sizing: auto` and `text-wrap: pretty`.

**Colour.** A 12-step perceptual scale in OKLCH, following the Radix role assignment: 1–2 backgrounds; 3/4/5 component background, hover, pressed; 6 subtle border, 7 interactive border, 8 strong border and focus ring; 9 solid accent, 10 accent hover; 11 low-contrast text, 12 high-contrast text. This maps cleanly onto diff (insert = green 3 background, green 8 border, green 11 text). OKLCH rather than HSL for the reason Linear moved to LCH: HSL drifts in hue and saturation as lightness changes, OKLCH holds "consistent blueness." Tailwind v4 ships its default palette in `oklch` and interpolates gradients in OKLAB.

Concretely: a **warm paper ground** around `oklch(0.985 0.004 85)` — a hair of yellow, never pure `#FFF`; **ink** near `oklch(0.22 0.01 260)` rather than black; **one accent that is not purple** (deep ink blue, oxblood, or forest green all read academic); green/red reserved exclusively for diff; an amber-to-oxblood ramp for risk; zero gradients anywhere. Ship a real dark mode but do not default to it.

**Density.** What makes Linear, a Bloomberg terminal and a JSTOR PDF feel serious is the same thing: consistent small type on a tight regular vertical rhythm, hairline rules instead of cards, no decorative whitespace. Encode it: a 4px spacing grid; 28–32px row heights in chrome; **13px UI text against 19px canvas text** — that contrast is what signals instrument-versus-document; 1px borders at step 6 instead of drop shadows; tabular numerals wherever a number can change; no card unless it is genuinely a separate object. Keyboard-first is part of density, because affordances that live in a `⌘K` palette do not have to be visible.

**Credibility signals for an academic-adjacent product.** This is where design does real work:

1. **A versioned model card**, per Mitchell et al. — intended use, evaluation procedure, and benchmarked performance across demographic groups, with a date and a changelog.
2. **Address the detector-bias literature head-on.** Liang, Yuksekgonul, Mao, Wu & Zou found GPT detectors "consistently misclassify non-native English writing samples as AI-generated, whereas native writing samples are accurately identified," and warn they could "inadvertently penalize or exclude non-native English speakers." For an ESL-heavy audience, citing this paper *is* the positioning: defending honest non-native writers from a broken instrument is a categorically different product from "beat the detector."
3. **Methodology in-product, not just in marketing** — the named metric behind every edit.
4. **State limitations and refuse to promise evasion.** Grammarly frames its detector as a "window… into what could be AI-generated text in their writing before they submit," explicitly not an enforcement tool. That hedge is the defensible institutional posture.
5. **Institutional over startup social proof** — Lex leads with Berkeley/Harvard/Yale, not YC logos. Critically: *real* relationships, unlike HIX's decorative Harvard and Columbia logos.
6. **A named, versioned house model.**

---

## 6. Positioning and market

**Pricing norms.** The market clusters tightly at **$8–$20/month billed annually**, with an anchoring monthly price roughly 2× that and a toggle advertising 43–86% off. Entry points: Twixify $5, BypassGPT $6.80, Rephrasy $8.25, HIX $9.99, Phrasly $10.99, WriteHuman $12, Undetectable $5 (10K words) to $42.50 (50K). Outliers: StealthGPT ($24.99–$249.99, no free tier) and Ryne (up to $99.99). Three metering models coexist — word buckets (Undetectable, HIX), request counts × word caps × variation counts (WriteHuman), and credits (Netus, ~1 credit ≈ 10 words). "Unlimited" is near-universal at the top tier and is a churn instrument, not a plan. Lifetime deals ($199 Rephrasy, $150 Twixify) and 24-hour passes ($12.99 Rephrasy, $2 Phrasly) signal the same thing: high churn, low trust.

**Buyers.** The landing pages segment identically: students first (HIX "1,280,000+ Students," Ryne "2.3M+ students"), then bloggers/SEO/webmasters, then agencies and marketers, then ESL writers (50+ language support and per-language landing pages are universal). Humbot's repositioning to "Your All-in-One AI Study Assistant" — with math solver, quiz generator, citation generator in 40+ styles — confirms where the money is.

**Ethics and liability framing splits three ways.**

*Explicit anti-academic-dishonesty language.* StealthGPT: "Our platform is committed to promoting academic integrity and prohibits any form of academic dishonesty, including plagiarism, cheating, collusion, fabrication, and facilitating dishonest acts." WriteHuman prohibits use "to facilitate or promote academic dishonesty — including paper-writing, homework-completion, or exam-taking assistance — or any other activity that enables users to mislead educational institutions." Phrasly, §31 Ethics: "we firmly condemn the use of our tool for academic dishonesty or cheating" and "does not support using our technology to circumvent educational AI detection systems when they are used to prevent academic misconduct." Ryne §3.2 permits academic use but says "Misuse of this tool for plagiarism or dishonesty is strictly prohibited." Note the tension: all four sell detector evasion as the headline feature.

*Responsible-use guidance in the product, not the contract.* Grammarly: "In academic settings or any context where AI use is restricted or must be disclosed, using an AI humanizer to disguise the use of AI may be considered unethical and could constitute cheating or plagiarism," plus "We recommend citing AI when in doubt." QuillBot: "make sure to follow AI guidelines from your institution or workplace... Trying to pass off someone else's idea as your own, even if it has been reworded, is just as unethical as directly copying a quote."

*Silence.* **Undetectable.ai, Humbot, BypassGPT, HIX Bypass and Twixify have no academic-integrity language at all** — only boilerplate ("illegal or unauthorized purpose") and liability caps limited to amounts paid. Twixify's ToS instead says usage "must comply with the permissions and restrictions of ChatGPT" and "Twixify is not responsible for any misuse of the service by users."

**SEO landscape.** This is a programmatic-SEO market. Undetectable.ai's sitemap carries **409 URLs**; HIX Bypass **495**; Humbot **1,101**. The structure is a small English page set multiplied by ~14 locales. HIX's English set is explicitly detector-targeted: `/bypass-gptzero`, `/bypass-turnitin`-equivalents, `/bypass-originality-ai`, `/bypass-copyleaks`, `/bypass-zerogpt`, `/bypass-winston-ai`, `/bypass-writer`, `/bypass-sapling`, `/bypass-crossplag`, `/bypass-scribbr`, `/bypass-quillbot-ai-detector`, `/bypass-contentatscale-ai`, plus `/openai-watermark-remover`, plus per-language humanizer pages (`/humanize-ai/tagalog`, `/french`, `/portuguese`, `/turkish`). Competing on that ground means writing hundreds of near-duplicate pages, which is precisely how everything ends up looking generated. The alternative is to compete on the strength of a small number of pages — a public benchmark, a methodology page, a model card — which is both cheaper and consistent with the credibility positioning.

**"Quality" vs "bypass" positioning.** Everyone claims both and is measured on one. WriteHuman is the most sophisticated: it publishes a per-detector benchmark table (Copyleaks 98, Turnitin 97, GPTZero 96, **Pangram 94**) and cites a third-party ranking. Naming Pangram at all is unusual and smart, because Pangram is the detector that catches everyone. Our claim — near-100% GPTZero pass at college-level writing quality — must be published the same way: a dated benchmark, named detectors, named quality metric, sample size, and the failure cases. Report 03's finding that evasion and quality are anti-correlated across every independent test means an unqualified claim will be tested and broken publicly.

---

## 7. Recommended front-end stack and the five UI decisions that will make this not look vibe-coded

**Stack.** Vite + TanStack Start (per-route `ssr: false`) or Next.js App Router with `immediatelyRender: false`. **ProseMirror + Tiptap v3 core (MIT only, no Pro add-ons) + `@handlewithcare/react-prosemirror`** for React widget decorations. `@handlewithcare/prosemirror-suggest-changes` or `prosemirror-suggestion-mode` for accept/reject; `prosemirror-changeset` for the revision trail. Diff = `Intl.Segmenter` + `jsdiff@9 diffArrays` for sentence alignment, `@sanity/diff-match-patch` + `cleanupSemantic` for intra-sentence words. **Base UI** (not Radix, not shadcn) for primitives; Floating UI virtual elements for Range-anchored hover cards; `motion@13` sparingly plus CSS `@starting-style`. Tailwind v4 CSS-first `@theme` with a hand-built 12-step OKLCH scale. Serif canvas (Source Serif 4 or Literata), sans chrome (Inter, 13px), mono for metrics.

**The five decisions.**

1. **Kill the textarea.** The product opens as a document, not a form field. A real ProseMirror document model is what makes per-sentence heatmap, per-sentence diff, per-sentence accept/reject, single-edit undo and a revision trail possible at all. Every incumbent's `<textarea>` is a confession that there is no engine behind it; ours should be the opposite confession.

2. **Two dials, never one.** Risk and quality side by side, always visible, tabular numerals, neither able to move without the other being seen. This is the single interaction that encodes our actual thesis — that fluent tools get caught and effective tools damage prose — and it is not copyable by anyone whose backend cannot measure quality.

3. **Name the measurement, every time.** No "Aggressive," no "Ghost," no "V7-N2." Every control and every flagged sentence states the metric and the human reference band: *"sentence-length variance 4.1; human academic writing 9–16."* Modes get descriptive names tied to what they change and what they cost. This is also the credibility architecture — model card, published benchmark, cited detector-bias literature, stated limitations.

4. **Simple Markup by default, green/red reserved for diff only.** Change bars in the gutter so the essay still reads as an essay; escalate to inline diff on demand; hold a key to see the original. Risk uses a separate sequential ramp with uncertainty as its own visual state. Batch like GitHub — review N rewrites, apply as one undoable transaction — with `J`/`K`, `⌘↵`, `⌘⌫`, and keep/discard/**try again** per edit.

5. **Paper and ink, no gradients, no purple.** Warm off-white ground, near-black ink, one non-violet accent, hairline rules instead of cards, 13px chrome against 19px serif canvas at a 65ch measure, all motion ≤200ms ease-out and interruptible. No gradient orbs, no glow shadows, no bento grid, no icon-topped feature cards, no fade-up-on-scroll, no generated-avatar testimonial wall, no centred Inter hero. And on the pricing page: one-click in-app cancellation, stated plainly — in a market where the top four tools average 3.4/5 on Trustpilot for billing dark patterns, that is a product feature.

---

## 8. Sources

**Incumbent landing, app, pricing and legal pages**
- Undetectable.ai — https://undetectable.ai/ · https://undetectable.ai/pricing · https://undetectable.ai/ai-humanizer · https://undetectable.ai/ai-detector · https://undetectable.ai/terms · https://undetectable.ai/sitemap.xml
- StealthGPT — https://stealthgpt.ai/ · https://stealthgpt.ai/terms
- HIX Bypass — https://hixbypass.com/ · https://hixbypass.com/pricing · https://hixbypass.com/terms-of-service · https://hixbypass.com/sitemap.xml
- Humbot — https://humbot.ai/ · https://humbot.ai/pricing · https://humbot.ai/terms-of-service · https://humbot.ai/sitemap.xml
- BypassGPT — https://bypassgpt.ai/ · https://bypassgpt.ai/pricing · https://bypassgpt.ai/terms-of-service
- WriteHuman — https://writehuman.ai/ · https://writehuman.ai/pricing · https://writehuman.ai/ai-detector · https://writehuman.ai/terms
- Phrasly — https://phrasly.ai/ · https://phrasly.ai/pricing · https://phrasly.ai/ai-detector · https://phrasly.ai/terms
- Rephrasy — https://www.rephrasy.ai/
- Walter Writes — https://walterwrites.ai/ · https://walterwrites.ai/ai-humanizer/
- Twixify — https://www.twixify.com/ · https://www.twixify.com/pricing · https://www.twixify.com/terms-conditions
- QuillBot — https://quillbot.com/ai-humanizer
- GPTinf — https://www.gptinf.com/
- Netus AI — https://netus.ai/ · https://netus.ai/pricing
- Ryne AI — https://ryne.ai/ · https://ryne.ai/terms
- Grammarly — https://www.grammarly.com/ai-humanizer · https://www.grammarly.com/docs · https://support.grammarly.com/hc/en-us/articles/115000091472
- Undetectable.ai (Wikipedia) — https://en.wikipedia.org/wiki/Undetectable.ai
- Originality.ai review — https://originality.ai/blog/undetectable-ai-review
- Trustpilot — https://www.trustpilot.com/review/undetectable.ai · /humbot.ai · /stealthgpt.ai · /writehuman.ai

**Detector product surfaces (reference implementations)**
- GPTZero — https://gptzero.me/ · https://gptzero.me/technology · https://gptzero.me/pricing · https://gptzero.me/news/ai-patterns/ · https://gptzero.me/ai-vocabulary
- Elicit — https://elicit.com/ · Consensus — https://consensus.app/

**Editor and diff libraries**
- ProseMirror — https://prosemirror.net/docs/ref/ · https://prosemirror.net/docs/guide/ · https://marijnhaverbeke.nl/blog/wordgard-0.1.html · https://news.ycombinator.com/item?id=48772573 · https://news.ycombinator.com/item?id=48774987
- Tiptap — https://tiptap.dev/pricing · https://tiptap.dev/docs/editor/getting-started/install/nextjs · https://github.com/ueberdosis/tiptap/issues/7613
- Lexical — https://lexical.dev/docs/concepts/decorators · https://github.com/facebook/lexical/issues/4960
- CodeMirror 6 — https://codemirror.net/docs/ref/#view.Decoration
- Suggestion plugins — https://github.com/handlewithcarecollective/prosemirror-suggest-changes · https://github.com/davefowler/prosemirror-suggestion-mode
- Diff — https://github.com/kpdecker/jsdiff · https://github.com/sanity-io/diff-match-patch · https://github.com/google/diff-match-patch
- Base UI — https://base-ui.com/react/overview/about · Floating UI virtual elements — https://floating-ui.com/docs/virtual-elements
- TanStack on RSC — https://tanstack.com/blog/we-stopped-using-rsc-on-tanstack-com

**Suggestion / diff UX conventions**
- Google Docs — https://support.google.com/docs/answer/6033474
- Microsoft Word — https://support.microsoft.com/en-us/office/track-changes-in-word-197ba630-0f5f-4a8e-9a77-3712475e806a
- CKEditor 5 — https://ckeditor.com/docs/ckeditor5/latest/features/collaboration/track-changes/track-changes.html
- GitHub — https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/reviewing-proposed-changes-in-a-pull-request · .../incorporating-feedback-in-your-pull-request
- Notion AI — https://www.notion.com/help/notion-ai-faqs

**Craft and design direction**
- Linear — https://linear.app/blog/a-design-reset · https://linear.app/blog/how-we-redesigned-the-linear-ui
- Rauno Freiberg — https://interfaces.rauno.me/ · https://rauno.me/craft/interaction-design
- Emil Kowalski — https://emilkowal.ski/ui/great-animations
- iA — https://ia.net/topics/a-typographic-christmas · https://ia.net/topics/ia-writer-in-paper
- Lex — https://lex.page/ · Sudowrite — https://www.sudowrite.com/
- Grammarly redesign — https://techcrunch.com/2025/08/18/grammarly-gets-a-design-overhaul-multiple-ai-features/
- AI design slop — https://www.adriankrebs.ch/blog/design-slop/ · https://news.ycombinator.com/item?id=47864393 · https://github.com/febbhav/signs-of-ai-design · https://www.vibecheck.fail/
- Typography — https://practicaltypography.com/typography-in-ten-minutes.html · https://practicaltypography.com/free-fonts.html · https://fonts.google.com/specimen/Source+Serif+4 · /Literata · /Newsreader · /Charis+SIL · /IBM+Plex+Serif · /JetBrains+Mono · https://rsms.me/inter/ · https://klim.co.nz/retail-fonts/tiempos-text/ · https://klim.co.nz/blog/signifier-design-information/ · https://klim.co.nz/retail-fonts/soehne/ · https://commercialtype.com/catalog/lyon · https://www.grillitype.com/typeface/gt-sectra · https://pangrampangram.com/products/editorial-new · https://vercel.com/font
- Colour and systems — https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale · https://tailwindcss.com/blog/tailwindcss-v4 · https://jakub.kr/components/oklch-colors · https://vercel.com/geist/introduction

**Credibility / academic literature**
- Mitchell et al., *Model Cards for Model Reporting* — https://arxiv.org/abs/1810.03993
- Liang, Yuksekgonul, Mao, Wu & Zou, *GPT detectors are biased against non-native English writers* — https://arxiv.org/abs/2304.02819
- International Center for Academic Integrity — https://academicintegrity.org/resources/blog

**Internal**
- `03-commercial-humanizers-and-evasion-papers.md` (pricing, modes, independent test evidence, DAMAGE/Pangram tiers)
- `01`, `04`, `06` (detector mechanics, feature catalogue, what flips GPTZero)
