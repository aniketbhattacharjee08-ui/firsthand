# 25. What makes a UI look human: X posts and the resources they point at

Date: 2026-09-08
Status: research note, read-only, no code changed
Question from the product owner: "scroll through X for posts by humans about websites and libraries that can be referenced to make vibe coded websites look better and have an optimal human looking UI."

## Method, and what I could not access

I cannot log into X. I used web search restricted to `x.com`, `xcancel.com`, `nitter.*`, `threads.com` and `bsky.app`, plus open searches for the phrases people use ("vibe coded", "AI slop", "deslopify", "looks AI generated"). For each X post I tried to open the post directly, then via xcancel, nitter.net, nitter.poast.org, nitter.tiekoetter.com, nitter.privacydev.net, sotwe and twstalker.

Results of that:

- `x.com` post pages return HTTP 402 to a fetcher without a login. Not one opened.
- `xcancel.com` serves a browser-verification page. Nothing opened.
- `nitter.net` is a notice page: Nitter received cease-and-desist letters from X Corp on 2026-08-24 and instances were offline at the time of this research. `nitter.poast.org` does not resolve. `nitter.tiekoetter.com` returns "Access Denied" from Anubis. `nitter.privacydev.net` refuses connections.
- `sotwe.com` and `twstalker.com` return 403.
- Threads post pages open, but only the top post text and counts. Replies are not in the served HTML.
- Medium and UX Planet return 403 to the fetcher.

So for every X post below the text I have is the excerpt that the search engine indexed (usually the first 200 to 300 characters of the post or thread) plus whatever articles quoted it. Where an article quoted a post at length I say so. I did not invent any post text. Posts marked "not opened" means I saw the URL and an indexed excerpt only.

The article layer (blogs, dev.to, GitHub READMEs, Hacker News) opened fine and is where most of the specific counts and tool names come from. Several of those articles are themselves summaries of X and Reddit threads, and one (the `vibecoded-design-tells` repo) is a frequency analysis of 3.2 million Reddit posts. I treat those as secondary sources and count them separately from posts.

Total sources read or excerpted: 29 posts (27 X, 2 Threads) and 25 articles or repos.

## (b) Posts and threads found

Dates are as shown in the search result or article. "n/a" means no date was visible. Status: "excerpt" means indexed excerpt only, post page not opened; "quoted" means an article I did open quoted the post.

| # | Author | Date | Advice in one line | URL | Status |
|---|--------|------|--------------------|-----|--------|
| 1 | @yuwen_lu_ | n/a (2026) | "Signs of vibe coded UI": nested cards, purple gradients, glass panels, generic feature grids, vague copy, repeated badges, weak hierarchy; "glassmorphism is becoming the new purple gradient", especially glass plus gradient plus a 1px light border | https://x.com/yuwen_lu_/article/2041187936738447565 | excerpt, plus quotes in vibemole and makeuseof |
| 2 | @pallavi_benawri | n/a (2026) | "deslopify your vibe coded apps": no eyebrow text, no bounce animations, no icons in containers, fewer icons, no emojis, cards sparingly, fewer badges and dot indicators, corners not over-rounded, primary buttons a reasonable size ("llms like huge buttons"), one consistent button style ("llms like to invent 5 buttons for one page"), fonts and colours chosen to fit the product | https://x.com/pallavi_benawri/status/2068408171484639311 | excerpt |
| 3 | @om_patel5 | n/a (2026) | "HOW TO VIBE CODE BEAUTIFUL UI": 1 sketch a wireframe in Excalidraw and tell the model "follow this structure exactly"; 2 screenshot one section you like from Dribbble or Mobbin and say "copy this style"; 3 give a mood board for colour; 4 define a design system (colours, type, spacing) before building | https://x.com/om_patel5/status/2025705776434520284 | excerpt |
| 4 | @MyWestLord | n/a (2026) | "Every vibe coded site has the same tell: a purple to white gradient, Inter font, 4 cards in a grid, 1 weak hover state." Then 45 SKILL.md style rule files that override model defaults before CSS is written | https://x.com/MyWestLord/article/2068064010600108267 | excerpt |
| 5 | @adamwathan (Tailwind creator) | Aug 2025 | "I'd like to formally apologize for making every button in Tailwind UI `bg-indigo-500` five years ago, leading to every AI generated UI on earth also being indigo." Over 1M views | https://x.com/adamwathan/status/1953510802159219096 | quoted in full by prg.sh, 925studios, dev.to alanwest |
| 6 | @PovilasKorop | n/a (2026) | "If you've used Claude frontend-design skill, you probably recognize these designs. I see A LOT of them in new startups built in 2026." Names the new default: Fraunces, editorial serif, warm brown italics. Calls it "2026 AI slop" | https://x.com/PovilasKorop/status/2050929501656268871 | excerpt |
| 7 | @nityeshaga | n/a | Anthropic's frontend-design skill "is just one file with 42 lines of instructions that read like the type of memo a frontend lead would write for their team" and that is enough to stop slop | https://x.com/nityeshaga/status/1989323006288830687 | excerpt |
| 8 | @feitong_yang | n/a (2026) | Found the frontend-design SKILL.md behind Claude's better web artefacts; it "guides creation of distinctive, production-grade frontend" | https://x.com/feitong_yang/status/2025104762035970267 | excerpt |
| 9 | @emilkowalski_ | Sep 2025 | Use `ease-out` for enter and exit animations; it accelerates at the start and feels snappier than `ease-in` at the same duration | https://x.com/emilkowalski_/status/1970144111261868487 | excerpt; the linked article opened |
| 10 | @emilkowalski | Sep/Oct 2025 | "7 Practical Animation Tips" (scale buttons to 0.97 on :active, never animate from scale(0), no delay on subsequent tooltips, ease-out for enter/exit, origin-aware transforms, keep under 300ms, small blur to hide seams) | https://x.com/emilkowalski/status/1973054048883351612 and https://emilkowal.ski/ui/7-practical-animation-tips | excerpt; article opened |
| 11 | @emilkowalski | n/a (2026) | His `/emil-design-eng` skill for coding agents has "more than 100k installs"; design engineering rules as a skill file | https://x.com/emilkowalski/status/2069742622252445866 | excerpt |
| 12 | @AdhamDannaway | Oct 2025 | Repost of the 7 tips with the ease-out vs ease-in dropdown comparison | https://x.com/AdhamDannaway/status/1974118742721470850 | excerpt |
| 13 | @shao__meng | n/a (2026) | Summary of Emil Kowalski's "Skills for Design Engineers": UI and animation principles packaged so Codex, Claude Code and Cursor write UI with taste | https://x.com/shao__meng/status/2072484635955900792 | excerpt |
| 14 | @101babich (Nick Babich) | May 2026 | "How To Spot AI-Generated Design": reused SaaS structure (centred hero, interface preview, feature grid, pricing, testimonials); oversized headings, gradients, soft shadows and glows, giant radii, safe type; copy that "sounds polished but says almost nothing" | https://x.com/101babich/status/2057097456857330089 | excerpt; UX Planet article 403 |
| 15 | @101babich | Apr 2024 | OKLCH palette generator: palettes that match perceived lightness, supported in modern browsers | https://x.com/101babich/status/1777294490283204671 | excerpt |
| 16 | @pie6k | n/a (2026) | In OKLCH you can move the hue slider at random and the colours stay in harmony | https://x.com/pie6k/status/2025505952124772854 | excerpt |
| 17 | @evilmartians | Oct 2022 | OKLCH for declaring colours plus native colour transforms in CSS (Sitnik article) | https://x.com/evilmartians/status/1585322808858517504 | excerpt |
| 18 | @radix_ui | Mar 2024 | Custom palettes land at the same APCA contrast ratios as Radix Colors scales; you will likely reproduce a Radix scale anyway | https://x.com/radix_ui/status/1771868155532468570 | excerpt |
| 19 | @shadcn | Jun 2025 | Thread on Radix, component libraries and shadcn/ui: "with Radix receiving fewer updates, it's a conversation worth having" | https://x.com/shadcn/status/1936082723904565435 | excerpt; the July 2026 changelog making Base UI the default opened |
| 20 | @marctaule | Sep 2024 | Developer resources list: GSAP for animating JS elements, Lenis as a lightweight smooth-scroll library, and others | https://x.com/marctaule/status/1841025310319366187 | excerpt |
| 21 | @filipz | 2026 | Stack for a Codrops piece: three.js, GSAP, Lenis, Blender via MCP, Webflow, coded with Claude | https://x.com/filipz/status/2079941027222147498 | excerpt |
| 22 | @froessell | n/a (2026) | "How I Build Apps That Don't Look Vibecoded": narrow scope with an LLM first, write a short PRD, then build; "AI can make beautiful designs, but it needs help and proper guidance" | https://x.com/froessell/article/2023339023381467556 | excerpt |
| 23 | @raunofreiberg | n/a (2026) | On Josh (jhey) Comeau's "Interface Craft": the reference for interface detail work | https://x.com/raunofreiberg/status/2067588848339603677 | excerpt |
| 24 | @MengTo | Jul 2025 | Tips after soloing an app to 15k MRR: "don't just vibe code, vibe design as well" | https://x.com/MengTo/status/1943717847236325519 | excerpt |
| 25 | @uxlinks | Nov 2024 | UI spacing cheat sheet built on the 8-point grid | https://x.com/uxlinks/status/1854074418512687481 | excerpt |
| 26 | @kadircalik | May 2026 | Five tells: purple gradient hero, three equal cards, Inter at weight 700, everything centred, gradient "Most popular" pricing pill. Fix: headline at weight 500 ("500 reads considered, 700 reads trying"), one primary plus two secondaries, at most three accent colours used at most three times each | https://rottoways.com/blog/fix-ai-slop-website (his own site, credits @kadircalik) | article opened |
| 27 | @jasonlk | 2025 | Designers who can make AI-generated platforms "look truly custom" should command premium pricing | https://xcancel.com/jasonlk | excerpt |
| 28 | @alexmacgregor__ (Threads) | 2026-01-12 | "99% of vibe coded websites look like this" with an image; 1.3K likes, 217 comments, 156K views. Image and replies not retrievable | https://www.threads.com/@alexmacgregor__/post/DTb2jHAk2OG | post opened, replies not |
| 29 | @huersestudios (Threads) | 2026-06-08 | "how to make vibecoded websites not look vibecoded?" 61 replies, not retrievable | https://www.threads.com/@huersestudios/post/DZUya-KAJS5/ | post opened, replies not |

Also seen but not counted as advice: prompt-selling posts by @viktoroddy, @alex_prompter, @aiedge_ and @shushant_l. They are useful as negative evidence. Their "premium" prompts ask for dark mode, glassmorphism, "glowing luminescent gradients in deep blue, purple, and emerald", parallax, and a `Lucide Sparkles` icon in amber next to the words "The Future Is Now". That is the recipe the design posts above are warning against.

Articles opened and used for counts (25): Fountain Institute "7 Signs a UI Has Been Vibe Coded" (Jeff Humble, Apr 2026); Developers Digest "16 Patterns" (2026-04-22); impeccable.style/slop (Paul Bakaus); 925studios "AI Slop Fonts and Gradients" (2026-06-14) and "AI Slop Web Design Guide" (2026-03-19); prg.sh "Why Your AI Keeps Building the Same Purple Gradient Website" (Oct 2025, updated Aug 2026); SmoothUI "AI Design Slop" (2026-06-24); The Crit "Vibe Coding Design Guide" (Jun 2026); Creative Mind Habits (2026-06-21); dev.to kiwibreaksme StyleSeed (Apr 2026); VibeMole (2026-07-01); Monet "7 Tips" (2025-12-11); Claude Code HQ "Unslop UI" and the JCarterJohnson `vibecoded-design-tells` README; bswen anti-patterns (2026-03-20); aiskill.market "Banning Inter" (2026-06-07); AIToolPick 30-point checklist (Apr 2026); Bruvora typography (2026-07-30); dev.to alanwest x2 (Mar and May 2026); Rottoways (2026-05-05); Battlecat (Nov 2024); Sinton Agency (2026-09-08); Hacker News 47865795 "slop fonts"; shadcn July 2026 changelog; motion.dev quick start; Radix Colors installation page.

## (c) Tells of a vibe-coded UI, consolidated

Count = number of distinct sources (posts plus articles, out of 54) that name the tell. Counts are from my reading of the excerpts and articles above, so treat them as approximate.

| Tell | Sources | Notes |
|------|---------|-------|
| Purple, indigo or blue-to-purple gradient, especially as hero background or button | 24 | The single most cited tell. Root cause traced to Tailwind UI's `bg-indigo-500` default (Wathan, post 5). Developers Digest gives it a name, "VibeCode Purple". |
| Inter (or Roboto, Arial, Open Sans) as the only typeface, at default weights | 19 | Not "Inter is bad". The tell is that nobody chose anything. Kadir Calik adds weight 700 headlines as a sub-tell. |
| Centred hero plus a row of three (or four) identical feature cards, icon on top, two lines of text | 15 | Nick Babich, MyWestLord and 925studios all describe the same skeleton. |
| Generic copy: "Build faster. Ship smarter.", "Seamless", "Unlock", "Transform your X" | 11 | Babich: "sounds polished but says almost nothing". VibeMole: "specific language beats polished language". |
| Glassmorphism, neon glow, radial halos, dark mode with glowing accents | 10 | yuwen_lu_: glass plus gradient plus 1px light border "is mostly certain that it is AI". |
| Emoji used as icons (rocket, sparkle, lock) in nav, headings or cards | 8 | |
| Over-rounded or uniformly rounded corners (`rounded-2xl` everywhere, pill everything) | 7 | |
| Stock or AI illustrations, fake dashboards, fake testimonials, fake "trusted by" logos | 7 | |
| Cards inside cards, everything wrapped in a card | 6 | Fountain Institute: use whitespace, proximity and type instead. |
| Badges, eyebrow labels and pills above the H1 | 6 | Pallavi: "no eyebrow text". |
| Animation spam: bounce or elastic easing, hover scale on everything, auto-scrolling marquees, fade-up on every section | 6 | |
| Monotonous equal spacing, too much padding, no rhythm | 6 | |
| Dark mode by default with medium-grey body text that barely passes contrast | 5 | |
| Gradient text on headings (`bg-clip-text text-transparent`) | 5 | |
| The 2026 replacement default: cream or beige background, Instrument Serif or Fraunces display, sage or forest green accent | 5 | PovilasKorop (post 6), Claude Code HQ lists it as tell number 0, Developers Digest, impeccable, HN thread. This is what Anthropic's own frontend-design skill produces. |
| Icons in tinted square containers ("icon tiles") | 4 | |
| Coloured left or top border on cards as decoration | 3 | |
| Pulsing status dots and meaningless dot indicators | 3 | Pallavi, Fountain Institute, impeccable. Fountain's rule: every indicator must map to a defined state. |
| Stat banner rows, numbered 1-2-3 step sequences, bento grids | 3 | |
| All-caps section labels, tiny label text | 2 | |
| Missing empty, loading and error states | 2 | |
| Accessibility gaps (focus states, contrast, alt text) | 3 | SmoothUI cites CHI 2025 work showing AI assistants "systematically generate inaccessible markup". |
| Oversized primary buttons; five different button styles on one page | 1 | Only Pallavi, but it is a concrete and checkable rule. |
| "AI" sparkle iconography | 2 | Named directly only in the prompt-selling posts and Claude Code HQ (sparkle emoji). The Lucide `Sparkles` icon appears in the recipes people are copying. |
| Justified text, line length over 75 characters, tight line height, skipped heading levels | 1 | impeccable's checklist; general typography hygiene that AI output often fails. |

Two things people disagree on. First, the fonts. The HN thread (47865795) pushes back on calling Space Grotesk, Instrument Serif, Geist, Syne and Fraunces "slop fonts": "lots of these are used by AI because they're good." The consensus reply is that the font is not the tell, the absence of a decision is. Second, shadcn defaults. Developers Digest and Monet recommend shadcn as a base; VibeMole and Claude Code HQ list unmodified shadcn tokens as the number one visual tell in Reddit complaints. Both are right: the base is fine, the untouched tokens are the tell.

## (d) What human-designed UIs do instead

Consolidated from the same sources. Ordered by how often it comes up.

1. Decide, then commit. One typeface pairing chosen for a reason, one accent colour, one layout idea repeated. Creative Mind Habits: "Taste isn't a gift. It's just noticing the defaults and overriding them." Claude Code HQ: "A tell is an unspecified default, not a banned colour."
2. Pick a display face and a body face and give the headline a point of view. Pairings named across sources: Fraunces or Instrument Serif over Geist or IBM Plex Sans; Clash Display over Satoshi; Monument Extended over General Sans; Editorial New over Söhne or Inter Tight; Space Grotesk headings with Instrument Sans body. Warning from post 6 and the Claude Code HQ data: Fraunces plus cream plus green is now itself a default. If you use a serif, do not use that combination.
3. Headlines at weight 500 or 600, not 700 (Kadir Calik). Tighter tracking on display sizes, looser line height on body (Bruvora). Body line length under 75 characters (impeccable).
4. One accent colour. If two things compete to be the accent, one becomes neutral (jacob perks via search excerpt; Rottoways caps it at three accents used three times each). Use OKLCH to build the scale so lightness is perceptual (posts 15, 16, 17). Radix Colors gives 12-step scales that already hit APCA targets (post 18).
5. Solid fills. At most one restrained gradient on the whole page, and only if it means something (Claude Code HQ, 925studios, Creative Mind Habits).
6. Left-align body text. Grids with edges, not one centred column (Creative Mind Habits, AIToolPick). Break symmetric three-card rows into one primary and two secondaries (Rottoways) or asymmetric CSS Grid (alanwest).
7. Fewer containers. Group with whitespace, proximity and type size before reaching for a card, a border or a badge (Fountain Institute, Pallavi, impeccable).
8. Real icon set, one family, one weight, one size. Lucide, Heroicons or Phosphor. No emoji (8 sources). Fewer icons overall (Pallavi).
9. Real content. Real screenshots, real numbers, real names. No lorem ipsum, no fake testimonials, no undraw-style blobs (7 sources). Monet: real content also makes the model size and space things correctly.
10. Motion that means something. Enter and exit with ease-out, under 300ms, origin-aware, no bounce, no hover scale on cards, honour `prefers-reduced-motion` (Emil Kowalski posts 9 to 12, Claude Code HQ, Pallavi).
11. Design the empty, loading and error states. That is where product apps differ from landing pages (SmoothUI, Creative Mind Habits).
12. Buttons at a reasonable size, one primary style, one secondary style, and stop there (Pallavi).
13. Write the rules down. Every source that got results did it by putting decisions in a file the model reads: Anthropic's 42-line frontend-design SKILL.md (posts 7, 8), Emil Kowalski's skills (posts 11, 13), MyWestLord's 45 skills (post 4), StyleSeed's 69 rules (dev.to), a DESIGN.md (Developers Digest), Tailwind config constraints (Monet), ESLint rules that ban `rounded-2xl` and gradient defaults (alanwest).
14. Show the model, do not describe. Wireframe in Excalidraw, screenshot one section from Mobbin or Dribbble, give a mood board (Om Patel post 3, Battlecat, The Crit).
15. Iterate by critique. The Jola Gil piece (title only, 403) and SmoothUI both argue taste enters at the evaluation step, not the first prompt.
16. Check contrast and focus states against WCAG AA before shipping (SmoothUI, Developers Digest, Sinton).

## (e) Libraries and tools recommended, ranked

"No build" means the tool can be used from a static page with a `<link>` or `<script type="module">` from a CDN, or is a browser tool that outputs plain CSS values. Sources = distinct posts or articles that recommend it.

| Rank | Tool | For | No build step? | Sources | Notes |
|------|------|-----|----------------|---------|-------|
| 1 | Rule files for the model: Anthropic frontend-design SKILL.md, emil-design-eng, unslop-ui, StyleSeed, DESIGN.md | Overriding model defaults before code is written | n/a (prompt artefact) | 11 | The most recommended "tool" is not a library. Caveat from post 6: a widely shared skill file becomes a new default. |
| 2 | shadcn/ui (now on Base UI by default since July 2026, Radix still supported) | Accessible React components to restyle | No. React plus a bundler. | 7 | Recommended as a base and flagged as the number one tell when left on default tokens. Not usable here. |
| 3 | Google Fonts, with named alternatives to Inter: DM Sans, Plus Jakarta Sans, Space Grotesk, Manrope, Sora, Bricolage Grotesque, Playfair Display, Crimson Pro, JetBrains Mono | Type with a decision behind it | Yes, `<link>` | 6 | Space Grotesk and Fraunces are now on the slop lists too. |
| 4 | Lucide icons | One consistent SVG icon family | Yes. Vanilla `lucide` package ships a UMD build and `createIcons()`; I could not fetch the exact CDN line this session. Copying SVGs is also fine. | 5 | VibeMole counts Lucide overuse as a tell. Use few icons. |
| 5 | Heroicons | Same | Yes, copy the SVG | 4 | |
| 6 | Motion (formerly Framer Motion; Motion One for vanilla) | Enter, exit and scroll animation | Yes, verified: `import { animate, scroll } from "https://cdn.jsdelivr.net/npm/motion@latest/+esm"` or a UMD `dist/motion.js` script. Pin a version. | 4 | This product already loads it from esm.sh. |
| 7 | GSAP | Timeline and scroll-driven animation on marketing sites | Yes, CDN UMD | 4 | Overkill for a one-screen tool. |
| 8 | Fontshare (Indian Type Foundry) fonts: Satoshi, Clash Display, Cabinet Grotesk, General Sans | Free display and body faces that are not Inter | Yes, hosted CSS link. Not verified this session; `api.fontshare.com` root only returned a status JSON. | 4 (by font name) | No source said "Fontshare" by name; four named its fonts. |
| 9 | Geist (Vercel) | Body sans alternative to Inter | Yes, Google Fonts or npm | 5 for, 3 against | Now on the slop lists at impeccable and Claude Code HQ. |
| 10 | OKLCH pickers and generators (oklch.com, Babich's generator, CSS `oklch()` directly) | Building a perceptual colour scale from one hue | Yes, browser tools or plain CSS | 3 | Posts 15, 16, 17. |
| 11 | Lenis | Smooth scroll on long pages | Yes, CDN | 3 | Irrelevant to a single-screen tool. |
| 12 | Radix primitives / Base UI / Ark UI | Headless accessible components | No. React (Ark also Vue, Solid, Svelte). Zag.js from the Ark team has a vanilla adapter but no source mentioned it. | 3 / 2 / 1 | |
| 13 | Radix Colors | 12-step colour scales with known contrast | Yes, verified: `<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@radix-ui/colors@3.0.0/gray.css">`. Docs say CDN is for prototyping; copy the CSS into your file for production. | 2 | |
| 14 | Mobbin, Dribbble, Awwwards, 21st.dev as reference screenshots | Show the model a section to copy | n/a | 3 | |
| 15 | Excalidraw wireframe first | Structure before style | n/a | 2 | |
| 16 | Phosphor icons | Icon family with weight variants | Yes, CDN or web component | 2 | |
| 17 | impeccable.style (site and Chrome extension) | Audit checklist of slop tells | Yes | 3 | |
| 18 | tweakcn | shadcn theme token generator | Outputs CSS variables | 1 | |
| 19 | easings.co | Custom easing curves | Yes | 1 | |
| 20 | 8-point spacing grid | Consistent spacing | n/a | 1 | |
| 21 | Coolors, Adobe Color | Palette pickers | Yes | 1 | |
| 22 | Playwright plus a DOM scorer, `devibe_scan.py` | Automated slop scoring | n/a | 2 | |

Asked about but not found in any post or article I read: anime.js, Rive, Lottie, Realtime Colors, Huemint, Open Props, Pangram Pangram, Ark UI as a recommendation (only in comparison articles). Absence here means I did not find a source, not that the tool is bad.

## (f) Applied to this product

The product is one editor, one primary button, one verdict, details folded, served as static files with vanilla JS and no bundler (`web/index.html`, `web/styles.css`, `web/app.js`, `web/anim.js`). Current choices: Literata for the draft and headings, Schibsted Grotesk for controls, one white sheet on a grey ground, one accent `#2540b3`, semantic verdict colours, radii of 4px and 8px, 180ms and 320ms transitions with a decelerating cubic-bezier, Motion One and AutoAnimate loaded as ESM from a CDN with a failsafe, reduced motion respected.

Measured against the lists above, the front end is already on the right side of most tells: no gradient, no cards-in-cards, no emoji, no feature grid, real copy, left-aligned body, 68ch line length, light mode. The recommendations that still apply:

1. Keep Literata and Schibsted Grotesk. Neither appears on any slop list I found. Do not move to Fraunces. The design memory for this project names Fraunces as part of "our identity", but post 6, Claude Code HQ (tell number 0), Developers Digest and the HN thread now list Fraunces plus cream plus green as the 2026 default. Sources: posts 6, 10; Developers Digest; Claude Code HQ.
2. Audit the accent against the indigo cluster. `#2540b3` is a deep blue, not Tailwind's indigo-600 (`#4f46e5`) or violet, so it passes. Rebuild its hover and wash steps in OKLCH so lightness steps are perceptual, and consider borrowing a Radix `blue` or `indigo` 12-step scale for the wash and line colours rather than hand-picked hexes. No build needed: copy the CSS from the Radix Colors CDN file into `styles.css`. Sources: posts 15 to 18; alanwest.
3. Reduce button styles to two. `styles.css` has `.btn`, `.btn-primary`, `.btn-quiet` and `.btn-link`. Pallavi's rule is one primary style and one secondary; the quiet and link variants should visually collapse into the secondary. Keep the primary at text size, not oversized. Source: post 2.
4. Remove decorative pulsing. `styles.css` has a `pulse` keyframe on the health pill and on status dots, and a `ringspin` on unknown-progress stages. Pulsing dots are a named tell in three sources. The stage rings map to real states, so keep them, but make the health pill static once the service answers, and drop the pulse on anything that is not actively waiting. Sources: post 2; Fountain Institute; impeccable.
5. Motion timing. Kowalski's numbers: enter and exit under 300ms with ease-out, no scale from 0, origin-aware transforms. `anim.js` uses 280ms and 450ms with a decelerating curve. The 450ms entrance is over the line; bring it to 300ms or under. The 1100ms `sentflash` is an attention cue, not an enter animation, so it is fine. Keep the CDN Motion import; motion.dev's own docs endorse the `+esm` CDN path, but pin a version instead of `@10` floating. Sources: posts 9, 10, 12; motion.dev quick start.
6. Keep icons to nearly none. The page has one inline SVG arrow. That is correct. If more are ever needed, pick one family (Lucide or Heroicons), one stroke weight, one size, pasted as inline SVG. No icon tiles, no sparkles. Sources: posts 2, 14; 8 sources on emoji icons.
7. Copy stays specific and honest. The verdict note ("A free local estimate of what GPTZero would say. Check the final draft with GPTZero itself.") and the empty-state text are the kind of copy every source asks for. Keep the rule: no "seamless", "unlock", "transform", no eyebrow label above the wordmark, no badges other than the single health pill. Sources: posts 1, 2, 14; 11 sources on generic copy.
8. Write the rules down where the model reads them. Every source that improved output did so with a rules file. The `web/AUDIT.md` and the header comment in `styles.css` already state the palette and type. Turn them into a short DESIGN.md (fonts, the five colours, the two radii, the two durations, "two button styles", "no gradients, no cards, no emoji, no pulse") and point any agent that edits `web/` at it. Sources: posts 4, 7, 8, 11, 13; Developers Digest; dev.to StyleSeed; alanwest.

Not applicable here: shadcn, Radix or Base UI primitives (React, bundler), GSAP and Lenis (no long scroll), dark mode work (stay light), asymmetric marketing grids (there is no marketing page).
