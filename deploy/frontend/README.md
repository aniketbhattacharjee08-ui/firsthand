# Account module for the ReadsHuman page

Sign-in by email link, the credit balance, buying packs, the account panel
(history, API keys, sign out, delete). One script, one stylesheet, no build
step, no framework, ES5 like `app.js`. It wraps `window.fetch`, so `app.js`
needs no edits to get the 401, 402, 503 and 429 behaviour.

Backend: `src/humanizer/billing/routes.py` and `gate.py`; the contract is in
`deploy/README.md` under "Frontend contract" and "Concurrency and refunds".

## Files

- `account.js`: the module. Copy to `web/account.js`.
- `account.css`: its styles, all under `.rha-`. Copy to `web/account.css`.
- `demo.html`: opens from disk and fakes the whole API so every state can
  be clicked through. Not shipped.

## Wiring into web/index.html

Add the stylesheet after `styles.css` in the head:

    <link rel="stylesheet" href="account.css?v=202609092346">

Add the script next to `app.js` at the end of the body, before it:

    <script src="account.js?v=202609092346" defer></script>
    <script src="app.js?v=202609092346" defer></script>

Use the same cache-busting `?v=` the page uses. Deferred scripts run in
document order, so putting `account.js` first guarantees `window.fetch` is
wrapped before `app.js` makes any request. After `app.js` also works,
because `app.js` looks up `fetch` at call time and its first calls (health)
are not gated, but first is simpler to reason about.

## What it does

- On load, `GET /api/billing/health` and `GET /api/me`. If `paywall` is
  false nothing is rendered; the public API still works.
- Mounts a pill into `.top-tools` (or a fixed top-right box if that is
  missing). Signed out: one Sign in button. Signed in: the email (clipped
  at 24ch), the balance in `.num`, and an Account button.
- Sign-in sheet: email in, `POST /api/auth/request-link`, then "Check your
  email" naming the address and the link's lifetime. Inline messages for
  400 `bad_email` and `blocked_email`, 429 `signup_limit` and rate limits,
  502 mail failures. In development mode (`dev_link` in the response) the
  link is offered on the sheet.
- Packs sheet: `GET /api/billing/packs`, the balance, and the blocked
  request's need ("This rewrite needs 3 credits; you have 1"). Buy posts
  `POST /api/billing/checkout {pack}` and navigates to `url`. If `stripe` is
  false it says purchases are not available yet.
- Account panel: email, credits, Buy credits, history from
  `GET /api/billing/ledger` (date, delta, kind, ref), API keys (list, create
  with a label, the key shown once with Copy, revoke), Sign out, Delete
  account behind typing "delete", links to `/legal/terms.html` and
  `/legal/privacy.html`.
- Fetch interception for same-origin `/api/` requests (or `window.API_BASE`
  or the page's `?api=` override): reads `X-Longhand-Credits-Balance` and
  updates the pill; 401 `login_required` opens sign-in; 402
  `insufficient_credits` opens packs with `needed` and `balance`; 503 `busy`
  shows "The rewriter is busy, retrying in N s" and retries once after
  `retry_after_s`, resolving the caller's promise with the retry; 429 shows
  a toast. The body is cloned before reading, so the caller still gets it.
  Streams (`text/event-stream`) are never read; only their headers.
  Requests to `/api/me`, `/api/auth/*` and `/api/billing/*` are not
  intercepted (they are the module's own).
- After any charged response it re-fetches `/api/me` 2.5 s later, because an
  unchanged rewrite is refunded after the response and the header does not
  show that.
- Query parameters on load: `?login_error=bad_token|used_token` opens the
  sheet with the matching message; `?purchase=success` polls `/api/me` once
  a second for up to 10 s until the balance rises, then confirms;
  `?purchase=cancelled` shows a quiet note. Both are removed from the URL
  with `history.replaceState`.

## Events

`readshuman:account`, a `CustomEvent` on `document`, with
`detail = {user}`. `user` is the `/api/me` payload (`id`, `email`,
`credits`, `is_admin`, `created`, `session_expires`) or `null`. Fired once
after boot and whenever the signed-in user or the balance changes.

    document.addEventListener('readshuman:account', function (e) {
      var user = e.detail.user;   // null when signed out
    });

## Public API

`window.readshumanAccount`:

- `costFor(wordCount)`: credits a rewrite of that many words costs, from
  `words_per_credit` in billing health; never less than 1.
- `refresh()`: re-fetch `/api/me`; resolves with the user or null.
- `open('signin' | 'packs' | 'account')`: open a sheet. `account` while
  signed out opens sign-in.
- `close()`: close the open sheet.
- `user()`: the current user or null.
- `health()`: the `/api/billing/health` payload or null.

## What it does not do

- It does not gate the Humanize button or show the cost before a run.
  `app.js` can do that with `costFor()` and `user()`; nothing here touches
  `app.js` or its state.
- It does not handle the email link itself. `GET /api/auth/verify` sets the
  cookie and redirects; the module only reads the `login_error` it may
  leave behind.
- It does not read or write `localStorage`, and it keeps no copy of the
  session; the HttpOnly cookie is the session.
- It does not retry anything except one 503 `busy`; network failures and
  other statuses pass through untouched.
- It does not inject CSS. If `account.css` is missing the sheet renders
  unstyled.
- It does not render when `paywall` is false, and it does not know about
  the local judge or the user's own GPTZero key.

## Two things the backend could add

- `GET /api/billing/health` does not include `magic_link_minutes`, so the
  "check your email" sheet takes the lifetime from the `request-link`
  response (`expires_in_minutes`), falling back to 15. If health ever
  carries `magic_link_minutes` the module will use it.
- The legal pages are Markdown in `deploy/legal/`. The links here point at
  `/legal/terms.html` and `/legal/privacy.html`, which need to be published
  at those paths.
