# Merge day: the exact edits to existing files

Everything in `src/humanizer/billing/`, `deploy/` and `tests/test_billing.py`
was written without touching any existing file. These are the edits that
finish the job once the frontend is done. Each is small; do them on a
branch, run `pytest`, then deploy. Line references are from 2026-09-10 and
may have moved; the surrounding code is quoted so the spot is findable.

## 1. `src/humanizer/cli.py`: `humanizer serve` runs the paywalled app

In `cmd_serve`, replace

```python
    try:
        from .api import run as run_server
```

with

```python
    try:
        from .billing.app import run as run_server
```

`billing.app.run` has the same signature (`host`, `port`, `reference`). With
`LONGHAND_PAYWALL` unset it serves the identical open app plus the auth
routes, so local development does not change.

## 2. `pyproject.toml`: a second console script

Under `[project.scripts]` add

```toml
readshuman = "humanizer.billing.__main__:main"
```

so the host can run `readshuman serve`, `readshuman check`, `readshuman
admin stats`. No new dependencies.

## 3. `src/humanizer/api/server.py`: do not cache a caller's GPTZero results

In `_request_judge`, change

```python
        detector=GPTZeroClient(api_key=key),
```

to

```python
        detector=GPTZeroClient(api_key=key, cache_dir=None),
```

Text scored with a customer's own key should not be written to the
operator's disk under `data/cache/gptzero`. The privacy policy then says
"nothing is stored" truthfully. The owner-key path used by benches keeps its
cache.

## 4. `src/humanizer/humanize/llm.py`: the base-model licence (decision D9)

`FREEFORM_MODEL` defaults to `mlx-community/Qwen2.5-3B-4bit`. Qwen2.5-3B is
released under the Qwen Research Licence, which does not permit commercial
use. The Apache-2.0 sizes are 0.5B, 1.5B, 7B, 14B and 32B. The repo's own
measurement (research/24 §6.5) has the 3B base passing 7 of 9 bench
paragraphs on GPTZero and the 7B base 4 of 9, so switching costs pass rate.
The options, for the owner to choose:

- **Use 7B** (`HUMANIZER_BASE_MODEL=mlx-community/Qwen2.5-7B-4bit` in
  `.env`, no code edit). Legal today; lower measured pass rate; slower.
- **Bench 1.5B** (`mlx-community/Qwen2.5-1.5B-4bit`) with the same few-shot
  prompt; Apache 2.0, unmeasured.
- **Ask Alibaba for a commercial licence** for 3B (the licence text gives a
  contact) and keep the best-measured checkpoint.

**Resolved and applied 2026-09-10:** `llm.py` now defaults `FREEFORM_MODEL`
to `mlx-community/Qwen2.5-7B-4bit` and `DEFAULT_BASE_ADAPTER` to
`data/adapters/hip7b-r4-it200` (used when the directory exists and
`HUMANIZER_BASE_ADAPTER` is unset; an empty string selects the plain base).
Adapter r4 scored 9 of 9 on own-key GPTZero against the plain 7B's 6 of 9 the
same day (research/26). `deploy/sync-weights.sh` copies `data/adapters/`
including it. Nothing left to do here except ship the adapter directory.

## 5. `web/index.html`: include the account module

Copy `deploy/frontend/account.css` and `account.js` into `web/`, then add
after the `styles.css` link and before `</body>` respectively:

```html
<link rel="stylesheet" href="account.css?v=1">
<script src="account.js?v=1" defer></script>
```

`deploy/frontend/README.md` documents what the module does and its public
API. It wraps `fetch` for `/api/` paths, so `app.js` needs no changes for
401, 402 and busy handling; it may optionally listen to the
`readshuman:account` event to show the balance in its own way.

## 6. Legal pages

```bash
.venv/bin/python deploy/legal/build.py --out web/legal
```

after the drafts in `deploy/legal/*.md` have been reviewed and the bracketed
items filled in. The footer of `index.html` should link `/legal/terms.html`
and `/legal/privacy.html`.

## 7. `deploy/Caddyfile`: enforce the CSP

After a day on staging with no console violations, rename
`Content-Security-Policy-Report-Only` to `Content-Security-Policy`.

## 8. `README.md`

In the Install section, after the pytest line, add a paragraph:

> The public deployment adds accounts, credits and a Stripe paywall from
> `humanizer.billing`; see `deploy/README.md`. `humanizer serve` runs the
> same app; set `LONGHAND_PAYWALL=1` to gate the LLM engines.

## 9. Verify

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m humanizer.billing check
humanizer serve --port 8000     # then open the page, sign in, run a rewrite
```

Then follow `deploy/LAUNCH-PLAN.md` section 5.

## 10. `src/humanizer/humanize/llm.py`: stop a freeform completion at the first blank line

Found benching the 7B adapter (2026-09-10): a base-model completion that does
not write the next `DRAFT:` label runs on into multilingual token noise, and
the fidelity gates let it through because `text.words` counts Latin letters
only. In `clean_completion`, after the stop-string cut, add

```python
    # A paragraph rewrite is one paragraph: cut at the first blank line.
    text = text.split("\n\n", 1)[0]
```

and in `pipeline._gate_candidate` add a hard rejection when more than 2% of
the candidate's characters are letters outside Latin scripts
(`unicodedata.name(ch).startswith(("CJK", "THAI", "HEBREW", "HANGUL", "ARABIC"))`
or simply `ord(ch) > 0x024F and ch.isalpha()`) or when it contains three or
more `!` in a row. Reason string: `"junk_text"`.
