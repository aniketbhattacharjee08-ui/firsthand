# Vervly without the Mac

Written 2026-10-05. The product runs on two rented services and a Vercel
front, none of which is the dev machine:

```
browser -> https://<domain>  (Vercel: TLS, custom domain, rewrite of every path)
        -> https://vervly-api.fly.dev   (Fly.io: the FastAPI app, surrogate detector, SQLite on a volume)
        -> https://<account>--vervly-llm-serve.modal.run/v1   (Modal: vLLM on an L4 GPU, scales to zero)
```

The engine change that makes this possible is `OpenAICompatBackend` in
`src/humanizer/humanize/llm.py`: when `HUMANIZER_LLM_URL` is set, every
style generates over an OpenAI-compatible HTTP API instead of local MLX, so
the API host needs no GPU. The shipped LoRA was converted from the mlx-lm
format to PEFT with `convert_adapter_to_peft.py` (verified against PEFT's
own LoRA code to 5e-4 relative error) and is served by vLLM under the name
`hip7b-r4-it200`.

Why two services: no per-token provider serves the Qwen2.5-7B *base* model
any more (Together dropped it, Fireworks only has instruct), and instruct
models read as AI on GPTZero (research/00 finding 5), so we rent GPU
seconds and run vLLM ourselves. Modal does that with a free monthly credit
and no card; it is a poor home for SQLite (its volumes have no file
locking), so the app itself lives on Fly.io, which has real disks.

## Owner steps, in order

Each one needs your account. Everything else is scripted.

1. **Modal** (GPU). `~/.local/deploy-venv/bin/modal token new` opens a
   browser; sign up with GitHub. Then:
   ```sh
   M=~/.local/deploy-venv/bin/modal
   $M secret create vervly-llm VLLM_API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')
   $M deploy deploy/cloud/modal_vllm.py        # first run downloads 15GB of weights, ~10 min
   ```
   It prints a URL ending in `.modal.run`. Put `<url>/v1` in `deploy/.env`
   as `HUMANIZER_LLM_URL` and the key as `HUMANIZER_LLM_API_KEY`
   (`$M secret list` shows the name; the value is what you generated).
2. **Fly.io** (API). `~/.fly/bin/flyctl auth login` opens a browser; a card
   is required. Then from the repo root:
   ```sh
   F=~/.fly/bin/flyctl
   $F apps create vervly-api
   $F volumes create vervly_data --size 5 --region sjc --app vervly-api --yes
   $F secrets set --app vervly-api $(grep -v '^#' deploy/.env | grep -v '^$' | grep -v '=$' | xargs)
   $F deploy --config deploy/cloud/fly.toml --remote-only
   ```
   First boot downloads the surrogate weights (480MB) into the volume.
   `curl https://vervly-api.fly.dev/api/humanize/llm/health` should say
   `"generation": "remote"` and `"available": true`.
3. **Vercel** (domain). Buy the domain in the Vercel dashboard (as for
   ashaberkeley.com) or anywhere else, then:
   ```sh
   export PATH=~/.local/node/bin:$PATH
   deploy/vercel/make-site.sh https://vervly-api.fly.dev
   cd deploy/vercel/site && vercel deploy --prod --yes --scope asha22
   vercel domains add <domain> --scope asha22      # Vercel-bought domains attach in one step
   ```
   Then set `LONGHAND_PUBLIC_URL` and `HUMANIZER_PUBLIC_URL` in
   `deploy/.env` to `https://<domain>`, re-run the `secrets set` line, and
   `fly deploy` again so sign-in cookies and OAuth redirects use the domain.

After step 3 the Mac can be switched off: `launchctl bootout
gui/$(id -u)/com.firsthand.tunnel` and the same for `com.firsthand.server`.

## Cost

Production makes no GPTZero calls (decided 2026-10-05): candidates are
ranked by the free local surrogate and the input is assumed AI-written
(`HUMANIZER_ASSUME_AI=1`), so the only usage-based bill is GPU seconds.

Launch traffic (tens of documents a day): Fly shared-cpu-2x with 4GB and a
5GB volume is about $26 a month; the GPU time fits inside Modal's $30
monthly credit. At 300 documents a day the GPU alone is $130 to $180 a
month on an L4; the levers are fewer candidates, FP8 weights, or a
dedicated A100 with scale-to-zero on DeepInfra (about $0.89 an hour).

## Files

- `Dockerfile`, `entrypoint.sh`, `../../.dockerignore`: the CPU image.
  Python 3.11, CPU torch, `pip install -e .[api,oauth]`, `web/` served by
  the app, `data/cache` symlinked into the `/data` volume.
- `fly.toml`: one machine, auto-stop off (rewrites stream for minutes),
  health check on `/api/health`.
- `modal_vllm.py`: vLLM serving `Qwen/Qwen2.5-7B` plus the LoRA, bf16,
  prefix caching, API key from the `vervly-llm` secret, idle release
  after 180 s. `VERVLY_GPU=A10G modal deploy ...` picks another card;
  `VERVLY_MIN_CONTAINERS=1` keeps one warm (about $580 a month on L4,
  so only once traffic justifies it).
- `convert_adapter_to_peft.py`: mlx-lm LoRA to PEFT. Output lives in
  `data/adapters-peft/` and is baked into the Modal image.

## What is not covered

- The `faithful` style (instruct model) and `register` style are not served
  by the Modal app; the product default `freeform` is. Requests for the
  others return a 503 naming the missing model. Add a second
  `vllm serve` function if they are ever exposed.
- Guided decoding's TOKEN mode needs in-process logits and stays Mac-only.
- The adapters were trained against the 4-bit MLX base and now run on
  bf16 weights. Bench the deployed server on GPTZero before announcing;
  `scripts/` has the bench, point it at the Fly URL.
- Paywall (`LONGHAND_PAYWALL=1`) needs Stripe keys and an SMTP URL; the
  entrypoint switches to `python -m humanizer.billing` automatically.
