# Vervly API on Linux, CPU only. Generation is delegated to the vLLM
# server named by HUMANIZER_LLM_URL (deploy/cloud/modal_vllm.py), so this
# image carries no MLX and no GPU runtime: FastAPI, the GPTZero surrogate
# (RoBERTa on CPU torch) and SQLite on a mounted volume at /data.
#
#   docker build -f deploy/cloud/Dockerfile -t vervly-api .
#   docker run --rm -p 8000:8000 -v vervly-data:/data --env-file deploy/.env vervly-api
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends curl unzip sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU wheels only: the default torch wheel on x86_64 pulls CUDA and is 2GB+.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<2.9" \
    && pip install "transformers>=4.40,<5" "safetensors>=0.4"

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY web ./web
COPY deploy/cloud/entrypoint.sh /usr/local/bin/vervly-entrypoint
# Editable install so detectors/local.py finds data/ under the repo root
# (the surrogate path is derived from the package file's location).
RUN pip install -e ".[api,oauth]" && chmod +x /usr/local/bin/vervly-entrypoint

# Reference distributions are small and gitignored; copy whatever is present.
COPY data/reference ./data/reference

# Runtime state lives on the volume: data/cache is a symlink into /data so
# the surrogate weights (downloaded once by the entrypoint), the GPTZero
# response cache and the billing DB all persist across deploys.
RUN mkdir -p /data && rm -rf data/cache && ln -s /data/cache data/cache
VOLUME ["/data"]

ENV HUMANIZER_AUTH_DB=/data/auth.sqlite \
    LONGHAND_DB=/data/billing.db \
    LONGHAND_LOG_DIR=/data/logs \
    HUMANIZER_YARDSTICK=surrogate \
    HUMANIZER_PROXY=surrogate \
    HUMANIZER_ASSUME_AI=1 \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://127.0.0.1:${PORT}/api/health || exit 1
ENTRYPOINT ["vervly-entrypoint"]
