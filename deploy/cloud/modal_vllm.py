"""Vervly's generation server: vLLM on a Modal GPU, OpenAI-compatible.

Serves the Apache-licensed `Qwen/Qwen2.5-7B` *base* model with the shipped
HIP LoRA (`data/adapters-peft/hip7b-r4-it200`, converted from the mlx-lm
adapter by `convert_adapter_to_peft.py`) under `/v1/completions`, which is
what `humanizer.humanize.llm.OpenAICompatBackend` calls when
HUMANIZER_LLM_URL points here. Scales to zero between rewrites and bills
per second while a request is running.

One-time setup (owner's Modal account, free tier is enough to start):

    pip install modal
    modal setup                                   # browser sign-in
    modal secret create vervly-llm VLLM_API_KEY=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')
    modal deploy deploy/cloud/modal_vllm.py       # prints the https URL

Then on the API host set HUMANIZER_LLM_URL=<that url>/v1 and
HUMANIZER_LLM_API_KEY to the same key. `modal app logs vervly-llm` tails it.
"""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "vervly-llm"
BASE_MODEL = os.environ.get("VERVLY_BASE_MODEL", "Qwen/Qwen2.5-7B")
#: Served LoRA name -> directory inside the image. The name is what the API
#: sends as `model` (HUMANIZER_BASE_ADAPTER on the API host).
ADAPTERS = {"hip7b-r4-it200": "/adapters/hip7b-r4-it200"}
GPU = os.environ.get("VERVLY_GPU", "L4")
#: Prompt plus completion. The freeform few-shot is ~2.5k tokens and
#: completions are capped at 900 by the pipeline.
MAX_MODEL_LEN = 6144
PORT = 8000

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, "..", ".."))

#: A CUDA *devel* base, not a slim one: recent vLLM JIT-compiles its sampler
#: kernels at first use and needs nvcc on the image (measured 2026-10-05:
#: vllm 0.31 on debian_slim died with "Could not find nvcc"). The vLLM
#: version is pinned so a `modal deploy` months from now builds the same
#: server; bump it deliberately and re-bench.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    # transformers is pinned below 5: an unpinned resolve on 2026-10-05 took
    # 5.x, which drops `all_special_tokens_extended` and breaks vLLM 0.11's
    # tokenizer wrapper at startup.
    .pip_install("vllm==0.11.0", "transformers>=4.56,<5", "tokenizers<0.23", "huggingface_hub>=0.30,<1")
    .env({"HF_HOME": "/hf", "VLLM_NO_USAGE_STATS": "1", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(os.path.join(_repo, "data", "adapters-peft"), remote_path="/adapters")
)

#: The Hugging Face cache, so the 15GB of base weights download once.
weights = modal.Volume.from_name("vervly-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf": weights},
    secrets=[modal.Secret.from_name("vervly-llm")],
    # Seconds idle before the GPU is released; the next request pays the
    # cold start (~5 s container + ~30 s weight load from the volume).
    scaledown_window=180,
    timeout=60 * 60,
    min_containers=int(os.environ.get("VERVLY_MIN_CONTAINERS", "0")),
    # A hard cap on GPU containers: the API server runs one rewrite at a time,
    # so one container is the whole need; a flood cannot fan out onto more GPUs.
    max_containers=int(os.environ.get("VERVLY_MAX_CONTAINERS", "1")),
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve() -> None:
    lora_args = []
    for name, path in ADAPTERS.items():
        if os.path.isdir(path):
            lora_args += ["--lora-modules", f"{name}={path}"]
    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--served-model-name", BASE_MODEL,
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    if lora_args:
        cmd += ["--enable-lora", "--max-lora-rank", "16", "--max-loras", "2"] + lora_args
    subprocess.Popen(cmd)
    # The volume keeps the download; commit so later containers see it.
    weights.commit()
