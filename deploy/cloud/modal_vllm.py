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
import shutil
import subprocess
import threading
import time

import modal

#: VERVLY_APP_NAME lets a trial server be deployed beside production
#: (`VERVLY_APP_NAME=vervly-llm-trial VERVLY_VLLM_VERSION=0.31.0 modal deploy ...`)
#: so a vLLM upgrade is proven on its own URL before the default below moves.
APP_NAME = os.environ.get("VERVLY_APP_NAME", "vervly-llm")
#: The vLLM release. 0.11.0 shipped 2026-10-05; the 2026-10-07 audit found 25
#: advisories against it, 14 fixed only in 0.30.0, so the upgrade trial
#: targets the current release. Bump deliberately and re-bench.
VLLM_VERSION = os.environ.get("VERVLY_VLLM_VERSION", "0.11.0")
BASE_MODEL = os.environ.get("VERVLY_BASE_MODEL", "Qwen/Qwen2.5-7B")
#: Served LoRA name -> directory inside the image. The name is what the API
#: sends as `model` (HUMANIZER_BASE_ADAPTER on the API host).
ADAPTERS = {"hip7b-r4-it200": "/adapters/hip7b-r4-it200"}
#: GPU types in order of preference. Modal schedules on the first with
#: capacity: on 2026-11-01 every wake sat "waiting to be scheduled on a
#: GPU_L4 worker" for up to 3.5 min before the boot even began, so the
#: alternatives (both 24 GB or more, Qwen 7B bf16 is 14.4 GiB) take over
#: when L4s are short. `VERVLY_GPU=L4` pins one card.
GPU = [g.strip() for g in os.environ.get("VERVLY_GPU", "L4,A10G,L40S").split(",") if g.strip()]
#: Seconds idle before the GPU is released. 180 s (the launch value) let
#: the server sleep between a visitor's measure and their Humanize press;
#: 10 min keeps a working session warm for about $0.13 of idle L4 time per
#: wake. `VERVLY_MIN_CONTAINERS=1` removes cold starts altogether.
SCALEDOWN_S = int(os.environ.get("VERVLY_SCALEDOWN_S", "600"))
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
# vLLM 0.11 needs transformers held below 5 (an unpinned resolve on 2026-10-05
# took 5.x, which drops `all_special_tokens_extended` and breaks its tokenizer
# wrapper at startup); 0.30+ ships against transformers 5 and pins its own
# torch, so it gets no extra constraints.
if VLLM_VERSION.startswith("0.11"):
    _PINS = ["vllm==" + VLLM_VERSION, "transformers>=4.56,<5", "tokenizers<0.23", "huggingface_hub>=0.30,<1"]
else:
    _PINS = ["vllm==" + VLLM_VERSION]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(*_PINS)
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
    scaledown_window=SCALEDOWN_S,
    timeout=60 * 60,
    min_containers=int(os.environ.get("VERVLY_MIN_CONTAINERS", "0")),
    # A hard cap on GPU containers: the API server runs one rewrite at a time,
    # so one container is the whole need; a flood cannot fan out onto more GPUs.
    max_containers=int(os.environ.get("VERVLY_MAX_CONTAINERS", "1")),
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve() -> None:
    _restore_caches()
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
    # Once the server answers, copy its compile caches to the volume too.
    threading.Thread(target=_save_caches_when_ready, name="cache-save", daemon=True).start()


#: vLLM's torch.compile artifacts, Inductor's kernels and Triton's, which
#: every fresh container rebuilt from nothing (62 s of torch.compile and
#: 14 s of graph capture on 2026-11-01). They are copied to and from the
#: weights volume rather than written there directly, because the compile
#: machinery takes file locks and the volume is a network mount.
_CACHES = {
    os.path.expanduser("~/.cache/vllm"): "/hf/caches/vllm",
    "/tmp/torchinductor_root": "/hf/caches/torchinductor",
    os.path.expanduser("~/.triton"): "/hf/caches/triton",
}


def _restore_caches() -> None:
    for local, saved in _CACHES.items():
        if os.path.isdir(saved):
            shutil.copytree(saved, local, dirs_exist_ok=True)


def _save_caches_when_ready() -> None:
    import urllib.request

    deadline = time.time() + 14 * 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(5)
    else:
        return
    for local, saved in _CACHES.items():
        if os.path.isdir(local):
            shutil.copytree(local, saved, dirs_exist_ok=True)
    weights.commit()
