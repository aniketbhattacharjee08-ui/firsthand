"""LoRA on a base checkpoint with the loss masked to the HUMAN completion.

Why not `python -m mlx_lm lora`: its prompt/completion loader runs the
tokenizer's chat template, which wraps every example in chat tokens the base
model never sees at inference and, in mlx_lm 0.29.1, crashes when computing
the mask offset. The `text` loader has no masking. This script keeps the
raw-text format the pipeline uses and masks the prompt by token offset.

    .venv/bin/python .hip7b/train.py --iters 240 --tag hip7b-r1

Writes adapters to .hip7b/adapters/<tag>/ in the layout `mlx_lm.load(...,
adapter_path=...)` expects (adapter_config.json + adapters.safetensors), with
a dated checkpoint every `--save-every` steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load, save_config

HERE = Path(__file__).resolve().parent


class MaskedCompletions:
    """(tokens, offset) pairs: raw prompt + completion, loss from `offset` on."""

    def __init__(self, rows, tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def process(self, d):
        prompt = self.tok.encode(d["prompt"])
        full = self.tok.encode(d["prompt"] + d["completion"])
        if full[-1] != self.tok.eos_token_id:
            full.append(self.tok.eos_token_id)
        offset = len(prompt)
        if full[:offset] != prompt:  # tokenizer merged across the boundary
            offset = max(0, offset - 1)
        if len(full) > self.max_len:
            full = full[: self.max_len]
        return (full, offset)

    def __getitem__(self, idx):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)


class Log(TrainingCallback):
    def __init__(self, path: Path):
        self.path = path
        self.rows = []

    def on_train_loss_report(self, info):
        info = dict(info, kind="train", t=time.time())
        self.rows.append(info)
        self.path.write_text(json.dumps(self.rows, indent=1))

    def on_val_loss_report(self, info):
        info = dict(info, kind="val", t=time.time())
        self.rows.append(info)
        self.path.write_text(json.dumps(self.rows, indent=1))
        print("VAL iter %s loss %.4f" % (info.get("iteration"), info.get("val_loss", float("nan"))), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-4bit")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--tag", default="hip7b")
    ap.add_argument("--iters", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-seq-length", type=int, default=1536)
    ap.add_argument("--steps-per-eval", type=int, default=40)
    ap.add_argument("--val-batches", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=40)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    model.freeze()
    lora_parameters = {"rank": args.rank, "dropout": args.dropout, "scale": args.scale}
    linear_to_lora_layers(model, args.num_layers, lora_parameters)
    if args.resume:
        model.load_weights(args.resume, strict=False)
    print_trainable_parameters(model)

    data = Path(args.data)
    train_rows = [json.loads(l) for l in (data / "train.jsonl").open()]
    valid_rows = [json.loads(l) for l in (data / "valid.jsonl").open()]
    print("train %d valid %d" % (len(train_rows), len(valid_rows)), flush=True)
    train_set = CacheDataset(MaskedCompletions(train_rows, tokenizer, args.max_seq_length))
    valid_set = CacheDataset(MaskedCompletions(valid_rows, tokenizer, args.max_seq_length))

    adapter_path = HERE / "adapters" / args.tag
    adapter_path.mkdir(parents=True, exist_ok=True)
    # The config `mlx_lm.load(adapter_path=...)` reads back.
    save_config(
        {
            "model": args.model,
            "fine_tune_type": "lora",
            "num_layers": args.num_layers,
            "lora_parameters": lora_parameters,
            "iters": args.iters,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "mask_prompt": True,
            "data": str(data),
            "seed": args.seed,
        },
        adapter_path / "adapter_config.json",
    )
    targs = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=10,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=str(adapter_path / "adapters.safetensors"),
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    opt = optim.Adam(learning_rate=args.learning_rate)
    train(
        model=model,
        args=targs,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        training_callback=Log(adapter_path / "losses.json"),
    )
    print("TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
