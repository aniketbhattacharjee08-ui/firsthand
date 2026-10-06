#!/usr/bin/env python
"""Convert an mlx-lm LoRA adapter directory into a Hugging Face PEFT adapter.

The output is a directory that vLLM can serve with
``--enable-lora --lora-modules NAME=/path/to/DST_DIR``.

Conventions (verified against mlx_lm/tuner/lora.py and peft):

    mlx   LoRALinear:  y = linear(x) + scale * ((x @ lora_a) @ lora_b)
                       lora_a: (in, r)   lora_b: (r, out)   scale used raw,
                       NOT divided by r (lora_parameters.scale, default 20.0)
    peft  LoRA Linear: y = base(x) + (lora_alpha / r) * lora_B(lora_A(x))
                       lora_A.weight: (r, in)   lora_B.weight: (out, r)

So   lora_A.weight = lora_a.T,  lora_B.weight = lora_b.T,  lora_alpha = scale * r.

Only numpy + safetensors + json are needed for the conversion itself.  If
torch and peft are importable, an additional end-to-end check runs one real
module through peft's own LoRA layer and compares it with the mlx formula.
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np

KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.lora_([ab])$")
ST_DTYPE = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}
# Relative tolerance for the round-trip check, per storage dtype.  Both A and
# B are rounded, so the product carries ~2 x the unit round-off.
RTOL = {"float16": 2e-3, "bfloat16": 2e-2, "float32": 1e-5}


# --------------------------------------------------------------------------
# minimal safetensors I/O (pure numpy, supports BF16 which numpy lacks)
# --------------------------------------------------------------------------
def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        s, e = info["data_offsets"]
        raw = blob[s:e]
        dt = info["dtype"]
        if dt == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype="<f2")
        elif dt == "BF16":
            arr = bf16_bits_to_f32(np.frombuffer(raw, dtype="<u2"))
        else:
            raise ValueError(f"{name}: unsupported dtype {dt}")
        out[name] = arr.reshape(info["shape"])
    return out


def f32_to_bf16_bits(x):
    """float32 -> bfloat16 bit pattern (uint16) with round-to-nearest-even."""
    b = np.ascontiguousarray(x, dtype="<f4").view("<u4").astype(np.uint32)
    lsb = (b >> 16) & 1
    rounded = (b + 0x7FFF + lsb) >> 16
    return rounded.astype("<u2")


def bf16_bits_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view("<f4")


def to_storage(x, dtype):
    """Return (raw little-endian bytes, round-tripped float32 array)."""
    x = np.ascontiguousarray(x, dtype="<f4")
    if dtype == "float32":
        return x.tobytes(), x
    if dtype == "float16":
        h = x.astype("<f2")
        return h.tobytes(), h.astype("<f4")
    if dtype == "bfloat16":
        bits = f32_to_bf16_bits(x)
        return bits.tobytes(), bf16_bits_to_f32(bits).reshape(x.shape)
    raise ValueError(dtype)


def write_safetensors(path, tensors, dtype, metadata=None):
    """tensors: {name: float32 ndarray}. Returns {name: round-tripped f32}."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    chunks, offset, roundtrip = [], 0, {}
    for name in sorted(tensors):
        raw, rt = to_storage(tensors[name], dtype)
        header[name] = {
            "dtype": ST_DTYPE[dtype],
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
        roundtrip[name] = rt
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(hbytes) % 8) % 8
    hbytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        for c in chunks:
            f.write(c)
    return roundtrip


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
def convert(src, dst, base, dtype):
    with open(os.path.join(src, "adapter_config.json")) as f:
        mlx_cfg = json.load(f)
    if mlx_cfg.get("fine_tune_type", "lora") != "lora":
        sys.exit(f"fine_tune_type={mlx_cfg.get('fine_tune_type')!r}; only 'lora' is supported")
    lp = mlx_cfg["lora_parameters"]
    r, scale, dropout = int(lp["rank"]), float(lp["scale"]), float(lp.get("dropout", 0.0))
    alpha = scale * r  # mlx applies `scale` raw; peft applies alpha / r

    tensors = read_safetensors(os.path.join(src, "adapters.safetensors"))

    modules = {}  # (layer, module) -> {"a": (in,r), "b": (r,out)}
    for name, arr in tensors.items():
        m = KEY_RE.match(name)
        if not m:
            sys.exit(f"unexpected tensor name {name!r}; only model.layers.N.<module>.lora_a/b handled")
        layer, mod, ab = int(m.group(1)), m.group(2), m.group(3)
        modules.setdefault((layer, mod), {})[ab] = arr.astype(np.float32)

    layers = sorted({k[0] for k in modules})
    target_modules = sorted({k[1].split(".")[-1] for k in modules})

    peft_tensors = {}
    for (layer, mod), ab in sorted(modules.items()):
        if set(ab) != {"a", "b"}:
            sys.exit(f"layer {layer} {mod}: missing lora_a or lora_b")
        a, b = ab["a"], ab["b"]
        if a.shape[1] != r or b.shape[0] != r:
            sys.exit(f"layer {layer} {mod}: rank mismatch a{a.shape} b{b.shape} r={r}")
        prefix = f"base_model.model.model.layers.{layer}.{mod}"
        peft_tensors[f"{prefix}.lora_A.weight"] = np.ascontiguousarray(a.T)  # (r, in)
        peft_tensors[f"{prefix}.lora_B.weight"] = np.ascontiguousarray(b.T)  # (out, r)

    os.makedirs(dst, exist_ok=True)
    rt = write_safetensors(
        os.path.join(dst, "adapter_model.safetensors"),
        peft_tensors,
        dtype,
        metadata={"format": "pt"},
    )

    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_modules,
        "layers_to_transform": layers,
        "layers_pattern": None,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "revision": None,
        "auto_mapping": None,
    }
    with open(os.path.join(dst, "adapter_config.json"), "w") as f:
        json.dump(peft_cfg, f, indent=2)
    # Provenance goes in a sidecar: extra keys in adapter_config.json make
    # peft emit an "unexpected keyword arguments" warning.
    with open(os.path.join(dst, "conversion_info.json"), "w") as f:
        json.dump(
            {
                "tool": "deploy/cloud/convert_adapter_to_peft.py",
                "source": os.path.abspath(src),
                "mlx_model": mlx_cfg.get("model"),
                "mlx_lora_parameters": lp,
                "mlx_num_layers": mlx_cfg.get("num_layers"),
                "dtype": dtype,
                "note": "mlx scale is applied raw (no /r); lora_alpha = scale * r",
            },
            f,
            indent=2,
        )

    return modules, rt, r, scale, alpha, dtype


# --------------------------------------------------------------------------
# check 1: numeric equivalence of every module (numpy only)
# --------------------------------------------------------------------------
def self_check(modules, rt, r, scale, alpha, dtype):
    rtol = RTOL[dtype]
    worst, n = 0.0, 0
    for (layer, mod), ab in sorted(modules.items()):
        a, b = ab["a"], ab["b"]
        prefix = f"base_model.model.model.layers.{layer}.{mod}"
        A, B = rt[f"{prefix}.lora_A.weight"], rt[f"{prefix}.lora_B.weight"]
        assert A.shape == (r, a.shape[0]) and B.shape == (b.shape[1], r), (layer, mod, A.shape, B.shape)
        delta_mlx = scale * (a @ b)        # (in, out), applied as x @ delta
        delta_peft = (alpha / r) * (B @ A)  # (out, in), applied as delta @ x
        diff = np.abs(delta_peft - delta_mlx.T)
        ref = np.abs(delta_mlx).max()
        rel = float(diff.max() / ref) if ref > 0 else 0.0
        if not np.isfinite(delta_peft).all():
            raise AssertionError(f"layer {layer} {mod}: non-finite values after {dtype} cast")
        if rel > rtol:
            raise AssertionError(
                f"layer {layer} {mod}: max|delta_peft - delta_mlx.T| / max|delta_mlx| = {rel:.3e} > {rtol}"
            )
        worst = max(worst, rel)
        n += 1
    print(
        f"[self-check] OK: {n} modules, delta_peft == delta_mlx.T within rtol={rtol} "
        f"(worst relative error {worst:.2e}, dtype={dtype}, scale={scale}, r={r}, alpha={alpha})"
    )


# --------------------------------------------------------------------------
# check 2: run one real module through peft's own LoRA layer
# --------------------------------------------------------------------------
def peft_check(dst, modules, scale, dtype):
    try:
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
    except Exception as e:  # noqa: BLE001
        print(f"[peft-check] skipped (torch/peft not importable: {e})")
        return

    with open(os.path.join(dst, "adapter_config.json")) as f:
        cfg = json.load(f)
    # pick the smallest module (k_proj: in=hidden, out=kv_heads*head_dim)
    key = min(modules, key=lambda k: modules[k]["a"].shape[0] * modules[k]["b"].shape[1])
    layer, mod = key
    short = mod.split(".")[-1]
    a, b = modules[key]["a"], modules[key]["b"]
    in_f, out_f = a.shape[0], b.shape[1]

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            setattr(self, short, nn.Linear(in_f, out_f, bias=True))

        def forward(self, x):
            return getattr(self, short)(x)

    torch.manual_seed(0)
    base = Tiny().float()
    lcfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=[short],
        bias=cfg["bias"],
        fan_in_fan_out=cfg["fan_in_fan_out"],
        init_lora_weights=True,
        inference_mode=True,
    )
    model = get_peft_model(base, lcfg)

    sd = load_file(os.path.join(dst, "adapter_model.safetensors"))
    prefix = f"base_model.model.model.layers.{layer}.{mod}"
    sub = {
        f"base_model.model.{short}.lora_A.weight": sd[f"{prefix}.lora_A.weight"].float(),
        f"base_model.model.{short}.lora_B.weight": sd[f"{prefix}.lora_B.weight"].float(),
    }
    res = set_peft_model_state_dict(model, sub)
    assert not res.unexpected_keys, res.unexpected_keys
    model.eval()

    x = torch.randn(5, in_f)
    with torch.no_grad():
        y_peft = model(x).numpy()
        W = getattr(base, short).weight.numpy()
        bias = getattr(base, short).bias.numpy()
    xn = x.numpy()
    y_mlx = xn @ W.T + bias + scale * (xn @ a @ b)
    y_base = xn @ W.T + bias
    lora_part = np.abs(y_mlx - y_base).max()
    err = np.abs(y_peft - y_mlx).max()
    rel = err / lora_part if lora_part > 0 else 0.0
    rtol = RTOL[dtype] * 2
    status = "OK" if rel <= rtol else "FAIL"
    print(
        f"[peft-check] {status}: layer {layer} {mod} ({in_f}->{out_f}) through peft {model.peft_config['default'].peft_type.value} "
        f"Linear vs numpy base(x) + {scale} * x @ lora_a @ lora_b: max abs err {err:.3e}, "
        f"lora contribution {lora_part:.3e}, rel {rel:.2e} (tol {rtol})"
    )
    if status != "OK":
        raise AssertionError("peft end-to-end check failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="mlx-lm adapter dir (adapter_config.json + adapters.safetensors)")
    ap.add_argument("dst", help="output PEFT adapter dir")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B", help="base_model_name_or_path for adapter_config.json")
    ap.add_argument("--dtype", default="float16", choices=sorted(ST_DTYPE))
    ap.add_argument("--no-peft-check", action="store_true", help="skip the torch/peft end-to-end check")
    args = ap.parse_args()

    modules, rt, r, scale, alpha, dtype = convert(args.src, args.dst, args.base, args.dtype)
    layers = sorted({k[0] for k in modules})
    print(
        f"[convert] {args.src} -> {args.dst}: {len(modules)} modules, layers {layers[0]}-{layers[-1]} "
        f"({len(layers)}), r={r}, mlx scale={scale} -> lora_alpha={alpha}, dtype={dtype}, "
        f"size={os.path.getsize(os.path.join(args.dst, 'adapter_model.safetensors')) / 1e6:.1f} MB"
    )
    self_check(modules, rt, r, scale, alpha, dtype)
    if not args.no_peft_check:
        peft_check(args.dst, modules, scale, dtype)


if __name__ == "__main__":
    main()
