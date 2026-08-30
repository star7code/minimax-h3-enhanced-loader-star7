"""Convert official FastH3 Dense shards to native ComfyUI INT8 ConvRot shards."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from fasth3_loader import INDEX_NAME, _rope_inv_freq, convert_fasth3_state_dict, detect_fasth3_checkpoint  # noqa: E402

MANIFEST_NAME = "star7_fasth3_int8.json"
QUANT_SUFFIXES = (
    ".attn.qkv_proj.weight", ".attn.out_proj.weight",
    ".mlp.fc1.weight", ".mlp.fc2.weight",
)


def _source_tensor(info, key):
    shard = info.weight_map[key]
    with safe_open(info.transformer / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _time_curve(info, grid, device):
    w1 = _source_tensor(info, "time_embedder.linear_1.weight").to(device, torch.float32)
    b1 = _source_tensor(info, "time_embedder.linear_1.bias").to(device, torch.float32)
    w2 = _source_tensor(info, "time_embedder.linear_2.weight").to(device, torch.float32)
    b2 = _source_tensor(info, "time_embedder.linear_2.bias").to(device, torch.float32)
    values = torch.linspace(0.0, 1.0, grid, device=device, dtype=torch.float32)
    half = int(info.config["freq_dim"]) // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
    emb = torch.cat([torch.cos(values[:, None] * freqs), torch.sin(values[:, None] * freqs)], dim=1)
    curve = torch.nn.functional.silu(emb @ w1.T + b1)
    return torch.nn.functional.silu(curve @ w2.T + b2)


def _curve_basis(info, grid, rank, device):
    curve = _time_curve(info, grid, device)
    torch.manual_seed(7)
    u, s, v = torch.pca_lowrank(curve, q=min(rank + 12, min(curve.shape)), center=False, niter=6)
    table = (u[:, :rank] * s[:rank]).contiguous()
    basis = v[:, :rank].contiguous()
    error = (curve - table @ basis.T).square().mean().sqrt()
    relative = error / curve.square().mean().sqrt().clamp_min(1e-12)
    print(f"[Star7 INT8] AdaLN curve grid={grid}, rank={rank}, relative_rmse={relative.item():.6e}")
    return table.cpu(), basis


def _quant_marker(group_size):
    raw = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}).encode()
    return torch.tensor(list(raw), dtype=torch.uint8)


def _quantize_weight(weight, device, group_size=256):
    from comfy_kitchen.tensor import TensorWiseINT8Layout
    source = weight.to(device=device, dtype=torch.float16)
    qdata, params = TensorWiseINT8Layout.quantize(
        source, is_weight=True, per_channel=True, convrot=True,
        convrot_groupsize=group_size, stochastic_rounding=0,
    )
    return qdata.cpu(), params.scale.cpu(), _quant_marker(group_size)


def _is_quant_target(key):
    return key.startswith(("blocks.", "token_refiner.blocks.")) and key.endswith(QUANT_SUFFIXES)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert(source, output, grid, rank, device):
    info = detect_fasth3_checkpoint(source)
    if info.requires_vsa:
        raise RuntimeError("Only FastH3 Dense is supported by this converter")
    output.mkdir(parents=True, exist_ok=True)
    transformer_out = output / "transformer"
    transformer_out.mkdir(exist_ok=True)
    work_device = torch.device(device)
    table, basis = _curve_basis(info, grid, rank, work_device)
    output_map = {}
    shard_names = sorted(set(info.weight_map.values()))
    quantized = source_keys = output_keys = 0
    for shard_index, shard_name in enumerate(shard_names, 1):
        print(f"[Star7 INT8] shard {shard_index}/{len(shard_names)}: {shard_name}", flush=True)
        with safe_open(info.transformer / shard_name, framework="pt", device="cpu") as f:
            source_state = {key: f.get_tensor(key) for key in f.keys()}
        source_keys += len(source_state)
        converted, unexpected = convert_fasth3_state_dict(source_state)
        if unexpected:
            raise ValueError(f"Unexpected source keys in {shard_name}: {unexpected[:5]}")
        del source_state
        for key in tuple(converted):
            if key.startswith("time_embedder."):
                del converted[key]
        out = {}
        for key in sorted(converted):
            tensor = converted[key]
            if key.endswith("adaln_proj.linear.weight"):
                out[key] = (tensor.to(work_device, torch.float32) @ basis).to(torch.float16).cpu()
            elif _is_quant_target(key):
                base = key[:-len(".weight")]
                out[key], out[f"{base}.weight_scale"], out[f"{base}.comfy_quant"] = _quantize_weight(tensor, work_device)
                quantized += 1
            else:
                out[key] = tensor
        if shard_index == 1:
            out["adaln_t_table"] = table
            out["rope.inv_freq"] = _rope_inv_freq(info.config)
        out_name = f"star7_fasth3_int8-{shard_index:05d}-of-{len(shard_names):05d}.safetensors"
        save_file(out, transformer_out / out_name)
        output_map.update({key: out_name for key in out})
        output_keys += len(out)
        del converted, out
        gc.collect()
        if work_device.type == "cuda":
            torch.cuda.empty_cache()
    total_size = sum(p.stat().st_size for p in transformer_out.glob("*.safetensors"))
    _write_json(transformer_out / INDEX_NAME, {"metadata": {"total_size": total_size}, "weight_map": output_map})
    _write_json(transformer_out / "config.json", info.config)
    _write_json(output / MANIFEST_NAME, {
        "schema_version": "star7-fasth3-native-int8-v1",
        "source_model_id": info.inference.get("model_id"), "variant": info.variant,
        "format": "comfy-native-sharded", "quantization": "int8_tensorwise_convrot",
        "convrot_groupsize": 256, "adaln_curve_grid": grid, "adaln_curve_rank": rank,
        "sampling_profile": info.sampling_profile, "source_keys": source_keys,
        "output_keys": output_keys, "quantized_linears": quantized,
    })
    for name in ("fastvideo_inference.json", "checkpoint_content.json", "checkpoint_metadata.json"):
        src = info.root / name
        if src.is_file():
            (output / name).write_bytes(src.read_bytes())
    print(f"[Star7 INT8] complete: {output} | {total_size / 2**30:.2f} GiB | quantized={quantized} | tensors={output_keys}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid", type=int, default=4097)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.grid < 2 or args.rank < 1:
        parser.error("--grid must be >=2 and --rank must be >=1")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    convert(args.source.resolve(), args.output.resolve(), args.grid, args.rank, args.device)


if __name__ == "__main__":
    main()
