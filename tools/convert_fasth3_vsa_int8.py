"""Merge the official FastH3 VSA adapter into a native H3 INT8 checkpoint.

The adapter is not a normal LoRA: it combines rank-64 factors, exact dense
deltas, and 50 replacement compression gates.  This converter applies every
payload at strength 1.0, requantizes matrix weights with Comfy Kitchen's
TensorWise INT8 ConvRot layout, and writes bounded-memory shards.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA = "star7-fasth3-vsa-native-int8-v1"
SAMPLING_PROFILE = "fasth3_4step_dmd_999_749_500_250_cfg1"
GROUP_SIZE = 256
HEAD_WIDTH = 56 * 128

GLOBAL_MAP = {
    "proj_in": "video_patch_proj",
    "audio_proj_in": "audio_patch_proj",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "token_refiner.final_norm": "token_refiner.final_norm",
    "norm_out.norm": "final_layer.norm",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "proj_out": "final_layer.video_out",
    "audio_proj_out": "final_layer.audio_out",
}

BLOCK_MAP = {
    "norm1": "norm1",
    "norm2": "norm2",
    "attn.norm_q": "attn.q_norm",
    "attn.norm_k": "attn.k_norm",
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
    "adaln_proj.linear": "adaln_proj.linear",
}


def _marker(group_size: int = GROUP_SIZE) -> torch.Tensor:
    value = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": int(group_size),
    }
    return torch.tensor(list(json.dumps(value).encode("utf-8")), dtype=torch.uint8)


def _adapter_to_target(name: str) -> tuple[str, tuple[int, int] | None]:
    for source, target in GLOBAL_MAP.items():
        if name == source:
            return target, None

    for source_prefix, target_prefix in (
        ("transformer_blocks.", "blocks."),
        ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ):
        if not name.startswith(source_prefix):
            continue
        tail = name[len(source_prefix):]
        index, suffix = tail.split(".", 1)
        if suffix.startswith("attn.to_") and suffix[-1:] in {"q", "k", "v"}:
            part = {"q": 0, "k": 1, "v": 2}[suffix[-1]]
            return f"{target_prefix}{index}.attn.qkv_proj", (part * HEAD_WIDTH, HEAD_WIDTH)
        mapped = BLOCK_MAP.get(suffix)
        if mapped is None:
            raise KeyError(f"Unsupported FastH3 adapter module: {name}")
        return f"{target_prefix}{index}.{mapped}", None
    raise KeyError(f"Unsupported FastH3 adapter module: {name}")


def _adapter_index(path: Path):
    low_rank: dict[str, list[tuple[int, int, str, str]]] = {}
    additive: dict[str, str] = {}
    gates: dict[int, str] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            if key.endswith(".lora_A.weight"):
                source = key[:-len(".lora_A.weight")]
                b_key = source + ".lora_B.weight"
                if b_key not in keys:
                    raise ValueError(f"Missing LoRA B tensor for {source}")
                target, span = _adapter_to_target(source)
                start, length = span or (0, -1)
                low_rank.setdefault(target + ".weight", []).append(
                    (start, length, key, b_key)
                )
            elif key.endswith(".diff_b"):
                source = key[:-len(".diff_b")]
                target, span = _adapter_to_target(source)
                if span is not None:
                    raise ValueError(f"Unexpected fused bias delta: {key}")
                additive[target + ".bias"] = key
            elif key.endswith(".diff"):
                source = key[:-len(".diff")]
                target, span = _adapter_to_target(source)
                if span is not None:
                    raise ValueError(f"Unexpected fused weight delta: {key}")
                additive[target + ".weight"] = key
            elif key.endswith(".attn.to_gate_compress.set_weight"):
                prefix = "transformer_blocks."
                if not key.startswith(prefix):
                    raise ValueError(f"Unexpected gate key: {key}")
                index = int(key[len(prefix):].split(".", 1)[0])
                gates[index] = key
    if len(gates) != 50:
        raise ValueError(f"Expected 50 VSA gates, found {len(gates)}")
    return low_rank, additive, gates


def _quant_params(
    scale: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    group_size: int,
):
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    return TensorWiseINT8Layout.Params(
        scale=scale.to(device),
        orig_dtype=torch.float16,
        orig_shape=shape,
        is_weight=True,
        convrot=True,
        convrot_groupsize=group_size,
    )


def _prepare_legacy_cuda_properties(device: torch.device) -> None:
    """Supply the shared-memory field omitted by this bundled Torch 2.6 build."""
    if device.type != "cuda":
        return
    props = torch.cuda.get_device_properties(device)
    if hasattr(props, "shared_memory_per_block"):
        return
    # Turing exposes 64 KiB opt-in shared memory, while 48 KiB is the portable
    # per-block limit.  The lower value only selects the staged ConvRot kernel;
    # it cannot change the quantized result.
    from comfy_kitchen.backends import cuda as kitchen_cuda

    index = device.index if device.index is not None else torch.cuda.current_device()
    kitchen_cuda._max_shared_memory_cache[index] = 48 * 1024


def _dequantize(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    group_size: int,
):
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    _prepare_legacy_cuda_properties(device)
    qdata = qdata.to(device)
    return TensorWiseINT8Layout.dequantize(
        qdata, _quant_params(scale, tuple(qdata.shape), device, group_size)
    ).to(torch.float32)


def _quantize(
    weight: torch.Tensor,
    device: torch.device,
    group_size: int = GROUP_SIZE,
):
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    _prepare_legacy_cuda_properties(device)
    source = weight.to(device=device, dtype=torch.float16)
    qdata, params = TensorWiseINT8Layout.quantize(
        source,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=group_size,
        stochastic_rounding=0,
    )
    return qdata.cpu(), params.scale.cpu(), _marker(group_size)


def _quant_group_size(marker: torch.Tensor) -> int:
    config = json.loads(marker.numpy().tobytes())
    params = config.get("params", {}) if isinstance(config.get("params"), dict) else {}
    return int(config.get("convrot_groupsize", params.get("convrot_groupsize", GROUP_SIZE)))


def _read_adapter(handle, key: str, device: torch.device) -> torch.Tensor:
    return handle.get_tensor(key).to(device=device, dtype=torch.float32)


def _merge_tensor(
    key: str,
    tensor: torch.Tensor,
    base_handle,
    adapter_handle,
    low_rank,
    additive,
    device: torch.device,
):
    patches = low_rank.get(key, ())
    delta_key = additive.get(key)
    quant_marker_key = key[:-len("weight")] + "comfy_quant" if key.endswith("weight") else ""
    is_quantized = bool(quant_marker_key and quant_marker_key in base_handle.keys())

    if is_quantized:
        scale_key = key[:-len("weight")] + "weight_scale"
        if not patches and delta_key is None:
            return {
                key: tensor,
                scale_key: base_handle.get_tensor(scale_key),
                quant_marker_key: base_handle.get_tensor(quant_marker_key),
            }
    elif not patches and delta_key is None:
        return {key: tensor}

    if is_quantized:
        scale_key = key[:-len("weight")] + "weight_scale"
        source_marker = base_handle.get_tensor(quant_marker_key)
        group_size = _quant_group_size(source_marker)
        weight = _dequantize(
            tensor, base_handle.get_tensor(scale_key), device, group_size
        )
    else:
        weight = tensor.to(device=device, dtype=torch.float32)

    if delta_key is not None:
        delta = _read_adapter(adapter_handle, delta_key, device)
        if delta.shape != weight.shape:
            raise ValueError(f"Shape mismatch for {key}: {weight.shape} vs {delta.shape}")
        weight.add_(delta)
        del delta

    for start, length, a_key, b_key in patches:
        a = _read_adapter(adapter_handle, a_key, device)
        b = _read_adapter(adapter_handle, b_key, device)
        target = weight if length < 0 else weight.narrow(0, start, length)
        if target.shape != (b.shape[0], a.shape[1]):
            raise ValueError(
                f"LoRA shape mismatch for {key}: target={tuple(target.shape)}, "
                f"B={tuple(b.shape)}, A={tuple(a.shape)}"
            )
        target.addmm_(b, a)
        del a, b

    if is_quantized:
        base = key[:-len(".weight")]
        qdata, scale, marker = _quantize(weight, device, group_size)
        return {
            key: qdata,
            f"{base}.weight_scale": scale,
            f"{base}.comfy_quant": marker,
        }
    return {key: weight.to(dtype=tensor.dtype).cpu()}


def _copy_group(
    keys: list[str],
    base_handle,
    adapter_handle,
    low_rank,
    additive,
    device: torch.device,
):
    output = {}
    consumed_quant = set()
    for key in keys:
        if key in consumed_quant or key.endswith((".weight_scale", ".comfy_quant")):
            continue
        tensor = base_handle.get_tensor(key)
        merged = _merge_tensor(
            key, tensor, base_handle, adapter_handle,
            low_rank, additive, device,
        )
        output.update(merged)
        if key.endswith(".weight") and f"{key[:-7]}.comfy_quant" in base_handle.keys():
            consumed_quant.update({f"{key[:-7]}.weight_scale", f"{key[:-7]}.comfy_quant"})
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(base: Path, adapter: Path, output: Path, device_name: str):
    output.mkdir(parents=True, exist_ok=False)
    transformer = output / "transformer"
    transformer.mkdir()
    device = torch.device(device_name)
    low_rank, additive, gates = _adapter_index(adapter)
    weight_map = {}

    with safe_open(base, framework="pt", device="cpu") as base_handle, safe_open(
        adapter, framework="pt", device="cpu"
    ) as adapter_handle:
        base_keys = list(base_handle.keys())
        groups = [("global", [k for k in base_keys if not k.startswith(("blocks.", "token_refiner.blocks."))])]
        groups += [
            (f"block-{i:02d}", [k for k in base_keys if k.startswith(f"blocks.{i}.")])
            for i in range(50)
        ]
        groups += [
            (f"refiner-{i}", [k for k in base_keys if k.startswith(f"token_refiner.blocks.{i}.")])
            for i in range(2)
        ]

        for shard_index, (label, keys) in enumerate(groups, 1):
            print(f"[Star7 VSA INT8] {shard_index}/{len(groups)} {label}", flush=True)
            shard = _copy_group(
                keys, base_handle, adapter_handle, low_rank, additive, device
            )
            if label.startswith("block-"):
                index = int(label.split("-")[1])
                gate = _read_adapter(adapter_handle, gates[index], device)
                qdata, scale, marker = _quantize(gate, device)
                prefix = f"blocks.{index}.attn.gate_compress"
                shard[f"{prefix}.weight"] = qdata
                shard[f"{prefix}.weight_scale"] = scale
                shard[f"{prefix}.comfy_quant"] = marker
                del gate

            name = f"star7-fasth3-vsa-{shard_index:05d}-of-{len(groups):05d}.safetensors"
            save_file(shard, transformer / name)
            weight_map.update({key: name for key in shard})
            del shard
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        metadata = base_handle.metadata() or {}
        config = json.loads(metadata.get("config", "{}"))

    index = {
        "metadata": {
            "total_size": sum(p.stat().st_size for p in transformer.glob("*.safetensors")),
            "schema": SCHEMA,
        },
        "weight_map": weight_map,
    }
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (transformer / "config.json").write_text(
        json.dumps(config.get("transformer", config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA,
        "variant": "fasth3_vsa_datafree_v1",
        "format": "comfy-native-sharded",
        "quantization": "int8_tensorwise_convrot",
        "convrot_groupsize": GROUP_SIZE,
        "sampling_profile": SAMPLING_PROFILE,
        "transformer_forwards": 4,
        "dmd_denoising_steps": [999, 749, 500, 250],
        "guidance_scale": 1.0,
        "attention_backend": "video_sparse_attn_h3",
        "tile_size": 64,
        "sparsity": 0.9,
        "adapter_sha256": _sha256(adapter),
        "base_file": base.name,
        "adapter_file": adapter.name,
        "tensors": len(weight_map),
    }
    (output / "star7_fasth3_vsa_int8.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[Star7 VSA INT8] complete | {index['metadata']['total_size'] / 2**30:.2f} GiB "
        f"| tensors={len(weight_map)} | output={output}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    convert(args.base.resolve(), args.adapter.resolve(), args.output.resolve(), args.device)


if __name__ == "__main__":
    main()
