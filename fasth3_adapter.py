"""Header-first parser for official FastH3 composite adapters.

FastH3 adapters are not plain LoRAs.  A published bundle may contain low-rank
factors, exact dense/bias deltas, and replacement VSA compression gates.  This
module turns the first two payload types into native ComfyUI ModelPatcher
patches and returns the replacement gates separately for runtime installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


HEAD_WIDTH = 56 * 128
SUPPORTED_FORMATS = {"fastvideo-lora-v2"}

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


@dataclass(frozen=True)
class FastH3AdapterInfo:
    path: Path
    metadata: dict[str, str]
    low_rank_pairs: int
    dense_deltas: int
    bias_deltas: int
    replacement_gates: int
    requires_vsa: bool


def adapter_target(name: str) -> tuple[str, tuple[int, int] | None]:
    """Map a Diffusers FastH3 module name to native ComfyUI H3."""
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
            return (
                f"{target_prefix}{index}.attn.qkv_proj",
                (part * HEAD_WIDTH, HEAD_WIDTH),
            )
        mapped = BLOCK_MAP.get(suffix)
        if mapped is None:
            raise KeyError(f"Unsupported FastH3 adapter module: {name}")
        return f"{target_prefix}{index}.{mapped}", None
    raise KeyError(f"Unsupported FastH3 adapter module: {name}")


def inspect_adapter(path: str | Path) -> FastH3AdapterInfo:
    """Validate the adapter using only its safetensors header."""
    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = {str(k): str(v) for k, v in (handle.metadata() or {}).items()}
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}

    format_name = metadata.get("format", "")
    if format_name and format_name not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported FastH3 adapter format {format_name!r}; "
            f"supported: {sorted(SUPPORTED_FORMATS)}"
        )

    low_rank = 0
    dense = 0
    bias = 0
    gates: set[int] = set()
    consumed: set[str] = set()
    for key in sorted(keys):
        if key.endswith(".lora_A.weight"):
            source = key[:-len(".lora_A.weight")]
            b_key = source + ".lora_B.weight"
            if b_key not in keys:
                raise ValueError(f"FastH3 Adapter is incomplete: missing {b_key}")
            _target, span = adapter_target(source)
            a_shape, b_shape = shapes[key], shapes[b_key]
            if len(a_shape) != 2 or len(b_shape) != 2 or b_shape[1] != a_shape[0]:
                raise ValueError(
                    f"Invalid FastH3 LoRA shapes for {source}: "
                    f"A={a_shape}, B={b_shape}"
                )
            if span is not None and b_shape[0] != span[1]:
                raise ValueError(
                    f"Invalid FastH3 fused QKV shape for {source}: "
                    f"B output={b_shape[0]}, expected {span[1]}"
                )
            low_rank += 1
            consumed.update((key, b_key))
        elif key.endswith(".diff_b"):
            adapter_target(key[:-len(".diff_b")])
            bias += 1
            consumed.add(key)
        elif key.endswith(".diff"):
            adapter_target(key[:-len(".diff")])
            dense += 1
            consumed.add(key)
        elif key.endswith(".attn.to_gate_compress.set_weight"):
            prefix = "transformer_blocks."
            if not key.startswith(prefix):
                raise ValueError(f"Unexpected FastH3 VSA gate key: {key}")
            index = int(key[len(prefix):].split(".", 1)[0])
            if shapes[key] != (HEAD_WIDTH, 5376):
                raise ValueError(
                    f"Invalid FastH3 VSA gate shape at block {index}: "
                    f"{shapes[key]}, expected {(HEAD_WIDTH, 5376)}"
                )
            gates.add(index)
            consumed.add(key)

    unknown = sorted(keys - consumed)
    if unknown:
        raise ValueError(
            "FastH3 Adapter contains unsupported payloads; first keys: "
            + ", ".join(unknown[:5])
        )
    if not (low_rank or dense or bias or gates):
        raise ValueError("Selected file is not a supported FastH3 Adapter")
    if gates and gates != set(range(50)):
        raise ValueError(
            f"FastH3 VSA Adapter is incomplete: expected gates 0..49, found {len(gates)}"
        )
    return FastH3AdapterInfo(
        path=path,
        metadata=metadata,
        low_rank_pairs=low_rank,
        dense_deltas=dense,
        bias_deltas=bias,
        replacement_gates=len(gates),
        requires_vsa=bool(gates),
    )


def load_adapter_payload(path: str | Path):
    """Load validated adapter payloads as ComfyUI patches plus VSA gates."""
    from comfy.weight_adapter import LoRAAdapter

    info = inspect_adapter(path)
    patches = {}
    gates: dict[int, torch.Tensor] = {}
    with safe_open(info.path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            if key.endswith(".lora_A.weight"):
                source = key[:-len(".lora_A.weight")]
                target, span = adapter_target(source)
                a = handle.get_tensor(key)
                b_key = source + ".lora_B.weight"
                b = handle.get_tensor(b_key)
                model_key = f"diffusion_model.{target}.weight"
                patch_key = model_key
                if span is not None:
                    patch_key = (model_key, (0, span[0], span[1]))
                patches[patch_key] = LoRAAdapter(
                    {key, b_key}, (b, a, None, None, None, None)
                )
            elif key.endswith(".diff_b"):
                target, span = adapter_target(key[:-len(".diff_b")])
                if span is not None:
                    raise ValueError(f"Unexpected fused bias delta: {key}")
                patches[f"diffusion_model.{target}.bias"] = (
                    "diff", (handle.get_tensor(key),)
                )
            elif key.endswith(".diff"):
                target, span = adapter_target(key[:-len(".diff")])
                if span is not None:
                    raise ValueError(f"Unexpected fused weight delta: {key}")
                patches[f"diffusion_model.{target}.weight"] = (
                    "diff", (handle.get_tensor(key),)
                )
            elif key.endswith(".attn.to_gate_compress.set_weight"):
                index = int(key[len("transformer_blocks."):].split(".", 1)[0])
                gates[index] = handle.get_tensor(key)
    return info, patches, gates
