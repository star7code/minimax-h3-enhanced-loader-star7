"""FastH3 directory-checkpoint detection and deterministic ComfyUI conversion."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch


OFFICIAL_DENSE_MODEL_ID = (
    "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
)
INFERENCE_SCHEMA = "fasth3-inference-contract-v1"
SAMPLING_PROFILE = "fasth3_4step_v1"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
QUANT_MANIFEST_NAME = "star7_fasth3_int8.json"
VSA_QUANT_MANIFEST_NAME = "star7_fasth3_vsa_int8.json"
VSA_QUANT_SCHEMA = "star7-fasth3-vsa-native-int8-v1"
ALT_INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class FastH3CheckpointInfo:
    root: Path
    transformer: Path
    config: dict
    inference: dict
    checkpoint_content: dict
    checkpoint_metadata: dict
    weight_map: dict[str, str]
    variant: str
    requires_vsa: bool
    attention_backend: str
    sampling_profile: str
    native_quantized: bool = False
    quantization: str = "bf16-dense"


@dataclass(frozen=True)
class StateDictValidationReport:
    source_keys: int
    converted_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


def _read_json(path: Path, *, required: bool = True) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required FastH3 file is missing: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid FastH3 JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"FastH3 JSON root must be an object: {path}")
    return value


def is_fasth3_directory(path: os.PathLike | str) -> bool:
    root = Path(path)
    return (
        (root / "transformer" / "config.json").is_file()
        and (
            (root / "transformer" / INDEX_NAME).is_file()
            or (root / "transformer" / ALT_INDEX_NAME).is_file()
        )
    )


def detect_fasth3_checkpoint(path: os.PathLike | str) -> FastH3CheckpointInfo:
    root = Path(path).resolve()
    transformer = root / "transformer"
    config = _read_json(transformer / "config.json")
    index_path = transformer / INDEX_NAME
    if not index_path.is_file():
        index_path = transformer / ALT_INDEX_NAME
    index = _read_json(index_path)
    vsa_quant_manifest = _read_json(
        root / VSA_QUANT_MANIFEST_NAME, required=False
    )
    if vsa_quant_manifest:
        if vsa_quant_manifest.get("schema_version") != VSA_QUANT_SCHEMA:
            raise ValueError("Unsupported Star7 FastH3 VSA quantized checkpoint schema")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("FastH3 VSA shard index has no weight_map")
        weight_map = {str(key): str(value) for key, value in weight_map.items()}
        gates = {
            int(match.group(1))
            for key in weight_map
            if (match := re.fullmatch(
                r"blocks\.(\d+)\.attn\.gate_compress\.weight", key
            ))
        }
        if gates != set(range(50)):
            raise ValueError(f"FastH3 VSA checkpoint needs gates 0..49, found {len(gates)}")
        return FastH3CheckpointInfo(
            root=root,
            transformer=transformer,
            config=config,
            inference=vsa_quant_manifest,
            checkpoint_content={},
            checkpoint_metadata={},
            weight_map=weight_map,
            variant="fasth3_vsa_datafree_v1",
            requires_vsa=True,
            attention_backend="video_sparse_attn_h3",
            sampling_profile=str(
                vsa_quant_manifest.get("sampling_profile", SAMPLING_PROFILE)
            ),
            native_quantized=True,
            quantization=str(
                vsa_quant_manifest.get("quantization", "int8_tensorwise_convrot")
            ),
        )

    inference = _read_json(root / "fastvideo_inference.json")
    content = _read_json(root / "checkpoint_content.json", required=False)
    metadata = _read_json(root / "checkpoint_metadata.json", required=False)
    quant_manifest = _read_json(root / QUANT_MANIFEST_NAME, required=False)
    native_quantized = bool(quant_manifest)
    if native_quantized and quant_manifest.get("schema_version") != "star7-fasth3-native-int8-v1":
        raise ValueError("Unsupported Star7 FastH3 quantized checkpoint schema")

    if config.get("_class_name") != "MiniMaxH3Transformer3DModel":
        raise ValueError(
            "Directory checkpoint is not a FastVideo MiniMax H3 transformer "
            f"({_class_name(config)})."
        )
    if inference.get("schema_version") != INFERENCE_SCHEMA:
        raise ValueError(
            "Unsupported or missing FastH3 inference contract: "
            f"{inference.get('schema_version')!r}"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("FastH3 shard index has no weight_map")
    weight_map = {str(key): str(value) for key, value in weight_map.items()}

    model_id = str(inference.get("model_id", ""))
    backend = str(inference.get("attention_backend", "")).upper()
    has_vsa_keys = any(
        "to_gate_compress" in key or ".set_weight" in key
        for key in weight_map
    )
    requires_vsa = (
        has_vsa_keys
        or "VSA" in backend
        or "VIDEO_SPARSE" in backend
        or "-VSA-" in model_id.upper()
        or bool(metadata.get("requires_vsa"))
    )
    if requires_vsa:
        variant = "fasth3_vsa_v1"
    elif model_id == OFFICIAL_DENSE_MODEL_ID and backend in {
        "FLASH_ATTN", "TORCH_SDPA", "DENSE"
    }:
        variant = "fasth3_dense_v1"
    else:
        raise ValueError(
            "This Enhanced Loader version supports only the official FastH3 "
            f"Dense Preview v1 checkpoint; detected model_id={model_id!r}, "
            f"attention_backend={backend!r}."
        )

    _validate_sampling_contract(inference)
    _validate_architecture_config(config)
    if not native_quantized:
        _validate_qkv_shard_locality(weight_map)
    return FastH3CheckpointInfo(
        root=root,
        transformer=transformer,
        config=config,
        inference=inference,
        checkpoint_content=content,
        checkpoint_metadata=metadata,
        weight_map=weight_map,
        variant=variant,
        requires_vsa=requires_vsa,
        attention_backend=backend.lower(),
        sampling_profile=SAMPLING_PROFILE,
        native_quantized=native_quantized,
        quantization=str(quant_manifest.get("quantization", "bf16-dense")),
    )


def _class_name(config: Mapping) -> str:
    return str(config.get("_class_name", "unknown class"))


def _validate_sampling_contract(inference: Mapping) -> None:
    expected = {
        "transformer_forwards": 4,
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
    }
    mismatches = [
        f"{name}={inference.get(name)!r} (expected {value!r})"
        for name, value in expected.items()
        if inference.get(name) != value
    ]
    if list(inference.get("dmd_denoising_steps", [])) != [999, 749, 500, 250]:
        mismatches.append(
            "dmd_denoising_steps must be [999, 749, 500, 250]"
        )
    if mismatches:
        raise ValueError("Unsupported FastH3 sampling contract: " + "; ".join(mismatches))


def _validate_architecture_config(config: Mapping) -> None:
    expected = {
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "hidden_size": 5376,
        "num_layers": 50,
        "num_refiner_layers": 2,
        "ffn_dim": 14336,
        "in_channels": 24,
        "audio_in_channels": 32,
        "text_dim": 5120,
        "freq_dim": 256,
        "time_embed_hidden_dim": 5376,
        "time_embed_dim": 2688,
        "rope_freq_dim": 16,
    }
    mismatches = [
        f"{name}={config.get(name)!r} (expected {value!r})"
        for name, value in expected.items()
        if config.get(name) != value
    ]
    if list(config.get("patch_size", [])) != [1, 2, 2]:
        mismatches.append("patch_size must be [1, 2, 2]")
    if mismatches:
        raise ValueError("Unsupported FastH3 architecture: " + "; ".join(mismatches))


def _validate_qkv_shard_locality(weight_map: Mapping[str, str]) -> None:
    groups: dict[str, dict[str, str]] = {}
    for key, shard in weight_map.items():
        match = re.match(r"^(.*\.attn)\.to_([qkv])\.weight$", key)
        if match:
            groups.setdefault(match.group(1), {})[match.group(2)] = shard
    for prefix, parts in groups.items():
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete FastH3 QKV group in index: {prefix}")
        if len(set(parts.values())) != 1:
            raise ValueError(
                f"FastH3 Q/K/V tensors span multiple shards and cannot be "
                f"converted with bounded memory: {prefix}"
            )


_GLOBAL_KEYS = {
    "proj_in.weight": "video_patch_proj.weight",
    "proj_in.bias": "video_patch_proj.bias",
    "audio_proj_in.weight": "audio_patch_proj.weight",
    "audio_proj_in.bias": "audio_patch_proj.bias",
    "context_embedder.weight": "condition_proj.weight",
    "context_embedder.bias": "condition_proj.bias",
    "time_embedder.linear_1.weight": "time_embedder.proj_in.weight",
    "time_embedder.linear_1.bias": "time_embedder.proj_in.bias",
    "time_embedder.linear_2.weight": "time_embedder.proj_out.weight",
    "time_embedder.linear_2.bias": "time_embedder.proj_out.bias",
    "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
    "norm_out.norm.weight": "final_layer.norm.weight",
    "norm_out.linear.weight": "final_layer.adaln_proj.linear.weight",
    "norm_out.linear.bias": "final_layer.adaln_proj.linear.bias",
    "proj_out.weight": "final_layer.video_out.weight",
    "proj_out.bias": "final_layer.video_out.bias",
    "audio_proj_out.weight": "final_layer.audio_out.weight",
    "audio_proj_out.bias": "final_layer.audio_out.bias",
}

_BLOCK_SUFFIXES = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "attn.norm_q.weight": "attn.q_norm.weight",
    "attn.norm_k.weight": "attn.k_norm.weight",
    "attn.to_out.0.weight": "attn.out_proj.weight",
    "ff.net.0.proj.weight": "mlp.fc1.weight",
    "ff.net.2.weight": "mlp.fc2.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
}


def _block_target(key: str) -> tuple[str, str] | None:
    match = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", key)
    if match:
        return f"blocks.{match.group(1)}", match.group(2)
    match = re.match(r"^token_refiner\.refiner_blocks\.(\d+)\.(.+)$", key)
    if match:
        return f"token_refiner.blocks.{match.group(1)}", match.group(2)
    return None


def convert_fasth3_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Convert one complete shard. Official Q/K/V groups are shard-local."""
    converted: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    qkv: dict[str, dict[str, torch.Tensor]] = {}

    for key, tensor in state_dict.items():
        target = _GLOBAL_KEYS.get(key)
        if target:
            converted[target] = tensor
            continue
        block = _block_target(key)
        if not block:
            unexpected.append(key)
            continue
        prefix, suffix = block
        qkv_match = re.fullmatch(r"attn\.to_([qkv])\.weight", suffix)
        if qkv_match:
            qkv.setdefault(prefix, {})[qkv_match.group(1)] = tensor
            continue
        mapped_suffix = _BLOCK_SUFFIXES.get(suffix)
        if mapped_suffix is None:
            unexpected.append(key)
            continue
        converted[f"{prefix}.{mapped_suffix}"] = tensor

    for prefix, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            unexpected.append(f"{prefix}.attn.incomplete_qkv")
            continue
        shapes = {tuple(parts[name].shape) for name in ("q", "k", "v")}
        if len(shapes) != 1:
            raise ValueError(f"FastH3 Q/K/V shapes differ at {prefix}: {sorted(shapes)}")
        converted[f"{prefix}.attn.qkv_proj.weight"] = torch.cat(
            (parts["q"], parts["k"], parts["v"]), dim=0
        )
    return converted, tuple(sorted(unexpected))


def _rope_inv_freq(config: Mapping) -> torch.Tensor:
    length = int(config["rope_freq_dim"])
    theta = float(config.get("rope_theta", 10000.0))
    return 1.0 / (
        theta ** (torch.arange(0, length, dtype=torch.float32) / float(length))
    )


def expected_comfy_keys(config: Mapping) -> set[str]:
    keys = set(_GLOBAL_KEYS.values()) | {"rope.inv_freq"}
    common = {
        "norm1.weight", "norm2.weight", "attn.q_norm.weight",
        "attn.k_norm.weight", "attn.qkv_proj.weight", "attn.out_proj.weight",
        "mlp.fc1.weight", "mlp.fc2.weight",
    }
    for index in range(int(config["num_refiner_layers"])):
        keys.update(f"token_refiner.blocks.{index}.{suffix}" for suffix in common)
    main = common | {"adaln_proj.linear.weight", "adaln_proj.linear.bias"}
    for index in range(int(config["num_layers"])):
        keys.update(f"blocks.{index}.{suffix}" for suffix in main)
    return keys


def validate_fasth3_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping,
    *,
    source_keys: int,
    unexpected: tuple[str, ...] = (),
) -> StateDictValidationReport:
    expected = expected_comfy_keys(config)
    actual = set(state_dict)
    return StateDictValidationReport(
        source_keys=source_keys,
        converted_keys=len(actual),
        missing_keys=tuple(sorted(expected - actual)),
        unexpected_keys=tuple(sorted(set(unexpected) | (actual - expected))),
    )


def load_fasth3_shards(
    info: FastH3CheckpointInfo,
    load_file: Callable[[str], Mapping[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], StateDictValidationReport]:
    if load_file is None:
        import comfy.utils

        load_file = lambda path: comfy.utils.load_torch_file(path)

    shards = sorted(set(info.weight_map.values()))
    missing_files = [name for name in shards if not (info.transformer / name).is_file()]
    if missing_files:
        preview = ", ".join(missing_files[:3])
        raise FileNotFoundError(
            f"FastH3 checkpoint is incomplete: {len(missing_files)} shard(s) "
            f"missing under {info.transformer}; first: {preview}"
        )

    output: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    loaded_source_keys = 0
    for index, shard in enumerate(shards, start=1):
        path = info.transformer / shard
        logging.info(
            "[Star7 H3 Enhanced] Loading FastH3 shard %d/%d: %s",
            index, len(shards), shard,
        )
        source = dict(load_file(str(path)))
        expected_source = {
            key for key, filename in info.weight_map.items() if filename == shard
        }
        missing_in_shard = expected_source - set(source)
        extra_in_shard = set(source) - expected_source
        if missing_in_shard or extra_in_shard:
            raise ValueError(
                f"FastH3 shard index mismatch in {shard}: "
                f"missing={len(missing_in_shard)}, unexpected={len(extra_in_shard)}"
            )
        loaded_source_keys += len(source)
        if info.native_quantized:
            converted, unknown = source, ()
        else:
            converted, unknown = convert_fasth3_state_dict(source)
        overlap = set(output).intersection(converted)
        if overlap:
            raise ValueError(f"Duplicate converted FastH3 keys: {sorted(overlap)[:3]}")
        output.update(converted)
        unexpected.extend(unknown)
        del source

    if not info.native_quantized:
        output["rope.inv_freq"] = _rope_inv_freq(info.config)
        report = validate_fasth3_state_dict(output, info.config, source_keys=loaded_source_keys, unexpected=tuple(unexpected))
    elif info.requires_vsa:
        required = {
            f"blocks.{index}.attn.{name}"
            for index in range(50)
            for name in (
                "qkv_proj.weight",
                "out_proj.weight",
                "gate_compress.weight",
                "gate_compress.weight_scale",
                "gate_compress.comfy_quant",
            )
        }
        actual = set(output)
        report = StateDictValidationReport(
            source_keys=loaded_source_keys,
            converted_keys=len(output),
            missing_keys=tuple(sorted(required - actual)),
            unexpected_keys=(),
        )
    else:
        expected = expected_comfy_keys(info.config)
        expected.difference_update(
            {key for key in expected if key.startswith("time_embedder.")}
        )
        expected.add("adaln_t_table")
        quant_bases = {
            key[:-len(".weight")]
            for key in expected
            if key.startswith(("blocks.", "token_refiner.blocks."))
            and key.endswith((
                ".attn.qkv_proj.weight", ".attn.out_proj.weight",
                ".mlp.fc1.weight", ".mlp.fc2.weight",
            ))
        }
        expected.update(f"{base}.weight_scale" for base in quant_bases)
        expected.update(f"{base}.comfy_quant" for base in quant_bases)
        actual = set(output)
        report = StateDictValidationReport(
            source_keys=loaded_source_keys, converted_keys=len(output),
            missing_keys=tuple(sorted(expected - actual)),
            unexpected_keys=tuple(sorted(actual - expected)),
        )
    if not report.valid:
        raise ValueError(
            "FastH3 conversion validation failed: "
            f"missing={len(report.missing_keys)} {report.missing_keys[:5]}, "
            f"unexpected={len(report.unexpected_keys)} {report.unexpected_keys[:5]}"
        )
    return output, report


def scan_fasth3_directories(roots: list[str]) -> list[str]:
    """Return portable, slash-suffixed directory entries for the model combo."""
    found: set[str] = set()
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for current, dirs, _files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in {".cache", ".git"}]
            current_path = Path(current)
            if is_fasth3_directory(current_path):
                relative = current_path.relative_to(root).as_posix().rstrip("/") + "/"
                found.add(relative)
                dirs[:] = []
    return sorted(found, key=str.casefold)
