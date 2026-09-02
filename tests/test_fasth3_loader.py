import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    name = "star7_fasth3_loader_tests"
    spec = importlib.util.spec_from_file_location(name, ROOT / "fasth3_loader.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_config():
    return {
        "_class_name": "MiniMaxH3Transformer3DModel",
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
        "rope_theta": 10000.0,
        "patch_size": [1, 2, 2],
    }


def official_index():
    global_keys = {
        "proj_in.weight", "proj_in.bias", "audio_proj_in.weight",
        "audio_proj_in.bias", "context_embedder.weight",
        "context_embedder.bias", "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias", "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias", "token_refiner.final_norm.weight",
        "norm_out.norm.weight", "norm_out.linear.weight",
        "norm_out.linear.bias", "proj_out.weight", "proj_out.bias",
        "audio_proj_out.weight", "audio_proj_out.bias",
    }
    common = {
        "norm1.weight", "norm2.weight", "attn.norm_q.weight",
        "attn.norm_k.weight", "attn.to_q.weight", "attn.to_k.weight",
        "attn.to_v.weight", "attn.to_out.0.weight",
        "ff.net.0.proj.weight", "ff.net.2.weight",
    }
    keys = set(global_keys)
    for index in range(2):
        keys.update(
            f"token_refiner.refiner_blocks.{index}.{suffix}"
            for suffix in common
        )
    for index in range(50):
        keys.update(f"transformer_blocks.{index}.{suffix}" for suffix in common)
        keys.add(f"transformer_blocks.{index}.adaln_proj.linear.weight")
        keys.add(f"transformer_blocks.{index}.adaln_proj.linear.bias")
    return {
        "metadata": {"total_size": 66245985792},
        "weight_map": {
            key: "diffusion_pytorch_model-00001-of-00013.safetensors"
            for key in sorted(keys)
        },
    }


def write_fixture(root, *, backend="FLASH_ATTN", model_id=None, vsa_key=False):
    module = load_module()
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    config = official_config()
    index = official_index()
    if vsa_key:
        index["weight_map"]["transformer_blocks.0.attn.to_gate_compress.weight"] = (
            "diffusion_pytorch_model-00001-of-00013.safetensors"
        )
    inference = {
        "schema_version": module.INFERENCE_SCHEMA,
        "model_id": model_id or module.OFFICIAL_DENSE_MODEL_ID,
        "attention_backend": backend,
        "transformer_forwards": 4,
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        "dmd_denoising_steps": [999, 749, 500, 250],
    }
    (transformer / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (transformer / module.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    (root / "fastvideo_inference.json").write_text(json.dumps(inference), encoding="utf-8")
    return module, config, index


def test_official_dense_directory_and_metadata():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        module, _config, _index = write_fixture(root)
        assert module.is_fasth3_directory(root)
        info = module.detect_fasth3_checkpoint(root)
        assert info.variant == "fasth3_dense_v1"
        assert info.sampling_profile == "fasth3_4step_v1"
        assert info.requires_vsa is False
        assert info.attention_backend == "flash_attn"


def test_official_index_converts_exactly_to_comfy_keys():
    module = load_module()
    config = official_config()
    index = official_index()
    source = {key: torch.zeros((1, 1)) for key in index["weight_map"]}
    converted, unexpected = module.convert_fasth3_state_dict(source)
    converted["rope.inv_freq"] = module._rope_inv_freq(config)
    report = module.validate_fasth3_state_dict(
        converted, config, source_keys=len(source), unexpected=unexpected
    )
    assert len(source) == 638
    assert report.converted_keys == 535
    assert report.valid
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()


def test_qkv_fusion_order_is_q_k_v():
    module = load_module()
    source = {
        "transformer_blocks.0.attn.to_q.weight": torch.full((1, 2), 1.0),
        "transformer_blocks.0.attn.to_k.weight": torch.full((1, 2), 2.0),
        "transformer_blocks.0.attn.to_v.weight": torch.full((1, 2), 3.0),
    }
    converted, unexpected = module.convert_fasth3_state_dict(source)
    assert unexpected == ()
    fused = converted["blocks.0.attn.qkv_proj.weight"]
    assert fused[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_missing_and_unexpected_keys_are_reported():
    module = load_module()
    config = official_config()
    state = {"video_patch_proj.weight": torch.zeros(1), "alien": torch.zeros(1)}
    report = module.validate_fasth3_state_dict(
        state, config, source_keys=2, unexpected=("source.alien",)
    )
    assert not report.valid
    assert "video_patch_proj.bias" in report.missing_keys
    assert "alien" in report.unexpected_keys
    assert "source.alien" in report.unexpected_keys


def test_vsa_is_detected_and_requires_complete_shards():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        module, _config, index = write_fixture(
            root,
            backend="VIDEO_SPARSE_ATTN",
            model_id="FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
            vsa_key=True,
        )
        info = module.detect_fasth3_checkpoint(root)
        assert info.variant == "fasth3_vsa_v1"
        assert info.requires_vsa
        try:
            module.load_fasth3_shards(info)
        except FileNotFoundError as exc:
            assert "checkpoint is incomplete" in str(exc)
        else:
            raise AssertionError("incomplete VSA fixture unexpectedly loaded")


def test_unknown_dense_variant_fails_clearly():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        module, _config, _index = write_fixture(
            root, model_id="example/not-supported"
        )
        try:
            module.detect_fasth3_checkpoint(root)
        except ValueError as exc:
            assert "verified official FastH3 Preview releases" in str(exc)
        else:
            raise AssertionError("Unknown model variant did not fail")


def test_official_v02_modular_directory_is_versioned():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        module = load_module()
        transformer = root / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text(
            json.dumps(official_config()), encoding="utf-8"
        )
        v02_index = official_index()
        for index in range(50):
            v02_index["weight_map"][
                f"transformer_blocks.{index}.attn.to_gate_compress.weight"
            ] = "diffusion_pytorch_model-00001-of-00014.safetensors"
        (transformer / module.INDEX_NAME).write_text(
            json.dumps(v02_index), encoding="utf-8"
        )
        modular = {
            "_class_name": "MiniMaxH3ModularPipeline",
            "transformer": [
                "diffusers", "MiniMaxH3Transformer3DModel",
                {
                    "pretrained_model_name_or_path":
                        "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2",
                    "subfolder": "transformer",
                },
            ],
        }
        (root / "modular_model_index.json").write_text(
            json.dumps(modular), encoding="utf-8"
        )
        info = module.detect_fasth3_checkpoint(root)
        assert info.variant == "fasth3_vsa_v0_2"
        assert info.sampling_profile == "fasth3_4step_v0_2"
        assert info.requires_vsa is True


def test_v02_gate_maps_to_native_vsa_weight():
    module = load_module()
    source = {
        "transformer_blocks.0.attn.to_gate_compress.weight": torch.zeros(4, 3)
    }
    converted, unexpected = module.convert_fasth3_state_dict(source)
    assert unexpected == ()
    assert "blocks.0.attn.gate_compress.weight" in converted


def test_cross_shard_qkv_uses_bounded_pending_group():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        module, _config, index = write_fixture(root)
        weight_map = index["weight_map"]
        shard_a = "diffusion_pytorch_model-00001-of-00002.safetensors"
        shard_b = "diffusion_pytorch_model-00002-of-00002.safetensors"
        for key in weight_map:
            weight_map[key] = shard_a
        weight_map["transformer_blocks.0.attn.to_v.weight"] = shard_b
        (root / "transformer" / module.INDEX_NAME).write_text(
            json.dumps(index), encoding="utf-8"
        )
        for name in (shard_a, shard_b):
            (root / "transformer" / name).touch()
        info = module.detect_fasth3_checkpoint(root)

        def fake_load(path):
            name = Path(path).name
            return {
                key: torch.zeros((1, 1))
                for key, filename in weight_map.items()
                if filename == name
            }

        state, report = module.load_fasth3_shards(info, load_file=fake_load)
        assert report.valid
        assert state["blocks.0.attn.qkv_proj.weight"].shape == (3, 1)


if __name__ == "__main__":
    test_official_dense_directory_and_metadata()
    test_official_index_converts_exactly_to_comfy_keys()
    test_qkv_fusion_order_is_q_k_v()
    test_missing_and_unexpected_keys_are_reported()
    test_vsa_is_detected_and_requires_complete_shards()
    test_unknown_dense_variant_fails_clearly()
    test_official_v02_modular_directory_is_versioned()
    test_v02_gate_maps_to_native_vsa_weight()
    test_cross_shard_qkv_uses_bounded_pending_group()
    print("FastH3 Enhanced Loader mapping tests passed")
