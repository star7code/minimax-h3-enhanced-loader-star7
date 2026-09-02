import importlib.util
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    name = "star7_fasth3_adapter_tests"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "fasth3_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_header_inspection_accepts_complete_dense_adapter():
    module = load_module()
    with tempfile.TemporaryDirectory() as value:
        path = Path(value) / "adapter.safetensors"
        save_file(
            {
                "proj_in.lora_A.weight": torch.zeros(2, 3),
                "proj_in.lora_B.weight": torch.zeros(4, 2),
                "proj_out.diff": torch.zeros(3, 4),
                "proj_out.diff_b": torch.zeros(3),
            },
            path,
            metadata={"format": "fastvideo-lora-v2"},
        )
        info = module.inspect_adapter(path)
        assert info.low_rank_pairs == 1
        assert info.dense_deltas == 1
        assert info.bias_deltas == 1
        assert info.replacement_gates == 0
        assert info.requires_vsa is False


def test_header_inspection_rejects_partial_vsa_adapter():
    module = load_module()
    with tempfile.TemporaryDirectory() as value:
        path = Path(value) / "adapter.safetensors"
        save_file(
            {
                "transformer_blocks.0.attn.to_gate_compress.set_weight":
                    torch.zeros(4, 3),
            },
            path,
            metadata={"format": "fastvideo-lora-v2"},
        )
        try:
            module.inspect_adapter(path)
        except ValueError as exc:
            assert "FastH3 VSA gate" in str(exc)
        else:
            raise AssertionError("partial VSA adapter was accepted")


def test_payload_maps_qkv_to_non_overlapping_fused_offsets():
    module = load_module()
    with tempfile.TemporaryDirectory() as value:
        path = Path(value) / "adapter.safetensors"
        tensors = {}
        for name in ("q", "k", "v"):
            base = f"transformer_blocks.0.attn.to_{name}"
            tensors[base + ".lora_A.weight"] = torch.zeros(2, 3)
            tensors[base + ".lora_B.weight"] = torch.zeros(module.HEAD_WIDTH, 2)
        save_file(tensors, path, metadata={"format": "fastvideo-lora-v2"})
        info, patches, gates = module.load_adapter_payload(path)
        offsets = sorted(
            key[1][1]
            for key in patches
            if isinstance(key, tuple)
        )
        assert offsets == [0, module.HEAD_WIDTH, module.HEAD_WIDTH * 2]
        assert info.low_rank_pairs == 3
        assert gates == {}


if __name__ == "__main__":
    test_header_inspection_accepts_complete_dense_adapter()
    test_header_inspection_rejects_partial_vsa_adapter()
    test_payload_maps_qkv_to_non_overlapping_fused_offsets()
    print("FastH3 Adapter tests passed")
