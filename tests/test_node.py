import importlib.util
import copy
import gc
import json
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = next(
    candidate
    for candidate in (
        Path(os.environ.get("COMFYUI_ROOT", "")),
        ROOT.parents[1],
        Path.cwd(),
        Path.cwd() / "ComfyUI",
    )
    if (candidate / "folder_paths.py").is_file()
)
if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))


def load_nodes():
    package_name = "star7_h3_test_package"
    package = sys.modules.get(package_name)
    if package is None:
        package = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
        )
        package.__path__ = [str(ROOT)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.nodes"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_registration():
    module = load_nodes()
    assert set(module.NODE_CLASS_MAPPINGS) == {
        "MiniMaxH3EnhancedLoaderStar7",
        "MiniMaxH3VSASwitchStar7",
    }
    for display_name in module.NODE_DISPLAY_NAME_MAPPINGS.values():
        assert display_name.endswith("Star7")


def test_loader_exposes_only_model_selector():
    module = load_nodes()
    with mock.patch.object(module, "_model_source_choices", return_value=["h3.safetensors"]):
        required = module.MiniMaxH3EnhancedLoaderStar7.INPUT_TYPES()["required"]
    assert set(required) == {"unet_name"}


def test_loader_clears_stale_chunk_rope_without_importing_chunk_project():
    module = load_nodes()
    import comfy.quant_ops as quant_ops

    def original(*args, **kwargs):
        return args, kwargs

    def stale_dispatch(*args, **kwargs):
        raise AssertionError("stale Chunk dispatcher should have been removed")

    stale_dispatch.__name__ = "_chunked_rms_rope_split_half_inplace"
    stale_dispatch.__module__ = "minimax-h3-chunk-star7.nodes"
    stale_dispatch._star7_original = original
    with mock.patch.object(quant_ops.ck, "rms_rope_split_half_", stale_dispatch):
        assert module._restore_upstream_rope_dispatch() is True
        assert quant_ops.ck.rms_rope_split_half_ is original


def test_loader_leaves_unrelated_rope_dispatch_untouched():
    module = load_nodes()
    import comfy.quant_ops as quant_ops

    def unrelated(*args, **kwargs):
        return args, kwargs

    with mock.patch.object(quant_ops.ck, "rms_rope_split_half_", unrelated):
        assert module._restore_upstream_rope_dispatch() is False
        assert quant_ops.ck.rms_rope_split_half_ is unrelated


def test_fasth3_preset_requires_fast_h3_metadata():
    module = load_nodes()
    model = SimpleNamespace(model_options={"transformer_options": {}})
    with mock.patch.object(model, "clone", return_value=model, create=True):
        try:
            module._apply_fasth3_vsa_switch(model)
        except ValueError as exc:
            assert "requires a FastH3 model" in str(exc)
        else:
            raise AssertionError("non-FastH3 model was accepted by the preset")


def test_fasth3_vsa_switch_validates_and_marks_vsa_model():
    module = load_nodes()

    class FakeModel:
        def __init__(self):
            self.model_options = {
                "transformer_options": {
                    "star7_h3": {
                        "variant": "fasth3_vsa_datafree_v1",
                        "sampling_profile": "fasth3_4step_dmd_999_749_500_250_cfg1",
                        "requires_vsa": True,
                    }
                }
            }
            self.model = SimpleNamespace(model_config=object())

        def clone(self):
            cloned = FakeModel()
            cloned.model_options = copy.deepcopy(self.model_options)
            cloned.model = self.model
            return cloned

    fake_model = FakeModel()
    with mock.patch.object(module, "_enable_vsa_runtime", side_effect=lambda model: model) as enable:
        patched = module._apply_fasth3_vsa_switch(fake_model)
    enable.assert_called_once_with(patched)
    switch = patched.model_options["transformer_options"]["star7_h3_vsa_switch"]
    assert switch == {
        "variant": "fasth3_vsa_datafree_v1",
        "vsa_acceleration": True,
        "attention_backend": "video_sparse_attn_h3",
    }


def test_scale_constants_are_powers_of_two():
    module = load_nodes()
    for value in (module.K_OUT_PROJ, module.K_FC2):
        integer = int(value)
        assert integer > 0 and integer & (integer - 1) == 0


def test_fasth3_mlp_uses_value_first_swiglu_layout():
    module = load_nodes()

    class FakeMLP:
        fc1 = torch.nn.Linear(2, 4, bias=False)
        fc2 = torch.nn.Linear(2, 2, bias=False)

    with torch.no_grad():
        FakeMLP.fc1.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
        FakeMLP.fc2.weight.copy_(torch.eye(2))

    wrapped = module._mlp_forward(lambda _self, _tensor: None, fast_h3=True)
    value = torch.tensor([[1.5, -0.5]])
    projected = FakeMLP.fc1(value)
    expected = torch.nn.functional.silu(projected[:, 2:]).mul(projected[:, :2])
    assert torch.allclose(wrapped(FakeMLP(), value), expected)


def test_fasth3_int8_fc2_fuses_value_first_swiglu_with_fp32_output():
    module = load_nodes()

    class FakeQuantizedTensor:
        _layout_cls = "TensorWiseINT8Layout"
        _params = SimpleNamespace(
            transposed=False, convrot=True, convrot_groupsize=256
        )

    fake_weight = FakeQuantizedTensor()
    fake_linear = SimpleNamespace(weight=fake_weight)
    fake_mlp = SimpleNamespace(fc2=fake_linear)
    projected = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
    captured = {}

    def int8_linear(x, qdata, scale, bias, out_dtype, **kwargs):
        captured.update(
            x=x.clone(), qdata=qdata, scale=scale, bias=bias,
            out_dtype=out_dtype, kwargs=kwargs,
        )
        return torch.ones(1, 2, dtype=out_dtype)

    with (
        mock.patch.object(module.comfy.ops, "QuantizedTensor", FakeQuantizedTensor),
        mock.patch.object(
            module.comfy.ops,
            "cast_bias_weight",
            return_value=(fake_weight, None, None),
        ),
        mock.patch.object(module.comfy.ops, "uncast_bias_weight") as uncast,
        mock.patch.object(
            module.comfy.ops.TensorWiseINT8Layout,
            "get_plain_tensors",
            return_value=("qdata", "scale"),
            create=True,
        ),
        mock.patch.object(
            module.comfy.ops.quant_ops.ck, "int8_linear", side_effect=int8_linear
        ),
    ):
        result = module._fast_h3_fused_int8_fc2(fake_mlp, projected)

    assert result.dtype is torch.float32
    assert torch.equal(
        captured["x"], torch.tensor([[3.0, 4.0, 1.0, 2.0]], dtype=torch.float16)
    )
    assert captured["out_dtype"] is torch.float32
    assert captured["kwargs"] == {
        "convrot": True,
        "convrot_groupsize": 256,
        "input_act": "swiglu",
    }
    uncast.assert_called_once_with(fake_linear, fake_weight, None, None)


def test_fasth3_int8_fc2_fusion_can_be_disabled_for_ab_testing():
    module = load_nodes()
    projected = torch.ones(1, 4, dtype=torch.float16)
    fake_mlp = SimpleNamespace(fc2=SimpleNamespace(weight=object()))
    with mock.patch.dict(
        os.environ, {module.FAST_H3_FUSED_INT8_FC2_ENV: "0"}
    ):
        assert module._fast_h3_fused_int8_fc2(fake_mlp, projected) is None


def test_fasth3_modulation_fusion_can_be_disabled_for_ab_testing():
    module = load_nodes()
    with mock.patch.dict(
        os.environ, {module.FAST_H3_FUSED_MODULATION_ENV: "0"}
    ):
        assert not module._fast_h3_fused_modulation_enabled()


def test_fasth3_modulation_fuses_only_scalar_segments():
    module = load_nodes()

    class FakeTensor:
        device = SimpleNamespace(type="cuda")
        dtype = torch.float32
        ndim = 2

        @staticmethod
        def stride(_dimension):
            return 1

    with mock.patch.object(module.fasth3_modulation, "triton", object()):
        assert module.fasth3_modulation.can_fuse(FakeTensor(), [(0, 8, 0)])
        assert not module.fasth3_modulation.can_fuse(
            FakeTensor(), [(0, 8, torch.zeros(8, dtype=torch.long))]
        )


def test_hardware_policy_uses_capability_not_product_names():
    module = load_nodes()
    assert module.detect_hardware_policy(
        cuda_available=True, hip=None, capability=(8, 0), force_fp16=False
    )["name"] == "native_bf16"
    assert module.detect_hardware_policy(
        cuda_available=True, hip=None, capability=(12, 0), force_fp16=False
    )["name"] == "native_bf16"
    sm80_fp16 = module.detect_hardware_policy(
        cuda_available=True, hip=None, capability=(8, 0), force_fp16=True
    )
    assert sm80_fp16["name"] == "native_bf16"
    assert sm80_fp16["apply_fp16_exact"] is False
    assert sm80_fp16["launcher_fp16_overridden"] is True
    assert "corrected to native BF16" in sm80_fp16["reason"]
    assert module.detect_hardware_policy(
        cuda_available=True, hip=None, capability=(7, 5)
    )["name"] == "fp16_compat"
    assert module.detect_hardware_policy(
        cuda_available=True, hip=None, capability=(6, 1)
    )["name"] == "native_default"
    assert module.detect_hardware_policy(
        cuda_available=True, hip="6.3", capability=(9, 4)
    )["name"] == "fp16_compat"


def test_native_bf16_policy_skips_fp16_wrappers():
    module = load_nodes()
    loaded = SimpleNamespace(clone=lambda: loaded)
    with (
        mock.patch.object(module, "detect_hardware_policy", return_value={
            "name": "native_bf16", "apply_fp16_exact": False, "reason": "sm80"
        }),
        mock.patch.object(module.folder_paths, "get_full_path_or_raise", return_value="h3.safetensors"),
        mock.patch.object(module.comfy.sd, "load_diffusion_model", return_value=loaded) as native_load,
        mock.patch.object(module, "_loaded_h3_object", return_value=object()),
        mock.patch.object(module, "detect_model_variant", return_value=("base_h3", "dense")),
        mock.patch.object(module, "_set_enhanced_metadata", side_effect=lambda model, **_kwargs: model),
        mock.patch.object(module, "_load_h3_native_fp16") as fp16_load,
    ):
        assert module.load_model_with_policy("h3.safetensors") is loaded
    native_load.assert_called_once_with(
        "h3.safetensors",
        model_options={"dtype": torch.bfloat16},
        disable_dynamic=False,
    )
    fp16_load.assert_not_called()


def test_compatibility_policy_keeps_fp16_exact_loader():
    module = load_nodes()
    loaded = object()
    with (
        mock.patch.object(module, "detect_hardware_policy", return_value={
            "name": "fp16_compat", "apply_fp16_exact": True, "reason": "sm75"
        }),
        mock.patch.object(module.folder_paths, "get_full_path_or_raise", return_value="h3.safetensors"),
        mock.patch.object(module, "_load_h3_native_fp16", return_value=loaded) as fp16_load,
        mock.patch.object(module, "detect_model_variant", return_value=("base_h3_quantized", "int8")),
        mock.patch.object(module, "_set_enhanced_metadata", side_effect=lambda model, **_kwargs: model),
        mock.patch.object(module.comfy.sd, "load_diffusion_model") as native_load,
    ):
        assert module.load_model_with_policy("h3.safetensors") is loaded
    fp16_load.assert_called_once_with("h3.safetensors", fast_h3=False)
    native_load.assert_not_called()


def test_model_selector_hides_individual_fasth3_shards():
    module = load_nodes()
    with (
        mock.patch.object(
            module.folder_paths,
            "get_filename_list",
            return_value=[
                "base_h3.safetensors",
                "FastH3-Dense-v1/transformer/diffusion_pytorch_model-00001-of-00013.safetensors",
            ],
        ),
        mock.patch.object(module.folder_paths, "get_folder_paths", return_value=["models"]),
        mock.patch.object(
            module,
            "scan_fasth3_directories",
            return_value=["FastH3-Dense-v1/"],
        ),
    ):
        assert module._model_source_choices() == [
            "base_h3.safetensors",
            "FastH3-Dense-v1/",
        ]


def test_fasth3_dense_load_records_sampling_metadata_without_scheduler_patch():
    module = load_nodes()

    class FakeModel:
        def __init__(self):
            self.model_options = {"transformer_options": {}}
            self.cached_patcher_init = None

        def clone(self):
            return self

    model = FakeModel()
    info = SimpleNamespace(
        root=Path("FastH3-Dense-v1"),
        requires_vsa=False,
        variant="fasth3_dense_v1",
        sampling_profile=module.SAMPLING_PROFILE,
    )
    report = SimpleNamespace(source_keys=638, converted_keys=535)
    policy = {
        "name": "native_bf16", "apply_fp16_exact": False, "reason": "sm80"
    }
    with (
        mock.patch.object(module, "detect_fasth3_checkpoint", return_value=info),
        mock.patch.object(module, "load_fasth3_shards", return_value=({"weights": 1}, report)),
        mock.patch.object(module, "_detect_h3_config", return_value=object()),
        mock.patch.object(module, "detect_hardware_policy", return_value=policy),
        mock.patch.object(module.comfy.sd, "load_diffusion_model_state_dict", return_value=model) as load_model,
        mock.patch.object(module, "_loaded_h3_object", return_value=object()),
        mock.patch.object(module, "_patch_h3_model", return_value=model) as patch_model,
    ):
        result = module._load_fasth3_directory("FastH3-Dense-v1")

    assert result is model
    assert result.model_options["transformer_options"]["star7_h3"] == {
        "loader": "enhanced",
        "version": module.NODE_VERSION,
        "variant": "fasth3_dense_v1",
        "source": "hf-directory",
        "sampling_profile": "fasth3_4step_v1",
        "requires_vsa": False,
        "precision_policy": "native_bf16",
        "quantization": "bf16-dense",
    }
    assert result.cached_patcher_init == (
        module._load_fasth3_directory,
        (str(info.root),),
    )
    assert load_model.call_args.kwargs["model_options"] == {
        "dtype": torch.bfloat16,
    }
    assert load_model.call_args.kwargs["disable_dynamic"] is False
    patch_model.assert_called_once_with(
        model,
        loader_native=True,
        fast_h3=True,
        apply_fp16_exact=False,
    )


def test_non_h3_model_fails_clearly():
    module = load_nodes()
    fake = SimpleNamespace(get_model_object=lambda _name: object())
    try:
        module._loaded_h3_object(fake)
    except TypeError as exc:
        assert "native ComfyUI MiniMaxH3Model" in str(exc)
    else:
        raise AssertionError("Non-H3 model was accepted")


def test_fasth3_converted_shapes_detect_native_comfy_h3_config():
    module = load_nodes()
    meta = lambda *shape: torch.empty(shape, device="meta")
    state = {
        "video_patch_proj.weight": meta(5376, 96),
        "audio_patch_proj.weight": meta(5376, 32),
        "final_layer.video_out.weight": meta(96, 5376),
        "final_layer.audio_out.weight": meta(32, 5376),
        "blocks.0.attn.q_norm.weight": meta(128),
        "blocks.0.attn.qkv_proj.weight": meta(21504, 5376),
        "blocks.0.mlp.fc1.weight": meta(28672, 5376),
        "condition_proj.weight": meta(5376, 5120),
        "time_embedder.proj_in.weight": meta(5376, 256),
        "time_embedder.proj_out.weight": meta(2688, 5376),
        "rope.inv_freq": meta(16),
    }
    for index in range(50):
        state[f"blocks.{index}.norm1.weight"] = meta(5376)
    for index in range(2):
        state[f"token_refiner.blocks.{index}.norm1.weight"] = meta(5376)
    config = module._detect_h3_config(state, {})
    assert isinstance(config, module.comfy.supported_models.MiniMaxH3)
    assert config.unet_config["num_layers"] == 50
    assert config.unet_config["token_refiner_num_layers"] == 2
    assert config.unet_config["num_attention_heads"] == 56
    assert config.unet_config["attention_head_dim"] == 128


def make_h3_patcher(module, quantized=False):
    import comfy.ldm.minimax.model as minimax

    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = torch.nn.Linear(2, 2, bias=False)

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(2, 4, bias=False)
            self.fc2 = torch.nn.Linear(2, 2, bias=False)

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TinyAttention()
            self.mlp = TinyMLP()

        def forward(self, *args, **kwargs):
            return args[0]

    diffusion = minimax.MiniMaxH3Model.__new__(minimax.MiniMaxH3Model)
    torch.nn.Module.__init__(diffusion)
    diffusion.condition_proj = torch.nn.Linear(2, 2, bias=False)
    diffusion.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
    if quantized:
        linear = diffusion.blocks[0].mlp.fc1
        linear.quant_format = "int8_tensorwise"
        linear.layout_type = "TensorWiseINT8Layout"
        linear.weight._params = SimpleNamespace(convrot=True)

    class FakePatcher:
        def __init__(self):
            self.diffusion = diffusion
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}
            self.compute_dtype = None
            self.force_cast_weights = False
            self.patches = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.model_options = copy.deepcopy(self.model_options)
            cloned.object_patches = self.object_patches.copy()
            return cloned

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return self.diffusion

        def set_model_compute_dtype(self, dtype):
            self.compute_dtype = dtype
            self.force_cast_weights = dtype is not None
            self.add_object_patch("manual_cast_dtype", dtype)

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

    return FakePatcher(), diffusion


def apply_node(module, patcher):
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)),
    ):
        return module._patch_h3_model(patcher)


def test_dense_model_patch_is_scoped_and_complete():
    module = load_nodes()
    patcher, diffusion = make_h3_patcher(module)
    patched = apply_node(module, patcher)
    assert patched is not patcher
    assert patched.compute_dtype is torch.float16
    assert patched.force_cast_weights is True
    assert patched.model_options["transformer_options"][module.PATCH_FLAG] == module.NODE_VERSION
    assert patched.model_options["transformer_options"][module.PATCH_MODE] == "postload-dense"
    assert len(patched.object_patches) == 2 + 3 * len(diffusion.blocks)


def test_quantized_model_preserves_native_dispatch():
    module = load_nodes()
    patcher, _diffusion = make_h3_patcher(module, quantized=True)
    patched = apply_node(module, patcher)
    assert patched.compute_dtype is torch.float16
    assert patched.force_cast_weights is False
    assert patched.object_patches["manual_cast_dtype"] is torch.float16
    assert patched.model_options["transformer_options"][module.PATCH_MODE] == "postload-quantized"


def test_model_wrappers_are_weakly_bound_and_idempotent():
    module = load_nodes()
    patcher, diffusion = make_h3_patcher(module)
    patched = apply_node(module, patcher)

    wrapper = patched.object_patches["diffusion_model.blocks.0.forward"]
    assert isinstance(wrapper.__self__, weakref.ProxyTypes)
    assert all(
        cell.cell_contents is not diffusion
        for cell in (wrapper.__func__.__closure__ or ())
    )

    reapplied = apply_node(module, patched)
    assert len(reapplied.object_patches) == len(patched.object_patches)
    assert reapplied.model_options["transformer_options"][module.PATCH_FLAG] == module.NODE_VERSION

    diffusion_ref = weakref.ref(diffusion)
    del reapplied, patched, patcher, diffusion
    gc.collect()
    assert diffusion_ref() is None


def test_quantization_summary_reports_convrot():
    module = load_nodes()
    _patcher, diffusion = make_h3_patcher(module, quantized=True)
    assert module._quantization_summary(diffusion) == {
        "int8_tensorwise+convrot": 1,
    }


def test_native_loader_builds_fp16_operations_before_model_creation():
    module = load_nodes()
    state_dict = {"marker": object()}
    metadata = {"version": 1}
    model_config = SimpleNamespace(quant_config={"layer": {"format": "int8_tensorwise"}})
    operations = object()
    base_model = SimpleNamespace()
    patched_model = SimpleNamespace()

    with (
        mock.patch.object(
            module.comfy.utils,
            "load_torch_file",
            return_value=(state_dict, metadata),
        ),
        mock.patch.object(
            module.comfy.utils,
            "convert_old_quants",
            return_value=(state_dict, metadata),
        ),
        mock.patch.object(module, "_detect_h3_config", return_value=model_config),
        mock.patch.object(
            module.comfy.model_management,
            "get_torch_device",
            return_value=torch.device("cuda"),
        ),
        mock.patch.object(
            module.comfy.ops,
            "pick_operations",
            return_value=operations,
        ) as pick_operations,
        mock.patch.object(
            module.comfy.sd,
            "load_diffusion_model_state_dict",
            return_value=base_model,
        ) as load_state_dict,
        mock.patch.object(
            module,
            "_patch_h3_model",
            return_value=patched_model,
        ) as patch_h3,
    ):
        result = module._load_h3_native_fp16("model.safetensors")

    assert result is patched_model
    assert result.cached_patcher_init == (
        module._load_h3_native_fp16,
        ("model.safetensors", False),
    )
    pick_operations.assert_called_once_with(
        torch.float16,
        torch.float16,
        load_device=torch.device("cuda"),
        model_config=model_config,
    )
    options = load_state_dict.call_args.kwargs["model_options"]
    assert options == {
        "dtype": torch.float16,
        "custom_operations": operations,
    }
    assert load_state_dict.call_args.kwargs["disable_dynamic"] is False
    patch_h3.assert_called_once_with(
        base_model, loader_native=True, fast_h3=False, apply_fp16_exact=True
    )


def test_model_detection_strips_diffusion_prefix():
    module = load_nodes()
    state_dict = {"model.blocks.0.weight": object()}
    stripped = {"blocks.0.weight": state_dict["model.blocks.0.weight"]}
    config = object.__new__(module.comfy.supported_models.MiniMaxH3)

    with (
        mock.patch.object(
            module.comfy.model_detection,
            "unet_prefix_from_state_dict",
            return_value="model.",
        ),
        mock.patch.object(
            module.comfy.utils,
            "state_dict_prefix_replace",
            return_value=stripped,
        ) as replace_prefix,
        mock.patch.object(
            module.comfy.model_detection,
            "model_config_from_unet",
            return_value=config,
        ) as detect_config,
    ):
        assert module._detect_h3_config(state_dict, {}) is config

    replace_prefix.assert_called_once_with(
        state_dict, {"model.": ""}, filter_keys=True
    )
    detect_config.assert_called_once_with(stripped, "", metadata={})


def test_normalize_keeps_embedded_quantization_configs():
    module = load_nodes()
    quant = torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)
    state_dict = {
        "video_patch_proj.weight": object(),
        "blocks.0.attn.qkv_proj.comfy_quant": quant,
    }

    normalized, metadata = module._normalize_h3_state_dict(state_dict, {})

    assert normalized == state_dict
    assert metadata == {}
    assert module.comfy.utils.detect_layer_quantization(normalized, "") == {
        "mixed_ops": True,
    }


def test_normalize_restores_unprefixed_legacy_quants_after_prefix_removal():
    module = load_nodes()
    prefix = "model.diffusion_model."
    state_dict = {
        f"{prefix}video_patch_proj.weight": object(),
        f"{prefix}audio_patch_proj.weight": object(),
        f"{prefix}blocks.0.attn.qkv_proj.weight": object(),
        f"{prefix}blocks.0.attn.q_norm.weight": object(),
        f"{prefix}blocks.0.attn.k_norm.weight": object(),
        f"{prefix}blocks.0.mlp.fc1.weight": object(),
    }
    layer_config = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }
    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": {
                "blocks.0.attn.qkv_proj": layer_config,
            }}
        )
    }

    normalized, normalized_metadata = module._normalize_h3_state_dict(
        state_dict, metadata
    )

    assert "video_patch_proj.weight" in normalized
    assert "audio_patch_proj.weight" in normalized
    assert "blocks.0.attn.qkv_proj.comfy_quant" in normalized
    assert not any(key.startswith(prefix) for key in normalized)
    restored = bytes(
        normalized["blocks.0.attn.qkv_proj.comfy_quant"].tolist()
    ).decode("utf-8")
    assert json.loads(restored) == layer_config
    assert normalized_metadata == metadata
    assert module.comfy.utils.detect_layer_quantization(normalized, "") == {
        "mixed_ops": True,
    }


def test_single_file_vsa_manifest_is_detected():
    module = load_nodes()
    manifest = {
        "variant": "fasth3_vsa_datafree_v1",
        "sampling_profile": "fasth3_4step_dmd_999_749_500_250_cfg1",
        "quantization": "int8_tensorwise_convrot",
    }

    handle = mock.MagicMock()
    handle.__enter__.return_value = handle
    handle.metadata.return_value = {
        "star7_fasth3_manifest": json.dumps(manifest),
    }
    with mock.patch.object(module, "safe_open", return_value=handle):
        detected = module._single_file_fasth3_manifest("FastH3-VSA.safetensors")

    assert detected == manifest


if __name__ == "__main__":
    test_registration()
    test_scale_constants_are_powers_of_two()
    test_hardware_policy_uses_capability_not_product_names()
    test_native_bf16_policy_skips_fp16_wrappers()
    test_compatibility_policy_keeps_fp16_exact_loader()
    test_model_selector_hides_individual_fasth3_shards()
    test_fasth3_dense_load_records_sampling_metadata_without_scheduler_patch()
    test_non_h3_model_fails_clearly()
    test_fasth3_converted_shapes_detect_native_comfy_h3_config()
    test_dense_model_patch_is_scoped_and_complete()
    test_quantized_model_preserves_native_dispatch()
    test_model_wrappers_are_weakly_bound_and_idempotent()
    test_quantization_summary_reports_convrot()
    test_native_loader_builds_fp16_operations_before_model_creation()
    test_model_detection_strips_diffusion_prefix()
    test_normalize_keeps_embedded_quantization_configs()
    test_normalize_restores_unprefixed_legacy_quants_after_prefix_removal()
    test_single_file_vsa_manifest_is_detected()
    print("MiniMax H3 Enhanced Loader - Star7 node tests passed")
