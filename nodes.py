import json
import logging
import math
import os
import time
import weakref
from collections import Counter
from types import MethodType

import folder_paths
import torch
import torch.nn.functional as F

import comfy.model_detection
import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.supported_models
import comfy.utils
from safetensors import safe_open

from .fasth3_loader import (
    SAMPLING_PROFILE,
    detect_fasth3_checkpoint,
    load_fasth3_shards,
    scan_fasth3_directories,
)
from . import fasth3_modulation
from . import fasth3_vsa


NODE_VERSION = "1.2.5"
PATCH_FLAG = "star7_minimax_h3_fp16_exact_fix"
PATCH_MODE = "star7_minimax_h3_fp16_mode"
FAST_H3_PATCH_FLAG = "star7_fasth3_mlp_layout_v2"
FAST_H3_VSA_PATCH_FLAG = "star7_fasth3_vsa_v1"
K_OUT_PROJ = 64.0
K_FC2 = 256.0
_AUTO_DETECT = object()
FAST_H3_FUSED_INT8_FC2_ENV = "STAR7_FASTH3_FUSED_INT8_FC2"
FAST_H3_FUSED_MODULATION_ENV = "STAR7_FASTH3_FUSED_MODULATION"
_ROPE_DISPATCH_RESTORED = False


def _restore_upstream_rope_dispatch():
    """Remove a stale global Chunk RoPE dispatcher before loading a model.

    The Chunk node installs its dispatcher when that node executes.  Older
    ComfyUI sessions can retain that process-wide function even after the node
    is bypassed.  The enhanced loader must remain standalone; a Chunk node
    connected downstream will execute later and install its own dispatcher
    again for that path.
    """
    global _ROPE_DISPATCH_RESTORED
    try:
        import comfy.quant_ops as quant_ops

        ck = quant_ops.ck
        current = ck.rms_rope_split_half_
    except (AttributeError, ImportError):
        return False

    if getattr(current, "__name__", "") != "_chunked_rms_rope_split_half_inplace":
        return False
    module_name = getattr(current, "__module__", "")
    if "minimax-h3-chunk-star7" not in module_name and "chunk_star7" not in module_name:
        return False

    original = getattr(current, "_star7_original", None)
    if original is None:
        original = getattr(current, "__globals__", {}).get(
            "_ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE"
        )
    if not callable(original) or original is current:
        return False

    ck.rms_rope_split_half_ = original
    if not _ROPE_DISPATCH_RESTORED:
        logging.info(
            "[Star7 H3 Enhanced] Cleared stale global Chunk RoPE dispatcher; "
            "standalone model path restored"
        )
        _ROPE_DISPATCH_RESTORED = True
    return True


def _weak_callable(value):
    """Keep a bound model method callable without retaining its owner."""
    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", value)
    if owner is None or isinstance(owner, weakref.ProxyTypes):
        return value

    owner_ref = weakref.ref(owner)

    def call(*args, **kwargs):
        current = owner_ref()
        if current is None:
            raise ReferenceError("Star7 FP16 wrapper owner was released")
        return function(current, *args, **kwargs)

    return call


def _weak_method(owner, function):
    """Bind a patch function through a weak proxy instead of the model module."""
    return MethodType(function, weakref.proxy(owner))


def _condition_proj_forward(original_forward):
    def forward(self, tensor):
        return original_forward(tensor.to(torch.float32))

    return forward


def _out_proj_forward(original_forward):
    def forward(self, tensor):
        scaled = (tensor / K_OUT_PROJ).to(torch.float16)
        return original_forward(scaled).to(torch.float32).mul_(K_OUT_PROJ)

    return forward


def _fast_h3_fused_int8_fc2_enabled():
    value = os.environ.get(FAST_H3_FUSED_INT8_FC2_ENV, "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _fast_h3_fused_modulation_enabled():
    value = os.environ.get(FAST_H3_FUSED_MODULATION_ENV, "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _fast_h3_fused_int8_fc2(self, projected):
    """Run value-first FastH3 SwiGLU inside the native INT8 down projection."""
    if not _fast_h3_fused_int8_fc2_enabled():
        return None
    weight = self.fc2.weight
    if (
        not isinstance(weight, comfy.ops.QuantizedTensor)
        or weight._layout_cls != "TensorWiseINT8Layout"
        or getattr(weight._params, "transposed", False)
        or projected.dtype not in (torch.float16, torch.bfloat16)
    ):
        return None

    value, gate = projected.chunk(2, dim=-1)
    # Comfy Kitchen expects [gate, value]. This FP16/BF16 reorder is much
    # cheaper than materializing FastH3's complete SwiGLU activation in FP32.
    native_order = torch.cat((gate, value), dim=-1)
    prepared, bias, offload_stream = comfy.ops.cast_bias_weight(
        self.fc2,
        native_order,
        offloadable=True,
        compute_dtype=native_order.dtype,
        want_requant=True,
    )
    try:
        if not isinstance(prepared, comfy.ops.QuantizedTensor):
            return None
        get_plain_tensors = getattr(
            comfy.ops.TensorWiseINT8Layout, "get_plain_tensors", None
        )
        if not callable(get_plain_tensors):
            # Older/incomplete Comfy Kitchen environments expose the layout
            # name but not its public unpacking API. Preserve correctness by
            # returning to the native MLP path instead of failing the model.
            return None
        qdata, scale = get_plain_tensors(prepared)
        return comfy.ops.quant_ops.ck.int8_linear(
            native_order,
            qdata,
            scale,
            bias,
            torch.float32,
            convrot=getattr(prepared._params, "convrot", False),
            convrot_groupsize=getattr(prepared._params, "convrot_groupsize", 256),
            input_act="swiglu",
        )
    finally:
        comfy.ops.uncast_bias_weight(self.fc2, prepared, bias, offload_stream)


def _mlp_forward(original_forward, *, fast_h3=False):
    def forward(self, tensor):
        if not fast_h3 and tensor.dtype != torch.float16:
            return original_forward(tensor)

        projected = self.fc1(tensor)
        if fast_h3:
            fused = _fast_h3_fused_int8_fc2(self, projected)
            if fused is not None:
                return fused
            # FastVideo stores SwiGLU as [value, gate], unlike ComfyUI's
            # native H3 [gate, value] convention.
            value, gate = projected.chunk(2, dim=-1)
        else:
            gate, value = projected.chunk(2, dim=-1)

        if tensor.dtype == torch.float16:
            activated = F.silu(gate.to(torch.float32)).mul_(value.to(torch.float32))
            scaled = (activated / K_FC2).to(torch.float16)
            return self.fc2(scaled).to(torch.float32).mul_(K_FC2)

        return self.fc2(F.silu(gate).mul_(value))

    return forward


def _block_forward(original_forward, minimax_module, *, fast_h3=False):
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)

        fuse_modulation = (
            fast_h3
            and _fast_h3_fused_modulation_enabled()
            and fasth3_modulation.can_fuse(x, mod_segments)
        )
        if fuse_modulation:
            with comfy.ops.CastBiasWeightContext(
                self.norm1, x, offloadable=True
            ) as (weight, _):
                h = fasth3_modulation.rmsnorm_modulate(
                    x,
                    weight,
                    scale_msa,
                    shift_msa,
                    mod_segments,
                    self.norm1.eps,
                ).to(torch.float16)
        else:
            h = minimax_module._mod_scale_shift(
                self.norm1(x), shift_msa, scale_msa, mod_segments
            ).to(torch.float16)
        attention = self.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        attention = attention.to(torch.float32)
        if fuse_modulation:
            with comfy.ops.CastBiasWeightContext(
                self.norm2, x, offloadable=True
            ) as (weight, _):
                x, h = fasth3_modulation.residual_gate_rmsnorm_modulate(
                    x,
                    attention,
                    gate_msa,
                    weight,
                    scale_mlp,
                    shift_mlp,
                    mod_segments,
                    self.norm2.eps,
                )
            h = h.to(torch.float16)
        else:
            x = minimax_module._mod_gate(x, gate_msa, attention, mod_segments)
            h = minimax_module._mod_scale_shift(
                self.norm2(x), shift_mlp, scale_mlp, mod_segments
            ).to(torch.float16)
        mlp = self.mlp(h)
        return minimax_module._mod_gate(
            x, gate_mlp, mlp.to(torch.float32), mod_segments
        )

    return forward


def _quantization_summary(diffusion_model):
    formats = Counter()
    for module in diffusion_model.modules():
        quant_format = getattr(module, "quant_format", None)
        layout_type = getattr(module, "layout_type", None)
        if quant_format is None or layout_type is None:
            continue

        weight = getattr(module, "weight", None)
        params = getattr(weight, "_params", None)
        label = quant_format
        if getattr(params, "convrot", False):
            label += "+convrot"
        formats[label] += 1
    return formats


def _format_quantization(formats):
    return ",".join(f"{name}:{count}" for name, count in sorted(formats.items()))


def detect_hardware_policy(
    *, cuda_available=None, hip=_AUTO_DETECT, capability=None, force_fp16=None
):
    """Select precision by capability features, never by product name."""
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if hip is _AUTO_DETECT:
        hip = torch.version.hip
    if not cuda_available:
        return {
            "name": "native_default",
            "apply_fp16_exact": False,
            "reason": "CUDA is unavailable",
        }
    if hip is not None:
        return {
            "name": "fp16_compat",
            "apply_fp16_exact": True,
            "reason": "ROCm compatibility path",
        }
    if capability is None:
        capability = torch.cuda.get_device_capability()
    capability = tuple(capability)
    label = f"sm{capability[0]}{capability[1]}"
    if capability[0] >= 8:
        if force_fp16 is None:
            force_fp16 = bool(
                getattr(comfy.model_management.args, "fp16_unet", False)
            )
        return {
            "name": "native_bf16",
            "apply_fp16_exact": False,
            "launcher_fp16_overridden": bool(force_fp16),
            "reason": (
                f"{label} launcher FP16 override corrected to native BF16 for H3"
                if force_fp16
                else f"{label} supports native BF16"
            ),
        }
    if capability == (6, 1):
        return {
            "name": "native_default",
            "apply_fp16_exact": False,
            "reason": "sm61 has very slow FP16 throughput",
        }
    return {
        "name": "fp16_compat",
        "apply_fp16_exact": True,
        "reason": label,
    }


def _native_model_options(policy):
    """Pin BF16-capable H3 hardware to BF16 despite global FP16 flags."""
    if policy.get("name") == "native_bf16":
        return {"dtype": torch.bfloat16}
    return {}


def _patch_h3_model(
    model, loader_native=False, *, fast_h3=False, apply_fp16_exact=True
):
    import comfy.ldm.minimax.model as minimax_module

    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_module.MiniMaxH3Model):
        raise TypeError("Connected model is not native ComfyUI MiniMax H3")

    transformer_options = patched.model_options.setdefault("transformer_options", {})
    if not fast_h3 and transformer_options.get(PATCH_FLAG):
        logging.info("[Star7 H3 FP16] Patch is already present; skipping duplicate.")
        return patched
    if fast_h3 and transformer_options.get(FAST_H3_PATCH_FLAG):
        logging.info("[Star7 H3 Enhanced] FastH3 MLP layout patch is already present; skipping duplicate.")
        return patched

    quant_formats = _quantization_summary(diffusion_model)
    is_quantized = bool(quant_formats)

    if apply_fp16_exact:
        patched.set_model_compute_dtype(torch.float16)
    if is_quantized or loader_native:
        # Keep the UUID update from set_model_compute_dtype without forcing
        # MixedPrecisionOps to dequantize its weights.
        patched.force_cast_weights = False

    if apply_fp16_exact and getattr(
        minimax_module.MiniMaxH3Model,
        "_star7_h3_global_fp16_patch",
        False,
    ) and getattr(diffusion_model.blocks[0], "_star7_h3_fp16_fix", False):
        mode = "loader-native" if loader_native else "postload"
        transformer_options[PATCH_FLAG] = NODE_VERSION
        transformer_options[PATCH_MODE] = mode
        logging.info("[Star7 H3 FP16] Global overflow fix already active | mode=%s", mode)
        return patched

    condition_proj = diffusion_model.condition_proj
    patched.add_object_patch(
        "diffusion_model.condition_proj.forward",
        _weak_method(
            condition_proj,
            _condition_proj_forward(_weak_callable(condition_proj.forward)),
        ),
    )

    for index, block in enumerate(diffusion_model.blocks):
        out_proj = block.attn.out_proj
        if apply_fp16_exact:
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.out_proj.forward",
                _weak_method(out_proj, _out_proj_forward(_weak_callable(out_proj.forward))),
            )
        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.mlp.forward",
            _weak_method(
                block.mlp,
                _mlp_forward(_weak_callable(block.mlp.forward), fast_h3=fast_h3),
            ),
        )
        if apply_fp16_exact:
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.forward",
                _weak_method(
                    block,
                    _block_forward(
                        _weak_callable(block.forward),
                        minimax_module,
                        fast_h3=fast_h3,
                    ),
                ),
            )

    if fast_h3:
        for index, block in enumerate(diffusion_model.token_refiner.blocks):
            patched.add_object_patch(
                f"diffusion_model.token_refiner.blocks.{index}.mlp.forward",
                _weak_method(
                    block.mlp,
                    _mlp_forward(_weak_callable(block.mlp.forward), fast_h3=True),
                ),
            )

    if loader_native:
        mode = "loader-quantized" if is_quantized else "loader-dense"
    else:
        mode = "postload-quantized" if is_quantized else "postload-dense"

    if fast_h3:
        transformer_options[FAST_H3_PATCH_FLAG] = NODE_VERSION
    if apply_fp16_exact:
        transformer_options[PATCH_FLAG] = NODE_VERSION
    transformer_options[PATCH_MODE] = mode

    weight_patches = len(getattr(patched, "patches", {}))
    backend = _format_quantization(quant_formats) if is_quantized else "dense-fp16"
    logging.info(
        "[Star7 H3 FP16] Enabled v%s | mode=%s | backend=%s | force-cast=%s | weight-patches=%d | blocks=%d",
        NODE_VERSION,
        mode,
        backend,
        bool(patched.force_cast_weights),
        weight_patches,
        len(diffusion_model.blocks),
    )
    if fast_h3:
        logging.info(
            "[Star7 FastH3] fused INT8 SwiGLU+FC2=%s | fused modulation=%s",
            "enabled" if _fast_h3_fused_int8_fc2_enabled() else "disabled",
            "enabled" if _fast_h3_fused_modulation_enabled() else "disabled",
        )
    if is_quantized and weight_patches:
        logging.warning(
            "[Star7 H3 FP16] Weight patches detected; dynamic low-VRAM LoRA may dequantize affected layers."
        )
    return patched


def _detect_h3_config(state_dict, metadata):
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    detection_state_dict = state_dict
    if prefix:
        stripped = comfy.utils.state_dict_prefix_replace(
            state_dict, {prefix: ""}, filter_keys=True
        )
        if stripped:
            detection_state_dict = stripped
    model_config = comfy.model_detection.model_config_from_unet(
        detection_state_dict, "", metadata=metadata
    )
    if not isinstance(model_config, comfy.supported_models.MiniMaxH3):
        raise ValueError("Selected file is not a native ComfyUI MiniMax H3 diffusion model")
    return model_config


def _normalize_h3_state_dict(state_dict, metadata):
    """Match ComfyUI's quant conversion around checkpoint prefix removal."""
    state_dict, metadata = comfy.utils.convert_old_quants(
        state_dict, "", metadata=metadata
    )
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    if prefix:
        stripped = comfy.utils.state_dict_prefix_replace(
            state_dict, {prefix: ""}, filter_keys=True
        )
        if stripped:
            state_dict = stripped
            if comfy.utils.detect_layer_quantization(state_dict, "") is None:
                state_dict, metadata = comfy.utils.convert_old_quants(
                    state_dict, "", metadata=metadata
                )
    return state_dict, metadata


def _single_file_fasth3_manifest(unet_path):
    if not str(unet_path).lower().endswith((".safetensors", ".sft")):
        return None
    try:
        with safe_open(unet_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        raw = metadata.get("star7_fasth3_manifest")
        manifest = json.loads(raw) if raw else {}
        if metadata.get("star7_variant"):
            manifest.setdefault("variant", metadata["star7_variant"])
        if metadata.get("star7_sampling_profile"):
            manifest.setdefault("sampling_profile", metadata["star7_sampling_profile"])
        if metadata.get("star7_quantization"):
            manifest.setdefault("quantization", metadata["star7_quantization"])
        return manifest if manifest.get("variant") in {
            "fasth3_dense_v1",
            "fasth3_vsa_datafree_v1",
        } else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_h3_native_fp16(unet_path, fast_h3=False):
    state_dict, metadata = comfy.utils.load_torch_file(
        unet_path, return_metadata=True
    )
    state_dict, metadata = _normalize_h3_state_dict(state_dict, metadata)
    model_config = _detect_h3_config(state_dict, metadata)
    load_device = comfy.model_management.get_torch_device()
    operations = comfy.ops.pick_operations(
        torch.float16,
        torch.float16,
        load_device=load_device,
        model_config=model_config,
    )
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options={
            "dtype": torch.float16,
            "custom_operations": operations,
        },
        metadata=metadata,
        disable_dynamic=False,
    )
    if model is None:
        raise RuntimeError("ComfyUI could not load the selected MiniMax H3 model")

    patched = _patch_h3_model(
        model,
        loader_native=True,
        fast_h3=fast_h3,
        apply_fp16_exact=True,
    )
    patched.cached_patcher_init = (
        _load_h3_native_fp16,
        (unet_path, fast_h3),
    )
    return patched


def _model_source_choices():
    directories = scan_fasth3_directories(
        folder_paths.get_folder_paths("diffusion_models")
    )
    files = []
    for name in folder_paths.get_filename_list("diffusion_models"):
        portable = str(name).replace("\\", "/")
        if any(portable.startswith(directory) for directory in directories):
            continue
        files.append(name)
    return sorted(set(files + directories), key=str.casefold)


def _split_vsa_gate_state(state_dict):
    gates = []
    for index in range(50):
        prefix = f"blocks.{index}.attn.gate_compress."
        gate = {}
        for name in ("weight", "weight_scale", "comfy_quant"):
            key = prefix + name
            if key not in state_dict:
                raise ValueError(f"FastH3 VSA gate tensor is missing: {key}")
            gate[name] = state_dict.pop(key)
        gates.append(gate)
    leftovers = [key for key in state_dict if ".gate_compress." in key]
    if leftovers:
        raise ValueError(f"Unexpected FastH3 VSA gate tensors: {leftovers[:3]}")
    return gates


def _attach_vsa_gates(model, gate_states):
    diffusion_model = _loaded_h3_object(model)
    if len(gate_states) != len(diffusion_model.blocks):
        raise ValueError(
            f"FastH3 VSA gate count {len(gate_states)} does not match "
            f"H3 blocks {len(diffusion_model.blocks)}"
        )
    for index, (block, state) in enumerate(zip(diffusion_model.blocks, gate_states)):
        template = block.attn.qkv_proj
        factory = getattr(template, "factory_kwargs", {})
        gate = type(template)(
            diffusion_model.hidden_size,
            block.attn.heads * block.attn.head_dim,
            bias=False,
            dtype=factory.get("dtype", torch.float16),
            device=factory.get("device", model.offload_device),
        )
        missing, unexpected = gate.load_state_dict(dict(state), strict=True)
        if missing or unexpected:
            raise ValueError(
                f"FastH3 VSA gate {index} load mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        block.attn.gate_compress = gate
    model.size = 0


def _vsa_block_segments(original_forward):
    original_forward = _weak_callable(original_forward)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        old_segments = getattr(self.attn, "_star7_sla_mod_segments", None)
        self.attn._star7_sla_mod_segments = mod_segments
        try:
            return original_forward(
                x, t_emb, mod_segments, rope_freqs,
                transformer_options=transformer_options,
            )
        finally:
            if old_segments is None:
                delattr(self.attn, "_star7_sla_mod_segments")
            else:
                self.attn._star7_sla_mod_segments = old_segments

    # The chunk node deliberately recognizes and replaces this wrapper while
    # preserving the VSA attention implementation.
    forward._star7_wrapper_kind = "sla-segment-block"
    forward._star7_original_forward = original_forward
    return forward


def _vsa_sampling_step(transformer_options):
    if not isinstance(transformer_options, dict):
        return None
    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigma = transformer_options.get("sigmas")
    if not torch.is_tensor(sample_sigmas) or not torch.is_tensor(current_sigma):
        return None
    schedule = sample_sigmas.detach().flatten()
    current = current_sigma.detach().flatten()
    if schedule.numel() < 2 or current.numel() < 1:
        return None
    value = current[0].to(schedule.device, dtype=schedule.dtype)
    matches = torch.isclose(schedule[:-1], value, rtol=1e-4, atol=1e-6)
    indices = torch.nonzero(matches, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    return int(indices[0].item()), int(schedule.numel() - 1)


def _vsa_model_context(original_forward):
    original_forward = _weak_callable(original_forward)

    def forward(
        self, x, timestep, context, transformer_options={}, minimax_payload=None,
        denoise_mask=None, audio_denoise_mask=None, **kwargs,
    ):
        video = x[0]
        pt, ph, pw = self.patch_size
        shape = (
            math.ceil(video.shape[2] / pt),
            math.ceil(video.shape[3] / ph),
            math.ceil(video.shape[4] / pw),
        )
        marker = "_star7_vsa_video_shape"
        previous = transformer_options.get(marker)
        transformer_options[marker] = shape
        step = _vsa_sampling_step(transformer_options)
        timing_device = x[0].device if isinstance(x, (list, tuple)) and x and torch.is_tensor(x[0]) else None
        start_event = None
        start_time = None
        if step is not None and timing_device is not None:
            if timing_device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(timing_device))
            else:
                start_time = time.perf_counter()
        try:
            result = original_forward(
                x, timestep, context,
                transformer_options=transformer_options,
                minimax_payload=minimax_payload,
                denoise_mask=denoise_mask,
                audio_denoise_mask=audio_denoise_mask,
                **kwargs,
            )
        finally:
            if step is not None and timing_device is not None:
                if start_event is not None:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record(torch.cuda.current_stream(timing_device))
                    end_event.synchronize()
                    elapsed = start_event.elapsed_time(end_event) / 1000.0
                elif start_time is not None:
                    elapsed = time.perf_counter() - start_time
                else:
                    elapsed = None
                if elapsed is not None:
                    logging.info(
                        "[Star7 FastH3 VSA] step %d/%d | %7.2fs/it | "
                        "backend=native-vsa | blocks=50 | tiles=64",
                        step[0] + 1, step[1], elapsed,
                    )
            if previous is None:
                transformer_options.pop(marker, None)
            else:
                transformer_options[marker] = previous
        return result

    return forward


def _vsa_attention_forward(self, x, rope_freqs=None, transformer_options={}):
    if isinstance(x, list):
        x = x.pop()
    sequence = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    gate = self.gate_compress(x)
    v = v.view(sequence, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, sequence, self.heads, self.head_dim)
        k = k.view(1, sequence, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw,
                epsilon=self.q_norm.eps, rot_dim=rot,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw,
                epsilon=self.q_norm.eps, rot_dim=rot,
            )
    else:
        q = self.q_norm(q.view(sequence, self.heads, self.head_dim)).unsqueeze(0)
        k = self.k_norm(k.view(sequence, self.heads, self.head_dim)).unsqueeze(0)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(0, 1).unsqueeze(0).contiguous()
    gate = gate.view(sequence, self.heads, self.head_dim)
    gate = gate.transpose(0, 1).unsqueeze(0).to(dtype=q.dtype).contiguous()

    segments = getattr(self, "_star7_sla_mod_segments", ())
    if len(segments) < 2:
        raise ValueError("FastH3 VSA did not receive the H3 packed segment table")
    prefix_segments = tuple(int(end) - int(start) for start, end, _ in segments[:-1])
    video_shape = transformer_options.get("_star7_vsa_video_shape")
    if video_shape is None:
        raise ValueError("FastH3 VSA did not receive the target video token shape")
    output = fasth3_vsa.sparse_attention_consume(
        [q, k, v, gate],
        prefix_segments=prefix_segments,
        video_shape=tuple(video_shape),
        sparsity=0.9,
        profile=(
            os.environ.get("STAR7_VSA_PROFILE", "0").strip().lower()
            in {"1", "true", "yes", "on"}
            and getattr(self, "_star7_vsa_layer_index", -1) == 0
        ),
    )
    output = output.transpose(1, 2).reshape(
        1, sequence, self.heads * self.head_dim
    ).squeeze(0)
    return self.out_proj(output.to(dtype=x.dtype))


_vsa_attention_forward._star7_consumes_input = True


def _enable_vsa_runtime(model):
    diffusion_model = _loaded_h3_object(model)
    options = model.model_options.setdefault("transformer_options", {})
    if options.get(FAST_H3_VSA_PATCH_FLAG):
        return model
    if not all(hasattr(block.attn, "gate_compress") for block in diffusion_model.blocks):
        raise ValueError("FastH3 VSA model is missing one or more compression gates")

    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
        except (RuntimeError, AssertionError):
            capability = None
        if capability == (7, 5):
            native, reason = fasth3_vsa.native_sm75_status()
            if not native:
                raise RuntimeError(
                    "FastH3 VSA requires the native SM75 launcher on this GPU, "
                    f"but it is unavailable: {reason}"
                )
            logging.info(
                "[Star7 FastH3 VSA] %s is ready; gate-compressed pooled "
                "branch remains FP16",
                reason,
            )
        elif capability is not None and capability < (8, 0):
            logging.warning(
                "[Star7 FastH3 VSA] SM%s%s uses the compatibility Triton tile-64 "
                "path; the official measured VSA CUDA kernel targets SM100a.",
                capability[0], capability[1],
            )

    for index, block in enumerate(diffusion_model.blocks):
        block.attn._star7_vsa_layer_index = index
        model.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            _weak_method(block.attn, _vsa_attention_forward),
        )
        block_path = f"diffusion_model.blocks.{index}.forward"
        upstream = model.object_patches.get(block_path, block.forward)
        model.add_object_patch(
            block_path,
            _weak_method(block, _vsa_block_segments(upstream)),
        )

    forward_path = "diffusion_model._forward"
    upstream_forward = model.object_patches.get(
        forward_path, diffusion_model._forward
    )
    model.add_object_patch(
        forward_path,
        _weak_method(diffusion_model, _vsa_model_context(upstream_forward)),
    )
    options[FAST_H3_VSA_PATCH_FLAG] = NODE_VERSION
    options["star7_h3_vsa_runner"] = fasth3_vsa.sparse_attention_consume
    logging.info(
        "[Star7 FastH3 VSA] Enabled official tile-64 semantics | "
        "sparsity=0.90 | gates=50 | kernel=architecture-specific VSA"
    )
    return model


def _resolve_directory_source(name):
    relative = str(name).replace("\\", "/").rstrip("/")
    if (
        not relative
        or relative == ".."
        or relative.startswith("../")
        or "/../" in relative
    ):
        raise ValueError(f"Invalid model directory selection: {name!r}")
    for root in folder_paths.get_folder_paths("diffusion_models"):
        root_path = os.path.realpath(root)
        candidate = os.path.realpath(os.path.join(root_path, relative))
        try:
            inside = os.path.normcase(
                os.path.commonpath((root_path, candidate))
            ) == os.path.normcase(root_path)
        except ValueError:
            inside = False
        if inside and os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"FastH3 model directory was not found: {name}")


def _loaded_h3_object(model):
    import comfy.ldm.minimax.model as minimax_module

    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_module.MiniMaxH3Model):
        raise TypeError(
            "Selected source did not create a native ComfyUI MiniMaxH3Model"
        )
    return diffusion_model


def detect_model_variant(model):
    diffusion_model = _loaded_h3_object(model)
    quant_formats = _quantization_summary(diffusion_model)
    return (
        "base_h3_quantized" if quant_formats else "base_h3",
        _format_quantization(quant_formats) if quant_formats else "dense",
    )


def _set_enhanced_metadata(
    model, *, variant, source, policy, quantization, requires_vsa=False,
    sampling_profile="base_h3",
):
    options = model.model_options.setdefault("transformer_options", {})
    options["star7_h3"] = {
        "loader": "enhanced",
        "version": NODE_VERSION,
        "variant": variant,
        "source": source,
        "sampling_profile": sampling_profile,
        "requires_vsa": bool(requires_vsa),
        "precision_policy": policy["name"],
        "quantization": quantization,
    }
    logging.info(
        "[Star7 H3 Enhanced] v%s | variant=%s | source=%s | attention=%s "
        "| sampling=%s | precision=%s | quantization=%s",
        NODE_VERSION,
        variant,
        source,
        (
            "vsa-h3-tile64"
            if requires_vsa
            else "dense" if variant == "fasth3_dense_v1" else "model-selected"
        ),
        sampling_profile,
        policy["name"],
        quantization,
    )
    if sampling_profile == SAMPLING_PROFILE or str(sampling_profile).startswith(
        "fasth3_4step"
    ):
        logging.warning(
            "[Star7 H3 Enhanced] FastH3 4-step sampling profile required: "
            "four Transformer forwards, timesteps 999/749/500/250, guidance 1.0. "
            "The loader records this contract but does not patch the scheduler."
        )
    return model


def _apply_fasth3_vsa_switch(model, vsa_acceleration=True):
    """Validate and enable the native VSA runtime on a FastH3 VSA model."""
    options = model.model_options.get("transformer_options", {})
    metadata = options.get("star7_h3", {})
    variant = str(metadata.get("variant", ""))
    profile = str(metadata.get("sampling_profile", ""))
    if not variant.startswith("fasth3_") or not profile.startswith("fasth3_4step"):
        raise ValueError(
            "MiniMax H3 FastH3 4-Step Preset requires a FastH3 model loaded by "
            "MiniMax H3 Enhanced Loader - Star7"
        )
    requires_vsa = bool(metadata.get("requires_vsa", False))
    if bool(vsa_acceleration) and not requires_vsa:
        raise ValueError(
            "VSA acceleration is enabled, but the connected model is not a "
            "VSA FastH3 checkpoint"
        )

    patched = model.clone()
    if bool(vsa_acceleration):
        patched = _enable_vsa_runtime(patched)
    transformer_options = dict(
        patched.model_options.get("transformer_options", {})
    )
    patched.model_options["transformer_options"] = transformer_options
    transformer_options["star7_h3_vsa_switch"] = {
        "variant": variant,
        "vsa_acceleration": bool(vsa_acceleration),
        "attention_backend": "video_sparse_attn_h3" if vsa_acceleration else "disabled",
    }
    return patched


class MiniMaxH3VSASwitchStar7:
    """Explicitly enable the native FastH3 VSA runtime."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vsa_acceleration": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Require and mark the connected FastH3 VSA model. The loader supplies tile-64 sparse attention; the chunk node must preserve existing attention.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "FastH3 VSA switch. Connect the VSA FastH3 model here and enable the "
        "switch to validate and install the tile-64 VSA runtime. Flow shift and "
        "sampling steps remain controlled by the original H3 nodes."
    )

    def apply(self, model, vsa_acceleration=True):
        return (_apply_fasth3_vsa_switch(model, vsa_acceleration),)


def _load_fasth3_directory(directory):
    info = detect_fasth3_checkpoint(directory)
    state_dict, report = load_fasth3_shards(info)
    gate_states = _split_vsa_gate_state(state_dict) if info.requires_vsa else None
    model_config = _detect_h3_config(state_dict, {})
    policy = detect_hardware_policy()
    model_options = _native_model_options(policy)
    if policy["apply_fp16_exact"]:
        load_device = comfy.model_management.get_torch_device()
        operations = comfy.ops.pick_operations(
            torch.float16,
            torch.float16,
            load_device=load_device,
            model_config=model_config,
        )
        model_options = {
            "dtype": torch.float16,
            "custom_operations": operations,
        }
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options=model_options,
        metadata={},
        disable_dynamic=False,
    )
    if model is None:
        raise RuntimeError("ComfyUI could not create MiniMax H3 from FastH3 Dense")
    model = _patch_h3_model(
        model,
        loader_native=True,
        # The official VSA adapter is merged into the native full H3 base and
        # therefore keeps native H3's MLP ordering.  Only HF Dense FastH3 uses
        # the value-first MLP compatibility path.
        fast_h3=not info.requires_vsa,
        apply_fp16_exact=policy["apply_fp16_exact"],
    )
    if info.requires_vsa:
        _attach_vsa_gates(model, gate_states)
        model = _enable_vsa_runtime(model)
    model.cached_patcher_init = (
        _load_fasth3_directory,
        (str(info.root),),
    )
    logging.info(
        "[Star7 H3 Enhanced] FastH3 conversion valid | source-keys=%d "
        "| converted-keys=%d | missing=0 | unexpected=0",
        report.source_keys,
        report.converted_keys,
    )
    return _set_enhanced_metadata(
        model,
        variant=info.variant,
        source="native-vsa-directory" if info.requires_vsa else "hf-directory",
        policy=policy,
        quantization=getattr(info, "quantization", "bf16-dense"),
        requires_vsa=info.requires_vsa,
        sampling_profile=info.sampling_profile,
    )


def _load_fasth3_vsa_single_file(unet_path, manifest):
    state_dict, metadata = comfy.utils.load_torch_file(
        unet_path, return_metadata=True
    )
    state_dict, metadata = _normalize_h3_state_dict(state_dict, metadata)
    gate_states = _split_vsa_gate_state(state_dict)
    model_config = _detect_h3_config(state_dict, metadata)
    policy = detect_hardware_policy()
    model_options = _native_model_options(policy)
    if policy["apply_fp16_exact"]:
        load_device = comfy.model_management.get_torch_device()
        operations = comfy.ops.pick_operations(
            torch.float16,
            torch.float16,
            load_device=load_device,
            model_config=model_config,
        )
        model_options = {
            "dtype": torch.float16,
            "custom_operations": operations,
        }
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options=model_options,
        metadata=metadata,
        disable_dynamic=False,
    )
    if model is None:
        raise RuntimeError("ComfyUI could not create MiniMax H3 from FastH3 VSA")
    model = _patch_h3_model(
        model,
        loader_native=True,
        fast_h3=False,
        apply_fp16_exact=policy["apply_fp16_exact"],
    )
    _attach_vsa_gates(model, gate_states)
    model = _enable_vsa_runtime(model)
    model.cached_patcher_init = (
        _load_fasth3_vsa_single_file,
        (unet_path, manifest),
    )
    return _set_enhanced_metadata(
        model,
        variant="fasth3_vsa_datafree_v1",
        source="native-vsa-single-file",
        policy=policy,
        quantization=manifest.get("quantization", "int8_tensorwise_convrot"),
        requires_vsa=True,
        sampling_profile=manifest.get("sampling_profile", SAMPLING_PROFILE),
    )


def load_model_with_policy(source_name):
    _restore_upstream_rope_dispatch()
    policy = detect_hardware_policy()
    if str(source_name).replace("\\", "/").endswith("/"):
        return _load_fasth3_directory(_resolve_directory_source(source_name))

    path = folder_paths.get_full_path_or_raise("diffusion_models", source_name)
    fast_h3_manifest = _single_file_fasth3_manifest(path)
    if (
        fast_h3_manifest is not None
        and fast_h3_manifest.get("variant") == "fasth3_vsa_datafree_v1"
    ):
        return _load_fasth3_vsa_single_file(path, fast_h3_manifest)
    is_fast_h3 = fast_h3_manifest is not None
    if policy["apply_fp16_exact"]:
        logging.info(
            "[Star7 H3 Enhanced] Loading native H3 with FP16 compatibility | %s",
            policy["reason"],
        )
        model = _load_h3_native_fp16(path, fast_h3=is_fast_h3)
    else:
        native_options = _native_model_options(policy)
        logging.info(
            "[Star7 H3 Enhanced] Loading native H3 with %s "
            "| %s | no FP16 block wrappers",
            "explicit BF16 compute" if native_options else "ComfyUI default precision",
            policy["reason"],
        )
        model = comfy.sd.load_diffusion_model(
            path,
            model_options=native_options,
            disable_dynamic=False,
        )
        if is_fast_h3:
            model = _patch_h3_model(
                model,
                loader_native=True,
                fast_h3=True,
                apply_fp16_exact=False,
            )
        else:
            model = model.clone()
            _loaded_h3_object(model)
    if is_fast_h3:
        variant = fast_h3_manifest["variant"]
        quantization = fast_h3_manifest.get("quantization", "int8_tensorwise_convrot")
        sampling_profile = fast_h3_manifest.get("sampling_profile", SAMPLING_PROFILE)
    else:
        variant, quantization = detect_model_variant(model)
        sampling_profile = "base_h3"
    return _set_enhanced_metadata(
        model,
        variant=variant,
        source="single-file",
        policy=policy,
        quantization=quantization,
        sampling_profile=sampling_profile,
    )


class MiniMaxH3EnhancedLoaderStar7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    _model_source_choices(),
                    {
                        "tooltip": "Select a native H3 file or a slash-suffixed FastH3 checkpoint directory."
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Star7/MiniMax H3"
    SEARCH_ALIASES = [
        "MiniMax H3 自适应优化载入",
        "MiniMax H3 增强载入",
        "MiniMax H3 优化载入",
        "MiniMax H3 智能增强载入",
        "H3 模型载入",
        "H3 模型加载",
        "FastH3 载入",
        "FastH3 加载",
        "enhanced loader",
        "optimized loader",
        "model loader",
    ]
    DESCRIPTION = (
        "Architecture-aware MiniMax H3 loader for official, third-party, "
        "quantized, FastH3 Dense, and FastH3 VSA checkpoints. Preserves native "
        "quantized dispatch, applies FP16 protection only where required, and "
        "records the effective model and sampling contract."
    )

    def load_model(self, unet_name):
        return (load_model_with_policy(unet_name),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3EnhancedLoaderStar7": MiniMaxH3EnhancedLoaderStar7,
    "MiniMaxH3VSASwitchStar7": MiniMaxH3VSASwitchStar7,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3EnhancedLoaderStar7": "MiniMax H3 Enhanced Loader - Star7",
    "MiniMaxH3VSASwitchStar7": "MiniMax H3 FastH3 VSA Switch - Star7",
}
