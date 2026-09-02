# MiniMax H3 Enhanced Loader - Star7

[中文](README.md) · [Example workflows](examples/workflows) · [Changelog](CHANGELOG.md)

A unified model loader and runtime adaptation layer for MiniMax H3 in ComfyUI. It detects model structure, quantization metadata, and GPU architecture, then selects the appropriate execution policy while preserving the checkpoint's native weights and quantized operators. It supports official H3, third-party H3 variants, hybrid/quantized checkpoints, FastH3 Dense, and FastH3 VSA.

This project incorporates the complete loading and numerical-protection functionality of [MiniMax H3 FP16 Exact Fix - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7) and extends it across supported architectures. New workflows can use this loader directly without chaining the former FP16 loader.

## Core features

| Feature | Description |
|---|---|
| Unified loading | Loads native H3 files, third-party full checkpoints, MixedPrecisionOps quantized models, and FastH3 from one node |
| Structural detection | Identifies base H3, quantized H3, FastH3 Dense, and FastH3 VSA from weights and metadata rather than a filename whitelist |
| Architecture-aware policy | Always uses native BF16 on SM80+ and overrides a global FP16 launcher setting for H3 only |
| Quantization preservation | Retains INT8, ConvRot, quantized layouts, and `_quantization_metadata` without expanding the full model to dense FP16 |
| Third-party compatibility | Supports structurally compatible fine-tuned, merged, pruned, and quantized full H3 checkpoints |
| FP16 protection | Protects H3 attention, residual, and MLP computations from FP16 overflow/non-finite values without replacing the sampler or attention backend |
| FastH3 support | Loads official Preview v1 and v0.1/v0.2 FastVideo shard directories, Star7 native single-file INT8 Dense models, and VSA DataFree INT8 models |
| SM75 VSA acceleration | Includes a VSA Switch and a native Q64/K64 CUDA path that preserves tile-64 routing and the compressed branch |
| Standard output | Produces a normal ComfyUI `MODEL` for standalone use or downstream LoRA, cache, and activation-chunk nodes |

## Supported models

The loader accepts complete checkpoints that ComfyUI can identify as a native `MiniMaxH3Model`, including:

- official dense BF16/FP16 H3;
- ComfyUI MixedPrecisionOps INT8, TensorWise INT8, and ConvRot H3;
- native layouts such as `convrot_w4a4` and `asym_w4a8_int8` when their operators are available in the installed ComfyUI/MixedPrecisionOps environment;
- H3 files with outer prefixes such as `model.diffusion_model.`;
- checkpoints with file-level `_quantization_metadata`;
- third-party full models such as `10ero` and `dasiwa`;
- merged/quantized models such as `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8`;
- other fine-tuned, merged, pruned, or quantized variants that preserve H3 module names and tensor shapes.

Compatibility is determined from structure and weight metadata, not model names. A checkpoint that changes layer count, hidden dimensions, QKV shapes, or custom operator interfaces is rejected with a specific diagnostic instead of receiving an unsafe fallback.

GGUF, GPTQ, bitsandbytes, and other non-MixedPrecisionOps formats remain the responsibility of their dedicated loaders.

## FastH3 support

| Type | Attention | Purpose |
|---|---|---|
| FastH3 Dense | Standard dense attention | Official four-step distilled checkpoints and native ComfyUI INT8 conversions |
| FastH3 VSA DataFree | Tile-64 Video Sparse Attention | Complete four-step sparse checkpoints with the VSA adaptation merged into the H3 base |

Official directories use versioned contract detection: Preview v1 is identified through `fastvideo_inference.json`, while Preview v0.1/v0.2 uses the official repository identity in `modular_model_index.json`. A folder name alone never classifies an arbitrary H3 checkpoint as FastH3.

FastH3 is a complete few-step distilled Transformer, not a Turbo LoRA that must be loaded at runtime. The loader records its inference contract but does not change the workflow's scheduler, flow shift, guidance, VAE, text encoder, frame count, or resolution.

Recommended FastH3 contract:

- four Transformer forwards;
- guidance `1.0`;
- `MiniMaxH3SigmaShift`: video `12`, audio `3`;
- scheduler, sampler, and remaining generation parameters stay under native ComfyUI nodes.

## Companion models

Both single-file models are available from
[suanyu/MiniMax-H3-Star7-INT8](https://huggingface.co/suanyu/MiniMax-H3-Star7-INT8/tree/main):

| File | Variant | Quantization | Usage |
|---|---|---|---|
| `FastH3-Dense-v1-Star7-INT8.safetensors` | `fasth3_dense_v1` | TensorWise INT8 + ConvRot | Select directly in the enhanced loader; do not use the VSA Switch |
| `FastH3-VSA-DataFree-v1-Star7-INT8.safetensors` | `fasth3_vsa_datafree_v1` | TensorWise INT8 + ConvRot | Load with the enhanced loader, then connect the VSA Switch |

Both are complete single-file Transformers and do not require users to download and convert the original shards. Embedded metadata identifies the variant, quantization format, and four-step sampling contract.

- [Download the FastH3 Dense single-file model](https://huggingface.co/suanyu/MiniMax-H3-Star7-INT8/blob/main/FastH3-Dense-v1-Star7-INT8.safetensors)
- [Download the FastH3 VSA DataFree single-file model](https://huggingface.co/suanyu/MiniMax-H3-Star7-INT8/blob/main/FastH3-VSA-DataFree-v1-Star7-INT8.safetensors)
- [Quark mirror](https://pan.quark.cn/s/27f28ba550dc)

## Example workflows

The repository includes four release workflows, with Chinese and English versions for both general H3 and FastH3 VSA:

- `MiniMax-H3-Enhanced-Loader-General-Star7.json`
- `MiniMax-H3-Enhanced-Loader-General-Star7-English.json`
- `MiniMax-H3-Enhanced-Loader-FastH3-VSA-Star7.json`
- `MiniMax-H3-Enhanced-Loader-FastH3-VSA-Star7-English.json`

## Quantization and LoRA

For quantized models, the loader preserves ComfyUI/MixedPrecisionOps weight objects and native operator dispatch. It disables forced weight casting that would otherwise expand the entire quantized model to the compute dtype. FP16 compatibility protects only the required intermediate computations; it does not convert INT8 attention or INT8 linear operations into dense FP16.

The output remains a standard ComfyUI `MODEL`, so shape-compatible H3 LoRAs can be applied normally. ComfyUI may create temporary dequantized weights for layers directly modified by a LoRA in low-VRAM mode; this is weight-patch behavior and does not mean the untouched base model has lost its quantized path.

A conventional base-H3 LoRA can be attached to FastH3, but its response is not guaranteed to match the four-step distilled checkpoint. Begin validation at a lower strength. Official FastH3 Adapter bundles containing `.diff`, `.diff_b`, or `.set_weight` are not conventional LoRA files.

This project now includes `MiniMax H3 FastH3 Adapter Loader - Star7`. It performs header-first validation, then applies every low-rank, dense/bias delta, and all 50 VSA replacement gates to a standard ComfyUI `MODEL`. A gate payload enables VSA automatically; conventional H3 LoRAs should still use a conventional LoRA loader.

```text
Enhanced Loader -> FastH3 Adapter Loader -> SigmaShift -> [optional: Activation Chunk(existing)] -> Scheduler / Guider
```

The published strength is `1.0`. The Adapter is approximately 5 GB and its BF16 VSA gates add about 3.6 GiB of model weights. Applying deltas to an INT8 base can also create temporary dequantized weights during model loading, so a pre-merged Star7 INT8 single file remains the more predictable distribution path for VRAM and startup time.

## FP16 numerical protection

The enhanced loader includes the former FP16 project's core protection path:

- FP32 input for condition projection;
- safely scaled FP16 attention `out_proj` with FP32 magnitude restoration;
- FP32 intermediate computation for residual gates, RMSNorm, modulation, and SwiGLU;
- safe scaling before the MLP down projection;
- ModelPatcher object patches, weakly bound forwards, and duplicate-wrapper markers to avoid global class modification and stale model retention;
- no replacement of CK, Sage, SLA, Sol, VSA, or another selected attention backend.

The protected `out_proj` scale is `64` and the MLP `fc2` scale is `256`. Both are powers of two, avoiding the additional rounding form of a general rescale. “Exact” describes this scaling property; it does not claim bitwise identity between FP16, BF16, INT8, and FP32 execution.

| Environment | Policy |
|---|---|
| NVIDIA SM80+ | Explicit BF16; a global `--fp16-unet` is corrected for H3 only, while quantized dispatch remains intact |
| NVIDIA SM60/SM70/SM75 and similar | FP16 compatibility protection |
| NVIDIA SM61 | Native default path to avoid its low FP16 throughput |
| AMD ROCm | FP16 compatibility path, subject to the installed PyTorch and ComfyUI stack |
| CPU / no CUDA | Native ComfyUI default path |

The enhanced loader can therefore replace `minimax-h3-fp16-exact-star7` in new workflows. SM80+ users get no FP16 block wrappers; use the separate Native FP16 Loader only when protected FP16 is explicitly desired.

## FastH3 VSA Switch

- Display name: `MiniMax H3 FastH3 VSA Switch - Star7`
- Class ID: `MiniMaxH3VSASwitchStar7`

The switch accepts only a FastH3 VSA model identified by the enhanced loader. It validates and enables the matching VSA runtime without changing steps, sigma, flow shift, guidance, or the VAE.

On SM75, the bundled native CUDA launcher computes selected blocks with All-INT8 Q/K/V while preserving the `gate_compress` contribution for non-selected regions. This retains FastH3 VSA semantics and is not equivalent to SLA or generic Sol-Attn.

VSA runs independently of the activation-chunk project. To reduce QKV, RoPE, and MLP peak VRAM, optionally connect [MiniMax H3 Activation Chunk - Star7](https://github.com/star7code/minimax-h3-chunk-star7) after the VSA Switch and select `attention_backend=existing` so the upstream VSA attention remains active.

```text
Enhanced Loader -> VSA Switch -> SigmaShift -> [optional: Activation Chunk(existing)] -> Scheduler / Guider
```

## Nodes

### MiniMax H3 Enhanced Loader - Star7

- Class ID: `MiniMaxH3EnhancedLoaderStar7`
- One H3 model selector;
- always uses ComfyUI dynamic loading;
- reports model variant, precision policy, quantization format, and FastH3 sampling contract.
- quantized-model logs also report the effective backend, `force-cast` state, weight-patch count, and DiT block count so INT8/ConvRot retention can be verified.

### MiniMax H3 FastH3 VSA Switch - Star7

- Class ID: `MiniMaxH3VSASwitchStar7`
- only for FastH3 VSA models;
- the SM75 native VSA binary is bundled with this project and does not depend on the activation-chunk installation.

### MiniMax H3 FastH3 Adapter Loader - Star7

- Class ID: `MiniMaxH3FastH3AdapterLoaderStar7`;
- accepts FastVideo composite Adapters, not conventional LoRAs;
- preserves low-rank, dense delta, bias delta, and replacement-gate payloads;
- rejects incomplete, unknown, or partial-gate payloads before model execution;
- enables the matching VSA runtime automatically and keeps a standard `MODEL` output.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-enhanced-loader-star7.git
```

Restart ComfyUI after installation or update. The historical FP16 project may remain installed for old workflows; the enhanced loader uses independent class IDs and does not overwrite it.

## Attribution and license

FastH3 models and reference code are provided by [FastVideo](https://github.com/hao-ai-lab/FastVideo). FP16 numerical protection derives from the MIT-licensed [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix). Star7 maintains unified loading, architecture policy, FastH3 mapping, quantization preservation, VSA integration, and ComfyUI compatibility.

Project code is MIT-licensed. Companion-model provenance and weight licenses will be documented separately on their Hugging Face model pages.
