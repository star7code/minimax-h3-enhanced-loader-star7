# MiniMax H3 自适应优化载入 - Star7

[English](README_EN.md) · [示例工作流](examples/workflows) · [更新记录](CHANGELOG.md)

面向 ComfyUI MiniMax H3 的统一模型载入与运行时适配节点。它根据模型结构、量化元数据和 GPU 架构自动选择正确路径，在保留模型原生权重与量化实现的前提下，统一支持官方 H3、第三方 H3、混合/量化模型以及 FastH3 Dense、FastH3 VSA 模型。

本项目完整包含 [MiniMax H3 FP16 Exact Fix - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7) 的载入与数值保护能力，并扩展到所有受支持架构。新工作流可直接使用本节点，不再需要额外串联旧 FP16 修复节点。

英文名称：`MiniMax H3 Enhanced Loader - Star7`

## 主要功能

| 功能 | 说明 |
|---|---|
| 统一模型载入 | 使用一个节点载入原生 H3、第三方完整模型、MixedPrecisionOps 量化模型和 FastH3 |
| 模型结构识别 | 根据权重结构与 metadata 识别普通 H3、量化 H3、FastH3 Dense 和 FastH3 VSA，不依赖固定文件名 |
| 架构自适应 | SM80+ 始终采用原生 BF16，并自动覆盖只针对 H3 不合适的全局 FP16 启动参数 |
| 量化路径保持 | 保留 INT8、ConvRot、量化 layout 与 `_quantization_metadata`，不会因启用兼容保护而把整模强制展开为 FP16 |
| 第三方模型兼容 | 支持结构符合 ComfyUI MiniMax H3 的第三方微调、融合、剪枝和量化完整模型 |
| FP16 数值保护 | 修复部分架构上 H3 attention、残差与 MLP 的 FP16 溢出/非有限值问题，不接管采样器或注意力算法 |
| FastH3 支持 | 支持官方 Preview v1 与 v0.1/v0.2 FastVideo 分片目录、Star7 原生单文件 INT8 Dense 模型及 VSA DataFree INT8 模型 |
| SM75 VSA 加速 | 附带 VSA Switch；SM75 使用本项目自带的原生 Q64/K64 CUDA 路径，并保留 VSA 的 tile-64 路由与压缩分支 |
| 标准模型输出 | 输出标准 `MODEL`，可单独使用，也可连接 LoRA、缓存节点和 MiniMax H3 分块节点 |

## 支持的模型

### 官方与第三方 MiniMax H3

节点支持 ComfyUI 能识别为原生 `MiniMaxH3Model` 的完整模型，包括：

- 官方稠密 BF16/FP16 H3；
- ComfyUI MixedPrecisionOps INT8、TensorWise INT8 与 ConvRot H3；
- 由当前 ComfyUI/MixedPrecisionOps 提供算子支持的 `convrot_w4a4`、`asym_w4a8_int8` 等原生量化布局；
- 带 `model.diffusion_model.` 等外层前缀的 H3；
- 使用文件级 `_quantization_metadata` 的量化 H3；
- `10ero`、`dasiwa` 等第三方完整 H3 模型；
- `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8` 等融合/混合量化模型；
- 其他保持 H3 模块名称和张量形状兼容的微调、融合、剪枝或量化版本。

兼容依据是模型结构与权重信息，而不是名称白名单。若第三方模型修改了 H3 层数、隐藏维度、QKV 形状或自定义算子接口，节点会明确报出不兼容位置，不会静默套用错误路径。

GGUF、GPTQ、bitsandbytes 等非 ComfyUI MixedPrecisionOps 格式不属于本节点的原生量化契约，应继续使用各自专用加载器。

### FastH3

| 类型 | Attention | 用途 |
|---|---|---|
| FastH3 Dense | 标准稠密 attention | 官方四步蒸馏模型及其原生 ComfyUI INT8 转换版 |
| FastH3 VSA DataFree | tile-64 Video Sparse Attention | 将 VSA 适配权重与完整 H3 基座合并后的四步稀疏模型 |

官方目录使用版本化契约识别：Preview v1 读取 `fastvideo_inference.json`，Preview v0.1/v0.2 读取 `modular_model_index.json` 中的官方仓库标识。节点不会仅凭文件夹名称把普通 H3 误判为 FastH3。

FastH3 是完整的少步蒸馏 Transformer，不是运行时必须额外加载的 Turbo LoRA。节点会记录其采样契约，但不会修改工作流中的 scheduler、flow shift、guidance、VAE、文本编码器、帧数或分辨率。

推荐的 FastH3 工作流参数：

- Transformer forward：`4` 步；
- guidance：`1.0`；
- `MiniMaxH3SigmaShift`：视频 `12`、音频 `3`；
- scheduler、sampler 和其他生成参数仍由 ComfyUI 原生节点控制。

## 配套模型

两个可由本节点直接识别的单文件模型已发布到
[suanyu/MiniMax-H3-Star7-INT8](https://huggingface.co/suanyu/MiniMax-H3-Star7-INT8/tree/main)：

| 文件 | 变体 | 量化 | 加载方式 |
|---|---|---|---|
| `FastH3-Dense-v1-Star7-INT8.safetensors` | `fasth3_dense_v1` | TensorWise INT8 + ConvRot | 直接在增强载入节点中选择；不使用 VSA Switch |
| `FastH3-VSA-DataFree-v1-Star7-INT8.safetensors` | `fasth3_vsa_datafree_v1` | TensorWise INT8 + ConvRot | 增强载入后连接 VSA Switch |

两者均为完整 Transformer 单文件，不要求用户下载原始分片后再转换。模型内 metadata 用于标识变体、量化方式和四步采样契约。

- [FastH3 Dense 单文件下载](https://huggingface.co/suanyu/MiniMax-H3-Star7-INT8/blob/main/FastH3-Dense-v1-Star7-INT8.safetensors)
- [FastH3 VSA DataFree 单文件下载](https://huggingface.co/suanyu/MiniMax-H3-Star7-INT8/blob/main/FastH3-VSA-DataFree-v1-Star7-INT8.safetensors)
- [夸克网盘备用地址](https://pan.quark.cn/s/27f28ba550dc)

## 示例工作流

仓库提供四份正式工作流，通用 H3 与 FastH3 VSA 各有中英文版本：

- `MiniMax-H3-Enhanced-Loader-General-Star7.json`
- `MiniMax-H3-Enhanced-Loader-General-Star7-English.json`
- `MiniMax-H3-Enhanced-Loader-FastH3-VSA-Star7.json`
- `MiniMax-H3-Enhanced-Loader-FastH3-VSA-Star7-English.json`

## 量化兼容与 LoRA

检测到量化模型时，节点保留 ComfyUI/MixedPrecisionOps 的量化权重对象和原生算子分派，并关闭会迫使量化权重整体转换计算 dtype 的强制 cast。FP16 兼容策略只保护需要的中间计算，不把 INT8 attention 或 INT8 linear 改成稠密 FP16，因此不会抵消量化模型的主要显存与带宽优势。

输出仍是标准 ComfyUI `MODEL`，可以继续连接形状兼容的 H3 LoRA。LoRA 只会影响它实际补丁覆盖的层；ComfyUI 在低显存模式下可能为这些被补丁修改的量化层创建临时反量化权重，这是 LoRA 权重合成行为，不代表基础模型的量化路径失效。

普通 H3 LoRA 可以连接 FastH3，但四步蒸馏权重与基础 H3 的响应不保证完全相同，应从较低强度开始验证。包含 `.diff`、`.diff_b`、`.set_weight` 的官方 FastH3 Adapter 不是普通 LoRA，不能按常规 LoRA 文件解释。

本项目提供独立的 `MiniMax H3 FastH3 Adapter Loader - Star7`。它会先读取 safetensors 头部，完整校验低秩权重、Dense/Bias Delta 和 50 个 VSA replacement gate，再把全部载荷应用到标准 ComfyUI `MODEL`。检测到 gate 时会自动启用 VSA，不需要再连接 VSA Switch；普通 H3 LoRA 仍应使用普通 LoRA 加载器。

```text
Enhanced Loader -> FastH3 Adapter Loader -> SigmaShift -> [可选：Activation Chunk(existing)] -> Scheduler / Guider
```

官方发布强度为 `1.0`。Adapter 文件约 5 GB，且 VSA gate 本身会增加约 3.6 GiB BF16 模型权重；对 INT8 基座应用 Delta 时，ComfyUI 可能在模型装载阶段产生临时反量化权重，因此预先合并的 Star7 INT8 单文件仍是显存和启动速度更稳定的分发方案。

## FP16 数值保护

本项目继承并整合旧 FP16 修复节点的全部核心能力：

- condition projection 使用 FP32 输入，避免条件投影提前溢出；
- attention `out_proj` 在安全缩放后执行 FP16 投影，再恢复 FP32 幅值；
- residual gate、RMSNorm、modulation 和 SwiGLU 激活保留 FP32 中间计算；
- MLP 下投影前进行安全缩放，降低长序列和高幅值条件下产生 NaN/Inf 的风险；
- 使用 ModelPatcher object patch、弱绑定 forward 和重复包装标记，避免修改全局模型类或长期强引用旧模型；
- 不替换用户选择的 CK、Sage、SLA、Sol、VSA 或其他 attention 后端。

`out_proj` 使用 `64`、MLP `fc2` 使用 `256` 进行保护，均为 2 的整数次幂缩放，可避免普通比例换算引入额外的舍入形式。这里的 Exact 指缩放方式，不表示 FP16、BF16、INT8 与 FP32 输出逐位一致。

策略按 CUDA compute capability 选择：

| 环境 | 策略 |
|---|---|
| NVIDIA SM80+ | 始终显式使用 BF16；检测到全局 `--fp16-unet` 时仅对 H3 自动纠正，量化分发保持不变 |
| NVIDIA SM60、SM70、SM75 等 | 使用 FP16 compatibility 数值保护 |
| NVIDIA SM61 | 保留默认路径，避免低 FP16 吞吐造成明显倒退 |
| AMD ROCm | 使用 FP16 compatibility 路径，具体可用性取决于 PyTorch 与 ComfyUI 环境 |
| CPU / 无 CUDA | 保留 ComfyUI 默认路径 |

因此，本节点可以完全替代 `minimax-h3-fp16-exact-star7` 在新工作流中的位置。SM80+ 不增加 FP16 block 包装；明确需要测试受保护 FP16 时可改用独立的 Native FP16 Loader。

## FastH3 VSA Switch

节点名称：`MiniMax H3 FastH3 VSA Switch - Star7`
内部 class ID：`MiniMaxH3VSASwitchStar7`

该节点只接受增强载入节点已识别的 FastH3 VSA 模型，用于校验并启用对应 VSA runtime。它不修改步数、sigma、flow shift、guidance 或 VAE。

SM75 路径使用项目自带的原生 CUDA launcher：选中块执行 Q/K/V All-INT8 计算，未选中部分继续使用 FastH3 VSA 的 `gate_compress` 压缩贡献。该实现保持 VSA 语义，不等同于 SLA 或通用 Sol-Attn。

VSA 可完全独立于分块节点运行。需要降低 QKV、RoPE、MLP 峰值显存时，可在 VSA Switch 后连接 [MiniMax H3 Activation Chunk - Star7](https://github.com/star7code/minimax-h3-chunk-star7)，并将 `attention_backend` 设为 `existing`，使分块节点保留上游 VSA attention。

推荐连接：

```text
Enhanced Loader -> VSA Switch -> SigmaShift -> [可选：Activation Chunk(existing)] -> Scheduler / Guider
```

## 节点与界面

### MiniMax H3 自适应优化载入 - Star7

- class ID：`MiniMaxH3EnhancedLoaderStar7`
- 仅保留一个“H3 模型”选择控件；
- 固定使用 ComfyUI 动态载入；
- 自动输出模型类型、精度策略、量化格式和 FastH3 采样契约日志。
- 量化日志同时报告实际 backend、`force-cast` 状态、权重补丁数量与 DiT block 数量，便于确认 INT8/ConvRot 是否保持生效。

### MiniMax H3 FastH3 VSA Switch - Star7

- class ID：`MiniMaxH3VSASwitchStar7`
- 仅用于 FastH3 VSA 模型；
- SM75 原生 VSA 二进制由本项目提供，不依赖分块项目的安装状态。

### MiniMax H3 FastH3 Adapter Loader - Star7

- class ID：`MiniMaxH3FastH3AdapterLoaderStar7`；
- 仅加载 FastVideo 官方复合 Adapter，不替代普通 LoRA Loader；
- 支持 LoRA、Dense Delta、Bias Delta 与 VSA replacement gate；
- 缺失、未知或只含部分 gate 的 Adapter 会在读取大权重前明确终止；
- VSA Adapter 自动启用正确的 VSA runtime，输出仍为标准 `MODEL`。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-enhanced-loader-star7.git
```

安装或更新后重启 ComfyUI。旧 FP16 项目可以继续保留以打开历史工作流，但新的增强载入节点使用独立 class ID，不会覆盖旧节点。

## 来源与许可

FastH3 模型与参考实现来自 [FastVideo](https://github.com/hao-ai-lab/FastVideo)。FP16 数值保护方法源自 MIT 许可的 [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix)。Star7 负责统一载入、架构策略、FastH3 权重映射、量化保持、VSA 集成及 ComfyUI 兼容实现。

项目代码采用 MIT License。配套模型的权重许可与来源将在各自 Hugging Face 模型页单独说明。
