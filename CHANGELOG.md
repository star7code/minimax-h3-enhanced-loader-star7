# Changelog

## 1.2.4 - 2026-08-30

- VSA 原生运行时和 SM75 DLL 均由增强载入项目自身提供，不依赖分块项目。
- 清理旧会话残留的全局 Chunk RoPE 包装：未连接分块时 VSA 保持独立运行；
  连接分块时仍由下游分块节点按当前参数重新启用 RoPE/QKV/MLP 分块。
- 重构中英文发布说明，明确第三方 H3、Hybrid INT8、原生量化保持、完整
  FP16 数值保护、FastH3 Dense/VSA 与旧 FP16 项目的替代关系。
- 中文节点名统一为 `MiniMax H3 自适应优化载入 - Star7`，并为 VSA Switch
  增加中文标题与控件标签。

## 1.2.3 - 2026-08-30

- 修复 SM75 原生 VSA 部分 tile 的边界屏蔽错误：此前把绝对序列偏移当成
  tile 内局部偏移，导致长序列 attention 实际只读取第一个 tile，视频输出为
  彩色噪点；现在按每个 tile 的真实有效 token 数计算边界。
- VSA attention 日志改为默认只输出加载时配置和每个采样 step 一行；逐次
  attention 的详细行仅在设置 `STAR7_VSA_VERBOSE=1` 时输出。
- VSA 工作流由 VSA 节点负责 step 计时，分块节点在检测到 VSA 上游后不再
  重复打印同一耗时。
- 修复最终版 VSA 示例工作流在当前 VHS 版本中使用日期占位符导致视频无法落盘的问题，输出路径改为稳定的 `MiniMax-h3/` 目录。
- 修正音频任务类型在示例工作流中的内部值为 `auto`，避免旧显示文本造成提交校验失败。

## 1.2.1 - 2026-08-30

- 将 VSA 节点收窄并重命名为 `MiniMax H3 FastH3 VSA Switch - Star7`
  (`MiniMaxH3VSASwitchStar7`)：只负责显式校验/
  启用 VSA，输出原样 `MODEL`；恢复原生 `MiniMaxH3SigmaShift` 与
  `BasicScheduler(steps=4)`，分块节点使用 `existing` 保留 VSA 后端。
- 将 VSA Triton 首轮配置收敛为单一 SM75 保守配置，避免在完整视频序列上
  自动测试多组 kernel 导致首步长时间无输出；不改变 VSA tile-64 路由。
- 移除“禁用动态载入”节点选项，所有模型加载路径固定使用 ComfyUI 动态载入。
- 不保留未发布旧工作流的兼容参数，示例工作流同步改为单一模型选择控件。
- 记录 22GB 显存实测：禁用动态载入触发大量 pin 失败，并将 warm 总任务从
  约 26.93 秒拖慢到约 64 秒。
- 接入本地 FastH3 VSA DataFree INT8 实验路径：53 个分片、50 个
  `gate_compress`、tile-64 block-sparse Triton attention；端到端速度尚未宣称。
- 修复 SM75 Triton 前向中概率块 BF16 与 V FP16 的 dtype 不匹配，已通过本机
  128-token block-sparse smoke test。

## 1.2.2 - 2026-08-30

- 新增 Windows SM75 原生 FastH3 VSA Q64/K64 CUDA 精确块路径；保留 VSA
  tile-64 路由、部分 tile 屏蔽和 `gate_compress` 压缩分支。
- SM75 VSA 路径不再调用 Triton 的 LUT 编译；原生 launcher 不可用时直接
  报出可诊断错误，避免长时间满载却没有采样输出。
- 现有 CK/Sol 后端仍保持独立：CK 负责通用量化 attention，Sol 负责通用
  exact-plus-centroid 稀疏 attention，均不会覆盖 FastVideo VSA 语义。

## 1.2.0 - 2026-08-30

- 将 FastH3 的 RMSNorm + modulation 以及注意力后的 residual gate +
  RMSNorm + modulation 合并为适配 ComfyUI 连续分段的 Triton kernel。
- RTX 2080 Ti、S=8773、全 4 步 SLA 的 warm A/B 中，稳态从
  约 4.27–4.28 秒/步降至 4.11–4.13 秒/步，约提升 3.5%。
- 新增 `STAR7_FASTH3_FUSED_MODULATION=0` 回退开关及真实尺寸微基准。

## 1.1.0 - 2026-08-30

- FastH3 INT8 的 value-first SwiGLU 现在重排为 Comfy Kitchen 原生输入，
  并融合到 INT8 `fc2` 动态量化/GEMM；FP32 输出保留 SM75 溢出保护。
- 新增 `FastH3-INT8-Speed-SM75.json`，将速度基准固定为全 4 步
  `sla_sm75_all_int8`，避免把 CK、Sol、LoRA 和模型变更混在一次对比中。

## 1.0.0 - 2026-08-29

- 新建独立的 `MiniMax H3 Enhanced Loader - Star7` 项目与 Registry 包。
- 支持原生 H3、原生量化 H3 和官方 FastH3 Dense Preview v1 分片目录。
- 加入基于 compute capability 的 BF16、默认精度与 FP16 compatibility 自动策略。
- 集成完整 FP16 Exact 数值保护、MixedPrecisionOps/INT8/ConvRot 保留及安全 ModelPatcher 生命周期。
- 加入 FastVideo 到 ComfyUI H3 的确定性权重映射和状态字典完整校验。
- 加入中英文节点标题与基础控件本地化。
