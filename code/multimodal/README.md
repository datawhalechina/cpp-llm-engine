# 多模态执行层

本目录建立图片进入视觉编码器之前的可验证数据路径。默认构建只需要 C++17；系统存在 `libjpeg` 时增加 JPEG 解码，PPM、原始 RGB、resize、normalize 和 patch 测试不依赖真实模型或 `llama.cpp`。

## 模块边界

多模态模块拥有图片解码、动态切图、视觉 Encoder、Projector、图文 embedding 组装和阶段指标。它复用 Chapter01 的 GGUF Reader、Chapter05 的量化 Linear，以及 Model 中聚焦的稠密 Llama CPU/CUDA 目标；这些实现通过 CMake target 链接，不在本目录重复编译源文件。

```text
Chapter01 Loader ─┐
Chapter05 Linear ─┼─> multimodal CPU/CUDA ─> 统一 embedding 序列
CUDA Backend ─────┘                              │
                                                v
稠密 Llama Model/Context <────────────── Engine / Chapter06 Runtime
```

图片路径扩展输入类型，Chapter06–10 继续负责生成、缓存、调度和服务。多模态目录不复制 Request、K/V Cache、Scheduler 或 Transport；真实检查器只在需要语言模型对照时链接 `cpp_llm_engine_llama_cpu_model` 或 `cpp_llm_engine_llama_cuda_model`。

## 资产总览

当前资产覆盖 IDEFICS3 动态切图、单张视觉 tile 预处理、mmproj schema、Float32 Patch/Position Embedding、使用真实 Q8_0 权重的 12 层 SigLIP Vision Encoder、IDEFICS3 Projector，以及图文 embedding 序列拼接。

可选 CUDA 适配已覆盖 Patch/Position Embedding、SigLIP Encoder、IDEFICS3 Projector 和语言 Context 交接。`CudaSiglipEncoder` 将 12 层真实 Q8_0 packed 权重常驻设备，复用一个 workspace 和两个 hidden buffer。`CudaSmolvlmVisionPipeline` 支持单 tile 与固定网格多图片 packed batch；Linear/MLP 合并执行，双向 Attention 和空间合并保持图片边界。完整 `32×32` tile 将 1024 个 Patch 变为 64 行视觉 embedding。真实分词器产生的 24 个文本词元与这 64 行视觉向量组成 `88x960` 序列，并已通过 Chapter06 Runtime 完成四词元 CPU/CUDA 贪心生成对照。同步 Completion、SSE、双图片视觉组批和双请求语言 Decode 组批已验证；图片 HTTP 传输与图文 Prefill 组批仍待接入。

| 资产 | 输入与输出 | 当前验证 |
| --- | --- | --- |
| `RgbImage` | 连续 HWC RGB8 | 维度与字节数契约 |
| `load_rgb_image` | PPM/JPEG 文件 -> RGB8 | PPM 默认测试；JPEG 可选 inspect |
| `calculate_idefics3_layout` | 原图尺寸与 token budget -> refined/grid | 640×488 -> 1536×1024、3×2 |
| `prepare_idefics3_tiles` | RGB -> overview + row-major tiles | 1 + 6 张 512×512 RGB |
| `resize_bilinear` | 任意有效 RGB -> 固定宽高 RGB | half-pixel 双线性与边界 clamp |
| `normalize_chw` | RGB8 -> Float32 CHW | `(pixel / 255 - mean) / std` |
| `extract_patches` | CHW -> `[patch_count, C*patch_h*patch_w]` | 行优先 patch、patch 内 C/Y/X |
| `preprocess_image` | RGB + 配置 -> CHW 与 patch tensor | SmolVLM 512/16 shape smoke |
| `multimodal_image_inspect` | 本地图像 -> shape/checksum | PPM 默认；真实 JPEG 可选 |
| `MmprojSchema::config_from_gguf` | GGUF metadata -> `VisionModelConfig` | 类型、范围和整除关系 |
| `MmprojSchema::validate_gguf` | tensor index -> `VisionTensorLayout` | 198 个名称和 shape |
| `multimodal_mmproj_inspect` | 真实 mmproj -> 配置与 schema 结果 | 只读 index，不加载 payload |
| `embed_patches` | patch + F32 patch/position 权重 -> hidden states | 合成精确值；真实 mmproj 首 patch |
| `CudaPatchEmbedding` | patch + position id -> 设备 `[tokens, hidden]` | 合成 CPU/CUDA 逐值对照；越界与 shape 失败路径 |
| `CudaSiglipBlock` | 设备 `[tokens, hidden]` -> 单视觉 Block 输出 | 合成 F32 与真实 mmproj Q8_0 CPU/CUDA golden |
| `multimodal_cuda_vision_block_inspect` | 真实 mmproj -> 第 0 个 CUDA Block 输出 | 4 token 最大误差 `8.82e-6`、余弦相似度 `1` |
| `CudaSiglipEncoder` | 设备 `[tokens,768]` -> 12 层 post-norm 输出 | 权重常驻、hidden ping-pong、单 workspace |
| `multimodal_cuda_vision_encoder_inspect` | 真实 mmproj -> 12 层 CUDA Encoder | 4 token 最大误差 `7.03e-5`、余弦相似度 `0.9999999999` |
| `CudaIdefics3Projector` | 设备视觉网格 -> 设备语言宽度 embedding | 空间合并顺序、F32 Linear 与失败路径 |
| `CudaSmolvlmVisionPipeline` | 单张或固定网格图片 batch -> 设备视觉 embedding | 三阶段间 hidden 不回传 Host；Attention 与空间合并按图片隔离 |
| `multimodal_cuda_vision_pipeline_inspect` | 真实图片 + mmproj -> `1x960` 或 `64x960` | 完整 tile 的单图片与 `batch=2` packed 输出通过 CPU/CUDA 对照 |
| `multimodal_cuda_smolvlm_context_test` | 真实视觉 embedding -> 32 层 CUDA LLM Context | 快速单行与完整 64 行 Prefill 均通过 CPU/CUDA logits、位置和贪心词元对照 |
| `run_siglip_block` | hidden states + F32 权重 -> 单 Block 输出 | 合成 LayerNorm、全注意力、GELU、双 residual |
| `multimodal_vision_embedding_inspect` | 真实图片 + mmproj -> 首 patch embedding | `1x768`、checksum、`payload=f32-valid` |
| `load_siglip_block_weights` | F32/Q8_0 payload -> `SiglipBlockWeights` | 合成装配测试；FFN legacy 名称交换 |
| `multimodal_vision_block_inspect` | 图片前 N 个 patch + 真实第 0 Block | 中间 checksum、`block=q8_0-valid` |
| `run_siglip_encoder` | 输入 hidden + 逐层权重 provider -> post-norm hidden | 合成两层串联与失败传播 |
| `multimodal_vision_encoder_inspect` | 图片前 N 个 patch + 真实 12 层 | 逐层 checksum、权重体积、`encoder=q8_0-valid` |
| `merge_spatial_patches` | `[grid_h*grid_w, hidden]` -> 空间合并 token | 合成非方形网格精确顺序 |
| `run_idefics3_projector` | merged hidden -> width 960 visual embedding | 真实 Q8_0 权重与 NumPy golden |
| `multimodal_vision_projector_inspect` | 左上窗口或完整 tile -> visual embedding | 标量 `1x960`；CBLAS `64x960` |
| `compose_multimodal_embeddings` | 文本占位行 + visual embedding -> 统一序列 | 多图片、顺序、来源映射和错误输入 |
| `multimodal_input_embedding_test` | 合成文本与两个视觉插入 | 默认 CTest，不依赖模型或 Python |
| `build_smolvlm_single_turn_prompt` | 问题 + tile 网格 -> 文本/视觉 chunks | 行优先 tiles、overview 最后、边界 token |
| `multimodal_smolvlm_prompt_test` | overview-only 与 `2x3` chunk 布局 | 默认 CTest，不依赖 tokenizer 或模型 |
| Core `DecoderEmbeddingSequence` | `[tokens, hidden]` -> Decoder prefill | token-prefill 与 embedding-prefill logits 一致 |
| SmolVLM Decoder 检查 | 真实 `3x960` -> 49,280 logits | 一层与 32 层均有 C++/NumPy golden |
| SmolVLM prompt 检查 | 真实 BPE + 1 visual row -> `25x960` | 32 层 C++/NumPy top-1 均为 token `378` |
| SmolVLM 单 tile 检查 | 真实 BPE + 64 visual rows -> `88x960` | 32 层 C++/NumPy top-1 均为 token `216` |
| SmolVLM 动态图片检查 | `2x3` tiles + overview -> `486x960` | 448 visual rows；top-1 均为 token `17600` |
| `MultimodalStageRecorder` | 图片输入到 LLM prefill 的请求阶段 -> 耗时/调用次数/峰值存活字节 | 注入时钟的默认 CTest；真实检查器阶段输出 |
| `benchmark_smolvlm_service.py` | 多进程重复执行真实动态图片服务 -> 阶段统计 JSON | hash、sync/SSE token、计数和释放闸门 |
| `run_real_service_benchmark.sh` | Release 构建与模型、BLAS、迭代参数 -> 可复查结果文件 | 默认 3 轮；不进入默认回归 |
| `results/smolvlm_service_cpu_benchmark.json` | 两轮真实动态图片服务样本与统计 | 结果结构、summary 重算和 correctness 可离线复查 |
| `reference/siglip_block_reference.py` | 同一输入的 NumPy/gguf-py 镜像 | 可选外部 golden，不进入默认回归 |

## 模型配置

SmolVLM 500M 的 mmproj metadata 给出 `image_size=512`、`patch_size=16`、mean/std 均为 `0.5`。这些值由模型配置提供，通用预处理函数不固定某个模型。

IDEFICS3 根据长宽比和 image-token budget 生成 overview 与多个 512×512 tile。固定 640×488 输入在 256-token budget 下得到 1536×1024 refined image、3×2 tile grid，以及 `overview + 6 tiles` 共 7 张视觉输入。

当前尺寸、padding 和顺序与外部基线一致；像素 resize 使用教程的双线性实现，参考实现使用 Pillow bicubic，因此本阶段不声明逐像素 checksum 一致。

## mmproj 加载契约

`GgufReader` 复用 Chapter01 的文件格式读取能力，`MmprojSchema` 单独表达视觉模型语义。这样不会把 Llama 文本模型的 `ModelSchema` 扩成同时猜测两类模型的接口。

| 阶段 | 输入 | 输出 | 是否读取权重数据 |
| --- | --- | --- | --- |
| GGUF index | mmproj 文件 | metadata、tensor name/shape/type/offset | 否 |
| 视觉配置 | metadata | image、patch、hidden、block、head、projector 参数 | 否 |
| tensor schema | 配置与 index | patch、12 个 block、post norm、projector 布局 | 否 |
| F32 embedding payload | 已验证布局 | Patch/Position Embedding 权重 | 是，MM-P4A 已实现 |
| Q8_0 block payload | 已验证布局 | SigLIP Block 可执行权重 | 是，MM-P4B 已实现 |
| Q8_0 projector payload | 已验证布局 | `12288 -> 960` Linear 权重 | 是，MM-P5B 已实现 |

## 单个 SigLIP Block

Patch projection 把每个展平 patch 映射到 hidden width，并加入对应 position embedding。随后单个视觉 Block 按 `LayerNorm -> 全注意力 -> residual -> LayerNorm -> GELU MLP -> residual` 执行；视觉 token 之间没有语言模型使用的 causal mask。

`run_siglip_block` 接收可注入的 Float32 权重，因此默认测试能用小矩阵逐项验证计算顺序。加载适配器将 Q8_0 解码为该接口需要的 Float32，并把 legacy IDEFICS3 的 raw `ffn_down` 映射为输入投影、raw `ffn_up` 映射为输出投影。

`CudaSiglipBlock` 沿用同一计算顺序。F32 Linear 权重转置为 cuBLAS 使用的 `[input, output]`；Q8_0 Linear 保留 GGUF 中 `[output, input]` 的 packed payload，由 CUDA fused GEMV 直接解码和累加。`CudaSiglipBlockWorkspace` 按 token 数预分配中间 Tensor 和局部 K/V；`CudaSiglipEncoder` 在 12 层之间复用该 workspace，并用两个 hidden buffer 交替承接层输出。

## 12 层 Vision Encoder

`run_siglip_encoder` 不接收 12 层权重数组，而是在每轮通过 provider 加载一个 Block。Block 执行完成后只把 hidden states 交给下一轮，当前权重和中间 trace 随循环作用域释放；第 12 层结束后执行 `v.post_ln`。

4-patch 验证中，每层解码后权重为 `28,351,488` bytes，进程最大 RSS 为 `46,600` KiB。该值验证少量 patch 的逐层释放策略，不代表完整 1024-token tile 的运行峰值。

CUDA 路径保留 12 层 packed Q8_0 权重。4-token 真实对照只重建一次 workspace，最大绝对误差为 `7.03e-5`。完整 1024-token tile 同样只构建一次 workspace，经过 12 层 Encoder 和 Projector 后得到 `64x960`，与 CPU CBLAS 参考实现的最大误差为 `2.365e-4`、余弦相似度为 `1`；显存和性能仍需由 Chapter09 单独测量。

`CudaSmolvlmVisionPipeline` 在 Host 完成图片解码、缩放、归一化和 patch 提取，随后把 Patch/Position、12 层 Encoder 与 Projector 串在同一 CUDA Context 中。三个视觉阶段之间的 hidden 不回传 Host；快速模式选取左上 `4×4` patch 子网格，完整模式执行全部 `32×32` patch。语言 Context 对照仍会把视觉结果下载到 Host，再由 `prefill_embeddings()` 上传到语言模型自己的 CUDA Context。这个中转固定两个执行层的接口和数值边界，设备直连与完整图文生成是下一阶段。

多图片组批先要求图片使用相同网格。Pipeline 把 `B` 张图片打包为 `[B×T,hidden]`：Patch Linear、LayerNorm、Q/K/V、MLP 和 Projector Linear 一次处理全部行；视觉 Attention 只读取同一图片的 `T` 个 Patch，空间合并也先定位图片再定位网格。真实完整 tile 的 `batch=2` 输出最大误差为 `4.88e-4`，两张图片均匹配 CPU reference。该接口完成视觉计算组批，还没有把不同网格做 padding，也没有把视觉 batch 与语言 Prefill 合成一个服务事务。

## IDEFICS3 Projector

Projector 先按 `dy -> dx -> hidden` 顺序把相邻 `4x4` 个 Encoder token 拼成一个向量，再执行 Q8_0 权重的 Linear。完整 tile 会从 `32x32x768` 变为 `8x8x12288`，最终得到 64 个 width 960 visual embedding。

标量 correctness 路径选择原始 `32x32` 网格左上角的 `4x4` patch，并使用真实 position id `0..3、32..35、64..67、96..99`。完整 tile 路径执行全部 1024 个 patch，通过可选 CBLAS backend 加速 Patch Linear、视觉 Attention、MLP 和 Projector；默认标量 API 与默认回归不依赖 BLAS。

## 图文 Embedding 序列

`compose_multimodal_embeddings` 接收已经完成 token embedding 的文本矩阵，以及若干“占位位置 + visual embedding”记录。普通文本行保持原顺序，占位行本身被移除，对应的一行或多行视觉向量在原位置展开；输出同时记录每一行来自哪个文本位置或第几个视觉输入。

该接口只负责连续向量序列，不解析 prompt，也不加载模型。真实检查器可额外读取 SmolVLM 文本 GGUF，将 BOS、图片占位和 EOS 组成 `3x960` 输入：BOS/EOS 来自真实 Q8_0 `token_embd.weight`，中间行来自真实 Projector。

共享 Decoder 已支持预计算 embedding prefill，Q8_0 也已进入统一 Linear policy。真实 GGUF chat template 和 llama.cpp BPE 产生图片 chunk 与问题 token；overview-only correctness 使用一个 visual row 组成 `25x960`，完整单 tile 使用 64 个 visual rows 组成 `88x960`。固定 640×488 JPEG 进一步形成 `2x3` tiles 与 overview，7 个视觉输入共展开 448 行，最终 Decoder 输入为 `486x960`。

动态 C++ 与 NumPy 路径都得到 top-1 token `17600`。JPEG 分别由系统 libjpeg 和 Pillow 解码，细微像素差异经过 resize 与 12 层视觉模型后会放大，因此结果文件分别保存两套数值 golden；跨实现严格核对 grid、chunk/token 顺序、shape 和 top-1，不把 JPEG 路径描述为逐值一致。无损 PPM 继续承担逐值数值对照。

## 请求阶段与 Runtime 衔接

`MultimodalStageRecorder` 依次记录图片解码、动态切图、prompt tokenize、Vision Encoder、Projector、embedding 组装和语言模型 prefill。Vision Encoder 与 Projector 会按视觉输入重复调用，因此指标同时保留累计耗时和调用次数。

`peak_live_bytes` 是调用方传入的请求中间数据估算，只用于比较同一负载下的生命周期变化；它不包含模型常驻权重、分配器开销或整个进程 RSS。统一 embedding sequence 通过 Engine 的 `EmbeddingRequestExecutor` 进入 Chapter06 Runtime，图片管线不复制 Request、Scheduler、KV Cache 或采样逻辑。

| 指标 | 含义 | 不包含 |
| --- | --- | --- |
| `stage_*_seconds` | 某阶段在单个请求内的累计 wall time | 跨请求排队时间 |
| `stage_*_count` | 阶段调用次数 | 视觉 token 数 |
| `stage_*_peak_live_bytes` | 阶段开始/结束时报告的最大请求存活字节 | 模型权重、allocator bookkeeping |
| `request_peak_live_bytes` | 全阶段请求存活字节峰值 | 进程 RSS、共享只读模型 |
| `visual_input_count` / `visual_row_count` | tile/overview 数与 Projector 输出行数 | 文本 token 数 |
| `prompt_position_count` | 交给 Decoder prefill 的统一序列长度 | 后续 decode token |

## 运行

默认检查使用仓库内的 2x2 PPM fixture：

```bash
bash code/multimodal/scripts/run.sh
```

有 NVIDIA GPU 时，可单独运行首个视觉 CUDA 组件对照。该命令不需要 SmolVLM 或 mmproj：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda.sh
```

成功条件是 `multimodal_cuda_vision_embedding_test`、`multimodal_cuda_siglip_block_test`、`multimodal_cuda_vision_encoder_test` 和 `multimodal_cuda_vision_projector_test` 通过。该入口验证 Patch Linear、乱序 position、标准 LayerNorm、非因果多头注意力、GELU、双残差、设备空间重排和 F32 Projector。

本机已有 SmolVLM mmproj 时，运行真实 Q8_0 第 0 Block 对照：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_real_block.sh
```

脚本默认使用 4 个确定性视觉 token。它检查真实 schema、F32 norm/bias、Q8_0 packed Linear、非因果 Attention 和最终输出。12 层 Encoder 对照使用：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_real_encoder.sh
```

真实图片的小窗口设备流水线使用：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_real_pipeline.sh
```

成功输出包含 `output_shape: 1x960`、`workspace_rebuilds: 1` 和 `cpu_cuda_match: true`。该入口读取默认 PPM fixture，也可以把 mmproj 与图片路径作为前两个参数传入。

完整视觉 tile 使用同一入口的 `full-tile` 模式，并启用 CPU CBLAS 参考实现：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_real_pipeline.sh \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  code/multimodal/tests/fixtures/two_by_two.ppm \
  full-tile
```

成功输出包含 `selected_grid: 32x32`、`output_shape: 64x960`、`workspace_rebuilds: 1` 和 `cpu_cuda_match: true`。

继续把该视觉 embedding 送入 SmolVLM 的 32 层 CUDA 语言 Context：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_smolvlm_context.sh
```

脚本默认通过 CTest 运行单行快速测试，核对 CPU/CUDA logits、贪心词元和 Context 位置。视觉与语言 Context 分别拥有执行资源，视觉结果经 Host 中转后进入语言 Prefill；默认模式用于快速固定接口，完整模式继续验证 64 行视觉 Prefill。

完整 64 行视觉 Prefill 使用 `full-tile` 模式：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_smolvlm_context.sh \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf \
  code/multimodal/tests/fixtures/two_by_two.ppm \
  full-tile
```

该入口要求 `visual_tokens: 64`、`context_positions: 64,64` 和 `cpu_cuda_match: true`。继续使用真实 SmolVLM Prompt、分词器和单步 Decode：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_smolvlm_context.sh \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf \
  code/multimodal/tests/fixtures/two_by_two.ppm \
  mixed-prompt \
  "What historic event is shown? Answer in one short sentence."
```

该模式把 6 个前缀文本词元、64 行视觉向量和 18 个后缀文本词元组成 `88x960` 序列。CPU/CUDA Prefill 的首个贪心词元均为 `216`，后续 Decode 生成 `41,31,33`，四词元序列一致；Runtime 最终把两端 Context 位置推进到 `91`。视觉输出目前经 Host 组装后进入语言 Context；设备直连、图片 HTTP 传输和 CUDA 批量服务属于下一阶段。

在相同图文输入上验证 Chapter10 同步 Completion、SSE 与终态释放：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_smolvlm_context.sh \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf \
  code/multimodal/tests/fixtures/two_by_two.ppm \
  mixed-service \
  "What historic event is shown? Answer in one short sentence."
```

该模式先为同步与 SSE 请求分别创建 CUDA Context，再把两个独立图文请求交给 Chapter08 连续调度。Engine 的 `CudaEmbeddingBatchRunner` 从每个 Executor 读取不转移所有权的 CUDA Context，并按 batch 大小持久复用工作区。成功输出包含 `service_sync_match: true`、`service_stream_match: true`、`service_metrics_match: true`、`service_model_batch_calls: 3`、`service_tensor_batch_requests: 6`、`service_batch_workspaces: 1` 和 `cpu_cuda_match: true`。两个请求分别完成 88-position Prefill，之后三轮 Decode 均以 `batch=2` 执行。视觉 Pipeline 已有独立的固定网格 `batch=2` 数值入口，但 Service 尚未收集待处理图片；图文 Prefill 仍按请求执行。

若本机已有 `llama.cpp/tools/mtmd/test-1.jpeg`，可以对 JPEG 做额外检查：

```bash
MULTIMODAL_BUILD_DIR=/tmp/cpp-llm-engine-multimodal-build \
  bash code/multimodal/scripts/run.sh
/tmp/cpp-llm-engine-multimodal-build/multimodal_image_inspect \
  llama.cpp/tools/mtmd/test-1.jpeg
```

若本机已有 SmolVLM mmproj，可以在同一入口追加真实 schema 验收：

```bash
MMPROJ_PATH=llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  bash code/multimodal/scripts/run.sh
```

同时提供图片路径时，入口会计算真实图片的首个 patch embedding，执行真实第 0 个 Block、12 层 Encoder，并用左上 `4x4` patch 验证 Projector：

```bash
MMPROJ_PATH=llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
MULTIMODAL_IMAGE_PATH=llama.cpp/tools/mtmd/test-1.jpeg \
  bash code/multimodal/scripts/run.sh
```

数值 golden 固定使用无损 PPM，避免 Pillow 与系统 libjpeg 的 JPEG 解码配置差异。结果与模型/input hash 保存在 `results/siglip_block_q8_0_golden.json`；可选 NumPy 镜像命令为：

```bash
python3 code/multimodal/reference/siglip_block_reference.py \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  code/multimodal/tests/fixtures/two_by_two.ppm 4
```

构建完成后可一次核对模型/input hash、C++ 输出、NumPy 输出和记录值：

```bash
python3 code/multimodal/reference/verify_siglip_block_golden.py \
  /tmp/cpp-llm-engine-multimodal-build/multimodal_vision_block_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

12 层对照使用独立结果文件和验证脚本：

```bash
python3 code/multimodal/reference/verify_siglip_encoder_golden.py \
  /tmp/cpp-llm-engine-multimodal-build/multimodal_vision_encoder_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

Projector 对照同时固定空间顺序、shape、数值和解码权重体积：

```bash
python3 code/multimodal/reference/verify_idefics3_projector_golden.py \
  /tmp/cpp-llm-engine-multimodal-build/multimodal_vision_projector_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

同时提供文本模型后，可以验证 Projector width、真实 token embedding 和图文来源顺序：

```bash
MMPROJ_PATH=llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
MULTIMODAL_MODEL_PATH=llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf \
MULTIMODAL_IMAGE_PATH=code/multimodal/tests/fixtures/two_by_two.ppm \
MULTIMODAL_BUILD_DIR=/tmp/cpp-llm-engine-multimodal-build \
  bash code/multimodal/scripts/run.sh

python3 code/multimodal/reference/verify_multimodal_input_golden.py \
  /tmp/cpp-llm-engine-multimodal-build/multimodal_vision_projector_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf
```

32 层验证继续复用同一个 `3x960` 最小序列，用于隔离层数变化；它不代表真实 SmolVLM prompt：

```bash
python3 code/multimodal/reference/verify_multimodal_decoder_32_layer_golden.py \
  /tmp/cpp-llm-engine-multimodal-build/multimodal_vision_projector_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf
```

真实 tokenizer 是可选集成，不进入默认构建。先构建 llama.cpp，再显式提供源码与构建目录：

```bash
cmake -S code/multimodal -B /tmp/cpp-llm-multimodal-real \
  -DCMAKE_BUILD_TYPE=Release \
  -DMULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DLLAMA_CPP_BUILD_DIR=/tmp/cpp-llm-engine-smolvlm-build
cmake --build /tmp/cpp-llm-multimodal-real -j 4

python3 code/multimodal/reference/verify_smolvlm_prompt_golden.py \
  /tmp/cpp-llm-multimodal-real/multimodal_vision_projector_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf
```

完整单 tile 需要显式启用 CBLAS。当前接口兼容 Netlib BLAS 与 OpenBLAS；后者适合降低 1024-token 视觉编码耗时，但不是默认构建依赖：

```bash
cmake -S code/multimodal -B /tmp/cpp-llm-multimodal-full-tile \
  -DCMAKE_BUILD_TYPE=Release \
  -DMULTIMODAL_ENABLE_CBLAS=ON \
  -DMULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DLLAMA_CPP_BUILD_DIR=/tmp/cpp-llm-engine-smolvlm-build
cmake --build /tmp/cpp-llm-multimodal-full-tile -j 4

python3 code/multimodal/reference/verify_smolvlm_single_tile_golden.py \
  /tmp/cpp-llm-multimodal-full-tile/multimodal_vision_projector_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf
```

如系统同时存在多个 BLAS 实现，可以用 `MULTIMODAL_CBLAS_LIBRARY` 固定具体动态库，避免 CMake 误选单线程 reference BLAS。动态图片验收复用同一构建：

```bash
cmake -S code/multimodal -B /tmp/cpp-llm-multimodal-dynamic \
  -DCMAKE_BUILD_TYPE=Release \
  -DMULTIMODAL_ENABLE_CBLAS=ON \
  -DMULTIMODAL_CBLAS_LIBRARY=/path/to/libopenblas.so \
  -DMULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DLLAMA_CPP_BUILD_DIR=/tmp/cpp-llm-engine-smolvlm-build
cmake --build /tmp/cpp-llm-multimodal-dynamic -j 4

python3 code/multimodal/reference/verify_smolvlm_dynamic_tiles_golden.py \
  /tmp/cpp-llm-multimodal-dynamic/multimodal_vision_projector_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/tools/mtmd/test-1.jpeg
```

## 真实 ServiceCore 与 SSE 验证

服务 smoke 复用动态图片产生的 `486x960` embedding，为同步 completion 和 SSE 分别创建独立的 32 层 SmolVLM Context。图片与 prompt 在进入 HttpAdapter 前已经转换为每请求 Executor；Chapter10 不解析 JPEG，也不持有 Vision Encoder。当前接口验证模型执行与服务生命周期，multipart/base64 图片协议属于后续 transport 设计。

该路径需要显式启用服务集成，并使用已有 OpenBLAS、llama.cpp build、语言模型、mmproj 和固定 JPEG。运行时间明显长于单次 logits 检查，因为同步与 SSE 不能共享请求 KV 状态。

```bash
cmake -S code/multimodal -B /tmp/cpp-llm-multimodal-service \
  -DCMAKE_BUILD_TYPE=Release \
  -DMULTIMODAL_ENABLE_CBLAS=ON \
  -DMULTIMODAL_CBLAS_LIBRARY=/path/to/libopenblas.so \
  -DMULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON \
  -DMULTIMODAL_ENABLE_SERVICE_INTEGRATION=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DLLAMA_CPP_BUILD_DIR="$PWD/llama.cpp/build"
cmake --build /tmp/cpp-llm-multimodal-service \
  --target multimodal_vision_projector_inspect -j 4

python3 code/multimodal/reference/verify_smolvlm_service_golden.py \
  /tmp/cpp-llm-multimodal-service/multimodal_vision_projector_inspect \
  llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf \
  llama.cpp/tools/mtmd/test-1.jpeg
```

## 真实服务重复 Benchmark

重复 benchmark 每轮启动一个独立进程，完整 wall time 包含权重读取和进程成本；程序内部的阶段计时从图片解码开始，不包含 mmproj、Decoder 权重加载，因此 JSON 会同时保留两种时间口径。每轮必须生成同步/SSE token `17600`，并满足 accepted/completed/released、generated token 和 inflight 守恒，失败时不会写出可用结果。

默认执行 3 个测量轮次、不执行 warmup。`MULTIMODAL_CBLAS_LIBRARY` 可以固定 OpenBLAS；比较两份结果时还必须保持模型、图片、CPU、线程环境和构建类型一致。

当前最小重复闭环使用 AMD Ryzen 9 9955HX、Release、OpenBLAS library-default 线程设置和两轮测量。wall mean 为 `177.15s`，输入阶段 mean 为 `176.98s`、CV 为 `0.0177%`；Vision Encoder mean 为 `16.35s`，同步/SSE 两次语言 prefill 合计 mean 为 `160.49s`。样本数只能证明采集闭环和本次短期稳定性，不能外推其他机器，也不能替代更长时间的稳态测试。归档结果可以不加载模型直接复查。

```bash
MULTIMODAL_CBLAS_LIBRARY=/path/to/libopenblas.so \
MULTIMODAL_SERVICE_BENCHMARK_WARMUP=0 \
MULTIMODAL_SERVICE_BENCHMARK_ITERATIONS=3 \
bash code/multimodal/scripts/run_real_service_benchmark.sh

python3 code/multimodal/reference/benchmark_smolvlm_service.py \
  --validate-result code/multimodal/results/smolvlm_service_cpu_benchmark.json
```
