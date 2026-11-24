# 01.5 模型元数据与 GGUF

真实模型文件同时保存模型元数据、分词器配置、张量索引和权重数据。本节沿 GGUF 读取器、元数据适配器、张量加载器和模型构建器，追踪这些信息如何变成后续章节使用的内部对象。

GGUF 数据区对齐、张量偏移和字节范围的统一计算见 [`F-LOAD-01`](../reference/formulas.md#f-load-01)。

模型元数据（metadata）是模型的配置说明，例如隐藏维度为多少、包含多少个解码器块、使用多少个注意力头。模型结构约束（schema）是检查规则：它根据架构和元数据，确认文件中确实存在所需权重，并检查每个权重的形状。两者的关系可以先记成：

```text
模型元数据：模型声明“我是怎样的模型”
模型结构约束：加载器核对“权重是否符合这份声明”
```

例如，元数据声明隐藏维度为 `1024`；结构约束随后检查 `token_embd.weight` 的第一维是否为 `1024`，并检查每个解码器块是否具有该架构要求的注意力和 FFN 权重。

## GGUF 加载链

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/gguf_reader.cpp"><code>gguf_reader.cpp</code></a> 读取文件结构，<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/model_metadata_adapter.cpp"><code>model_metadata_adapter.cpp</code></a> 归一化配置，<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/model_builder.cpp"><code>model_builder.cpp</code></a> 检查配置与权重是否匹配。

![Chapter 01 GGUF 四层加载链](../assets/diagrams/chapter01-gguf-loading-chain.svg)

| 层 | 主要输入 | 主要输出 | 检查内容 |
| --- | --- | --- | --- |
| GGUF 读取器 | 文件字节 | 文件头、键值、张量索引、偏移 | 二进制结构与数据区位置 |
| 元数据适配器 | GGUF 键值 | `ModelMetadata` | 架构字段、层数、维度和特殊词元 |
| 张量加载器 | 张量索引和数据区 | 张量映射 | 名称、形状、数据类型、偏移和权重数据 |
| 模型构建器 | 模型元数据、张量映射、分词器 | `LoadedModel`、`DecoderInput` | 词表、模型结构约束和权重映射 |

分词器元数据来自同一份 GGUF。独立适配器读取词表、分数、词元类型、BOS/EOS 和可选聊天模板；张量加载器则按索引读取权重。

不同外部格式经各自的适配器转换为同一种内部配置：

```text
GGUF Reader -> GGUF Metadata Adapter -> ModelMetadata
其他 Reader -> 对应 Metadata Adapter -> ModelMetadata
                                      -> ModelConfig
```

替换文件格式时，解码器和生成运行时继续消费 `ModelConfig` 和张量映射，不读取 GGUF 键或文件偏移。

## 配置、权重与模型布局

内部配置保存检查模型结构所需的字段：

| 字段 | 用途 |
| --- | --- |
| `architecture` | 选择对应模型结构约束 |
| `vocab_size` | 校验分词器、词嵌入和输出层 |
| `hidden_size` | 校验张量形状和解码器隐藏状态 |
| `max_position` | 限制输入长度和位置编号 |
| `layer_count` | 校验解码器块数量 |
| `attention_head_count` / `attention_head_count_kv` | 计算 Q 头和共享 K/V 头的投影宽度 |
| `attention_key_length` / `attention_value_length` | 确定 GQA 的头维度，或 MLA 每个词元保存的压缩 K/V 宽度 |
| `bos_token_id` / `eos_token_id` | 标记输入输出中的特殊词元 |

TinyLlama 1.1B 提供本教程的第一条真实模型基线。`block_count=22` 对应 `blk.0` 到 `blk.21`；Llama 结构约束检查每个解码器块的九个权重张量，以及块堆栈之外的词嵌入、末尾归一化和输出权重。

Qwen3 复用同一批主体名称，但结构约束并不完全相同：

| 检查项 | Llama / TinyLlama | Qwen3 |
| --- | --- | --- |
| 注意力与 FFN 主体权重 | Q/K/V、输出投影、Gate/Up/Down | 复用相同主体名称 |
| 投影宽度 | Q 输出通常等于隐藏维度 | 按头数乘以头维度计算 |
| Q/K 归一化 | 没有独立权重 | 每层要求 `attn_q_norm.weight` 和 `attn_k_norm.weight` |
| 输出权重 | `output.weight` 必需 | 可以缺省并与 `token_embd.weight` 共享 |
| BOS | TinyLlama 文件提供 | 可以缺省，内部使用 `-1` 表示不存在 |

Qwen3-0.6B 的隐藏维度是 1024，但每个 Q/K/V 头包含 128 个数值。16 个查询头得到宽度 2048 的 Q 投影，8 个 K/V 头得到宽度 1024 的 K/V 投影；输出投影再把 2048 维注意力结果映射回 1024 维隐藏状态。默认合成测试和真实 GGUF 都会核对这些关系；Chapter04 再用首层执行验证 Q/K Norm、NeoX RoPE 与增量 K/V，并用 28 层 CPU Decoder 对照 llama.cpp。

Qwen1.5-MoE 使用另一份结构约束。真实 GGUF 声明 24 层、60 个专家，并为每个词元选择 4 个专家。每层除了注意力权重，还包含路由器、三组打包专家权重、共享专家门控和三组共享专家权重。`MoeDecoderTensorLayout` 保存这些名称与形状；Chapter04/05 解释 Top-K 路由和专家 FFN，`code/model` 再连接 24 层模型与请求上下文。

DeepSeek-V2-Lite 同时改变注意力和 FFN 布局。它的多头潜在注意力（MLA）先把 K/V 历史压缩到低秩表示，再在计算注意力时展开；因此 Loader 需要同时保存压缩缓存宽度、展开后的 K/V 头宽度和 K/V 低秩维度。

| DeepSeek2 字段 | 固定模型值 | Loader 用途 |
| --- | --- | --- |
| K/V 低秩维度 | `512` | 每个词元保存的潜在 K/V 内容 |
| RoPE 维度 | `64` | 与 K/V 低秩内容共同形成 576 维压缩 K |
| MLA K/V 展开维度 | `192 / 128` | 检查注意力投影输出形状 |
| 前置稠密层数 | `1` | `blk.0` 使用普通 Gate/Up/Down FFN |
| 路由专家 | `64 选 6` | `blk.1` 以后检查 Router、专家和共享专家 |

DeepSeek2 GGUF 可以用通用 K/V 长度字段或 MLA 专用字段表示 `192 / 128`。元数据适配器把两种表示转换为相同的内部字段：576/512 的压缩缓存宽度和 192/128 的展开宽度。结构约束同时识别合并的 `attn_kv_b.weight` 和分离的 K-B、V-B 权重。

文件解析和模型布局通过两条路径汇合：

```text
metadata adapter -> ModelMetadata -> ModelConfig
张量加载器       -> 张量映射      -> 稠密 Decoder 张量布局 -> LoadedModel / DecoderInput
                                  -> MoE Decoder 张量布局  -> 真实 MoE Model / Context
                                  -> DeepSeek2 张量布局     -> MLA/MoE 结构输入
```

`DecoderInput` 保存张量名称映射，权重载荷由 `LoadedModel.tensors` 持有。Chapter04 根据配置和权重映射执行真实解码器；Chapter05 负责解码 Q4_K、Q6_K 等量化载荷。

## 模型路径、架构与后端怎样选择

模型切换包含三个不同决定：

| 决定 | 信息来源 | 当前行为 |
| --- | --- | --- |
| 读取哪个模型 | 命令行模型路径或 `CPP_LLM_ENGINE_MODEL_PATH` | 调用者选择 GGUF 文件 |
| 使用哪种模型架构 | GGUF 的 `general.architecture` | Loader 自动识别 `llama`、`qwen3`、`qwen2moe` 或 `deepseek2`，并选择对应结构约束 |
| 在哪里执行 | CPU/CUDA 入口或设备配置 | 执行入口根据模型家族和可用后端创建对应执行器；支持范围见[模型支持表](../reference/model_support.md) |

架构由模型文件中的 `general.architecture` 声明。可选的 `--architecture` 参数用于核对预期架构，加载器仍以 GGUF 声明选择结构约束。稠密模型进入 Llama 或 Qwen3 数学路径，Qwen2MoE 与 DeepSeek2 使用各自的模型类和执行器工厂。Chapter01 输出元数据与权重布局，Chapter04–06 和 `code/model` 继续执行注意力、专家网络与完整解码器。

## 验证与衔接

默认回归使用小型测试文件检查 GGUF 文件头、模型元数据、张量索引、数据对齐，以及 Llama、Qwen3、Qwen2MoE 和 DeepSeek2 结构约束：

```bash
bash code/chapter01_tensor_loader/run.sh
```

已下载 TinyLlama GGUF 时，可以直接检查真实模型索引：

```bash
cmake -S code/chapter01_tensor_loader \
  -B /tmp/cpp-llm-engine-chapter01-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-chapter01-build \
  --target gguf_inspect -j2
/tmp/cpp-llm-engine-chapter01-build/gguf_inspect \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --validate-schema
```

Qwen3-0.6B 使用同一个工具验证。下载脚本固定模型资产并检查文件校验值：

```bash
/tmp/cpp-llm-engine-chapter01-build/gguf_inspect \
  llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf \
  --validate-schema
```

完整的 Qwen3 Loader 基线可直接运行：

```bash
bash code/chapter01_tensor_loader/run_qwen3_baseline.sh
```

Qwen1.5-MoE 下载完成后，运行固定资产入口验证专家布局：

```bash
bash code/chapter01_tensor_loader/run_qwen15_moe_schema.sh
```

预期输出包含 `schema: qwen2moe tensors valid`、`decoder layers: 24` 和 `experts selected/total: 4/60`。

DeepSeek-V2-Lite 下载完成后，运行真实 MLA 与混合层结构检查：

```bash
bash code/chapter01_tensor_loader/run_deepseek_v2_lite_schema.sh
```

预期输出包含 `schema: deepseek2 tensors valid`、`MLA key/value length: 192/128`、`leading dense layers: 1` 和 `experts selected/total: 6/64`。该入口验证 GGUF 元数据与张量索引；DeepSeek 解码器由 Chapter04–06 的真实模型入口验证。

需要单独核对某个权重时，使用 `--tensor` 查看数据类型和形状：

```bash
/tmp/cpp-llm-engine-chapter01-build/gguf_inspect \
  llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf \
  --tensor blk.0.attn_q.weight
```

验证结果应覆盖：模型元数据可以转换为 `ModelConfig`，张量加载器可以按索引读取权重，结构约束可以发现缺失或形状错误，模型构建器可以生成 `LoadedModel`。TinyLlama 配置与权重由 [`Chapter04`](../chapter04_attention/06_real_tinyllama_decoder.md) 的 `code/model` 使用；Qwen3 在 [`首层与完整 CPU 解码器`](../chapter04_attention/05_load_first_layer.md) 继续执行。 [`Chapter02`](../chapter02_tensor_ops/intro.md) 使用独立的 CPU Float32 教学张量学习形状、存储和视图。
