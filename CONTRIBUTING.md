# 贡献指南

这份文档说明如何修改、验证和提交 `cpp-llm-engine`。学习和运行教程请先看 [USAGE.md](USAGE.md)。

## 可以贡献什么

| 类型 | 适合的修改 | 至少需要验证 |
| --- | --- | --- |
| 教程正文 | 修正文意、补数据流、公式、图表或代码映射 | 文档结构、风格和链接检查 |
| CPU 代码 | 修复实现、测试、构建脚本或章节示例 | 对应章节测试和默认回归 |
| CUDA 代码 | 新增或修复 kernel、backend、设备资源和数值对照 | CPU reference、CUDA CTest 和真实设备执行 |
| Runtime / Serving | 修改 Request、Scheduler、KV、streaming、metrics 或 transport | 单测、并发/生命周期测试和默认回归 |
| Benchmark / Profiling | 新增 workload、指标、结果或 A/B 实验 | 固定环境、可复现命令和机器可读结果 |
| 框架对照 | 补充成熟框架的实现坐标和工程差异 | 对应源码或官方资料，明确版本和核验日期 |

不要在一个补丁中同时重排章节、替换运行时架构并引入未经验证的优化。大范围改动应先拆出契约、reference 实现和验证入口。

## 目录职责

同一事实只在一个位置维护，其他位置使用链接：

| 目录或文件 | 权威内容 |
| --- | --- |
| `source/` | 中文教程正文和章节导读 |
| `code/<chapter>/` | 章节实现、测试、示例和资产说明 |
| `exercises/` | 练习题、starter、答案边界和题目验证 |
| `source/reference/` | 术语、公式、框架对照和跨章节资产表 |
| `code/*/results/` | 可复现的实验结果，不存放解释性正文 |
| `USAGE.md` | 学习路线、运行入口和验证层级 |
| `CONTRIBUTING.md` | 修改、测试和提交规则 |
| `source/maintenance/` | 面向维护者的定位、模板和长期治理规则 |

普通文字修正不要求通读维护材料。只有调整教程定位、章节边界、公共模板或全局测试体系时，才需要阅读[教程定位](source/maintenance/tutorial_positioning.md)和[维护文档入口](source/maintenance/README.md)。

## 工作流程

1. 检查工作区，确认修改不会覆盖他人尚未提交的内容。
2. 找到事实的权威位置，避免在 README、指南和章节正文中复制同一段状态说明。
3. 做能够解决问题的最小修改，同时更新直接关联的测试或文档。
4. 先运行局部验证，再运行对应类别的全局检查。
5. 用 `git diff --check` 检查空白字符，用 `git diff` 检查修改范围。
6. 按职责拆分提交，不把模型、构建产物或无关格式化混入提交。

## 代码要求

实现要求、审核顺序和问题等级见[代码质量与审核规则](source/maintenance/code_quality_guidelines.md)。

- 默认使用 C++17，遵循现有 CMake target 和命名方式。
- CPU 默认路径不得依赖 GGUF 模型、网络下载、Python、CUDA 或外部 BLAS。
- 优化实现必须保留可比较的 reference 或 golden，不能让实现本身同时充当验收答案。
- 新增公共接口时，先写清输入 shape、数据类型、所有权、错误条件和输出语义。
- 教学代码文件头使用中文，至少说明用途、输入输出和验证入口。
- CUDA 修改必须区分“编译通过”“数值正确”和“真实设备执行通过”。

## 文档要求

- 中文正文优先，文件名、命令、API 和框架专有名词保留原文。
- 先说明学习问题和数据流，再给实现细节；避免空泛的“这一层”“这一页”和连续否定式边界说明。
- 章节正文不绑定阶段任务编号，学习计划可以独立调整。
- 面向维护者的状态、计划和写作规则只放在 `source/maintenance/`。
- 图表前先给必要说明，图表后不追加新的关键信息。
- 框架对照必须说明比较的是算法骨架、教学实现还是生产工程，不能用名称相似代替源码证据。

## 验证矩阵

只运行与修改类型匹配的检查不代表可以跳过明显受影响的回归。

Pull request 会自动运行 `.github/workflows/ci.yml`。CI 只覆盖无模型、无网络服务和无 GPU 的默认路径；涉及真实模型、CUDA、transport 或性能实验时，贡献者仍需按下表提供对应的本地验证结果。

| 修改范围 | 必须运行 |
| --- | --- |
| Markdown、SVG、导航或链接 | `npm run docs:check-structure`、`npm run docs:check-links`、`npm run docs:check-style`、`npm run docs:check-reference` |
| 章节 C++ 代码 | 对应章节 `run.sh`、`npm run code:check-headers`、`npm run test:regression` |
| Foundation 练习 | `bash exercises/foundation/run_all.sh`、`npm run test:regression` |
| Engine / ServiceCore | `bash code/engine/scripts/run.sh` 或相关 CTest、`npm run test:regression` |
| CUDA backend 或 kernel | `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh`，并记录 GPU、驱动和 CUDA 版本 |
| 真实 CUDA Model、Runtime 或 Serving | 完成 [USAGE GPU 纵向验收](USAGE.md#gpu-纵向验收)中的三项命令，并记录模型、GPU、驱动和 CUDA 版本 |
| 多模态 | `bash code/multimodal/scripts/run.sh`，涉及真实模型时补对应 golden |
| 测试注册、CTest label 或跨章公共契约 | `npm run test:check-contracts`、`npm run test:release-contracts` |
| 准备发布或大范围重构 | 文档检查、`npm run test:regression`、`npm run test:release-contracts` |
| 站点构建 | `npm run docs:check-links` 已包含完整 VitePress 构建；需要单独查看构建输出时使用 `npm run docs:build` |

文档修改的最小检查集：

```bash
npm run docs:check-structure
npm run docs:check-links
npm run docs:check-style
npm run docs:check-reference
git diff --check
```

代码修改通常还需要：

```bash
npm run code:check-headers
npm run test:regression
```

改动测试登记、公共契约、章节边界或发布前，再运行：

```bash
npm run test:release-contracts
```

`test:regression` 验证默认开发路径；`test:release-contracts` 使用全新 Release 构建目录按 owner、label 和 target 重新验证发布契约。两者都不替代真实 GGUF、CUDA 设备、HTTP transport 或性能实验。

## 实验与结果

- 正确性结果记录输入、shape、dtype、容差、reference 和比较方法。
- 性能结果记录硬件、编译器、构建类型、线程数、模型、context、batch、warmup 和重复次数。
- 原始结果放入对应 `results/`，长期索引更新到[实验资产表](source/reference/experiment_assets.md)。
- 不提交只在单次终端输出中出现、无法复现或缺少环境信息的性能数字。
- 性能提高不能证明实现正确；先通过 correctness，再做 benchmark 和 profiling。

## 不要提交

- GGUF、Safetensors 等模型权重或下载缓存。
- `build/`、CMake 临时文件、二进制和编辑器缓存。
- 只有编译成功、没有真实设备运行证据的 GPU 完成声明。
- 与当前修改无关的大范围格式化或文件重排。
- 没有来源、版本或核验日期的成熟框架实现结论。

## 提交前检查

- 修改位于正确的职责目录，没有重复维护同一事实。
- 新功能有测试，性能修改有固定 workload，文档修改有可访问链接。
- 默认 CPU 路径仍可在无模型、无网络、无 GPU 环境下运行。
- `git diff --check` 没有报错，提交中不含构建产物和大模型文件。
- 提交信息描述实际改动，例如 `docs: clarify usage and contribution boundaries`。

## English Summary

Contributions are welcome for documentation, CPU/CUDA implementations, runtime and serving, reproducible experiments, and framework comparisons. Keep each change small, preserve the dependency-free CPU regression, add an independent correctness reference, and run the validation commands required by the matrix above.
