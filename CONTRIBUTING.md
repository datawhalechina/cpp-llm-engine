# 贡献指南

这份指南面向准备参与教程建设的学习者和开发者。学习或运行项目请先看 [USAGE.md](USAGE.md)；这里集中说明怎样选择任务、完成修改并提供可验证的结果。

## 第一次贡献

建议从能够独立验证的小修改开始：修正一处解释或链接、补充一个失败路径测试、改善一条报错信息，或为现有实现增加可复现示例。第一次提交不建议同时重排章节、替换架构和引入性能优化。

一次贡献按以下顺序完成：

1. 用 `git status --short` 确认工作区状态。
2. 找到问题对应的正文、代码或测试，只修改解决问题所需的内容。
3. 先运行局部测试，再从验证表中选择受影响的全局检查。
4. 用 `git diff --check` 和 `git diff` 检查空白、构建产物和修改范围。
5. 在提交或 Pull Request 中说明问题、改动、验证命令和结果。

## 可以贡献什么

| 方向 | 适合的修改 | 需要提供的证据 |
| --- | --- | --- |
| 教程正文 | 修正文意，补数据流、公式、图表或代码跳转 | 文档和链接检查 |
| CPU 代码 | 修复实现、测试、构建脚本或章节示例 | 对应章节测试和默认回归 |
| CUDA 代码 | kernel、backend、设备资源或数值对照 | CPU reference、CUDA CTest 和真实设备结果 |
| Runtime / Serving | Request、Scheduler、KV、streaming、metrics 或 transport | 单测、生命周期测试和默认回归 |
| 性能实验 | workload、指标、结果或 A/B 实验 | 固定环境、复现命令和机器可读结果 |
| 框架对照 | 成熟框架的实现坐标和工程差异 | 官方源码或资料、版本和核验日期 |

教程正文位于 `source/`，章节实现和测试位于 `code/<chapter>/`，练习位于 `exercises/`，跨章术语和资产索引位于 `source/reference/`。同一事实只在一个位置维护，其他位置使用链接。

## 基本要求

- 默认使用 C++17，并沿用现有 CMake target、命名和文件头风格。
- 默认 CPU 回归不得依赖模型下载、网络、Python、CUDA 或外部 BLAS。
- 优化实现保留独立的 reference 或 golden；性能提高不能代替正确性验证。
- 公共接口需要明确 shape、dtype、所有权、错误条件和输出语义。
- CUDA 结果需要区分编译通过、数值正确和真实设备执行通过。
- 中文教程先说明学习问题和数据流，再连接实现代码。

调整教程定位、章节边界或公共模板时，再阅读[维护规则](source/maintenance/README.md)。详细写作和实现要求分别见[教程写作规则](source/maintenance/tutorial_writing_guidelines.md)与[代码质量规则](source/maintenance/code_quality_guidelines.md)。

## 选择验证命令

Pull Request 中的 CI 覆盖无模型、无网络服务和无 GPU 的默认路径。真实模型、CUDA、transport 和性能实验需要附上本地验证结果。

| 修改范围 | 必须运行 |
| --- | --- |
| Markdown、SVG、导航或链接 | `npm run docs:check-structure`、`npm run docs:check-style`、`npm run docs:check-reference`、`npm run docs:check-links` |
| 章节 C++ 代码 | 对应章节 `run.sh`、`npm run code:check-headers`、`npm run test:regression` |
| Foundation 练习 | `bash exercises/foundation/run_all.sh`、`npm run test:regression` |
| 引擎集成 / `ServiceCore` | 对应 CTest 或 `bash code/engine/scripts/run.sh`、`npm run test:regression` |
| CUDA backend 或 kernel | `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh`，并记录 GPU、驱动和 CUDA 版本 |
| 真实 CUDA Runtime / Serving | [GPU 纵向验收](USAGE.md#gpu-纵向验收)中的三项命令和环境信息 |
| 多模态 | `bash code/multimodal/scripts/run.sh`；真实模型结果另附 golden |
| 测试注册或公共接口约定 | `npm run test:check-contracts`、`npm run test:release-contracts` |
| 发布或大范围重构 | 全部文档检查、`npm run test:regression`、`npm run test:release-contracts` |

`test:regression` 验证默认开发路径；`test:release-contracts` 在全新 Release 构建目录中验证测试登记和发布验收规则。二者都不替代真实 GGUF、CUDA 设备、HTTP transport 或性能实验。

## 结果与提交

- 正确性结果记录输入、shape、dtype、容差、reference 和比较方法。
- 性能结果记录硬件、编译器、构建类型、线程数、模型、context、batch、warmup 和重复次数；原始结果放入对应 `results/`，长期入口登记到[实验资产表](source/reference/experiment_assets.md)。
- 不提交模型权重、下载缓存、`build/`、CMake 临时文件、二进制或编辑器缓存。
- 不提交缺少真实设备证据的 GPU 完成声明，或缺少来源和版本的框架结论。
- 按职责拆分提交，提交信息描述实际改动，例如 `docs: clarify contribution workflow`。

提交前确认：修改位于正确目录，新功能有测试，文档链接可访问，默认 CPU 路径仍可运行，`git diff --check` 无报错。
