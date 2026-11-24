# 仓库脚本分级

根 `scripts/` 按用途保存跨章节工具。单章构建与实验继续放在 `code/<module>/scripts/`，避免根目录成为所有命令的集合。

个人使用的一次性扫描、临时迁移脚本和审核笔记放在 `scripts/local/`。该目录由 `.gitignore` 忽略，不进入脚本清单、CI 或发布验收。脚本只有在规则客观、输入输出稳定且需要团队共同执行时，才迁入下面的共享分级。

| 目录 | 内容 |
| --- | --- |
| `checks/` | 快速、确定性的原子检查 |
| `workflows/` | 学习路径和可选真实资产流程 |
| `gates/` | 合并前回归、故障注入和质量门禁 |
| `release/` | 阶段验收和 Release 构建验证 |
| `maintenance/` | 文档迁移与同步工具 |
| `local/` | 个人脚本与笔记，Git 忽略 |

## 分级规则

| 分级 | 运行特征 | 使用场景 |
| --- | --- | --- |
| L0 原子检查 | 快速、确定性、默认不下载模型或使用 GPU | 本地提交前检查、CI 基础门禁 |
| L1 学习工作流 | 编排一段学习路径，可能编译多个组件或读取可选模型 | 学习者按主题运行 |
| L2 集成门禁 | 聚合多个检查或注入失败，运行时间高于 L0 | 分支合并前回归 |
| L3 发布验收 | 覆盖全模块、真实资产或发布构建 | 发布候选和阶段验收 |
| 维护工具 | 修改或迁移仓库内容，不属于测试层级 | 维护者显式运行 |

L0–L3 描述验证范围和成本，不表示实现质量高低。调用方优先使用 `package.json` 中的稳定入口；脚本内部仍可组合较低层级命令。

## 脚本清单

<!-- script-catalog:start -->
| 脚本 | 分级 | 主要职责 | 默认外部依赖 |
| --- | --- | --- | --- |
| `checks/check_code_headers.mjs` | L0 | 检查源码文件头 | Node.js |
| `checks/check_docs_links.mjs` | L0 | 构建文档并检查本地与仓库链接 | Node.js、npm |
| `checks/check_docs_structure.mjs` | L0 | 检查章节、侧栏和代码 README 结构 | Node.js |
| `checks/check_foundation_metadata.sh` | L0 | 检查 Foundation 题目元数据 | Bash |
| `checks/check_foundation_starters.sh` | L0 | 检查 Foundation starter 边界 | Bash、CMake、C++17 |
| `checks/check_line_endings.mjs` | L0 | 检查 Shell 脚本换行符 | Node.js |
| `checks/check_model_capabilities.mjs` | L0 | 检查模型能力矩阵与测试证据 | Node.js |
| `checks/check_multimodal_target_boundaries.mjs` | L0 | 检查多模态 CMake target 边界 | Node.js |
| `checks/check_reference_assets.mjs` | L0 | 检查实验资产索引 | Node.js |
| `checks/check_script_catalog.mjs` | L0 | 检查本清单覆盖所有共享脚本 | Node.js |
| `checks/check_test_assets.mjs` | L0 | 检查测试资产登记 | Node.js |
| `checks/check_test_contracts.mjs` | L0 | 检查测试 owner、状态和迁移债务 | Node.js |
| `workflows/check_foundation_answer.sh` | L1 | 编译并运行单题答案 | Bash、C++17 |
| `workflows/run_cpu_service_learning.sh` | L1 | 编排 CPU 服务学习路径 | Bash、CMake、C++17 |
| `workflows/run_generation_runtime_learning.sh` | L1 | 编排生成运行时学习路径 | Bash、CMake、C++17 |
| `workflows/run_real_decoder_learning.sh` | L1 | 编排真实 Decoder 学习路径 | Bash、CMake、可选 GGUF |
| `workflows/run_smolvlm_cpu_baseline.sh` | L1 | 运行 SmolVLM CPU 基线 | Bash、CMake、SmolVLM 资产 |
| `workflows/run_tensor_operator_learning.sh` | L1 | 编排 Tensor 与算子学习路径 | Bash、CMake、C++17 |
| `gates/run_cpp_quality_gate.sh` | L2 | 聚合 C++ 静态与构建质量检查 | Bash、CMake、C++17 |
| `gates/run_fault_injection_gate.sh` | L2 | 聚合缓存、调度与服务故障注入 | Bash、CMake、C++17 |
| `gates/run_regression_baseline.sh` | L2 | 聚合默认无模型、无 GPU 回归 | Bash、Node.js、CMake、C++17 |
| `release/run_cpu_v1_acceptance.sh` | L3 | 运行 CPU 阶段验收 | Bash、CMake、真实模型资产 |
| `release/run_release_contracts.sh` | L3 | 以 Release 构建验证关键模块契约 | Bash、CMake、C++17 |
| `maintenance/migrate_chapter_to_source.sh` | 维护工具 | 迁移章节文档到统一源目录 | Bash |
| `maintenance/sync_english_sections.py` | 维护工具 | 同步英文摘要区块 | Python 3 |
| `maintenance/sync_shared_sections.py` | 维护工具 | 同步共享文档区块 | Python 3 |
<!-- script-catalog:end -->

## 目录约束

新增脚本先判断作用范围：单章脚本进入 `code/<module>/scripts/`，跨章节脚本进入上表对应目录，个人原型进入 `local/`。共享脚本移动后需要同时更新 `package.json`、内部调用和学习文档，并通过清单检查；根目录不保留兼容包装器。

运行清单检查：

```bash
npm run test:check-script-catalog
```
