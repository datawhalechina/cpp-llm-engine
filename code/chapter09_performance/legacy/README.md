# 历史性能资产

本目录保留 Chapter06-10 重构前的 Benchmark 与 Profiling 实现，供结果追溯和迁移验证。当前学习、构建和回归入口统一使用 `code/chapter09_performance`，历史目录不再参与默认回归。

旧结果中只有包含逐轮样本的文件可以通过 `scripts/import_legacy_evidence.mjs` 转换为当前证据格式；只有聚合值的结果保存在 `results/legacy/`，不能作为新的可重复基线。
