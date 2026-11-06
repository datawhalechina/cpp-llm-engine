# Maintenance 维护文档

这里保存教程维护者需要长期执行的定位、模板和治理规则。普通学习者不需要阅读；常规贡献流程以仓库根目录的 `CONTRIBUTING.md` 为准。

## 内容范围

- [tutorial_positioning.md](./tutorial_positioning.md)：教程定位、职责边界和结构变更基准
- [tutorial_maintenance.md](./tutorial_maintenance.md)：教程维护文档
- [tutorial_writing_guidelines.md](./tutorial_writing_guidelines.md)：教程编写规范
- [chapter_template.md](./chapter_template.md)：章节模板
- [body_page_template.md](./body_page_template.md)：正文页面层级与框架对照页模板
- [algorithm_mirror_template.md](./algorithm_mirror_template.md)：Python 风格算法镜像、C++ 映射与验证规则
- [framework_comparison_card_template.md](./framework_comparison_card_template.md)：章级成熟框架对照卡与生态项目卡分工
- [diagram_review_guidelines.md](./diagram_review_guidelines.md)：图片语义、连线和渲染审核规则
- [chapter00_04_evidence_matrix.md](./chapter00_04_evidence_matrix.md)：Chapter00–04 代码、测试、正文与图片证据矩阵
- [chapter02_nd_tensor_design.md](./chapter02_nd_tensor_design.md)：Chapter02 N 维 Tensor 重构规格
- [chapter04_tensor_handoff_design.md](./chapter04_tensor_handoff_design.md)：Chapter02 到 Chapter04 的 Attention Tensor 交接规格
- [tensor_type_migration.md](./tensor_type_migration.md)：跨章 Tensor 类型、公共 View、adapter 收口和逐步验证路线
- [chapter06_10_restructure_plan.md](./chapter06_10_restructure_plan.md)：Chapter06–10 目标小节、旧资产迁移和分阶段验收矩阵
- [chapter01_10_content_design.md](./chapter01_10_content_design.md)：Chapter01–10 主图、正文小节和章间交接矩阵
- [chapter00_10_teaching_audit.md](./chapter00_10_teaching_audit.md)：Chapter00–10 教学职责、成熟度和发布门槛审核
- [chapter04_06_test_ownership_audit.md](./chapter04_06_test_ownership_audit.md)：Chapter04–08 测试归属、有效证据和迁移结论
- [summary_template.md](./summary_template.md)：章节收口模板

## 怎么用

- 修改教程结构、写新章节或调整实现范围时，先看定位与职责边界
- 只学习主线内容时，可以跳过这里
- 这些页面不重复学习路线、运行命令和提交检查表

## 职责关系

- 学习路线和运行入口：`USAGE.md`
- 修改、验证和提交：`CONTRIBUTING.md`
- 教程定位和长期治理：当前目录

## 维护门禁

维护文档不复制各章运行命令。只在调整章节结构、公共契约、站点映射或发布状态时使用下列全局命令。

| 目的 | 命令 |
| --- | --- |
| 审核修改范围 | `git status --short`、`git diff --check`、`git diff --stat` |
| 文档治理 | `npm run docs:check-structure`、`npm run docs:check-style`、`npm run docs:check-reference`、`npm run docs:check-links` |
| 默认无外部依赖回归 | `npm run test:regression` |
| 测试资产和 owner 审核 | `npm run test:check-assets`、`npm run test:check-contracts` |
| 发布前契约验收 | `npm run test:release-contracts` |

`docs:check-links` 会构建 VitePress 并检查生成页面；`test:regression` 使用各模块默认脚本；`test:release-contracts` 使用全新 Release 构建目录重新编译并按 CTest 契约执行。真实模型、CUDA、网络 transport 和 benchmark 仍需按修改范围单独验证。
