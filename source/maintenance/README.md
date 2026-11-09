# Maintenance 维护规则

这里仅保存跨章节、长期有效的治理规则。普通学习者不需要阅读；贡献流程、测试命令和提交要求以仓库根目录的 [`CONTRIBUTING.md`](../../CONTRIBUTING.md) 为准。

## 权威文件

| 文件 | 负责的问题 |
| --- | --- |
| [tutorial_positioning.md](./tutorial_positioning.md) | 教程定位、范围、章节职责和结构调整条件 |
| [tutorial_writing_guidelines.md](./tutorial_writing_guidelines.md) | 导读、正文、代码 README、算法镜像、框架对照和图片规则 |
| [code_quality_guidelines.md](./code_quality_guidelines.md) | C++/CUDA 质量要求、审核顺序、证据和问题等级 |

阶段计划、迁移过程、单章审核、终端输出和当前优先级不进入本目录。此类内容保存在被 Git 忽略的交接文档或本地归档；其中形成跨章节稳定结论后，再提炼到上表对应文件。

## 使用顺序

修改教程范围或章节分工时先看定位；修改学习者页面或教学资产时看写作规则；新增模型、执行策略或修改实现、测试、公共接口时看代码质量与准入规则。`CONTRIBUTING.md` 只摘要贡献者必须执行的要求，不复制完整维护规则。

## 维护门禁

维护文档不复制各章运行命令。全局修改按范围执行下面的检查；真实模型、CUDA、网络 transport 和性能实验仍使用对应模块入口单独验证。

| 目的 | 命令 |
| --- | --- |
| 修改范围与空白 | `git status --short`、`git diff --check`、`git diff --stat` |
| 文档结构与站点 | `npm run docs:check-structure`、`npm run docs:check-style`、`npm run docs:check-reference`、`npm run docs:check-links` |
| 默认无外部依赖回归 | `npm run test:regression` |
| 测试 owner 与发布验收 | `npm run test:check-contracts`、`npm run test:release-contracts` |

新增维护文件前先判断内容能否合入现有四类规则。没有新的长期职责时，不为一次任务、一个章节或一轮审核创建永久文件。
