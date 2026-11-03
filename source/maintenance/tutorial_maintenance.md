# Tutorial Maintenance 教程维护文档

本文档面向负责教程结构和长期治理的维护者，定义跨章节的稳定维护规范。学习路线和运行入口以仓库根目录的 `USAGE.md` 为准，常规修改与提交要求以 `CONTRIBUTING.md` 为准。

下面的“当前交接快照”是早期历史记录，不作为当前实现状态或任务优先级的权威来源；后续维护应将稳定规则保留在本文，将阶段状态留在版本历史和项目管理工具中。

## 0. 当前交接快照

更新时间：2026-08-04。

### 0.1 项目目标

教程定位、能力范围、模块职责和结构变更条件统一维护在 [`tutorial_positioning.md`](./tutorial_positioning.md)。本文件记录仓库维护方法和历史状态，不再维护第二份定位定义。

主线目标保持三件事：

- 先跑通真实 baseline，再逐章实现最小推理引擎部件
- 每章都能回答“解决什么、代码在哪、怎么验证”
- 把 `vLLM`、`nano-vLLM`、`SGLang`、`TensorRT-LLM`、`llama.cpp` 放在工程参照位置，而不是替代教程主线

### 0.2 当前架构

当前仓库按三层组织：

- `source/` 是中文教程正文源头，后续正文修改优先改这里
- `docs/` 是 VitePress 站点入口和在线映射层，大部分章节目录通过软链接指向 `source/`
- `code/` 是每章最小实现源码，文档里的“源码直达”应指向这里的具体文件
- `code/` 下的源码、头文件、测试和示例使用统一文件头；当前检查器先覆盖已完成的 `core` 和 `engine`，各章节补齐后再扩大检查范围

站点相关结构：

- `docs/.vitepress/config.mts` 维护站点配置、sidebar、clean URL 策略和 `_code` 静态资源入口
- `docs/.vitepress/public/_code -> ../../../code` 用于把真实源码发布成站点静态资源
- `source/_code -> ../code` 用于让直接阅读 `source/*.md` 时也能点到真实源码文件
- `docs/chapterXX_* -> ../source/chapterXX_*` 保持站点章节和源码正文共用一份内容

### 0.3 已完成内容

本轮已经完成的重点：

- 明确中文教程优先，英文版整体计划先挂起
- 将教程定位整理为“推理引擎原理 + 最小实现 + 验证”
- 在 README / guide / reference / maintenance 中补充推理服务、推理引擎、推理框架之间的关系说明
- 为 `chapter04`、`chapter06`、`chapter05` 增加与 `vLLM` / `nano-vLLM` 的工程坐标桥接
- 在 `guides` 维护可选 `vLLM` serving baseline，不加入 Chapter00 必读路径
- 为 `chapter03` 增加 backend / executor / runtime 视角桥接
- 为 `chapter06` 增加 request lifecycle、continuous batching、block KV cache 相关小节
- 为 `chapter05` 增加量化在 serving runtime 中位置的小节
- 新增 `chapter08_profiling`，覆盖 profiling、latency breakdown、memory bandwidth、CUDA Graph、A/B 实验和优化闭环
- 收敛 `chapter00` 到 `chapter08` 的 intro 风格，统一章节导读、源码顺序和验证入口
- 将同章小节链接改为源码可点击的 `./xxx.md`
- 将跨章文档链接改为源码可点击的 `../chapter_xxx/xxx.md`
- 将正文中的“源码直达”从 `/code/chapterXX` 或 `/code/chapterXX#anchor` 改为具体源码文件链接
- 完成 `source/` 全仓库文档类绝对路径审计，清理旧 `/chapter...`、`/foundation...`、`/reference...`、`/maintenance...`、`/guides...`、`/appendix...` 链接
- 完成 `source/` 全仓库旧 `/code/chapterXX...` 链接审计，正文不再依赖章节代码索引页作为源码直达入口
- 重新定义并压缩 `docs/code/chapterXX.md`：代码索引页只做站内源码导航补充，不再承担正文源码直达主入口
- 整理 `tutorial_maintenance.md` 历史编号和重复段落，重新归并站点构建、同步规则、代码映射、写作流程和验收清单
- 完成 `chapter01` 到 `chapter07` 正文入口质量抽查：压短 intro 的源码直达，正文页只保留与本页任务直接相关的 1 到 3 个源码链接
- 完成 `chapter01` 到 `chapter04` 正文小节层级合并：在不删减内容的前提下，将正文页中过细的二级 / 三级标题收敛为更少的主层级
- 完成 `chapter06` 正文小节层级合并：将 runtime 主线正文页的编号标题和三级标题收敛为 `核心设计 / 核心基线 / 测试设计 / 验证与衔接` 等主层级
- 完成 `chapter05`、`chapter07` 正文小节层级合并：将量化和 benchmark 主线正文页的厚模板收敛为 `核心设计 / 验证设计 / 指标设计 / 判断流程 / 验证与排查` 等主层级
- 完成 `chapter03` 到 `chapter07` 正文页入口合并：将独立 `代码直达` 标题并入 `阅读入口` 的 `源码直达` 项，减少正文开头停顿
- 完成 `chapter00` 到 `chapter08` summary 专项收敛：统一小结标题顺序为 `本章解决了什么 / 为什么重要 / 下一步 / 复盘路径`，清理编号标题和过厚的回看源码层级
- 完成 `chapter06`、`chapter05`、`chapter07` 的阅读入口瘦身：intro 不再重复设置第二套“代码直达”，正文页不再在“阅读入口”和“代码直达”里重复列同一批源码
- 完成 `chapter01`、`chapter02`、`chapter03` 的工程桥接层补强：三章已分别收口为 `权重进入 runtime`、`Tensor view / shared storage`、`MatMul backend` 主线，不再只是组件说明
- 完成 `chapter01` 到 `chapter06` 顶层命名统一：章节标题和导航已切换到系统层次命名，分别为 `Model Weights and Tensor Objects`、`Tensor Views and Shared Storage`、`MatMul and Compute Backends`、`Decoder Structure and Attention`、`Generation Runtime and Serving Basics`
- 完成 `chapter04_attention` 结构收口：`intro`、`attention`、`decoder layer`、`structure bridge`、`summary` 已明确收束为 `chapter01` 到 `chapter03` 的第一次结构汇合点，并作为 `chapter06_generation` runtime 主线的结构前置章
- 完成 `chapter08_profiling` 第二轮质量评审：补充源码锚点、延迟拆解表、profiling 记录格式、A/B 报告模板和优化保留边界
- 完成 `chapter07_benchmark` 最小阶段计时补强：baseline demo 现可输出 `input / prefill / decode / sampling / output` 的真实阶段时间、占比和吞吐
- 完成 `chapter07_benchmark` 最小事件级 trace 补强：baseline demo 现可输出事件名、起始时间、持续时间和注释，供 `chapter08_profiling` 引用
- 完成 `chapter07_benchmark` 最小 hotspot 汇总补强：baseline demo 现可输出函数级热点排序，供 `chapter08_profiling` 解释“阶段 -> 事件 -> 热点”三层证据链
- 完成 `chapter07_benchmark` 最小 JSON trace 样例补强：baseline demo 现可输出 `phases / trace_events / hotspots` 结构化结果，供 `chapter08_profiling` 解释“文本阅读”与“工具消费”之间的过渡
- 完成 `chapter07_benchmark` 简化时间轴视图补强：baseline demo 现可输出最小 Chrome-trace 风格结果，供 `chapter08_profiling` 解释“结构化结果”与“时间轴视图”之间的过渡
- 完成 `chapter07_benchmark` 最小 `tid / sync point` 语义补强：简化时间轴视图现可区分 `host-thread`、`compute-stream`，并标出一个最小等待点
- 完成 `chapter07_benchmark` 最小多请求时间轴样例：简化时间轴现可展示 `req-A / req-B` 重叠、stream 分离和一次 batch merge 等待点
- 增强 `scripts/check_docs_links.mjs`：构建前清理旧 dist，并检查生成 HTML 中的内部相对链接

### 0.4 未完成内容

当前仍未完成或需要后续继续推进：

- `docs/README.md`、`docs/guide.md`、`docs/index.md` 属于站点入口页，目前仍可保留站点绝对路由；如果要求这些页面也能在源码预览里逐项点击，需要单独制定入口页链接策略
- `docs/code/chapterXX.md` 已改为站内源码导航补充页，后续还可以继续压缩长路径展示，但不再作为正文源码直达主入口
- 英文版教程计划已挂起，英文入口和旧英文页面不作为当前质量优化重点
- `chapter08_profiling` 已补齐工程记录模板，并已接入 `chapter07_benchmark_demo` 的真实 baseline 输出；当前阶段计时已经细化到 `input / prefill / decode / sampling / output`，并补入最小事件级 trace、hotspot 汇总、JSON 样例、简化时间轴视图、最小 `tid / sync point` 语义，以及多请求时间轴样例；下一步如继续扩展，应停止在这里或只做极小文字收口，不再扩成真正工具实现
- 源码直达目前使用 HTML `<a href="../_code/..."><code>...</code></a>`，主要是为了避免 VitePress 把 `.cpp` / `.h` 改写成 `.cpp.html`；后续如果换文档框架，需要重新验证
- 代码层本身没有在本轮做功能实现变更，主要变更是教程结构、链接、导读和 reference

### 0.5 当前分支状态

当前分支：`main`。

当前工作区不是干净状态，存在大量未提交文档修改和新增文件。继续工作前必须先看：

```bash
git status --short
git diff --stat
```

不要执行 `git reset --hard`、`git checkout --` 等会丢弃改动的命令，除非明确确认这些改动都可以删除。

### 0.6 关键文件路径

正文与入口：

- `README.md`
- `docs/README.md`
- `docs/index.md`
- `docs/guide.md`
- `source/foundation/intro.md`
- `source/chapter00_quickstart/intro.md`
- `source/chapter01_tensor_loader/intro.md`
- `source/chapter02_tensor_ops/intro.md`
- `source/chapter03_matmul/intro.md`
- `source/chapter04_attention/intro.md`
- `source/chapter04_attention/07_framework_comparison.md`
- `source/chapter06_generation/intro.md`
- `source/chapter05_quantization/intro.md`
- `source/chapter07_benchmark/intro.md`
- `source/chapter08_profiling/intro.md`

维护与参考：

- `source/maintenance/tutorial_maintenance.md`
- `source/maintenance/tutorial_positioning.md`
- `source/maintenance/tutorial_writing_guidelines.md`
- `source/reference/README.md`
- `source/reference/framework_comparison.md`
- `source/reference/inference_engine_systems.md`
- `source/reference/runtime_systems.md`
- `source/reference/input_pipeline.md`
- `source/reference/service_basics.md`
- `source/reference/profiling.md`

构建与链接：

- `docs/.vitepress/config.mts`
- `scripts/check_docs_links.mjs`
- `source/_code`
- `docs/.vitepress/public/_code`

源码：

- `code/chapter00_quickstart/`
- `code/chapter01_tensor_loader/`
- `code/chapter02_tensor_ops/`
- `code/chapter03_matmul/`
- `code/chapter04_attention/`
- `code/chapter06_generation/`
- `code/chapter05_quantization/`
- `code/chapter07_benchmark/`

### 0.7 重要设计决策

- `source/` 是正文唯一源头，不从 `docs/` 复制第二份正文
- `docs/` 只承担 VitePress 站点入口、配置、英文入口和在线映射
- 正文同章链接使用 `./xxx.md`
- 正文跨章链接使用 `../chapter_xxx/xxx.md`
- 正文源码直达使用具体源码文件，不再只跳章节代码文件夹
- 源码直达使用 HTML `<a>`，避免 VitePress 把源码文件后缀改写成文档页面后缀
- `cleanUrls` 保持 `false`，构建后链接显式落到 `.html`
- `preserveSymlinks` 保持开启，配合 `docs/` 到 `source/` 的软链接结构
- `_code` 静态入口同时服务源码阅读和站点构建
- `reference` 保留完整展开，正文只放 1 到 3 句短桥接并互相挂链接
- `chapter01` 到 `chapter06` 的顶层命名统一按“系统层次命名”维护，不再使用单纯动作式英文标题；当前标准名称见对应 `intro.md` 与 `docs/.vitepress/config.mts`

### 0.8 已知问题

- `docs/code/chapterXX.md` 已经改成轻量源码导航页，但如果后续觉得仍然偏厚，可以继续压缩展示字段
- 如果直接在某些 Markdown 预览器中打开 HTML `<a>` 链接，显示样式可能和普通 Markdown 链接略有差异，但跳转路径是正确的
- `docs/.vitepress/public/_code` 会把真实 `code/` 发布为站点静态资源，需要注意不要把构建产物或大模型文件放进 `code/`
- 当前链接检查主要验证构建和生成 HTML 内部链接，不替代人工阅读质量评审
- 当前工作区依然是脏状态，且存在大量历史文档修改与新增文件；继续工作时必须只在目标文件上做最小改动，避免误覆盖前序成果

### 0.9 下一步计划

建议后续按优先级推进：

1. 优先审查并精修 `chapter06_generation`：确认它已经稳定承担“运行时主线章”的职责，而不是再次滑回组件说明；重点看 `intro / request lifecycle / KV cache / vLLM bridge`
2. 检查 `chapter00` 到 `chapter08` 的兼容页和旧链接，确认章节收口已经统一迁移到 intro
3. 按 `chapter01` 到 `chapter04` 的方式继续审查 `chapter05`、`chapter07` 正文小节层级，优先合并重复标题，不删减正文信息
4. `chapter08_profiling` 的最小工程证据链已基本收口；后续优先转向其他章节，而不是继续扩成 profiler 工具
5. 做一次全教程标题层级审计，确认正文、intro、summary、sidebar 四类入口的命名规则已经一致
6. 如果需要源码预览友好的站点入口页，再单独处理 `docs/README.md`、`docs/guide.md`、`docs/index.md`
7. 每轮修改后固定运行 `npm run docs:check-links` 和 `git diff --check`

## 1. 维护目标

这套教程的目标不是追求“内容最多”，而是保持下面几件事长期成立：

- 主线清晰，读者可以按章节顺序学下去
- 每章都有明确产物和验证方式
- 文档描述与代码实现保持一致
- 术语、风格和命名尽量统一
- 章节之间的桥接稳定，读者能顺着读完主线
- 进阶内容固定挂在尾部，不打散主线

## 2. 当前阶段

当前阶段已经进入 V2，不再是“最小可用实现”阶段。

V2 的重点是：

- 让主线更顺
- 让章节边界更清楚
- 让框架对照升级为场景决策树
- 让进阶内容有固定落点
- 让不同层次读者都能找到自己的入口

## 3. 工程原则

这套项目在工程上希望长期保持下面这些原则：

### 3.1 中文主线优先

- 正文、章节说明和维护说明以中文为主
- 必要英文只保留术语、代码标识和少量入口

### 3.2 主线优先于附录

- 先保证 `foundation`、`chapter00` 到 `chapter07` 的主线完整
- `guides`、`reference` 和 `maintenance` 各司其职，不做第二主线

### 3.3 背景和中间推导优先于结论堆砌

- 每章先讲背景，再讲推导，再讲实现和验证
- 初学者最缺的是“为什么这样走到这里”

### 3.4 章节职责必须单一

- `foundation` 负责进入主线前的准备
- `chapter00` 负责第一次跑通
- `chapter01` 到 `chapter07` 负责主线推进

### 3.5 概念和实现同步出现

- 只有在需要时才引入新概念
- 不提前堆过多背景，避免主线失焦

### 3.6 正确性和性能分开

- 单测和 e2e 负责“对不对”
- benchmark 负责“快不快”
- profiling 负责“为什么慢”

### 3.7 测试分层推进

- `chapter00` 用 smoke test
- `chapter01` 到 `chapter05` 用 `assert` 或最小测试
- `chapter07` 用统一测试 harness 收口
- 未来若复杂度上升，再从 `chapter07` 迁移到 Google Test

### 3.8 外部框架只做参考，不做主线

- `llama.cpp`、`vLLM`、`SGLang`、`TensorRT-LLM`、`Ollama`、`Gemma` 都是坐标，不是教程本体

### 3.9 保持可验证、可比较、可迁移

- 每章都要能验证
- 每次优化都要能比较
- 整条主线要能迁移到自己的工程里

### 3.10 维护文档与学习文档分工明确

- `USAGE.md` 给学习者
- `tutorial_maintenance.md` 给未来的项目贡献者
- `chapter07` 的测试收口同时服务学习者和贡献者，但描述要分层

### 3.11 先收口，再扩展

- 先把当前主线压实
- 新内容只挂在明确的桥接点、进阶点和附录点上

### 3.12 保持教程是教程，不是百科

- 只写当前主线真正需要的内容
- 框架对比和背景材料都应该服务于主线，而不是替代主线

## 4. 维护原则

### 4.1 先保证主线，再补充分支

优先维护以下主线内容：

- 环境准备
- Tensor 与模型加载
- 基础算子
- MatMul 和 Transformer 核心模块
- 生成、量化、测试与 benchmark

只有当主线完整后，才考虑增加更细的分支内容，例如：

- 更多模型格式
- 更多后端实现
- 更多工程化优化
- 更多框架对比

### 4.2 先补桥，再补深度

章节维护的优先顺序是：

1. 先把上一章和下一章的桥接写顺
2. 再补这一章内部的深度说明
3. 最后再考虑进阶内容和案例扩展

如果桥接还不稳，不要先加新功能或新案例。

### 4.3 每章必须回答三个问题

每一章都应该明确写出：

- 本章要解决什么问题
- 本章完成后可以验证什么
- 本章和前后章节是什么依赖关系

如果一章不能回答这三个问题，说明章节粒度需要重构。

### 4.4 每章必须有收口

每章结尾必须能明确回答：

- 这一章解决了什么
- 这一章完成后能做什么
- 下一章为什么必须接上来
- 边界是什么
- 进阶扩展放在哪里

收口要短、硬、统一，不要一章一个风格。

### 4.5 概念和实现同步出现

不要提前堆太多概念。

- 在需要 `Tensor` 的地方再讲 `Tensor`
- 在需要 view 语义时再讲 `reshape` 和 `slice`
- 在需要性能时再讲 `KV Cache`
- 在需要压缩模型时再讲量化

这样读者更容易建立真实的心智模型。

### 4.6 框架对比只做参照，不做主线

教程里外部框架的角色建议固定如下：

- `llama.cpp` 作为本地推理 baseline
- `Gemma` 作为模型家族示例
- `Ollama` 作为本地包装和交付参照
- `vLLM` 作为云端 serving 和调度参照
- `SGLang` 作为应用编排和 structured serving 参照
- `TensorRT-LLM` 作为高性能 GPU engine 参照

不要把这些项目写成同一层级的章节主线。

### 4.7 框架参考必须按场景决策

框架参考页不写“框架百科”，只写“场景决策树”。

读者应该能从页面里直接回答：

- 我是先跑起来，还是先做服务？
- 我更关心吞吐、编排，还是 GPU 性能？
- 我是在端侧、本地、云端，还是分布式场景？

### 4.8 测试也要分层

测试框架不需要在所有章节里一开始就完全统一，建议按阶段分层：

- `chapter00_quickstart`
  - 使用 smoke test 和命令行验证
- `chapter01` 到 `chapter05`
  - 使用章节内最小可执行测试或 `assert` 级测试作为过渡
- `chapter07`
  - 统一引入正式测试体系，承担命名单元测试和端到端测试

原则是：

- 正确性测试和 benchmark 分离
- 章节内测试先保证当前模块可用
- 最终统一测试体系放在 benchmark 章节收口

| 阶段 | 主要测试方式 | 目标 | 说明 |
| --- | --- | --- | --- |
| `chapter00_quickstart` | smoke test / 命令行验证 | 快速确认环境和链路能跑通 | 关注启动、路径、模型下载和首次推理 |
| `chapter01` - `chapter05` | 章节内最小测试 / `assert` / 小脚本 | 验证当前模块语义正确 | 关注 Tensor、算子、量化等局部正确性 |
| `chapter07` | 正式测试体系 / 端到端测试 | 统一正式测试体系 | 关注回归、可维护性和跨模块一致性 |

测试方式的选择原则：

- Smoke test 先解决“能不能跑”
- 章节测试再解决“当前模块对不对”
- 正式测试体系负责“长期可维护的标准测试”
- Benchmark 只回答“快不快”，不回答“对不对”

## 5. 当前测试现状

截至目前，仓库里的测试形态大致是这样：

- `chapter00_quickstart`
  - 以 smoke test 和脚本验证为主
  - 关注环境、路径和首次推理链路
- `chapter01` 到 `chapter05`
  - 以章节内最小可执行测试和 `assert` 级测试为主
  - 关注局部语义正确性和回归稳定性
- `chapter07`
  - 以命名测试用例 + 统一测试入口的 harness 为主
  - 关注端到端正确性、benchmark 结果和收口判断

当前最需要记住的一点是：

- 前面章节的测试主要负责“当前模块对不对”
- `chapter07` 负责把“对不对”和“快不快”收口到同一套可重复流程里
- 如果后续要升级测试体系，优先从 `chapter07` 开始最合适

## 6. 测试框架说明

当前仓库里，章节测试的实际形态可以概括为：

- `chapter00_quickstart`
  - 通过脚本和命令行做 smoke test
- `chapter01` 到 `chapter05`
  - 通过章节内最小可执行测试和 `assert` 做局部验证
- `chapter07`
  - 通过命名测试用例 + 统一测试入口的 harness 做收口
  - 回归样例、边界输入和长上下文退化要作为长期基线维护

这里之所以不写成 Google Test，是因为仓库当前并没有接入 GTest 依赖，文档应该和代码保持一致。
如果后续确实要升级，最合理的起点仍然是 `chapter07`。

## 7. Google Test 迁移预案

如果未来 `chapter07` 的测试复杂度继续上升，GTest 的迁移预案已经单独放在 [gtest_migration.md](./gtest_migration.md)。

那页会回答：

- 什么时候该迁移
- 先迁移哪些文件
- 迁移顺序是什么
- 哪些章节暂时不动

当前维护原则仍然是：如果测试规模还小、测试形态还清楚，就继续用现有 harness，不要为了“更标准”而提前增加依赖。

## 8. 站点构建与本地验证

仓库已经补了一个最小的 VitePress 站点骨架。站点入口由 `docs/index.md`、`docs/guide.md` 和 `docs/.vitepress/config.mts` 共同定义。

当前在线站点适配已经完成物理分层：全部教程正文统一放在 `source/`，`docs/` 继续承担站点入口和在线映射层。

### 8.1 安装依赖

```bash
npm install
```

如果默认镜像里没有 `vitepress`，可以改成：

```bash
npm install --registry=https://registry.npmjs.org
```

### 8.2 本地构建

```bash
npm run docs:build
```

### 8.3 本地预览

```bash
npm run docs:preview
```

### 8.4 本地开发

```bash
npm run docs:dev
```

### 8.5 构建检查点

- 首页是否能正常打开
- 教程主入口是否能正常跳转
- 阅读指南是否能正常跳转
- 附录页和章节页是否能正常渲染
- Mermaid、表格和代码块是否正常显示

## 9. 文档同步与链接规则

为了避免 `README`、使用文档、正文和维护文档各写各的，当前同步和链接规则固定如下。

### 9.1 章节目录使用镜像脚本

- `scripts/migrate_chapter_to_source.sh` 负责章节目录镜像
- `docs/chapterXX_*`、`docs/foundation`、`docs/guides`、`docs/reference`、`docs/maintenance`、`docs/appendix` 都应通过脚本映射到 `source/` 下的对应目录
- 章节目录不要手工在 `docs/` 里复制一份后再长期维护

### 9.2 共享入口文档使用同步脚本

- `scripts/sync_shared_sections.py` 负责中文共享入口同步
- 当前它维护两组映射：
  - `README.md` -> `docs/README.md`
  - `USAGE.md` -> `docs/guide.md`
- 这些页里凡是属于共享章节的内容，先改源文件，再运行同步脚本
- 站点入口页可以使用站点路由写法；`source/` 正文页优先使用源码可点击的相对路径

### 9.3 英文入口单独处理

- 英文类文档使用单独脚本，不和中文同步规则混用
- `scripts/sync_english_sections.py` 负责 `docs/en/README.md` -> `docs/README.en.md`
- 英文入口页可以保留自己的措辞，但章节顺序、路由和覆盖状态要保持可检查
- 英文版教程计划当前挂起，新增中文内容不要求同步扩展英文正文

### 9.4 站点层内容保持手工维护

- `docs/index.md`
- `docs/guide.md`
- `docs/code/`
- `docs/.vitepress/config.mts`

这些内容和正文同步不是一类问题，应该单独维护或单独生成，不要混进章节镜像脚本。

### 9.5 文档互链规则

- `source/` 正文同章链接使用 `./xxx.md`
- `source/` 正文跨章链接使用 `../chapter_xxx/xxx.md`
- `source/` 到 `foundation`、`reference`、`guides`、`maintenance` 的链接也使用相对 `.md` 路径
- `docs/README.md`、`docs/guide.md`、`docs/index.md` 属于站点入口页，可以保留站点路由
- 修改链接后必须运行 `npm run docs:check-links`

### 9.6 文档到代码映射

当前在线文档和源码 Markdown 的代码跳转，统一指向 `code/` 下的具体源码文件，而不是只跳到章节代码目录或代码索引页。

当前约定如下：

- `source/_code -> ../code` 用于源码 Markdown 直接阅读
- `docs/.vitepress/public/_code -> ../../../code` 用于 VitePress 构建后的站点访问
- 正文中的源码链接写成 `<a href="../_code/chapterXX_xxx/path/to/file"><code>path/to/file</code></a>`
- 不使用普通 Markdown 链接指向 `.cpp`、`.h`、`.sh`，避免 VitePress 把它们改写成 `.cpp.html`、`.h.html`、`.sh.html`
- 链接目标可以保留真实完整路径，但展示文本要尽量短；头文件不要展示 `include/chapterXX_xxx/foo.h`，优先展示 `foo.h`
- `docs/code/chapterXX.md` 只做站内源码导航补充，正文里的“源码直达”不再优先跳到这些索引页
- 如果某一页只知道章节，不知道具体文件，应该先补齐具体文件选择，再写链接

这样做的目的，是让读者在源码文件预览和在线站点里都能从文档直接进入真实源码，形成“文档 -> 具体代码 -> 测试 / 脚本”的稳定路径。

## 10. 章节写作与维护流程

当你要新增或修改章节时，建议按下面顺序处理：

1. 先写清楚章节要解决什么问题
2. 再确认前置章节依赖
3. 补充一个最小可验证结果
4. 最后再写实现细节

如果涉及桥接或收口，先统一章节边界，再补正文。

如果章节涉及代码修改，建议同时检查：

- 文档中的目标是否和代码行为一致
- 示例命令是否仍然可执行
- 验证结果是否还成立
- 在线文档镜像是否已经同步
- 本地构建是否通过
- dead link 是否清理干净

如果是按优先级推进主线章节，建议直接按下面顺序落地：

1. 先做 `chapter01` 到 `chapter07` 的阅读入口质量抽查
2. 继续做 `chapter08_profiling` 的阶段计时与 profiling 样例补强
3. 最后处理站点入口页的源码预览友好策略

每完成一章，就做一次完成验收，不要攒到最后一起查。

### 10.1 收口标准固定

每章收口统一按以下五项写，不再临时发挥：

1. 本章解决了什么
2. 本章完成后能做什么
3. 下一章为什么必须接上来
4. 边界说明
5. 进阶扩展

建议句式：

- 这一章解决的是……
- 这一章完成后，读者可以……
- 如果没有这一步，下一章会缺少……
- 这一章的增量重点是……
- 下一章会在这个基础上……

### 10.2 增量规则固定

增量期遵循下面几条：

- 主线不动，增量只挂指定位置
- 先补桥，再补深度
- 先补主线必需的增量，再补进阶分支
- 每次新增内容都要能回答“它补的是哪一个前置缺口”
- `chapter06` 主要承接 scheduler、continuous batching、streaming 进阶
- `chapter07` 主要承接 memory pool、profiling、A/B benchmark 进阶
- `chapter08` 主要承接 profiling、A/B 实验和优化闭环

正文模板规则固定如下：

- `intro.md` 必须有可跳转的章节导读，并承载定位、顺序和验证
- 章节结论与验证统一写入 `intro.md`，不再创建独立 `summary.md`
- 章节正文的标题编号要和章节号保持一致，避免 `06.x`、`04.x`、`07.x` 混写
- 同一章内，intro 的结构保持一致，减少读者重新适应成本

### 10.3 语言规则固定

- 正文页面以中文为主
- 必要技术名词可以保留英文，例如 `Tensor`、`MatMul`、`KV Cache`
- `docs/README.md` 和生态索引页可以中英混排
- 其他正文不要用大段英文模板
- 语言策略的完整约定见 [language_strategy.md](./language_strategy.md)

### 10.4 章节间代码复用方式

这套教程的代码目录是“按章节拆分”，不是“按功能彻底隔离”。也就是说：

- 每章保留自己的最小可运行代码
- 真正稳定、会被多章反复使用的基础能力，可以抽成公共层
- 不要为了“共享”而把章节骨架揉在一起

推荐的复用顺序是：

1. 先看正文里的源码直达链接
2. 再查看本章代码导航页，确认源码、测试和脚本分布
3. 如果某一段逻辑后续章节会继续复用，优先在文档里说明“它属于哪一章的基础能力”
4. 如果确实要抽公共代码，优先放到明确的公共目录或公共 target，而不是直接跨章节拷贝

这样做的目的是让读者先知道“怎么用”，再决定“代码放哪儿”。

### 10.5 新增章节模板

建议每个章节至少包含以下部分：

- `intro.md`
  - 问题陈述
  - 为什么重要
  - 本章产出
  - 实现路径
- 正文
  - 一个概念点对应一个实现步骤
  - 每一步都有可运行或可验证的结果
- `intro.md`
  - 本章位置与成功标准
  - 内容地图
  - 阅读与源码顺序
  - 验证与下一章提示

## 11. 章节完成验收

每一章完成后，都要做下面三项验收，不要只看内容改完没有：

- 在线文档镜像映射
  - 确认 `source/` 对应章节已经在 `docs/` 站点层正确映射
  - 确认在线入口、导航和章节锚点都能跳到新内容
- 本地构建验证
  - 至少跑一次 `npm run docs:check-links`
  - 如果章节结构有明显变化，再补一次本地预览检查
- dead link 检查
  - 检查章节内锚点链接是否仍然可跳转
  - 检查跨章节链接、附录链接和维护文档链接是否失效
  - 如果有镜像页或索引页同步，顺手检查这些页面的引用是否还正确

## 12. 文档更新检查清单

在提交文档修改前，建议检查以下项：

- 标题和章节编号是否一致
- 中英文术语是否统一
- 拼写错误是否修正
- 命令行示例是否仍可执行
- 是否说明了验证方式
- 是否说明了失败时怎么排查
- 是否避免把多个层级的内容混在一起
- 是否按收口标准写清了当前章边界
- 是否把进阶内容放进了章节尾部
- 是否完成在线文档镜像映射
- 是否完成 `npm run docs:check-links`
- 是否完成 dead link 检查

## 13. 版本维护建议

如果教程内容开始稳定，建议再做两类管理：

- 目录索引页
  - 明确章节顺序
  - 标注每章适用场景
- 变更记录
  - 记录新增章节
  - 记录重大术语调整
  - 记录依赖版本变化
  - 记录收口规则变化

这样后续更新时，不会把教程改散。

## 14. 什么时候需要重构教程

出现以下情况时，建议考虑重构而不是局部修补：

- 同一章同时承担太多功能
- 读者在某章需要回头补太多前置知识
- 章节之间的顺序已经不再自然
- 代码实现和文档叙述偏离太多
- 框架对比内容开始挤占主线篇幅

## 15. 建议的维护节奏

- 小修订：修正拼写、命令、链接、术语
- 中修订：补充验证步骤、重排章节说明
- 大修订：重构目录、拆分章节、调整学习路径

建议优先顺序：

1. 入口链条
2. 章节桥接
3. 收口标准
4. 进阶分支
5. 生态对照

## 16. 最后原则

这套教程的维护目标不是“把每个技术点都写满”，而是保持它始终像一条可走通的学习路径。
