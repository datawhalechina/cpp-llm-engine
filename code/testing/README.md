# 公共测试基础设施

本目录为各章提供统一测试断言、CMake 注册规则和测试资产登记。章节测试仍放在各自 `tests/` 目录；这里不集中保存业务测试，也不替代每章的 `run.sh`。

## 代码架构

| 资产 | 职责 |
| --- | --- |
| `include/cpp_llm_engine/testing/test.h` | 提供不受 `NDEBUG` 影响的命名检查、前置条件和异常断言 |
| `cmake/CppLlmTesting.cmake` | 为测试目标接入公共头文件，并登记 owner、证据层级和迁移状态标签 |
| `chapter_test_contracts.json` | 记录测试源码、CMake target、所属章节、依赖章节和证据层级 |
| `tests/testing_self_test.cpp` | 验证公共断言自身在 Release 构建中仍然工作 |

`CHECK` 记录失败后继续当前用例，`REQUIRE` 在前置条件失败时停止当前用例，避免后续索引或解引用掩盖最初错误。CTest label 用于筛选证据范围，不作为测试是否通过的替代条件。

## 新增测试的顺序

1. 在所属章节 `tests/` 中增加正常、边界和失败路径。
2. 在章节 CMake 中创建 target、调用 `cpp_llm_prepare_test()` 并注册 CTest。
3. 使用 `cpp_llm_set_test_contract()` 标记 owner、level 和 active 状态。
4. 在 `chapter_test_contracts.json` 登记源码、target 和跨章依赖。
5. 先运行章节 `run.sh`，再运行资产检查和全局回归。

| 证据层级 | 适用范围 |
| --- | --- |
| `unit` | 单个函数、算法或局部状态 |
| `component` | 同一章多个对象协作 |
| `contract` | 接口、不变量和错误行为 |
| `integration` | 两章或多个模块的真实交接 |
| `real_model` | 需要本地模型文件的权重、logits 或词元证据 |
| `e2e` | 从入口到最终输出的完整路径 |

## 验证入口

```bash
npm run test:check-assets
npm run test:check-contracts
npm run test:regression
npm run test:release-contracts
```

日常修改先运行所属章节入口；默认回归不下载模型、不访问网络、不要求 GPU。真实 GGUF、CUDA、Socket 和性能实验按 [`USAGE.md`](../../USAGE.md) 的专项入口执行，提交要求见 [`CONTRIBUTING.md`](../../CONTRIBUTING.md)。
