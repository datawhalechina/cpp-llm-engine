# 推理引擎输入链路

输入链路把用户文本转换为带有模板、特殊 token、停止条件和长度限制的 token 序列。

## 1. 请求如何进入引擎

模型权重在加载阶段进入只读 Model；用户请求在运行阶段经过 tokenizer、模板、特殊 token 和长度检查进入 Runtime。

## 2. 输入链路的基本步骤

一条典型输入链路可以拆成这些环节：

1. 文本进入系统
2. tokenizer 把文本切成 token ids
3. prompt template 把请求整理成统一格式
4. special tokens 补充系统约定
5. stop criteria 定义什么时候结束
6. max length 约束上下文长度
7. pad / truncation 处理 batch 中的长度差异
8. token ids 进入 embedding 和位置编码

这些步骤的目标不是增加复杂度，而是把“用户看到的请求”变成“引擎内部可稳定处理的请求”。

## 3. tokenizer

tokenizer 的作用，是把自然语言字符串转换成 token ids。  
它决定了：

- 词元边界怎么切
- 特殊符号怎么保留
- 词表里没有的东西怎么处理

对于推理引擎来说，tokenizer 不是前端细节，而是输入语义的一部分。  
一旦 tokenizer 变了，后面的长度、停止条件和 cache 行为都可能跟着变。

## 4. prompt template

prompt template 的作用，是把不同来源的请求整理成统一结构。

这层通常会处理：

- system / user / assistant 角色
- 固定前缀
- 对话格式
- 模型家族要求的特殊包装

如果没有这层，不同调用方发进来的请求会长得不一样，后面很难做稳定验证和复用。

## 5. special tokens、stop criteria 和长度约束

这三件事决定输入链路能不能和生成链路闭合：

- special tokens 决定模型如何识别开始和结束
- stop criteria 决定输出什么时候停
- max length 决定上下文不会无限长

它们不是模型外的装饰，而是生成系统的契约。

### padding 和 truncation

当 batch 中不同请求长度不一致时，就会遇到 padding 和 truncation。

- padding 让短请求对齐 batch
- truncation 让超长请求被限制在可控范围内

这两者都要和 tokenizer、cache 和 scheduler 一起看，不能孤立讨论。

## 6. 和主线章节的关系

| 章节 | 已经做了什么 | 输入链路页继续补什么 |
| --- | --- | --- |
| `chapter01_tensor_loader` | 权重、`Tensor`、`ModelLoader` | 输入请求如何进入系统 |
| `chapter06_generation` | embedding、位置编码、生成 runtime | 请求契约如何进入 runtime |
| `chapter09_performance` | 测试和性能记录 | 不同输入契约如何影响比较结果 |

输入链路为 Runtime 提供结构稳定、长度合法且终止条件明确的请求。

最小学习顺序是：`chapter00` 先用固定 prompt 跑通外部输入，`chapter01` 固定权重和 `Tensor` 对象，`chapter06` 再把 token ids、停止条件和长度限制放进 `Request`。这样输入链路与模型加载、生成调度各自保持清楚。

相关阅读：[`00.4 First Inference`](../chapter00_quickstart/04_first_inference.md)、[`01.0 Model Weights and Tensor Objects`](../chapter01_tensor_loader/intro.md)、[`06.4 Request 生命周期`](../chapter06_generation/next/04_request_lifecycle.md)、[`推理引擎回归测试`](./regression_testing.md)

## 7. 如何使用输入链路参考页

- 先看 `chapter06_generation/next/01_request_contract`
- 再回到这里补充 tokenizer 和请求契约
- 如果后面看起来像模型有问题，先检查输入链路是否已经统一

## 核心结论

输入链路确定模型开始计算时收到的 token、位置、长度和停止条件。
