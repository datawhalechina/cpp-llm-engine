# F05 面向对象与设计基础

推理引擎的类设计应直接表达对象的职责、状态和资源关系。类越大，越难单独测试；接口越宽，后续章节越容易互相依赖。

## 状态、资源与最小接口

| 设计问题 | 建议 | 主线对象 |
| --- | --- | --- |
| 谁拥有数据 | 在类型中明确所有者 | `Tensor`、`Workspace` |
| 谁改变状态 | 让动作集中在状态拥有者 | `Request`、`Scheduler` |
| 接口暴露多少 | 只暴露下一步需要的能力 | `Operator`、`Backend` |
| 如何组合功能 | 优先组合小对象 | `DecoderLayer`、生成运行时 |
| 何时继承 | 只有替换关系稳定时使用 | 后端接口 |

## 对象职责

| 对象 | 持有 | 提供 | 交给其他对象的工作 |
| --- | --- | --- | --- |
| `Tensor` | 存储、形状、数据类型 | 索引和视图 | 请求调度由调度器处理 |
| `Operator` | 算子参数 | `run` 或 `forward` | 生成状态由运行时管理 |
| `Request` | 输入、状态、输出 | 状态迁移 | GPU 缓冲区由后端管理 |
| `Scheduler` | 请求队列和预算 | 选择下一步 | 矩阵乘法由算子后端实现 |

## 接口、后端与工厂

教程使用三个角色隔离“调用什么”和“怎样实现”：

| 角色 | 负责 | 示例 |
| --- | --- | --- |
| 接口 | 定义调用者可依赖的最小操作 | `Backend::run(input)` |
| 后端 | 在 CPU、CUDA 或测试环境实现接口 | `CpuBackend`、`CudaBackend`、`FakeBackend` |
| 工厂 | 根据模型元数据和运行参数创建合适实现 | `make_backend(config)` |

工厂模式指“把对象选择和创建集中在一个入口”。算子执行由后端负责，请求状态由调用者持有。调用者依赖接口后，测试可以注入 `FakeBackend`，运行时则由工厂选择 CPU 或 CUDA 后端。存在多个稳定实现或创建过程需要共享资源时使用工厂；单一实现可以直接构造。

## 最小示例

```cpp
class Linear {
 public:
  Linear(Tensor weights, Tensor bias)
      : weights_(std::move(weights)), bias_(std::move(bias)) {}

  Tensor run(const Tensor& input) const;

 private:
  Tensor weights_;
  Tensor bias_;
};
```

权重属于 `Linear`，输入是借用的视图；请求队列、模型文件和 CUDA 流由外层组件管理。

## 验证与练习

检查每个类：字段属于谁？状态由谁改变？接口能否单独测试？是否能用组合替代继承？对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 `FQ01`、`FQ08`，再用 `FQ17` 验证工程组合；FQ17 的选做层继续练习后端接口、工厂（Factory）和测试替身（Fake）。
