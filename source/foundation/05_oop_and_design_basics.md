# F05 面向对象与设计基础

推理引擎的类设计应直接表达对象的职责、状态和资源关系。类越大，越难单独测试；接口越宽，后续章节越容易互相依赖。

## 状态、资源与最小接口

| 设计问题 | 建议 | 主线对象 |
| --- | --- | --- |
| 谁拥有数据 | 在类型中明确 owner | `Tensor`、`Workspace` |
| 谁改变状态 | 让动作集中在状态拥有者 | `Request`、`Scheduler` |
| 接口暴露多少 | 只暴露下一步需要的能力 | `Operator`、`Backend` |
| 如何组合功能 | 优先组合小对象 | `DecoderLayer`、runtime |
| 何时继承 | 只有替换关系稳定时使用 | backend 接口 |

## 对象职责

| 对象 | 持有 | 提供 | 不负责 |
| --- | --- | --- | --- |
| `Tensor` | storage、shape、dtype | 索引和 view | 调度请求 |
| `Operator` | 算子参数 | `run` 或 `forward` | 管理全局 runtime |
| `Request` | 输入、状态、输出 | 状态迁移 | 直接分配 GPU buffer |
| `Scheduler` | 请求队列和预算 | 选择下一步 | 实现 MatMul |

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

权重属于 `Linear`，输入是借用的 view；`Linear` 不负责请求队列、模型文件或 CUDA stream。

## 验证与练习

检查每个类：字段属于谁？状态由谁改变？接口能否单独测试？是否能用组合替代继承？对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 `FQ01`、`FQ08`，再用 `FQ17` 验证工程组合。
