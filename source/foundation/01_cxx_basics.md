# F01 C++ 基础

Foundation 的 C++ 内容只补主线会反复使用的对象语义：所有权、生命周期、移动和容器选择。目标不是复习全部语法，而是能读懂 `Tensor`、`Workspace` 和 `Request` 的资源关系。

## 主线位置与学习目标

| 读完后 | 在主线中的用途 |
| --- | --- |
| 能区分拥有资源和借用资源 | `Tensor` storage、workspace 和 view |
| 能解释构造、析构、移动的结果 | 避免 buffer 重复释放和隐式复制 |
| 能按访问方式选择容器 | 参数表、连续数据和请求队列 |

## 核心语义

| 语义 | 默认做法 | 推理引擎中的例子 |
| --- | --- | --- |
| 单一所有权 | RAII、`std::unique_ptr` | buffer、workspace |
| 借用数据 | 引用、指针或 view，不释放底层 storage | Tensor slice |
| 大对象转移 | 删除复制，支持移动 | 返回 Tensor、交接 request |
| 连续数值数据 | `std::vector` 或专用 buffer | Tensor payload |
| 名称到对象 | `std::unordered_map<std::string, T>` | 权重和 metadata 查找 |

## 最小示例

```cpp
class Buffer {
 public:
  explicit Buffer(std::size_t n) : data_(std::make_unique<float[]>(n)), size_(n) {}
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) noexcept = default;

 private:
  std::unique_ptr<float[]> data_;
  std::size_t size_;
};
```

复制被禁止，移动表示所有权转移；析构自动释放。view 不应拥有 `data_`，否则一个 storage 会被释放多次。

## 验证与练习

验证时回答三个问题：资源由谁释放？移动后源对象是什么状态？返回大对象是否需要手写 `new/delete`？

对应练习见 [Foundation 正文与练习映射](./exercise_map.md)：`FQ01` 验证所有权，`FQ17` 验证完整最小工程。
