# 01.1 张量对象（`Tensor`）

模型文件中的权重进入程序后，需要用同一种对象表达名称、形状、类型和底层字节。`Tensor` 将文件记录转换为后续视图和算子都能检查的数据对象。

## Tensor 保存哪些信息

`Tensor` 保存以下字段：

| 信息 | 当前实现 | 后续用途 |
| --- | --- | --- |
| 名称 | `name_` | 按模型结构约束定位权重 |
| 逻辑布局 | `shape_`、`dtype_` | 校验元素数量和算子输入 |
| 执行位置 | `device_`，当前为 CPU | 后续选择 CPU/GPU 计算后端 |
| 物理数据 | `shared_ptr<vector<byte>>` | 共享底层存储并管理生命周期 |
| 视图起点 | `offset_bytes_` | 定位当前权重数据的起始字节 |
| 有效载荷 | `payload_bytes_` | 记录量化权重的真实字节数 |

形状（`shape`）描述各维长度，底层存储保存物理字节。加载器分别保存这两类信息，既能检查布局，也能保留量化权重的原始数据。Chapter02 会用独立的 Float32 教学张量继续学习形状、底层存储、偏移和视图。

## 从输入数据构造 Tensor

接口和实现分别位于 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/tensor.h"><code>tensor.h</code></a> 与 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/tensor.cpp"><code>tensor.cpp</code></a>。`from_vector<T>()` 适合测试和普通数值，写入前会核对形状与元素数量；`to_vector<T>()` 读取前会核对数据类型和剩余存储长度。`from_raw_bytes()` 保留 GGUF 量化权重的原始字节，由对应格式加载器验证载荷。复制张量时共享底层存储；移动张量时转移对象状态，资源释放由 RAII 完成。

## 验证与衔接

从仓库根目录运行：

```bash
bash code/chapter01_tensor_loader/run.sh
```

在测试中确认以下结果：

1. `numel()` 与形状一致。
2. `bytes()` 和原始权重数据大小一致。
3. `raw_data()` 已包含 `offset_bytes`。
4. 类型不匹配或存储长度不足时，类型化读取会失败。
5. 非法形状、元素数量或越界偏移会明确失败。

本章的 `Tensor` 面向模型文件中的权重。Chapter02 使用独立教学类型演示视图操作，两类对象通过公共 `TensorContract` 向算子提供形状、类型和存储范围。

