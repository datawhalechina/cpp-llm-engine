# 01.1 张量对象（`Tensor`）

模型文件中的权重进入程序后，需要用同一种对象表达名称、形状、类型和底层字节。`Tensor` 将文件加载结果转换为后续 View 和算子都能检查的数据对象。

## Tensor 保存哪些信息

`Tensor` 用一组稳定字段连接模型加载器、算子和运行时：

| 信息 | 当前实现 | 后续用途 |
| --- | --- | --- |
| 名称 | `name_` | 按模型 schema 定位权重 |
| 逻辑布局 | `shape_`、`dtype_` | 校验元素数量和算子输入 |
| 执行位置 | `device_`，当前为 CPU | 后续选择 CPU/GPU 计算后端 |
| 物理数据 | `shared_ptr<vector<byte>>` | 共享底层存储并管理生命周期 |
| 视图起点 | `offset_bytes_` | 定位当前权重数据的起始字节 |
| 有效载荷 | `payload_bytes_` | 支持量化 tensor 的真实字节数 |

形状（`shape`）描述逻辑对象，底层存储保存物理字节。两者分开后，加载器可以稳定保存权重的布局和原始数据。Chapter02 会用独立的 Float32 教学张量继续学习形状、底层存储、偏移和视图概念。

## 从输入数据构造 Tensor

接口和实现分别位于 <a href="../_code/chapter01_tensor_loader/src/tensor.h"><code>tensor.h</code></a> 与 <a href="../_code/chapter01_tensor_loader/src/tensor.cpp"><code>tensor.cpp</code></a>。`from_vector<T>()` 适合测试和普通数值，写入前会核对形状与元素数量；`to_vector<T>()` 读取前会核对数据类型和剩余存储长度。`from_raw_bytes()` 保留 GGUF 量化权重的原始字节，由格式 Loader 负责验证 payload。复制张量时共享底层存储；移动张量时转移对象状态，资源释放由 RAII 完成。

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

这些字段固定了加载结果的模型语义。Chapter02 不直接扩展这个权重 Tensor，而是通过独立教学类型建立 View 语义；后续算子再通过公共 `TensorContract` 连接两类对象。

