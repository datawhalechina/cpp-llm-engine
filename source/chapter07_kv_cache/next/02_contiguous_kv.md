# 07.2 用连续存储建立缓存基线

连续 KV Cache 为每层预留一段固定容量，词元按序写入相邻位置。它容易检查地址和数值，是分页实现必须对照的 reference；缺点是请求长度变化时难以灵活扩容和复用空闲空间。

## 连续地址怎样计算

Chapter04 的 `KvCache` 使用 `[token, kv_head, head_dimension]` 布局。给定词元位置 `t`、KV head `h` 和维度 `d`，元素偏移是：

```text
offset = (t * Hkv + h) * Dh + d
```

key 和 value 使用两块独立的 `float` storage，地址公式相同。实现位于 <a href="/_code/chapter04_attention/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>。

| 操作 | Storage 变化 | `size` 变化 |
| --- | --- | --- |
| 构造 | 一次分配 `capacity * Hkv * Dh` | 设为 0 |
| `append` | 把新行复制到 `size` 之后 | 增加本次 token 数 |
| 超容量 `append` | 不写入 | 保持不变 |
| `clear` | 保留已分配内存 | 重置为 0 |

`clear` 不需要把全部数值清零，因为后续 Attention 只读取 `0..size-1`。物理内存仍属于同一个缓存对象，下一次请求可以覆盖旧位置。

## 真实模型怎样使用连续缓存

`RealLlamaCpuContext` 为请求保存当前位置和每层 K/V。Model 保存共享只读权重，Context 不能跨请求共享。

| 对象 | 生命周期 | 保存内容 |
| --- | --- | --- |
| `RealLlamaCpuModel` | 多请求共享 | 量化权重和模型配置 |
| `RealLlamaCpuContext` | 单请求独占 | position、每层 K/V、执行统计 |
| `Request` | 单请求独占 | prompt、输出词元和终态 |

连续实现让 `token_index` 可以直接转换为数组偏移，但每个请求都要持有按最大容量或增长策略分配的连续区域。大量不同长度请求并发时，空闲容量不能直接借给另一请求，扩容还可能移动数据。

## 验证追加和隔离

先运行 Chapter04 连续缓存测试，再运行真实 TinyLlama 的重复请求隔离：

```bash
cmake -S code/chapter04_attention -B /tmp/cpp-llm-chapter07-contiguous \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-contiguous \
  --target chapter04_attention_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-contiguous \
  -R '^chapter04_attention_test$' --output-on-failure

ctest --test-dir /tmp/cpp-llm-real-chapter06 \
  -R '^real_tinyllama_generation_test$' --output-on-failure
```

第一项固定连续 storage 的追加语义，第二项确认共享权重时两次请求不会共享 Context。分页缓存必须保持相同 logits 和输出，只改变位置到物理内存的映射方式。
