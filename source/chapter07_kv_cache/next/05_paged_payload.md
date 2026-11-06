# 07.5 在分页槽位中保存真实 K/V

页表只给出物理位置，`CpuPagedKvCache` 负责保存每层的 key 和 value。构造函数一次分配固定大小的数据区；请求获得物理块后，写入操作再把一行 K/V 放到对应层和槽位。

## 控制面与数据面怎样配合

一次写入需要 `request_id`、`layer`、`token_index` 和一行 K/V。执行顺序是：

```text
request_id + token_index
        -> BlockTable
        -> {physical_block, slot}
        -> [layer, physical_block, slot, kv_head, head_dimension]
        -> key/value payload
```

| 检查 | 拒绝的错误 |
| --- | --- |
| `layer` 在范围内 | 写入不存在的模型层 |
| key/value 指针有效 | 空输入 |
| 元素数等于 `Hkv * Dh` | 形状不匹配 |
| 请求已经预留该槽位 | 越过页表容量 |
| 位置尚未提交 | 覆盖已生效的历史 K/V |

`written_` 为每个“层、物理块、槽位”记录写入状态。它不保存数值，只用于检查一个词元是否已经在所有模型层完成写入。

## 请求释放后为什么要清理数据

物理块可以交给另一个请求复用。`release` 先归还页表中的块，再清零这些块的 key、value 和写入标记。新请求即使获得相同物理块，也不能读到旧请求的数值或继承旧的完成状态。

| 生命周期动作 | 页表 | K/V 数据 | 写入标记 |
| --- | --- | --- | --- |
| `reserve` | 增加容量 | 保持初始值 | `false` |
| `write_token` | 不变 | 写入指定层和槽位 | 对应位置改为 `true` |
| `commit_tokens` | 增加已提交词元数 | 不变 | 保留 |
| `release` | 删除请求页表 | 清零归还块 | 清零归还块 |

实现位于 <a href="/_code/chapter07_kv_cache/include/chapter07_kv_cache/paged_kv_cache.h"><code>paged_kv_cache.h</code></a> 和 <a href="/_code/chapter07_kv_cache/src/paged_kv_cache.cpp"><code>paged_kv_cache.cpp</code></a>。

## 验证分层写入与复用隔离

```bash
cmake -S code/chapter07_kv_cache -B /tmp/cpp-llm-chapter07-payload \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-payload \
  --target chapter07_paged_kv_cache_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-payload \
  -R '^chapter07_paged_kv_cache_test$' --output-on-failure
```

测试先验证缺少任意模型层时不能提交，再释放请求 A 并让请求 B 复用相同物理块，最后检查请求 B 的 Attention 输出只来自新写入的数据。
