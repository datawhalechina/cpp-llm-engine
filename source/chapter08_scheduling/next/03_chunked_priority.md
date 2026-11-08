# 08.3 让长输入和普通请求都能推进

完整 Prefill 可能一次占满词元预算。Chunked Prefill 把长输入拆成多个调度片段；优先级老化让等待请求逐步提高选择机会；Chapter07 的前缀命中长度帮助同级请求选择先后。三者共同处理“谁先执行”和“每次执行多少”的问题。

## 分块 Prefill 保留模型进度

`Request::prefill_offset()` 指向下一段起点。中间片段只建立 K/V，最后一段完成后才采样首个输出词元。

```text
remaining = prompt_size - prefill_offset
chunk = min(remaining, remaining_budget, max_chunk_tokens)
```

| 片段 | 是否开始新序列 | 是否产生 logits | 提交结果 |
| --- | --- | --- | --- |
| 第一段 | 是 | 仅当同时为最后一段 | 增加 Prefill 进度 |
| 中间段 | 否 | 否 | 继续增加进度和 K/V |
| 最后一段 | 否，单段时为是 | 是 | 完成 Prefill 并采样首词元 |

## 优先级老化固定候选顺序

基础优先级较大的请求先执行；未实际推进的请求累积等待步数。增益有上限，避免等待时间无限覆盖业务优先级。

```text
effective_priority = base_priority
                   + min(waiting_steps / aging_interval, max_aging_boost)
```

| 排序信息 | 规则 | 用途 |
| --- | --- | --- |
| 有效优先级 | 较大者优先 | 表达业务优先级和老化结果 |
| 等待步数 | 较大者优先 | 有效优先级相同时照顾久等请求 |
| 缓存前缀词元数 | 较大者优先 | 同级 Prefill 先利用较长命中 |
| Prefill 进度 | 较少者优先 | 让多个长输入轮换推进 |
| 入队序号 | 较小者优先 | 固定平局结果 |

`cached_prefix_tokens` 在两个阶段有不同含义。准入时，ServiceCore 保存 Chapter07 给出的无副作用估算，只用它排列 waiting 请求，不增加块引用，也不推进 Request。请求获得运行槽时，Factory 重新查询并采用仍然有效的前缀，同时推进 `BlockTable`、`Request::prefill_offset()` 和模型 Context；随后交给 Planner 的才是实际采用长度。估算过期只会改变候选顺序，不能直接改变执行状态。

## 当前公平性保证到哪里

`PriorityAgingPolicy` 由调用方跨 step 保存，Planner 只读策略状态。缓存不足或 Executor 没有推进时，请求继续老化；实际推进后等待步数清零。Overlap 计划同时记录策略版本，版本变化后旧计划不能执行。

| 已验证 | 后续能力 |
| --- | --- |
| Decode 优先并为可分块 Prefill 保留最小机会 | 自动 Deadline 与到期队列 |
| 多个长输入按进度轮换 | step 边界协作式抢占 |
| 小优先级差可由有界老化追平 | 缓存压力和持续 admission 下的长期公平性 |
| 同优先级结果可复现 | 配额与多租户策略 |
| 同级 Prefill 按估算命中长度选择，运行前重新采用 | 缓存压力与命中收益联合评分 |

## 验证分块和老化

```bash
cmake -S code/chapter08_scheduling -B /tmp/cpp-llm-chapter08-policy \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter08-policy \
  --target chapter08_step_scheduler_test chapter08_priority_aging_test -j2
ctest --test-dir /tmp/cpp-llm-chapter08-policy \
  -R '^chapter08_(step_scheduler|priority_aging)_test$' --output-on-failure
```

| 通过条件 | 观察对象 |
| --- | --- |
| 中间片段不采样，最后片段只采样一次 | Prefill 语义 |
| 两个长输入轮换并完成 | `prefill_offset` |
| 高优先级先执行，普通请求等待后获得机会 | 有效优先级 |
| 老化不超过配置上限 | `max_aging_boost` |
| 同级请求中缓存命中较长者先进入计划 | `cached_prefix_tokens` |

真实纵向验收由 `real_tinyllama_serving_test` 完成：冷请求先入队，共享四词元 K/V 前缀的请求后入队，同优先级下命中请求先产生输出。
