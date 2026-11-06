# C++ LLM Inference Engine Tutorial

_A Practical Tutorial for Building a C++ LLM Inference Engine_

This is the canonical English entry for the tutorial. The Chinese main line remains the source of truth, and the English side starts with entry pages first.

## Roles

- foundation: background you should know before the main line
- chapter00_quickstart: the first runnable baseline
- guide.md: how to read the tutorial
- guides: engineering references for build, environment, and migration
- reference: terminology, concept mapping, ecosystem comparison, and engine-layer analysis

## Engine Layer

The engine layer is the execution core of inference systems. It turns a static model into a runnable runtime with clear boundaries for tensor representation, weight loading, operator execution, decoder structure, runtime state, KV cache, quantization, backend adaptation, testing, and benchmarking.

You can think of Python inference frameworks as the "control tower" and the C++ inference engine as the "engine":

- the control tower organizes requests and workflows
- the engine executes the hot path, manages memory and state, and determines latency and throughput

This tutorial teaches both sides together, but the main line focuses on the engine layer first.

## Start here

1. Read [guide.md](guide) first if you want a route map.
2. Read [Foundation](foundation/intro) if you need background.
3. Open [chapter00_quickstart](chapter00_quickstart/intro) when you want the first runnable baseline.
4. Use [guides/README.md](guides/README) for build or environment help.
5. Use [reference/README.md](reference/README) when you need concept mapping or ecosystem context.
6. Read [training vs inference frameworks](reference/framework_comparison) for the engine-layer map.

## Coverage status

- English navigation and entry pages: available
- Selected chapter introductions: available
- The Chinese source remains authoritative while full chapter translation is incomplete

## Main line

- foundation
- chapter00_quickstart
- chapter01_tensor_loader
- chapter02_tensor_ops
- chapter03_matmul
- chapter04_attention
- chapter05_quantization
- chapter06_generation
- chapter07_kv_cache through chapter10_serving (Chinese source of truth)

## Notes

- If you only need the route map, start from [guide.md](guide).
- If you need the full Chinese walkthrough, switch back to the Chinese main entry.

## Daily checks

If you are changing docs or links, run:

```bash
npm run docs:check-links
```

Then run:

```bash
npm run docs:build
```
