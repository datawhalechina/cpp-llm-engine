# Chapter 03 MatMul and Compute Backends

Build a MatMul operator with a clean CPU baseline and a GPU backend.

## What you will do

- Design the operator interface
- Write the CPU reference implementation
- Add the cuBLAS path and compare it with PyTorch

## What you will have at the end

- a correct MatMul baseline plus a GPU backend
- A clear bridge to the next chapter
- A visible checkpoint for the English route

## Status

This is the English landing page for the chapter. The detailed Chinese pages remain the source of truth for now.

## Next step

Continue to [summary](summary) or move to [the next chapter](/en/chapter04_attention/intro).

## Related Reading

- Prerequisites: read [Foundation 01](/en/foundation/01_cxx_basics) and [Foundation 02](/en/foundation/02_cmake_build_basics) first, so the interface, build, and layout constraints are easier to follow.
- Later use: `chapter04_attention` will reuse this `MatMulOp` boundary directly, and `chapter06_generation` will keep using the same operator interface in the runtime.
- The full 03–06 reading path is summarized on this page. Use it as the roadmap for the chapters that follow.

### 03–06 Reading Path

| Chapter | Read first | What it gives you |
| --- | --- | --- |
| `chapter03_matmul` | `Operator`, CPU reference, cuBLAS comparison | A stable, replaceable, and checkable matmul boundary |
| `chapter04_attention` | `Multi-head Attention`, `DecoderLayer`, first-layer weight wiring | A reusable model structure built on top of the operator boundary |
| `chapter06_generation` | `Naive Generation`, `KV Cache`, `Sampler` | A persistent generation loop built on the decoder structure |
| `chapter05_quantization` | quantization basics, dequantization, accuracy checks | A resource-vs-accuracy trade-off on top of the stable runtime |

### Source Reading Suggestion

- `chapter03_matmul`: read `Operator` first, then the implemented CPU reference, then the planned cuBLAS backend design
- `chapter04_attention`: read `Multi-head Attention`, then `DecoderLayer`, then first-layer weight validation
- `chapter06_generation`: read `Naive Generation`, then `KV Cache`, then `Sampler`
- `chapter05_quantization`: read quantization basics, then `DequantizeOp`, then accuracy validation
