# F07 Common CUDA Libraries

## Goal

Give the reader a CUDA library map that can be used to make tutorial-route decisions, not just a list of names.

## What you should be able to do after reading

- Understand what different CUDA libraries solve
- Understand why the tutorial focuses on `cuBLAS` first
- Understand which libraries are optional and which are common in inference systems

## Why this matters

CUDA library names can be more confusing than the code itself. This page is not about memorizing the whole ecosystem; it is about knowing which library maps to which engineering problem in this tutorial.

## What comes next

- `cuBLAS` for matrix multiplication
- `CUDA Graph` for launch overhead
- memory-pool style allocation management
- optional awareness of `cuDNN`, `CUTLASS`, `Thrust`, `CUB`, and `NCCL`

## Where they appear

- `chapter03_matmul`: `cuBLAS` as the dense matmul backend reference
- `chapter06_generation`: `NCCL` only if the tutorial grows toward multi-GPU serving
- `chapter07_benchmark`: `CUDA Graph`, memory pools, and async copy as optimization ideas

## Chapter relationship

| Chapter | Layer | Question |
| --- | --- | --- |
| `F03` | Environment | Can the GPU be seen? Are driver and runtime usable? |
| `F07` | Library | What problem does `cuBLAS` solve? What problem does `CUDA Graph` solve? |
| `chapter03_matmul` | Application | How do we use `cuBLAS` to implement MatMul? |

## Core ideas

### 1. Separate runtime and driver

Runtime lets the program issue GPU work. Driver makes the GPU visible and usable to the system.

### 2. `cuBLAS` is the dense matmul path

`cuBLAS` is NVIDIA’s standard dense linear algebra library. In this tutorial, it is the most direct GPU backend reference for `MatMul`.

### 3. `CUDA Graph` reduces repeated launch overhead

`CUDA Graph` captures repeated GPU launch patterns. It belongs more naturally to performance optimization than to the first implementation pass.

### 4. `cuDNN`, `CUTLASS`, `Thrust`, `CUB`, and `NCCL` each solve different extension problems

- `cuDNN`: deep-learning primitives
- `CUTLASS`: high-performance custom-kernel reference
- `Thrust`: GPU-side algorithm primitives
- `CUB`: lower-level parallel primitives
- `NCCL`: multi-GPU communication

### 5. Choose by question, not by fame

Start with `cuBLAS`, then `CUDA Graph`, and only then move to the optional extensions.

### 6. Async copy is an optimization tool, not a core dependency

Async copy helps overlap data transfer and kernel execution. It matters later, not at the first step.

## Minimal map

```text
cuBLAS  -> dense matmul / GEMM
CUDA Graph -> repeated launch overhead
cuDNN   -> optional deep-learning primitives
CUTLASS -> custom-kernel learning / extension
Thrust  -> GPU-side sort / reduction primitives
CUB     -> lower-level parallel primitives
NCCL    -> multi-GPU communication
```

## How to validate

- Explain what `cuBLAS` does in this tutorial
- Explain why `CUDA Graph` appears later
- Explain which libraries are optional
- Explain which chapter each library supports
- Explain why we look at one library before another

## Summary

If you can explain why each library appears on the route map, then the CUDA ecosystem is clear enough for the chapters that follow.
