# Reading Guide

Complete the CPU path first, then choose CUDA or multimodal extensions. For each chapter, read the introduction and concept pages before opening its code README, scripts, and tests.

## Start Here

1. Use [Chapter00](/chapter00_quickstart/intro) to establish a real TinyLlama baseline with llama.cpp.
2. Visit [Foundation](/foundation/intro) only when you need C++, CMake, testing, or CUDA prerequisites.
3. Continue through Chapter01–10 from model input to execution, runtime, cache, scheduling, performance, and serving.

| Stage | Chapters | Result |
| --- | --- | --- |
| Input and execution | Chapter01–03 | Model, Tensor/View, Operator, and Backend |
| Model and generation | Chapter04–06 | Decoder, quantized Linear, Request, and Generation Runtime |
| Cache and scheduling | Chapter07–08 | Paged K/V, Radix Cache, Scheduler, and tensor batches |
| Evidence and serving | Chapter09–10 | Benchmark, Profiling, ServiceCore, and HTTP/SSE |

## Choose an Extension

| Path | Prerequisite | Entry |
| --- | --- | --- |
| CPU main line | C++17, CMake 3.20+, and Git | [Chapter00](/chapter00_quickstart/intro) |
| GPU extension | Stable CPU contracts, NVIDIA Driver, and CUDA Toolkit | [CUDA setup](/guides/setup_cuda) |
| Multimodal extension | Decoder and Runtime baseline | [Multimodal guide](/guides/multimodal_inference) |

Linux is the default environment. On Windows, use WSL2 Ubuntu. CUDA is optional and is not required for the CPU path.

## Run and Verify

Use the `run.sh` or `scripts/run.sh` entry under each chapter. Start with the dependency-free regression:

```bash
npm install
npm run test:regression
```

This regression does not replace real GGUF, CUDA device, network transport, or performance experiments. Use [Guides](/guides/README) for cross-chapter workflows and [Reference](/reference/README) for terms, formulas, assets, and framework coordinates.
