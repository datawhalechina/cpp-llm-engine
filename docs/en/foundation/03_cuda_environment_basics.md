# F03 CUDA and Environment Basics

## Goal

Understand the minimum CUDA environment knowledge needed by the tutorial.

## What you should be able to do after reading

- Know what CUDA is used for in this project
- Understand GPU, driver, toolkit, and runtime at a high level
- Know when CPU fallback is acceptable
- Understand basic device checks and environment variables

## Why this matters

Later chapters use CUDA as the acceleration path for matrix multiplication and model execution. To understand that path, readers need to know what lives in the driver, what lives in the toolkit, and what happens when GPU access is not available.

## What comes next

- cuBLAS-based MatMul
- GPU acceleration paths
- CUDA compilation and debugging
- CPU fallback as a valid baseline

## Core ideas

### 1. Driver and toolkit are different layers

The driver makes the GPU visible and usable to the system. The toolkit provides the compiler, headers, and libraries needed to build CUDA code.

### 2. GPU memory and device selection matter

Weights and activations live in device memory. The chosen device must match both the build-time and run-time assumptions.

### 3. Build flags control backend choice

Build flags decide whether a project follows a CPU-only or CUDA-enabled path.

This tutorial uses a generic `ENABLE_CUDA` style for its own examples. External references such as `llama.cpp` may use their own names, such as `GGML_CUDA=ON`.

### 4. Run-time checks are better than “it looks installed”

`nvidia-smi` and `nvcc --version` are useful, but they only prove that the driver and compiler are present. To know whether the runtime can actually use the GPU, you should check whether the program can enumerate devices and query device properties.

```cpp
#include <cuda_runtime.h>
#include <iostream>

bool check_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cerr << "CUDA not available: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    return true;
}
```

If CUDA samples are installed, `deviceQuery` is also a useful reference point for run-time availability.

`CUDA_VISIBLE_DEVICES` is helpful on multi-GPU machines:

```bash
CUDA_VISIBLE_DEVICES=0 ./my_program
CUDA_VISIBLE_DEVICES="" ./my_program
```

## CPU-first chapters

These chapters are intentionally designed to be CPU-friendly:

- `chapter00_quickstart`
- `chapter01_tensor_loader`
- `chapter02_tensor_ops`
- `chapter04_attention`
- `chapter06_generation`
- `chapter05_quantization`
- `chapter07_benchmark`

CUDA becomes necessary when you want to compare backends or optimize the execution path.

## Minimal example

```bash
nvidia-smi
nvcc --version

cmake -B build -DCMAKE_BUILD_TYPE=Release
# or
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
```

This example shows:

- the system can see the GPU
- the CUDA compiler is installed
- the build path can switch between CPU-only and CUDA-enabled
- a run-time check is still needed

## How to validate

- Confirm the system can see the GPU
- Confirm the current build path matches the intended backend
- Confirm the tutorial still runs on CPU when CUDA is absent
- Confirm the program can enumerate devices at run time
- Confirm `CUDA_VISIBLE_DEVICES` changes the observed behavior

## Summary

This page explains the CUDA background needed by the later chapters. If you can explain driver, toolkit, and runtime, and tell whether a failure belongs to CPU logic or CUDA configuration, the environment layer is good enough.
