#!/usr/bin/env bash
# Engine/Chapter 04：构建真实 TinyLlama layer-0，并核对 CPU/CUDA hidden 与 K/V。
# 默认自动寻找仓库内模型和 CUDA 12.8；环境变量可覆盖模型、Toolkit 和构建目录。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CUDA_REAL_LAYER_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-cuda-real-layer.XXXXXX")}"
cleanup_build=false
if [[ -z "${CUDA_REAL_LAYER_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found: ${cuda_home}/bin/nvcc" >&2
    echo "set CUDA_HOME to a compatible CUDA Toolkit" >&2
    exit 2
fi
if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    echo "set CPP_LLM_ENGINE_MODEL_PATH to a TinyLlama GGUF file" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_CUDA_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_cuda_first_layer_test --parallel 2
ctest --test-dir "${build_dir}" \
    -R '^real_tinyllama_cuda_first_layer_test$' --output-on-failure
