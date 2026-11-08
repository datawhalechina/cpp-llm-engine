#!/usr/bin/env bash
# Engine/Chapter 06：验证真实文本经过 tokenizer、CUDA Runtime 和 detokenizer。
# 默认使用仓库内 llama.cpp build、TinyLlama GGUF 和 CUDA 12.8。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CUDA_REAL_GENERATION_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-cuda-real-generation.XXXXXX")}"
cleanup_build=false
if [[ -z "${CUDA_REAL_GENERATION_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found: ${cuda_home}/bin/nvcc" >&2
    exit 2
fi
if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi
if [[ ! -f "${llama_root}/include/llama.h" ||
      ! -d "${llama_build_dir}/bin" ]]; then
    echo "llama.cpp tokenizer build not found: ${llama_build_dir}/bin" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_CUDA_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_cuda_generation_smoke --parallel 2
ctest --test-dir "${build_dir}" \
    -R '^real_tinyllama_cuda_generation_smoke$' --output-on-failure
