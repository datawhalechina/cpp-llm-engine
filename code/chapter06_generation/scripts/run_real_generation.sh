#!/usr/bin/env bash
# Chapter 06：验证 TinyLlama 单请求 Generation Core，不运行 batch 或 Serving。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CHAPTER06_REAL_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter06-real.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER06_REAL_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_tinyllama_generation_smoke --parallel 2
ctest --test-dir "${build_dir}" \
    -R '^real_tinyllama_generation_test$' --output-on-failure
