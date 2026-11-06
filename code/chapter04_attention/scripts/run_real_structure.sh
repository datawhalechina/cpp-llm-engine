#!/usr/bin/env bash
# Chapter 04：验证 TinyLlama 首层真实权重结构，不执行量化 Linear。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
build_dir="${CHAPTER04_REAL_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter04-real-structure}"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/chapter01_tensor_loader" -B "${build_dir}" \
    -DCHAPTER01_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel 2 --target gguf_first_layer_inspect
ctest --test-dir "${build_dir}" \
    -R '^real_tinyllama_first_layer_structure_test$' \
    --output-on-failure
