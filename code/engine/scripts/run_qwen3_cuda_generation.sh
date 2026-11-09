#!/usr/bin/env bash
# Engine/Chapter 06：验证 Qwen3 真实文本、CUDA Runtime 和请求 Context 隔离。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_QWEN_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"

CUDA_REAL_GENERATION_BUILD_DIR="${CUDA_QWEN_GENERATION_BUILD_DIR:-${CUDA_REAL_GENERATION_BUILD_DIR:-}}" \
bash "${script_dir}/run_real_cuda_generation.sh" \
    "${model_path}" "Hello Qwen3" "11,358,1366,311"
