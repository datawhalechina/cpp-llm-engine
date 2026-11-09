#!/usr/bin/env bash
# Engine/Chapter 10：验证 Qwen3 CUDA Completion、SSE、批量调度和终态释放。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_QWEN_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"

CUDA_REAL_SERVING_BUILD_DIR="${CUDA_QWEN_SERVING_BUILD_DIR:-${CUDA_REAL_SERVING_BUILD_DIR:-}}" \
bash "${script_dir}/run_real_cuda_serving.sh" "${model_path}"
