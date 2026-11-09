#!/usr/bin/env bash
# Engine/Chapter 06：验证 Qwen3 CUDA Executor、Runtime、分页 K/V 与 Radix 前缀复用。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"

CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
CUDA_REAL_RUNTIME_BUILD_DIR="${CUDA_QWEN_RUNTIME_BUILD_DIR:-${CUDA_REAL_RUNTIME_BUILD_DIR:-}}" \
bash "${script_dir}/run_real_cuda_runtime.sh" "${model_path}"
