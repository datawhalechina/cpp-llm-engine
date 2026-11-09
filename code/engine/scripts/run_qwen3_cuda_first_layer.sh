#!/usr/bin/env bash
# Engine/Chapter 04：验证 Qwen3-0.6B CUDA 首层与 CPU 参考结果。
# 默认读取仓库内 Qwen3 GGUF；可用第一个参数覆盖模型路径。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_QWEN_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"

exec bash "${script_dir}/run_real_cuda_first_layer.sh" "${model_path}"
