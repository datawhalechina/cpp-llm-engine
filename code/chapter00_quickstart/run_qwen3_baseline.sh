#!/usr/bin/env bash
# Chapter 00：使用 llama.cpp 建立 Qwen3-0.6B 外部推理基线。
# 边界：该入口只验证固定 GGUF；教程自己的执行链从 Chapter 01 继续。
set -euo pipefail

chapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${chapter_dir}/../.." && pwd)"
model_path="${CPP_LLM_ENGINE_QWEN3_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}"
prompt="${1:-${QWEN3_PROMPT:-Hello Qwen3}}"
max_new_tokens="${2:-${QWEN3_MAX_NEW_TOKENS:-4}}"

if [[ ! -f "${model_path}" ]]; then
    echo "Qwen3 GGUF not found: ${model_path}" >&2
    echo "run: bash code/chapter00_quickstart/download_qwen3.sh" >&2
    exit 2
fi

CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
CHAPTER00_PROMPT="${prompt}" \
CHAPTER00_MAX_NEW_TOKENS="${max_new_tokens}" \
bash "${chapter_dir}/run_real_inference.sh" "${prompt}" "${max_new_tokens}"
