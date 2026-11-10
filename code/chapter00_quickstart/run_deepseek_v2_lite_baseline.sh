#!/usr/bin/env bash
# Chapter 00：使用 llama.cpp 建立 DeepSeek-V2-Lite-Chat 外部推理基线。
# 边界：该入口只验证 llama.cpp 外部结果；教程自己的执行链从 Chapter 01 继续。
set -euo pipefail

chapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${chapter_dir}/../.." && pwd)"
model_path="${CPP_LLM_ENGINE_DEEPSEEK_V2_LITE_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}"
prompt="${1:-${DEEPSEEK_V2_LITE_PROMPT:-The capital of China is}}"
max_new_tokens="${2:-${DEEPSEEK_V2_LITE_MAX_NEW_TOKENS:-4}}"

if [[ ! -f "${model_path}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${model_path}" >&2
    echo "run: bash code/chapter00_quickstart/download_deepseek_v2_lite.sh" >&2
    exit 2
fi

CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
CHAPTER00_PROMPT="${prompt}" \
CHAPTER00_MAX_NEW_TOKENS="${max_new_tokens}" \
bash "${chapter_dir}/run_real_inference.sh" "${prompt}" "${max_new_tokens}"
