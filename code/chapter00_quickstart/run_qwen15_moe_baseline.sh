#!/usr/bin/env bash
# Chapter 00：使用 llama.cpp 建立 Qwen1.5-MoE-A2.7B-Chat 外部推理基线。
# 边界：该入口只验证外部框架；教程自己的 MoE Decoder 使用后续章节入口。
set -euo pipefail

chapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${chapter_dir}/../.." && pwd)"
model_path="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
prompt="${1:-${QWEN15_MOE_PROMPT:-The capital of China is}}"
max_new_tokens="${2:-${QWEN15_MOE_MAX_NEW_TOKENS:-8}}"

if [[ ! -f "${model_path}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${model_path}" >&2
    echo "run: bash code/chapter00_quickstart/download_qwen15_moe.sh" >&2
    exit 2
fi

CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
CHAPTER00_PROMPT="${prompt}" \
CHAPTER00_MAX_NEW_TOKENS="${max_new_tokens}" \
bash "${chapter_dir}/run_real_inference.sh" "${prompt}" "${max_new_tokens}"
