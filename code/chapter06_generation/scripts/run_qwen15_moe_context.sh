#!/usr/bin/env bash
# Chapter 06：验证 Qwen1.5-MoE 真实模型请求 Context 的 Prefill/Decode 连续性。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
BUILD_DIR="${CPP_LLM_QWEN15_MOE_CONTEXT_BUILD_DIR:-/tmp/cpp-llm-qwen15-moe-context-build}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${MODEL_PATH}" >&2
    echo "run: bash code/chapter00_quickstart/download_qwen15_moe.sh" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization"
cmake --build "${BUILD_DIR}" --target qwen15_moe_context_smoke -j2
"${BUILD_DIR}/qwen15_moe_context_smoke" "${MODEL_PATH}"
