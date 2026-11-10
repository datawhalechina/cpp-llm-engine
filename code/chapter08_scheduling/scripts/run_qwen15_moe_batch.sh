#!/usr/bin/env bash
# Chapter 08：构建并验证 Qwen1.5-MoE 双请求 Continuous Batch。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
BUILD_DIR="${CHAPTER08_QWEN15_MOE_BUILD_DIR:-/tmp/cpp-llm-chapter08-qwen15-moe-build}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${MODEL_PATH}" >&2
    exit 2
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_QWEN15_MOE_MODEL="${MODEL_PATH}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
    --target real_qwen15_moe_scheduling_smoke -j2
ctest --test-dir "${BUILD_DIR}" \
    -R '^real_qwen15_moe_scheduling_smoke$' --output-on-failure
