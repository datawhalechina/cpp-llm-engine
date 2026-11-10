#!/usr/bin/env bash
# Chapter08：构建并验证 DeepSeek-V2-Lite 双请求 Continuous Batch。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}"
BUILD_DIR="${CHAPTER08_DEEPSEEK2_BUILD_DIR:-/tmp/cpp-llm-chapter08-deepseek2-build}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${MODEL_PATH}" >&2
    exit 2
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${MODEL_PATH}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
    --target real_deepseek2_scheduling_smoke -j2
ctest --test-dir "${BUILD_DIR}" \
    -R '^real_deepseek2_scheduling_smoke$' --output-on-failure
