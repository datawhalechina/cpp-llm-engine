#!/usr/bin/env bash
# Chapter 10：验证 DeepSeek-V2-Lite 动态准入、请求组批和终态释放。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}"
BUILD_DIR="${CPP_LLM_DEEPSEEK2_SERVICE_BUILD_DIR:-/tmp/cpp-llm-deepseek2-service-build}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${MODEL_PATH}" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${MODEL_PATH}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target \
    real_deepseek2_service_smoke real_deepseek2_http_adapter_smoke -j2
ctest --test-dir "${BUILD_DIR}" \
    -R '^real_deepseek2_(service|http_adapter)_smoke$' --output-on-failure
