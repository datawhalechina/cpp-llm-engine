#!/usr/bin/env bash
# Engine/Chapter06：运行 DeepSeek-V2-Lite CPU 文本生成闭环。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}"
LLAMA_ROOT="${LLAMA_CPP_ROOT:-${ROOT_DIR}/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${LLAMA_ROOT}/build}"
BUILD_DIR="${CPP_LLM_DEEPSEEK2_GENERATION_BUILD_DIR:-/tmp/cpp-llm-deepseek2-generation-build}"
PROMPT="${1:-Hello}"

if [[ ! -f "${MODEL_PATH}" || ! -d "${LLAMA_BUILD_DIR}/bin" ]]; then
    echo "DeepSeek-V2-Lite model or llama.cpp tokenizer build is missing" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${LLAMA_ROOT}" \
    -DLLAMA_CPP_BUILD_DIR="${LLAMA_BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target real_deepseek2_generation_smoke -j2
"${BUILD_DIR}/real_deepseek2_generation_smoke" "${MODEL_PATH}" "${PROMPT}"
