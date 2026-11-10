#!/usr/bin/env bash
# Engine/Chapter 06：运行 Qwen1.5-MoE CPU 文本生成闭环。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
LLAMA_ROOT="${LLAMA_CPP_ROOT:-${ROOT_DIR}/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${LLAMA_ROOT}/build}"
BUILD_DIR="${CPP_LLM_QWEN15_MOE_GENERATION_BUILD_DIR:-/tmp/cpp-llm-qwen15-moe-generation-build}"
PROMPT="${1:-Hello}"

if [[ ! -f "${MODEL_PATH}" || ! -d "${LLAMA_BUILD_DIR}/bin" ]]; then
    echo "Qwen1.5-MoE model or llama.cpp tokenizer build is missing" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${LLAMA_ROOT}" \
    -DLLAMA_CPP_BUILD_DIR="${LLAMA_BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target real_qwen15_moe_generation_smoke -j2
"${BUILD_DIR}/real_qwen15_moe_generation_smoke" "${MODEL_PATH}" "${PROMPT}"
