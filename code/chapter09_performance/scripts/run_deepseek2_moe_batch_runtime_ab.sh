#!/usr/bin/env bash
# Chapter 09：比较 DeepSeek-V2-Lite 双请求完整 Decode 的逐行与分组 MoE。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}"
BUILD_DIR="${CPP_LLM_DEEPSEEK2_MOE_AB_BUILD_DIR:-/tmp/cpp-llm-deepseek2-moe-ab-build}"
RESULT_DIR="${CPP_LLM_DEEPSEEK2_MOE_AB_RESULT_DIR:-/tmp/cpp-llm-deepseek2-moe-ab-result}"
ITERATIONS="${CPP_LLM_DEEPSEEK2_MOE_AB_ITERATIONS:-5}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${MODEL_PATH}" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
    --target real_deepseek2_moe_batch_runtime_benchmark -j2

mkdir -p "${RESULT_DIR}"
cat >"${RESULT_DIR}/metadata.txt" <<EOF
experiment_id=EXP-CPU-19
workload_id=WL-CPU-DEEPSEEK2-MOE-RUNTIME-BATCH-V1
model=DeepSeek-V2-Lite-Chat-Q2_K
backend=CPU
build_type=Release
batch=2
iterations=${ITERATIONS}
cache_layers=27
baseline=row-wise-moe
experiment=grouped-by-expert-moe
timing_scope=27-layer-decode-and-lm-head
EOF
uname -srmo >>"${RESULT_DIR}/metadata.txt"
"${CXX:-c++}" --version >>"${RESULT_DIR}/metadata.txt"

"${BUILD_DIR}/real_deepseek2_moe_batch_runtime_benchmark" \
    "${MODEL_PATH}" "${ITERATIONS}" | tee "${RESULT_DIR}/paired.txt"
echo "deepseek2_moe_batch_runtime_result_dir=${RESULT_DIR}"
