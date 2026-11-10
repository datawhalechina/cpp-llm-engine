#!/usr/bin/env bash
# Chapter 09：检查 MoE 分组执行在不同 batch 与 Decode 步数下的适用范围。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
BUILD_DIR="${CPP_LLM_QWEN15_MOE_MATRIX_BUILD_DIR:-/tmp/cpp-llm-qwen15-moe-runtime-matrix-build}"
RESULT_DIR="${CPP_LLM_QWEN15_MOE_MATRIX_RESULT_DIR:-/tmp/cpp-llm-qwen15-moe-runtime-matrix-result}"
BATCHES="${CPP_LLM_QWEN15_MOE_MATRIX_BATCHES:-1,2,4}"
DECODE_STEPS="${CPP_LLM_QWEN15_MOE_MATRIX_DECODE_STEPS:-1}"
ITERATIONS="${CPP_LLM_QWEN15_MOE_MATRIX_ITERATIONS:-2}"
EXPERIMENT_ID="${CPP_LLM_QWEN15_MOE_MATRIX_EXPERIMENT_ID:-EXP-CPU-15}"
WORKLOAD_ID="${CPP_LLM_QWEN15_MOE_MATRIX_WORKLOAD_ID:-WL-CPU-QWEN15-MOE-RUNTIME-MATRIX-V1}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${MODEL_PATH}" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
    --target real_qwen15_moe_batch_runtime_benchmark -j2

mkdir -p "${RESULT_DIR}"
cat >"${RESULT_DIR}/metadata.txt" <<EOF
experiment_id=${EXPERIMENT_ID}
workload_id=${WORKLOAD_ID}
model=Qwen1.5-MoE-A2.7B-Chat-Q3_K_M
backend=CPU
build_type=Release
batches=${BATCHES}
decode_steps=${DECODE_STEPS}
iterations=${ITERATIONS}
cache_layers=24
baseline=row-wise-moe
experiment=grouped-by-expert-moe
timing_scope=24-layer-decode-and-lm-head
EOF
uname -srmo >>"${RESULT_DIR}/metadata.txt"
"${CXX:-c++}" --version >>"${RESULT_DIR}/metadata.txt"

"${BUILD_DIR}/real_qwen15_moe_batch_runtime_benchmark" \
    "${MODEL_PATH}" "${ITERATIONS}" "${BATCHES}" "${DECODE_STEPS}" \
    | tee "${RESULT_DIR}/matrix.txt"
echo "qwen15_moe_batch_runtime_matrix_result_dir=${RESULT_DIR}"
