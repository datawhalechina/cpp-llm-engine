#!/usr/bin/env bash
# Chapter 09：比较真实 Qwen1.5-MoE 第一层逐行与按专家分组执行。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
BUILD_DIR="${CPP_LLM_QWEN15_MOE_BATCH_BUILD_DIR:-/tmp/cpp-llm-qwen15-moe-batch-build}"
RESULT_DIR="${CPP_LLM_QWEN15_MOE_BATCH_RESULT_DIR:-/tmp/cpp-llm-qwen15-moe-batch-result}"
BATCH="${CPP_LLM_QWEN15_MOE_BATCH_SIZE:-2}"
WARMUP="${CPP_LLM_QWEN15_MOE_BATCH_WARMUP:-1}"
ITERATIONS="${CPP_LLM_QWEN15_MOE_BATCH_ITERATIONS:-5}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${MODEL_PATH}" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
    --target qwen15_moe_grouped_batch_benchmark -j2

mkdir -p "${RESULT_DIR}"
cat >"${RESULT_DIR}/metadata.txt" <<EOF
experiment_id=EXP-CPU-13
workload_id=WL-CPU-QWEN15-MOE-LAYER0-BATCH-V1
model=Qwen1.5-MoE-A2.7B-Chat-Q3_K_M
backend=CPU
build_type=Release
layer=0
batch=${BATCH}
warmup=${WARMUP}
iterations=${ITERATIONS}
baseline=row-wise
experiment=grouped-by-expert
timing_scope=router-experts-shared-expert
EOF
uname -srmo >>"${RESULT_DIR}/metadata.txt"
"${CXX:-c++}" --version >>"${RESULT_DIR}/metadata.txt"

"${BUILD_DIR}/qwen15_moe_grouped_batch_benchmark" \
    "${MODEL_PATH}" "${BATCH}" "${WARMUP}" "${ITERATIONS}" \
    | tee "${RESULT_DIR}/paired.txt"
echo "qwen15_moe_grouped_batch_result_dir=${RESULT_DIR}"
