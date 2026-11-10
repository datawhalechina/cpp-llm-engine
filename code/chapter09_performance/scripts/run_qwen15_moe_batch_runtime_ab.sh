#!/usr/bin/env bash
# Chapter 09：比较 Qwen1.5-MoE 双请求完整 Decode 的逐行与分组执行。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
BUILD_DIR="${CPP_LLM_QWEN15_MOE_RUNTIME_AB_BUILD_DIR:-/tmp/cpp-llm-qwen15-moe-runtime-ab-build}"
RESULT_DIR="${CPP_LLM_QWEN15_MOE_RUNTIME_AB_RESULT_DIR:-/tmp/cpp-llm-qwen15-moe-runtime-ab-result}"
ITERATIONS="${CPP_LLM_QWEN15_MOE_RUNTIME_AB_ITERATIONS:-3}"

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
experiment_id=EXP-CPU-14
workload_id=WL-CPU-QWEN15-MOE-RUNTIME-BATCH-V1
model=Qwen1.5-MoE-A2.7B-Chat-Q3_K_M
backend=CPU
build_type=Release
batch=2
iterations=${ITERATIONS}
cache_layers=24
baseline=row-wise-moe
experiment=grouped-by-expert-moe
timing_scope=24-layer-decode-and-lm-head
EOF
uname -srmo >>"${RESULT_DIR}/metadata.txt"
"${CXX:-c++}" --version >>"${RESULT_DIR}/metadata.txt"

"${BUILD_DIR}/real_qwen15_moe_batch_runtime_benchmark" \
    "${MODEL_PATH}" "${ITERATIONS}" | tee "${RESULT_DIR}/paired.txt"
echo "qwen15_moe_batch_runtime_result_dir=${RESULT_DIR}"
