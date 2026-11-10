#!/usr/bin/env bash
# Chapter 09：比较固定请求队列在 Chapter08 调度中的逐行与分组 MoE。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${CPP_LLM_ENGINE_QWEN15_MOE_MODEL_PATH:-${ROOT_DIR}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}"
BUILD_DIR="${CPP_LLM_QWEN15_MOE_SCHEDULING_BUILD_DIR:-/tmp/cpp-llm-qwen15-moe-scheduling-ab-build}"
RESULT_DIR="${CPP_LLM_QWEN15_MOE_SCHEDULING_RESULT_DIR:-/tmp/cpp-llm-qwen15-moe-scheduling-ab-result}"
REQUESTS="${CPP_LLM_QWEN15_MOE_SCHEDULING_REQUESTS:-4}"
NEW_TOKENS="${CPP_LLM_QWEN15_MOE_SCHEDULING_NEW_TOKENS:-4}"
ITERATIONS="${CPP_LLM_QWEN15_MOE_SCHEDULING_ITERATIONS:-2}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${MODEL_PATH}" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}/code/engine" -B "${BUILD_DIR}" \
    -DCHAPTER05_SOURCE_DIR="${ROOT_DIR}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
    --target real_qwen15_moe_scheduling_benchmark -j2

mkdir -p "${RESULT_DIR}"
cat >"${RESULT_DIR}/metadata.txt" <<EOF
experiment_id=EXP-CPU-17
workload_id=WL-CPU-QWEN15-MOE-SCHEDULING-V1
model=Qwen1.5-MoE-A2.7B-Chat-Q3_K_M
backend=CPU
build_type=Release
requests=${REQUESTS}
new_tokens=${NEW_TOKENS}
iterations=${ITERATIONS}
cache_layers=24
baseline=row-wise-moe
experiment=grouped-by-expert-moe
timing_scope=chapter08-post-prefill-scheduling-and-decode
admission=fixed-cohort
EOF
uname -srmo >>"${RESULT_DIR}/metadata.txt"
"${CXX:-c++}" --version >>"${RESULT_DIR}/metadata.txt"

"${BUILD_DIR}/real_qwen15_moe_scheduling_benchmark" \
    "${MODEL_PATH}" "${REQUESTS}" "${NEW_TOKENS}" "${ITERATIONS}" \
    | tee "${RESULT_DIR}/paired.txt"
echo "qwen15_moe_scheduling_result_dir=${RESULT_DIR}"
