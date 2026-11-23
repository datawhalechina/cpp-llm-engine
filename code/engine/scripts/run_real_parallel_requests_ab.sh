#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${CPP_LLM_ENGINE_REAL_MODEL:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}}"
result_dir="${PARALLEL_AB_RESULT_DIR:-${repo_root}/code/chapter09_performance/results}"
execution_policy="${PARALLEL_AB_EXECUTION_POLICY:-optimized-avx2}"
warmup="${PARALLEL_AB_WARMUP:-1}"
iterations="${PARALLEL_AB_ITERATIONS:-3}"
build_dir="${PARALLEL_AB_BUILD_DIR:-/tmp/cpp-llm-engine-real-parallel-ab}"
if [[ "${execution_policy}" == "optimized-avx2" ]]; then
    policy_suffix="avx2"
else
    policy_suffix="${execution_policy//[^a-zA-Z0-9_-]/_}"
fi

run_mode() {
    local mode="$1"
    local output="$2"
    REAL_SERVING_BUILD_DIR="${build_dir}" \
    REAL_SERVING_RESULT_FILE="${output}" \
    REAL_SERVING_EXECUTION_POLICY="${execution_policy}" \
    REAL_SERVING_DISPATCH_MODE="${mode}" \
    REAL_SERVING_WARMUP="${warmup}" \
    REAL_SERVING_ITERATIONS="${iterations}" \
        bash "${script_dir}/run_real_serving_benchmark.sh" "${model_path}"
}

mkdir -p "${result_dir}"
sequential_result="${result_dir}/tinyllama_service_sequential_${policy_suffix}.json"
parallel_result="${result_dir}/tinyllama_service_parallel_requests_${policy_suffix}.json"
run_mode sequential "${sequential_result}"
run_mode parallel-requests "${parallel_result}"

echo "sequential_result=${sequential_result}"
echo "parallel_result=${parallel_result}"
