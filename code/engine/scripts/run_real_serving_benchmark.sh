#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${CPP_LLM_ENGINE_REAL_MODEL:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}}"
build_dir="${REAL_SERVING_BUILD_DIR:-/tmp/cpp-llm-engine-real-serving-benchmark}"
result_file="${REAL_SERVING_RESULT_FILE:-${repo_root}/code/chapter09_serving/results/tinyllama_service_cpu_baseline.json}"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi

mkdir -p "$(dirname "${result_file}")"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${repo_root}/llama.cpp" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_tinyllama_serving_benchmark -j2
"${build_dir}/real_tinyllama_serving_benchmark" "${model_path}" \
    --warmup "${REAL_SERVING_WARMUP:-1}" \
    --iterations "${REAL_SERVING_ITERATIONS:-3}" \
    --execution-policy "${REAL_SERVING_EXECUTION_POLICY:-scalar}" \
    --dispatch-mode "${REAL_SERVING_DISPATCH_MODE:-sequential}" \
    --output "${result_file}"
