#!/usr/bin/env bash
# 真实 TinyLlama sequential/scalar batch 配对 benchmark。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
if [[ -n "${LLAMA_CPP_BUILD_DIR:-}" ]]; then
    llama_build_dir="${LLAMA_CPP_BUILD_DIR}"
elif [[ -d "${llama_root}/build-cpu/bin" ]]; then
    llama_build_dir="${llama_root}/build-cpu"
else
    llama_build_dir="${llama_root}/build"
fi
build_dir="${REAL_BATCH_BENCHMARK_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-engine-real-batch-benchmark}"
result_path="${REAL_BATCH_BENCHMARK_RESULT:-${repo_root}/code/chapter07_benchmark/results/tinyllama_real_batch_scalar.json}"
warmup="${REAL_BATCH_BENCHMARK_WARMUP:-1}"
iterations="${REAL_BATCH_BENCHMARK_ITERATIONS:-3}"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_tinyllama_batch_benchmark -j2
mkdir -p "$(dirname "${result_path}")"
"${build_dir}/real_tinyllama_batch_benchmark" "${model_path}" \
    --warmup "${warmup}" \
    --iterations "${iterations}" \
    --output "${result_path}"
