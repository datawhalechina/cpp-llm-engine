#!/usr/bin/env bash
# Chapter 09：在真实 TinyLlama Decode 中比较固定 tile=4 与自适应 GEMV。
# 门禁：两轮独立进程采用相反执行顺序，并由 benchmark 核对 logits 与 greedy token。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_CUDA_ADAPTIVE_GEMV_RUNTIME_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-adaptive-gemv-runtime-build}"
result_dir="${CHAPTER09_CUDA_ADAPTIVE_GEMV_RUNTIME_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-adaptive-gemv-runtime-result}"

for command in cmake node nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required command not found: ${command}" >&2
        exit 2
    fi
done
if [[ ! -x "${cuda_home}/bin/nvcc" || ! -f "${model_path}" ]]; then
    echo "CUDA toolkit or model is missing" >&2
    exit 2
fi

mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_CUDA_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_cuda_batch_benchmark --parallel 2

run_path() {
    local path="$1"
    local run="$2"
    "${build_dir}/real_tinyllama_cuda_batch_benchmark" "${model_path}" \
        --gemv-path "${path}" >"${result_dir}/${path}-run-${run}.txt"
}

run_path fixed-four 1
run_path adaptive 1
run_path adaptive 2
run_path fixed-four 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-15
workload_id=WL-GPU-TINYLLAMA-ADAPTIVE-GEMV-DECODE-AB-V1
model=TinyLlama-1.1B-Chat-v1.0.Q4_K_M
backend=CUDA
build_type=Release
requests=2
prompt_tokens=1,2
decode_steps=1
block_size=8
warmup=2
iterations=10
independent_runs=2
execution_order=fixed-adaptive,adaptive-fixed
baseline=fixed-four
experiment=adaptive-1-2-4
timing_scope=excludes-model-load-context-create-request-bind-and-prefill
correctness=PASS
execution_status=PASS
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"
node "${script_dir}/decide_cuda_adaptive_gemv.mjs" \
    "${result_dir}" "${result_dir}/decision.json"
echo "cuda_adaptive_gemv_runtime_result_dir=${result_dir}"
echo "cuda_adaptive_gemv_runtime_execution=PASS"
