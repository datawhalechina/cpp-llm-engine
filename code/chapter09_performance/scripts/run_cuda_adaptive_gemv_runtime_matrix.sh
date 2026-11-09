#!/usr/bin/env bash
# Chapter 09：采集 Adaptive GEMV 的 batch、上下文长度与 Decode 步数矩阵。
# 两轮共享相同 workload，但反转每个单元的路径执行顺序。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_CUDA_ADAPTIVE_MATRIX_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-adaptive-matrix-build}"
result_dir="${CHAPTER09_CUDA_ADAPTIVE_MATRIX_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-adaptive-matrix-result}"
batches="${CHAPTER09_CUDA_ADAPTIVE_MATRIX_BATCHES:-1,2,4,8}"
contexts="${CHAPTER09_CUDA_ADAPTIVE_MATRIX_CONTEXTS:-16,128}"
decode_steps="${CHAPTER09_CUDA_ADAPTIVE_MATRIX_DECODE_STEPS:-1,8}"
warmup="${CHAPTER09_CUDA_ADAPTIVE_MATRIX_WARMUP:-1}"
iterations="${CHAPTER09_CUDA_ADAPTIVE_MATRIX_ITERATIONS:-3}"

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
    --target real_tinyllama_cuda_adaptive_gemv_matrix --parallel 2

run_matrix() {
    local run="$1"
    local order="$2"
    echo "cuda_adaptive_gemv_matrix_run=${run} path_order=${order}" >&2
    "${build_dir}/real_tinyllama_cuda_adaptive_gemv_matrix" "${model_path}" \
        --batches "${batches}" --contexts "${contexts}" \
        --decode-steps "${decode_steps}" --warmup "${warmup}" \
        --iterations "${iterations}" --path-order "${order}" \
        >"${result_dir}/run-${run}.csv"
}

run_matrix 1 fixed-first
run_matrix 2 adaptive-first
node "${script_dir}/summarize_cuda_adaptive_gemv_matrix.mjs" \
    "${result_dir}" "${result_dir}/summary.json"

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-16
workload_id=WL-GPU-TINYLLAMA-ADAPTIVE-GEMV-RUNTIME-MATRIX-V1
model=TinyLlama-1.1B-Chat-v1.0.Q4_K_M
backend=CUDA
build_type=Release
batches=${batches}
context_tokens=${contexts}
decode_steps=${decode_steps}
warmup=${warmup}
iterations=${iterations}
independent_runs=2
execution_order=fixed-first,adaptive-first
timing_scope=excludes-model-load-context-create-request-bind-and-prefill
execution_status=PASS
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"
echo "cuda_adaptive_gemv_matrix_result_dir=${result_dir}"
echo "cuda_adaptive_gemv_matrix_execution=PASS"
