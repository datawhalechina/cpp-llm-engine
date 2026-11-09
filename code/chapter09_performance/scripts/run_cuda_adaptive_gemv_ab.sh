#!/usr/bin/env bash
# Chapter 09：比较固定 tile=4 与自适应 tile 的 Q4_K/Q6_K GEMV 矩阵。
# 门禁：先运行 CUDA 正确性测试，再在同一构建中采集两条路径。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_CUDA_ADAPTIVE_GEMV_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-adaptive-gemv-build}"
result_dir="${CHAPTER09_CUDA_ADAPTIVE_GEMV_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-adaptive-gemv-result}"
iterations="${CHAPTER09_CUDA_ADAPTIVE_GEMV_ITERATIONS:-100}"
profile="${CHAPTER09_CUDA_ADAPTIVE_GEMV_PROFILE:-0}"

for command in node nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required command not found: ${command}" >&2
        exit 2
    fi
done
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found under ${cuda_home}" >&2
    exit 2
fi
mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/cuda" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target cpp_llm_engine_cuda_quantized_linear_test \
             cpp_llm_engine_cuda_quantized_gemv_matrix --parallel 2

"${build_dir}/cpp_llm_engine_cuda_quantized_linear_test" \
    >"${result_dir}/correctness.txt"
run_path() {
    local path="$1"
    local run="$2"
    "${build_dir}/cpp_llm_engine_cuda_quantized_gemv_matrix" \
        --iterations "${iterations}" --path "${path}" \
        >"${result_dir}/${path}-run-${run}.txt"
}

run_path fixed-four 1
run_path adaptive 1
run_path adaptive 2
run_path fixed-four 2

if [[ "${profile}" == "1" ]]; then
    ncu_bin="${cuda_home}/bin/ncu"
    if [[ ! -x "${ncu_bin}" ]]; then
        echo "ncu not found under ${cuda_home}" >&2
        exit 2
    fi
    for format in q4 q6; do
        for batch in 1 2; do
            for path in fixed-four adaptive; do
                "${ncu_bin}" --set basic --csv \
                    --section ComputeWorkloadAnalysis \
                    --section InstructionStats \
                    --section MemoryWorkloadAnalysis \
                    --section SchedulerStats \
                    --section WarpStateStats \
                    --kernel-name regex:quantized_gemv_kernel \
                    --launch-count 1 \
                    "${build_dir}/cpp_llm_engine_cuda_quantized_gemv_matrix" \
                    --format "${format}" --batch "${batch}" \
                    --iterations 1 --path "${path}" \
                    >"${result_dir}/ncu-${format}-b${batch}-${path}.csv" 2>&1
            done
        done
    done
    node "${script_dir}/summarize_cuda_adaptive_gemv_ncu.mjs" \
        "${result_dir}" "${result_dir}/ncu-summary.json"
fi

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-14
workload_id=WL-GPU-QUANTIZED-GEMV-ADAPTIVE-TILE-AB-V1
formats=Q4_K,Q6_K
batches=1,2,4,8
rows=2048
cols=2048
warmup=5
iterations=${iterations}
independent_runs=2
execution_order=fixed-adaptive,adaptive-fixed
ncu_profile=${profile}
baseline=fixed-four
experiment=adaptive-1-2-4
timing_scope=kernel-submit-and-final-synchronize
correctness=PASS
execution_status=PASS
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"
echo "cuda_adaptive_gemv_result_dir=${result_dir}"
echo "cuda_adaptive_gemv_execution=PASS"
