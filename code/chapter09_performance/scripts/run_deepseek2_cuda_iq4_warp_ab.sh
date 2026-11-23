#!/usr/bin/env bash
# Chapter 09：比较 IQ4_NL 通用 256 线程归约与单 Warp 专用归约。
# 门禁：先运行 CPU/CUDA 正确性对照，再做两轮反向顺序配对。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_DEEPSEEK2_IQ4_WARP_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-iq4-warp-build}"
result_dir="${CHAPTER09_DEEPSEEK2_IQ4_WARP_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-iq4-warp-result}"
iterations="${CHAPTER09_DEEPSEEK2_IQ4_WARP_ITERATIONS:-100}"
profile="${CHAPTER09_DEEPSEEK2_IQ4_WARP_PROFILE:-0}"

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found under ${cuda_home}" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "required command not found: nvidia-smi" >&2
    exit 2
fi
if [[ "${profile}" != "0" && "${profile}" != "1" ]]; then
    echo "CHAPTER09_DEEPSEEK2_IQ4_WARP_PROFILE must be 0 or 1" >&2
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
        --format iq4 --path "${path}" --iterations "${iterations}" \
        >"${result_dir}/${path}-run-${run}.txt"
}
run_path adaptive 1
run_path iq4-warp 1
run_path iq4-warp 2
run_path adaptive 2

if [[ "${profile}" == "1" ]]; then
    ncu_bin="${cuda_home}/bin/ncu"
    if [[ ! -x "${ncu_bin}" ]]; then
        echo "ncu not found under ${cuda_home}" >&2
        exit 2
    fi
    for batch in 1 4; do
        for path in adaptive iq4-warp; do
            "${ncu_bin}" --csv \
                --section ComputeWorkloadAnalysis \
                --section InstructionStats \
                --section MemoryWorkloadAnalysis \
                --section SchedulerStats \
                --section WarpStateStats \
                --kernel-name regex:gemv_kernel \
                --launch-count 1 \
                "${build_dir}/cpp_llm_engine_cuda_quantized_gemv_matrix" \
                --format iq4 --batch "${batch}" --iterations 1 --path "${path}" \
                >"${result_dir}/ncu-b${batch}-${path}.csv" 2>&1
        done
    done
fi

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-23
workload_id=WL-GPU-DEEPSEEK2-IQ4-WARP-AB-V1
format=IQ4_NL
batches=1,2,4,8
rows=2048
cols=2048
warmup=5
iterations=${iterations}
independent_runs=2
execution_order=adaptive-iq4-warp,iq4-warp-adaptive
baseline=adaptive-256-thread
experiment=adaptive-iq4-single-warp
ncu_profile=${profile}
timing_scope=kernel-submit-and-final-synchronize
correctness=PASS
execution_status=PASS
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

echo "deepseek2_cuda_iq4_warp_result_dir=${result_dir}"
echo "deepseek2_cuda_iq4_warp_execution=PASS"
