#!/usr/bin/env bash
# Chapter 09：采集 DeepSeek Q2_K/IQ4_NL 小 batch GEMV 基线与可选 NCU 原因指标。
# 口径：固定 2048x2048 微基准，只测量已通过 CPU 对照的 Adaptive kernel。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_DEEPSEEK2_GEMV_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-gemv-build}"
result_dir="${CHAPTER09_DEEPSEEK2_GEMV_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-gemv-result}"
iterations="${CHAPTER09_DEEPSEEK2_GEMV_ITERATIONS:-100}"
profile="${CHAPTER09_DEEPSEEK2_GEMV_PROFILE:-0}"

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found under ${cuda_home}" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "required command not found: nvidia-smi" >&2
    exit 2
fi
if [[ "${profile}" != "0" && "${profile}" != "1" ]]; then
    echo "CHAPTER09_DEEPSEEK2_GEMV_PROFILE must be 0 or 1" >&2
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
: >"${result_dir}/matrix.txt"
for format in q2 iq4; do
    "${build_dir}/cpp_llm_engine_cuda_quantized_gemv_matrix" \
        --format "${format}" --path adaptive --iterations "${iterations}" \
        >>"${result_dir}/matrix.txt"
done

if [[ "${profile}" == "1" ]]; then
    ncu_bin="${cuda_home}/bin/ncu"
    if [[ ! -x "${ncu_bin}" ]]; then
        echo "ncu not found under ${cuda_home}" >&2
        exit 2
    fi
    for format in q2 iq4; do
        for batch in 1 4; do
            "${ncu_bin}" --csv \
                --section ComputeWorkloadAnalysis \
                --section InstructionStats \
                --section MemoryWorkloadAnalysis \
                --section SchedulerStats \
                --section WarpStateStats \
                --kernel-name regex:quantized_gemv_kernel \
                --launch-count 1 \
                "${build_dir}/cpp_llm_engine_cuda_quantized_gemv_matrix" \
                --format "${format}" --batch "${batch}" \
                --iterations 1 --path adaptive \
                >"${result_dir}/ncu-${format}-b${batch}.csv" 2>&1
        done
    done
fi

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-22
workload_id=WL-GPU-DEEPSEEK2-QUANTIZED-GEMV-BASELINE-V1
formats=Q2_K,IQ4_NL
batches=1,2,4,8
rows=2048
cols=2048
path=adaptive
warmup=5
iterations=${iterations}
ncu_profile=${profile}
timing_scope=kernel-submit-and-final-synchronize
correctness=PASS
execution_status=PASS
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

echo "deepseek2_cuda_gemv_result_dir=${result_dir}"
echo "deepseek2_cuda_gemv_execution=PASS"
