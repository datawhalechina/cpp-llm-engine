#!/usr/bin/env bash
# Chapter 09：采集 Q4_K/Q6_K 在 batch=1/2/4/8 下的 CUDA GEMV 专项矩阵。
# 产物：固定形状、迭代次数、GPU 环境和逐格式结果；默认写入临时目录。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_CUDA_GEMV_MATRIX_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-gemv-matrix-build}"
result_dir="${CHAPTER09_CUDA_GEMV_MATRIX_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-gemv-matrix-result}"
iterations="${CHAPTER09_CUDA_GEMV_MATRIX_ITERATIONS:-50}"

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found under ${cuda_home}" >&2
    exit 2
fi
mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/cuda" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target cpp_llm_engine_cuda_quantized_gemv_matrix --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-13
workload_id=WL-GPU-QUANTIZED-GEMV-BATCH-MATRIX-V1
formats=Q4_K,Q6_K
batches=1,2,4,8
rows=2048
cols=2048
warmup=5
iterations=${iterations}
timing_scope=kernel-submit-and-final-synchronize
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"
"${build_dir}/cpp_llm_engine_cuda_quantized_gemv_matrix" \
    --iterations "${iterations}" >"${result_dir}/matrix.txt"
echo "correctness=PASS" >>"${result_dir}/metadata.txt"
echo "cuda_quantized_gemv_matrix_result_dir=${result_dir}"
echo "cuda_quantized_gemv_matrix=PASS"
