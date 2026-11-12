#!/usr/bin/env bash
# Chapter 09: profile one grouped DeepSeek2 CUDA Decode after Prefill.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_CUDA_MOE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-cuda-moe-build}"
result_dir="${CHAPTER09_DEEPSEEK2_CUDA_MOE_PROFILE_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-cuda-moe-profile}"
profile_base="${result_dir}/grouped-decode"

if [[ ! -x "${cuda_home}/bin/nvcc" ||
      ! -x "${cuda_home}/bin/nsys" || ! -f "${model_path}" ]]; then
    echo "CUDA Toolkit, Nsight Systems, or model is missing" >&2
    exit 2
fi

mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${model_path}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_deepseek2_cuda_grouped_moe_benchmark --parallel 2

"${cuda_home}/bin/nsys" profile \
    --trace=cuda,osrt \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="${profile_base}" \
    "${build_dir}/real_deepseek2_cuda_grouped_moe_benchmark" \
    "${model_path}" --profile-grouped >"${result_dir}/program-output.txt"

"${cuda_home}/bin/nsys" stats --force-export=true \
    --report cuda_gpu_kern_sum --format csv \
    "${profile_base}.nsys-rep" >"${result_dir}/kernel-summary.csv"
"${cuda_home}/bin/nsys" stats --force-export=true \
    --report cuda_api_sum --format csv \
    "${profile_base}.nsys-rep" >"${result_dir}/cuda-api-summary.csv"
grep -q 'profile_correctness: PASS' "${result_dir}/program-output.txt"
echo "deepseek2_cuda_grouped_moe_profile_result_dir=${result_dir}"
echo "deepseek2_cuda_grouped_moe_profile=PASS"
