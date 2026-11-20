#!/usr/bin/env bash
# Chapter 09: attribute DeepSeek2 CUDA Tensor Prefill at two prompt lengths.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-tensor-prefill-build}"
result_dir="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_PROFILE_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-tensor-prefill-profile}"
lengths="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_PROFILE_LENGTHS:-32,128}"
mla_path="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_PROFILE_MLA_PATH:-absorbed}"

case "${mla_path}" in
    legacy)
        experiment_id="EXP-GPU-19"
        attachment="deepseek2-tensor-prefill-length-profile"
        ;;
    absorbed)
        experiment_id="EXP-GPU-20"
        attachment="deepseek2-absorbed-mla-post-profile"
        ;;
    *)
        echo "MLA path must be legacy or absorbed: ${mla_path}" >&2
        exit 2
        ;;
esac

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
    --target real_deepseek2_cuda_tensor_prefill_benchmark --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=${experiment_id}
attachment=${attachment}
workload_id=WL-GPU-DEEPSEEK2-TENSOR-PREFILL-PROFILE-V1
model=DeepSeek-V2-Lite-Chat-Q2_K
backend=CUDA
build_type=Release
profile_lengths=${lengths}
mla_path=${mla_path}
capture_range=one-complete-tensor-prefill
cache=continuous-compressed-mla
timing_scope=27-layer-prefill-and-lm-head
excludes=model-load-context-create-and-correctness-reference
EOF
"${cuda_home}/bin/nsys" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

IFS=',' read -r -a profile_lengths <<<"${lengths}"
for length in "${profile_lengths[@]}"; do
    if [[ ! "${length}" =~ ^[1-9][0-9]*$ ]]; then
        echo "profile length must be a positive integer: ${length}" >&2
        exit 2
    fi
    profile_base="${result_dir}/tensor-prefill-${length}"
    "${cuda_home}/bin/nsys" profile \
        --trace=cuda,osrt \
        --sample=none \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        --output="${profile_base}" \
        "${build_dir}/real_deepseek2_cuda_tensor_prefill_benchmark" \
        "${model_path}" --profile-tensor "${length}" "${mla_path}" \
        >"${result_dir}/program-output-${length}.txt"

    "${cuda_home}/bin/nsys" stats --force-export=true \
        --report cuda_gpu_kern_sum --format csv \
        "${profile_base}.nsys-rep" |
        sed -n '/^Time (%)/,$p' |
        sed -e 's/[[:space:]]*$//' -e '/^$/d' \
            >"${result_dir}/kernel-summary-${length}.csv"
    "${cuda_home}/bin/nsys" stats --force-export=true \
        --report cuda_api_sum --format csv \
        "${profile_base}.nsys-rep" |
        sed -n '/^Time (%)/,$p' |
        sed -e 's/[[:space:]]*$//' -e '/^$/d' \
            >"${result_dir}/cuda-api-summary-${length}.csv"
    "${cuda_home}/bin/nsys" stats --force-export=true \
        --report cuda_gpu_mem_time_sum --format csv \
        "${profile_base}.nsys-rep" |
        sed -n '/^Time (%)/,$p' |
        sed -e 's/[[:space:]]*$//' -e '/^$/d' \
            >"${result_dir}/memory-time-summary-${length}.csv"
    "${cuda_home}/bin/nsys" stats --force-export=true \
        --report cuda_gpu_mem_size_sum --format csv \
        "${profile_base}.nsys-rep" |
        sed -n '/^Total/,$p' |
        sed -e 's/[[:space:]]*$//' -e '/^$/d' \
            >"${result_dir}/memory-size-summary-${length}.csv"
    grep -q 'profile_correctness: PASS' \
        "${result_dir}/program-output-${length}.txt"
done

echo "deepseek2_cuda_tensor_prefill_profile_result_dir=${result_dir}"
echo "deepseek2_cuda_tensor_prefill_profile=PASS"
