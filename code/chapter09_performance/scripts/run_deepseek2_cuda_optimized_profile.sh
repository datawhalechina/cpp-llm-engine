#!/usr/bin/env bash
# Chapter 09: profile the current default DeepSeek2 CUDA Prefill path.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-optimized-profile-build}"
result_dir="${CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-optimized-profile}"
lengths="${CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_LENGTHS:-32,128}"
gemv_path="${CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_GEMV_PATH:-block-warp}"
moe_path="${CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_MOE_PATH:-packed}"

case "${gemv_path}:${moe_path}" in
    block-warp:packed)
        experiment_id="EXP-GPU-27"
        workload_id="WL-GPU-DEEPSEEK2-OPTIMIZED-PREFILL-PROFILE-V1"
        ;;
    all-warp:packed)
        experiment_id="EXP-GPU-30"
        workload_id="WL-GPU-DEEPSEEK2-ALL-WARP-POST-PROFILE-V1"
        ;;
    all-warp:device-router)
        experiment_id="EXP-GPU-32"
        workload_id="WL-GPU-DEEPSEEK2-DEVICE-ROUTER-POST-PROFILE-V1"
        ;;
    *)
        echo "supported GEMV/MoE paths are block-warp/packed, "\
             "all-warp/packed, and all-warp/device-router: "\
             "${gemv_path}/${moe_path}" >&2
        exit 2
        ;;
esac

if [[ ! -x "${cuda_home}/bin/nvcc" ||
      ! -x "${cuda_home}/bin/nsys" || ! -f "${model_path}" ]]; then
    echo "CUDA Toolkit, Nsight Systems, or DeepSeek model is missing" >&2
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
workload_id=${workload_id}
model=DeepSeek-V2-Lite-Chat-Q2_K
backend=CUDA
build_type=Release
profile_lengths=${lengths}
mla_path=absorbed-three-stage
moe_prefill_path=${moe_path}
gemv_path=${gemv_path}
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
    profile_base="${result_dir}/optimized-prefill-${length}"
    "${cuda_home}/bin/nsys" profile \
        --trace=cuda,osrt \
        --sample=none \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        --output="${profile_base}" \
        "${build_dir}/real_deepseek2_cuda_tensor_prefill_benchmark" \
        "${model_path}" --profile-tensor "${length}" \
        absorbed "${gemv_path}" "${moe_path}" \
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
    rm -f "${profile_base}.nsys-rep" "${profile_base}.sqlite"
done

echo "deepseek2_cuda_optimized_profile_result_dir=${result_dir}"
echo "deepseek2_cuda_optimized_profile=PASS"
