#!/usr/bin/env bash
# Chapter 09: compare row-copy and packed DeepSeek2 CUDA MoE Prefill dispatch.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_MOE_DISPATCH_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-moe-dispatch-build}"
result_dir="${CHAPTER09_DEEPSEEK2_MOE_DISPATCH_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-moe-dispatch-result}"
lengths="${CHAPTER09_DEEPSEEK2_MOE_DISPATCH_LENGTHS:-32,128}"
warmup="${CHAPTER09_DEEPSEEK2_MOE_DISPATCH_WARMUP:-1}"
iterations="${CHAPTER09_DEEPSEEK2_MOE_DISPATCH_ITERATIONS:-4}"

if [[ ! -x "${cuda_home}/bin/nvcc" || ! -f "${model_path}" ]]; then
    echo "CUDA Toolkit or DeepSeek model is missing" >&2
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
    --target real_deepseek2_cuda_moe_dispatch_benchmark --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-21
workload_id=WL-GPU-DEEPSEEK2-MOE-DISPATCH-AB-V1
model=DeepSeek-V2-Lite-Chat-Q2_K
backend=CUDA
build_type=Release
lengths=${lengths}
warmup=${warmup}
iterations=${iterations}
mla_path=absorbed
baseline=row-copies
candidate=packed-dispatch
timing_scope=27-layer-prefill-and-lm-head
excludes=model-load-context-create
EOF
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

"${build_dir}/real_deepseek2_cuda_moe_dispatch_benchmark" \
    "${model_path}" "${lengths}" "${warmup}" "${iterations}" |
    tee "${result_dir}/paired.txt"
grep -q 'moe_dispatch_correctness=PASS' "${result_dir}/paired.txt"

echo "deepseek2_cuda_moe_dispatch_result_dir=${result_dir}"
echo "deepseek2_cuda_moe_dispatch_ab=PASS"
