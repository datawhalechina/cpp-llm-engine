#!/usr/bin/env bash
# Chapter 09: compare sequential and grouped DeepSeek2 CUDA Decode.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_CUDA_MOE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-cuda-moe-build}"
result_dir="${CHAPTER09_DEEPSEEK2_CUDA_MOE_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-cuda-moe-result}"
iterations="${CHAPTER09_DEEPSEEK2_CUDA_MOE_ITERATIONS:-2}"

for command in cmake nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required command not found: ${command}" >&2
        exit 2
    fi
done
if [[ ! -x "${cuda_home}/bin/nvcc" || ! -f "${model_path}" ]]; then
    echo "CUDA toolkit or DeepSeek-V2-Lite model is missing" >&2
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

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-17
workload_id=WL-GPU-DEEPSEEK2-GROUPED-MOE-DECODE-AB-V1
model=DeepSeek-V2-Lite-Chat-Q2_K
backend=CUDA
build_type=Release
batch=2
iterations=${iterations}
execution_order=sequential-grouped,grouped-sequential
baseline=two-sequential-decodes
experiment=one-grouped-decode
timing_scope=27-layer-decode-and-lm-head
excludes=model-load-context-create-and-prefill
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

"${build_dir}/real_deepseek2_cuda_grouped_moe_benchmark" \
    "${model_path}" "${iterations}" | tee "${result_dir}/paired.txt"
echo "deepseek2_cuda_grouped_moe_result_dir=${result_dir}"
echo "deepseek2_cuda_grouped_moe_execution=PASS"
