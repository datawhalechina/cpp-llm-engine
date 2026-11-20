#!/usr/bin/env bash
# Chapter 09: compare sequential and Tensor Prefill for DeepSeek2 CUDA.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-tensor-prefill-build}"
result_dir="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-tensor-prefill-result}"
lengths="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_LENGTHS:-2,8,32,128}"
warmup="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_WARMUP:-1}"
iterations="${CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_ITERATIONS:-4}"

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
    --target real_deepseek2_cuda_tensor_prefill_benchmark --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-19
workload_id=WL-GPU-DEEPSEEK2-SEQUENTIAL-TENSOR-PREFILL-AB-V1
model=DeepSeek-V2-Lite-Chat-Q2_K
backend=CUDA
build_type=Release
prompt_lengths=${lengths}
warmup=${warmup}
iterations=${iterations}
execution_order=sequential-tensor,tensor-sequential
baseline=one-token-prefill-chunks
experiment=multi-token-tensor-prefill
cache=continuous-compressed-mla
timing_scope=27-layer-prefill-and-lm-head
excludes=model-load-and-context-create
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

"${build_dir}/real_deepseek2_cuda_tensor_prefill_benchmark" \
    "${model_path}" "${lengths}" "${warmup}" "${iterations}" \
    | tee "${result_dir}/paired.txt"
echo "deepseek2_cuda_tensor_prefill_result_dir=${result_dir}"
echo "deepseek2_cuda_tensor_prefill_execution=PASS"
