#!/usr/bin/env bash
# Chapter 09: compare continuous and paged DeepSeek2 CUDA MLA.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_PAGED_MLA_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-paged-mla-build}"
result_dir="${CHAPTER09_DEEPSEEK2_PAGED_MLA_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-deepseek2-paged-mla-result}"
warmup="${CHAPTER09_DEEPSEEK2_PAGED_MLA_WARMUP:-1}"
iterations="${CHAPTER09_DEEPSEEK2_PAGED_MLA_ITERATIONS:-4}"

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
    --target real_deepseek2_cuda_paged_mla_benchmark --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-18
workload_id=WL-GPU-DEEPSEEK2-CONTINUOUS-PAGED-MLA-AB-V1
model=DeepSeek-V2-Lite-Chat-Q2_K
backend=CUDA
build_type=Release
prompt_tokens=2
decode_steps=1
block_size=2
warmup=${warmup}
iterations=${iterations}
execution_order=continuous-paged,paged-continuous
baseline=continuous-compressed-mla
experiment=paged-compressed-mla
timing_scope=27-layer-prefill-or-decode-and-lm-head
excludes=model-load-context-create-and-terminal-release
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

"${build_dir}/real_deepseek2_cuda_paged_mla_benchmark" \
    "${model_path}" "${warmup}" "${iterations}" \
    | tee "${result_dir}/paired.txt"
echo "deepseek2_cuda_paged_mla_result_dir=${result_dir}"
echo "deepseek2_cuda_paged_mla_execution=PASS"
