#!/usr/bin/env bash
# Chapter 09：采集真实 TinyLlama 分页逐 token 与 Tensor Prefill 的 A/B 样本。
# 产物：环境元数据和两轮独立程序输出；默认写入临时目录。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_CUDA_PAGED_PREFILL_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-paged-prefill-build}"
result_dir="${CHAPTER09_CUDA_PAGED_PREFILL_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-paged-prefill-result}"

for command in cmake nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required command not found: ${command}" >&2
        exit 2
    fi
done
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found under ${cuda_home}" >&2
    exit 2
fi
if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi

mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_CUDA_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_cuda_paged_prefill_benchmark --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-05
workload_id=WL-GPU-TINYLLAMA-PAGED-PREFILL-AB-V1
model=TinyLlama-1.1B-Chat-v1.0.Q4_K_M
backend=CUDA
build_type=Release
prompt_tokens=32
block_size=8
warmup=2
iterations=10
independent_runs=2
statistic=median
baseline=paged-sequential-prefill
experiment=paged-tensor-prefill
timing_scope=excludes-model-load-context-create-and-request-bind
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

for run in 1 2; do
    "${build_dir}/real_tinyllama_cuda_paged_prefill_benchmark" "${model_path}" \
        >"${result_dir}/run-${run}.txt"
done
echo "correctness=PASS" >>"${result_dir}/metadata.txt"
echo "cuda_paged_prefill_ab_result_dir=${result_dir}"
echo "cuda_paged_prefill_ab=PASS"
