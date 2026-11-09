#!/usr/bin/env bash
# Chapter 09：采集真实 TinyLlama 双请求顺序 Decode 与 Tensor Batch A/B 样本。
# 产物：环境元数据和两轮独立程序输出；默认写入临时目录。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_CUDA_BATCH_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-batch-build}"
result_dir="${CHAPTER09_CUDA_BATCH_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-batch-result}"
experiment_id="${CHAPTER09_CUDA_BATCH_EXPERIMENT_ID:-EXP-GPU-12}"
optimization="${CHAPTER09_CUDA_BATCH_OPTIMIZATION:-native-half-batch-tile4}"

for command in cmake nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required command not found: ${command}" >&2
        exit 2
    fi
done
if [[ ! -x "${cuda_home}/bin/nvcc" || ! -f "${model_path}" ]]; then
    echo "CUDA toolkit or model is missing" >&2
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
    --target real_tinyllama_cuda_batch_benchmark --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=${experiment_id}
workload_id=WL-GPU-TINYLLAMA-PAGED-BATCH-AB-V1
optimization=${optimization}
model=TinyLlama-1.1B-Chat-v1.0.Q4_K_M
backend=CUDA
build_type=Release
requests=2
prompt_tokens=1,2
decode_steps=1
block_size=8
warmup=2
iterations=10
independent_runs=2
statistic=median
baseline=paged-sequential-decode
experiment=paged-tensor-batch-decode
timing_scope=excludes-model-load-context-create-request-bind-and-prefill
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

for run in 1 2; do
    "${build_dir}/real_tinyllama_cuda_batch_benchmark" "${model_path}" \
        --gemv-path fixed-four \
        >"${result_dir}/run-${run}.txt"
done
echo "correctness=PASS" >>"${result_dir}/metadata.txt"
echo "cuda_batch_ab_result_dir=${result_dir}"
echo "cuda_batch_ab=PASS"
