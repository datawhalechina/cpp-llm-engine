#!/usr/bin/env bash
# Chapter 09：用 Nsight Systems 记录真实 TinyLlama CUDA 生成的 kernel 基线。
# 产物：环境元数据、程序输出和 kernel/memory CSV；不保存体积较大的 nsys-rep。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CHAPTER09_CUDA_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-profile-build}"
result_dir="${CHAPTER09_CUDA_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-profile}"
experiment_id="${CHAPTER09_CUDA_EXPERIMENT_ID:-EXP-GPU-03}"
optimization="${CHAPTER09_CUDA_OPTIMIZATION:-fused-q4k-q6k-gemv-v1}"
profile_dir="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-nsys-profile.XXXXXX")"
profile_prefix="${profile_dir}/tinyllama-generation"
trap 'rm -rf "${profile_dir}"' EXIT

for command in cmake nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required command not found: ${command}" >&2
        exit 2
    fi
done
if [[ ! -x "${cuda_home}/bin/nvcc" || ! -x "${cuda_home}/bin/nsys" ]]; then
    echo "CUDA Toolkit or Nsight Systems not found under ${cuda_home}" >&2
    exit 2
fi
if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi
if [[ ! -f "${llama_root}/include/llama.h" ||
      ! -d "${llama_build_dir}/bin" ]]; then
    echo "llama.cpp tokenizer build not found: ${llama_build_dir}/bin" >&2
    exit 2
fi

mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_CUDA_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_cuda_generation_smoke --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=${experiment_id}
workload_id=WL-GPU-TINYLLAMA-GENERATION-4TOK-V1
optimization=${optimization}
model=TinyLlama-1.1B-Chat-v1.0.Q4_K_M
backend=CUDA
build_type=Release
prompt=Hello TinyLlama
output_tokens=4
profiler=Nsight Systems
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

"${cuda_home}/bin/nsys" profile \
    --trace=cuda,cublas \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="${profile_prefix}" \
    "${build_dir}/real_tinyllama_cuda_generation_smoke" \
    "${model_path}" 4 "Hello TinyLlama" \
    >"${result_dir}/program-output.txt"

"${cuda_home}/bin/nsys" stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "${result_dir}/kernel" \
    --force-overwrite true \
    "${profile_prefix}.nsys-rep"
"${cuda_home}/bin/nsys" stats \
    --report cuda_gpu_mem_time_sum \
    --format csv \
    --output "${result_dir}/memory" \
    --force-export true \
    --force-overwrite true \
    "${profile_prefix}.nsys-rep"

grep -q "generated_tokens: 29892 306 29915 29885" \
    "${result_dir}/program-output.txt"
echo "correctness=PASS" >>"${result_dir}/metadata.txt"
echo "cuda_profile_result_dir=${result_dir}"
echo "cuda_profile=PASS"
