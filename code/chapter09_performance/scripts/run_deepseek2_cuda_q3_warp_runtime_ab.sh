#!/usr/bin/env bash
# Chapter 09: keep Q2_K/IQ4_NL optimized and compare only Q3_K paths.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CHAPTER09_DEEPSEEK2_Q3_RUNTIME_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-q3-runtime-build}"
result_dir="${CHAPTER09_DEEPSEEK2_Q3_RUNTIME_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-q3-runtime-result}"
lengths="${CHAPTER09_DEEPSEEK2_Q3_RUNTIME_LENGTHS:-32,128}"
warmup="${CHAPTER09_DEEPSEEK2_Q3_RUNTIME_WARMUP:-1}"
iterations="${CHAPTER09_DEEPSEEK2_Q3_RUNTIME_ITERATIONS:-4}"

if [[ ! -f "${model_path}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${model_path}" >&2
    exit 2
fi
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found under ${cuda_home}" >&2
    exit 2
fi

mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${model_path}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_deepseek2_cuda_iq4_warp_benchmark --parallel 2

"${build_dir}/real_deepseek2_cuda_iq4_warp_benchmark" \
    "${model_path}" "${lengths}" "${warmup}" "${iterations}" q3 \
    >"${result_dir}/paired.txt"
grep -Fq "gemv_path_correctness=PASS" "${result_dir}/paired.txt"

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-29
workload_id=WL-GPU-DEEPSEEK2-Q3-WARP-RUNTIME-AB-V1
model=DeepSeek-V2-Lite-Chat-Q2_K
lengths=${lengths}
warmup=${warmup}
iterations=${iterations}
execution_order=baseline-experiment,experiment-baseline
mla_path=absorbed-three-stage
moe_prefill_path=packed-dispatch
q2_path=single-warp
iq4_path=single-warp
baseline=adaptive-256-thread-q3
experiment=adaptive-single-warp-q3
timing_scope=executor-prefill
correctness=PASS
execution_status=PASS
EOF
sha256sum "${model_path}" >>"${result_dir}/metadata.txt"
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"

echo "deepseek2_cuda_q3_runtime_result_dir=${result_dir}"
echo "deepseek2_cuda_q3_runtime_execution=PASS"
