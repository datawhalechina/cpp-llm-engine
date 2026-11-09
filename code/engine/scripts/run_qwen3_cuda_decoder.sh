#!/usr/bin/env bash
# Engine/Chapter 04：验证 Qwen3-0.6B 的 28 层 CUDA Decoder。
# 验收 CPU/CUDA hidden、逐层 K/V、logits、融合 Attention 与 Tensor Prefill。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_QWEN_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CUDA_QWEN_DECODER_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-qwen-cuda-decoder.XXXXXX")}"
cleanup_build=false
if [[ -z "${CUDA_QWEN_DECODER_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found: ${cuda_home}/bin/nvcc" >&2
    exit 2
fi
if [[ ! -f "${model_path}" ]]; then
    echo "Qwen3 model not found: ${model_path}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_cuda_multilayer_test --parallel 2
"${build_dir}/real_tinyllama_cuda_multilayer_test" "${model_path}" 28
