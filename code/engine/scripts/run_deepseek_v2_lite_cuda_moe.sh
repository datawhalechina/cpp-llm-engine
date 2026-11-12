#!/usr/bin/env bash
# Engine/Chapter 05: compare one real DeepSeek-V2-Lite MoE FFN on CPU/CUDA.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CPP_LLM_ENGINE_DEEPSEEK2_CUDA_MOE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-cuda-moe}"

if [[ ! -f "${model_path}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${model_path}" >&2
    exit 1
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${model_path}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc"
cmake --build "${build_dir}" \
    --target \
        real_deepseek2_cuda_moe_feed_forward_test \
        real_deepseek2_cuda_moe_layer_test -j2
if ! ctest --test-dir "${build_dir}" -N \
    -R '^real_deepseek2_cuda_moe_(feed_forward|layer)_test$' | \
    grep -q 'Total Tests: 2'; then
    echo "DeepSeek CUDA MoE tests were not registered; use a fresh build directory" >&2
    exit 1
fi
ctest --test-dir "${build_dir}" \
    -R '^real_deepseek2_cuda_moe_(feed_forward|layer)_test$' \
    --output-on-failure
