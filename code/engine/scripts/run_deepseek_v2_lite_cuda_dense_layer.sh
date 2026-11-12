#!/usr/bin/env bash
# Engine/Chapter 04：验证 DeepSeek-V2-Lite CUDA 稠密首层与 CPU 参考结果。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CPP_LLM_ENGINE_DEEPSEEK2_CUDA_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek2-cuda-layer}"

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
    --target real_deepseek2_cuda_dense_layer_test -j2
if ! ctest --test-dir "${build_dir}" -N \
    -R '^real_deepseek2_cuda_dense_layer_test$' | \
    grep -q 'Total Tests: 1'; then
    echo "DeepSeek CUDA dense-layer test was not registered; use a fresh build directory" >&2
    exit 1
fi
ctest --test-dir "${build_dir}" \
    -R '^real_deepseek2_cuda_dense_layer_test$' --output-on-failure
