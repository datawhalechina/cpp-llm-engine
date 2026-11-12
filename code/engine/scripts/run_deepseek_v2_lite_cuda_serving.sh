#!/usr/bin/env bash
# Engine/Chapter10: validate DeepSeek-V2-Lite CUDA service lifecycle.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${CPP_LLM_DEEPSEEK2_CUDA_SERVING_BUILD_DIR:-/tmp/cpp-llm-deepseek2-cuda-serving-build}"

if [[ ! -f "${model_path}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${model_path}" >&2
    exit 1
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${model_path}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_deepseek2_cuda_serving_test -j2
if ! ctest --test-dir "${build_dir}" -N \
    -R '^real_deepseek2_cuda_serving_test$' | grep -q 'Total Tests: 1'; then
    echo "DeepSeek CUDA serving test was not registered; use a fresh build directory" >&2
    exit 1
fi
ctest --test-dir "${build_dir}" \
    -R '^real_deepseek2_cuda_serving_test$' --output-on-failure
