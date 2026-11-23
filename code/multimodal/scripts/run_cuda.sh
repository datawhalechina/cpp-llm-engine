#!/usr/bin/env bash
# Multi-modal CUDA: build and run the optional vision adapter tests.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${MULTIMODAL_CUDA_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-multimodal-cuda}"

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "CUDA compiler is missing: ${cuda_home}/bin/nvcc" >&2
    exit 2
fi

cmake -S "${repo_root}/code/multimodal" -B "${build_dir}" \
    -DMULTIMODAL_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target multimodal_cuda_vision_embedding_test \
        multimodal_cuda_vision_projector_test \
        multimodal_cuda_siglip_block_test \
        multimodal_cuda_vision_encoder_test --parallel 2
ctest --test-dir "${build_dir}" \
    -R '^multimodal_cuda_(vision_(embedding|projector|encoder)|siglip_block)_test$' \
    --output-on-failure
