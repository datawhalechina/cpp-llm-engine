#!/usr/bin/env bash
# 多模态 CUDA：把真实视觉 embedding 送入 SmolVLM CUDA 语言 Context。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${source_dir}/../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
mmproj="${1:-${repo_root}/llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf}"
model="${2:-${repo_root}/llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf}"
image="${3:-${source_dir}/tests/fixtures/two_by_two.ppm}"
build_dir="${MULTIMODAL_CUDA_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-multimodal-cuda}"

for asset in "${mmproj}" "${model}" "${image}"; do
    if [[ ! -f "${asset}" ]]; then
        echo "required asset not found: ${asset}" >&2
        exit 2
    fi
done

cmake -S "${source_dir}" -B "${build_dir}" \
    -DMULTIMODAL_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DMULTIMODAL_CUDA_SMOLVLM_MMPROJ="${mmproj}" \
    -DMULTIMODAL_CUDA_SMOLVLM_MODEL="${model}" \
    -DMULTIMODAL_CUDA_SMOLVLM_IMAGE="${image}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target multimodal_cuda_smolvlm_context_inspect -j2
ctest --test-dir "${build_dir}" \
    -R '^multimodal_cuda_smolvlm_context_test$' --output-on-failure
