#!/usr/bin/env bash
# 多模态 CUDA：运行真实图片 Patch/Position、12 层 Encoder 与 Projector 对照。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${source_dir}/../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
model="${1:-${repo_root}/llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf}"
image="${2:-${source_dir}/tests/fixtures/two_by_two.ppm}"
mode="${3:-top-left-window}"
build_dir="${MULTIMODAL_CUDA_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-multimodal-cuda}"

if [[ ! -f "${model}" ]]; then
    echo "mmproj model not found: ${model}" >&2
    exit 2
fi
if [[ ! -f "${image}" ]]; then
    echo "image not found: ${image}" >&2
    exit 2
fi
if [[ "${mode}" != "top-left-window" && "${mode}" != "full-tile" ]]; then
    echo "vision mode must be top-left-window or full-tile" >&2
    exit 2
fi

enable_cblas="OFF"
run_arguments=("${model}" "${image}")
if [[ "${mode}" == "full-tile" ]]; then
    enable_cblas="ON"
    run_arguments+=("--full-tile")
fi

cmake -S "${source_dir}" -B "${build_dir}" \
    -DMULTIMODAL_ENABLE_CUDA=ON \
    -DMULTIMODAL_ENABLE_CBLAS="${enable_cblas}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target multimodal_cuda_vision_pipeline_inspect -j2
"${build_dir}/multimodal_cuda_vision_pipeline_inspect" "${run_arguments[@]}"
