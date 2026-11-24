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
mode="${4:-single-row}"
question="${5:-What historic event is shown? Answer in one short sentence.}"
build_dir="${MULTIMODAL_CUDA_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-multimodal-cuda}"
llama_cpp_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_cpp_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_cpp_root}/build}"

for asset in "${mmproj}" "${model}" "${image}"; do
    if [[ ! -f "${asset}" ]]; then
        echo "required asset not found: ${asset}" >&2
        exit 2
    fi
done
if [[ "${mode}" != "single-row" && "${mode}" != "full-tile" && \
      "${mode}" != "mixed-prompt" && "${mode}" != "mixed-service" ]]; then
    echo "context mode must be single-row, full-tile, mixed-prompt, or mixed-service" >&2
    exit 2
fi

cmake_options=(
    -DMULTIMODAL_ENABLE_CUDA=ON
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc"
    -DMULTIMODAL_CUDA_SMOLVLM_MMPROJ="${mmproj}"
    -DMULTIMODAL_CUDA_SMOLVLM_MODEL="${model}"
    -DMULTIMODAL_CUDA_SMOLVLM_IMAGE="${image}"
    -DCMAKE_BUILD_TYPE=Release
)
if [[ "${mode}" == "mixed-prompt" || "${mode}" == "mixed-service" ]]; then
    if [[ ! -f "${llama_cpp_root}/include/llama.h" || \
          ! -d "${llama_cpp_build_dir}/bin" ]]; then
        echo "mixed-prompt requires a built llama.cpp checkout" >&2
        exit 2
    fi
    cmake_options+=(
        -DMULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON
        -DLLAMA_CPP_ROOT="${llama_cpp_root}"
        -DLLAMA_CPP_BUILD_DIR="${llama_cpp_build_dir}"
    )
fi
if [[ "${mode}" == "mixed-service" ]]; then
    cmake_options+=(
        -DMULTIMODAL_ENABLE_SERVICE_INTEGRATION=ON
    )
fi

cmake -S "${source_dir}" -B "${build_dir}" \
    "${cmake_options[@]}"
cmake --build "${build_dir}" \
    --target multimodal_cuda_smolvlm_context_inspect -j2
if [[ "${mode}" == "mixed-prompt" ]]; then
    "${build_dir}/multimodal_cuda_smolvlm_context_inspect" \
        "${mmproj}" "${model}" "${image}" --mixed-prompt "${question}"
elif [[ "${mode}" == "mixed-service" ]]; then
    "${build_dir}/multimodal_cuda_smolvlm_context_inspect" \
        "${mmproj}" "${model}" "${image}" --mixed-service "${question}"
elif [[ "${mode}" == "full-tile" ]]; then
    "${build_dir}/multimodal_cuda_smolvlm_context_inspect" \
        "${mmproj}" "${model}" "${image}" --full-tile
else
    ctest --test-dir "${build_dir}" \
        -R '^multimodal_cuda_smolvlm_context_test$' --output-on-failure
fi
