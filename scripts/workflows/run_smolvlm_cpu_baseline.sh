#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
build_dir="${SMOLVLM_BUILD_DIR:-/tmp/cpp-llm-engine-smolvlm-build}"
model_dir="${SMOLVLM_MODEL_DIR:-${llama_root}/models/smolvlm-500m}"
model_path="${SMOLVLM_MODEL_PATH:-${model_dir}/SmolVLM-500M-Instruct-Q8_0.gguf}"
mmproj_path="${SMOLVLM_MMPROJ_PATH:-${model_dir}/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf}"
image_path="${SMOLVLM_IMAGE_PATH:-${llama_root}/tools/mtmd/test-1.jpeg}"
result_dir="${SMOLVLM_RESULT_DIR:-/tmp/cpp-llm-engine-smolvlm}"

for path in "${llama_root}/CMakeLists.txt" "${model_path}" "${mmproj_path}" "${image_path}"; do
    if [[ ! -e "${path}" ]]; then
        echo "missing required baseline asset: ${path}" >&2
        exit 1
    fi
done

cmake -S "${llama_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_CURL=OFF
cmake --build "${build_dir}" --target llama-mtmd-cli -j "${BUILD_JOBS:-8}"

mkdir -p "${result_dir}"
log_path="${result_dir}/smolvlm-cpu-baseline.log"
manifest_path="${result_dir}/manifest.txt"

"${build_dir}/bin/llama-mtmd-cli" \
    -m "${model_path}" \
    --mmproj "${mmproj_path}" \
    --image "${image_path}" \
    -p "What historic event is shown? Answer in one short sentence." \
    --no-mmproj-offload -ngl 0 \
    --image-min-tokens 256 --image-max-tokens 256 \
    --temp 0 --seed 42 -n 32 -t 8 -tb 8 --no-warmup \
    >"${log_path}" 2>&1

if ! grep -Fq "Apollo 11" "${log_path}"; then
    tail -80 "${log_path}" >&2
    echo "SmolVLM baseline did not contain the expected answer" >&2
    exit 1
fi

{
    echo "repo_commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "llama_cpp_commit=$(git -C "${llama_root}" rev-parse HEAD)"
    echo "model_sha256=$(sha256sum "${model_path}" | awk '{print $1}')"
    echo "mmproj_sha256=$(sha256sum "${mmproj_path}" | awk '{print $1}')"
    echo "image_sha256=$(sha256sum "${image_path}" | awk '{print $1}')"
    echo "backend=cpu"
    echo "threads=8"
    echo "image_token_budget=256"
    echo "expected_answer=Apollo 11"
    echo "log=${log_path}"
} >"${manifest_path}"

echo "SmolVLM CPU baseline passed"
echo "log=${log_path}"
echo "manifest=${manifest_path}"
