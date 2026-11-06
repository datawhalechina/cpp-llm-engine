#!/usr/bin/env bash
# 真实 TinyLlama CPU Decoder 学习入口。
# 串联教学 Decoder、量化 backend 和 22 层真实 generation correctness。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
if [[ -n "${LLAMA_CPP_BUILD_DIR:-}" ]]; then
    llama_build_dir="${LLAMA_CPP_BUILD_DIR}"
elif [[ -d "${llama_root}/build-cpu/bin" ]]; then
    llama_build_dir="${llama_root}/build-cpu"
else
    llama_build_dir="${llama_root}/build"
fi
build_dir="${REAL_DECODER_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-real-decoder.XXXXXX")}"
cleanup_build=false
if [[ -z "${REAL_DECODER_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    echo "set CPP_LLM_ENGINE_MODEL_PATH to a TinyLlama GGUF file" >&2
    exit 2
fi
if [[ ! -f "${llama_root}/CMakeLists.txt" ]]; then
    echo "llama.cpp source not found: ${llama_root}" >&2
    echo "set LLAMA_CPP_ROOT to the llama.cpp checkout" >&2
    exit 2
fi

echo "[real-decoder] chapter04 structure"
bash "${repo_root}/code/chapter04_attention/scripts/run.sh"

echo "[real-decoder] chapter05 quantized backends"
bash "${repo_root}/code/chapter05_quantization/scripts/run.sh"

echo "[real-decoder] configure engine"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target gguf_decoder_multilayer real_tinyllama_generation_smoke \
    real_tinyllama_batch_smoke -j2
ctest --test-dir "${build_dir}" \
    -R '^(gguf_decoder_multilayer|real_tinyllama_(generation|batch)_test)$' \
    --output-on-failure

echo "[real-decoder] learning path passed"
