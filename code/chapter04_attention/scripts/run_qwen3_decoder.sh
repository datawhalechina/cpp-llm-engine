#!/usr/bin/env bash
# Chapter 04：验证 Qwen3-0.6B 的 28 层 CPU Decoder 与 llama.cpp 基线。
# 覆盖完整 logits、候选排序和连续四步贪心解码，不进入 Runtime 与服务层。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${CPP_LLM_ENGINE_QWEN_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CHAPTER04_QWEN_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter04-qwen.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER04_QWEN_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi
if [[ ! -f "${llama_build_dir}/bin/libllama.so" ]]; then
    echo "llama.cpp build not found: ${llama_build_dir}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target gguf_decoder_multilayer --parallel 2
"${build_dir}/gguf_decoder_multilayer" "${model_path}" 28
