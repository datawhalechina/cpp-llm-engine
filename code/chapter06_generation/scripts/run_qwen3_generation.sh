#!/usr/bin/env bash
# Chapter 06：验证 Qwen3-0.6B 的 CPU Generation Runtime 与请求 Context 隔离。
# 模型架构由 GGUF 自动识别；脚本固定 prompt、贪心词元和重复请求结果。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${CPP_LLM_ENGINE_QWEN_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CHAPTER06_QWEN_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter06-qwen.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER06_QWEN_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
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
cmake --build "${build_dir}" --target real_tinyllama_generation_smoke --parallel 2
"${build_dir}/real_tinyllama_generation_smoke" \
    "${model_path}" 4 "Hello Qwen3" "11,358,1366,311"
