#!/usr/bin/env bash
# Chapter 10：验证 Qwen3-0.6B CPU ServiceCore 与 HTTP/SSE 组合闭环。
# 覆盖双请求批调度、分页 K/V、Radix 命中、终态释放、Completion 和 SSE。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${CPP_LLM_ENGINE_QWEN_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CHAPTER10_QWEN_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter10-qwen.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER10_QWEN_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
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
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_serving_smoke real_tinyllama_http_adapter_smoke \
    --parallel 2
"${build_dir}/real_tinyllama_serving_smoke" "${model_path}"
"${build_dir}/real_tinyllama_http_adapter_smoke" "${model_path}"
