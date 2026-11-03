#!/usr/bin/env bash
# Chapter 01：构建并验证真实 llama.cpp tokenizer adapter。
# 验证 TinyLlama GGUF 的词表规模、固定 prompt token ids 和 detokenize 结果。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${repo_root}/code/chapter01_tensor_loader"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${CHAPTER01_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
if [[ -n "${LLAMA_CPP_BUILD_DIR:-}" ]]; then
    llama_build_dir="${LLAMA_CPP_BUILD_DIR}"
elif [[ -d "${llama_root}/build-cpu/bin" ]]; then
    llama_build_dir="${llama_root}/build-cpu"
else
    llama_build_dir="${llama_root}/build"
fi
build_dir="${CHAPTER01_REAL_TOKENIZER_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter01-real-tokenizer.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER01_REAL_TOKENIZER_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}"
cmake --build "${build_dir}"

output="$("${build_dir}/llama_tokenizer_compare" "${model_path}" 'Hello TinyLlama' 2>/dev/null)"
grep -q '^vocab_size=32000$' <<<"${output}"
grep -q '^token_count=6$' <<<"${output}"
grep -q '^tokens=15043,18439,1460,29931,5288,29874$' <<<"${output}"
grep -q '^text=Hello TinyLlama$' <<<"${output}"
printf '%s\n' "${output}"

inspect_output="$("${build_dir}/gguf_inspect" "${model_path}" --validate-schema)"
grep -q '^schema: llama tensors valid$' <<<"${inspect_output}"
grep -q '^decoder input: ready$' <<<"${inspect_output}"
printf '%s\n' "${inspect_output}"
echo "real tokenizer contract passed"
