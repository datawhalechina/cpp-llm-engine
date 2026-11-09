#!/usr/bin/env bash
# Chapter 01：验证固定 Qwen3-0.6B GGUF 的分词器、元数据和张量布局。
# 运行：bash code/chapter01_tensor_loader/run_qwen3_baseline.sh [model.gguf]。
# 通过条件：文件校验值、固定词元、聊天模板和 Qwen3 schema 全部匹配。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${repo_root}/code/chapter01_tensor_loader"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
model_path="${1:-${CHAPTER01_QWEN3_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"
expected_sha256="9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14"

if [[ -n "${LLAMA_CPP_BUILD_DIR:-}" ]]; then
    llama_build_dir="${LLAMA_CPP_BUILD_DIR}"
elif [[ -d "${llama_root}/build-cpu/bin" ]]; then
    llama_build_dir="${llama_root}/build-cpu"
else
    llama_build_dir="${llama_root}/build"
fi

if [[ ! -f "${model_path}" ]]; then
    echo "Qwen3 model not found: ${model_path}" >&2
    exit 1
fi

actual_sha256="$(sha256sum "${model_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Qwen3 model checksum mismatch: ${actual_sha256}" >&2
    exit 1
fi

build_dir="${CHAPTER01_QWEN3_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter01-qwen3.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER01_QWEN3_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}"
cmake --build "${build_dir}" --target llama_tokenizer_compare gguf_inspect -j2

tokenizer_output="$("${build_dir}/llama_tokenizer_compare" \
    "${model_path}" 'Hello Qwen3' 2>/dev/null)"
grep -q '^vocab_size=151936$' <<<"${tokenizer_output}"
grep -q '^token_count=4$' <<<"${tokenizer_output}"
grep -q '^tokens=9707,1207,16948,18$' <<<"${tokenizer_output}"
grep -q '^text=Hello Qwen3$' <<<"${tokenizer_output}"
printf '%s\n' "${tokenizer_output}"

inspect_output="$("${build_dir}/gguf_inspect" "${model_path}" --validate-schema)"
grep -q '^architecture: qwen3$' <<<"${inspect_output}"
grep -q '^layers: 28$' <<<"${inspect_output}"
grep -q '^attention heads: 16$' <<<"${inspect_output}"
grep -q '^attention kv heads: 8$' <<<"${inspect_output}"
grep -q '^attention key/value length: 128/128$' <<<"${inspect_output}"
grep -q '^chat template: yes$' <<<"${inspect_output}"
grep -q '^tensors: 311$' <<<"${inspect_output}"
grep -q '^schema: qwen3 tensors valid$' <<<"${inspect_output}"
grep -q '^decoder input: ready$' <<<"${inspect_output}"
printf '%s\n' "${inspect_output}"
echo "Qwen3 Chapter01 baseline passed"
