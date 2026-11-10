#!/usr/bin/env bash
# Chapter 01：验证固定 DeepSeek-V2-Lite GGUF 的 MLA 与混合 Dense/MoE 布局。
# 运行：bash code/chapter01_tensor_loader/run_deepseek_v2_lite_schema.sh [model.gguf]。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${repo_root}/code/chapter01_tensor_loader"
model_path="${1:-${CHAPTER01_DEEPSEEK_V2_LITE_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
expected_bytes=6430464704
expected_sha256="a77d1dab7fae74a0377a3803b7428e60643fd231d309e8718bcee4c8ecf9499d"

if [[ ! -f "${model_path}" ]]; then
    echo "DeepSeek-V2-Lite model not found: ${model_path}" >&2
    echo "run: bash code/chapter00_quickstart/download_deepseek_v2_lite.sh" >&2
    exit 2
fi

actual_bytes="$(wc -c <"${model_path}")"
if [[ "${CHAPTER01_DEEPSEEK_ALLOW_HEADER_ONLY:-0}" == "1" ]]; then
    if (( actual_bytes < 3996352 )); then
        echo "DeepSeek-V2-Lite header fixture is incomplete: ${actual_bytes} bytes" >&2
        exit 1
    fi
else
    if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
        echo "DeepSeek-V2-Lite model size mismatch: ${actual_bytes} bytes" >&2
        exit 1
    fi
    actual_sha256="$(sha256sum "${model_path}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "DeepSeek-V2-Lite model checksum mismatch: ${actual_sha256}" >&2
        exit 1
    fi
fi

build_dir="${CHAPTER01_DEEPSEEK_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter01-deepseek.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER01_DEEPSEEK_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}" --target gguf_inspect -j2

inspect_output="$("${build_dir}/gguf_inspect" "${model_path}" --validate-schema)"
grep -q '^architecture: deepseek2$' <<<"${inspect_output}"
grep -q '^layers: 27$' <<<"${inspect_output}"
grep -q '^attention key/value length: 576/512$' <<<"${inspect_output}"
grep -q '^MLA key/value length: 192/128$' <<<"${inspect_output}"
grep -q '^KV LoRA rank: 512$' <<<"${inspect_output}"
grep -q '^leading dense layers: 1$' <<<"${inspect_output}"
grep -q '^experts selected/total: 6/64$' <<<"${inspect_output}"
grep -q '^tensors: 377$' <<<"${inspect_output}"
grep -q '^schema: deepseek2 tensors valid$' <<<"${inspect_output}"
grep -q '^decoder input: DeepSeek2 schema ready$' <<<"${inspect_output}"
printf '%s\n' "${inspect_output}"
echo "DeepSeek-V2-Lite Chapter01 schema baseline passed"
