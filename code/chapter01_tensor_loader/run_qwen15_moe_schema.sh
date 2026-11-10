#!/usr/bin/env bash
# Chapter 01：验证固定 Qwen1.5-MoE GGUF 的元数据和专家张量布局。
# 运行：bash code/chapter01_tensor_loader/run_qwen15_moe_schema.sh [model.gguf]。
# 通过条件：文件校验值、24 层、411 个张量和 60 选 4 专家布局全部匹配。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${repo_root}/code/chapter01_tensor_loader"
model_path="${1:-${CHAPTER01_QWEN15_MOE_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}}"
expected_sha256="e97354d815afecb935502776280da88932a93c4933ff169c7f463b2b354ecaf0"

if [[ ! -f "${model_path}" ]]; then
    echo "Qwen1.5-MoE model not found: ${model_path}" >&2
    echo "run: bash code/chapter00_quickstart/download_qwen15_moe.sh" >&2
    exit 2
fi

actual_sha256="$(sha256sum "${model_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Qwen1.5-MoE model checksum mismatch: ${actual_sha256}" >&2
    exit 1
fi

build_dir="${CHAPTER01_QWEN15_MOE_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter01-qwen15-moe.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER01_QWEN15_MOE_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}" --target gguf_inspect -j2

inspect_output="$("${build_dir}/gguf_inspect" "${model_path}" --validate-schema)"
grep -q '^architecture: qwen2moe$' <<<"${inspect_output}"
grep -q '^layers: 24$' <<<"${inspect_output}"
grep -q '^experts selected/total: 4/60$' <<<"${inspect_output}"
grep -q '^tensors: 411$' <<<"${inspect_output}"
grep -q '^schema: qwen2moe tensors valid$' <<<"${inspect_output}"
grep -q '^decoder input: MoE schema ready$' <<<"${inspect_output}"
printf '%s\n' "${inspect_output}"
echo "Qwen1.5-MoE Chapter01 schema baseline passed"
