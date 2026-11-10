#!/usr/bin/env bash
# Chapter 00：下载 Qwen1.5-MoE-A2.7B-Chat Q3_K_M GGUF，并校验固定资产。
# 验证：断点续传成功后检查文件大小；首次固定基线时应登记输出的 SHA256。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${QWEN15_MOE_MODEL_DIR:-${repo_root}/llama.cpp/models/qwen15-moe-a2.7b-chat}"
model_name="Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf"
model_path="${model_dir}/${model_name}"
model_url="https://huggingface.co/tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF/resolve/main/${model_name}"
expected_bytes=7440000000
expected_sha256="e97354d815afecb935502776280da88932a93c4933ff169c7f463b2b354ecaf0"
reserve_bytes=2000000000

mkdir -p "${model_dir}"
existing_bytes=0
if [[ -f "${model_path}" ]]; then
    existing_bytes="$(wc -c <"${model_path}")"
fi
remaining_bytes=$((expected_bytes > existing_bytes ? expected_bytes - existing_bytes : 0))
required_bytes=$((remaining_bytes + reserve_bytes))
available_kib="$(df -Pk "${model_dir}" | awk 'NR == 2 {print $4}')"
available_bytes=$((available_kib * 1024))
if (( available_bytes < required_bytes )); then
    echo "insufficient disk space for Qwen1.5-MoE download" >&2
    echo "available: ${available_bytes} bytes" >&2
    echo "required:  ${required_bytes} bytes (remaining model plus 2 GB reserve)" >&2
    exit 2
fi

curl -L --fail --retry 3 -C - -o "${model_path}" "${model_url}"

actual_bytes="$(wc -c <"${model_path}")"
if (( actual_bytes < 7000000000 )); then
    echo "downloaded GGUF is smaller than expected: ${actual_bytes} bytes" >&2
    exit 1
fi

actual_sha256="$(sha256sum "${model_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Qwen1.5-MoE GGUF checksum mismatch" >&2
    echo "expected: ${expected_sha256}" >&2
    echo "actual:   ${actual_sha256}" >&2
    exit 1
fi

ls -lh "${model_path}"
echo "${actual_sha256}  ${model_path}"
echo "[chapter00] Qwen1.5-MoE model ready: ${model_path}"
