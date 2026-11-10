#!/usr/bin/env bash
# Chapter 00：下载固定版本的 DeepSeek-V2-Lite-Chat Q2_K GGUF。
# 验证：断点续传完成后严格检查文件大小和 SHA256。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${DEEPSEEK_V2_LITE_MODEL_DIR:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat}"
model_name="DeepSeek-V2-Lite-Chat-Q2_K.gguf"
model_path="${model_dir}/${model_name}"
model_revision="d04e3f5a5775f7015d1c87e6a8ef2da4876baf0d"
model_url="https://huggingface.co/tensorblock/DeepSeek-V2-Lite-Chat-GGUF/resolve/${model_revision}/${model_name}"
expected_bytes=6430464704
expected_sha256="a77d1dab7fae74a0377a3803b7428e60643fd231d309e8718bcee4c8ecf9499d"
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
    echo "insufficient disk space for DeepSeek-V2-Lite download" >&2
    echo "available: ${available_bytes} bytes" >&2
    echo "required:  ${required_bytes} bytes (remaining model plus 2 GB reserve)" >&2
    exit 2
fi

curl -L --fail --retry 3 -C - -o "${model_path}" "${model_url}"

actual_bytes="$(wc -c <"${model_path}")"
if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "DeepSeek-V2-Lite GGUF size mismatch" >&2
    echo "expected: ${expected_bytes} bytes" >&2
    echo "actual:   ${actual_bytes} bytes" >&2
    exit 1
fi

actual_sha256="$(sha256sum "${model_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "DeepSeek-V2-Lite GGUF checksum mismatch" >&2
    echo "expected: ${expected_sha256}" >&2
    echo "actual:   ${actual_sha256}" >&2
    exit 1
fi

ls -lh "${model_path}"
echo "${actual_sha256}  ${model_path}"
echo "[chapter00] DeepSeek-V2-Lite model ready: ${model_path}"
