#!/usr/bin/env bash
# Chapter 00：下载固定版本的 Qwen3-0.6B Q4_K_M GGUF。
# 验证：断点续传完成后严格检查文件大小和 SHA256。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${QWEN3_MODEL_DIR:-${repo_root}/llama.cpp/models/qwen3-0.6b}"
model_name="Qwen3-0.6B-Q4_K_M.gguf"
model_path="${model_dir}/${model_name}"
model_revision="f9184d8b4c371cb52af5b8cb3d38e55e3cf0f4a0"
remote_name="Qwen_Qwen3-0.6B-Q4_K_M.gguf"
model_url="https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/${model_revision}/${remote_name}"
expected_bytes=484220320
expected_sha256="9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14"
reserve_bytes=1000000000

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
    echo "insufficient disk space for Qwen3 download" >&2
    echo "available: ${available_bytes} bytes" >&2
    echo "required:  ${required_bytes} bytes (remaining model plus 1 GB reserve)" >&2
    exit 2
fi

curl -L --fail --retry 3 -C - -o "${model_path}" "${model_url}"

actual_bytes="$(wc -c <"${model_path}")"
if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "Qwen3 GGUF size mismatch" >&2
    echo "expected: ${expected_bytes} bytes" >&2
    echo "actual:   ${actual_bytes} bytes" >&2
    exit 1
fi

actual_sha256="$(sha256sum "${model_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Qwen3 GGUF checksum mismatch" >&2
    echo "expected: ${expected_sha256}" >&2
    echo "actual:   ${actual_sha256}" >&2
    exit 1
fi

ls -lh "${model_path}"
echo "${actual_sha256}  ${model_path}"
echo "[chapter00] Qwen3 model ready: ${model_path}"
