#!/usr/bin/env bash
# Chapter 00：使用本地 llama.cpp 和 TinyLlama 建立真实推理基线。
# 验证：路径、参数和进程退出状态有效，并输出真实模型生成结果。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
if [[ -n "${LLAMA_CPP_BUILD_DIR:-}" ]]; then
    llama_build_dir="${LLAMA_CPP_BUILD_DIR}"
elif [[ -x "${llama_root}/build-cpu/bin/llama-simple" ]]; then
    llama_build_dir="${llama_root}/build-cpu"
else
    llama_build_dir="${llama_root}/build"
fi
executable="${CHAPTER00_LLAMA_EXECUTABLE:-${llama_build_dir}/bin/llama-simple}"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${CHAPTER00_MODEL_PATH:-${llama_root}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
prompt="${1:-${CHAPTER00_PROMPT:-Hello, world}}"
max_new_tokens="${2:-${CHAPTER00_MAX_NEW_TOKENS:-8}}"
gpu_layers="${CHAPTER00_GPU_LAYERS:-0}"
verbose="${CHAPTER00_VERBOSE:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash code/chapter00_quickstart/run_real_inference.sh [prompt] [max_new_tokens]

Environment overrides:
  LLAMA_CPP_ROOT               llama.cpp source checkout
  LLAMA_CPP_BUILD_DIR          llama.cpp build directory
  CHAPTER00_LLAMA_EXECUTABLE   llama-simple executable
  CPP_LLM_ENGINE_MODEL_PATH    GGUF model path
  CHAPTER00_MODEL_PATH         Chapter00-specific GGUF model path
  CHAPTER00_PROMPT             default prompt
  CHAPTER00_MAX_NEW_TOKENS     default generated token count
  CHAPTER00_GPU_LAYERS         layers offloaded to GPU; default 0
  CHAPTER00_VERBOSE            set to 1 to show full llama.cpp logs
EOF
    exit 0
fi

if [[ ! -x "${executable}" ]]; then
    echo "llama-simple executable not found: ${executable}" >&2
    echo "build llama.cpp first, or set CHAPTER00_LLAMA_EXECUTABLE" >&2
    exit 2
fi

if [[ ! -f "${model_path}" ]]; then
    echo "GGUF model not found: ${model_path}" >&2
    echo "download TinyLlama first, or set CPP_LLM_ENGINE_MODEL_PATH" >&2
    exit 2
fi

if [[ -z "${prompt}" ]]; then
    echo "prompt must not be empty" >&2
    exit 2
fi

if [[ ! "${max_new_tokens}" =~ ^[1-9][0-9]*$ ]]; then
    echo "max_new_tokens must be a positive integer: ${max_new_tokens}" >&2
    exit 2
fi

if [[ ! "${gpu_layers}" =~ ^[0-9]+$ ]]; then
    echo "CHAPTER00_GPU_LAYERS must be a non-negative integer: ${gpu_layers}" >&2
    exit 2
fi

echo "[chapter00] executable: ${executable}"
echo "[chapter00] model: ${model_path}"
echo "[chapter00] max new tokens: ${max_new_tokens}"
echo "[chapter00] GPU layers: ${gpu_layers}"

command=(
    "${executable}"
    -m "${model_path}"
    -n "${max_new_tokens}"
    -ngl "${gpu_layers}"
    "${prompt}"
)

log_file="$(mktemp "${TMPDIR:-/tmp}/cpp-llm-engine-chapter00.XXXXXX.log")"
trap 'rm -f "${log_file}"' EXIT
if "${command[@]}" 2>"${log_file}"; then
    if [[ "${verbose}" == "1" ]]; then
        cat "${log_file}" >&2
    else
        grep -E '^(main: decoded|llama_perf_context_print: *(load|prompt eval|eval|total) time)' \
            "${log_file}" || true
    fi
else
    status=$?
    cat "${log_file}" >&2
    exit "${status}"
fi

decoded_tokens="$(sed -n 's/^main: decoded \([0-9][0-9]*\) tokens.*/\1/p' "${log_file}" | tail -n 1)"
if [[ -z "${decoded_tokens}" || "${decoded_tokens}" -eq 0 ]]; then
    echo "llama.cpp exited without generating a token" >&2
    exit 1
fi

echo "[chapter00] real inference passed: decoded_tokens=${decoded_tokens}"
