#!/usr/bin/env bash
# Chapter 04：用 llama.cpp 中间张量核对 Qwen1.5-MoE 24 层 Decoder。
# 运行：bash code/chapter04_attention/scripts/run_qwen15_moe_decoder_compare.sh [model.gguf]。
# 通过条件：逐层局部 Top-K、隐藏状态、累计输出和 24 层 K/V 通过门禁。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${1:-${CHAPTER04_QWEN15_MOE_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CHAPTER04_QWEN15_MOE_COMPARE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-qwen15-moe-compare-build}"

if [[ ! -f "${model_path}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${model_path}" >&2
    exit 2
fi
if [[ ! -f "${llama_build_dir}/bin/libllama.so" &&
      ! -f "${llama_build_dir}/bin/libllama.so.0" ]]; then
    if ! compgen -G "${llama_build_dir}/bin/libllama.so.*" >/dev/null; then
        echo "llama.cpp shared library not found under ${llama_build_dir}/bin" >&2
        exit 2
    fi
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target qwen15_moe_ffn_compare -j2

output="$("${build_dir}/qwen15_moe_ffn_compare" "${model_path}" \
    --tokens 2 --layers 24)"
grep -q '^reference: llama.cpp cb_eval$' <<<"${output}"
grep -q '^token count: 2$' <<<"${output}"
grep -q '^layer count: 24$' <<<"${output}"
grep -q '^captured layer output size: 2048$' <<<"${output}"
grep -q '^key payload size: 98304$' <<<"${output}"
grep -q '^value payload size: 98304$' <<<"${output}"
grep -q '^local reference route mismatch tokens: 0$' <<<"${output}"
grep -q '^greedy token: 151643$' <<<"${output}"
grep -q '^reference greedy token: 151643$' <<<"${output}"
grep -q '^numeric comparison: PASS$' <<<"${output}"
printf '%s\n' "${output}"
