#!/usr/bin/env bash
# Chapter 04：捕获 llama.cpp DeepSeek-V2-Lite 首层逐节点参考。
# 通过条件：双词元首层逐节点对照满足误差门槛。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${1:-${DEEPSEEK_V2_LITE_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
build_dir="${CHAPTER04_DEEPSEEK_CAPTURE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-deepseek-layer-capture}"

if [[ ! -f "${model_path}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${model_path}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target deepseek_v2_lite_layer_capture -j2

output="$("${build_dir}/deepseek_v2_lite_layer_capture" "${model_path}")"
grep -q '^embd: elements=4096 ' <<<"${output}"
grep -q '^q-0: elements=6144 ' <<<"${output}"
grep -q '^kv_cmpr_pe-0: elements=1152 ' <<<"${output}"
grep -q '^l_out-0: elements=4096 ' <<<"${output}"
grep -q '^ffn_norm-1: elements=4096 ' <<<"${output}"
grep -q '^ffn_inp-1: elements=4096 ' <<<"${output}"
grep -q '^ffn_moe_out-1: elements=4096 ' <<<"${output}"
grep -q '^ffn_shexp-1: elements=4096 ' <<<"${output}"
grep -q '^ffn_out-1: elements=4096 ' <<<"${output}"
grep -q '^l_out-1: elements=4096 ' <<<"${output}"
grep -q '^deepseek_v2_lite_tutorial_moe_compare=PASS$' <<<"${output}"
grep -q '^deepseek_v2_lite_tutorial_layer_compare=PASS$' <<<"${output}"
grep -q '^deepseek_v2_lite_layer_capture=PASS$' <<<"${output}"
printf '%s\n' "${output}"
