#!/usr/bin/env bash
# Chapter 04：运行 Qwen3-0.6B 首层单词元与多词元 CPU 闭环。
# 运行：bash code/chapter04_attention/scripts/run_qwen3_first_layer.sh [model.gguf]。
# 通过条件：Q/K Norm、头维度、增量 KV 和因果注意力输出符合固定契约。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${1:-${CHAPTER04_QWEN3_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"
build_dir="${CHAPTER04_QWEN3_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter04-qwen3-first-layer}"

if [[ ! -f "${model_path}" ]]; then
    echo "Qwen3 model not found: ${model_path}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel 2 \
    --target gguf_first_layer_forward gguf_first_layer_multitoken

single_output="$("${build_dir}/gguf_first_layer_forward" "${model_path}")"
grep -q '^architecture: qwen3$' <<<"${single_output}"
grep -q '^query_heads: 16$' <<<"${single_output}"
grep -q '^kv_heads: 8$' <<<"${single_output}"
grep -q '^head_dimension: 128$' <<<"${single_output}"
grep -q '^qk_norm: applied$' <<<"${single_output}"
grep -q '^query_size: 2048$' <<<"${single_output}"
grep -q '^kv_size: 1024$' <<<"${single_output}"
grep -q '^first_layer_forward: ready$' <<<"${single_output}"
printf '%s\n' "${single_output}"

multi_output="$("${build_dir}/gguf_first_layer_multitoken" "${model_path}")"
grep -q '^architecture: qwen3$' <<<"${multi_output}"
grep -q '^token_count: 4$' <<<"${multi_output}"
grep -q '^qk_norm: applied$' <<<"${multi_output}"
grep -q '^rope_dim: 128$' <<<"${multi_output}"
grep -q '^kv_cache_tokens: 4$' <<<"${multi_output}"
grep -q '^attention_score_count: 160$' <<<"${multi_output}"
grep -q '^first_layer_multitoken: ready$' <<<"${multi_output}"
printf '%s\n' "${multi_output}"
echo "Qwen3 Chapter04 first-layer execution passed"
