#!/usr/bin/env bash
# Chapter 04：执行真实 Qwen1.5-MoE layer-0 的量化 FFN 子层。
# 运行：bash code/chapter04_attention/scripts/run_qwen15_moe_ffn.sh [model.gguf]。
# 通过条件：真实 packed 权重完成 60 选 4、共享专家合并和有限输出。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${1:-${CHAPTER04_QWEN15_MOE_MODEL_PATH:-${repo_root}/llama.cpp/models/qwen15-moe-a2.7b-chat/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf}}"
build_dir="${CHAPTER04_QWEN15_MOE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-qwen15-moe-ffn-build}"

if [[ ! -f "${model_path}" ]]; then
    echo "Qwen1.5-MoE GGUF not found: ${model_path}" >&2
    echo "run: bash code/chapter00_quickstart/download_qwen15_moe.sh" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target qwen15_moe_ffn_inspect -j2

output="$("${build_dir}/qwen15_moe_ffn_inspect" "${model_path}")"
grep -q '^architecture: qwen2moe$' <<<"${output}"
grep -q '^experts selected/total: 4/60$' <<<"${output}"
grep -q '^output size: 2048$' <<<"${output}"
grep -q '^output finite: true$' <<<"${output}"
printf '%s\n' "${output}"
echo "Qwen1.5-MoE real FFN smoke passed"
