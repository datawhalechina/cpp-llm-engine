#!/usr/bin/env bash
# Chapter 08：真实 TinyLlama Q6_K FFN down scalar/AVX2 单变量 A/B。
# 契约：同一 Release 可执行文件、模型、prompt、token 和迭代数，只切换 down backend。
# 验证：golden 与计时闭合通过，且 down 和 Decoder 总时间同时改善才 KEEP。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_REAL_MODEL:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
build_dir="${CHAPTER08_Q6K_AB_BUILD_DIR:-/tmp/cpp-llm-engine-q6k-down-ab}"
result_dir="${CHAPTER08_Q6K_AB_RESULT_DIR:-${script_dir}/../results}"
iterations="${CHAPTER08_Q6K_AB_ITERATIONS:-3}"
baseline_file="${result_dir}/tinyllama_q6k_down_scalar.txt"
experiment_file="${result_dir}/tinyllama_q6k_down_avx2.txt"
report_file="${result_dir}/tinyllama_q6k_down_ab.txt"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi

mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${repo_root}/llama.cpp" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_tinyllama_decoder_profile -j2

# 先触碰模型页和执行路径，正式结果仍由后续独立 Context 采集。
"${build_dir}/real_tinyllama_decoder_profile" "${model_path}" 1 scalar >/dev/null

baseline_output=$("${build_dir}/real_tinyllama_decoder_profile" \
    "${model_path}" "${iterations}" scalar)
experiment_output=$("${build_dir}/real_tinyllama_decoder_profile" \
    "${model_path}" "${iterations}" avx2)
printf '%s\n' "${baseline_output}" > "${baseline_file}"
printf '%s\n' "${experiment_output}" > "${experiment_file}"
grep -Fq "iterations=${iterations}" <<<"${baseline_output}"
grep -Fq "iterations=${iterations}" <<<"${experiment_output}"

report=$(node "${script_dir}/compare_q6k_down_profiles.mjs" \
    "${baseline_file}" "${experiment_file}")
printf '%s\n' "${report}" | tee "${report_file}"
grep -Fq "correctness=PASS" <<<"${report}"
grep -Fq "decision=KEEP" <<<"${report}"
