#!/usr/bin/env bash
# Chapter 08：真实 TinyLlama Q4_K FFN gate/up scalar/AVX2 单变量 A/B。
# 契约：同一 Release 可执行文件、模型、prompt、token 和迭代数，只切换 gate/up。
# 验证：无论 KEEP 或 REJECT 都归档报告，正确性或可比性失败才中止。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_REAL_MODEL:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
build_dir="${CHAPTER08_Q4K_AB_BUILD_DIR:-/tmp/cpp-llm-engine-q4k-ffn-ab}"
result_dir="${CHAPTER08_Q4K_AB_RESULT_DIR:-${repo_root}/code/chapter08_profiling/results}"
iterations="${CHAPTER08_Q4K_AB_ITERATIONS:-3}"
baseline_file="${result_dir}/tinyllama_q4k_ffn_scalar.txt"
experiment_file="${result_dir}/tinyllama_q4k_ffn_avx2.txt"
report_file="${result_dir}/tinyllama_q4k_ffn_ab.txt"

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

"${build_dir}/real_tinyllama_decoder_profile" "${model_path}" 1 scalar >/dev/null
baseline_output=$("${build_dir}/real_tinyllama_decoder_profile" \
    "${model_path}" "${iterations}" scalar)
experiment_output=$("${build_dir}/real_tinyllama_decoder_profile" \
    "${model_path}" "${iterations}" q4-avx2)
printf '%s\n' "${baseline_output}" > "${baseline_file}"
printf '%s\n' "${experiment_output}" > "${experiment_file}"
grep -Fq "iterations=${iterations}" <<<"${baseline_output}"
grep -Fq "iterations=${iterations}" <<<"${experiment_output}"

report=$(node "${script_dir}/compare_q4k_ffn_profiles.mjs" \
    "${baseline_file}" "${experiment_file}")
printf '%s\n' "${report}" | tee "${report_file}"
grep -Fq "correctness=PASS" <<<"${report}"
grep -Eq "decision=(KEEP|REJECT)" <<<"${report}"
