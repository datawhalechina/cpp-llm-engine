#!/usr/bin/env bash
# Chapter 08：组合 Q4 Attention、Q4 FFN 和 Q6 FFN AVX2 policy 的真实 A/B。
# 验证：同一构建同时通过 Decoder golden/profile 和双请求 ServiceCore benchmark。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_REAL_MODEL:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
build_dir="${CHAPTER08_OPTIMIZED_AB_BUILD_DIR:-/tmp/cpp-llm-engine-optimized-policy-ab}"
result_dir="${CHAPTER08_OPTIMIZED_AB_RESULT_DIR:-${repo_root}/code/chapter08_profiling/results}"
iterations="${CHAPTER08_OPTIMIZED_AB_ITERATIONS:-3}"
warmup="${CHAPTER08_OPTIMIZED_AB_WARMUP:-1}"
scalar_profile="${result_dir}/tinyllama_optimized_decoder_scalar.txt"
optimized_profile="${result_dir}/tinyllama_optimized_decoder_avx2.txt"
scalar_service="${result_dir}/tinyllama_optimized_service_scalar.json"
optimized_service="${result_dir}/tinyllama_optimized_service_avx2.json"
report_file="${result_dir}/tinyllama_optimized_policy_ab.txt"

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
cmake --build "${build_dir}" \
    --target real_tinyllama_decoder_profile real_tinyllama_serving_benchmark -j2

"${build_dir}/real_tinyllama_decoder_profile" "${model_path}" 1 scalar >/dev/null
"${build_dir}/real_tinyllama_decoder_profile" \
    "${model_path}" "${iterations}" scalar > "${scalar_profile}"
"${build_dir}/real_tinyllama_decoder_profile" \
    "${model_path}" "${iterations}" optimized-avx2 > "${optimized_profile}"

"${build_dir}/real_tinyllama_serving_benchmark" "${model_path}" \
    --warmup "${warmup}" --iterations "${iterations}" \
    --execution-policy scalar --output "${scalar_service}"
"${build_dir}/real_tinyllama_serving_benchmark" "${model_path}" \
    --warmup "${warmup}" --iterations "${iterations}" \
    --execution-policy optimized-avx2 --output "${optimized_service}"

report=$(node "${script_dir}/compare_optimized_policy.mjs" \
    "${scalar_profile}" "${optimized_profile}" \
    "${scalar_service}" "${optimized_service}")
printf '%s\n' "${report}" | tee "${report_file}"
grep -Fq "correctness=PASS" <<<"${report}"
grep -Eq "decision=(KEEP|REJECT)" <<<"${report}"
