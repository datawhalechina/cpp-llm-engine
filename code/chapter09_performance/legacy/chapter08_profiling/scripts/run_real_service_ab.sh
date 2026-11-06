#!/usr/bin/env bash
# Chapter 08：重新采集真实 TinyLlama ServiceCore experiment，并与归档 baseline 比较。
# 契约：这是同条件回归/可重复性 A/B，不把工作树变化写成单变量优化收益。
# 验证：correctness、环境和 workload 可比后，总延迟回退不超过默认 5%。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${source_dir}/../../../.." && pwd)"
build_dir="${CHAPTER08_REAL_BUILD_DIR:-/tmp/cpp-llm-engine-chapter08-real-ab}"
baseline_json="${CHAPTER08_REAL_BASELINE_JSON:-${repo_dir}/code/chapter09_performance/results/legacy/tinyllama_service_cpu_baseline.json}"
experiment_json="${CHAPTER08_REAL_EXPERIMENT_JSON:-${build_dir}/tinyllama_service_experiment.json}"
report_file="${CHAPTER08_REAL_AB_REPORT_FILE:-}"

if [[ ! -f "${baseline_json}" ]]; then
    echo "baseline not found: ${baseline_json}" >&2
    exit 2
fi

mkdir -p "${build_dir}"
if [[ -z "${CHAPTER08_REAL_EXPERIMENT_JSON:-}" ]]; then
    REAL_SERVING_BUILD_DIR="${REAL_SERVING_BUILD_DIR:-${build_dir}/engine}" \
    REAL_SERVING_RESULT_FILE="${experiment_json}" \
    REAL_SERVING_WARMUP="${CHAPTER08_REAL_WARMUP:-1}" \
    REAL_SERVING_ITERATIONS="${CHAPTER08_REAL_ITERATIONS:-3}" \
        bash "${repo_dir}/code/engine/scripts/run_real_serving_benchmark.sh"
fi

if [[ ! -f "${experiment_json}" ]]; then
    echo "experiment not found: ${experiment_json}" >&2
    exit 2
fi

cmake -S "${source_dir}" -B "${build_dir}/profile" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}/profile" --target chapter08_profile_report -j2
node "${script_dir}/select_real_service_profile.mjs" \
    "${baseline_json}" "${build_dir}/baseline.txt" "EXP-CPU-02"
node "${script_dir}/select_real_service_profile.mjs" \
    "${experiment_json}" "${build_dir}/experiment.txt" "EXP-CPU-03"

baseline_report=$("${build_dir}/profile/chapter08_profile_report" \
    < "${build_dir}/baseline.txt")
experiment_report=$("${build_dir}/profile/chapter08_profile_report" \
    < "${build_dir}/experiment.txt")
comparison_report=$("${build_dir}/profile/chapter08_profile_report" --compare \
    "${build_dir}/baseline.txt" "${build_dir}/experiment.txt")

full_report=$(printf '%s\n' \
    "[baseline]" "${baseline_report}" \
    "[experiment]" "${experiment_report}" \
    "[comparison]" "${comparison_report}")
printf '%s\n' "${full_report}"
if [[ -n "${report_file}" ]]; then
    mkdir -p "$(dirname "${report_file}")"
    printf '%s\n' "${full_report}" > "${report_file}"
fi

grep -Fq "correctness=PASS" <<<"${comparison_report}"
grep -Fq "decision=KEEP" <<<"${comparison_report}"
