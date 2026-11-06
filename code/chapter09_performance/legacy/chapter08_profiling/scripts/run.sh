#!/usr/bin/env bash
# Build Chapter 08 evidence tools and consume Chapter 07 multi-request JSON output.
# Same-build A/B only verifies the evidence chain; it is not an optimization claim.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${source_dir}/../../../.." && pwd)"
build_dir="${CHAPTER08_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-chapter08-build.XXXXXX)}"
chapter07_build_dir="${build_dir}/chapter07"
cleanup_build=false
if [[ -z "${CHAPTER08_BUILD_DIR:-}" ]]; then
    cleanup_build=true
fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j2
ctest --test-dir "${build_dir}" --output-on-failure

cmake -S "${repo_dir}/code/chapter09_performance/legacy/chapter07_benchmark" -B "${chapter07_build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${chapter07_build_dir}" --target chapter07_multi_request_benchmark -j2
"${chapter07_build_dir}/chapter07_multi_request_benchmark" \
    --warmup "${CHAPTER07_WARMUP:-1}" \
    --iterations "${CHAPTER07_ITERATIONS:-3}" \
    --output "${build_dir}/chapter07_cpu_baseline.json" \
    > "${build_dir}/chapter07_multi_request.log"
node "${source_dir}/scripts/select_multi_request_profile.mjs" \
    "${build_dir}/chapter07_cpu_baseline.json" 4 "${build_dir}/baseline.txt"
cp "${build_dir}/baseline.txt" "${build_dir}/experiment.txt"

echo "[chapter08] measured benchmark report"
measured_report=$("${build_dir}/chapter08_profile_report" < "${build_dir}/baseline.txt")
printf '%s\n' "${measured_report}"
grep -Fq "experiment_id=EXP-CPU-01" <<<"${measured_report}"
grep -Fq "workload_id=WL-CPU-TEACHING-BATCH-V1:requests_4" <<<"${measured_report}"
echo "[chapter08] same-build A/B smoke decision (not an optimization claim)"
comparison_output=$("${build_dir}/chapter08_profile_report" --compare \
    "${build_dir}/baseline.txt" "${build_dir}/experiment.txt"
)
printf '%s\n' "${comparison_output}"
grep -Fq "correctness=PASS" <<<"${comparison_output}"
grep -Fq "decision=KEEP" <<<"${comparison_output}"

node "${source_dir}/scripts/select_real_service_profile.mjs" \
    "${repo_dir}/code/chapter09_performance/results/legacy/tinyllama_service_cpu_baseline.json" \
    "${build_dir}/tinyllama_service_profile.txt"
echo "[chapter08] archived TinyLlama ServiceCore profile"
real_report=$("${build_dir}/chapter08_profile_report" \
    < "${build_dir}/tinyllama_service_profile.txt")
printf '%s\n' "${real_report}"
grep -Fq "experiment_id=EXP-CPU-02" <<<"${real_report}"
grep -Fq "workload_id=WL-CPU-REAL-SERVICE-V1" <<<"${real_report}"
grep -Fq "bottleneck_scope=model_compute" <<<"${real_report}"
grep -Fq "execution_pattern=single_thread_round_robin" <<<"${real_report}"

optimized_report=$(node "${source_dir}/scripts/compare_optimized_policy.mjs" \
    "${source_dir}/results/tinyllama_optimized_decoder_scalar.txt" \
    "${source_dir}/results/tinyllama_optimized_decoder_avx2.txt" \
    "${source_dir}/results/tinyllama_optimized_service_scalar.json" \
    "${source_dir}/results/tinyllama_optimized_service_avx2.json")
echo "[chapter08] archived combined optimized policy A/B"
printf '%s\n' "${optimized_report}"
grep -Fq "correctness=PASS" <<<"${optimized_report}"
grep -Fq "decision=KEEP" <<<"${optimized_report}"

batch_profile_report=$(node "${source_dir}/scripts/analyze_real_batch_profile.mjs" \
    "${repo_dir}/code/chapter09_performance/legacy/chapter07_benchmark/results/tinyllama_real_batch_scalar.json")
echo "[chapter08] archived real scalar batch profile"
printf '%s\n' "${batch_profile_report}"
grep -Fq "experiment_id=EXP-CPU-10" <<<"${batch_profile_report}"
grep -Fq "workload_id=WL-CPU-REAL-BATCH-V1" <<<"${batch_profile_report}"
grep -Fq "correctness=PASS" <<<"${batch_profile_report}"
grep -Fq "bottleneck_scope=scalar_batch_kernel" <<<"${batch_profile_report}"
grep -Fq "decision=CORRECTNESS_ONLY" <<<"${batch_profile_report}"
