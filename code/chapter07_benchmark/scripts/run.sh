#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
cleanup_build=false
if [[ -n "${CHAPTER07_BUILD_DIR:-}" ]]; then
    build_dir="${CHAPTER07_BUILD_DIR}"
else
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter07.XXXXXX")"
    cleanup_build=true
fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then
        rm -rf "${build_dir}"
    fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${CHAPTER07_BUILD_TYPE:-Release}"
cmake --build "${build_dir}"
"${build_dir}/chapter07_benchmark_demo"
"${build_dir}/chapter07_runtime_benchmark"
result_dir="${CHAPTER07_RESULT_DIR:-${source_dir}/results}"
mkdir -p "${result_dir}"
result_file="${result_dir}/chapter07_cpu_baseline.json"
"${build_dir}/chapter07_multi_request_benchmark" \
    --warmup "${CHAPTER07_WARMUP:-1}" \
    --iterations "${CHAPTER07_ITERATIONS:-3}" \
    --output "${result_file}"
ctest --test-dir "${build_dir}" --output-on-failure
