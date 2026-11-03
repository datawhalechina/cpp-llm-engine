#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
cleanup_build=false
if [[ -n "${CHAPTER05_BUILD_DIR:-}" ]]; then
    build_dir="${CHAPTER05_BUILD_DIR}"
else
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter05.XXXXXX")"
    cleanup_build=true
fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then
        rm -rf "${build_dir}"
    fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/chapter05_quantization_demo"
ctest --test-dir "${build_dir}" --output-on-failure
