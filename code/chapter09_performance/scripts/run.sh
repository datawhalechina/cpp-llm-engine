#!/usr/bin/env bash
# Chapter 09：构建并验证统一性能证据、统计和 A/B 决策契约。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER09_PERFORMANCE_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter09.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER09_PERFORMANCE_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel 2
ctest --test-dir "${build_dir}" -R '^chapter09_' --output-on-failure
