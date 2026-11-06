#!/usr/bin/env bash
# Chapter 08：构建并验证调度、批处理与跨章缓存事务。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER08_SCHEDULING_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter08.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER08_SCHEDULING_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel 2
ctest --test-dir "${build_dir}" -R '^chapter08_' --output-on-failure
