#!/usr/bin/env bash
# Chapter 10：构建并验证服务生命周期、调度交接和 HTTP/SSE 契约。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER10_SERVING_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter10.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER10_SERVING_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel 2
ctest --test-dir "${build_dir}" -R '^chapter10_' --output-on-failure
