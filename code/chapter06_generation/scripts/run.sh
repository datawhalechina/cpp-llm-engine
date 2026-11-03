#!/usr/bin/env bash
# Chapter 06：使用隔离目录构建并验证，再运行 generation 和 streaming demo。
# 验证入口：generation、Scheduler、KV Cache、Sampler 和 streaming 测试。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
cleanup_build=false
if [[ -n "${CHAPTER06_BUILD_DIR:-}" ]]; then
    build_dir="${CHAPTER06_BUILD_DIR}"
else
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter06.XXXXXX")"
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
"${build_dir}/chapter06_generation_demo"
ctest --test-dir "${build_dir}" --output-on-failure
"${build_dir}/chapter06_streaming_demo"
