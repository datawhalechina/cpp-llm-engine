#!/usr/bin/env bash
# Chapter 05 默认入口：隔离构建量化示例并运行本章及依赖契约测试。
# 不需要真实模型文件；构建目录默认在 /tmp，退出时自动清理。
# 验证入口：bash code/chapter05_quantization/scripts/run.sh。
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
