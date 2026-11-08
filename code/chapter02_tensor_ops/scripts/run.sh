#!/usr/bin/env bash
# Chapter 02：Tensor view 验证脚本
# 执行：隔离配置、构建示例，并运行 Chapter02 的两个 CTest。
# 参数：CHAPTER02_BUILD_DIR 可指定持久 build 目录。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER02_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-chapter02.XXXXXX)}"
cleanup_build=false
if [[ -z "${CHAPTER02_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/chapter02_tensor_ops_demo"
ctest --test-dir "${build_dir}" -R '^chapter02_' --output-on-failure
