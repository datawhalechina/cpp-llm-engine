#!/usr/bin/env bash
# Chapter 00：构建并运行不依赖模型的仓库连通性检查。
# 输入：可选的 CHAPTER00_BUILD_DIR；输出：Chapter00 CTest 结果。
# 验证：确认 C++17、CMake、脚本路径和最小可执行程序可以工作。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 使用临时目录，避免旧路径中的 CMake 缓存影响检查结果。
build_dir="${CHAPTER00_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-chapter00.XXXXXX)}"
cleanup_build=false
if [[ -z "${CHAPTER00_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${script_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" -R '^chapter00_' --output-on-failure
