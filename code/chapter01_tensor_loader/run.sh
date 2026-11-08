#!/usr/bin/env bash
# Chapter 01：隔离构建并运行本章 Loader、Schema 和 Tokenizer/Payload 测试。
# 默认使用临时目录并在退出时清理；设置 CHAPTER01_BUILD_DIR 可保留构建产物。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${CHAPTER01_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-chapter01.XXXXXX)}"
cleanup_build=false
if [[ -z "${CHAPTER01_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${script_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" --output-on-failure -R '^chapter01_'
