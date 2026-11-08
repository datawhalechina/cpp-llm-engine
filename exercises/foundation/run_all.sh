#!/usr/bin/env bash
# Foundation：隔离构建并运行 17 道参考实现的 CTest。
# 输入：可选 FOUNDATION_BUILD_DIR；输出：完整题库测试结果。
# 验证：参考答案全部通过，不替代学习者 starter 的答案检查。
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 使用临时目录，避免其他工作区留下的 CMake 缓存影响结果。
build_dir="${FOUNDATION_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-foundation.XXXXXX)}"
cleanup_build=false
if [[ -z "${FOUNDATION_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${root_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" --output-on-failure
