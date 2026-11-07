#!/usr/bin/env bash
# CPU CMake 质量门：分别执行严格警告构建和 UBSan 测试。
# 用法：bash scripts/run_cpp_quality_gate.sh <source-dir> [ctest-regex]
# 支持 ASan 的环境可额外设置 CPP_LLM_ENABLE_ASAN=1。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${1:-}"
test_regex="${2:-}"

if [[ -z "${source_dir}" ]]; then
    echo "usage: $0 <source-dir> [ctest-regex]" >&2
    exit 2
fi
if [[ "${source_dir}" != /* ]]; then
    source_dir="${repo_root}/${source_dir}"
fi
if [[ ! -f "${source_dir}/CMakeLists.txt" ]]; then
    echo "CMake source directory not found: ${source_dir}" >&2
    exit 2
fi

build_root="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-quality.XXXXXX")"
trap 'rm -rf "${build_root}"' EXIT

run_tests() {
    local build_dir="$1"
    local -a args=(--test-dir "${build_dir}" --output-on-failure)
    if [[ -n "${test_regex}" ]]; then
        args+=(-R "${test_regex}")
    fi
    ctest "${args[@]}"
}

echo "[cpp-quality] warnings"
cmake \
    -S "${source_dir}" \
    -B "${build_root}/warnings" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Werror"
cmake --build "${build_root}/warnings" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
run_tests "${build_root}/warnings"

sanitizers="undefined"
if [[ "${CPP_LLM_ENABLE_ASAN:-0}" == "1" ]]; then
    sanitizers="address,undefined"
    export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}"
fi

echo "[cpp-quality] sanitizers=${sanitizers}"
cmake \
    -S "${source_dir}" \
    -B "${build_root}/sanitizers" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=${sanitizers} -fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=${sanitizers}"
cmake --build "${build_root}/sanitizers" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
run_tests "${build_root}/sanitizers"

echo "[cpp-quality] passed"
