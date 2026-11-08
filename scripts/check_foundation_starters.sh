#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
starter_root="${root_dir}/exercises/foundation/problems"
tmp_dir="$(mktemp -d /tmp/cpp-llm-engine-foundation-starters.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT

checked=0
for starter in "${starter_root}"/*/starter/main.cpp; do
  name="$(basename "$(dirname "$(dirname "${starter}")")")"
  output="${tmp_dir}/${name}"
  c++ -std=c++17 -Wall -Wextra -Wpedantic "${starter}" -o "${output}"
  if "${output}" >/dev/null 2>&1; then
    echo "Foundation starter unexpectedly passed answer validation: ${name}" >&2
    exit 1
  fi
  checked=$((checked + 1))
done

test -f "${starter_root}/fq04_cmake_ctest/starter/CMakeLists.txt"
test -f "${starter_root}/fq04_cmake_ctest/starter/include/arithmetic.h"
test -f "${starter_root}/fq04_cmake_ctest/starter/src/arithmetic.cpp"
test -f "${starter_root}/fq04_cmake_ctest/starter/app/main.cpp"
test -f "${starter_root}/fq04_cmake_ctest/starter/tests/arithmetic_test.cpp"
test -f "${starter_root}/fq10_vector_add/starter/main.cu"

fq04_build="${tmp_dir}/fq04-build"
cmake -S "${starter_root}/fq04_cmake_ctest/starter" -B "${fq04_build}"
cmake --build "${fq04_build}"
ctest --test-dir "${fq04_build}" -N | grep -Fq 'Total Tests: 0'

fq17_build="${tmp_dir}/fq17-build"
cmake -S "${starter_root}/fq17_minimal_project/starter" -B "${fq17_build}"
cmake --build "${fq17_build}"
if ctest --test-dir "${fq17_build}" --output-on-failure \
    >"${tmp_dir}/fq17-ctest.log" 2>&1; then
  cat "${tmp_dir}/fq17-ctest.log" >&2
  echo "FQ17 starter unexpectedly passed before TODOs were completed" >&2
  exit 1
fi

echo "Foundation starter contract passed: ${checked} single-file starters compile and fail until completed."
echo "FQ04 starts with zero tests; FQ17 builds but its behavior test fails until completed."
