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
  checked=$((checked + 1))
done

test -f "${starter_root}/fq04_cmake_ctest/starter/CMakeLists.txt"
test -f "${starter_root}/fq04_cmake_ctest/starter/include/arithmetic.h"
test -f "${starter_root}/fq04_cmake_ctest/starter/src/arithmetic.cpp"
test -f "${starter_root}/fq04_cmake_ctest/starter/app/main.cpp"
test -f "${starter_root}/fq04_cmake_ctest/starter/tests/arithmetic_test.cpp"
test -f "${starter_root}/fq10_vector_add/starter/main.cu"
fq17_build="${tmp_dir}/fq17-build"
cmake -S "${starter_root}/fq17_minimal_project/starter" -B "${fq17_build}"
cmake --build "${fq17_build}"
ctest --test-dir "${fq17_build}" --output-on-failure
echo "Foundation starter compile check passed: ${checked} C++ starters; FQ04/FQ10 CUDA starter files present."
echo "Foundation starter project check passed: FQ17 library/app/test starter verified."
