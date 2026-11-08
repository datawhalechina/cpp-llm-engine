#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
problem_root="${root_dir}/exercises/foundation/problems"
expected=17
count=0

for problem in "${problem_root}"/fq*; do
  test -d "${problem}"
  test -f "${problem}/README.md"
  if [[ "${problem}" == *"/fq17_minimal_project" ]]; then
    test -f "${problem}/starter/CMakeLists.txt"
    test -f "${problem}/starter/app/main.cpp"
    test -f "${problem}/starter/tests/engine_test.cpp"
  else
    test -f "${problem}/starter/main.cpp" || test -f "${problem}/starter/README.md"
  fi
  for heading in '## 任务' '## 完成标准' '## 验证' '## 提示' '## 主线映射'; do
    grep -Fqx "${heading}" "${problem}/README.md"
  done
  if [[ "${problem}" == *"/fq04_cmake_ctest" ||
        "${problem}" == *"/fq17_minimal_project" ]]; then
    grep -Fq 'ctest --test-dir' "${problem}/README.md"
  else
    grep -Fq 'check_foundation_answer.sh' "${problem}/README.md"
  fi
  grep -Fq 'exercises/foundation/run_all.sh' "${problem}/README.md"
  count=$((count + 1))
done

test "${count}" -eq "${expected}"
echo "Foundation metadata check passed: ${count} problems with standard sections."
