#!/usr/bin/env bash
# Foundation：编译并运行学习者完成的单文件 C++ 答案。
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <answer.cpp>" >&2
  exit 2
fi

answer="$1"
if [[ ! -f "${answer}" ]]; then
  echo "answer not found: ${answer}" >&2
  exit 2
fi

output="$(mktemp "${TMPDIR:-/tmp}/cpp-llm-foundation-answer.XXXXXX")"
trap 'rm -f "${output}"' EXIT

c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror "${answer}" -o "${output}"
"${output}"
