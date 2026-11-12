#!/usr/bin/env bash
# 故障注入专项门：用严格警告和 Sanitizer 验证注入器、事务回滚与终态释放。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
quality_gate="${repo_root}/scripts/run_cpp_quality_gate.sh"

echo "[fault-injection] testing infrastructure"
bash "${quality_gate}" \
    code/testing \
    '^cpp_llm_testing_self_test$'

echo "[fault-injection] scheduler transaction rollback"
bash "${quality_gate}" \
    code/chapter08_scheduling \
    '^chapter08_cache_transaction_test$'

echo "[fault-injection] serving terminal release"
bash "${quality_gate}" \
    code/chapter10_serving \
    '^(chapter10_terminal_release_test|chapter10_http_contract_test)$'

echo "[fault-injection] passed"
