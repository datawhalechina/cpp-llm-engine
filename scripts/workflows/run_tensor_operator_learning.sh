#!/usr/bin/env bash
# Tensor/View/Operator/MatMul CPU 学习入口。
# 先验证共享 storage 和 view，再验证公共 contract 与 CPU MatMul backend。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[tensor-operator] chapter02 tensor views"
bash "${repo_root}/code/chapter02_tensor_ops/scripts/run.sh"

echo "[tensor-operator] chapter03 matmul backend"
bash "${repo_root}/code/chapter03_matmul/scripts/run.sh"

echo "[tensor-operator] learning path passed"
