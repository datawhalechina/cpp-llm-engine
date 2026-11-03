#!/usr/bin/env bash
# Release 契约回归：验证基础对象、Runtime、ServiceCore 和 Engine 的关键契约。
# 范围：不下载模型，不启用 GPU，不运行 benchmark。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-release-contracts.XXXXXX")"
trap 'rm -rf "${build_root}"' EXIT

run_suite() {
    local name="$1"
    local source_dir="$2"
    shift 2
    local build_dir="${build_root}/${name}"

    echo "[release-contracts] ${name}"
    cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" --parallel 2
    ctest --test-dir "${build_dir}" --output-on-failure "$@"
}

run_suite testing "${repo_root}/code/testing"
run_suite core "${repo_root}/code/core" -L contract
run_suite chapter01 "${repo_root}/code/chapter01_tensor_loader" -L loader
run_suite chapter02 "${repo_root}/code/chapter02_tensor_ops" -L tensor
run_suite chapter03 "${repo_root}/code/chapter03_matmul" -L operator
run_suite chapter04 "${repo_root}/code/chapter04_attention" -L attention
run_suite chapter05 "${repo_root}/code/chapter05_quantization" -L quantization
run_suite chapter06 "${repo_root}/code/chapter06_generation" -L contract
run_suite chapter09 "${repo_root}/code/chapter09_serving" -R '^chapter09_'
run_suite engine "${repo_root}/code/engine" -L engine
run_suite multimodal "${repo_root}/code/multimodal" -L multimodal

echo "[release-contracts] passed"
