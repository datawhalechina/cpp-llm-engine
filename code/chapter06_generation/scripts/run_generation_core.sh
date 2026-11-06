#!/usr/bin/env bash
# Chapter 06 Generation Core：构建并验证单请求生成，不运行 KV 分页或调度测试。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER06_CORE_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter06-core.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER06_CORE_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel 2 --target \
    chapter06_generation_demo \
    chapter06_streaming_demo \
    chapter06_generation_test \
    chapter06_sampler_test \
    chapter06_token_history_test \
    chapter06_request_contract_test
ctest --test-dir "${build_dir}" \
    -R '^chapter06_(generation|sampler|token_history|request_contract)_test$' \
    --output-on-failure
"${build_dir}/chapter06_generation_demo"
"${build_dir}/chapter06_streaming_demo"
