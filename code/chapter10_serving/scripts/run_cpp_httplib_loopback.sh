#!/usr/bin/env bash
# Chapter 10：构建 cpp-httplib Transport 并执行本地 HTTP/SSE 回环。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER10_HTTP_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-chapter10-http.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER10_HTTP_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
port="${CHAPTER10_HTTP_PORT:-18083}"
include_dir="${CHAPTER10_CPPHTTPLIB_INCLUDE_DIR:-}"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake_args=(-DCHAPTER10_ENABLE_CPPHTTPLIB=ON)
if [[ -n "${include_dir}" ]]; then
    cmake_args+=("-DCHAPTER10_CPPHTTPLIB_INCLUDE_DIR=${include_dir}")
fi
cmake -S "${source_dir}" -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel 2

"${build_dir}/chapter10_cpp_httplib_server" "${port}" &
server_pid=$!
sleep 1

health="$(curl --fail --silent --show-error --max-time 10 \
    "http://127.0.0.1:${port}/health")"
[[ "${health}" == '{"status":"ok"}' ]]

metrics="$(curl --fail --silent --show-error --max-time 10 \
    "http://127.0.0.1:${port}/metrics")"
grep -q '^serving_accepted_total 0$' <<<"${metrics}"

completion="$(curl --fail --silent --show-error --max-time 10 \
    -H 'Content-Type: application/json' \
    -d '{"request_id":21,"prompt_tokens":[1,2]}' \
    "http://127.0.0.1:${port}/v1/completions")"
grep -q '"finish_reason":"stop"' <<<"${completion}"

stream="$(curl --fail --silent --show-error --max-time 10 -N \
    -H 'Content-Type: application/json' \
    -d '{"request_id":22,"prompt_tokens":[1,2],"stream":true}' \
    "http://127.0.0.1:${port}/v1/completions")"
grep -q 'data: {"token":' <<<"${stream}"
grep -q 'finish_reason":"stop' <<<"${stream}"

echo "Chapter10 cpp-httplib loopback passed"
