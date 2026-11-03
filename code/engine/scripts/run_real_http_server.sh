#!/usr/bin/env bash
# Engine/Chapter 09：构建并启动真实 TinyLlama CPU HTTP 服务。
# 使用 --check 时执行 health、同步 completion、SSE 和 metrics 的本地 curl 闭环后退出。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
check_mode=false
if [[ "${1:-}" == "--check" ]]; then
    check_mode=true
    shift
fi

model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${CPP_LLM_ENGINE_REAL_MODEL:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}}"
build_dir="${REAL_HTTP_BUILD_DIR:-/tmp/cpp-llm-engine-real-http-server}"
port="${REAL_HTTP_PORT:-18082}"
bind_address="${REAL_HTTP_BIND_ADDRESS:-127.0.0.1}"
startup_timeout_seconds="${REAL_HTTP_STARTUP_TIMEOUT_SECONDS:-180}"
httplib_include="${CHAPTER09_CPPHTTPLIB_INCLUDE_DIR:-${repo_root}/llama.cpp/vendor/cpp-httplib}"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi
}
trap cleanup EXIT

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi
if [[ ! -f "${httplib_include}/httplib.h" ]]; then
    echo "httplib.h not found under: ${httplib_include}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${repo_root}/llama.cpp" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCHAPTER09_ENABLE_CPPHTTPLIB=ON \
    -DCHAPTER09_CPPHTTPLIB_INCLUDE_DIR="${httplib_include}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_tinyllama_http_server -j2

server="${build_dir}/real_tinyllama_http_server"
if [[ "${check_mode}" != true ]]; then
    exec "${server}" "${model_path}" "${port}" "${bind_address}"
fi

server_log="${build_dir}/real_tinyllama_http_server.log"
"${server}" "${model_path}" "${port}" "${bind_address}" >"${server_log}" 2>&1 &
server_pid=$!
ready=false
for ((attempt = 0; attempt < startup_timeout_seconds; ++attempt)); do
    if curl --fail --silent --max-time 1 "http://${bind_address}:${port}/health" >/dev/null; then
        ready=true
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}" || true
        echo "real HTTP server exited before readiness" >&2
        tail -n 40 "${server_log}" >&2
        exit 1
    fi
    sleep 1
done
if [[ "${ready}" != true ]]; then
    echo "real HTTP server was not ready after ${startup_timeout_seconds}s" >&2
    tail -n 40 "${server_log}" >&2
    exit 1
fi

base_url="http://${bind_address}:${port}"
health="$(curl --fail --silent --show-error --max-time 10 "${base_url}/health")"
[[ "${health}" == '{"status":"ok"}' ]]

completion="$(curl --fail --silent --show-error --max-time 120 \
    -H 'Content-Type: application/json' \
    -d '{"request_id":101,"prompt_tokens":[15043,18439,1460,29931,5288,29874]}' \
    "${base_url}/v1/completions")"
grep -Fq '"generated_token_ids":[29892,306,29915,29885]' <<<"${completion}"

stream="$(curl --fail --silent --show-error --max-time 120 -N \
    -H 'Content-Type: application/json' \
    -d '{"request_id":102,"prompt_tokens":[15043,18439,1460,29931,5288,29874],"stream":true}' \
    "${base_url}/v1/completions")"
for token in 29892 306 29915 29885; do
    grep -Fq "\"token\":${token}" <<<"${stream}"
done
grep -Fq '"finish_reason":"stop"' <<<"${stream}"

metrics="$(curl --fail --silent --show-error --max-time 10 "${base_url}/metrics")"
grep -Fq 'serving_accepted_total 2' <<<"${metrics}"
grep -Fq 'serving_completed_total 2' <<<"${metrics}"
grep -Fq 'serving_released_total 2' <<<"${metrics}"
grep -Fq 'serving_generated_tokens_total 8' <<<"${metrics}"

echo "real HTTP loopback passed: health, sync golden, SSE golden, metrics"
