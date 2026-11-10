#!/usr/bin/env bash
# Engine/Chapter10：启动 DeepSeek-V2-Lite HTTP 服务，或执行本地 socket 回环。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
check_mode=false
if [[ "${1:-}" == "--check" ]]; then
    check_mode=true
    shift
fi

model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
build_dir="${DEEPSEEK2_HTTP_BUILD_DIR:-/tmp/cpp-llm-deepseek2-http-server}"
port="${DEEPSEEK2_HTTP_PORT:-18084}"
bind_address="${DEEPSEEK2_HTTP_BIND_ADDRESS:-127.0.0.1}"
startup_timeout_seconds="${DEEPSEEK2_HTTP_STARTUP_TIMEOUT_SECONDS:-180}"
httplib_include="${CHAPTER10_CPPHTTPLIB_INCLUDE_DIR:-${repo_root}/llama.cpp/vendor/cpp-httplib}"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi
}
trap cleanup EXIT

if [[ ! -f "${model_path}" ]]; then
    echo "DeepSeek-V2-Lite GGUF not found: ${model_path}" >&2
    exit 2
fi
if [[ ! -f "${httplib_include}/httplib.h" ]]; then
    echo "httplib.h not found under: ${httplib_include}" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${model_path}" \
    -DCHAPTER10_ENABLE_CPPHTTPLIB=ON \
    -DCHAPTER10_CPPHTTPLIB_INCLUDE_DIR="${httplib_include}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_deepseek2_http_server -j2

server="${build_dir}/real_deepseek2_http_server"
if [[ "${check_mode}" != true ]]; then
    exec "${server}" "${model_path}" "${port}" "${bind_address}"
fi

server_log="${build_dir}/real_deepseek2_http_server.log"
"${server}" "${model_path}" "${port}" "${bind_address}" >"${server_log}" 2>&1 &
server_pid=$!
ready=false
for ((attempt = 0; attempt < startup_timeout_seconds; ++attempt)); do
    if curl --fail --silent --max-time 1 \
        "http://${bind_address}:${port}/health" >/dev/null; then
        ready=true
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}" || true
        echo "DeepSeek HTTP server exited before readiness" >&2
        tail -n 40 "${server_log}" >&2
        exit 1
    fi
    sleep 1
done
if [[ "${ready}" != true ]]; then
    echo "DeepSeek HTTP server readiness timed out" >&2
    tail -n 40 "${server_log}" >&2
    exit 1
fi

base_url="http://${bind_address}:${port}"
health="$(curl --fail --silent --show-error --max-time 10 \
    "${base_url}/health")"
[[ "${health}" == '{"status":"ok"}' ]]

completion="$(curl --fail --silent --show-error --max-time 120 \
    -H 'Content-Type: application/json' \
    -d '{"request_id":301,"prompt_tokens":[0]}' \
    "${base_url}/v1/completions")"
grep -Fq '"generated_token_ids":[185,85770]' <<<"${completion}"

stream="$(curl --fail --silent --show-error --max-time 120 -N \
    -H 'Content-Type: application/json' \
    -d '{"request_id":302,"prompt_tokens":[0],"stream":true}' \
    "${base_url}/v1/completions")"
for token in 185 85770; do
    grep -Fq "\"token\":${token}" <<<"${stream}"
done
grep -Fq '"finish_reason":"stop"' <<<"${stream}"

metrics="$(curl --fail --silent --show-error --max-time 10 \
    "${base_url}/metrics")"
grep -Fq 'serving_accepted_total 2' <<<"${metrics}"
grep -Fq 'serving_completed_total 2' <<<"${metrics}"
grep -Fq 'serving_released_total 2' <<<"${metrics}"
grep -Fq 'serving_generated_tokens_total 4' <<<"${metrics}"

echo "DeepSeek HTTP loopback passed: health, completion, SSE, metrics"
