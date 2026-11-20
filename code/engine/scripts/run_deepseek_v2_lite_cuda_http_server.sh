#!/usr/bin/env bash
# Engine/Chapter10: run DeepSeek-V2-Lite CUDA HTTP or check a text/SSE loopback.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
check_mode=false
if [[ "${1:-}" == "--check" ]]; then
    check_mode=true
    shift
fi

model_path="${1:-${CPP_LLM_ENGINE_DEEPSEEK2_MODEL_PATH:-${repo_root}/llama.cpp/models/deepseek-v2-lite-chat/DeepSeek-V2-Lite-Chat-Q2_K.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_root}/build}"
httplib_include="${CHAPTER10_CPPHTTPLIB_INCLUDE_DIR:-${llama_root}/vendor/cpp-httplib}"
build_dir="${DEEPSEEK2_CUDA_HTTP_BUILD_DIR:-/tmp/cpp-llm-deepseek2-cuda-http-server}"
port="${DEEPSEEK2_CUDA_HTTP_PORT:-18085}"
bind_address="${DEEPSEEK2_CUDA_HTTP_BIND_ADDRESS:-127.0.0.1}"
startup_timeout_seconds="${DEEPSEEK2_CUDA_HTTP_STARTUP_TIMEOUT_SECONDS:-240}"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi
}
trap cleanup EXIT

if [[ ! -x "${cuda_home}/bin/nvcc" || ! -f "${model_path}" ]]; then
    echo "CUDA compiler or DeepSeek-V2-Lite model is missing" >&2
    exit 2
fi
if [[ ! -f "${llama_root}/include/llama.h" ||
      ! -d "${llama_build_dir}/bin" ||
      ! -f "${httplib_include}/httplib.h" ]]; then
    echo "llama.cpp tokenizer build or cpp-httplib header is missing" >&2
    exit 2
fi

cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_DEEPSEEK2_MODEL="${model_path}" \
    -DCHAPTER10_ENABLE_CPPHTTPLIB=ON \
    -DCHAPTER10_CPPHTTPLIB_INCLUDE_DIR="${httplib_include}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_deepseek2_cuda_http_server -j2

server="${build_dir}/real_deepseek2_cuda_http_server"
if [[ "${check_mode}" != true ]]; then
    exec "${server}" "${model_path}" "${port}" "${bind_address}"
fi

server_log="${build_dir}/real_deepseek2_cuda_http_server.log"
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
        echo "DeepSeek CUDA HTTP server exited before readiness" >&2
        tail -n 40 "${server_log}" >&2
        exit 1
    fi
    sleep 1
done
if [[ "${ready}" != true ]]; then
    echo "DeepSeek CUDA HTTP server readiness timed out" >&2
    tail -n 40 "${server_log}" >&2
    exit 1
fi

base_url="http://${bind_address}:${port}"
completion="$(curl --fail --silent --show-error --max-time 180 \
    -H 'Content-Type: application/json' \
    -d '{"request_id":401,"prompt":"Hello DeepSeek"}' \
    "${base_url}/v1/completions")"
grep -Eq '"generated_token_ids":\[[0-9]+,[0-9]+\]' <<<"${completion}"

stream="$(curl --fail --silent --show-error --max-time 180 -N \
    -H 'Content-Type: application/json' \
    -d '{"request_id":402,"prompt":"Hi","stream":true}' \
    "${base_url}/v1/completions")"
[[ "$(grep -c '"token":' <<<"${stream}")" -eq 2 ]]
grep -Fq '"finish_reason":"stop"' <<<"${stream}"

metrics="$(curl --fail --silent --show-error --max-time 10 \
    "${base_url}/metrics")"
grep -Fq 'serving_accepted_total 2' <<<"${metrics}"
grep -Fq 'serving_completed_total 2' <<<"${metrics}"
grep -Fq 'serving_released_total 2' <<<"${metrics}"
grep -Fq 'serving_inflight 0' <<<"${metrics}"

echo "DeepSeek CUDA HTTP loopback passed: text tokenizer, completion, SSE, release"
