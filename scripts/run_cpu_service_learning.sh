#!/usr/bin/env bash
# CPU 推理服务学习入口。
# 验证 benchmark/profiling、ServiceCore、真实 HTTP socket 和双请求基线。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
result_file="${CPU_SERVICE_LEARNING_RESULT:-${TMPDIR:-/tmp}/cpp-llm-engine-service-learning.json}"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    echo "set CPP_LLM_ENGINE_MODEL_PATH to a TinyLlama GGUF file" >&2
    exit 2
fi

echo "[cpu-service] benchmark contracts"
CHAPTER07_RESULT_DIR="${TMPDIR:-/tmp}/cpp-llm-engine-chapter07-learning-results" \
    bash "${repo_root}/code/chapter07_kv_cache/scripts/run.sh"

echo "[cpu-service] profiling method"
bash "${repo_root}/code/chapter08_scheduling/scripts/run.sh"

echo "[cpu-service] service and HTTP contracts"
bash "${repo_root}/code/chapter09_performance/scripts/run.sh"
bash "${repo_root}/code/chapter10_serving/scripts/run.sh"

echo "[cpu-service] real HTTP socket loopback"
CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
    bash "${repo_root}/code/engine/scripts/run_real_http_server.sh" --check

echo "[cpu-service] real two-request benchmark"
CPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
REAL_SERVING_RESULT_FILE="${result_file}" \
    bash "${repo_root}/code/engine/scripts/run_real_serving_benchmark.sh"

echo "[cpu-service] result: ${result_file}"
echo "[cpu-service] learning path passed"
