#!/usr/bin/env bash
# CPU 推理服务学习入口。
# 验证性能证据、ServiceCore、真实 HTTP socket 和顺序/并发 A/B。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
result_dir="${CPU_SERVICE_LEARNING_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-engine-service-learning}"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    echo "set CPP_LLM_ENGINE_MODEL_PATH to a TinyLlama GGUF file" >&2
    exit 2
fi

echo "[cpu-service] chapter09 evidence contracts"
bash "${repo_root}/code/chapter09_performance/scripts/run.sh"

echo "[cpu-service] chapter10 service and HTTP contracts"
bash "${repo_root}/code/chapter10_serving/scripts/run.sh"

echo "[cpu-service] real HTTP socket loopback"
CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
    bash "${repo_root}/code/engine/scripts/run_real_http_server.sh" --check

echo "[cpu-service] sequential/parallel paired benchmark"
CPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
PARALLEL_AB_RESULT_DIR="${result_dir}" \
PARALLEL_AB_EXECUTION_POLICY="${CPU_SERVICE_EXECUTION_POLICY:-scalar}" \
    bash "${repo_root}/code/engine/scripts/run_real_parallel_requests_ab.sh"

echo "[cpu-service] results: ${result_dir}"
echo "[cpu-service] learning path passed"
