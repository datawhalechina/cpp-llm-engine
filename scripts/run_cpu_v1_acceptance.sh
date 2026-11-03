#!/usr/bin/env bash
# 兼容入口：统一验收真实 CPU 基线、Decoder、Runtime 和最小服务。
# 产物：在临时结果目录保存 commit、模型、环境和各子任务生成的结果路径。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
llama_simple="${LLAMA_SIMPLE_BIN:-${llama_root}/build/bin/llama-simple}"
result_dir="${CPU_V1_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-engine-cpu-v1}"
manifest="${result_dir}/acceptance.txt"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    echo "set CPP_LLM_ENGINE_MODEL_PATH to a TinyLlama GGUF file" >&2
    exit 2
fi
if [[ ! -x "${llama_simple}" ]]; then
    echo "llama-simple not found: ${llama_simple}" >&2
    echo "build llama.cpp for CPU or set LLAMA_SIMPLE_BIN" >&2
    exit 2
fi

mkdir -p "${result_dir}"
{
    echo "started_status=RUNNING"
    echo "commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "model=${model_path}"
    echo "model_bytes=$(stat -c '%s' "${model_path}")"
    echo "compiler=${CXX:-c++} $("${CXX:-c++}" -dumpfullversion -dumpversion)"
    echo "started_at=$(date --iso-8601=seconds)"
} >"${manifest}"

echo "[cpu-v1] task 1: real CPU baseline"
bash "${repo_root}/code/chapter00_quickstart/run.sh"
"${llama_simple}" -m "${model_path}" -n 1 -ngl 0 "Hello TinyLlama"

echo "[cpu-v1] task 2: tokenizer and GGUF loader"
CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
LLAMA_CPP_ROOT="${llama_root}" \
    bash "${repo_root}/code/chapter01_tensor_loader/run_real_tokenizer.sh"

echo "[cpu-v1] task 3: tensor and operator"
bash "${repo_root}/scripts/run_tensor_operator_learning.sh"

echo "[cpu-v1] task 4: real 22-layer decoder"
CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
LLAMA_CPP_ROOT="${llama_root}" \
    bash "${repo_root}/scripts/run_real_decoder_learning.sh"

echo "[cpu-v1] task 5: generation runtime"
bash "${repo_root}/code/chapter06_generation/scripts/run.sh"

echo "[cpu-v1] task 6: real CPU service"
CPP_LLM_ENGINE_MODEL_PATH="${model_path}" \
CPU_SERVICE_LEARNING_RESULT="${result_dir}/service-benchmark.json" \
    bash "${repo_root}/scripts/run_cpu_service_learning.sh"

{
    echo "status=PASS"
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "service_result=${result_dir}/service-benchmark.json"
} >>"${manifest}"

echo "[cpu-v1] manifest: ${manifest}"
echo "[cpu-v1] acceptance passed"
