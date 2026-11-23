#!/usr/bin/env bash
# 真实 TinyLlama CPU Generation Runtime 学习入口。
# 验证 Chapter06-08 组件，再运行真实文本生成和双请求 batch。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
if [[ -n "${LLAMA_CPP_BUILD_DIR:-}" ]]; then
    llama_build_dir="${LLAMA_CPP_BUILD_DIR}"
elif [[ -d "${llama_root}/build-cpu/bin" ]]; then
    llama_build_dir="${llama_root}/build-cpu"
else
    llama_build_dir="${llama_root}/build"
fi
build_dir="${GENERATION_RUNTIME_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-generation-runtime.XXXXXX")}"
cleanup_build=false
if [[ -z "${GENERATION_RUNTIME_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    echo "set CPP_LLM_ENGINE_MODEL_PATH to a TinyLlama GGUF file" >&2
    exit 2
fi
if [[ ! -f "${llama_root}/CMakeLists.txt" ]]; then
    echo "llama.cpp source not found: ${llama_root}" >&2
    echo "set LLAMA_CPP_ROOT to the llama.cpp checkout" >&2
    exit 2
fi

echo "[generation-runtime] chapter06 generation semantics"
bash "${repo_root}/code/chapter06_generation/scripts/run_generation_core.sh"

echo "[generation-runtime] chapter07 paged K/V transactions"
bash "${repo_root}/code/chapter07_kv_cache/scripts/run.sh"

echo "[generation-runtime] chapter08 scheduling and tensor batch"
bash "${repo_root}/code/chapter08_scheduling/scripts/run.sh"

echo "[generation-runtime] configure real model integration"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_generation_smoke real_tinyllama_batch_smoke -j2
ctest --test-dir "${build_dir}" \
    -R '^real_tinyllama_(generation|batch)_test$' \
    --output-on-failure

echo "[generation-runtime] learning path passed"
