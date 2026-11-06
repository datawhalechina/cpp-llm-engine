#!/usr/bin/env bash
# Engine/Chapter 08：构建并运行真实 TinyLlama CPU Decoder 子阶段 profiling。
# 验证 golden token 后，归档 embedding、projection、attention、FFN 和 LM Head 耗时。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_REAL_MODEL:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
llama_root="${LLAMA_CPP_ROOT:-${repo_root}/llama.cpp}"
if [[ -n "${LLAMA_CPP_BUILD_DIR:-}" ]]; then
    llama_build_dir="${LLAMA_CPP_BUILD_DIR}"
elif [[ -d "${llama_root}/build-cpu/bin" ]]; then
    llama_build_dir="${llama_root}/build-cpu"
else
    llama_build_dir="${llama_root}/build"
fi
build_dir="${REAL_DECODER_PROFILE_BUILD_DIR:-/tmp/cpp-llm-engine-real-decoder-profile}"
result_file="${REAL_DECODER_PROFILE_RESULT_FILE:-${repo_root}/code/chapter09_performance/legacy/chapter08_profiling/results/tinyllama_decoder_phase_profile.txt}"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi

mkdir -p "$(dirname "${result_file}")"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
    -DLLAMA_CPP_ROOT="${llama_root}" \
    -DLLAMA_CPP_BUILD_DIR="${llama_build_dir}" \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target real_tinyllama_decoder_profile -j2
profile_output=$("${build_dir}/real_tinyllama_decoder_profile" \
    "${model_path}" "${REAL_DECODER_PROFILE_ITERATIONS:-3}" \
    "${REAL_DECODER_PROFILE_BACKEND:-${REAL_DECODER_PROFILE_FFN_DOWN_BACKEND:-scalar}}")
printf '%s\n' "${profile_output}"
printf '%s\n' "${profile_output}" > "${result_file}"
grep -Fq "correctness=PASS" <<<"${profile_output}"
grep -Fq "profile_complete=PASS" <<<"${profile_output}"
grep -Fq "ffn_profile_complete=PASS" <<<"${profile_output}"
