#!/usr/bin/env bash
# Chapter 09：生成真实 TinyLlama 性能原始样本，并转换为统一证据记录。
# 默认结果写入 /tmp；模式可选 service、profile、ab 或 all。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
mode="${1:-service}"
model_path="${2:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
result_root="${CHAPTER09_REAL_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-real-evidence}"
raw_dir="${result_root}/raw"
evidence_dir="${result_root}/evidence"
profile_dir="${result_root}/profile"
engine_scripts="${repo_root}/code/engine/scripts"
importer="${script_dir}/import_legacy_evidence.mjs"
compare_bridge="${script_dir}/compare_evidence_pair.mjs"
build_dir="${CHAPTER09_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-build}"

if [[ ! -f "${model_path}" ]]; then
    echo "model not found: ${model_path}" >&2
    exit 2
fi
case "${mode}" in
    service|profile|ab|all) ;;
    *)
        echo "usage: run_real_evidence.sh [service|profile|ab|all] [model.gguf]" >&2
        exit 2
        ;;
esac

mkdir -p "${raw_dir}" "${evidence_dir}" "${profile_dir}"

import_result() {
    local input="$1"
    node "${importer}" --input "${input}" --output-dir "${evidence_dir}"
}

run_service() {
    local raw_file="${raw_dir}/tinyllama-service-scalar-sequential.json"
    REAL_SERVING_RESULT_FILE="${raw_file}" \
    REAL_SERVING_WARMUP="${CHAPTER09_REAL_WARMUP:-1}" \
    REAL_SERVING_ITERATIONS="${CHAPTER09_REAL_ITERATIONS:-3}" \
    REAL_SERVING_EXECUTION_POLICY=scalar \
    REAL_SERVING_DISPATCH_MODE=sequential \
        bash "${engine_scripts}/run_real_serving_benchmark.sh" "${model_path}"
    import_result "${raw_file}"
}

run_profile() {
    REAL_DECODER_PROFILE_RESULT_FILE="${profile_dir}/tinyllama-decoder-profile.txt" \
    REAL_DECODER_PROFILE_ITERATIONS="${CHAPTER09_PROFILE_ITERATIONS:-3}" \
        bash "${engine_scripts}/run_real_decoder_profile.sh" "${model_path}"
}

run_ab() {
    PARALLEL_AB_RESULT_DIR="${raw_dir}" \
    PARALLEL_AB_WARMUP="${CHAPTER09_REAL_WARMUP:-1}" \
    PARALLEL_AB_ITERATIONS="${CHAPTER09_REAL_ITERATIONS:-3}" \
        bash "${engine_scripts}/run_real_parallel_requests_ab.sh" "${model_path}"
    local baseline_path experiment_path
    baseline_path="$(import_result "${raw_dir}/tinyllama_service_sequential_avx2.json" | tail -n 1)"
    experiment_path="$(import_result "${raw_dir}/tinyllama_service_parallel_requests_avx2.json" | tail -n 1)"
    cmake -S "${repo_root}/code/chapter09_performance" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" --target chapter09_compare_evidence --parallel 2
    node "${compare_bridge}" \
        "${build_dir}/chapter09_compare_evidence" \
        "${baseline_path}" "${experiment_path}" "${evidence_dir}/real-service-ab-decision.json"
}

case "${mode}" in
    service) run_service ;;
    profile) run_profile ;;
    ab) run_ab ;;
    all)
        run_service
        run_profile
        run_ab
        ;;
esac

echo "chapter09_mode=${mode}"
echo "chapter09_result_root=${result_root}"
echo "chapter09_real_evidence=PASS"
