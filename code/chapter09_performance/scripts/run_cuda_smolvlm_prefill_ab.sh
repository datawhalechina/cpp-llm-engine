#!/usr/bin/env bash
# Chapter09：重复采集 SmolVLM 双请求串行/组批 Prefill A/B。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
runner="${repo_root}/code/multimodal/scripts/run_cuda_smolvlm_context.sh"
result_dir="${CHAPTER09_SMOLVLM_PREFILL_RESULT_DIR:-/tmp/cpp-llm-smolvlm-prefill-ab}"
iterations="${CHAPTER09_SMOLVLM_PREFILL_ITERATIONS:-2}"
mmproj="${SMOLVLM_MMPROJ_PATH:-${repo_root}/llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf}"
model="${SMOLVLM_MODEL_PATH:-${repo_root}/llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf}"
image="${SMOLVLM_IMAGE_PATH:-${repo_root}/llama.cpp/tools/mtmd/test-1.jpeg}"

if ! [[ "${iterations}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHAPTER09_SMOLVLM_PREFILL_ITERATIONS must be a positive integer" >&2
    exit 2
fi
mkdir -p "${result_dir}"
printf 'run\tserial_prefill_ms\tbatched_prefill_ms\tratio\n' > "${result_dir}/paired.tsv"

for ((run = 1; run <= iterations; ++run)); do
    output_path="${result_dir}/run-${run}.txt"
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
        bash "${runner}" "${mmproj}" "${model}" "${image}" mixed-service \
        > "${output_path}"
    grep -q '^service_metrics_match: true$' "${output_path}"
    grep -q '^service_batch_match: true$' "${output_path}"
    grep -q '^service_async_packed_prefill_calls: 1$' "${output_path}"
    grep -q '^service_packed_prefill_calls: 1$' "${output_path}"
    grep -q '^service_prefill_workspaces: 1$' "${output_path}"
    serial="$(awk '/^service_serial_prefill_ms:/ {print $2}' "${output_path}")"
    batched="$(awk '/^service_batched_prefill_ms:/ {print $2}' "${output_path}")"
    ratio="$(awk -v serial="${serial}" -v batched="${batched}" \
        'BEGIN { printf "%.6f", batched / serial }')"
    printf '%s\t%s\t%s\t%s\n' \
        "${run}" "${serial}" "${batched}" "${ratio}" \
        >> "${result_dir}/paired.tsv"
done

cat > "${result_dir}/metadata.txt" <<EOF
experiment=smolvlm-cuda-packed-prefill
workload=two identical 88-token multimodal embedding requests
comparison=serial context prefill vs equal-length packed tensor prefill
iterations=${iterations}
order=serial-first within each process
model_asset=${model##*/}
mmproj_asset=${mmproj##*/}
image_asset=${image##*/}
note=packed linear and FFN rows with per-request multi-query attention; adaptive remains serial until paired latency improves
EOF

echo "smolvlm_cuda_prefill_result_dir=${result_dir}"
echo "smolvlm_cuda_prefill_ab=PASS"
