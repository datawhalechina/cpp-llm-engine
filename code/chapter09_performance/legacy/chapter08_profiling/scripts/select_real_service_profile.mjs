#!/usr/bin/env node

// Chapter 08：把真实 TinyLlama ServiceCore benchmark JSON 适配为 ProfileRecord。
// 职责：校验环境、正确性和阶段字段，再输出统一 key=value profiling 输入。
// 验证：Chapter08 run.sh 消费归档基线，真实 A/B 脚本可再传入 experiment JSON。

import fs from "node:fs";

const [inputPath, outputPath, experimentId = "EXP-CPU-02"] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error(
    "usage: select_real_service_profile.mjs INPUT_JSON OUTPUT_TXT [EXPERIMENT_ID]",
  );
  process.exit(2);
}

const document = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const mean = (name) => {
  const value = document.metrics_ms?.[name]?.mean;
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`missing metrics_ms.${name}.mean`);
  }
  return value;
};

if (document.benchmark !== "tinyllama_service_core_cpu") {
  throw new Error(`unsupported benchmark: ${document.benchmark}`);
}
if (!document.correctness?.all_iterations ||
    !document.correctness?.context_and_cache_isolation ||
    !document.correctness?.terminal_resources_released) {
  throw new Error("real service benchmark failed correctness gate");
}
if (!Number.isInteger(document.request_count) || document.request_count <= 0 ||
    !Number.isInteger(document.max_new_tokens_per_request) ||
    document.max_new_tokens_per_request <= 0) {
  throw new Error("invalid real service request/token configuration");
}

const ttftA = mean("request_a_ttft");
const ttftB = mean("request_b_ttft");
const completionA = mean("request_a_completion");
const completionB = mean("request_b_completion");
const throughput = document.throughput_tokens_per_second?.mean;
if (!Number.isFinite(throughput) || throughput <= 0) {
  throw new Error("invalid throughput mean");
}
const memory = document.memory_bytes ?? {};
for (const field of ["model_weight_payload", "kv_committed_total", "process_peak_rss"]) {
  if (!Number.isInteger(memory[field]) || memory[field] <= 0) {
    throw new Error(`missing memory_bytes.${field}`);
  }
}

const lines = [
  `experiment_id=${experimentId}`,
  "workload_id=WL-CPU-REAL-SERVICE-V1",
  `backend=${document.device}`,
  `model=${document.model}`,
  `cpu_model=${document.environment?.cpu_model ?? "unknown"}`,
  `build_type=${document.environment?.build_type ?? "unknown"}`,
  "execution_pattern=single_thread_round_robin",
  "correctness=PASS",
  `iterations=${document.iterations}`,
  `request_count=${document.request_count}`,
  `generated_tokens_per_iteration=${document.request_count * document.max_new_tokens_per_request}`,
  `prefill_ms=${mean("prefill_compute")}`,
  `decode_ms=${mean("decode_compute")}`,
  `runtime_overhead_ms=${mean("runtime_overhead")}`,
  `total_ms=${mean("dispatch")}`,
  `ttft_ms=${Math.max(ttftA, ttftB)}`,
  `completion_ms=${Math.max(completionA, completionB)}`,
  `ttft_spread_ms=${Math.abs(ttftB - ttftA)}`,
  `completion_spread_ms=${Math.abs(completionB - completionA)}`,
  `reported_tokens_per_second=${throughput}`,
  `model_weight_payload_bytes=${memory.model_weight_payload}`,
  `kv_cache_bytes=${memory.kv_committed_total}`,
  `peak_memory_mb=${memory.process_peak_rss / (1024 * 1024)}`,
];
fs.writeFileSync(outputPath, `${lines.join("\n")}\n`);
