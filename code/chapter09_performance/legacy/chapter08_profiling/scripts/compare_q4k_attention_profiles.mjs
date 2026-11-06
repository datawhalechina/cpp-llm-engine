#!/usr/bin/env node

// Chapter 08：比较真实 Decoder 的 Q4_K Attention projection scalar/AVX2 profile。
// 契约：只允许 QKV/out backend 改变，FFN、LM Head、workload 和正确性保持一致。
// 验证：run_real_q4k_attention_ab.sh 生成三轮原始结果和最终判定。
import fs from "node:fs";

const [baselinePath, experimentPath] = process.argv.slice(2);
if (!baselinePath || !experimentPath) {
  console.error("usage: compare_q4k_attention_profiles.mjs BASELINE EXPERIMENT");
  process.exit(2);
}

function parse(path) {
  const values = new Map();
  for (const line of fs.readFileSync(path, "utf8").split(/\r?\n/)) {
    const separator = line.indexOf("=");
    if (separator > 0) values.set(line.slice(0, separator), line.slice(separator + 1));
  }
  return values;
}

function number(values, name) {
  const value = Number(values.get(name));
  if (!Number.isFinite(value) || value < 0) throw new Error(`invalid ${name}`);
  return value;
}

function requirePass(values, name) {
  if (values.get(name) !== "PASS") throw new Error(`${name} did not pass`);
}

const baseline = parse(baselinePath);
const experiment = parse(experimentPath);
for (const values of [baseline, experiment]) {
  requirePass(values, "correctness");
  requirePass(values, "profile_complete");
  requirePass(values, "ffn_profile_complete");
}
if (baseline.get("decoder_profile") !== experiment.get("decoder_profile") ||
    baseline.get("iterations") !== experiment.get("iterations") ||
    baseline.get("forward_tokens_per_iteration") !==
      experiment.get("forward_tokens_per_iteration") ||
    baseline.get("attention_q4_backend") !== "scalar" ||
    experiment.get("attention_q4_backend") !== "avx2" ||
    baseline.get("ffn_q4_backend") !== "scalar" ||
    experiment.get("ffn_q4_backend") !== "scalar" ||
    baseline.get("ffn_down_backend") !== "scalar" ||
    experiment.get("ffn_down_backend") !== "scalar" ||
    baseline.get("lm_head_backend") !== "scalar" ||
    experiment.get("lm_head_backend") !== "scalar") {
  throw new Error("baseline and experiment are not comparable");
}

const baselineQkv = number(baseline, "qkv_projection_ms");
const experimentQkv = number(experiment, "qkv_projection_ms");
const baselineOut = number(baseline, "attention_output_ms");
const experimentOut = number(experiment, "attention_output_ms");
const baselineProjection = baselineQkv + baselineOut;
const experimentProjection = experimentQkv + experimentOut;
const baselineTotal = number(baseline, "total_ms");
const experimentTotal = number(experiment, "total_ms");
const delta = (before, after) => (after - before) * 100 / before;
const qkvDelta = delta(baselineQkv, experimentQkv);
const outDelta = delta(baselineOut, experimentOut);
const projectionDelta = delta(baselineProjection, experimentProjection);
const totalDelta = delta(baselineTotal, experimentTotal);
const accepted = qkvDelta < 0 && outDelta < 0 && totalDelta < 0;

console.log([
  "experiment=tinyllama_q4k_attention_projection_single_variable",
  "correctness=PASS",
  `iterations=${baseline.get("iterations")}`,
  `forward_tokens_per_iteration=${baseline.get("forward_tokens_per_iteration")}`,
  "baseline_backend=scalar",
  "experiment_backend=avx2",
  `baseline_qkv_projection_ms=${baselineQkv.toFixed(4)}`,
  `experiment_qkv_projection_ms=${experimentQkv.toFixed(4)}`,
  `qkv_projection_delta_percent=${qkvDelta.toFixed(4)}`,
  `baseline_attention_output_ms=${baselineOut.toFixed(4)}`,
  `experiment_attention_output_ms=${experimentOut.toFixed(4)}`,
  `attention_output_delta_percent=${outDelta.toFixed(4)}`,
  `baseline_attention_projection_ms=${baselineProjection.toFixed(4)}`,
  `experiment_attention_projection_ms=${experimentProjection.toFixed(4)}`,
  `attention_projection_delta_percent=${projectionDelta.toFixed(4)}`,
  `baseline_total_ms=${baselineTotal.toFixed(4)}`,
  `experiment_total_ms=${experimentTotal.toFixed(4)}`,
  `total_delta_percent=${totalDelta.toFixed(4)}`,
  `decision=${accepted ? "KEEP" : "REJECT"}`,
  `reason=${accepted
    ? "QKV, attention output, and total latency improved"
    : "QKV, attention output, or total latency did not improve"}`,
].join("\n"));
