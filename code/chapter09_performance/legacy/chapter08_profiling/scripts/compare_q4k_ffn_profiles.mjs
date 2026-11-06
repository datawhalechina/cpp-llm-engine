#!/usr/bin/env node

// Chapter 08：比较真实 Decoder 的 Q4_K gate/up scalar 与 AVX2 profile。
// 契约：正确性、计时闭合和 workload 一致，gate、up、总耗时都改善才 KEEP。
// 验证：run_real_q4k_ffn_ab.sh 生成三轮原始结果和最终判定。
import fs from "node:fs";

const [baselinePath, experimentPath] = process.argv.slice(2);
if (!baselinePath || !experimentPath) {
  console.error("usage: compare_q4k_ffn_profiles.mjs BASELINE EXPERIMENT");
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
    baseline.get("ffn_q4_backend") !== "scalar" ||
    experiment.get("ffn_q4_backend") !== "avx2" ||
    baseline.get("ffn_down_backend") !== "scalar" ||
    experiment.get("ffn_down_backend") !== "scalar") {
  throw new Error("baseline and experiment are not comparable");
}

const baselineGate = number(baseline, "ffn_gate_ms");
const experimentGate = number(experiment, "ffn_gate_ms");
const baselineUp = number(baseline, "ffn_up_ms");
const experimentUp = number(experiment, "ffn_up_ms");
const baselineFfn = number(baseline, "ffn_ms");
const experimentFfn = number(experiment, "ffn_ms");
const baselineTotal = number(baseline, "total_ms");
const experimentTotal = number(experiment, "total_ms");
const delta = (before, after) => (after - before) * 100 / before;
const gateDelta = delta(baselineGate, experimentGate);
const upDelta = delta(baselineUp, experimentUp);
const ffnDelta = delta(baselineFfn, experimentFfn);
const totalDelta = delta(baselineTotal, experimentTotal);
const accepted = gateDelta < 0 && upDelta < 0 && totalDelta < 0;

console.log([
  "experiment=tinyllama_q4k_ffn_single_variable",
  "correctness=PASS",
  `iterations=${baseline.get("iterations")}`,
  `forward_tokens_per_iteration=${baseline.get("forward_tokens_per_iteration")}`,
  "baseline_backend=scalar",
  "experiment_backend=avx2",
  `baseline_ffn_gate_ms=${baselineGate.toFixed(4)}`,
  `experiment_ffn_gate_ms=${experimentGate.toFixed(4)}`,
  `ffn_gate_delta_percent=${gateDelta.toFixed(4)}`,
  `baseline_ffn_up_ms=${baselineUp.toFixed(4)}`,
  `experiment_ffn_up_ms=${experimentUp.toFixed(4)}`,
  `ffn_up_delta_percent=${upDelta.toFixed(4)}`,
  `baseline_ffn_ms=${baselineFfn.toFixed(4)}`,
  `experiment_ffn_ms=${experimentFfn.toFixed(4)}`,
  `ffn_delta_percent=${ffnDelta.toFixed(4)}`,
  `baseline_total_ms=${baselineTotal.toFixed(4)}`,
  `experiment_total_ms=${experimentTotal.toFixed(4)}`,
  `total_delta_percent=${totalDelta.toFixed(4)}`,
  `decision=${accepted ? "KEEP" : "REJECT"}`,
  `reason=${accepted
    ? "gate, up, and total latency improved"
    : "gate, up, or total latency did not improve"}`,
].join("\n"));
