#!/usr/bin/env node

// Chapter 08：比较组合 optimized policy 的 Decoder profile 与 ServiceCore benchmark。
// 契约：模型、workload、环境和 LM Head 不变，只组合三个已通过的 AVX2 作用域。
import fs from "node:fs";

const [scalarProfilePath, optimizedProfilePath, scalarServicePath,
  optimizedServicePath] = process.argv.slice(2);
if (!optimizedServicePath) {
  console.error(
    "usage: compare_optimized_policy.mjs SCALAR_PROFILE OPTIMIZED_PROFILE " +
    "SCALAR_SERVICE OPTIMIZED_SERVICE");
  process.exit(2);
}

function parseProfile(path) {
  const values = new Map();
  for (const line of fs.readFileSync(path, "utf8").split(/\r?\n/)) {
    const separator = line.indexOf("=");
    if (separator > 0) values.set(line.slice(0, separator), line.slice(separator + 1));
  }
  return values;
}

function requirePass(values, name) {
  if (values.get(name) !== "PASS") throw new Error(`${name} did not pass`);
}

function number(values, name) {
  const value = Number(values.get(name));
  if (!Number.isFinite(value) || value < 0) throw new Error(`invalid ${name}`);
  return value;
}

function serviceMetric(document, name) {
  const value = document.metrics_ms?.[name]?.mean;
  if (!Number.isFinite(value) || value < 0) throw new Error(`invalid service ${name}`);
  return value;
}

function same(left, right, name) {
  if (JSON.stringify(left) !== JSON.stringify(right)) {
    throw new Error(`service ${name} does not match`);
  }
}

const scalarProfile = parseProfile(scalarProfilePath);
const optimizedProfile = parseProfile(optimizedProfilePath);
for (const profile of [scalarProfile, optimizedProfile]) {
  requirePass(profile, "correctness");
  requirePass(profile, "profile_complete");
  requirePass(profile, "ffn_profile_complete");
}
if (scalarProfile.get("iterations") !== optimizedProfile.get("iterations") ||
    scalarProfile.get("forward_tokens_per_iteration") !==
      optimizedProfile.get("forward_tokens_per_iteration")) {
  throw new Error("Decoder profiles are not comparable");
}
for (const field of ["attention_q4_backend", "ffn_q4_backend", "ffn_down_backend"]) {
  if (scalarProfile.get(field) !== "scalar" || optimizedProfile.get(field) !== "avx2") {
    throw new Error(`unexpected ${field} transition`);
  }
}
if (scalarProfile.get("lm_head_backend") !== "scalar" ||
    optimizedProfile.get("lm_head_backend") !== "scalar") {
  throw new Error("LM Head must remain scalar");
}

const scalarService = JSON.parse(fs.readFileSync(scalarServicePath, "utf8"));
const optimizedService = JSON.parse(fs.readFileSync(optimizedServicePath, "utf8"));
if (scalarService.execution_policy !== "scalar" ||
    optimizedService.execution_policy !== "optimized-avx2") {
  throw new Error("unexpected service execution policy");
}
for (const document of [scalarService, optimizedService]) {
  if (!document.correctness?.all_iterations ||
      !document.correctness?.context_and_cache_isolation ||
      !document.correctness?.terminal_resources_released) {
    throw new Error("service correctness did not pass");
  }
}
for (const field of ["benchmark", "model", "device", "request_count",
  "max_new_tokens_per_request", "prompt_tokens", "iterations", "warmup"]) {
  same(scalarService[field], optimizedService[field], field);
}
for (const field of ["hardware_concurrency", "cpu_model", "compiler", "build_type"]) {
  same(scalarService.environment?.[field], optimizedService.environment?.[field], field);
}

const delta = (before, after) => (after - before) * 100 / before;
const scalarDecoder = number(scalarProfile, "total_ms");
const optimizedDecoder = number(optimizedProfile, "total_ms");
const scalarDispatch = serviceMetric(scalarService, "dispatch");
const optimizedDispatch = serviceMetric(optimizedService, "dispatch");
const scalarPrefill = serviceMetric(scalarService, "prefill_compute");
const optimizedPrefill = serviceMetric(optimizedService, "prefill_compute");
const scalarDecode = serviceMetric(scalarService, "decode_compute");
const optimizedDecode = serviceMetric(optimizedService, "decode_compute");
const scalarThroughput = scalarService.throughput_tokens_per_second?.mean;
const optimizedThroughput = optimizedService.throughput_tokens_per_second?.mean;
if (!Number.isFinite(scalarThroughput) || !Number.isFinite(optimizedThroughput)) {
  throw new Error("invalid service throughput");
}

const decoderDelta = delta(scalarDecoder, optimizedDecoder);
const dispatchDelta = delta(scalarDispatch, optimizedDispatch);
const prefillDelta = delta(scalarPrefill, optimizedPrefill);
const decodeDelta = delta(scalarDecode, optimizedDecode);
const throughputDelta = delta(scalarThroughput, optimizedThroughput);
const accepted = decoderDelta < 0 && dispatchDelta < 0 &&
  prefillDelta < 0 && decodeDelta < 0 && throughputDelta > 0;

console.log([
  "experiment=tinyllama_combined_optimized_policy",
  "correctness=PASS",
  `iterations=${scalarProfile.get("iterations")}`,
  "scalar_policy=attention:q4-scalar,ffn:q4-scalar+q6-scalar,lm_head:scalar",
  "optimized_policy=attention:q4-avx2,ffn:q4-avx2+q6-avx2,lm_head:scalar",
  `scalar_decoder_ms=${scalarDecoder.toFixed(4)}`,
  `optimized_decoder_ms=${optimizedDecoder.toFixed(4)}`,
  `decoder_delta_percent=${decoderDelta.toFixed(4)}`,
  `scalar_service_dispatch_ms=${scalarDispatch.toFixed(4)}`,
  `optimized_service_dispatch_ms=${optimizedDispatch.toFixed(4)}`,
  `service_dispatch_delta_percent=${dispatchDelta.toFixed(4)}`,
  `service_prefill_delta_percent=${prefillDelta.toFixed(4)}`,
  `service_decode_delta_percent=${decodeDelta.toFixed(4)}`,
  `scalar_tokens_per_second=${scalarThroughput.toFixed(4)}`,
  `optimized_tokens_per_second=${optimizedThroughput.toFixed(4)}`,
  `throughput_delta_percent=${throughputDelta.toFixed(4)}`,
  `decision=${accepted ? "KEEP" : "REJECT"}`,
  `reason=${accepted
    ? "Decoder and service latency improved with correctness preserved"
    : "Decoder or service improvement criteria were not met"}`,
].join("\n"));
