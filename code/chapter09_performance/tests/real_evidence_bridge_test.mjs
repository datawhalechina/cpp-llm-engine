// Chapter 09：验证 JSON bridge 不绕过 C++ 证据契约和可比性检查。
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const [bridge, comparator] = process.argv.slice(2);
assert.ok(bridge && comparator, "bridge and comparator paths are required");
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "chapter09-bridge-test-"));

function record(id) {
  return {
    schema_version: 1,
    experiment_id: id,
    workload_id: "WL-REAL-SERVICE",
    environment: { model: "TinyLlama", backend: "cpu", device: "x86_64", build_type: "Release", compiler: "gcc-13" },
    workload: { request_count: 2, prompt_tokens_per_request: 0, prompt_tokens: [6, 1], output_tokens_per_request: 4, warmup_iterations: 1, measurement_iterations: 2, execution_mode: "service_generation" },
    correctness_pass: true,
    correctness_checks: ["golden tokens"],
    samples: [
      { total_ms: 10, prefill_ms: 2, decode_ms: 8, ttft_ms: 3, itl_ms: 1, tokens_per_second: 800, peak_memory_mb: 500 },
      { total_ms: 12, prefill_ms: 2, decode_ms: 10, ttft_ms: 3, itl_ms: 1, tokens_per_second: 700, peak_memory_mb: 510 },
    ],
  };
}

function compare(baseline, experiment, name) {
  const baselineFile = path.join(temporary, `${name}-baseline.json`);
  const experimentFile = path.join(temporary, `${name}-experiment.json`);
  const outputFile = path.join(temporary, `${name}-decision.json`);
  fs.writeFileSync(baselineFile, JSON.stringify(baseline));
  fs.writeFileSync(experimentFile, JSON.stringify(experiment));
  const result = spawnSync(process.execPath,
    [bridge, comparator, baselineFile, experimentFile, outputFile], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(fs.readFileSync(outputFile, "utf8"));
}

try {
  const baseline = record("baseline");
  const faster = record("faster");
  faster.samples.forEach((sample) => {
    sample.total_ms *= 0.9;
    sample.tokens_per_second *= 1.1;
  });
  assert.equal(compare(baseline, faster, "keep").decision, "KEEP");

  const otherWorkload = record("other-workload");
  otherWorkload.workload.prompt_tokens = [6, 2];
  assert.equal(compare(baseline, otherWorkload, "not-comparable").decision, "NOT_COMPARABLE");

  const otherProtocol = record("other-protocol");
  otherProtocol.workload.warmup_iterations = 0;
  assert.equal(compare(baseline, otherProtocol, "protocol-not-comparable").decision, "NOT_COMPARABLE");

  const invalid = record("invalid");
  delete invalid.environment.compiler;
  assert.equal(compare(baseline, invalid, "invalid").decision, "INVALID");
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}

process.stdout.write("real evidence bridge tests passed\n");
