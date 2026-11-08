// Chapter 09：把 JSON 证据无损规范化，完整校验与决策仍由 C++ 完成。
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

function number(value) {
  return Number.isFinite(value) ? String(value) : "nan";
}

function size(value) {
  return Number.isSafeInteger(value) && value >= 0 ? String(value) : "0";
}

function text(value) {
  return encodeURIComponent(typeof value === "string" ? value : "");
}

function writeWire(record, file) {
  const environment = record?.environment ?? {};
  const workload = record?.workload ?? {};
  const samples = Array.isArray(record?.samples) ? record.samples : [];
  const lines = [
    ["schema_version", size(record?.schema_version)],
    ["experiment_id", text(record?.experiment_id)],
    ["workload_id", text(record?.workload_id)],
    ["model", text(environment.model)],
    ["backend", text(environment.backend)],
    ["device", text(environment.device)],
    ["hardware", text(environment.hardware)],
    ["hardware_concurrency", size(environment.hardware_concurrency)],
    ["build_type", text(environment.build_type)],
    ["compiler", text(environment.compiler)],
    ["request_count", size(workload.request_count)],
    ["prompt_tokens_per_request", size(workload.prompt_tokens_per_request)],
    ["prompt_tokens", Array.isArray(workload.prompt_tokens) ? workload.prompt_tokens.map(size).join(",") : ""],
    ["output_tokens_per_request", size(workload.output_tokens_per_request)],
    ["warmup_iterations", size(workload.warmup_iterations)],
    ["measurement_iterations", size(workload.measurement_iterations)],
    ["execution_mode", text(workload.execution_mode)],
    ["correctness_pass", record?.correctness_pass === true ? "1" : "0"],
    ["correctness_check_count", size(Array.isArray(record?.correctness_checks) ? record.correctness_checks.length : 0)],
  ];
  for (const sample of samples) {
    const fields = ["total_ms", "prefill_ms", "decode_ms", "ttft_ms", "itl_ms",
      "tokens_per_second", "peak_memory_mb"];
    lines.push(["sample", fields.map((field) => number(sample?.[field])).join(",")]);
  }
  fs.writeFileSync(file, `${lines.map(([key, value]) => `${key}\t${value}`).join("\n")}\n`);
}

const [comparator, baselineFile, experimentFile, outputFile] = process.argv.slice(2);
if (!comparator || !baselineFile || !experimentFile || !outputFile) {
  throw new Error("usage: compare_evidence_pair.mjs COMPARATOR BASELINE EXPERIMENT OUTPUT");
}

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "chapter09-evidence-"));
try {
  const baselineWire = path.join(temporary, "baseline.wire");
  const experimentWire = path.join(temporary, "experiment.wire");
  writeWire(JSON.parse(fs.readFileSync(baselineFile, "utf8")), baselineWire);
  writeWire(JSON.parse(fs.readFileSync(experimentFile, "utf8")), experimentWire);
  const result = spawnSync(comparator, [baselineWire, experimentWire], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || "C++ comparator failed");
  fs.writeFileSync(outputFile, result.stdout);
  process.stdout.write(result.stdout);
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}
