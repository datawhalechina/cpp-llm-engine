#!/usr/bin/env node
// Chapter 08：校验真实 scalar batch profile，并输出可复查的瓶颈摘要。
import fs from 'node:fs';

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath) {
  console.error('usage: analyze_real_batch_profile.mjs <result.json> [report.txt]');
  process.exit(2);
}

const result = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
if (result.schema_version < 2 || !Array.isArray(result.samples) || result.samples.length === 0) {
  throw new Error('real batch profile requires schema v2 samples');
}
if (result.experiment_id !== 'EXP-CPU-10' ||
    result.workload_id !== 'WL-CPU-REAL-BATCH-V1') {
  throw new Error('real batch result identity does not match EXP-CPU-10');
}
if (!result.correctness?.all_iterations ||
    !result.correctness?.sequential_batch_tokens_match ||
    !result.correctness?.golden_tokens_match) {
  throw new Error('real batch profile correctness gate failed');
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const fieldMean = (mode, field) => mean(result.samples.map((sample) => {
  if (!sample.correctness) throw new Error('real batch sample correctness failed');
  const value = sample[`${mode}_profile`]?.[field];
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`invalid ${mode} profile field: ${field}`);
  }
  return value;
}));

const sequentialWallMs = mean(result.samples.map((sample) => sample.sequential_decode_ms));
const batchWallMs = mean(result.samples.map((sample) => sample.batch_decode_ms));
const batchTotalMs = fieldMean('batch', 'total_ms');
const packMs = fieldMean('batch', 'batch_input_pack_ms');
const kernelMs = fieldMean('batch', 'batch_kernel_ms');
const unpackMs = fieldMean('batch', 'batch_output_unpack_ms');
const batchCalls = fieldMean('batch', 'batch_linear_calls');
const speedup = sequentialWallMs / batchWallMs;
const measuredLinearMs = packMs + kernelMs + unpackMs;

if (batchTotalMs <= 0 || kernelMs <= 0 || batchCalls <= 0 ||
    batchTotalMs > batchWallMs * 1.05 || batchTotalMs < batchWallMs * 0.90 ||
    measuredLinearMs > batchTotalMs * 1.01) {
  throw new Error('real batch profile timing boundary is inconsistent');
}

const percent = (part, total) => total > 0 ? part * 100 / total : 0;
const fixed = (value) => value.toFixed(4);
const decision = speedup > 1.0 ? 'KEEP_FOR_PERFORMANCE' : 'CORRECTNESS_ONLY';
const bottleneck = percent(kernelMs, batchTotalMs) >= 80.0
  ? 'scalar_batch_kernel'
  : 'outside_scalar_batch_kernel';

const report = [
  `experiment_id=${result.experiment_id}`,
  `workload_id=${result.workload_id}`,
  'correctness=PASS',
  `samples=${result.samples.length}`,
  `sequential_decode_mean_ms=${fixed(sequentialWallMs)}`,
  `batch_decode_mean_ms=${fixed(batchWallMs)}`,
  `paired_speedup=${fixed(speedup)}`,
  `batch_profile_total_mean_ms=${fixed(batchTotalMs)}`,
  `batch_linear_calls_mean=${fixed(batchCalls)}`,
  `batch_input_pack_mean_ms=${fixed(packMs)}`,
  `batch_kernel_mean_ms=${fixed(kernelMs)}`,
  `batch_output_unpack_mean_ms=${fixed(unpackMs)}`,
  `batch_pack_unpack_share_percent=${fixed(percent(packMs + unpackMs, batchTotalMs))}`,
  `batch_kernel_share_percent=${fixed(percent(kernelMs, batchTotalMs))}`,
  `bottleneck_scope=${bottleneck}`,
  `decision=${decision}`,
].join('\n') + '\n';

if (outputPath) fs.writeFileSync(outputPath, report);
process.stdout.write(report);
