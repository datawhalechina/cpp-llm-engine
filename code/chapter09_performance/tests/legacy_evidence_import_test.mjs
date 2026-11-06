// Chapter 09：验证旧 benchmark 的无损迁移及聚合历史结果的拒绝规则。
import assert from "node:assert/strict";
import {
  LegacyMigrationError,
  migrateLegacyDocument,
} from "../scripts/import_legacy_evidence.mjs";

const environment = { compiler: "gcc 13", build_type: "Release" };
const teaching = {
  experiment_id: "EXP-TEACH",
  workload_id: "WL-TEACH",
  device: "cpu",
  environment,
  warmup: 1,
  scenarios: [{
    execution_mode: "tensor_batch",
    request_count: 2,
    prompt_tokens: 4,
    max_new_tokens: 3,
    correctness: true,
    samples: [{
      total_ms: 2,
      prefill_ms: 0.5,
      decode_ms: 1.5,
      ttft_ms: 0.7,
      throughput_tokens_per_second: 3000,
      correctness: true,
    }],
  }],
};
const teachingRecords = migrateLegacyDocument(teaching);
assert.equal(teachingRecords.length, 1);
assert.equal(teachingRecords[0].samples[0].total_ms, 2);
assert.equal(teachingRecords[0].workload.execution_mode, "multi_request_generation");

const pair = {
  experiment_id: "EXP-PAIR",
  workload_id: "WL-PAIR",
  model: "TinyLlama",
  device: "cpu",
  execution_policy: "scalar",
  environment,
  warmup_pairs: 1,
  workload: {
    request_count: 2,
    prompt_tokens: [6, 1],
    generated_tokens_per_request: 4,
  },
  samples: [{
    sequential_decode_ms: 8,
    batch_decode_ms: 4,
    correctness: true,
  }],
  correctness: { all_iterations: true },
};
const pairRecords = migrateLegacyDocument(pair);
assert.equal(pairRecords.length, 2);
assert.deepEqual(pairRecords[0].workload.prompt_tokens, [6, 1]);
assert.equal(pairRecords[1].samples[0].tokens_per_second, 2000);

const aggregateOnly = {
  benchmark: "tinyllama_service_core_cpu",
  model: "TinyLlama",
  device: "cpu",
  execution_policy: "scalar",
  dispatch_mode: "sequential",
  environment,
  iterations: 3,
  metrics_ms: { end_to_end: { mean: 10 } },
};
assert.throws(
  () => migrateLegacyDocument(aggregateOnly),
  (error) => error instanceof LegacyMigrationError && /no raw/.test(error.message),
);

const service = {
  ...aggregateOnly,
  request_count: 2,
  prompt_tokens: [6, 1],
  max_new_tokens_per_request: 4,
  warmup: 1,
  memory_bytes: { process_peak_rss: 1048576 },
  correctness: { all_iterations: true },
  samples: [{
    prefill_compute_ms: 4,
    decode_compute_ms: 6,
    end_to_end_ms: 10,
    ttft_ms: [3, 4],
    throughput_tokens_per_second: 800,
    correctness: true,
  }],
};
const serviceRecord = migrateLegacyDocument(service)[0];
assert.equal(serviceRecord.samples[0].peak_memory_mb, 1);
assert.equal(serviceRecord.samples[0].ttft_ms, 4);
assert.equal(serviceRecord.environment.backend, "cpp-llm-engine-cpu");
assert.equal(serviceRecord.workload.execution_mode, "service_generation");

process.stdout.write("legacy evidence import tests passed\n");
