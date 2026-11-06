// Chapter 09：把保留原始样本的旧 benchmark JSON 转成统一性能证据。
// 聚合结果不能反推逐轮样本；遇到这类历史文件时明确拒绝，而不是伪造数据。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export class LegacyMigrationError extends Error {}

function requirePositive(value, field) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new LegacyMigrationError(`invalid ${field}`);
  }
  return value;
}

function environmentOf(document, options) {
  const environment = document.environment ?? {};
  return {
    model: document.model ?? options.model ?? "teaching-transformer",
    backend: options.backend ?? (document.model ? "cpp-llm-engine-cpu" : "teaching-cpu"),
    device: document.device ?? "cpu",
    build_type: environment.build_type,
    compiler: environment.compiler,
  };
}

function commonRecord(document, options, overrides) {
  return {
    schema_version: 1,
    experiment_id: overrides.experiment_id,
    workload_id: overrides.workload_id,
    environment: environmentOf(document, options),
    workload: overrides.workload,
    correctness_pass: overrides.correctness_pass,
    correctness_checks: overrides.correctness_checks,
    samples: overrides.samples,
  };
}

function migrateTeachingScenarios(document, options) {
  return document.scenarios.map((scenario) => {
    if (!Array.isArray(scenario.samples) || scenario.samples.length === 0) {
      throw new LegacyMigrationError("teaching benchmark misses raw samples");
    }
    const suffix = `${scenario.execution_mode}-requests-${scenario.request_count}`;
    return commonRecord(document, options, {
      experiment_id: `${document.experiment_id}:${suffix}`,
      workload_id: `${document.workload_id}:requests-${scenario.request_count}`,
      workload: {
        request_count: scenario.request_count,
        prompt_tokens_per_request: scenario.prompt_tokens,
        prompt_tokens: [],
        output_tokens_per_request: scenario.max_new_tokens,
        warmup_iterations: document.warmup ?? 0,
        measurement_iterations: scenario.samples.length,
        execution_mode: "multi_request_generation",
      },
      correctness_pass: scenario.correctness === true &&
        scenario.samples.every((sample) => sample.correctness === true),
      correctness_checks: [
        "legacy scenario correctness",
        `execution policy: ${scenario.execution_mode}`,
      ],
      samples: scenario.samples.map((sample) => ({
        total_ms: requirePositive(sample.total_ms, "sample.total_ms"),
        prefill_ms: sample.prefill_ms ?? 0,
        decode_ms: sample.decode_ms ?? 0,
        ttft_ms: sample.ttft_ms ?? 0,
        itl_ms: 0,
        tokens_per_second: requirePositive(
          sample.throughput_tokens_per_second,
          "sample.throughput_tokens_per_second",
        ),
        peak_memory_mb: 0,
      })),
    });
  });
}

function migrateRealBatchPair(document, options) {
  const workload = document.workload;
  const generatedTokens = workload.request_count * workload.generated_tokens_per_request;
  const makeRecord = (mode, field) => commonRecord(document, options, {
    experiment_id: `${document.experiment_id}:${mode}`,
    workload_id: document.workload_id,
    workload: {
      request_count: workload.request_count,
      prompt_tokens_per_request: 0,
      prompt_tokens: workload.prompt_tokens,
      output_tokens_per_request: workload.generated_tokens_per_request,
      warmup_iterations: document.warmup_pairs ?? 0,
      measurement_iterations: document.samples.length,
      execution_mode: "real_decode_pair",
    },
    correctness_pass: document.correctness?.all_iterations === true &&
      document.samples.every((sample) => sample.correctness === true),
    correctness_checks: [
      "sequential and batch tokens match",
      "golden tokens match",
      `execution policy: ${mode}`,
    ],
    samples: document.samples.map((sample) => {
      const totalMs = requirePositive(sample[field], `sample.${field}`);
      return {
        total_ms: totalMs,
        prefill_ms: 0,
        decode_ms: totalMs,
        ttft_ms: 0,
        itl_ms: 0,
        tokens_per_second: generatedTokens * 1000 / totalMs,
        peak_memory_mb: 0,
      };
    }),
  });
  return [makeRecord("sequential", "sequential_decode_ms"),
    makeRecord("tensor-batch", "batch_decode_ms")];
}

function migrateServiceSamples(document, options) {
  if (!Array.isArray(document.samples) || document.samples.length === 0) {
    throw new LegacyMigrationError(
      "service result has aggregate statistics but no raw per-iteration samples",
    );
  }
  const correctness = document.correctness ?? {};
  const peakMemoryMb = (document.memory_bytes?.process_peak_rss ?? 0) / (1024 * 1024);
  return [commonRecord(document, options, {
    experiment_id: `${document.benchmark}:${document.execution_policy}:${document.dispatch_mode}`,
    workload_id: `${document.benchmark}:requests-${document.request_count}`,
    workload: {
      request_count: document.request_count,
      prompt_tokens_per_request: 0,
      prompt_tokens: document.prompt_tokens,
      output_tokens_per_request: document.max_new_tokens_per_request,
      warmup_iterations: document.warmup ?? 0,
      measurement_iterations: document.samples.length,
      execution_mode: "service_generation",
    },
    correctness_pass: correctness.all_iterations === true &&
      document.samples.every((sample) => sample.correctness === true),
    correctness_checks: [
      "golden output tokens",
      "context and cache isolation",
      "terminal resources released",
      `execution policy: ${document.execution_policy}`,
      `dispatch policy: ${document.dispatch_mode}`,
    ],
    samples: document.samples.map((sample) => ({
      total_ms: requirePositive(sample.end_to_end_ms, "sample.end_to_end_ms"),
      prefill_ms: sample.prefill_compute_ms ?? 0,
      decode_ms: sample.decode_compute_ms ?? 0,
      ttft_ms: Array.isArray(sample.ttft_ms)
        ? Math.max(...sample.ttft_ms)
        : (sample.ttft_ms ?? 0),
      itl_ms: 0,
      tokens_per_second: requirePositive(
        sample.throughput_tokens_per_second,
        "sample.throughput_tokens_per_second",
      ),
      peak_memory_mb: peakMemoryMb,
    })),
  })];
}

export function migrateLegacyDocument(document, options = {}) {
  if (Array.isArray(document.scenarios)) return migrateTeachingScenarios(document, options);
  if (Array.isArray(document.samples) && document.workload?.generated_tokens_per_request) {
    return migrateRealBatchPair(document, options);
  }
  if (document.benchmark === "tinyllama_service_core_cpu") {
    return migrateServiceSamples(document, options);
  }
  throw new LegacyMigrationError("unsupported legacy evidence format");
}

function parseCli(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!name?.startsWith("--") || value === undefined) {
      throw new LegacyMigrationError("usage: --input FILE --output-dir DIR [--model NAME] [--backend NAME]");
    }
    options[name.slice(2).replaceAll("-", "_")] = value;
  }
  if (!options.input || !options.output_dir) {
    throw new LegacyMigrationError("--input and --output-dir are required");
  }
  return options;
}

function safeName(value) {
  return value.replace(/[^A-Za-z0-9._-]+/g, "-");
}

function main() {
  const options = parseCli(process.argv.slice(2));
  const document = JSON.parse(fs.readFileSync(options.input, "utf8"));
  const records = migrateLegacyDocument(document, options);
  fs.mkdirSync(options.output_dir, { recursive: true });
  for (const record of records) {
    const outputPath = path.join(options.output_dir, `${safeName(record.experiment_id)}.json`);
    fs.writeFileSync(outputPath, `${JSON.stringify(record, null, 2)}\n`);
    process.stdout.write(`${outputPath}\n`);
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`legacy evidence migration failed: ${error.message}\n`);
    process.exitCode = 2;
  }
}
