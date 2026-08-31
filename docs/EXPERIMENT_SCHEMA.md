# Experiment Record Schema

Each experiment lives under `results/<experiment-id>/` with immutable raw evidence in `logs/<experiment-id>/` or content-addressed `artifacts/`.

Required `experiment.json` fields:

- `schema_version`, `experiment_id`, `started_utc`, `ended_utc`, `operator`;
- `status`: `PLANNED`, `RUNNING`, `VALID`, `VOID`, `FAILED`, or `BLOCKED`;
- `evidence_class`: `MEASURED`, `PROXY`, `ANALYTICAL`, or `EXTERNAL`;
- `model.repo`, `model.revision`, `model.artifact_sha256`, `model.source_hash`;
- `runtime.repo`, `runtime.commit`, `runtime.executable_sha256`, `runtime.build`;
- `quant.recipe_id` and tensor-map reference;
- exact `command`, working directory, and relevant environment;
- machine/topology snapshot reference;
- raw stdout, stderr, telemetry, and output paths;
- machine-readable metrics;
- `quality_verdict`, `memory_verdict`, and `speed_verdict`;
- contamination checks and `void_reasons`;
- limitations and analyst notes.

An experiment can be `FAILED` and still produce valid evidence. `VOID` means contamination prevents using its metrics for comparative conclusions.

## Public benchmark rows

The public benchmark package uses the stricter append-only row contract in [`benchmarks/result.schema.json`](../benchmarks/result.schema.json). Its manifest, release-stage dependencies, fixtures, validity rules, and current execution gates live in [`benchmarks/manifest.json`](../benchmarks/manifest.json). These rows do not replace an experiment record; they provide a consistent per-run shape for performance, occupied-context, resource, correctness, perplexity, and capability results.
