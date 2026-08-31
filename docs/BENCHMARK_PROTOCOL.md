# Benchmark protocol

The executable suite contract lives in `benchmarks/manifest.json`; individual result rows follow `benchmarks/result.schema.json`.

Visible CLI, diagnostic-panel, screenshot, redaction, and evidence-manifest requirements are defined in [BENCHMARK_EVIDENCE_CAPTURE.md](BENCHMARK_EVIDENCE_CAPTURE.md). Screenshots corroborate the raw and structured evidence; they never replace it.

## Run identity

Every meaningful run records:

- UTC timestamp, run ID, suite version, and evidence class;
- model repositories, revisions, artifact bytes, SHA-256 manifests, and quant recipe;
- runtime repository, commit, build flags, executable hash, driver, CUDA, and backend;
- exact command, environment overrides, server mode, and working directory;
- machine and topology snapshot;
- fixture, prompt hash, prompt token target, and actual prompt tokens;
- context, KV types, batch, ubatch, threads, GPU layers, MoE placement, mmap/load mode, and PLE placement;
- raw response, stdout, stderr, server log, and telemetry paths.

## Metrics

Performance:

- startup and model-load time;
- prompt tokens, prompt milliseconds, and prefill tokens/s;
- time to first streamed token;
- completion tokens, decode milliseconds, and single-stream decode tokens/s;
- end-to-end wall time;
- cold versus warm state and repeat number.

Speculative decoding:

- target-only result first;
- nmax, p_min, draft device, draft KV, and draft placement;
- drafted and accepted tokens, acceptance rate, accepted tokens by position, and verification steps;
- exact output text hash and retokenized-token hash against the matched target result.

Resources:

- available RAM minimum, owned RSS peak, free VRAM minimum, used VRAM peak;
- pagefile growth, page faults, disk bytes, network bytes, and GPU utilization;
- load/warm state and unrelated-process snapshot.

Quality and capability:

- task and dataset revision;
- prompt/chat template and scorer revision;
- few-shot count, seed, temperature, batch, concurrency, exclusions, and sample count;
- metric, standard error or confidence interval where supported;
- raw per-sample output or an immutable pointer to it.

## Test order

1. Artifact, runtime, process, port, and resource preflight.
2. Deterministic target-only compatibility, code, and prose fixtures.
3. Matched MTP fixtures only when target hashes are valid.
4. Occupied-context target-only ladder at 8K, 16K, 32K, and 64K.
5. MTP occupied-context rows only after target safety and exact-parity prerequisites pass.
6. KV and resource matrix, separate from speed promotion.
7. Quant-regression perplexity and next-token probes.
8. Development smoke suite.
9. Pinned public capability pilot.
10. Full public capability run only after pilot stability and runtime-budget review.

## Validity and contamination

A run is `VOID` if any of these affect the claim:

- wrong runtime, model, sidecar, port, or command identity;
- another GPU workload, model download, unrelated disk load, or network transfer;
- paging storm or at least 1 GiB pagefile growth;
- undeclared cold/warm mismatch;
- missing telemetry required for the claim;
- output truncation where a natural stop is required;
- prompt occupancy below 99% of a stated occupied-context target;
- missing or failed retrieval needle;
- server or watchdog cleanup failure.

A run is `FAILED`, not `VOID`, when the controlled system itself fails under an otherwise valid setup. Both dispositions remain in the evidence tree.

## Safety floors

Preflight:

- at least 40 GiB available RAM;
- at least 8 GiB free VRAM;
- at most 15% GPU utilization;
- no exact project server conflict.

Hard stops:

- available RAM below 6 GiB;
- free VRAM below 768 MiB;
- owned RSS above 50 GiB;
- pagefile growth at least 1 GiB;
- CUDA, driver, server, or output corruption.

Promotion floor:

- at least 1,024 MiB minimum free VRAM for a daily profile;
- three valid repeats for release performance;
- dispersion reported, never a lone best run.

## Reporting rules

- Separate `MEASURED`, `PROXY`, `ANALYTICAL`, and `EXTERNAL` values.
- Report target-only and MTP separately.
- Report single-stream user-visible decode, never aggregate concurrent throughput as t/s.
- Label allocated context and occupied prompt length in the same row.
- Do not headline a long-context speed from a short prompt.
- Do not promote MTP speed when exact text or token identity fails.
- Do not collapse the public capability suite into an unqualified single intelligence score.
- Preserve negative and failed results.

## Current project classification

- 16K warm short-prompt speed: `MEASURED`, repeated.
- 64K through 224K short-prompt allocation profiles: `MEASURED CAPACITY`, not occupied-context speed.
- 60K occupied-context attempt: `FAILED`.
- Promoted-artifact MTP all-accepted fixtures: `MEASURED EXPECTED-HASH PASS`, useful for mechanics but not rollback correctness.
- Promoted-artifact rejection-bearing code: `NOT YET MEASURED` as a matched target/MTP A/B.
- Promoted-artifact prose/rejection fixture: `MEASURED BLOCKED_EXACT_PARITY`; output and token hashes differ.
- Earlier Q4_K_M target/sidecar state-fix rows: `MEASURED HISTORICAL`, excluded from promoted-artifact performance claims.
- broad intelligence and capability: `NOT YET MEASURED` on the exact retained release candidate.
