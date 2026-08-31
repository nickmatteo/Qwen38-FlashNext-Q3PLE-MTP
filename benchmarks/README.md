# Public benchmark suite

Suite ID: `PUBLIC-BENCH-001`

This package defines what must be measured before the Q3_PLE target or learned-MTP sidecar is presented as a release candidate.

It does not turn one benchmark into an intelligence score. The release report is a vector of capability, correctness, speed, context, and resource results.

## What is implemented

- `manifest.json`: artifacts, runtime identity, safety floors, fixtures, capability tasks, dependency graph, and gate status.
- `result.schema.json`: JSON Schema for append-only result rows.
- `fixtures/performance.jsonl`: compatibility, code, and prose fixtures.
- `fixtures/context_needles.jsonl`: actual 8K, 16K, 32K, and 64K occupied-context targets.
- `fixtures/tool_use.jsonl`: deterministic tool-call parsing cases.
- `scripts/benchmark.py`: standard-library contract validator, dependency planner, result policy checker, and summary tool.
- `tests/test_benchmark_contract.py`: focused contract tests.

The tool does not start servers, download tasks, or execute model-generated code. Existing guarded harnesses retain ownership of GPU process launch and telemetry. This separation prevents a dry run from turning into an 80 GB model load or a network download.

Future promoted runs should use visible, titled server, client, and diagnostic consoles. Follow the [benchmark evidence capture protocol](../docs/BENCHMARK_EVIDENCE_CAPTURE.md) for timestamped `nvidia-smi`, RAM/pagefile/process diagnostics, screenshot stages, redaction, and SHA-256 manifesting. Benchmark execution remains blocked until the manifest's MTP parity prerequisite passes.

## Validate the package

Contract only:

```powershell
python scripts/benchmark.py validate --manifest benchmarks/manifest.json
python -m unittest discover -v -s tests -p "test_benchmark_contract.py"
```

Contract plus the retained local target, sidecar, shard sizes, and executable hashes:

```powershell
python scripts/benchmark.py validate --manifest benchmarks/manifest.json --require-local-artifacts
```

The local-artifact validation intentionally trusts the existing 33-file SHA-256 authority while checking every current shard's presence and byte size. It freshly hashes the 2.20 GB promoted sidecar and the small server executable. A full fresh 78.5 GB rehash is a separate release action.

## Inspect execution plans

```powershell
python scripts/benchmark.py plan --manifest benchmarks/manifest.json --profile smoke
python scripts/benchmark.py plan --manifest benchmarks/manifest.json --profile performance
python scripts/benchmark.py plan --manifest benchmarks/manifest.json --profile capability
python scripts/benchmark.py plan --manifest benchmarks/manifest.json --profile release --json
```

Blocked stages stay in the plan. They are not silently omitted.

## Result layout

New public-suite results belong under:

```text
results/PUBLIC-BENCH-001/
  manifest.snapshot.json
  runs.jsonl
  summary.json
  raw/
  logs/
  telemetry/
```

Each line of `runs.jsonl` is one immutable `q38-public-benchmark-v1` object. Never rewrite a poor row. Append a superseding run and preserve the original status.

Validate rows:

```powershell
python scripts/benchmark.py check-results `
  --manifest benchmarks/manifest.json `
  --results results/PUBLIC-BENCH-001/runs.jsonl
```

The checker rejects a `VALID` MTP row unless it reports nonzero drafts and exact target text and token identity. It also rejects occupied-context rows below 99% prompt occupancy, without a passing needle, or across the declared hard resource floors.

## Stage 1: artifact integrity

Verify before each release run:

- official Qwen base revision;
- AtomicChat immediate source revision;
- 33 target shards and aggregate bytes;
- target manifest identity;
- promoted sidecar bytes and SHA-256;
- runtime commit and executable SHA-256;
- clean process, port, GPU, RAM, and disk preflight.

## Stage 2: deterministic performance and MTP correctness

Run target-only first on three fixtures:

- exact-copy compatibility;
- approximately 200-token code;
- approximately 180-word prose.

Use three release repeats with explicit cold/warm state. Stream the response to timestamp the first content token. Record:

- prompt tokens and prefill;
- TTFT;
- completion tokens and decode;
- wall time;
- finish reason;
- output and retokenized-token hashes;
- target and draft placement;
- drafted and accepted tokens;
- RAM, VRAM, RSS, pagefile, page faults, disk, network, and GPU utilization;
- raw response and server log.

MTP rows are valid only after the matching target hashes exist. An all-accepted fixture proves mechanics, not rollback correctness. At least one rejection-bearing fixture must pass exact parity.

Current gate:

- promoted-artifact all-accepted fixtures: expected-hash pass, mechanics only;
- promoted-artifact code with rejections: not yet established in a matched target/MTP A/B;
- promoted-artifact prose with rejections: blocked because output and token hashes differ;
- earlier +19.1% code and -18.8% prose rows: historical Q4_K_M target/sidecar evidence, excluded from current-artifact claims by the [MTP provenance boundary](../results/QWEN38-MTP-PROTOTYPE-001/PROVENANCE_BOUNDARY_2026-08-30.md).

## Stage 3: actual occupied context

Test target-only at 8K, 16K, 32K, and 64K first. The generated prompt must be at least 99% of its token target and include a deterministic needle near the beginning.

Record actual prompt tokens in the same row as allocated context. A large `--ctx-size` with a 232-token prompt is capacity evidence only.

The 80K and 128K occupied profiles are follow-ups after 64K passes with adequate headroom. The current 59,996-token attempt failed the 6 GiB RAM floor, so MTP long-context release work is blocked.

## Stage 4: KV and resource safety

Keep this separate from the speed headline:

- F16, Q8_0, and Q4_0 target KV at 8K and 16K;
- Q4_0 64K capacity only after short profiles pass;
- load time, first-request time, RSS, available RAM, VRAM, pagefile, page faults, disk bytes, and cleanup;
- promoted daily row requires at least 1,024 MiB minimum free VRAM.

## Stage 5: quant regression

Use `llama-perplexity` on the project corpus and, when present locally, the excluded llama.cpp-derived corpus with exact corpus hashes, tokenizer, context, stride, chunk count, and scored tokens. The second corpus is deliberately absent from a fresh clone; its tracked manifest records source hashes and the expected corpus hash.

These corpora are project-specific. They can detect quant regressions but cannot establish general intelligence.

The frozen evidence build currently lacks `llama-perplexity` and `llama-bench`. Build them from the exact chosen runtime revision before running this stage. Do not substitute an unrelated binary without recording its commit and hash.

## Stage 6: capability and intelligence vector

The manifest pins `lm-evaluation-harness` source commit `c1b3b3a33e0e17bcb329a3e4dc7825b77cb5d373` and LongBench source commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e` as of 2026-08-30. The harness is not installed and dataset revisions remain deliberately blocked until execution. Pin every dataset/task revision before running, then use the exact local chat template with temperature 0, seed 38027, batch 1, and concurrency 1.

The planned vector is:

| Facet | Task |
| --- | --- |
| Instruction following | IFEval |
| Broad knowledge and reasoning | selected MMLU subjects plus ARC-Challenge |
| Commonsense completion | HellaSwag |
| Math | GSM8K |
| Code | MBPP, in an isolated scorer |
| Tool use | local strict JSON/function fixtures |
| Multilingual reading | pinned Belebele subset |
| Truthfulness limitation | TruthfulQA MC2 |
| Long context | local occupied needles first, then a licensed pinned LongBench subset |
| Quant regression | local perplexity and next-token comparisons |

Pilot limits in the manifest are plumbing and variance checks. Do not publish them as full benchmark scores. Run full task sets only after the pilot confirms template, scorer, and runtime stability.

Generated-code evaluation is marked unsafe and must run in an isolated environment with explicit approval. The suite does not enable it automatically.

## Stage 7: human review

Automated scores miss style, usefulness, calibration, and failure severity. Before release, run a small blinded pairwise review against the immediate AtomicChat source or another exact local control on:

- explanation quality;
- code correctness and maintainability;
- tool-call restraint;
- multilingual fluency;
- long-answer coherence;
- refusal and over-refusal behavior.

Publish the rubric, randomization, sample count, ties, and reviewer count. Do not turn a one-person development review into a statistically general claim.

## Current limitations

- No broad public capability results exist yet for the retained target.
- Public task revisions remain intentionally marked `PIN_BEFORE_RUN`.
- The frozen evidence build has no `llama-bench` or `llama-perplexity` executable configured.
- TTFT is missing from the old non-streaming harness rows.
- The 60K occupied-context run failed.
- MTP exact parity is fixture-dependent.
- The historical IQ3 comparison artifact is not currently present. Its evidence remains `PROXY` or `EXTERNAL`; no 82 GB download is authorized.
