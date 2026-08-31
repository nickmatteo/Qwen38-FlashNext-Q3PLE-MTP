# Decision: llama.cpp, Q3_PLE, and learned MTP are the project

Decision ID: `CLUSTER-DEC-007`

Updated: 2026-08-30

## Decision

Retire the FreeToken Qwen4 experiment and remove its model, worktrees, generated artifacts, logs, results, and active documentation.

Make the retained llama.cpp plus Q3_PLE plus learned-MTP result the sole active research lane.

Publish nothing yet. Prepare a small standalone reproducibility repository first. Treat a Hugging Face model release and any llama.cpp upstream contribution as separate later decisions with separate gates.

## Why

The retained lane has produced the strongest result on the actual machine:

- 31.45 t/s mean warm decode at a 16K allocation;
- 25.81 t/s at an 80K allocation;
- successful short-prompt generation with allocations through 224K;
- exact text and token hashes on the promoted compatibility fixture;
- a frozen 33-shard, 78,525,318,176-byte target manifest and a frozen winning MTP sidecar hash.

The removed lane consumed 219.659 GiB and never produced a practical end-to-end profile that beat this result. Keeping it as the primary project made the repository harder to understand and diverted attention from the working system.

## Promoted configuration

Hardware:

- RTX 5070 12 GB;
- Ryzen 9 5900XT;
- 64 GB DDR4-3000;
- Windows 11;
- target and PLE files on Z: NVMe.

Target:

- `AtomicChat/Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64`;
- 33 GGUF shards;
- 78,525,318,176 bytes;
- 1,223 non-PLE tensor payloads preserved byte-for-byte from the verified AtomicChat source;
- only `per_layer_token_embd.weight` changed from Q5_1 to the project Q3_PLE type.

Winning learned-MTP sidecar:

- file: `mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf`;
- bytes: 2,202,883,264;
- SHA-256: `7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E`;
- corrected FC/HC tensor structure;
- Q4_0 down experts and Q4 output path;
- routed draft experts in pinned host memory for the promoted profile.

Runtime identity:

- branch: `cluster/q3ple-mtp`;
- commit: `4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc`;
- llama.cpp build: 10690;
- `llama-server.exe` SHA-256: `72BB9839C156ABBBA5D55B0CA3F2D7F89A931ACAA8A32BA40A8D76BBB4B67436`.

## Measured boundary

| Allocated context | Target placement | Warm decode | Actual prompt tokens | Classification |
| ---: | --- | ---: | ---: | --- |
| 16,384 | n45 | 31.45 t/s mean | 232 | repeated short-prompt measurement |
| 65,536 | n47 | 26.45 t/s | 232 | capacity profile |
| 81,920 | n47 | 25.81 t/s | 232 | capacity profile |
| 131,072 | n47 | 27.41 t/s | 232 | capacity profile |
| 196,608 | n48 | 26.65 t/s | 232 | capacity profile |
| 229,376 | n48 | 26.89 t/s | 232 | capacity profile |
| 262,144 | n48 | unsafe | none | rejected at initialization |

No row above 16K is evidence of sustained decode with a filled long context. The attempted 59,996-token content run is `FAILED`: minimum available RAM crossed just below 6 GiB, the server connection reset, and no valid completion was recorded.

## MTP correctness decision

MTP feasibility is confirmed, but the current implementation is `PASS_CONDITIONAL`.

Promoted-artifact evidence:

- the compatibility and headline short-prompt fixtures match their preserved expected text and token hashes when every draft is accepted;
- the promoted Q3_PLE target and 2.20 GB sidecar run together in the frozen runtime;
- on the promoted-artifact prose A/B, target-only warm decode was 18.3953 t/s and MTP warm decode was 18.8486 t/s with 101 of 112 drafts accepted.

Blocking evidence:

- the promoted-artifact prose output and retokenized-token hashes differ from target-only, so its speed is not promotable;
- no matched rejection-bearing code A/B has established exact parity for the promoted target, sidecar, and runtime together;
- low-bit verifier geometry remains sensitive to batch shape near greedy logit ties.

Historical implementation evidence is preserved but excluded from current-artifact claims. The 13.7895 to 16.4212 t/s code result and the 14.7941 to 12.0157 t/s prose result used an earlier Q4_K_M target, earlier runtime worktree, and 2.786 GB Q4_K_M sidecar. They demonstrated useful rollback behavior during development, but they do not benchmark the promoted Q3_PLE configuration.

Therefore:

- do not call MTP universally faster;
- do not call the current runtime production-ready;
- do not average code and prose into one headline number;
- require exact text and retokenized-token parity per fixture before promoting that fixture's speed result.

## Open-source decision

Use two repositories eventually, with a third upstream path only if it earns its way in:

1. A small standalone research repository containing scripts, patches, source pins, manifests, benchmark contracts, selected raw evidence, and limitations. Do not include local build trees or giant weights.
2. A separate Hugging Face model repository for the Q3_PLE target and MTP sidecar after provenance, license, quality, runtime, and upload gates pass.
3. An optional narrow llama.cpp contribution after the existing Qwen4Exp MTP draft work is reconciled and the generic change passes upstream expectations. Do not make a duplicate MTP PR and do not use a wholesale llama.cpp fork as the project's public home.

The current upstream state supports this sequence: base Qwen3.8 support is merged, but Qwen4Exp MTP remains a draft contribution and the stock converter still disables MTP export.

## Release gates

The research repository can be prepared before the model is released. The model repository stays blocked until all of these are true:

- every shipped file has known origin and license treatment;
- Qwen Community License 1.0 notice and conditions are included;
- AtomicChat and Cafe/llama.cpp provenance is documented;
- the target and every promoted sidecar have complete byte and SHA-256 manifests;
- a clean occupied-context run passes at the claimed working-set size;
- MTP exact parity passes compatibility, code, prose, and rejection-bearing fixtures;
- target-only and MTP results have matched repeated runs and resource telemetry;
- the public capability suite covers instruction following, knowledge, reasoning, math, code, tool use, multilingual behavior, long-context retrieval, quant regression, and safety limitations;
- the user explicitly authorizes the upload.

## Next executable work

1. Finish the compact benchmark contract and validate its manifests.
2. Run target-only baselines before any new MTP comparison.
3. Diagnose or mitigate the 60K occupied-context RAM failure without weakening watchdog floors.
4. Rerun matched rejection-bearing code and prose fixtures on the exact promoted target, sidecar, and runtime, then close parity before publishing any general MTP speed claim.
5. Run the pinned public capability suite and retain exact task revisions, prompts, scorer versions, exclusions, and raw outputs.
6. Prepare, but do not publish, the standalone repository and Hugging Face model card.

No download, paid compute, push, PR, Hugging Face upload, or community post is authorized by this decision alone.
