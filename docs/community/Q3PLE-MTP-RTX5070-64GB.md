# Qwen3.8-Flash-Next on an RTX 5070 12 GB and 64 GB RAM

## Status

Preliminary reproducibility report, updated 2026-08-30.

The short-prompt measurements are real and preserved. The long rows describe allocated context capacity with a 232-token prompt. They do not prove sustained speed at a filled 64K, 80K, 128K, 192K, or 224K working set.

The current learned-MTP runtime is research-grade. The copied all-accepted fixture matches preserved expected hashes, but new rejection-bearing code and prose pairs on the 80K daily-driver allocation do not match target-only output and are not repeat-stable under MTP. The 100% acceptance and exactness claim is therefore fixture-specific, not general.

## Hardware

- GPU: NVIDIA GeForce RTX 5070, 12 GB, PCIe 4.0 x16 at x16.
- CPU: AMD Ryzen 9 5900XT, 16 cores / 32 threads.
- RAM: 64 GB, 4 x 16 GB DDR4 at 3000 MT/s.
- Storage: Z: NVMe for GGUF and PLE paging.
- OS: Windows 11 Home, build family 26200.

## Reproducible runtime

- branch: `cluster/q3ple-mtp`;
- commit: `4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc`;
- llama.cpp build: 10690;
- `llama-server.exe` SHA-256: `72BB9839C156ABBBA5D55B0CA3F2D7F89A931ACAA8A32BA40A8D76BBB4B67436`;
- CUDA 13.1, SM120a, Flash Attention on.

## Target and draft

Target:

- AtomicChat `Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64`;
- 33 shards, 78,525,318,176 bytes;
- PLE tensor: 22,400,107,520 bytes, Q3_PLE type 43, lazy mmap-backed CPU row lookup;
- authoritative shard manifest: `results/ATOMICCHAT-4.27-Q3PLE-001/gate_b_derivative_manifest.json`.

Promoted MTP sidecar:

- `mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf`;
- 2,202,883,264 bytes;
- SHA-256 `7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E`;
- corrected FC/HC tensor split and naming;
- Q4_0 down experts, Q4 output path, Q4_K gate/up experts.

## Winning 16K layout

This is a preserved historical throughput result, not the recommended live desktop profile. Its 776-780 MiB free-VRAM margin is below the project's 1,024 MiB release-headroom policy. A later smoke in the current desktop state reached 546 MiB during MTP startup and was correctly stopped by the unchanged 768 MiB emergency floor.

- context allocation: 16,384;
- target MoE: `--n-cpu-moe 45`;
- target KV: Q4_0 K and Q4_0 V;
- target threads: 11;
- target ubatch: 256 in the promoted family;
- MTP: `draft-mtp`, nmax 3, p_min 0.75;
- draft threads: 8;
- draft ubatch: 64;
- MTP output and dense glue tensors on CUDA;
- MTP routed expert banks in pinned host memory;
- draft-only token-batch expert CUDA streaming enabled;
- expert LRU off;
- extra scheduler DMA prefetch off;
- unified KV off.

## Measured short-prompt results

Every row used a 232-token prompt and a 171-token completion. Every output matched the preserved expected text and token hashes. The first two rows are independent top-profile runs; the remaining rows are one warm row each.

| Allocated context | n-cpu-moe | Warm decode | Warm prefill | MTP accepted | Min free VRAM | Evidence class |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16K, run 1 | 45 | **31.5021 t/s** | 92.8814 t/s | 129/129 | 780 MiB | measured repeat |
| 16K, run 2 | 45 | **31.3968 t/s** | 92.8484 t/s | 129/129 | 776 MiB | measured repeat |
| **16K mean** | **45** | **31.4495 t/s** | **92.8649 t/s** | **100%** | tight | promoted short profile |
| 64K | 47 | **26.4499 t/s** | 71.2189 t/s | 128/128 | 1,745 MiB | capacity profile |
| 80K | 47 | **25.8109 t/s** | 82.6519 t/s | 128/128 | 1,569 MiB | capacity profile |
| 128K | 47 | **27.4126 t/s** | 82.9789 t/s | 128/128 | 1,008 MiB | capacity profile |
| 192K | 48 | **26.6527 t/s** | 78.0875 t/s | 128/129 | 1,365 MiB | capacity profile |
| 224K | 48 | **26.8854 t/s** | 79.1947 t/s | 128/129 | 936 MiB | capacity profile |

The 128K row is faster than 80K in this one warm short-prompt run. That is not evidence that more context makes inference faster. No repeat set or occupied-context workload supports that interpretation.

## Capacity boundary

- 128K allocation generated correctly with n47 and Q4 target/draft KV.
- 192K and 224K generated correctly with n48.
- cold first requests became much slower at the largest allocations.
- 262,144 failed the unchanged 768 MiB VRAM safety floor during initialization.

## Filled-context and agentic operation

The first real 80K-profile attempt built 59,996 content tokens inside an 81,920-token allocation. It is not a successful benchmark:

- minimum available RAM fell just below the 6 GiB hard floor;
- minimum free VRAM remained 1,126 MiB;
- the server reset the connection;
- no valid completion, needle result, TTFT, prefill, decode, or MTP parity row was produced.

The file `q3ple_ctx80k_filled60k_r2.json` is correctly marked `FAILED`. A safer n48 plus 38 GiB working-set profile remained resource-safe through 12,309 prompt tokens, but advanced at only 35.62 prompt tokens/s, projecting roughly 28-33 minutes for a cold 60K rebuild. That path is operationally rejected for an agent harness.

The usable design is incremental prefix reuse plus persisted target and MTP state. An isolated experimental runtime patch stores the draft sequence state and MTP carryover beside the target slot, validates their exact prompt identity, and fails closed on a missing or mismatched companion. It is not a crash-atomic production file format.

Measured growing-prefix results on the 81,920-token n48 agent profile:

| Saved prefix | Prefix steps | Median incremental prefill | Increment latency | Save / restore | Restart parity | Min free VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 8 | not promoted | 493 tokens at 42.03 t/s | 0.14 / 0.11 s | exact, 9/9 drafts both sides | 2,507 MiB |
| 8K | 16 | 42.35 t/s | 512 tokens at 43.59 t/s | 0.25 / 0.27 s | exact, active MTP | 2,485 MiB |
| 16K | 32 | 42.01 t/s overall; 43.75 t/s warm | 494 tokens at 35.30 t/s | 0.43 / 0.32 s | exact, 8/8 drafts both sides | 2,445 MiB |
| 32K literal continuation | 33 from saved 16K | 36.67 t/s median; 35.34 t/s mean for full 512-token turns | 13.98 s median per 512 tokens | 0.82 / 0.63 s | exact, 9/9 drafts both sides | 2,403 MiB |
| 48K chained continuation | 32 from saved 32K | 34.08 t/s median; 33.72 t/s mean | 15.06 s median per 512 tokens | 1.19 / 0.98 s | exact, 9/9 drafts both sides | 2,312 MiB |
| 60K chained continuation | 22 from saved 48K | 34.12 t/s full median; 34.78 t/s warm median; 33.66 t/s full mean | 14.77 s warm median per 512 tokens | 1.45 / 1.16 s | exact, 9/9 drafts both sides | 2,250 MiB |

These are suffix-prefill measurements while a conversation grows, not cold full-prompt throughput. They show why a long-lived agent can be practical even though replaying 60K from zero is not.

The first 16K-to-32K attempt deliberately failed before inference because a freshly retokenized longer user message did not preserve the arbitrary BPE boundary of the sealed 16,366-token state. The harness did not trim the cache or hide that mismatch. The passing 32K diagnostic instead kept the saved token vector immutable and appended a deterministic block tokenized from previously unused local source text. It saved exactly 32,768 reusable tokens, plus a separate 53-token retrieval suffix used only for validation. The new 443.8 MB target slot and 19.7 MB MTP companion parsed to identical token vectors, then reproduced the exact A01 output and 9/9 draft counters after a fresh server restart. Minimum available RAM was 7.85 GiB, pagefile growth was zero, and all 16,402 appended tokens took 483.25 seconds in total.

That 32K row is explicitly a `synthetic_literal_slot_continuation`: it proves low-level target/MTP state continuation and restart safety. It is not a canonical chat-template boundary, not a one-shot filled-context benchmark, and not yet a production/crash-atomic persistence format. Raw result SHA-256: `E4B93425D0943CB7091318C0AD2FEDFE655DC73010A4EA1BB30AFFA273EEE431`.

The next run proved that the state is genuinely chainable rather than a disguised 16K replay. It loaded the saved 32,768-token pair and the first extension request reported exactly `cache_n=32768`, `prompt_n=512`. Thirty-two exact additions produced a 49,152-token state. The first addition after the fresh server load was cold at 9.04 t/s / 56.64 seconds; later turns reached as high as 43.33 t/s. The new target and `.dft` files were 600,245,784 and 29,532,256 bytes, parsed to identical token vectors, and restored the exact A01 output with 9/9 MTP. Minimum available RAM was 7.23 GiB, minimum free VRAM was 2,312 MiB, and pagefile growth was zero. Raw result SHA-256: `C64BB9FE723220DCA66F3BBBC0EC27C6C8160EE074C164775BCE8C5FCA3972D9`.

The final chained gate loaded that 49,152-token pair and extended only the unseen suffix to exactly 60,000 reusable tokens: 21 full 512-token additions plus a 96-token tail. The first request again proved exact reuse (`cache_n=49152`, `prompt_n=512`) and paid the cold-residency cost at 9.83 t/s / 52.10 seconds. The remaining full additions averaged 34.85 t/s with a 34.78 t/s / 14.77-second median. All 10,848 appended tokens completed in 358.08 seconds. The 703,822,488-byte target state and 36,041,056-byte `.dft` saved/restored in 1.45/1.16 seconds and reproduced exact A01 text/token hashes with 9/9 MTP after a fresh process restart. Minimum available RAM was 6.90 GiB, minimum free VRAM was 2,250 MiB, peak owned RSS was 38.09 GiB, and pagefile growth was zero. Raw result SHA-256: `1891BBA32A92F9819ADC65844FE96EADEAEC4D048D895D739B4392B7A6C99FF7`.

The cold first-turn result matters operationally. Disk persistence is roughly one to one-and-a-half seconds at 48K-60K, but restarting the process also discards useful OS/model residency. A daily agent should keep the server warm when practical; slot persistence is for recovery and bounded checkpoints, not a promise that the first post-restart turn is as fast as steady state. These 60K results remain explicitly tagged as synthetic literal-token state continuation, not a canonical retokenized chat prompt or one-shot filled-context benchmark.

## Canonical 59,750-token result

The synthetic limitation is now resolved by a separate measured run. One canonical chat transcript reached a complete assistant-message boundary at 59,750 occupied tokens. It was sealed at 15,872, 31,700, 47,303, and 59,750 tokens. Every later stage started a fresh process, restored the preceding target and `.dft` pair, and processed only the unseen suffix. The exact restore proofs were `15,872 + 512`, `31,700 + 512`, and `47,303 + 512` in `cache_n + prompt_n` accounting.

The staged build passed with 2,113 MiB minimum free VRAM, about 9.20 GiB minimum available RAM, a 37.00 GiB peak owned working set, zero workflow-wide pagefile growth, and no watchdog violation.

At the same 59,750-token depth, retrieval passed three target-only and three MTP repeats with exact text and retokenized-token parity. Warm decode means were 18.25 tok/s target-only and 18.34 tok/s MTP. Each MTP repeat accepted 10 of 11 drafts. The 0.5% numerical difference is not a meaningful general speed claim; the exact rejection-bearing parity is the important result.

The code fixture passed target-only and remained stable. MTP was numerically faster at 22.30 tok/s warm versus 18.15 tok/s target-only, but changed exact text and tokens in every repeat. That lane is explicitly non-promotable. The prose target produced a stable 90-word response to a 170 to 210-word request, so the broad target gate remains blocked and no prose MTP row was run.

See [the complete canonical evidence note](Q3PLE-CANONICAL-60K-2026-08-31.md) for result hashes and claim boundaries.

## Target-only versus MTP

The all-accepted compatibility fixture is useful for mechanics, but it does not test rollback under rejections.

For the exact promoted Q3_PLE target, 2.20 GB sidecar, and frozen runtime, the new 81,920-allocation rejection-bearing code pair measured:

- second target-only row: 22.0386 t/s;
- second MTP row: 24.0769 t/s (+9.25% numerically);
- accepted: 210 of 224 drafts, or 93.75%;
- minimum free VRAM under MTP: 1,344 MiB;
- exact output and retokenized-token parity: fail;
- MTP repeat stability: fail.

All four code outputs satisfy the corrected static functional contract, but comments and docstring wording changed. Strict raw-text and token parity therefore still disqualify the speed row.

The matched 81,920-allocation prose pair measured:

- second target-only row: 21.8279 t/s;
- second MTP row: 19.7634 t/s (-9.46%);
- accepted: 130 of 146 drafts, or 89.04%;
- minimum free VRAM under MTP: 1,311 MiB;
- exact output and retokenized-token parity: fail;
- MTP repeat stability: fail; the second MTP response also exceeded the requested word range.

An earlier 16K promoted-artifact prose A/B measured:

- target-only warm decode: 18.3953 t/s;
- MTP warm decode: 18.8486 t/s;
- accepted: 101 of 112 drafts, or 90.18%;
- exact output and retokenized-token parity: fail.

That older MTP row was numerically 2.46% faster, but it was not a valid speed win because the generated continuation changed.

Earlier state-fix work measured code at 13.7895 to 16.4212 t/s with exact parity and prose at 14.7941 to 12.0157 t/s without parity. Those runs used a different Q4_K_M target, a different runtime worktree, and the earlier 2.786 GB Q4_K_M sidecar. They remain historical correctness evidence only and must not be attributed to the promoted Q3_PLE configuration. See the [MTP measurement provenance boundary](../../results/QWEN38-MTP-PROTOTYPE-001/PROVENANCE_BOUNDARY_2026-08-30.md).

The correct conclusion is functioning learned-MTP mechanics with a quantized-target verification/repeatability gate. MTP helps this code fixture, hurts this prose fixture, and is not exact on either. There is no general MTP speed claim.

## Negative configuration results

These were measured and should not be copied into the recommended profile:

- removing explicit PLE CPU placement reduced 80K warm decode to 20.96 t/s;
- unified KV reduced 80K warm decode to 22.19 t/s;
- strict physical-core affinity reduced it to 23.40 t/s;
- 12 target threads reduced it to 24.49 t/s and crossed the VRAM floor in the packed 16K profile;
- MTP nmax 4 was slightly slower than nmax 3;
- 12 draft threads were slower than 8;
- 128, 256, and 512 MiB expert LRU improved cold behavior but not warm decode;
- extra scheduler DMA prefetch was slower than pinned-host experts alone;
- global Q3 draft quantization was slower than surgical draft quantization.

## Attribution

This work combines Qwen's model, upstream llama.cpp, AtomicChat's 4.27-bpw quant, Cafe/Ark MTP work, and the project's Q3_PLE conversion, FC/HC correction, sidecar quant experiments, and hardware placement. Pinned-host MoE and expert-streaming ideas also have historical Cafe and FreeToken lineage.

Do not present this as a solo invention or a record. Independent reproduction does not exist yet.
