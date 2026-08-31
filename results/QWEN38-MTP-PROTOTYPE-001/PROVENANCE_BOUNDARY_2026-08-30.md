# MTP measurement provenance boundary

This note corrects an attribution hazard without rewriting immutable result JSON.

## Historical state-fix measurements

`mtp_statefix_benchmark_20260829.json` summarizes code and prose measurements from an earlier implementation stage. The raw command records are:

- `logs/QWEN38-MTP-PROTOTYPE-001/runtime-bench-code-mtp-r1.json`;
- `logs/QWEN38-MTP-PROTOTYPE-001/runtime-bench-final-mtp-r1.json`.

Those commands used:

- target: `AtomicChat/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`;
- runtime worktree: `llama.cpp-qwen38-bf16-dispatch-diag`;
- sidecar: `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, approximately 2.786 GB.

The 13.7895 to 16.4212 t/s code row and 14.7941 to 12.0157 t/s prose row are valid historical implementation evidence. They are not measurements of the promoted Q3_PLE target or promoted 2.20 GB sidecar.

## Promoted artifact set

Current release-candidate work is scoped to:

- target: `AtomicChat/Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64`;
- runtime: `cluster/q3ple-mtp` at `4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc`;
- sidecar: `mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf`, 2,202,883,264 bytes, SHA-256 `7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E`.

`q3ple_champion_prose_ab.json` is a matched promoted-artifact prose A/B. Its warm rows measured 18.3953 t/s target-only and 18.8486 t/s with MTP, with 101 of 112 drafts accepted. Output and retokenized-token hashes differ, so the MTP row fails the correctness gate and is not a promotable speed result.

No matched rejection-bearing code A/B currently establishes exact target/MTP parity for this promoted artifact set.

## Reporting rule

Every performance statement must name or resolve to the exact target, draft sidecar, runtime commit, and raw command. Historical state-fix measurements may explain implementation progress, but they must not be merged into a current-artifact benchmark table.
