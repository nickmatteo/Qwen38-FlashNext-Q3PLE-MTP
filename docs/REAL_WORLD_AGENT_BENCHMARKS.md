# Real-world coding-agent benchmarks

Status: Pi cache and restart lifecycle proven; real coding-task pilot in progress.

This lane measures whether Qwen3.8-Flash-Next Q3_PLE can finish recognizable
repository work on the RTX 5070 host. Pi is the primary workload and DeepSeek
Harness is the only second harness. The project adapter and task verifier, not
the agent UI, determine whether a result is valid.

## Result hierarchy

1. Objective task pass/fail and verifier output.
2. Human interventions, malformed tools, loops, and unrelated changes.
3. Total wall time and time to first useful edit.
4. Exact cache reuse and clean restart recovery.
5. Reasoning/generation/tool/build/test time breakdown.
6. Resource safety and only then raw token throughput.

The initial baseline is target-only with reasoning enabled. MTP comparisons
are prohibited until the target-only lifecycle and task suite are stable.

## Machine-readable contracts

- `benchmarks/agent/task.schema.json` defines a sealed task, permissions,
  limits, disposable-worktree setup, and objective verifier.
- `benchmarks/agent/result.schema.json` defines the promoted result row.
- `benchmarks/agent/pilot-manifest.json` tracks the pilot gates and task pack.
- `scripts/q3ple_pi_adapter.py` writes per-turn JSONL plus exact raw request
  and response evidence.

Every task forbids network access, pushes, merges, and external side effects.
The model can edit only the disposable worktree and can run only the declared
commands.

## Evidence boundary

No result is promoted merely because Pi or DSH reports completion. The custom
harness must independently run the verifier, capture the final diff, detect
unrelated changes, bind the exact model/runtime/profile/adapter identities,
and join the adapter turn rows with the runtime watchdog telemetry.

Raw adapter evidence can contain source code, tool output, and reasoning. It
stays private until a separate redaction and release-manifest pass.

## First live Pi results

The target-only reasoning profile completed a headless 20-turn Pi session with
exact cache accounting on every turn. The first turn was cold. Turns 2 through
20 reused the existing prefix, replayed only the two-token template boundary,
and submitted 36 prompt tokens each.

| Measurement | Result |
| --- | ---: |
| Cold first-turn wall time | 42.62 s |
| Cold first-turn TTFT | 39.65 s |
| Warm wall-time median | **3.79 s** |
| Warm TTFT median | **1.57 s** |
| Warm wall-time range | 3.15 to 4.68 s |
| Exact cache-accounting turns | 20/20 |
| Minimum free VRAM | 3,146 MiB |
| Minimum available RAM | 12.38 GiB |
| Maximum owned RSS | 37.00 GiB |
| Maximum pagefile growth | 37.70 MiB |
| Watchdog violations | 0 |

The separate restart test aligned a complete assistant-message boundary,
saved 535 canonical tokens, stopped the exact-owned process, launched a fresh
process, and restored the state in 47.1 ms. The next request reported
`cache_n=535` and `prompt_n=32`, proving that it extended the restored prefix
instead of replaying it. Its 60.52-second wall time is the expected first-turn
page-residency penalty, not the warm-turn target.

The first read-only Pi tool job also passed its objective named-symbol checks.
It was slow because the agent read a large source file and returned 9,601 new
prompt tokens on its second model call. That is useful negative evidence: the
next tool-policy change should bound and slice tool output before attempting
long autonomous coding jobs.

These are target-only agent-harness results. They are not MTP results and they
are not yet a broad coding-intelligence score.

## Current gates

- Separate target-only reasoning profile: PASS (static).
- Pi/DSH OpenAI-compatible adapter: PASS (static, HTTP, and live Pi integration).
- Deterministic fake 20-turn append-only contract: PASS.
- Live 20-turn cache proof: PASS.
- Save/stop/restart/restore canonical continuation: PASS.
- Read-only Pi tool job: PASS, with a large-context efficiency warning.
- Tool-enabled restart continuation: PENDING.
- NVMe snapshot catalog and recovery: PENDING.
- First 3-5 coding tasks: PENDING.
- Isolated n47 placement comparison: PENDING; n48 remains the frozen baseline.
- DeepSeek Harness integration: PENDING; not installed or measured yet.
- Expanded 10-15 task suite: PENDING.
- 60-120 minute autonomous task: PENDING.

The Hugging Face release upload is complete. Further live work still requires
the normal idle and preflight gates, and every placement candidate gets its own
profile, state namespace, and result directory.
