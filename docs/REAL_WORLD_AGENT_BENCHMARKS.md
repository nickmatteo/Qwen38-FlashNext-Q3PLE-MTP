# Real-world coding-agent benchmarks

Status: Pi cache and restart lifecycle proven, DeepSeek Harness integrated and
passing its first read-only contract, n47 measured and rejected; real
coding-task pilot not started.

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

## DeepSeek Harness integration

DSH is pinned at `@deepseek-ai/dsh` 0.1.2-alpha.4, tag `dsh-v0.1.2-alpha.4`,
commit `4e84901e6471b79ec0338099867ebb4606d12bb5`, MIT licensed, run on Node
v26.5.1 through the shipped `headless` profile. It reaches the model through
the same loopback adapter Pi uses, so both harnesses are measured against one
gateway and one accounting contract.

`benchmarks/dsh/cordis.patch.yml` is the whole integration. It names the
`q3ple-local` route, points it at the adapter, declares the reasoning levels
the profile actually serves, pins a `read-only-headless` permission preset, and
disables the session-title plugin.

The first live DSH job was a bounded read-only symbol lookup. It passed.

| Measurement | Result |
| --- | ---: |
| Verifier | PASS, both line numbers exact |
| Turns | 2 |
| Cold turn 1 prompt tokens | 8,313 |
| Cold turn 1 wall time | 182.06 s |
| Turn 2 cached tokens | 8,381 |
| Turn 2 new prompt tokens | **93** |
| Turn 2 wall time | 8.27 s |
| Cache accounting | exact on both turns |
| Minimum free VRAM | 3,014 MiB |
| Watchdog violations | 0 |

Two findings are worth keeping.

DSH's system prompt plus 26 tool schemas cost 8,313 prompt tokens before the
agent does anything. That is the price of a full harness and it is paid once
per session, not per turn.

The second turn is the interesting number. DSH answered from a search result
and submitted 93 new prompt tokens, where the earlier Pi tool job read a whole
file and submitted 9,601. That is a tool-policy difference, not a model
difference, and it is the clearest evidence so far that bounding tool output is
worth more than any placement tuning.

### What DSH does not enforce on Windows

DSH's own documentation is explicit that its Windows sandbox backend is an ACL
restricted-token runner that reports **partial** enforcement in read-only mode,
and that network access is outside the sandbox vocabulary entirely. So the
`read-only` mode is defence in depth, not the fence. The disposable worktree,
the declared command allowlist, and the verifier remain the authoritative
boundary for any task that writes.

### Two integration failures worth recording

Both were sealed as negative evidence rather than deleted.

`dsh-readonly-v1` failed with `request rewrote or removed a prior canonical
chat message`. The cause was DSH's session-title plugin issuing its own
64-token completion down the same provider route. That is a second
conversation on a single-slot session, and the adapter was right to refuse it.
The fix is the `session-title-llm` disable in the patch file.

`dsh-readonly-v2` failed with `fresh session reused unexpected cache tokens:
8309`. A fresh adapter episode expects an empty slot, but the server still held
the previous attempt's prefix. The operator step is to
`POST /slots/0?action=erase` before starting a fresh episode against a server
that is already running.

## The n47 placement candidate is rejected

The question was whether the roughly 8.5 GiB the model uses leaves room for one
more MoE layer on the GPU. It does fit. It is not worth taking.

Both runs were cold-started back to back on an idle machine with the same
runner, the same deterministic 20 prompts, the same adapter, and the same
runtime build. The only difference was `--n-cpu-moe 48` against `47`.

| Measurement | n48 control | n47 candidate | Delta |
| --- | ---: | ---: | ---: |
| 20-turn wall time | 128.15 s | 134.20 s | **+6.05 s** |
| Warm wall mean | 3.339 s | 3.447 s | +0.108 s |
| Warm wall median | 3.322 s | 3.166 s | -0.156 s |
| Cold first turn | 37.49 s | 42.53 s | +5.04 s |
| Minimum free VRAM | 3,066 MiB | 2,131 MiB | **-935 MiB** |
| Peak VRAM used | 8,878 MiB | 9,813 MiB | +935 MiB |
| Exact outputs | 20/20 | 20/20 | same |
| Exact cache accounting | 20/20 | 20/20 | same |
| Watchdog violations | 0 | 0 | same |

n47 passed four of the five promotion gates. It produced every expected output
with exact accounting, no watchdog violation, and 2,131 MiB of free VRAM, well
above the 1,024 MiB floor. It failed the fifth: it did not win on matched
end-to-end time.

Read the latency honestly. The warm median favours n47 by about 5 percent and
the warm mean and total wall favour n48 by about 3 to 5 percent. One paired run
that disagrees with itself is noise, not a speedup. What is not noise is the
935 MiB of free VRAM margin n47 spends to get it, which matches the 0.8 to
1.2 GiB loss seen in earlier occupied-context n47 work.

Paying real headroom for a result indistinguishable from noise is a bad trade
in a session that has to survive KV growth and a large tool result late on.
**n48 remains the frozen baseline.** The candidate profile, both run
directories, and the gate-by-gate comparison are retained as negative evidence
at `results/REAL-WORLD-AGENT-001/n47-candidate/comparison-summary.json`.

The occupied-context safety probe was not run. Promotion had already failed on
the latency gate, so there was nothing left for it to decide.

### How a candidate stays isolated

Both the controller and the adapter used to hard-pin `--n-cpu-moe 48`, port
18090, and the baseline state namespace. Rather than loosen those pins, a
profile now has to declare itself:

```json
"candidate_of": "q3ple_daily_80k_reasoning_v1",
"placement_candidate": {
  "n_cpu_moe": 47,
  "baseline_profile_id": "q3ple_daily_80k_reasoning_v1",
  "baseline_port": 18090,
  "baseline_state_directory": "results/QWEN38-MTP-PROTOTYPE-001/state/q3ple_daily_reasoning_v1"
}
```

A declared candidate must have its own profile id, its own server port, and its
own state directory, and it still has to satisfy every other clause of the
reasoning contract. An undeclared profile that simply changes the placement is
still refused. Tests cover both directions.


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
- Isolated n47 placement comparison: DONE, REJECTED; n48 remains the frozen baseline.
- DeepSeek Harness pinned and wired to the adapter: PASS.
- DSH read-only task contract: PASS.
- DSH multi-turn cache proof: PENDING.
- Expanded 10-15 task suite: PENDING.
- 60-120 minute autonomous task: PENDING.

The Hugging Face release upload is complete. Further live work still requires
the normal idle and preflight gates, and every placement candidate gets its own
profile, state namespace, and result directory.
