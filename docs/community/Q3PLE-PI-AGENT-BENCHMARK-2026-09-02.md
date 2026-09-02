# A 125B-class MoE coding agent on a 12 GB RTX 5070

Qwen3.8-Flash-Next Q3_PLE completed a headless 20-turn Pi session on one
Windows PC with an RTX 5070 12 GB and 64 GB DDR4. After the cold first turn,
the median turn finished in **3.79 seconds** with a **1.57-second median TTFT**.

That headline is intentionally narrower than a coding-quality claim. This run
measured whether a real agent harness could reuse one warm slot correctly and
survive a clean process restart. It used the target model only. Learned MTP was
disabled.

## The 20-turn result

| Measurement | Result |
| --- | ---: |
| Turns | 20/20 PASS |
| Cold first-turn wall time | 42.62 s |
| Cold first-turn TTFT | 39.65 s |
| Warm wall-time mean | 3.71 s |
| Warm wall-time median | **3.79 s** |
| Warm TTFT mean | 1.58 s |
| Warm TTFT median | **1.57 s** |
| Warm wall-time range | 3.15 to 4.68 s |
| Cache accounting | exact on every turn |
| Template-boundary replay | 2 tokens per warm turn |
| New prompt tokens per warm turn | 36 |
| Minimum free VRAM | 3,146 MiB |
| Minimum available RAM | 12.38 GiB |
| Watchdog violations | none |

The first turn processed 440 prompt tokens from a cold model state. Every
later turn reused the cached conversation. Qwen's template changed two marker
tokens at the previous boundary, so the harness allowed and recorded exactly
that two-token replay. It still required `cache_n + prompt_n` to equal the full
rendered request length on every turn.

## Restart without replaying the conversation

A second run rendered a complete assistant-message boundary, aligned the live
slot to that canonical token vector, and saved 535 tokens. After a clean stop
and fresh model process:

- state restore took **47.1 ms**;
- the next request reported `cache_n=535`;
- only 32 new prompt tokens were processed;
- the requested post-restart response was exact;
- the first post-restart request took 60.52 seconds because model page
  residency was cold again.

This separates two costs that are often conflated. Restoring the conversation
state is fast. Warming the model's memory residency after a process restart is
not. The practical daily design is therefore one persistent process, one slot,
incremental suffix processing, and periodic state saves.

## The first tool job

Pi then completed a bounded read-only repository investigation and found all
four required code symbols. It took two model calls. The second call received
9,601 new prompt tokens because the agent read a large source file. That turn
took 321.97 seconds.

We count the job as a verifier PASS and the context behavior as an efficiency
warning. Before expanding to long autonomous coding work, tool output needs to
be sliced, summarized, or requested in narrower ranges. A correct but
unbounded file read is not a good agent loop.

## Why the GPU was not filled to 12 GB

The run used about 8.5 GiB of VRAM at one observed point and retained at least
3,146 MiB across the sealed 20-turn telemetry. That reserve is shared by KV
growth, CUDA work buffers, transient allocations, and the safety margin needed
to avoid failing late in a long agent session.

Moving one more MoE layer to the GPU is a valid candidate, but it is not yet a
result. The n48 profile remains the baseline. An n47 candidate will be promoted
only if it preserves the exact task result, stays above 1,024 MiB free VRAM,
passes all RAM/RSS/pagefile gates, and improves matched end-to-end agent time.

## Claim boundaries

- This is a target-only Pi lifecycle benchmark, not an MTP benchmark.
- It proves cache reuse and restart continuation, not broad intelligence.
- The warm turns are short, deterministic agent messages, not 60K occupied
  context.
- The model is Qwen3.8-Flash-Next, whose official architecture has a 125B main
  MoE plus separate N-gram embeddings and activates about 6B parameters per
  token. The Q3_PLE artifact is a project derivative of an AtomicChat source
  quant, not an official Qwen or AtomicChat release.
- Raw requests, model reasoning, tool output, and machine-local paths remain
  private. Public numbers come from sanitized sealed summaries.

The useful result is not merely that a huge model ran on a small GPU. It is
that a real agent harness stopped replaying its entire history, proved the
cache accounting turn by turn, and resumed an exact canonical state after a
fresh process.
