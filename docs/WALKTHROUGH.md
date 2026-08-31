# How we fit and ran Qwen3.8-Flash-Next on a 12 GB RTX 5070

This is the complete path from the source quant to the published Q3_PLE target,
the learned-MTP sidecar, the 31.45 tok/s controlled result, and the canonical
59,750-token restartable state. It also records the results we rejected.

![Q3_PLE canonical benchmark card](community/assets/q3ple-canonical-59750-card.png)

## 1. Start with the real bottleneck

Qwen3.8-Flash-Next is a large multimodal MoE. The official model has 125B main
parameters, a 51.2B-element n-gram/PLE table, and roughly 6B active parameters
per token. Our machine was intentionally ordinary for this scale:

- Windows 11;
- AMD Ryzen 9 5900XT, 16 cores / 32 threads;
- 64 GB DDR4-3000;
- NVIDIA RTX 5070 with 12 GB VRAM;
- one 1 TB NVMe project drive.

The starting point was AtomicChat's 4.27-bpw GGUF. Its neural weights already
had a useful mixed quantization. The outsized component was
`per_layer_token_embd.weight`, the PLE/n-gram lookup table. Requantizing every
tensor would have destroyed the useful source layout and made provenance much
harder to prove, so we changed exactly one tensor family.

## 2. Replace only the PLE table

Q3_PLE is a project-specific 3.5-bit block format:

- 32 values per block;
- one BF16 scale per block;
- 12 bytes of packed 3-bit codes;
- 14 bytes per 32 values;
- private GGML type code 43 in the measured runtime.

The conversion streamed rows from the original Q5_1 PLE tensor and preserved
the existing 33-shard split. It used the equivalent of:

```text
llama-quantize --allow-requantize --keep-split --max-buffer-size 52 \
  --tensor-type "^per_layer_token_embd\.weight$=q3_ple" \
  <AtomicChat-first-shard.gguf> <Q3_PLE-output-prefix> COPY 16
```

The output target contains 33 shards totaling 78,525,318,176 bytes. The PLE
payload is 22,400,107,520 bytes. The other 1,223 tensor payloads, totaling
56,114,183,680 bytes, were verified byte-identical to the AtomicChat source.
The public target manifest contains every filename, size, and SHA-256.

## 3. Keep learned MTP separate

Qwen's learned MTP/NextN head was extracted as a detached GGUF so that the
target did not need to be rebuilt for every draft experiment. The promoted
sidecar is:

```text
mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf
```

- size: 2,202,883,264 bytes;
- SHA-256:
  `7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E`;
- down experts and output path selectively quantized;
- gate/up experts retained at Q4_K;
- FC/HC tensor layout corrected;
- routed draft experts placed in pinned host memory for the promoted profile.

Several larger and more aggressively quantized sidecars were tested locally.
They are not in the model release because they were not the promoted artifact.

## 4. Reconstruct the exact runtime

Stock llama.cpp does not understand this private Q3_PLE type or this complete
MTP/state path. The public repository includes the exact 36-commit patch series
used by the measured build:

```powershell
git clone https://github.com/quimmedes/cafe-llama.cpp.git
Set-Location cafe-llama.cpp
git checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
git am <research-repo>\patches\llama.cpp\cafe-035e227-to-73b803\*.patch
git rev-parse HEAD
```

The resulting head must be
`73b803464f25fc9054046728bf2ebed5a372737e`. Our measured Windows build used
CUDA 13.1, SM120a, and Flash Attention. Rebuilding the same source with another
toolchain is a reproduction attempt, not the identical binary.

## 5. Find the 16K throughput profile

We varied target expert placement while keeping the target, sidecar, prompt,
sampling, and output checks fixed. The best controlled profile used:

- 16,384 allocated context;
- 232 prompt tokens and 171 completion tokens;
- target `n_cpu_moe=45`;
- Q4_0 target KV;
- MTP `nmax=3`, `p_min=0.75`;
- pinned-host draft experts.

Two clean warm runs measured 31.5021 and 31.3968 tok/s, for a 31.4495 tok/s
mean. Both produced the expected text and token hashes and accepted 129 of 129
drafts. Minimum free VRAM was only 780 and 776 MiB, so this is a benchmark
profile, not the recommended daily configuration.

## 6. Separate allocation from occupancy

We then initialized larger context allocations with the same 232-token prompt:

| Allocated context | Placement | Warm decode |
| ---: | --- | ---: |
| 65,536 | n47 | 26.45 tok/s |
| 81,920 | n47 | 25.81 tok/s |
| 131,072 | n47 | 27.41 tok/s |
| 196,608 | n48 | 26.65 tok/s |
| 229,376 | n48 | 26.89 tok/s |

These rows prove allocation capacity and short-prompt decode behavior. They do
not measure a filled 64K to 224K prompt. The 262,144 allocation was rejected by
the VRAM safety floor.

## 7. Reject cold replay

The first monolithic attempt to occupy roughly 60K tokens crossed the 6 GiB
available-RAM floor and returned no valid completion. A safer attempt projected
roughly 28 to 33 minutes to replay the entire prefix. That is not an acceptable
agent design.

The replacement was stateful:

1. keep one server and one slot warm;
2. process only tokens that were not in the saved prefix;
3. save the target slot and its MTP `.dft` companion together;
4. restore both after a process restart;
5. fail closed if the pair does not match.

An earlier synthetic diagnostic grew a saved state from 49,152 to 60,000 tokens
by processing only 10,848 new tokens. Warm 512-token additions had a 34.78
tok/s median, the final save took 1.45 seconds, and restore took 1.16 seconds.
That proved the mechanism, but it was not yet a canonical chat boundary.

## 8. Build a canonical 59,750-token state

The final harness rendered one deterministic chat transcript and checkpointed
only after complete assistant messages. It built four sealed boundaries:

| Stage | Saved tokens | What the next fresh process processed |
| ---: | ---: | --- |
| 1 | 15,872 | initial prefix |
| 2 | 31,700 | suffix after restoring 15,872 |
| 3 | 47,303 | suffix after restoring 31,700 |
| 4 | 59,750 | suffix after restoring 47,303 |

At each restore, the first non-empty suffix proved cache accounting. The final
target and `.dft` contained the same token vector. The build recorded 9.20 GiB
minimum available RAM, 2,113 MiB minimum free VRAM, about 37.00 GiB peak owned
RSS, zero workflow-wide pagefile growth, and no watchdog violation.

## 9. Benchmark at actual depth

Every actual-depth comparison restored the same sealed state, ran target-only
first, then MTP, and used three repeats.

### Retrieval passed exact parity

| Mode | Cold decode | Warm decode mean | Warm TTFT | Result |
| --- | ---: | ---: | ---: | --- |
| Target-only | 11.54 tok/s | 18.25 tok/s | 1.82 s | exact and stable |
| MTP | 12.12 tok/s | 18.34 tok/s | 1.89 s | exact target parity, 10/11 accepted |

The 0.5% warm difference is too small to promote as a speedup. The useful result
is rejection-bearing exact parity at 59,750-token depth.

### Code was faster but not exact

| Mode | Cold decode | Warm decode mean | Warm TTFT | Result |
| --- | ---: | ---: | ---: | --- |
| Target-only | 12.55 tok/s | 18.15 tok/s | 2.89 s | valid and stable |
| MTP | 18.84 tok/s | 22.30 tok/s | 3.09 s | valid, but different text and tokens |

MTP was numerically 22.9% faster and accepted 117 of 119 drafts. We rejected
the speed claim because exact output and token parity failed in every repeat.

### Prose failed target-only instruction following

The target produced a stable 90-word response to a 170 to 210-word request.
All required facts were present, but the length constraint failed. MTP was not
run because the target-first gate did not pass.

## 10. Daily operating profile

The practical profile allocates 81,920 tokens, uses `n_cpu_moe=48`, a 37 GiB
working-set cap, Q4_0 target and draft KV, one persistent slot, and explicit
target-only or MTP startup modes. It keeps roughly 21,920 tokens after a 60K
state for subsequent prompts, tool output, and generation.

Use `profiles/q3ple_daily_80k.json` and
`scripts/q3ple_daily_profile.py`. MTP is never a silent retry after a partial or
incorrect generation. Correctness-sensitive work should launch target-only
before generation.

## 11. What to reproduce

The most useful independent checks are:

1. load all 33 target shards with the patched runtime;
2. reproduce the exact 16K fixture and its output hashes;
3. build a canonical staged state without replaying earlier prefixes;
4. verify paired target/`.dft` token identity after restart;
5. run rejection-bearing target-first comparisons;
6. report RAM, VRAM, RSS, pagefile growth, prompt occupancy, and failures.

Do not reduce this project to one tokens-per-second number. The negative results
are part of the release because they mark the boundary between a compelling
demo and a dependable runtime.

## Evidence map

- [Canonical 59,750-token report](community/Q3PLE-CANONICAL-60K-2026-08-31.md)
- [Hardware and safety floors](HARDWARE.md)
- [Benchmark protocol](BENCHMARK_PROTOCOL.md)
- [Daily profile](Q3PLE_DAILY_PROFILE.md)
- [Runtime patch series](../patches/llama.cpp/cafe-035e227-to-73b803/README.md)
- [Third-party provenance](../THIRD_PARTY_NOTICES.md)
- [Hugging Face model](https://huggingface.co/nmatteo3294/Qwen3.8-Flash-Next-Q3_PLE-MTP-GGUF)
