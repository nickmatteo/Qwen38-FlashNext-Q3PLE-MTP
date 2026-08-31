---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - qwen
  - llama.cpp
  - gguf
  - speculative-decoding
  - mtp
  - long-context
---

# Qwen3.8-Flash-Next Q3_PLE + learned MTP GGUF

This is an experimental, hardware-oriented Qwen3.8-Flash-Next derivative for
patched llama.cpp. It combines a 78.5 GB, 33-shard Q3_PLE target with one
compatible 2.20 GB detached learned-MTP sidecar.

The controlled headline is **31.45 tok/s mean sustained decode** on an RTX 5070
12 GB plus 64 GB DDR4. That measurement used a 16K allocation, a 232-token
prompt, and a 171-token completion. It is not a 60K occupied-context speed.

The deeper result is a canonical 59,750-token chat state built and restored at
complete message boundaries without replaying the earlier 16K, 32K, or 47K
prefixes. Retrieval at that depth preserved exact target/MTP text and token
parity with 10 of 11 drafts accepted.

## Files

- `Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64-00001-of-00033.gguf` through
  `...-00033-of-00033.gguf`: target, 78,525,318,176 bytes total.
- `mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf`: promoted draft sidecar,
  2,202,883,264 bytes,
  SHA-256 `7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E`.
- `target-shards.json`: per-shard bytes and SHA-256.
- `compatibility.json`: source, runtime, and target/sidecar compatibility pins.
- `SHA256SUMS`: release-file checksums.

Total GGUF payload: 80,728,201,440 bytes, approximately 75.18 GiB.

## Source chain

- Official base:
  `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Immediate quant source:
  `AtomicChat/Qwen3.8-Flash-Next-GGUF@142262902a46f7daed19c79d0771534c8106ad59`.
- Runtime base:
  `quimmedes/cafe-llama.cpp@035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`.
- Required patched runtime head:
  `73b803464f25fc9054046728bf2ebed5a372737e`.
- Reproduction repository:
  https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP

AtomicChat published the immediate source quant. It did not publish this Q3_PLE
derivative. Q3_PLE conversion, sidecar selection, runtime patch packaging, and
benchmarking are documented in the reproduction repository.

## What changed

Only `per_layer_token_embd.weight`, the 51.2B-element PLE/n-gram table, was
replaced in the target:

- source tensor type: Q5_1;
- destination tensor type: project Q3_PLE, private GGML type 43;
- Q3_PLE block: 32 values, BF16 scale, 12 packed code bytes, 14 bytes total;
- destination PLE payload: 22,400,107,520 bytes;
- the other 1,223 tensor payloads, 56,114,183,680 bytes, were verified
  byte-identical to the immediate source.

The detached MTP sidecar uses the corrected FC/HC layout and a selective
down/output quantization recipe. Alternate experimental sidecars are not part
of this release.

## Required runtime

Stock llama.cpp cannot load private Q3_PLE type 43 or reproduce the paired MTP
slot-state behavior. Apply the exact 36-patch series from:

https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/tree/main/patches/llama.cpp/cafe-035e227-to-73b803

The measured Windows binary was built with CUDA 13.1 for SM120a. The public
repository contains the source patch series, build identity, profiles, harness,
and tests, but not prebuilt DLLs or executables.

## Tested hardware

- Windows 11;
- AMD Ryzen 9 5900XT, 16 cores / 32 threads;
- 64 GB DDR4-3000;
- NVIDIA RTX 5070 12 GB;
- NVMe storage.

This target is mmap-heavy and relies on host memory. A 12 GB GPU alone is not
sufficient. The tested machine uses both VRAM and tens of GiB of system RAM.

## Measured results

### Controlled short-prompt throughput

| Allocation | Actual prompt | Warm decode | Repeats | Exact output | Minimum free VRAM |
| ---: | ---: | ---: | ---: | --- | ---: |
| 16,384 | 232 | **31.4495 tok/s mean** | 2 | pass | 776 MiB worst case |

The two runs were 31.5021 and 31.3968 tok/s. Both generated 171 tokens and
accepted 129 of 129 drafts on this fixture. The VRAM margin is extremely tight,
so this is a benchmark profile.

### Short-prompt allocation capacity

| Allocated context | Actual prompt | Placement | Warm decode |
| ---: | ---: | --- | ---: |
| 65,536 | 232 | n47 | 26.45 tok/s |
| 81,920 | 232 | n47 | 25.81 tok/s |
| 131,072 | 232 | n47 | 27.41 tok/s |
| 196,608 | 232 | n48 | 26.65 tok/s |
| 229,376 | 232 | n48 | 26.89 tok/s |

These are allocation-capacity rows, not filled-context performance.

### Canonical 59,750-token state

The state was built at complete assistant-message boundaries of 15,872,
31,700, 47,303, and 59,750 tokens. Every later stage used a fresh process,
restored the previous target plus `.dft` pair, and processed only the unseen
suffix.

- minimum available RAM: about 9.20 GiB;
- minimum free VRAM: 2,113 MiB;
- peak owned RSS: about 37.00 GiB;
- workflow-wide pagefile growth: 0 bytes;
- watchdog violations: none.

At actual 59,750-token depth, three-repeat retrieval measured 18.25 tok/s warm
target-only and 18.34 tok/s warm with MTP. Text and retokenized tokens matched
exactly, and MTP accepted 10 of 11 drafts. The 0.5% difference is not presented
as a general speedup.

## Important negative results

- At 59,750-token depth, code measured 18.15 tok/s warm target-only and 22.30
  tok/s warm with MTP. Both answers were semantically valid, but exact text and
  token parity failed in every repeat. The 22.9% numerical difference is not a
  promoted speed claim.
- The target-only prose fixture returned 90 words for a requested 170 to 210
  words. MTP was not run because the target-first gate failed.
- Earlier cold filled-context attempts either crossed the 6 GiB available-RAM
  floor or projected roughly 28 to 33 minutes of replay. The operating design
  uses persistent paired state instead.

Learned MTP is workload-dependent on this quantized verifier path. Select
target-only mode before correctness-sensitive work. Do not use MTP as a silent
retry after partial output.

## Running the model

Clone the reproduction repository, apply the runtime patches, and place the
target shards plus sidecar where the supplied profile expects them. Then:

```powershell
python scripts/q3ple_daily_profile.py validate
python scripts/q3ple_daily_profile.py launch --mode mtp
python scripts/q3ple_daily_profile.py status
```

Explicit target-only operation:

```powershell
python scripts/q3ple_daily_profile.py launch --mode target
```

Generation requests for the stateful profile require one slot and prompt-cache
reuse:

```json
{
  "id_slot": 0,
  "cache_prompt": true
}
```

Read the full walkthrough and exact profile before running:

https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/blob/main/docs/WALKTHROUGH.md

## Intended use

- local inference research;
- Qwen4Exp/Q3_PLE runtime reproduction;
- learned-MTP verifier and state-persistence research;
- hardware-placement experiments on systems with constrained VRAM and large
  host memory.

This release is not a general-purpose drop-in Transformers checkpoint, not
stock llama.cpp compatible, and not validated for the vision path.

## License

The model data is subject to Qwen Community License 1.0, included as `LICENSE`.
The license includes conditions for certain high-scale commercial products and
for commercial Model-as-a-Service or AI Work Assistant use. It does not grant
trademark rights. Review the complete license before redistribution or
commercial deployment.

Runtime patches retain their separate upstream MIT and applicable Apache-2.0
notices in the reproduction repository. Those code licenses do not replace the
model-data license.

## Citation and attribution

Please credit Qwen, AtomicChat, llama.cpp, Cafe, and this research package when
reusing the artifacts or measurements. Reproduction reports are especially
useful when they include exact GPU, RAM, storage, runtime commit, command,
allocated context, actual prompt occupancy, hashes, and resource floors.
