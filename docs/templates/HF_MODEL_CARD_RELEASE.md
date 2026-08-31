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
  - qwen4_exp
  - llama.cpp
  - gguf
  - q3_ple
  - speculative-decoding
  - mtp
  - long-context
  - rtx-5070
---

# Qwen3.8-Flash-Next Q3_PLE + Learned MTP

**A 33-shard Q3_PLE target and matching learned-MTP sidecar built for constrained-VRAM, host-memory inference with patched llama.cpp.**

![Qwen3.8-Flash-Next on an RTX 5070 12 GB](https://raw.githubusercontent.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/main/docs/community/assets/q3ple-consumer-hardware-hero.svg)

> **Compatibility warning:** this is not a stock llama.cpp model. Private Q3_PLE type 43, the detached learned-MTP path, and paired target/`.dft` state persistence require the exact public patch series linked below.

## Why this release exists

[Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next) is Qwen's early preview of the architecture being developed for Qwen4: a 125B-parameter main model, a 51.2B-element n-gram embedding table, and roughly 6B active parameters per token.

This release asks a practical question: **can that class of model be made useful on one ordinary consumer desktop with only 12 GB of VRAM?**

The measured system used:

| Component | Hardware |
| --- | --- |
| GPU | NVIDIA RTX 5070, 12 GB |
| CPU | AMD Ryzen 9 5900XT, 16C / 32T |
| RAM | 64 GB DDR4-3000 |
| Storage | NVMe SSD |
| OS | Windows 11 |

The controlled headline is **31.45 tok/s mean sustained decode**. The deeper systems result is a **59,750-token occupied chat state** that was saved at complete message boundaries, restored in fresh processes, and extended without replaying the earlier prefix.

These are separate measurements.

## Release at a glance

| Item | Value |
| --- | ---: |
| Target | 33 GGUF shards |
| Target bytes | 78,525,318,176 |
| Promoted MTP sidecar | 1 GGUF |
| Sidecar bytes | 2,202,883,264 |
| Complete GGUF payload | 80,728,201,440 bytes |
| Approximate binary size | 75.18 GiB |
| Best controlled decode | 31.4495 tok/s mean |
| Canonical occupied state | 59,750 tokens |
| Required runtime head | `73b803464f25fc9054046728bf2ebed5a372737e` |

## Files you need

A complete download contains **34 GGUF files**:

- `Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64-00001-of-00033.gguf`
  through `...-00033-of-00033.gguf`;
- `mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf`.

It also contains:

- `SHA256SUMS`;
- `target-shards.json`;
- `compatibility.json`;
- `PROVENANCE.md`;
- `LICENSE`.

The promoted sidecar SHA-256 is:

```text
7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E
```

Do not attempt to launch from a partial shard set. Verify all file sizes and checksums first.

## What changed

Only one target tensor family was requantized:

```text
per_layer_token_embd.weight
```

That tensor is the model's 51.2B-element PLE/n-gram lookup table.

| Property | Value |
| --- | --- |
| Source tensor type | Q5_1 |
| Destination type | project Q3_PLE, private GGML type 43 |
| Q3_PLE block | 32 values |
| Scale | BF16 |
| Packed codes | 12 bytes |
| Total block size | 14 bytes |
| Destination PLE payload | 22,400,107,520 bytes |

The other **1,223 tensor payloads**, totaling **56,114,183,680 bytes**, were verified byte-identical to the immediate AtomicChat source.

![How the target is placed across NVMe, DDR4, and 12 GB VRAM](https://raw.githubusercontent.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/main/docs/community/assets/q3ple-how-it-fits.svg)

This is a host-memory deployment with aggressive placement. It is not a claim that a 78.5 GB target resides inside 12 GB of VRAM.

## Required runtime

The exact patch series is published here:

https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/tree/main/patches/llama.cpp/cafe-035e227-to-73b803

Reconstruct it from the pinned Cafe base:

```powershell
git clone https://github.com/quimmedes/cafe-llama.cpp.git
Set-Location cafe-llama.cpp
git checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
git am <research-repo>\patches\llama.cpp\cafe-035e227-to-73b803\*.patch
git rev-parse HEAD
```

Expected head:

```text
73b803464f25fc9054046728bf2ebed5a372737e
```

The measured Windows binary used CUDA 13.1 and SM120a. No prebuilt executable or DLL bundle is distributed in this model repository.

## Quick start

Clone the reproduction repository:

```powershell
git clone https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP.git
Set-Location Qwen38-FlashNext-Q3PLE-MTP
```

The supplied profile expects the reconstructed runtime under ignored `workstreams/` paths and the model files under ignored `artifacts/models/` paths. Either reproduce that layout or edit `profiles/q3ple_daily_80k.json` deliberately while retaining the artifact hashes, placement, resource floors, and one-slot contract.

Validate before launch:

```powershell
python scripts/q3ple_daily_profile.py validate
```

Start in explicit target-only mode for correctness-sensitive work:

```powershell
python scripts/q3ple_daily_profile.py launch --mode target
```

Start experimental MTP mode only when you intend to evaluate it:

```powershell
python scripts/q3ple_daily_profile.py launch --mode mtp
```

Check status:

```powershell
python scripts/q3ple_daily_profile.py status
```

Stateful requests require one slot and prompt-cache reuse:

```json
{
  "id_slot": 0,
  "cache_prompt": true
}
```

## Measured results

### Controlled short-prompt throughput

Two clean runs used:

- 16,384-token allocation;
- 232-token actual prompt;
- 171-token completion;
- Q4_0 target KV;
- target placement `n_cpu_moe=45`;
- MTP `nmax=3`, `p_min=0.75`;
- pinned-host routed draft experts.

| Run | Warm decode | Draft acceptance | Minimum free VRAM |
| ---: | ---: | ---: | ---: |
| 1 | 31.5021 tok/s | 129 / 129 | 780 MiB |
| 2 | 31.3968 tok/s | 129 / 129 | 776 MiB |
| **Mean** | **31.4495 tok/s** | **100% on this fixture** | **776 MiB worst case** |

Both runs reproduced the preserved expected text and token hashes. The margin is extremely tight, so this is a benchmark profile—not the recommended daily configuration.

### Short-prompt allocation capacity

Every row below used the same 232-token prompt:

| Allocated context | Placement | Warm decode |
| ---: | --- | ---: |
| 65,536 | n47 | 26.45 tok/s |
| 81,920 | n47 | 25.81 tok/s |
| 131,072 | n47 | 27.41 tok/s |
| 196,608 | n48 | 26.65 tok/s |
| 229,376 | n48 | 26.89 tok/s |

These rows show allocation capacity and short-prompt generation. They do **not** show filled 64K–224K throughput. A 262,144-token allocation was rejected by the 768 MiB free-VRAM safety floor.

### Canonical 59,750-token state

The state was built only at complete assistant-message boundaries:

```text
15,872 → 31,700 → 47,303 → 59,750 tokens
```

Each later stage:

1. started a fresh process;
2. restored the prior target state plus its `.dft` companion;
3. verified target/draft token-vector identity;
4. processed only the unseen suffix;
5. sealed the next complete-message boundary.

Recorded resource minima:

| Metric | Result |
| --- | ---: |
| Minimum available RAM | about 9.20 GiB |
| Minimum free VRAM | 2,113 MiB |
| Peak owned RSS | about 37.00 GiB |
| Workflow-wide pagefile growth | 0 bytes |
| Watchdog violations | none |

### Actual-depth target versus MTP

![Exactness boundary for retrieval, code, and prose at 59,750 tokens](https://raw.githubusercontent.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/main/docs/community/assets/q3ple-evidence-boundary.svg)

| Workload | Target-only | MTP | Verdict |
| --- | ---: | ---: | --- |
| Retrieval | 18.25 tok/s warm | 18.34 tok/s warm | Exact text/token parity; 10/11 drafts accepted; no general speed claim |
| Code | 18.15 tok/s warm | 22.30 tok/s warm | Numerical +22.9%, but exact output changed; speedup rejected |
| Prose | Target returned 90 words for a requested 170–210 | Not run | Target-first correctness gate failed |

Learned MTP is functional and restartable, but the current low-bit verifier path remains workload-dependent. **Select target-only mode before correctness-sensitive generation. Never use MTP as a silent retry after partial output.**

## What this release proves

- The 33-shard Q3_PLE target loads and generates with the published runtime lineage.
- The target runs with the promoted 2.20 GB learned-MTP sidecar.
- The controlled short-prompt fixture reaches 31.45 tok/s mean with preserved output and token hashes.
- Context allocations through 229,376 initialize and generate above the resource floors.
- Paired target and MTP state can survive fresh-process restarts.
- A 59,750-token canonical state can continue without replaying previously checkpointed prefixes.
- Retrieval at actual depth can preserve exact target/MTP text and token parity.

## What this release does not prove

- Broad capability retention versus the official base checkpoint.
- Safe filled-context operation across every large allocation.
- Universal MTP exactness or acceleration.
- Stock llama.cpp compatibility.
- Production readiness or crash-atomic two-file state.
- Independent reproduction.
- Vision-path compatibility.

## Source and provenance

- Official base: `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Immediate quant source: `AtomicChat/Qwen3.8-Flash-Next-GGUF@142262902a46f7daed19c79d0771534c8106ad59`
- Runtime base: `quimmedes/cafe-llama.cpp@035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`
- Required patched runtime: `73b803464f25fc9054046728bf2ebed5a372737e`
- Reproduction repository: https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP

AtomicChat published the immediate source quant. It did not publish this Q3_PLE derivative. Q3_PLE conversion, sidecar selection, runtime patch packaging, state-persistence work, and benchmarking are documented in the reproduction repository.

## Intended use

- local inference research;
- constrained-VRAM and host-memory placement experiments;
- Qwen4Exp / Q3_PLE runtime reproduction;
- learned-MTP verifier research;
- paired target/draft state-persistence research;
- independent benchmark reproduction.

This is not a drop-in Transformers checkpoint and is not validated for vision input.

## License

The model data is distributed under **Qwen Community License 1.0**, included as `LICENSE`. Review the complete license before redistribution or commercial deployment, including its conditions for certain high-scale products, Model-as-a-Service, and AI Work Assistant use.

Runtime patches retain their applicable upstream MIT and Apache-2.0 notices in the reproduction repository. Those code licenses do not replace the model-data license.

## Citation and attribution

Please credit Qwen, AtomicChat, llama.cpp, Cafe, and this research package when reusing the artifacts or measurements.

The most useful reproduction reports include:

- exact GPU, CPU, RAM, and storage;
- target and sidecar hashes;
- runtime commit and build toolchain;
- allocated context and actual prompt occupancy;
- commands and sampling settings;
- RAM, VRAM, RSS, and pagefile floors;
- exact output/token comparison;
- failed and rejected runs, not only the best number.

Full technical walkthrough:

https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/blob/main/docs/WALKTHROUGH.md
