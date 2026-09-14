# Qwen3.8-Flash-Next on a 12 GB RTX 5070

**Q3_PLE quantization · learned MTP · restartable long-context state · patched llama.cpp**

![Q3_PLE AD-4.27bpw occupied-context result on an RTX 5070 12 GB](docs/community/assets/q3ple-61440-native-vision-card.png)

This started with a stubborn question:

> Can Qwen's new Qwen4 architecture preview become genuinely useful on the consumer hardware people already own—not a 24 GB 3090 showcase, not a DDR5 workstation, and not a multi-GPU server?

On this machine, the answer became **yes, with clearly measured limits**.

The published configuration runs a 33-shard Q3_PLE target plus a detached learned-MTP sidecar on **one RTX 5070 12 GB, 64 GB DDR4-3000, a Ryzen 9 5900XT, and NVMe storage**. It reached **31.45 tok/s mean sustained decode** on a controlled short-prompt fixture and built a **59,750-token occupied chat state** that survives clean process restarts without replaying the earlier prefix.

The later occupied-context qualification retained **61,440 real tokens** and
returned the same exact 512-token continuation throughout a **27-request
matrix: three conditions, three independent processes per condition, and
three requests per process**. On that harder test, target-only measured 19.276
tok/s, corrected MTP measured 19.649 tok/s, and corrected MTP with paired CPU
quant kernels measured **20.253 tok/s**. The final daily route also runs native
vision through the same target and learned-MTP process, with a 65,536-token
allocation and an on-demand JIT lifecycle. Its matched warm text row measured
**15.82 tok/s**; after a full unload, the first cold request took **267.99 s**.

**[Read the technical breakdown →](docs/WALKTHROUGH.md)**  
[Hugging Face release page — public after shard verification](https://huggingface.co/nmatteo3294/Qwen3.8-Flash-Next-Q3_PLE-MTP-GGUF) · [Audit the canonical 59,750-token run](docs/community/Q3PLE-CANONICAL-60K-2026-08-31.md) · [Rebuild the exact occupied-60K runtime](patches/llama.cpp/73b803-to-c9d0-occupied60k/README.md)

> [!IMPORTANT]
> **31.45 tok/s and 59,750 occupied tokens are two different results.** The speed headline used a 232-token prompt inside a 16K allocation. The long-context result used a real 59,750-token saved state. This repository keeps them separate everywhere.

## Why this model was worth the grind

Qwen released [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next) on August 26, 2026 as an early preview of the architecture being developed for Qwen4. The official design combines a **125B-parameter main model**, a **51.2B-element n-gram embedding table**, and roughly **6B active parameters per token**.

Third-party leaderboard values for the base model have changed since release
and are not used as evidence for this derivative. **This project has not yet
established that the Q3_PLE target retains the base model's broad capability.**

The work here attacks a different problem: **how far can a frontier-ranked open-weight model be pushed on a small, imperfect, real desktop?**

## The result in plain English

| What was tested | What actually happened | Claim boundary |
| --- | --- | --- |
| Controlled throughput | Two clean 16K-allocation runs averaged **31.4495 tok/s** with a 232-token prompt and 171-token completion | Benchmark profile; only **776 MiB** free VRAM in the tighter run |
| Daily-capacity profile | An **81,920-token allocation** generated at **25.81 tok/s** with the same short prompt | Allocation capacity, **not** 80K occupied-context speed |
| Restartable agent state | A canonical chat state was sealed at **15,872 → 31,700 → 47,303 → 59,750 tokens** | Every later stage started fresh and processed only the unseen suffix |
| Actual-depth retrieval | At 59,750 tokens, target-only measured **18.25 tok/s** and MTP measured **18.34 tok/s** | Exact text/token parity; 10/11 drafts accepted; **no general speedup claimed** |
| Occupied-60K qualification | At 61,440 retained tokens, paired MTP measured **20.253 tok/s**, summarized as the median of three process medians | 27/27 exact 512-token outputs across three conditions; separate experimental runtime and sidecar |
| Daily native vision | The 64K route returned the requested pagoda fields through the normal JIT gateway | 15.82 tok/s matched warm text; 267.99 s cold first request; CPU projector; bounded labeled-image smoke test |
| Maximum safe allocation tested | **229,376 tokens** initialized and generated at 26.89 tok/s with a 232-token prompt | 262,144 was rejected by the VRAM safety floor |

## How it fits

![How Q3_PLE places Qwen3.8-Flash-Next across NVMe, DDR4, and 12 GB VRAM](docs/community/assets/q3ple-how-it-fits.svg)

The important move was **not** requantizing the entire model.

1. Start from AtomicChat's useful 4.27-bpw mixed quant.
2. Preserve the other **1,223 tensor payloads—56,114,183,680 bytes—byte for byte**.
3. Replace only `per_layer_token_embd.weight`, the giant 51.2B-element PLE/n-gram table, with the project Q3_PLE format.
4. Keep learned MTP detached so target and draft experiments remain independently verifiable. The daily route uses the promoted **2,202,883,264-byte** sidecar; the 20.253 tok/s qualification used a separate **1,487,721,536-byte shared-tensor sidecar** recorded in its exact profile.
5. Place latency-critical work on the GPU, routed work in host memory, and cold backing on NVMe through mmap.

The resulting target is **33 shards / 78,525,318,176 bytes**. The complete target plus promoted MTP sidecar is **80,728,201,440 bytes**, or about **75.18 GiB**.

This is therefore **a 12 GB GPU deployment**, not a claim that a 78.5 GB target somehow lives inside 12 GB of VRAM.

It also does not rely on unified memory. The RTX 5070 has its own 12 GB VRAM;
the machine has separate 64 GB DDR4 system memory, and the oversized PLE table
is mapped from NVMe. The result comes from explicit placement and paging across
three distinct storage tiers.

## The speedup that did *not* make the headline

![MTP evidence boundary at an actual 59,750-token context depth](docs/community/assets/q3ple-evidence-boundary.svg)

The project used one rule throughout the final gate:

> **A faster number does not count if exact output changes.**

At 59,750-token depth, the code fixture measured **22.30 tok/s with MTP versus 18.15 tok/s target-only**—a numerical increase of 22.9%. Both answers were usable, but exact text and retokenized-token identity changed in every repeat. The speedup was rejected.

That negative result is public on purpose. Learned MTP is working, but this quantized verifier path is still workload-dependent. Use target-only mode for correctness-sensitive work.

## Pick your path

| You want to… | Start here |
| --- | --- |
| Understand the complete engineering story | **[Technical walkthrough](docs/WALKTHROUGH.md)** |
| See the exact hardware, profiles, and safety floors | [RTX 5070 / 64 GB report](docs/community/Q3PLE-MTP-RTX5070-64GB.md) |
| Audit the real 59,750-token state | [Canonical 60K evidence report](docs/community/Q3PLE-CANONICAL-60K-2026-08-31.md) |
| Reproduce benchmarks | [Benchmark suite](benchmarks/README.md) |
| See the real Pi cache and restart benchmark | [20-turn coding-agent report](docs/community/Q3PLE-PI-AGENT-BENCHMARK-2026-09-02.md) |
| Audit the 27-request 61,440-token qualification | [Occupied-60K report](docs/community/Q3PLE-OCCUPIED-61440-2026-09-06.md) |
| Reproduce the daily native-vision route | [64K daily native-vision guide](docs/Q3PLE_DAILY_NATIVE_VISION.md) |
| Reconstruct the custom llama.cpp runtime | [Base 36-patch series](patches/llama.cpp/cafe-035e227-to-73b803/README.md) · [occupied-60K 24-patch delta](patches/llama.cpp/73b803-to-c9d0-occupied60k/README.md) |
| Inspect provenance and licensing | [Third-party notices](THIRD_PARTY_NOTICES.md) |
| Download and verify the GGUF release after publication | [Hugging Face model repository](https://huggingface.co/nmatteo3294/Qwen3.8-Flash-Next-Q3_PLE-MTP-GGUF) |

## Reproduction quick start

This is an experimental research release, not a stock `llama.cpp` one-liner.

### 1. Rebuild the measured runtime

```powershell
git clone https://github.com/quimmedes/cafe-llama.cpp.git
Set-Location cafe-llama.cpp
git checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
$basePatches = Get-ChildItem <research-repo>\patches\llama.cpp\cafe-035e227-to-73b803\*.patch | Sort-Object Name
git am --committer-date-is-author-date $basePatches.FullName
$dailyPatches = Get-ChildItem <research-repo>\patches\llama.cpp\73b803-to-4cae4d\*.patch | Sort-Object Name
git am --committer-date-is-author-date $dailyPatches.FullName
git rev-parse 'HEAD^{tree}'
```

The reconstructed daily source tree must be:

```text
95417c378aee171d51ac09808c16f0b33b0fbdc0
```

The measured source commit was `4cae4d05122198c65895ee714780f4426aa76684`.
A reconstructed commit ID can differ because email patches do not retain
original committer identity. The measured Windows build used CUDA 13.1 and
SM120a. A build made with another toolchain is a reproduction attempt, not the
byte-identical measured binary.

### 2. Download and verify all model files

A complete release contains:

- 33 target GGUF shards;
- 1 learned-MTP sidecar;
- `SHA256SUMS`;
- `target-shards.json`;
- `compatibility.json`;
- provenance and license files.

Do not treat a partial shard set as a usable model. Verify every file before launch.

### 3. Match the profile layout—or edit it deliberately

The supplied profile expects reconstructed runtime and model paths under the repository's ignored `workstreams/` and `artifacts/models/` directories. Either place them there or update [profiles/q3ple_dsh_daily_64k_mtp.json](profiles/q3ple_dsh_daily_64k_mtp.json) to your paths while preserving the hashes, placement, safety floors, native-vision projector, and one-slot contract.

```powershell
python scripts/q3ple_daily_profile.py --profile profiles/q3ple_dsh_daily_64k_mtp.json validate
python scripts/q3ple_daily_profile.py --profile profiles/q3ple_dsh_daily_64k_mtp.json launch --mode target
```

Experimental MTP mode must be selected explicitly:

```powershell
python scripts/q3ple_daily_profile.py --profile profiles/q3ple_dsh_daily_64k_mtp.json launch --mode mtp
```

Stateful requests require:

```json
{
  "id_slot": 0,
  "cache_prompt": true
}
```

The 20.253 tok/s occupied-context result is a separate research configuration.
Apply the [24-patch qualification delta](patches/llama.cpp/73b803-to-c9d0-occupied60k/README.md)
to reproduce source tree `7b3a784de1922d023ee5f040b19873903b26aeb4`, then use
[`profiles/q3ple_occupied_61440_pair.json`](profiles/q3ple_occupied_61440_pair.json).
That profile identifies the exact 1.488 GB shared-tensor sidecar, runtime bundle,
server arguments, and opt-in environment flags used by the measured pair row.

## What is proven

- The complete 33-shard Q3_PLE target loads and generates in the retained patched runtime.
- The promoted target and 2.20 GB learned-MTP sidecar run together on the daily route.
- The controlled fixture reached **31.45 tok/s mean** with exact preserved output and token hashes.
- Allocations through **229,376 tokens** initialized and generated above the watchdog floors.
- A canonical **59,750-token** target + `.dft` state pair survived fresh-process restores and continued without prefix replay.
- Retrieval at actual depth preserved exact target/MTP text and token parity across three repeats.
- The occupied-context matrix returned 27 exact 512-token outputs at 61,440 retained tokens and reached 20.253 tok/s with corrected MTP plus paired CPU quant kernels. Each condition used three independent processes.
- The pinned 64K daily route accepted native image input through its normal OpenAI-compatible JIT gateway and returned the requested visual fields from a labeled pagoda test image.

## What is *not* proven

- That this quant retains the official base model's full benchmark quality.
- That every 64K–224K allocation can be filled safely on this machine.
- That learned MTP is universally exact or faster.
- That stock llama.cpp can load private Q3_PLE type 43 or reproduce paired MTP state.
- That this runtime is production-ready, crash-atomic, or independently reproduced.
- That one pagoda smoke test establishes broad multimodal quality.
- That another published 12 GB result cannot exist; the performance-and-quality comparison reflects the public evidence found as of September 14, 2026.

## Hardware used

| Component | Measured system |
| --- | --- |
| GPU | NVIDIA RTX 5070, 12 GB |
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads |
| System memory | 64 GB DDR4-3000 |
| Storage | NVMe SSD |
| OS | Windows 11 |
| Runtime | Patched Cafe/llama.cpp lineage; daily native-vision head `4cae4d05122198c65895ee714780f4426aa76684` |

## Evidence language

- `MEASURED` — produced locally by a preserved command and raw record on identified hardware and artifacts.
- `PROXY` — an observed surrogate that is not the claimed end metric.
- `ANALYTICAL` — calculated from explicit inputs and assumptions.
- `EXTERNAL` — reported elsewhere and not locally reproduced.
- `VOID` — preserved run whose contamination or incompleteness prevents comparative use.

## Credits and license boundary

This project stands on work from **Qwen**, **AtomicChat**, **llama.cpp**, and **Cafe/Ark**. Historical pinned-host expert-streaming ideas also came from Cafe and FreeToken. The repository does not claim those components as original work.

Original project code and documentation are MIT-licensed under the scoped root [LICENSE](LICENSE). Qwen-derived model weights and GGUF tensors remain subject to the **Qwen Community License 1.0**. Upstream code retains its own notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact boundary.

---

**The headline is speed. The project is really about discipline:** fitting the model, surviving long state, preserving provenance, publishing the failures, and refusing to turn a changed answer into a benchmark win.
