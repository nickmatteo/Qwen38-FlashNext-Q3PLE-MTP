# Hardware and topology

## Main Windows host

This is the only promoted benchmark machine.

- OS: Windows 11 Home, build family 26200.
- CPU: AMD Ryzen 9 5900XT, 16 cores / 32 threads.
- Physical memory: 68,640,825,344 bytes, marketed as 64 GB.
- Memory configuration: 4 x 16 GB DDR4, currently 3000 MT/s, mixed Corsair 3000C15 and 3600C18 kits.
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB reported, PCIe 4.0 x16 negotiated at x16.
- Project storage: EDILOCA EN870 1 TB NVMe mounted as Z:.
- System storage: Samsung 860 QVO SATA SSD mounted as C:.
- Project free space after the 2026-08-30 cleanup: 451.97 GiB.

The model, PLE table, and local benchmark artifacts stay on Z:. Any claimed SSD-backed PLE result mapped from C: or another volume is invalid for this hardware profile.

## Retained runtime profile

- llama.cpp branch: `cluster/q3ple-mtp`.
- commit: `4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc`.
- build: llama.cpp 10690, CUDA 13.1, SM120a, Flash Attention enabled.
- server SHA-256: `72BB9839C156ABBBA5D55B0CA3F2D7F89A931ACAA8A32BA40A8D76BBB4B67436`.
- target: 78,525,318,176-byte AtomicChat Q3_PLE derivative.
- promoted draft: 2,202,883,264-byte corrected FC/HC sidecar.

The canonical staged-state and actual-depth benchmarks use the slot-state
candidate at commit `73b803464f25fc9054046728bf2ebed5a372737e` and server
SHA-256 `49B51341F0E29B7AA6F73C1723F105E407AFB06EAB3BCFDCE6490F4985DB949E`.
That candidate adds paired target/`.dft` persistence on top of the frozen
performance runtime. The exact source reconstruction is published as a
36-commit patch series.

## Measured placement envelope

| Allocation | Target `n-cpu-moe` | Warm decode | Minimum available RAM | Minimum free VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 16K, repeat 1 | 45 | 31.50 t/s | 14.31 GiB | 780 MiB |
| 16K, repeat 2 | 45 | 31.40 t/s | 14.31 GiB | 776 MiB |
| 64K | 47 | 26.45 t/s | 12.82 GiB | 1,745 MiB |
| 80K | 47 | 25.81 t/s | 12.76 GiB | 1,569 MiB |
| 128K | 47 | 27.41 t/s | 12.74 GiB | 1,008 MiB |
| 192K | 48 | 26.65 t/s | 12.92 GiB | 1,365 MiB |
| 224K | 48 | 26.89 t/s | 13.10 GiB | 936 MiB |

All rows used a 232-token prompt. The RAM values are available-memory minima from those short-prompt allocation profiles, not predictions for filled contexts.

## Occupied-context history

The first monolithic 81,920-token allocation with 59,996 content tokens is a
preserved `FAILED` run:

- preflight available RAM: about 45.16 GiB;
- minimum available RAM: 6,442,377,216 bytes, just below the declared 6 GiB floor;
- minimum free VRAM: 1,126 MiB;
- maximum owned RSS: about 40.59 GiB;
- pagefile growth: zero;
- result: connection reset and no valid completion.

That failure pointed to main-memory residency, not VRAM, and ruled out cold
full-prefix replay as the operating design.

The replacement staged build reached a canonical complete-message boundary at
59,750 tokens. It sealed state at 15,872, 31,700, 47,303, and 59,750 tokens;
each later stage started a fresh process, restored the preceding target and MTP
pair, and processed only the unseen suffix. Its resource envelope was:

- minimum available RAM: 9,882,427,392 bytes, about 9.20 GiB;
- minimum free VRAM: 2,113 MiB;
- maximum owned RSS: 39,728,500,736 bytes, about 37.00 GiB;
- workflow-wide pagefile growth: zero;
- watchdog violations: none.

This is a stateful occupied-context result. It does not make arbitrary cold 60K
transcripts load instantly.

## Linux devbox

The retained secondary machine is an Arch-based Ryzen 5 3600 host with 32 GB
RAM, a GTX 1080 8 GB, and a direct 1 GbE link. Its private address is omitted
from the public package because that machine is outside the promoted path.

It remains outside the promoted path. Earlier matched topology evidence showed remote participation can add capacity while slowing warmed local inference. It must re-earn inclusion with identical-artifact, identical-output, end-to-end measurements. Aggregate RPC throughput is not a substitute for user-visible single-stream decode.

## Safety thresholds

Preflight for promoted runs:

- available RAM at least 40 GiB;
- free VRAM at least 8 GiB;
- GPU utilization at most 15%;
- benchmark port free;
- exact runtime and model identities verified.

Hard run stops:

- available RAM below 6 GiB;
- free VRAM below 768 MiB;
- owned RSS above 50 GiB;
- pagefile growth at least 1 GiB;
- CUDA, driver, server, output-corruption, or watchdog failure.

A daily profile needs at least 1,024 MiB minimum free VRAM. Rows below that can remain capacity evidence but are not recommended daily settings.
