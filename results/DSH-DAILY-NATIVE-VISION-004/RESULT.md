# 64K daily Q3_PLE + MTP native-vision result

## Decision

`MEASURED` - native image input is qualified for the same single-slot 64K
Q3_PLE + learned-MTP process used by the daily OpenAI-compatible route. The
projector stays on CPU, every image receives exactly 1,024 context tokens, and
the bounded text check showed no material steady-state decode change.

This is a discrete-memory Windows desktop: 12 GB of GPU VRAM, 64 GB of DDR4,
and an NVMe drive. It does not use a unified-memory accelerator or a second
GPU.

## Exact stack

- runtime: llama.cpp build 10696, commit
  `4cae4d05122198c65895ee714780f4426aa76684`;
- target: 33-shard AtomicChat-derived Q3_PLE M64, 78,525,318,176 bytes;
- learned-MTP sidecar: `DOWNQ4-FC-HC-OUTQ4`, 2,202,883,264 bytes;
- F16 vision projector: 904,003,520 bytes, SHA-256
  `AFB86DC1041DFC72C90C510F33FAB52838F01EF8E3C5732666EA44FA28C96ACA`;
- context: 65,536 tokens, one slot, non-unified KV;
- placement: projector on CPU, 48 target expert layers on CPU, dense target
  work and selected draft tensors on the RTX 5070;
- model working-set cap: 34 GiB;
- transport: localhost OpenAI-compatible `/v1/chat/completions` through a JIT
  gateway with a 30-minute model idle timeout.

The profile validates the runtime executable and adjacent DLL hashes, all 33
target shards, the MTP sidecar, and the projector before launch.

## Measured requests

The text and image requests used the same model process and MTP slot.

| Request | Prompt | Output | Prompt rate | Decode rate | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold text warm-up | 60 | 512 | 0.27 tok/s | 11.02 tok/s | 267.99 s |
| Warm text | 60 | 512 | 16.84 tok/s | 15.82 tok/s | 35.87 s |
| Native pagoda image | 1,054 | 119 | 14.01 tok/s | 11.31 tok/s | 85.95 s |

The cold row includes demand paging of the large mapped target and is not a
steady-state decode result. The preceding text-only daily log had 115 completed
decode rows with a 15.99 tok/s median. The matched vision-enabled warm row was
15.82 tok/s. This single matched comparison did not show a large decode-speed
regression; it is not a precise estimate of a general vision cost.

The image response returned:

```json
{"tier_count":5,"roof_color":"teal","frame_color":"red","entry_feature":"torii_gate"}
```

Those facts were absent from the text prompt. The test image itself visibly
contains the pagoda and its styling, and it labels the tier count. This is a
provider-path smoke test rather than an independent counting or broad vision
benchmark.

## Resource envelope

| Point | Process RSS | Available RAM | Used VRAM | Free VRAM |
| --- | ---: | ---: | ---: | ---: |
| Before cold text | 9.23 GiB | 33.06 GiB | 10,201 MiB | 1,742 MiB |
| After warm text | 34.00 GiB | 7.73 GiB | 10,234 MiB | 1,710 MiB |
| After image | 34.00 GiB | 7.63 GiB | 10,394 MiB | 1,550 MiB |

The image turn added about 160 MiB of observed VRAM over its immediate
pre-turn snapshot even with the projector weights on CPU. The final readings
remained above the 6 GiB available-RAM floor, 768 MiB emergency VRAM floor,
and 1,024 MiB daily VRAM reserve. Swap growth was about 20 MiB, below the 1 GiB
stop threshold.

## Full daily-gateway path

The same inline PNG was then sent to the route configured for DSH:
`http://127.0.0.1:18092/v1/chat/completions`. The gateway JIT-loaded the exact
target, MTP sidecar, and projector. A deliberately small 160-token cold request
hit its output ceiling after correctly identifying the image in hidden
reasoning, so that attempt remains `PARTIAL` and is not counted as a completed
client response.

The immediate 384-token retry returned HTTP 200 with `finish_reason: stop` in
8.25 seconds. It reused 1,020 of 1,024 image prompt tokens and decoded 119
tokens at 15.79 tok/s. This proves the normal provider/gateway/model path. It
does not independently test any particular browser attachment control.

## Published evidence

- `text-measured.json`: warm text response, timings, and resource readings;
- `image-measured.json`: uncached native image response and resource readings;
- `gateway-image-measured.json`: completed full-path gateway response;
- `profiles/q3ple_dsh_daily_64k_mtp.json`: exact paths, hashes, placement, and
  safety limits;
- `patches/llama.cpp/73b803-to-4cae4d/`: source delta from the previously
  published runtime to the measured daily head.

Machine-local logs, process manifests, private network names, and credentials
are intentionally excluded. Paths inside the three compact JSON receipts were
normalized for publication; response text, token counts, timings, resource
measurements, hashes, and system fingerprints were not changed.

The compact receipt did not freeze the gateway/controller Python source hash.
The published scripts are the reviewed reproduction implementation and include
later request-validation fixes. The completed measured request explicitly used
384 output tokens, which remains within the published profile's 1,024-token
completion cap. The HTTP receipt proves the measured route behavior; it does
not prove byte-identical gateway source.

## Limits

- The 31.45 tok/s headline elsewhere in this repository is a 232-token fixture
  in a 16K allocation and is not this daily route's long-context speed.
- The separately qualified 20.253 tok/s result at 61,440 retained tokens used
  an experimental paired-CPU-dot runtime and a separate 1,487,721,536-byte
  shared-tensor sidecar, not this native-vision daily build.
- Broad multimodal quality, independent reproduction, and arbitrary cold 60K
  prompt replay remain open.
- Learned MTP is still conditional: exact parity passes on the qualified
  occupied-context fixture, while other rejection-bearing workloads remain
  non-promotable.
