# LinkedIn copy: September 2026 research refresh

## Project title

Q3_PLE: Qwen3.8-Flash-Next AD-4.27bpw on a 12 GB RTX 5070

## Project description

I built a hardware-specific Qwen3.8-Flash-Next runtime for one discrete RTX
5070 with 12 GB VRAM, 64 GB of DDR4-3000, a Ryzen 9 5900XT, and one NVMe drive.
No unified memory or second GPU.

The model has a 125B main MoE, a 51.2B n-gram/PLE table, and roughly 6B active
parameters per token. I started from AtomicChat's high-quality AD-4.27bpw mixed
GGUF, preserved the other 1,223 tensor payloads byte-for-byte, and replaced only
the giant PLE table with Q3_PLE, a custom 3.5-bit format.

The best repeated short fixture averaged 31.45 tok/s with exact text and token
hashes. The harder qualification retained 61,440 real tokens. Its 27-request
matrix covered three conditions, three independent processes per condition,
and three requests per process. Every response returned the same exact
512-token continuation. Median process decode was 19.276 tok/s target-only,
19.649 with corrected MTP, and 20.253 with corrected MTP plus paired CPU quant
kernels.

The final daily route is separate: it uses the promoted 2.20 GB MTP sidecar, a
65,536-token allocation, JIT loading with a 30-minute idle unload, and native
vision. Its matched warm text row measured 15.82 tok/s. A cold first request
after unload took 267.99 seconds. The projector stays on CPU and each image uses
exactly 1,024 context tokens.

The limits are published too: 31.45 is a 232-token-prompt result at a 16K
allocation; 20.253 is a separate occupied-context runtime with a 1.488 GB
shared-tensor sidecar; the 25 tok/s long-context goal remains open; and faster
MTP rows are rejected whenever exact output changes.

GitHub: https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP

## Post draft - do not publish without final approval

**20.253 tok/s with 61,440 actual retained tokens on a 12 GB RTX 5070.**

The interesting part is the combination: AtomicChat's AD-4.27bpw neural quant,
exact long-context output gates, learned MTP, native vision, and no unified
memory.

The system is a normal Windows desktop: one RTX 5070 12 GB, 64 GB DDR4-3000, a
Ryzen 9 5900XT, and NVMe. The GPU has discrete VRAM. System RAM and storage are
separate tiers.

I preserved 1,223 tensor payloads from AtomicChat's AD-4.27bpw GGUF byte for
byte and replaced only the model's enormous 51.2B-element n-gram/PLE table with
Q3_PLE, a custom 3.5-bit format. The 33-shard target is 78.5 GB. The daily
learned-MTP head remains a separately verified 2.20 GB sidecar. The occupied-60K
qualification used a different 1.488 GB shared-tensor sidecar, whose exact hash
and profile are in the repository.

Two results matter, and they are deliberately separate:

- **31.45 tok/s** mean across two repeated 232-token-prompt runs at a 16K
  allocation, with exact text and token hashes.
- **20.253 tok/s at 61,440 retained tokens**, summarized as the median of three
  process medians for the paired condition. The full three-condition matrix
  returned 27/27 exact 512-token outputs. Target-only was 19.276; corrected MTP
  was 19.649.

The separate daily 64K route accepts native images through its target and
promoted MTP process. Warm text measured 15.82 tok/s. It JIT-loads on first use
and releases the model after 30 idle minutes; a measured cold first request
took about 4 minutes 28 seconds.

As of September 14, 2026, from the public results I could find, I have not seen
another 12 GB discrete-GPU result report this decode speed while retaining a
genuine 60K-plus prefix and publishing hash-identical output gates around an
AD-4.27bpw-derived target. I am avoiding an absolute "first" claim, and I would
be glad to add a comparable result if someone has one.

The failures are part of the release. The 31.45 number is not a 60K result. The
25 tok/s occupied-context goal is still open. Some faster MTP rows changed the
continuation, so I rejected them.

Patch series, manifests, compact receipts, reproduction steps, and negative
results:
https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP

#LocalAI #LLM #Qwen #llamacpp #MachineLearning #NVIDIA #OpenSource

## Suggested post media

`docs/community/assets/q3ple-61440-native-vision-card.png` (1200 x 627)
