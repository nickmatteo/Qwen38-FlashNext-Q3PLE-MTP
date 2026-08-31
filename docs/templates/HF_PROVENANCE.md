# Provenance and compatibility

## Model lineage

- Official base: `Qwen/Qwen3.8-Flash-Next` at
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Immediate quant source: `AtomicChat/Qwen3.8-Flash-Next-GGUF` at
  `142262902a46f7daed19c79d0771534c8106ad59`.
- Project derivative: 33-shard `AD-4.27bpw-Q3_PLE-M64` target.
- Detached learned-MTP artifact:
  `mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf`.

AtomicChat published the immediate source quant. AtomicChat did not publish
this derivative.

## Target transformation

The conversion preserved the source shard split and copied all tensors except
`per_layer_token_embd.weight`. That table was streamed from Q5_1 to the project
Q3_PLE format. The conversion record identifies:

- converter SHA-256:
  `C10B228F612110FAB3021E307C759791D88493682AD83FC8D9F4BE5F0D7EC166`;
- runtime patch SHA-256:
  `AB0FB4EC59D7EE22EF9C5F490A49CBD3E581E8F9EF7E28331A1F9743DF2E61AB`;
- 1,223 unchanged tensor payloads;
- 56,114,183,680 unchanged payload bytes;
- exact payload equivalence for every unchanged tensor.

The release `target-shards.json` contains the independently checked size and
SHA-256 of every output shard.

## Runtime compatibility

Required source reconstruction:

- Cafe/llama.cpp base:
  `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`;
- project patch series head:
  `73b803464f25fc9054046728bf2ebed5a372737e`;
- patch series:
  https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/tree/main/patches/llama.cpp/cafe-035e227-to-73b803

The model is not stock llama.cpp compatible. Private Q3_PLE type 43, the tested
Qwen4Exp learned-MTP path, host-placement behavior, and paired `.dft` slot-state
persistence are part of the published patch series.

## License boundary

The GGUF target and sidecar contain Qwen-derived model tensors and are released
under Qwen Community License 1.0. Runtime source has separate MIT and applicable
Apache-2.0 notices. No code license relicenses the model tensors.
