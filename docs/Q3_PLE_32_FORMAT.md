# Q3_PLE_32 Format Proposal

Status: **REFERENCE CODEC IMPLEMENTED — LLAMA.CPP INTEGRATION NOT IMPLEMENTED**

## Scope

`Q3_PLE_32` is a PLE/`GET_ROWS`-only tensor encoding for `per_layer_token_embd.weight`. It is not initially a general matmul quant and must not be accepted for arbitrary tensors.

## Block layout

One block represents 32 scalar values in exactly 14 bytes:

```text
bytes 0..1   BF16 scale d, little-endian
bytes 2..13  32 packed unsigned 3-bit codes, little-endian bitstream
```

Effective size: `14 × 8 / 32 = 3.5 bpw`.

Codes map linearly to signed integer quants:

```text
q = code - 4          # code 0..7 -> q -4..3
x_hat = fp32(d) * q
```

Packing is deterministic: element `i` begins at bit `3*i` of the 12-byte code stream; the low bit of a code is written first. Groups of eight elements occupy exactly three bytes. Implementations must test across byte boundaries, not rely on host bitfields or struct packing.

## Reference quantization

For each 32-value block:

1. Reject/record non-finite source values; production conversion must never silently encode them.
2. Initialize positive `d = max(-min(x)/4, max(x)/3)`. An all-zero block uses `d=0` and code 4 for every value.
3. Quantize `q_i = clamp(round_ties_to_even(x_i/d), -4, 3)`.
4. Optionally perform a fixed, deterministic number of least-squares refinements: `d = sum(x_i*q_i)/sum(q_i^2)`, then requantize. Keep `d >= 0` and stop on stable codes.
5. Round `d` to BF16, clamping a positive underflow to the smallest positive BF16 scale, then requantize once against the stored scale so the converter's error measurement matches runtime dequantization.

The original FP16-scale proposal is rejected. In `Q3-PLE-REAL-001`, five of 120 sampled blocks underflowed to a zero FP16 scale and one nonzero row became entirely zero. The same 16-bit BF16 scale preserved all sampled blocks with no size change. This is a measured numeric-format decision, not evidence of model quality.

The first implementation compared zero, one, and two refinement passes on sampled real PLE rows. Two BF16-scale refinement passes are the current provisional converter rule; the encoded format remains identical across pass counts, while the exact converter rule is pinned in the recipe.

## Shape compatibility

The GGUF PLE tensor has `ne[0] = 160`, exactly five 32-value blocks per row. No row padding or cross-row block is needed. The full table has 1,600,007,680 blocks and therefore exactly 22,400,107,520 payload bytes.

## Required implementation surface

- GGML type enum/name/traits with block size 32 and type size 14.
- Reference CPU quantize/dequantize functions.
- CPU `GET_ROWS` conversion to the graph's required output type.
- Quantizer allow-list restricted to exact `per_layer_token_embd.weight` or an explicit tensor regex.
- GGUF reader/writer round-trip and clear rejection on unsupported backends.
- No CUDA, RPC, or matmul kernel in the first correctness patch.

## Tests

- Compile-time `sizeof(block_q3_ple_32) == 14` and block-size assertions.
- Known byte vectors for packing/unpacking all codes and cross-byte boundaries.
- All-zero, extrema, monotonic, random-normal, random-uniform, and non-finite input cases.
- Quantize → serialize → load → `GET_ROWS` on repeated, boundary, and out-of-order row indices.
- Reference dequant equality and output SHA-256 determinism.
- Error metrics: MSE, max absolute error, cosine similarity, and row-norm drift.
- Regression: existing IQ4_NL PLE and standard non-PLE models unchanged.

## Promotion gate

This format becomes a real candidate only after tiny-fixture correctness, sampled-real-row error, and a full public-IQ3 neural-map `A0` load. Quality—not the 6.4 GB saving—decides whether it survives.
