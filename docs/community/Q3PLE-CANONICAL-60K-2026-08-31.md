# Q3_PLE canonical 59,750-token benchmark

Date: 2026-08-31

Status: measured locally and authorized for external publication on 2026-08-31

The sealed JSON files retain `publishable: false` because publication had not
been authorized when those immutable runs were captured. That field records the
capture-time authorization state, not a failed measurement gate. The user
authorized external publication after the final review; the raw files remain
unchanged so their published SHA-256 values continue to verify.

Share card: [PNG](assets/q3ple-canonical-59750-card.png) | [editable SVG](assets/q3ple-canonical-59750-card.svg)

## What passed

One canonical chat transcript was built to a complete assistant-message boundary at 59,750 occupied tokens inside an 81,920-token allocation. The build used four sealed boundaries:

| Stage | Saved tokens | Fresh-process restore proof |
|---:|---:|---:|
| 1 | 15,872 | Initial build |
| 2 | 31,700 | `cache_n=15,872`, `prompt_n=512` |
| 3 | 47,303 | `cache_n=31,700`, `prompt_n=512` |
| 4 | 59,750 | `cache_n=47,303`, `prompt_n=512` |

Every boundary ended after a complete assistant message. Every later stage restored the preceding target and MTP state pair in a new process and processed only the unseen suffix. The final target and `.dft` files contained the same 59,750-token vector.

Build safety summary:

| Measurement | Result |
|---|---:|
| Minimum available RAM | 9,882,427,392 bytes, about 9.20 GiB |
| Minimum free VRAM | 2,113 MiB |
| Peak owned RSS | 39,728,500,736 bytes, about 37.00 GiB |
| Workflow-wide pagefile growth | 0 bytes |
| Watchdog violations | None |

The sealed build result is `results/QWEN38-MTP-PROTOTYPE-001/q3ple_canonical_60k_publishable_20260831_v7_staged_r1.json`.

SHA-256: `E78C00B662198C4E6ED1D1D47DA761C56923DCBF6764C6C86FADF66AE878DAC6`

## Actual-depth retrieval

The retrieval fixture used the same 59,750-token state. Target-only ran first. MTP then ran from a copied pair with identical hashes. Each lane used three repeats.

| Mode | Cold decode | Warm decode mean | Warm TTFT mean | Result |
|---|---:|---:|---:|---|
| Target-only | 11.54 tok/s | 18.25 tok/s | 1.82 s | Exact retrieval, stable output |
| MTP | 12.12 tok/s | 18.34 tok/s | 1.89 s | Exact target parity, 10/11 drafts accepted |

MTP was approximately 0.5% faster on the two warm decode repeats. That difference is too small to present as a meaningful general speedup. The important result is exact target/MTP text and token parity with a real rejection at 59,750-token depth.

The retrieval result is `results/QWEN38-MTP-PROTOTYPE-001/q3ple_canonical_60k_canonical_retrieval_ab_20260831_v1_r1.json`.

SHA-256: `04C4749A5674F8AA531B0DFA0D2AAC839BF3D40E1A33EB9B3D655B4D575AEBC2`

## Actual-depth code

The target-only implementation was valid and byte-stable across all three repeats. The original scorer falsely rejected `tempfile.mkstemp(dir=str(directory))` even though `directory` was assigned from `path.parent`. The scorer was corrected to follow that local alias, covered by a regression test, and the code fixture was rerun from the same sealed state.

| Mode | Cold decode | Warm decode mean | Warm TTFT mean | Result |
|---|---:|---:|---:|---|
| Target-only | 12.55 tok/s | 18.15 tok/s | 2.89 s | Valid and stable |
| MTP | 18.84 tok/s | 22.30 tok/s | 3.09 s | Semantically valid, but exact text and tokens differ |

MTP was numerically 22.9% faster on the two warm repeats and accepted 117 of 119 drafted tokens. This is not a promotable speed result because MTP changed the exact output and retokenized token vector in every repeat. It remains useful negative evidence for the verifier-numerics problem.

The code result is `results/QWEN38-MTP-PROTOTYPE-001/q3ple_canonical_60k_canonical_code_ab_20260831_v1_r1.json`.

SHA-256: `F0CB3572626752FAAD46BA0383B7DE40F7252D331A3B19C1228C08E817A98928`

## Prose target failure

The broad three-fixture target-first run remains blocked. Retrieval passed. The code output was semantically correct but was rejected by the old scorer described above. The prose output contained every required factual element and was stable, but it was 90 words instead of the requested 170 to 210 words. That is a real target-only instruction-following failure, so MTP was not run for prose.

The original broad result remains unchanged at `results/QWEN38-MTP-PROTOTYPE-001/q3ple_canonical_60k_canonical_ab_20260831_v1_r1.json`.

SHA-256: `2F352780DCED0332061E92A598272CCD96F905340BF996BB0A35A0DAECD6C5F3`

## Claim boundaries

These measurements support:

- A canonical, complete-message, 59,750-token state can be built incrementally and restored across fresh processes without replaying prior stages.
- Actual-depth retrieval can preserve exact target/MTP text and token parity while exercising rejection-bearing MTP.
- Target-only code generation can remain stable and valid at this depth.
- MTP performance and exactness remain workload-dependent.

They do not support:

- 31.45 tok/s at 59,750-token occupied depth. The 31.45 tok/s headline belongs to the separate repeated 16K-allocation fixture.
- A general MTP speedup.
- General long-context intelligence or agent competence.
- Instant loading of an arbitrary transcript that was not previously checkpointed.
- Crash-atomic or power-loss-safe two-file persistence.

Screenshots are corroborating evidence only. The structured JSON, hashes, token accounting, runtime identity, and telemetry remain authoritative.
