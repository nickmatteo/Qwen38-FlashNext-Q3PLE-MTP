# Upstream pins

Refreshed: 2026-08-30

Moving branch names are never sufficient for a measured or released result. Local measurements remain attached to their exact historical revisions even when upstream moves.

## Active pins

| Component | Repository or source | Revision | Project role | Status |
| --- | --- | --- | --- | --- |
| Official model and license | `Qwen/Qwen3.8-Flash-Next` | `f5d08274bafd880402bd16f5e3e6c514136ec06c` | base model, config, license | PINNED |
| Official architecture repository | `QwenLM/Qwen3.8-Flash-Next` | `513aa6e18a335296fc13e538232a8735b230877d` | architecture and documentation reference | PINNED |
| Immediate target quant source | `AtomicChat/Qwen3.8-Flash-Next-GGUF` | `142262902a46f7daed19c79d0771534c8106ad59` | 4.27-bpw, 33-shard source | PINNED |
| Historical A0 source | `unsloth/Qwen3.8-Flash-Next-GGUF` | `8bdc666649440e9bdc97e16f3f75782c98478ff5` | UD-IQ3_XXS control | PINNED HISTORICAL |
| Frozen measured runtime | local `cluster/q3ple-mtp` | `4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc` | promoted benchmark baseline | PINNED MEASURED |
| Frozen server executable | local build 10690 | SHA-256 `72BB9839C156ABBBA5D55B0CA3F2D7F89A931ACAA8A32BA40A8D76BBB4B67436` | exact measured binary | PINNED MEASURED |
| Upstream Qwen4Exp support | llama.cpp PR #27742 | merged 2026-08-27 | base architecture support | MERGED, NOT THE MEASURED BUILD |
| Upstream Qwen4Exp MTP draft | llama.cpp PR #27836 | `1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af` | generic MTP reference | DRAFT |
| Cafe runtime source used in local lineage | `quimmedes/cafe-llama.cpp` | `3fa3ab4faea0d496968a8ede1cdbb7cca21338fc` | MTP and shared-memory reference | PINNED FOR PROVENANCE |
| Community MTP sidecar source | `quimmedes/Qwen3.8-Flash-Next-MTP-GGUF` | `fb84e51` | original detached sidecar lineage | PINNED FOR PROVENANCE |
| llama.cpp current `master` snapshot | `ggml-org/llama.cpp` | `9723942adc518b43c4b95dc4dce6906903eb5e09` | current-state comparison only | REFRESHED 2026-08-30 |

The Qwen and AtomicChat repositories have moved since some of these snapshots. That does not change the base of the local artifact.

## Verified model facts at the pinned official revision

- model type: `qwen4_exp`;
- 48 text layers;
- hidden size 2,560;
- 512 routed experts, top 10 per token;
- repeating schedule of three linear-attention layers and one full/QSA layer;
- 51.2B-element PLE table split across 128 source parts;
- maximum position embeddings: 262,144;
- one learned MTP / NextN block in the official checkpoint;
- multimodal base with a 27-layer vision encoder.

The retained project artifact is text-only. Do not imply that the local Q3_PLE target or benchmark package validates the vision path.

## Current upstream state

Base Qwen3.8-Flash-Next support was merged into llama.cpp in PR #27742 on 2026-08-27.

Qwen4Exp MTP is not merged at this snapshot:

- PR #27836 is still `Draft` with three commits;
- its converter path adds `--mtp` at the PR head;
- current stock `conversion/qwen4exp.py` still declares `supports_mtp_export = False` and `no_mtp = True`.

The project should track and collaborate with the existing MTP work rather than open a duplicate feature PR.

## Provenance boundary

- Qwen-derived weights, parameters, config, and GGUF tensors: Qwen Community License 1.0.
- llama.cpp and Cafe-derived code: MIT, with the original notice retained.
- project-original scripts and documentation: scoped MIT license in the repository root.
- no active project file depends on the retired FreeToken codebase.

Historical documents may name older or removed source pins. They are evidence snapshots, not active dependency instructions.
