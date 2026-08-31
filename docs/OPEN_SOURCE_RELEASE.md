# Open-source and Hugging Face release decision

Checked against primary sources on 2026-08-30.

## Short answer

Yes, this work can be opened in a useful way.

The best structure is:

1. A small standalone Git repository for reproducibility, evidence, scripts, patches, and limitations.
2. A separate Hugging Face model repository for the Q3_PLE target and learned-MTP sidecar after the release gates pass.
3. A narrow upstream llama.cpp contribution only if the generic runtime work is still needed after the existing MTP draft PR evolves.

Do not make a wholesale llama.cpp fork the project's main public identity. A fork is useful as an implementation vehicle, but it buries the research under a very large codebase and creates a permanent synchronization burden.

## Why a standalone repository first

The novel and reproducible unit is larger than one llama.cpp patch:

- the exact Q3_PLE substitution and byte-equivalence proof;
- detached MTP conversion and FC/HC correction;
- sidecar quantization and placement choices;
- the hardware profile and resource watchdog;
- target-only versus MTP correctness policy;
- allocated-context versus occupied-context reporting;
- raw positive and negative results.

A small repository can explain those pieces without pretending that every local hardware optimization belongs upstream.

Recommended public tree:

```text
qwen38-q3ple-mtp/
  README.md
  DECISION.md
  LICENSES/
    PROJECT-LICENSE
    THIRD-PARTY-NOTICES.md
  benchmarks/
    manifest.json
    result.schema.json
    fixtures/
  docs/
    HARDWARE.md
    BENCHMARK_PROTOCOL.md
    MODEL_CARD.md
    LIMITATIONS.md
  patches/
    llama.cpp/<base-commit>/*.patch
  results/
    public-summary.json
    selected-raw-json/
  scripts/
    conversion/
    benchmark.py
    validation/
  manifests/
    target-shards.json
    mtp-sidecar.json
    source-pins.json
```

Keep these out of Git:

- local GGUF payloads;
- build trees;
- full llama.cpp worktrees;
- caches and temporary files;
- raw logs containing unnecessary machine paths;
- benchmark datasets whose terms do not permit redistribution.

## Should this be a llama.cpp fork?

Not as the main project.

Use an implementation fork or branch only for code that must live inside llama.cpp. Publish small patch files against exact upstream revisions in the research repository. That keeps the scientific package readable while making the runtime delta auditable.

Base Qwen3.8-Flash-Next support is already merged in [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742). Qwen4Exp learned-MTP support is already represented by [draft PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836). As of this check, that PR is still a three-commit draft, and stock [`conversion/qwen4exp.py`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/conversion/qwen4exp.py) still sets `supports_mtp_export = False` and `no_mtp = True`.

Opening a duplicate MTP PR would waste maintainer time. The correct upstream path is to understand the delta, test against the current draft, and work with that thread if a generic fix remains.

Hardware-specific choices such as n45/n47/n48 placement, Windows pinned-host behavior, or a local VRAM-fit heuristic belong in the research package unless upstream maintainers ask for them.

## Upstream contribution rules

llama.cpp's current [contribution guide](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CONTRIBUTING.md) says to search existing work, begin features with an issue, run full local CI, and verify perplexity and performance with `llama-perplexity` and `llama-bench`. New quant types have a higher bar, including a small public GGUF, perplexity against native precision and similar-size types, KL divergence, and CPU performance comparisons.

Its [agent policy](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/AGENTS.md) allows assisted code under human ownership but prohibits automated submissions and AI-written PR descriptions, commit messages, or reviewer replies. The human contributor must understand, explain, and maintain every submitted line.

Therefore this project will not auto-create a llama.cpp PR or write the human's upstream conversation. It can prepare evidence and a reviewable local patch.

## Can the model be published on Hugging Face?

Technically, yes, subject to the license and release gates.

The official pinned checkpoint uses [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE). Its grant expressly includes model weights, parameters, configuration, inference code, documentation, publication, distribution, and derivative works.

That grant has conditions:

- include Qwen's copyright and permission notice in copies or substantial portions;
- prominently display the model name in qualifying commercial products or services above 100 million monthly active users or US$20 million monthly revenue;
- obtain a separate Qwen license before commercial Model-as-a-Service or AI Work Assistant use, except the license's stated internal-use case with no third-party access;
- comply with law and third-party intellectual-property rights;
- accept the license's no-warranty terms.

The Q3_PLE target and MTP sidecar contain Qwen-derived tensors. Treat them as Qwen Community License 1.0 model data. Do not relabel them Apache-2.0 or MIT merely because the converter or runtime code uses those licenses.

The immediate target source, [AtomicChat/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF), also identifies Qwen Community License 1.0. The model card must name both the official Qwen base revision and the exact AtomicChat source revision.

The community MTP repository currently displays `apache-2.0` metadata on [its Hugging Face page](https://huggingface.co/quimmedes/Qwen3.8-Flash-Next-MTP-GGUF/tree/main). That is a provenance warning, not safe precedent for Qwen-derived tensor data. Our release should correct the classification rather than copy it.

This is a technical license reading, not a substitute for legal advice where commercial redistribution or trademark use carries material risk.

## Hugging Face repository shape

Hugging Face supports arbitrary custom model files, not only Transformers checkpoints, and recommends a model card for the repository. See [Uploading models](https://huggingface.co/docs/hub/models-uploading).

The Hub's [model card documentation](https://huggingface.co/docs/hub/en/model-cards) supports explicit `base_model`, `base_model_relation: quantized`, and custom license metadata. Its [license guidance](https://huggingface.co/docs/hub/repositories-licenses) also says repository authors must respect the source project's license.

Recommended metadata:

```yaml
---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
tags:
  - qwen
  - llama.cpp
  - gguf
  - speculative-decoding
  - mtp
---
```

Required card content:

- official base: `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`;
- immediate source: `AtomicChat/Qwen3.8-Flash-Next-GGUF@142262902a46f7daed19c79d0771534c8106ad59`;
- target conversion tool and patch hashes;
- exact tensor changed, old and new types, shape, and byte delta;
- all 33 target shard hashes and aggregate size;
- every sidecar filename, recipe, bytes, and SHA-256;
- required llama.cpp branch or patch revision;
- minimum and recommended RAM, VRAM, storage, and context profiles;
- target-only and MTP commands;
- measured results with actual prompt occupancy;
- failed occupied-context and prose-parity results;
- intended use and limitations;
- Qwen license text and attribution;
- AtomicChat, llama.cpp, Cafe/Ark, and other material provenance.

## One model repository or two?

Use one Hugging Face repository for the target shards and compatible sidecars if the total repository remains understandable. The model card should make it impossible to confuse the target with a draft sidecar.

Split into two repositories only if:

- sidecars need independent release cadence;
- multiple targets become compatible with the same draft family;
- Hub clients mis-detect or download the wrong files;
- license or provenance differs materially.

The current recommendation is one repository with clear filename families and a machine-readable compatibility manifest.

## Project code licensing

Original project scripts and documentation use the scoped MIT license in the repository root. The accompanying third-party notice and license inventory explicitly exclude Qwen model material, third-party source, and derived patches from that grant.

Do not place one blanket MIT or Apache license over the whole repository if it contains third-party patches, model metadata copied from Qwen, or weights.

File-level treatment:

| Material | Treatment |
| --- | --- |
| Original benchmark and manifest tools | scoped project MIT license |
| llama.cpp or Cafe source/patch-derived lines | retain MIT copyright and permission notices |
| Qwen config, weights, tensors, and derived GGUF | Qwen Community License 1.0 |
| AtomicChat source-model lineage | attribute exact repository and revision; model data remains Qwen-licensed |
| Historical FreeToken ideas named in provenance | attribution only; no active FreeToken code is intended for release |
| Benchmark datasets | retain upstream revision/license references; do not redistribute task contents by default |

## Publication gates

The standalone research repository is ready to publish only after:

- active documentation contains no dead-path instructions;
- project validation and benchmark-contract tests pass;
- public files are scrubbed for credentials and unnecessary absolute paths;
- every source and patch file has provenance classification;
- selected raw evidence is sufficient to reproduce every headline number;
- limitations lead the public summary rather than being buried.

The Hugging Face model publication gate is satisfied for an explicitly
experimental release because:

- the canonical staged build reached a complete 59,750-token boundary under the
  watchdog without replaying prior stages;
- retrieval passed exact rejection-bearing target/MTP parity at actual depth;
- code and prose failures are disclosed and the sidecar is labeled
  experimental with no general speed claim;
- the target and promoted sidecar manifests are complete;
- the full Qwen license and provenance notices are staged;
- the user explicitly authorized the approximately 80.7 GB GGUF upload on
  2026-08-31.

Broad capability evaluation and independent reproduction remain open. They are
release limitations, not hidden claims.

## Recommended release language

Good:

> A hardware-specific Qwen3.8-Flash-Next GGUF experiment for an RTX 5070 12 GB plus 64 GB RAM system. The best repeated 16K short-prompt profile measured 31.45 t/s. Allocations through 224K generated under watchdog, but those rows used a 232-token prompt. A 60K occupied-context attempt failed the 6 GiB RAM floor. Learned MTP is currently workload-dependent and experimental.

Do not publish:

> Qwen3.8 runs at 27 t/s with 224K context on a 12 GB GPU.

That sentence hides prompt occupancy and would overstate the evidence.

## Current decision

- Standalone research repository: authorized for a clean-history public release.
- Hugging Face target and promoted sidecar: authorized as an experimental release under the Qwen license boundary.
- Wholesale llama.cpp fork as the public project: not recommended.
- Narrow upstream work: possible later, coordinated with existing PR #27836 and written/submitted by the human contributor.

Provider URLs and immutable release identities are recorded after the upload is verified.
