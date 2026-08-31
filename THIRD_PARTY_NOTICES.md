# Third-party notices and provenance

The root [LICENSE](LICENSE) applies only to original project code and
documentation unless a file says otherwise. It does not relicense third-party
material or model data.

## Qwen3.8-Flash-Next

The base model, configuration, parameters, weights, and tensors derived from
them are Qwen material. The pinned source is
`Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`.
That revision identifies the license as Qwen Community License 1.0. A copy is
included at [LICENSES/Qwen-Community-License-1.0.txt](LICENSES/Qwen-Community-License-1.0.txt).

The ignored local Q3_PLE GGUF and learned-MTP sidecars contain Qwen-derived
tensors and are not covered by the root MIT license. Any later model repository
must preserve Qwen's notice and conditions and document its exact base and
immediate source revisions.

## llama.cpp and Cafe runtime lineage

Runtime code, small source excerpts, and patches derived from llama.cpp or the
Cafe fork remain under the upstream MIT License. The retained measured runtime
is pinned in [docs/UPSTREAM_PINS.md](docs/UPSTREAM_PINS.md). The applicable
notice is preserved at [LICENSES/llama.cpp-MIT.txt](LICENSES/llama.cpp-MIT.txt).

Relevant sources include:

- `ggml-org/llama.cpp`;
- `quimmedes/cafe-llama.cpp@3fa3ab4faea0d496968a8ede1cdbb7cca21338fc`.

## AtomicChat source quant

The immediate source of the retained 33-shard target is
`AtomicChat/Qwen3.8-Flash-Next-GGUF@142262902a46f7daed19c79d0771534c8106ad59`.
The target is a transformed Qwen-derived model artifact. Attribution to
AtomicChat does not replace the Qwen Community License boundary.

## FreeToken-derived runtime lane and historical attribution

The FreeToken experiment and model payload are retired and are not included in
this repository. The exact measured llama.cpp patch series does, however,
retain a pinned-host MoE and elastic-cache commit attributed to FreeToken. That
patch remains under its applicable Apache License 2.0 terms, preserved at
[LICENSES/FreeToken-Apache-2.0.txt](LICENSES/FreeToken-Apache-2.0.txt), alongside
the MIT terms governing llama.cpp and Cafe-derived code. Historical documents
may retain additional attribution; they are not instructions to fetch or run
the retired FreeToken model lane.

## Benchmark data

Public benchmark task names and planned revisions are catalogued in the
benchmark manifest. Dataset contents are not redistributed by default. Each
future evidence bundle must record the exact task revision and upstream terms.
The local `independent_corpus_v2.txt` quant-regression corpus is excluded from
Git because it was assembled from llama.cpp source and documentation; only its
provenance manifest and hash may be committed.

This inventory is a technical provenance record, not legal advice.
