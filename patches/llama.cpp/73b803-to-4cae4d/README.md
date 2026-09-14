# Daily-runtime delta: `73b803` to `4cae4d`

These four commits extend the published 36-commit Cafe-to-`73b803` series to
the exact Windows runtime used for the measured 64K daily native-vision route.

## Identity

- prerequisite head: `73b803464f25fc9054046728bf2ebed5a372737e`;
- measured daily head: `4cae4d05122198c65895ee714780f4426aa76684`;
- llama.cpp fingerprint: build 10696 / commit `4cae4d051`;
- patch count in this delta: 4.

The delta adds the Windows direct PLE reader, incorporates the shared-KV
draft-MTP fix, separates the lazy reader's source view, and fixes overlapped
I/O plus Unicode-safe Windows path handling. Native multimodal support comes
from the underlying Qwen3.8/Cafe runtime; the measured server also loads the
pinned F16 projector described by the daily profile.

## Apply

First reconstruct `73b803` with the [base 36-commit series](../cafe-035e227-to-73b803/README.md).
Then apply this directory in commit order:

```powershell
$patches = Get-ChildItem <research-repo>\patches\llama.cpp\73b803-to-4cae4d\*.patch | Sort-Object Name
git am --committer-date-is-author-date $patches.FullName
git rev-parse 'HEAD^{tree}'
```

The resulting tree must be
`95417c378aee171d51ac09808c16f0b33b0fbdc0`. The measured source commit was
`4cae4d05122198c65895ee714780f4426aa76684`; a reconstructed commit ID may
differ because email patches do not retain original committer identity.
Rebuilding the same source with a different compiler, CUDA toolkit, driver, or
GPU is a reproduction attempt and does not produce the identical measured
binary.

## Provenance

Patch 1 records the relevant upstream pull-request lineage in its commit
message. The patch headers preserve the authorship and commit messages from the
measured worktree. llama.cpp and Cafe-derived code remain under their upstream
MIT license; see the repository third-party notices.
