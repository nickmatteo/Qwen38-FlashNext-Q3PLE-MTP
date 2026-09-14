# Exact measured runtime patch series

This directory contains the 36-commit patch series used to construct the
Windows CUDA runtime measured by this project.

## Identity

- source repository: `quimmedes/cafe-llama.cpp`;
- base commit: `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`;
- measured head: `73b803464f25fc9054046728bf2ebed5a372737e`;
- patch count: 36;
- generated with `git format-patch`, in commit order.

The series includes the Qwen4Exp runtime work inherited from Cafe, Q3_PLE
private type 43 support, lazy PLE loading, learned-MTP support, Windows/CUDA
host-placement work, and paired target plus `.dft` slot-state persistence.

## Apply

```powershell
git clone https://github.com/quimmedes/cafe-llama.cpp.git
Set-Location cafe-llama.cpp
git checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
$patches = Get-ChildItem <path-to-this-repository>\patches\llama.cpp\cafe-035e227-to-73b803\*.patch | Sort-Object Name
git am --committer-date-is-author-date $patches.FullName
git rev-parse 'HEAD^{tree}'
```

The resulting tree must be
`db305f143fefd46da53db8a8d751b882da38274f`. The measured source commit was
`73b803464f25fc9054046728bf2ebed5a372737e`; a reconstructed commit ID may
differ because email patches do not retain original committer identity. Patch
application is source reconstruction, not proof that another compiler, driver,
or GPU reproduces the published measurements.

## License and provenance

llama.cpp and Cafe-derived code remain under the upstream MIT license preserved
at `LICENSES/llama.cpp-MIT.txt`. Patch 0032 contains the pinned-host MoE/cache
lane attributed to FreeToken and is accompanied by the Apache License 2.0 text
at `LICENSES/FreeToken-Apache-2.0.txt`. The patch headers retain their original
authors and commit messages.

This repository does not claim the entire patch series as original project
work and does not relicense it under the root project MIT grant.
