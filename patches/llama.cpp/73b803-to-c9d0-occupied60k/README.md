# Occupied-60K qualification runtime delta

This directory contains the exact 24-commit source delta from the previously
published Q3_PLE runtime at
`73b803464f25fc9054046728bf2ebed5a372737e` to the occupied-context
qualification head at
`c9d0b6139b2a312af70c25e1b05974ee3ddcf1ce`.

The series includes the four-commit Windows direct-reader work also used by the
daily native-vision runtime, default-off diagnostic and parity experiments,
the paired IQ CPU kernel, and the gathered-attention serialization repair that
qualified exact output at 61,440 retained tokens. Experimental paths remain
opt-in through the environment flags recorded in
`profiles/q3ple_occupied_61440_pair.json`.

Apply it after the original 36-patch Q3_PLE series:

```powershell
git clone https://github.com/quimmedes/cafe-llama.cpp.git
Set-Location cafe-llama.cpp
git checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
$basePatches = Get-ChildItem <research-repo>\patches\llama.cpp\cafe-035e227-to-73b803\*.patch | Sort-Object Name
git am --committer-date-is-author-date $basePatches.FullName
$qualificationPatches = Get-ChildItem <research-repo>\patches\llama.cpp\73b803-to-c9d0-occupied60k\*.patch | Sort-Object Name
git am --committer-date-is-author-date $qualificationPatches.FullName
git rev-parse 'HEAD^{tree}'
```

The reconstructed source tree must be:

```text
7b3a784de1922d023ee5f040b19873903b26aeb4
```

The measured source commit was
`c9d0b6139b2a312af70c25e1b05974ee3ddcf1ce`. A reconstructed commit ID may
differ because email patches do not retain original committer identity; the
tree hash verifies the source content. The published executable and DLL hashes
identify the measured Windows build. A rebuild from this source is a
reproduction attempt unless its bundle hashes match the qualification profile.
