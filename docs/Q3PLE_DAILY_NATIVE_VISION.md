# Reproduce the measured 64K daily native-vision route

This guide reconstructs the public parts of the measured Windows path. It is
for a machine with an NVIDIA RTX 5070 12 GB, 64 GB of system RAM, and fast NVMe
storage. Other hardware may require a different expert placement or working-set
cap and is a new experiment.

## 1. Obtain the artifacts

Download the published Q3_PLE target and learned-MTP sidecar from the project
[Hugging Face release](https://huggingface.co/nmatteo3294/Qwen3.8-Flash-Next-Q3_PLE-MTP-GGUF).
Place the 33 target shards and sidecar at the relative paths recorded in
`profiles/q3ple_dsh_daily_64k_mtp.json`.

The F16 projector is derived from Qwen3.8-Flash-Next. The measured file is
904,003,520 bytes with SHA-256
`AFB86DC1041DFC72C90C510F33FAB52838F01EF8E3C5732666EA44FA28C96ACA`.
Place it at the profile's `artifacts.vision.path`. Qwen-derived files remain
under the Qwen Community License 1.0.

## 2. Reconstruct the runtime

Apply the original 36-commit series, then the four-commit daily delta:

```powershell
git clone https://github.com/quimmedes/cafe-llama.cpp.git
Set-Location cafe-llama.cpp
git checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
$basePatches = Get-ChildItem <research-repo>\patches\llama.cpp\cafe-035e227-to-73b803\*.patch | Sort-Object Name
git am --committer-date-is-author-date $basePatches.FullName
$dailyPatches = Get-ChildItem <research-repo>\patches\llama.cpp\73b803-to-4cae4d\*.patch | Sort-Object Name
git am --committer-date-is-author-date $dailyPatches.FullName
git rev-parse 'HEAD^{tree}'
```

The final source tree must be
`95417c378aee171d51ac09808c16f0b33b0fbdc0`. The measured source commit was
`4cae4d05122198c65895ee714780f4426aa76684`; patch reconstruction can produce
a different commit ID because email patches omit original committer identity.
Build for Windows CUDA and update the profile only when the resulting
executable and DLL hashes have been recorded. Changing the hashes without
requalification defeats the fail-closed profile check.

## 3. Validate without loading the model

Install Python dependencies from `requirements.txt`, then run:

```powershell
python scripts\q3ple_daily_profile.py `
  --profile profiles\q3ple_dsh_daily_64k_mtp.json validate
```

This checks schema, placement, safety policy, source commit, file sizes, and
artifact/runtime hashes. It does not start the model.

## 4. Start the JIT gateway

The public gateway owns a single upstream model process through the profile
controller. It starts unloaded, loads on the first valid chat request, and
unloads the model after 30 idle minutes:

```powershell
python scripts\q3ple_dsh_gateway.py serve `
  --profile profiles\q3ple_dsh_daily_64k_mtp.json `
  --listen-host 127.0.0.1 --listen-port 18092 --idle-minutes 30
```

Keep ports 18092 and 18098 on loopback. The gateway serializes inference for
the one-slot server, caps output, validates inline image data, reserves 1,024
tokens per image, and never silently changes from MTP to target-only mode.

## 5. Configure DSH or another OpenAI-compatible client

Merge `config/dsh/q3ple-daily-local.example.yml` into your settings and define
the referenced local client variable outside the repository. The preserved
route ID is `q3ple-fast-80k-mtp`, but the measured runtime and declared context
window are 65,536 tokens.

The profile's server default is low reasoning with a 512-token budget. The DSH
example explicitly requests medium reasoning, while the gateway caps the whole
completion at the profile's 1,024-token agent budget. Change the example's
`reasoning` field to `low` when you want the measured profile default.

The gateway accepts up to four inline PNG, JPEG, WebP, or GIF data URIs. Limits
are 4 MiB per decoded image and 8 MiB in aggregate. Remote image URLs are
rejected. This keeps request validation deterministic and avoids fetching
untrusted remote content inside the model service.

## 6. Stop by recorded identity

The profile controller stores the executable path, PID, creation time, command,
mode, and watchdog identity. Stop only that owned process:

```powershell
python scripts\q3ple_daily_profile.py `
  --profile profiles\q3ple_dsh_daily_64k_mtp.json stop
```

The gateway performs the same identity checks during idle unload. It will not
kill an unrelated process that happens to use the same executable name.

## Evidence boundary

The [measured result](../results/DSH-DAILY-NATIVE-VISION-004/RESULT.md) contains
the exact timings, resource envelope, full-path receipt, and limitations. It
qualifies one Windows system and one pinned artifact/runtime combination. It
does not promise the same fit, speed, or output on another machine. The matched
warm text row measured 15.82 tok/s. After a full unload, the first cold request
took 267.99 seconds (about 4 minutes 28 seconds), so JIT unload trades idle
resource release for substantial first-turn latency.
