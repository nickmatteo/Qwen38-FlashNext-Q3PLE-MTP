# Benchmark evidence capture

This protocol governs the visible CLI and screenshot evidence requested for
future benchmark runs. It does not authorize a benchmark. Execution remains
blocked until the promoted target, sidecar, and runtime pass the MTP
rejection-parity gate in `benchmarks/manifest.json`.

## Evidence hierarchy

The source of truth is, in order:

1. immutable structured result rows and artifact hashes;
2. raw stdout, stderr, response, and timestamped telemetry;
3. a run manifest linking every file by SHA-256;
4. lossless screenshots that make the run legible to a human.

A screenshot strengthens provenance, but it cannot rescue a run with missing
raw data, a wrong artifact, contamination, failed parity, or a crossed safety
floor.

## Visible console layout

Use visible Windows Terminal or PowerShell windows whenever the tool supports
it. Keep three clearly titled consoles on screen:

- `SERVER <run-id>`: the exact owned `llama-server` command and server output;
- `CLIENT <run-id>`: the benchmark fixture, repeat, settings, and result;
- `DIAGNOSTICS <run-id>`: the safe diagnostic panel from
  `scripts/Capture-BenchmarkDiagnostics.ps1`.

Do not hide the server in a background task for a release run. A harness may
still own and monitor the process, but its live stdout should be visible or
tailed into the named server console. Only the harness may stop the child it
launched.

Before the first frame, close or move unrelated chats, mail, browser tabs,
account panels, and notifications away from the capture area. Do not expose
usernames, tokens, private paths, device serials, unrelated process command
lines, or private network addresses.

## Required frames

Capture these lossless PNG frames for every promoted performance row:

1. `00-preflight`: UTC timestamp, run ID, Git commit, clean tracked state,
   executable hash, artifact manifest hash, free RAM, pagefile use, and idle GPU
   state before model load.
2. `10-loaded`: server ready, owned PID, model and sidecar identity, loaded VRAM,
   RAM, RSS, and port.
3. `20-steady`: visible client repeat plus GPU utilization, power, temperature,
   VRAM, RAM, RSS, and pagefile during generation.
4. `30-result`: completion state, target or MTP mode, token counts, prefill,
   TTFT, decode, acceptance counters, output hashes, and validity verdict.
5. `40-postflight`: owned process stopped, port free, GPU memory returned, and
   post-run RAM/pagefile state.

Capture a failure frame whenever a watchdog fires. Do not crop away the failed
status, safety metric, or traceback.

## Diagnostic panel

The helper prints a screenshot-safe panel and optionally saves the same facts
as JSON:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/Capture-BenchmarkDiagnostics.ps1 `
  -RunId PUBLIC-BENCH-001-perf-0001 `
  -Stage preflight
```

After the server starts, add its verified PID:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/Capture-BenchmarkDiagnostics.ps1 `
  -RunId PUBLIC-BENCH-001-perf-0001 `
  -Stage loaded `
  -OwnedPid 12345
```

The helper intentionally uses a restricted `nvidia-smi` query instead of the
full process table. This shows GPU name, driver, P-state, temperature, power,
utilization, and memory without exposing unrelated executable paths. The raw
benchmark telemetry must still record enough process information to prove that
the GPU was uncontaminated.

## File layout and manifest

Use this structure:

```text
results/PUBLIC-BENCH-001/evidence/<run-id>/
  manifest.json
  diagnostics/
    preflight-<utc>.json
    loaded-<utc>.json
    steady-<utc>.json
    result-<utc>.json
    postflight-<utc>.json
  screenshots/
    00-preflight.png
    10-loaded.png
    20-steady.png
    30-result.png
    40-postflight.png
  raw/
  telemetry/
```

The evidence manifest must record, for every screenshot and diagnostic file:

- relative path and SHA-256;
- capture UTC time and stage;
- run ID and result-row ID;
- target or MTP mode and repeat number;
- the Git, runtime, executable, target-manifest, and sidecar identities;
- whether a redacted public copy differs from the sealed local original.

Never silently edit a sealed screenshot. If public redaction is required, keep
the original locally, create a derivative with a new filename and hash, and
record both in the manifest. Redaction may hide private metadata only; it must
not obscure settings, safety telemetry, errors, or the validity verdict.

## `nvidia-smi` credibility rules

- Pair every GPU frame with the run ID and UTC timestamp in the same visible
  terminal layout.
- Capture before load, after load, during a measured repeat, and after the owned
  server exits.
- Do not use a lone post-hoc `nvidia-smi` screenshot as performance evidence.
- Store sampled GPU telemetry throughout the run; a screenshot is only one
  point on that timeline.
- Record driver and executable hashes once per evidence bundle, then reference
  them from each result row.
- Preserve negative frames such as paging, throttling, OOM, low-VRAM watchdog,
  or foreign GPU activity and mark the affected row `FAILED` or `VOID`.

## Screenshot acceptance checklist

- Run ID and UTC are visible.
- The screenshot stage matches a diagnostic JSON record.
- Text is readable at native resolution.
- No unrelated application content or identifying path is visible.
- The model mode and relevant settings are visible.
- Safety telemetry and failure indicators are not cropped.
- PNG SHA-256 is present in the evidence manifest.
- Structured and raw evidence independently support the claimed number.
