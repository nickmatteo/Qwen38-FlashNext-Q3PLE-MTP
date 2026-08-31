# Q3PLE daily 80K lifecycle

`profiles/q3ple_daily_80k.json` is the **EXPERIMENTAL** daily profile for the
slot-state candidate runtime (`73b803464f25fc9054046728bf2ebed5a372737e`). It
uses the pinned target Q3_PLE bundle, the pinned DOWNQ4/FC-HC/OUTQ4 MTP
sidecar, one API slot (`id_slot=0`), context 81,920, 11 host threads, 2,048
batch / 256 micro-batch, q4_0 target and draft KV, draft `n_max=3`,
`p_min=0.75`, draft threads 8, the pinned tensor-placement override, and a
37 GiB Windows working-set cap. State, logs, slot files, manifests, and the
`latest.json` pointer are kept under:

```
results/QWEN38-MTP-PROTOTYPE-001/state/q3ple_daily/
```

The directory is local runtime state and is ignored by the repository. No GPU
launch is performed by this document or by the `smoke` command. The profile is
not a release recommendation until canonical actual-depth target/MTP parity
gates pass.

## Lifecycle

Run these commands from the repository root. `validate` checks the candidate
commit, clean worktree, executable bundle hashes, target first-shard hash,
sidecar hash, and target shard count/aggregate. `--no-hash` is a quick path
check when a full payload rehash is not appropriate.

```powershell
python scripts/q3ple_daily_profile.py show
python scripts/q3ple_daily_profile.py validate
python scripts/q3ple_daily_profile.py launch --mode mtp
python scripts/q3ple_daily_profile.py status
python scripts/q3ple_daily_profile.py save
python scripts/q3ple_daily_profile.py restore                 # latest.json
python scripts/q3ple_daily_profile.py stop
```

`launch --mode target` is an explicit target-only fallback. Mode is never
silently changed. A target server may restore the target part of an MTP
manifest, but the result records an explicit `target-only` fallback and does
not claim that draft state was restored. An MTP server refuses a target-only
manifest. Restore validates the manifest, byte lengths, SHA-256 values, mode,
and (for MTP) the target/.dft token-vector hash before making the one API call.

`stop --save-before-stop` performs the same validated save and manifest
promotion first. If save, parsing, hashing, or promotion fails, it fails closed
and leaves the owned server running.

The controller records the PID, process creation time, resolved executable,
full command line, runtime commit, complete nine-file runtime bundle identity,
and immutable environment/client fields in `server.json`. A detached
exact-owned watchdog continuously records RAM, RSS, swap, and GPU telemetry. It
fail-closes on missing telemetry or a hard-floor violation and terminates only
the PID whose creation time, executable, and command line still match. Every
API/termination action revalidates those ownership fields. Stop also stops the
exact watchdog PID; it never uses name-wide termination and never kills a
replacement process that reused a PID.

## Client contract

The server has one slot and clients must pass the slot and cache fields on each
completion request. Parallel requests are intentionally unsupported:

```json
{
  "model": "q3ple-daily",
  "prompt": "...",
  "max_tokens": 256,
  "temperature": 0,
  "stream": false,
  "id_slot": 0,
  "cache_prompt": true
}
```

For OpenAI-compatible chat clients, put the same values in the request body or
their `extra_body` facility. The native llama.cpp endpoint is `POST
/completion`; `/v1/chat/completions` requires a separately configured
OpenAI-compatibility adapter.

```python
client.chat.completions.create(
    model="q3ple-daily",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=256,
    extra_body={"id_slot": 0, "cache_prompt": True, "parallel": 1},
)
```

Generic OpenAI clients that cannot pass `id_slot`, `cache_prompt`, and
`parallel` need a small future wrapper. Do not assume that a client default
will select slot 0 or preserve the prompt cache.

Keep the server warm between turns. The first request after a clean restart can
take about 52--57 seconds while the 80K context and files become resident;
this is a restart/warm-up caveat, not a per-request latency target.

## Safety and persistence gates

The 37 GiB cap is the active safety candidate after the canonical v3 build at
38 GiB reached healthy model execution but dipped to 5.974 GiB available RAM,
just below the unchanged 6 GiB hard floor. This trades roughly 1 GiB of
resident mmap cache for operating margin; it remains experimental until a
canonical build and the lifecycle smoke measure its throughput and safety.

The launch preflight is deliberately strict: at least 40 GiB available RAM,
8,192 MiB free VRAM, and at most 15% GPU utilization. After launch, the
watchdog enforces hard runtime floors of 6 GiB available RAM, 768 MiB free
VRAM, at most 50 GiB owned RSS, and less than 1 GiB swap growth. The daily
promotion floor remains 1,024 MiB free VRAM. A failed preflight, health wait,
or watchdog gate does not trigger a mode switch or a retry after partial
generation.

An MTP save must produce both `<basename>.slot.bin` and a non-empty
`<basename>.slot.bin.dft`. The existing slot parser checks the state envelope,
sizes, SHA-256 values, and equality of the serialized target/draft prompt token
vectors. Only after those checks pass does the controller atomically write a
versioned `*.manifest.json` and then atomically promote `latest.json` to point
at it. Manifests bind the profile hash, runtime commit and all nine bundle
hashes, artifact identity, slot filename/path, environment, client contract,
and both target/.dft byte/hash/token-vector records. The target and `.dft`
files themselves are written by the server as two separate files and are
therefore not atomic together; unique basenames plus the validated manifest
pointer limit what a reader can select after a partial save but do not claim
power-loss or crash consistency.

Use the bounded non-live plan when checking a client or automation path without
loading the model:

```powershell
python scripts/q3ple_daily_profile.py smoke --mode mtp
```

It prints one deterministic completion request (slot 0, cache enabled) and
does not launch a process, generate text, write a slot, or stop a server.
