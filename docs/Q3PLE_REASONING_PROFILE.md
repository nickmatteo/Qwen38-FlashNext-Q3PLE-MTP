# Q3_PLE target-only reasoning profile

`profiles/q3ple_daily_80k_reasoning.json` is the isolated starting profile for
the real-world coding-agent benchmark. It does not replace or modify the
deterministic `q3ple_daily_80k.json` throughput profile.

The first intelligence baseline is target-only. MTP is pinned for later
matched experiments but is disabled by the profile policy and rejected by the
lifecycle CLI.

## Fixed profile

- profile id: `q3ple_daily_80k_reasoning_v1`;
- patched runtime: `73b803464f25fc9054046728bf2ebed5a372737e`;
- context allocation: 81,920;
- upstream llama-server: `127.0.0.1:18090`;
- Pi/DSH adapter: `127.0.0.1:18091`;
- one server slot: id 0;
- target-only `--spec-type none`;
- reasoning enabled, medium effort, 8,192-token reasoning ceiling;
- reasoning output format: `deepseek` (`reasoning_content` separated from the
  final response);
- preserved thinking disabled until the exact template-prefix test passes;
- independent state namespace:
  `results/QWEN38-MTP-PROTOTYPE-001/state/q3ple_daily_reasoning_v1`.

The request sampling policy follows the pinned official Qwen thinking-mode
recommendation:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0
}
```

## Static validation

These commands do not launch or load the model:

```powershell
python scripts\q3ple_daily_profile.py `
  --profile profiles\q3ple_daily_80k_reasoning.json `
  validate --no-files

python scripts\q3ple_daily_profile.py `
  --profile profiles\q3ple_daily_80k_reasoning.json `
  smoke --mode target

python scripts\q3ple_pi_adapter.py validate
python scripts\q3ple_pi_adapter.py smoke --turns 20 --fake-server
```

Passing `--mode mtp` with this profile is an intentional error. There is no
implicit fallback or post-generation retry in another mode.

## Controlled live sequence

Do not start this sequence while another upload, benchmark, build, or model
process owns the machine. The first live run uses a unique run id and local
evidence directory.

```powershell
$runId = "pi-cache-20turn-v1"
$runRoot = "results\REAL-WORLD-AGENT-001\$runId"

python scripts\q3ple_daily_profile.py `
  --profile profiles\q3ple_daily_80k_reasoning.json `
  validate

python scripts\q3ple_daily_profile.py `
  --profile profiles\q3ple_daily_80k_reasoning.json `
  launch --mode target

python scripts\q3ple_pi_adapter.py serve `
  --session-id $runId `
  --run-dir $runRoot
```

Use an isolated Pi configuration instead of editing the user's normal Pi
configuration:

```powershell
$env:PI_CODING_AGENT_DIR = (Join-Path $PWD "$runRoot\pi-config")
New-Item -ItemType Directory -Force -Path $env:PI_CODING_AGENT_DIR | Out-Null
Copy-Item benchmarks\pi\models.q3ple.json `
  (Join-Path $env:PI_CODING_AGENT_DIR "models.json")

pi --offline `
  --provider q3ple-local `
  --model q3ple-daily-reasoning `
  --thinking medium `
  --session-dir (Join-Path $PWD "$runRoot\pi-sessions")
```

Pi is the first harness. DeepSeek Harness is the only planned second harness.
Qwen Code is out of scope.

## Evidence and privacy

The adapter stores append-only rows plus exact raw request and response bytes
under the selected local run directory. Those records can contain repository
source, tool output, and model reasoning. They are private evidence by default
and must not be committed or published without a separate redaction and
release-manifest pass.

Every promoted turn must prove:

- the new rendered/tokenized request preserves the preceding prompt prefix;
- `cache_n` is at least the previous request-prompt token count;
- `cache_n + prompt_n` equals the current request-prompt token count;
- the forced slot/cache/parallel fields were used;
- the response ended normally and its raw evidence hashes resolve;
- no resource watchdog gate fired.

The custom project harness remains the authority for cache, resource, and
verifier claims. Pi and DSH are workloads that exercise that contract.
