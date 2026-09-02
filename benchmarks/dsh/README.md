# DeepSeek Harness benchmark configuration

DSH is the second agent harness for the Q3_PLE coding-agent lane. It talks to
the same loopback adapter Pi uses, so both harnesses are measured against one
gateway and one cache-accounting contract.

## Pinned versions

| Item | Value |
| --- | --- |
| Package | `@deepseek-ai/dsh` |
| Version | `0.1.2-alpha.4` |
| Tag | `dsh-v0.1.2-alpha.4` |
| Commit | `4e84901e6471b79ec0338099867ebb4606d12bb5` |
| License | MIT |
| Node | `^22.19.0 \|\| >=24` (measured on v26.5.1) |
| Profile | `headless` |

Install the scoped package, not the unscoped `dsh` on npm, which is a
different project:

```sh
npm install --save-exact @deepseek-ai/dsh@0.1.2-alpha.4
```

## Running one bounded job

```sh
export DSH_HOME=<a directory outside the repository>
export DSH_PERMISSION_MODE=read-only
export Q3PLE_LOCAL_API_KEY=local-q3ple
dsh --profile headless --patch benchmarks/dsh/cordis.patch.yml "<job>"
```

`--profile headless` runs one fresh persisted session, prints the final
answer, and exits. Use `--dump-config` instead of a job to print the composed
plugin tree and confirm the patch applied without booting anything.

The model server and the adapter are started separately. `dsh` never launches
either one.

## What the patch layer does

`cordis.patch.yml` is the entire integration. Each row replaces the targeted
row's complete `config`; the loader does not deep-merge keys.

- `llm-pi-ai` registers the `q3ple-local` route against the adapter on
  `127.0.0.1:18091`, with the wire-compat switches the gateway needs and the
  reasoning levels the pinned profile actually serves.
- `agent-default-model` selects that route.
- `sandbox-policy` and `approval` set read-only file effects with no approval
  prompt, and `permission` names that pair as the `read-only-headless` preset.
  The permission service refuses to boot if the composed pair has no name.
- `session-title-llm` is disabled. It otherwise issues its own short completion
  down the same route, which is a second conversation on a single-slot session
  and is correctly rejected by the adapter's append-only history contract.

## Operating notes

Erase the slot before starting a fresh adapter episode against a server that is
already running, or the adapter will refuse the first turn for reusing cache it
did not create:

```sh
curl -X POST "http://127.0.0.1:18090/slots/0?action=erase"
```

DSH's sandbox governs file effects only. On Windows its backend is an ACL
restricted-token runner that reports partial enforcement, and network access is
outside the sandbox vocabulary entirely. Treat `read-only` as defence in depth.
The disposable worktree, the declared command allowlist, and the objective
verifier remain the authoritative boundary for any task that writes.
