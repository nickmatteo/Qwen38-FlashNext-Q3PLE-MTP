# Contributing

This is an evidence-gated systems research project. Small, reviewable changes
with exact provenance are preferred over broad refactors.

## Before changing code

- Read [README.md](README.md), [DECISION.md](DECISION.md), and the
  [full walkthrough](docs/WALKTHROUGH.md).
- Keep model payloads, build trees, external worktrees, credentials, and local
  process state out of Git.
- Preserve unrelated local experiments and never stop a process unless its PID
  and executable path prove that the current run owns it.
- Record third-party source revisions and retain their notices.

## Evidence rules

- Label observations as `MEASURED`, `PROXY`, `ANALYTICAL`, `EXTERNAL`, or
  `VOID`.
- Keep allocated context distinct from actual prompt occupancy.
- Run the matched target-only row before an MTP row.
- Do not promote MTP performance unless exact output text and retokenized token
  IDs match the target on a rejection-bearing fixture.
- Keep failed and negative results. Append a superseding record instead of
  rewriting history.
- Treat screenshots as corroboration. Machine-readable rows, raw command
  output, hashes, and telemetry remain authoritative.

## Validation

Create a Python 3.10 or newer virtual environment and install the pinned
runtime and test dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Then, from the repository root:

```powershell
python scripts/benchmark.py validate --manifest benchmarks/manifest.json
python -m pytest -q
python -m compileall -q scripts tests
python scripts/validate_public_release.py
gitleaks git --staged --no-banner --redact --no-color
```

The repository Gitleaks configuration extends the default rules. Its allowlist
is limited to named token-ID SHA-256 evidence fields, fixed retrieval fixtures,
and the one sealed retrieval result whose token hash otherwise triggers the
generic API-key heuristic. Do not broaden it to other paths or arbitrary
hashes.

Local artifact validation additionally reads the ignored target, sidecar, and
runtime build:

```powershell
python scripts/benchmark.py validate `
  --manifest benchmarks/manifest.json `
  --require-local-artifacts
```

## Upstream work

Do not automatically open, push, or comment on an upstream llama.cpp change.
Follow its current contribution and agent policies. A human contributor must
understand and own the patch, commit message, pull-request text, and reviewer
conversation.
