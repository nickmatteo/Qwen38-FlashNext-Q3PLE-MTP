# Security policy

## Supported scope

This repository contains experimental local inference tooling and evidence. It
is not a hosted service and does not currently publish a stable security support
window.

## Reporting a vulnerability

Do not place credentials, private model URLs, exploit details, or sensitive
machine information in a public issue. Contact the repository owner privately
through the account hosting this repository and include:

- the affected revision and file;
- a minimal reproduction;
- expected and observed behavior;
- impact and any safe mitigation already tested.

## Operational safety

- Bind experimental servers to loopback unless remote access is intentional and
  independently secured.
- Verify PID, executable path, port ownership, and command line before stopping
  a process.
- Never commit tokens, cookies, `.env` files, private keys, or unredacted account
  data.
- Treat model files and benchmark prompts as untrusted input. Do not execute
  model-generated code during capability evaluation.
- Keep the RAM, VRAM, RSS, and pagefile watchdog floors enabled during model
  loads and long-context work.
