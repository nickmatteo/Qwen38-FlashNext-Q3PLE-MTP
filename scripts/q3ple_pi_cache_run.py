#!/usr/bin/env python3
"""Run a bounded multi-turn Pi cache proof without opening console windows."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PI_CLI = Path(os.environ.get("APPDATA", "")) / "npm/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
NODE = shutil.which("node.exe") or shutil.which("node")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def append_sync(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> dict:
    if NODE is None or not PI_CLI.is_file():
        raise RuntimeError("Pi node entrypoint is unavailable")
    if args.turns < 1 or args.turns > 100:
        raise RuntimeError("turns must be between 1 and 100")
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = run_dir / "pi-config"
    sessions = run_dir / "pi-sessions"
    config.mkdir(exist_ok=True)
    sessions.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "benchmarks/pi/models.json", config / "models.json")
    environment = os.environ.copy()
    environment.update({"PI_CODING_AGENT_DIR": str(config), "PI_TELEMETRY": "0"})
    progress = run_dir / "runner-progress.jsonl"
    started = time.perf_counter()
    for turn in range(1, args.turns + 1):
        tag = f"{turn:02d}"
        prompt = f"Reply with exactly {args.response_prefix}_{tag} and no other text."
        output = run_dir / f"pi-turn-{tag}.jsonl"
        command = [
            NODE,
            str(PI_CLI),
            "--offline",
            "--provider", "q3ple-local",
            "--model", "q3ple-daily-reasoning",
            "--thinking", "medium",
            "--mode", "json",
            "--print",
            "--session-id", args.session_id,
            "--session-dir", str(sessions),
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-tools",
            prompt,
        ]
        turn_started = time.perf_counter()
        with output.open("wb") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=args.turn_timeout,
                check=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        raw = output.read_text(encoding="utf-8", errors="replace")
        failed = completed.returncode != 0 or '"stopReason":"error"' in raw
        row = {
            "utc": utc_now(),
            "turn": turn,
            "elapsed_s": round(time.perf_counter() - turn_started, 3),
            "exit_code": completed.returncode,
            "failed": failed,
            "output": str(output.relative_to(run_dir)),
        }
        append_sync(progress, row)
        if failed:
            raise RuntimeError(f"Pi turn {turn} failed; see {output}")
    summary = {
        "schema": "q3ple-pi-cache-runner-v1",
        "status": "PASS",
        "utc": utc_now(),
        "session_id": args.session_id,
        "turns": args.turns,
        "wall_s": round(time.perf_counter() - started, 3),
        "headless": True,
        "provider": "q3ple-local",
        "model": "q3ple-daily-reasoning",
    }
    temporary = run_dir / "runner-summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, run_dir / "runner-summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--response-prefix", default="CACHE_TURN")
    parser.add_argument("--turn-timeout", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), indent=2))
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
