#!/usr/bin/env python3
"""Fail-closed static preflight for disposable real-agent benchmark tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePath
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "benchmarks" / "agent" / "task.schema.json"
FIXTURE_ROOT = (ROOT / "benchmarks" / "agent" / "fixtures").resolve()

DANGEROUS_COMMANDS = re.compile(
    r"(?:^|\s)(?:remove-item|rm|del|erase|format|shutdown|restart-computer|"
    r"invoke-webrequest|invoke-restmethod|curl|wget|git\s+(?:push|merge|clone|fetch|pull))(?:\s|$)",
    re.IGNORECASE,
)


class TaskError(RuntimeError):
    pass


def _relative_under(path_text: str, base: Path, root: Path) -> Path:
    candidate = PurePath(path_text)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise TaskError(f"path must be repository-relative without traversal: {path_text!r}")
    resolved = (base / Path(path_text)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise TaskError(f"path is outside the allowed root {root}: {path_text!r}") from error
    return resolved


def _command_text(command: str | Iterable[str]) -> str:
    if isinstance(command, str):
        return " ".join(command.split())
    return " ".join(str(part) for part in command)


def _require_safe_command(command: str | Iterable[str], label: str) -> str:
    text = _command_text(command)
    if not text or DANGEROUS_COMMANDS.search(text):
        raise TaskError(f"{label} is empty or contains a forbidden operation: {text!r}")
    if any(marker in text for marker in (";", "&&", "||", "|", ">", "<")):
        raise TaskError(f"{label} must be one command without shell composition: {text!r}")
    return text


def validate_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        import jsonschema
    except ImportError as error:
        raise TaskError("jsonschema is required for task preflight") from error
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(task)
    fixture = _relative_under(str(task["fixture"]["source"]), ROOT, FIXTURE_ROOT)
    for path in task["permissions"]["allowed_paths"]:
        _relative_under(str(path), fixture, fixture)
    setup = task["fixture"].get("setup_command")
    if setup:
        _require_safe_command(setup, "fixture setup command")
    allowed = [_require_safe_command(value, "allowed command") for value in task["permissions"]["allowed_commands"]]
    verifier = _require_safe_command(task["verifier"]["command"], "verifier command")
    if not any(verifier == value or verifier.startswith(value + " ") for value in allowed):
        raise TaskError("verifier command is not covered by permissions.allowed_commands")
    return {
        "valid": True,
        "task_id": task["task_id"],
        "fixture": str(fixture.relative_to(ROOT)).replace("\\", "/"),
        "verifier": verifier,
        "network": False,
        "push": False,
        "merge": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="task JSON file")
    args = parser.parse_args(argv)
    try:
        path = Path(args.task)
        if not path.is_absolute():
            path = ROOT / path
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TaskError("task JSON must be an object")
        print(json.dumps(validate_task(value), indent=2))
        return 0
    except (OSError, json.JSONDecodeError, TaskError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
