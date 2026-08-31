#!/usr/bin/env python3
"""Validate the clean-history public research export without local model data."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "docs/WALKTHROUGH.md",
    "docs/community/Q3PLE-CANONICAL-60K-2026-08-31.md",
    "manifests/target-shards.json",
    "manifests/compatibility.json",
    "patches/llama.cpp/cafe-035e227-to-73b803/README.md",
    "profiles/q3ple_daily_80k.json",
    "scripts/q3ple_canonical_60k.py",
    "scripts/q3ple_daily_profile.py",
    "tests/test_q3ple_canonical_60k.py",
)
FORBIDDEN_SUFFIXES = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}
SECRET_PATTERNS = (
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_json_files() -> int:
    count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
            count += 1
        elif path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        fail(f"invalid JSONL {path.relative_to(ROOT)}:{line_number}: {exc}")
            count += 1
    return count


def validate_links() -> int:
    checked = 0
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8-sig")
        for target in MARKDOWN_LINK.findall(text):
            clean = target.strip().strip("<>").split("#", 1)[0]
            if not clean or clean.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                fail(f"missing local Markdown link: {path.relative_to(ROOT)} -> {target}")
            checked += 1
    return checked


def main() -> int:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            fail(f"missing required public file: {relative}")

    payloads = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES]
    if payloads:
        fail(f"model/binary payload entered Git export: {payloads[0].relative_to(ROOT)}")

    patches = list((ROOT / "patches/llama.cpp/cafe-035e227-to-73b803").glob("*.patch"))
    if len(patches) != 36:
        fail(f"runtime patch count mismatch: {len(patches)}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in {".png", ".npy"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                fail(f"credential-shaped value in {path.relative_to(ROOT)}")

    json_count = validate_json_files()
    link_count = validate_links()
    print(f"PUBLIC_RELEASE_OK json_files={json_count} local_links={link_count} patches={len(patches)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PUBLIC_RELEASE_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
