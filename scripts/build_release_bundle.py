#!/usr/bin/env python3
"""Build and verify the immutable Hugging Face release directory.

The large GGUF files are hard-linked on the same volume, so staging does not
duplicate approximately 75 GiB of payload. The script refuses to overwrite an
existing output directory and verifies all target hashes against the sealed
conversion manifest before creating the release manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "artifacts/models/AtomicChat/Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64"
SOURCE_MANIFEST = ROOT / "results/ATOMICCHAT-4.27-Q3PLE-001/gate_b_derivative_manifest.json"
SIDECAR = ROOT / "artifacts/models/Qwen3.8-Flash-Next-MTP-Q4_K_M-FC-HC/mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf"
CARD = ROOT / "docs/templates/HF_MODEL_CARD_RELEASE.md"
PROVENANCE = ROOT / "docs/templates/HF_PROVENANCE.md"
QWEN_LICENSE = ROOT / "LICENSES/Qwen-Community-License-1.0.txt"
SIDECAR_SHA256 = "7e9f2b282dc62534313b30738e0ad114c14e1a58b9c1e7bb9715dcf9c4ca676e"
SIDECAR_BYTES = 2_202_883_264


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def link_file(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    os.link(source, destination)


def compatibility_manifest(target_manifest_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "release": "Qwen3.8-Flash-Next-Q3_PLE-MTP-GGUF",
        "base_model": {
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
        },
        "immediate_source": {
            "repo": "AtomicChat/Qwen3.8-Flash-Next-GGUF",
            "revision": "142262902a46f7daed19c79d0771534c8106ad59",
        },
        "target": {
            "family": "Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64",
            "shards": 33,
            "aggregate_bytes": 78_525_318_176,
            "manifest_sha256": target_manifest_sha256,
            "ggml_type": "Q3_PLE",
            "ggml_type_code": 43,
        },
        "draft": {
            "filename": SIDECAR.name,
            "bytes": SIDECAR_BYTES,
            "sha256": SIDECAR_SHA256,
            "spec_type": "draft-mtp",
        },
        "runtime": {
            "base_repo": "quimmedes/cafe-llama.cpp",
            "base_commit": "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3",
            "required_head": "73b803464f25fc9054046728bf2ebed5a372737e",
            "stock_llama_cpp_compatible": False,
            "patches": "https://github.com/nickmatteo/Qwen38-FlashNext-Q3PLE-MTP/tree/main/patches/llama.cpp/cafe-035e227-to-73b803",
        },
        "license": "Qwen Community License 1.0",
    }


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing release directory: {output}")
    output.mkdir(parents=True)

    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected_rows = source["files"]
    if len(expected_rows) != 33 or source["aggregate_bytes"] != 78_525_318_176:
        raise RuntimeError("sealed target manifest identity does not match the promoted target")

    public_rows: list[dict] = []
    aggregate = 0
    for row in expected_rows:
        filename = row["filename"]
        source_file = TARGET_DIR / filename
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        size = source_file.stat().st_size
        if size != row["bytes"]:
            raise RuntimeError(f"size mismatch for {filename}: {size} != {row['bytes']}")
        actual_hash = sha256_file(source_file)
        if actual_hash.lower() != row["sha256"].lower():
            raise RuntimeError(f"SHA-256 mismatch for {filename}")
        link_file(source_file, output / filename)
        public_rows.append({"number": row["number"], "filename": filename, "bytes": size, "sha256": actual_hash})
        aggregate += size
        print(f"verified {row['number']:02d}/33 {filename} {size} {actual_hash}", flush=True)

    if aggregate != 78_525_318_176:
        raise RuntimeError(f"target aggregate mismatch: {aggregate}")

    sidecar_size = SIDECAR.stat().st_size
    sidecar_hash = sha256_file(SIDECAR)
    if sidecar_size != SIDECAR_BYTES or sidecar_hash != SIDECAR_SHA256:
        raise RuntimeError("promoted MTP sidecar identity mismatch")
    link_file(SIDECAR, output / SIDECAR.name)

    target_manifest = {
        "schema_version": 1,
        "artifact": "Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64",
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "file_count": 33,
        "aggregate_bytes": aggregate,
        "files": public_rows,
    }
    target_manifest_path = output / "target-shards.json"
    target_manifest_path.write_text(json.dumps(target_manifest, indent=2) + "\n", encoding="utf-8")
    target_manifest_sha = sha256_file(target_manifest_path)

    compatibility_path = output / "compatibility.json"
    compatibility_path.write_text(
        json.dumps(compatibility_manifest(target_manifest_sha), indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(CARD, output / "README.md")
    shutil.copy2(PROVENANCE, output / "PROVENANCE.md")
    shutil.copy2(QWEN_LICENSE, output / "LICENSE")

    checksum_files = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in checksum_files]
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    report = {
        "output": str(output),
        "files": len(checksum_files) + 1,
        "target_bytes": aggregate,
        "sidecar_bytes": sidecar_size,
        "gguf_bytes": aggregate + sidecar_size,
        "target_manifest_sha256": target_manifest_sha,
        "sha256sums_sha256": sha256_file(output / "SHA256SUMS"),
    }
    print(json.dumps(report, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new immutable release directory")
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
