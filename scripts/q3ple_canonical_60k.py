"""Canonical complete-message Q3_PLE+MTP benchmark harness.

This module defines the publication gate for a roughly 60K-token *canonical*
history. The final history is rendered through llama.cpp's chat template and
ends at a complete assistant boundary. Exact token prefixes of that one sealed
vector are submitted to ``/completion`` as transport chunks; those chunks may
end inside a message and are never described as semantic checkpoints. The old
synthetic repeated-token transcript is deliberately not used here.

The module is import-safe: importing it does not load a model, inspect a GPU,
open a socket, parse command-line arguments, or start a server.  Live work is
behind ``main`` and the small functions below are intentionally injectable so
unit tests can use fakes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import psutil
except ImportError:  # pragma: no cover - offline fixture checks do not need it
    psutil = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "QWEN38-MTP-PROTOTYPE-001"
FIXTURE_PATH = ROOT / "benchmarks" / "fixtures" / "q3ple_canonical_60k.jsonl"
HISTORY_FIXTURE = ROOT / "benchmarks" / "fixtures" / "q3ple_canonical_history.json"
DAILY_PROFILE_SCRIPT = ROOT / "scripts" / "q3ple_daily_profile.py"
MIN_BOUNDARY_TOKENS = 59_000
MAX_BOUNDARY_TOKENS = 60_000
CONTEXT_SIZE = 81_920
PORT = 18_089
SEED = 38_027
MIN_REPEATS = 3
EVIDENCE_CLASS = "MEASURED_DIAGNOSTIC"
CANDIDATE_WORKTREE = ROOT / "workstreams" / "llama.cpp-mtp-slot-state"
CANDIDATE_BIN = CANDIDATE_WORKTREE / "build-win-cuda-mtp-slot" / "bin"
CANDIDATE_EXE = CANDIDATE_BIN / "llama-server.exe"
TARGET_MODEL = ROOT / "artifacts" / "models" / "AtomicChat" / "Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64" / "Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64-00001-of-00033.gguf"
MTP_SIDECAR = ROOT / "artifacts" / "models" / "Qwen3.8-Flash-Next-MTP-Q4_K_M-FC-HC" / "mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf"
CANDIDATE_RUNTIME_COMMIT = "73b803464f25fc9054046728bf2ebed5a372737e"
SOURCE_WORKTREE_COMMIT = "4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc"
CANDIDATE_RUNTIME_FILES = {
    "ggml.dll": "A6E1A30EF7DA4B9D65D0496E0BD796D1871A4C1BD04F78DC41EDEE6B82716741",
    "ggml-base.dll": "345CC0849FA85522B476D1ACAE99C9DF2FB255CD3129D644D4937835E23B7F85",
    "ggml-cpu.dll": "66756DB9E03189B26F85442177D07EB33D0ECEFA267668D7811B05060C9B0B85",
    "ggml-cuda.dll": "444C23292C53E891B1D5C3A74B21233A015256CC92B91D35E6C6F757DDCBE0C2",
    "llama.dll": "8779092AC497E995DDEBE0327DA53EB4774EF4513E13E7E20EA9CDFFAC10613D",
    "llama-common.dll": "E192780F16EF88F341FA3E12300F7F0E118F40D3E3F6F23AE659BACA7F6E65BE",
    "llama-server.exe": "49B51341F0E29B7AA6F73C1723F105E407AFB06EAB3BCFDCE6490F4985DB949E",
    "llama-server-impl.dll": "F071A8864507E83AF9E67B1E6EC3D13D786B5BBBEA243DC34CA175C7303BE566",
    "mtmd.dll": "11B609FEE9DC975911662E170439FC66B7553C48B43B2A04F988A2636D61A597",
}
CANDIDATE_TARGET_SHARD_COUNT = 33
CANDIDATE_TARGET_BYTES = 78_525_318_176
CANDIDATE_TARGET_FIRST_SHA256 = "D160140839732E03F0FDAD1BF27B7512FFBEE4DC411BCE0F986367A005D9A726"
CANDIDATE_SIDECAR_SHA256 = "7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E"
CANDIDATE_SIDECAR_BYTES = 2_202_883_264
WORKING_SET_CAP_GIB = 37
WORKING_SET_CAP_BYTES = WORKING_SET_CAP_GIB * 1024**3
TARGET_KV = "q4_0"
TARGET_THREADS = 11
TARGET_BATCH = 2048
TARGET_UBATCH = 256
TARGET_N_CPU_MOE = 48
MTP_DRAFT_N_MAX = 3
MTP_DRAFT_P_MIN = 0.75
MTP_DRAFT_THREADS = 8
CHUNK_TOKENS = 512
HARD_RAM_FLOOR = 6 * 1024**3
HARD_RSS_CEILING = 50 * 1024**3
HARD_SWAP_GROWTH = 1 * 1024**3
HARD_VRAM_FLOOR_MIB = 768
PUBLISHABLE_VRAM_FLOOR_MIB = 1024
REQUEST_TIMEOUT_SECONDS = 7200
SYSTEM_PROMPT = (
    "You are a deterministic engineering assistant. Follow the exact requested "
    "format, preserve quoted markers, and do not invent evidence. Prefer small, "
    "stdlib-only, reviewable solutions. Return only the requested answer."
)
ASSISTANT_ACK = "ACK_CONTEXT_RECEIVED"
EARLY_RETRIEVAL_NEEDLE = "A01 canonical boundary retrieval: preserve this exact line."
BOUNDARY_PROBE_MARKER = "Q3PLE_CANONICAL_BOUNDARY_PROBE_38027_DO_NOT_SAVE"
RESTART_PROBE_USER = "Reply with exactly: Q3PLE_RESTART_CACHE_OK"
PINNED_OVERRIDE = (
    r"^output\.weight$=CUDA0,^blk\.48\.attn_.*=CUDA0,^blk\.48\.hc_attn_.*=CUDA0,"
    r"^blk\.48\.hc_ffn_.*=CUDA0,^blk\.48\.nextn\..*=CUDA0,^blk\.48\.ffn_gate_inp.*=CUDA0,"
    r"^blk\.48\.ffn_(gate|up|down)_shexp.*=CUDA0,^blk\.48\.ffn_(gate|up|down)_exps.*=CUDA_Host"
)


def _load_module(path: Path, name: str):
    """Load a reference script without invoking its CLI entry point."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_modules() -> tuple[Any, Any, Any]:
    """Return cache-probe, slot-extension, and realistic scorer modules."""

    return (
        _load_module(ROOT / "scripts" / "q3ple_agentic_cache_probe.py", "q3ple_cache_probe_ref"),
        _load_module(ROOT / "scripts" / "q3ple_agentic_slot_extend.py", "q3ple_slot_extend_ref"),
        _load_module(ROOT / "scripts" / "q3ple_realistic_ab.py", "q3ple_realistic_ref"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_tokens(tokens: Iterable[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def harness_source_identity(fixture_path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Hash the small source/config inputs that define one live run."""

    inputs = [
        Path(__file__).resolve(),
        DAILY_PROFILE_SCRIPT,
        ROOT / "profiles" / "q3ple_daily_80k.json",
        ROOT / "scripts" / "q3ple_agentic_cache_probe.py",
        ROOT / "scripts" / "q3ple_agentic_slot_extend.py",
        ROOT / "scripts" / "q3ple_mtp_ab.py",
        ROOT / "scripts" / "q3ple_realistic_ab.py",
        Path(fixture_path).resolve(),
    ]
    if HISTORY_FIXTURE.is_file():
        inputs.append(HISTORY_FIXTURE)
    records: dict[str, Any] = {}
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"harness input is missing: {path}")
        key = repo_relative(path)
        if key in records:
            continue
        records[key] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": "q3ple-harness-source-identity-v1",
        "files": records,
        "files_sha256": sha256_json(records),
    }


def daily_profile_module():
    return _load_module(DAILY_PROFILE_SCRIPT, "q3ple_canonical_daily_profile")


def expand_message(message: Mapping[str, Any]) -> dict[str, str]:
    """Expand a fixture message's deterministic ``repeat`` shorthand."""

    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
        raise ValueError(f"message role must be system/user/assistant, got {role!r}")
    if not isinstance(content, str) or not content:
        raise ValueError("message content must be a non-empty string")
    repeat = message.get("repeat", 1)
    if not isinstance(repeat, int) or repeat < 1:
        raise ValueError("message repeat must be a positive integer")
    if repeat == 1:
        expanded = content
    else:
        # Keep each repetition on a stable newline boundary so the source is
        # deterministic and does not depend on Python's repr or locale.
        expanded = "\n".join(f"[{index:03d}] {content}" for index in range(1, repeat + 1))
    return {"role": role, "content": expanded}


def load_history_fixture(path: Path = HISTORY_FIXTURE) -> dict[str, Any]:
    """Load and validate the canonical history fixture, expanding repeats."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        fixtures = load_benchmark_fixtures(path)
        # JSONL is the authoritative task manifest.  This compatibility view is
        # intentionally small; the long source-backed history is built live.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for row in fixtures:
            messages.extend([
                {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": ASSISTANT_ACK},
            ])
        return {
            "schema": "q3ple-canonical-history-v1",
            "id": "q3ple-canonical-60k-jsonl-compat",
            "messages": messages,
            "boundary_tokens": {"minimum": MIN_BOUNDARY_TOKENS, "maximum": MAX_BOUNDARY_TOKENS},
            "benchmark_fixture_ids": [row["id"] for row in fixtures],
            "messages_sha256": sha256_json(messages),
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "q3ple-canonical-history-v1":
        raise ValueError("canonical fixture schema is not q3ple-canonical-history-v1")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("canonical fixture must contain a non-empty messages list")
    expanded = [expand_message(item) for item in messages]
    if expanded[-1]["role"] != "assistant":
        raise ValueError("canonical history must end at a complete assistant message")
    bounds = raw.get("boundary_tokens") or {}
    minimum = bounds.get("minimum", MIN_BOUNDARY_TOKENS)
    maximum = bounds.get("maximum", MAX_BOUNDARY_TOKENS)
    if not isinstance(minimum, int) or not isinstance(maximum, int) or minimum >= maximum:
        raise ValueError("canonical fixture boundary_tokens must be an increasing integer range")
    return {
        **raw,
        "messages": expanded,
        "boundary_tokens": {"minimum": minimum, "maximum": maximum},
        "messages_sha256": sha256_json(expanded),
    }


def _validate_benchmark_fixture_contract(row: Mapping[str, Any]) -> None:
    """Validate category-specific scorer inputs before any model is loaded."""

    fixture_id = str(row["id"])
    max_tokens = row.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError(f"fixture {fixture_id} needs a positive integer max_tokens")
    requirements = row.get("semantic_requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError(f"fixture {fixture_id} needs semantic_requirements")
    category = row["category"]
    scorer = row["scorer"]
    if category == "retrieval":
        needle = row.get("needle")
        if scorer != "retrieval_needle" or not isinstance(needle, str) or not needle:
            raise ValueError(f"fixture {fixture_id} has an invalid retrieval scorer/needle")
        if requirements.get("exact_line") != needle:
            raise ValueError(f"fixture {fixture_id} retrieval exact_line differs from needle")
    elif category == "code":
        if scorer != "code_atomic_json":
            raise ValueError(f"fixture {fixture_id} has an invalid code scorer")
        if requirements.get("function_name") != "atomic_json" or requirements.get("stdlib_only") is not True:
            raise ValueError(f"fixture {fixture_id} lacks the atomic_json/stdlib contract")
    elif category == "prose":
        if scorer != "prose_filled_context_incident":
            raise ValueError(f"fixture {fixture_id} has an invalid prose scorer")
        bounds = requirements.get("word_count")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in bounds)
            or bounds[0] < 1
            or bounds[0] > bounds[1]
        ):
            raise ValueError(f"fixture {fixture_id} has invalid prose word_count bounds")
        if not isinstance(requirements.get("end_marker"), str) or not requirements["end_marker"]:
            raise ValueError(f"fixture {fixture_id} needs a prose end_marker")


def load_benchmark_fixtures(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Parse the canonical retrieval/code/prose JSONL task manifest."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL fixture line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"fixture line {line_number} needs a string id")
            if row["id"] in seen:
                raise ValueError(f"duplicate fixture id {row['id']}")
            for field in ("category", "system", "user", "scorer"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"fixture {row['id']} missing string {field}")
            if row["category"] not in {"retrieval", "code", "prose"}:
                raise ValueError(f"fixture {row['id']} has unsupported category")
            if not isinstance(row.get("provenance"), dict) or not row["provenance"].get("static"):
                raise ValueError(f"fixture {row['id']} must include static provenance")
            _validate_benchmark_fixture_contract(row)
            seen.add(row["id"]); rows.append(row)
    categories = {row["category"] for row in rows}
    if categories != {"retrieval", "code", "prose"}:
        raise ValueError(f"fixture categories must be retrieval/code/prose, got {sorted(categories)}")
    return rows


def canonical_history(path: Path = HISTORY_FIXTURE) -> list[dict[str, str]]:
    """Return the expanded deterministic message list (convenience alias)."""

    return list(load_history_fixture(path)["messages"])


def validate_fixture(path: Path = HISTORY_FIXTURE) -> dict[str, Any]:
    """Pure validation used by CI and ``--validate-fixtures``."""

    if Path(path).suffix.lower() == ".jsonl":
        fixtures = load_benchmark_fixtures(path)
        return {
            "valid": True,
            "fixture_count": len(fixtures),
            "categories": sorted({row["category"] for row in fixtures}),
            "fixture_ids": [row["id"] for row in fixtures],
            "candidate_runtime_commit": CANDIDATE_RUNTIME_COMMIT,
            "boundary_tokens": {"minimum": MIN_BOUNDARY_TOKENS, "maximum": MAX_BOUNDARY_TOKENS},
        }
    fixture = load_history_fixture(path)
    messages = fixture["messages"]
    roles = [message["role"] for message in messages]
    if roles[-1] != "assistant":
        raise ValueError("final canonical role is not assistant")
    if not any(role == "user" for role in roles):
        raise ValueError("canonical history must include a user turn")
    if not any(role == "system" for role in roles):
        raise ValueError("canonical history must include a system turn")
    if any(message.get("repeat") for message in messages):
        raise AssertionError("load_history_fixture must remove repeat shorthand")
    return {
        "valid": True,
        "id": fixture["id"],
        "message_count": len(messages),
        "final_role": roles[-1],
        "messages_sha256": fixture["messages_sha256"],
        "boundary_tokens": fixture["boundary_tokens"],
        "benchmark_fixture_ids": list(fixture.get("benchmark_fixture_ids", [])),
    }


def render_messages(
    messages: Sequence[Mapping[str, str]],
    renderer: Callable[[Sequence[Mapping[str, str]]], str],
) -> str:
    """Render a complete message list through an injected chat-template call."""

    rendered = renderer(list(messages))
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat-template renderer returned empty/non-string prompt")
    return rendered


def render_boundary_records(
    messages: Sequence[Mapping[str, str]],
    renderer: Callable[[Sequence[Mapping[str, str]]], str],
    tokenizer: Callable[[str], Sequence[int]],
) -> list[dict[str, Any]]:
    """Render/tokenize every complete-message prefix and prove exact ancestry."""

    records: list[dict[str, Any]] = []
    previous: list[int] = []
    for index in range(1, len(messages) + 1):
        prefix_messages = [dict(item) for item in messages[:index]]
        raw_prompt = render_messages(prefix_messages, renderer)
        tokens = list(tokenizer(raw_prompt))
        if not tokens:
            raise ValueError(f"empty token vector at message boundary {index}")
        exact_prefix = tokens[: len(previous)] == previous
        common_prefix = _common_prefix(previous, tokens)
        record = {
            "message_index": index,
            "message_count": index,
            "messages": prefix_messages,
            "last_role": prefix_messages[-1]["role"],
            "raw_prompt": raw_prompt,
            "raw_prompt_sha256": sha256_text(raw_prompt),
            "tokens": tokens,
            "token_count": len(tokens),
            "token_ids_sha256": sha256_tokens(tokens),
            "previous_token_count": len(previous),
            "common_prefix_tokens": common_prefix,
            "exact_previous_prefix": exact_prefix,
            "complete_assistant_boundary": prefix_messages[-1]["role"] == "assistant",
        }
        records.append(record)
        if not exact_prefix:
            raise ValueError(
                f"message boundary {index} is not an exact prefix of boundary {index - 1} "
                f"(common={common_prefix}, previous={len(previous)})"
            )
        previous = tokens
    return records


def assert_exact_message_prefixes(records: Sequence[Mapping[str, Any]]) -> bool:
    """Raise on a non-prefix boundary and return ``True`` for a valid chain."""

    previous: list[int] = []
    for record in records:
        tokens = list(record.get("tokens", []))
        if not tokens[: len(previous)] == previous:
            raise ValueError("complete-message token vectors are not an exact prefix chain")
        previous = tokens
    return True


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def select_canonical_boundary(
    records: Sequence[Mapping[str, Any]],
    minimum: int = MIN_BOUNDARY_TOKENS,
    maximum: int = MAX_BOUNDARY_TOKENS,
) -> dict[str, Any]:
    """Select the largest whole-message assistant boundary in the target range."""

    if minimum < 1 or maximum <= minimum:
        raise ValueError("boundary range must be positive and increasing")
    candidates = [
        dict(record)
        for record in records
        if minimum <= int(record.get("token_count", -1)) <= maximum
        and record.get("last_role") == "assistant"
        and record.get("exact_previous_prefix") is True
        and record.get("complete_assistant_boundary") is True
    ]
    if not candidates:
        raise ValueError(
            f"no complete assistant boundary in {minimum}-{maximum} tokens"
        )
    selected = max(candidates, key=lambda item: item["token_count"])
    # Never expose a mutable alias to a caller that may append benchmark tokens.
    selected["tokens"] = list(selected["tokens"])
    selected["selected"] = True
    selected["boundary_range"] = {"minimum": minimum, "maximum": maximum}
    selected["canonical_chat_template_boundary"] = True
    selected["synthetic_literal_continuation"] = False
    return selected


def build_boundary_proof(
    records: Sequence[Mapping[str, Any]], boundary: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a compact, serializable proof for the selected boundary."""

    selected_index = int(boundary["message_index"])
    prior = [dict(item) for item in records[:selected_index]]
    all_exact = all(item.get("exact_previous_prefix") is True for item in prior)
    return {
        "message_index": selected_index,
        "message_count": int(boundary["message_count"]),
        "last_role": boundary["last_role"],
        "token_count": int(boundary["token_count"]),
        "token_ids_sha256": boundary["token_ids_sha256"],
        "raw_prompt_sha256": boundary["raw_prompt_sha256"],
        "all_complete_message_prefixes_exact": all_exact,
        "base_token_vector_is_exact_prefix": all_exact,
        "canonical_chat_template_boundary": True,
    }


def completion_request_body(tokens: Sequence[int], *, n_predict: int = 0) -> dict[str, Any]:
    """Build the token-array request used for canonical prefix warming."""

    return {
        "prompt": list(tokens),
        "n_predict": int(n_predict),
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
        "seed": SEED,
        "stream": False,
    }


def verify_completion_accounting(
    response: Mapping[str, Any], request_tokens: int, expected_cache: int
) -> dict[str, Any]:
    """Verify llama-server's cache_n/prompt_n split for one prefix request."""

    timings = response.get("timings") if isinstance(response, Mapping) else None
    timings = timings if isinstance(timings, Mapping) else {}
    cache_n = timings.get("cache_n", response.get("cache_n"))
    prompt_n = timings.get("prompt_n", response.get("prompt_n"))
    if cache_n is None or prompt_n is None:
        raise ValueError("completion response lacks cache_n/prompt_n")
    cache_n = int(cache_n)
    prompt_n = int(prompt_n)
    expected_prompt = request_tokens - expected_cache
    checks = {
        "cache_n_exact": cache_n == expected_cache,
        "prompt_n_exact": prompt_n == expected_prompt,
        "accounting_exact": cache_n + prompt_n == request_tokens,
    }
    return {
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "request_tokens": request_tokens,
        "expected_cache_n": expected_cache,
        "expected_prompt_n": expected_prompt,
        "checks": checks,
        "pass": all(checks.values()),
    }


def offline_pair_gate(
    target_path: Path, expected_tokens: Sequence[int], *, expected_count: int | None = None
) -> dict[str, Any]:
    """Parse and verify a target/``.dft`` pair without loading llama.cpp."""

    _, extender, _ = reference_modules()
    parsed = extender.parse_slot_pair(
        Path(target_path), expected_count or len(expected_tokens), expected_tokens
    )
    parsed["offline_pair_gate"] = True
    parsed["target_token_vector_exact"] = parsed["expected_tokens_equal"]
    parsed["pair_promotion_pass"] = bool(
        parsed.get("target_draft_tokens_equal") and parsed.get("expected_tokens_equal")
    )
    return parsed


def _durable_copy_new(source: Path, destination: Path) -> None:
    """Copy one file without overwrite and flush the new bytes before return."""

    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def copy_slot_pair(
    source_target: Path,
    destination_dir: Path,
    destination_filename: str,
    expected_tokens: Sequence[int],
    *,
    pair_gate: Callable[..., dict[str, Any]] = offline_pair_gate,
) -> dict[str, Any]:
    """Copy one immutable adjacent target/``.dft`` pair and revalidate it."""

    source_target = Path(source_target).resolve()
    source_draft = Path(f"{source_target}.dft").resolve()
    destination_dir = Path(destination_dir).resolve()
    if not destination_dir.is_dir():
        raise ValueError("destination slot directory does not exist")
    if (
        not isinstance(destination_filename, str)
        or not destination_filename
        or Path(destination_filename).name != destination_filename
        or not destination_filename.endswith(".slot.bin")
    ):
        raise ValueError("destination slot filename must be a basename ending in .slot.bin")
    destination_target = (destination_dir / destination_filename).resolve()
    destination_draft = Path(f"{destination_target}.dft").resolve()
    if destination_target.parent != destination_dir or destination_draft.parent != destination_dir:
        raise ValueError("destination slot pair escaped its run directory")
    if not source_target.is_file() or not source_draft.is_file():
        raise FileNotFoundError("source target/.dft pair is incomplete")
    if source_target == destination_target or source_draft == destination_draft:
        raise ValueError("source and destination slot pairs must differ")
    if destination_target.exists() or destination_draft.exists():
        raise FileExistsError("destination slot pair already exists")

    tokens = list(expected_tokens)
    source_pair = pair_gate(source_target, tokens, expected_count=len(tokens))
    if not source_pair.get("pair_promotion_pass"):
        raise ValueError("source target/.dft pair failed the offline gate")
    try:
        _durable_copy_new(source_target, destination_target)
        _durable_copy_new(source_draft, destination_draft)
        destination_pair = pair_gate(destination_target, tokens, expected_count=len(tokens))
        if not destination_pair.get("pair_promotion_pass"):
            raise ValueError("copied target/.dft pair failed the offline gate")
        for field in ("target_bytes", "draft_bytes", "target_sha256", "draft_sha256"):
            if destination_pair.get(field) != source_pair.get(field):
                raise ValueError(f"copied slot pair changed {field}")
        return {
            "schema_version": "q3ple-slot-pair-copy-v1",
            "source": {
                "target_path": str(source_target),
                "draft_path": str(source_draft),
                "target_sha256": source_pair.get("target_sha256"),
                "draft_sha256": source_pair.get("draft_sha256"),
            },
            "destination": {
                "target_path": str(destination_target),
                "draft_path": str(destination_draft),
                "target_sha256": destination_pair.get("target_sha256"),
                "draft_sha256": destination_pair.get("draft_sha256"),
            },
            "pair": destination_pair,
            "bytes_and_hashes_exact": True,
        }
    except Exception:
        for path in (destination_draft, destination_target):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def append_generation_prompt(
    base_messages: Sequence[Mapping[str, str]], fixture: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Append one complete user fixture while preserving the saved base."""

    user = fixture.get("user")
    instruction = fixture.get("system", "Return only the requested answer.")
    if not isinstance(user, str) or not user:
        raise ValueError("benchmark fixture has no user prompt")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("benchmark fixture has no evaluation instruction")
    content = f"EVALUATION_INSTRUCTION\n{instruction}\n\nTASK\n{user}"
    return [dict(item) for item in base_messages] + [{"role": "user", "content": content}]


def build_benchmark_prompt(
    boundary: Mapping[str, Any],
    fixture: Mapping[str, Any],
    renderer: Callable[..., str],
    tokenizer: Callable[[str], Sequence[int]],
) -> dict[str, Any]:
    """Build the complete user-turn benchmark prompt from a saved boundary."""

    base_messages = boundary.get("messages")
    if base_messages is None:
        raise ValueError("boundary must include complete base messages")
    messages = append_generation_prompt(base_messages, fixture)
    prompt = generation_prompt_tokens(messages, renderer, tokenizer)
    base_tokens = list(boundary.get("tokens", []))
    prompt["messages"] = messages
    prompt["base_token_count"] = len(base_tokens)
    prompt["base_token_ids_sha256"] = sha256_tokens(base_tokens)
    prompt["base_is_exact_prefix"] = prompt["tokens"][: len(base_tokens)] == base_tokens
    if not prompt["base_is_exact_prefix"]:
        raise ValueError("benchmark generation prompt does not preserve saved canonical prefix")
    return prompt


def generation_prompt_tokens(
    messages: Sequence[Mapping[str, str]],
    renderer: Callable[..., str],
    tokenizer: Callable[[str], Sequence[int]],
) -> dict[str, Any]:
    """Render a benchmark turn with a generation prompt and tokenize it."""

    try:
        raw = renderer(list(messages), add_generation_prompt=True)
    except TypeError:
        # A fake or older renderer may expose only ``messages``.  The caller can
        # still add the generation marker itself, while the result records this.
        raw = renderer(list(messages))
    if not isinstance(raw, str) or not raw:
        raise ValueError("generation renderer returned empty/non-string prompt")
    tokens = list(tokenizer(raw))
    return {
        "raw_prompt": raw,
        "raw_prompt_sha256": sha256_text(raw),
        "tokens": tokens,
        "token_count": len(tokens),
        "token_ids_sha256": sha256_tokens(tokens),
        "add_generation_prompt": True,
    }


def source_backed_messages(
    filler: str,
    *,
    target_chars: int | None = None,
    turn_chars: int = 12_000,
) -> list[dict[str, str]]:
    """Build deterministic complete user/assistant ingestion turns from source."""

    if turn_chars < 256:
        raise ValueError("turn_chars must be at least 256")
    source = filler if target_chars is None else filler[: max(1, int(target_chars))]
    if not source:
        raise ValueError("llama.cpp source corpus is empty")
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    chunks = [source[index : index + turn_chars] for index in range(0, len(source), turn_chars)]
    for index, chunk in enumerate(chunks, 1):
        early = (
            f"AGENTIC_RETRIEVAL_NEEDLE_BEGIN\n{EARLY_RETRIEVAL_NEEDLE}\n"
            "AGENTIC_RETRIEVAL_NEEDLE_END\n"
            if index == 1 else ""
        )
        messages.append({
            "role": "user",
            "content": (
                f"CONTEXT_INGESTION_TURN_{index:03d}\n"
                "Read this complete local llama.cpp source excerpt as context; do not answer yet.\n"
                f"{early}--- SOURCE EXCERPT {index:03d} ---\n{chunk}"
            ),
        })
        messages.append({"role": "assistant", "content": ASSISTANT_ACK})
    return messages


def post_apply_template(
    port: int,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """POST /apply-template with explicit generation-prompt semantics."""

    probe, _, _ = reference_modules()
    response = probe.post_json(
        port,
        "/apply-template",
        {"messages": [dict(item) for item in messages], "add_generation_prompt": bool(add_generation_prompt)},
        120,
    )
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("prompt", "content", "result"):
            if isinstance(response.get(key), str) and response[key]:
                return response[key]
    raise ValueError(f"/apply-template returned no prompt string: {response!r}")


def render_complete_assistant_boundary(
    port: int, messages: Sequence[Mapping[str, str]]
) -> tuple[str, dict[str, Any]]:
    """Render a history ending after a complete assistant message.

    The pinned Qwen template rejects a transcript whose final role is assistant
    when asked to render it directly. Append a unique throwaway user turn,
    render the valid transcript, then cut at that final user header. The saved
    bytes therefore end immediately after the preceding complete assistant
    message. Every measured task independently proves this byte/token prefix.
    """

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("canonical boundary source must end with an assistant message")
    probe_messages = [dict(item) for item in messages] + [
        {"role": "user", "content": BOUNDARY_PROBE_MARKER}
    ]
    rendered = post_apply_template(port, probe_messages, add_generation_prompt=True)
    marker_index = rendered.find(BOUNDARY_PROBE_MARKER)
    if marker_index < 0 or rendered.find(BOUNDARY_PROBE_MARKER, marker_index + 1) >= 0:
        raise ValueError("canonical boundary probe marker is missing or ambiguous")
    user_header = "<|im_start|>user"
    boundary_end = rendered.rfind(user_header, 0, marker_index)
    if boundary_end < 0:
        raise ValueError("pinned Qwen template did not expose the final user header")
    boundary = rendered[:boundary_end]
    if not boundary or not rendered.startswith(boundary):
        raise ValueError("canonical assistant-boundary extraction failed")
    if ASSISTANT_ACK not in boundary[-512:]:
        raise ValueError("canonical boundary does not end after the expected assistant ACK")
    return boundary, {
        "method": "append_unique_user_then_cut_before_final_chatml_user_header",
        "probe_marker_sha256": sha256_text(BOUNDARY_PROBE_MARKER),
        "rendered_probe_sha256": sha256_text(rendered),
        "boundary_char_count": len(boundary),
        "boundary_end": boundary_end,
        "final_role": "assistant",
        "complete_assistant_boundary": True,
        "probe_content_saved": False,
    }


def build_restart_suffix_probe(port: int, base: Mapping[str, Any]) -> dict[str, Any]:
    """Render one canonical user suffix for the fresh-process restore proof."""

    base_tokens = list(base.get("tokens") or [])
    messages = [dict(item) for item in base.get("messages") or []]
    if not base_tokens or not messages or messages[-1].get("role") != "assistant":
        raise ValueError("restart probe requires a complete assistant boundary")
    messages.append({"role": "user", "content": RESTART_PROBE_USER})
    raw_prompt = post_apply_template(port, messages, add_generation_prompt=True)
    probe, _, _ = reference_modules()
    prompt_tokens = list(probe.tokenize(port, raw_prompt))
    if prompt_tokens[: len(base_tokens)] != base_tokens:
        raise ValueError("restart probe does not preserve the sealed token prefix")
    suffix_tokens = prompt_tokens[len(base_tokens) :]
    if not suffix_tokens:
        raise ValueError("restart probe produced no unseen suffix")
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_token_count": len(prompt_tokens),
        "prompt_token_ids_sha256": sha256_tokens(prompt_tokens),
        "suffix_token_count": len(suffix_tokens),
        "suffix_token_ids_sha256": sha256_tokens(suffix_tokens),
        "user_sha256": sha256_text(RESTART_PROBE_USER),
        "base_prefix_exact": True,
    }


def build_canonical_state(
    port: int,
    *,
    minimum: int = MIN_BOUNDARY_TOKENS,
    maximum: int = MAX_BOUNDARY_TOKENS,
    turn_chars: int = 12_000,
) -> dict[str, Any]:
    """Build and prove a 59K-60K source-backed complete-message boundary."""

    probe, _, _ = reference_modules()
    base = _load_module(ROOT / "scripts" / "q3ple_mtp_ab.py", "q3ple_canonical_base_ref")
    _historical_needle, filler, corpus_manifest = probe.build_source_material(base)
    target_chars = min(len(filler), max(turn_chars, int(((minimum + maximum) / 2) * 4)))
    sizing: list[dict[str, int]] = []
    messages: list[dict[str, str]] = []
    raw_prompt = ""
    tokens: list[int] = []
    extraction: dict[str, Any] = {}
    for _attempt in range(5):
        messages = source_backed_messages(filler, target_chars=target_chars, turn_chars=turn_chars)
        raw_prompt, extraction = render_complete_assistant_boundary(port, messages)
        tokens = list(probe.tokenize(port, raw_prompt))
        sizing.append({"chars": target_chars, "tokens": len(tokens)})
        if minimum <= len(tokens) <= maximum:
            break
        if not tokens:
            raise ValueError("canonical tokenizer returned an empty vector")
        target_chars = min(len(filler), max(turn_chars, int(target_chars * ((minimum + maximum) / 2) / len(tokens))))
    if not minimum <= len(tokens) <= maximum:
        raise ValueError(f"no canonical boundary in {minimum}-{maximum} tokens; sizing={sizing}")
    if messages[-1]["role"] != "assistant":
        raise ValueError("canonical base must end on a complete assistant acknowledgement")
    # Render every complete assistant-message prefix with the same pinned
    # append-a-probe/cut method used for the final boundary.  These vectors are
    # semantic checkpoints (not arbitrary 512-token transport chunks) and are
    # retained only in the private checkpoint.
    assistant_records = derive_assistant_boundary_records(
        port, messages, final_tokens=tokens
    )
    message_manifest = [
        {
            "index": index,
            "role": message["role"],
            "content_chars": len(message["content"]),
            "content_sha256": sha256_text(message["content"]),
        }
        for index, message in enumerate(messages, 1)
    ]
    return {
        "messages": messages,
        "raw_prompt": raw_prompt,
        "tokens": tokens,
        "token_count": len(tokens),
        "token_ids_sha256": sha256_tokens(tokens),
        "raw_prompt_sha256": sha256_text(raw_prompt),
        "messages_sha256": sha256_json(messages),
        "message_manifest": message_manifest,
        "source_manifest": {
            **corpus_manifest,
            "filler_sha256": sha256_text(filler),
            "selected_chars": target_chars,
            "redistribute_source_text": False,
        },
        "sizing": sizing,
        "complete_assistant_boundary": True,
        "canonical_chat_template_boundary": True,
        "synthetic_literal_continuation": False,
        "boundary_proof": {
            **extraction,
            "message_count": len(messages),
            "token_count": len(tokens),
            "token_ids_sha256": sha256_tokens(tokens),
            "raw_prompt_sha256": sha256_text(raw_prompt),
            "rendered_once_per_sizing_attempt": True,
            "base_token_vector_task_prefix_gate_required": True,
        },
        "_assistant_boundary_records": assistant_records,
    }


def derive_assistant_boundary_records(
    port: int,
    messages: Sequence[Mapping[str, str]],
    *,
    final_tokens: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Derive complete assistant boundaries and prove exact final ancestry.

    The Qwen template cannot render a transcript ending in ``assistant``
    directly, so each prefix goes through :func:`render_complete_assistant_boundary`.
    Every resulting token vector must be an exact prefix of the final vector;
    otherwise staging would silently change the prompt history.
    """

    if not messages:
        raise ValueError("cannot derive boundaries from an empty message list")
    final_vector = list(final_tokens) if final_tokens is not None else None
    probe, _, _ = reference_modules()
    records: list[dict[str, Any]] = []
    previous_count = 0
    for message_index, message in enumerate(messages, 1):
        if message.get("role") != "assistant":
            continue
        raw_prompt, proof = render_complete_assistant_boundary(
            port, messages[:message_index]
        )
        tokens = list(probe.tokenize(port, raw_prompt))
        if not tokens:
            raise ValueError(f"empty token vector at assistant boundary {message_index}")
        if len(tokens) <= previous_count:
            raise ValueError("assistant boundaries are not strictly increasing")
        if final_vector is not None and final_vector[: len(tokens)] != tokens:
            common = _common_prefix(final_vector, tokens)
            raise ValueError(
                f"assistant boundary {message_index} is not an exact prefix of final vector "
                f"(common={common}, boundary={len(tokens)}, final={len(final_vector)})"
            )
        records.append(
            {
                "message_index": message_index,
                "message_count": message_index,
                "last_role": "assistant",
                "raw_prompt": raw_prompt,
                "raw_prompt_sha256": sha256_text(raw_prompt),
                "tokens": tokens,
                "token_count": len(tokens),
                "token_ids_sha256": sha256_tokens(tokens),
                "previous_token_count": previous_count,
                "exact_previous_prefix": True,
                "exact_final_prefix": final_vector is None or final_vector[: len(tokens)] == tokens,
                "complete_assistant_boundary": bool(proof.get("complete_assistant_boundary")),
                "boundary_proof": proof,
            }
        )
        previous_count = len(tokens)
    if not records:
        raise ValueError("canonical history contains no assistant boundaries")
    if final_vector is not None and records[-1]["tokens"] != final_vector:
        raise ValueError("final assistant boundary does not equal canonical vector")
    return records


def select_stage_boundaries(
    records: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[int] = (16_384, 32_768, 49_152),
    minimum: int = MIN_BOUNDARY_TOKENS,
    maximum: int = MAX_BOUNDARY_TOKENS,
) -> list[dict[str, Any]]:
    """Select strictly increasing assistant boundaries for staged warming.

    For each target, choose the largest complete assistant boundary not above
    that target.  The final stage is the largest boundary in the canonical
    59K--60K range.  All vectors are copied so callers cannot mutate the
    private boundary chain accidentally.
    """

    if not targets or any(int(target) <= 0 for target in targets):
        raise ValueError("stage targets must be positive")
    if list(targets) != sorted(set(int(item) for item in targets)):
        raise ValueError("stage targets must be strictly increasing")
    assert_exact_message_prefixes(records)
    valid = [
        record for record in records
        if record.get("last_role") == "assistant"
        and record.get("exact_previous_prefix") is True
        and record.get("complete_assistant_boundary") is True
    ]
    if not valid:
        raise ValueError("no complete assistant boundaries available for staging")
    selected: list[dict[str, Any]] = []
    previous_count = 0
    for target in targets:
        candidates = [item for item in valid if previous_count < int(item.get("token_count", 0)) <= int(target)]
        if not candidates:
            raise ValueError(f"no increasing assistant boundary at or below stage target {target}")
        picked = max(candidates, key=lambda item: int(item["token_count"]))
        copied = dict(picked)
        copied["tokens"] = list(picked["tokens"])
        copied["stage"] = len(selected) + 1
        copied["stage_target_tokens"] = int(target)
        copied["selected_reason"] = "largest_complete_assistant_boundary_at_or_below_target"
        selected.append(copied)
        previous_count = int(copied["token_count"])
    final = select_canonical_boundary(records, minimum=minimum, maximum=maximum)
    if int(final["token_count"]) <= previous_count:
        raise ValueError("final canonical boundary is not strictly after intermediate stages")
    final["stage"] = len(selected) + 1
    final["stage_target_tokens"] = int(final["token_count"])
    final["selected_reason"] = "largest_complete_assistant_boundary_in_canonical_range"
    selected.append(final)
    for left, right in zip(selected, selected[1:]):
        if list(right["tokens"])[: int(left["token_count"])] != list(left["tokens"]):
            raise ValueError("selected stage token vectors are not exact prefixes")
    return selected


def suffix_chunk_ranges(
    previous_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    chunk_tokens: int = CHUNK_TOKENS,
) -> list[dict[str, Any]]:
    """Return unseen suffix chunks while proving the prior prefix is intact."""

    previous = list(previous_tokens)
    target = list(target_tokens)
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    if target[: len(previous)] != previous:
        raise ValueError("target stage is not an exact prefix extension")
    ranges: list[dict[str, Any]] = []
    start = len(previous)
    while start < len(target):
        end = min(len(target), start + int(chunk_tokens))
        suffix = target[start:end]
        ranges.append({
            "start": start,
            "end": end,
            "delta_tokens": len(suffix),
            "suffix_tokens": suffix,
            # llama-server's cache accounting compares the complete prompt;
            # only this suffix is processed (cache_n remains ``start``).
            "prompt_tokens": target[:end],
            "prompt_token_count": end,
            "expected_cache_n": start,
            "expected_prompt_n": len(suffix),
        })
        start = end
    return ranges


def submit_unseen_suffix_chunks(
    port: int,
    previous_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    chunk_tokens: int = CHUNK_TOKENS,
    poster: Callable[[int, str, dict[str, Any], int], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Submit cache-aware chunks and reject any prefix replay/accounting drift."""

    if poster is None:
        probe, _, _ = reference_modules()
        poster = probe.post_json
    submitted: list[dict[str, Any]] = []
    for item in suffix_chunk_ranges(
        previous_tokens, target_tokens, chunk_tokens=chunk_tokens
    ):
        response = poster(
            port,
            "/completion",
            completion_request_body(item["prompt_tokens"]),
            REQUEST_TIMEOUT_SECONDS,
        )
        accounting = verify_completion_accounting(
            response,
            item["prompt_token_count"],
            item["expected_cache_n"],
        )
        record = {
            **{
                key: value
                for key, value in item.items()
                if key not in {"prompt_tokens", "suffix_tokens"}
            },
            "prompt_token_ids_sha256": sha256_tokens(item["prompt_tokens"]),
            "suffix_token_ids_sha256": sha256_tokens(item["suffix_tokens"]),
            "accounting": accounting,
            "pass": accounting["pass"],
        }
        submitted.append(record)
        if not accounting["pass"]:
            raise ValueError(
                f"suffix cache accounting failed at {item['start']}..{item['end']}: {accounting}"
            )
    return submitted


def build_generation_task(
    base: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    renderer: Callable[..., str],
    tokenizer: Callable[[str], Sequence[int]],
) -> dict[str, Any]:
    """Render a final user task and require the saved base vector as a prefix."""

    messages = append_generation_prompt(base["messages"], fixture)
    generated = generation_prompt_tokens(messages, renderer, tokenizer)
    base_tokens = list(base["tokens"])
    if generated["tokens"][: len(base_tokens)] != base_tokens:
        raise ValueError("base token vector is not an exact prefix of task prompt")
    generated.update({"base_token_count": len(base_tokens), "base_token_ids_sha256": sha256_tokens(base_tokens), "base_token_prefix_exact": True})
    return generated


def candidate_profile_args(mode: str, slot_dir: Path | None = None) -> list[str]:
    """Construct the canonical command from the one daily-profile source."""

    if mode not in {"target", "mtp"}:
        raise ValueError("mode must be target or mtp")
    profile_tool = daily_profile_module()
    args = list(profile_tool.build_command(mode))
    if slot_dir is not None:
        flag = "--slot-save-path"
        if flag not in args:
            raise ValueError("daily profile command has no --slot-save-path")
        args[args.index(flag) + 1] = str(Path(slot_dir).resolve())
    return args


def profile_environment(mode: str) -> dict[str, str]:
    """Keep MTP environment knobs explicit and clear them for target-only."""

    if mode not in {"target", "mtp"}:
        raise ValueError("mode must be target or mtp")
    profile = daily_profile_module().load_profile()
    env = dict(os.environ)
    for name in ("QWEN38_MTP_UBATCH", "GGML_CUDA_MOE_CACHE_MB", "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD", "QWEN38_WORKING_SET_CAP_GIB"):
        env.pop(name, None)
    env["QWEN38_WORKING_SET_CAP_GIB"] = str(WORKING_SET_CAP_GIB)
    if mode == "mtp":
        for name in ("QWEN38_MTP_UBATCH", "GGML_CUDA_MOE_CACHE_MB", "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD"):
            env[name] = str(profile["environment"][name])
    return env


def git_output(*args: str) -> str:
    """Return a candidate-worktree git value, preserving command evidence."""

    return subprocess.check_output(
        ["git", "-C", str(CANDIDATE_WORKTREE), *args],
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    ).strip()


def validate_candidate_identity() -> dict[str, Any]:
    """Validate every runtime and model artifact before an owned launch."""

    if not CANDIDATE_WORKTREE.is_dir() or not CANDIDATE_BIN.is_dir():
        raise FileNotFoundError(f"missing candidate runtime worktree/bin: {CANDIDATE_WORKTREE}")
    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    if commit != CANDIDATE_RUNTIME_COMMIT:
        raise RuntimeError(f"unexpected candidate runtime commit: {commit}")
    if status:
        raise RuntimeError(f"candidate runtime worktree is dirty: {status}")
    runtime_files: dict[str, Any] = {}
    for name, expected in CANDIDATE_RUNTIME_FILES.items():
        path = CANDIDATE_BIN / name
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate runtime dependency: {path}")
        actual = sha256_file(path)
        if actual.upper() != expected.upper():
            raise RuntimeError(f"runtime bundle hash mismatch for {name}: {actual}")
        runtime_files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}

    shards = sorted(TARGET_MODEL.parent.glob("*.gguf"))
    if not TARGET_MODEL.is_file() or len(shards) != CANDIDATE_TARGET_SHARD_COUNT:
        raise RuntimeError(f"target shard count mismatch: observed {len(shards)}")
    target_bytes = sum(path.stat().st_size for path in shards)
    if target_bytes != CANDIDATE_TARGET_BYTES:
        raise RuntimeError(f"target aggregate bytes mismatch: {target_bytes}")
    first_hash = sha256_file(TARGET_MODEL)
    if first_hash.upper() != CANDIDATE_TARGET_FIRST_SHA256.upper():
        raise RuntimeError(f"target first-shard hash mismatch: {first_hash}")
    if not MTP_SIDECAR.is_file():
        raise FileNotFoundError(f"missing MTP sidecar: {MTP_SIDECAR}")
    if MTP_SIDECAR.stat().st_size != CANDIDATE_SIDECAR_BYTES:
        raise RuntimeError(f"MTP sidecar byte length mismatch: {MTP_SIDECAR.stat().st_size}")
    sidecar_hash = sha256_file(MTP_SIDECAR)
    if sidecar_hash.upper() != CANDIDATE_SIDECAR_SHA256.upper():
        raise RuntimeError(f"MTP sidecar hash mismatch: {sidecar_hash}")
    return {
        "commit": commit,
        "worktree": str(CANDIDATE_WORKTREE),
        "worktree_status": status,
        "runtime_files": runtime_files,
        "target": {
            "first_shard": str(TARGET_MODEL),
            "first_shard_bytes": TARGET_MODEL.stat().st_size,
            "first_shard_sha256": first_hash,
            "shard_count": len(shards),
            "aggregate_bytes": target_bytes,
            "expected_shard_count": CANDIDATE_TARGET_SHARD_COUNT,
            "expected_aggregate_bytes": CANDIDATE_TARGET_BYTES,
        },
        "sidecar": {
            "path": str(MTP_SIDECAR),
            "bytes": MTP_SIDECAR.stat().st_size,
            "sha256": sidecar_hash,
        },
    }


def validate_live_inputs(fixture_path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Fail closed on profile, runtime, artifact, and source-worktree drift."""

    profile_tool = daily_profile_module()
    # The detailed identity pass below hashes every runtime/model artifact once.
    # This profile pass validates settings, paths, commit, shard count, and sizes
    # without rereading the multi-gigabyte payloads a second time.
    validation = dict(profile_tool.validate_profile(check_files=True, check_hashes=False))
    probe, _, _ = reference_modules()
    source_worktree = Path(probe.WORKTREE)
    source_head = subprocess.check_output(
        ["git", "-C", str(source_worktree), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    source_dirty = subprocess.check_output(
        ["git", "-C", str(source_worktree), "status", "--porcelain"],
        text=True,
        encoding="utf-8",
    ).strip()
    if source_dirty:
        raise RuntimeError("canonical source worktree is dirty")
    if source_head != SOURCE_WORKTREE_COMMIT:
        raise RuntimeError(
            f"canonical source worktree drifted: {source_head} != {SOURCE_WORKTREE_COMMIT}"
        )
    validation.update(
        {
            "source_worktree": repo_relative(source_worktree),
            "source_commit": source_head,
            "expected_source_commit": SOURCE_WORKTREE_COMMIT,
            "runtime_executable_sha256": sha256_file(CANDIDATE_EXE),
            "harness_source": harness_source_identity(fixture_path),
            "validated_utc_unix": time.time(),
        }
    )
    return validation


def post_json(port: int, path: str, body: Mapping[str, Any], timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    probe, _, _ = reference_modules()
    response = probe.post_json(port, path, dict(body), timeout)
    return dict(response) if isinstance(response, Mapping) else {"result": response}


def slot_save(port: int, filename: str, slot_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    response = post_json(port, "/slots/0?action=save", {"filename": filename})
    path = Path(slot_dir) / filename
    dft = Path(f"{path}.dft")
    record = {
        "response": response,
        "filename": filename,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "n_saved": response.get("n_saved"),
        "n_written": response.get("n_written"),
        "wall_s": time.perf_counter() - started,
        "companion_dft_after_save": {
            "path": str(dft), "exists": dft.is_file(),
            "bytes": dft.stat().st_size if dft.is_file() else None,
            "sha256": sha256_file(dft) if dft.is_file() else None,
        },
    }
    if not record["exists"]:
        raise RuntimeError(f"slot save did not create {path}")
    return record


def slot_restore(port: int, filename: str, slot_dir: Path) -> dict[str, Any]:
    path = Path(slot_dir) / filename
    started = time.perf_counter()
    response = post_json(port, "/slots/0?action=restore", {"filename": filename})
    return {
        "response": response, "filename": filename, "path": str(path),
        "exists": path.is_file(), "n_restored": response.get("n_restored"),
        "n_read": response.get("n_read"), "restore_ms": (response.get("timings") or {}).get("restore_ms") or response.get("restore_ms"),
        "wall_s": time.perf_counter() - started,
    }


def require_exact_restore(record: Mapping[str, Any], expected_tokens: int, *, expected_target_bytes: int | None = None, expected_draft_bytes: int | None = None) -> None:
    if not record.get("exists"):
        raise RuntimeError("slot restore target file is missing")
    if int(record.get("n_restored") or -1) != int(expected_tokens):
        raise RuntimeError(
            f"slot restore count {record.get('n_restored')} != {expected_tokens}"
        )
    if expected_target_bytes is not None:
        restore_path = record.get("path")
        if not restore_path or not Path(str(restore_path)).is_file() or Path(str(restore_path)).stat().st_size != int(expected_target_bytes):
            raise RuntimeError("slot restore target byte count changed")
        if int(record.get("n_read") or -1) != int(expected_target_bytes):
            raise RuntimeError(
                f"slot restore read {record.get('n_read')} bytes, expected {expected_target_bytes}"
            )
    if expected_draft_bytes is not None:
        draft = Path(f"{record.get('path')}.dft")
        if not draft.is_file() or draft.stat().st_size != int(expected_draft_bytes):
            raise RuntimeError("slot restore draft byte count changed")


def _gpu_snapshot() -> dict[str, int]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()
    if not output: raise RuntimeError("nvidia-smi returned no GPU")
    used, free, util = [int(value.strip()) for value in output[0].split(",")[:3]]
    return {"used_mib": used, "free_mib": free, "util_pct": util}


def resource_snapshot(process: Any = None) -> dict[str, Any]:
    if psutil is None:
        return {"timestamp_unix": time.time(), "error": "psutil unavailable"}
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    sample: dict[str, Any] = {"timestamp_unix": time.time(), "ram_available_bytes": int(memory.available), "ram_total_bytes": int(memory.total), "pagefile_used_bytes": int(swap.used), "pagefile_total_bytes": int(swap.total)}
    try:
        sample["gpu"] = _gpu_snapshot()
    except Exception as exc:
        sample["gpu_error"] = f"{type(exc).__name__}: {exc}"
    sample["network_io"] = {name: {"bytes_sent": int(value.bytes_sent), "bytes_recv": int(value.bytes_recv)} for name, value in psutil.net_io_counters(pernic=True).items()}
    sample["disk_io"] = {name: {"read_bytes": int(value.read_bytes), "write_bytes": int(value.write_bytes)} for name, value in psutil.disk_io_counters(perdisk=True).items()}
    if process is not None:
        try:
            info = process.memory_info(); sample["rss_bytes"] = int(info.rss); sample["vms_bytes"] = int(info.vms)
            io = process.io_counters(); sample["process_io"] = {"read_bytes": int(io.read_bytes), "write_bytes": int(io.write_bytes)}
        except Exception:
            sample["rss_bytes"] = None
    return sample


def preflight_snapshot() -> dict[str, Any]:
    sample = resource_snapshot()
    if not port_free(PORT):
        raise RuntimeError(f"fixed port {PORT} is busy")
    if sample.get("ram_available_bytes", 0) < 40 * 1024**3:
        raise RuntimeError("preflight RAM available below 40 GiB")
    if sample.get("gpu_error"):
        raise RuntimeError(f"GPU telemetry unavailable: {sample['gpu_error']}")
    gpu = sample.get("gpu") or {}
    if gpu.get("free_mib", 0) < 8192:
        raise RuntimeError("preflight free VRAM below 8192 MiB")
    if gpu.get("util_pct", 101) > 15:
        raise RuntimeError("preflight GPU utilization above 15 percent")
    return sample


def port_free(port: int = PORT) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port)); return True
    except OSError: return False
    finally: sock.close()


def stop_owned_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None: return
    process.terminate()
    try: process.wait(15)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(10)


class OwnedServer:
    """A single owned Popen with hard-stop floors and full telemetry."""

    def __init__(
        self,
        mode: str,
        args: list[str],
        log_path: Path,
        telemetry_path: Path,
        env: Mapping[str, str] | None = None,
        *,
        workflow_pagefile_baseline_bytes: int | None = None,
    ):
        if mode not in {"target", "mtp"}:
            raise ValueError("server mode must be target or mtp")
        self.mode, self.args, self.log_path, self.telemetry_path, self.env = mode, args, Path(log_path), Path(telemetry_path), dict(env) if env is not None else os.environ.copy()
        self.process: subprocess.Popen[str] | None = None; self.samples: list[dict[str, Any]] = []; self.violations: list[str] = []
        self._done = threading.Event(); self._thread: threading.Thread | None = None; self.preflight: dict[str, Any] = {}
        self.identity: dict[str, Any] = {}; self.working_set_cap: dict[str, Any] = {}; self.teardown: dict[str, Any] = {}
        self.workflow_pagefile_baseline_bytes = workflow_pagefile_baseline_bytes

    def start(self) -> None:
        self.preflight = preflight_snapshot(); self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("x", encoding="utf-8")
        self.process = subprocess.Popen(self.args, cwd=str(CANDIDATE_BIN), env=self.env, stdout=self._log, stderr=subprocess.STDOUT, text=True)
        try:
            profile_tool = daily_profile_module()
            self.working_set_cap = dict(profile_tool.set_working_set_cap(self.process))
            self.working_set_cap.update({"requested_gib": WORKING_SET_CAP_GIB, "applied_at_unix": time.time()})
        except BaseException:
            stop_owned_process(self.process)
            self._log.close()
            raise
        proc_info = psutil.Process(self.process.pid) if psutil is not None else None
        self.identity = {
            "pid": self.process.pid,
            "create_time": proc_info.create_time() if proc_info is not None else None,
            "executable": str(Path(proc_info.exe()).resolve()) if proc_info is not None else str(CANDIDATE_EXE.resolve()),
            "command_sha256": sha256_json(self.args),
            "runtime_executable_sha256": sha256_file(CANDIDATE_EXE),
            "mode": self.mode,
        }
        def monitor() -> None:
            with self.telemetry_path.open("a", encoding="utf-8") as handle:
                while not self._done.wait(0.5) and self.process is not None and self.process.poll() is None:
                    sample = resource_snapshot(proc_info); sample["mode"] = self.mode; self.samples.append(sample); handle.write(json.dumps(sample) + "\n"); handle.flush()
                    ram, rss = sample.get("ram_available_bytes"), sample.get("rss_bytes"); gpu = sample.get("gpu") or {}; page = sample.get("pagefile_used_bytes")
                    if sample.get("gpu_error"): self.violations.append("gpu_telemetry_unavailable")
                    if ram is not None and ram < HARD_RAM_FLOOR: self.violations.append("ram_available<6GiB")
                    if gpu.get("free_mib") is not None and gpu["free_mib"] < HARD_VRAM_FLOOR_MIB: self.violations.append("vram_free<768MiB")
                    if rss is not None and rss > HARD_RSS_CEILING: self.violations.append("owned_rss>50GiB")
                    if page is not None:
                        if page - self.preflight.get("pagefile_used_bytes", page) >= HARD_SWAP_GROWTH:
                            self.violations.append("pagefile_growth>=1GiB")
                        if (
                            self.workflow_pagefile_baseline_bytes is not None
                            and page - self.workflow_pagefile_baseline_bytes >= HARD_SWAP_GROWTH
                        ):
                            self.violations.append("workflow_pagefile_growth>=1GiB")
                    if self.violations: stop_owned_process(self.process); return
        self._thread = threading.Thread(target=monitor, daemon=True); self._thread.start()

    def close(self) -> None:
        self._done.set(); stop_owned_process(self.process)
        if self._thread is not None: self._thread.join(timeout=3)
        try: self._log.close()
        except Exception: pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not port_free(PORT):
            time.sleep(0.1)
        self.teardown = {
            "owned_pid": self.identity.get("pid"),
            "exit_code": self.process.poll() if self.process is not None else None,
            "port_free": port_free(PORT),
            "intentional_owned_termination": True,
            "violations": sorted(set(self.violations)),
            "resource_summary": self.resource_summary(),
        }

    def resource_summary(self) -> dict[str, Any]:
        ram = [int(item["ram_available_bytes"]) for item in self.samples if item.get("ram_available_bytes") is not None]
        vram = [int((item.get("gpu") or {})["free_mib"]) for item in self.samples if (item.get("gpu") or {}).get("free_mib") is not None]
        rss = [int(item["rss_bytes"]) for item in self.samples if item.get("rss_bytes") is not None]
        page = [int(item["pagefile_used_bytes"]) for item in self.samples if item.get("pagefile_used_bytes") is not None]
        stage_pagefile_growth = (
            max(page) - int(self.preflight.get("pagefile_used_bytes", max(page)))
            if page else None
        )
        workflow_pagefile_growth = (
            max(page) - self.workflow_pagefile_baseline_bytes
            if page and self.workflow_pagefile_baseline_bytes is not None else None
        )
        return {
            "sample_count": len(self.samples),
            "min_ram_available_bytes": min(ram) if ram else None,
            "min_vram_free_mib": min(vram) if vram else None,
            "max_owned_rss_bytes": max(rss) if rss else None,
            "pagefile_growth_bytes": stage_pagefile_growth,
            "workflow_pagefile_growth_bytes": workflow_pagefile_growth,
            "workflow_pagefile_baseline_bytes": self.workflow_pagefile_baseline_bytes,
            "max_pagefile_used_bytes": max(page) if page else None,
            "violations": sorted(set(self.violations)),
            "pass": bool(self.samples) and not self.violations,
        }


class CanonicalBuildFailure(RuntimeError):
    """A build failure carrying the already-allocated evidence paths.

    A staged build can fail after several large checkpoint files have been
    sealed.  The normal CLI failure writer must therefore point at that run's
    checkpoint instead of allocating a second, empty run directory.
    """

    def __init__(
        self,
        message: str,
        *,
        paths: Mapping[str, Any] | None = None,
        stage: str | None = None,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.run_paths = dict(paths or {})
        self.stage = stage
        self.cause = cause


def wait_ready(process: subprocess.Popen[str], port: int = PORT, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None: raise RuntimeError(f"server exited with code {process.poll()}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status in {200, 204}: return
        except Exception: pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready on {port}")


def stream_completion(port: int, prompt_tokens: Sequence[int], *, n_predict: int = 256) -> dict[str, Any]:
    """Stream /completion SSE and timestamp the first content token (TTFT)."""

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps({"prompt": list(prompt_tokens), "n_predict": int(n_predict), "id_slot": 0,
                         "cache_prompt": True, "temperature": 0, "seed": SEED, "stream": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter(); first: float | None = None; chunks: list[str] = []; final: dict[str, Any] = {}; raw_events: list[Any] = []
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        for line in response:
            text = line.decode("utf-8", errors="replace").strip()
            if not text.startswith("data:"): continue
            payload = text[5:].strip()
            if payload == "[DONE]": break
            try: item = json.loads(payload)
            except json.JSONDecodeError: continue
            if not isinstance(item, Mapping): continue
            raw_events.append(dict(item))
            final.update(item)
            content = item.get("content") or item.get("token") or ""
            if not content and isinstance(item.get("choices"), list) and item["choices"]:
                choice = item["choices"][0] or {}; delta = choice.get("delta") or {}
                content = delta.get("content") or choice.get("text") or ""
            if content:
                if first is None: first = time.perf_counter()
                chunks.append(str(content))
    output = "".join(chunks)
    finish = final.get("stop_type") or final.get("finish_reason")
    return {"output": output, "raw_response": final, "raw_sse_events": raw_events, "ttft_s": first - started if first else None,
            "wall_s": time.perf_counter() - started, "finish_reason": finish,
            "natural_stop": finish in {"eos", "word", "stop"}, "output_sha256": sha256_text(output),
            "timings": final.get("timings") if isinstance(final.get("timings"), Mapping) else {}}


def benchmark_one_task(
    port: int,
    base: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Render/tokenize one final task, stream output, retokenize, and score."""

    probe, _, realistic = reference_modules()
    messages = append_generation_prompt(base["messages"], fixture)
    raw = post_apply_template(port, messages, add_generation_prompt=True)
    task_tokens = list(probe.tokenize(port, raw))
    base_tokens = list(base["tokens"])
    if task_tokens[: len(base_tokens)] != base_tokens:
        raise ValueError("base token vector is not an exact prefix of final task prompt")
    suffix_tokens = task_tokens[len(base_tokens) :]
    if not suffix_tokens:
        raise ValueError("benchmark task produced an empty suffix")
    before = resource_snapshot()
    streamed = stream_completion(port, task_tokens, n_predict=int(fixture.get("max_tokens", 256)))
    output_tokens = list(probe.tokenize(port, streamed["output"]))
    scorer = realistic.score_fixture
    semantic = dict(scorer(dict(fixture), streamed["output"])) if fixture.get("scorer") != "retrieval_needle" else {"valid": streamed["output"].strip() == fixture.get("needle"), "semantic_vector": {"needle_exact": streamed["output"].strip() == fixture.get("needle")}}
    timings = streamed.get("timings") or {}; response = streamed.get("raw_response") or {}
    def metric(*keys: str) -> Any:
        for key in keys:
            if timings.get(key) is not None: return timings[key]
            if response.get(key) is not None: return response[key]
        return None
    cache_n = metric("cache_n")
    prompt_n = metric("prompt_n")
    cache_exact = cache_n is not None and int(cache_n) == len(base_tokens)
    suffix_exact = prompt_n is not None and int(prompt_n) == len(suffix_tokens)
    return {"mode": mode, "fixture_id": fixture.get("id"), "fixture_category": fixture.get("category"), "prompt_tokens": len(task_tokens), "base_tokens": len(base_tokens),
            "prompt_token_ids_sha256": sha256_tokens(task_tokens), "output": streamed["output"], "output_sha256": streamed["output_sha256"],
            "base_token_ids_sha256": sha256_tokens(base_tokens), "suffix_tokens": len(suffix_tokens), "suffix_token_ids_sha256": sha256_tokens(suffix_tokens),
            "output_token_count": len(output_tokens), "output_token_ids_sha256": sha256_tokens(output_tokens), "raw_sse_events": streamed.get("raw_sse_events", []),
            "retokenized_token_ids_sha256": sha256_tokens(output_tokens), "ttft_s": streamed.get("ttft_s"),
            "wall_s": streamed.get("wall_s"), "finish_reason": streamed.get("finish_reason"), "natural_stop": streamed.get("natural_stop"),
            "timings": timings, "prefill_tps": metric("prompt_per_second", "prefill_tps"), "decode_tps": metric("predicted_per_second", "decode_tps"),
            "prompt_n": prompt_n, "cache_n": cache_n, "expected_cache_n": len(base_tokens), "expected_prompt_n": len(suffix_tokens),
            "cache_reuse_exact": cache_exact, "suffix_processing_exact": suffix_exact,
            "draft_n": int(metric("draft_n", "n_draft_tokens") or 0),
            "draft_n_accepted": int(metric("draft_n_accepted", "n_draft_accepted") or 0), "semantic_score": semantic,
            "semantic_vector": semantic.get("semantic_vector"), "resource_before": before, "resource_after": resource_snapshot(),
            "request_ok": True}


def allocate_run_paths(tag: str) -> dict[str, Path | str]:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip(".-")[:96]
    if not clean: raise ValueError("tag must contain an alphanumeric character")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True); log_root = ROOT / "logs" / "QWEN38-MTP-PROTOTYPE-001" / "q3ple_canonical_60k"; log_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        run_tag = f"canonical-60k-{clean}-r{index}"; run_dir = log_root / run_tag; result = RESULT_ROOT / f"q3ple_{run_tag.replace('-', '_')}.json"
        if run_dir.exists() or result.exists(): continue
        run_dir.mkdir(); slots = run_dir / "slots"; slots.mkdir()
        return {"run_dir": run_dir, "result": result, "checkpoint": run_dir / "checkpoint.json", "slot_dir": slots, "slot_filename": f"{run_tag}.slot.bin", "target_log": run_dir / "target.log", "mtp_log": run_dir / "mtp.log", "restart_log": run_dir / "restart.log", "client_log": run_dir / "client.jsonl", "telemetry": run_dir / "telemetry.jsonl", "evidence_dir": run_dir / "evidence"}
    raise RuntimeError("no unique canonical run path")


def public_base_record(base: Mapping[str, Any]) -> dict[str, Any]:
    """Remove source text and raw token IDs while retaining their sealed hashes."""

    private_fields = {
        "messages", "raw_prompt", "tokens", "_prefix_records",
        "_assistant_boundary_records",
    }
    result = {key: value for key, value in base.items() if key not in private_fields}
    result["source_text_in_public_result"] = False
    result["raw_token_ids_in_public_result"] = False
    return result


def public_pair_record(pair: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(pair)
    for key in ("target_path", "draft_path"):
        if result.get(key):
            result[key] = repo_relative(result[key])
    return result


def public_stage_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Drop literal token arrays from staged evidence exposed in the result."""

    private_keys = {"tokens", "prompt_tokens", "suffix_tokens", "raw_prompt"}
    result: dict[str, Any] = {}
    for key, value in record.items():
        if key in private_keys:
            continue
        if key == "suffix_submissions" and isinstance(value, Sequence):
            result[key] = [
                {subkey: subvalue for subkey, subvalue in item.items() if subkey not in private_keys}
                for item in value if isinstance(item, Mapping)
            ]
        else:
            result[key] = value
    return result


def staged_ingestion_proof(
    stage_records: Sequence[Mapping[str, Any]], final_tokens: Sequence[int]
) -> dict[str, Any]:
    """Recompute the staged exact-prefix/cache proof from private evidence."""

    final = list(final_tokens)
    if not final or not stage_records:
        raise ValueError("staged ingestion needs final tokens and at least one stage")
    previous_count = 0
    boundaries: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(stage_records, 1):
        if int(stage.get("stage", -1)) != stage_index:
            raise ValueError("stage indices are not contiguous")
        count = int(stage.get("token_count", -1))
        if count <= previous_count or count > len(final):
            raise ValueError(f"stage {stage_index} token count is not increasing")
        expected_stage_hash = sha256_tokens(final[:count])
        if str(stage.get("token_ids_sha256", "")).lower() != expected_stage_hash.lower():
            raise ValueError(f"stage {stage_index} token hash differs from final prefix")
        if stage.get("complete_assistant_boundary") is not True:
            raise ValueError(f"stage {stage_index} is not a complete assistant boundary")
        if stage.get("pass") is not True or stage.get("sealed") is not True:
            raise ValueError(f"stage {stage_index} did not pass its seal gate")
        pair = stage.get("pair")
        if not isinstance(pair, Mapping) or pair.get("pair_promotion_pass") is not True:
            raise ValueError(f"stage {stage_index} lacks a promoted target/.dft pair")
        if stage_index > 1:
            restore = stage.get("restore")
            if not isinstance(restore, Mapping) or int(restore.get("n_restored") or -1) != previous_count:
                raise ValueError(f"stage {stage_index} did not restore the prior boundary exactly")
            restore_probe = stage.get("restore_suffix_probe")
            if not isinstance(restore_probe, Mapping) or restore_probe.get("pass") is not True:
                raise ValueError(f"stage {stage_index} first-suffix restore probe failed")
            if (
                int(restore_probe.get("start", -1)) != previous_count
                or int(restore_probe.get("expected_cache_n", -1)) != previous_count
                or int(restore_probe.get("expected_prompt_n", -1)) <= 0
            ):
                raise ValueError(f"stage {stage_index} first-suffix restore probe drifted")
        submissions = stage.get("suffix_submissions")
        if not isinstance(submissions, list) or not submissions:
            raise ValueError(f"stage {stage_index} lacks suffix submissions")
        if stage_index > 1:
            first = submissions[0]
            restore_probe = stage["restore_suffix_probe"]
            for key in (
                "start", "end", "delta_tokens", "expected_cache_n",
                "expected_prompt_n", "prompt_token_ids_sha256",
                "suffix_token_ids_sha256",
            ):
                if restore_probe.get(key) != first.get(key):
                    raise ValueError(f"stage {stage_index} first-suffix restore probe differs from first submission")
        cursor = previous_count
        for chunk_index, submission in enumerate(submissions, 1):
            if not isinstance(submission, Mapping):
                raise ValueError(f"stage {stage_index} chunk {chunk_index} is malformed")
            start = int(submission.get("start", -1))
            end = int(submission.get("end", -1))
            delta = int(submission.get("delta_tokens", -1))
            if start != cursor or end <= start or end > count or delta != end - start:
                raise ValueError(f"stage {stage_index} chunk {chunk_index} is not contiguous")
            if int(submission.get("expected_cache_n", -1)) != start or int(submission.get("expected_prompt_n", -1)) != delta:
                raise ValueError(f"stage {stage_index} chunk {chunk_index} expected accounting drifted")
            accounting = submission.get("accounting")
            if (
                submission.get("pass") is not True
                or not isinstance(accounting, Mapping)
                or accounting.get("pass") is not True
                or int(accounting.get("cache_n", -1)) != start
                or int(accounting.get("prompt_n", -1)) != delta
            ):
                raise ValueError(f"stage {stage_index} chunk {chunk_index} cache accounting failed")
            if str(submission.get("prompt_token_ids_sha256", "")).lower() != sha256_tokens(final[:end]).lower():
                raise ValueError(f"stage {stage_index} chunk {chunk_index} prompt hash drifted")
            if str(submission.get("suffix_token_ids_sha256", "")).lower() != sha256_tokens(final[start:end]).lower():
                raise ValueError(f"stage {stage_index} chunk {chunk_index} suffix hash drifted")
            cursor = end
        if cursor != count:
            raise ValueError(f"stage {stage_index} suffix submissions do not reach its boundary")
        boundaries.append(
            {
                "stage": stage_index,
                "message_index": int(stage.get("message_index", -1)),
                "token_count": count,
                "token_ids_sha256": expected_stage_hash,
                "suffix_submission_count": len(submissions),
                "restored_previous_boundary": stage_index > 1,
                "complete_assistant_boundary": True,
            }
        )
        previous_count = count
    if previous_count != len(final):
        raise ValueError("final staged checkpoint does not equal the canonical vector")
    return {
        "schema_version": "q3ple-canonical-staged-ingestion-proof-v1",
        "stage_count": len(boundaries),
        "stage_boundaries": boundaries,
        "final_token_count": len(final),
        "final_token_ids_sha256": sha256_tokens(final),
        "all_complete_assistant_boundaries": True,
        "all_prefix_hashes_exact": True,
        "all_cache_accounting_exact": True,
        "restart_between_stages": True,
        "transport_chunks_may_end_inside_messages": True,
        "synthetic_transcript_used": False,
    }


def load_sealed_base(base_result: Path) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    payload = json.loads(Path(base_result).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q3ple-canonical-60k-build-v2" or payload.get("status") != "PASS" or not payload.get("local_build_gate"):
        raise ValueError("base result did not pass the canonical build gate")
    checkpoint_value = payload.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError("base result has no private checkpoint reference")
    checkpoint_path = resolve_repo_path(checkpoint_value)
    private_root = (ROOT / "logs" / "QWEN38-MTP-PROTOTYPE-001" / "q3ple_canonical_60k").resolve()
    try:
        checkpoint_path.relative_to(private_root)
    except ValueError as error:
        raise ValueError("private checkpoint is outside the canonical run root") from error
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != "q3ple-canonical-60k-private-v2" or checkpoint.get("phase") != "complete":
        raise ValueError("private checkpoint is not a completed staged build")
    base = checkpoint.get("base")
    pair = checkpoint.get("slot_pair")
    if not isinstance(base, dict) or not isinstance(base.get("tokens"), list):
        raise ValueError("private checkpoint lacks canonical base tokens")
    if (
        not isinstance(base.get("messages"), list)
        or base.get("complete_assistant_boundary") is not True
    ):
        raise ValueError("private checkpoint is not a complete assistant boundary")
    if not isinstance(pair, dict) or not pair.get("pair_promotion_pass"):
        raise ValueError("private checkpoint lacks a validated target/.dft pair")
    if not MIN_BOUNDARY_TOKENS <= len(base["tokens"]) <= MAX_BOUNDARY_TOKENS:
        raise ValueError("private checkpoint token count is outside the canonical range")
    if base.get("canonical_chat_template_boundary") is not True or base.get("synthetic_literal_continuation") is not False:
        raise ValueError("private checkpoint does not prove the canonical non-synthetic boundary")
    public_base = payload.get("base") or {}
    if public_base.get("token_ids_sha256") != sha256_tokens(base["tokens"]):
        raise ValueError("public base token hash differs from private checkpoint")
    stages = checkpoint.get("stage_records")
    if not isinstance(stages, list):
        raise ValueError("private checkpoint lacks staged canonical evidence")
    ingestion_proof = staged_ingestion_proof(stages, base["tokens"])
    stored_proof = checkpoint.get("ingestion_proof")
    if not isinstance(stored_proof, Mapping) or sha256_json(stored_proof) != sha256_json(ingestion_proof):
        raise ValueError("private checkpoint ingestion proof is missing or inconsistent")
    public_ingestion = payload.get("ingestion_proof")
    if not isinstance(public_ingestion, Mapping) or sha256_json(public_ingestion) != sha256_json(ingestion_proof):
        raise ValueError("public result ingestion proof differs from private checkpoint")
    final_stage = stages[-1] if stages else None
    final_stage_pair = final_stage.get("pair") if isinstance(final_stage, Mapping) else None
    if not isinstance(final_stage_pair, Mapping) or any(
        final_stage_pair.get(field) != pair.get(field)
        for field in ("target_sha256", "draft_sha256", "target_bytes", "draft_bytes")
    ):
        raise ValueError("final staged pair differs from the checkpoint pair")
    public_pair = payload.get("slot_pair")
    if not isinstance(public_pair, Mapping) or any(
        public_pair.get(field) != pair.get(field)
        for field in ("target_sha256", "draft_sha256", "target_bytes", "draft_bytes")
    ):
        raise ValueError("public slot pair differs from the private checkpoint")
    target_path = resolve_repo_path(pair["target_path"])
    try:
        target_path.relative_to(private_root)
    except ValueError as error:
        raise ValueError("sealed slot pair is outside the canonical run root") from error
    if target_path.parent.name != "slots" or target_path.suffixes[-2:] != [".slot", ".bin"]:
        raise ValueError("sealed slot pair filename is invalid")
    if target_path.parent != checkpoint_path.parent / "slots":
        raise ValueError("sealed slot pair is not owned by its private run checkpoint")
    expected_slot_filename = target_path.name
    if checkpoint.get("slot_filename") != expected_slot_filename:
        raise ValueError("private checkpoint slot filename differs from its sealed pair")
    if payload.get("slot_filename") != expected_slot_filename:
        raise ValueError("public result slot filename differs from its sealed pair")
    if not isinstance(final_stage, Mapping) or final_stage.get("slot_filename") != expected_slot_filename:
        raise ValueError("final staged slot filename differs from the sealed pair")
    return payload, base, target_path.parent, target_path.name


def combined_resource_summary(
    *summaries: Mapping[str, Any],
    workflow_pagefile_baseline_bytes: int | None = None,
) -> dict[str, Any]:
    """Aggregate stage/restart resources with one workflow pagefile baseline.

    Per-process ``pagefile_growth_bytes`` is useful diagnostically, but it can
    hide cumulative growth when every fresh stage takes its own baseline.  Live
    staged builds therefore pass the pre-stage-1 pagefile usage and each server
    summary's maximum observed usage to this function.
    """

    # Accept either flat summaries or stage records containing ``resources``;
    # this keeps the helper convenient for offline tests and old callers.
    valid: list[dict[str, Any]] = []
    for item in summaries:
        if not item:
            continue
        nested = item.get("resources") if isinstance(item, Mapping) else None
        valid.append(dict(nested) if isinstance(nested, Mapping) else dict(item))
    vram = [int(item["min_vram_free_mib"]) for item in valid if item.get("min_vram_free_mib") is not None]
    ram = [int(item["min_ram_available_bytes"]) for item in valid if item.get("min_ram_available_bytes") is not None]
    rss = [int(item["max_owned_rss_bytes"]) for item in valid if item.get("max_owned_rss_bytes") is not None]
    page = [int(item["pagefile_growth_bytes"]) for item in valid if item.get("pagefile_growth_bytes") is not None]
    max_pagefile = [int(item["max_pagefile_used_bytes"]) for item in valid if item.get("max_pagefile_used_bytes") is not None]
    global_pagefile_growth: int | None = None
    if workflow_pagefile_baseline_bytes is not None and max_pagefile:
        global_pagefile_growth = max(max_pagefile) - int(workflow_pagefile_baseline_bytes)
        page.append(global_pagefile_growth)
    violation_set = {reason for item in valid for reason in item.get("violations", [])}
    if global_pagefile_growth is not None and global_pagefile_growth >= HARD_SWAP_GROWTH:
        violation_set.add("workflow_pagefile_growth>=1GiB")
    violations = sorted(violation_set)
    return {
        "sample_count": sum(int(item.get("sample_count", 0)) for item in valid),
        "min_vram_free_mib": min(vram) if vram else None,
        "min_ram_available_bytes": min(ram) if ram else None,
        "max_owned_rss_bytes": max(rss) if rss else None,
        "pagefile_growth_bytes": max(page) if page else None,
        "workflow_pagefile_baseline_bytes": workflow_pagefile_baseline_bytes,
        "max_pagefile_used_bytes": max(max_pagefile) if max_pagefile else None,
        "global_pagefile_growth_bytes": global_pagefile_growth,
        "violations": violations,
        "pass": bool(valid)
        and all(bool(item.get("pass")) for item in valid)
        and not violations
        and (global_pagefile_growth is None or global_pagefile_growth < HARD_SWAP_GROWTH),
    }


def metric_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "stdev": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _staged_stage_paths(paths: Mapping[str, Any], stage: int, token_count: int) -> dict[str, Path | str]:
    run_dir = Path(paths["run_dir"])
    stem = f"{run_dir.name}.stage{int(stage):02d}-{int(token_count)}"
    return {
        "slot_filename": f"{stem}.slot.bin",
        "log": run_dir / f"stage{int(stage):02d}-{int(token_count)}.log",
        "telemetry": run_dir / f"stage{int(stage):02d}-{int(token_count)}.telemetry.jsonl",
    }


def _execute_staged_stage(
    paths: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    stage_index: int,
    previous: Mapping[str, Any] | None,
    workflow_pagefile_baseline_bytes: int,
) -> dict[str, Any]:
    """Start one fresh server, extend only the unseen suffix, and seal it."""

    stage_count = int(stage["token_count"])
    evidence = _staged_stage_paths(paths, stage_index, stage_count)
    slot_dir = Path(paths["slot_dir"])
    server = OwnedServer(
        "mtp", candidate_profile_args("mtp", slot_dir), Path(evidence["log"]),
        Path(evidence["telemetry"]), profile_environment("mtp"),
        workflow_pagefile_baseline_bytes=workflow_pagefile_baseline_bytes,
    )
    record: dict[str, Any] = {
        "stage": stage_index,
        "stage_target_tokens": stage.get("stage_target_tokens"),
        "token_count": stage_count,
        "token_ids_sha256": stage.get("token_ids_sha256") or sha256_tokens(stage["tokens"]),
        "message_index": stage.get("message_index"),
        "complete_assistant_boundary": stage.get("complete_assistant_boundary", True),
        "slot_filename": evidence["slot_filename"],
        "slot_dir": str(slot_dir),
    }
    error: BaseException | None = None
    try:
        server.start(); wait_ready(server.process)  # type: ignore[arg-type]
        old_tokens = list(previous.get("tokens", [])) if previous else []
        if previous is not None:
            old_pair = previous.get("pair") or {}
            restore = slot_restore(PORT, str(previous["slot_filename"]), slot_dir)
            require_exact_restore(
                restore, len(old_tokens),
                expected_target_bytes=int(old_pair.get("target_bytes", 0)) or None,
                expected_draft_bytes=int(old_pair.get("draft_bytes", 0)) or None,
            )
            record["restore"] = restore
        suffix = submit_unseen_suffix_chunks(PORT, old_tokens, list(stage["tokens"]))
        record["suffix_submissions"] = suffix
        if previous is not None:
            if not suffix:
                raise RuntimeError(f"stage {stage_index} has no unseen suffix after restore")
            restore_probe = dict(suffix[0])
            if (
                restore_probe.get("pass") is not True
                or int(restore_probe.get("start", -1)) != len(old_tokens)
                or int(restore_probe.get("expected_cache_n", -1)) != len(old_tokens)
                or int(restore_probe.get("expected_prompt_n", -1)) <= 0
            ):
                raise RuntimeError(f"stage {stage_index} first-suffix restore probe failed: {restore_probe}")
            record["restore_suffix_probe"] = restore_probe
        save = slot_save(PORT, str(evidence["slot_filename"]), slot_dir)
        if int(save.get("n_saved") or -1) != stage_count:
            raise RuntimeError(f"stage {stage_index} save count {save.get('n_saved')} != {stage_count}")
        pair = offline_pair_gate(slot_dir / str(evidence["slot_filename"]), stage["tokens"], expected_count=stage_count)
        if not pair.get("pair_promotion_pass"):
            raise RuntimeError(f"stage {stage_index} offline target/.dft pair gate failed")
        if int(save.get("n_written") or -1) != int(pair.get("target_bytes") or -1):
            raise RuntimeError(f"stage {stage_index} save byte count differs from parsed target")
        companion = save.get("companion_dft_after_save") or {}
        if not companion.get("exists") or int(companion.get("bytes") or 0) <= 0:
            raise RuntimeError(f"stage {stage_index} save did not produce a non-empty .dft")
        record.update({"slot_save": save, "pair": pair, "sealed": True})
    except BaseException as exc:
        error = exc
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        server.close()
    record["server"] = {
        "identity": server.identity,
        "working_set_cap": server.working_set_cap,
        "resources": server.resource_summary(),
        "postflight": server.teardown,
    }
    record["pass"] = bool(
        error is None and record.get("sealed")
        and record["server"]["resources"].get("pass")
        and record["server"]["postflight"].get("port_free")
    )
    return record


def _build_live_staged(tag: str, *, turn_chars: int = 12_000) -> dict[str, Any]:
    """Staged exact-prefix canonical build implementation."""

    paths = allocate_run_paths(tag)
    identity = validate_live_inputs()
    identity["candidate_runtime"] = validate_candidate_identity()
    workflow_preflight = preflight_snapshot()
    pagefile_baseline = int(workflow_preflight.get("pagefile_used_bytes", 0))
    checkpoint: dict[str, Any] = {
        "schema_version": "q3ple-canonical-60k-private-v2",
        "phase": "allocated", "identity": identity,
        "workflow_preflight": workflow_preflight,
        "workflow_pagefile_baseline_bytes": pagefile_baseline,
        "stage_records": [],
        "paths": {key: str(value) for key, value in paths.items() if isinstance(value, Path)},
    }
    atomic_write_json(Path(paths["checkpoint"]), checkpoint)
    base: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    probe_server: OwnedServer | None = None
    try:
        # Build/render on a disposable probe process.  Stage 1 itself is always
        # a fresh process started below and begins with an empty slot.
        probe_server = OwnedServer(
            "mtp", candidate_profile_args("mtp", Path(paths["slot_dir"])),
            Path(paths["run_dir"]) / "boundary-probe.log",
            Path(paths["run_dir"]) / "boundary-probe.telemetry.jsonl",
            profile_environment("mtp"),
            workflow_pagefile_baseline_bytes=pagefile_baseline,
        )
        probe_error: BaseException | None = None
        try:
            probe_server.start()
            wait_ready(probe_server.process)  # type: ignore[arg-type]
            base = build_canonical_state(PORT, turn_chars=turn_chars)
        except BaseException as error:
            probe_error = error
        finally:
            probe_server.close()
        checkpoint["boundary_probe_server"] = {
            "identity": probe_server.identity,
            "working_set_cap": probe_server.working_set_cap,
            "resources": probe_server.resource_summary(),
            "postflight": probe_server.teardown,
        }
        probe_workflow = combined_resource_summary(
            checkpoint["boundary_probe_server"]["resources"],
            workflow_pagefile_baseline_bytes=pagefile_baseline,
        )
        checkpoint["workflow_resources_after_probe"] = probe_workflow
        atomic_write_json(Path(paths["checkpoint"]), checkpoint)
        if probe_error is not None:
            raise CanonicalBuildFailure(
                f"canonical boundary probe failed: {probe_error}",
                paths=paths,
                stage="boundary-probe",
                cause=probe_error,
            ) from probe_error
        if (
            not probe_workflow.get("pass")
            or not checkpoint["boundary_probe_server"]["postflight"].get("port_free")
        ):
            raise CanonicalBuildFailure(
                f"canonical boundary probe failed resource gate: {probe_workflow.get('violations', [])}",
                paths=paths,
                stage="boundary-probe",
                cause=RuntimeError(str(probe_workflow.get("violations", []))),
            )
        if base is None:
            raise RuntimeError("canonical tokenization probe produced no base")
        stages = select_stage_boundaries(base.get("_assistant_boundary_records") or [])
        checkpoint.update({
            "phase": "boundaries_selected", "base": base,
            "stage_boundaries": [
                {key: value for key, value in stage.items() if key not in {"messages", "raw_prompt", "tokens"}}
                for stage in stages
            ],
        })
        atomic_write_json(Path(paths["checkpoint"]), checkpoint)
        for index, stage in enumerate(stages, 1):
            record = _execute_staged_stage(
                paths,
                stage,
                stage_index=index,
                previous=previous,
                workflow_pagefile_baseline_bytes=pagefile_baseline,
            )
            checkpoint["stage_records"].append(record)
            rolling_resources = combined_resource_summary(
                checkpoint["boundary_probe_server"]["resources"],
                *(item["server"]["resources"] for item in checkpoint["stage_records"]),
                workflow_pagefile_baseline_bytes=pagefile_baseline,
            )
            record["workflow_resources_after_stage"] = rolling_resources
            atomic_write_json(Path(paths["checkpoint"]), checkpoint)
            if not record.get("pass") or not rolling_resources.get("pass"):
                details = record.get("error") or {"message": "stage evidence gate failed"}
                if not rolling_resources.get("pass"):
                    details = {
                        "message": "workflow resource gate failed: "
                        f"{rolling_resources.get('violations', [])}"
                    }
                raise CanonicalBuildFailure(
                    f"canonical stage {index} failed closed: {details.get('message', details)}",
                    paths=paths, stage=f"stage-{index}",
                    cause=RuntimeError(str(details.get("message", details))),
                )
            previous = {"tokens": list(stage["tokens"]), "slot_filename": record["slot_filename"], "pair": record["pair"]}
        if previous is None or base is None:
            raise RuntimeError("staged build produced no sealed final pair")
        checkpoint.update({"phase": "sealed", "base": base, "slot_pair": previous["pair"], "slot_filename": previous["slot_filename"]})
        atomic_write_json(Path(paths["checkpoint"]), checkpoint)

        restart = OwnedServer(
            "mtp", candidate_profile_args("mtp", Path(paths["slot_dir"])),
            Path(paths["restart_log"]), Path(paths["run_dir"]) / "restart.telemetry.jsonl",
            profile_environment("mtp"),
            workflow_pagefile_baseline_bytes=pagefile_baseline,
        )
        try:
            restart.start(); wait_ready(restart.process)  # type: ignore[arg-type]
            restored = slot_restore(PORT, str(previous["slot_filename"]), Path(paths["slot_dir"]))
            require_exact_restore(
                restored, len(base["tokens"]),
                expected_target_bytes=int(previous["pair"].get("target_bytes", 0)) or None,
                expected_draft_bytes=int(previous["pair"].get("draft_bytes", 0)) or None,
            )
            restart_probe = build_restart_suffix_probe(PORT, base)
            proof = post_json(
                PORT,
                "/completion",
                completion_request_body(restart_probe["prompt_tokens"]),
            )
            accounting = verify_completion_accounting(
                proof, restart_probe["prompt_token_count"], len(base["tokens"])
            )
            if not accounting["pass"]:
                raise RuntimeError(f"restart cache check failed: {accounting}")
            pair_after = offline_pair_gate(Path(paths["slot_dir"]) / str(previous["slot_filename"]), base["tokens"], expected_count=len(base["tokens"]))
            if pair_after["target_sha256"] != previous["pair"]["target_sha256"] or pair_after["draft_sha256"] != previous["pair"]["draft_sha256"]:
                raise RuntimeError("sealed target/.dft pair changed across restart restore")
            checkpoint.update({
                "phase": "restart_restored",
                "slot_restore": restored,
                "restart_suffix_probe": {
                    key: value for key, value in restart_probe.items()
                    if key != "prompt_tokens"
                },
                "restart_prefix_response": proof,
                "restart_cache_check": accounting,
                "slot_pair_after_restore": pair_after,
            })
            atomic_write_json(Path(paths["checkpoint"]), checkpoint)
        finally:
            restart.close()
        checkpoint["restart_server"] = {
            "identity": restart.identity,
            "working_set_cap": restart.working_set_cap,
            "resources": restart.resource_summary(),
            "postflight": restart.teardown,
        }
        summaries = [checkpoint["boundary_probe_server"]["resources"]]
        summaries.extend(record["server"]["resources"] for record in checkpoint["stage_records"])
        summaries.append(checkpoint["restart_server"]["resources"])
        resources = combined_resource_summary(*summaries, workflow_pagefile_baseline_bytes=pagefile_baseline)
        ingestion_proof = staged_ingestion_proof(checkpoint["stage_records"], base["tokens"])
        checkpoint.update({"resources": resources, "ingestion_proof": ingestion_proof, "phase": "complete"})
        atomic_write_json(Path(paths["checkpoint"]), checkpoint)
        local_gate = bool(
            resources.get("pass") and int(resources.get("min_vram_free_mib") or 0) >= PUBLISHABLE_VRAM_FLOOR_MIB
            and checkpoint["restart_server"]["postflight"].get("port_free")
            and checkpoint.get("restart_cache_check", {}).get("pass")
        )
        result = {
            "schema_version": "q3ple-canonical-60k-build-v2", "status": "PASS" if local_gate else "FAILED",
            "evidence_class": EVIDENCE_CLASS, "publishable": False, "publication_authorized": False,
            "candidate_runtime_commit": CANDIDATE_RUNTIME_COMMIT, "identity": identity,
            "base": public_base_record(base), "checkpoint": repo_relative(paths["checkpoint"]),
            "ingestion_proof": ingestion_proof,
            "slot_pair": public_pair_record(previous["pair"]), "slot_filename": previous["slot_filename"],
            "stage_boundaries": [
                {key: value for key, value in stage.items() if key not in {"messages", "raw_prompt", "tokens"}}
                for stage in stages
            ],
            "stages": [public_stage_record(item) for item in checkpoint["stage_records"]],
            "restart_restore": checkpoint.get("slot_restore"),
            "restart_cache_check": checkpoint.get("restart_cache_check"), "resources": resources,
            "local_build_gate": local_gate,
            "paths": {key: repo_relative(value) for key, value in paths.items() if isinstance(value, Path)},
        }
        atomic_write_json(Path(paths["result"]), result)
        return result
    except CanonicalBuildFailure:
        checkpoint.update({"phase": "failed", "base": base, "stage_boundaries": stages})
        atomic_write_json(Path(paths["checkpoint"]), checkpoint)
        raise
    except BaseException as error:
        checkpoint.update({"phase": "failed", "base": base, "stage_boundaries": stages, "error": {"type": type(error).__name__, "message": str(error)}})
        atomic_write_json(Path(paths["checkpoint"]), checkpoint)
        raise CanonicalBuildFailure(str(error), paths=paths) from error


def build_live(tag: str, *, turn_chars: int = 12_000) -> dict[str, Any]:
    """Build, prefix-warm, save/hash, then restart/restore a canonical base."""

    return _build_live_staged(tag, turn_chars=turn_chars)


def select_benchmark_fixtures(
    fixtures: Sequence[Mapping[str, Any]], fixture_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Select an explicit diagnostic subset without changing the source fixture."""

    available = {str(row["id"]): dict(row) for row in fixtures}
    if not fixture_ids:
        selected = list(available.values())
    else:
        requested = list(dict.fromkeys(str(item) for item in fixture_ids))
        missing = [item for item in requested if item not in available]
        if missing:
            raise ValueError(f"unknown fixture ids: {missing}")
        selected = [available[item] for item in requested]
    if not selected:
        raise ValueError("benchmark fixture selection is empty")
    return sorted(selected, key=lambda row: (row["category"], row["id"]))


def benchmark_live(
    base_result: Path,
    tag: str,
    *,
    repeats: int = MIN_REPEATS,
    fixture_path: Path = FIXTURE_PATH,
    fixture_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run target-first then MTP, restoring the same sealed base before every repeat."""

    if repeats < MIN_REPEATS: raise ValueError(f"repeats must be at least {MIN_REPEATS}")
    payload, base, source_slot_dir, source_slot_filename = load_sealed_base(Path(base_result))
    source_pair = offline_pair_gate(source_slot_dir / source_slot_filename, base["tokens"], expected_count=len(base["tokens"]))
    expected_public_pair = payload.get("slot_pair") or {}
    if source_pair["target_sha256"].upper() != str(expected_public_pair.get("target_sha256", "")).upper() or source_pair["draft_sha256"].upper() != str(expected_public_pair.get("draft_sha256", "")).upper():
        raise ValueError("sealed slot pair hashes differ from base result")
    fixtures = select_benchmark_fixtures(
        load_benchmark_fixtures(fixture_path), fixture_ids
    )
    paths = allocate_run_paths(tag)
    identity = validate_live_inputs(fixture_path)
    identity["candidate_runtime"] = validate_candidate_identity()
    copied = copy_slot_pair(
        source_slot_dir / source_slot_filename,
        Path(paths["slot_dir"]),
        str(paths["slot_filename"]),
        base["tokens"],
    )
    run_slot_dir = Path(paths["slot_dir"])
    run_slot_filename = str(paths["slot_filename"])
    run_pair = dict(copied["pair"])
    benchmark_checkpoint: dict[str, Any] = {
        "schema_version": "q3ple-canonical-60k-benchmark-private-v2",
        "phase": "allocated",
        "identity": identity,
        "base_result": repo_relative(base_result),
        "sealed_source_pair": public_pair_record(source_pair),
        "benchmark_pair_copy": copied,
        "selected_fixture_ids": [row["id"] for row in fixtures],
    }
    atomic_write_json(Path(paths["checkpoint"]), benchmark_checkpoint)
    target_rows: list[dict[str, Any]] = []
    mtp_rows: list[dict[str, Any]] = []
    server_records: dict[str, Any] = {}
    benchmark_pagefile_baseline = int(resource_snapshot().get("pagefile_used_bytes", 0))

    def run_mode(mode: str, destination: list[dict[str, Any]], logfile: Path) -> dict[str, Any]:
        server = OwnedServer(mode, candidate_profile_args(mode, run_slot_dir), Path(logfile), Path(paths["telemetry"]), profile_environment(mode))
        error: str | None = None
        pair_before_lane: dict[str, Any] | None = None
        pair_after_lane: dict[str, Any] | None = None
        try:
            pair_before_lane = offline_pair_gate(
                run_slot_dir / run_slot_filename,
                base["tokens"],
                expected_count=len(base["tokens"]),
            )
            if (
                pair_before_lane.get("target_sha256") != run_pair.get("target_sha256")
                or pair_before_lane.get("draft_sha256") != run_pair.get("draft_sha256")
            ):
                raise RuntimeError(f"{mode} lane slot pair changed before launch")
            server.start(); wait_ready(server.process)  # type: ignore[arg-type]
            for repeat in range(1, repeats + 1):
                for fixture in fixtures:
                    restore = slot_restore(PORT, run_slot_filename, run_slot_dir)
                    require_exact_restore(restore, len(base["tokens"]), expected_target_bytes=int(run_pair.get("target_bytes", 0)) or None, expected_draft_bytes=int(run_pair.get("draft_bytes", 0)) or None)
                    sample_start = len(server.samples)
                    row = benchmark_one_task(PORT, base, fixture, mode=mode)
                    row.update({
                        "repeat": repeat,
                        "restore": restore,
                        "restore_exact": int(restore.get("n_restored") or -1) == len(base["tokens"]),
                        "resource_sample_start": sample_start,
                        "resource_sample_end": len(server.samples),
                        "violations": sorted(set(server.violations)),
                    })
                    destination.append(row)
                    with Path(paths["client_log"]).open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if server.violations: raise RuntimeError(f"resource hard-stop: {server.violations}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            server.close()
        try:
            pair_after_lane = offline_pair_gate(
                run_slot_dir / run_slot_filename,
                base["tokens"],
                expected_count=len(base["tokens"]),
            )
            if (
                pair_after_lane.get("target_sha256") != run_pair.get("target_sha256")
                or pair_after_lane.get("draft_sha256") != run_pair.get("draft_sha256")
            ):
                raise RuntimeError(f"{mode} lane slot pair changed after execution")
        except Exception as pair_error:
            pair_message = f"{type(pair_error).__name__}: {pair_error}"
            error = f"{error}; {pair_message}" if error else pair_message
        return {
            "identity": server.identity,
            "working_set_cap": server.working_set_cap,
            "postflight": server.teardown,
            "resource_summary": server.resource_summary(),
            "pair_before": public_pair_record(pair_before_lane or {}),
            "pair_after": public_pair_record(pair_after_lane or {}),
            "pair_unchanged": bool(
                pair_before_lane
                and pair_after_lane
                and pair_before_lane.get("target_sha256") == pair_after_lane.get("target_sha256") == run_pair.get("target_sha256")
                and pair_before_lane.get("draft_sha256") == pair_after_lane.get("draft_sha256") == run_pair.get("draft_sha256")
            ),
            "error": error,
        }

    server_records["target"] = run_mode("target", target_rows, Path(paths["target_log"]))
    target_checks: list[dict[str, Any]] = []
    for fixture in fixtures:
        rows = [row for row in target_rows if row["fixture_id"] == fixture["id"]]
        stable = len(rows) >= repeats and len({row.get("output_sha256") for row in rows}) == 1 and len({row.get("output_token_ids_sha256") for row in rows}) == 1
        valid = len(rows) >= repeats and all(bool(row.get("natural_stop")) and bool((row.get("semantic_score") or {}).get("valid")) and bool(row.get("cache_reuse_exact")) and bool(row.get("suffix_processing_exact")) and bool(row.get("restore_exact")) for row in rows)
        target_checks.append({"fixture_id": fixture["id"], "repeat_count": len(rows), "stable": stable, "valid": valid})
    target_record = server_records["target"]
    target_resources = target_record.get("resource_summary", {})
    target_release_headroom = bool(
        target_resources.get("min_vram_free_mib") is not None
        and int(target_resources["min_vram_free_mib"]) >= PUBLISHABLE_VRAM_FLOOR_MIB
    )
    target_ready = bool(
        target_record.get("error") is None
        and target_record.get("pair_unchanged") is True
        and target_record.get("postflight", {}).get("port_free")
        and target_resources.get("pass")
        and target_release_headroom
        and all(item["stable"] and item["valid"] for item in target_checks)
    )
    if target_ready:
        server_records["mtp"] = run_mode("mtp", mtp_rows, Path(paths["mtp_log"]))
    else:
        server_records["mtp"] = {"skipped": True, "reason": "target lane failed exact stability, cache, semantic, safety, or teardown gate"}

    by_fixture: list[dict[str, Any]] = []
    resources = combined_resource_summary(
        server_records["target"].get("resource_summary", {}),
        server_records["mtp"].get("resource_summary", {}),
        workflow_pagefile_baseline_bytes=benchmark_pagefile_baseline,
    )
    mtp_record = server_records["mtp"]
    if target_ready and not mtp_record.get("error") and not mtp_record.get("skipped"):
        for fixture in fixtures:
            fid = fixture["id"]
            target = [row for row in target_rows if row["fixture_id"] == fid]
            mtp = [row for row in mtp_rows if row["fixture_id"] == fid]
            gate = benchmark_promotion_gate(target, mtp, minimum_repeats=repeats, resources=resources, require_rejection=fixture["category"] in {"code", "prose"})
            gate["metrics"] = {
                "target_decode_tps": metric_summary(target, "decode_tps"),
                "mtp_decode_tps": metric_summary(mtp, "decode_tps"),
                "target_ttft_s": metric_summary(target, "ttft_s"),
                "mtp_ttft_s": metric_summary(mtp, "ttft_s"),
                "target_wall_s": metric_summary(target, "wall_s"),
                "mtp_wall_s": metric_summary(mtp, "wall_s"),
            }
            by_fixture.append({"fixture_id": fid, "category": fixture["category"], **gate})
    pair_after = offline_pair_gate(run_slot_dir / run_slot_filename, base["tokens"], expected_count=len(base["tokens"]))
    pair_unchanged = run_pair["target_sha256"] == pair_after["target_sha256"] and run_pair["draft_sha256"] == pair_after["draft_sha256"]
    exact = bool(target_ready and by_fixture and all(item["promotable"] for item in by_fixture) and pair_unchanged and port_free(PORT))
    target_only_publishable = bool(target_ready and pair_unchanged and port_free(PORT))
    if not target_ready:
        verdict = "BLOCKED_TARGET_GATE"
    elif mtp_record.get("error") or mtp_record.get("skipped"):
        verdict = "PASS_TARGET_ONLY_MTP_EXECUTION_FAILED" if target_only_publishable else "FAILED_MTP_EXECUTION"
    else:
        verdict = "PASS_EXACT_PARITY" if exact else (
            "PASS_TARGET_ONLY_MTP_NON_PROMOTABLE" if target_only_publishable else "FAILED_NOT_PROMOTABLE"
        )
    result = {
        "schema_version": "q3ple-canonical-60k-benchmark-v2",
        "candidate_runtime_commit": CANDIDATE_RUNTIME_COMMIT,
        "identity": identity,
        "base_result": repo_relative(base_result),
        "base": public_base_record(base),
        "ingestion_proof": payload.get("ingestion_proof"),
        "sealed_source_pair": public_pair_record(source_pair),
        "benchmark_pair_before": public_pair_record(run_pair),
        "benchmark_pair_after": public_pair_record(pair_after),
        "sealed_pair_unchanged": pair_unchanged,
        "repeats": repeats,
        "target_first": True,
        "target_gate": {
            "pass": target_ready,
            "release_headroom": target_release_headroom,
            "minimum_free_vram_mib": target_resources.get("min_vram_free_mib"),
            "fixtures": target_checks,
        },
        "target_rows": target_rows,
        "mtp_rows": mtp_rows,
        "server_records": server_records,
        "resources": resources,
        "fixtures": [row["id"] for row in fixtures],
        "comparisons": by_fixture,
        "target_only_publishable": target_only_publishable,
        "mtp_speed_promotable": exact,
        "verdict": verdict,
        "evidence_class": EVIDENCE_CLASS,
        "publishable": False,
        "publication_authorized": False,
        "postflight_port_free": port_free(PORT),
        "paths": {key: repo_relative(value) for key, value in paths.items() if isinstance(value, Path)},
    }
    benchmark_checkpoint.update(
        {
            "phase": "complete",
            "target_rows": target_rows,
            "mtp_rows": mtp_rows,
            "server_records": server_records,
            "benchmark_pair_after": pair_after,
            "result": result,
        }
    )
    atomic_write_json(Path(paths["result"]), result)
    atomic_write_json(Path(paths["checkpoint"]), benchmark_checkpoint)
    return result


def compare_target_mtp(target: Mapping[str, Any], mtp: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one target/MTP pair with exact output and semantic gates."""

    exact_text = target.get("output_sha256") == mtp.get("output_sha256")
    exact_tokens = target.get("output_token_ids_sha256") == mtp.get("output_token_ids_sha256")
    semantic_equal = target.get("semantic_vector") == mtp.get("semantic_vector")
    draft_n = int(mtp.get("draft_n", 0))
    accepted = int(mtp.get("draft_n_accepted", 0))
    checks = {
        "target_request_ok": bool(target.get("request_ok", True)),
        "mtp_request_ok": bool(mtp.get("request_ok", True)),
        "target_natural_stop": bool(target.get("natural_stop")),
        "mtp_natural_stop": bool(mtp.get("natural_stop")),
        "target_semantic_valid": bool((target.get("semantic_score") or {}).get("valid")),
        "mtp_semantic_valid": bool((mtp.get("semantic_score") or {}).get("valid")),
        "exact_text": exact_text,
        "exact_token_ids": exact_tokens,
        "semantic_equal": semantic_equal,
        "same_prompt_tokens": target.get("prompt_token_ids_sha256") == mtp.get("prompt_token_ids_sha256"),
        "same_base_tokens": target.get("base_token_ids_sha256") == mtp.get("base_token_ids_sha256"),
        "same_suffix_tokens": target.get("suffix_token_ids_sha256") == mtp.get("suffix_token_ids_sha256"),
        "target_cache_exact": bool(target.get("cache_reuse_exact", True) and target.get("suffix_processing_exact", True)),
        "mtp_cache_exact": bool(mtp.get("cache_reuse_exact", True) and mtp.get("suffix_processing_exact", True)),
        "target_restore_exact": bool(target.get("restore_exact", True)),
        "mtp_restore_exact": bool(mtp.get("restore_exact", True)),
        "mtp_active": draft_n > 0,
        "mtp_counters_valid": 0 <= accepted <= draft_n,
    }
    return {
        "checks": checks,
        "exact": all(checks.values()),
        "mtp_promotable": all(checks.values()),
        "rejection_observed": draft_n > accepted,
    }


def benchmark_promotion_gate(
    target_runs: Sequence[Mapping[str, Any]],
    mtp_runs: Sequence[Mapping[str, Any]],
    *,
    minimum_repeats: int = MIN_REPEATS,
    resources: Mapping[str, Any] | None = None,
    require_rejection: bool = False,
) -> dict[str, Any]:
    """Apply the conservative publication rule to repeated target/MTP rows."""

    pair_count = min(len(target_runs), len(mtp_runs))
    comparisons = [compare_target_mtp(target_runs[i], mtp_runs[i]) for i in range(pair_count)]
    repeat_gate = len(target_runs) >= minimum_repeats and len(mtp_runs) >= minimum_repeats
    target_valid = all(
        bool(run.get("request_ok", True))
        and bool(run.get("natural_stop"))
        and bool((run.get("semantic_score") or {}).get("valid"))
        and bool(run.get("cache_reuse_exact", True))
        and bool(run.get("suffix_processing_exact", True))
        and bool(run.get("restore_exact", True))
        for run in target_runs
    )
    mtp_valid = all(
        bool(run.get("request_ok", True))
        and bool(run.get("natural_stop"))
        and bool((run.get("semantic_score") or {}).get("valid"))
        and bool(run.get("cache_reuse_exact", True))
        and bool(run.get("suffix_processing_exact", True))
        and bool(run.get("restore_exact", True))
        for run in mtp_runs
    )
    exact_all = bool(comparisons) and all(item["exact"] for item in comparisons)
    target_stable = len({run.get("output_sha256") for run in target_runs}) == 1 and len({run.get("output_token_ids_sha256") for run in target_runs}) == 1
    mtp_stable = len({run.get("output_sha256") for run in mtp_runs}) == 1 and len({run.get("output_token_ids_sha256") for run in mtp_runs}) == 1
    mtp_active = bool(mtp_runs) and all(int(run.get("draft_n", 0)) > 0 for run in mtp_runs)
    counters_valid = all(
        0 <= int(run.get("draft_n_accepted", 0)) <= int(run.get("draft_n", 0))
        for run in mtp_runs
    )
    rejection_observed = any(
        int(run.get("draft_n", 0)) > int(run.get("draft_n_accepted", 0))
        for run in mtp_runs
    )
    rejection_gate = rejection_observed if require_rejection else True
    resources_ok = True if resources is None else bool(resources.get("pass", False))
    if resources is not None:
        # Promotion requires the release headroom floor, while the watchdog's
        # lower 768 MiB floor remains a hard-stop during execution.
        # Lightweight unit callers historically supplied only ``pass``. Live
        # runs include sample_count/min_vram and therefore receive the full
        # release-headroom check.
        if "sample_count" in resources:
            resources_ok = resources_ok and int(resources.get("sample_count", 0)) > 0
        if "min_vram_free_mib" in resources:
            resources_ok = resources_ok and int(resources.get("min_vram_free_mib") or 0) >= PUBLISHABLE_VRAM_FLOOR_MIB
    checks = {
        "minimum_repeats": repeat_gate,
        "target_valid": target_valid,
        "mtp_valid": mtp_valid,
        "exact_text_tokens_semantics": exact_all,
        "target_repeat_stable": target_stable,
        "mtp_repeat_stable": mtp_stable,
        "mtp_active_every_repeat": mtp_active,
        "mtp_counters_valid": counters_valid,
        "rejection_observed_when_required": rejection_gate,
        "resources": resources_ok,
        "matched_pair_count": pair_count >= minimum_repeats,
    }
    reasons: list[str] = []
    if not repeat_gate:
        reasons.append("fewer than minimum measured repeats")
    if not target_valid:
        reasons.append("target natural-stop or semantic gate failed")
    if not mtp_valid:
        reasons.append("MTP natural-stop or semantic gate failed")
    if not exact_all:
        reasons.append("MTP output/token/semantic mismatch; no uplift promotion")
    if not target_stable or not mtp_stable:
        reasons.append("target or MTP repeats are not text/token stable")
    if not mtp_active:
        reasons.append("MTP drafts were not active in every repeat")
    if not counters_valid:
        reasons.append("MTP accepted counters are outside 0..draft_n")
    if require_rejection and not rejection_observed:
        reasons.append("no rejected MTP proposal was observed on a rejection-required fixture")
    if not resources_ok:
        reasons.append("resource safety gate failed")
    promotable = all(checks.values())
    return {
        "checks": checks,
        "comparisons": comparisons,
        "reasons": reasons,
        "promotable": promotable,
        "mtp_promotable": promotable,
        "classification": "PROMOTABLE" if promotable else "MTP_NON_PROMOTABLE",
        "evidence_class": EVIDENCE_CLASS,
        "rejection_observed": rejection_observed,
        "rejection_required": require_rejection,
        "publishable": False,
    }


def daily_profile_summary() -> dict[str, Any]:
    """Describe the intended 81,920-token one-slot operating profile."""

    return {
        "context_size": CONTEXT_SIZE,
        "port": PORT,
        "slots": 1,
        "state": "target-plus-dft-pair",
        "warm_process_required": True,
        "target_only_fallback": "explicit-before-generation",
        "restart": "recovery-path; first post-restart turn may be cold",
        "publication_status": "local-only-until-explicit-authorization",
    }


def write_failure_result(tag: str, mode: str, error: BaseException) -> Path:
    """Persist a unique, machine-readable failure instead of losing evidence."""

    if isinstance(error, CanonicalBuildFailure) and error.run_paths:
        paths = dict(error.run_paths)
    else:
        paths = allocate_run_paths(f"{tag}-{mode}-failure")
    result_path = Path(paths["result"])
    checkpoint_path = Path(paths["checkpoint"])
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite sealed failure result: {result_path}")
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.is_file():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                checkpoint = loaded
        except (OSError, json.JSONDecodeError):
            checkpoint = {}
    cause = error.cause if isinstance(error, CanonicalBuildFailure) and error.cause is not None else error
    stage = error.stage if isinstance(error, CanonicalBuildFailure) else mode
    server_records = {
        key: checkpoint.get(key)
        for key in ("build_server", "restart_server")
        if isinstance(checkpoint.get(key), Mapping)
    }
    stage_records = checkpoint.get("stage_records")
    if isinstance(stage_records, list):
        # Keep failure evidence pointed at the real run without copying raw
        # token arrays into the top-level result.
        server_records["stages"] = [
            public_stage_record(item) for item in stage_records if isinstance(item, Mapping)
        ]
    violation_set: set[str] = set()
    for record in server_records.values():
        if isinstance(record, Mapping) and isinstance(record.get("resources"), Mapping):
            violation_set.update(str(item) for item in record["resources"].get("violations", []))
        elif isinstance(record, list):
            for stage_record in record:
                if not isinstance(stage_record, Mapping):
                    continue
                server_record = stage_record.get("server")
                resources = server_record.get("resources") if isinstance(server_record, Mapping) else None
                if isinstance(resources, Mapping):
                    violation_set.update(str(item) for item in resources.get("violations", []))
    watchdog_violations = sorted(violation_set)
    result = {
        "schema_version": "q3ple-canonical-60k-failure-v2",
        "mode": mode,
        "tag": tag,
        "verdict": "FAILED_NOT_PROMOTABLE",
        "publishable": False,
        "publication_authorized": False,
        "evidence_class": EVIDENCE_CLASS,
        "failure_stage": stage,
        "error": {
            "type": type(cause).__name__,
            "message": str(cause),
            "wrapper_type": type(error).__name__,
            "wrapper_message": str(error),
        },
        "watchdog_violations": watchdog_violations,
        "server_records": server_records,
        "stage_records": server_records.get("stages", []),
        "checkpoint": repo_relative(checkpoint_path),
        "paths": {
            key: repo_relative(value)
            for key, value in paths.items()
            if isinstance(value, Path)
        },
        "created_unix": time.time(),
    }
    checkpoint.update(
        {
            "phase": "failed",
            "failure_stage": stage,
            "failure": result["error"],
            "failure_result": repo_relative(result_path),
            "watchdog_violations": watchdog_violations,
        }
    )
    atomic_write_json(checkpoint_path, checkpoint)
    atomic_write_json(result_path, result)
    return result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate a canonical complete-message Q3_PLE+MTP 60K boundary."
    )
    parser.add_argument(
        "--validate-fixtures",
        action="store_true",
        help="validate the canonical history fixture offline and exit",
    )
    parser.add_argument("--static", action="store_true", help="offline static validation alias")
    parser.add_argument("--mode", choices=("static", "build", "benchmark"), default="static", help="static validation (default), live canonical build, or live A/B benchmark")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH, help="canonical JSONL fixture path")
    parser.add_argument("--min-boundary", type=int, default=MIN_BOUNDARY_TOKENS)
    parser.add_argument("--max-boundary", type=int, default=MAX_BOUNDARY_TOKENS)
    parser.add_argument("--turn-chars", type=int, default=12_000, help="source chars per complete ingestion turn")
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS, help="target/MTP repeats (minimum three)")
    parser.add_argument("--fixture-id", action="append", dest="fixture_ids", help="run only this fixture id; repeat to select multiple explicit diagnostic fixtures")
    parser.add_argument("--base-result", type=Path, help="sealed canonical build result for --mode benchmark")
    parser.add_argument("--tag", help="unique live-run label")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_fixtures or args.static or args.mode == "static":
        print(json.dumps(validate_fixture(args.fixture), indent=2))
        return 0
    if not args.tag:
        raise SystemExit("live execution requires --tag; static validation is the default")
    try:
        if args.mode == "build":
            result = build_live(args.tag, turn_chars=args.turn_chars)
            print(json.dumps({"result": result.get("paths", {}).get("result"), "token_count": result.get("base", {}).get("token_count"), "status": result.get("status"), "local_build_gate": result.get("local_build_gate")}, indent=2))
            return 0 if result.get("status") == "PASS" and result.get("local_build_gate") is True else 2
        if args.mode == "benchmark":
            if args.base_result is None:
                raise SystemExit("--base-result is required for --mode benchmark")
            result = benchmark_live(
                args.base_result,
                args.tag,
                repeats=args.repeats,
                fixture_path=args.fixture,
                fixture_ids=args.fixture_ids,
            )
            print(json.dumps({"result": result.get("paths", {}).get("result"), "verdict": result.get("verdict"), "target_only_publishable": result.get("target_only_publishable"), "mtp_speed_promotable": result.get("mtp_speed_promotable")}, indent=2))
            return 0 if str(result.get("verdict", "")).startswith("PASS_") else 2
    except Exception as error:
        failure = write_failure_result(args.tag, args.mode, error)
        print(json.dumps({"result": str(failure), "verdict": "FAILED_NOT_PROMOTABLE", "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
