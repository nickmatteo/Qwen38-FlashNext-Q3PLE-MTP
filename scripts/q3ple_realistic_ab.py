"""Controlled target-vs-MTP runs for frozen Q3_PLE profiles.

The module is deliberately import-safe: importing it only defines constants and
pure helpers.  A live model run happens only through ``main`` (or an explicit
call to ``run_suite``).  Generated code from the code fixture is inspected with
``ast`` and is never imported or executed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil
except ImportError:  # pragma: no cover - a live run reports this clearly
    psutil = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "benchmarks" / "fixtures" / "q3ple_realistic.jsonl"
RESULT_ROOT = ROOT / "results" / "QWEN38-MTP-PROTOTYPE-001"
LOG_ROOT = ROOT / "logs" / "QWEN38-MTP-PROTOTYPE-001" / "q3ple_realistic_ab"

# 18086 is used by q3ple_mtp_ab.py and 18087 by the agentic-cache probe.  Keep
# this lane on its own fixed port and fail closed when it is not available.
PORT = 18088
SEED = 38027
CONTEXT_SIZE = 16_384
N_CPU_MOE = 45
TARGET_THREADS = 11
TARGET_UBATCH = 256
TARGET_BATCH = 2_048
TARGET_KV = "q4_0"
MTP_DRAFT_THREADS = 8
MTP_DRAFT_NMAX = 3
MTP_DRAFT_PMIN = 0.75
MTP_UBATCH = 64
DEFAULT_PROFILE = "agentic80k"
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "headline16k": {
        "context_size": 16_384,
        "n_cpu_moe": 45,
        "draft_kv": "f16",
        "purpose": "historical maximum-throughput headline profile",
    },
    "agentic80k": {
        "context_size": 81_920,
        "n_cpu_moe": 47,
        "draft_kv": "q4_0",
        "purpose": "practical long-context daily-driver allocation",
    },
}
MTP_ENV = {
    "QWEN38_MTP_UBATCH": "64",
    "GGML_CUDA_MOE_CACHE_MB": "0",
    "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD": "1",
}
PINNED_TENSOR_OVERRIDE = (
    r"^output\.weight$=CUDA0,^blk\.48\.attn_.*=CUDA0,"
    r"^blk\.48\.hc_attn_.*=CUDA0,^blk\.48\.hc_ffn_.*=CUDA0,"
    r"^blk\.48\.nextn\..*=CUDA0,^blk\.48\.ffn_gate_inp.*=CUDA0,"
    r"^blk\.48\.ffn_(gate|up|down)_shexp.*=CUDA0,"
    r"^blk\.48\.ffn_(gate|up|down)_exps.*=CUDA_Host"
)

RUNTIME_COMMIT = "4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc"
RUNTIME_BUILD = 10690
RUNTIME_SHA256 = "72bb9839c156abbba5d55b0ca3f2d7f89a931acaa8a32ba40a8d76bbb4b67436"
RUNTIME_BUNDLE_SHA256 = {
    "ggml.dll": "66f4975d9da0d36e5c9ac29bf7f73acae52335a56d67cb87b75181cb6a7f523e",
    "ggml-base.dll": "2226a00b3b0df079c0b8b5073e05d4d748e837031e58ffa7dbcfe7358c4f455c",
    "ggml-cpu.dll": "ce262ab6d9e3ec6854743e52262d8a2579f7f1367250ed4454f2e5603a8aa1d1",
    "ggml-cuda.dll": "723ddcc921d5f7305ccca170d6b7d1e450be36f37c90ff5cd6b8085faeba8411",
    "llama.dll": "e8b917f37af14e980b138e749d8d6709a7f3fbb38508fd6fdc9555fb449327b8",
    "llama-common.dll": "ece579d0d73a91462c2be3d1cc0f031d378e1a97921364ca01f668977de7584d",
    "llama-server.exe": RUNTIME_SHA256,
    "llama-server-impl.dll": "210325fc337930b105adc127d409db56a06532f5a6bc653638b4a9bc73679068",
    "mtmd.dll": "a91aa7b71e8e2a2e6b50558e921debbf7719c3c64d94e34c289411ece7691c86",
}
SIDECAR_SHA256 = "7e9f2b282dc62534313b30738e0ad114c14e1a58b9c1e7bb9715dcf9c4ca676e"
TARGET_SHARD_COUNT = 33
TARGET_BYTES = 78_525_318_176

# Import the existing profile module without invoking its CLI.  Its constants
# and base argument builder are the source of truth for the frozen target.
_BASE_SPEC = importlib.util.spec_from_file_location(
    "q3ple_mtp_ab_base", ROOT / "scripts" / "q3ple_mtp_ab.py"
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load scripts/q3ple_mtp_ab.py")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

EXE = Path(_BASE.EXE)
MODEL = Path(_BASE.MODEL)
SIDECAR = ROOT / "artifacts" / "models" / "Qwen3.8-Flash-Next-MTP-Q4_K_M-FC-HC" / "mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf"

SYSTEM_CONDITIONING = "Return only the requested answer."


def load_fixtures(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load JSONL fixtures and reject malformed or duplicate IDs."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid fixture JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"fixture line {line_number} must be an object with string id")
            if row["id"] in seen:
                raise ValueError(f"duplicate fixture id: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    return rows


def get_fixture(fixture_id: str, fixtures: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return one fixture by ID, with a useful error for CLI callers."""

    for fixture in fixtures if fixtures is not None else load_fixtures():
        if fixture.get("id") == fixture_id:
            return fixture
    raise KeyError(f"unknown fixture id: {fixture_id}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


def _call_name(node: ast.Call) -> str:
    def dotted(value: ast.AST) -> str:
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            parent = dotted(value.value)
            return f"{parent}.{value.attr}" if parent else value.attr
        return ""

    return dotted(node.func)


def _all_calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def score_code_atomic_json(output: str) -> dict[str, Any]:
    """Statically score the atomic-json contract; never execute ``output``."""

    markdown_free = not re.search(r"```|^\s*```", output, flags=re.MULTILINE)
    source = output.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*\n(?P<code>.*)\n```", source, flags=re.I | re.S)
    if fenced:
        source = fenced.group("code")
    try:
        tree = ast.parse(source)
        parse_error = None
    except SyntaxError as exc:
        tree = ast.Module(body=[], type_ignores=[])
        parse_error = str(exc)

    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    fn = next((node for node in functions if node.name == "atomic_json"), None)
    signature = bool(fn and [arg.arg for arg in fn.args.args] == ["path", "value"] and not fn.args.vararg and not fn.args.kwarg)
    calls = _all_calls(fn or tree)
    names = [_call_name(call) for call in calls]

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    allowed_stdlib = {
        "__future__", "json", "os", "pathlib", "tempfile", "typing", "contextlib",
        "io", "errno", "stat", "sys", "collections", "dataclasses",
    }
    stdlib_only = all(module in allowed_stdlib for module in imports)

    temp_calls = [call for call, name in zip(calls, names) if name in {"tempfile.NamedTemporaryFile", "tempfile.mkstemp"}]
    same_directory_names: set[str] = set()
    for node in ast.walk(fn or tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or not any(
            isinstance(item, ast.Attribute)
            and item.attr == "parent"
            and isinstance(item.value, ast.Name)
            and item.value.id == "path"
            for item in ast.walk(value)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        same_directory_names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )

    def is_same_directory(expression: ast.AST) -> bool:
        return any(
            (
                isinstance(item, ast.Attribute)
                and item.attr == "parent"
                and isinstance(item.value, ast.Name)
                and item.value.id == "path"
            )
            or (isinstance(item, ast.Name) and item.id in same_directory_names)
            for item in ast.walk(expression)
        )

    same_directory_temp = any(
        any(
            keyword.arg == "dir" and is_same_directory(keyword.value)
            for keyword in call.keywords
            if keyword.arg
        )
        for call in temp_calls
    )
    dump_calls = [call for call, name in zip(calls, names) if name == "json.dump"]
    formatting = bool(
        dump_calls
        and all(any(keyword.arg == "indent" for keyword in call.keywords) for call in dump_calls)
        and all(any(keyword.arg == "sort_keys" for keyword in call.keywords) for call in dump_calls)
    )

    def line(node: ast.AST) -> tuple[int, int]:
        return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))

    flush_calls = [call for call, name in zip(calls, names) if name.endswith(".flush")]
    fsync_calls = [call for call, name in zip(calls, names) if name == "os.fsync"]
    replace_calls = [call for call, name in zip(calls, names) if name == "os.replace"]
    flush_before_fsync = bool(flush_calls and fsync_calls and min(map(line, flush_calls)) < min(map(line, fsync_calls)))
    replace_after_fsync = bool(replace_calls and fsync_calls and min(map(line, fsync_calls)) < min(map(line, replace_calls)))

    cleanup_every_failure = False
    if fn:
        for candidate in ast.walk(fn):
            if not isinstance(candidate, ast.Try):
                continue
            final_calls = [
                _call_name(node)
                for node in ast.walk(ast.Module(body=candidate.finalbody, type_ignores=[]))
                if isinstance(node, ast.Call)
            ]
            final_cleanup = any(
                name in {"os.unlink", "os.remove", "Path.unlink", "pathlib.Path.unlink"}
                or name.endswith(".unlink")
                for name in final_calls
            )
            handler_cleanup = False
            for handler in candidate.handlers:
                handler_tree = ast.Module(body=handler.body, type_ignores=[])
                handler_calls = [
                    _call_name(node)
                    for node in ast.walk(handler_tree)
                    if isinstance(node, ast.Call)
                ]
                removes_temp = any(
                    name in {"os.unlink", "os.remove", "Path.unlink", "pathlib.Path.unlink"}
                    or name.endswith(".unlink")
                    for name in handler_calls
                )
                reraises = any(isinstance(node, ast.Raise) for node in ast.walk(handler_tree))
                if removes_temp and reraises:
                    handler_cleanup = True
                    break
            if final_cleanup or handler_cleanup:
                cleanup_every_failure = True
                break

    no_execution = not any(name in {"exec", "eval", "compile", "builtins.exec", "builtins.eval"} for name in names)
    semantic = {
        "function_name": bool(fn),
        "signature": signature,
        "stdlib_only": stdlib_only,
        "same_directory_temp": same_directory_temp,
        "json_dump_formatting": formatting,
        "flush_before_fsync": flush_before_fsync,
        "replace_after_fsync": replace_after_fsync,
        "cleanup_every_failure": cleanup_every_failure,
        "no_execution": no_execution,
        "markdown_free": markdown_free,
    }
    # Formatting compliance is recorded, but exact target/MTP comparison uses
    # the unmodified raw text. A single fenced block may still be valid code.
    functional_semantic = {key: value for key, value in semantic.items() if key != "markdown_free"}
    valid = parse_error is None and all(functional_semantic.values())
    return {
        "valid": valid,
        "parse_error": parse_error,
        "semantic": semantic,
        "semantic_vector": semantic,
        "no_generated_code_execution": True,
    }


_WORD_RE = re.compile(r"\b[\w][\w’'/-]*\b", flags=re.UNICODE)
_PROSE_FACTS: dict[str, re.Pattern[str]] = {
    "allocation_81920": re.compile(r"\b81[\s,]?920\b", re.I),
    "occupied_59996": re.compile(r"\b59[\s,]?996\b", re.I),
    "ram_below_6gib": re.compile(r"(?:ram|memory).{0,50}(?:below|under|less than|fell|dropped).{0,30}\b6\s*(?:gib|gb)\b", re.I | re.S),
    "connection_reset": re.compile(r"connection\s+reset", re.I),
    "no_completion": re.compile(r"\bno\b.{0,50}(?:completion|completed\s+(?:response|request))", re.I | re.S),
    "no_retrieval_verdict": re.compile(r"\bno\b.{0,50}retrieval\s+verdict", re.I | re.S),
    "not_capacity_proof": re.compile(
        r"(?:not\s+(?:a\s+)?capacity\s+proof|does\s+not\s+(?:prove|establish|constitute).{0,30}capacity|"
        r"does\s+not\s+(?:serve|count|qualify)\s+as\s+(?:a\s+)?proof\s+of\s+capacity)",
        re.I | re.S,
    ),
    "next_retrieval_telemetry_gate": re.compile(r"(?:next|valid|rerun).{0,100}retrieval.{0,180}(?:telemetry|timings|rss|pagefile|vram|ram)", re.I | re.S),
}


def score_prose_filled_context_incident(output: str) -> dict[str, Any]:
    """Score incident prose by word bounds, factual assertions, and marker."""

    stripped = output.strip()
    words = _WORD_RE.findall(stripped)
    marker = "BENCH-REAL-PROSE-1"
    markdown = bool(re.search(r"```|^\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)", stripped, re.M))
    plain_prose = bool(stripped) and not markdown and not any(char in stripped for char in "{}[]")
    facts = {name: bool(pattern.search(stripped)) for name, pattern in _PROSE_FACTS.items()}
    semantic = {
        "plain_prose": plain_prose,
        "facts": facts,
        "end_marker": stripped.endswith(marker),
    }
    word_count = len(words)
    valid = 170 <= word_count <= 210 and all(facts.values()) and semantic["plain_prose"] and semantic["end_marker"]
    return {
        "valid": valid,
        "word_count": word_count,
        "word_count_ok": 170 <= word_count <= 210,
        "facts": facts,
        "end_marker": semantic["end_marker"],
        "plain_prose": plain_prose,
        # Word count is intentionally not part of semantic equality: two
        # independently generated, in-range syntheses can be semantically equal.
        "semantic": semantic,
        "semantic_vector": semantic,
    }


def score_fixture(fixture: dict[str, Any], output: str) -> dict[str, Any]:
    scorer = fixture.get("scorer")
    if scorer == "code_atomic_json" or fixture.get("id") == "code_atomic_json_v1":
        return score_code_atomic_json(output)
    if scorer == "prose_filled_context_incident" or fixture.get("id") == "prose_filled_context_incident_v1":
        return score_prose_filled_context_incident(output)
    raise ValueError(f"no scorer registered for fixture {fixture.get('id')!r}")


def semantic_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("semantic_vector") == right.get("semantic_vector")


def _replace_arg(args: list[str], flag: str, value: str | int) -> None:
    index = args.index(flag)
    args[index + 1] = str(value)


def profile_args(mode: str, profile_name: str = DEFAULT_PROFILE) -> list[str]:
    """Build exact frozen target/MTP arguments for hidden static assertions."""

    if mode not in {"target", "mtp"}:
        raise ValueError("mode must be target or mtp")
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError(f"unknown profile: {profile_name}")
    profile = PROFILE_CONFIGS[profile_name]
    args = list(_BASE.base_args())
    _replace_arg(args, "--port", PORT)
    _replace_arg(args, "--ctx-size", profile["context_size"])
    _replace_arg(args, "--threads", TARGET_THREADS)
    _replace_arg(args, "--threads-batch", TARGET_THREADS)
    _replace_arg(args, "--batch-size", TARGET_BATCH)
    _replace_arg(args, "--ubatch-size", TARGET_UBATCH)
    _replace_arg(args, "--cache-type-k", TARGET_KV)
    _replace_arg(args, "--cache-type-v", TARGET_KV)
    _replace_arg(args, "--n-cpu-moe", profile["n_cpu_moe"])
    if mode == "target":
        return args + ["--spec-type", "none"]
    return args + [
        "-md", str(SIDECAR),
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(MTP_DRAFT_NMAX),
        "--spec-draft-p-min", str(MTP_DRAFT_PMIN),
        "--spec-draft-device", "CUDA0",
        "--spec-draft-ngl", "0",
        "--spec-draft-threads", str(MTP_DRAFT_THREADS),
        "--spec-draft-threads-batch", str(MTP_DRAFT_THREADS),
        "--spec-draft-type-k", str(profile["draft_kv"]),
        "--spec-draft-type-v", str(profile["draft_kv"]),
        "--spec-draft-override-tensor", PINNED_TENSOR_OVERRIDE,
    ]


# Alias used by a few existing probe scripts and convenient for callers.
args_for = profile_args


def profile_environment(mode: str, base: dict[str, str] | None = None) -> dict[str, str]:
    if mode not in {"target", "mtp"}:
        raise ValueError("mode must be target or mtp")
    env = dict(os.environ if base is None else base)
    for name in MTP_ENV:
        env.pop(name, None)
    if mode == "mtp":
        env.update(MTP_ENV)
    return env


def _gpu_snapshot() -> dict[str, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip().splitlines()
    if not output:
        raise RuntimeError("nvidia-smi returned no GPUs")
    fields = [field.strip() for field in output[0].split(",")]
    if len(fields) < 3:
        raise RuntimeError(f"unparseable nvidia-smi output: {output[0]!r}")
    return {"used_mib": int(fields[0]), "free_mib": int(fields[1]), "util_pct": int(fields[2])}


def resource_snapshot(process: Any | None = None) -> dict[str, Any]:
    if psutil is None:
        raise RuntimeError("psutil is required for owned-process telemetry")
    memory = psutil.virtual_memory()
    pagefile = psutil.swap_memory()
    sample: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "ram_available_bytes": int(memory.available),
        "pagefile_used_bytes": int(pagefile.used),
        "gpu": _gpu_snapshot(),
    }
    if process is not None:
        try:
            info = process.memory_info()
            sample["rss_bytes"] = int(info.rss)
            sample["vms_bytes"] = int(info.vms)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            sample["rss_bytes"] = None
    return sample


def port_free(port: int = PORT) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def preflight_snapshot() -> dict[str, Any]:
    sample = resource_snapshot()
    if sample["ram_available_bytes"] < 40 * 1024**3:
        raise RuntimeError("preflight RAM available is below 40 GiB")
    if sample["gpu"]["free_mib"] < 8192:
        raise RuntimeError("preflight free VRAM is below 8192 MiB")
    if sample["gpu"]["util_pct"] > 15:
        raise RuntimeError("preflight GPU utilization exceeds 15%")
    if not port_free():
        raise RuntimeError(f"fixed port {PORT} is busy")
    return sample


def _file_sha256(path: Path, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity() -> dict[str, Any]:
    """Return cheap identity metadata; expensive hashes are checked at run time."""

    model_shards = sorted(MODEL.parent.glob("*.gguf"))
    return {
        "target": {
            "first_shard": str(MODEL),
            "shard_count_observed": len(model_shards),
            "bytes_observed": sum(path.stat().st_size for path in model_shards if path.is_file()),
            "quant_recipe": "atomicchat-4.27bpw-q3_ple-m64-v1",
        },
        "draft": {
            "path": str(SIDECAR),
            "expected_sha256": SIDECAR_SHA256,
            "quant_recipe": "mtp-downq4-fc-hc-outq4-v1",
        },
        "runtime": {
            "executable": str(EXE),
            "expected_sha256": RUNTIME_SHA256,
            "expected_bundle_sha256": RUNTIME_BUNDLE_SHA256,
            "commit": RUNTIME_COMMIT,
            "build": RUNTIME_BUILD,
        },
    }


def validate_artifacts() -> dict[str, Any]:
    for path in (EXE, MODEL, SIDECAR):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen artifact: {path}")
    executable_sha = _file_sha256(EXE)
    if executable_sha.lower() != RUNTIME_SHA256.lower():
        raise RuntimeError(f"runtime executable hash mismatch: {executable_sha}")
    identity = artifact_identity()
    observed_bundle: dict[str, str] = {}
    for name, expected_sha in RUNTIME_BUNDLE_SHA256.items():
        path = EXE.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen runtime dependency: {path}")
        observed_sha = _file_sha256(path)
        observed_bundle[name] = observed_sha
        if observed_sha.lower() != expected_sha.lower():
            raise RuntimeError(f"runtime bundle hash mismatch for {name}: {observed_sha}")
    if identity["target"]["shard_count_observed"] != TARGET_SHARD_COUNT:
        raise RuntimeError("target shard count is not the frozen 33-shard artifact")
    if identity["target"]["bytes_observed"] != TARGET_BYTES:
        raise RuntimeError("target artifact byte count differs from frozen manifest")
    identity["runtime"]["observed_sha256"] = executable_sha
    identity["runtime"]["observed_bundle_sha256"] = observed_bundle
    identity["draft"]["observed_sha256"] = _file_sha256(SIDECAR)
    if identity["draft"]["observed_sha256"].lower() != SIDECAR_SHA256.lower():
        raise RuntimeError("MTP sidecar hash mismatch")
    return identity


def post_json(port: int, path: str, body: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def stop_owned(process: subprocess.Popen[str] | None) -> None:
    """Stop only the exact Popen object created by this runner."""

    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_ready(process: subprocess.Popen[str], deadline: float) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"owned server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    raise TimeoutError("owned server did not become healthy")


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON crash-safely without replacing an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        temporary = None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _request_record(
    fixture: dict[str, Any], mode: str, pair_index: int, measurement_index: int,
    body: dict[str, Any], process: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        telemetry_before = resource_snapshot(process)
    except Exception as exc:
        telemetry_before = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        response = post_json(PORT, "/v1/chat/completions", body)
        elapsed = time.perf_counter() - started
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        output = message.get("content") or ""
        token_ids: list[int] | None = None
        token_error = None
        try:
            tokenized = post_json(PORT, "/tokenize", {"content": output, "add_special": False}, timeout=60)
            token_ids = tokenized.get("tokens")
        except Exception as exc:  # preserve output even when retokenization fails
            token_error = str(exc)
        semantic = score_fixture(fixture, output)
        timings = response.get("timings") or {}
        usage = response.get("usage") or {}
        try:
            telemetry_after = resource_snapshot(process)
        except Exception as exc:
            telemetry_after = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "mode": mode,
            "pair_index": pair_index,
            "measurement_index": measurement_index,
            "elapsed_s": elapsed,
            "raw_output": output,
            "raw_response": response,
            "output_sha256": _sha256_text(output),
            "token_ids_sha256": _sha256_json(token_ids) if token_ids is not None else None,
            "token_count": len(token_ids) if token_ids is not None else None,
            "tokenize_error": token_error,
            "semantic_vector": semantic.get("semantic_vector"),
            "semantic_score": semantic,
            "usage": usage,
            "timings": timings,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_tps": timings.get("prompt_per_second"),
            "decode_tps": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n", 0),
            "draft_n_accepted": timings.get("draft_n_accepted", 0),
            "finish_reason": choice.get("finish_reason"),
            "natural_stop": choice.get("finish_reason") == "stop",
            "request_ok": True,
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        }
    except Exception as exc:
        return {
            "mode": mode,
            "pair_index": pair_index,
            "measurement_index": measurement_index,
            "elapsed_s": time.perf_counter() - started,
            "request_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry_before": telemetry_before,
        }


def run_condition(
    mode: str, fixture: dict[str, Any], pair_index: int, order_index: int,
    run_id: str, profile_name: str,
) -> dict[str, Any]:
    """Run one fresh owned server and its conditioning + two measured calls."""

    preflight = preflight_snapshot()
    identity = validate_artifacts()
    run_log_dir = LOG_ROOT / run_id
    run_log_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_log_dir / f"{mode}.log"
    log_handle = log_path.open("x", encoding="utf-8")
    process: subprocess.Popen[str] | None = None
    owned_process: Any | None = None
    monitor_done = threading.Event()
    samples: list[dict[str, Any]] = []
    violations: list[str] = []
    monitor_thread: threading.Thread | None = None
    return_record: dict[str, Any] = {
        "mode": mode,
        "pair_index": pair_index,
        "order_index": order_index,
        "run_id": run_id,
        "profile_name": profile_name,
        "log_path": str(log_path),
        "artifact_identity": identity,
        "preflight": preflight,
        "runs": [],
        "samples": samples,
        "violations": violations,
        "error": None,
    }

    def monitor() -> None:
        while not monitor_done.wait(0.25) and process is not None and process.poll() is None:
            try:
                sample = resource_snapshot(owned_process)
                samples.append(sample)
                if sample["ram_available_bytes"] < 6 * 1024**3:
                    violations.append("ram_available<6GiB")
                elif sample["gpu"]["free_mib"] < 768:
                    violations.append("vram_free<768MiB")
                elif sample.get("rss_bytes") is not None and sample["rss_bytes"] > 50 * 1024**3:
                    violations.append("owned_rss>50GiB")
                elif sample["pagefile_used_bytes"] - preflight["pagefile_used_bytes"] >= 1 * 1024**3:
                    violations.append("pagefile_growth>=1GiB")
                if violations:
                    stop_owned(process)
                    return
            except Exception as exc:
                violations.append(f"telemetry_error:{exc}")
                stop_owned(process)
                return

    try:
        command = profile_args(mode, profile_name)
        environment = profile_environment(mode)
        temp_root = ROOT / "artifacts" / ".cache" / "q3ple-realistic-ab" / run_id
        temp_root.mkdir(parents=True, exist_ok=True)
        environment.update({"TEMP": str(temp_root), "TMP": str(temp_root), "HF_HOME": str(temp_root / "hf")})
        process = subprocess.Popen(command, cwd=str(_BASE.BIN), env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        owned_process = psutil.Process(process.pid)
        monitor_thread = threading.Thread(target=monitor, name=f"q3ple-realistic-{run_id}", daemon=True)
        monitor_thread.start()
        _wait_ready(process, time.monotonic() + 300)

        request = {
            "model": "model",
            "messages": [
                {"role": "system", "content": fixture["system"]},
                {"role": "user", "content": fixture["user"]},
            ],
            "temperature": 0,
            "seed": SEED,
            "max_tokens": int(fixture.get("max_tokens", 256)),
            "stream": False,
            "cache_prompt": False,
        }
        conditioning = dict(request)
        conditioning["max_tokens"] = 1
        conditioning_started = time.perf_counter()
        conditioning_response: dict[str, Any] | None = None
        conditioning_error: str | None = None
        try:
            conditioning_response = post_json(PORT, "/v1/chat/completions", conditioning)
        except Exception as exc:
            conditioning_error = f"{type(exc).__name__}: {exc}"

        runs: list[dict[str, Any]] = []
        if conditioning_error is None and not violations:
            for measurement_index in (1, 2):
                record = _request_record(fixture, mode, pair_index, measurement_index, request, owned_process)
                runs.append(record)
                if violations or not record.get("request_ok"):
                    break
        return_record.update({
            "command": command,
            "command_string": subprocess.list2cmdline(command),
            "environment_overrides": {
                name: environment.get(name)
                for name in MTP_ENV
            },
            "conditioning": {
                "counted": False,
                "elapsed_s": time.perf_counter() - conditioning_started,
                "raw_response": conditioning_response,
                "error": conditioning_error,
            },
            "runs": runs,
        })
    except Exception as exc:
        return_record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor_done.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=3)
        stop_owned(process)
        try:
            log_handle.close()
        except Exception:
            pass

        postflight_port_free = False
        for _ in range(20):
            if port_free():
                postflight_port_free = True
                break
            time.sleep(0.25)

        ram_values = [preflight["ram_available_bytes"]] + [
            int(sample["ram_available_bytes"])
            for sample in samples
            if isinstance(sample.get("ram_available_bytes"), int)
        ]
        vram_values = [preflight["gpu"]["free_mib"]] + [
            int(sample["gpu"]["free_mib"])
            for sample in samples
            if isinstance(sample.get("gpu"), dict) and isinstance(sample["gpu"].get("free_mib"), int)
        ]
        rss_values = [
            int(sample["rss_bytes"])
            for sample in samples
            if isinstance(sample.get("rss_bytes"), int)
        ]
        pagefile_values = [preflight["pagefile_used_bytes"]] + [
            int(sample["pagefile_used_bytes"])
            for sample in samples
            if isinstance(sample.get("pagefile_used_bytes"), int)
        ]
        peak = {
            "min_ram_available_bytes": min(ram_values),
            "min_vram_free_mib": min(vram_values),
            "max_owned_rss_bytes": max(rss_values, default=0),
            "max_pagefile_used_bytes": max(pagefile_values),
            "pagefile_growth_bytes": max(pagefile_values) - preflight["pagefile_used_bytes"],
        }
        safety_ok = bool(
            not violations
            and peak["min_ram_available_bytes"] >= 6 * 1024**3
            and peak["min_vram_free_mib"] >= 768
            and peak["max_owned_rss_bytes"] <= 50 * 1024**3
            and peak["pagefile_growth_bytes"] < 1 * 1024**3
            and postflight_port_free
        )
        return_record.update({
            "samples": samples,
            "violations": violations,
            "peak": peak,
            "postflight": {
                "owned_process_exit_code": process.poll() if process is not None else None,
                "port_free": postflight_port_free,
            },
            "safety_ok": safety_ok,
        })

    # The return is assigned in the try block; this guard also keeps static
    # analyzers honest when an exception is raised before a request completes.
    return return_record


def compare_runs(target: dict[str, Any], mtp: dict[str, Any]) -> dict[str, Any]:
    requests_ok = bool(target.get("request_ok") and mtp.get("request_ok"))
    semantic_valid = bool(
        target.get("semantic_score", {}).get("valid")
        and mtp.get("semantic_score", {}).get("valid")
    )
    exact_text = target.get("raw_output") == mtp.get("raw_output")
    token_hash_equal = target.get("token_ids_sha256") == mtp.get("token_ids_sha256")
    semantic = semantic_equal(target, mtp)
    exact = bool(requests_ok and semantic_valid and exact_text and token_hash_equal and semantic)
    return {
        "target_measurement_index": target.get("measurement_index"),
        "mtp_measurement_index": mtp.get("measurement_index"),
        "exact_text_equal": exact_text,
        "retokenized_token_hash_equal": token_hash_equal,
        "semantic_equal": semantic,
        "requests_ok": requests_ok,
        "semantic_valid_both": semantic_valid,
        "exact_parity": exact,
        "classification": "PASS" if exact else "FAILED_NOT_PROMOTABLE",
    }


def run_suite(
    fixture: dict[str, Any], pairs: int, tag: str,
    profile_name: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("tag may contain only letters, numbers, dot, underscore, and dash")
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError(f"unknown profile: {profile_name}")
    profile = PROFILE_CONFIGS[profile_name]
    pair_records: list[dict[str, Any]] = []
    all_comparisons: list[dict[str, Any]] = []
    for pair_index in range(1, pairs + 1):
        order = ("target", "mtp") if pair_index % 2 else ("mtp", "target")
        conditions: dict[str, dict[str, Any]] = {}
        for order_index, mode in enumerate(order, 1):
            run_id = f"{tag}-p{pair_index:02d}-{mode}-{uuid.uuid4().hex[:10]}"
            try:
                conditions[mode] = run_condition(mode, fixture, pair_index, order_index, run_id, profile_name)
            except Exception as exc:
                conditions[mode] = {
                    "mode": mode,
                    "pair_index": pair_index,
                    "order_index": order_index,
                    "run_id": run_id,
                    "runs": [],
                    "violations": [],
                    "safety_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        target_runs = conditions["target"].get("runs", [])
        mtp_runs = conditions["mtp"].get("runs", [])
        comparisons = [compare_runs(t, m) for t, m in zip(target_runs, mtp_runs)]
        all_comparisons.extend(comparisons)
        pair_records.append({"pair_index": pair_index, "order": list(order), "target": conditions["target"], "mtp": conditions["mtp"], "comparisons": comparisons})

    mtp_rejections = sum(
        1
        for pair in pair_records
        for run in pair["mtp"].get("runs", [])
        if isinstance(run.get("draft_n"), (int, float)) and run.get("draft_n", 0) > run.get("draft_n_accepted", 0)
    )
    expected_measured = pairs * 2 * 2
    observed_measured = sum(
        len(condition.get("runs", []))
        for pair in pair_records
        for condition in (pair["target"], pair["mtp"])
    )
    natural_stop = observed_measured == expected_measured and all(
        run.get("natural_stop")
        for pair in pair_records
        for condition in (pair["target"], pair["mtp"])
        for run in condition.get("runs", [])
    )
    condition_records = [
        condition
        for pair in pair_records
        for condition in (pair["target"], pair["mtp"])
    ]
    safety_ok = bool(condition_records) and all(
        condition.get("safety_ok") and not condition.get("error")
        for condition in condition_records
    )
    release_headroom_ok = safety_ok and all(
        condition.get("peak", {}).get("min_vram_free_mib", 0) >= 1024
        for condition in condition_records
    )
    semantic_valid_all = observed_measured == expected_measured and all(
        run.get("semantic_score", {}).get("valid")
        for pair in pair_records
        for condition in (pair["target"], pair["mtp"])
        for run in condition.get("runs", [])
    )
    mtp_active = observed_measured == expected_measured and all(
        isinstance(run.get("draft_n"), (int, float)) and run.get("draft_n", 0) > 0
        for pair in pair_records
        for run in pair["mtp"].get("runs", [])
    )
    deterministic: dict[str, bool] = {}
    canonical_hashes: dict[str, dict[str, list[str]]] = {}
    for mode in ("target", "mtp"):
        mode_runs = [
            run
            for pair in pair_records
            for run in pair[mode].get("runs", [])
            if run.get("request_ok")
        ]
        output_hashes = sorted({str(run.get("output_sha256")) for run in mode_runs})
        token_hashes = sorted({str(run.get("token_ids_sha256")) for run in mode_runs})
        canonical_hashes[mode] = {"output_sha256": output_hashes, "token_ids_sha256": token_hashes}
        deterministic[mode] = bool(
            len(mode_runs) == pairs * 2
            and len(output_hashes) == 1
            and len(token_hashes) == 1
            and token_hashes != ["None"]
        )
    if observed_measured != expected_measured:
        verdict = "FAILED_INCOMPLETE"
    elif not safety_ok:
        verdict = "FAILED_SAFETY"
    elif not release_headroom_ok:
        verdict = "FAILED_RELEASE_HEADROOM"
    elif not mtp_active:
        verdict = "FAILED_MTP_INACTIVE"
    elif not semantic_valid_all:
        verdict = "FAILED_SEMANTIC_CONTRACT"
    elif any(item["classification"] == "FAILED_NOT_PROMOTABLE" for item in all_comparisons):
        verdict = "FAILED_NOT_PROMOTABLE"
    elif not all(deterministic.values()):
        verdict = "FAILED_NONDETERMINISTIC"
    elif not natural_stop:
        verdict = "FAILED_NATURAL_STOP"
    elif fixture.get("requires_rejection") and not mtp_rejections:
        verdict = "WEAK_FIXTURE"
    else:
        verdict = "PASS"
    fixture_identity = {
        key: value
        for key, value in fixture.items()
        if key not in {"system", "user"}
    }
    fixture_identity.update({
        "manifest_path": str(FIXTURE_PATH),
        "manifest_sha256": _file_sha256(FIXTURE_PATH),
        "system_sha256": _sha256_text(str(fixture["system"])),
        "user_sha256": _sha256_text(str(fixture["user"])),
        "messages_sha256": _sha256_json([
            {"role": "system", "content": fixture["system"]},
            {"role": "user", "content": fixture["user"]},
        ]),
    })
    return {
        "schema_version": "q3ple-realistic-ab-v1",
        "tag": tag,
        "fixture": fixture_identity,
        "profile": {
            "name": profile_name,
            "purpose": profile["purpose"],
            "context_size": profile["context_size"],
            "n_cpu_moe": profile["n_cpu_moe"],
            "target_threads": TARGET_THREADS,
            "target_threads_batch": TARGET_THREADS,
            "batch": TARGET_BATCH,
            "ubatch": TARGET_UBATCH,
            "target_kv": TARGET_KV,
            "mtp_draft_nmax": MTP_DRAFT_NMAX,
            "mtp_draft_pmin": MTP_DRAFT_PMIN,
            "mtp_draft_device": "CUDA0",
            "mtp_draft_ngl": 0,
            "mtp_draft_threads": MTP_DRAFT_THREADS,
            "mtp_draft_threads_batch": MTP_DRAFT_THREADS,
            "mtp_draft_kv": profile["draft_kv"],
            "mtp_ubatch_env": MTP_UBATCH,
            "ggml_cuda_moe_cache_mb": 0,
            "draft_expert_offload": 1,
            "fixed_port": PORT,
            "seed": SEED,
        },
        "runtime": {"commit": RUNTIME_COMMIT, "build": RUNTIME_BUILD, "executable": str(EXE), "sha256": RUNTIME_SHA256},
        "pairs": pair_records,
        "comparisons": all_comparisons,
        "mtp_rejections": mtp_rejections,
        "mtp_active_all": mtp_active,
        "semantic_valid_all": semantic_valid_all,
        "safety_ok_all": safety_ok,
        "release_headroom_ok_all": release_headroom_ok,
        "deterministic_by_mode": deterministic,
        "canonical_hashes": canonical_hashes,
        "natural_stop_all": natural_stop,
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _unique_result_path(tag: str) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        candidate = RESULT_ROOT / f"q3ple_realistic_ab_{tag}_{stamp}_{uuid.uuid4().hex[:8]}.json"
        if not candidate.exists():
            return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled frozen Q3_PLE target-vs-MTP realistic fixture runner")
    parser.add_argument("--fixture-id", required=True, help="fixture ID from benchmarks/fixtures/q3ple_realistic.jsonl")
    parser.add_argument("--pairs", type=int, default=1, help="number of fresh target/MTP pairs (default: 1)")
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_CONFIGS), default=DEFAULT_PROFILE,
        help=f"runtime profile (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument("--tag", required=True, help="unique evidence tag used in result and log paths")
    args = parser.parse_args(argv)
    fixture = get_fixture(args.fixture_id)
    record = run_suite(fixture, args.pairs, args.tag, args.profile)
    output_path = _unique_result_path(args.tag)
    _atomic_write_json(output_path, record)
    print(json.dumps({"verdict": record["verdict"], "result": str(output_path), "mtp_rejections": record["mtp_rejections"]}, indent=2))
    return 0 if record["verdict"] in {"PASS", "WEAK_FIXTURE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
