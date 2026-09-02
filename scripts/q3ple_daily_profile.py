"""Small, fail-closed lifecycle controller for the Q3_PLE daily profile.

The command is intentionally boring: one pinned ``llama-server`` process, one
slot (id 0), and versioned slot manifests.  It never searches for a process by
name and never changes MTP/target mode implicitly.  Importing this module only
defines constants and helpers; no profile, GPU, process, or network work is
performed until a command is selected.

The live command is suitable for the Windows host that owns this checkout.  A
``smoke`` command is always a bounded, non-live request plan, which is useful in
CI and during review without loading a 78 GB target bundle.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable
import shutil
import threading
import uuid

import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = ROOT / "profiles/q3ple_daily_80k.json"
PROFILE_PATH = DEFAULT_PROFILE_PATH
STATE_DIR = ROOT / "results/QWEN38-MTP-PROTOTYPE-001/state/q3ple_daily"
STATE_PATH = STATE_DIR / "server.json"
SLOTS_DIR = STATE_DIR / "slots"
LOGS_DIR = STATE_DIR / "logs"
PARSER_HELPER = ROOT / "scripts/q3ple_agentic_slot_extend.py"

CANDIDATE_RUNTIME_COMMIT = "73b803464f25fc9054046728bf2ebed5a372737e"
BASELINE_N_CPU_MOE = 48
DEFAULT_PORT = 18089
MIN_RAM_BYTES = 6 * 1024**3
LAUNCH_MIN_RAM_BYTES = 40 * 1024**3
HARD_GPU_FREE_MIB = 768
LAUNCH_GPU_FREE_MIB = 8192
DAILY_GPU_FREE_MIB = 1024
MAX_RSS_BYTES = 50 * 1024**3
MAX_SWAP_GROWTH_BYTES = 1 * 1024**3
WS_CAP_BYTES = 37 * 1024**3
HEALTH_TIMEOUT_SECONDS = 300
REQUEST_TIMEOUT_SECONDS = 7200
WATCHDOG_INTERVAL_SECONDS = 2.0


class ProfileError(RuntimeError):
    """A profile or ownership gate failed; no live side effect is attempted."""


class PrefixMismatch(ProfileError):
    """A new prompt is not an exact extension of the cached token vector."""


class ResyncRequired(ProfileError):
    """A target-only generation made an MTP pair stale; explicit resync is required."""


def assert_exact_prefix(prefix: Iterable[int], full: Iterable[int]) -> list[int]:
    """Return the unseen suffix, rejecting even one changed cached token."""

    left = list(prefix)
    right = list(full)
    if len(right) < len(left) or right[: len(left)] != left:
        raise PrefixMismatch(f"cached token prefix mismatch (cached={len(left)}, new={len(right)})")
    return right[len(left) :]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    selected = PROFILE_PATH if path is None else resolve_path(path)
    with selected.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict) or profile.get("schema") != 1:
        raise ProfileError(f"unsupported profile schema in {selected}")
    return profile


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value)))
    return path if path.is_absolute() else ROOT / path


def _config(profile: dict[str, Any]) -> dict[str, Path | int | str]:
    runtime = profile["runtime"]
    artifacts = profile["artifacts"]
    server = profile["server"]
    state = profile["state"]
    return {
        "executable": resolve_path(runtime["executable"]),
        "worktree": resolve_path(runtime["worktree"]),
        "target": resolve_path(artifacts["target"]["first_shard"]),
        "target_dir": resolve_path(artifacts["target"]["directory"]),
        "sidecar": resolve_path(artifacts["sidecar"]["path"]),
        "state_dir": resolve_path(state["directory"]),
        "slots_dir": resolve_path(server["slot_save_path"]),
        "host": str(server["host"]),
        "port": int(server["port"]),
    }


def configure_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Select one profile and derive its isolated state paths before use."""

    global PROFILE_PATH, STATE_DIR, STATE_PATH, SLOTS_DIR, LOGS_DIR
    PROFILE_PATH = DEFAULT_PROFILE_PATH if path is None else resolve_path(path)
    profile = load_profile(PROFILE_PATH)
    config = _config(profile)
    STATE_DIR = Path(config["state_dir"])
    STATE_PATH = STATE_DIR / "server.json"
    SLOTS_DIR = Path(config["slots_dir"])
    LOGS_DIR = STATE_DIR / "logs"
    return profile


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    """Write and flush a JSON file, then atomically replace its destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _git(*args: str, cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProfileError(f"git check failed in {cwd}: {error}") from error


def validate_profile(profile: dict[str, Any] | None = None, *, check_files: bool = True, check_hashes: bool = True) -> dict[str, Any] | list[str]:
    """Validate immutable profile pins and (optionally) local artifact bytes.

    The positional-profile form is intentionally a pure compatibility helper:
    it returns an error list (empty for a valid profile) and never performs
    payload I/O.  The CLI form loads the pinned profile and returns structured
    evidence, raising :class:`ProfileError` on failure.
    """

    supplied = profile is not None
    profile = load_profile() if profile is None else profile
    if supplied:
        check_files = False
        check_hashes = False
    try:
        config = _config(profile)
    except (KeyError, TypeError, ValueError) as error:
        if supplied:
            return [f"profile shape is invalid: {error}"]
        raise ProfileError(f"profile shape is invalid: {error}") from error
    errors: list[str] = []
    runtime = profile.get("runtime", {})
    artifacts = profile.get("artifacts", {})
    server = profile.get("server", {})
    safety = profile.get("safety", {})

    if runtime.get("commit") != CANDIDATE_RUNTIME_COMMIT:
        errors.append("runtime commit is not the slot-state candidate commit")
    if int(server.get("slot_count", 0)) != 1 or int(server.get("slot_id", -1)) != 0:
        errors.append("profile must have exactly one slot with id 0")
    if int(server.get("port", 0)) <= 0 or not str(server.get("host")):
        errors.append("server host/port is invalid")
    base_args = server.get("base_args", [])
    try:
        ctx_value = int(base_args[base_args.index("--ctx-size") + 1])
    except (ValueError, IndexError, TypeError):
        ctx_value = None
    if ctx_value != 81920:
        errors.append("base args do not pin ctx-size 81920")
    # The n48 placement is the frozen baseline. A profile may test a different
    # MoE split, but only by declaring itself a candidate and naming the
    # baseline it deviates from, so the deviation is explicit in the file and
    # not a silent edit. A candidate must also be a separate identity: its own
    # profile id, its own server port, and its own state directory. Sharing any
    # of those would let a candidate run overwrite the baseline's state.
    expected_n_cpu_moe = str(BASELINE_N_CPU_MOE)
    candidate = profile.get("placement_candidate")
    if candidate is not None:
        if not isinstance(candidate, dict):
            errors.append("placement_candidate must be an object")
        else:
            expected_n_cpu_moe = str(candidate.get("n_cpu_moe"))
            baseline_id = candidate.get("baseline_profile_id")
            baseline_port = candidate.get("baseline_port")
            baseline_state = candidate.get("baseline_state_directory")
            if not baseline_id or not baseline_port or not baseline_state:
                errors.append(
                    "placement_candidate must name baseline_profile_id, baseline_port and baseline_state_directory"
                )
            if profile.get("candidate_of") != baseline_id:
                errors.append("candidate_of must equal placement_candidate.baseline_profile_id")
            if profile.get("profile_id") == baseline_id:
                errors.append("a placement candidate must not reuse the baseline profile id")
            if int(server.get("port", 0)) == int(baseline_port or 0):
                errors.append("a placement candidate must not reuse the baseline server port")
            if str(profile.get("state", {}).get("directory", "")) == str(baseline_state):
                errors.append("a placement candidate must not reuse the baseline state directory")
    required_values = {
        "--parallel": "1",
        "--threads": "11",
        "--threads-batch": "11",
        "--batch-size": "2048",
        "--ubatch-size": "256",
        "--cache-type-k": "q4_0",
        "--cache-type-v": "q4_0",
        "--n-cpu-moe": expected_n_cpu_moe,
    }
    args = base_args
    for flag, expected in required_values.items():
        try:
            actual = str(args[args.index(flag) + 1])
        except (ValueError, IndexError):
            actual = None
        if actual != expected:
            errors.append(f"base args {flag} is {actual!r}, expected {expected!r}")
    mtp = server.get("modes", {}).get("mtp", {})
    for key, expected in {
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 3,
        "spec_draft_p_min": 0.75,
        "spec_draft_threads": 8,
        "spec_draft_threads_batch": 8,
        "spec_draft_type_k": "q4_0",
        "spec_draft_type_v": "q4_0",
    }.items():
        if mtp.get(key) != expected:
            errors.append(f"MTP {key} is {mtp.get(key)!r}, expected {expected!r}")
    expected_environment = {
        "QWEN38_MTP_UBATCH": "64",
        "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD": "1",
        "GGML_CUDA_MOE_CACHE_MB": "0",
        "QWEN38_WORKING_SET_CAP_GIB": "37",
    }
    if profile.get("environment") != expected_environment:
        errors.append("environment pins are not the validated daily values")
    client_contract = profile.get("client_contract", {})
    if client_contract.get("id_slot") != 0 or client_contract.get("cache_prompt") is not True or client_contract.get("parallel") != 1:
        errors.append("client contract must pin id_slot=0, cache_prompt=true, parallel=1")
    if int(safety.get("working_set_cap_bytes", 0)) != WS_CAP_BYTES:
        errors.append("working-set cap is not 37 GiB")

    # All nine DLL/EXE entries are part of the immutable runtime identity.  A
    # hash-only map is deliberate: bundle files live beside the pinned EXE.
    bundle = runtime.get("bundle", {})
    if not isinstance(bundle, dict) or set(bundle) != {
        "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll",
        "llama.dll", "llama-common.dll", "llama-server.exe",
        "llama-server-impl.dll", "mtmd.dll",
    }:
        errors.append("runtime bundle must contain exactly the nine pinned files")

    # The profile is deliberately explicit about safety policy; silently
    # inheriting a default would make a copied profile unsafe to operate.
    safety_requirements = {
        "hard_ram_available_bytes": MIN_RAM_BYTES,
        "hard_gpu_free_mib": HARD_GPU_FREE_MIB,
        "daily_gpu_free_mib": DAILY_GPU_FREE_MIB,
        "max_owned_rss_bytes": MAX_RSS_BYTES,
        "max_swap_growth_bytes": MAX_SWAP_GROWTH_BYTES,
    }
    for key, expected in safety_requirements.items():
        try:
            actual = int(safety.get(key))
        except (TypeError, ValueError):
            actual = None
        if actual != expected:
            errors.append(f"safety {key} is {actual!r}, expected {expected!r}")
    if str(profile.get("status", "EXPERIMENTAL")) != "EXPERIMENTAL":
        errors.append("profile status must remain EXPERIMENTAL until parity gates pass")

    if check_files:
        worktree = config["worktree"]
        executable = config["executable"]
        target = config["target"]
        sidecar = config["sidecar"]
        for label, path in (("runtime worktree", worktree), ("executable", executable), ("target", target), ("sidecar", sidecar)):
            if not Path(path).exists():
                errors.append(f"missing {label}: {path}")
        if Path(worktree).is_dir():
            try:
                commit = _git("rev-parse", "HEAD", cwd=Path(worktree))
                dirty = _git("status", "--porcelain", cwd=Path(worktree))
                if commit != runtime.get("commit"):
                    errors.append(f"runtime HEAD is {commit}, expected {runtime.get('commit')}")
                if dirty:
                    errors.append("runtime worktree is dirty")
            except ProfileError as error:
                errors.append(str(error))
        if Path(target).is_file() and check_hashes:
            actual = sha256_file(target)
            expected = str(artifacts["target"].get("first_shard_sha256", "")).upper()
            if actual != expected:
                errors.append(f"target first-shard hash is {actual}, expected {expected}")
        if Path(sidecar).is_file():
            try:
                expected_bytes = int(artifacts["sidecar"].get("bytes", -1))
                if Path(sidecar).stat().st_size != expected_bytes:
                    errors.append(f"sidecar bytes are {Path(sidecar).stat().st_size}, expected {expected_bytes}")
            except (TypeError, ValueError, KeyError):
                errors.append("sidecar bytes pin is missing or invalid")
            if check_hashes:
                actual = sha256_file(sidecar)
                expected = str(artifacts["sidecar"].get("sha256", "")).upper()
                if actual != expected:
                    errors.append(f"sidecar hash is {actual}, expected {expected}")
        if Path(executable).is_file():
            for name, expected_hash in bundle.items():
                path = Path(executable).parent / str(name)
                if not path.is_file():
                    errors.append(f"missing runtime bundle file: {path}")
                    continue
                if check_hashes:
                    actual = sha256_file(path)
                    if actual != str(expected_hash).upper():
                        errors.append(f"runtime bundle hash for {name} is {actual}, expected {expected_hash}")
        if Path(config["target_dir"]).is_dir():
            shards = sorted(Path(config["target_dir"]).glob("*.gguf"))
            if len(shards) != int(artifacts["target"].get("shard_count", 0)):
                errors.append(f"target shard count is {len(shards)}, expected {artifacts['target'].get('shard_count')}")
            aggregate = sum(path.stat().st_size for path in shards)
            if aggregate != int(artifacts["target"].get("aggregate_bytes", -1)):
                errors.append(f"target aggregate bytes are {aggregate}, expected {artifacts['target'].get('aggregate_bytes')}")

    result = {
        "valid": not errors,
        "profile": str(PROFILE_PATH),
        "profile_sha256": sha256_file(PROFILE_PATH),
        "candidate_commit": runtime.get("commit"),
        "resolved": {key: str(value) for key, value in config.items()},
        "checked_files": bool(check_files),
        "checked_hashes": bool(check_files and check_hashes),
        "errors": errors,
    }
    if supplied:
        return errors
    if errors:
        raise ProfileError(json.dumps(result, indent=2))
    return result


def _replace_placeholders(value: str, config: dict[str, Any]) -> str:
    replacements = {
        "{target_first_shard}": str(config["target"]),
        "{sidecar}": str(config["sidecar"]),
        "{slot_save_path}": str(config["slots_dir"]),
    }
    for marker, replacement in replacements.items():
        value = value.replace(marker, replacement)
    return value


def build_command(mode: str, profile: dict[str, Any] | None = None) -> list[str]:
    """Construct the exact command for ``mode`` without starting it."""

    if mode not in ("mtp", "target"):
        raise ValueError("mode must be mtp or target")
    profile = profile or load_profile()
    allowed_modes = profile.get("policy", {}).get("allowed_modes", ["mtp", "target"])
    if mode not in allowed_modes:
        raise ProfileError(f"mode {mode!r} is disabled by profile {profile.get('profile_id')!r}")
    config = _config(profile)
    server = profile["server"]
    command = [str(config["executable"])]
    command.extend(_replace_placeholders(str(item), config) for item in server["base_args"])
    mode_spec = server["modes"][mode]
    if mode == "mtp":
        command.extend(
            [
                "-md", str(config["sidecar"]),
                "--spec-type", str(mode_spec["spec_type"]),
                "--spec-draft-n-max", str(mode_spec["spec_draft_n_max"]),
                "--spec-draft-p-min", str(mode_spec["spec_draft_p_min"]),
                "--spec-draft-device", str(mode_spec["spec_draft_device"]),
                "--spec-draft-ngl", str(mode_spec["spec_draft_ngl"]),
                "--spec-draft-threads", str(mode_spec["spec_draft_threads"]),
                "--spec-draft-threads-batch", str(mode_spec["spec_draft_threads_batch"]),
                "--spec-draft-type-k", str(mode_spec["spec_draft_type_k"]),
                "--spec-draft-type-v", str(mode_spec["spec_draft_type_v"]),
                "--spec-draft-override-tensor", str(mode_spec["spec_draft_override_tensor"]),
            ]
        )
    else:
        command.extend(["--spec-type", str(mode_spec["spec_type"])])
    return command


def preview(mode: str, state_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a non-live launch preview for tests, CI, and operators."""

    profile = load_profile()
    if mode not in ("mtp", "target"):
        return {"valid": False, "mode": mode, "errors": ["mode must be mtp or target"], "command": []}
    errors = validate_profile(profile)
    if errors:
        return {"valid": False, "mode": mode, "errors": errors, "command": []}
    command = build_command(mode, profile)
    return {
        "valid": True,
        "mode": mode,
        "command": command,
        "state_dir": str(Path(state_dir) if state_dir is not None else _config(profile)["state_dir"]),
        "live": False,
        "status": profile.get("status", "EXPERIMENTAL"),
    }


def _gpu_snapshot() -> dict[str, int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProfileError(f"nvidia-smi preflight failed: {error}") from error
    if not output:
        raise ProfileError("nvidia-smi returned no GPU rows")
    first = output.splitlines()[0].split(",")
    try:
        return {"used_mib": int(first[0]), "free_mib": int(first[1]), "util_pct": int(first[2])}
    except (IndexError, ValueError) as error:
        raise ProfileError(f"unparseable nvidia-smi output: {output!r}") from error


def preflight() -> dict[str, Any]:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu = _gpu_snapshot()
    snapshot = {
        "utc": utc_now(),
        "ram_available_bytes": virtual.available,
        "ram_total_bytes": virtual.total,
        "swap_used_bytes": swap.used,
        "gpu": gpu,
        "launch_ram_floor_bytes": LAUNCH_MIN_RAM_BYTES,
        "launch_gpu_floor_mib": LAUNCH_GPU_FREE_MIB,
        "runtime_ram_floor_bytes": MIN_RAM_BYTES,
        "runtime_gpu_floor_mib": HARD_GPU_FREE_MIB,
        "daily_gpu_floor_mib": DAILY_GPU_FREE_MIB,
        "launch_gpu_util_ceiling_pct": 15,
    }
    failures = []
    if virtual.available < LAUNCH_MIN_RAM_BYTES:
        failures.append("available RAM below 40 GiB launch floor")
    if gpu["free_mib"] < LAUNCH_GPU_FREE_MIB:
        failures.append("free VRAM below 8192 MiB launch floor")
    if gpu["free_mib"] < DAILY_GPU_FREE_MIB:
        failures.append("free VRAM below 1024 MiB daily floor")
    if gpu["util_pct"] > 15:
        failures.append("GPU utilization above 15%")
    if failures:
        snapshot["failures"] = failures
        raise ProfileError(json.dumps(snapshot, indent=2))
    snapshot["failures"] = []
    return snapshot


def _port_free(host: str, port: int) -> bool:
    sock = socket.socket()
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def set_working_set_cap(process: subprocess.Popen[str], maximum_bytes: int = WS_CAP_BYTES) -> dict[str, Any]:
    """Apply the 37 GiB Windows working-set cap to the owned process.

    The profile is Windows-specific.  Returning an explicit ``skipped`` record
    on another platform keeps imports and synthetic tests harmless while making
    it impossible to mistake a non-Windows dry run for a capped launch.
    """

    if os.name != "nt":
        return {"skipped": "windows-only", "requested_max": maximum_bytes}
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_ws = kernel32.SetProcessWorkingSetSizeEx
    set_ws.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32]
    set_ws.restype = ctypes.c_int
    handle = ctypes.c_void_p(int(process._handle))
    flags = 0x00000002 | 0x00000004  # QUOTA_LIMITS_HARDWS_MIN/MAX_ENABLE
    ctypes.set_last_error(0)
    if not set_ws(handle, ctypes.c_size_t(64 * 1024), ctypes.c_size_t(maximum_bytes), flags):
        raise ProfileError(f"SetProcessWorkingSetSizeEx failed: {ctypes.WinError(ctypes.get_last_error())}")
    get_ws = kernel32.GetProcessWorkingSetSizeEx
    get_ws.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint32)]
    get_ws.restype = ctypes.c_int
    minimum = ctypes.c_size_t()
    maximum = ctypes.c_size_t()
    actual_flags = ctypes.c_uint32()
    if not get_ws(handle, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(actual_flags)):
        raise ProfileError(f"GetProcessWorkingSetSizeEx failed: {ctypes.WinError(ctypes.get_last_error())}")
    if maximum.value != maximum_bytes:
        raise ProfileError(f"working-set cap readback {maximum.value} != requested {maximum_bytes}")
    return {"requested_max": maximum_bytes, "actual_min": minimum.value, "actual_max": maximum.value, "flags": actual_flags.value}


def _process_identity(pid: int) -> dict[str, Any]:
    try:
        process = psutil.Process(int(pid))
        return {
            "pid": process.pid,
            "create_time": process.create_time(),
            "exe": str(Path(process.exe()).resolve()),
            "command": process.cmdline(),
            "running": process.is_running(),
        }
    except (psutil.Error, OSError) as error:
        raise ProfileError(f"unable to inspect owned PID {pid}: {error}") from error


def identity_matches(record: dict[str, Any], *, strict_command: bool = True) -> bool:
    """Return true only if PID, create time, executable, and command still match."""

    try:
        current = _process_identity(int(record["pid"]))
    except (ProfileError, KeyError, ValueError, TypeError):
        return False
    if not current["running"]:
        return False
    try:
        if abs(float(current["create_time"]) - float(record["create_time"])) > 1.0:
            return False
        if os.path.normcase(str(Path(current["exe"]).resolve())) != os.path.normcase(str(Path(record["exe"]).resolve())):
            return False
        if strict_command and list(current["command"]) != list(record.get("command", [])):
            return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def _read_state(path: str | Path | None = None) -> dict[str, Any] | None:
    state_path = Path(path) if path is not None else STATE_PATH
    if not state_path.is_file():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"invalid server state {state_path}: {error}") from error
    if not isinstance(value, dict):
        raise ProfileError(f"invalid server state object: {state_path}")
    return value


def _require_owned_state() -> dict[str, Any]:
    state = _read_state()
    if not state or state.get("status") != "running":
        raise ProfileError("no running q3ple_daily server is recorded")
    if not identity_matches(state):
        raise ProfileError("owned PID identity revalidation failed; refusing API or stop action")
    return state


def _http_json(host: str, port: int, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    url = f"http://{host}:{port}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        raise ProfileError(f"HTTP {method} {path} failed: {error}") from error
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProfileError(f"non-JSON response from {path}: {raw[:300]}") from error
    if not isinstance(value, dict):
        raise ProfileError(f"unexpected response object from {path}")
    return value


def _wait_health(host: str, port: int, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProfileError(f"llama-server exited with code {process.returncode} before health")
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.25)
    raise ProfileError(f"health did not reach HTTP 200 within {timeout}s")


def _stop_process_exact(state: dict[str, Any]) -> None:
    if not identity_matches(state):
        raise ProfileError("owned PID identity revalidation failed; refusing termination")
    process = psutil.Process(int(state["pid"]))
    process.terminate()
    try:
        process.wait(timeout=15)
    except psutil.TimeoutExpired:
        if not identity_matches(state):
            raise ProfileError("PID identity changed during termination; refusing kill")
        process.kill()
        process.wait(timeout=10)


def _stop_watcher_exact(state: dict[str, Any]) -> None:
    watcher = state.get("watchdog")
    if not isinstance(watcher, dict) or watcher.get("status") != "running":
        return
    if not identity_matches(watcher):
        # The watcher already exited (or its PID was reused).  Never kill by
        # name; simply mark the record inactive.
        watcher["status"] = "not-owned-or-exited"
        return
    process = psutil.Process(int(watcher["pid"]))
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        if identity_matches(watcher):
            process.kill()
            process.wait(timeout=10)
    watcher["status"] = "stopped"


def _runtime_snapshot(pid: int, initial_swap: int) -> dict[str, Any]:
    process = psutil.Process(int(pid))
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu = _gpu_snapshot()
    rss = int(process.memory_info().rss)
    return {
        "utc": utc_now(),
        "pid": int(pid),
        "ram_available_bytes": int(virtual.available),
        "rss_bytes": rss,
        "swap_used_bytes": int(swap.used),
        "swap_growth_bytes": max(0, int(swap.used) - int(initial_swap)),
        "gpu": gpu,
        "failures": [],
    }


def _watchdog_violation(snapshot: dict[str, Any]) -> list[str]:
    failures: list[str] = list(snapshot.get("failures", []))
    if int(snapshot.get("ram_available_bytes", 0)) < MIN_RAM_BYTES:
        failures.append("available RAM below 6 GiB runtime floor")
    if int(snapshot.get("gpu", {}).get("free_mib", 0)) < HARD_GPU_FREE_MIB:
        failures.append("free VRAM below 768 MiB runtime floor")
    if int(snapshot.get("rss_bytes", 0)) > MAX_RSS_BYTES:
        failures.append("owned RSS above 50 GiB runtime ceiling")
    if int(snapshot.get("swap_growth_bytes", 0)) > MAX_SWAP_GROWTH_BYTES:
        failures.append("swap growth above 1 GiB runtime ceiling")
    return failures


def _append_telemetry(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()


def _spawn_watchdog(state: dict[str, Any], state_path: Path | None = None) -> dict[str, Any]:
    state_path = STATE_PATH if state_path is None else state_path
    telemetry = LOGS_DIR / f"watchdog-{state['pid']}-{int(float(state['create_time']))}.jsonl"
    command = [sys.executable, str(Path(__file__).resolve()), "watchdog", "--state-path", str(state_path.resolve()), "--pid", str(state["pid"])]
    # Publish the telemetry path before spawning so the child cannot race ahead
    # and silently write to a different fallback file.
    state["watchdog"] = {
        "status": "starting",
        "telemetry": str(telemetry),
        "interval_seconds": WATCHDOG_INTERVAL_SECONDS,
    }
    atomic_json(state_path, state)
    handle = open(os.devnull, "w", encoding="utf-8")
    try:
        watcher = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=_creation_flags(),
            text=True,
        )
    finally:
        handle.close()
    identity = _process_identity(watcher.pid)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if watcher.poll() is not None:
            raise ProfileError(f"watchdog exited during startup with code {watcher.returncode}")
        if telemetry.is_file() and telemetry.stat().st_size > 0:
            break
        time.sleep(0.1)
    else:
        if identity_matches(identity):
            psutil.Process(watcher.pid).terminate()
        raise ProfileError("watchdog produced no telemetry within 10 seconds")
    return {"pid": identity["pid"], "create_time": identity["create_time"], "exe": identity["exe"], "command": identity["command"], "telemetry": str(telemetry), "status": "running", "interval_seconds": WATCHDOG_INTERVAL_SECONDS}


def _watchdog_main(state_path: Path, pid: int) -> int:
    target: dict[str, Any] | None = None
    try:
        state = _read_state(state_path)
        if not state or int(state.get("pid", -1)) != int(pid) or not identity_matches(state):
            return 2
        initial_swap = int(state.get("preflight", {}).get("swap_used_bytes", 0))
        telemetry_path = Path(state.get("watchdog", {}).get("telemetry", str(LOGS_DIR / f"watchdog-{pid}.jsonl")))
        target = dict(state)
        while True:
            current = _read_state(state_path)
            if not current or current.get("status") not in ("starting", "running"):
                return 0
            if not identity_matches(target):
                return 0
            try:
                snapshot = _runtime_snapshot(pid, initial_swap)
            except Exception as error:
                snapshot = {"utc": utc_now(), "pid": pid, "failures": [f"runtime telemetry unavailable: {error}"]}
            failures = _watchdog_violation(snapshot)
            snapshot["failures"] = failures
            _append_telemetry(telemetry_path, snapshot)
            if failures:
                # Revalidate immediately before the only destructive action.
                if identity_matches(target):
                    _stop_process_exact(target)
                latest = _read_state(state_path) or target
                latest["status"] = "failed_closed"
                latest["watchdog_violation"] = {"utc": snapshot["utc"], "failures": failures, "telemetry": str(telemetry_path)}
                atomic_json(state_path, latest)
                return 3
            time.sleep(WATCHDOG_INTERVAL_SECONDS)
    except Exception as error:
        try:
            _append_telemetry(LOGS_DIR / f"watchdog-{pid}-error.jsonl", {"utc": utc_now(), "pid": pid, "failures": [str(error)]})
        except Exception:
            pass
        try:
            if target is not None and identity_matches(target):
                _stop_process_exact(target)
            latest = _read_state(state_path) or target or {"pid": pid}
            latest["status"] = "failed_closed"
            latest["watchdog_violation"] = {
                "utc": utc_now(),
                "failures": [f"watchdog failed: {type(error).__name__}: {error}"],
            }
            atomic_json(state_path, latest)
        except Exception:
            pass
        return 4


def _artifact_identity(profile: dict[str, Any]) -> dict[str, Any]:
    artifacts = profile["artifacts"]
    target = artifacts["target"]
    sidecar = artifacts["sidecar"]
    return {
        "target": {
            "directory": str(target["directory"]),
            "first_shard": str(target["first_shard"]),
            "first_shard_sha256": str(target.get("first_shard_sha256", "")).upper(),
            "shard_count": int(target["shard_count"]),
            "aggregate_bytes": int(target["aggregate_bytes"]),
        },
        "sidecar": {
            "path": str(sidecar["path"]),
            "sha256": str(sidecar.get("sha256", "")).upper(),
            "bytes": int(sidecar["bytes"]),
        },
    }


def _immutable_profile_identity(profile: dict[str, Any]) -> dict[str, Any]:
    server = profile["server"]
    return {
        "profile_id": profile.get("profile_id"),
        "runtime_commit": profile["runtime"].get("commit"),
        "runtime_bundle": dict(profile["runtime"].get("bundle", {})),
        "artifacts": _artifact_identity(profile),
        "base_args": list(server.get("base_args", [])),
        "modes": json.loads(json.dumps(server.get("modes", {}), sort_keys=True)),
        "environment": dict(profile.get("environment", {})),
        "client_contract": json.loads(json.dumps(profile.get("client_contract", {}), sort_keys=True)),
        "base_args_sha256": sha256_json(list(server.get("base_args", []))),
        "modes_sha256": sha256_json(server.get("modes", {})),
        "environment_sha256": sha256_json(dict(profile.get("environment", {}))),
        "client_contract_sha256": sha256_json(profile.get("client_contract", {})),
    }


def launch(mode: str) -> dict[str, Any]:
    profile = load_profile()
    validation = validate_profile(check_files=True, check_hashes=True)
    config = _config(profile)
    if mode not in ("mtp", "target"):
        raise ProfileError("launch mode must be mtp or target")
    allowed_modes = profile.get("policy", {}).get("allowed_modes", ["mtp", "target"])
    if mode not in allowed_modes:
        raise ProfileError(f"launch mode {mode!r} is disabled by profile {profile.get('profile_id')!r}")
    prior = _read_state()
    if prior and prior.get("status") == "running" and identity_matches(prior):
        raise ProfileError("an owned q3ple_daily server is already running")
    if not _port_free(str(config["host"]), int(config["port"])):
        raise ProfileError(f"port {config['port']} is already in use")
    pre = preflight()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    log_path = LOGS_DIR / f"server-{mode}-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.log"
    command = build_command(mode, profile)
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in profile.get("environment", {}).items()})
    log_handle = log_path.open("a", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(config["executable"].parent),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=_creation_flags(),
            text=True,
        )
    except Exception:
        log_handle.close()
        raise
    try:
        identity = _process_identity(process.pid)
        state = {
            "schema": 1,
            "status": "starting",
            "mode": mode,
            "pid": process.pid,
            "create_time": identity["create_time"],
            "exe": identity["exe"],
            "command": identity["command"],
            "runtime_commit": profile["runtime"]["commit"],
            "runtime_executable_sha256": profile["runtime"]["bundle"]["llama-server.exe"],
            "port": int(config["port"]),
            "host": str(config["host"]),
            "started_utc": started,
            "log": str(log_path),
            "preflight": pre,
            "profile_sha256": validation["profile_sha256"],
            "command_sha256": sha256_json(command),
            "immutable_profile": _immutable_profile_identity(profile),
            "environment": dict(profile.get("environment", {})),
            "client_contract": json.loads(json.dumps(profile.get("client_contract", {}), sort_keys=True)),
        }
        state["working_set_cap"] = set_working_set_cap(process, WS_CAP_BYTES)
        atomic_json(STATE_PATH, state)
        _wait_health(str(config["host"]), int(config["port"]), process, HEALTH_TIMEOUT_SECONDS)
        state["status"] = "running"
        state["health_utc"] = utc_now()
        atomic_json(STATE_PATH, state)
        try:
            watchdog = _spawn_watchdog(state)
            latest = _read_state()
            if latest and latest.get("status") == "failed_closed":
                raise ProfileError("watchdog failed closed during launch startup")
            if not identity_matches(state):
                raise ProfileError("server exited while watchdog was starting")
            state["watchdog"] = watchdog
        except Exception as error:
            # A server without supervision is not a valid daily launch.
            if identity_matches(state):
                _stop_process_exact(state)
            raise ProfileError(f"watchdog launch failed; server terminated: {error}") from error
        atomic_json(STATE_PATH, state)
        if not identity_matches(state) or not identity_matches(state["watchdog"]):
            latest = _read_state() or state
            latest["status"] = "failed_closed"
            latest["error"] = "server or watchdog exited during launch finalization"
            latest["failed_utc"] = utc_now()
            atomic_json(STATE_PATH, latest)
            raise ProfileError(latest["error"])
        return state
    except Exception as error:
        try:
            if identity_matches(state):
                _stop_process_exact(state)
        except Exception:
            pass
        current = _read_state()
        failed = current or locals().get("state", {"schema": 1, "status": "failed", "mode": mode, "pid": process.pid})
        failure_status = "failed_closed" if failed.get("status") == "failed_closed" else "failed"
        failed.update({"status": failure_status, "error": str(error), "failed_utc": utc_now()})
        atomic_json(STATE_PATH, failed)
        raise
    finally:
        log_handle.close()


def _load_slot_helper():
    spec = importlib.util.spec_from_file_location("q3ple_daily_slot_helper", PARSER_HELPER)
    if spec is None or spec.loader is None:
        raise ProfileError(f"unable to load slot parser helper: {PARSER_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_target(path: Path) -> dict[str, Any]:
    helper = _load_slot_helper()
    try:
        data = path.read_bytes()
        parsed = helper.parse_serialized_prompt(data)
    except (OSError, ValueError) as error:
        raise ProfileError(f"target slot parser rejected {path}: {error}") from error
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "token_count": parsed["count"],
        "token_ids_sha256": parsed["token_ids_sha256"].upper(),
    }


class GenerationStore:
    """Immutable local generation store used by the daily controller.

    The server writes target and draft state separately.  This store copies
    only a fully parsed/validated pair into a unique generation directory and
    atomically advances a small pointer after both files are durable.
    """

    def __init__(self, root: str | Path, parser: Any) -> None:
        self.root = Path(root)
        self.parser = parser
        self.generations = self.root / "generations"
        self.pointer = self.root / "latest.json"
        self.generations.mkdir(parents=True, exist_ok=True)

    def current(self) -> dict[str, Any] | None:
        if not self.pointer.is_file():
            return None
        try:
            value = json.loads(self.pointer.read_text(encoding="utf-8"))
            generation = str(value["generation"])
            if not generation or Path(generation).name != generation:
                return None
            return value
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _new_generation(self, mode: str, target: Path, draft: Path | None, parsed: dict[str, Any], tokens: list[int]) -> dict[str, Any]:
        generation = f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:10]}"
        destination = self.generations / generation
        destination.mkdir(parents=True, exist_ok=False)
        out_target = destination / "target.slot.bin"
        shutil.copyfile(target, out_target)
        out_draft = None
        if draft is not None:
            out_draft = destination / "target.slot.bin.dft"
            shutil.copyfile(draft, out_draft)
        record: dict[str, Any] = {
            "schema": 1,
            "generation": generation,
            "mode": mode,
            "created_utc": utc_now(),
            "target": {"path": str(out_target), "bytes": out_target.stat().st_size, "sha256": sha256_file(out_target), "token_count": len(tokens), "token_ids_sha256": sha256_tokens(tokens)},
            "mtp_state": "VALID" if draft is not None else "STALE/RESYNC_REQUIRED",
        }
        if out_draft is not None:
            record["draft"] = {"path": str(out_draft), "bytes": out_draft.stat().st_size, "sha256": sha256_file(out_draft)}
        atomic_json(destination / "manifest.json", record)
        atomic_json(self.pointer, {"schema": 1, "generation": generation, "mode": mode, "manifest": str(destination / "manifest.json"), "mtp_state": record["mtp_state"]})
        return record

    def commit_pair(self, source_target: str | Path, source_draft: str | Path, expected_tokens: Iterable[int]) -> dict[str, Any]:
        source_target = Path(source_target)
        source_draft = Path(source_draft)
        tokens = list(expected_tokens)
        if not source_target.is_file() or not source_draft.is_file():
            raise ProfileError("cannot commit an incomplete target/.dft pair")
        current = self.current()
        if current and current.get("mtp_state") == "STALE/RESYNC_REQUIRED":
            raise ResyncRequired("target-only generation is stale; explicitly resync before committing MTP")
        try:
            parsed = self.parser.parse_slot_pair(source_target, expected_count=len(tokens), expected_tokens=tokens)
        except FileNotFoundError as error:
            raise ProfileError(f"missing target/.dft pair: {error}") from error
        # ValueError is intentionally preserved for a malformed or mismatched
        # vector: callers/tests must be able to distinguish corruption.
        if not parsed.get("target_draft_tokens_equal", True):
            raise ValueError("target/.dft token vectors differ")
        return self._new_generation("mtp", source_target, source_draft, parsed, tokens)

    def commit_target_only(self, source_target: str | Path, expected_tokens: Iterable[int]) -> dict[str, Any]:
        source_target = Path(source_target)
        tokens = list(expected_tokens)
        if not source_target.is_file():
            raise ProfileError("cannot commit missing target-only state")
        try:
            parsed = self.parser.parse_serialized_prompt(source_target.read_bytes(), expected_count=len(tokens))
        except FileNotFoundError as error:
            raise ProfileError(f"missing target state: {error}") from error
        if list(tokens) != list(parsed.get("tokens", tokens)):
            raise ValueError("target token vector differs")
        return self._new_generation("target", source_target, None, parsed, tokens)


class DailySession:
    """Small backend-neutral request wrapper with fail-closed persistence."""

    def __init__(self, root: str | Path, backend: Any, *, mode: str, store: GenerationStore) -> None:
        if mode not in ("target", "mtp"):
            raise ValueError("mode must be target or mtp")
        self.root = Path(root)
        self.backend = backend
        self.mode = mode
        self.store = store

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages", [])
        prompt = self.backend.render(messages)
        tokens = list(self.backend.tokenize(prompt))
        self.backend.pending_tokens = tokens
        response = self.backend.generate(tokens, [], mode=self.mode, n_predict=int(request.get("n_predict", request.get("max_tokens", 256))), seed=int(request.get("seed", 0)))
        if not response or response.get("partial") or not response.get("complete", True):
            raise ProfileError("partial/failed generation is not checkpointable; no fallback or save performed")
        saved = self.backend.save(mode=self.mode)
        if self.mode == "mtp":
            target = saved.get("target") if isinstance(saved, dict) else None
            draft = saved.get("draft") if isinstance(saved, dict) else None
            if not target or not draft:
                raise ProfileError("MTP generation did not produce a complete target/.dft pair")
            pointer = self.store.commit_pair(target, draft, tokens)
        else:
            target = saved.get("target") if isinstance(saved, dict) else None
            if not target:
                raise ProfileError("target generation did not produce target state")
            pointer = self.store.commit_target_only(target, tokens)
        return {"response": response, "generation": pointer}


def sha256_tokens(tokens: Iterable[int]) -> str:
    return sha256_json([int(token) for token in tokens])


def _slot_basename(mode: str) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for number in range(1, 1000):
        suffix = f"-r{number}" if number > 1 else ""
        name = f"q3ple-daily-{mode}-{stamp}{suffix}.slot.bin"
        if not (SLOTS_DIR / name).exists() and not (SLOTS_DIR / f"{name}.dft").exists() and not (STATE_DIR / f"{name}.manifest.json").exists():
            return name
    raise ProfileError("unable to allocate unique versioned slot basename")


def _manifest_path(value: str | None) -> Path:
    if value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = STATE_DIR / candidate
    else:
        pointer = STATE_DIR / "latest.json"
        if not pointer.is_file():
            raise ProfileError("latest.json does not exist")
        try:
            with pointer.open("r", encoding="utf-8") as handle:
                pointer_data = json.load(handle)
            candidate = STATE_DIR / str(pointer_data["manifest"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ProfileError(f"invalid latest.json pointer: {error}") from error
    candidate = candidate.resolve()
    if candidate.parent != STATE_DIR.resolve() or candidate.name == "latest.json":
        raise ProfileError("manifest must be a direct child of the q3ple_daily state directory")
    if not candidate.is_file():
        raise ProfileError(f"manifest does not exist: {candidate}")
    return candidate


def _validate_manifest(manifest_path: Path, *, running_mode: str, requested_mode: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"invalid slot manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ProfileError("slot manifest must be a JSON object")
    profile = load_profile()
    identity = _immutable_profile_identity(profile)
    if manifest.get("profile_id") != profile.get("profile_id"):
        raise ProfileError("slot manifest profile_id does not match q3ple_daily")
    if manifest.get("profile_sha256") != sha256_file(PROFILE_PATH):
        raise ProfileError("slot manifest profile hash does not match current profile")
    if manifest.get("runtime_commit") != profile["runtime"].get("commit"):
        raise ProfileError("slot manifest runtime commit does not match current profile")
    if manifest.get("runtime_bundle") != identity["runtime_bundle"]:
        raise ProfileError("slot manifest runtime bundle identity does not match current profile")
    if manifest.get("artifacts") != identity["artifacts"]:
        raise ProfileError("slot manifest artifact identity does not match current profile")
    if manifest.get("environment") != identity["environment"] or manifest.get("client_contract") != identity["client_contract"]:
        raise ProfileError("slot manifest immutable environment/client contract changed")
    for field in ("base_args_sha256", "modes_sha256", "environment_sha256", "client_contract_sha256"):
        if manifest.get(field) != identity[field]:
            raise ProfileError(f"slot manifest {field} does not match current profile")
    manifest_mode = manifest.get("mode")
    if manifest_mode not in ("mtp", "target"):
        raise ProfileError("slot manifest mode is invalid")
    if requested_mode and requested_mode != running_mode:
        raise ProfileError(f"requested restore mode {requested_mode!r} does not match running mode {running_mode!r}")
    fallback = None
    if running_mode == "mtp" and manifest_mode == "target":
        raise ProfileError("MTP runtime cannot restore a target-only manifest")
    if running_mode == "target" and manifest_mode == "mtp":
        fallback = {
            "explicit": True,
            "kind": "target-only",
            "reason": "target runtime requested an MTP manifest; only its validated target state is restored",
        }
    slot_filename = manifest.get("slot_filename")
    if not isinstance(slot_filename, str) or not slot_filename or Path(slot_filename).name != slot_filename or not slot_filename.endswith(".slot.bin"):
        raise ProfileError("manifest slot filename is malformed")
    expected_slot = (SLOTS_DIR / slot_filename).resolve()
    slot_path = manifest.get("slot_path")
    if slot_path is not None and Path(str(slot_path)).resolve() != expected_slot:
        raise ProfileError("manifest slot path does not match slot filename")
    target = manifest.get("target")
    if not isinstance(target, dict):
        raise ProfileError("manifest target record is malformed")
    try:
        target_path = Path(str(target.get("path", ""))).resolve()
    except (TypeError, ValueError, OSError) as error:
        raise ProfileError(f"manifest target path is malformed: {error}") from error
    if target_path != expected_slot or not target_path.is_file():
        raise ProfileError("manifest target path is missing or does not match slot filename")
    try:
        target_bytes = int(target["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProfileError(f"manifest target byte pin is malformed: {error}") from error
    if target_bytes != target_path.stat().st_size:
        raise ProfileError("manifest target byte length changed")
    if sha256_file(target_path) != str(target.get("sha256", "")).upper():
        raise ProfileError("manifest target hash changed")
    try:
        target_count = int(target.get("token_count", -1))
    except (TypeError, ValueError) as error:
        raise ProfileError(f"manifest target token vector pin is malformed: {error}") from error
    if not target.get("token_ids_sha256") or target_count < 0:
        raise ProfileError("manifest target token vector pin is malformed")
    if manifest_mode == "mtp":
        draft = manifest.get("draft")
        if not isinstance(draft, dict):
            raise ProfileError("manifest MTP draft record is malformed")
        try:
            draft_path = Path(str(draft.get("path", ""))).resolve()
        except (TypeError, ValueError, OSError) as error:
            raise ProfileError(f"manifest draft path is malformed: {error}") from error
        if draft_path != Path(f"{target_path}.dft").resolve() or not draft_path.is_file():
            raise ProfileError("manifest MTP draft companion is missing or misplaced")
        try:
            draft_bytes = int(draft["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProfileError(f"manifest draft byte pin is malformed: {error}") from error
        if draft_bytes != draft_path.stat().st_size or draft_path.stat().st_size <= 0:
            raise ProfileError("manifest MTP draft companion is empty or changed")
        if sha256_file(draft_path) != str(draft.get("sha256", "")).upper():
            raise ProfileError("manifest MTP draft hash changed")
        try:
            draft_count = int(draft.get("token_count", -1))
        except (TypeError, ValueError) as error:
            raise ProfileError(f"manifest draft token vector pin is malformed: {error}") from error
        if target.get("token_ids_sha256") != draft.get("token_ids_sha256") or draft_count != target_count:
            raise ProfileError("manifest target/.dft token-vector hashes differ")
    return manifest, fallback


def save() -> dict[str, Any]:
    state = _require_owned_state()
    profile = load_profile()
    config = _config(profile)
    # A promoted daily checkpoint must leave the stricter 1 GiB VRAM reserve;
    # the 768 MiB value is only the emergency runtime floor.
    try:
        promotion_gpu = _gpu_snapshot()
    except ProfileError:
        raise
    if promotion_gpu["free_mib"] < DAILY_GPU_FREE_MIB:
        raise ProfileError(f"save/promotion refused below daily VRAM floor: {promotion_gpu['free_mib']} MiB")
    mode = str(state["mode"])
    basename = _slot_basename(mode)
    _http_json(str(config["host"]), int(config["port"]), "POST", "/slots/0?action=save", {"filename": basename})
    target_path = SLOTS_DIR / basename
    if not target_path.is_file():
        raise ProfileError("save API returned but target slot file is missing")
    target = _parse_target(target_path)
    manifest: dict[str, Any] = {
        "schema": 1,
        "profile_id": profile["profile_id"],
        "profile_sha256": sha256_file(PROFILE_PATH),
        "mode": mode,
        "created_utc": utc_now(),
        "slot_filename": basename,
        "slot_path": str(target_path.resolve()),
        "target": target,
        "runtime_commit": profile["runtime"]["commit"],
        "runtime_bundle": dict(profile["runtime"].get("bundle", {})),
        "artifacts": _artifact_identity(profile),
        "environment": dict(profile.get("environment", {})),
        "client_contract": json.loads(json.dumps(profile.get("client_contract", {}), sort_keys=True)),
        "base_args_sha256": sha256_json(list(profile["server"].get("base_args", []))),
        "modes_sha256": sha256_json(profile["server"].get("modes", {})),
        "environment_sha256": sha256_json(dict(profile.get("environment", {}))),
        "client_contract_sha256": sha256_json(profile.get("client_contract", {})),
        "server_pid": state["pid"],
    }
    if mode == "mtp":
        helper = _load_slot_helper()
        try:
            parsed = helper.parse_slot_pair(target_path)
        except (OSError, ValueError) as error:
            raise ProfileError(f"MTP target/.dft parser rejected saved pair: {error}") from error
        draft_path = Path(f"{target_path}.dft")
        if draft_path.stat().st_size <= 0:
            raise ProfileError("MTP save produced an empty .dft companion")
        if not parsed.get("target_draft_tokens_equal") or parsed.get("target_token_ids_sha256") != parsed.get("draft_prompt_token_ids_sha256"):
            raise ProfileError("MTP save target/.dft token vectors are not identical")
        manifest["target"].update({
            "token_count": parsed["target_token_count"],
            "token_ids_sha256": parsed["target_token_ids_sha256"].upper(),
        })
        manifest["draft"] = {
            "path": str(draft_path),
            "bytes": draft_path.stat().st_size,
            "sha256": sha256_file(draft_path),
            "token_count": parsed["draft_prompt_token_count"],
            "token_ids_sha256": parsed["draft_prompt_token_ids_sha256"].upper(),
            "spec_bytes": parsed.get("draft_spec_bytes"),
        }
    manifest_path = STATE_DIR / f"{basename}.manifest.json"
    atomic_json(manifest_path, manifest)
    # latest.json is a pointer, promoted only after the immutable manifest and
    # all slot bytes have passed validation above.
    atomic_json(STATE_DIR / "latest.json", {
        "schema": 1,
        "manifest": manifest_path.name,
        "mode": mode,
        "slot_filename": basename,
        "promoted_utc": utc_now(),
    })
    state["last_save"] = {"manifest": str(manifest_path), "slot_filename": basename, "mode": mode, "saved_utc": manifest["created_utc"]}
    atomic_json(STATE_PATH, state)
    return manifest


def restore(manifest: str | None = None, requested_mode: str | None = None) -> dict[str, Any]:
    state = _require_owned_state()
    profile = load_profile()
    config = _config(profile)
    manifest_path = _manifest_path(manifest)
    loaded, fallback = _validate_manifest(manifest_path, running_mode=str(state["mode"]), requested_mode=requested_mode)
    basename = str(loaded["slot_filename"])
    response = _http_json(str(config["host"]), int(config["port"]), "POST", "/slots/0?action=restore", {"filename": basename})
    result = {
        "manifest": str(manifest_path),
        "mode": state["mode"],
        "manifest_mode": loaded["mode"],
        "slot_filename": basename,
        "response": response,
        "fallback": fallback,
        "restored_utc": utc_now(),
    }
    state["last_restore"] = result
    atomic_json(STATE_PATH, state)
    return result


def stop(save_before: bool = False) -> dict[str, Any]:
    state = _require_owned_state()
    if save_before:
        # Save first and fail closed: if validation/promotion fails, the owned
        # server remains running and this function never reaches termination.
        saved = save()
    else:
        saved = None
    _stop_watcher_exact(state)
    _stop_process_exact(state)
    state["status"] = "stopped"
    state["stopped_utc"] = utc_now()
    if saved is not None:
        state["save_before_stop"] = {"slot_filename": saved["slot_filename"], "manifest": str(STATE_DIR / f"{saved['slot_filename']}.manifest.json")}
    atomic_json(STATE_PATH, state)
    return {"status": "stopped", "pid": state["pid"], "save_before_stop": saved}


def status() -> dict[str, Any]:
    profile = load_profile()
    state = _read_state()
    config = _config(profile)
    result: dict[str, Any] = {
        "profile_id": profile["profile_id"],
        "state_path": str(STATE_PATH),
        "port": int(config["port"]),
        "port_free": _port_free(str(config["host"]), int(config["port"])),
        "state": state,
    }
    if state:
        result["owned_identity"] = identity_matches(state) if state.get("status") in ("starting", "running") else False
        watcher = state.get("watchdog")
        if isinstance(watcher, dict):
            result["watchdog_identity"] = identity_matches(watcher)
            result["watchdog_telemetry"] = watcher.get("telemetry")
    if state and state.get("status") == "running":
        if result["owned_identity"]:
            try:
                result["health"] = _http_json(str(config["host"]), int(config["port"]), "GET", "/health", timeout=3)
            except ProfileError as error:
                result["health_error"] = str(error)
    return result


def smoke_plan(mode: str = "mtp") -> dict[str, Any]:
    if mode not in ("mtp", "target"):
        raise ValueError("mode must be mtp or target")
    profile = load_profile()
    allowed_modes = profile.get("policy", {}).get("allowed_modes", ["mtp", "target"])
    if mode not in allowed_modes:
        raise ProfileError(f"smoke mode {mode!r} is disabled by profile {profile.get('profile_id')!r}")
    config = _config(profile)
    return {
        "bounded": True,
        "live": False,
        "mode": mode,
        "port": config["port"],
        "state_dir": str(config["state_dir"]),
        "max_requests": 2,
        "request": {
            "path": "/completion",
            "body": {
                "prompt": "smoke: return exactly SMOKE_OK",
                "n_predict": 4,
                "temperature": 0,
                "seed": 38027,
                "stream": False,
                "id_slot": 0,
                "cache_prompt": True,
                "parallel": 1,
            },
        },
        "client_contract": profile["client_contract"],
        "note": "Plan only; no launch, generation, save, restore, or stop is performed.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE_PATH.relative_to(ROOT)),
        help="profile JSON path relative to the repository root",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="print the pinned profile JSON")
    validate = sub.add_parser("validate", help="validate profile pins and local artifact hashes")
    validate.add_argument("--no-files", action="store_true", help="validate schema/args without touching local artifacts")
    validate.add_argument("--no-hash", action="store_true", help="check paths but skip expensive payload hashes")
    sub.add_parser("status", help="show owned process and latest state")
    launch_parser = sub.add_parser("launch", help="launch one hidden owned server")
    launch_parser.add_argument("--mode", choices=("mtp", "target"), required=True)
    sub.add_parser("save", help="save and validate a versioned slot manifest")
    restore_parser = sub.add_parser("restore", help="validate and restore a saved slot")
    restore_parser.add_argument("--manifest", help="manifest basename/path; defaults to latest.json")
    restore_parser.add_argument("--mode", choices=("mtp", "target"), help="assert the running mode; required for explicit target fallback documentation")
    stop_parser = sub.add_parser("stop", help="stop only the exact owned PID")
    stop_parser.add_argument("--save-before-stop", action="store_true", help="save first; any save failure leaves the server running")
    smoke_parser = sub.add_parser("smoke", help="print a bounded non-live smoke plan")
    smoke_parser.add_argument("--mode", choices=("mtp", "target"), default="mtp")
    watchdog = sub.add_parser("watchdog", help=argparse.SUPPRESS)
    watchdog.add_argument("--state-path", required=True)
    watchdog.add_argument("--pid", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_profile(args.profile)
        if args.command == "show":
            print(PROFILE_PATH.read_text(encoding="utf-8"), end="")
        elif args.command == "validate":
            print(json.dumps(validate_profile(check_files=not args.no_files, check_hashes=not args.no_hash), indent=2))
        elif args.command == "status":
            print(json.dumps(status(), indent=2))
        elif args.command == "launch":
            print(json.dumps(launch(args.mode), indent=2))
        elif args.command == "save":
            print(json.dumps(save(), indent=2))
        elif args.command == "restore":
            print(json.dumps(restore(args.manifest, args.mode), indent=2))
        elif args.command == "stop":
            print(json.dumps(stop(args.save_before_stop), indent=2))
        elif args.command == "smoke":
            print(json.dumps(smoke_plan(args.mode), indent=2))
        elif args.command == "watchdog":
            return _watchdog_main(Path(args.state_path).resolve(), int(args.pid))
        return 0
    except (ProfileError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
