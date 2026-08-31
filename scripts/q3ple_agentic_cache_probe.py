"""Agentic prompt-cache and slot-persistence probe for the frozen Q3_PLE+MTP runtime.

The probe deliberately keeps all runtime work behind :func:`main`.  Importing this
module only defines helpers and constants; it does not import the runner scripts,
start a server, inspect a GPU, or parse command-line arguments.

The API assumptions that cannot be verified without starting llama-server are
recorded in the result: ``/apply-template`` returns a ``prompt`` field, the
``/completion`` response exposes prompt timings (possibly under ``timings``),
and slot save/restore accepts a basename relative to ``--slot-save-path``.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT / "workstreams/llama.cpp-q3ple-mtp"
CANDIDATE_WORKTREE = ROOT / "workstreams/llama.cpp-mtp-slot-state"
CANDIDATE_BIN = CANDIDATE_WORKTREE / "build-win-cuda-mtp-slot/bin"
BASE_SCRIPT = ROOT / "scripts/q3ple_mtp_ab.py"
CONTEXT_SCRIPT = ROOT / "scripts/q3ple_80k_filled_context.py"
RESULTS_DIR = ROOT / "results/QWEN38-MTP-PROTOTYPE-001"
LOGS_DIR = ROOT / "logs/QWEN38-MTP-PROTOTYPE-001/q3ple_agentic_cache"
PORT = 18087

RUNTIME_CHOICES = ("frozen", "mtp-slot-state")
CANDIDATE_RUNTIME_COMMIT = "73b803464f25fc9054046728bf2ebed5a372737e"
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

EXPECTED_SIDECAR_SHA256 = (
    "7E9F2B282DC62534313B30738E0AD114C14E1A58B9C1E7BB9715DCF9C4CA676E"
)

CTX_SIZE = 81920
TARGET_N_CPU_MOE = 48
THREADS = 11
BATCH_SIZE = 2048
UBATCH_SIZE = 256
DRAFT_THREADS = 8
DRAFT_N_MAX = 3
DRAFT_P_MIN = 0.75
MTP_UBATCH = "64"
WORKING_SET_CAP_GIB = 38
WORKING_SET_CAP_BYTES = WORKING_SET_CAP_GIB * 1024**3
HARD_VRAM_FLOOR_MIB = 768
PUBLISHABLE_VRAM_FLOOR_MIB = 1024
HARD_RAM_FLOOR = 6 * 1024**3
HARD_RSS_CEILING = 50 * 1024**3
HARD_SWAP_GROWTH = 1 * 1024**3
REQUEST_TIMEOUT_SECONDS = 7200

DEFAULT_TARGET_CONTENT_TOKENS = 4096
DEFAULT_CHUNK_TOKENS = 512

DENSE_OVERRIDE = (
    r"^output\.weight$=CUDA0,"
    r"^blk\.48\.attn_.*=CUDA0,"
    r"^blk\.48\.hc_attn_.*=CUDA0,"
    r"^blk\.48\.hc_ffn_.*=CUDA0,"
    r"^blk\.48\.nextn\..*=CUDA0,"
    r"^blk\.48\.ffn_gate_inp.*=CUDA0,"
    r"^blk\.48\.ffn_(gate|up|down)_shexp.*=CUDA0"
)
PINNED_OVERRIDE = (
    DENSE_OVERRIDE + r",^blk\.48\.ffn_(gate|up|down)_exps.*=CUDA_Host"
)

API_ASSUMPTIONS = {
    "apply_template": (
        "POST /apply-template accepts a messages array and returns the final raw "
        "chat prompt in a prompt field."
    ),
    "completion": (
        "POST /completion accepts prompt, n_predict, id_slot, cache_prompt, "
        "temperature, seed, and stream=false."
    ),
    "slot_save": (
        "POST /slots/0?action=save accepts {filename} and writes beneath the "
        "directory passed to --slot-save-path."
    ),
    "slot_restore": (
        "POST /slots/0?action=restore accepts the same basename and returns "
        "n_restored/n_read plus timings.restore_ms."
    ),
    "timings": (
        "Prompt and speculative-decoding counters may be nested under timings; "
        "the probe checks both nested and top-level forms."
    ),
}

KNOWN_RUNTIME_LIMITATION = {
    "id": "slot-save-target-only-mtp-draft-state-not-persisted",
    "classification": "SOURCE_CONFIRMED_BLOCKER",
    "detail": (
        "The frozen server saves and restores ctx_tgt only. It does not serialize "
        "ctx_dft or the speculative driver's pending MTP state, so observational "
        "output/counter equality cannot promote restart-safe MTP persistence."
    ),
    "source": {
        "file": "tools/server/server-context.cpp",
        "save_lines": "2459-2461",
        "restore_lines": "2504-2508",
        "prompt_restore_lines": "2524-2525",
    },
}

CANDIDATE_RUNTIME_LIMITATION = {
    "id": "slot-save-mtp-draft-state-clean-restart-only",
    "classification": "EXPERIMENTAL_DIAGNOSTIC",
    "detail": (
        "The candidate writes and restores target state plus a companion .dft "
        "MTP draft-state file during this clean save/stop/restart probe. This "
        "does not establish crash atomicity, torn-write recovery, fsync/power-loss "
        "durability, concurrent-save behavior, or format migration compatibility."
    ),
    "source": {
        "worktree": str(CANDIDATE_WORKTREE),
        "commit": CANDIDATE_RUNTIME_COMMIT,
        "files": ["tools/server/server-context.cpp", "common/speculative.cpp"],
    },
}


def load_module(path, name):
    """Load a reference script without executing its guarded ``main``."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_tokens(tokens):
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path, value):
    """Write JSON through a same-directory temporary file and atomic replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
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


def safe_tag(tag):
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip(".-")
    if not clean:
        raise ValueError("--tag must contain at least one alphanumeric character")
    return clean[:96]


def allocate_run(tag):
    """Return unique paths, refusing to reuse an existing result or run directory."""

    clean = safe_tag(tag)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1000):
        run_tag = f"agentic-cache-{clean}-r{number}"
        result_path = RESULTS_DIR / f"q3ple_{run_tag.replace('-', '_')}.json"
        run_dir = LOGS_DIR / run_tag
        if result_path.exists() or run_dir.exists():
            continue
        run_dir.mkdir(parents=False, exist_ok=False)
        slot_dir = run_dir / "slots"
        slot_dir.mkdir(parents=False, exist_ok=False)
        return {
            "tag": clean,
            "run_tag": run_tag,
            "run_dir": run_dir,
            "result": result_path,
            "checkpoint": run_dir / "checkpoint.json",
            "raw_log": run_dir / "server-initial.log",
            "restart_log": run_dir / "server-restart.log",
            "telemetry": run_dir / "telemetry-initial.jsonl",
            "restart_telemetry": run_dir / "telemetry-restart.jsonl",
            "slot_dir": slot_dir,
            "slot_file": f"{run_tag}.slot.bin",
        }
    raise RuntimeError("no unused agentic-cache run path")


def git_output(*args):
    return subprocess.check_output(
        ["git", "-C", str(WORKTREE), *args],
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    ).strip()


def git_output_at(worktree, *args):
    return subprocess.check_output(
        ["git", "-C", str(worktree), *args],
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    ).strip()


def select_runtime(runtime_name, base):
    """Return the executable/worktree selected by the explicit runtime flag."""

    if runtime_name == "frozen":
        return {
            "name": runtime_name,
            "worktree": WORKTREE,
            "bin": Path(base.BIN),
            "executable": Path(base.EXE),
            "candidate": False,
        }
    if runtime_name == "mtp-slot-state":
        return {
            "name": runtime_name,
            "worktree": CANDIDATE_WORKTREE,
            "bin": CANDIDATE_BIN,
            "executable": CANDIDATE_BIN / "llama-server.exe",
            "candidate": True,
        }
    raise ValueError(f"unsupported --runtime: {runtime_name}")


def _candidate_runtime_identity(selection, base, context):
    # The candidate changes only slot persistence. Keep the accepted target,
    # sidecar, and frozen reference build pinned as a prerequisite as well.
    frozen_reference = context.validate_frozen_runtime(base)
    worktree = selection["worktree"]
    runtime_bin = selection["bin"]
    executable = selection["executable"]
    if not worktree.is_dir():
        raise RuntimeError(f"missing candidate runtime worktree: {worktree}")
    if not runtime_bin.is_dir():
        raise RuntimeError(f"missing candidate runtime bin: {runtime_bin}")
    commit = git_output_at(worktree, "rev-parse", "HEAD")
    dirty = git_output_at(worktree, "status", "--porcelain")
    if commit != CANDIDATE_RUNTIME_COMMIT:
        raise RuntimeError(f"unexpected candidate runtime commit: {commit}")
    if dirty:
        raise RuntimeError(f"candidate runtime worktree is dirty:\n{dirty}")

    runtime_files = {}
    for filename, expected_sha256 in CANDIDATE_RUNTIME_FILES.items():
        path = runtime_bin / filename
        if not path.is_file():
            raise RuntimeError(f"missing candidate runtime file: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"unexpected candidate {filename} SHA256: {actual_sha256}"
            )
        runtime_files[filename] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }

    if not base.MODEL.is_file():
        raise RuntimeError(f"missing target model: {base.MODEL}")
    if not context.SIDE.is_file():
        raise RuntimeError(f"missing MTP sidecar: {context.SIDE}")
    sidecar_sha256 = sha256_file(context.SIDE)
    if sidecar_sha256 != EXPECTED_SIDECAR_SHA256:
        raise RuntimeError(f"unexpected MTP sidecar SHA256: {sidecar_sha256}")
    return {
        "commit": commit,
        "binary": str(executable),
        "binary_sha256": runtime_files["llama-server.exe"]["sha256"],
        "target_model": str(base.MODEL),
        "target_first_shard_bytes": base.MODEL.stat().st_size,
        "mtp_sidecar": str(context.SIDE),
        "mtp_sidecar_bytes": context.SIDE.stat().st_size,
        "runtime_worktree": str(worktree),
        "runtime_worktree_status": dirty,
        "runtime_bin": str(runtime_bin),
        "runtime_files": runtime_files,
        "frozen_reference_runtime": frozen_reference,
        "binary_stat_bytes": executable.stat().st_size,
        "binary_sha256_rechecked": runtime_files["llama-server.exe"]["sha256"],
        "target_model_sha256": sha256_file(base.MODEL),
        "mtp_sidecar_sha256": sidecar_sha256,
    }


def runtime_identity(selection, base=None, context=None):
    """Capture exact selected runtime and artifact identity before launch."""

    # Keep the pre-selector helper call usable for existing read-only checks.
    if context is None and base is not None and hasattr(selection, "EXE"):
        context = base
        base = selection
        selection = select_runtime("frozen", base)
    if base is None or context is None:
        raise TypeError("runtime_identity requires selection, base, and context")
    if selection["candidate"]:
        identity = _candidate_runtime_identity(selection, base, context)
    else:
        # The context runner owns the accepted commit and binary digest.  Calling
        # it here prevents accidentally measuring a nearby frozen build.
        identity = context.validate_frozen_runtime(base)
        identity.update(
            {
                "runtime_worktree": str(selection["worktree"]),
                "runtime_worktree_status": git_output("status", "--porcelain"),
                "runtime_bin": str(selection["bin"]),
            }
        )
    sidecar_sha256 = identity.get("mtp_sidecar_sha256") or sha256_file(context.SIDE)
    if sidecar_sha256 != EXPECTED_SIDECAR_SHA256:
        raise RuntimeError(f"unexpected MTP sidecar SHA256: {sidecar_sha256}")
    identity.update(
        {
            "runtime": selection["name"],
            "runtime_candidate": selection["candidate"],
            "binary": str(selection["executable"]),
            "binary_stat_bytes": selection["executable"].stat().st_size,
            "binary_sha256_rechecked": sha256_file(selection["executable"]),
            "target_model_sha256": identity.get("target_model_sha256") or sha256_file(base.MODEL),
            "mtp_sidecar_sha256": sidecar_sha256,
        }
    )
    return identity


def server_args(base, context, slot_dir, runtime=None):
    """Build the one exact target+MTP profile, with prompt caching enabled."""

    args = [item for item in base.base_args() if item != "--no-cache-prompt"]
    if runtime is not None:
        args[0] = str(runtime["executable"])
    replacements = {
        "--port": PORT,
        "--ctx-size": CTX_SIZE,
        "--threads": THREADS,
        "--threads-batch": THREADS,
        "--batch-size": BATCH_SIZE,
        "--ubatch-size": UBATCH_SIZE,
        "--n-cpu-moe": TARGET_N_CPU_MOE,
    }
    for flag, value in replacements.items():
        index = args.index(flag)
        args[index + 1] = str(value)
    for flag in ("--cache-type-k", "--cache-type-v"):
        index = args.index(flag)
        args[index + 1] = "q4_0"
    args.extend(
        [
            "--slot-save-path",
            str(slot_dir),
            "-md",
            str(context.SIDE),
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            str(DRAFT_N_MAX),
            "--spec-draft-p-min",
            str(DRAFT_P_MIN),
            "--spec-draft-device",
            "CUDA0",
            "--spec-draft-ngl",
            "0",
            "--spec-draft-threads",
            str(DRAFT_THREADS),
            "--spec-draft-threads-batch",
            str(DRAFT_THREADS),
            "--spec-draft-type-k",
            "q4_0",
            "--spec-draft-type-v",
            "q4_0",
            "--spec-draft-override-tensor",
            PINNED_OVERRIDE,
        ]
    )
    if "--no-cache-prompt" in args:
        raise RuntimeError("profile accidentally disabled prompt caching")
    return args


def environment_for_run():
    environment = os.environ.copy()
    environment.update(
        {
            "QWEN38_MTP_UBATCH": MTP_UBATCH,
            "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD": "1",
            "GGML_CUDA_MOE_CACHE_MB": "0",
            "QWEN38_WORKING_SET_CAP_GIB": str(WORKING_SET_CAP_GIB),
        }
    )
    return environment


def port_free(port=PORT):
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def post_json(port, path, body, timeout=REQUEST_TIMEOUT_SECONDS):
    url = f"http://127.0.0.1:{port}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"non-JSON response from {path}: {payload[:500]}") from error


def get_json(port, path, timeout=5):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=timeout
    ) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def gpu_snapshot():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
    ).strip()
    first = output.splitlines()[0].split(",")
    return {
        "used_mib": int(first[0].strip()),
        "free_mib": int(first[1].strip()),
        "util_pct": int(first[2].strip()),
    }


def process_snapshot(process=None):
    virtual = psutil.virtual_memory()
    pagefile = psutil.swap_memory()
    sample = {
        "t": time.time(),
        "ram_available": virtual.available,
        "ram_total": virtual.total,
        "swap_used": pagefile.used,
        "swap_total": pagefile.total,
        "pagefile": {
            "used": pagefile.used,
            "free": max(pagefile.total - pagefile.used, 0),
            "total": pagefile.total,
            "percent": pagefile.percent,
        },
        "gpu": gpu_snapshot(),
    }
    if process is not None:
        try:
            sample["rss"] = process.memory_info().rss
            sample["read_bytes"] = process.io_counters().read_bytes
        except (psutil.Error, OSError):
            pass
    return sample


def stop_owned_process(process):
    """Stop only the exact Popen handle created by this probe."""

    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(10)


class ServerSession:
    """One owned server process and its interruption-safe telemetry monitor."""

    def __init__(self, process, log_handle, telemetry_handle, preflight):
        self.process = process
        self.log_handle = log_handle
        self.telemetry_handle = telemetry_handle
        self.preflight = preflight
        self.process_info = psutil.Process(process.pid)
        self.done = threading.Event()
        self.samples = []
        self.violations = []
        self.working_set_cap_events = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def start_monitor(self):
        self.thread.start()

    def _monitor(self):
        next_telemetry_write = 0.0
        while not self.done.wait(0.25) and self.process.poll() is None:
            try:
                sample = process_snapshot(self.process_info)
            except BaseException as error:  # telemetry failure is fail-closed
                self.violations.append(f"telemetry-failed:{type(error).__name__}:{error}")
                stop_owned_process(self.process)
                return
            self.samples.append(sample)
            if sample["t"] >= next_telemetry_write:
                self.telemetry_handle.write(json.dumps(sample) + "\n")
                self.telemetry_handle.flush()
                next_telemetry_write = sample["t"] + 1.0
            if (
                not self.working_set_cap_events
                and sample.get("rss", 0) >= WORKING_SET_CAP_BYTES
            ):
                try:
                    event = set_working_set_cap(self.process, WORKING_SET_CAP_BYTES)
                    event.update({"t": time.time(), "rss_before": sample.get("rss", 0)})
                    self.working_set_cap_events.append(event)
                except BaseException as error:
                    self.violations.append(f"working-set-cap-failed:{error}")
                    stop_owned_process(self.process)
                    return
            if sample["ram_available"] < HARD_RAM_FLOOR:
                self.violations.append("ram<6GiB")
                stop_owned_process(self.process)
                return
            if sample["gpu"]["free_mib"] < HARD_VRAM_FLOOR_MIB:
                self.violations.append("vram<768MiB")
                stop_owned_process(self.process)
                return
            if sample.get("rss", 0) > HARD_RSS_CEILING:
                self.violations.append("rss>50GiB")
                stop_owned_process(self.process)
                return
            if sample["swap_used"] - self.preflight["swap_used"] > HARD_SWAP_GROWTH:
                self.violations.append("swap-growth>1GiB")
                stop_owned_process(self.process)
                return

    def close(self):
        self.done.set()
        self.thread.join(2)
        stop_owned_process(self.process)
        try:
            self.log_handle.flush()
        except (OSError, ValueError):
            pass
        try:
            self.telemetry_handle.flush()
        except (OSError, ValueError):
            pass
        self.log_handle.close()
        self.telemetry_handle.close()

    def peak(self):
        samples = self.samples
        ram = [self.preflight["ram_available"]] + [item["ram_available"] for item in samples]
        vram = [self.preflight["gpu"]["free_mib"]] + [item["gpu"]["free_mib"] for item in samples]
        swaps = [self.preflight["swap_used"]] + [item["swap_used"] for item in samples]
        return {
            "min_ram_available": min(ram),
            "min_vram_free_mib": min(vram),
            "max_owned_rss": max([0] + [item.get("rss", 0) for item in samples]),
            "max_swap_used": max(swaps),
            "swap_growth": max(swaps) - self.preflight["swap_used"],
            "sample_count": len(samples),
        }


def set_working_set_cap(process, max_bytes):
    """Use the existing SetProcessWorkingSetSizeEx helper from q3ple_mtp_ab."""

    # This function is replaced with the reference helper in main.  Keeping a
    # narrow wrapper makes the monitor testable and makes the ownership explicit.
    raise RuntimeError("reference working-set helper was not installed")


def launch_server(
    base,
    args,
    environment,
    log_path,
    telemetry_path,
    preflight,
    runtime_bin=None,
):
    log_handle = Path(log_path).open("w", encoding="utf-8", buffering=1)
    telemetry_handle = Path(telemetry_path).open(
        "w", encoding="utf-8", buffering=1
    )
    try:
        process = subprocess.Popen(
            args,
            cwd=Path(runtime_bin) if runtime_bin is not None else base.BIN,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except BaseException:
        log_handle.close()
        telemetry_handle.close()
        raise
    session = ServerSession(process, log_handle, telemetry_handle, preflight)
    session.start_monitor()
    return session


def wait_ready(session, port):
    for _ in range(1200):
        if session.violations or session.process.poll() is not None:
            break
        try:
            status, _ = get_json(port, "/health", timeout=0.5)
            if status == 200:
                return
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError(
        f"server not ready; exit={session.process.poll()} "
        f"violations={session.violations}"
    )


def extract_tokens(response):
    tokens = response.get("tokens") if isinstance(response, dict) else None
    if isinstance(tokens, list):
        return tokens
    raise RuntimeError(f"/tokenize response has no tokens list: {response}")


def tokenize(port, text):
    return extract_tokens(post_json(port, "/tokenize", {"content": text, "add_special": False}, 120))


def build_source_material(base):
    files = sorted(
        path
        for directory in ("src", "common", "ggml/src")
        for path in (WORKTREE / directory).rglob("*")
        if path.is_file() and path.suffix.lower() in (".cpp", ".c", ".h", ".hpp", ".cu", ".cuh")
    )
    selected = files[:300]
    filler_parts = []
    for path in selected:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        filler_parts.append(
            f"\n--- FILE {path.relative_to(WORKTREE)} ---\n{text}"
        )
    filler = "".join(filler_parts)
    needle = "AGENTIC_NEEDLE_BEGIN\n" + base.EXPECTED + "\nAGENTIC_NEEDLE_END\n"
    manifest = {
        "eligible_files": len(files),
        "selected_files": len(selected),
        "read_files": len(filler_parts),
        "first_file": str(selected[0].relative_to(WORKTREE)) if selected else None,
        "last_file": str(selected[-1].relative_to(WORKTREE)) if selected else None,
        "filler_chars": len(filler),
    }
    if not filler:
        raise RuntimeError("no local llama.cpp source files were found")
    return needle, filler, manifest


def size_source_content(base, port, needle, filler, target_tokens):
    """Size deterministic source text close to the requested content-token target."""

    tail = (
        "\nEND_CODE_CORPUS\n"
        "AGENTIC_RETRIEVAL_BEGIN\n"
        "Return exactly the A01 line from AGENTIC_NEEDLE_BEGIN and "
        "AGENTIC_NEEDLE_END, excluding all marker lines. No commentary.\n"
        "AGENTIC_RETRIEVAL_END"
    )
    # Four source characters per token is a useful starting point.  The small
    # fixed iteration count avoids an unbounded server interaction on bad input.
    chars = min(len(filler), max(1024, target_tokens * 4))
    sizing = []
    for _ in range(5):
        user_content = needle + "\nBEGIN_CODE_CORPUS\n" + filler[:chars] + tail
        token_count = len(tokenize(port, user_content))
        sizing.append({"chars": len(user_content), "tokens": token_count})
        if token_count <= 0:
            break
        proposed = int(chars * target_tokens / token_count)
        if proposed == chars:
            break
        chars = max(1024, min(len(filler), proposed))
    user_content = needle + "\nBEGIN_CODE_CORPUS\n" + filler[:chars] + tail
    return user_content, sizing, len(tokenize(port, user_content)), tail


def apply_template(port, messages):
    response = post_json(port, "/apply-template", {"messages": messages}, 120)
    if isinstance(response, str):
        return response
    for key in ("prompt", "content", "result"):
        value = response.get(key) if isinstance(response, dict) else None
        if isinstance(value, str):
            return value
    raise RuntimeError(f"/apply-template response has no raw prompt field: {response}")


def common_token_prefix_length(left, right):
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def build_prefixes(full_prompt_tokens, base_token_count, chunk_tokens):
    """Return exact, monotonically growing token-array prefixes.

    Text slices can retokenize their final word when more text is appended.  That
    would force a partial cache trim and confound this probe, so every step is a
    literal prefix of the one final token sequence.
    """

    if base_token_count <= 0:
        raise RuntimeError("template produced an empty base token prefix")
    prefixes = []
    end = min(chunk_tokens, base_token_count)
    while end < base_token_count:
        prefixes.append(list(full_prompt_tokens[:end]))
        end += chunk_tokens
    prefixes.append(list(full_prompt_tokens[:base_token_count]))
    return prefixes


def nested_value(response, timings, *keys):
    for key in keys:
        value = timings.get(key) if isinstance(timings, dict) else None
        if value is None and isinstance(response, dict):
            value = response.get(key)
        if value is not None:
            return value
    return None


def completion_metrics(response, wall_seconds, request_prompt_tokens):
    timings = response.get("timings") or {}
    cache_n = nested_value(response, timings, "cache_n")
    prompt_n = nested_value(
        response,
        timings,
        "prompt_n",
    )
    prompt_ms = nested_value(response, timings, "prompt_ms", "prompt_time_ms")
    prefill_tps = nested_value(
        response, timings, "prompt_per_second", "prefill_tps", "prompt_tps"
    )
    predicted_n = nested_value(
        response, timings, "predicted_n"
    )
    cache_n = int(cache_n) if cache_n is not None else None
    prompt_n = int(prompt_n) if prompt_n is not None else None
    predicted_n = int(predicted_n) if predicted_n is not None else None
    return {
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prompt_ms": prompt_ms,
        "prefill_tps": prefill_tps,
        "wall_s": wall_seconds,
        "cache_ratio": (
            cache_n / request_prompt_tokens
            if cache_n is not None and request_prompt_tokens
            else None
        ),
        "unseen_suffix_tokens": prompt_n,
        "expected_unseen_suffix_tokens": (
            request_prompt_tokens - cache_n if cache_n is not None else None
        ),
        "context_tokens_after_request": (
            cache_n + prompt_n + (predicted_n or 0)
            if cache_n is not None and prompt_n is not None
            else None
        ),
        "timings": timings,
    }


def run_prefix_steps(port, prefixes):
    records = []
    previous_prefix_tokens = 0
    for step, prefix in enumerate(prefixes, 1):
        started = time.time()
        response = post_json(
            port,
            "/completion",
            {
                "prompt": list(prefix),
                "n_predict": 0,
                "id_slot": 0,
                "cache_prompt": True,
                "temperature": 0,
                "seed": 38027,
                "stream": False,
            },
            REQUEST_TIMEOUT_SECONDS,
        )
        wall = time.time() - started
        metrics = completion_metrics(response, wall, len(prefix))
        expected_cache_n = previous_prefix_tokens
        expected_prompt_n = len(prefix) - expected_cache_n
        metrics.update(
            {
                "step": step,
                "prefix_tokens": len(prefix),
                "prefix_token_ids_sha256": sha256_tokens(prefix),
                "expected_cache_n": expected_cache_n,
                "expected_prompt_n": expected_prompt_n,
                "cache_reuse_exact": metrics["cache_n"] == expected_cache_n,
                "suffix_processing_exact": metrics["prompt_n"] == expected_prompt_n,
            }
        )
        records.append(metrics)
        previous_prefix_tokens = len(prefix)
    return records


def companion_dft_record(slot_path):
    """Capture the optional MTP draft-state companion beside a target slot file."""

    path = Path(f"{slot_path}.dft")
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "bytes": size,
        "sha256": sha256_file(path) if exists else None,
        "nonempty": bool(size),
    }


def slot_save(port, slot_filename, slot_dir):
    started = time.time()
    response = post_json(
        port,
        "/slots/0?action=save",
        {"filename": slot_filename},
        REQUEST_TIMEOUT_SECONDS,
    )
    wall = time.time() - started
    timings = response.get("timings") or {}
    path = slot_dir / slot_filename
    companion = companion_dft_record(path)
    record = {
        "response": response,
        "save_ms": timings.get("save_ms") or response.get("save_ms"),
        "wall_s": wall,
        "n_saved": response.get("n_saved"),
        "n_written": response.get("n_written"),
        "filename": slot_filename,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "companion_dft_after_save": companion,
    }
    if not record["exists"]:
        raise RuntimeError(f"slot save did not create expected file: {path}")
    return record


def slot_restore(port, slot_filename, slot_dir):
    path = slot_dir / slot_filename
    companion_before = companion_dft_record(path)
    started = time.time()
    response = post_json(
        port,
        "/slots/0?action=restore",
        {"filename": slot_filename},
        REQUEST_TIMEOUT_SECONDS,
    )
    wall = time.time() - started
    timings = response.get("timings") or {}
    companion_after = companion_dft_record(path)
    return {
        "response": response,
        "restore_ms": timings.get("restore_ms") or response.get("restore_ms"),
        "wall_s": wall,
        "n_restored": response.get("n_restored"),
        "n_read": response.get("n_read"),
        "filename": slot_filename,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "companion_dft_before_restore": companion_before,
        "companion_dft_after_restore": companion_after,
    }


def completion_with_output(port, prompt_tokens, previous_tokens):
    started = time.time()
    response = post_json(
        port,
        "/completion",
        {
            "prompt": list(prompt_tokens),
            "n_predict": 256,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0,
            "seed": 38027,
            "stream": False,
        },
        REQUEST_TIMEOUT_SECONDS,
    )
    wall = time.time() - started
    content = response.get("content") or response.get("text") or ""
    tokens = tokenize(port, content)
    timings = response.get("timings") or {}
    draft_n = nested_value(response, timings, "draft_n", "n_draft_tokens") or 0
    accepted = nested_value(
        response, timings, "draft_n_accepted", "n_draft_accepted"
    ) or 0
    result = completion_metrics(response, wall, len(prompt_tokens))
    finish_reason = response.get("stop_type") or response.get("finish_reason")
    result.update(
        {
            "response": response,
            "output": content,
            "output_sha256": sha256_text(content),
            "output_token_ids_sha256": sha256_tokens(tokens),
            "output_token_count": len(tokens),
            "draft_n": draft_n,
            "draft_n_accepted": accepted,
            "acceptance": accepted / draft_n if draft_n else None,
            "finish_reason": finish_reason,
            "natural_stop": finish_reason in ("eos", "word", "stop"),
            "expected_cache_n": previous_tokens,
            "expected_prompt_n": len(prompt_tokens) - previous_tokens,
            "cache_reuse_exact": result["cache_n"] == previous_tokens,
            "suffix_processing_exact": (
                result["prompt_n"] == len(prompt_tokens) - previous_tokens
            ),
        }
    )
    return result


def summarize_status(session):
    if session is None:
        return {"violations": [], "peak": {}}
    return {
        "violations": list(session.violations),
        "peak": session.peak(),
        "working_set_cap_events": list(session.working_set_cap_events),
        "server_exit_code": session.process.poll(),
    }


def runtime_limitation(runtime_name):
    """Return the evidence boundary appropriate for the selected runtime."""

    return (
        CANDIDATE_RUNTIME_LIMITATION
        if runtime_name == "mtp-slot-state"
        else KNOWN_RUNTIME_LIMITATION
    )


def candidate_draft_state_gate(save_record, restore_record):
    """Check that the candidate's companion .dft survived the clean restart."""

    after_save = save_record.get("companion_dft_after_save") or {}
    before_restore = restore_record.get("companion_dft_before_restore") or {}
    after_restore = restore_record.get("companion_dft_after_restore") or {}
    checks = {
        "companion_dft_exists_after_save": after_save.get("exists") is True,
        "companion_dft_nonempty_after_save": (
            isinstance(after_save.get("bytes"), int) and after_save["bytes"] > 0
        ),
        "companion_dft_sha256_after_save": bool(after_save.get("sha256")),
        "companion_dft_exists_before_restore": before_restore.get("exists") is True,
        "companion_dft_nonempty_before_restore": (
            isinstance(before_restore.get("bytes"), int)
            and before_restore["bytes"] > 0
        ),
        "companion_dft_sha256_before_restore": bool(before_restore.get("sha256")),
        "companion_dft_bytes_stable": (
            after_save.get("bytes") == before_restore.get("bytes")
        ),
        "companion_dft_sha256_stable": (
            after_save.get("sha256") == before_restore.get("sha256")
            and bool(after_save.get("sha256"))
        ),
        "companion_dft_exists_after_restore": after_restore.get("exists") is True,
        "companion_dft_nonempty_after_restore": (
            isinstance(after_restore.get("bytes"), int)
            and after_restore["bytes"] > 0
        ),
        "companion_dft_sha256_after_restore": bool(after_restore.get("sha256")),
        "companion_dft_sha256_stable_after_restore": (
            before_restore.get("sha256") == after_restore.get("sha256")
            and bool(before_restore.get("sha256"))
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def parser():
    argument_parser = argparse.ArgumentParser(
        description="Probe Q3_PLE+MTP prompt caching and slot save/restore parity."
    )
    argument_parser.add_argument("--tag", required=True, help="unique run label")
    argument_parser.add_argument(
        "--runtime",
        choices=RUNTIME_CHOICES,
        default="frozen",
        help="runtime selector (default: frozen)",
    )
    argument_parser.add_argument(
        "--target-content-tokens",
        type=int,
        default=DEFAULT_TARGET_CONTENT_TOKENS,
        help=f"source-content target token count (default: {DEFAULT_TARGET_CONTENT_TOKENS})",
    )
    argument_parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"approximate cache-prefix increment (default: {DEFAULT_CHUNK_TOKENS})",
    )
    return argument_parser


def failure_record(paths, checkpoint, profile, runtime, args, environment, status, error, context):
    runtime_name = runtime.get("runtime", "frozen") if isinstance(runtime, dict) else "frozen"
    return {
        "schema": 1,
        "status": status,
        "evidence_class": "REJECTED",
        "publishable": False,
        "run_tag": paths.get("run_tag"),
        "runner_pid": os.getpid(),
        "profile": profile,
        "runtime": runtime,
        "args": args,
        "environment_overrides": environment,
        "error_type": type(error).__name__ if error else None,
        "error": str(error) if error else None,
        "traceback": context,
        "checkpoint": str(checkpoint),
        "raw_log": str(paths.get("raw_log")),
        "restart_log": str(paths.get("restart_log")),
        "api_assumptions": API_ASSUMPTIONS,
        "known_runtime_limitation": runtime_limitation(runtime_name),
    }


def main(argv=None):
    cli = parser().parse_args(argv)
    if cli.target_content_tokens < 1:
        parser().error("--target-content-tokens must be positive")
    if cli.chunk_tokens < 1:
        parser().error("--chunk-tokens must be positive")

    paths = None
    checkpoint = None
    result_path = None
    process_sessions = []
    current_session = None
    restart_session = None
    caught_error = None
    runtime = {
        "runtime": cli.runtime,
        "runtime_candidate": cli.runtime == "mtp-slot-state",
    }
    runtime_selection = None
    args = None
    preflight = None
    checkpoint_data = {}
    profile = {
        "ctx_size": CTX_SIZE,
        "n_cpu_moe": TARGET_N_CPU_MOE,
        "threads": THREADS,
        "threads_batch": THREADS,
        "batch_size": BATCH_SIZE,
        "ubatch_size": UBATCH_SIZE,
        "target_kv": "q4_0",
        "draft_kv": "q4_0",
        "draft_experts": "pinned",
        "draft_override": PINNED_OVERRIDE,
        "draft_n_max": DRAFT_N_MAX,
        "draft_p_min": DRAFT_P_MIN,
        "draft_threads": DRAFT_THREADS,
        "parallel_slots": 1,
        "slot_local_cache_prompt": True,
        "global_cache_ram_mib": 0,
        "working_set_cap_gib": WORKING_SET_CAP_GIB,
        "target_content_tokens": cli.target_content_tokens,
        "chunk_tokens": cli.chunk_tokens,
        "runtime": cli.runtime,
    }
    environment_overrides = {
        "QWEN38_MTP_UBATCH": MTP_UBATCH,
        "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD": "1",
        "GGML_CUDA_MOE_CACHE_MB": "0",
        "QWEN38_WORKING_SET_CAP_GIB": str(WORKING_SET_CAP_GIB),
    }

    try:
        paths = allocate_run(cli.tag)
        checkpoint = paths["checkpoint"]
        result_path = paths["result"]
        checkpoint_data = {
            "schema": 1,
            "phase": "initializing",
            "run_tag": paths["run_tag"],
            "runner_pid": os.getpid(),
            "started_at": time.time(),
            "result": str(result_path),
            "checkpoint": str(checkpoint),
            "raw_log": str(paths["raw_log"]),
            "restart_log": str(paths["restart_log"]),
            "telemetry": str(paths["telemetry"]),
            "restart_telemetry": str(paths["restart_telemetry"]),
            "slot_save_path": str(paths["slot_dir"]),
            "slot_filename": paths["slot_file"],
            "profile": profile,
        }
        atomic_json(checkpoint, checkpoint_data)

        base = load_module(BASE_SCRIPT, "q3ple_agentic_base")
        context = load_module(CONTEXT_SCRIPT, "q3ple_agentic_context")
        # Install the existing Windows helper into the monitor without changing
        # the reference script or introducing a second implementation.
        global set_working_set_cap
        set_working_set_cap = base.set_working_set_cap

        runtime_selection = select_runtime(cli.runtime, base)
        runtime = runtime_identity(runtime_selection, base, context)
        args = server_args(base, context, paths["slot_dir"], runtime_selection)
        if (
            not runtime_selection["executable"].is_file()
            or not base.MODEL.is_file()
            or not context.SIDE.is_file()
        ):
            raise RuntimeError(f"{cli.runtime} runtime artifact is missing")
        preflight = process_snapshot()
        if preflight["ram_available"] < 40 * 1024**3:
            raise RuntimeError(f"preflight RAM below 40 GiB: {preflight}")
        if preflight["gpu"]["free_mib"] < 8192:
            raise RuntimeError(f"preflight VRAM below 8192 MiB: {preflight}")
        if preflight["gpu"]["util_pct"] > 15:
            raise RuntimeError(f"preflight GPU utilization above 15%: {preflight}")
        if not port_free(PORT):
            raise RuntimeError(f"port {PORT} is not free")
        checkpoint_data.update(
            {"phase": "preflight_passed", "runtime": runtime, "args": args, "preflight": preflight}
        )
        atomic_json(checkpoint, checkpoint_data)

        environment = environment_for_run()
        current_session = launch_server(
            base,
            args,
            environment,
            paths["raw_log"],
            paths["telemetry"],
            preflight,
            runtime_bin=runtime_selection["bin"],
        )
        process_sessions.append(current_session)
        checkpoint_data.update({"phase": "server_initial_launched", "server_pid": current_session.process.pid})
        atomic_json(checkpoint, checkpoint_data)
        wait_ready(current_session, PORT)
        checkpoint_data["phase"] = "server_initial_ready"
        atomic_json(checkpoint, checkpoint_data)

        needle, filler, corpus = build_source_material(base)
        user_content, sizing, content_tokens, retrieval_suffix = size_source_content(
            base,
            PORT,
            needle,
            filler,
            cli.target_content_tokens,
        )
        messages = [
            {
                "role": "system",
                "content": "Retrieve the requested A01 line from the supplied local source corpus. No commentary.",
            },
            {"role": "user", "content": user_content},
        ]
        raw_prompt = apply_template(PORT, messages)
        marker = raw_prompt.find("AGENTIC_RETRIEVAL_BEGIN")
        if marker < 0:
            raise RuntimeError("final raw prompt has no retrieval boundary")
        full_prompt_tokens = tokenize(PORT, raw_prompt)
        pre_marker_tokens = tokenize(PORT, raw_prompt[:marker])
        base_prefix_tokens = common_token_prefix_length(
            pre_marker_tokens, full_prompt_tokens
        )
        total_prompt_tokens = len(full_prompt_tokens)
        if base_prefix_tokens >= total_prompt_tokens:
            raise RuntimeError("retrieval suffix did not add any prompt tokens")
        prefixes = build_prefixes(
            full_prompt_tokens, base_prefix_tokens, cli.chunk_tokens
        )
        checkpoint_data.update(
            {
                "phase": "prompt_built",
                "corpus": corpus,
                "sizing": sizing,
                "content_tokens": content_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "base_prefix_tokens": base_prefix_tokens,
                "pre_marker_token_count": len(pre_marker_tokens),
                "boundary_token_backoff": len(pre_marker_tokens) - base_prefix_tokens,
                "raw_prompt_chars": len(raw_prompt),
                "raw_prompt_sha256": sha256_text(raw_prompt),
                "full_prompt_token_ids_sha256": sha256_tokens(full_prompt_tokens),
                "messages_sha256": sha256_text(json.dumps(messages, sort_keys=True)),
            }
        )
        atomic_json(checkpoint, checkpoint_data)

        prefix_steps = run_prefix_steps(PORT, prefixes)
        checkpoint_data.update({"phase": "prefix_cache_warmed", "prefix_steps": prefix_steps})
        atomic_json(checkpoint, checkpoint_data)
        save_record = slot_save(PORT, paths["slot_file"], paths["slot_dir"])
        save_record["runtime"] = runtime
        checkpoint_data.update({"phase": "slot_saved", "slot_save": save_record})
        atomic_json(checkpoint, checkpoint_data)
        initial = completion_with_output(
            PORT,
            full_prompt_tokens,
            base_prefix_tokens,
        )
        initial["expected_a01_text"] = base.EXPECTED.splitlines()[0]
        initial["expected_a01_exact"] = initial["output"] == initial["expected_a01_text"]
        checkpoint_data.update({"phase": "initial_retrieval_complete", "initial": initial})
        atomic_json(checkpoint, checkpoint_data)

        current_session.close()
        current_session = None
        time.sleep(2)
        if not port_free(PORT):
            raise RuntimeError("port remained busy after owned initial server stopped")

        restart_preflight = process_snapshot()
        if restart_preflight["ram_available"] < HARD_RAM_FLOOR:
            raise RuntimeError(f"restart RAM below 6 GiB: {restart_preflight}")
        if restart_preflight["gpu"]["free_mib"] < 8192:
            raise RuntimeError(f"restart VRAM below 8192 MiB: {restart_preflight}")
        if restart_preflight["gpu"]["util_pct"] > 15:
            raise RuntimeError(f"restart GPU utilization above 15%: {restart_preflight}")
        current_session = launch_server(
            base,
            args,
            environment,
            paths["restart_log"],
            paths["restart_telemetry"],
            preflight,
            runtime_bin=runtime_selection["bin"],
        )
        process_sessions.append(current_session)
        checkpoint_data.update({"phase": "server_restart_launched", "restart_pid": current_session.process.pid})
        atomic_json(checkpoint, checkpoint_data)
        wait_ready(current_session, PORT)
        checkpoint_data["phase"] = "server_restart_ready"
        atomic_json(checkpoint, checkpoint_data)
        restore_record = slot_restore(PORT, paths["slot_file"], paths["slot_dir"])
        restore_record["runtime"] = runtime
        checkpoint_data.update({"phase": "slot_restored", "slot_restore": restore_record})
        atomic_json(checkpoint, checkpoint_data)
        restored = completion_with_output(
            PORT,
            full_prompt_tokens,
            base_prefix_tokens,
        )
        restored["expected_a01_text"] = base.EXPECTED.splitlines()[0]
        restored["expected_a01_exact"] = restored["output"] == restored["expected_a01_text"]

        # Finish the owned process before promoting a result so postflight state
        # is part of the result itself rather than only the checkpoint sidecar.
        restart_session = current_session
        current_session.close()
        current_session = None
        time.sleep(2)
        postflight_port_free = port_free(PORT)
        if not postflight_port_free:
            raise RuntimeError("port remained busy after owned restart server stopped")

        prefix_cache_exact = all(
            step["cache_reuse_exact"] and step["suffix_processing_exact"]
            for step in prefix_steps
        )
        comparisons = {
            "prefix_cache_incremental_exact": prefix_cache_exact,
            "output_sha256_equal": initial["output_sha256"] == restored["output_sha256"],
            "output_token_ids_sha256_equal": initial["output_token_ids_sha256"] == restored["output_token_ids_sha256"],
            "expected_a01_exact": restored["expected_a01_exact"] and initial["expected_a01_exact"],
            "natural_stop_before_restart": initial["natural_stop"],
            "natural_stop_after_restart": restored["natural_stop"],
            "initial_cache_reuse_exact": initial["cache_reuse_exact"],
            "initial_suffix_processing_exact": initial["suffix_processing_exact"],
            "restored_cache_reuse_exact": restored["cache_reuse_exact"],
            "restored_suffix_processing_exact": restored["suffix_processing_exact"],
            "processed_suffix_tokens_equal": initial["prompt_n"] == restored["prompt_n"],
            "reused_prefix_tokens_equal": initial["cache_n"] == restored["cache_n"],
            "mtp_active_before_restart": initial["draft_n"] > 0,
            "mtp_active_after_restart": restored["draft_n"] > 0,
            "mtp_draft_n_equal": initial["draft_n"] == restored["draft_n"],
            "mtp_accepted_equal": initial["draft_n_accepted"] == restored["draft_n_accepted"],
            "restore_bytes_present": (
                isinstance(restore_record.get("n_read"), int)
                and isinstance(save_record.get("n_written"), int)
            ),
            "restore_bytes_match_save": (
                isinstance(restore_record.get("n_read"), int)
                and restore_record["n_read"] == save_record.get("n_written")
            ),
            "slot_token_counts_present": (
                isinstance(save_record.get("n_saved"), int)
                and isinstance(restore_record.get("n_restored"), int)
            ),
            "saved_slot_tokens_match_base": save_record.get("n_saved") == base_prefix_tokens,
            "restored_slot_tokens_match_base": restore_record.get("n_restored") == base_prefix_tokens,
            "restored_cache_matches_slot": (
                isinstance(restore_record.get("n_restored"), int)
                and restored["cache_n"] == restore_record["n_restored"]
            ),
            "initial_safety_clear": not process_sessions[0].violations,
            "restart_safety_clear": not restart_session.violations,
        }
        parity = all(comparisons.values())
        draft_state = candidate_draft_state_gate(save_record, restore_record)
        is_candidate = cli.runtime == "mtp-slot-state"
        candidate_pass = parity and draft_state["pass"]
        # Frozen source inspection proves that its disk format omits
        # draft/speculative state; do not let coincidentally identical output
        # promote restart safety. The candidate only promotes after every
        # existing parity/safety gate and the .dft gate pass.
        restart_safe_mtp = candidate_pass if is_candidate else False
        publishable = False
        status = (
            ("PASS" if candidate_pass else "FAILED")
            if is_candidate
            else ("BLOCKED" if parity else "FAILED")
        )
        result = {
            "schema": 1,
            "status": status,
            "evidence_class": "MEASURED_DIAGNOSTIC",
            "publishable": publishable,
            "functional_parity_observed": parity,
            "restart_safe_mtp": restart_safe_mtp,
            "known_runtime_limitation": runtime_limitation(cli.runtime),
            "run_tag": paths["run_tag"],
            "runner_pid": os.getpid(),
            "profile": profile,
            "runtime": runtime,
            "args": args,
            "environment_overrides": environment_overrides,
            "raw_log": str(paths["raw_log"]),
            "restart_log": str(paths["restart_log"]),
            "telemetry": str(paths["telemetry"]),
            "restart_telemetry": str(paths["restart_telemetry"]),
            "initial_preflight": preflight,
            "restart_preflight": restart_preflight,
            "slot_save_path": str(paths["slot_dir"]),
            "slot_filename": paths["slot_file"],
            "corpus": corpus,
            "sizing": sizing,
            "content_tokens": content_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "base_prefix_tokens": base_prefix_tokens,
            "pre_marker_token_count": len(pre_marker_tokens),
            "boundary_token_backoff": len(pre_marker_tokens) - base_prefix_tokens,
            "raw_prompt_chars": len(raw_prompt),
            "raw_prompt_sha256": sha256_text(raw_prompt),
            "full_prompt_token_ids_sha256": sha256_tokens(full_prompt_tokens),
            "prefix_steps": prefix_steps,
            "slot_save": save_record,
            "initial": initial,
            "slot_restore": restore_record,
            "restored": restored,
            "comparisons": comparisons,
            "draft_state": draft_state,
            "candidate_pass": candidate_pass if is_candidate else None,
            "initial_resources": summarize_status(process_sessions[0]),
            "restart_resources": summarize_status(restart_session),
            "postflight_port_free": postflight_port_free,
            "postflight_server_exit_codes": [
                session.process.poll() for session in process_sessions
            ],
            "safety": {
                "hard_vram_floor_mib": HARD_VRAM_FLOOR_MIB,
                "publishable_vram_floor_mib": PUBLISHABLE_VRAM_FLOOR_MIB,
                "hard_ram_floor_bytes": HARD_RAM_FLOOR,
                "hard_rss_ceiling_bytes": HARD_RSS_CEILING,
                "hard_swap_growth_bytes": HARD_SWAP_GROWTH,
                "working_set_cap_bytes": WORKING_SET_CAP_BYTES,
            },
            "api_assumptions": API_ASSUMPTIONS,
        }
        # The result path was reserved by allocate_run and is not overwritten.
        atomic_json(result_path, result)
        checkpoint_data.update({"phase": "result_written", "result": result})
        atomic_json(checkpoint, checkpoint_data)
        print(json.dumps(result, indent=2), flush=True)
        if is_candidate:
            return 0 if candidate_pass else 1
        return 2 if parity else 1
    except BaseException as error:
        caught_error = error
        trace = traceback.format_exc()
        status = "BLOCKED" if isinstance(error, FileNotFoundError) else "FAILED"
        if paths is not None and checkpoint is not None:
            checkpoint_data.update(
                {
                    "phase": "error",
                    "status": status,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": trace,
                }
            )
            atomic_json(checkpoint, checkpoint_data)
            if result_path is not None and not result_path.exists():
                atomic_json(
                    result_path,
                    failure_record(
                        paths,
                        checkpoint,
                        profile,
                        runtime,
                        args,
                        environment_overrides,
                        status,
                        error,
                        trace,
                    ),
                )
        print(f"{status}: {error}", flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if current_session is not None:
            current_session.close()
        if checkpoint is not None:
            checkpoint_data.update(
                {
                    "finished_at": time.time(),
                    "phase": (
                        "complete" if caught_error is None else "error_cleaned"
                    ),
                    "postflight_port_free": port_free(PORT),
                    "postflight_server_exit_codes": [
                        session.process.poll() for session in process_sessions
                    ],
                }
            )
            atomic_json(checkpoint, checkpoint_data)


if __name__ == "__main__":
    raise SystemExit(main())
