import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "workstreams/llama.cpp-q3ple-mtp"
BASE_SCRIPT = ROOT / "scripts/q3ple_mtp_ab.py"
SIDE = ROOT / (
    "artifacts/models/Qwen3.8-Flash-Next-MTP-Q4_K_M-FC-HC/"
    "mtp-Qwen3.8-Flash-Next-DOWNQ4-FC-HC-OUTQ4.gguf"
)
EXPECTED_COMMIT = "4c3ee4526a5fb7555c9c5ef02c09ef1ff0cf25cc"
EXPECTED_BINARY_SHA256 = (
    "72BB9839C156ABBBA5D55B0CA3F2D7F89A931ACAA8A32BA40A8D76BBB4B67436"
)
RESULTS_DIR = ROOT / "results/QWEN38-MTP-PROTOTYPE-001"
LOGS_DIR = ROOT / "logs/QWEN38-MTP-PROTOTYPE-001/q3ple_ctx"
RUN_PREFIX = "ctx80k-filled60k"
HARD_VRAM_FLOOR_MIB = 768
PUBLISHABLE_VRAM_FLOOR_MIB = 1024
HARD_RAM_FLOOR = 6 * 1024**3
HARD_RSS_CEILING = 50 * 1024**3
HARD_SWAP_GROWTH = 1 * 1024**3
REQUEST_TIMEOUT_SECONDS = 7200

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


def load_base_module():
    spec = importlib.util.spec_from_file_location("q3base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def next_run_number(run_prefix):
    for number in range(1, 1000):
        tag = f"{run_prefix}-r{number}"
        result = RESULTS_DIR / f"q3ple_{tag.replace('-', '_')}.json"
        log_dir = LOGS_DIR / tag
        if not result.exists() and not log_dir.exists():
            return number
    raise RuntimeError("no unused filled-context run number")


def git_output(*args):
    return subprocess.check_output(
        ["git", "-C", str(WT), *args], text=True, encoding="utf-8"
    ).strip()


def validate_frozen_runtime(base):
    if not base.EXE.is_file():
        raise RuntimeError(f"missing binary: {base.EXE}")
    if not base.MODEL.is_file():
        raise RuntimeError(f"missing target model: {base.MODEL}")
    if not SIDE.is_file():
        raise RuntimeError(f"missing MTP sidecar: {SIDE}")

    commit = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--porcelain")
    binary_sha256 = sha256_file(base.EXE)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"unexpected runtime commit: {commit}")
    if dirty:
        raise RuntimeError(f"frozen runtime worktree is dirty:\n{dirty}")
    if binary_sha256 != EXPECTED_BINARY_SHA256:
        raise RuntimeError(f"unexpected binary SHA256: {binary_sha256}")
    return {
        "commit": commit,
        "binary": str(base.EXE),
        "binary_sha256": binary_sha256,
        "target_model": str(base.MODEL),
        "target_first_shard_bytes": base.MODEL.stat().st_size,
        "mtp_sidecar": str(SIDE),
        "mtp_sidecar_bytes": SIDE.stat().st_size,
    }


def server_args(
    base,
    n_cpu_moe,
    draft_experts,
    fit_target_mib,
    token_embd_cpu,
    gpu_layers,
):
    args = base.base_args()
    for flag, value in (
        ("--ctx-size", 81920),
        ("--threads", 11),
        ("--threads-batch", 11),
        ("--ubatch-size", 256),
        ("--n-cpu-moe", n_cpu_moe),
        ("--fit-target", fit_target_mib),
    ):
        index = args.index(flag)
        args[index + 1] = str(value)
    for flag in ("--cache-type-k", "--cache-type-v"):
        index = args.index(flag)
        args[index + 1] = "q4_0"
    if gpu_layers is not None:
        index = args.index("--n-gpu-layers")
        args[index + 1] = str(gpu_layers)
    if token_embd_cpu:
        index = args.index("--override-tensor")
        args[index + 1] += r",^token_embd\.weight$=CPU"
    return args + [
        "-md",
        str(SIDE),
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "3",
        "--spec-draft-p-min",
        "0.75",
        "--spec-draft-device",
        "CUDA0",
        "--spec-draft-ngl",
        "0",
        "--spec-draft-threads",
        "8",
        "--spec-draft-threads-batch",
        "8",
        "--spec-draft-type-k",
        "q4_0",
        "--spec-draft-type-v",
        "q4_0",
        "--spec-draft-override-tensor",
        PINNED_OVERRIDE if draft_experts == "pinned" else DENSE_OVERRIDE,
    ]


def build_content(base):
    files = sorted(
        path
        for directory in ("src", "common", "ggml/src")
        for path in (WT / directory).rglob("*")
        if path.suffix in (".cpp", ".c", ".h", ".hpp", ".cu", ".cuh")
    )
    selected = files[:300]
    filler = "".join(
        f"\n--- FILE {path.relative_to(WT)} ---\n"
        + path.read_text(errors="ignore")
        for path in selected
    )
    prefix = (
        "AGENTIC_NEEDLE_BEGIN\n"
        + base.EXPECTED
        + "\nAGENTIC_NEEDLE_END\n\nBEGIN_CODE_CORPUS\n"
    )
    suffix = (
        "\nEND_CODE_CORPUS\nReturn exactly the text between "
        "AGENTIC_NEEDLE_BEGIN and AGENTIC_NEEDLE_END, excluding the marker "
        "lines. No commentary."
    )
    manifest = {
        "eligible_files": len(files),
        "selected_files": len(selected),
        "first_file": str(selected[0].relative_to(WT)),
        "last_file": str(selected[-1].relative_to(WT)),
        "filler_chars": len(filler),
    }
    return filler, prefix, suffix, manifest


def stop_owned_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(10)


def summarize_peak(preflight, samples):
    if preflight is None:
        return {}
    ram_values = [preflight["ram_available"]] + [
        sample["ram_available"] for sample in samples
    ]
    vram_values = [preflight["gpu"]["free_mib"]] + [
        sample["gpu"]["free_mib"] for sample in samples
    ]
    swap_values = [preflight["swap_used"]] + [
        sample["swap_used"] for sample in samples
    ]
    return {
        "min_ram_available": min(ram_values),
        "min_vram_free_mib": min(vram_values),
        "max_owned_rss": max(
            [0] + [sample.get("rss", 0) for sample in samples]
        ),
        "max_swap_used": max(swap_values),
        "swap_growth": max(swap_values) - preflight["swap_used"],
        "sample_count": len(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-experts", choices=("pinned", "cpu"), default="pinned"
    )
    parser.add_argument("--n-cpu-moe", type=int, choices=(47, 48), default=47)
    parser.add_argument("--working-set-cap-gib", type=float, default=0)
    parser.add_argument("--fit-target-mib", type=int, default=1024)
    parser.add_argument("--token-embd-cpu", action="store_true")
    parser.add_argument("--gpu-layers", type=int)
    cli = parser.parse_args()
    if cli.working_set_cap_gib and not 32 <= cli.working_set_cap_gib <= 42:
        parser.error("--working-set-cap-gib must be between 32 and 42")
    if not 1024 <= cli.fit_target_mib <= 2048:
        parser.error("--fit-target-mib must be between 1024 and 2048")
    if cli.gpu_layers is not None and not 1 <= cli.gpu_layers <= 49:
        parser.error("--gpu-layers must be between 1 and 49")
    working_set_cap_bytes = int(cli.working_set_cap_gib * 1024**3)
    base = load_base_module()
    profile_suffix = (
        "" if cli.draft_experts == "pinned" and cli.n_cpu_moe == 47
        else f"-n{cli.n_cpu_moe}-{cli.draft_experts}experts"
    )
    if working_set_cap_bytes:
        cap_label = str(cli.working_set_cap_gib).rstrip("0").rstrip(".")
        profile_suffix += "-ws" + cap_label.replace(".", "p")
    if cli.fit_target_mib != 1024:
        profile_suffix += f"-fit{cli.fit_target_mib}"
    if cli.token_embd_cpu:
        profile_suffix += "-tembedcpu"
    if cli.gpu_layers is not None:
        profile_suffix += f"-ngl{cli.gpu_layers}"
    run_prefix = RUN_PREFIX + profile_suffix
    run_number = next_run_number(run_prefix)
    run_tag = f"{run_prefix}-r{run_number}"
    log_dir = LOGS_DIR / run_tag
    log_dir.mkdir(parents=True, exist_ok=False)
    server_log = log_dir / "mtp.log"
    telemetry_path = log_dir / "telemetry.jsonl"
    checkpoint_path = log_dir / "checkpoint.json"
    result_path = RESULTS_DIR / f"q3ple_{run_tag.replace('-', '_')}.json"

    checkpoint = {
        "schema": 2,
        "run_tag": run_tag,
        "runner_pid": os.getpid(),
        "profile": {
            "n_cpu_moe": cli.n_cpu_moe,
            "draft_experts": cli.draft_experts,
            "working_set_cap_gib": cli.working_set_cap_gib or None,
            "fit_target_mib": cli.fit_target_mib,
            "token_embd_cpu": cli.token_embd_cpu,
            "gpu_layers": cli.gpu_layers if cli.gpu_layers is not None else "auto",
        },
        "phase": "initializing",
        "started_at": time.time(),
        "result": str(result_path),
        "server_log": str(server_log),
        "telemetry": str(telemetry_path),
    }
    atomic_json(checkpoint_path, checkpoint)

    process = None
    process_info = None
    server_handle = None
    telemetry_handle = None
    monitor_thread = None
    done = threading.Event()
    samples = []
    violations = []
    working_set_cap_events = []
    caught_error = None
    runtime = None
    args = None
    preflight = None
    raw_tokens = None
    content = None
    corpus = None
    sizing = []

    try:
        runtime = validate_frozen_runtime(base)
        args = server_args(
            base,
            cli.n_cpu_moe,
            cli.draft_experts,
            cli.fit_target_mib,
            cli.token_embd_cpu,
            cli.gpu_layers,
        )
        preflight = base.snap()
        if preflight["ram_available"] < 40 * 1024**3:
            raise RuntimeError(f"preflight RAM below 40 GiB: {preflight}")
        if preflight["gpu"]["free_mib"] < 8192:
            raise RuntimeError(f"preflight VRAM below 8192 MiB: {preflight}")
        if preflight["gpu"]["util_pct"] > 15:
            raise RuntimeError(f"preflight GPU utilization above 15%: {preflight}")
        if not base.port_free():
            raise RuntimeError(f"port {base.PORT} is not free")

        checkpoint.update(
            {
                "phase": "preflight_passed",
                "runtime": runtime,
                "preflight": preflight,
                "args": args,
            }
        )
        atomic_json(checkpoint_path, checkpoint)

        environment = os.environ.copy()
        environment["QWEN38_MTP_UBATCH"] = "64"
        environment["GGML_CUDA_MOE_CACHE_MB"] = "0"
        if cli.draft_experts == "pinned":
            environment["QWEN38_MTP_DRAFT_EXPERT_OFFLOAD"] = "1"
        else:
            environment.pop("QWEN38_MTP_DRAFT_EXPERT_OFFLOAD", None)

        server_handle = server_log.open("w", encoding="utf-8", buffering=1)
        telemetry_handle = telemetry_path.open(
            "w", encoding="utf-8", buffering=1
        )
        process = subprocess.Popen(
            args,
            cwd=base.BIN,
            env=environment,
            stdout=server_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process_info = psutil.Process(process.pid)
        checkpoint.update({"phase": "server_launched", "server_pid": process.pid})
        atomic_json(checkpoint_path, checkpoint)

        def monitor():
            next_write = 0.0
            while not done.wait(0.25) and process.poll() is None:
                sample = base.snap(process_info)
                samples.append(sample)
                if sample["t"] >= next_write:
                    telemetry_handle.write(json.dumps(sample) + "\n")
                    next_write = sample["t"] + 1.0
                if (
                    working_set_cap_bytes
                    and not working_set_cap_events
                    and sample.get("rss", 0) >= working_set_cap_bytes
                ):
                    try:
                        event = base.set_working_set_cap(
                            process, working_set_cap_bytes
                        )
                        event.update(
                            {"t": time.time(), "rss_before": sample.get("rss", 0)}
                        )
                        working_set_cap_events.append(event)
                    except Exception as error:
                        violations.append(f"working-set-cap-failed:{error}")
                        stop_owned_process(process)
                        return
                if sample["ram_available"] < HARD_RAM_FLOOR:
                    violations.append("ram<6GiB")
                    stop_owned_process(process)
                    return
                if sample["gpu"]["free_mib"] < HARD_VRAM_FLOOR_MIB:
                    violations.append("vram<768MiB")
                    stop_owned_process(process)
                    return
                if sample.get("rss", 0) > HARD_RSS_CEILING:
                    violations.append("rss>50GiB")
                    stop_owned_process(process)
                    return
                if sample["swap_used"] - preflight["swap_used"] > HARD_SWAP_GROWTH:
                    violations.append("swap-growth>1GiB")
                    stop_owned_process(process)
                    return

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

        ready = False
        for _ in range(1200):
            if violations or process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{base.PORT}/health", timeout=0.5
                ) as response:
                    if response.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(0.25)
        if not ready:
            raise RuntimeError(
                f"server not ready; exit={process.poll()} violations={violations}"
            )
        checkpoint["phase"] = "server_ready"
        atomic_json(checkpoint_path, checkpoint)

        filler, prefix, suffix, corpus = build_content(base)
        nchars = min(220000, len(filler))
        sizing = []
        for _ in range(3):
            content = prefix + filler[:nchars] + suffix
            token_count = len(
                base.post(
                    "/tokenize",
                    {"content": content, "add_special": False},
                    timeout=120,
                )["tokens"]
            )
            sizing.append({"chars": len(content), "tokens": token_count})
            nchars = max(
                1000,
                min(len(filler), int(nchars * 60000 / max(token_count, 1))),
            )

        content = prefix + filler[:nchars] + suffix
        raw_tokens = len(
            base.post(
                "/tokenize",
                {"content": content, "add_special": False},
                timeout=120,
            )["tokens"]
        )
        checkpoint.update(
            {
                "phase": "corpus_sized",
                "corpus": corpus,
                "sizing": sizing,
                "content_chars": len(content),
                "raw_content_tokens": raw_tokens,
            }
        )
        atomic_json(checkpoint_path, checkpoint)
        print(
            f"run={run_tag} raw_tokens={raw_tokens} chars={len(content)} "
            f"server_pid={process.pid}",
            flush=True,
        )

        request = {
            "model": "model",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Follow the user instruction exactly. Retrieve the "
                        "requested block from the supplied code corpus."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "seed": 38027,
            "max_tokens": 256,
            "stream": False,
            "cache_prompt": False,
        }
        checkpoint.update({"phase": "request_started", "request_at": time.time()})
        atomic_json(checkpoint_path, checkpoint)
        started = time.time()
        response = base.post(
            "/v1/chat/completions", request, timeout=REQUEST_TIMEOUT_SECONDS
        )
        wall_seconds = time.time() - started
        checkpoint.update(
            {"phase": "response_received", "response_at": time.time()}
        )
        atomic_json(checkpoint_path, checkpoint)

        output = response["choices"][0]["message"].get("content") or ""
        output_tokens = base.post(
            "/tokenize", {"content": output, "add_special": False}, timeout=120
        )["tokens"]
        expected_tokens = base.post(
            "/tokenize",
            {"content": base.EXPECTED, "add_special": False},
            timeout=120,
        )["tokens"]
        timings = response.get("timings") or {}
        usage = response.get("usage") or {}
        draft_n = timings.get("draft_n", 0)
        accepted = timings.get("draft_n_accepted", 0)

        done.set()
        if monitor_thread is not None:
            monitor_thread.join(2)
        peak = summarize_peak(preflight, samples)
        exact_text = output == base.EXPECTED
        exact_tokens = output_tokens == expected_tokens
        publishable = (
            exact_text
            and exact_tokens
            and not violations
            and draft_n > 0
            and peak["min_vram_free_mib"] >= PUBLISHABLE_VRAM_FLOOR_MIB
        )
        result = {
            "schema": 2,
            "run_tag": run_tag,
            "runner_pid": os.getpid(),
            "profile": {
                "n_cpu_moe": cli.n_cpu_moe,
                "draft_experts": cli.draft_experts,
                "working_set_cap_gib": cli.working_set_cap_gib or None,
            },
            "evidence_class": "MEASURED" if publishable else "REJECTED",
            "publishable": publishable,
            "runtime": runtime,
            "args": args,
            "environment_overrides": {
                "QWEN38_MTP_UBATCH": "64",
                "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD": (
                    "1" if cli.draft_experts == "pinned" else None
                ),
                "GGML_CUDA_MOE_CACHE_MB": "0",
                "QWEN38_WORKING_SET_CAP_GIB": (
                    str(cli.working_set_cap_gib)
                    if working_set_cap_bytes
                    else None
                ),
            },
            "allocated_ctx": 81920,
            "raw_content_tokens": raw_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_tps": timings.get("prompt_per_second"),
            "decode_tps": timings.get("predicted_per_second"),
            "draft_n": draft_n,
            "draft_n_accepted": accepted,
            "acceptance": accepted / draft_n if draft_n else None,
            "wall_s": wall_seconds,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "exact_text": exact_text,
            "exact_tokens": exact_tokens,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "expected_sha256": hashlib.sha256(base.EXPECTED.encode()).hexdigest(),
            "output_token_ids_sha256": hashlib.sha256(
                json.dumps(output_tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "expected_token_ids_sha256": hashlib.sha256(
                json.dumps(expected_tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "corpus": corpus,
            "sizing": sizing,
            "peak": peak,
            "working_set_cap_events": working_set_cap_events,
            "safety": {
                "hard_vram_floor_mib": HARD_VRAM_FLOOR_MIB,
                "publishable_vram_floor_mib": PUBLISHABLE_VRAM_FLOOR_MIB,
                "hard_ram_floor_bytes": HARD_RAM_FLOOR,
                "hard_rss_ceiling_bytes": HARD_RSS_CEILING,
                "hard_swap_growth_bytes": HARD_SWAP_GROWTH,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            },
            "violations": violations,
            "server_pid": process.pid,
            "server_log": str(server_log),
            "telemetry": str(telemetry_path),
            "checkpoint": str(checkpoint_path),
        }
        atomic_json(result_path, result)
        checkpoint.update(
            {
                "phase": "result_written",
                "result_written_at": time.time(),
                "publishable": publishable,
            }
        )
        atomic_json(checkpoint_path, checkpoint)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        caught_error = error
        error_traceback = traceback.format_exc()
        checkpoint.update(
            {
                "phase": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": error_traceback,
                "violations": violations,
            }
        )
        atomic_json(checkpoint_path, checkpoint)
        if not result_path.exists():
            failure = {
                "schema": 2,
                "run_tag": run_tag,
                "runner_pid": os.getpid(),
                "profile": {
                    "n_cpu_moe": cli.n_cpu_moe,
                    "draft_experts": cli.draft_experts,
                    "working_set_cap_gib": cli.working_set_cap_gib or None,
                },
                "evidence_class": "REJECTED",
                "publishable": False,
                "status": "FAILED",
                "runtime": runtime,
                "args": args,
                "environment_overrides": {
                    "QWEN38_MTP_UBATCH": "64",
                    "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD": (
                        "1" if cli.draft_experts == "pinned" else None
                    ),
                    "GGML_CUDA_MOE_CACHE_MB": "0",
                    "QWEN38_WORKING_SET_CAP_GIB": (
                        str(cli.working_set_cap_gib)
                        if working_set_cap_bytes
                        else None
                    ),
                },
                "allocated_ctx": 81920,
                "raw_content_tokens": raw_tokens,
                "content_chars": len(content) if content is not None else None,
                "corpus": corpus,
                "sizing": sizing,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": error_traceback,
                "peak": summarize_peak(preflight, samples),
                "working_set_cap_events": working_set_cap_events,
                "violations": violations,
                "server_pid": process.pid if process is not None else None,
                "server_log": str(server_log),
                "telemetry": str(telemetry_path),
                "checkpoint": str(checkpoint_path),
            }
            atomic_json(result_path, failure)
        raise
    finally:
        done.set()
        if monitor_thread is not None:
            monitor_thread.join(2)
        stop_owned_process(process)
        if telemetry_handle is not None:
            telemetry_handle.close()
        if server_handle is not None:
            server_handle.close()
        time.sleep(2)
        postflight = {
            "server_exit_code": process.poll() if process is not None else None,
            "port_free": base.port_free(),
            "finished_at": time.time(),
        }
        checkpoint.update(
            {
                "phase": "complete" if caught_error is None else "error_cleaned",
                "postflight": postflight,
            }
        )
        atomic_json(checkpoint_path, checkpoint)
        if result_path.exists():
            result_record = json.loads(result_path.read_text(encoding="utf-8"))
            result_record["postflight"] = postflight
            atomic_json(result_path, result_record)


if __name__ == "__main__":
    main()
