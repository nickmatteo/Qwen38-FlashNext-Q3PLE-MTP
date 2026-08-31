"""Resume an immutable Q3_PLE+MTP slot without replaying its prefix.

This is an experimental, import-safe diagnostic harness for the ``mtp-slot-state``
runtime (commit ``73b803464f25fc9054046728bf2ebed5a372737e``).  It restores the
known-good 16K target slot and its ``.dft`` speculative-state companion into a
new run directory, reconstructs the source-backed prompt through the server's
tokenizer, proves that the saved 16,366-token prefix is byte-for-byte the same
token vector, and appends a deterministic literal-token source block in growing
chunks.  It does not pretend that retokenizing a longer version of the original
message preserves an arbitrary BPE boundary.
The server must report the prior prefix as ``cache_n`` and only the new chunk as
``prompt_n`` at every step.  The original slot files are copied read-only in
spirit; they are never modified or physically appended to.

The two-file target/``.dft`` persistence format is intentionally called out in
the result as non-atomic.  This probe demonstrates a clean restore/extension,
not crash consistency, fsync/power-loss durability, concurrent-save safety, or
format migration compatibility.

Importing this module only defines constants and helpers.  It does not load a
runner, inspect the GPU, parse command-line arguments, start a server, or run a
benchmark.  The live path is guarded by ``main``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import time
import traceback
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_SCRIPT = ROOT / "scripts/q3ple_agentic_cache_probe.py"
BASE_SCRIPT = ROOT / "scripts/q3ple_mtp_ab.py"
CONTEXT_SCRIPT = ROOT / "scripts/q3ple_80k_filled_context.py"
RESULTS_DIR = ROOT / "results/QWEN38-MTP-PROTOTYPE-001"
LOGS_DIR = ROOT / "logs/QWEN38-MTP-PROTOTYPE-001/q3ple_agentic_extend"

# A separate port is owned by this harness only.  It is deliberately distinct
# from the cache probe's 18087 port and the old 80K runner's 18086 port.
PORT = 18088
CTX_SIZE = 81920
BASE_PREFIX_TOKENS = 16366
BASE_CONTENT_TARGET = 16384
DEFAULT_TARGET_TOKENS = 32768
DEFAULT_CHUNK_TOKENS = 512

CANDIDATE_RUNTIME_COMMIT = "73b803464f25fc9054046728bf2ebed5a372737e"
SOURCE_RESULT = RESULTS_DIR / (
    "q3ple_agentic_cache_cache16k_mtp_slot_state_v2_r1.json"
)
# Sealed bytes from the successful 16K run.  Keeping these digests here makes
# the "immutable source pair" gate fail closed if another process edits or
# replaces either file after the original evidence was recorded.
EXPECTED_BASE_TARGET_SLOT_SHA256 = (
    "B1C1AC5F3061FB6315FCF7AE62937698A9D2E86D4ED5A80DB4A6D90FBED37E71"
)
EXPECTED_BASE_DRAFT_SLOT_SHA256 = (
    "46B2C55EA4295F2175F0A500D2EF367A35576BF15C97C45000F5012078250ACF"
)
EXPECTED_BASE_TARGET_SLOT_BYTES = 287205056
EXPECTED_BASE_DRAFT_SLOT_BYTES = 9860656

HARD_VRAM_FLOOR_MIB = 768
RELEASE_VRAM_FLOOR_MIB = 1024
HARD_RAM_FLOOR = 6 * 1024**3
PREFLIGHT_RAM_FLOOR = 40 * 1024**3
HARD_RSS_CEILING = 50 * 1024**3
HARD_SWAP_GROWTH = 1 * 1024**3
WORKING_SET_CAP_GIB = 38
WORKING_SET_CAP_BYTES = WORKING_SET_CAP_GIB * 1024**3
REQUEST_TIMEOUT_SECONDS = 7200

NON_ATOMIC_LIMITATION = {
    "id": "two-file-target-and-draft-persistence-is-not-atomic",
    "classification": "EXPERIMENTAL_DIAGNOSTIC",
    "detail": (
        "The target slot and its .dft speculative-state companion are written "
        "as two independent files. This clean extension does not establish "
        "crash atomicity, fsync/power-loss durability, torn-write recovery, "
        "concurrent-save safety, or format migration compatibility."
    ),
}


def load_module(path: Path, name: str):
    """Import an import-safe reference module without invoking its main."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_tokens(tokens) -> str:
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value) -> None:
    """Persist evidence through a same-directory temporary file and replace."""

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


def safe_tag(value: str) -> str:
    clean = "".join(char if (char.isalnum() or char in "_.-") else "-" for char in value)
    clean = clean.strip(".-")[:96]
    if not clean:
        raise ValueError("--tag must contain at least one alphanumeric character")
    return clean


def allocate_paths(tag: str) -> dict:
    clean = safe_tag(tag)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1000):
        run_tag = f"agentic-extend-{clean}-r{number}"
        result = RESULTS_DIR / f"q3ple_{run_tag.replace('-', '_')}.json"
        run_dir = LOGS_DIR / run_tag
        if result.exists() or run_dir.exists():
            continue
        run_dir.mkdir(parents=False, exist_ok=False)
        slots = run_dir / "slots"
        slots.mkdir(parents=False, exist_ok=False)
        return {
            "tag": clean,
            "run_tag": run_tag,
            "result": result,
            "run_dir": run_dir,
            "slots": slots,
            "raw_log": run_dir / "server.log",
            "telemetry": run_dir / "telemetry.jsonl",
            "restart_log": run_dir / "server-restart.log",
            "restart_telemetry": run_dir / "telemetry-restart.jsonl",
            "checkpoint": run_dir / "checkpoint.json",
            "slot_filename": f"{run_tag}.slot.bin",
        }
    raise RuntimeError("no unused agentic extension run path")


def port_free(port: int = PORT) -> bool:
    import socket

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def post_json(port: int, path: str, body: dict, timeout=REQUEST_TIMEOUT_SECONDS):
    import urllib.request

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if not raw:
        return {}
    return json.loads(raw)


def get_json(port: int, path: str, timeout=5):
    import urllib.request

    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=timeout
    ) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def completion_metrics(probe, response: dict, wall_seconds: float, request_tokens: int):
    metrics = probe.completion_metrics(response, wall_seconds, request_tokens)
    return metrics


def completion_prefix(probe, port: int, prompt_tokens, expected_cache: int) -> dict:
    started = time.time()
    response = post_json(
        port,
        "/completion",
        {
            "prompt": list(prompt_tokens),
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
    metrics = completion_metrics(probe, response, wall, len(prompt_tokens))
    expected_prompt = len(prompt_tokens) - expected_cache
    metrics.update(
        {
            "prefix_tokens": len(prompt_tokens),
            "prefix_token_ids_sha256": sha256_tokens(prompt_tokens),
            "expected_cache_n": expected_cache,
            "expected_prompt_n": expected_prompt,
            "cache_reuse_exact": metrics.get("cache_n") == expected_cache,
            "suffix_processing_exact": metrics.get("prompt_n") == expected_prompt,
            "response": response,
        }
    )
    return metrics


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ValueError("slot payload is truncated")
    return struct.unpack_from("<I", data, offset)[0]


def parse_serialized_prompt(
    data: bytes, expected_count: int | None = None, *, state_header: bool = True
) -> dict:
    """Parse the server_tokens serialization at a GGCC-state payload boundary.

    ``server_tokens::serialize`` is marker (-1), version (1), a u32 token count,
    the int32 token IDs, and a u32 media-key count.  The state file's first 12
    bytes are the llama state header; no GPU or llama runtime is needed here.
    """

    # ``LLAMA_STATE_SEQ_SAVE_FILE_MAGIC`` is the little-endian bytes ``qsgg``
    # (the human-readable constant is commonly described as GGCC/GGUF state).
    if state_header:
        if len(data) < 24 or data[:4] != b"qsgg":
            raise ValueError("slot state does not begin with the llama qsgg magic")
        payload = 12
    else:
        if len(data) < 16:
            raise ValueError("serialized prompt payload is truncated")
        payload = 0
    marker = _u32(data, payload)
    version = _u32(data, payload + 4)
    count = _u32(data, payload + 8)
    if marker != 0xFFFFFFFF or version != 1:
        raise ValueError(
            f"unexpected server token header marker=0x{marker:08x} version={version}"
        )
    if expected_count is not None and count != expected_count:
        raise ValueError(f"saved token count {count} != expected {expected_count}")
    token_start = payload + 12
    token_end = token_start + count * 4
    media_count_offset = token_end
    media_count = _u32(data, media_count_offset)
    if media_count != 0:
        raise ValueError("16K source slot unexpectedly contains media tokens")
    if token_end + 4 > len(data):
        raise ValueError("slot token vector is truncated")
    tokens = list(struct.unpack_from(f"<{count}i", data, token_start))
    packed_bytes = token_end + 4 - payload
    return {
        "count": count,
        "tokens": tokens,
        "token_ids_sha256": sha256_tokens(tokens),
        "packed_bytes": packed_bytes,
        "media_count": media_count,
        "payload_offset": payload,
    }


def parse_slot_pair(
    target_path: Path,
    expected_count: int | None = None,
    expected_tokens=None,
) -> dict:
    """Parse and cross-check one target/``.dft`` pair without llama.cpp."""

    target_path = Path(target_path)
    draft_path = Path(f"{target_path}.dft")
    if not target_path.is_file() or not draft_path.is_file():
        raise FileNotFoundError(f"missing target/.dft slot pair: {target_path}")
    target_bytes = target_path.read_bytes()
    draft_bytes = draft_path.read_bytes()
    target_sha = sha256_file(target_path)
    draft_sha = sha256_file(draft_path)
    target = parse_serialized_prompt(target_bytes, expected_count)
    if len(draft_bytes) < 28 or draft_bytes[:4] != b"qsgg":
        raise ValueError("draft state does not begin with the llama qsgg magic")
    draft_payload = 12
    if _u32(draft_bytes, draft_payload) != 0x54464453:
        raise ValueError("draft state is missing SDFT envelope")
    if _u32(draft_bytes, draft_payload + 4) != 1:
        raise ValueError("unsupported SDFT envelope version")
    prompt_bytes = _u32(draft_bytes, draft_payload + 8)
    spec_bytes = _u32(draft_bytes, draft_payload + 12)
    prompt_start = draft_payload + 16
    prompt_end = prompt_start + prompt_bytes
    spec_end = prompt_end + spec_bytes
    if spec_bytes <= 0 or spec_end > len(draft_bytes):
        raise ValueError("truncated or empty SDFT speculative-state payload")
    draft_prompt = parse_serialized_prompt(
        draft_bytes[prompt_start:prompt_end], expected_count, state_header=False
    )
    if draft_prompt["packed_bytes"] != prompt_bytes:
        raise ValueError("SDFT prompt byte length does not match serialized prompt")
    target_prompt = target["tokens"]
    if draft_prompt["tokens"] != target_prompt:
        raise ValueError("target and .dft serialized token vectors differ")
    if expected_tokens is not None and list(expected_tokens) != target_prompt:
        raise ValueError("saved target/.dft token vector differs from expected prefix")
    return {
        "target_path": str(target_path),
        "target_bytes": len(target_bytes),
        "target_sha256": target_sha,
        "draft_path": str(draft_path),
        "draft_bytes": len(draft_bytes),
        "draft_sha256": draft_sha,
        "target_token_count": target["count"],
        "target_token_ids_sha256": target["token_ids_sha256"],
        "draft_prompt_token_count": draft_prompt["count"],
        "draft_prompt_token_ids_sha256": draft_prompt["token_ids_sha256"],
        "draft_prompt_bytes": prompt_bytes,
        "draft_spec_bytes": spec_bytes,
        "target_draft_tokens_equal": True,
        "expected_tokens_equal": expected_tokens is None or list(expected_tokens) == target_prompt,
    }


def offline_slot_pair_check(source_result: dict) -> dict:
    """Read/hash an immutable target and ``.dft`` pair without a server."""

    save = source_result.get("slot_save") or {}
    target_path = Path(save.get("path", ""))
    count = source_prefix_count(source_result)
    parsed = parse_slot_pair(target_path, count)
    expected_target_sha = save.get("sha256") or (
        EXPECTED_BASE_TARGET_SLOT_SHA256 if count == BASE_PREFIX_TOKENS else None
    )
    expected_draft = save.get("companion_dft_after_save") or {}
    expected_draft_sha = expected_draft.get("sha256") or (
        EXPECTED_BASE_DRAFT_SLOT_SHA256 if count == BASE_PREFIX_TOKENS else None
    )
    expected_target_bytes = save.get("bytes") or (
        EXPECTED_BASE_TARGET_SLOT_BYTES if count == BASE_PREFIX_TOKENS else None
    )
    expected_draft_bytes = expected_draft.get("bytes") or (
        EXPECTED_BASE_DRAFT_SLOT_BYTES if count == BASE_PREFIX_TOKENS else None
    )
    if not all(
        value is not None
        for value in (
            expected_target_sha,
            expected_draft_sha,
            expected_target_bytes,
            expected_draft_bytes,
        )
    ):
        raise ValueError("source result lacks a complete target/.dft identity")
    if parsed["target_sha256"] != expected_target_sha:
        raise ValueError("immutable target slot hash changed from source evidence")
    if parsed["draft_sha256"] != expected_draft_sha:
        raise ValueError("immutable .dft companion hash changed from source evidence")
    if parsed["target_bytes"] != expected_target_bytes:
        raise ValueError("immutable target slot byte length changed")
    if parsed["draft_bytes"] != expected_draft_bytes:
        raise ValueError("immutable .dft byte length changed")
    expected_token_sha = (
        source_result.get("target_base_prefix_token_ids_sha256")
        or save.get("token_ids_sha256")
        or (save.get("offline_pair") or {}).get("target_token_ids_sha256")
    )
    if expected_token_sha and parsed["target_token_ids_sha256"] != expected_token_sha:
        raise ValueError("immutable slot token-vector hash changed from source evidence")
    parsed["source_prefix_tokens"] = count
    return parsed


def source_prompt(probe, base, context, port: int, content_target: int):
    """Rebuild a prompt from local llama.cpp sources through the live tokenizer."""

    needle, filler, corpus = probe.build_source_material(base)
    content, sizing, content_tokens, retrieval_suffix = probe.size_source_content(
        base, port, needle, filler, content_target
    )
    messages = [
        {
            "role": "system",
            "content": "Retrieve the requested A01 line from the supplied local source corpus. No commentary.",
        },
        {"role": "user", "content": content},
    ]
    raw_prompt = probe.apply_template(port, messages)
    full_tokens = probe.tokenize(port, raw_prompt)
    marker = raw_prompt.find("AGENTIC_RETRIEVAL_BEGIN")
    if marker < 0:
        raise RuntimeError("reconstructed prompt has no retrieval boundary")
    pre_marker_tokens = probe.tokenize(port, raw_prompt[:marker])
    base_prefix = probe.common_token_prefix_length(pre_marker_tokens, full_tokens)
    return {
        "needle": needle,
        "filler": filler,
        "corpus": corpus,
        "content": content,
        "sizing": sizing,
        "content_tokens": content_tokens,
        "retrieval_suffix": retrieval_suffix,
        "messages": messages,
        "raw_prompt": raw_prompt,
        "full_tokens": full_tokens,
        "pre_marker_tokens": pre_marker_tokens,
        "base_prefix_tokens": base_prefix,
        "raw_prompt_sha256": sha256_text(raw_prompt),
        "full_prompt_token_ids_sha256": sha256_tokens(full_tokens),
        "prefix_token_ids_sha256": sha256_tokens(full_tokens[:BASE_PREFIX_TOKENS]),
    }


def build_literal_extension(
    probe,
    port: int,
    rebuilt_base: dict,
    source_tokens,
    target_prefix_tokens: int,
    source_used_chars: int | None = None,
) -> dict:
    """Append unused source tokens without retokenizing the sealed boundary.

    The original sealed 16K ancestor ends just before
    ``AGENTIC_RETRIEVAL_BEGIN``; later saved states are exact descendants of
    that token vector. A longer rendering of the same text is not guaranteed to
    retain the final BPE token at an arbitrary cut. Agent state, however, is
    already a token sequence. This helper therefore keeps the selected source
    sequence immutable, adds a clearly delimited token-array block sourced from
    the *unused* continuation of the same local corpus, and reuses the exact
    original retrieval suffix only for the validation request.
    """

    source_tokens = list(source_tokens)
    if len(source_tokens) < BASE_PREFIX_TOKENS:
        raise ValueError("source token vector is shorter than the sealed base")
    if target_prefix_tokens <= len(source_tokens):
        raise ValueError("literal extension target must exceed the source prefix")

    content = rebuilt_base["content"]
    filler = rebuilt_base["filler"]
    head = rebuilt_base["needle"] + "\nBEGIN_CODE_CORPUS\n"
    tail = rebuilt_base["retrieval_suffix"]
    if not content.startswith(head) or not content.endswith(tail):
        raise ValueError("rebuilt 16K source prompt has an unexpected composition")
    base_used_filler = content[len(head) : len(content) - len(tail)]
    if not filler.startswith(base_used_filler):
        raise ValueError("rebuilt 16K prompt is not a prefix of the source corpus")
    if source_used_chars is None:
        source_used_chars = len(base_used_filler)
    if not isinstance(source_used_chars, int) or not (
        len(base_used_filler) <= source_used_chars <= len(filler)
    ):
        raise ValueError("source corpus offset is outside the unused-text range")
    unused_filler = filler[source_used_chars:]
    if not unused_filler:
        raise ValueError("source corpus has no unused text for extension")

    header_text = "\nBEGIN_CODE_CORPUS_EXTENSION\n"
    footer_text = "\nEND_CODE_CORPUS_EXTENSION\n"
    header_tokens = probe.tokenize(port, header_text)
    footer_tokens = probe.tokenize(port, footer_text)
    body_needed = (
        target_prefix_tokens
        - len(source_tokens)
        - len(header_tokens)
        - len(footer_tokens)
    )
    if body_needed <= 0:
        raise ValueError("target prefix is too small for the extension delimiters")

    chars = min(len(unused_filler), max(4096, body_needed * 4))
    sizing = []
    body_tokens = []
    body_text = ""
    for _ in range(8):
        body_text = unused_filler[:chars]
        body_tokens = probe.tokenize(port, body_text)
        sizing.append({"source_chars": chars, "tokens": len(body_tokens)})
        if len(body_tokens) >= body_needed:
            break
        if chars >= len(unused_filler):
            break
        proposed = max(chars + 1024, int(chars * body_needed / max(len(body_tokens), 1) * 1.05))
        chars = min(len(unused_filler), proposed)
    if len(body_tokens) < body_needed:
        raise ValueError(
            f"unused source corpus produced {len(body_tokens)} tokens; need {body_needed}"
        )
    body_tokens_available = len(body_tokens)
    body_tokens = list(body_tokens[:body_needed])
    extension_tokens = list(header_tokens) + body_tokens + list(footer_tokens)
    target_base_tokens = source_tokens + extension_tokens
    if len(target_base_tokens) != target_prefix_tokens:
        raise AssertionError("literal extension did not reach the exact target count")

    base_full_tokens = rebuilt_base["full_tokens"]
    retrieval_suffix_tokens = list(base_full_tokens[BASE_PREFIX_TOKENS:])
    if not retrieval_suffix_tokens:
        raise ValueError("sealed prompt has no retrieval suffix tokens")
    target_full_tokens = target_base_tokens + retrieval_suffix_tokens
    return {
        "mode": "literal-token-append",
        "prompt_construction": "synthetic_literal_slot_continuation",
        "canonical_retokenized_long_prompt": False,
        "canonical_chat_template_boundary": False,
        "filled_context_claim": False,
        "state_semantics": (
            "The saved target/MTP token state is immutable; only new literal "
            "tokens are evaluated. This is a low-level slot continuation, not "
            "a claim that a chat template can retokenize the old message."
        ),
        "source_corpus": rebuilt_base["corpus"],
        "source_filler_chars": len(filler),
        "source_filler_sha256": sha256_text(filler),
        "target_prefix_tokens": target_prefix_tokens,
        "source_prefix_tokens": len(source_tokens),
        "target_base_tokens": target_base_tokens,
        "target_full_tokens": target_full_tokens,
        "retrieval_suffix_tokens": retrieval_suffix_tokens,
        "retrieval_suffix_token_count": len(retrieval_suffix_tokens),
        "retrieval_suffix_token_ids_sha256": sha256_tokens(retrieval_suffix_tokens),
        "extension_tokens": extension_tokens,
        "extension_token_count": len(extension_tokens),
        "extension_token_ids_sha256": sha256_tokens(extension_tokens),
        "header_token_count": len(header_tokens),
        "footer_token_count": len(footer_tokens),
        "body_tokens_needed": body_needed,
        "body_tokens_available": body_tokens_available,
        "body_token_ids_sha256": sha256_tokens(body_tokens),
        "used_source_chars_before_extension": source_used_chars,
        "extension_source_chars_tokenized": len(body_text),
        "extension_source_text_sha256": sha256_text(body_text),
        "unused_source_chars_available": len(unused_filler),
        "sizing": sizing,
    }


def build_extension_prefixes(full_tokens, base_prefix: int, target_tokens: int, chunk: int):
    if target_tokens <= base_prefix:
        raise ValueError(
            f"target prompt ({target_tokens}) must exceed saved prefix ({base_prefix})"
        )
    if target_tokens > CTX_SIZE - 8:
        raise ValueError(f"target prompt must fit below ctx-size {CTX_SIZE}")
    if chunk < 1:
        raise ValueError("chunk size must be positive")
    prefixes = []
    end = min(base_prefix + chunk, target_tokens)
    while True:
        prefixes.append((end, list(full_tokens[:end])))
        if end >= target_tokens:
            break
        end = min(end + chunk, target_tokens)
    return prefixes


def source_prefix_count(result: dict) -> int:
    value = result.get("target_base_prefix_tokens")
    if value is None:
        value = result.get("base_prefix_tokens")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("source result has no valid saved-prefix token count")
    return value


def read_source_result(path: Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source result is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or not result.get("restart_safe_mtp"):
        raise ValueError("source result is not a PASS/restart-safe run")
    if result.get("candidate_pass") is False:
        raise ValueError("source result did not pass its candidate parity gate")
    count = source_prefix_count(result)
    save = result.get("slot_save") or {}
    if save.get("n_saved") != count:
        raise ValueError("source result slot count does not match its saved prefix")
    if result.get("runtime", {}).get("commit") != CANDIDATE_RUNTIME_COMMIT:
        raise ValueError("source result was not produced by candidate runtime")
    result["_source_result_path"] = str(path)
    result["_source_result_sha256"] = sha256_file(path)
    return result


def source_corpus_offset(result: dict) -> int | None:
    literal = result.get("literal_extension")
    if not literal:
        return None
    before = literal.get("used_source_chars_before_extension")
    tokenized = literal.get("extension_source_chars_tokenized")
    if not isinstance(before, int) or not isinstance(tokenized, int):
        raise ValueError("source result lacks literal-extension corpus offsets")
    if before < 0 or tokenized <= 0:
        raise ValueError("source result has invalid literal-extension corpus offsets")
    # Skip the entire tokenized slice, including its unused tail, so resumed
    # extensions cannot accidentally duplicate source text.
    return before + tokenized


def candidate_args(probe, base, context, slot_dir: Path, runtime_selection: dict):
    args = probe.server_args(base, context, slot_dir, runtime_selection)
    index = args.index("--port")
    args[index + 1] = str(PORT)
    if str(runtime_selection["executable"]) != args[0]:
        raise RuntimeError("candidate executable was not installed in server args")
    return args


def wait_ready(session, port: int):
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
        f"server not ready; exit={session.process.poll()} violations={session.violations}"
    )


def copy_immutable_pair(source_check: dict, destination_dir: Path, filename: str):
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_target = Path(source_check["target_path"])
    source_draft = Path(source_check["draft_path"])
    target = destination_dir / filename
    draft = Path(f"{target}.dft")
    if target.exists() or draft.exists():
        raise FileExistsError(f"destination slot pair already exists: {target}")
    # Copying into the fresh run directory lets the HTTP slot API address a
    # basename.  The sealed source pair remains untouched and is hash-checked.
    shutil.copy2(source_target, target)
    shutil.copy2(source_draft, draft)
    record = {
        "source_target": str(source_target),
        "source_draft": str(source_draft),
        "target": str(target),
        "draft": str(draft),
        "target_sha256": sha256_file(target),
        "draft_sha256": sha256_file(draft),
        "target_bytes": target.stat().st_size,
        "draft_bytes": draft.stat().st_size,
        "copy_is_not_physical_append": True,
    }
    if (
        record["target_sha256"] != source_check["target_sha256"]
        or record["draft_sha256"] != source_check["draft_sha256"]
        or record["target_bytes"] != source_check["target_bytes"]
        or record["draft_bytes"] != source_check["draft_bytes"]
    ):
        raise RuntimeError(f"copied slot pair differs from sealed source: {record}")
    record["copy_matches_sealed_source"] = True
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Resume an immutable Q3_PLE+MTP slot with cached literal token chunks "
            "using the mtp-slot-state diagnostic runtime."
        )
    )
    parser.add_argument("--tag", required=True, help="unique run label")
    parser.add_argument(
        "--source-result",
        type=Path,
        default=SOURCE_RESULT,
        help="PASS/restart-safe slot result to resume (default: sealed 16K result)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help=f"exact reusable-prefix token count (default: {DEFAULT_TARGET_TOKENS})",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"literal cache extension chunk (default: {DEFAULT_CHUNK_TOKENS})",
    )
    return parser.parse_args(argv)


def failure_record(
    paths,
    profile,
    error,
    trace,
    offline=None,
    evidence=None,
    checkpoint=None,
):
    record = {
        "schema": 1,
        "status": "FAILED",
        "evidence_class": "REJECTED",
        "publishable": False,
        "run_tag": paths.get("run_tag") if paths else None,
        "runner_pid": os.getpid(),
        "profile": profile,
        "error_type": type(error).__name__ if error else None,
        "error": str(error) if error else None,
        "traceback": trace,
        "offline_slot_check": offline,
        "known_limitation": NON_ATOMIC_LIMITATION,
    }
    if evidence:
        record["partial_evidence"] = evidence
    if checkpoint:
        # Snapshot rather than retain the live dict: the exception path later
        # embeds this result back into the checkpoint.
        record["last_checkpoint"] = json.loads(json.dumps(checkpoint))
    return record


def main(argv=None):
    cli = parse_args(argv)
    if cli.target_tokens <= BASE_PREFIX_TOKENS:
        raise SystemExit("--target-tokens must exceed the sealed 16K base")
    if cli.target_tokens > CTX_SIZE - 8:
        raise SystemExit(f"--target-tokens must be <= {CTX_SIZE - 8}")
    if cli.chunk_tokens < 1:
        raise SystemExit("--chunk-tokens must be positive")

    paths = None
    session = None
    initial_session = None
    restart_session = None
    probe = None
    source_result = None
    offline = None
    live_evidence = {}
    profile = {
        "ctx_size": CTX_SIZE,
        "runtime": "mtp-slot-state",
        "runtime_commit": CANDIDATE_RUNTIME_COMMIT,
        "source_result": str(cli.source_result),
        "saved_prefix_tokens": None,
        "base_content_target_tokens": BASE_CONTENT_TARGET,
        "target_prefix_tokens": cli.target_tokens,
        "chunk_tokens": cli.chunk_tokens,
        "target_n_cpu_moe": 48,
        "threads": 11,
        "threads_batch": 11,
        "batch_size": 2048,
        "ubatch_size": 256,
        "target_kv": "q4_0",
        "draft_kv": "q4_0",
        "draft_n_max": 3,
        "draft_p_min": 0.75,
        "draft_threads": 8,
        "draft_experts": "pinned CUDA_Host",
        "global_cache_ram_mib": 0,
        "working_set_cap_gib": WORKING_SET_CAP_GIB,
        "cache_prompt": True,
        "parallel_slots": 1,
        "port": PORT,
    }
    checkpoint = {}
    try:
        paths = allocate_paths(cli.tag)
        checkpoint = {
            "schema": 1,
            "phase": "initializing",
            "run_tag": paths["run_tag"],
            "runner_pid": os.getpid(),
            "result": str(paths["result"]),
            "checkpoint": str(paths["checkpoint"]),
            "target_slot": str(paths["slots"] / paths["slot_filename"]),
        }
        atomic_json(paths["checkpoint"], checkpoint)

        # These imports are intentionally inside main: import safety is part of
        # the harness contract, and neither reference module's main may run here.
        probe = load_module(PROBE_SCRIPT, "q3ple_agentic_cache_probe_for_extend")
        base = load_module(BASE_SCRIPT, "q3ple_agentic_base_for_extend")
        context = load_module(CONTEXT_SCRIPT, "q3ple_agentic_context_for_extend")
        probe.set_working_set_cap = base.set_working_set_cap

        source_result = read_source_result(cli.source_result)
        sealed_reference = read_source_result(SOURCE_RESULT)
        source_count = source_prefix_count(source_result)
        if cli.target_tokens <= source_count:
            raise ValueError(
                f"target prefix {cli.target_tokens} must exceed source prefix {source_count}"
            )
        source_offset = source_corpus_offset(source_result)
        profile["source_result"] = source_result["_source_result_path"]
        profile["source_result_sha256"] = source_result["_source_result_sha256"]
        profile["saved_prefix_tokens"] = source_count
        profile["source_corpus_offset"] = source_offset
        offline = offline_slot_pair_check(source_result)
        if offline["target_token_ids_sha256"] != offline["draft_prompt_token_ids_sha256"]:
            raise ValueError("offline target/.dft token hashes differ")
        # Extract the source token vector before model load so parsing a large
        # slot file cannot add a transient allocation to the loaded-model peak.
        source_tokens = parse_serialized_prompt(
            Path(offline["target_path"]).read_bytes(), source_count
        )["tokens"]
        checkpoint.update(
            {
                "phase": "offline_slot_checked",
                "offline": offline,
                "source_result": source_result["_source_result_path"],
                "source_result_sha256": source_result["_source_result_sha256"],
                "source_prefix_tokens": source_count,
                "source_corpus_offset": source_offset,
            }
        )
        atomic_json(paths["checkpoint"], checkpoint)

        source_slot_copy = copy_immutable_pair(
            offline, paths["slots"], paths["slot_filename"]
        )
        selection = probe.select_runtime("mtp-slot-state", base)
        runtime = probe.runtime_identity(selection, base, context)
        args = candidate_args(probe, base, context, paths["slots"], selection)
        environment = probe.environment_for_run()
        preflight = probe.process_snapshot()
        if preflight["ram_available"] < PREFLIGHT_RAM_FLOOR:
            raise RuntimeError(f"preflight RAM below 40 GiB: {preflight}")
        if preflight["gpu"]["free_mib"] < 8192:
            raise RuntimeError(f"preflight VRAM below 8192 MiB: {preflight}")
        if preflight["gpu"]["util_pct"] > 15:
            raise RuntimeError(f"preflight GPU utilization above 15%: {preflight}")
        if not port_free(PORT):
            raise RuntimeError(f"port {PORT} is not free")
        checkpoint.update(
            {
                "phase": "preflight_passed",
                "runtime": runtime,
                "args": args,
                "preflight": preflight,
                "environment_overrides": {
                    key: environment[key]
                    for key in (
                        "QWEN38_MTP_UBATCH",
                        "QWEN38_MTP_DRAFT_EXPERT_OFFLOAD",
                        "GGML_CUDA_MOE_CACHE_MB",
                        "QWEN38_WORKING_SET_CAP_GIB",
                    )
                    if key in environment
                },
                "source_slot_copy": source_slot_copy,
            }
        )
        atomic_json(paths["checkpoint"], checkpoint)

        session = probe.launch_server(
            base,
            args,
            environment,
            paths["raw_log"],
            paths["telemetry"],
            preflight,
            runtime_bin=selection["bin"],
        )
        checkpoint.update({"phase": "server_launched", "server_pid": session.process.pid})
        atomic_json(paths["checkpoint"], checkpoint)
        wait_ready(session, PORT)
        checkpoint["phase"] = "server_ready"
        atomic_json(paths["checkpoint"], checkpoint)

        rebuilt_base = source_prompt(
            probe, base, context, PORT, BASE_CONTENT_TARGET
        )
        expected_total = sealed_reference["total_prompt_tokens"]
        expected_raw = sealed_reference["raw_prompt_sha256"]
        expected_full = sealed_reference["full_prompt_token_ids_sha256"]
        expected_prefix = (
            sealed_reference.get("prefix_steps", [])[-1].get("prefix_token_ids_sha256")
            if sealed_reference.get("prefix_steps")
            else None
        )
        base_checks = {
            "base_total_prompt_tokens_equal": len(rebuilt_base["full_tokens"]) == expected_total,
            "base_raw_prompt_sha256_equal": rebuilt_base["raw_prompt_sha256"] == expected_raw,
            "base_full_prompt_token_ids_sha256_equal": rebuilt_base["full_prompt_token_ids_sha256"] == expected_full,
            "base_prefix_token_ids_sha256_equal": rebuilt_base["prefix_token_ids_sha256"] == expected_prefix,
            "base_prefix_count_equal": rebuilt_base["base_prefix_tokens"] == BASE_PREFIX_TOKENS,
            "base_pre_marker_count_equal": len(rebuilt_base["pre_marker_tokens"]) == BASE_PREFIX_TOKENS,
            "base_common_prefix_identity": rebuilt_base["base_prefix_tokens"] == BASE_PREFIX_TOKENS,
            "sealed_base_prefix_hash_equal": (
                sha256_tokens(source_tokens[:BASE_PREFIX_TOKENS])
                == rebuilt_base["prefix_token_ids_sha256"]
            ),
            "source_begins_with_sealed_base": (
                source_tokens[:BASE_PREFIX_TOKENS]
                == rebuilt_base["full_tokens"][:BASE_PREFIX_TOKENS]
            ),
        }
        if not all(base_checks.values()):
            raise ValueError(f"rebuilt 16K prompt failed sealed identity checks: {base_checks}")

        literal_extension = build_literal_extension(
            probe,
            PORT,
            rebuilt_base,
            source_tokens,
            cli.target_tokens,
            source_used_chars=source_offset,
        )
        target_full_tokens = literal_extension["target_full_tokens"]
        target_base_count = literal_extension["target_prefix_tokens"]
        target_base_tokens = literal_extension["target_base_tokens"]
        target_prefix_hash = sha256_tokens(target_base_tokens[:source_count])
        source_literal = source_result.get("literal_extension") or {}
        target_checks = {
            "target_contains_saved_prefix": (
                target_base_tokens[:source_count] == source_tokens
            ),
            "target_saved_prefix_hash_equal": (
                target_prefix_hash == offline["target_token_ids_sha256"]
            ),
            "target_prompt_fits_ctx_with_output": (
                len(target_full_tokens) + 256 <= CTX_SIZE
            ),
            "target_base_is_longer_than_saved_prefix": (
                target_base_count > source_count
            ),
            "target_retrieval_suffix_present": target_base_count < len(target_full_tokens),
            "target_prefix_count_exact": target_base_count == cli.target_tokens,
            "extension_nonempty": literal_extension["extension_token_count"] > 0,
            "extension_uses_unused_source": (
                literal_extension["used_source_chars_before_extension"] > 0
                and literal_extension["extension_source_chars_tokenized"] > 0
            ),
            "retrieval_suffix_matches_sealed_prompt": (
                literal_extension["retrieval_suffix_tokens"]
                == rebuilt_base["full_tokens"][BASE_PREFIX_TOKENS:]
            ),
            "source_retrieval_suffix_count_preserved": (
                source_literal.get("retrieval_suffix_token_count") is None
                or source_literal["retrieval_suffix_token_count"]
                == literal_extension["retrieval_suffix_token_count"]
            ),
            "source_retrieval_suffix_hash_preserved": (
                source_literal.get("retrieval_suffix_token_ids_sha256") is None
                or source_literal["retrieval_suffix_token_ids_sha256"]
                == literal_extension["retrieval_suffix_token_ids_sha256"]
            ),
            "source_filler_hash_preserved": (
                source_literal.get("source_filler_sha256") is None
                or source_literal["source_filler_sha256"]
                == literal_extension["source_filler_sha256"]
            ),
        }
        if not all(target_checks.values()):
            raise ValueError(f"target prompt failed saved-prefix checks: {target_checks}")
        checkpoint.update(
            {
                "phase": "prompt_identity_proven",
                "base_prompt": {
                    key: value
                    for key, value in rebuilt_base.items()
                    if key not in ("needle", "filler", "content", "full_tokens", "pre_marker_tokens")
                },
                "literal_extension": {
                    key: value
                    for key, value in literal_extension.items()
                    if key
                    not in (
                        "target_base_tokens",
                        "target_full_tokens",
                        "retrieval_suffix_tokens",
                        "extension_tokens",
                    )
                },
                "base_checks": base_checks,
                "target_checks": target_checks,
            }
        )
        atomic_json(paths["checkpoint"], checkpoint)

        # Restore the copied immutable pair.  The API receives a basename, and
        # the candidate runtime restores both target and .dft speculative state.
        restore = probe.slot_restore(PORT, paths["slot_filename"], paths["slots"])
        restore["expected_saved_prefix_tokens"] = source_count
        if restore.get("n_restored") != source_count:
            raise RuntimeError(f"restore token count mismatch: {restore}")
        checkpoint.update({"phase": "slot_restored", "restore": restore})
        atomic_json(paths["checkpoint"], checkpoint)

        # The first actual extension request is also the restore proof.  Avoid a
        # zero-suffix request because some server cache paths intentionally
        # re-evaluate one trailing token solely to produce logits.
        steps = []
        previous = source_count
        for end, prefix in build_extension_prefixes(
            target_base_tokens,
            source_count,
            target_base_count,
            cli.chunk_tokens,
        ):
            record = completion_prefix(probe, PORT, prefix, previous)
            record["step"] = len(steps) + 1
            record["appended_tokens"] = end - previous
            if not (
                record["cache_reuse_exact"]
                and record["suffix_processing_exact"]
            ):
                raise RuntimeError(f"extension replayed at {end} tokens: {record}")
            steps.append(record)
            previous = end
            checkpoint.update(
                {
                    "phase": "extending_prefix",
                    "last_prefix_tokens": end,
                    "extension_steps": steps,
                }
            )
            atomic_json(paths["checkpoint"], checkpoint)

        save = probe.slot_save(PORT, paths["slot_filename"], paths["slots"])
        saved_draft = save.get("companion_dft_after_save") or {}
        if not saved_draft.get("exists") or not saved_draft.get("nonempty"):
            raise RuntimeError("extended slot did not produce a nonempty .dft companion")
        final_slot_tokens = save.get("n_saved")
        if final_slot_tokens != target_base_count:
            raise RuntimeError(
                f"extended slot saved {final_slot_tokens} tokens, expected {target_base_count}"
            )

        extended_offline = parse_slot_pair(
            Path(save["path"]), target_base_count, target_base_tokens
        )
        save["sha256"] = extended_offline["target_sha256"]
        save["token_ids_sha256"] = extended_offline["target_token_ids_sha256"]
        save["offline_pair"] = extended_offline
        checkpoint.update(
            {"phase": "extended_slot_saved_and_parsed", "slot_save": save}
        )
        atomic_json(paths["checkpoint"], checkpoint)

        # The persisted state ends immediately before the retrieval marker.  The
        # query itself remains a suffix, so this file is reusable by later turns.
        initial = probe.completion_with_output(
            PORT, target_full_tokens, target_base_count
        )
        initial["expected_a01_text"] = base.EXPECTED.splitlines()[0]
        initial["expected_a01_exact"] = (
            initial["output"] == initial["expected_a01_text"]
        )
        checkpoint.update({"phase": "initial_retrieval_complete", "initial": initial})
        atomic_json(paths["checkpoint"], checkpoint)

        initial_session = session
        initial_resources_before_close = probe.summarize_status(initial_session)
        if initial_resources_before_close["violations"]:
            raise RuntimeError(
                f"initial server safety violation: {initial_resources_before_close}"
            )
        initial_session.close()
        session = None
        initial_resources = probe.summarize_status(initial_session)
        live_evidence["initial_resources"] = initial_resources
        time.sleep(2)
        initial_port_free = port_free(PORT)
        if not initial_port_free:
            raise RuntimeError("owned initial server left port busy after clean stop")

        restart_preflight = probe.process_snapshot()
        if restart_preflight["ram_available"] < HARD_RAM_FLOOR:
            raise RuntimeError(f"restart RAM below 6 GiB: {restart_preflight}")
        if restart_preflight["gpu"]["free_mib"] < 8192:
            raise RuntimeError(f"restart VRAM below 8192 MiB: {restart_preflight}")
        if restart_preflight["gpu"]["util_pct"] > 15:
            raise RuntimeError(f"restart GPU utilization above 15%: {restart_preflight}")
        live_evidence["restart_preflight"] = restart_preflight
        session = probe.launch_server(
            base,
            args,
            environment,
            paths["restart_log"],
            paths["restart_telemetry"],
            restart_preflight,
            runtime_bin=selection["bin"],
        )
        restart_session = session
        checkpoint.update(
            {
                "phase": "restart_server_launched",
                "restart_server_pid": session.process.pid,
                "restart_preflight": restart_preflight,
            }
        )
        atomic_json(paths["checkpoint"], checkpoint)
        wait_ready(session, PORT)

        restore_extended = probe.slot_restore(
            PORT, paths["slot_filename"], paths["slots"]
        )
        if restore_extended.get("n_restored") != target_base_count:
            raise RuntimeError(
                f"extended restore token count mismatch: {restore_extended}"
            )
        restored = probe.completion_with_output(
            PORT, target_full_tokens, target_base_count
        )
        restored["expected_a01_text"] = base.EXPECTED.splitlines()[0]
        restored["expected_a01_exact"] = (
            restored["output"] == restored["expected_a01_text"]
        )
        checkpoint.update(
            {
                "phase": "restart_retrieval_complete",
                "slot_restore_extended": restore_extended,
                "restored": restored,
            }
        )
        atomic_json(paths["checkpoint"], checkpoint)

        restart_resources_before_close = probe.summarize_status(restart_session)
        if restart_resources_before_close["violations"]:
            raise RuntimeError(
                f"restart server safety violation: {restart_resources_before_close}"
            )
        restart_session.close()
        session = None
        restart_resources = probe.summarize_status(restart_session)
        live_evidence["restart_resources"] = restart_resources
        time.sleep(2)
        postflight = probe.process_snapshot()
        postflight_port_free = port_free(PORT)
        if not postflight_port_free:
            raise RuntimeError("owned restart server left port busy after clean stop")
        if postflight["gpu"]["free_mib"] < RELEASE_VRAM_FLOOR_MIB:
            raise RuntimeError(f"postflight VRAM release below 1024 MiB: {postflight}")

        comparisons = {
            "all_extension_cache_reuse_exact": all(
                item["cache_reuse_exact"] and item["suffix_processing_exact"]
                for item in steps
            ),
            "saved_pair_matches_target_base_tokens": (
                extended_offline["expected_tokens_equal"]
                and extended_offline["target_draft_tokens_equal"]
                and extended_offline["target_token_count"] == target_base_count
            ),
            "initial_expected_a01_exact": initial["expected_a01_exact"],
            "restored_expected_a01_exact": restored["expected_a01_exact"],
            "output_sha256_equal": (
                initial["output_sha256"] == restored["output_sha256"]
            ),
            "output_token_ids_sha256_equal": (
                initial["output_token_ids_sha256"]
                == restored["output_token_ids_sha256"]
            ),
            "natural_stop_before_restart": initial["natural_stop"],
            "natural_stop_after_restart": restored["natural_stop"],
            "initial_cache_reuse_exact": initial["cache_reuse_exact"],
            "initial_suffix_processing_exact": initial["suffix_processing_exact"],
            "restored_cache_reuse_exact": restored["cache_reuse_exact"],
            "restored_suffix_processing_exact": restored["suffix_processing_exact"],
            "reused_prefix_tokens_equal": (
                initial["cache_n"] == restored["cache_n"] == target_base_count
            ),
            "processed_suffix_tokens_equal": (
                initial["prompt_n"] == restored["prompt_n"]
            ),
            "mtp_active_before_restart": initial["draft_n"] > 0,
            "mtp_active_after_restart": restored["draft_n"] > 0,
            "mtp_acceptance_valid_before_restart": (
                isinstance(initial["draft_n"], int)
                and isinstance(initial["draft_n_accepted"], int)
                and 0 < initial["draft_n_accepted"] <= initial["draft_n"]
            ),
            "mtp_acceptance_valid_after_restart": (
                isinstance(restored["draft_n"], int)
                and isinstance(restored["draft_n_accepted"], int)
                and 0 < restored["draft_n_accepted"] <= restored["draft_n"]
            ),
            "mtp_draft_n_equal": initial["draft_n"] == restored["draft_n"],
            "mtp_accepted_equal": (
                initial["draft_n_accepted"] == restored["draft_n_accepted"]
            ),
            "save_restore_bytes_match": (
                isinstance(save.get("n_written"), int)
                and restore_extended.get("n_read") == save["n_written"]
            ),
            "save_restore_token_counts_match": (
                save.get("n_saved")
                == restore_extended.get("n_restored")
                == target_base_count
            ),
            "draft_companion_stable_across_restart": (
                (save.get("companion_dft_after_save") or {}).get("sha256")
                == (restore_extended.get("companion_dft_before_restore") or {}).get(
                    "sha256"
                )
                == (restore_extended.get("companion_dft_after_restore") or {}).get(
                    "sha256"
                )
                == extended_offline["draft_sha256"]
            ),
            "initial_safety_clear": not initial_resources["violations"],
            "restart_safety_clear": not restart_resources["violations"],
            "initial_release_headroom": (
                initial_resources["peak"].get("min_vram_free_mib", 0)
                >= RELEASE_VRAM_FLOOR_MIB
            ),
            "restart_release_headroom": (
                restart_resources["peak"].get("min_vram_free_mib", 0)
                >= RELEASE_VRAM_FLOOR_MIB
            ),
            "initial_telemetry_sampled": (
                initial_resources["peak"].get("sample_count", 0) > 0
            ),
            "restart_telemetry_sampled": (
                restart_resources["peak"].get("sample_count", 0) > 0
            ),
            "initial_port_free_after_stop": initial_port_free,
            "postflight_port_free": postflight_port_free,
        }
        candidate_pass = all(comparisons.values())
        if not candidate_pass:
            raise RuntimeError(
                f"{target_base_count}-token extension parity gate failed: {comparisons}"
            )

        safety = {
            "hard_vram_floor_mib": HARD_VRAM_FLOOR_MIB,
            "release_vram_floor_mib": RELEASE_VRAM_FLOOR_MIB,
            "hard_ram_floor_bytes": HARD_RAM_FLOOR,
            "hard_rss_ceiling_bytes": HARD_RSS_CEILING,
            "hard_swap_growth_bytes": HARD_SWAP_GROWTH,
            "working_set_cap_bytes": WORKING_SET_CAP_BYTES,
        }
        result = {
            "schema": 1,
            "status": "PASS",
            "evidence_class": "MEASURED_DIAGNOSTIC",
            "publishable": False,
            "restart_safe_mtp": True,
            "run_tag": paths["run_tag"],
            "runner_pid": os.getpid(),
            "profile": profile,
            "runtime": runtime,
            "runner_script": str(Path(__file__).resolve()),
            "runner_script_sha256": sha256_file(Path(__file__).resolve()),
            "source_result": {
                "path": source_result["_source_result_path"],
                "sha256": source_result["_source_result_sha256"],
                "prefix_tokens": source_count,
                "corpus_offset": source_offset,
            },
            "args": args,
            "initial_preflight": preflight,
            "postflight": postflight,
            "postflight_port_free": postflight_port_free,
            "offline_slot_check": offline,
            "source_slot_copy": source_slot_copy,
            "base_checks": base_checks,
            "target_checks": target_checks,
            "restore": restore,
            "extension_steps": steps,
            "target_base_prefix_tokens": target_base_count,
            "target_base_prefix_token_ids_sha256": sha256_tokens(target_base_tokens),
            "target_full_prompt_tokens": len(target_full_tokens),
            "target_full_prompt_token_ids_sha256": sha256_tokens(target_full_tokens),
            "literal_extension": {
                key: value
                for key, value in literal_extension.items()
                if key
                not in (
                    "target_base_tokens",
                    "target_full_tokens",
                    "retrieval_suffix_tokens",
                    "extension_tokens",
                )
            },
            "slot_save": save,
            "extended_slot_offline_check": extended_offline,
            "initial_retrieval": initial,
            "restart_preflight": restart_preflight,
            "slot_restore_extended": restore_extended,
            "restored_retrieval": restored,
            "comparisons": comparisons,
            "candidate_pass": candidate_pass,
            "all_cache_reuse_exact": comparisons["all_extension_cache_reuse_exact"],
            "saved_slot_tokens": final_slot_tokens,
            "saved_slot_matches_target_base": final_slot_tokens == target_base_count,
            "initial_resources": initial_resources,
            "restart_resources": restart_resources,
            "safety": safety,
            "non_atomic_persistence_limitation": NON_ATOMIC_LIMITATION,
            "raw_log": str(paths["raw_log"]),
            "restart_log": str(paths["restart_log"]),
            "telemetry": str(paths["telemetry"]),
            "restart_telemetry": str(paths["restart_telemetry"]),
            "checkpoint": str(paths["checkpoint"]),
        }
        atomic_json(paths["result"], result)
        checkpoint.update({"phase": "result_written", "result": result})
        atomic_json(paths["checkpoint"], checkpoint)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as error:
        trace = traceback.format_exc()
        if session is not None:
            try:
                if probe is not None:
                    live_evidence["active_session_before_cleanup"] = (
                        probe.summarize_status(session)
                    )
                session.close()
                if probe is not None:
                    live_evidence["active_session_after_cleanup"] = (
                        probe.summarize_status(session)
                    )
            except BaseException:
                pass
        if paths is not None:
            if probe is not None:
                try:
                    live_evidence["failure_postflight"] = probe.process_snapshot()
                    live_evidence["failure_port_free"] = port_free(PORT)
                except BaseException as postflight_error:
                    live_evidence["failure_postflight_error"] = repr(postflight_error)
            result = failure_record(
                paths,
                profile,
                error,
                trace,
                offline,
                evidence=live_evidence,
                checkpoint=checkpoint,
            )
            try:
                atomic_json(paths["result"], result)
                checkpoint.update({"phase": "failed", "result": result})
                atomic_json(paths["checkpoint"], checkpoint)
            except BaseException:
                pass
        print(trace, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
