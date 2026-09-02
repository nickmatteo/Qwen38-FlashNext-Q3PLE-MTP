"""Fail-closed Pi/DSH gateway for the Q3_PLE reasoning profile.

The gateway does not launch or stop llama-server. It exposes the OpenAI Chat
Completions surface used by Pi, forces the one-slot request contract, renders
and tokenizes each request through the pinned server, rejects rewritten prompt
prefixes, and writes append-only machine-readable evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "profiles/q3ple_daily_80k_reasoning.json"
DEFAULT_PI_MODELS = ROOT / "benchmarks/pi/models.json"
DEFAULT_LISTEN_PORT = 18091
MODEL_ID = "q3ple-daily-reasoning"
MAX_CANONICAL_BOUNDARY_REPLAY_TOKENS = 8
BOUNDARY_PROBE_MARKER = "Q3PLE_PI_CANONICAL_BOUNDARY_PROBE_7F2D1B9E"


class AdapterError(RuntimeError):
    """The adapter contract failed and the request must not be retried."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def write_bytes_sync(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdapterError(f"expected JSON object in {path}")
    return value


def resolve_profile(path: str | Path = DEFAULT_PROFILE) -> tuple[Path, dict[str, Any]]:
    selected = Path(path)
    if not selected.is_absolute():
        selected = ROOT / selected
    selected = selected.resolve()
    return selected, load_json(selected)


def profile_contract(profile: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if profile.get("profile_id") != "q3ple_daily_80k_reasoning_v1":
        errors.append("unexpected reasoning profile id")
    policy = profile.get("policy", {})
    if policy.get("default_mode") != "target" or policy.get("allowed_modes") != ["target"]:
        errors.append("reasoning profile must allow only target mode")
    server = profile.get("server", {})
    if server.get("port") != 18090 or server.get("slot_count") != 1 or server.get("slot_id") != 0:
        errors.append("reasoning server must use port 18090 and slot 0/1")
    if server.get("modes", {}).get("mtp", {}).get("enabled") is not False:
        errors.append("MTP must be disabled for the reasoning baseline")
    if server.get("modes", {}).get("target", {}).get("enabled") is not True:
        errors.append("target mode must be enabled")
    reasoning = profile.get("reasoning", {})
    if reasoning.get("enabled") is not True or reasoning.get("effort") != "medium":
        errors.append("medium reasoning must be explicit")
    if reasoning.get("budget_tokens") != 8192 or reasoning.get("preserve") is not False:
        errors.append("reasoning budget/preservation gate is incorrect")
    expected_sampling = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    if reasoning.get("sampling") != expected_sampling:
        errors.append("sampling does not match the pinned Qwen thinking policy")
    contract = profile.get("client_contract", {})
    if contract.get("required_extra_body") != {"id_slot": 0, "cache_prompt": True, "parallel": 1}:
        errors.append("one-slot client contract is incomplete")
    state_dir = str(profile.get("state", {}).get("directory", ""))
    if "q3ple_daily_reasoning_v1" not in state_dir:
        errors.append("reasoning state namespace is not isolated")
    return errors


def validate_pi_models(path: str | Path = DEFAULT_PI_MODELS) -> list[str]:
    errors: list[str] = []
    selected = Path(path)
    if not selected.is_absolute():
        selected = ROOT / selected
    value = load_json(selected.resolve())
    provider = value.get("providers", {}).get("q3ple-local", {})
    if provider.get("baseUrl") != "http://127.0.0.1:18091/v1":
        errors.append("Pi provider must target adapter port 18091")
    if provider.get("api") != "openai-completions":
        errors.append("Pi provider must use OpenAI Chat Completions")
    models = provider.get("models", [])
    if len(models) != 1 or models[0].get("id") != MODEL_ID:
        errors.append("Pi provider must expose only the pinned local model")
    if not models or models[0].get("reasoning") is not True:
        errors.append("Pi model must advertise reasoning")
    compat = provider.get("compat", {})
    if compat.get("thinkingFormat") != "qwen-chat-template":
        errors.append("Pi must use Qwen chat-template thinking controls")
    kwargs = compat.get("chatTemplateKwargs", {})
    if kwargs.get("preserve_thinking") is not False:
        errors.append("preserved thinking must remain disabled until its prefix gate passes")
    return errors


def force_request(body: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise AdapterError("messages must be a non-empty array")
    if body.get("n", 1) != 1:
        raise AdapterError("parallel/multiple completions are not supported")
    forced = json.loads(json.dumps(body))
    forced["model"] = MODEL_ID
    forced["id_slot"] = 0
    forced["cache_prompt"] = True
    forced["parallel"] = 1
    forced["n"] = 1
    reasoning = profile["reasoning"]
    forced.update(reasoning["sampling"])
    forced["reasoning_effort"] = reasoning["effort"]
    kwargs = forced.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise AdapterError("chat_template_kwargs must be an object")
    kwargs.update({"enable_thinking": True, "preserve_thinking": False})
    forced["chat_template_kwargs"] = kwargs
    requested_max = forced.setdefault("max_tokens", 16384)
    if isinstance(requested_max, bool) or not isinstance(requested_max, int):
        raise AdapterError("max_tokens must be an integer")
    if requested_max < 1 or requested_max > 16384:
        raise AdapterError("max_tokens must be between 1 and 16384")
    return forced


def common_prefix(left: Iterable[int], right: Iterable[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def find_timings(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        timings = value.get("timings")
        if isinstance(timings, dict):
            return dict(timings)
        for nested in value.values():
            found = find_timings(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_timings(nested)
            if found:
                return found
    return {}


def extract_delta(value: Mapping[str, Any]) -> tuple[str, str, int, str | None]:
    content = ""
    reasoning = ""
    tool_calls = 0
    finish: str | None = None
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        if not delta and isinstance(choice.get("message"), dict):
            delta = choice["message"]
        content = str(delta.get("content") or "")
        reasoning = str(delta.get("reasoning_content") or delta.get("reasoning") or "")
        calls = delta.get("tool_calls")
        tool_calls = len(calls) if isinstance(calls, list) else 0
        finish = choice.get("finish_reason")
    return content, reasoning, tool_calls, str(finish) if finish is not None else None


def require_terminal(*, stream: bool, saw_done: bool, finish: str | None) -> None:
    if stream and not saw_done:
        raise AdapterError("upstream SSE ended without data: [DONE]")
    if finish not in {"stop", "tool_calls", "eos", "word"}:
        kind = "SSE" if stream else "response"
        raise AdapterError(f"upstream {kind} lacked a normal terminal finish_reason: {finish!r}")


class SessionContract:
    """Durable prompt-prefix state for one adapter/server slot."""

    def __init__(self, path: Path, profile_sha256: str, session_id: str):
        self.path = path
        self.profile_sha256 = profile_sha256
        self.session_id = session_id
        self.turn = 0
        self.last_prompt_tokens: list[int] = []
        self.last_messages: list[dict[str, Any]] = []
        self.pending_expected_cache_n: int | None = None
        self.pending_messages: list[dict[str, Any]] | None = None
        self.failed = False
        if path.is_file():
            value = load_json(path)
            if value.get("profile_sha256") != profile_sha256 or value.get("session_id") != session_id:
                raise AdapterError("adapter session identity does not match the durable state")
            self.turn = int(value.get("turn", 0))
            self.last_prompt_tokens = [int(token) for token in value.get("last_prompt_tokens", [])]
            stored_messages = value.get("last_messages", [])
            if not isinstance(stored_messages, list) or not all(isinstance(item, dict) for item in stored_messages):
                raise AdapterError("adapter session contains invalid message history")
            self.last_messages = json.loads(json.dumps(stored_messages))
            self.failed = bool(value.get("failed", False))

    def prepare(self, prompt_tokens: list[int], messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.failed:
            raise AdapterError("session is failed closed; start a new explicit episode")
        prior = self.last_prompt_tokens
        prefix = common_prefix(prior, prompt_tokens)
        if messages is not None:
            if not all(isinstance(item, dict) for item in messages):
                raise AdapterError("messages must contain only objects")
            if self.last_messages and messages[: len(self.last_messages)] != self.last_messages:
                self.failed = True
                self._save()
                raise AdapterError("request rewrote or removed a prior canonical chat message")
            if prior and prefix < max(0, len(prior) - 16):
                self.failed = True
                self._save()
                raise AdapterError(
                    f"canonical template prefix changed too early (common={prefix}, previous={len(prior)}, current={len(prompt_tokens)})"
                )
            self.pending_messages = json.loads(json.dumps(messages))
        elif prior and prefix != len(prior):
            self.failed = True
            self._save()
            raise AdapterError(
                f"prompt rewrote the cached prefix (common={prefix}, previous={len(prior)}, current={len(prompt_tokens)})"
            )
        if not prior:
            self.pending_expected_cache_n = 0
        elif messages is None:
            self.pending_expected_cache_n = prefix
        else:
            self.pending_expected_cache_n = max(0, prefix - MAX_CANONICAL_BOUNDARY_REPLAY_TOKENS)
        return {
            "turn": self.turn + 1,
            "prompt_tokens": len(prompt_tokens),
            "prompt_token_ids_sha256": sha256_json(prompt_tokens),
            "previous_prompt_tokens": len(prior),
            "expected_min_cache_n": self.pending_expected_cache_n,
            "stable_previous_token_prefix": prefix,
            "max_canonical_boundary_replay_tokens": MAX_CANONICAL_BOUNDARY_REPLAY_TOKENS,
            "exact_previous_message_prefix": messages is None or not self.last_messages or messages[: len(self.last_messages)] == self.last_messages,
        }

    def commit(self, prompt_tokens: list[int], timings: Mapping[str, Any]) -> dict[str, Any]:
        expected_min = self.pending_expected_cache_n if self.pending_expected_cache_n is not None else len(self.last_prompt_tokens)
        stable_prefix = common_prefix(self.last_prompt_tokens, prompt_tokens)
        cache_n = timings.get("cache_n")
        prompt_n = timings.get("prompt_n")
        errors: list[str] = []
        if cache_n is None or prompt_n is None:
            errors.append("response lacks cache_n/prompt_n")
            cache_value = prompt_value = -1
        else:
            if type(cache_n) is not int or type(prompt_n) is not int:
                errors.append("cache_n/prompt_n must be integer counts")
                cache_value = prompt_value = -1
            else:
                cache_value = cache_n
                prompt_value = prompt_n
            if cache_value < 0 or prompt_value < 0:
                errors.append(f"cache_n/prompt_n cannot be negative: {cache_value}/{prompt_value}")
            if cache_value > len(prompt_tokens):
                errors.append(f"cache_n={cache_value} exceeds request length {len(prompt_tokens)}")
            if not self.last_prompt_tokens and cache_value != 0:
                errors.append(f"fresh session reused unexpected cache tokens: {cache_value}")
            if self.last_prompt_tokens and cache_value < expected_min:
                errors.append(f"cache_n={cache_value} is below expected minimum {expected_min}")
            if cache_value + prompt_value != len(prompt_tokens):
                errors.append(
                    f"cache/prompt accounting {cache_value}+{prompt_value} does not equal request {len(prompt_tokens)}"
                )
        if errors:
            self.failed = True
            self._save()
            raise AdapterError("; ".join(errors))
        self.turn += 1
        self.last_prompt_tokens = list(prompt_tokens)
        if self.pending_messages is not None:
            self.last_messages = self.pending_messages
        self.pending_messages = None
        self.pending_expected_cache_n = None
        self._save()
        return {
            "cache_n": cache_value,
            "prompt_n": prompt_value,
            "expected_min_cache_n": expected_min,
            "accounting_exact": cache_value + prompt_value == len(prompt_tokens),
            "cache_reuse_floor_pass": cache_value >= expected_min,
            "boundary_replay_tokens": max(0, stable_prefix - cache_value),
            "boundary_replay_within_limit": max(0, stable_prefix - cache_value) <= MAX_CANONICAL_BOUNDARY_REPLAY_TOKENS,
        }

    def promote_boundary(
        self,
        boundary_tokens: list[int],
        completed_messages: list[dict[str, Any]],
        timings: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.failed:
            raise AdapterError("session is failed closed; boundary promotion is forbidden")
        if completed_messages[: len(self.last_messages)] != self.last_messages:
            raise AdapterError("completed boundary rewrote prior messages")
        stable_prefix = common_prefix(self.last_prompt_tokens, boundary_tokens)
        cache_n = timings.get("cache_n")
        prompt_n = timings.get("prompt_n")
        if type(cache_n) is not int or type(prompt_n) is not int:
            raise AdapterError("boundary alignment lacks integer cache_n/prompt_n")
        expected_min = max(0, stable_prefix - MAX_CANONICAL_BOUNDARY_REPLAY_TOKENS)
        if cache_n < expected_min or cache_n < 0 or prompt_n < 0 or cache_n + prompt_n != len(boundary_tokens):
            raise AdapterError(
                f"boundary alignment accounting failed: cache={cache_n}, prompt={prompt_n}, tokens={len(boundary_tokens)}, expected_min={expected_min}"
            )
        self.last_prompt_tokens = list(boundary_tokens)
        self.last_messages = json.loads(json.dumps(completed_messages))
        self._save()
        return {
            "token_count": len(boundary_tokens),
            "token_ids_sha256": sha256_json(boundary_tokens),
            "cache_n": cache_n,
            "prompt_n": prompt_n,
            "stable_previous_token_prefix": stable_prefix,
            "boundary_replay_tokens": max(0, stable_prefix - cache_n),
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema": "q3ple-pi-session-v1",
            "profile_sha256": self.profile_sha256,
            "session_id": self.session_id,
            "turn": self.turn,
            "failed": self.failed,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_prompt_token_ids_sha256": sha256_json(self.last_prompt_tokens),
            "last_messages": self.last_messages,
            "last_messages_sha256": sha256_json(self.last_messages),
            "updated_utc": utc_now(),
        }
        temporary = self.path.with_suffix(self.path.suffix + f".{uuid.uuid4().hex}.tmp")
        write_bytes_sync(temporary, canonical_json(value) + b"\n")
        os.replace(temporary, self.path)


def post_json(host: str, port: int, path: str, body: Mapping[str, Any], timeout: int = 120) -> Any:
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=canonical_json(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise AdapterError(f"upstream {path} returned HTTP {error.code}: {detail[:1000]}") from error
    value = json.loads(raw.decode("utf-8"))
    return value


def prompt_tokens(host: str, port: int, body: Mapping[str, Any]) -> list[int]:
    template_body: dict[str, Any] = {
        "messages": body["messages"],
        "add_generation_prompt": True,
    }
    if isinstance(body.get("tools"), list):
        template_body["tools"] = body["tools"]
    if isinstance(body.get("chat_template_kwargs"), dict):
        template_body["chat_template_kwargs"] = body["chat_template_kwargs"]
    rendered = post_json(host, port, "/apply-template", template_body)
    if isinstance(rendered, dict):
        text = next((rendered.get(key) for key in ("prompt", "content", "result") if isinstance(rendered.get(key), str)), None)
    else:
        text = rendered if isinstance(rendered, str) else None
    if not text:
        raise AdapterError("/apply-template returned no prompt string")
    tokenized = post_json(host, port, "/tokenize", {"content": text, "add_special": False})
    tokens = tokenized.get("tokens") if isinstance(tokenized, dict) else None
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise AdapterError("/tokenize returned no integer token vector")
    return tokens


def extract_complete_assistant_boundary(rendered: str, marker: str = BOUNDARY_PROBE_MARKER) -> str:
    marker_index = rendered.find(marker)
    if marker_index < 0 or rendered.find(marker, marker_index + 1) >= 0:
        raise AdapterError("canonical boundary probe marker is missing or ambiguous")
    boundary_end = rendered.rfind("<|im_start|>user", 0, marker_index)
    if boundary_end < 0:
        raise AdapterError("Qwen template did not expose the final probe user header")
    boundary = rendered[:boundary_end]
    if not boundary:
        raise AdapterError("canonical assistant boundary is empty")
    return boundary


def completed_boundary_tokens(host: str, port: int, messages: list[dict[str, Any]]) -> list[int]:
    if not messages or messages[-1].get("role") != "assistant":
        raise AdapterError("canonical boundary must end with an assistant message")
    if BOUNDARY_PROBE_MARKER in json.dumps(messages, ensure_ascii=False):
        raise AdapterError("canonical boundary probe marker collides with the conversation")
    rendered = post_json(
        host,
        port,
        "/apply-template",
        {
            "messages": messages + [{"role": "user", "content": BOUNDARY_PROBE_MARKER}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    )
    if isinstance(rendered, dict):
        rendered = next((rendered.get(key) for key in ("prompt", "content", "result") if isinstance(rendered.get(key), str)), None)
    if not isinstance(rendered, str):
        raise AdapterError("/apply-template returned no boundary prompt string")
    boundary = extract_complete_assistant_boundary(rendered)
    tokenized = post_json(host, port, "/tokenize", {"content": boundary, "add_special": False})
    tokens = tokenized.get("tokens") if isinstance(tokenized, dict) else None
    if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
        raise AdapterError("/tokenize returned no canonical boundary token vector")
    return tokens


class EvidenceStore:
    def __init__(self, root: Path):
        self.root = root
        self.rows = root / "turns.jsonl"
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)

    def record(self, row: Mapping[str, Any], raw_request: bytes, raw_response: bytes) -> None:
        turn = int(row["turn"])
        request_path = self.raw / f"turn-{turn:04d}-request.json"
        response_path = self.raw / f"turn-{turn:04d}-response.raw"
        if request_path.exists() or response_path.exists():
            raise AdapterError(f"turn {turn} raw evidence already exists")
        write_bytes_sync(request_path, raw_request)
        write_bytes_sync(response_path, raw_response)
        complete = dict(row)
        complete["artifacts"] = {
            "request": str(request_path.relative_to(self.root)),
            "request_sha256": sha256_bytes(raw_request),
            "response": str(response_path.relative_to(self.root)),
            "response_sha256": sha256_bytes(raw_response),
        }
        with self.rows.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(complete, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_failure(self, row: Mapping[str, Any], raw_request: bytes, raw_response: bytes) -> None:
        failure_id = uuid.uuid4().hex
        request_path = self.raw / f"failure-{failure_id}-request.raw"
        response_path = self.raw / f"failure-{failure_id}-response.raw"
        write_bytes_sync(request_path, raw_request)
        write_bytes_sync(response_path, raw_response)
        complete = dict(row)
        complete["artifacts"] = {
            "request": str(request_path.relative_to(self.root)),
            "request_sha256": sha256_bytes(raw_request),
            "response": str(response_path.relative_to(self.root)),
            "response_sha256": sha256_bytes(raw_response),
        }
        with (self.root / "failures.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(complete, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class AdapterServer(HTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        profile: dict[str, Any],
        upstream_host: str,
        upstream_port: int,
        contract: SessionContract,
        evidence: EvidenceStore,
    ):
        super().__init__(address, AdapterHandler)
        self.profile = profile
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.contract = contract
        self.evidence = evidence
        self.last_completed_messages: list[dict[str, Any]] | None = None


class AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: AdapterServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        raw = canonical_json(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": MODEL_ID, "failed_closed": self.server.contract.failed})
            return
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
            return
        self._json(404, {"error": {"message": "not found", "type": "adapter_error"}})

    def do_POST(self) -> None:
        if self.path == "/q3ple/align-boundary":
            self._align_boundary()
            return
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found", "type": "adapter_error"}})
            return
        response_started = False
        inference_started = False
        raw_request = b""
        response_raw = bytearray()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_request = self.rfile.read(length)
            incoming = json.loads(raw_request.decode("utf-8"))
            if not isinstance(incoming, dict):
                raise AdapterError("request body must be an object")
            forced = force_request(incoming, self.server.profile)
            tokens = prompt_tokens(self.server.upstream_host, self.server.upstream_port, forced)
            prepared = self.server.contract.prepare(tokens, forced["messages"])
            started = time.perf_counter()
            raw_forced = canonical_json(forced)
            connection = http.client.HTTPConnection(
                self.server.upstream_host,
                self.server.upstream_port,
                timeout=int(self.server.profile["safety"]["request_timeout_seconds"]),
            )
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=raw_forced,
                headers={"Content-Type": "application/json", "Content-Length": str(len(raw_forced))},
            )
            inference_started = True
            upstream = connection.getresponse()
            if upstream.status != 200:
                detail = upstream.read()
                connection.close()
                raise AdapterError(f"upstream chat returned HTTP {upstream.status}: {detail[:1000]!r}")
            stream = bool(forced.get("stream", False))
            events: list[dict[str, Any]] = []
            nonstream_raw: bytes | None = None
            first_content: float | None = None
            content_chars = reasoning_chars = tool_calls = 0
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            finish: str | None = None
            saw_done = False
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                response_started = True
                while True:
                    line = upstream.readline()
                    if not line:
                        break
                    response_raw.extend(line)
                    self.wfile.write(line)
                    self.wfile.flush()
                    stripped = line.decode("utf-8", errors="replace").strip()
                    if not stripped.startswith("data:"):
                        continue
                    payload = stripped[5:].strip()
                    if payload == "[DONE]":
                        saw_done = True
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    events.append(event)
                    content, reasoning, calls, stopped = extract_delta(event)
                    if (content or reasoning or calls) and first_content is None:
                        first_content = time.perf_counter()
                    content_chars += len(content)
                    reasoning_chars += len(reasoning)
                    content_parts.append(content)
                    reasoning_parts.append(reasoning)
                    tool_calls += calls
                    finish = stopped or finish
                require_terminal(stream=True, saw_done=saw_done, finish=finish)
                self.close_connection = True
            else:
                raw = upstream.read()
                nonstream_raw = raw
                response_raw.extend(raw)
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise AdapterError("upstream returned a non-object response")
                events.append(value)
                content, reasoning, calls, stopped = extract_delta(value)
                content_chars = len(content)
                reasoning_chars = len(reasoning)
                content_parts.append(content)
                reasoning_parts.append(reasoning)
                tool_calls = calls
                finish = stopped
                require_terminal(stream=False, saw_done=False, finish=finish)
            connection.close()
            timings = find_timings(events)
            accounting = self.server.contract.commit(tokens, timings)
            assistant: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
            if reasoning_parts:
                assistant["reasoning_content"] = "".join(reasoning_parts)
            self.server.last_completed_messages = (
                json.loads(json.dumps(forced["messages"] + [assistant])) if tool_calls == 0 else None
            )
            wall_ms = (time.perf_counter() - started) * 1000.0
            row = {
                "schema": "q3ple-real-agent-turn-v1",
                "status": "VALID",
                "utc": utc_now(),
                "session_id": self.server.contract.session_id,
                **prepared,
                "request_sha256": sha256_bytes(raw_forced),
                "messages_sha256": sha256_json(forced["messages"]),
                "tools_sha256": sha256_json(forced.get("tools", [])),
                "enforced_contract": {"id_slot": 0, "cache_prompt": True, "parallel": 1},
                "sampling": self.server.profile["reasoning"]["sampling"],
                "reasoning_effort": self.server.profile["reasoning"]["effort"],
                "accounting": accounting,
                "ttft_ms": (first_content - started) * 1000.0 if first_content is not None else None,
                "wall_ms": wall_ms,
                "content_chars": content_chars,
                "reasoning_chars": reasoning_chars,
                "tool_call_fragments": tool_calls,
                "finish_reason": finish,
            }
            self.server.evidence.record(row, raw_forced, bytes(response_raw))
            if not stream:
                if nonstream_raw is None:
                    raise AdapterError("non-streaming response bytes are missing")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(nonstream_raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                response_started = True
                self.wfile.write(nonstream_raw)
                self.close_connection = True
        except (AdapterError, json.JSONDecodeError, OSError, ValueError) as error:
            if inference_started:
                self.server.contract.failed = True
                self.server.contract._save()
            try:
                self.server.evidence.record_failure(
                    {
                        "schema": "q3ple-real-agent-failure-v1",
                        "status": "FAILED",
                        "utc": utc_now(),
                        "session_id": self.server.contract.session_id,
                        "turn": self.server.contract.turn + 1,
                        "error": str(error),
                        "response_started": response_started,
                        "inference_started": inference_started,
                    },
                    raw_request,
                    bytes(response_raw),
                )
            except OSError:
                pass
            if not response_started and not self.wfile.closed:
                try:
                    self._json(409, {"error": {"message": str(error), "type": "q3ple_contract_error"}})
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.close_connection = True

    def _align_boundary(self) -> None:
        try:
            messages = self.server.last_completed_messages
            if messages is None:
                raise AdapterError("no tool-free completed assistant turn is available for boundary alignment")
            tokens = completed_boundary_tokens(self.server.upstream_host, self.server.upstream_port, messages)
            response = post_json(
                self.server.upstream_host,
                self.server.upstream_port,
                "/completion",
                {"prompt": tokens, "n_predict": 0, "cache_prompt": True, "id_slot": 0},
            )
            result = self.server.contract.promote_boundary(tokens, messages, find_timings(response))
            record = {"schema": "q3ple-pi-boundary-alignment-v1", "status": "PASS", "utc": utc_now(), **result}
            with (self.server.evidence.root / "boundary-alignments.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._json(200, record)
        except (AdapterError, OSError, ValueError) as error:
            self.server.contract.failed = True
            self.server.contract._save()
            self._json(409, {"error": {"message": str(error), "type": "q3ple_boundary_error"}})


def validate(profile_path: str | Path, pi_models_path: str | Path) -> dict[str, Any]:
    selected, profile = resolve_profile(profile_path)
    errors = profile_contract(profile) + validate_pi_models(pi_models_path)
    if DEFAULT_LISTEN_PORT == int(profile["server"]["port"]):
        errors.append("adapter and upstream ports must differ")
    return {
        "valid": not errors,
        "profile": str(selected),
        "profile_sha256": sha256_bytes(selected.read_bytes()),
        "pi_models": str((Path(pi_models_path) if Path(pi_models_path).is_absolute() else ROOT / Path(pi_models_path)).resolve()),
        "pi_version": "0.83.0",
        "listen": "127.0.0.1:18091",
        "upstream": "127.0.0.1:18090",
        "model": MODEL_ID,
        "allowed_harnesses": ["pi", "deepseek-harness"],
        "errors": errors,
    }


def validate_listen_endpoint(host: str, port: int, profile: Mapping[str, Any]) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise AdapterError("adapter listen host must be loopback-only")
    if port != DEFAULT_LISTEN_PORT:
        raise AdapterError(f"adapter listen port must remain pinned to {DEFAULT_LISTEN_PORT}")
    if port == int(profile["server"]["port"]):
        raise AdapterError("adapter listen port must differ from the upstream model port")


def fake_smoke(turns: int, profile: Mapping[str, Any]) -> dict[str, Any]:
    if turns < 1 or turns > 100:
        raise AdapterError("fake smoke turns must be between 1 and 100")
    previous: list[int] = []
    rows: list[dict[str, Any]] = []
    for turn in range(1, turns + 1):
        request = force_request({"messages": [{"role": "user", "content": f"turn {turn}"}]}, profile)
        tokens = list(previous) + [10_000 + turn, 20_000 + turn]
        prefix = common_prefix(previous, tokens)
        cache_n = len(previous)
        prompt_n = len(tokens) - cache_n
        rows.append({
            "turn": turn,
            "forced": all((request["id_slot"] == 0, request["cache_prompt"] is True, request["parallel"] == 1)),
            "prefix_exact": prefix == len(previous),
            "cache_n": cache_n,
            "prompt_n": prompt_n,
            "accounting_exact": cache_n + prompt_n == len(tokens),
        })
        previous = tokens
    return {
        "schema": "q3ple-pi-fake-smoke-v1",
        "live": False,
        "turns": turns,
        "valid": all(row["forced"] and row["prefix_exact"] and row["accounting_exact"] for row in rows),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE.relative_to(ROOT)))
    parser.add_argument("--pi-models", default=str(DEFAULT_PI_MODELS.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate static profile and Pi provider contracts")
    smoke = sub.add_parser("smoke", help="run a non-live deterministic contract smoke")
    smoke.add_argument("--turns", type=int, default=20)
    smoke.add_argument("--fake-server", action="store_true", required=True)
    serve = sub.add_parser("serve", help="serve the local OpenAI-compatible adapter")
    serve.add_argument("--listen-host", default="127.0.0.1")
    serve.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    serve.add_argument("--session-id", required=True)
    serve.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate(args.profile, args.pi_models)
        if not result["valid"]:
            raise AdapterError("; ".join(result["errors"]))
        _, profile = resolve_profile(args.profile)
        if args.command == "validate":
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "smoke":
            print(json.dumps(fake_smoke(args.turns, profile), indent=2))
            return 0
        validate_listen_endpoint(args.listen_host, args.listen_port, profile)
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        profile_path, _ = resolve_profile(args.profile)
        contract = SessionContract(run_dir / "session.json", sha256_bytes(profile_path.read_bytes()), args.session_id)
        evidence = EvidenceStore(run_dir)
        upstream = profile["server"]
        server = AdapterServer(
            (args.listen_host, args.listen_port),
            profile,
            str(upstream["host"]),
            int(upstream["port"]),
            contract,
            evidence,
        )
        print(json.dumps({**result, "status": "serving", "session_id": args.session_id, "run_dir": str(run_dir)}), flush=True)
        server.serve_forever(poll_interval=0.25)
        return 0
    except (AdapterError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
