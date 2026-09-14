"""Small localhost OpenAI-compatible gateway for everyday DSH use.

This process deliberately has no benchmark/session machinery.  It forwards a
single serialized chat request at a time to the one-slot llama-server and
keeps the public contract boring for DSH: one model, one slot, prompt caching,
low reasoning, and bounded output.  Importing the module only defines the
gateway; the server is started by the ``serve`` CLI command.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import contextmanager
import http.client
import hmac
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "profiles/q3ple_dsh_daily_64k_mtp.json"
DEFAULT_UPSTREAM_HOST = "127.0.0.1"
DEFAULT_UPSTREAM_PORT = 18098
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 18092
MODEL_ID = "q3ple-fast-32k"
DEFAULT_MODEL_ID = "q3ple-fast-80k-mtp"
DEFAULT_OUTPUT_TOKENS = 4096
HARD_MAX_OUTPUT_TOKENS = 8192
MAX_OUTPUT_TOKENS = DEFAULT_OUTPUT_TOKENS
DEFAULT_CONTEXT_WINDOW = 81920

REASONING_BUDGETS = {
    "low": 512,
    "medium": 1536,
    "high": 3072,
}
DEFAULT_REASONING_EFFORT = "medium"
MAX_REASONING_BUDGET_TOKENS = 3072
MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 3600
DEFAULT_IDLE_MINUTES = 30.0
LOAD_FAILURE_COOLDOWN_SECONDS = 60.0

DEFAULT_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


class GatewayError(RuntimeError):
    """A static profile or gateway configuration is not usable."""


class RequestError(ValueError):
    """A client request cannot be sent to the upstream model."""


def resolve_profile(path: str | os.PathLike[str] = DEFAULT_PROFILE) -> tuple[Path, dict[str, Any]]:
    """Resolve and load a JSON profile without doing any live work.

    Relative paths are rooted at this checkout.  Environment variables are
    expanded because the daily profile intentionally uses ``%USERPROFILE%``
    for user-owned state paths.
    """

    selected = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not selected.is_absolute():
        selected = ROOT / selected
    try:
        selected = selected.resolve(strict=True)
        with selected.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise GatewayError(f"could not load profile {selected}: {error}") from error
    if not isinstance(profile, dict):
        raise GatewayError(f"profile {selected} must contain a JSON object")
    if profile.get("schema") not in (None, 1):
        raise GatewayError(f"unsupported profile schema in {selected}")
    return selected, profile


def profile_upstream(profile: Mapping[str, Any]) -> tuple[str, int]:
    """Return the model endpoint from a profile, with safe daily defaults."""

    server = profile.get("server")
    if not isinstance(server, Mapping):
        server = {}
    host = str(server.get("host") or DEFAULT_UPSTREAM_HOST)
    try:
        port = int(server.get("port", DEFAULT_UPSTREAM_PORT))
    except (TypeError, ValueError) as error:
        raise GatewayError("profile server.port must be an integer") from error
    if not (1 <= port <= 65535):
        raise GatewayError("profile server.port must be between 1 and 65535")
    if not host:
        raise GatewayError("profile server.host must not be empty")
    return host, port


def profile_model_id(profile: Mapping[str, Any]) -> str:
    """Return the model ID from a profile, with safe daily defaults."""
    if "model_id" in profile:
        return str(profile["model_id"])
    pid = str(profile.get("profile_id", ""))
    if "80k" in pid:
        return DEFAULT_MODEL_ID
    if "64k" in pid:
        return "q3ple-fast-64k-mtp"
    if "32k" in pid:
        return MODEL_ID
    server = profile.get("server")
    if isinstance(server, Mapping):
        base_args = server.get("base_args")
        if isinstance(base_args, list) and "81920" in base_args:
            return DEFAULT_MODEL_ID
    return MODEL_ID


def profile_output_limits(profile: Mapping[str, Any]) -> tuple[int, int]:
    """Return the profile's default and hard completion limits."""

    budget = profile.get("agent_budget")
    if not isinstance(budget, Mapping):
        return DEFAULT_OUTPUT_TOKENS, HARD_MAX_OUTPUT_TOKENS
    hard_value = budget.get("hard_ceiling_response_tokens", budget.get("max_response_tokens", HARD_MAX_OUTPUT_TOKENS))
    try:
        hard = int(hard_value)
        default_value = budget.get("default_response_tokens", min(DEFAULT_OUTPUT_TOKENS, hard))
        default = int(default_value)
    except (TypeError, ValueError) as error:
        raise GatewayError("profile agent_budget response limits must be integers") from error
    if hard <= 0 or hard > HARD_MAX_OUTPUT_TOKENS:
        raise GatewayError(f"profile response ceiling must be between 1 and {HARD_MAX_OUTPUT_TOKENS}")
    if default <= 0 or default > hard:
        raise GatewayError("profile default response tokens must be positive and no greater than its ceiling")
    return default, hard


def profile_context_window(profile: Mapping[str, Any]) -> int:
    """Return the context allocated by the profile's llama-server command."""

    server = profile.get("server")
    if not isinstance(server, Mapping):
        return DEFAULT_CONTEXT_WINDOW
    base_args = server.get("base_args")
    if not isinstance(base_args, list) or "--ctx-size" not in base_args:
        return DEFAULT_CONTEXT_WINDOW
    index = base_args.index("--ctx-size") + 1
    if index >= len(base_args):
        raise GatewayError("profile server.base_args --ctx-size requires a value")
    try:
        context_window = int(base_args[index])
    except (TypeError, ValueError) as error:
        raise GatewayError("profile server.base_args --ctx-size must be an integer") from error
    if context_window < 1:
        raise GatewayError("profile server.base_args --ctx-size must be positive")
    return context_window


def profile_sampling(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Read the profile's pinned sampling policy and fill known safe defaults."""

    reasoning = profile.get("reasoning")
    if not isinstance(reasoning, Mapping):
        reasoning = {}
    sampling = reasoning.get("sampling")
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, Mapping):
        raise GatewayError("profile reasoning.sampling must be an object")
    result = dict(DEFAULT_SAMPLING)
    result.update(dict(sampling))
    return result


def estimate_prompt_tokens(messages: list[Any], profile: Mapping[str, Any]) -> int:
    """Validate message content and estimate text plus native image tokens.

    Base64 image bytes are transport data, not language tokens. The selected
    daily profile pins the projector to a fixed token budget per image, so this
    estimate can reserve the same amount without treating a data URI as text.
    """

    contract = profile.get("client_contract")
    if not isinstance(contract, Mapping):
        contract = {}
    vision = contract.get("vision")
    if not isinstance(vision, Mapping):
        vision = {}
    vision_enabled = vision.get("enabled") is True
    max_images = int(vision.get("max_images", 0)) if vision_enabled else 0
    max_image_bytes = int(vision.get("max_image_bytes", 0)) if vision_enabled else 0
    max_total_image_bytes = int(vision.get("max_total_image_bytes", 0)) if vision_enabled else 0
    tokens_per_image = int(vision.get("tokens_per_image", 0)) if vision_enabled else 0
    accepted_media_types = {str(value).lower() for value in vision.get("accepted_media_types", [])}

    text_chars = 0
    image_count = 0
    total_image_bytes = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            text_chars += len(content)
            continue
        if not isinstance(content, list):
            raise RequestError("message content must be text, an array of content parts, or null")
        for part in content:
            if not isinstance(part, Mapping):
                raise RequestError("each message content part must be an object")
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise RequestError("text content parts must contain a string")
                text_chars += len(text)
                continue
            if part_type != "image_url":
                raise RequestError(f"unsupported message content type: {part_type!r}")
            if not vision_enabled:
                raise RequestError("native image input is not enabled for this profile")
            if message.get("role") != "user":
                raise RequestError("image content is supported only in user messages")
            image_url = part.get("image_url")
            if isinstance(image_url, Mapping):
                url = image_url.get("url")
            else:
                url = image_url
            if not isinstance(url, str) or not url.startswith("data:"):
                raise RequestError("image_url must be an inline base64 data URI")
            header, separator, encoded = url.partition(",")
            if not separator or not header.lower().endswith(";base64"):
                raise RequestError("image_url must use base64 data URI encoding")
            media_type = header[5:-7].lower()
            if media_type not in accepted_media_types:
                raise RequestError(f"unsupported image media type: {media_type!r}")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise RequestError("image_url contains invalid base64 data") from error
            image_count += 1
            total_image_bytes += len(decoded)
            if image_count > max_images:
                raise RequestError(f"at most {max_images} images are supported per request")
            if len(decoded) > max_image_bytes:
                raise RequestError(f"each image must be at most {max_image_bytes} decoded bytes")
            if total_image_bytes > max_total_image_bytes:
                raise RequestError(f"images must total at most {max_total_image_bytes} decoded bytes")

    return text_chars // 3 + image_count * tokens_per_image


def _bounded_integer(value: Any, *, name: str, cap: int, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{name} must be an integer")
    if value < 1:
        raise RequestError(f"{name} must be at least 1")
    return min(value, cap)


def _canonical_json(value: Any) -> Any:
    """Return JSON data with deterministic object-key ordering."""

    if isinstance(value, dict):
        return {key: _canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


def canonicalize_tools(value: Any) -> list[dict[str, Any]]:
    """Make the model-visible tool prefix stable across DSH process starts."""

    if not isinstance(value, list):
        raise RequestError("tools must be an array")
    tools: list[dict[str, Any]] = []
    for tool in value:
        if not isinstance(tool, dict):
            raise RequestError("each tool must be an object")
        tools.append(_canonical_json(tool))

    def sort_key(tool: dict[str, Any]) -> tuple[str, str, str]:
        function = tool.get("function")
        name = function.get("name", "") if isinstance(function, dict) else tool.get("name", "")
        return (
            str(tool.get("type", "")),
            str(name),
            json.dumps(tool, ensure_ascii=False, separators=(",", ":")),
        )

    return sorted(tools, key=sort_key)


def force_request(body: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a DSH request and apply the fixed one-slot daily contract."""

    if not isinstance(body, Mapping):
        raise RequestError("request body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            raise RequestError("each message must be an object with a role")
    n = body.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        raise RequestError("only n=1 is supported")

    try:
        forced = json.loads(json.dumps(body, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise RequestError(f"request contains values that cannot be encoded as JSON: {error}") from error

    forced["model"] = profile_model_id(profile)
    forced["id_slot"] = 0
    forced["cache_prompt"] = True
    forced["parallel"] = 1
    forced["n"] = 1
    if "tools" in forced:
        forced["tools"] = canonicalize_tools(forced["tools"])
    forced.update(profile_sampling(profile))

    reasoning = profile.get("reasoning")
    if not isinstance(reasoning, Mapping):
        reasoning = {}
    default_effort = str(reasoning.get("effort", DEFAULT_REASONING_EFFORT)).lower()
    if default_effort not in REASONING_BUDGETS:
        raise GatewayError(f"profile reasoning effort is unsupported: {default_effort!r}")

    # Validate and forward reasoning effort. The selected profile supplies the
    # default; callers may explicitly request another supported effort.
    effort = body.get("reasoning_effort")
    if effort is None:
        effort = default_effort
    if not isinstance(effort, str) or effort.lower() not in REASONING_BUDGETS:
        raise RequestError(f"unsupported reasoning_effort: {effort!r} (must be low, medium, or high)")
    forced["reasoning_effort"] = effort.lower()

    kwargs = forced.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise RequestError("chat_template_kwargs must be an object")
    kwargs.update({"enable_thinking": True, "preserve_thinking": False})
    forced["chat_template_kwargs"] = kwargs

    default_output_tokens, hard_output_tokens = profile_output_limits(profile)
    forced["max_tokens"] = _bounded_integer(
        forced.get("max_tokens"),
        name="max_tokens",
        cap=hard_output_tokens,
        default=default_output_tokens,
    )
    if "max_completion_tokens" in forced:
        forced["max_completion_tokens"] = _bounded_integer(
            forced.get("max_completion_tokens"),
            name="max_completion_tokens",
            cap=hard_output_tokens,
            default=default_output_tokens,
        )

    # Reserve against the context actually allocated by the selected profile.
    context_window = profile_context_window(profile)
    est_prompt_tokens = estimate_prompt_tokens(messages, profile)
    if est_prompt_tokens + forced["max_tokens"] > context_window:
        raise RequestError(
            f"prompt ({est_prompt_tokens} est tokens) plus completion reserve ({forced['max_tokens']}) "
            f"exceeds context window ({context_window})"
        )

    # Determine reasoning budget based on effort
    budget = REASONING_BUDGETS[forced["reasoning_effort"]]
    budgets_cfg = reasoning.get("budgets")
    if isinstance(budgets_cfg, Mapping) and forced["reasoning_effort"] in budgets_cfg:
        custom_b = budgets_cfg[forced["reasoning_effort"]]
        if isinstance(custom_b, int) and custom_b > 0:
            budget = custom_b
    elif forced["reasoning_effort"] == default_effort:
        custom_b = reasoning.get("budget_tokens", reasoning.get("default_budget_tokens"))
        if isinstance(custom_b, int) and custom_b > 0:
            budget = custom_b

    forced["reasoning_budget_tokens"] = min(budget, MAX_REASONING_BUDGET_TOKENS, forced["max_tokens"])
    return forced


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class ModelLifecycle:
    """Serialize loading, inference, and idle stop for one explicit profile.

    The operation lock is held throughout prefill and streaming. The separate
    state lock keeps status responsive while a cold load or inference runs.
    """

    def __init__(self, profile_path: Path, idle_seconds: float, *, controller=None, clock=time.monotonic):
        if not math.isfinite(idle_seconds) or idle_seconds < 0:
            raise GatewayError("idle timeout must be finite and nonnegative")
        self.profile_path = profile_path.resolve()
        self.idle_seconds = idle_seconds
        self.controller = controller or self._controller
        self.clock = clock
        self.operation_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = "unloaded"
        self.pending_requests = 0
        self.last_activity = clock()
        self.retry_after = 0.0
        self.error: str | None = None
        self.model_pid: int | None = None
        self.accepting_requests = True

    def _controller(self, *arguments: str) -> dict[str, Any]:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/q3ple_daily_profile.py"),
             "--profile", str(self.profile_path), *arguments],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0,
        )
        if completed.returncode:
            raise GatewayError(completed.stderr.strip() or completed.stdout.strip() or "model controller failed")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise GatewayError("model controller returned invalid JSON") from error

    def snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "managed_by": "gateway", "state": self.state,
                "profile": str(self.profile_path), "model_pid": self.model_pid,
                "pending_requests": self.pending_requests,
                "accepting_requests": self.accepting_requests,
                "idle_seconds": round(max(0.0, self.clock() - self.last_activity), 1),
                "idle_timeout_seconds": self.idle_seconds,
                "retry_after_seconds": round(max(0.0, self.retry_after - self.clock()), 1),
                "error": self.error,
            }

    def _set(self, state: str, pid: int | None = None) -> None:
        with self.state_lock:
            self.state, self.model_pid, self.error = state, pid, None

    def failed(self, error: Exception | str) -> None:
        with self.state_lock:
            self.state = "error"
            self.error = str(error)
            self.retry_after = self.clock() + LOAD_FAILURE_COOLDOWN_SECONDS

    @staticmethod
    def _owned(status: dict[str, Any]) -> bool:
        return (status.get("state") or {}).get("status") == "running" and status.get("owned_identity") is True

    def observe(self) -> None:
        """Read existing ownership on startup without loading a model."""
        status = self.controller("status")
        if self._owned(status):
            if status.get("health_error"):
                raise GatewayError(str(status["health_error"]))
            self._set("ready", status["state"]["pid"])
        elif status.get("port_free") is not True:
            raise GatewayError("model port is occupied without a healthy owned model; refusing replacement")
        else:
            self._set("unloaded")

    def health_snapshot(self) -> dict[str, Any]:
        if self.operation_lock.acquire(blocking=False):
            try:
                if self.snapshot()["pending_requests"] == 0:
                    if self.clock() >= self.retry_after:
                        try:
                            self.observe()
                        except Exception as error:
                            self.failed(error)
            finally:
                self.operation_lock.release()
        return self.snapshot()

    def _ensure_ready(self) -> None:
        with self.state_lock:
            if self.clock() < self.retry_after:
                raise GatewayError(f"Model unavailable: {self.error}. Retry after {self.retry_after - self.clock():.0f} seconds.")
        self.observe()
        if self.snapshot()["state"] == "ready":
            return
        self._set("loading")
        result = self.controller("launch", "--mode", "mtp")
        if result.get("status") != "running":
            raise GatewayError("model launch did not reach running state")
        self._set("ready", result.get("pid"))

    @contextmanager
    def request(self):
        with self.state_lock:
            if not self.accepting_requests:
                raise GatewayError("gateway is stopping; retry after startup finishes")
            self.pending_requests += 1
        try:
            with self.operation_lock:
                try:
                    self._ensure_ready()
                except Exception as error:
                    # One failed cold load also rejects queued requests during
                    # this cooldown instead of starting a load for each waiter.
                    if self.clock() >= self.retry_after:
                        self.failed(error)
                    raise GatewayError(str(error)) from error
                with self.state_lock:
                    self.state = "busy"
                try:
                    yield
                finally:
                    with self.state_lock:
                        if self.state == "busy":
                            self.state = "ready"
        finally:
            with self.state_lock:
                self.pending_requests -= 1
                self.last_activity = self.clock()

    def unload_if_idle(self) -> bool:
        if self.idle_seconds <= 0 or not self.operation_lock.acquire(blocking=False):
            return False
        try:
            with self.state_lock:
                if not self.accepting_requests or self.pending_requests or self.clock() - self.last_activity < self.idle_seconds:
                    return False
                if self.clock() < self.retry_after or self.state == "unloaded":
                    return False
            status = self.controller("status")
            if not self._owned(status):
                self.observe()
                return False
            self._set("unloading", status["state"]["pid"])
            self.controller("stop")  # existing exact-identity guard; model only
            self._set("unloaded")
            return True
        except Exception as error:
            self.failed(error)
            return False
        finally:
            self.operation_lock.release()

    def drain(self) -> bool:
        """Atomically stop admissions only when all model operations are idle."""
        if not self.operation_lock.acquire(blocking=False):
            return False
        try:
            with self.state_lock:
                if self.pending_requests:
                    return False
                self.accepting_requests = False
                return True
        finally:
            self.operation_lock.release()


class GatewayServer(ThreadingHTTPServer):
    """Responsive status with serialized access to the one model slot."""

    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        profile: Mapping[str, Any],
        *,
        upstream_host: str | None = None,
        upstream_port: int | None = None,
        upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
        lifecycle: ModelLifecycle | None = None,
        control_token: str = "",
    ) -> None:
        super().__init__(address, GatewayRequestHandler)
        profile_copy = json.loads(json.dumps(profile))
        self.profile: dict[str, Any] = profile_copy
        profile_host, profile_port = profile_upstream(profile_copy)
        self.upstream_host = upstream_host or profile_host
        self.upstream_port = int(upstream_port if upstream_port is not None else profile_port)
        self.upstream_timeout = float(upstream_timeout)
        self.lifecycle = lifecycle
        self.control_token = control_token
        self.inference_lock = threading.Lock()
        self.idle_stop = threading.Event()
        self.idle_thread: threading.Thread | None = None
        if lifecycle is not None and lifecycle.idle_seconds > 0:
            self.idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
            self.idle_thread.start()

    def _idle_loop(self) -> None:
        interval = min(15.0, max(0.1, self.lifecycle.idle_seconds / 4))
        while not self.idle_stop.wait(interval):
            if self.lifecycle.unload_if_idle():
                print(json.dumps({"event": "model-idle-unloaded", **self.lifecycle.snapshot()}), flush=True)

    def server_close(self) -> None:
        self.idle_stop.set()
        super().server_close()
        if self.idle_thread is not None:
            self.idle_thread.join(timeout=5)


class GatewayRequestHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP surface used by the daily DSH preset."""

    server: GatewayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the daily console useful without leaking prompts or credentials.
        return

    def _send_json(self, status: int, value: Any) -> None:
        payload = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _error(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": error_type}})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/v1/models":
            active_id = profile_model_id(self.server.profile)
            entries = [{"id": active_id, "object": "model", "owned_by": "local"}]
            if active_id == "q3ple-fast-64k-mtp":
                # Keep the saved DSH route discoverable across the 80K -> 64K
                # runtime migration without rewriting existing sessions.
                entries.append({"id": DEFAULT_MODEL_ID, "object": "model", "owned_by": "local"})
            if active_id != MODEL_ID:
                entries.append({"id": MODEL_ID, "object": "model", "owned_by": "local"})
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": entries,
                },
            )
            return
        if self.path == "/health":
            self._health()
            return
        self._error(404, "not found", "not_found_error")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/internal/drain":
            expected = self.server.control_token
            if not expected or not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {expected}"):
                self._error(401, "unauthorized", "authentication_error")
            elif self.server.lifecycle is None or not self.server.lifecycle.drain():
                self._error(409, "model is busy; retry when the current request finishes", "model_busy")
            else:
                self._send_json(200, {"status": "drained"})
            return
        if self.path != "/v1/chat/completions":
            self._error(404, "not found", "not_found_error")
            return
        try:
            body = self._read_json_body()
            forced = force_request(body, self.server.profile)
        except RequestError as error:
            self._error(400, str(error))
            return
        except (GatewayError, json.JSONDecodeError) as error:
            self._error(400, str(error))
            return
        if self.server.lifecycle is not None:
            try:
                with self.server.lifecycle.request():
                    self._proxy_chat(forced)
            except GatewayError as error:
                self._error(503, f"Model could not be made ready: {error}", "model_unavailable")
        else:
            with self.server.inference_lock:
                self._proxy_chat(forced)

    def do_PUT(self) -> None:  # noqa: N802
        self._error(405, "method not allowed", "method_not_allowed")

    do_PATCH = do_PUT
    do_DELETE = do_PUT

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise RequestError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise RequestError("Content-Length must be an integer") from error
        if length < 0:
            raise RequestError("Content-Length must not be negative")
        if length > MAX_REQUEST_BYTES:
            raise RequestError("request body is too large")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RequestError("request body ended before Content-Length")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RequestError(f"invalid JSON request: {error}") from error
        if not isinstance(value, dict):
            raise RequestError("request body must be a JSON object")
        return value

    def _health(self) -> None:
        if self.server.lifecycle is not None:
            model = self.server.lifecycle.health_snapshot()
            self._send_json(200, {
                "status": "ok", "available": model["state"] in ("ready", "busy"),
                "upstream": f"{self.server.upstream_host}:{self.server.upstream_port}",
                "model": model,
            })
            return
        connection: http.client.HTTPConnection | None = None
        try:
            connection = http.client.HTTPConnection(
                self.server.upstream_host,
                self.server.upstream_port,
                timeout=min(self.server.upstream_timeout, 5.0),
            )
            connection.request("GET", "/health", headers={"Connection": "close"})
            response = connection.getresponse()
            status = int(response.status)
            response.read()
            available = 200 <= status < 300
            self._send_json(
                200 if available else 503,
                {
                    "status": "ok" if available else "unavailable",
                    "available": available,
                    "upstream_status": status,
                    "upstream": f"{self.server.upstream_host}:{self.server.upstream_port}",
                },
            )
        except (OSError, socket.timeout, http.client.HTTPException) as error:
            self._send_json(
                503,
                {
                    "status": "unavailable",
                    "available": False,
                    "upstream": f"{self.server.upstream_host}:{self.server.upstream_port}",
                    "error": str(error),
                },
            )
        finally:
            if connection is not None:
                connection.close()

    def _proxy_chat(self, forced: dict[str, Any]) -> None:
        payload = _json_bytes(forced)
        connection: http.client.HTTPConnection | None = None
        upstream: http.client.HTTPResponse | None = None
        response_started = False
        try:
            connection = http.client.HTTPConnection(
                self.server.upstream_host,
                self.server.upstream_port,
                timeout=self.server.upstream_timeout,
            )
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(payload)),
                    "Connection": "close",
                },
            )
            upstream = connection.getresponse()
            if upstream.status >= 500 and self.server.lifecycle is not None:
                self.server.lifecycle.failed(f"model returned HTTP {upstream.status}")
            self.send_response(upstream.status)
            copied_content_length = False
            content_type = upstream.getheader("Content-Type", "")
            is_event_stream = content_type.lower().startswith("text/event-stream")
            for name, value in upstream.getheaders():
                lower = name.lower()
                if lower in {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}:
                    continue
                if lower == "content-length":
                    copied_content_length = True
                self.send_header(name, value)
            if not copied_content_length:
                # HTTP/1.1 framing is unambiguous because we close after the
                # upstream body; streamed responses are still forwarded as
                # bytes arrive instead of being buffered in memory.
                self.send_header("Connection", "close")
            else:
                self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            while True:
                # HTTPResponse.read(64 KiB) is efficient for ordinary JSON,
                # but an SSE response can remain below that amount while the
                # model is still generating.  Read one SSE line at a time so
                # the first token/event reaches DSH immediately; readline()
                # also handles the blank line that terminates each event.
                chunk = upstream.readline() if is_event_stream else upstream.read(64 * 1024)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # Closing both objects is the cancellation signal for a
                    # long-running llama-server request.
                    upstream.close()
                    connection.close()
                    self.close_connection = True
                    return
                # OpenAI SSE has an explicit terminal event. Do not wait for a
                # keep-alive upstream socket to reach EOF after completion.
                if is_event_stream and chunk.strip() == b"data: [DONE]":
                    break
            self.close_connection = True
        except (OSError, socket.timeout, http.client.HTTPException) as error:
            if self.server.lifecycle is not None:
                self.server.lifecycle.failed(error)
            if not response_started:
                self._error(502, f"upstream request failed: {error}", "upstream_error")
            else:
                self.close_connection = True
        finally:
            if upstream is not None:
                upstream.close()
            if connection is not None:
                connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve the daily DSH gateway")
    serve.add_argument("--profile", default=str(DEFAULT_PROFILE), help="daily profile JSON")
    serve.add_argument("--listen-host", default=DEFAULT_LISTEN_HOST)
    serve.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    serve.add_argument("--idle-minutes", type=float, default=DEFAULT_IDLE_MINUTES,
                       help="unload the model after idle minutes; 0 keeps it loaded")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "serve":
        return 2
    try:
        profile_path, profile = resolve_profile(args.profile)
        if not math.isfinite(args.idle_minutes) or args.idle_minutes < 0:
            raise GatewayError("--idle-minutes must be finite and nonnegative")
        upstream_host, upstream_port = profile_upstream(profile)
        lifecycle = ModelLifecycle(profile_path, args.idle_minutes * 60)
        try:
            lifecycle.observe()
        except GatewayError as error:
            lifecycle.failed(error)
        server = GatewayServer(
            (args.listen_host, args.listen_port),
            profile,
            upstream_host=upstream_host,
            upstream_port=upstream_port,
            lifecycle=lifecycle,
            control_token=os.environ.get("Q3PLE_GATEWAY_CONTROL_TOKEN", ""),
        )
        print(
            json.dumps(
                {
                    "status": "serving",
                    "profile": str(profile_path),
                    "listen": f"{args.listen_host}:{args.listen_port}",
                    "upstream": f"{upstream_host}:{upstream_port}",
                    "model": profile_model_id(profile),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            server.server_close()
        return 0
    except (GatewayError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
