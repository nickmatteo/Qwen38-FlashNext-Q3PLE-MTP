from __future__ import annotations

import base64
import http.client
import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "q3ple_dsh_gateway_under_test", ROOT / "scripts/q3ple_dsh_gateway.py"
)
assert SPEC and SPEC.loader
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


PROFILE = {
    "schema": 1,
    "server": {"host": "127.0.0.1", "port": 1},
    "reasoning": {
        "effort": "low",
        "budget_tokens": 512,
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.91,
            "top_k": 11,
            "min_p": 0.03,
            "presence_penalty": 0.2,
            "repetition_penalty": 1.05,
        },
    },
}
VISION_PROFILE = json.loads(
    (ROOT / "profiles/q3ple_dsh_daily_64k_mtp.json").read_text(encoding="utf-8")
)


class UpstreamHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    health_status = 200
    response_content_type = "application/x-test"
    response_body = b'{"id":"ok","choices":[]}'

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            self.send_response(type(self).health_status)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length))
        type(self).requests.append(value)
        self.send_response(201)
        self.send_header("Content-Type", type(self).response_content_type)
        self.send_header("Content-Length", str(len(type(self).response_body)))
        self.end_headers()
        self.wfile.write(type(self).response_body)

    def do_PUT(self):
        self.send_error(405)


@pytest.fixture
def upstream():
    UpstreamHandler.requests = []
    UpstreamHandler.health_status = 200
    server = HTTPServer(("127.0.0.1", 0), UpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def gateway_server(upstream):
    server = gateway.GatewayServer(
        ("127.0.0.1", 0),
        {**PROFILE, "server": {"host": "127.0.0.1", "port": upstream.server_port}},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, method, path, body=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    encoded = None if body is None else json.dumps(body).encode()
    headers = {} if encoded is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    content_type = response.getheader("Content-Type")
    connection.close()
    return status, content_type, raw


def test_force_request_applies_daily_contract_and_caps():
    forced = gateway.force_request(
        {
            "model": "caller-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 1,
            "id_slot": 99,
            "cache_prompt": False,
            "parallel": 8,
            "max_tokens": 99999,
            "reasoning_effort": "high",
            "reasoning_budget_tokens": 9999,
            "chat_template_kwargs": {"preserve_thinking": True, "custom": 1},
            "temperature": 0.01,
        },
        PROFILE,
    )
    assert forced["model"] == gateway.MODEL_ID
    assert forced["id_slot"] == 0
    assert forced["cache_prompt"] is True
    assert forced["parallel"] == forced["n"] == 1
    assert forced["max_tokens"] == gateway.HARD_MAX_OUTPUT_TOKENS
    assert forced["reasoning_budget_tokens"] == gateway.REASONING_BUDGETS["high"]
    assert forced["reasoning_effort"] == "high"
    assert forced["chat_template_kwargs"] == {
        "preserve_thinking": False,
        "enable_thinking": True,
        "custom": 1,
    }
    assert forced["temperature"] == 0.7
    assert forced["top_k"] == 11


def test_force_request_uses_profile_reasoning_default_and_global_output_default():
    forced = gateway.force_request(
        {"messages": [{"role": "user", "content": "hello"}]},
        PROFILE,
    )
    assert forced["reasoning_effort"] == "low"
    assert forced["reasoning_budget_tokens"] == 512
    assert forced["max_tokens"] == gateway.DEFAULT_OUTPUT_TOKENS


def test_on_disk_daily_profile_caps_default_output_to_agent_budget():
    forced = gateway.force_request(
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 99999},
        VISION_PROFILE,
    )
    assert forced["max_tokens"] == 1024
    assert forced["reasoning_effort"] == "low"
    assert forced["reasoning_budget_tokens"] == 512


@pytest.mark.parametrize(
    "budget",
    [
        {"max_response_tokens": 0},
        {"max_response_tokens": gateway.HARD_MAX_OUTPUT_TOKENS + 1},
        {"max_response_tokens": 1024, "default_response_tokens": 2048},
    ],
)
def test_profile_output_limits_reject_invalid_budgets(budget):
    with pytest.raises(gateway.GatewayError):
        gateway.profile_output_limits({"agent_budget": budget})


@pytest.mark.parametrize(
    "effort,expected_budget",
    [
        ("low", 512),
        ("medium", 1536),
        ("high", 3072),
        ("LOW", 512),
        ("Medium", 1536),
        ("HIGH", 3072),
    ],
)
def test_force_request_honors_supported_reasoning_efforts(effort, expected_budget):
    forced = gateway.force_request(
        {"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": effort},
        PROFILE,
    )
    assert forced["reasoning_effort"] == effort.lower()
    assert forced["reasoning_budget_tokens"] == expected_budget


def test_force_request_rejects_unsupported_reasoning_effort():
    with pytest.raises(gateway.RequestError, match="unsupported reasoning_effort"):
        gateway.force_request(
            {"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "extreme"},
            PROFILE,
        )


def test_reasoning_budget_clamped_to_max_tokens():
    forced = gateway.force_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "medium",
            "max_tokens": 256,
        },
        PROFILE,
    )
    assert forced["max_tokens"] == 256
    assert forced["reasoning_budget_tokens"] == 256


def test_profile_model_id_derivation():
    assert gateway.profile_model_id({"profile_id": "q3ple_dsh_daily_80k_mtp"}) == "q3ple-fast-80k-mtp"
    assert gateway.profile_model_id({"profile_id": "q3ple_dsh_daily_64k_mtp"}) == "q3ple-fast-64k-mtp"
    assert gateway.profile_model_id({"profile_id": "q3ple_dsh_daily_32k"}) == "q3ple-fast-32k"
    assert gateway.profile_model_id({"model_id": "custom-id"}) == "custom-id"
    assert gateway.profile_model_id({}) == gateway.MODEL_ID


def test_profile_context_window_reads_ctx_size_from_server_args():
    assert gateway.profile_context_window(
        {"server": {"base_args": ["--port", "1", "--ctx-size", "65536", "--parallel", "1"]}}
    ) == 65536
    assert gateway.profile_context_window({}) == gateway.DEFAULT_CONTEXT_WINDOW


@pytest.mark.parametrize("value", [None, "bad", "0", "-1"])
def test_profile_context_window_rejects_invalid_ctx_size(value):
    args = ["--ctx-size"] if value is None else ["--ctx-size", value]
    with pytest.raises(gateway.GatewayError, match="--ctx-size"):
        gateway.profile_context_window({"server": {"base_args": args}})


@pytest.mark.parametrize(
    "profile_name,expected",
    [
        ("q3ple_dsh_daily_64k_mtp.json", 65536),
    ],
)
def test_on_disk_profile_context_window_matches_server_allocation(profile_name, expected):
    profile = json.loads((ROOT / "profiles" / profile_name).read_text(encoding="utf-8"))
    assert gateway.profile_context_window(profile) == expected


def test_force_request_reserve_uses_selected_profile_context_window():
    messages = [{"role": "user", "content": "A" * (62000 * 3)}]
    profile_64k = {
        **PROFILE,
        "server": {**PROFILE["server"], "base_args": ["--ctx-size", "65536"]},
    }
    profile_80k = {
        **PROFILE,
        "server": {**PROFILE["server"], "base_args": ["--ctx-size", "81920"]},
    }
    with pytest.raises(gateway.RequestError, match=r"context window \(65536\)"):
        gateway.force_request({"messages": messages, "max_tokens": 4096}, profile_64k)
    assert gateway.force_request(
        {"messages": messages, "max_tokens": 4096}, profile_80k
    )["max_tokens"] == 4096


def test_force_request_accepts_inline_native_image_without_counting_base64_as_text():
    image = base64.b64encode(b"small-png-payload").decode("ascii")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect this"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ],
            }
        ],
        "max_tokens": 256,
    }
    forced = gateway.force_request(body, VISION_PROFILE)
    assert forced["messages"] == body["messages"]
    assert gateway.estimate_prompt_tokens(body["messages"], VISION_PROFILE) == len("inspect this") // 3 + 1024


@pytest.mark.parametrize(
    "url,error",
    [
        ("https://example.test/image.png", "inline base64"),
        ("data:image/svg+xml;base64,PHN2Zz4=", "unsupported image media type"),
        ("data:image/png;base64,%%%", "invalid base64"),
    ],
)
def test_force_request_rejects_unsafe_native_image_urls(url, error):
    with pytest.raises(gateway.RequestError, match=error):
        gateway.force_request(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
                ]
            },
            VISION_PROFILE,
        )


def test_force_request_canonicalizes_tool_order_and_schema_keys():
    first = {
        "type": "function",
        "function": {
            "parameters": {
                "type": "object",
                "properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
            },
            "description": "second",
            "name": "z_tool",
        },
    }
    second = {
        "function": {
            "name": "a_tool",
            "description": "first",
            "parameters": {
                "properties": {"b": {"type": "boolean"}, "a": {"type": "string"}},
                "type": "object",
            },
        },
        "type": "function",
    }
    left = gateway.force_request(
        {"messages": [{"role": "user", "content": "hello"}], "tools": [first, second]},
        PROFILE,
    )["tools"]
    right = gateway.force_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                json.loads(json.dumps(second, sort_keys=True)),
                json.loads(json.dumps(first, sort_keys=True)),
            ],
        },
        PROFILE,
    )["tools"]

    assert [tool["function"]["name"] for tool in left] == ["a_tool", "z_tool"]
    assert left == right
    assert json.dumps(left, separators=(",", ":")) == json.dumps(right, separators=(",", ":"))


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": "not-an-array"},
        {"messages": [{"content": "missing role"}]},
        {"messages": [{"role": "user", "content": "x"}], "n": 2},
        {"messages": [{"role": "user", "content": "x"}], "n": True},
    ],
)
def test_invalid_messages_or_n_are_rejected(gateway_server, body):
    status, _, raw = request(gateway_server, "POST", "/v1/chat/completions", body)
    assert status == 400
    assert json.loads(raw)["error"]["type"] == "invalid_request_error"


def test_models_endpoint(gateway_server):
    status, content_type, raw = request(gateway_server, "GET", "/v1/models")
    assert status == 200
    assert content_type == "application/json"
    value = json.loads(raw)
    assert value["object"] == "list"
    assert value["data"][0]["id"] == gateway.MODEL_ID


def test_64k_models_endpoint_preserves_the_existing_dsh_route_alias(gateway_server):
    gateway_server.profile["profile_id"] = "q3ple_dsh_daily_64k_mtp"
    status, _, raw = request(gateway_server, "GET", "/v1/models")
    assert status == 200
    ids = {model["id"] for model in json.loads(raw)["data"]}
    assert "q3ple-fast-64k-mtp" in ids
    assert "q3ple-fast-80k-mtp" in ids


def test_health_reports_available_upstream(gateway_server, upstream):
    status, _, raw = request(gateway_server, "GET", "/health")
    assert status == 200
    value = json.loads(raw)
    assert value["available"] is True
    assert value["upstream_status"] == 200
    UpstreamHandler.health_status = 503
    status, _, raw = request(gateway_server, "GET", "/health")
    assert status == 503
    assert json.loads(raw)["available"] is False


def test_health_unavailable_is_503():
    # Reserve and release a local port without starting an upstream server.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    server = gateway.GatewayServer(
        ("127.0.0.1", 0),
        {**PROFILE, "server": {"host": "127.0.0.1", "port": port}},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, raw = request(server, "GET", "/health")
        assert status == 503
        assert json.loads(raw)["available"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_chat_response_status_and_content_type_are_proxied(gateway_server, upstream):
    status, content_type, raw = request(
        gateway_server,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello"}], "stream": False},
    )
    assert status == 201
    assert content_type == UpstreamHandler.response_content_type
    assert raw == UpstreamHandler.response_body
    assert len(UpstreamHandler.requests) == 1
    assert UpstreamHandler.requests[0]["id_slot"] == 0
    assert UpstreamHandler.requests[0]["cache_prompt"] is True


class StreamingUpstreamHandler(BaseHTTPRequestHandler):
    first_event_sent = threading.Event()
    release = threading.Event()

    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(b"data: first-token\n\n")
        self.wfile.flush()
        type(self).first_event_sent.set()
        # Keep the upstream open until the test has observed the first event.
        type(self).release.wait(timeout=5)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def test_event_stream_forwards_first_event_before_upstream_finishes():
    StreamingUpstreamHandler.first_event_sent = threading.Event()
    StreamingUpstreamHandler.release = threading.Event()
    upstream = HTTPServer(("127.0.0.1", 0), StreamingUpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    gateway_server = gateway.GatewayServer(
        ("127.0.0.1", 0),
        {**PROFILE, "server": {"host": "127.0.0.1", "port": upstream.server_port}},
    )
    gateway_thread = threading.Thread(target=gateway_server.serve_forever, daemon=True)
    gateway_thread.start()
    client = http.client.HTTPConnection(*gateway_server.server_address, timeout=2)
    try:
        client.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(
                {"messages": [{"role": "user", "content": "stream"}], "stream": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        assert response.status == 200
        first = response.readline()
        assert first == b"data: first-token\n"
        assert StreamingUpstreamHandler.first_event_sent.is_set()
        # The upstream is intentionally waiting, so this proves the gateway
        # did not buffer until EOF before exposing the first SSE event.
        assert not StreamingUpstreamHandler.release.is_set()
    finally:
        StreamingUpstreamHandler.release.set()
        client.close()
        gateway_server.shutdown()
        gateway_server.server_close()
        gateway_thread.join(timeout=2)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)


def test_import_has_no_server_side_effects():
    assert gateway.DEFAULT_LISTEN_PORT == 18092
    assert gateway.DEFAULT_UPSTREAM_PORT == 18098
    assert gateway.MODEL_ID == "q3ple-fast-32k"
    assert gateway.DEFAULT_MODEL_ID == "q3ple-fast-80k-mtp"
