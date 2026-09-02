import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading
import tempfile
import urllib.request
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_adapter():
    path = ROOT / "scripts" / "q3ple_pi_adapter.py"
    spec = importlib.util.spec_from_file_location("q3ple_pi_adapter_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PiAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()
        _, cls.profile = cls.adapter.resolve_profile()

    def test_static_contract_is_valid(self):
        result = self.adapter.validate(
            ROOT / "profiles" / "q3ple_daily_80k_reasoning.json",
            ROOT / "benchmarks" / "pi" / "models.json",
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["allowed_harnesses"], ["pi", "deepseek-harness"])

    def _candidate(self):
        _, profile = self.adapter.resolve_profile(
            ROOT / "profiles" / "q3ple_daily_80k_reasoning_n47.json"
        )
        return profile

    def test_declared_placement_candidate_serves_its_own_upstream(self):
        result = self.adapter.validate(
            ROOT / "profiles" / "q3ple_daily_80k_reasoning_n47.json",
            ROOT / "benchmarks" / "pi" / "models.json",
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["upstream"], "127.0.0.1:18092")

    def test_undeclared_profile_may_not_move_off_the_baseline_upstream(self):
        profile = json.loads(json.dumps(self.profile))
        profile["server"]["port"] = 18092
        self.assertNotEqual(self.adapter.profile_contract(profile), [])

    def test_placement_candidate_may_not_borrow_baseline_identity(self):
        for key in ("profile_id", "candidate_of", "port", "state"):
            with self.subTest(shared=key):
                profile = json.loads(json.dumps(self._candidate()))
                if key == "profile_id":
                    profile["profile_id"] = self.adapter.BASELINE_PROFILE_ID
                elif key == "candidate_of":
                    profile["candidate_of"] = "something-else"
                elif key == "port":
                    profile["server"]["port"] = self.adapter.BASELINE_UPSTREAM_PORT
                else:
                    profile["state"]["directory"] = (
                        "results/QWEN38-MTP-PROTOTYPE-001/state/q3ple_daily_reasoning_v1"
                    )
                self.assertNotEqual(self.adapter.profile_contract(profile), [])

    def test_placement_candidate_still_obeys_the_reasoning_contract(self):
        profile = json.loads(json.dumps(self._candidate()))
        profile["reasoning"]["effort"] = "high"
        self.assertNotEqual(self.adapter.profile_contract(profile), [])

    def test_request_contract_and_sampling_are_forced(self):
        forced = self.adapter.force_request(
            {
                "model": "wrong",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
                "stream": True,
            },
            self.profile,
        )
        self.assertEqual(forced["model"], self.adapter.MODEL_ID)
        self.assertEqual(forced["id_slot"], 0)
        self.assertIs(forced["cache_prompt"], True)
        self.assertEqual(forced["parallel"], 1)
        self.assertEqual(forced["temperature"], 1.0)
        self.assertEqual(forced["top_p"], 0.95)
        self.assertEqual(forced["top_k"], 20)
        self.assertEqual(forced["reasoning_effort"], "medium")
        self.assertEqual(
            forced["chat_template_kwargs"],
            {"enable_thinking": True, "preserve_thinking": False},
        )

    def test_multiple_completions_are_rejected(self):
        with self.assertRaises(self.adapter.AdapterError):
            self.adapter.force_request(
                {"messages": [{"role": "user", "content": "hello"}], "n": 2},
                self.profile,
            )

    def test_generation_limit_is_bounded(self):
        for value in (0, 16385, "100"):
            with self.subTest(value=value), self.assertRaises(self.adapter.AdapterError):
                self.adapter.force_request(
                    {"messages": [{"role": "user", "content": "hello"}], "max_tokens": value},
                    self.profile,
                )

    def test_listen_endpoint_is_loopback_and_pinned(self):
        self.adapter.validate_listen_endpoint("127.0.0.1", self.adapter.DEFAULT_LISTEN_PORT, self.profile)
        for host, port in (("0.0.0.0", 18091), ("127.0.0.1", 18090), ("127.0.0.1", 19000)):
            with self.subTest(host=host, port=port), self.assertRaises(self.adapter.AdapterError):
                self.adapter.validate_listen_endpoint(host, port, self.profile)

    def test_prompt_rewrite_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self.adapter.SessionContract(Path(temporary) / "session.json", "PROFILE", "SESSION")
            first = [1, 2, 3]
            state.prepare(first)
            state.commit(first, {"cache_n": 0, "prompt_n": 3})
            with self.assertRaises(self.adapter.AdapterError):
                state.prepare([1, 9, 3, 4])
            self.assertTrue(state.failed)

    def test_append_only_accounting_is_durable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.json"
            state = self.adapter.SessionContract(path, "PROFILE", "SESSION")
            first = [1, 2, 3]
            state.prepare(first)
            state.commit(first, {"cache_n": 0, "prompt_n": 3})
            second = [1, 2, 3, 4, 5]
            prepared = state.prepare(second)
            self.assertEqual(prepared["expected_min_cache_n"], 3)
            accounting = state.commit(second, {"cache_n": 3, "prompt_n": 2})
            self.assertEqual(accounting["expected_min_cache_n"], 3)
            self.assertTrue(accounting["boundary_replay_within_limit"])
            self.assertTrue(accounting["cache_reuse_floor_pass"])
            restored = self.adapter.SessionContract(path, "PROFILE", "SESSION")
            self.assertEqual(restored.turn, 2)
            self.assertEqual(restored.last_prompt_tokens, second)
            self.assertFalse(restored.failed)

    def test_canonical_assistant_marker_transition_preserves_message_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.json"
            state = self.adapter.SessionContract(path, "PROFILE", "SESSION")
            first_messages = [{"role": "user", "content": "one"}]
            first = [1, 2, 3, 90, 91]
            state.prepare(first, first_messages)
            state.commit(first, {"cache_n": 0, "prompt_n": 5})
            second_messages = first_messages + [
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": "two"},
            ]
            second = [1, 2, 3, 4, 5, 6]
            prepared = state.prepare(second, second_messages)
            self.assertEqual(prepared["stable_previous_token_prefix"], 3)
            self.assertTrue(prepared["exact_previous_message_prefix"])
            accounting = state.commit(second, {"cache_n": 3, "prompt_n": 3})
            self.assertEqual(accounting["expected_min_cache_n"], 0)
            self.assertTrue(accounting["boundary_replay_within_limit"])
            restored = self.adapter.SessionContract(path, "PROFILE", "SESSION")
            self.assertEqual(restored.last_messages, second_messages)

    def test_canonical_message_rewrite_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self.adapter.SessionContract(Path(temporary) / "session.json", "PROFILE", "SESSION")
            state.prepare([1, 2, 3], [{"role": "user", "content": "original"}])
            state.commit([1, 2, 3], {"cache_n": 0, "prompt_n": 3})
            with self.assertRaises(self.adapter.AdapterError):
                state.prepare([1, 2, 3, 4], [{"role": "user", "content": "changed"}])
            self.assertTrue(state.failed)

    def test_complete_assistant_boundary_probe_cut(self):
        rendered = (
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\nanswer<|im_end|>\n"
            f"<|im_start|>user\n{self.adapter.BOUNDARY_PROBE_MARKER}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        boundary = self.adapter.extract_complete_assistant_boundary(rendered)
        self.assertTrue(boundary.endswith("<|im_start|>assistant\nanswer<|im_end|>\n"))
        self.assertNotIn(self.adapter.BOUNDARY_PROBE_MARKER, boundary)

    def test_boundary_promotion_is_exact_and_does_not_increment_turn(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self.adapter.SessionContract(Path(temporary) / "session.json", "PROFILE", "SESSION")
            request_messages = [{"role": "user", "content": "hello"}]
            state.prepare(list(range(20)), request_messages)
            state.commit(list(range(20)), {"cache_n": 0, "prompt_n": 20})
            completed = request_messages + [{"role": "assistant", "content": "answer"}]
            boundary = list(range(18)) + [30, 31, 32]
            result = state.promote_boundary(boundary, completed, {"cache_n": 16, "prompt_n": 5})
            self.assertEqual(result["boundary_replay_tokens"], 2)
            self.assertEqual(state.turn, 1)
            self.assertEqual(state.last_prompt_tokens, boundary)
            self.assertEqual(state.last_messages, completed)

    def test_missing_accounting_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self.adapter.SessionContract(Path(temporary) / "session.json", "PROFILE", "SESSION")
            with self.assertRaises(self.adapter.AdapterError):
                state.commit([1, 2], {})
            self.assertTrue(state.failed)

    def test_impossible_accounting_fails_closed(self):
        for timings in (
            {"cache_n": 0, "prompt_n": -1},
            {"cache_n": 4, "prompt_n": -1},
            {"cache_n": 3, "prompt_n": 1},
            {"cache_n": True, "prompt_n": 2},
        ):
            with self.subTest(timings=timings), tempfile.TemporaryDirectory() as temporary:
                state = self.adapter.SessionContract(Path(temporary) / "session.json", "PROFILE", "SESSION")
                with self.assertRaises(self.adapter.AdapterError):
                    state.commit([1, 2, 3], timings)
                self.assertTrue(state.failed)

    def test_stream_terminal_contract_rejects_truncation(self):
        with self.assertRaises(self.adapter.AdapterError):
            self.adapter.require_terminal(stream=True, saw_done=False, finish="stop")
        with self.assertRaises(self.adapter.AdapterError):
            self.adapter.require_terminal(stream=True, saw_done=True, finish=None)
        self.adapter.require_terminal(stream=True, saw_done=True, finish="tool_calls")

    def test_nested_timings_are_found(self):
        value = {"choices": [{"delta": {"content": "x"}}], "meta": {"timings": {"cache_n": 12, "prompt_n": 3}}}
        self.assertEqual(self.adapter.find_timings(value), {"cache_n": 12, "prompt_n": 3})

    def test_fake_twenty_turn_smoke(self):
        result = self.adapter.fake_smoke(20, self.profile)
        self.assertTrue(result["valid"])
        self.assertEqual(result["turns"], 20)
        self.assertEqual(len(result["rows"]), 20)
        self.assertTrue(all(row["forced"] for row in result["rows"]))

    def test_pi_models_file_is_valid_json(self):
        value = json.loads((ROOT / "benchmarks" / "pi" / "models.json").read_text(encoding="utf-8"))
        self.assertIn("q3ple-local", value["providers"])

    def test_non_streaming_http_proxy_records_valid_evidence(self):
        adapter = self.adapter
        prompt = "rendered prompt"
        tokens = [ord(char) for char in prompt]

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/apply-template":
                    value = {"prompt": prompt}
                elif self.path == "/tokenize":
                    value = {"tokens": tokens}
                elif self.path == "/v1/chat/completions":
                    self.assert_contract(body)
                    value = {
                        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "timings": {"cache_n": 0, "prompt_n": len(tokens)},
                    }
                else:
                    self.send_error(404)
                    return
                raw = json.dumps(value).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            @staticmethod
            def assert_contract(body):
                if body.get("id_slot") != 0 or body.get("cache_prompt") is not True or body.get("parallel") != 1:
                    raise AssertionError("adapter did not force the one-slot contract")

        upstream = HTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                contract = adapter.SessionContract(root / "session.json", "PROFILE", "SESSION")
                evidence = adapter.EvidenceStore(root)
                gateway = adapter.AdapterServer(
                    ("127.0.0.1", 0),
                    self.profile,
                    "127.0.0.1",
                    upstream.server_address[1],
                    contract,
                    evidence,
                )
                gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
                gateway_thread.start()
                try:
                    raw = json.dumps(
                        {"model": "ignored", "messages": [{"role": "user", "content": "hello"}], "stream": False}
                    ).encode("utf-8")
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{gateway.server_address[1]}/v1/chat/completions",
                        data=raw,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        value = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(value["choices"][0]["message"]["content"], "ok")
                    self.assertEqual(contract.turn, 1)
                    rows = [json.loads(line) for line in (root / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["accounting"]["cache_n"], 0)
                    self.assertTrue(rows[0]["accounting"]["accounting_exact"])
                    self.assertEqual(rows[0]["content_chars"], 2)
                    self.assertEqual(rows[0]["reasoning_chars"], 0)
                    self.assertEqual(rows[0]["finish_reason"], "stop")
                    self.assertTrue((root / rows[0]["artifacts"]["request"]).is_file())
                    self.assertTrue((root / rows[0]["artifacts"]["response"]).is_file())
                finally:
                    gateway.shutdown()
                    gateway.server_close()
                    gateway_thread.join(timeout=5)
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

    def test_streaming_http_proxy_preserves_sse_and_terminal_accounting(self):
        adapter = self.adapter
        prompt = "stream prompt"
        tokens = [ord(char) for char in prompt]

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                if self.path == "/apply-template":
                    raw = json.dumps({"prompt": prompt}).encode("utf-8")
                    content_type = "application/json"
                elif self.path == "/tokenize":
                    raw = json.dumps({"tokens": tokens}).encode("utf-8")
                    content_type = "application/json"
                elif self.path == "/v1/chat/completions":
                    events = [
                        {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]},
                        {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]},
                        {"timings": {"cache_n": 0, "prompt_n": len(tokens)}, "choices": []},
                    ]
                    raw = b"".join(f"data: {json.dumps(event)}\n\n".encode("utf-8") for event in events) + b"data: [DONE]\n\n"
                    content_type = "text/event-stream"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        upstream = HTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                contract = adapter.SessionContract(root / "session.json", "PROFILE", "STREAM")
                gateway = adapter.AdapterServer(
                    ("127.0.0.1", 0),
                    self.profile,
                    "127.0.0.1",
                    upstream.server_address[1],
                    contract,
                    adapter.EvidenceStore(root),
                )
                gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
                gateway_thread.start()
                try:
                    raw = json.dumps(
                        {"model": "ignored", "messages": [{"role": "user", "content": "hello"}], "stream": True}
                    ).encode("utf-8")
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{gateway.server_address[1]}/v1/chat/completions",
                        data=raw,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        streamed = response.read()
                    self.assertIn(b"reasoning_content", streamed)
                    self.assertIn(b"data: [DONE]", streamed)
                    self.assertEqual(contract.turn, 1)
                    row = json.loads((root / "turns.jsonl").read_text(encoding="utf-8").splitlines()[0])
                    self.assertEqual(row["reasoning_chars"], 5)
                    self.assertEqual(row["content_chars"], 4)
                    self.assertEqual(row["finish_reason"], "stop")
                    self.assertTrue(row["accounting"]["accounting_exact"])
                finally:
                    gateway.shutdown()
                    gateway.server_close()
                    gateway_thread.join(timeout=5)
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
