from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "q3ple_daily_profile", ROOT / "scripts" / "q3ple_daily_profile.py"
)
assert SPEC and SPEC.loader
daily = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(daily)


class FakeParser:
    """Tiny parser that preserves the store's real copy/hash/pointer behavior."""

    @staticmethod
    def _tokens(path: Path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def sha256_file(path):
        import hashlib

        return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

    @staticmethod
    def sha256_tokens(tokens):
        return daily.sha256_tokens(tokens)

    @classmethod
    def parse_serialized_prompt(cls, data, expected_count=None, **_kwargs):
        tokens = json.loads(data.decode("utf-8"))
        if expected_count is not None and len(tokens) != expected_count:
            raise ValueError("saved token count mismatch")
        return {"tokens": tokens, "count": len(tokens), "token_ids_sha256": daily.sha256_tokens(tokens)}

    @classmethod
    def parse_slot_pair(cls, target, expected_count=None, expected_tokens=None):
        tokens = cls._tokens(target)
        if expected_count is not None and len(tokens) != expected_count:
            raise ValueError("saved token count mismatch")
        if expected_tokens is not None and list(expected_tokens) != tokens:
            raise ValueError("saved vector mismatch")
        draft = Path(f"{target}.dft")
        if not draft.is_file():
            raise FileNotFoundError("missing draft")
        if cls._tokens(draft) != tokens:
            raise ValueError("pair mismatch")
        return {
            "target_path": str(target),
            "target_bytes": Path(target).stat().st_size,
            "target_sha256": cls.sha256_file(target),
            "draft_path": str(draft),
            "draft_bytes": draft.stat().st_size,
            "draft_sha256": cls.sha256_file(draft),
            "target_token_count": len(tokens),
            "target_token_ids_sha256": daily.sha256_tokens(tokens),
            "target_draft_tokens_equal": True,
        }


class FakeBackend:
    def __init__(self, root: Path, output: str = "ok"):
        self.root = root
        self.output = output
        self.partial = False
        self.counter = 0

    def render(self, messages):
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)

    def tokenize(self, prompt):
        return [ord(char) for char in prompt]

    def generate(self, prompt_tokens, unseen, *, mode, n_predict, seed):
        if self.partial:
            return {"partial": True, "output": "partial"}
        return {"complete": True, "output": self.output}

    def save(self, *, mode):
        self.counter += 1
        tokens = self.pending_tokens
        target = self.root / f"source-{self.counter}.slot.bin"
        target.write_text(json.dumps(tokens), encoding="utf-8")
        if mode == "mtp":
            draft = self.root / f"source-{self.counter}.slot.bin.dft"
            draft.write_text(json.dumps(tokens), encoding="utf-8")
        else:
            draft = None
        return {"target": target, "draft": draft}


class DailyProfileTests(unittest.TestCase):
    def test_gpu_watchdog_query_is_headless_on_windows(self):
        with mock.patch.object(daily.subprocess, "check_output", return_value="100, 2000, 3") as check:
            self.assertEqual(daily._gpu_snapshot(), {"used_mib": 100, "free_mib": 2000, "util_pct": 3})
        expected = int(getattr(daily.subprocess, "CREATE_NO_WINDOW", 0)) if daily.os.name == "nt" else 0
        self.assertEqual(check.call_args.kwargs["creationflags"], expected)

    def _reasoning_profile(self):
        return daily.load_profile(daily.ROOT / "profiles/q3ple_daily_80k_reasoning.json")

    def test_undeclared_placement_change_is_rejected(self):
        profile = self._reasoning_profile()
        args = profile["server"]["base_args"]
        args[args.index("--n-cpu-moe") + 1] = "47"
        errors = daily.validate_profile(profile)
        self.assertTrue(any("--n-cpu-moe" in error for error in errors), errors)

    def test_declared_placement_candidate_is_accepted(self):
        candidate = daily.load_profile(daily.ROOT / "profiles/q3ple_daily_80k_reasoning_n47.json")
        args = candidate["server"]["base_args"]
        self.assertEqual(args[args.index("--n-cpu-moe") + 1], "47")
        self.assertEqual(daily.validate_profile(candidate), [])

    def test_placement_candidate_may_not_share_baseline_identity(self):
        for key, value in (
            ("profile_id", "q3ple_daily_80k_reasoning_v1"),
            ("port", 18090),
            ("state_directory", "results/QWEN38-MTP-PROTOTYPE-001/state/q3ple_daily_reasoning_v1"),
        ):
            with self.subTest(shared=key):
                candidate = daily.load_profile(daily.ROOT / "profiles/q3ple_daily_80k_reasoning_n47.json")
                if key == "profile_id":
                    candidate["profile_id"] = value
                elif key == "port":
                    candidate["server"]["port"] = value
                else:
                    candidate["state"]["directory"] = value
                self.assertNotEqual(daily.validate_profile(candidate), [])

    def test_profile_and_dry_run_are_pinned(self):
        profile = daily.load_profile()
        self.assertEqual(daily.validate_profile(profile), [])
        target = daily.preview("target", Path("tmp/test-daily"))
        mtp = daily.preview("mtp", Path("tmp/test-daily"))
        self.assertTrue(target["valid"])
        self.assertIn("--spec-type", target["command"])
        self.assertEqual(target["command"][target["command"].index("--spec-type") + 1], "none")
        self.assertEqual(mtp["command"][mtp["command"].index("--spec-type") + 1], "draft-mtp")

    def test_pair_generations_preserve_previous_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_target = root / "source.slot.bin"
            source_draft = root / "source.slot.bin.dft"
            source_target.write_text(json.dumps([1, 2]), encoding="utf-8")
            source_draft.write_text(json.dumps([1, 2]), encoding="utf-8")
            store = daily.GenerationStore(root / "state", FakeParser())
            first = store.commit_pair(source_target, source_draft, [1, 2])
            source_target.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            source_draft.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            second = store.commit_pair(source_target, source_draft, [1, 2, 3])
            self.assertNotEqual(first["generation"], second["generation"])
            self.assertTrue((root / "state" / "generations" / first["generation"] / "target.slot.bin").is_file())
            self.assertEqual(store.current()["generation"], second["generation"])

    def test_missing_or_mismatched_pair_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "target.slot.bin"
            target.write_text(json.dumps([1, 2]), encoding="utf-8")
            store = daily.GenerationStore(root / "state", FakeParser())
            with self.assertRaises(daily.ProfileError):
                store.commit_pair(target, root / "missing.dft", [1, 2])
            draft = root / "target.slot.bin.dft"
            draft.write_text(json.dumps([9, 9]), encoding="utf-8")
            with self.assertRaises(ValueError):
                store.commit_pair(target, draft, [1, 2])

    def test_exact_prefix_and_target_only_marks_mtp_stale(self):
        self.assertEqual(daily.assert_exact_prefix([1, 2], [1, 2, 3]), [3])
        with self.assertRaises(daily.PrefixMismatch):
            daily.assert_exact_prefix([1, 2], [1, 9, 3])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "target.slot.bin"
            target.write_text(json.dumps([1, 2]), encoding="utf-8")
            store = daily.GenerationStore(root / "state", FakeParser())
            pointer = store.commit_target_only(target, [1, 2])
            self.assertEqual(pointer["mtp_state"], "STALE/RESYNC_REQUIRED")
            draft = root / "target.slot.bin.dft"
            draft.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(daily.ResyncRequired):
                store.commit_pair(target, draft, [1, 2, 3])

    def test_no_silent_fallback_after_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend = FakeBackend(root, output="should-not-save")
            backend.partial = True
            session = daily.DailySession(root / "state", backend, mode="target", store=daily.GenerationStore(root / "state", FakeParser()))
            with self.assertRaises(daily.ProfileError):
                session.process({"mode": "target", "messages": [{"role": "user", "content": "hello"}]})
            self.assertIsNone(session.store.current())

    def test_import_does_not_execute_main_or_spawn_process(self):
        code = "import importlib.util; s=importlib.util.spec_from_file_location('d', r'%s'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('imported')" % (ROOT / "scripts" / "q3ple_daily_profile.py")
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(completed.stdout.strip(), "imported")


if __name__ == "__main__":
    unittest.main()
