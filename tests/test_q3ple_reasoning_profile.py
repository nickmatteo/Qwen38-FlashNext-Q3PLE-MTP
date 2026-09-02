import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = ROOT / "profiles" / "q3ple_daily_80k.json"
REASONING_PROFILE_PATH = ROOT / "profiles" / "q3ple_daily_80k_reasoning.json"


def load_daily_module():
    path = ROOT / "scripts" / "q3ple_daily_profile.py"
    spec = importlib.util.spec_from_file_location("q3ple_daily_profile_reasoning_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReasoningProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = load_daily_module()
        cls.default = json.loads(DEFAULT_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.reasoning = json.loads(REASONING_PROFILE_PATH.read_text(encoding="utf-8"))

    def test_profile_passes_existing_static_contract(self):
        self.assertEqual(self.daily.validate_profile(self.reasoning), [])

    def test_profile_uses_independent_identity_port_and_state(self):
        self.assertNotEqual(self.reasoning["profile_id"], self.default["profile_id"])
        self.assertNotEqual(self.reasoning["server"]["port"], self.default["server"]["port"])
        self.assertNotEqual(self.reasoning["state"]["directory"], self.default["state"]["directory"])
        self.assertNotEqual(self.reasoning["server"]["slot_save_path"], self.default["server"]["slot_save_path"])

    def test_target_and_runtime_pins_are_unchanged(self):
        self.assertEqual(self.reasoning["runtime"], self.default["runtime"])
        self.assertEqual(self.reasoning["artifacts"]["target"], self.default["artifacts"]["target"])
        self.assertEqual(self.reasoning["safety"], self.default["safety"])

    def test_only_target_mode_is_allowed(self):
        self.assertEqual(self.reasoning["policy"]["default_mode"], "target")
        self.assertEqual(self.reasoning["policy"]["allowed_modes"], ["target"])
        self.assertTrue(self.reasoning["server"]["modes"]["target"]["enabled"])
        self.assertFalse(self.reasoning["server"]["modes"]["mtp"]["enabled"])
        self.assertFalse(self.reasoning["artifacts"]["sidecar"]["enabled"])
        with self.assertRaises(self.daily.ProfileError):
            self.daily.build_command("mtp", self.reasoning)

    def test_reasoning_flags_are_explicit_and_preservation_is_gated(self):
        command = self.daily.build_command("target", self.reasoning)
        joined = " ".join(command)
        self.assertIn("--reasoning on", joined)
        self.assertIn("--reasoning-effort medium", joined)
        self.assertIn("--reasoning-budget 8192", joined)
        self.assertIn("--reasoning-format deepseek", joined)
        self.assertIn("--no-reasoning-preserve", command)
        self.assertIn("--spec-type none", joined)
        self.assertNotIn("-md", command)

    def test_sampling_matches_official_thinking_recommendation(self):
        self.assertEqual(
            self.reasoning["reasoning"]["sampling"],
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
        )

    def test_one_slot_contract_includes_parallel(self):
        contract = self.reasoning["client_contract"]
        self.assertEqual(contract["id_slot"], 0)
        self.assertIs(contract["cache_prompt"], True)
        self.assertEqual(contract["parallel"], 1)
        self.assertEqual(contract["required_extra_body"], {"id_slot": 0, "cache_prompt": True, "parallel": 1})

    def test_cli_selects_reasoning_profile_without_live_work(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "q3ple_daily_profile.py"),
                "--profile",
                "profiles/q3ple_daily_80k_reasoning.json",
                "smoke",
                "--mode",
                "target",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["live"])
        self.assertEqual(payload["mode"], "target")
        self.assertEqual(payload["port"], 18090)
        self.assertIn("q3ple_daily_reasoning_v1", payload["state_dir"])

    def test_cli_rejects_mtp_for_reasoning_profile(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "q3ple_daily_profile.py"),
                "--profile",
                "profiles/q3ple_daily_80k_reasoning.json",
                "smoke",
                "--mode",
                "mtp",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("disabled by profile", completed.stderr)


if __name__ == "__main__":
    unittest.main()
