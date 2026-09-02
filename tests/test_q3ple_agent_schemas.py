import copy
import importlib.util
import json
from pathlib import Path
import unittest

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "benchmarks" / "agent"


def load_task_preflight():
    path = ROOT / "scripts" / "q3ple_agent_task.py"
    spec = importlib.util.spec_from_file_location("q3ple_agent_task_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AgentSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_schema = json.loads((AGENT_ROOT / "task.schema.json").read_text(encoding="utf-8"))
        cls.result_schema = json.loads((AGENT_ROOT / "result.schema.json").read_text(encoding="utf-8"))
        cls.task_validator = jsonschema.Draft202012Validator(cls.task_schema)
        cls.result_validator = jsonschema.Draft202012Validator(cls.result_schema)
        cls.preflight = load_task_preflight()
        cls.valid_task = {
            "schema": "q3ple-real-agent-task-v1",
            "task_id": "pilot.seeded-bug-001",
            "title": "Fix the seeded cache invalidation bug",
            "category": "seeded-bug",
            "prompt": "Fix the seeded defect, add a regression test, and run the declared verifier.",
            "fixture": {
                "source": "benchmarks/agent/fixtures/pilot-repo",
                "revision": "0123456789abcdef",
                "setup_sha256": "A" * 64,
                "disposable_worktree": True,
                "setup_command": ["python", "setup_task.py"],
            },
            "permissions": {
                "allowed_paths": ["src", "tests"],
                "allowed_commands": ["python -m pytest"],
                "network": False,
                "push": False,
                "merge": False,
                "external_side_effects": False,
            },
            "limits": {
                "wall_seconds": 1800,
                "max_turns": 50,
                "max_human_interventions": 0,
                "restart_after_turn": None,
            },
            "verifier": {
                "command": ["python", "-m", "pytest", "-q"],
                "expected_exit": 0,
                "result_parser": "exit-code",
            },
        }
        cls.valid_result = {
            "schema": "q3ple-real-agent-result-v1",
            "run_id": "pilot-run-001",
            "task_id": "pilot.seeded-bug-001",
            "status": "VALID",
            "evidence_class": "MEASURED",
            "harness": {"name": "pi", "version": "0.83.0", "adapter_sha256": "A" * 64},
            "model": {"target_manifest_sha256": "B" * 64, "target_bytes": 78525318176, "mode": "target"},
            "runtime": {
                "commit": "73b803464f25fc9054046728bf2ebed5a372737e",
                "executable_sha256": "C" * 64,
            },
            "profile": {
                "id": "q3ple_daily_80k_reasoning_v1",
                "sha256": "D" * 64,
                "reasoning_effort": "medium",
                "reasoning_budget": 8192,
            },
            "outcome": {"task_pass": True, "verifier_exit": 0, "human_interventions": 0, "unrelated_changes": []},
            "agent": {
                "turns": 3,
                "tool_calls": 2,
                "malformed_tool_calls": 0,
                "wall_ms": 1000,
                "time_to_first_edit_ms": 300,
                "longest_autonomous_ms": 900,
            },
            "cache": {"all_turns_exact": True, "turn_rows_sha256": "E" * 64, "restart_restore_pass": True},
            "resources": {
                "min_ram_available_bytes": 1,
                "min_vram_free_mib": 1,
                "max_owned_rss_bytes": 1,
                "pagefile_growth_bytes": 0,
                "watchdog_violations": [],
            },
            "artifacts": {
                "task_manifest": "evidence/task.json",
                "turns_jsonl": "evidence/turns.jsonl",
                "verifier_log": "evidence/verifier.log",
                "diff": "evidence/task.diff",
                "telemetry": "evidence/telemetry.jsonl",
            },
        }

    def test_schemas_are_valid_draft_2020_12(self):
        jsonschema.Draft202012Validator.check_schema(self.task_schema)
        jsonschema.Draft202012Validator.check_schema(self.result_schema)

    def test_valid_task_passes(self):
        self.task_validator.validate(self.valid_task)

    def test_network_enabled_task_is_rejected(self):
        task = copy.deepcopy(self.valid_task)
        task["permissions"]["network"] = True
        with self.assertRaises(jsonschema.ValidationError):
            self.task_validator.validate(task)

    def test_non_disposable_task_is_rejected(self):
        task = copy.deepcopy(self.valid_task)
        task["fixture"]["disposable_worktree"] = False
        with self.assertRaises(jsonschema.ValidationError):
            self.task_validator.validate(task)

    def test_result_rejects_negative_resources_and_null_artifacts(self):
        self.result_validator.validate(self.valid_result)
        negative = copy.deepcopy(self.valid_result)
        negative["resources"]["min_vram_free_mib"] = -1
        with self.assertRaises(jsonschema.ValidationError):
            self.result_validator.validate(negative)
        null_artifact = copy.deepcopy(self.valid_result)
        null_artifact["artifacts"]["telemetry"] = None
        with self.assertRaises(jsonschema.ValidationError):
            self.result_validator.validate(null_artifact)

    def test_task_paths_and_commands_fail_closed(self):
        traversal = copy.deepcopy(self.valid_task)
        traversal["fixture"]["source"] = "../private"
        with self.assertRaises(jsonschema.ValidationError):
            self.task_validator.validate(traversal)
        destructive = copy.deepcopy(self.valid_task)
        destructive["permissions"]["allowed_commands"] = ["Remove-Item -Recurse ."]
        destructive["verifier"]["command"] = ["Remove-Item", "-Recurse", "."]
        with self.assertRaises(self.preflight.TaskError):
            self.preflight.validate_task(destructive)

    def test_verifier_must_be_covered_by_allowed_commands(self):
        task = copy.deepcopy(self.valid_task)
        task["permissions"]["allowed_commands"] = ["python -m unittest"]
        with self.assertRaises(self.preflight.TaskError):
            self.preflight.validate_task(task)

    def test_pilot_manifest_is_pi_then_dsh_and_unpromoted(self):
        manifest = json.loads((AGENT_ROOT / "pilot-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["harness_order"], ["pi", "deepseek-harness"])
        self.assertEqual(manifest["initial_mode"], "target")
        self.assertFalse(manifest["promotion_gates"]["live_20_turn_cache_proof"])
        self.assertEqual(manifest["tasks"], [])


if __name__ == "__main__":
    unittest.main()
