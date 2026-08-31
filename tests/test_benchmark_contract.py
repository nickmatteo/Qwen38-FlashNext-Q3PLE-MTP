from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("q38_benchmark", ROOT / "scripts" / "benchmark.py")
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def load_manifest():
    return benchmark.load_json(ROOT / "benchmarks" / "manifest.json")


def valid_result_row():
    digest = "a" * 64
    return {
        "schema_version": "q38-public-benchmark-v1",
        "run_id": "test-target-1",
        "timestamp_utc": "2026-08-30T00:00:00Z",
        "suite_id": "PUBLIC-BENCH-001",
        "stage_id": "target_performance",
        "mode": "target",
        "evidence_class": "MEASURED",
        "status": "VALID",
        "model": {"repo": "local", "revision": "pin", "quant_recipe": "recipe"},
        "runtime": {"repo": "local", "commit": "pin", "executable_sha256": digest},
        "fixture": {"id": "fixture", "prompt_sha256": digest},
        "settings": {},
        "metrics": {"decode_tps": 1.0, "drafted_tokens": None},
        "resources": {
            "ram_available_min_bytes": 8 * 1024**3,
            "vram_free_min_mib": 1024,
            "pagefile_growth_bytes": 0,
            "cold_warm": "warm",
        },
        "correctness": {"output_sha256": digest, "token_ids_sha256": digest},
        "artifacts": {},
        "contamination": {"void_reasons": []},
    }


class BenchmarkContractTests(unittest.TestCase):
    def test_repository_manifest_is_valid(self):
        errors, _warnings = benchmark.validate_manifest(load_manifest(), ROOT)
        self.assertEqual(errors, [])

    def test_promoted_artifact_parity_gate_excludes_legacy_statefix_rows(self):
        manifest = load_manifest()
        mtp_stage = next(stage for stage in manifest["stages"] if stage["id"] == "mtp_performance")
        self.assertEqual(
            mtp_stage["gate_status"],
            "BLOCKED_PROMOTED_ARTIFACT_REJECTION_PARITY",
        )
        exclusions = manifest["evidence_policy"]["historical_measurement_exclusions"]
        self.assertTrue(exclusions)
        self.assertIn("earlier AtomicChat Q4_K_M target", exclusions[0]["reason"])
        self.assertIn("2.786 GB Q4_K_M sidecar", exclusions[0]["reason"])

    def test_promoted_performance_requires_visual_evidence_protocol(self):
        policy = load_manifest()["evidence_policy"]["visual_evidence"]
        self.assertTrue(policy["required_for_promoted_performance"])
        self.assertTrue(policy["screenshots_are_corroborating"])
        self.assertEqual(
            policy["required_stages"],
            ["preflight", "loaded", "steady", "result", "postflight"],
        )
        self.assertEqual(policy["protocol"], "docs/BENCHMARK_EVIDENCE_CAPTURE.md")

    def test_third_party_ppl_corpus_is_local_only(self):
        fixtures = load_manifest()["fixtures"]
        self.assertNotIn(
            "results/A0-QUALITY-001/independent_corpus_v2.txt",
            fixtures["ppl_corpora"],
        )
        local_only = fixtures["local_only_ppl_corpora"]
        self.assertEqual(local_only[0]["distribution"], "EXCLUDED_THIRD_PARTY_SOURCE_CORPUS")

    def test_release_plan_is_dependency_ordered(self):
        manifest = load_manifest()
        stages = {item["id"]: item for item in manifest["stages"]}
        selected = set(manifest["profiles"]["release"]["stages"])
        order, errors = benchmark._topological_order(stages, selected)
        self.assertEqual(errors, [])
        positions = {stage_id: index for index, stage_id in enumerate(order)}
        for stage_id in order:
            for dependency in stages[stage_id]["depends_on"]:
                self.assertLess(positions[dependency], positions[stage_id])

    def test_valid_target_result_passes_policy(self):
        self.assertEqual(benchmark.validate_result_row(valid_result_row(), load_manifest()), [])

    def test_valid_mtp_result_requires_exact_parity_and_drafts(self):
        row = valid_result_row()
        row["run_id"] = "test-mtp-1"
        row["stage_id"] = "mtp_performance"
        row["mode"] = "mtp"
        errors = benchmark.validate_result_row(row, load_manifest())
        self.assertTrue(any("exact_text=true" in error for error in errors))
        self.assertTrue(any("exact_token_ids=true" in error for error in errors))
        self.assertTrue(any("drafted_tokens > 0" in error for error in errors))

    def test_occupied_context_requires_ratio_and_needle(self):
        row = valid_result_row()
        row["stage_id"] = "occupied_context_target"
        row["fixture"].update({"prompt_tokens_target": 8192, "actual_prompt_tokens": 7000})
        errors = benchmark.validate_result_row(row, load_manifest())
        self.assertTrue(any("prompt occupancy" in error for error in errors))
        self.assertTrue(any("needle_pass=true" in error for error in errors))

    def test_fixture_files_are_jsonl_objects_with_unique_ids(self):
        manifest = load_manifest()
        for key in ("performance", "context_needles", "development_smoke"):
            rows = benchmark.read_jsonl(ROOT / manifest["fixtures"][key])
            ids = [row["id"] for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(isinstance(json.dumps(row), str) for row in rows))


if __name__ == "__main__":
    unittest.main()
