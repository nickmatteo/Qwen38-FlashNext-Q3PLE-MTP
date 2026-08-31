from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "q3ple_canonical_60k.py"
SPEC = importlib.util.spec_from_file_location("q3ple_canonical_60k_test", SCRIPT)
assert SPEC and SPEC.loader
canonical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canonical)


class CanonicalHarnessTests(unittest.TestCase):
    def test_import_has_no_execution_side_effects(self):
        # The module is already loaded above. Re-executing it with process and
        # socket entry points blocked proves import does not launch a runtime.
        fresh_spec = importlib.util.spec_from_file_location("q3ple_import_probe", SCRIPT)
        assert fresh_spec and fresh_spec.loader
        fresh = importlib.util.module_from_spec(fresh_spec)
        with mock.patch("subprocess.Popen", side_effect=AssertionError("spawned on import")):
            with mock.patch("socket.socket", side_effect=AssertionError("opened socket on import")):
                fresh_spec.loader.exec_module(fresh)
        self.assertTrue(hasattr(fresh, "select_canonical_boundary"))

    def test_fixture_is_deterministic_and_ends_with_assistant(self):
        result = canonical.validate_fixture(canonical.ROOT / "benchmarks" / "fixtures" / "q3ple_canonical_history.json")
        self.assertTrue(result["valid"])
        self.assertEqual(result["final_role"], "assistant")
        self.assertGreater(result["message_count"], 10)
        self.assertEqual(result["boundary_tokens"], {"minimum": 59000, "maximum": 60000})

    def test_whole_message_prefix_stability(self):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]

        def renderer(rows):
            return "|".join(row["content"] for row in rows)

        def tokenizer(text):
            return [ord(char) for char in text]

        records = canonical.render_boundary_records(messages, renderer, tokenizer)
        self.assertEqual([row["token_count"] for row in records], [1, 3, 5])
        self.assertTrue(canonical.assert_exact_message_prefixes(records))
        selected = canonical.select_canonical_boundary(records, 5, 5 + 1)
        self.assertEqual(selected["last_role"], "assistant")
        self.assertTrue(selected["canonical_chat_template_boundary"])

    def test_mid_message_and_nonprefix_vectors_are_rejected(self):
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]

        def renderer(rows):
            return " ".join(row["content"] for row in rows)

        def unstable_tokenizer(text):
            # Appending a message changes the last token, simulating a BPE
            # boundary that was cut in the middle of a message.
            return [len(text)]

        with self.assertRaises(ValueError):
            canonical.render_boundary_records(messages, renderer, unstable_tokenizer)

        bad_records = [
            {"message_index": 1, "message_count": 1, "last_role": "user", "tokens": [1], "token_count": 1, "exact_previous_prefix": True},
            {"message_index": 2, "message_count": 2, "last_role": "assistant", "tokens": [2, 3], "token_count": 2, "exact_previous_prefix": False},
        ]
        with self.assertRaises(ValueError):
            canonical.assert_exact_message_prefixes(bad_records)
        with self.assertRaises(ValueError):
            canonical.select_canonical_boundary(bad_records, 1, 3)

    def test_boundary_requires_range_and_final_assistant(self):
        records = [
            {"message_index": 1, "message_count": 1, "last_role": "user", "tokens": [1, 2], "token_count": 2, "exact_previous_prefix": True},
            {"message_index": 2, "message_count": 2, "last_role": "assistant", "tokens": [1, 2, 3], "token_count": 3, "exact_previous_prefix": True, "complete_assistant_boundary": True},
        ]
        with self.assertRaises(ValueError):
            canonical.select_canonical_boundary(records, 10, 20)
        picked = canonical.select_canonical_boundary(records, 3, 4)
        self.assertEqual(picked["message_index"], 2)
        self.assertEqual(picked["last_role"], "assistant")

    def test_completion_accounting_and_suffix_submission(self):
        def poster(_port, _path, body, _timeout):
            length = len(body["prompt"])
            return {"timings": {"cache_n": length - 1, "prompt_n": 1}}

        submitted = canonical.submit_unseen_suffix_chunks(
            18089, [1, 2], [1, 2, 3, 4], chunk_tokens=1, poster=poster
        )
        self.assertEqual([(row["start"], row["end"]) for row in submitted], [(2, 3), (3, 4)])
        self.assertTrue(all(row["pass"] for row in submitted))
        self.assertTrue(all("prompt_tokens" not in row for row in submitted))

    def test_restart_restore_probe_requires_a_real_canonical_suffix(self):
        base = {
            "tokens": [1, 2, 3],
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        rendered = "canonical user suffix"
        tokenized = [1, 2, 3, 4, 5]
        fake_probe = mock.Mock()
        fake_probe.tokenize.return_value = tokenized
        with mock.patch.object(canonical, "post_apply_template", return_value=rendered):
            with mock.patch.object(canonical, "reference_modules", return_value=(fake_probe, None, None)):
                record = canonical.build_restart_suffix_probe(18089, base)
        self.assertEqual(record["suffix_token_count"], 2)
        self.assertTrue(record["base_prefix_exact"])
        self.assertEqual(record["prompt_tokens"], tokenized)

        fake_probe.tokenize.return_value = [9, 2, 3, 4]
        with mock.patch.object(canonical, "post_apply_template", return_value=rendered):
            with mock.patch.object(canonical, "reference_modules", return_value=(fake_probe, None, None)):
                with self.assertRaisesRegex(ValueError, "preserve the sealed token prefix"):
                    canonical.build_restart_suffix_probe(18089, base)

    def test_transport_chunks_are_exact_suffixes_of_one_vector(self):
        records = canonical.suffix_chunk_ranges([0, 1], list(range(11)), chunk_tokens=4)
        self.assertEqual([(row["start"], row["end"]) for row in records], [(2, 6), (6, 10), (10, 11)])
        self.assertEqual([row["expected_cache_n"] for row in records], [2, 6, 10])
        self.assertEqual([row["expected_prompt_n"] for row in records], [4, 4, 1])
        with self.assertRaises(ValueError):
            canonical.suffix_chunk_ranges([9], [1, 2, 3])

    def test_fixture_contract_rejects_missing_category_requirements(self):
        row = {
            "id": "bad-retrieval",
            "category": "retrieval",
            "system": "s",
            "user": "u",
            "scorer": "retrieval_needle",
            "max_tokens": 0,
            "needle": "n",
            "provenance": {"static": True},
            "semantic_requirements": {"exact_line": "n"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive integer max_tokens"):
                canonical.load_benchmark_fixtures(path)

    def test_explicit_fixture_subset_is_fail_closed(self):
        fixtures = [
            {"id": "retrieval", "category": "retrieval"},
            {"id": "code", "category": "code"},
        ]
        selected = canonical.select_benchmark_fixtures(fixtures, ["retrieval"])
        self.assertEqual([row["id"] for row in selected], ["retrieval"])
        with self.assertRaisesRegex(ValueError, "unknown fixture ids"):
            canonical.select_benchmark_fixtures(fixtures, ["missing"])

    def test_atomic_json_scorer_accepts_a_path_parent_alias(self):
        realistic = canonical.reference_modules()[2]
        output = """import json
import os
import pathlib
import tempfile

def atomic_json(path: pathlib.Path, value: object) -> None:
    path = pathlib.Path(path)
    directory = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(directory), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
"""
        score = realistic.score_code_atomic_json(output)
        self.assertTrue(score["semantic"]["same_directory_temp"])
        self.assertTrue(score["valid"])

    def test_slot_pair_copy_is_no_overwrite_and_revalidated(self):
        tokens = [1, 2, 3]

        def fake_gate(target, expected, *, expected_count):
            target = Path(target)
            draft = Path(f"{target}.dft")
            self.assertEqual(list(expected), tokens)
            self.assertEqual(expected_count, len(tokens))
            return {
                "target_path": str(target),
                "draft_path": str(draft),
                "target_bytes": target.stat().st_size,
                "draft_bytes": draft.stat().st_size,
                "target_sha256": canonical.sha256_file(target),
                "draft_sha256": canonical.sha256_file(draft),
                "pair_promotion_pass": True,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.slot.bin"
            source.write_bytes(b"target-state")
            Path(f"{source}.dft").write_bytes(b"draft-state")
            destination = root / "destination"
            destination.mkdir()
            copied = canonical.copy_slot_pair(
                source, destination, "copy.slot.bin", tokens, pair_gate=fake_gate
            )
            self.assertTrue(copied["bytes_and_hashes_exact"])
            self.assertEqual((destination / "copy.slot.bin").read_bytes(), b"target-state")
            self.assertEqual(Path(f"{destination / 'copy.slot.bin'}.dft").read_bytes(), b"draft-state")
            with self.assertRaises(FileExistsError):
                canonical.copy_slot_pair(
                    source, destination, "copy.slot.bin", tokens, pair_gate=fake_gate
                )

    def test_pair_gate_and_benchmark_gate_are_conservative(self):
        # Avoid a live slot parser here; the pair gate is exercised through the
        # benchmark gate's exact, rejection-bearing checks below.
        target = {
            "output_sha256": "same",
            "output_token_ids_sha256": "same-token",
            "semantic_vector": {"ok": True},
            "semantic_score": {"valid": True},
            "natural_stop": True,
            "draft_n": 0,
        }
        mtp = dict(target, draft_n=3, draft_n_accepted=3)
        good = canonical.benchmark_promotion_gate([target] * 3, [mtp] * 3, resources={"pass": True})
        self.assertTrue(good["promotable"])
        self.assertEqual(good["classification"], "PROMOTABLE")

        mismatch = dict(mtp, output_sha256="different")
        rejected = canonical.benchmark_promotion_gate([target] * 3, [mismatch] * 3, resources={"pass": True})
        self.assertFalse(rejected["promotable"])
        self.assertEqual(rejected["classification"], "MTP_NON_PROMOTABLE")
        self.assertTrue(any("mismatch" in reason for reason in rejected["reasons"]))

        rejection_required = canonical.benchmark_promotion_gate(
            [target] * 3,
            [dict(mtp, draft_n_accepted=2)] * 3,
            resources={"pass": True},
            require_rejection=True,
        )
        self.assertTrue(rejection_required["promotable"])
        all_accepted = canonical.benchmark_promotion_gate(
            [target] * 3,
            [mtp] * 3,
            resources={"pass": True},
            require_rejection=True,
        )
        self.assertFalse(all_accepted["promotable"])

    def test_public_base_record_is_hash_only(self):
        base = {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "assistant", "content": "a"},
            ],
            "raw_prompt": "private source text",
            "tokens": [1, 2],
            "token_count": 2,
            "token_ids_sha256": canonical.sha256_tokens([1, 2]),
            "message_manifest": [
                {"index": 1, "role": "system"},
                {"index": 2, "role": "assistant"},
            ],
            "_assistant_boundary_records": [{"tokens": [1, 2]}],
        }
        public = canonical.public_base_record(base)
        self.assertNotIn("messages", public)
        self.assertNotIn("raw_prompt", public)
        self.assertNotIn("tokens", public)
        self.assertEqual(public["token_ids_sha256"], canonical.sha256_tokens([1, 2]))

    def test_generation_prompt_must_preserve_boundary(self):
        boundary = {
            "messages": [{"role": "assistant", "content": "done"}],
            "tokens": [1, 2],
        }
        fixture = {"id": "fixture", "system": "instruction", "user": "task"}

        def renderer(messages, add_generation_prompt=False):
            self.assertTrue(add_generation_prompt)
            return "done task"

        self.assertTrue(canonical.build_benchmark_prompt(boundary, fixture, renderer, lambda _: [1, 2, 3])["base_is_exact_prefix"])
        with self.assertRaises(ValueError):
            canonical.build_benchmark_prompt(boundary, fixture, renderer, lambda _: [9, 2, 3])

    def test_stage_selection_uses_strict_complete_prefix_boundaries(self):
        vector = list(range(1, 60_001))
        records = []
        for index, count in enumerate((16_000, 32_000, 48_000, 59_500), 1):
            records.append({
                "message_index": index,
                "message_count": index,
                "last_role": "assistant",
                "tokens": vector[:count],
                "token_count": count,
                "token_ids_sha256": canonical.sha256_tokens(vector[:count]),
                "exact_previous_prefix": True,
                "complete_assistant_boundary": True,
            })
        stages = canonical.select_stage_boundaries(records)
        self.assertEqual([stage["token_count"] for stage in stages], [16_000, 32_000, 48_000, 59_500])
        self.assertEqual([stage["stage"] for stage in stages], [1, 2, 3, 4])
        self.assertTrue(all(stage["last_role"] == "assistant" for stage in stages))
        self.assertTrue(all(
            stages[index + 1]["tokens"][: stages[index]["token_count"]] == stages[index]["tokens"]
            for index in range(len(stages) - 1)
        ))

    def test_stage_selection_rejects_non_prefix_and_non_increasing_vectors(self):
        good = [{
            "message_index": 1, "last_role": "assistant", "tokens": [1, 2],
            "token_count": 2, "exact_previous_prefix": True,
            "complete_assistant_boundary": True,
        }, {
            "message_index": 2, "last_role": "assistant", "tokens": [1, 9, 3],
            "token_count": 3, "exact_previous_prefix": True,
            "complete_assistant_boundary": True,
        }]
        with self.assertRaises(ValueError):
            canonical.select_stage_boundaries(good, targets=(2,), minimum=3, maximum=4)
        with self.assertRaises(ValueError):
            canonical.suffix_chunk_ranges([1, 2], [1, 9, 3])

    def test_stage_selection_requires_explicit_complete_assistant_proof(self):
        records = [{
            "message_index": 1,
            "last_role": "assistant",
            "tokens": [1, 2],
            "token_count": 2,
            "exact_previous_prefix": True,
        }, {
            "message_index": 2,
            "last_role": "assistant",
            "tokens": [1, 2, 3],
            "token_count": 3,
            "exact_previous_prefix": True,
            "complete_assistant_boundary": True,
        }]
        with self.assertRaises(ValueError):
            canonical.select_stage_boundaries(
                records, targets=(2,), minimum=3, maximum=4
            )

        missing_final_proof = [{
            "message_index": 1,
            "last_role": "assistant",
            "tokens": [1, 2],
            "token_count": 2,
            "exact_previous_prefix": True,
            "complete_assistant_boundary": True,
        }, {
            "message_index": 2,
            "last_role": "assistant",
            "tokens": [1, 2, 3],
            "token_count": 3,
            "exact_previous_prefix": True,
        }]
        with self.assertRaises(ValueError):
            canonical.select_stage_boundaries(
                missing_final_proof, targets=(2,), minimum=3, maximum=4
            )

    def test_suffix_submission_has_no_replay_in_accounting(self):
        calls = []
        expected_starts = [2, 4, 6]

        def poster(_port, _path, body, _timeout):
            prompt = body["prompt"]
            start = expected_starts[len(calls)]
            calls.append(prompt)
            return {"timings": {"cache_n": start, "prompt_n": len(prompt) - start}}

        rows = canonical.submit_unseen_suffix_chunks(18089, [1, 2], list(range(1, 8)), chunk_tokens=2, poster=poster)
        self.assertEqual([row["start"] for row in rows], [2, 4, 6])
        self.assertEqual([row["delta_tokens"] for row in rows], [2, 2, 1])
        self.assertEqual([row["expected_cache_n"] for row in rows], [2, 4, 6])
        self.assertTrue(all(row["pass"] for row in rows))
        self.assertEqual([len(prompt) for prompt in calls], [4, 6, 7])

    def test_stage_filenames_are_unique_and_resources_aggregate_globally(self):
        paths = {"run_dir": canonical.ROOT / "logs" / "stage-test"}
        first = canonical._staged_stage_paths(paths, 1, 16_000)
        second = canonical._staged_stage_paths(paths, 2, 32_000)
        self.assertNotEqual(first["slot_filename"], second["slot_filename"])
        baseline = 10 * 1024**3
        summary = canonical.combined_resource_summary(
            {"sample_count": 2, "min_ram_available_bytes": 8 * 1024**3,
             "min_vram_free_mib": 1500, "max_owned_rss_bytes": 40 * 1024**3,
             "pagefile_growth_bytes": 128, "max_pagefile_used_bytes": baseline + 512,
             "pass": True},
            {"sample_count": 2, "min_ram_available_bytes": 7 * 1024**3,
             "min_vram_free_mib": 1200, "max_owned_rss_bytes": 41 * 1024**3,
             "pagefile_growth_bytes": 256, "max_pagefile_used_bytes": baseline + 2 * 1024**3,
             "pass": True},
            workflow_pagefile_baseline_bytes=baseline,
        )
        self.assertEqual(summary["global_pagefile_growth_bytes"], 2 * 1024**3)
        self.assertFalse(summary["pass"])
        self.assertIn("workflow_pagefile_growth>=1GiB", summary["violations"])

    def test_staged_ingestion_proof_recomputes_every_prefix_and_cache_gate(self):
        final_tokens = [1, 2, 3, 4]

        def submission(start, end):
            delta = end - start
            return {
                "start": start,
                "end": end,
                "delta_tokens": delta,
                "expected_cache_n": start,
                "expected_prompt_n": delta,
                "prompt_token_ids_sha256": canonical.sha256_tokens(final_tokens[:end]),
                "suffix_token_ids_sha256": canonical.sha256_tokens(final_tokens[start:end]),
                "accounting": {"cache_n": start, "prompt_n": delta, "pass": True},
                "pass": True,
            }

        stages = [
            {
                "stage": 1,
                "message_index": 3,
                "token_count": 2,
                "token_ids_sha256": canonical.sha256_tokens(final_tokens[:2]),
                "complete_assistant_boundary": True,
                "suffix_submissions": [submission(0, 2)],
                "pair": {"pair_promotion_pass": True},
                "pass": True,
                "sealed": True,
            },
            {
                "stage": 2,
                "message_index": 5,
                "token_count": 4,
                "token_ids_sha256": canonical.sha256_tokens(final_tokens),
                "complete_assistant_boundary": True,
                "restore": {"n_restored": 2},
                "restore_suffix_probe": submission(2, 4),
                "suffix_submissions": [submission(2, 4)],
                "pair": {"pair_promotion_pass": True},
                "pass": True,
                "sealed": True,
            },
        ]

        proof = canonical.staged_ingestion_proof(stages, final_tokens)
        self.assertEqual(proof["stage_count"], 2)
        self.assertEqual(proof["final_token_ids_sha256"], canonical.sha256_tokens(final_tokens))
        self.assertTrue(proof["all_complete_assistant_boundaries"])
        self.assertTrue(proof["all_cache_accounting_exact"])

        tampered = json.loads(json.dumps(stages))
        tampered[1]["suffix_submissions"][0]["suffix_token_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from first submission"):
            canonical.staged_ingestion_proof(tampered, final_tokens)

        missing_probe = json.loads(json.dumps(stages))
        del missing_probe[1]["restore_suffix_probe"]
        with self.assertRaisesRegex(ValueError, "first-suffix restore probe failed"):
            canonical.staged_ingestion_proof(missing_probe, final_tokens)

        divergent_probe = json.loads(json.dumps(stages))
        divergent_probe[1]["restore_suffix_probe"]["suffix_token_ids_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "differs from first submission"):
            canonical.staged_ingestion_proof(divergent_probe, final_tokens)

    def test_build_main_returns_nonzero_when_local_gate_fails(self):
        failed = {
            "status": "FAILED",
            "local_build_gate": False,
            "base": {"token_count": 59_500},
            "paths": {"result": "failed.json"},
        }
        with mock.patch.object(canonical, "build_live", return_value=failed):
            self.assertEqual(
                canonical.main(["--mode", "build", "--tag", "offline-failed"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
