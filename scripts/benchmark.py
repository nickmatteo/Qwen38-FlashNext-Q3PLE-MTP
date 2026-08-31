#!/usr/bin/env python3
"""Validate and expand the Qwen3.8 public benchmark contract.

This tool is intentionally orchestration-only. It does not download datasets,
start model servers, execute generated code, or manufacture benchmark results.
The existing guarded runtime harnesses produce append-only result rows; this
tool makes their dependencies and evidence policy machine-checkable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "q38-benchmark-manifest-v1"
RESULT_SCHEMA = "q38-public-benchmark-v1"


class ContractError(ValueError):
    """Raised when a benchmark contract cannot be parsed."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSONL file: {path}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON at {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"expected an object at {path}:{number}")
        row.setdefault("_source_line", number)
        rows.append(row)
    return rows


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_ids(items: Iterable[dict[str, Any]], label: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{label}[{index}] has no non-empty id")
            continue
        if item_id in found:
            errors.append(f"duplicate {label} id: {item_id}")
            continue
        found[item_id] = item
    return found


def _topological_order(
    stages: dict[str, dict[str, Any]], selected: set[str] | None = None
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    chosen = set(stages) if selected is None else set(selected)

    def add_dependencies(stage_id: str) -> None:
        stage = stages.get(stage_id)
        if stage is None:
            return
        for dependency in stage.get("depends_on", []):
            if dependency not in chosen:
                chosen.add(dependency)
                add_dependencies(dependency)

    for stage_id in list(chosen):
        add_dependencies(stage_id)

    permanent: set[str] = set()
    temporary: set[str] = set()
    order: list[str] = []

    def visit(stage_id: str, chain: list[str]) -> None:
        if stage_id in permanent:
            return
        if stage_id in temporary:
            errors.append("stage dependency cycle: " + " -> ".join(chain + [stage_id]))
            return
        stage = stages.get(stage_id)
        if stage is None:
            errors.append(f"unknown stage dependency: {stage_id}")
            return
        temporary.add(stage_id)
        for dependency in stage.get("depends_on", []):
            if dependency in chosen:
                visit(dependency, chain + [stage_id])
        temporary.remove(stage_id)
        permanent.add(stage_id)
        order.append(stage_id)

    for stage_id in stages:
        if stage_id in chosen:
            visit(stage_id, [])
    return order, errors


def validate_manifest(
    manifest: dict[str, Any], root: Path = REPO_ROOT, require_local_artifacts: bool = False
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if not isinstance(manifest.get("suite_id"), str) or not manifest.get("suite_id"):
        errors.append("suite_id must be a non-empty string")

    for key in (
        "evidence_policy",
        "artifacts",
        "runtime",
        "resource_policy",
        "fixtures",
        "stages",
        "profiles",
        "required_facets",
        "capability_tasks",
    ):
        if key not in manifest:
            errors.append(f"missing manifest section: {key}")

    stages_raw = manifest.get("stages", [])
    if not isinstance(stages_raw, list):
        errors.append("stages must be an array")
        stages_raw = []
    stages = _unique_ids(stages_raw, "stage", errors)
    for stage_id, stage in stages.items():
        dependencies = stage.get("depends_on")
        if not isinstance(dependencies, list) or any(not isinstance(x, str) for x in dependencies):
            errors.append(f"stage {stage_id} depends_on must be an array of strings")
            continue
        for dependency in dependencies:
            if dependency not in stages:
                errors.append(f"stage {stage_id} depends on unknown stage {dependency}")
        if not isinstance(stage.get("facets"), list) or not stage.get("facets"):
            errors.append(f"stage {stage_id} must declare at least one facet")
        if not isinstance(stage.get("gate_status"), str) or not stage.get("gate_status"):
            errors.append(f"stage {stage_id} must declare gate_status")
    _, dependency_errors = _topological_order(stages)
    errors.extend(dependency_errors)

    profiles = manifest.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        errors.append("profiles must be a non-empty object")
        profiles = {}
    for profile_id, profile in profiles.items():
        selected = profile.get("stages") if isinstance(profile, dict) else None
        if not isinstance(selected, list) or not selected:
            errors.append(f"profile {profile_id} must contain a non-empty stages array")
            continue
        for stage_id in selected:
            if stage_id not in stages:
                errors.append(f"profile {profile_id} references unknown stage {stage_id}")

    capability_raw = manifest.get("capability_tasks", [])
    if not isinstance(capability_raw, list):
        errors.append("capability_tasks must be an array")
        capability_raw = []
    capability_tasks = _unique_ids(capability_raw, "capability task", errors)
    for task_id, task in capability_tasks.items():
        if task.get("adapter") not in {"lm_eval", "local"}:
            errors.append(f"capability task {task_id} has unsupported adapter")
        if not isinstance(task.get("facets"), list) or not task.get("facets"):
            errors.append(f"capability task {task_id} must declare facets")
        if task.get("revision") == "PIN_BEFORE_RUN":
            warnings.append(f"capability task {task_id} is not revision-pinned")
        if task.get("unsafe_code"):
            warnings.append(f"capability task {task_id} requires an isolated unsafe-code scorer")

    required_facets = manifest.get("required_facets", [])
    if not isinstance(required_facets, list) or any(not isinstance(x, str) for x in required_facets):
        errors.append("required_facets must be an array of strings")
        required_facets = []
    covered: set[str] = set()
    for stage in stages.values():
        covered.update(x for x in stage.get("facets", []) if isinstance(x, str))
    for task in capability_tasks.values():
        covered.update(x for x in task.get("facets", []) if isinstance(x, str))
    for facet in required_facets:
        if facet not in covered:
            errors.append(f"required facet has no stage or task coverage: {facet}")

    fixtures = manifest.get("fixtures", {})
    if isinstance(fixtures, dict):
        for fixture_name in ("performance", "context_needles", "development_smoke"):
            value = fixtures.get(fixture_name)
            if not isinstance(value, str):
                errors.append(f"fixture path {fixture_name} must be a string")
                continue
            path = resolve_path(root, value)
            if not path.is_file():
                errors.append(f"missing fixture file {fixture_name}: {path}")
                continue
            try:
                rows = read_jsonl(path)
            except ContractError as exc:
                errors.append(str(exc))
                continue
            _unique_ids(rows, f"fixture {fixture_name}", errors)
    else:
        errors.append("fixtures must be an object")

    result_schema_value = manifest.get("result_schema")
    if not isinstance(result_schema_value, str):
        errors.append("result_schema must be a path string")
    else:
        schema_path = resolve_path(root, result_schema_value)
        try:
            schema = load_json(schema_path)
        except ContractError as exc:
            errors.append(str(exc))
        else:
            if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                errors.append("result schema must declare JSON Schema draft 2020-12")
            required = schema.get("required")
            if not isinstance(required, list) or not {"run_id", "status", "evidence_class"}.issubset(required):
                errors.append("result schema is missing core required fields")

    resource = manifest.get("resource_policy", {})
    if isinstance(resource, dict):
        numeric_keys = (
            "preflight_ram_available_gib",
            "preflight_vram_free_mib",
            "hard_stop_ram_available_gib",
            "hard_stop_vram_free_mib",
            "promotion_vram_free_mib",
            "hard_stop_owned_rss_gib",
            "hard_stop_pagefile_growth_gib",
            "occupied_prompt_ratio_min",
            "release_repeats",
        )
        for key in numeric_keys:
            value = resource.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"resource_policy.{key} must be positive")
        if isinstance(resource.get("promotion_vram_free_mib"), (int, float)) and isinstance(
            resource.get("hard_stop_vram_free_mib"), (int, float)
        ):
            if resource["promotion_vram_free_mib"] < resource["hard_stop_vram_free_mib"]:
                errors.append("promotion VRAM floor cannot be below the hard-stop floor")

    policy = manifest.get("evidence_policy", {})
    if isinstance(policy, dict):
        for key in (
            "target_before_mtp",
            "require_exact_mtp_text",
            "require_exact_mtp_tokens",
            "preserve_negative_results",
            "forbid_allocated_context_as_occupied",
        ):
            if policy.get(key) is not True:
                errors.append(f"evidence_policy.{key} must be true")

        visual = policy.get("visual_evidence")
        if not isinstance(visual, dict):
            errors.append("evidence_policy.visual_evidence must be an object")
        else:
            for key in (
                "required_for_promoted_performance",
                "screenshots_are_corroborating",
                "manifest_sha256_required",
            ):
                if visual.get(key) is not True:
                    errors.append(f"evidence_policy.visual_evidence.{key} must be true")
            expected_stages = ["preflight", "loaded", "steady", "result", "postflight"]
            if visual.get("required_stages") != expected_stages:
                errors.append("visual evidence stages must preserve the five-stage capture order")
            for key in ("protocol", "diagnostic_helper"):
                value = visual.get(key)
                if not isinstance(value, str) or not resolve_path(root, value).is_file():
                    errors.append(f"missing visual evidence {key}: {value!r}")

    if require_local_artifacts:
        artifacts = manifest.get("artifacts", {})
        runtime = manifest.get("runtime", {})
        try:
            target = artifacts["target"]
            draft = artifacts["draft"]
        except (KeyError, TypeError):
            errors.append("cannot validate local artifacts: malformed artifacts section")
        else:
            first_shard = resolve_path(root, target.get("local_first_shard", ""))
            target_manifest_path = resolve_path(root, target.get("manifest", ""))
            draft_path = resolve_path(root, draft.get("local_path", ""))
            server_path = resolve_path(root, runtime.get("server", ""))
            for label, path in (
                ("target first shard", first_shard),
                ("target manifest", target_manifest_path),
                ("draft sidecar", draft_path),
                ("server executable", server_path),
            ):
                if not path.is_file():
                    errors.append(f"missing local {label}: {path}")

            if target_manifest_path.is_file():
                try:
                    target_manifest = load_json(target_manifest_path)
                except ContractError as exc:
                    errors.append(str(exc))
                else:
                    files = target_manifest.get("files")
                    if not isinstance(files, list):
                        errors.append("target manifest has no files array")
                    else:
                        if len(files) != target.get("file_count"):
                            errors.append("target file count does not match benchmark manifest")
                        total = sum(item.get("bytes", 0) for item in files if isinstance(item, dict))
                        if total != target.get("bytes"):
                            errors.append("target aggregate bytes do not match benchmark manifest")
                        target_dir = first_shard.parent
                        for item in files:
                            if not isinstance(item, dict):
                                continue
                            path = target_dir / str(item.get("filename", ""))
                            if not path.is_file():
                                errors.append(f"missing target shard: {path}")
                            elif path.stat().st_size != item.get("bytes"):
                                errors.append(f"target shard size mismatch: {path.name}")

            if draft_path.is_file():
                if draft_path.stat().st_size != draft.get("bytes"):
                    errors.append("draft sidecar size mismatch")
                elif sha256_file(draft_path).lower() != str(draft.get("sha256", "")).lower():
                    errors.append("draft sidecar SHA-256 mismatch")
            if server_path.is_file():
                expected = str(runtime.get("server_sha256", "")).lower()
                if sha256_file(server_path).lower() != expected:
                    errors.append("server executable SHA-256 mismatch")

    runtime = manifest.get("runtime", {})
    if isinstance(runtime, dict):
        for tool in ("llama_bench", "llama_perplexity"):
            if runtime.get(tool) is None:
                warnings.append(f"runtime tool not configured: {tool}")

    return errors, warnings


def validate_result_row(
    row: dict[str, Any], manifest: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    line = row.get("_source_line", "?")
    prefix = f"row {line}"
    required = (
        "schema_version",
        "run_id",
        "timestamp_utc",
        "suite_id",
        "stage_id",
        "mode",
        "evidence_class",
        "status",
        "model",
        "runtime",
        "fixture",
        "settings",
        "metrics",
        "resources",
        "correctness",
        "artifacts",
        "contamination",
    )
    for key in required:
        if key not in row:
            errors.append(f"{prefix}: missing {key}")
    if row.get("schema_version") != RESULT_SCHEMA:
        errors.append(f"{prefix}: wrong result schema_version")
    if row.get("suite_id") != manifest.get("suite_id"):
        errors.append(f"{prefix}: wrong suite_id")

    stage_ids = {stage.get("id") for stage in manifest.get("stages", [])}
    if row.get("stage_id") not in stage_ids:
        errors.append(f"{prefix}: unknown stage_id {row.get('stage_id')!r}")
    if row.get("evidence_class") not in manifest.get("evidence_policy", {}).get("classes", []):
        errors.append(f"{prefix}: invalid evidence_class")
    if row.get("status") not in manifest.get("evidence_policy", {}).get("statuses", []):
        errors.append(f"{prefix}: invalid status")
    if row.get("mode") not in {"target", "mtp", "llama_bench", "perplexity", "quality", "integrity"}:
        errors.append(f"{prefix}: invalid mode")

    contamination = row.get("contamination")
    if isinstance(contamination, dict):
        void_reasons = contamination.get("void_reasons")
        if not isinstance(void_reasons, list):
            errors.append(f"{prefix}: contamination.void_reasons must be an array")
        elif row.get("status") == "VOID" and not void_reasons:
            errors.append(f"{prefix}: VOID row must state at least one void reason")

    if row.get("status") == "VALID" and row.get("mode") == "mtp":
        correctness = row.get("correctness")
        metrics = row.get("metrics")
        if not isinstance(correctness, dict):
            errors.append(f"{prefix}: VALID MTP row has no correctness object")
        else:
            if correctness.get("exact_text") is not True:
                errors.append(f"{prefix}: VALID MTP row must have exact_text=true")
            if correctness.get("exact_token_ids") is not True:
                errors.append(f"{prefix}: VALID MTP row must have exact_token_ids=true")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("drafted_tokens"), int) or metrics.get(
            "drafted_tokens", 0
        ) <= 0:
            errors.append(f"{prefix}: VALID MTP row must report drafted_tokens > 0")

    fixture = row.get("fixture")
    if row.get("status") == "VALID" and isinstance(fixture, dict):
        target_tokens = fixture.get("prompt_tokens_target")
        actual_tokens = fixture.get("actual_prompt_tokens")
        if isinstance(target_tokens, int) and target_tokens > 0:
            if not isinstance(actual_tokens, int):
                errors.append(f"{prefix}: occupied-context target requires actual_prompt_tokens")
            else:
                ratio = actual_tokens / target_tokens
                minimum = manifest.get("resource_policy", {}).get("occupied_prompt_ratio_min", 0.99)
                if ratio < minimum:
                    errors.append(f"{prefix}: prompt occupancy {ratio:.4f} is below {minimum:.4f}")
                correctness = row.get("correctness")
                if not isinstance(correctness, dict) or correctness.get("needle_pass") is not True:
                    errors.append(f"{prefix}: occupied-context VALID row requires needle_pass=true")

    if row.get("status") == "VALID":
        resources = row.get("resources")
        policy = manifest.get("resource_policy", {})
        if isinstance(resources, dict):
            vram = resources.get("vram_free_min_mib")
            if isinstance(vram, (int, float)) and vram < policy.get("hard_stop_vram_free_mib", 0):
                errors.append(f"{prefix}: VALID row crossed the VRAM hard-stop floor")
            ram = resources.get("ram_available_min_bytes")
            floor = policy.get("hard_stop_ram_available_gib", 0) * 1024**3
            if isinstance(ram, int) and ram < floor:
                errors.append(f"{prefix}: VALID row crossed the RAM hard-stop floor")
            pagefile = resources.get("pagefile_growth_bytes")
            pagefile_limit = policy.get("hard_stop_pagefile_growth_gib", 0) * 1024**3
            if isinstance(pagefile, int) and pagefile >= pagefile_limit:
                errors.append(f"{prefix}: VALID row crossed the pagefile-growth floor")
    return errors


def command_validate(args: argparse.Namespace) -> int:
    path = Path(args.manifest).resolve()
    try:
        manifest = load_json(path)
    except ContractError as exc:
        print(json.dumps({"status": "FAIL", "errors": [str(exc)]}, indent=2))
        return 1
    errors, warnings = validate_manifest(
        manifest, REPO_ROOT, require_local_artifacts=args.require_local_artifacts
    )
    output = {
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(path),
        "suite_id": manifest.get("suite_id"),
        "stage_count": len(manifest.get("stages", [])),
        "capability_task_count": len(manifest.get("capability_tasks", [])),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(output, indent=2))
    return 0 if not errors else 1


def command_plan(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(Path(args.manifest).resolve())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    errors, warnings = validate_manifest(manifest, REPO_ROOT)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    profile = manifest.get("profiles", {}).get(args.profile)
    if not isinstance(profile, dict):
        choices = ", ".join(sorted(manifest.get("profiles", {})))
        print(f"ERROR: unknown profile {args.profile!r}; choose one of {choices}", file=sys.stderr)
        return 1
    stage_map = {stage["id"]: stage for stage in manifest["stages"]}
    order, dependency_errors = _topological_order(stage_map, set(profile["stages"]))
    if dependency_errors:
        print(json.dumps({"status": "FAIL", "errors": dependency_errors}, indent=2))
        return 1
    plan = []
    for number, stage_id in enumerate(order, 1):
        stage = stage_map[stage_id]
        plan.append(
            {
                "order": number,
                "id": stage_id,
                "kind": stage.get("kind"),
                "gate_status": stage.get("gate_status"),
                "depends_on": stage.get("depends_on", []),
                "facets": stage.get("facets", []),
                "command": stage.get("command"),
            }
        )
    output = {
        "status": "PASS",
        "suite_id": manifest["suite_id"],
        "profile": args.profile,
        "description": profile.get("description"),
        "warnings": warnings,
        "plan": plan,
    }
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"{manifest['suite_id']} profile={args.profile}")
        print(profile.get("description", ""))
        for stage in plan:
            dependencies = ", ".join(stage["depends_on"]) or "none"
            print(f"\n{stage['order']:02d}. {stage['id']} [{stage['gate_status']}]")
            print(f"    depends: {dependencies}")
            print(f"    facets: {', '.join(stage['facets'])}")
            print(f"    action: {stage['command']}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
    return 0


def _results_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "status_counts": dict(sorted(Counter(row.get("status", "MISSING") for row in rows).items())),
        "evidence_counts": dict(
            sorted(Counter(row.get("evidence_class", "MISSING") for row in rows).items())
        ),
        "mode_counts": dict(sorted(Counter(row.get("mode", "MISSING") for row in rows).items())),
        "stage_counts": dict(sorted(Counter(row.get("stage_id", "MISSING") for row in rows).items())),
    }


def command_check_results(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(Path(args.manifest).resolve())
        rows = read_jsonl(Path(args.results).resolve())
    except ContractError as exc:
        print(json.dumps({"status": "FAIL", "errors": [str(exc)]}, indent=2))
        return 1
    manifest_errors, warnings = validate_manifest(manifest, REPO_ROOT)
    errors = list(manifest_errors)
    run_ids: set[str] = set()
    for row in rows:
        run_id = row.get("run_id")
        if isinstance(run_id, str):
            if run_id in run_ids:
                errors.append(f"duplicate run_id: {run_id}")
            run_ids.add(run_id)
        errors.extend(validate_result_row(row, manifest))
    output = {
        "status": "PASS" if not errors else "FAIL",
        "results": str(Path(args.results).resolve()),
        "summary": _results_summary(rows),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(output, indent=2))
    return 0 if not errors else 1


def command_summarize(args: argparse.Namespace) -> int:
    try:
        rows = read_jsonl(Path(args.results).resolve())
    except ContractError as exc:
        print(json.dumps({"status": "FAIL", "errors": [str(exc)]}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "summary": _results_summary(rows)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the suite contract")
    validate.add_argument("--manifest", default="benchmarks/manifest.json")
    validate.add_argument(
        "--require-local-artifacts",
        action="store_true",
        help="also verify local shard sizes plus draft/server hashes",
    )
    validate.set_defaults(func=command_validate)

    plan = subparsers.add_parser("plan", help="expand a profile in dependency order")
    plan.add_argument("--manifest", default="benchmarks/manifest.json")
    plan.add_argument("--profile", default="smoke")
    plan.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    plan.set_defaults(func=command_plan)

    check = subparsers.add_parser("check-results", help="validate append-only result rows")
    check.add_argument("--manifest", default="benchmarks/manifest.json")
    check.add_argument("--results", required=True)
    check.set_defaults(func=command_check_results)

    summarize = subparsers.add_parser("summarize", help="summarize result rows without promotion")
    summarize.add_argument("--results", required=True)
    summarize.set_defaults(func=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
