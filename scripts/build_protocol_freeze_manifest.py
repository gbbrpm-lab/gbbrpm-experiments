"""Build and verify the immutable GBBRPM v2.0.0 Protocol Freeze manifest.

Generates ``results/protocol_freeze/protocol_freeze_manifest_v2.0.0.json``
capturing:
- Held-out isolation status read directly from Step 4 development pilot and
  development controls summaries (``eval_seeds_evaluated == 0``).
- Linked repository URLs and ``main`` commit SHAs across ``gbbrpm-experiments``,
  ``gbbrpm-prototype``, and ``gbbrpm-manuscript``.
- Runtime environment versions and lockfile SHA-256.
- Frozen scenario inventory, seed manifest, solver/screening contract, and
  domain eligibility gates (Gate A / Gate B).
- SHA-256 digests of all tracked core modules, benchmark scripts, frozen public
  inputs, and Step 4 development pilot/controls artifacts.
- External SHA-256 digests ONLY (filename, byte size, SHA-256, and
  ``"redistribution": "restricted_not_in_git"``) for local restricted DPWH and
  Caltech artifacts when present on disk; never exposes restricted data.

Usage:
    python scripts/build_protocol_freeze_manifest.py
    python scripts/build_protocol_freeze_manifest.py --verify
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gbbrpm.generators import (
    BRANCH_B,
    BRANCH_CONDITIONS,
    BRANCH_ELL,
    BRANCH_M,
    BRANCH_P,
    DAG_B_SETTINGS,
    DAG_FAMILIES,
    DAG_SIZES,
    DAG_WEIGHTS,
    DEV_SEEDS,
    EVAL_SEEDS,
    EXPRESS_B_SETTINGS,
    EXPRESS_EXPECTED_EDGES,
    EXPRESS_EXPECTED_NODES,
    EXPRESS_EXPECTED_ROOTS,
    EXPRESS_WEIGHTS,
    HETERO_DEV_SEEDS,
    HETERO_EVAL_SEEDS,
    HETERO_WEIGHT_LEVELS,
)

FREEZE_TAG = "v2.0.0-protocol-freeze"
PROTOCOL_VERSION = "2.0.0"
MANIFEST_REL_PATH = "results/protocol_freeze/protocol_freeze_manifest_v2.0.0.json"
LOCKFILE_REL_PATH = "requirements-lock-v2.0.0.txt"

# Canonical linked repository commits recorded at protocol freeze time.
DEFAULT_EXPERIMENTS_BASE_SHA = "422dfdcfa38c18e168268c8f77da83263bc20c49"
DEFAULT_PROTOTYPE_MAIN_SHA = "e99adf4a382479a0e50df875a3b358f9f152c4e9"
DEFAULT_MANUSCRIPT_MAIN_SHA = "2aaf2e4c8b8e3a84fef11e40a6675a370f571a44"

EXTERNAL_RESTRICTED_REL_PATHS: tuple[str, ...] = (
    "data/drainage/dpwh_cdo_bitanag_kolambog/nodes.csv",
    "data/drainage/dpwh_cdo_bitanag_kolambog/edges.csv",
    "data/drainage/dpwh_cdo_bitanag_kolambog/manifest.json",
    "results/electrical/caltech_component2_private/manifest.json",
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using streaming 64 KiB reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev_parse(repo_dir: Path, ref: str) -> str | None:
    if not repo_dir.exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", ref],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def _package_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_tracked_files(repo_root: Path) -> list[tuple[str, str]]:
    """Return sorted list of (posix_rel_path, category) for frozen tracked files."""
    entries: list[tuple[str, str]] = []

    for py_path in sorted((repo_root / "gbbrpm").glob("*.py")):
        entries.append((py_path.relative_to(repo_root).as_posix(), "core_package"))

    for rel in (
        "model.py",
        "reconvergence.py",
        "scripts/run_dev_pilot.py",
        "scripts/build_protocol_freeze_manifest.py",
        LOCKFILE_REL_PATH,
    ):
        entries.append((rel, "core_and_benchmark_scripts"))

    for data_path in sorted((repo_root / "data" / "reconstructed").iterdir()):
        if data_path.is_file():
            entries.append((data_path.relative_to(repo_root).as_posix(), "frozen_public_input"))

    for rel in (
        "data/software/express_4.18.2/nodes.csv",
        "data/software/express_4.18.2/edges.csv",
        "data/software/express_4.18.2/manifest.json",
        "data/electrical/socal28_sample/manifest.json",
    ):
        entries.append((rel, "frozen_public_input"))

    for pilot_path in sorted((repo_root / "results" / "dev_pilot").iterdir()):
        if pilot_path.is_file():
            entries.append((pilot_path.relative_to(repo_root).as_posix(), "dev_pilot_artifact"))

    # Deduplicate while preserving deterministic sorted order by path
    dedup = {rel: cat for rel, cat in entries}
    return sorted(dedup.items(), key=lambda item: item[0])


def build_manifest(
    repo_root: Path = REPO_ROOT,
    base_experiments_sha: str | None = None,
    freeze_date_utc: str | None = None,
) -> dict[str, Any]:
    """Build the complete protocol freeze manifest dictionary."""
    pilot_summary_path = repo_root / "results" / "dev_pilot" / "dev_compute_pilot_summary.json"
    controls_summary_path = repo_root / "results" / "dev_pilot" / "dev_controls_summary.json"

    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    controls_summary = json.loads(controls_summary_path.read_text(encoding="utf-8"))

    pilot_eval_seeds = int(pilot_summary["protocol"]["eval_seeds_evaluated"])
    controls_eval_seeds = int(controls_summary["protocol"]["eval_seeds_evaluated"])
    if pilot_eval_seeds != 0 or controls_eval_seeds != 0:
        raise RuntimeError(
            f"Held-out isolation check failed: pilot={pilot_eval_seeds}, controls={controls_eval_seeds}"
        )

    extrap = pilot_summary["full_inventory_extrapolation"]
    feasible = bool(extrap["feasible_without_reduction"])
    if not feasible:
        raise RuntimeError("Dev compute pilot is not feasible without reduction.")

    single_core_min_exact = float(extrap["single_core_sec"]["total_minutes"])
    parallel_8w_min_exact = float(extrap["parallel_estimate"]["estimated_wall_minutes"])
    global_assertions = controls_summary["global_assertions"]

    held_out_isolation = {
        "eval_seeds_evaluated": 0,
        "eval_outcomes_inspected": False,
        "dev_compute_pilot_feasible_without_reduction": feasible,
        "extrapolated_single_core_minutes": round(single_core_min_exact, 2),
        "extrapolated_single_core_minutes_exact": single_core_min_exact,
        "extrapolated_8worker_minutes": round(parallel_8w_min_exact, 2),
        "extrapolated_8worker_minutes_exact": parallel_8w_min_exact,
        "dev_pilot_configs_timed": int(pilot_summary["protocol"]["n_pilot_configs"]),
        "dev_controls_total_verified": int(global_assertions["total_scenarios_verified"]),
        "dev_controls_all_hard_assertions_passed": bool(
            global_assertions["all_hard_assertions_passed"]
        ),
        "dev_controls_bracket_violations_total": int(
            global_assertions["bracket_violations_total"]
        ),
        "dev_controls_false_certifications_total": int(
            global_assertions["false_certifications_total"]
        ),
        "dev_controls_empty_hoeffding_intersections_total_nodes": int(
            global_assertions["empty_hoeffding_intersections_total_nodes"]
        ),
        "source_summaries": [
            "results/dev_pilot/dev_compute_pilot_summary.json",
            "results/dev_pilot/dev_controls_summary.json",
        ],
    }

    sibling_root = repo_root.parent
    exp_sha = base_experiments_sha or _git_rev_parse(repo_root, "main") or DEFAULT_EXPERIMENTS_BASE_SHA
    proto_sha = _git_rev_parse(sibling_root / "gbbrpm-prototype", "main") or DEFAULT_PROTOTYPE_MAIN_SHA
    ms_sha = _git_rev_parse(sibling_root / "GBBRPM_Manuscript", "main") or DEFAULT_MANUSCRIPT_MAIN_SHA

    linked_repositories = {
        "gbbrpm-experiments": {
            "url": "https://github.com/gbbrpm-lab/gbbrpm-experiments",
            "branch": "main",
            "base_pre_freeze_commit_sha": exp_sha,
            "step4_merged_pr": "https://github.com/gbbrpm-lab/gbbrpm-experiments/pull/18",
        },
        "gbbrpm-prototype": {
            "url": "https://github.com/gbbrpm-lab/gbbrpm-prototype",
            "branch": "main",
            "commit_sha": proto_sha,
            "step4_merged_pr": "https://github.com/gbbrpm-lab/gbbrpm-prototype/pull/7",
        },
        "gbbrpm-manuscript": {
            "url": "https://github.com/gbbrpm-lab/gbbrpm-manuscript",
            "branch": "main",
            "commit_sha": ms_sha,
        },
    }

    lockfile_path = repo_root / LOCKFILE_REL_PATH
    if not lockfile_path.is_file():
        raise FileNotFoundError(f"Missing lockfile: {lockfile_path}")

    environment = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": int(os.cpu_count() or 1),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
            "pytest": pytest.__version__,
            "matplotlib": matplotlib.__version__,
            "swmm_toolkit": _package_version("swmm-toolkit"),
        },
        "requirements_lock_file": LOCKFILE_REL_PATH,
        "requirements_lock_sha256": sha256_file(lockfile_path),
    }

    matched_branch_total = (
        len(BRANCH_M) * len(BRANCH_ELL) * len(BRANCH_P) * len(BRANCH_B) * len(BRANCH_CONDITIONS)
    )
    dag_uniform_dev = len(DEV_SEEDS) * len(DAG_FAMILIES) * len(DAG_SIZES) * len(DAG_WEIGHTS) * len(DAG_B_SETTINGS)
    dag_uniform_eval = len(EVAL_SEEDS) * len(DAG_FAMILIES) * len(DAG_SIZES) * len(DAG_WEIGHTS) * len(DAG_B_SETTINGS)
    dag_uniform_total = dag_uniform_dev + dag_uniform_eval
    dag_hetero_dev = len(HETERO_DEV_SEEDS) * len(DAG_FAMILIES) * len(DAG_SIZES) * len(DAG_B_SETTINGS)
    dag_hetero_eval = len(HETERO_EVAL_SEEDS) * len(DAG_FAMILIES) * len(DAG_SIZES) * len(DAG_B_SETTINGS)
    dag_hetero_total = dag_hetero_dev + dag_hetero_eval
    express_total = len(EXPRESS_WEIGHTS) * len(EXPRESS_B_SETTINGS)
    primary_total = matched_branch_total + dag_uniform_total + dag_hetero_total + express_total
    precision_audit_weights = [0.05, 0.10]
    precision_audits_total = len(DAG_FAMILIES) * len(DAG_SIZES) * len(precision_audit_weights) * len(DAG_B_SETTINGS)

    scenario_inventory = {
        "matched_branch_controls": {
            "total": matched_branch_total,
            "shared": matched_branch_total // 2,
            "independent": matched_branch_total // 2,
            "reference_formula": "Eq. (3-15) exact closed-form",
            "m": list(BRANCH_M),
            "ell": list(BRANCH_ELL),
            "p": list(BRANCH_P),
            "b": list(BRANCH_B),
            "conditions": list(BRANCH_CONDITIONS),
        },
        "uniform_generated_dags": {
            "total": dag_uniform_total,
            "dev_total": dag_uniform_dev,
            "dev_seeds": list(DEV_SEEDS),
            "eval_total": dag_uniform_eval,
            "eval_seeds": list(EVAL_SEEDS),
            "families": list(DAG_FAMILIES),
            "sizes": list(DAG_SIZES),
            "weights": list(DAG_WEIGHTS),
            "b_settings": list(DAG_B_SETTINGS),
        },
        "heterogeneous_weight_dags": {
            "total": dag_hetero_total,
            "dev_total": dag_hetero_dev,
            "dev_seeds": list(HETERO_DEV_SEEDS),
            "eval_total": dag_hetero_eval,
            "eval_seeds": list(HETERO_EVAL_SEEDS),
            "families": list(DAG_FAMILIES),
            "sizes": list(DAG_SIZES),
            "b_settings": list(DAG_B_SETTINGS),
            "edge_ordering_rule": "UTF-8 SHA-256 of family|size|seed|source|destination|hetero-v1",
            "weight_levels": list(HETERO_WEIGHT_LEVELS),
            "weight_quarters": {
                "bottom_quarter": 0.10,
                "middle_half": 0.20,
                "top_quarter": 0.30,
            },
            "exact_mean_weight": 0.20,
        },
        "express_4_18_2_stochastic": {
            "total": express_total,
            "nodes": EXPRESS_EXPECTED_NODES,
            "edges": EXPRESS_EXPECTED_EDGES,
            "advisory_roots": EXPRESS_EXPECTED_ROOTS,
            "weights": list(EXPRESS_WEIGHTS),
            "b_settings": list(EXPRESS_B_SETTINGS),
        },
        "precision_audits": {
            "total": precision_audits_total,
            "mc_trials": 1_000_000,
            "eval_seed": 1001,
            "families": list(DAG_FAMILIES),
            "sizes": list(DAG_SIZES),
            "weights": precision_audit_weights,
            "b_settings": list(DAG_B_SETTINGS),
        },
        "totals": {
            "primary_scenarios": primary_total,
            "precision_audits": precision_audits_total,
            "dev_controls_evaluated": matched_branch_total + dag_uniform_dev + dag_hetero_dev,
            "held_out_eval_scenarios": dag_uniform_eval + dag_hetero_eval,
            "total_edge_trials_billions": float(extrap["inventory_counts"]["total_edge_trials_billions"]),
        },
    }

    seed_manifest = {
        "dev_seeds": list(DEV_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "hetero_dev_seeds": list(HETERO_DEV_SEEDS),
        "hetero_eval_seeds": list(HETERO_EVAL_SEEDS),
        "precision_audit_seed": 1001,
        "robustness_seed": 41,
        "tie_permutation_seed": 140,
        "tie_permutation_n_perms": 100,
        "mc_seed_rule": "derive_scenario_seed: big-endian uint32 from first 4 bytes of SHA-256(scenario_id.encode('utf-8'))",
        "mc_trials_primary": 100_000,
        "mc_trials_precision_audit": 1_000_000,
        "mc_batch_size": 10_000,
        "mc_backend": str(pilot_summary["protocol"]["mc_backend"]),
        "hoeffding_alpha": 0.05,
        "tolerance_epsilon": 1e-12,
        "candidate_set": "B_j == 0",
        "budget_fractions": [0.10, 0.20, 0.30],
        "empty_intersection_policy": "record_without_repair",
    }

    domain_eligibility_gates = {
        "urban_drainage": {
            "case_id": "drainage_dpwh_cdo_bitanag_kolambog_v1",
            "gate_a_structural_readiness": "PASS",
            "gate_a_details": (
                "11 nodes, 11 edges, valid DAG, S_ij in [0.4639, 0.8471], "
                "Manning verification <1% error, J_AGORA_CONF reconvergence junction, Tier-2 B_i"
            ),
            "gate_b_independent_outcome_validation": "FAIL",
            "gate_b_details": (
                "June 2025 DEO flooding hotspots are exploratory only; "
                "no independent field-flooding validation claim."
            ),
            "redistribution": "restricted_not_in_git",
        },
        "software": {
            "case_id": "software_npm_express_4.18.2",
            "gate_a_structural_readiness": "PASS",
            "gate_a_details": (
                "71 exact package-version nodes, 128 reversed dependency edges, "
                "1 weakly connected DAG, 7 OSV advisory roots (13 advisory matches)"
            ),
            "gate_b_independent_outcome_validation": "N/A",
            "gate_b_details": (
                "Cross-domain applicability and stochastic structural benchmark; "
                "OSV supplies local-disturbance inputs rather than independent outcomes."
            ),
            "redistribution": "public_in_git",
        },
        "electrical": {
            "case_id": "electrical_caltech_socal28_component2",
            "gate_a_structural_readiness": "PASS",
            "gate_a_details": (
                "12-node arborescence (Component 2), 24 phasor snapshots, "
                "43-scenario measured-L/C suite"
            ),
            "gate_b_independent_outcome_validation": "FAIL",
            "gate_b_details": (
                "Telemetry covers a normal-operation window after the Nov 13, 2024 "
                "switching event; no fault-prediction claim."
            ),
            "redistribution": "restricted_not_in_git",
        },
    }

    tracked_file_hashes: dict[str, dict[str, Any]] = {}
    for rel_path, category in _collect_tracked_files(repo_root):
        abs_path = repo_root / rel_path
        if not abs_path.is_file():
            raise FileNotFoundError(f"Expected tracked file missing on disk: {rel_path}")
        tracked_file_hashes[rel_path] = {
            "category": category,
            "bytes": abs_path.stat().st_size,
            "sha256": sha256_file(abs_path),
        }

    # Record ONLY filename, byte size, SHA-256, and redistribution flag for restricted local files.
    external_restricted_hashes: dict[str, dict[str, Any]] = {}
    for rel_path in EXTERNAL_RESTRICTED_REL_PATHS:
        abs_path = repo_root / rel_path
        if abs_path.is_file():
            external_restricted_hashes[rel_path] = {
                "filename": abs_path.name,
                "bytes": abs_path.stat().st_size,
                "sha256": sha256_file(abs_path),
                "redistribution": "restricted_not_in_git",
            }

    manifest = {
        "freeze_tag": FREEZE_TAG,
        "freeze_date_utc": freeze_date_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_version": PROTOCOL_VERSION,
        "held_out_isolation": held_out_isolation,
        "linked_repositories": linked_repositories,
        "environment": environment,
        "scenario_inventory": scenario_inventory,
        "seed_manifest": seed_manifest,
        "domain_eligibility_gates": domain_eligibility_gates,
        "tracked_file_hashes": tracked_file_hashes,
        "external_restricted_hashes": external_restricted_hashes,
    }
    return manifest


def verify_manifest(manifest_path: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """Verify every tracked SHA-256 hash and protocol invariant in the manifest.

    Returns a list of error strings (empty list means 100% PASS).
    """
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"Manifest file not found: {manifest_path}"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("freeze_tag") != FREEZE_TAG:
        errors.append(f"Unexpected freeze_tag: {manifest.get('freeze_tag')!r}")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(f"Unexpected protocol_version: {manifest.get('protocol_version')!r}")

    iso = manifest.get("held_out_isolation", {})
    if iso.get("eval_seeds_evaluated") != 0:
        errors.append(f"eval_seeds_evaluated must be 0, got {iso.get('eval_seeds_evaluated')!r}")
    if iso.get("eval_outcomes_inspected") is not False:
        errors.append(f"eval_outcomes_inspected must be False, got {iso.get('eval_outcomes_inspected')!r}")
    if iso.get("dev_compute_pilot_feasible_without_reduction") is not True:
        errors.append("dev_compute_pilot_feasible_without_reduction must be True")

    tracked = manifest.get("tracked_file_hashes", {})
    if not tracked:
        errors.append("tracked_file_hashes is empty")
    for rel_path, meta in tracked.items():
        abs_path = repo_root / rel_path
        if not abs_path.is_file():
            errors.append(f"Missing tracked file: {rel_path}")
            continue
        actual_sha = sha256_file(abs_path)
        actual_bytes = abs_path.stat().st_size
        if actual_sha != meta.get("sha256"):
            errors.append(
                f"SHA-256 mismatch for {rel_path}: expected {meta.get('sha256')}, got {actual_sha}"
            )
        if actual_bytes != meta.get("bytes"):
            errors.append(
                f"Byte size mismatch for {rel_path}: expected {meta.get('bytes')}, got {actual_bytes}"
            )

    lock_meta = tracked.get(LOCKFILE_REL_PATH, {})
    if manifest.get("environment", {}).get("requirements_lock_sha256") != lock_meta.get("sha256"):
        errors.append("environment.requirements_lock_sha256 does not match tracked_file_hashes entry")

    # Verify external restricted hashes if present on disk; always verify metadata safety.
    for rel_path, meta in manifest.get("external_restricted_hashes", {}).items():
        if meta.get("redistribution") != "restricted_not_in_git":
            errors.append(f"External file {rel_path} missing restricted_not_in_git redistribution tag")
        allowed_keys = {"filename", "bytes", "sha256", "redistribution"}
        extra_keys = set(meta.keys()) - allowed_keys
        if extra_keys:
            errors.append(f"External file {rel_path} contains disallowed keys: {sorted(extra_keys)}")
        abs_path = repo_root / rel_path
        if abs_path.is_file():
            actual_sha = sha256_file(abs_path)
            if actual_sha != meta.get("sha256"):
                errors.append(
                    f"External SHA-256 mismatch for {rel_path}: expected {meta.get('sha256')}, got {actual_sha}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify the GBBRPM v2.0.0 protocol freeze manifest.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the existing manifest on disk without overwriting it.",
    )
    parser.add_argument(
        "--base-experiments-sha",
        type=str,
        default=None,
        help="Override base pre-freeze commit SHA for gbbrpm-experiments.",
    )
    args = parser.parse_args()

    manifest_path = REPO_ROOT / MANIFEST_REL_PATH
    if args.verify:
        errors = verify_manifest(manifest_path, REPO_ROOT)
        if errors:
            for err in errors:
                print(f"[FAIL] {err}", file=sys.stderr)
            return 1
        print(
            f"[PASS] Verified {MANIFEST_REL_PATH} "
            f"(sha256={sha256_file(manifest_path)})"
        )
        return 0

    manifest = build_manifest(
        repo_root=REPO_ROOT,
        base_experiments_sha=args.base_experiments_sha,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")

    errors = verify_manifest(manifest_path, REPO_ROOT)
    if errors:
        for err in errors:
            print(f"[FAIL] {err}", file=sys.stderr)
        return 1

    print(
        f"[OK] Wrote and verified {MANIFEST_REL_PATH} "
        f"({len(manifest['tracked_file_hashes'])} tracked files, "
        f"{len(manifest['external_restricted_hashes'])} external restricted digests, "
        f"manifest_sha256={sha256_file(manifest_path)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
