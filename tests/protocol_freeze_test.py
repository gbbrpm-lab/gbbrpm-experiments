"""Automated integrity and privacy verification for GBBRPM v2.0.0 Protocol Freeze."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gbbrpm
from scripts.build_protocol_freeze_manifest import (
    LOCKFILE_REL_PATH,
    MANIFEST_REL_PATH,
    sha256_file,
    verify_manifest,
)

MANIFEST_PATH = REPO_ROOT / MANIFEST_REL_PATH

FORBIDDEN_GIT_PATTERNS: tuple[str, ...] = (
    "data/drainage/",
    "results/drainage/",
    "data_caltech",
    "caltech_component2_private",
    "dataset_api_credentials.json",
)


def _load_manifest() -> dict:
    assert MANIFEST_PATH.is_file(), f"Missing protocol freeze manifest: {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_parses_and_every_tracked_hash_matches_disk() -> None:
    manifest = _load_manifest()
    assert manifest["freeze_tag"] == "v2.0.0-protocol-freeze"
    assert manifest["protocol_version"] == "2.0.0"

    errors = verify_manifest(MANIFEST_PATH, REPO_ROOT)
    assert errors == [], f"Protocol freeze manifest verification failed: {errors}"

    tracked = manifest["tracked_file_hashes"]
    assert len(tracked) >= 30, f"Unexpectedly few tracked files hashed: {len(tracked)}"
    for rel_path, meta in tracked.items():
        abs_path = REPO_ROOT / rel_path
        assert abs_path.is_file(), f"Tracked file missing on disk: {rel_path}"
        assert sha256_file(abs_path) == meta["sha256"], f"SHA-256 mismatch on {rel_path}"
        assert abs_path.stat().st_size == meta["bytes"], f"Byte size mismatch on {rel_path}"

    assert manifest["environment"]["requirements_lock_file"] == LOCKFILE_REL_PATH
    assert manifest["environment"]["requirements_lock_sha256"] == tracked[LOCKFILE_REL_PATH]["sha256"]


def test_held_out_isolation_and_dev_pilot_signoff() -> None:
    manifest = _load_manifest()
    iso = manifest["held_out_isolation"]
    assert iso["eval_seeds_evaluated"] == 0
    assert iso["eval_outcomes_inspected"] is False
    assert iso["dev_compute_pilot_feasible_without_reduction"] is True
    assert iso["dev_controls_all_hard_assertions_passed"] is True
    assert iso["dev_controls_bracket_violations_total"] == 0
    assert iso["dev_controls_false_certifications_total"] == 0
    assert iso["dev_controls_total_verified"] == 1304

    pilot_summary = json.loads(
        (REPO_ROOT / "results" / "dev_pilot" / "dev_compute_pilot_summary.json").read_text(
            encoding="utf-8"
        )
    )
    controls_summary = json.loads(
        (REPO_ROOT / "results" / "dev_pilot" / "dev_controls_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert pilot_summary["protocol"]["eval_seeds_evaluated"] == 0
    assert controls_summary["protocol"]["eval_seeds_evaluated"] == 0
    assert pilot_summary["full_inventory_extrapolation"]["feasible_without_reduction"] is True


def test_scenario_inventory_and_seed_contract() -> None:
    manifest = _load_manifest()
    inv = manifest["scenario_inventory"]
    seeds = manifest["seed_manifest"]

    assert inv["matched_branch_controls"]["total"] == 560
    assert inv["uniform_generated_dags"]["dev_total"] == 720
    assert inv["uniform_generated_dags"]["eval_total"] == 2880
    assert inv["uniform_generated_dags"]["total"] == 3600
    assert inv["heterogeneous_weight_dags"]["dev_total"] == 24
    assert inv["heterogeneous_weight_dags"]["eval_total"] == 120
    assert inv["heterogeneous_weight_dags"]["total"] == 144
    assert inv["express_4_18_2_stochastic"]["total"] == 12
    assert inv["precision_audits"]["total"] == 48
    assert inv["totals"]["primary_scenarios"] == 4316
    assert inv["totals"]["precision_audits"] == 48
    assert (
        inv["matched_branch_controls"]["total"]
        + inv["uniform_generated_dags"]["total"]
        + inv["heterogeneous_weight_dags"]["total"]
        + inv["express_4_18_2_stochastic"]["total"]
        == 4316
    )

    assert seeds["dev_seeds"] == [41, 42, 43, 44, 45]
    assert seeds["eval_seeds"] == list(range(1001, 1021))
    assert seeds["robustness_seed"] == 41
    assert seeds["tie_permutation_seed"] == 140
    assert seeds["tie_permutation_n_perms"] == 100
    assert seeds["mc_trials_primary"] == 100_000
    assert seeds["mc_trials_precision_audit"] == 1_000_000
    assert seeds["mc_batch_size"] == 10_000
    assert seeds["mc_backend"] == "numpy_uint64_bitpacked"
    assert seeds["hoeffding_alpha"] == 0.05
    assert seeds["tolerance_epsilon"] == 1e-12
    assert seeds["candidate_set"] == "B_j == 0"
    assert seeds["budget_fractions"] == [0.10, 0.20, 0.30]
    assert seeds["empty_intersection_policy"] == "record_without_repair"


def test_domain_eligibility_gates_and_restricted_hash_safety() -> None:
    manifest = _load_manifest()
    gates = manifest["domain_eligibility_gates"]
    assert gates["urban_drainage"]["gate_a_structural_readiness"] == "PASS"
    assert gates["urban_drainage"]["gate_b_independent_outcome_validation"] == "FAIL"
    assert gates["software"]["gate_a_structural_readiness"] == "PASS"
    assert gates["software"]["gate_b_independent_outcome_validation"] == "N/A"
    assert gates["electrical"]["gate_a_structural_readiness"] == "PASS"
    assert gates["electrical"]["gate_b_independent_outcome_validation"] == "FAIL"

    for rel_path, meta in manifest["external_restricted_hashes"].items():
        assert meta["redistribution"] == "restricted_not_in_git", f"Bad tag for {rel_path}"
        assert set(meta.keys()) == {"filename", "bytes", "sha256", "redistribution"}


def test_zero_restricted_files_tracked_in_git() -> None:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    tracked_files = proc.stdout.splitlines()
    violations = [
        path
        for path in tracked_files
        if any(pattern in path for pattern in FORBIDDEN_GIT_PATTERNS)
    ]
    assert violations == [], f"Restricted files found in git ls-files: {violations}"


def test_public_package_api_version_preserved() -> None:
    assert gbbrpm.__version__ == "0.1.0"
    assert set(gbbrpm.__all__) == {"derive_susceptibility", "evaluate_gbbrpm"}
