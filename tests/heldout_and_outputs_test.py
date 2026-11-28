"""Comprehensive verification suite for Step 6 held-out evaluation and dual-target v2 outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_protocol_freeze_manifest import MANIFEST_REL_PATH, verify_manifest

DEV_PILOT_DIR = REPO_ROOT / "results" / "dev_pilot"
HELDOUT_DIR = REPO_ROOT / "results" / "heldout_eval"
PAPER_V2_DIR = REPO_ROOT / "results" / "paper_v2"
PAPER_V2_MANIFEST = PAPER_V2_DIR / "paper_v2_manifest.json"
FREEZE_MANIFEST = REPO_ROOT / MANIFEST_REL_PATH

EXPECTED_V2_FIGURE_STEMS = {
    "fig_v2_matched_branch_controls",
    "fig_v2_bracket_tightness",
    "fig_v2_fixed_budget_ranking",
    "fig_v2_certified_screening",
    "fig_v2_w_self_diagnosis",
    "fig_v2_hetero_and_precision_audit",
    "fig_v2_scalability_sparse_dense",
    "fig_v2_domain_synthesis",
}

EXPECTED_V2_TABLES = {
    "table_v2_inventory_and_assertions.tex",
    "table_v2_bracket_tightness.tex",
    "table_v2_comparator_ranking.tex",
    "table_v2_certified_screening.tex",
    "table_v2_hetero_and_audits.tex",
    "table_v2_scalability.tex",
    "table_v2_domain_cases.tex",
}

FORBIDDEN_GIT_PATTERNS: tuple[str, ...] = (
    "data/drainage/",
    "results/drainage/",
    "data_caltech",
    "caltech_component2_private",
    "dataset_api_credentials.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_heldout_csv_row_counts_and_seed_coverage() -> None:
    uniform_df = pd.read_csv(HELDOUT_DIR / "heldout_uniform_2880.csv")
    hetero_df = pd.read_csv(HELDOUT_DIR / "heldout_hetero_120.csv")
    express_df = pd.read_csv(HELDOUT_DIR / "heldout_express_12.csv")
    audit_df = pd.read_csv(HELDOUT_DIR / "heldout_precision_audits_48.csv")
    scal_df = pd.read_csv(HELDOUT_DIR / "scalability_v2.csv")
    dom_express_df = pd.read_csv(HELDOUT_DIR / "domain_express_v2.csv")

    assert len(uniform_df) == 2880
    assert len(hetero_df) == 120
    assert len(express_df) == 12
    assert len(audit_df) == 48
    assert len(scal_df) >= 100
    assert len(dom_express_df) == 5

    assert sorted(uniform_df["graph_seed"].unique().tolist()) == list(range(1001, 1021))
    assert sorted(hetero_df["graph_seed"].unique().tolist()) == list(range(1001, 1006))
    assert sorted(audit_df["graph_seed"].unique().tolist()) == [1001]
    assert sorted(audit_df["weight_value"].unique().tolist()) == [0.05, 0.1]
    assert set(scal_df["series"].unique()) == {"sparse", "dense", "paired_mc_reference"}


def test_heldout_zero_violations_and_zero_false_certifications() -> None:
    summary = json.loads((HELDOUT_DIR / "heldout_eval_summary.json").read_text(encoding="utf-8"))
    ga = summary["global_assertions"]

    assert ga["all_hard_assertions_passed"] is True
    assert ga["bracket_violations_total"] == 0
    assert ga["false_certifications_total"] == 0
    assert ga["false_certifications_analytic_total"] == 0
    assert ga["false_certifications_interval_total"] == 0
    assert ga["total_evaluations_verified"] == 3060
    assert summary["protocol"]["empty_intersection_policy"] == "record_without_repair"

    for suite_key in ("heldout_uniform_2880", "heldout_hetero_120", "heldout_express_12"):
        assert summary[suite_key]["bracket_violations"] == 0
        assert summary[suite_key]["false_certifications_analytic"] == 0
        assert summary[suite_key]["false_certifications_interval"] == 0

    pa = summary["precision_audits_48"]
    assert pa["bracket_violations"] == 0
    assert pa["false_certifications_100k"] == 0
    assert pa["false_certifications_1m"] == 0
    assert pa["suite1_100k_determinism_parity_verified"] is True


def test_domain_cases_v2_summary_assertions_and_gates() -> None:
    dsum = json.loads((HELDOUT_DIR / "domain_cases_v2_summary.json").read_text(encoding="utf-8"))
    ga = dsum["global_assertions"]

    assert ga["all_input_hashes_verified"] is True
    assert ga["bracket_violations_total"] == 0
    assert ga["abbe_violations_total"] == 0
    assert ga["false_certifications_total"] == 0
    assert ga["express_root_tau1_exact_match"] is True
    assert ga["caltech_tree_equivalence_verified"] is True

    ud = dsum["urban_drainage"]
    assert ud["gate_a_structural_readiness"] == "PASS"
    assert ud["gate_b_independent_outcome_validation"] == "FAIL"
    assert ud["graph_summary"]["reconvergent_junction"] == "J_AGORA_CONF"
    assert ud["j_agora_conf_reconvergence"]["exact_max_R_minus_pi"] > 0.05

    sw = dsum["software_dependency"]
    assert sw["gate_a_structural_readiness"] == "PASS"
    assert sw["gate_b_independent_outcome_validation"] == "N/A"
    assert abs(sw["verified_root_risk_tau1"] - 0.997528076171875) < 1e-15

    el = dsum["electrical_distribution"]
    assert el["gate_a_structural_readiness"] == "PASS"
    assert el["gate_b_independent_outcome_validation"] == "FAIL"
    assert el["graph_summary"]["is_arborescence"] is True
    assert el["tree_equivalence_verification"]["scenario_count"] == 43
    assert el["tree_equivalence_verification"]["max_abs_R_minus_pi"] < 1e-12


def test_paper_v2_manifest_full_sha256_parity_and_inventory() -> None:
    assert PAPER_V2_MANIFEST.is_file(), f"Missing manifest: {PAPER_V2_MANIFEST}"
    manifest = json.loads(PAPER_V2_MANIFEST.read_text(encoding="utf-8"))

    # Protocol freeze manifest hash matches disk
    pf_entry = manifest["protocol_freeze_manifest"]
    pf_path = REPO_ROOT / pf_entry["path"]
    assert pf_path.resolve() == FREEZE_MANIFEST.resolve()
    assert _sha256(pf_path) == pf_entry["sha256"]

    # Every file in results/dev_pilot/ and results/heldout_eval/ is hashed and matches disk
    input_hashes = manifest["input_sha256"]
    expected_inputs = {FREEZE_MANIFEST.relative_to(REPO_ROOT).as_posix()}
    for folder in (DEV_PILOT_DIR, HELDOUT_DIR):
        for p in folder.iterdir():
            if p.is_file():
                expected_inputs.add(p.relative_to(REPO_ROOT).as_posix())
    assert expected_inputs.issubset(set(input_hashes.keys()))

    for rel_path, expected_hash in input_hashes.items():
        abs_path = REPO_ROOT / rel_path
        assert abs_path.is_file(), f"Missing hashed input: {rel_path}"
        assert _sha256(abs_path) == expected_hash, f"Input SHA-256 mismatch: {rel_path}"

    # Verify both targets: 8 stems x 2 formats = 16 figures, 7 tables each
    assert manifest["counts"]["thesis_figures"] == 16
    assert manifest["counts"]["thesis_tables"] == 7
    assert manifest["counts"]["elsevier_paper1_figures"] == 16
    assert manifest["counts"]["elsevier_paper1_tables"] == 7
    assert manifest["counts"]["total_figures"] == 32
    assert manifest["counts"]["total_tables"] == 14

    for target_name in ("thesis", "elsevier_paper1"):
        t_data = manifest["targets"][target_name]
        figs = t_data["figures"]
        tabs = t_data["tables"]
        assert len(figs) == 16
        assert len(tabs) == 7
        assert {item["stem"] for item in figs} == EXPECTED_V2_FIGURE_STEMS
        assert {item["format"] for item in figs} == {"pdf", "png"}
        assert {item["name"] for item in tabs} == EXPECTED_V2_TABLES

        for item in figs + tabs:
            abs_path = REPO_ROOT / item["path"]
            assert abs_path.is_file() and abs_path.stat().st_size > 0
            assert abs_path.stat().st_size == item["bytes"]
            assert _sha256(abs_path) == item["sha256"]


def test_v2_latex_tables_booktabs_and_target_environments() -> None:
    for target_name in ("thesis", "elsevier_paper1"):
        tab_dir = PAPER_V2_DIR / target_name / "tables"
        for name in EXPECTED_V2_TABLES:
            path = tab_dir / name
            assert path.is_file(), f"Missing table {path}"
            text = path.read_text(encoding="utf-8")
            assert "\\toprule" in text
            assert "\\midrule" in text
            assert "\\bottomrule" in text
            assert "\\caption{" in text
            assert "\\label{" in text
            assert "Auto-generated" in text
            if target_name == "thesis":
                assert "\\begin{table}" in text
                assert "\\begin{table*}" not in text
            else:
                assert "\\begin{table*}" in text


def test_zero_restricted_files_in_git_and_protocol_freeze_intact() -> None:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    tracked = proc.stdout.splitlines()
    violations = [
        path for path in tracked if any(pat in path for pat in FORBIDDEN_GIT_PATTERNS)
    ]
    assert violations == [], f"Restricted files tracked in git: {violations}"

    errors = verify_manifest(FREEZE_MANIFEST, REPO_ROOT)
    assert errors == [], f"Protocol freeze immutability broken: {errors}"
