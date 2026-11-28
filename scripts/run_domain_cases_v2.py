#!/usr/bin/env python3
"""Step 6 Task B: Eligible Real-Domain Instantiations (DPWH Drainage, Express 4.18.2, Caltech Comp. 2).

Executes rigorous v2.0 methodological evaluations across all three Gate-A-verified
domain instantiations while strictly enforcing data-governance & privacy boundaries:
1. Urban Drainage (`drainage_dpwh_cdo_bitanag_kolambog_v1`, Gate A PASS / Gate B FAIL):
   - SHA-256 verification against `results/protocol_freeze/protocol_freeze_manifest_v2.0.0.json`.
   - Controlled Tier-2 blockage scenarios (single-source upstream & Agora split,
     multi-source pairs & all-headwaters, uniform-baseline tau stress sweeps).
   - Exact Eq. (3-10) reference marginals (`evaluate_exact_marginals`) wherever
     non-degenerate primitives <= 16 (fallback to `run_mc_reference` N=100,000 only
     if > 16), all matched comparators (`M_j`, `T_j`, `H_j`, `abbe_lower`,
     `local_upstream`, `ancestry_diagnostic`), and Two-Stage Certified Screening.
   - Detailed node/scenario outputs written STRICTLY to gitignored
     `results/drainage/dpwh_private/`.
2. Software Dependency (`software_npm_express_4.18.2`, Gate A PASS / Gate B N/A):
   - SHA-256 verification against frozen manifest.
   - Full `tau in {0.0, 0.25, 0.50, 0.75, 1.0}` sweep with all comparators,
     `mc_ref` (`N=100,000`), Two-Stage Certified Screening, explicit verification
     of Express root risk `0.997528076171875` at `tau=1.0`, and shared-path
     reconvergence quantification along `qs` and `send`.
   - Public summary table saved to `results/heldout_eval/domain_express_v2.csv`.
3. Electrical Distribution (`electrical_caltech_socal28_component2`, Gate A PASS / Gate B FAIL):
   - SHA-256 verification of private `manifest.json` against frozen manifest.
   - Arborescence confirmation (`nx.is_arborescence`, zero reconvergence) and full
     43-scenario exact tree-equivalence verification (`evaluate_exact_marginals` vs
     `GBBRPM`, verifying `|R_j - pi_j| < 1e-12`, `|T_j - R_j| < 1e-12`, and
     `M_j <= T_j == pi_j == R_j <= H_j` across all 43 scenarios).
   - Detailed verification saved STRICTLY to gitignored
     `results/electrical/caltech_component2_private/`.

Safe, redacted aggregate synthesis across all three domains (ZERO raw DPWH conduit
tables, ZERO raw Caltech phasor/bus/meter tables, ZERO PII) is written to:
- `results/heldout_eval/domain_cases_v2_summary.json`
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from electrical_scenarios import define_electrical_scenarios  # noqa: E402
from gbbrpm.benchmark_comparators import (  # noqa: E402
    compute_abbe_lower,
    compute_all_tree_bounds,
    compute_ancestry_diagnostic,
    compute_clipped_sum,
    compute_gbbrpm_fast,
    compute_local_upstream,
    compute_recursive_max,
)
from gbbrpm.benchmark_metrics import intersect_intervals  # noqa: E402
from gbbrpm.certified_screening import (  # noqa: E402
    assert_zero_false_certifications,
    certify_top_k_two_stage,
)
from gbbrpm.core import evaluate_gbbrpm  # noqa: E402
from gbbrpm.csr_graph import PreparedDAG, prepare_dag  # noqa: E402
from gbbrpm.exact_ref import evaluate_exact_marginals  # noqa: E402
from gbbrpm.generators import derive_scenario_seed  # noqa: E402
from gbbrpm.mc_ref import run_mc_reference  # noqa: E402
from reconvergence import evaluate_path_pruned_reference  # noqa: E402

TOL = 1e-12
EXPRESS_EXPECTED_ROOT_TAU1 = 0.997528076171875
MC_TRIALS = 100_000
MC_BATCH_SIZE = 10_000
HOEFFDING_ALPHA = 0.05


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_file_hash(path: Path, expected_sha: str, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file for {label}: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for {label} ({path.name}): expected {expected_sha}, got {actual}"
        )
    return {"filename": path.name, "sha256": actual, "verified": True}


def _count_nondegenerate_primitives(dag: PreparedDAG) -> int:
    n_nodes_nd = int(np.sum((dag.B > 0.0) & (dag.B < 1.0)))
    n_edges_nd = int(np.sum((dag.in_weights > 0.0) & (dag.in_weights < 1.0)))
    return n_nodes_nd + n_edges_nd


def _evaluate_scenario_bundle(
    scenario_id: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    exact_max_primitives: int = 16,
    mc_trials: int = MC_TRIALS,
    check_df_parity: bool = True,
    check_pruned_parity: bool = True,
) -> dict[str, Any]:
    """Run GBBRPM, all comparators, exact/MC reference, and Two-Stage certification."""
    dag = prepare_dag(nodes_df, edges_df)

    R = compute_gbbrpm_fast(dag)
    M = compute_recursive_max(dag)
    T = compute_all_tree_bounds(dag)
    H = compute_clipped_sum(dag)
    local_up = compute_local_upstream(dag)
    abbe = compute_abbe_lower(dag)
    anc_meta = compute_ancestry_diagnostic(dag)
    anc = anc_meta["scores"]
    W = R - M

    if check_df_parity:
        df_risk, _ = evaluate_gbbrpm(nodes_df, edges_df)
        df_vec = np.array([df_risk[nid] for nid in dag.node_ids], dtype=np.float64)
        df_err = float(np.max(np.abs(df_vec - R)))
        if df_err >= TOL:
            raise AssertionError(
                f"{scenario_id}: DataFrame vs CSR GBBRPM parity error {df_err:.3e}"
            )
    else:
        df_err = 0.0

    if check_pruned_parity:
        pruned_map = evaluate_path_pruned_reference(nodes_df, edges_df)
        pruned_vec = np.array([pruned_map[nid] for nid in dag.node_ids], dtype=np.float64)
        pruned_err = float(np.max(np.abs(pruned_vec - anc)))
        if pruned_err >= TOL:
            raise AssertionError(
                f"{scenario_id}: reconvergence.py vs ancestry_diagnostic parity error {pruned_err:.3e}"
            )
    else:
        pruned_err = 0.0

    # Deterministic bracket check: M <= T <= R <= H
    bracket_viol = int(
        np.sum((M > T + TOL) | (T > R + TOL) | (R > H + TOL))
    )
    if bracket_viol != 0:
        raise AssertionError(
            f"{scenario_id}: deterministic bracket M <= T <= R <= H violated at {bracket_viol} nodes"
        )

    # Exact or MC reference
    n_nondeg = _count_nondegenerate_primitives(dag)
    if n_nondeg <= exact_max_primitives:
        ref_mode = "exact"
        ex = evaluate_exact_marginals(dag, max_primitives=exact_max_primitives)
        pi_ref = ex["marginals"]
        pi_lo = pi_ref.copy()
        pi_hi = pi_ref.copy()
        hoeffding_h = 0.0
        mc_seed = None
        n_empty_intersections = 0

        # Exact bracket & Abbe checks: M <= T <= pi <= R <= H and abbe <= pi
        exact_bracket_viol = int(
            np.sum((T > pi_ref + TOL) | (pi_ref > R + TOL))
        )
        abbe_viol = int(np.sum(abbe > pi_ref + TOL))
        if exact_bracket_viol != 0:
            raise AssertionError(
                f"{scenario_id}: exact bracket T <= pi <= R violated at {exact_bracket_viol} nodes"
            )
        if abbe_viol != 0:
            raise AssertionError(
                f"{scenario_id}: Abbe lower bound violated exact pi at {abbe_viol} nodes"
            )
    else:
        ref_mode = "mc_100k"
        mc_seed = derive_scenario_seed(scenario_id)
        mc = run_mc_reference(
            dag,
            n_trials=mc_trials,
            seed=mc_seed,
            batch_size=MC_BATCH_SIZE,
            alpha=HOEFFDING_ALPHA,
        )
        pi_ref = mc["pi_hat"]
        hoeffding_h = float(mc["hoeffding_h"])
        pi_lo, pi_hi, empty_mask = intersect_intervals(mc["lower"], mc["upper"], T, R)
        n_empty_intersections = int(np.sum(empty_mask))
        # Keep non-repaired intersections for audit; if any node is empty, record transparently
        exact_bracket_viol = 0
        # Check Abbe against analytic upper bound R and MC upper bound
        abbe_viol = int(np.sum((abbe > R + TOL) | (abbe > mc["upper"] + TOL)))
        if abbe_viol != 0:
            raise AssertionError(
                f"{scenario_id}: Abbe lower bound violated upper bound at {abbe_viol} nodes"
            )

    # Two-stage certified screening across standard budgets (0.10, 0.20, 0.30)
    screening = certify_top_k_two_stage(
        dag,
        fractions=(0.10, 0.20, 0.30),
        epsilon=TOL,
        precomputed_M=M,
        precomputed_R=R,
    )
    if ref_mode == "exact":
        cert_check = assert_zero_false_certifications(
            screening, pi_exact=pi_ref, epsilon=TOL
        )
    else:
        # Always verify against valid analytic bounds [T, R], plus intersected Hoeffding if nonempty
        if n_empty_intersections == 0:
            cert_check = assert_zero_false_certifications(
                screening, pi_lower=pi_lo, pi_upper=pi_hi, epsilon=TOL
            )
        else:
            cert_check = assert_zero_false_certifications(
                screening, pi_lower=T, pi_upper=R, epsilon=TOL
            )

    return {
        "dag": dag,
        "R": R,
        "M": M,
        "T": T,
        "H": H,
        "local_upstream": local_up,
        "abbe_lower": abbe,
        "ancestry": anc,
        "ancestry_meta": anc_meta,
        "W": W,
        "ref_mode": ref_mode,
        "n_nondeg_primitives": n_nondeg,
        "pi_ref": pi_ref,
        "pi_lo": pi_lo,
        "pi_hi": pi_hi,
        "hoeffding_h": hoeffding_h,
        "mc_seed": mc_seed,
        "n_empty_intersections": n_empty_intersections,
        "bracket_violations": bracket_viol + exact_bracket_viol,
        "abbe_violations": abbe_viol,
        "false_certifications": int(cert_check["n_false_in"] + cert_check["n_false_out"]),
        "df_parity_err": df_err,
        "pruned_parity_err": pruned_err,
        "screening": screening,
    }


# ---------------------------------------------------------------------------
# Domain 1: Urban Drainage (DPWH CDO Bitan-ag-Kolambog)
# ---------------------------------------------------------------------------


def run_dpwh_drainage_domain(freeze_manifest: dict[str, Any]) -> dict[str, Any]:
    """Run Tier-2 controlled scenarios on local restricted DPWH Bitan-ag-Kolambog DAG."""
    dpwh_dir = ROOT / "data" / "drainage" / "dpwh_cdo_bitanag_kolambog"
    private_out_dir = ROOT / "results" / "drainage" / "dpwh_private"
    private_out_dir.mkdir(parents=True, exist_ok=True)

    ext_hashes = freeze_manifest["external_restricted_hashes"]
    hash_checks = {}
    for rel in (
        "data/drainage/dpwh_cdo_bitanag_kolambog/nodes.csv",
        "data/drainage/dpwh_cdo_bitanag_kolambog/edges.csv",
        "data/drainage/dpwh_cdo_bitanag_kolambog/manifest.json",
    ):
        hash_checks[Path(rel).name] = _verify_file_hash(
            ROOT / rel, ext_hashes[rel]["sha256"], rel
        )

    nodes_base = pd.read_csv(dpwh_dir / "nodes.csv")
    edges_base = pd.read_csv(dpwh_dir / "edges.csv")

    # Structural check
    G = nx.DiGraph()
    for nid in nodes_base["node"].astype(str):
        G.add_node(nid)
    for row in edges_base.itertuples(index=False):
        G.add_edge(str(row.source), str(row.target))
    assert nx.is_directed_acyclic_graph(G)
    assert nx.number_weakly_connected_components(G) == 1

    u_vals = (edges_base["L"] / edges_base["C"]).to_numpy(dtype=np.float64)
    s_vals = np.clip(u_vals, 0.0, 1.0)

    headwaters = ["J_BIT_HEAD", "J_TRIB1_HEAD", "J_KOL_HEAD"]
    single_targets = ["J_BIT_HEAD", "J_TRIB1_HEAD", "J_KOL_HEAD", "J_CAM_WEIR", "J_AGORA_SPLIT"]
    b_levels = (0.25, 0.50, 0.75, 1.00)
    tau_levels = (0.25, 0.50, 0.75, 1.00)

    # Assemble controlled Tier-2 scenarios (unique scenario_id)
    scenario_specs: list[dict[str, Any]] = []

    # 1) Single-source upstream & split injections across B levels (tau = 1.0)
    for src in single_targets:
        for b in b_levels:
            scenario_specs.append(
                {
                    "scenario_id": f"dpwh|single|{src}|B={b:.2f}|tau=1.00",
                    "family": "single_source",
                    "disturbances": {src: b},
                    "tau": 1.00,
                }
            )

    # 2) Multi-source pairs among upstream/split nodes at B in {0.50, 0.75} (tau = 1.0)
    pair_pool = ["J_BIT_HEAD", "J_TRIB1_HEAD", "J_KOL_HEAD", "J_AGORA_SPLIT"]
    for left, right in combinations(pair_pool, 2):
        for b in (0.50, 0.75):
            scenario_specs.append(
                {
                    "scenario_id": f"dpwh|pair|{left}+{right}|B={b:.2f}|tau=1.00",
                    "family": "multi_source_pair",
                    "disturbances": {left: b, right: b},
                    "tau": 1.00,
                }
            )

    # 3) All-headwaters multi-source injection across B levels (tau = 1.0)
    for b in b_levels:
        scenario_specs.append(
            {
                "scenario_id": f"dpwh|all_headwaters|B={b:.2f}|tau=1.00",
                "family": "all_headwaters",
                "disturbances": {h: b for h in headwaters},
                "tau": 1.00,
            }
        )

    # 4) Uniform-baseline tau stress sweeps (tau in {0.25, 0.50, 0.75, 1.00})
    for src in ("J_BIT_HEAD", "J_AGORA_SPLIT"):
        for b in b_levels:
            for tau in tau_levels:
                if tau == 1.00:
                    continue  # already included in single_source
                scenario_specs.append(
                    {
                        "scenario_id": f"dpwh|tau_sweep|{src}|B={b:.2f}|tau={tau:.2f}",
                        "family": "tau_sweep_single",
                        "disturbances": {src: b},
                        "tau": tau,
                    }
                )

    # All-headwaters B=0.50 & uniform-all-nodes B=0.25 across tau in {0.25, 0.50, 0.75, 1.00}
    for tau in tau_levels:
        if tau != 1.00:
            scenario_specs.append(
                {
                    "scenario_id": f"dpwh|tau_sweep|all_headwaters|B=0.50|tau={tau:.2f}",
                    "family": "tau_sweep_headwaters",
                    "disturbances": {h: 0.50 for h in headwaters},
                    "tau": tau,
                }
            )
        scenario_specs.append(
            {
                "scenario_id": f"dpwh|uniform_all_nodes|B=0.25|tau={tau:.2f}",
                "family": "uniform_all_nodes_stress",
                "disturbances": {str(n): 0.25 for n in nodes_base["node"]},
                "tau": tau,
            }
        )

    scenario_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []

    total_bracket_viol = 0
    total_abbe_viol = 0
    total_false_cert = 0
    total_empty_inter = 0
    max_df_parity_err = 0.0
    max_pruned_parity_err = 0.0

    for spec in scenario_specs:
        nodes_sc = nodes_base[["node", "B"]].copy()
        nodes_sc["B"] = 0.0
        for nid, b_val in spec["disturbances"].items():
            nodes_sc.loc[nodes_sc["node"] == nid, "B"] = float(b_val)

        edges_sc = edges_base[["source", "target", "L", "C", "tau"]].copy()
        edges_sc["tau"] = float(spec["tau"])

        ev = _evaluate_scenario_bundle(
            spec["scenario_id"],
            nodes_sc,
            edges_sc,
            exact_max_primitives=16,
        )
        dag = ev["dag"]
        total_bracket_viol += ev["bracket_violations"]
        total_abbe_viol += ev["abbe_violations"]
        total_false_cert += ev["false_certifications"]
        total_empty_inter += ev["n_empty_intersections"]
        max_df_parity_err = max(max_df_parity_err, ev["df_parity_err"])
        max_pruned_parity_err = max(max_pruned_parity_err, ev["pruned_parity_err"])

        idx_agora = dag.node_to_idx["J_AGORA_CONF"]
        idx_outfall = dag.node_to_idx["J_BIT_OUTFALL"]
        b20 = ev["screening"]["budgets"][0.20]

        agora_R = float(ev["R"][idx_agora])
        agora_pi = float(ev["pi_ref"][idx_agora])
        agora_M = float(ev["M"][idx_agora])
        agora_T = float(ev["T"][idx_agora])
        agora_H = float(ev["H"][idx_agora])
        agora_abbe = float(ev["abbe_lower"][idx_agora])
        agora_anc = float(ev["ancestry"][idx_agora])

        outfall_R = float(ev["R"][idx_outfall])
        outfall_pi = float(ev["pi_ref"][idx_outfall])
        outfall_anc = float(ev["ancestry"][idx_outfall])

        scenario_rows.append(
            {
                "scenario_id": spec["scenario_id"],
                "family": spec["family"],
                "n_active_sources": len(spec["disturbances"]),
                "tau": float(spec["tau"]),
                "ref_mode": ev["ref_mode"],
                "n_nondeg_primitives": ev["n_nondeg_primitives"],
                "hoeffding_h": ev["hoeffding_h"],
                "mean_R": float(np.mean(ev["R"])),
                "max_R": float(np.max(ev["R"])),
                "mean_pi": float(np.mean(ev["pi_ref"])),
                "max_abs_R_minus_pi": float(np.max(np.abs(ev["R"] - ev["pi_ref"]))),
                "mean_W": float(np.mean(ev["W"])),
                "max_W": float(np.max(ev["W"])),
                "agora_R": agora_R,
                "agora_pi": agora_pi,
                "agora_M": agora_M,
                "agora_T": agora_T,
                "agora_H": agora_H,
                "agora_abbe": agora_abbe,
                "agora_ancestry": agora_anc,
                "agora_R_minus_pi": agora_R - agora_pi,
                "agora_R_minus_ancestry": agora_R - agora_anc,
                "agora_W": agora_R - agora_M,
                "outfall_R": outfall_R,
                "outfall_pi": outfall_pi,
                "outfall_R_minus_pi": outfall_R - outfall_pi,
                "outfall_R_minus_ancestry": outfall_R - outfall_anc,
                "res_s1_f20": float(b20["res_s1"]),
                "res_s2_f20": float(b20["res_s2"]),
                "n_stage2_trees_f20": int(b20["n_stage2_trees"]),
                "bracket_violations": ev["bracket_violations"],
                "abbe_violations": ev["abbe_violations"],
                "false_certifications": ev["false_certifications"],
                "empty_intersections": ev["n_empty_intersections"],
            }
        )

        for j in range(dag.n_nodes):
            node_rows.append(
                {
                    "scenario_id": spec["scenario_id"],
                    "family": spec["family"],
                    "tau": float(spec["tau"]),
                    "node": dag.node_ids[j],
                    "B": float(dag.B[j]),
                    "M": float(ev["M"][j]),
                    "T": float(ev["T"][j]),
                    "pi_ref": float(ev["pi_ref"][j]),
                    "R": float(ev["R"][j]),
                    "H": float(ev["H"][j]),
                    "abbe_lower": float(ev["abbe_lower"][j]),
                    "local_upstream": float(ev["local_upstream"][j]),
                    "ancestry": float(ev["ancestry"][j]),
                    "W": float(ev["W"][j]),
                    "R_minus_pi": float(ev["R"][j] - ev["pi_ref"][j]),
                    "R_minus_ancestry": float(ev["R"][j] - ev["ancestry"][j]),
                    "ref_mode": ev["ref_mode"],
                }
            )

    sc_df = pd.DataFrame(scenario_rows)
    nd_df = pd.DataFrame(node_rows)

    # Save detailed private tables STRICTLY in gitignored dpwh_private/
    sc_df.to_csv(private_out_dir / "dpwh_v2_scenarios.csv", index=False, lineterminator="\n")
    nd_df.to_csv(private_out_dir / "dpwh_v2_nodes.csv", index=False, lineterminator="\n")

    exact_sub = sc_df[sc_df["ref_mode"] == "exact"]
    mc_sub = sc_df[sc_df["ref_mode"] != "exact"]

    # Key reconvergence highlights at J_AGORA_CONF (exact scenarios only!)
    reconv_positive = exact_sub[exact_sub["agora_R_minus_pi"] > 1e-12]
    max_agora_idx = exact_sub["agora_R_minus_pi"].idxmax()
    max_agora_row = exact_sub.loc[max_agora_idx]
    max_oi_idx = exact_sub["agora_R_minus_ancestry"].idxmax()
    max_oi_row = exact_sub.loc[max_oi_idx]

    # Single-source J_BIT_HEAD and J_AGORA_SPLIT B-sweep at tau=1.0 (safe aggregate view)
    def _extract_sweep(prefix: str) -> list[dict[str, Any]]:
        rows_out = []
        for b in b_levels:
            sid = f"dpwh|single|{prefix}|B={b:.2f}|tau=1.00"
            r = sc_df[sc_df["scenario_id"] == sid].iloc[0]
            rows_out.append(
                {
                    "B": float(b),
                    "tau": 1.0,
                    "ref_mode": str(r["ref_mode"]),
                    "agora_M": float(r["agora_M"]),
                    "agora_T": float(r["agora_T"]),
                    "agora_pi": float(r["agora_pi"]),
                    "agora_R": float(r["agora_R"]),
                    "agora_H": float(r["agora_H"]),
                    "agora_ancestry": float(r["agora_ancestry"]),
                    "agora_R_minus_pi": float(r["agora_R_minus_pi"]),
                    "agora_R_minus_ancestry": float(r["agora_R_minus_ancestry"]),
                    "agora_W": float(r["agora_W"]),
                }
            )
        return rows_out

    def _extract_tau_sweep(src: str, b_fixed: float = 0.50) -> list[dict[str, Any]]:
        rows_out = []
        for tau in tau_levels:
            sid = (
                f"dpwh|single|{src}|B={b_fixed:.2f}|tau=1.00"
                if tau == 1.00
                else f"dpwh|tau_sweep|{src}|B={b_fixed:.2f}|tau={tau:.2f}"
            )
            r = sc_df[sc_df["scenario_id"] == sid].iloc[0]
            rows_out.append(
                {
                    "tau": float(tau),
                    "B": float(b_fixed),
                    "ref_mode": str(r["ref_mode"]),
                    "agora_pi": float(r["agora_pi"]),
                    "agora_R": float(r["agora_R"]),
                    "agora_ancestry": float(r["agora_ancestry"]),
                    "agora_R_minus_pi": float(r["agora_R_minus_pi"]),
                    "agora_R_minus_ancestry": float(r["agora_R_minus_ancestry"]),
                    "agora_W": float(r["agora_W"]),
                }
            )
        return rows_out

    # Private full summary (can include scenario-level breakdown)
    private_summary = {
        "case_id": "drainage_dpwh_cdo_bitanag_kolambog_v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hash_verification": hash_checks,
        "scenario_count": int(len(sc_df)),
        "exact_scenario_count": int(len(exact_sub)),
        "mc_fallback_scenario_count": int(len(mc_sub)),
        "family_counts": {str(k): int(v) for k, v in sc_df["family"].value_counts().sort_index().items()},
        "bracket_violations_total": int(total_bracket_viol),
        "abbe_violations_total": int(total_abbe_viol),
        "false_certifications_total": int(total_false_cert),
        "empty_intersections_total": int(total_empty_inter),
        "max_df_vs_csr_parity_err": float(max_df_parity_err),
        "max_pruned_vs_ancestry_parity_err": float(max_pruned_parity_err),
        "j_bit_head_b_sweep_tau1": _extract_sweep("J_BIT_HEAD"),
        "j_agora_split_b_sweep_tau1": _extract_sweep("J_AGORA_SPLIT"),
        "j_bit_head_tau_sweep_b050": _extract_tau_sweep("J_BIT_HEAD", 0.50),
        "j_agora_split_tau_sweep_b050": _extract_tau_sweep("J_AGORA_SPLIT", 0.50),
    }
    (private_out_dir / "dpwh_v2_summary.json").write_text(
        json.dumps(private_summary, indent=2) + "\n", encoding="utf-8"
    )

    # Safe, redacted public aggregate summary (ZERO raw conduit tables / ZERO PII)
    safe_public = {
        "case_id": "drainage_dpwh_cdo_bitanag_kolambog_v1",
        "gate_a_structural_readiness": "PASS",
        "gate_b_independent_outcome_validation": "FAIL",
        "gate_b_note": (
            "June 2025 DEO flooding hotspots (B01-B16) are exploratory only; "
            "no confirmatory field-flooding validation claim."
        ),
        "redistribution_policy": "restricted_not_in_git_aggregate_summary_only",
        "input_hash_verification_passed": True,
        "graph_summary": {
            "node_count": int(G.number_of_nodes()),
            "edge_count": int(G.number_of_edges()),
            "dag": True,
            "weak_components": 1,
            "s_min": round(float(s_vals.min()), 6),
            "s_max": round(float(s_vals.max()), 6),
            "s_mean": round(float(s_vals.mean()), 6),
            "reconvergent_junction": "J_AGORA_CONF",
            "reconvergent_pcswmm_ref": "J5",
            "reconvergence_diamond": {
                "split_junction": "J_AGORA_SPLIT",
                "reconvergence_junction": "J_AGORA_CONF",
                "path_1_hops": 2,
                "path_2_hops": 1,
                "description": (
                    "Agora diversion split (J_AGORA_SPLIT) bifurcates into the main "
                    "Bitan-ag trunk (2 hops via J_TRIB1_CONF) and the Agora diversion "
                    "channel (1 hop), reconverging at J_AGORA_CONF (PCSWMM J5)."
                ),
            },
        },
        "tier2_scenario_evaluation": {
            "scenario_count": int(len(sc_df)),
            "exact_reference_scenarios": int(len(exact_sub)),
            "mc_100k_fallback_scenarios": int(len(mc_sub)),
            "family_counts": private_summary["family_counts"],
            "bracket_violations_total": int(total_bracket_viol),
            "abbe_violations_total": int(total_abbe_viol),
            "false_certifications_total": int(total_false_cert),
            "empty_intersections_total": int(total_empty_inter),
            "max_df_vs_csr_parity_err": float(max_df_parity_err),
            "max_pruned_vs_ancestry_parity_err": float(max_pruned_parity_err),
            "exact_scenarios_max_abs_R_minus_pi": float(exact_sub["max_abs_R_minus_pi"].max()),
            "exact_scenarios_mean_max_W": float(exact_sub["max_W"].mean()),
            "two_stage_screening_f20_mean_res_s1": float(sc_df["res_s1_f20"].mean()),
            "two_stage_screening_f20_mean_res_s2": float(sc_df["res_s2_f20"].mean()),
        },
        "j_agora_conf_reconvergence": {
            "exact_scenarios_with_positive_inflation": int(len(reconv_positive)),
            "exact_max_R_minus_pi": float(max_agora_row["agora_R_minus_pi"]),
            "exact_max_R_minus_pi_scenario_family": str(max_agora_row["family"]),
            "exact_max_R_minus_pi_tau": float(max_agora_row["tau"]),
            "exact_max_R_minus_ancestry": float(max_oi_row["agora_R_minus_ancestry"]),
            "exact_max_R_minus_ancestry_scenario_family": str(max_oi_row["family"]),
            "j_bit_head_b_sweep_tau1": _extract_sweep("J_BIT_HEAD"),
            "j_agora_split_b_sweep_tau1": _extract_sweep("J_AGORA_SPLIT"),
            "j_bit_head_tau_sweep_b050": _extract_tau_sweep("J_BIT_HEAD", 0.50),
            "j_agora_split_tau_sweep_b050": _extract_tau_sweep("J_AGORA_SPLIT", 0.50),
        },
    }
    print(
        f"[Domain 1 DPWH] {len(sc_df)} scenarios ({len(exact_sub)} exact, {len(mc_sub)} MC), "
        f"0 violations, max J_AGORA_CONF (R - pi)={safe_public['j_agora_conf_reconvergence']['exact_max_R_minus_pi']:.6f}, "
        f"max (R - ancestry)={safe_public['j_agora_conf_reconvergence']['exact_max_R_minus_ancestry']:.6f}",
        flush=True,
    )
    return safe_public


# ---------------------------------------------------------------------------
# Domain 2: Software Dependency (Express 4.18.2)
# ---------------------------------------------------------------------------


def run_express_software_domain(
    freeze_manifest: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Run full tau sweep + qs/send reconvergence analysis on verified Express 4.18.2."""
    exp_dir = ROOT / "data" / "software" / "express_4.18.2"
    tracked_hashes = freeze_manifest["tracked_file_hashes"]
    hash_checks = {}
    for rel in (
        "data/software/express_4.18.2/nodes.csv",
        "data/software/express_4.18.2/edges.csv",
        "data/software/express_4.18.2/manifest.json",
    ):
        hash_checks[Path(rel).name] = _verify_file_hash(
            ROOT / rel, tracked_hashes[rel]["sha256"], rel
        )

    nodes_raw = pd.read_csv(exp_dir / "nodes.csv")
    edges_raw = pd.read_csv(exp_dir / "edges.csv")
    root_id = "npm:express@4.18.2"
    qs_id = "npm:qs@6.11.0"
    send_id = "npm:send@0.18.0"

    # Build DiGraph for structural path verification (qs & send)
    G = nx.DiGraph()
    for nid in nodes_raw["node"].astype(str):
        G.add_node(nid)
    for row in edges_raw.itertuples(index=False):
        G.add_edge(str(row.source), str(row.target))
    assert nx.is_directed_acyclic_graph(G)
    assert G.number_of_nodes() == 71 and G.number_of_edges() == 128

    qs_paths = [list(p) for p in nx.all_simple_paths(G, qs_id, root_id)]
    send_paths = [list(p) for p in nx.all_simple_paths(G, send_id, root_id)]
    advisory_nodes = sorted(
        nodes_raw.loc[pd.to_numeric(nodes_raw["B"]) > 0.0, "node"].astype(str).tolist()
    )

    tau_values = (0.0, 0.25, 0.50, 0.75, 1.0)
    sweep_rows: list[dict[str, Any]] = []
    tau_summaries: dict[str, Any] = {}

    total_bracket_viol = 0
    total_abbe_viol = 0
    total_false_cert = 0
    total_empty_inter = 0

    for tau in tau_values:
        nodes_sc = nodes_raw[["node", "B"]].copy()
        edges_sc = edges_raw[["source", "target", "S"]].copy()
        edges_sc["tau"] = float(tau)
        sid = f"domain_express_v2|tau={tau:.2f}"

        # At tau=0 or tau=1, edge weights are degenerate (0 or 1), leaving <= 7 nondeg node B primitives -> exact!
        # At tau in {0.25, 0.50, 0.75}, 128 edges + 7 nodes -> mc_ref N=100,000!
        # Wait: brief also asks for mc_ref (N=100,000) across the tau in {0.25, 0.50, 0.75, 1.0} sweep;
        # so at tau=1.0 we compute BOTH exact (7 non-deg primitives!) AND mc_ref (N=100,000) for complete parity!
        ev = _evaluate_scenario_bundle(
            sid,
            nodes_sc,
            edges_sc,
            exact_max_primitives=16,
            mc_trials=MC_TRIALS,
        )
        dag = ev["dag"]
        root_idx = dag.node_to_idx[root_id]

        # Run explicit N=100,000 MC for every tau in {0.25, 0.50, 0.75, 1.0}
        if tau > 0.0:
            if ev["ref_mode"] == "mc_100k":
                mc_pi_hat = ev["pi_ref"]
                mc_h = ev["hoeffding_h"]
                mc_seed = ev["mc_seed"]
                exact_root = None
            else:
                mc_seed = derive_scenario_seed(f"{sid}|mc_100k")
                mc_extra = run_mc_reference(
                    dag,
                    n_trials=MC_TRIALS,
                    seed=mc_seed,
                    batch_size=MC_BATCH_SIZE,
                    alpha=HOEFFDING_ALPHA,
                )
                mc_pi_hat = mc_extra["pi_hat"]
                mc_h = float(mc_extra["hoeffding_h"])
                exact_root = float(ev["pi_ref"][root_idx])
        else:
            mc_pi_hat = ev["pi_ref"]
            mc_h = 0.0
            mc_seed = None
            exact_root = float(ev["pi_ref"][root_idx])

        root_R = float(ev["R"][root_idx])
        root_M = float(ev["M"][root_idx])
        root_T = float(ev["T"][root_idx])
        root_H = float(ev["H"][root_idx])
        root_abbe = float(ev["abbe_lower"][root_idx])
        root_local_up = float(ev["local_upstream"][root_idx])
        root_anc = float(ev["ancestry"][root_idx])
        root_pi_mc = float(mc_pi_hat[root_idx])
        root_pi_ref = float(ev["pi_ref"][root_idx])

        if tau == 1.0:
            if abs(root_R - EXPRESS_EXPECTED_ROOT_TAU1) >= 1e-15:
                raise AssertionError(
                    f"Express root risk at tau=1.0 mismatch: got {root_R:.16f}, "
                    f"expected {EXPRESS_EXPECTED_ROOT_TAU1:.16f}"
                )

        # Shared-path reconvergence isolation for qs and send at this tau
        def _isolate_single_source_reconvergence(src_id: str) -> dict[str, Any]:
            n_iso = nodes_raw[["node", "B"]].copy()
            b_orig = float(n_iso.loc[n_iso["node"] == src_id, "B"].iloc[0])
            n_iso["B"] = 0.0
            n_iso.loc[n_iso["node"] == src_id, "B"] = b_orig
            d_iso = prepare_dag(n_iso, edges_sc)
            r_iso = compute_gbbrpm_fast(d_iso)
            anc_iso = compute_ancestry_diagnostic(d_iso)["scores"]
            t_iso = compute_all_tree_bounds(d_iso)
            # Only edges on paths from src_id to root_id matter -> exact small subgraph!
            sub_nodes = nx.descendants(G, src_id) | {src_id}
            sub_n_df = n_iso[n_iso["node"].isin(sub_nodes)].copy()
            sub_e_df = edges_sc[
                edges_sc["source"].isin(sub_nodes) & edges_sc["target"].isin(sub_nodes)
            ].copy()
            sub_dag = prepare_dag(sub_n_df, sub_e_df)
            ex_iso = evaluate_exact_marginals(sub_dag, max_primitives=16)
            sub_root_idx = sub_dag.node_to_idx[root_id]
            d_root_idx = d_iso.node_to_idx[root_id]
            pi_exact_val = float(ex_iso["marginals"][sub_root_idx])
            r_val = float(r_iso[d_root_idx])
            anc_val = float(anc_iso[d_root_idx])
            t_val = float(t_iso[d_root_idx])
            return {
                "source_node": src_id,
                "source_B": b_orig,
                "root_R": r_val,
                "root_pi_exact": pi_exact_val,
                "root_T": t_val,
                "root_ancestry": anc_val,
                "root_R_minus_pi_exact": r_val - pi_exact_val,
                "root_R_minus_ancestry": r_val - anc_val,
            }

        qs_iso = _isolate_single_source_reconvergence(qs_id)
        send_iso = _isolate_single_source_reconvergence(send_id)

        b10 = ev["screening"]["budgets"][0.10]
        b20 = ev["screening"]["budgets"][0.20]
        b30 = ev["screening"]["budgets"][0.30]

        total_bracket_viol += ev["bracket_violations"]
        total_abbe_viol += ev["abbe_violations"]
        total_false_cert += ev["false_certifications"]
        total_empty_inter += ev["n_empty_intersections"]

        # Rank of root node (1 = highest)
        root_rank_R = int(1 + np.sum(ev["R"] > root_R + TOL))
        nonzero_R_count = int(np.sum(ev["R"] > 0.0))

        row = {
            "tau": float(tau),
            "ref_mode": ev["ref_mode"],
            "n_nodes": dag.n_nodes,
            "n_edges": dag.n_edges,
            "nonzero_R_nodes": nonzero_R_count,
            "root_rank_R": root_rank_R,
            "root_M": root_M,
            "root_T": root_T,
            "root_pi_ref": root_pi_ref,
            "root_pi_mc_100k": root_pi_mc,
            "root_pi_exact": exact_root if exact_root is not None else np.nan,
            "root_R": root_R,
            "root_H": root_H,
            "root_abbe_lower": root_abbe,
            "root_local_upstream": root_local_up,
            "root_ancestry": root_anc,
            "root_W": root_R - root_M,
            "root_R_minus_pi_ref": root_R - root_pi_ref,
            "root_R_minus_ancestry": root_R - root_anc,
            "hoeffding_h_100k": mc_h,
            "mean_R": float(np.mean(ev["R"])),
            "mean_W": float(np.mean(ev["W"])),
            "max_W": float(np.max(ev["W"])),
            "max_R_minus_ancestry": float(np.max(ev["R"] - ev["ancestry"])),
            "overlap_node_count": int(ev["ancestry_meta"]["overlap_node_count"]),
            "qs_isolated_root_R": qs_iso["root_R"],
            "qs_isolated_root_pi_exact": qs_iso["root_pi_exact"],
            "qs_isolated_root_R_minus_pi": qs_iso["root_R_minus_pi_exact"],
            "qs_isolated_root_R_minus_ancestry": qs_iso["root_R_minus_ancestry"],
            "send_isolated_root_R": send_iso["root_R"],
            "send_isolated_root_pi_exact": send_iso["root_pi_exact"],
            "send_isolated_root_R_minus_pi": send_iso["root_R_minus_pi_exact"],
            "send_isolated_root_R_minus_ancestry": send_iso["root_R_minus_ancestry"],
            "res_s1_f10": float(b10["res_s1"]),
            "res_s2_f10": float(b10["res_s2"]),
            "res_s1_f20": float(b20["res_s1"]),
            "res_s2_f20": float(b20["res_s2"]),
            "n_stage2_trees_f20": int(b20["n_stage2_trees"]),
            "res_s1_f30": float(b30["res_s1"]),
            "res_s2_f30": float(b30["res_s2"]),
            "bracket_violations": ev["bracket_violations"],
            "abbe_violations": ev["abbe_violations"],
            "false_certifications": ev["false_certifications"],
            "empty_intersections": ev["n_empty_intersections"],
        }
        sweep_rows.append(row)
        tau_summaries[f"{tau:.2f}"] = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in row.items()
        }

    exp_df = pd.DataFrame(sweep_rows)
    csv_path = out_dir / "domain_express_v2.csv"
    exp_df.to_csv(csv_path, index=False, lineterminator="\n")

    tau1 = tau_summaries["1.00"]
    print(
        f"[Domain 2 Express 4.18.2] tau=1.0 root R={tau1['root_R']:.15f} "
        f"(exact pi={tau1['root_pi_exact']:.15f}, R-pi={tau1['root_R_minus_pi_ref']:.6f}, "
        f"R-anc={tau1['root_R_minus_ancestry']:.6f}), "
        f"qs_iso R-pi={tau1['qs_isolated_root_R_minus_pi']:.6f}, "
        f"send_iso R-pi={tau1['send_isolated_root_R_minus_pi']:.6f}, 0 violations",
        flush=True,
    )

    return {
        "case_id": "software_npm_express_4.18.2",
        "gate_a_structural_readiness": "PASS",
        "gate_b_independent_outcome_validation": "N/A",
        "gate_b_note": (
            "OSV advisory severities define local-disturbance inputs B_i, "
            "not independent exploit/incident outcome labels."
        ),
        "redistribution_policy": "public_in_git",
        "input_hash_verification": hash_checks,
        "graph_summary": {
            "node_count": 71,
            "edge_count": 128,
            "dag": True,
            "weak_components": 1,
            "root_package": root_id,
            "advisory_root_count": len(advisory_nodes),
            "advisory_roots": advisory_nodes,
        },
        "verified_root_risk_tau1": float(tau1["root_R"]),
        "verified_root_risk_tau1_expected": EXPRESS_EXPECTED_ROOT_TAU1,
        "verified_root_risk_tau1_exact_match": bool(
            abs(float(tau1["root_R"]) - EXPRESS_EXPECTED_ROOT_TAU1) < 1e-15
        ),
        "shared_path_reconvergence": {
            "qs_paths_to_root": qs_paths,
            "send_paths_to_root": send_paths,
            "qs_tau1_isolated_R_minus_pi_exact": float(tau1["qs_isolated_root_R_minus_pi"]),
            "send_tau1_isolated_R_minus_pi_exact": float(tau1["send_isolated_root_R_minus_pi"]),
            "combined_tau1_root_R_minus_pi_exact": float(tau1["root_R_minus_pi_ref"]),
            "combined_tau1_root_R_minus_ancestry": float(tau1["root_R_minus_ancestry"]),
        },
        "tau_sweep": tau_summaries,
        "bracket_violations_total": int(total_bracket_viol),
        "abbe_violations_total": int(total_abbe_viol),
        "false_certifications_total": int(total_false_cert),
        "empty_intersections_total": int(total_empty_inter),
        "public_csv": "results/heldout_eval/domain_express_v2.csv",
    }


# ---------------------------------------------------------------------------
# Domain 3: Electrical Distribution (Caltech SoCal 28-Bus Component 2)
# ---------------------------------------------------------------------------


def run_caltech_electrical_domain(freeze_manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify arborescence & exact tree equivalence across all 43 Component 2 scenarios."""
    cal_dir = ROOT / "results" / "electrical" / "caltech_component2_private"
    exp_dir = cal_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    ext_hashes = freeze_manifest["external_restricted_hashes"]
    rel_manifest = "results/electrical/caltech_component2_private/manifest.json"
    manifest_check = _verify_file_hash(
        ROOT / rel_manifest, ext_hashes[rel_manifest]["sha256"], rel_manifest
    )

    nodes_base = pd.read_csv(cal_dir / "model_nodes.csv")
    edges_base = pd.read_csv(cal_dir / "model_edges.csv")

    G = nx.from_pandas_edgelist(edges_base, "source", "target", create_using=nx.DiGraph)
    G.add_nodes_from(nodes_base["node"].astype(str))
    is_dag = nx.is_directed_acyclic_graph(G)
    is_arb = nx.is_arborescence(G)
    max_in_deg = max(dict(G.in_degree()).values())
    if not (is_dag and is_arb and max_in_deg <= 1):
        raise AssertionError(
            f"Caltech Component 2 expected arborescence: dag={is_dag}, arb={is_arb}, max_in_deg={max_in_deg}"
        )

    scenarios, meta = define_electrical_scenarios(nodes_base, edges_base)
    if len(scenarios) != 43:
        raise AssertionError(f"Expected 43 electrical scenarios, got {len(scenarios)}")

    u_base = (pd.to_numeric(edges_base["L"]) / pd.to_numeric(edges_base["C"])).to_numpy(dtype=np.float64)
    s_base = np.clip(u_base, 0.0, 1.0)

    per_scenario_verif: list[dict[str, Any]] = []
    max_abs_R_minus_pi = 0.0
    max_abs_T_minus_R = 0.0
    max_abs_R_minus_anc = 0.0
    max_df_parity_err = 0.0
    max_pruned_parity_err = 0.0
    total_bracket_viol = 0
    total_abbe_viol = 0
    total_false_cert = 0

    risk_sums: list[float] = []
    max_W_vals: list[float] = []
    res_s1_f20_vals: list[float] = []
    res_s2_f20_vals: list[float] = []

    for sc in scenarios:
        sc_nodes = nodes_base[["node", "B"]].copy()
        sc_nodes["B"] = 0.0
        for nid, sev in sc.disturbances.items():
            sc_nodes.loc[sc_nodes["node"].astype(str) == nid, "B"] = float(sev)

        sc_edges = edges_base[["source", "target", "L", "C", "tau"]].copy()
        sc_edges["L"] = pd.to_numeric(sc_edges["L"]) * float(sc.load_scale)
        sc_edges["tau"] = float(sc.tau)

        # Allow up to 18 non-deg primitives so even multi_all_internal (11 edges + 6 nodes = 17)
        # is evaluated to EXACT machine precision via evaluate_exact_marginals!
        ev = _evaluate_scenario_bundle(
            f"caltech_comp2|{sc.scenario_id}",
            sc_nodes,
            sc_edges,
            exact_max_primitives=18,
        )
        if ev["ref_mode"] != "exact":
            raise AssertionError(
                f"Expected exact evaluation for {sc.scenario_id}, got {ev['ref_mode']}"
            )

        err_R_pi = float(np.max(np.abs(ev["R"] - ev["pi_ref"])))
        err_T_R = float(np.max(np.abs(ev["T"] - ev["R"])))
        err_R_anc = float(np.max(np.abs(ev["R"] - ev["ancestry"])))

        if err_R_pi >= TOL:
            raise AssertionError(
                f"Tree equivalence violated on {sc.scenario_id}: max |R - pi| = {err_R_pi:.3e}"
            )
        if err_T_R >= TOL:
            raise AssertionError(
                f"Tree bound parity violated on {sc.scenario_id}: max |T - R| = {err_T_R:.3e}"
            )
        if err_R_anc >= TOL:
            raise AssertionError(
                f"Tree zero-overlap parity violated on {sc.scenario_id}: max |R - ancestry| = {err_R_anc:.3e}"
            )

        max_abs_R_minus_pi = max(max_abs_R_minus_pi, err_R_pi)
        max_abs_T_minus_R = max(max_abs_T_minus_R, err_T_R)
        max_abs_R_minus_anc = max(max_abs_R_minus_anc, err_R_anc)
        max_df_parity_err = max(max_df_parity_err, ev["df_parity_err"])
        max_pruned_parity_err = max(max_pruned_parity_err, ev["pruned_parity_err"])
        total_bracket_viol += ev["bracket_violations"]
        total_abbe_viol += ev["abbe_violations"]
        total_false_cert += ev["false_certifications"]

        r_sum = float(np.sum(ev["R"]))
        w_max = float(np.max(ev["W"]))
        b20 = ev["screening"]["budgets"][0.20]
        risk_sums.append(r_sum)
        max_W_vals.append(w_max)
        res_s1_f20_vals.append(float(b20["res_s1"]))
        res_s2_f20_vals.append(float(b20["res_s2"]))

        per_scenario_verif.append(
            {
                "scenario_id": sc.scenario_id,
                "family": sc.family,
                "n_nondeg_primitives": ev["n_nondeg_primitives"],
                "risk_sum": r_sum,
                "max_risk": float(np.max(ev["R"])),
                "max_abs_R_minus_pi": err_R_pi,
                "max_abs_T_minus_R": err_T_R,
                "max_abs_R_minus_ancestry": err_R_anc,
                "max_W": w_max,
                "res_s1_f20": float(b20["res_s1"]),
                "res_s2_f20": float(b20["res_s2"]),
                "bracket_violations": ev["bracket_violations"],
                "abbe_violations": ev["abbe_violations"],
                "false_certifications": ev["false_certifications"],
            }
        )

    private_verif = {
        "case_id": "electrical_caltech_socal28_component2",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_hash_verification": manifest_check,
        "node_count": int(G.number_of_nodes()),
        "edge_count": int(G.number_of_edges()),
        "is_dag": True,
        "is_arborescence": True,
        "max_in_degree": int(max_in_deg),
        "reconvergent_node_count": 0,
        "scenario_count": int(len(scenarios)),
        "exact_scenarios_verified": int(len(scenarios)),
        "family_counts": meta["family_counts"],
        "exact_tree_equivalence_verified": True,
        "max_abs_R_minus_pi": float(max_abs_R_minus_pi),
        "max_abs_T_minus_R": float(max_abs_T_minus_R),
        "max_abs_R_minus_ancestry": float(max_abs_R_minus_anc),
        "max_df_vs_csr_parity_err": float(max_df_parity_err),
        "max_pruned_vs_ancestry_parity_err": float(max_pruned_parity_err),
        "bracket_violations_total": int(total_bracket_viol),
        "abbe_violations_total": int(total_abbe_viol),
        "false_certifications_total": int(total_false_cert),
        "scenarios": per_scenario_verif,
    }

    # Save strictly inside gitignored caltech_component2_private/ (both canonical locations)
    verif_json = json.dumps(private_verif, indent=2) + "\n"
    (cal_dir / "caltech_v2_verification.json").write_text(verif_json, encoding="utf-8")
    (exp_dir / "tree_equivalence_verification.json").write_text(verif_json, encoding="utf-8")

    print(
        f"[Domain 3 Caltech Comp. 2] 43/43 exact arborescence scenarios verified: "
        f"max |R - pi|={max_abs_R_minus_pi:.3e}, max |T - R|={max_abs_T_minus_R:.3e}, "
        f"max |R - anc|={max_abs_R_minus_anc:.3e}, 0 violations",
        flush=True,
    )

    # Safe, redacted public aggregate summary (ZERO raw phasors, meter IDs, or bus tables!)
    return {
        "case_id": "electrical_caltech_socal28_component2",
        "gate_a_structural_readiness": "PASS",
        "gate_b_independent_outcome_validation": "FAIL",
        "gate_b_note": (
            "Telemetry covers a normal-operation window after the Nov 13, 2024 "
            "switching event; no fault-prediction or operational validation claim."
        ),
        "redistribution_policy": "restricted_not_in_git_aggregate_summary_only",
        "manifest_hash_verification_passed": True,
        "graph_summary": {
            "node_count": int(G.number_of_nodes()),
            "edge_count": int(G.number_of_edges()),
            "dag": True,
            "is_arborescence": True,
            "max_in_degree": int(max_in_deg),
            "reconvergent_node_count": 0,
            "baseline_s_min": round(float(s_base.min()), 6),
            "baseline_s_max": round(float(s_base.max()), 6),
            "baseline_s_mean": round(float(s_base.mean()), 6),
        },
        "tree_equivalence_verification": {
            "scenario_count": int(len(scenarios)),
            "exact_scenarios_verified": int(len(scenarios)),
            "family_counts": meta["family_counts"],
            "exact_tree_equivalence_verified": True,
            "tolerance": TOL,
            "max_abs_R_minus_pi": float(max_abs_R_minus_pi),
            "max_abs_T_minus_R": float(max_abs_T_minus_R),
            "max_abs_R_minus_ancestry": float(max_abs_R_minus_anc),
            "max_df_vs_csr_parity_err": float(max_df_parity_err),
            "max_pruned_vs_ancestry_parity_err": float(max_pruned_parity_err),
            "bracket_violations_total": int(total_bracket_viol),
            "abbe_violations_total": int(total_abbe_viol),
            "false_certifications_total": int(total_false_cert),
            "aggregate_risk_sum_min": float(min(risk_sums)),
            "aggregate_risk_sum_max": float(max(risk_sums)),
            "aggregate_max_W_across_scenarios": float(max(max_W_vals)),
            "two_stage_screening_f20_mean_res_s1": float(np.mean(res_s1_f20_vals)),
            "two_stage_screening_f20_mean_res_s2": float(np.mean(res_s2_f20_vals)),
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    freeze_path = ROOT / "results" / "protocol_freeze" / "protocol_freeze_manifest_v2.0.0.json"
    freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))

    out_dir = ROOT / "results" / "heldout_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    drainage_summary = run_dpwh_drainage_domain(freeze_manifest)
    express_summary = run_express_software_domain(freeze_manifest, out_dir)
    electrical_summary = run_caltech_electrical_domain(freeze_manifest)

    total_bracket_viol = (
        drainage_summary["tier2_scenario_evaluation"]["bracket_violations_total"]
        + express_summary["bracket_violations_total"]
        + electrical_summary["tree_equivalence_verification"]["bracket_violations_total"]
    )
    total_abbe_viol = (
        drainage_summary["tier2_scenario_evaluation"]["abbe_violations_total"]
        + express_summary["abbe_violations_total"]
        + electrical_summary["tree_equivalence_verification"]["abbe_violations_total"]
    )
    total_false_cert = (
        drainage_summary["tier2_scenario_evaluation"]["false_certifications_total"]
        + express_summary["false_certifications_total"]
        + electrical_summary["tree_equivalence_verification"]["false_certifications_total"]
    )

    combined = {
        "evaluation_version": "2.0.0",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_runtime_sec": float(time.perf_counter() - t0),
        "privacy_and_redistribution_guardrail": (
            "Contains ONLY safe aggregate metrics and scientific findings. "
            "Detailed DPWH drainage scenario/node tables remain strictly in "
            "gitignored results/drainage/dpwh_private/, and detailed Caltech "
            "Component 2 outputs remain strictly in gitignored "
            "results/electrical/caltech_component2_private/. Zero raw agency "
            "tables, zero raw phasor measurements, and zero PII are included."
        ),
        "protocol_freeze_manifest_sha256": _sha256_file(freeze_path),
        "global_assertions": {
            "all_input_hashes_verified": True,
            "bracket_violations_total": int(total_bracket_viol),
            "abbe_violations_total": int(total_abbe_viol),
            "false_certifications_total": int(total_false_cert),
            "express_root_tau1_exact_match": express_summary["verified_root_risk_tau1_exact_match"],
            "caltech_tree_equivalence_verified": electrical_summary["tree_equivalence_verification"]["exact_tree_equivalence_verified"],
        },
        "urban_drainage": drainage_summary,
        "software_dependency": express_summary,
        "electrical_distribution": electrical_summary,
        "cross_domain_synthesis": [
            {
                "domain": "urban_drainage",
                "case_id": drainage_summary["case_id"],
                "nodes": drainage_summary["graph_summary"]["node_count"],
                "edges": drainage_summary["graph_summary"]["edge_count"],
                "reconvergent": True,
                "gate_a": "PASS",
                "gate_b": "FAIL",
                "scenarios_evaluated": drainage_summary["tier2_scenario_evaluation"]["scenario_count"],
                "key_mechanism_finding": (
                    "Real heterogeneous L/C propagation with measurable shared-ancestry "
                    "reconvergence at J_AGORA_CONF (PCSWMM J5); zero bracket or certification violations."
                ),
            },
            {
                "domain": "software_dependency",
                "case_id": express_summary["case_id"],
                "nodes": express_summary["graph_summary"]["node_count"],
                "edges": express_summary["graph_summary"]["edge_count"],
                "reconvergent": True,
                "gate_a": "PASS",
                "gate_b": "N/A",
                "scenarios_evaluated": len(express_summary["tau_sweep"]),
                "key_mechanism_finding": (
                    "Verified Express root risk 0.997528076171875 at tau=1.0; explicit "
                    "shared-path reconvergence quantified along qs and send dependency paths."
                ),
            },
            {
                "domain": "electrical_distribution",
                "case_id": electrical_summary["case_id"],
                "nodes": electrical_summary["graph_summary"]["node_count"],
                "edges": electrical_summary["graph_summary"]["edge_count"],
                "reconvergent": False,
                "gate_a": "PASS",
                "gate_b": "FAIL",
                "scenarios_evaluated": electrical_summary["tree_equivalence_verification"]["scenario_count"],
                "key_mechanism_finding": (
                    "Verified 12-node arborescence with exact tree equivalence "
                    "(max |R - pi| < 1e-12 and T_j == pi_j == R_j across all 43 scenarios)."
                ),
            },
        ],
    }

    summary_path = out_dir / "domain_cases_v2_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"[Domain Cases v2] Wrote safe aggregate synthesis to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
