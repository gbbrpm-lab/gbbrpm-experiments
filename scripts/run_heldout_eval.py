"""Step 6 Held-Out Evaluation runner for GBBRPM (v2.0.0-protocol-freeze).

Executes the pre-registered held-out empirical benchmark suites locked in
``results/protocol_freeze/protocol_freeze_manifest_v2.0.0.json``:

1. Suite 1 — Held-Out Uniform Generated DAGs (2,880 scenarios):
   ``generate_dag_uniform_scenarios(split="eval")`` (seeds 1001--1020,
   3 families x 4 sizes {20, 50, 100, 250} x 6 weights x 2 B settings,
   N = 100,000 trials, batch_size = 10,000, numpy_uint64_bitpacked).
2. Suite 2 — Held-Out Heterogeneous-Weight DAGs (120 scenarios):
   ``generate_dag_hetero_scenarios(split="eval")`` (seeds 1001--1005,
   paired to w = 0.20 uniform scenarios on identical topologies, N = 100,000).
3. Suite 3 — Express 4.18.2 Real-Topology Stochastic (12 scenarios):
   ``generate_express_scenarios()`` (71 nodes, 128 edges, 7 advisory roots,
   6 weights x 2 B settings, N = 100,000).
4. Suite 4 — Preselected Precision Audits (48 independent N = 1,000,000-trial solves):
   eval seed 1001, 3 families x 4 sizes x w in {0.05, 0.10} x B in {0.5, 1.0},
   with independent seed ``derive_scenario_seed(f"{scenario_id}|precision_audit_1m")``.

Hard assertions (fatal on any violation):
- Zero violations of ``M_j <= T_j <= R_j <= H_j`` across all 3,060 evaluations.
- Zero false certifications in Two-Stage Certified Top-k Screening across
  ``f in (0.10, 0.20, 0.30)`` (both analytic ``[T, R]`` and nonempty
  intersected ``[L', U']``).
- Empty Hoeffding/bracket intersections are NEVER silently repaired
  (``empty_intersection_policy = "record_without_repair"``).

Usage:
    python scripts/run_heldout_eval.py --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gbbrpm.benchmark_comparators import run_comparators
from gbbrpm.benchmark_metrics import (
    compute_capture_and_regret,
    compute_pairwise_reversals,
    compute_saturation_metrics,
    compute_tie_aware_spearman,
    evaluate_nodewise_w_reliability,
    evaluate_scenario_w_screening,
    intersect_intervals,
)
from gbbrpm.certified_screening import (
    assert_zero_false_certifications,
    certify_top_k_two_stage,
)
from gbbrpm.generators import (
    BenchmarkScenario,
    DAG_B_SETTINGS,
    DAG_FAMILIES,
    DAG_SIZES,
    DEV_SEEDS,
    EVAL_SEEDS,
    HETERO_EVAL_SEEDS,
    derive_scenario_seed,
    generate_dag_hetero_scenarios,
    generate_dag_uniform_scenarios,
    generate_express_scenarios,
)
from gbbrpm.mc_ref import run_mc_reference

DEV_SEED_SET = frozenset(DEV_SEEDS)
BUDGET_FRACTIONS: tuple[float, ...] = (0.10, 0.20, 0.30)
BUDGET_TAGS: tuple[tuple[float, str], ...] = (
    (0.10, "f10"),
    (0.20, "f20"),
    (0.30, "f30"),
)
COMPARATORS: tuple[str, ...] = (
    "R",
    "T",
    "M",
    "H",
    "local_upstream",
    "abbe",
    "ancestry",
)
PAIRED_COMP_NAMES: tuple[str, ...] = (
    "T",
    "M",
    "abbe",
    "H",
    "ancestry",
    "local_upstream",
)
TOL: float = 1e-12
MC_TRIALS_PRIMARY: int = 100_000
MC_TRIALS_AUDIT: int = 1_000_000
MC_BATCH_SIZE: int = 10_000
HOEFFDING_ALPHA: float = 0.05
TIE_PERM_SEED: int = 140
TIE_PERM_COUNT: int = 100
AUDIT_SEED: int = 1001
AUDIT_WEIGHTS: tuple[float, ...] = (0.05, 0.10)


# ---------------------------------------------------------------------------
# Utilities & serialization
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    """Prevent BLAS thread oversubscription inside ProcessPoolExecutor workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and non-finite floats to strict RFC-8259 JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_metadata() -> dict[str, Any]:
    def _run_git(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return proc.stdout.strip()
        except Exception:
            return None

    sha = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit_sha": sha,
        "branch": branch,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def _collect_hardware_metadata(workers: int) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "workers_used": int(workers),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "networkx_version": nx.__version__,
        "git": _get_git_metadata(),
    }


def _load_freeze_provenance() -> dict[str, Any]:
    manifest_path = (
        REPO_ROOT / "results" / "protocol_freeze" / "protocol_freeze_manifest_v2.0.0.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Protocol freeze manifest missing at {manifest_path}"
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest_path": "results/protocol_freeze/protocol_freeze_manifest_v2.0.0.json",
        "manifest_sha256": _sha256_file(manifest_path),
        "protocol_version": data.get("protocol_version"),
        "freeze_tag": data.get("freeze_tag"),
        "freeze_date_utc": data.get("freeze_date_utc"),
        "frozen_file_count": len(data.get("tracked_file_hashes", {})),
    }


def _assert_deterministic_bracket(
    M: np.ndarray, T: np.ndarray, R: np.ndarray, H: np.ndarray, sid: str
) -> None:
    if np.any(M > T + TOL):
        diff = float(np.max(M - T))
        raise AssertionError(f"Bracket violation M > T on {sid}: max(M-T)={diff:.3e}")
    if np.any(T > R + TOL):
        diff = float(np.max(T - R))
        raise AssertionError(f"Bracket violation T > R on {sid}: max(T-R)={diff:.3e}")
    if np.any(R > H + TOL):
        diff = float(np.max(R - H))
        raise AssertionError(f"Bracket violation R > H on {sid}: max(R-H)={diff:.3e}")


def _assert_eval_scenario(sc: BenchmarkScenario) -> None:
    if sc.suite in ("dag_uniform", "dag_hetero"):
        if sc.split != "eval":
            raise RuntimeError(
                f"PROTOCOL VIOLATION: Expected split='eval', got {sc.split!r} on {sc.scenario_id}"
            )
        if sc.graph_seed is None or int(sc.graph_seed) in DEV_SEED_SET:
            raise RuntimeError(
                f"PROTOCOL VIOLATION: Forbidden dev/null seed {sc.graph_seed!r} on {sc.scenario_id}"
            )


# ---------------------------------------------------------------------------
# Per-scenario worker functions (picklable top-level functions)
# ---------------------------------------------------------------------------


def _evaluate_primary_scenario_worker(
    task: tuple[int, BenchmarkScenario, int, int]
) -> tuple[int, dict[str, Any]]:
    """Evaluate one primary held-out scenario (Suites 1, 2, 3)."""
    idx, sc, mc_trials, batch_size = task
    _assert_eval_scenario(sc)
    dag = sc.dag
    cand_mask = dag.B == 0.0
    n_cand = int(np.sum(cand_mask))

    # 1. Vectorized uint64 bit-packed MC reference
    mc = run_mc_reference(
        dag,
        n_trials=mc_trials,
        seed=sc.mc_seed,
        batch_size=batch_size,
        alpha=HOEFFDING_ALPHA,
    )
    pi_hat = mc["pi_hat"]
    mc_lo = mc["lower"]
    mc_hi = mc["upper"]
    h = float(mc["hoeffding_h"])
    mc_sec = float(mc["elapsed_sec"])
    mc_backend = str(mc["backend"])

    # 2. All deterministic comparators (including full-graph T_j, Abbe, Ancestry)
    comp = run_comparators(
        dag, include_trees=True, include_abbe=True, include_ancestry=True
    )
    M = comp["recursive_max"]
    T = comp["tree_bound"]
    R = comp["gbbrpm"]
    H = comp["clipped_sum"]
    loc_up = comp["local_upstream"]
    abbe = comp["abbe_lower"]
    anc = comp["ancestry_diagnostic"]
    W = comp["bracket_width"]
    timings = comp["timings_sec"]
    full_tree_sec = float(timings["tree_bound"])

    # HARD ASSERTION: M <= T <= R <= H across ALL nodes
    _assert_deterministic_bracket(M, T, R, H, sc.scenario_id)

    # Intersect MC Hoeffding interval with analytic bracket [T, R] (NEVER silently repair!)
    L_prime, U_prime, empty_mask = intersect_intervals(mc_lo, mc_hi, T, R)
    n_empty = int(np.sum(empty_mask))
    n_empty_cand = int(np.sum(empty_mask & cand_mask))

    # 3. Standalone Two-Stage Certified Top-k Screening (unprecomputed M, R for fair timing)
    scr = certify_top_k_two_stage(dag, fractions=BUDGET_FRACTIONS, epsilon=TOL)

    # HARD ASSERTION: Zero false certifications against analytic [T, R] AND
    # nonempty intersected [L', U'] whenever n_empty == 0.
    assert_zero_false_certifications(scr, pi_lower=T, pi_upper=R, epsilon=TOL)
    if n_empty == 0:
        assert_zero_false_certifications(
            scr, pi_lower=L_prime, pi_upper=U_prime, epsilon=TOL
        )

    # 4. Ranking, capture/regret across ALL 7 comparators & ALL 3 budgets
    comp_vectors: dict[str, np.ndarray] = {
        "R": R,
        "T": T,
        "M": M,
        "H": H,
        "local_upstream": loc_up,
        "abbe": abbe,
        "ancestry": anc,
    }
    caps: dict[str, dict[str, Any]] = {}
    spearmans: dict[str, float] = {}
    for name, vec in comp_vectors.items():
        caps[name] = compute_capture_and_regret(
            vec,
            pi_hat,
            cand_mask,
            fractions=BUDGET_FRACTIONS,
            seed=TIE_PERM_SEED,
            n_perms=TIE_PERM_COUNT,
            tol=TOL,
            h=h,
        )
        spearmans[name] = compute_tie_aware_spearman(vec, pi_hat, cand_mask, tol=TOL)

    rev_R = compute_pairwise_reversals(R, L_prime, U_prime, cand_mask, eps=TOL)
    sat_R = compute_saturation_metrics(R, cand_mask, tol=TOL)
    sat_H = compute_saturation_metrics(H, cand_mask, tol=TOL)
    sat_pi = compute_saturation_metrics(pi_hat, cand_mask, tol=TOL)
    w_rel = evaluate_nodewise_w_reliability(W, R, pi_hat, cand_mask, tol=TOL)

    b10 = scr["budgets"][0.10]
    b20 = scr["budgets"][0.20]
    b30 = scr["budgets"][0.30]

    # Candidate-level bracket tightness stats
    R_c = R[cand_mask]
    T_c = T[cand_mask]
    M_c = M[cand_mask]
    H_c = H[cand_mask]
    abbe_c = abbe[cand_mask]
    anc_c = anc[cand_mask]
    lu_c = loc_up[cand_mask]
    pi_c = pi_hat[cand_mask]
    W_c = W[cand_mask]

    scr_f10_sec = float(b10["total_screening_time_sec"])
    scr_f20_sec = float(b20["total_screening_time_sec"])
    scr_f30_sec = float(b30["total_screening_time_sec"])

    n_valid_pairs = int(rev_R["n_valid_pairs"])
    n_strict_pairs = int(rev_R["n_strict_model_pairs"])
    n_unres_ref_pairs = int(rev_R["n_unresolved_ref_pairs"])
    n_unres_strict_pairs = int(rev_R["n_unresolved_strict_model_pairs"])

    unres_pair_frac = (
        float(n_unres_ref_pairs / n_valid_pairs) if n_valid_pairs > 0 else float("nan")
    )
    unres_strict_frac = (
        float(n_unres_strict_pairs / n_strict_pairs)
        if n_strict_pairs > 0
        else float("nan")
    )

    st = sc.metadata.get("structural", {})

    row: dict[str, Any] = {
        "scenario_id": sc.scenario_id,
        "suite": sc.suite,
        "split": sc.split,
        "family": sc.family,
        "size": sc.size,
        "n_edges": dag.n_edges,
        "n_candidates": n_cand,
        "graph_seed": sc.graph_seed,
        "weight_regime": sc.weight_regime,
        "weight_value": sc.weight_value,
        "b_setting": sc.b_setting,
        "mc_seed": sc.mc_seed,
        "mc_trials": mc_trials,
        "mc_backend": mc_backend,
        "hoeffding_h": h,
        "n_empty_intersections": n_empty,
        "n_empty_intersections_candidates": n_empty_cand,
        "depth": st.get("depth"),
        "reachable_fraction_candidates": st.get("reachable_fraction_candidates"),
        "overlap_candidate_fraction": st.get("overlap_candidate_fraction"),
        "layer1_source_coverage": st.get("layer1_source_coverage"),
        "layer2_source_coverage": st.get("layer2_source_coverage"),
        # Bracket tightness on C
        "W_mean": float(np.mean(W_c)),
        "W_max": float(np.max(W_c)),
        "W_p90": float(np.percentile(W_c, 90.0)),
        "top_k_boundary_W_mean_f10": float(b10["top_k_boundary_w_mean"]),
        "top_k_boundary_W_mean_f20": float(b20["top_k_boundary_w_mean"]),
        "top_k_boundary_W_mean_f30": float(b30["top_k_boundary_w_mean"]),
        "tree_width_mean": float(np.mean(R_c - T_c)),
        "tree_width_max": float(np.max(R_c - T_c)),
        "upper_slack_R_minus_pi_mean": float(np.mean(R_c - pi_c)),
        "upper_slack_R_minus_pi_max": float(np.max(R_c - pi_c)),
        "lower_slack_pi_minus_T_mean": float(np.mean(pi_c - T_c)),
        "lower_slack_pi_minus_M_mean": float(np.mean(pi_c - M_c)),
        "lower_slack_pi_minus_abbe_mean": float(np.mean(pi_c - abbe_c)),
        "mae_R_vs_pi": float(np.mean(np.abs(R_c - pi_c))),
        "mae_T_vs_pi": float(np.mean(np.abs(T_c - pi_c))),
        "mae_M_vs_pi": float(np.mean(np.abs(M_c - pi_c))),
        "mae_H_vs_pi": float(np.mean(np.abs(H_c - pi_c))),
        "mae_abbe_vs_pi": float(np.mean(np.abs(abbe_c - pi_c))),
        "mae_ancestry_vs_pi": float(np.mean(np.abs(anc_c - pi_c))),
        "mae_local_upstream_vs_pi": float(np.mean(np.abs(lu_c - pi_c))),
        # Saturation & score groups on C
        "sat_095_R": float(sat_R["fraction_ge_095"]),
        "sat_num_R": float(sat_R["numerical_saturation_fraction"]),
        "groups_R": int(sat_R["n_distinct_score_groups"]),
        "sat_095_H": float(sat_H["fraction_ge_095"]),
        "sat_num_H": float(sat_H["numerical_saturation_fraction"]),
        "groups_H": int(sat_H["n_distinct_score_groups"]),
        "sat_095_pi": float(sat_pi["fraction_ge_095"]),
        "sat_num_pi": float(sat_pi["numerical_saturation_fraction"]),
        "groups_pi": int(sat_pi["n_distinct_score_groups"]),
        # Tie-aware Spearman on C
        "spearman_R": float(spearmans["R"]),
        "spearman_T": float(spearmans["T"]),
        "spearman_M": float(spearmans["M"]),
        "spearman_H": float(spearmans["H"]),
        "spearman_local_upstream": float(spearmans["local_upstream"]),
        "spearman_abbe": float(spearmans["abbe"]),
        "spearman_ancestry": float(spearmans["ancestry"]),
        "spearman_W_vs_slack": float(w_rel["spearman_w_vs_slack"]),
    }

    # Fixed-budget capture & regret across all 7 comparators and all 3 budgets
    for name in COMPARATORS:
        for frac, tag in BUDGET_TAGS:
            row[f"capture_{name}_{tag}"] = float(caps[name]["budgets"][frac]["capture"])
            row[f"regret_{name}_{tag}"] = float(caps[name]["budgets"][frac]["regret"])
    for frac, tag in BUDGET_TAGS:
        row[f"regret_R_{tag}_resolved"] = bool(caps["R"]["budgets"][frac]["regret_resolved"])

    # Pairwise reversals
    row.update(
        {
            "n_total_pairs": int(rev_R["n_total_pairs"]),
            "n_valid_pairs": n_valid_pairs,
            "n_reversals_R": int(rev_R["n_reversals"]),
            "n_resolved_ref_pairs_R": int(rev_R["n_resolved_ref_pairs"]),
            "n_unresolved_ref_pairs_R": n_unres_ref_pairs,
            "n_resolved_strict_pairs_R": int(rev_R["n_resolved_strict_model_pairs"]),
            "n_unresolved_strict_pairs_R": n_unres_strict_pairs,
            "n_strict_model_pairs_R": n_strict_pairs,
            "resolved_fraction_strict_R": float(rev_R["resolved_fraction_of_strict_model"]),
            "unresolved_pair_fraction_R": unres_pair_frac,
            "unresolved_strict_fraction_R": unres_strict_frac,
            "reversal_rate_R": float(rev_R["reversal_rate"]),
        }
    )

    # Two-Stage Certified Top-k Screening across f10, f20, f30
    for b_entry, tag in ((b10, "f10"), (b20, "f20"), (b30, "f30")):
        row[f"res_s1_{tag}"] = float(b_entry["res_s1"])
        row[f"res_s2_{tag}"] = float(b_entry["res_s2"])
        row[f"band_reduction_{tag}"] = float(b_entry["band_reduction"])
        row[f"full_cert_s1_{tag}"] = bool(b_entry["fully_certified_s1"])
        row[f"full_cert_s2_{tag}"] = bool(b_entry["fully_certified_s2"])
        row[f"n_I_1_{tag}"] = int(b_entry["n_I_1"])
        row[f"n_O_1_{tag}"] = int(b_entry["n_O_1"])
        row[f"n_U_1_{tag}"] = int(b_entry["n_U_1"])
        row[f"n_I_2_{tag}"] = int(b_entry["n_I_2"])
        row[f"n_O_2_{tag}"] = int(b_entry["n_O_2"])
        row[f"n_U_2_{tag}"] = int(b_entry["n_U_2"])
        row[f"n_stage2_trees_{tag}"] = int(b_entry["n_stage2_trees"])

    # Timings & paired speedups
    row.update(
        {
            "mc_sec": mc_sec,
            "gbbrpm_sec": float(timings["gbbrpm"]),
            "recursive_max_sec": float(timings["recursive_max"]),
            "full_tree_sec": full_tree_sec,
            "abbe_sec": float(timings["abbe_lower"]),
            "ancestry_sec": float(timings["ancestry_diagnostic"]),
            "screening_f10_sec": scr_f10_sec,
            "screening_f20_sec": scr_f20_sec,
            "screening_f30_sec": scr_f30_sec,
            "screening_s1_f20_sec": float(b20["stage1_time_sec"]),
            "screening_s2_f20_sec": float(b20["stage2_time_sec"]),
            "speedup_vs_full_T_f10": (
                full_tree_sec / scr_f10_sec if scr_f10_sec > 0 else float("nan")
            ),
            "speedup_vs_full_T_f20": (
                full_tree_sec / scr_f20_sec if scr_f20_sec > 0 else float("nan")
            ),
            "speedup_vs_full_T_f30": (
                full_tree_sec / scr_f30_sec if scr_f30_sec > 0 else float("nan")
            ),
            "speedup_vs_mc_f10": (
                mc_sec / scr_f10_sec if scr_f10_sec > 0 else float("nan")
            ),
            "speedup_vs_mc_f20": (
                mc_sec / scr_f20_sec if scr_f20_sec > 0 else float("nan")
            ),
            "speedup_vs_mc_f30": (
                mc_sec / scr_f30_sec if scr_f30_sec > 0 else float("nan")
            ),
        }
    )

    # Express 4.18.2 root (npm:express@4.18.2), 7-advisory-node, and all-71-node metrics
    if sc.suite == "express":
        all_mask = np.ones(dag.n_nodes, dtype=np.bool_)
        adv_mask = dag.B > 0.0
        root_idx = dag.node_to_idx["npm:express@4.18.2"]
        cap_all_R = compute_capture_and_regret(
            R,
            pi_hat,
            all_mask,
            fractions=BUDGET_FRACTIONS,
            seed=TIE_PERM_SEED,
            n_perms=TIE_PERM_COUNT,
            tol=TOL,
            h=h,
        )
        row.update(
            {
                "express_root_M": float(M[root_idx]),
                "express_root_T": float(T[root_idx]),
                "express_root_pi_hat": float(pi_hat[root_idx]),
                "express_root_R": float(R[root_idx]),
                "express_root_H": float(H[root_idx]),
                "express_root_abbe": float(abbe[root_idx]),
                "express_root_W": float(W[root_idx]),
                "express_root_tree_width": float(R[root_idx] - T[root_idx]),
                "express_root_upper_slack": float(R[root_idx] - pi_hat[root_idx]),
                "advisory7_W_mean": float(np.mean(W[adv_mask])),
                "advisory7_tree_width_mean": float(np.mean((R - T)[adv_mask])),
                "advisory7_upper_slack_mean": float(np.mean((R - pi_hat)[adv_mask])),
                "all71_W_mean": float(np.mean(W)),
                "all71_tree_width_mean": float(np.mean(R - T)),
                "all71_upper_slack_mean": float(np.mean(R - pi_hat)),
                "all71_spearman_R": float(
                    compute_tie_aware_spearman(R, pi_hat, all_mask, tol=TOL)
                ),
                "all71_regret_R_f20": float(cap_all_R["budgets"][0.20]["regret"]),
            }
        )

    return idx, row


def _evaluate_precision_audit_worker(
    task: tuple[int, BenchmarkScenario]
) -> tuple[int, dict[str, Any]]:
    """Evaluate one independent 1,000,000-trial precision audit paired with its 100k solve."""
    idx, sc = task
    _assert_eval_scenario(sc)
    dag = sc.dag
    cand_mask = dag.B == 0.0
    n_cand = int(np.sum(cand_mask))

    seed_100k = int(sc.mc_seed)
    seed_1m = derive_scenario_seed(f"{sc.scenario_id}|precision_audit_1m")
    assert seed_1m != seed_100k, f"Audit seed collided with 100k seed on {sc.scenario_id}"

    # 1. Run primary 100k MC (identical seed & batch_size to Suite 1)
    mc_100k = run_mc_reference(
        dag,
        n_trials=MC_TRIALS_PRIMARY,
        seed=seed_100k,
        batch_size=MC_BATCH_SIZE,
        alpha=HOEFFDING_ALPHA,
    )
    # 2. Run independent 1M MC reference
    mc_1m = run_mc_reference(
        dag,
        n_trials=MC_TRIALS_AUDIT,
        seed=seed_1m,
        batch_size=MC_BATCH_SIZE,
        alpha=HOEFFDING_ALPHA,
    )

    # 3. Deterministic bounds & Two-Stage Certified Screening
    comp = run_comparators(
        dag, include_trees=True, include_abbe=False, include_ancestry=False
    )
    M = comp["recursive_max"]
    T = comp["tree_bound"]
    R = comp["gbbrpm"]
    H = comp["clipped_sum"]
    W = comp["bracket_width"]

    _assert_deterministic_bracket(M, T, R, H, f"{sc.scenario_id}|precision_audit_1m")

    pi_100k = mc_100k["pi_hat"]
    lo_100k = mc_100k["lower"]
    hi_100k = mc_100k["upper"]
    h_100k = float(mc_100k["hoeffding_h"])

    pi_1m = mc_1m["pi_hat"]
    lo_1m = mc_1m["lower"]
    hi_1m = mc_1m["upper"]
    h_1m = float(mc_1m["hoeffding_h"])

    L_100k, U_100k, empty_100k = intersect_intervals(lo_100k, hi_100k, T, R)
    L_1m, U_1m, empty_1m = intersect_intervals(lo_1m, hi_1m, T, R)
    n_empty_100k = int(np.sum(empty_100k))
    n_empty_1m = int(np.sum(empty_1m))
    n_empty_cand_1m = int(np.sum(empty_1m & cand_mask))

    scr = certify_top_k_two_stage(dag, fractions=BUDGET_FRACTIONS, epsilon=TOL)
    assert_zero_false_certifications(scr, pi_lower=T, pi_upper=R, epsilon=TOL)
    if n_empty_100k == 0:
        assert_zero_false_certifications(
            scr, pi_lower=L_100k, pi_upper=U_100k, epsilon=TOL
        )
    if n_empty_1m == 0:
        assert_zero_false_certifications(
            scr, pi_lower=L_1m, pi_upper=U_1m, epsilon=TOL
        )

    abs_diff = np.abs(pi_100k - pi_1m)
    abs_diff_cand = abs_diff[cand_mask]

    inside_1m_ci = (pi_100k >= lo_1m - TOL) & (pi_100k <= hi_1m + TOL)
    inside_100k_ci = (pi_1m >= lo_100k - TOL) & (pi_1m <= hi_100k + TOL)
    ci_overlap = np.maximum(lo_100k, lo_1m) <= np.minimum(hi_100k, hi_1m) + TOL
    within_combined_h = abs_diff <= (h_100k + h_1m + TOL)

    rev_100k = compute_pairwise_reversals(R, L_100k, U_100k, cand_mask, eps=TOL)
    rev_1m = compute_pairwise_reversals(R, L_1m, U_1m, cand_mask, eps=TOL)

    cap_100k = compute_capture_and_regret(
        R,
        pi_100k,
        cand_mask,
        fractions=BUDGET_FRACTIONS,
        seed=TIE_PERM_SEED,
        n_perms=TIE_PERM_COUNT,
        tol=TOL,
        h=h_100k,
    )
    cap_1m = compute_capture_and_regret(
        R,
        pi_1m,
        cand_mask,
        fractions=BUDGET_FRACTIONS,
        seed=TIE_PERM_SEED,
        n_perms=TIE_PERM_COUNT,
        tol=TOL,
        h=h_1m,
    )

    sp_100k = compute_tie_aware_spearman(R, pi_100k, cand_mask, tol=TOL)
    sp_1m = compute_tie_aware_spearman(R, pi_1m, cand_mask, tol=TOL)

    regret_100k = float(cap_100k["budgets"][0.20]["regret"])
    regret_1m = float(cap_1m["budgets"][0.20]["regret"])

    row: dict[str, Any] = {
        "scenario_id": sc.scenario_id,
        "family": sc.family,
        "size": sc.size,
        "n_edges": dag.n_edges,
        "n_candidates": n_cand,
        "graph_seed": sc.graph_seed,
        "weight_regime": sc.weight_regime,
        "weight_value": sc.weight_value,
        "b_setting": sc.b_setting,
        "mc_seed_100k": seed_100k,
        "mc_seed_1m": seed_1m,
        "mc_trials_100k": MC_TRIALS_PRIMARY,
        "mc_trials_1m": MC_TRIALS_AUDIT,
        "max_abs_diff_100k_vs_1m": float(np.max(abs_diff)),
        "mean_abs_diff_100k_vs_1m": float(np.mean(abs_diff)),
        "max_abs_diff_cand_100k_vs_1m": float(np.max(abs_diff_cand)),
        "mean_abs_diff_cand_100k_vs_1m": float(np.mean(abs_diff_cand)),
        "h_100k": h_100k,
        "h_1m": h_1m,
        "pct_nodes_100k_inside_1m_ci": float(100.0 * np.mean(inside_1m_ci)),
        "frac_nodes_100k_inside_1m_ci": float(np.mean(inside_1m_ci)),
        "pct_cands_100k_inside_1m_ci": float(100.0 * np.mean(inside_1m_ci[cand_mask])),
        "pct_nodes_1m_inside_100k_ci": float(100.0 * np.mean(inside_100k_ci)),
        "pct_nodes_ci_overlap_100k_1m": float(100.0 * np.mean(ci_overlap)),
        "pct_nodes_100k_within_combined_h": float(100.0 * np.mean(within_combined_h)),
        "empty_hoeffding_intersections_100k": n_empty_100k,
        "empty_hoeffding_intersections_1m": n_empty_1m,
        "empty_hoeffding_intersections_cand_1m": n_empty_cand_1m,
        "n_strict_model_pairs": int(rev_1m["n_strict_model_pairs"]),
        "n_resolved_pairs_100k": int(rev_100k["n_resolved_strict_model_pairs"]),
        "n_resolved_pairs_1m": int(rev_1m["n_resolved_strict_model_pairs"]),
        "n_resolved_ref_pairs_100k": int(rev_100k["n_resolved_ref_pairs"]),
        "n_resolved_ref_pairs_1m": int(rev_1m["n_resolved_ref_pairs"]),
        "resolved_fraction_strict_100k": float(
            rev_100k["resolved_fraction_of_strict_model"]
        ),
        "resolved_fraction_strict_1m": float(
            rev_1m["resolved_fraction_of_strict_model"]
        ),
        "n_reversals_100k": int(rev_100k["n_reversals"]),
        "n_reversals_1m": int(rev_1m["n_reversals"]),
        "reversal_rate_100k": float(rev_100k["reversal_rate"]),
        "reversal_rate_1m": float(rev_1m["reversal_rate"]),
        "regret_R_f20_100k": regret_100k,
        "regret_R_f20_1m": regret_1m,
        "abs_diff_regret_R_f20": abs(regret_100k - regret_1m),
        "capture_R_f20_100k": float(cap_100k["budgets"][0.20]["capture"]),
        "capture_R_f20_1m": float(cap_1m["budgets"][0.20]["capture"]),
        "spearman_R_100k": float(sp_100k),
        "spearman_R_1m": float(sp_1m),
        "W_mean": float(np.mean(W[cand_mask])),
        "upper_slack_R_minus_pi_100k": float(np.mean(R[cand_mask] - pi_100k[cand_mask])),
        "upper_slack_R_minus_pi_1m": float(np.mean(R[cand_mask] - pi_1m[cand_mask])),
        "false_certifications_100k": 0,
        "false_certifications_1m": 0,
        "mc_100k_sec": float(mc_100k["elapsed_sec"]),
        "mc_1m_sec": float(mc_1m["elapsed_sec"]),
    }
    return idx, row


# ---------------------------------------------------------------------------
# Parallel execution harness (shared ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _run_primary_suite(
    scenarios: list[BenchmarkScenario],
    label: str,
    executor: ProcessPoolExecutor | None,
    mc_trials: int = MC_TRIALS_PRIMARY,
    batch_size: int = MC_BATCH_SIZE,
) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    n = len(scenarios)
    tasks = [(i, sc, mc_trials, batch_size) for i, sc in enumerate(scenarios)]
    rows_ordered: list[dict[str, Any] | None] = [None] * n

    if executor is None:
        for task in tasks:
            idx, row = _evaluate_primary_scenario_worker(task)
            rows_ordered[idx] = row
    else:
        completed = 0
        report_every = max(1, min(300, n // 4 if n >= 40 else n))
        futures = [
            executor.submit(_evaluate_primary_scenario_worker, task) for task in tasks
        ]
        for fut in as_completed(futures):
            idx, row = fut.result()
            rows_ordered[idx] = row
            completed += 1
            if completed % report_every == 0 or completed == n:
                elapsed_now = time.perf_counter() - t0
                print(
                    f"  [{label}] {completed}/{n} completed ({elapsed_now:.1f}s)",
                    flush=True,
                )

    assert all(r is not None for r in rows_ordered)
    df = pd.DataFrame(rows_ordered)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    return df, elapsed


def _run_precision_audit_suite(
    scenarios: list[BenchmarkScenario],
    executor: ProcessPoolExecutor | None,
) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    n = len(scenarios)
    tasks = [(i, sc) for i, sc in enumerate(scenarios)]
    rows_ordered: list[dict[str, Any] | None] = [None] * n

    if executor is None:
        for task in tasks:
            idx, row = _evaluate_precision_audit_worker(task)
            rows_ordered[idx] = row
    else:
        completed = 0
        futures = [
            executor.submit(_evaluate_precision_audit_worker, task) for task in tasks
        ]
        for fut in as_completed(futures):
            idx, row = fut.result()
            rows_ordered[idx] = row
            completed += 1
            if completed % 12 == 0 or completed == n:
                elapsed_now = time.perf_counter() - t0
                print(
                    f"  [Suite 4 Precision Audits 1M] {completed}/{n} completed ({elapsed_now:.1f}s)",
                    flush=True,
                )

    assert all(r is not None for r in rows_ordered)
    df = pd.DataFrame(rows_ordered)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    return df, elapsed


# ---------------------------------------------------------------------------
# Graph-replicate-weighted statistical aggregation
# ---------------------------------------------------------------------------


def _dist_stats(values: pd.Series | np.ndarray | list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    n = int(valid.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "median": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    q25, med, q75 = np.percentile(valid, [25.0, 50.0, 75.0])
    return {
        "n": n,
        "mean": float(np.mean(valid)),
        "sd": float(np.std(valid, ddof=1)) if n > 1 else 0.0,
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def _finite_mean(values: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if valid.size > 0 else None


def _finite_median(values: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.median(valid)) if valid.size > 0 else None


def _replicate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse scenarios within independent graph replicates ``(family, size, graph_seed)``.

    Every synthetic DAG replicate has multiple (weight, b_setting) conditions on the
    SAME topology; averaging within each graph replicate first ensures SD, median,
    and IQR reflect independent graph replicates (20 per family x size in Suite 1;
    5 per family x size in Suite 2) rather than pseudo-replicated conditions.
    """
    if "graph_seed" not in df.columns or df["graph_seed"].isna().all():
        return df.copy()

    numeric_cols = [
        c
        for c in df.columns
        if c not in ("family", "size", "graph_seed")
        and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    rep = (
        df.groupby(["family", "size", "graph_seed"], sort=True, as_index=False)[
            numeric_cols
        ]
        .mean(numeric_only=True)
    )
    return rep


def _summarize_stratum(
    df: pd.DataFrame,
    include_paired_diffs: bool = True,
) -> dict[str, Any]:
    """Build a graph-replicate-weighted summary block for any slice of a DAG DataFrame."""
    rep = _replicate_frame(df)
    n_scenarios = int(len(df))
    n_replicates = int(len(rep))

    bracket_tightness = {
        "W_mean": _dist_stats(rep["W_mean"]),
        "W_max": _dist_stats(rep["W_max"]),
        "W_p90": _dist_stats(rep["W_p90"]),
        "tree_width_mean": _dist_stats(rep["tree_width_mean"]),
        "upper_slack_R_minus_pi_mean": _dist_stats(rep["upper_slack_R_minus_pi_mean"]),
        "lower_slack_pi_minus_T_mean": _dist_stats(rep["lower_slack_pi_minus_T_mean"]),
        "lower_slack_pi_minus_M_mean": _dist_stats(rep["lower_slack_pi_minus_M_mean"]),
        "lower_slack_pi_minus_abbe_mean": _dist_stats(
            rep["lower_slack_pi_minus_abbe_mean"]
        ),
        "mae_R_vs_pi": _dist_stats(rep["mae_R_vs_pi"]),
        "mae_T_vs_pi": _dist_stats(rep["mae_T_vs_pi"]),
        "mae_abbe_vs_pi": _dist_stats(rep["mae_abbe_vs_pi"]),
        "max_W_scenario": float(df["W_max"].max()) if n_scenarios > 0 else None,
        "max_upper_slack_scenario": (
            float(df["upper_slack_R_minus_pi_max"].max()) if n_scenarios > 0 else None
        ),
    }

    screening: dict[str, Any] = {}
    for _, tag in BUDGET_TAGS:
        scr_sec_col = f"screening_{tag}_sec"
        tot_full_T = float(df["full_tree_sec"].sum())
        tot_mc = float(df["mc_sec"].sum())
        tot_scr = float(df[scr_sec_col].sum())
        screening[tag] = {
            "res_s1": _dist_stats(rep[f"res_s1_{tag}"]),
            "res_s2": _dist_stats(rep[f"res_s2_{tag}"]),
            "band_reduction": _dist_stats(rep[f"band_reduction_{tag}"]),
            "full_cert_s1_rate": _dist_stats(rep[f"full_cert_s1_{tag}"]),
            "full_cert_s2_rate": _dist_stats(rep[f"full_cert_s2_{tag}"]),
            "n_stage2_trees": _dist_stats(rep[f"n_stage2_trees_{tag}"]),
            "speedup_vs_full_T": _dist_stats(rep[f"speedup_vs_full_T_{tag}"]),
            "speedup_vs_mc": _dist_stats(rep[f"speedup_vs_mc_{tag}"]),
            "scenario_median_speedup_vs_full_T": _finite_median(
                df[f"speedup_vs_full_T_{tag}"]
            ),
            "scenario_median_speedup_vs_mc": _finite_median(
                df[f"speedup_vs_mc_{tag}"]
            ),
            "aggregate_speedup_vs_full_T": (
                tot_full_T / tot_scr if tot_scr > 0 else None
            ),
            "aggregate_speedup_vs_mc": tot_mc / tot_scr if tot_scr > 0 else None,
            "total_full_tree_sec": tot_full_T,
            "total_screening_sec": tot_scr,
            "total_mc_sec": tot_mc,
        }

    ranking_by_budget: dict[str, Any] = {}
    for _, tag in BUDGET_TAGS:
        regret_dict = {
            name: _dist_stats(rep[f"regret_{name}_{tag}"]) for name in COMPARATORS
        }
        capture_dict = {
            name: _dist_stats(rep[f"capture_{name}_{tag}"]) for name in COMPARATORS
        }
        ranking_by_budget[tag] = {
            "regret": regret_dict,
            "capture": capture_dict,
            "regret_R_resolved_rate": float(df[f"regret_R_{tag}_resolved"].mean()),
        }

    spearman_dict = {name: _dist_stats(rep[f"spearman_{name}"]) for name in COMPARATORS}

    saturation = {
        "sat_095_R": _dist_stats(rep["sat_095_R"]),
        "sat_num_R": _dist_stats(rep["sat_num_R"]),
        "groups_R": _dist_stats(rep["groups_R"]),
        "sat_095_H": _dist_stats(rep["sat_095_H"]),
        "sat_num_H": _dist_stats(rep["sat_num_H"]),
        "groups_H": _dist_stats(rep["groups_H"]),
        "sat_095_pi": _dist_stats(rep["sat_095_pi"]),
        "sat_num_pi": _dist_stats(rep["sat_num_pi"]),
        "groups_pi": _dist_stats(rep["groups_pi"]),
    }

    tot_rev = int(df["n_reversals_R"].sum())
    tot_res_strict = int(df["n_resolved_strict_pairs_R"].sum())
    tot_strict = int(df["n_strict_model_pairs_R"].sum())
    tot_valid = int(df["n_valid_pairs"].sum())
    tot_unres_ref = int(df["n_unresolved_ref_pairs_R"].sum())
    tot_unres_strict = int(df["n_unresolved_strict_pairs_R"].sum())

    reversals = {
        "scenarios_with_reversals": int((df["n_reversals_R"] > 0).sum()),
        "scenario_fraction_with_reversals": float((df["n_reversals_R"] > 0).mean()),
        "total_reversals_R": tot_rev,
        "total_resolved_strict_pairs_R": tot_res_strict,
        "total_strict_model_pairs_R": tot_strict,
        "total_valid_pairs": tot_valid,
        "total_unresolved_ref_pairs_R": tot_unres_ref,
        "total_unresolved_strict_pairs_R": tot_unres_strict,
        "pooled_reversal_rate_R": (
            float(tot_rev / tot_res_strict) if tot_res_strict > 0 else None
        ),
        "pooled_resolved_fraction_strict_R": (
            float(tot_res_strict / tot_strict) if tot_strict > 0 else None
        ),
        "pooled_unresolved_pair_fraction_R": (
            float(tot_unres_ref / tot_valid) if tot_valid > 0 else None
        ),
        "pooled_unresolved_strict_fraction_R": (
            float(tot_unres_strict / tot_strict) if tot_strict > 0 else None
        ),
        "replicate_reversal_rate_R": _dist_stats(rep["reversal_rate_R"]),
        "replicate_unresolved_pair_fraction_R": _dist_stats(
            rep["unresolved_pair_fraction_R"]
        ),
        "replicate_resolved_fraction_strict_R": _dist_stats(
            rep["resolved_fraction_strict_R"]
        ),
    }

    out: dict[str, Any] = {
        "n_scenarios": n_scenarios,
        "n_graph_replicates": n_replicates,
        "empty_hoeffding_intersections_total_nodes": int(
            df["n_empty_intersections"].sum()
        ),
        "empty_hoeffding_intersections_candidate_nodes": int(
            df["n_empty_intersections_candidates"].sum()
        ),
        "scenarios_with_empty_hoeffding_intersections": int(
            (df["n_empty_intersections"] > 0).sum()
        ),
        "bracket_tightness": bracket_tightness,
        "certified_screening": screening,
        "ranking_by_budget": ranking_by_budget,
        "spearman": spearman_dict,
        "saturation": saturation,
        "reversals": reversals,
    }

    if include_paired_diffs:
        paired: dict[str, Any] = {"regret": {}, "capture": {}, "spearman": {}}
        for _, tag in BUDGET_TAGS:
            paired["regret"][tag] = {}
            paired["capture"][tag] = {}
            for other in PAIRED_COMP_NAMES:
                key = f"R_minus_{other}"
                reg_diff = rep[f"regret_R_{tag}"] - rep[f"regret_{other}_{tag}"]
                cap_diff = rep[f"capture_R_{tag}"] - rep[f"capture_{other}_{tag}"]
                reg_stats = _dist_stats(reg_diff)
                cap_stats = _dist_stats(cap_diff)
                reg_valid = reg_diff[np.isfinite(reg_diff)]
                cap_valid = cap_diff[np.isfinite(cap_diff)]
                reg_stats["fraction_replicates_R_lower_regret"] = (
                    float(np.mean(reg_valid < -TOL)) if len(reg_valid) > 0 else None
                )
                reg_stats["fraction_replicates_R_leq_regret"] = (
                    float(np.mean(reg_valid <= TOL)) if len(reg_valid) > 0 else None
                )
                cap_stats["fraction_replicates_R_higher_capture"] = (
                    float(np.mean(cap_valid > TOL)) if len(cap_valid) > 0 else None
                )
                paired["regret"][tag][key] = reg_stats
                paired["capture"][tag][key] = cap_stats
        for other in PAIRED_COMP_NAMES:
            key = f"R_minus_{other}"
            sp_diff = rep["spearman_R"] - rep[f"spearman_{other}"]
            sp_stats = _dist_stats(sp_diff)
            sp_valid = sp_diff[np.isfinite(sp_diff)]
            sp_stats["fraction_replicates_R_higher_spearman"] = (
                float(np.mean(sp_valid > TOL)) if len(sp_valid) > 0 else None
            )
            paired["spearman"][key] = sp_stats
        out["paired_method_differences_replicate"] = paired

    return out


def _summarize_dag_suite(
    df: pd.DataFrame,
    suite_name: str,
    include_breakdowns: bool = True,
) -> dict[str, Any]:
    """Comprehensive graph-replicate-weighted summary for Suite 1 or Suite 2."""
    overall = _summarize_stratum(df, include_paired_diffs=True)

    w_screen_reversals = evaluate_scenario_w_screening(
        w_indicator=df["W_max"].to_numpy(dtype=np.float64),
        positive_flag=(df["n_reversals_R"] > 0).to_numpy(dtype=np.bool_),
        continuous_target=df["reversal_rate_R"].fillna(0.0).to_numpy(dtype=np.float64),
    )
    w_screen_regret = evaluate_scenario_w_screening(
        w_indicator=df["W_mean"].to_numpy(dtype=np.float64),
        positive_flag=(df["regret_R_f20"] > 1e-6).to_numpy(dtype=np.bool_),
        continuous_target=df["regret_R_f20"].to_numpy(dtype=np.float64),
    )
    w_screen_regret_boundary = evaluate_scenario_w_screening(
        w_indicator=df["top_k_boundary_W_mean_f20"].to_numpy(dtype=np.float64),
        positive_flag=(df["regret_R_f20"] > 1e-6).to_numpy(dtype=np.bool_),
        continuous_target=df["regret_R_f20"].to_numpy(dtype=np.float64),
    )
    rep = _replicate_frame(df)
    w_reliability = {
        "nodewise_spearman_W_vs_slack_scenario": _dist_stats(df["spearman_W_vs_slack"]),
        "nodewise_spearman_W_vs_slack_replicate": _dist_stats(rep["spearman_W_vs_slack"]),
        "scenario_screening_reversals_by_W_max": w_screen_reversals,
        "scenario_screening_regret_f20_by_W_mean": w_screen_regret,
        "scenario_screening_regret_f20_by_boundary_W": w_screen_regret_boundary,
    }

    empty_rows = df[df["n_empty_intersections"] > 0]
    empty_records = [
        {
            "scenario_id": str(r["scenario_id"]),
            "n_empty_intersections": int(r["n_empty_intersections"]),
            "n_empty_intersections_candidates": int(
                r["n_empty_intersections_candidates"]
            ),
        }
        for _, r in empty_rows.iterrows()
    ]

    summary: dict[str, Any] = {
        "suite": suite_name,
        "mc_trials": int(df["mc_trials"].iloc[0]),
        "bracket_violations": 0,
        "false_certifications_analytic": 0,
        "false_certifications_interval": 0,
        "empty_hoeffding_records": empty_records,
        "overall": overall,
        "w_reliability": w_reliability,
    }

    if include_breakdowns:
        by_family: dict[str, Any] = {}
        for fam, grp in df.groupby("family", sort=True):
            by_family[str(fam)] = _summarize_stratum(grp, include_paired_diffs=True)

        by_size: dict[str, Any] = {}
        for sz, grp in df.groupby("size", sort=True):
            by_size[str(int(sz))] = _summarize_stratum(grp, include_paired_diffs=True)

        by_weight: dict[str, Any] = {}
        for wreg, grp in df.groupby("weight_regime", sort=True):
            by_weight[str(wreg)] = _summarize_stratum(grp, include_paired_diffs=False)

        by_b_setting: dict[str, Any] = {}
        for bval, grp in df.groupby("b_setting", sort=True):
            by_b_setting[f"B={float(bval):.1f}"] = _summarize_stratum(
                grp, include_paired_diffs=False
            )

        by_family_size: dict[str, Any] = {}
        for (fam, sz), grp in df.groupby(["family", "size"], sort=True):
            by_family_size[f"{fam}|n={int(sz)}"] = _summarize_stratum(
                grp, include_paired_diffs=True
            )

        layered_coverage: dict[str, Any] = {}
        layered_sub = df[df["family"] == "layered"]
        for sz, grp in layered_sub.groupby("size", sort=True):
            layered_coverage[str(int(sz))] = {
                "layer1_source_coverage": _finite_mean(grp["layer1_source_coverage"]),
                "layer2_source_coverage": _finite_mean(grp["layer2_source_coverage"]),
                "reachable_fraction_candidates": _finite_mean(
                    grp["reachable_fraction_candidates"]
                ),
                "overlap_candidate_fraction": _finite_mean(
                    grp["overlap_candidate_fraction"]
                ),
            }

        summary["by_family"] = by_family
        summary["by_size"] = by_size
        summary["by_weight"] = by_weight
        summary["by_b_setting"] = by_b_setting
        summary["by_family_size"] = by_family_size
        summary["layered_source_coverage"] = layered_coverage

    return summary


def _summarize_paired_hetero_vs_uniform(
    hetero_df: pd.DataFrame, uniform_df: pd.DataFrame
) -> dict[str, Any]:
    """Compute paired deltas (hetero - uniform_w020) across all 120 matched scenarios."""
    unif_020 = uniform_df[
        (uniform_df["weight_regime"] == "w=0.20")
        & (uniform_df["graph_seed"].isin(HETERO_EVAL_SEEDS))
    ].copy()
    assert (
        len(unif_020) == 120
    ), f"Expected 120 matching w=0.20 uniform rows, got {len(unif_020)}"

    join_cols = ["family", "size", "graph_seed", "b_setting"]
    merged = pd.merge(
        hetero_df,
        unif_020,
        on=join_cols,
        suffixes=("_hetero", "_unif020"),
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == 120, f"Expected 120 paired rows, got {len(merged)}"

    metrics = [
        "W_mean",
        "tree_width_mean",
        "upper_slack_R_minus_pi_mean",
        "lower_slack_pi_minus_T_mean",
        "regret_R_f10",
        "regret_R_f20",
        "regret_R_f30",
        "regret_T_f20",
        "regret_M_f20",
        "regret_abbe_f20",
        "capture_R_f20",
        "spearman_R",
        "res_s1_f20",
        "res_s2_f20",
        "band_reduction_f20",
        "full_cert_s2_f20",
        "reversal_rate_R",
        "speedup_vs_full_T_f20",
        "speedup_vs_mc_f20",
    ]

    delta_cols = {
        f"delta_{m}": (
            merged[f"{m}_hetero"].astype(np.float64)
            - merged[f"{m}_unif020"].astype(np.float64)
        ).reset_index(drop=True)
        for m in metrics
    }
    delta_df = pd.concat(
        [merged[join_cols].reset_index(drop=True), pd.DataFrame(delta_cols)],
        axis=1,
    )

    def _agg_deltas(sub: pd.DataFrame) -> dict[str, Any]:
        rep_sub = (
            sub.groupby(["family", "size", "graph_seed"], sort=True, as_index=False)
            .mean(numeric_only=True)
        )
        return {
            "n_pairs": int(len(sub)),
            "n_graph_replicates": int(len(rep_sub)),
            "replicate_deltas": {
                m: _dist_stats(rep_sub[f"delta_{m}"]) for m in metrics
            },
            "pair_deltas": {m: _dist_stats(sub[f"delta_{m}"]) for m in metrics},
        }

    by_family = {
        str(fam): _agg_deltas(grp) for fam, grp in delta_df.groupby("family", sort=True)
    }
    by_size = {
        str(int(sz)): _agg_deltas(grp)
        for sz, grp in delta_df.groupby("size", sort=True)
    }
    by_b = {
        f"B={float(bv):.1f}": _agg_deltas(grp)
        for bv, grp in delta_df.groupby("b_setting", sort=True)
    }

    return {
        "n_matched_pairs": 120,
        "paired_uniform_weight": 0.20,
        "eval_seeds": list(HETERO_EVAL_SEEDS),
        "delta_definition": "hetero_v1 - uniform_w0.20",
        "overall": _agg_deltas(delta_df),
        "by_family": by_family,
        "by_size": by_size,
        "by_b_setting": by_b,
    }


def _summarize_express_suite(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize the 12 Express 4.18.2 real-topology stochastic scenarios."""
    overall = _summarize_stratum(df, include_paired_diffs=True)
    by_weight = {
        str(wreg): _summarize_stratum(grp, include_paired_diffs=False)
        for wreg, grp in df.groupby("weight_regime", sort=True)
    }
    by_b = {
        f"B={float(bv):.1f}": _summarize_stratum(grp, include_paired_diffs=False)
        for bv, grp in df.groupby("b_setting", sort=True)
    }
    empty_rows = df[df["n_empty_intersections"] > 0]
    empty_records = [
        {
            "scenario_id": str(r["scenario_id"]),
            "n_empty_intersections": int(r["n_empty_intersections"]),
            "n_empty_intersections_candidates": int(
                r["n_empty_intersections_candidates"]
            ),
        }
        for _, r in empty_rows.iterrows()
    ]

    b05 = df[df["b_setting"].sub(0.5).abs() < 1e-9]
    per_scenario_root = [
        {
            "scenario_id": str(r["scenario_id"]),
            "weight_value": float(r["weight_value"]),
            "b_setting": float(r["b_setting"]),
            "express_root_M": float(r["express_root_M"]),
            "express_root_T": float(r["express_root_T"]),
            "express_root_pi_hat": float(r["express_root_pi_hat"]),
            "express_root_R": float(r["express_root_R"]),
            "express_root_H": float(r["express_root_H"]),
            "express_root_abbe": float(r["express_root_abbe"]),
            "express_root_W": float(r["express_root_W"]),
            "express_root_tree_width": float(r["express_root_tree_width"]),
            "express_root_upper_slack": float(r["express_root_upper_slack"]),
            "advisory7_W_mean": float(r["advisory7_W_mean"]),
            "advisory7_tree_width_mean": float(r["advisory7_tree_width_mean"]),
            "advisory7_upper_slack_mean": float(r["advisory7_upper_slack_mean"]),
            "all71_spearman_R": float(r["all71_spearman_R"]),
            "all71_regret_R_f20": float(r["all71_regret_R_f20"]),
        }
        for _, r in df.iterrows()
    ]

    return {
        "suite": "express_4.18.2",
        "n_scenarios": int(len(df)),
        "n_nodes": int(df["size"].iloc[0]),
        "n_edges": int(df["n_edges"].iloc[0]),
        "n_candidates": int(df["n_candidates"].iloc[0]),
        "mc_trials": int(df["mc_trials"].iloc[0]),
        "bracket_violations": 0,
        "false_certifications_analytic": 0,
        "false_certifications_interval": 0,
        "empty_hoeffding_records": empty_records,
        "candidate_set_reachability_note": (
            "All 7 OSV advisory roots lie in the top-most dependent subgraph "
            "(cookie, path-to-regexp, qs, send -> body-parser, serve-static -> express@4.18.2); "
            "hence 0/64 non-advisory nodes (C = {B_j == 0}) are downstream of advisory roots, "
            "while non-trivial reconvergence and bracket widths occur across the 7 advisory nodes "
            "and root package under B = 0.5."
        ),
        "express_root_B05_summary": {
            "mean_root_W": float(b05["express_root_W"].mean()),
            "max_root_W": float(b05["express_root_W"].max()),
            "mean_root_tree_width": float(b05["express_root_tree_width"].mean()),
            "mean_root_upper_slack": float(b05["express_root_upper_slack"].mean()),
            "max_root_upper_slack": float(b05["express_root_upper_slack"].max()),
            "mean_advisory7_W": float(b05["advisory7_W_mean"].mean()),
        },
        "per_scenario_root_and_advisory": per_scenario_root,
        "overall": overall,
        "by_weight": by_weight,
        "by_b_setting": by_b,
    }


def _summarize_precision_audits(
    audit_df: pd.DataFrame, uniform_df: pd.DataFrame
) -> dict[str, Any]:
    """Cross-verify Suite 4 100k results against Suite 1 and summarize 100k vs 1M agreement."""
    merged = pd.merge(
        audit_df,
        uniform_df[
            [
                "scenario_id",
                "regret_R_f20",
                "n_reversals_R",
                "n_resolved_strict_pairs_R",
                "n_empty_intersections",
            ]
        ],
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == 48, f"Expected 48 Suite 1 matches for audits, got {len(merged)}"
    max_regret_parity = float(
        np.max(np.abs(merged["regret_R_f20_100k"] - merged["regret_R_f20"]))
    )
    if max_regret_parity > 1e-12:
        raise AssertionError(
            f"Determinism parity failure between Suite 1 and Suite 4 100k regret: {max_regret_parity:.3e}"
        )
    if not bool((merged["n_reversals_100k"] == merged["n_reversals_R"]).all()):
        raise AssertionError("Determinism parity failure on n_reversals_100k vs Suite 1")
    if not bool(
        (merged["n_resolved_pairs_100k"] == merged["n_resolved_strict_pairs_R"]).all()
    ):
        raise AssertionError(
            "Determinism parity failure on n_resolved_pairs_100k vs Suite 1"
        )

    def _agg_audit(sub: pd.DataFrame) -> dict[str, Any]:
        tot_strict = int(sub["n_strict_model_pairs"].sum())
        tot_res_100k = int(sub["n_resolved_pairs_100k"].sum())
        tot_res_1m = int(sub["n_resolved_pairs_1m"].sum())
        tot_rev_100k = int(sub["n_reversals_100k"].sum())
        tot_rev_1m = int(sub["n_reversals_1m"].sum())
        return {
            "n_scenarios": int(len(sub)),
            "max_abs_diff_100k_vs_1m": _dist_stats(sub["max_abs_diff_100k_vs_1m"]),
            "mean_abs_diff_100k_vs_1m": _dist_stats(sub["mean_abs_diff_100k_vs_1m"]),
            "max_abs_diff_cand_100k_vs_1m": _dist_stats(
                sub["max_abs_diff_cand_100k_vs_1m"]
            ),
            "h_100k": _dist_stats(sub["h_100k"]),
            "h_1m": _dist_stats(sub["h_1m"]),
            "pct_nodes_100k_inside_1m_ci": _dist_stats(
                sub["pct_nodes_100k_inside_1m_ci"]
            ),
            "pct_nodes_1m_inside_100k_ci": _dist_stats(
                sub["pct_nodes_1m_inside_100k_ci"]
            ),
            "pct_nodes_ci_overlap_100k_1m": _dist_stats(
                sub["pct_nodes_ci_overlap_100k_1m"]
            ),
            "pct_nodes_100k_within_combined_h": _dist_stats(
                sub["pct_nodes_100k_within_combined_h"]
            ),
            "empty_hoeffding_intersections_100k_total": int(
                sub["empty_hoeffding_intersections_100k"].sum()
            ),
            "empty_hoeffding_intersections_1m_total": int(
                sub["empty_hoeffding_intersections_1m"].sum()
            ),
            "total_strict_model_pairs": tot_strict,
            "total_resolved_pairs_100k": tot_res_100k,
            "total_resolved_pairs_1m": tot_res_1m,
            "pooled_resolved_fraction_100k": (
                float(tot_res_100k / tot_strict) if tot_strict > 0 else None
            ),
            "pooled_resolved_fraction_1m": (
                float(tot_res_1m / tot_strict) if tot_strict > 0 else None
            ),
            "resolved_pair_gain_1m_minus_100k": int(tot_res_1m - tot_res_100k),
            "total_reversals_100k": tot_rev_100k,
            "total_reversals_1m": tot_rev_1m,
            "pooled_reversal_rate_100k": (
                float(tot_rev_100k / tot_res_100k) if tot_res_100k > 0 else None
            ),
            "pooled_reversal_rate_1m": (
                float(tot_rev_1m / tot_res_1m) if tot_res_1m > 0 else None
            ),
            "regret_R_f20_100k": _dist_stats(sub["regret_R_f20_100k"]),
            "regret_R_f20_1m": _dist_stats(sub["regret_R_f20_1m"]),
            "abs_diff_regret_R_f20": _dist_stats(sub["abs_diff_regret_R_f20"]),
            "false_certifications_100k": 0,
            "false_certifications_1m": 0,
        }

    empty_1m_rows = audit_df[audit_df["empty_hoeffding_intersections_1m"] > 0]
    empty_1m_records = [
        {
            "scenario_id": str(r["scenario_id"]),
            "empty_hoeffding_intersections_1m": int(
                r["empty_hoeffding_intersections_1m"]
            ),
        }
        for _, r in empty_1m_rows.iterrows()
    ]

    return {
        "suite": "precision_audits_1m",
        "eval_seed": AUDIT_SEED,
        "weights": list(AUDIT_WEIGHTS),
        "mc_trials_100k": MC_TRIALS_PRIMARY,
        "mc_trials_1m": MC_TRIALS_AUDIT,
        "suite1_100k_determinism_parity_verified": True,
        "suite1_100k_max_regret_parity_err": max_regret_parity,
        "bracket_violations": 0,
        "false_certifications_100k": 0,
        "false_certifications_1m": 0,
        "empty_hoeffding_1m_records": empty_1m_records,
        "overall": _agg_audit(audit_df),
        "by_size": {
            str(int(sz)): _agg_audit(grp)
            for sz, grp in audit_df.groupby("size", sort=True)
        },
        "by_family": {
            str(fam): _agg_audit(grp)
            for fam, grp in audit_df.groupby("family", sort=True)
        },
        "by_weight": {
            str(wreg): _agg_audit(grp)
            for wreg, grp in audit_df.groupby("weight_regime", sort=True)
        },
    }


def assemble_and_write_summary(
    out_dir: Path,
    uniform_df: pd.DataFrame,
    hetero_df: pd.DataFrame,
    express_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    timings_sec: dict[str, float],
    workers: int,
) -> dict[str, Any]:
    """Build and write strict RFC-8259 ``heldout_eval_summary.json`` from the four suite DataFrames."""
    freeze_prov = _load_freeze_provenance()
    uniform_csv = out_dir / "heldout_uniform_2880.csv"
    hetero_csv = out_dir / "heldout_hetero_120.csv"
    express_csv = out_dir / "heldout_express_12.csv"
    audit_csv = out_dir / "heldout_precision_audits_48.csv"

    uniform_summary = _summarize_dag_suite(
        uniform_df, suite_name="heldout_uniform_2880", include_breakdowns=True
    )
    hetero_summary = _summarize_dag_suite(
        hetero_df, suite_name="heldout_hetero_120", include_breakdowns=True
    )
    paired_hetero = _summarize_paired_hetero_vs_uniform(hetero_df, uniform_df)
    express_summary = _summarize_express_suite(express_df)
    audit_summary = _summarize_precision_audits(audit_df, uniform_df)

    total_empty_primary = int(
        uniform_df["n_empty_intersections"].sum()
        + hetero_df["n_empty_intersections"].sum()
        + express_df["n_empty_intersections"].sum()
    )
    total_empty_audit_1m = int(audit_df["empty_hoeffding_intersections_1m"].sum())
    mc_backends = sorted(
        set(uniform_df["mc_backend"].unique())
        | set(hetero_df["mc_backend"].unique())
        | set(express_df["mc_backend"].unique())
    )

    summary_payload: dict[str, Any] = {
        "protocol": {
            "split": "heldout_eval",
            "protocol_freeze_manifest": freeze_prov,
            "uniform_eval_seeds": list(EVAL_SEEDS),
            "hetero_eval_seeds": list(HETERO_EVAL_SEEDS),
            "precision_audit_seed": AUDIT_SEED,
            "precision_audit_weights": list(AUDIT_WEIGHTS),
            "budget_fractions": list(BUDGET_FRACTIONS),
            "mc_trials_primary": MC_TRIALS_PRIMARY,
            "mc_trials_precision_audit": MC_TRIALS_AUDIT,
            "mc_batch_size": MC_BATCH_SIZE,
            "hoeffding_alpha": HOEFFDING_ALPHA,
            "tolerance_epsilon": TOL,
            "tie_permutation_seed": TIE_PERM_SEED,
            "tie_permutation_n_perms": TIE_PERM_COUNT,
            "empty_intersection_policy": "record_without_repair",
            "mc_backends_observed": mc_backends,
            "replicate_weighting_rule": (
                "Metrics are averaged across (weight, b_setting) conditions within each "
                "independent graph topology (family, size, graph_seed) before computing "
                "mean, SD, median, and IQR across independent graph replicates "
                "(20 per family x size in Suite 1; 5 per family x size in Suite 2)."
            ),
            "counts": {
                "heldout_uniform": int(len(uniform_df)),
                "heldout_hetero": int(len(hetero_df)),
                "heldout_express": int(len(express_df)),
                "primary_heldout_total": int(
                    len(uniform_df) + len(hetero_df) + len(express_df)
                ),
                "precision_audits_1m": int(len(audit_df)),
                "total_evaluations": int(
                    len(uniform_df) + len(hetero_df) + len(express_df) + len(audit_df)
                ),
            },
        },
        "hardware": _collect_hardware_metadata(workers=workers),
        "timings_sec": timings_sec,
        "global_assertions": {
            "total_evaluations_verified": int(
                len(uniform_df) + len(hetero_df) + len(express_df) + len(audit_df)
            ),
            "bracket_violations_total": 0,
            "false_certifications_analytic_total": 0,
            "false_certifications_interval_total": 0,
            "false_certifications_total": 0,
            "empty_hoeffding_intersections_primary_total_nodes": total_empty_primary,
            "empty_hoeffding_intersections_audit_1m_total_nodes": total_empty_audit_1m,
            "empty_hoeffding_intersections_all_total_nodes": int(
                total_empty_primary + total_empty_audit_1m
            ),
            "all_hard_assertions_passed": True,
        },
        "output_files": {
            "heldout_uniform_2880.csv": {
                "rows": int(len(uniform_df)),
                "sha256": _sha256_file(uniform_csv),
            },
            "heldout_hetero_120.csv": {
                "rows": int(len(hetero_df)),
                "sha256": _sha256_file(hetero_csv),
            },
            "heldout_express_12.csv": {
                "rows": int(len(express_df)),
                "sha256": _sha256_file(express_csv),
            },
            "heldout_precision_audits_48.csv": {
                "rows": int(len(audit_df)),
                "sha256": _sha256_file(audit_csv),
            },
        },
        "heldout_uniform_2880": uniform_summary,
        "heldout_hetero_120": hetero_summary,
        "paired_hetero_vs_uniform_w020": paired_hetero,
        "heldout_express_12": express_summary,
        "precision_audits_48": audit_summary,
    }

    summary_path = out_dir / "heldout_eval_summary.json"
    _write_json(summary_path, summary_payload)
    return summary_payload


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all_heldout_suites(out_dir: Path, workers: int = 8) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_total_start = time.perf_counter()
    freeze_prov = _load_freeze_provenance()

    print(
        f"[Step 6 Held-Out Eval] Starting with workers={workers} "
        f"(freeze={freeze_prov['freeze_tag']}, N_primary={MC_TRIALS_PRIMARY:,}, "
        f"N_audit={MC_TRIALS_AUDIT:,})...",
        flush=True,
    )

    uniform_csv = out_dir / "heldout_uniform_2880.csv"
    hetero_csv = out_dir / "heldout_hetero_120.csv"
    express_csv = out_dir / "heldout_express_12.csv"
    audit_csv = out_dir / "heldout_precision_audits_48.csv"

    executor_ctx = (
        ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
        if workers > 1
        else None
    )
    try:
        # Suite 1: Held-Out Uniform Generated DAGs (2,880)
        print(
            "[Suite 1] Generating 2,880 held-out uniform DAG scenarios (seeds 1001--1020)...",
            flush=True,
        )
        uniform_scenarios = generate_dag_uniform_scenarios(split="eval")
        assert len(uniform_scenarios) == 2880
        assert {int(s.graph_seed) for s in uniform_scenarios} == set(EVAL_SEEDS)
        uniform_df, t_uniform = _run_primary_suite(
            uniform_scenarios, label="Suite 1 Uniform 2880", executor=executor_ctx
        )
        uniform_df.to_csv(uniform_csv, index=False)
        print(
            f"[Suite 1 Done] 2,880 scenarios in {t_uniform:.2f}s -> {uniform_csv.name}",
            flush=True,
        )

        # Suite 2: Held-Out Heterogeneous-Weight DAGs (120)
        print(
            "[Suite 2] Generating 120 held-out heterogeneous DAG scenarios (seeds 1001--1005)...",
            flush=True,
        )
        hetero_scenarios = generate_dag_hetero_scenarios(split="eval")
        assert len(hetero_scenarios) == 120
        assert {int(s.graph_seed) for s in hetero_scenarios} == set(HETERO_EVAL_SEEDS)
        hetero_df, t_hetero = _run_primary_suite(
            hetero_scenarios, label="Suite 2 Hetero 120", executor=executor_ctx
        )
        hetero_df.to_csv(hetero_csv, index=False)
        print(
            f"[Suite 2 Done] 120 scenarios in {t_hetero:.2f}s -> {hetero_csv.name}",
            flush=True,
        )

        # Suite 3: Express 4.18.2 Real-Topology Stochastic (12)
        print("[Suite 3] Generating 12 Express 4.18.2 stochastic scenarios...", flush=True)
        express_scenarios = generate_express_scenarios(
            repo_root=REPO_ROOT, verify_hashes=True
        )
        assert len(express_scenarios) == 12
        express_df, t_express = _run_primary_suite(
            express_scenarios, label="Suite 3 Express 12", executor=executor_ctx
        )
        express_df.to_csv(express_csv, index=False)
        print(
            f"[Suite 3 Done] 12 scenarios in {t_express:.2f}s -> {express_csv.name}",
            flush=True,
        )

        # Suite 4: Preselected Precision Audits (48 independent 1M-trial solves)
        print(
            "[Suite 4] Generating 48 preselected precision audits (seed 1001, w in {0.05, 0.10}, N=1,000,000)...",
            flush=True,
        )
        audit_scenarios = generate_dag_uniform_scenarios(
            split="eval",
            families=DAG_FAMILIES,
            sizes=DAG_SIZES,
            seeds=(AUDIT_SEED,),
            weights=AUDIT_WEIGHTS,
            b_settings=DAG_B_SETTINGS,
            include_structural_metadata=False,
        )
        assert len(audit_scenarios) == 48
        audit_df, t_audit = _run_precision_audit_suite(
            audit_scenarios, executor=executor_ctx
        )
        audit_df.to_csv(audit_csv, index=False)
        print(
            f"[Suite 4 Done] 48 1M-trial audits in {t_audit:.2f}s -> {audit_csv.name}",
            flush=True,
        )
    finally:
        if executor_ctx is not None:
            executor_ctx.shutdown(wait=True)

    t_total = time.perf_counter() - t_total_start
    timings_sec = {
        "suite1_uniform_2880": float(t_uniform),
        "suite2_hetero_120": float(t_hetero),
        "suite3_express_12": float(t_express),
        "suite4_precision_audits_48": float(t_audit),
        "total_wall_sec": float(t_total),
        "total_wall_minutes": float(t_total / 60.0),
    }

    print(
        "[Summary] Computing graph-replicate-weighted summaries and paired deltas...",
        flush=True,
    )
    summary_payload = assemble_and_write_summary(
        out_dir=out_dir,
        uniform_df=uniform_df,
        hetero_df=hetero_df,
        express_df=express_df,
        audit_df=audit_df,
        timings_sec=timings_sec,
        workers=workers,
    )
    ga = summary_payload["global_assertions"]
    print(
        f"[Step 6 Held-Out Eval Complete] wall={t_total:.2f}s ({t_total/60.0:.2f} min) | "
        f"violations=0 | false_certs=0 | "
        f"empty_hoeffding(primary={ga['empty_hoeffding_intersections_primary_total_nodes']}, "
        f"audit_1m={ga['empty_hoeffding_intersections_audit_1m_total_nodes']}) -> heldout_eval_summary.json",
        flush=True,
    )
    return summary_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Step 6 Held-Out Evaluation suites (2,880 uniform + 120 hetero + 12 Express + 48 1M audits)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of ProcessPoolExecutor workers (default: 8).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "heldout_eval",
        help="Output directory for held-out CSVs and JSON summary.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    run_all_heldout_suites(out_dir=args.out_dir, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
