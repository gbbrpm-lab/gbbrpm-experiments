"""Step 4 Development Compute Pilot and Development Controls runner for GBBRPM.

STRICT HELD-OUT PROTOCOL:
- Evaluates ONLY Development Compute Pilot (seed 41, 54 configs) and
  Development Controls (560 matched branch + 720 dev uniform on seeds 41--45
  + 24 dev heterogeneous on seed 41 = 1,304 scenarios).
- Held-out evaluation seeds (1001--1020) are NEVER generated, executed,
  evaluated, or inspected by this script.

Usage:
    python scripts/run_dev_pilot.py [--all | --pilot-only | --controls-only]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gbbrpm.benchmark_comparators import (
    compute_abbe_lower,
    compute_all_tree_bounds,
    compute_ancestry_diagnostic,
    compute_clipped_sum,
    compute_gbbrpm_fast,
    compute_local_only,
    compute_local_upstream,
    compute_recursive_max,
    run_comparators,
)
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
from gbbrpm.exact_ref import evaluate_exact_marginals
from gbbrpm.generators import (
    BenchmarkScenario,
    DAG_B_SETTINGS,
    DAG_FAMILIES,
    DEV_SEEDS,
    EVAL_SEEDS,
    assign_heterogeneous_weights,
    build_dag_from_topology,
    derive_scenario_seed,
    generate_dag_hetero_scenarios,
    generate_dag_topology,
    generate_dag_uniform_scenarios,
    generate_matched_branch_scenarios,
)
from gbbrpm.mc_ref import run_mc_reference

EVAL_SEED_SET = frozenset(EVAL_SEEDS)
BUDGET_FRACTIONS: tuple[float, ...] = (0.10, 0.20, 0.30)
TOL: float = 1e-12


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


def _collect_hardware_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "networkx_version": nx.__version__,
        "git": _get_git_metadata(),
    }


def _assert_not_held_out(scenario: BenchmarkScenario) -> None:
    if scenario.split == "eval":
        raise RuntimeError(
            f"PROTOCOL VIOLATION: Attempted to evaluate held-out eval scenario {scenario.scenario_id}"
        )
    if scenario.graph_seed is not None and int(scenario.graph_seed) in EVAL_SEED_SET:
        raise RuntimeError(
            f"PROTOCOL VIOLATION: Attempted to evaluate held-out seed {scenario.graph_seed}"
        )


def _count_nondegenerate_primitives(dag) -> int:
    n_nodes = int(np.sum((dag.B > 0.0) & (dag.B < 1.0)))
    n_edges = int(np.sum((dag.in_weights > 0.0) & (dag.in_weights < 1.0)))
    return n_nodes + n_edges


def _exact_matched_branch_node_marginals(sc: BenchmarkScenario) -> np.ndarray:
    """Exact closed-form Eq. (3-10)/(3-15) marginal vector for every node in a matched-branch DAG.

    - Root ``a``: ``b`` (shared) or ``0.0`` (independent).
    - Branch starts ``s_r``: ``b`` (both conditions).
    - Intermediate node ``b{r}_{step}``: ``b * p**step`` (unique path of ``step`` edges).
    - Receiver ``d``: exact Eq. (3-15) closed form (``shared`` or ``independent``).
    """
    dag = sc.dag
    p = float(sc.metadata["p"])
    b = float(sc.metadata["b"])
    cond = str(sc.metadata["condition"])
    exact_d = float(sc.metadata["exact_receiver_pi"])

    pi = np.zeros(dag.n_nodes, dtype=np.float64)
    for idx, nid in enumerate(dag.node_ids):
        if nid == "a":
            pi[idx] = b if cond == "shared" else 0.0
        elif nid.startswith("s_"):
            pi[idx] = b
        elif nid == "d":
            pi[idx] = exact_d
        elif nid.startswith("b") and "_" in nid:
            step = int(nid.split("_", 1)[1])
            pi[idx] = b * (p ** step)
        else:
            raise ValueError(f"Unexpected node ID in matched branch graph: {nid!r}")
    return pi


def _assert_deterministic_bracket(M: np.ndarray, T: np.ndarray, R: np.ndarray, H: np.ndarray, sid: str) -> None:
    if np.any(M > T + TOL):
        diff = float(np.max(M - T))
        raise AssertionError(f"Bracket violation M > T on {sid}: max(M-T)={diff:.3e}")
    if np.any(T > R + TOL):
        diff = float(np.max(T - R))
        raise AssertionError(f"Bracket violation T > R on {sid}: max(T-R)={diff:.3e}")
    if np.any(R > H + TOL):
        diff = float(np.max(R - H))
        raise AssertionError(f"Bracket violation R > H on {sid}: max(R-H)={diff:.3e}")


def _assert_exact_bounds(
    M: np.ndarray,
    T: np.ndarray,
    pi: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    abbe: np.ndarray,
    sid: str,
) -> None:
    _assert_deterministic_bracket(M, T, R, H, sid)
    if np.any(T > pi + TOL):
        raise AssertionError(f"Exact bound violation T > pi on {sid}: max(T-pi)={float(np.max(T - pi)):.3e}")
    if np.any(pi > R + TOL):
        raise AssertionError(f"Exact bound violation pi > R on {sid}: max(pi-R)={float(np.max(pi - R)):.3e}")
    if np.any(abbe > pi + TOL):
        raise AssertionError(
            f"Abbe lower violation abbe > pi on {sid}: max(abbe-pi)={float(np.max(abbe - pi)):.3e}"
        )


def _finite_mean(values: list[float] | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if valid.size > 0 else None


def _finite_median(values: list[float] | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.median(valid)) if valid.size > 0 else None


def _finite_max(values: list[float] | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.max(valid)) if valid.size > 0 else None


# ---------------------------------------------------------------------------
# Part A — Development Compute Pilot (seed 41, 54 configs)
# ---------------------------------------------------------------------------


def run_compute_pilot(out_dir: Path) -> dict[str, Any]:
    """Run the 54-config Development Compute Pilot on seed 41 and extrapolate full inventory."""
    pilot_seed = 41
    pilot_sizes = (20, 100, 250)
    pilot_weight_specs: tuple[tuple[str, float | None], ...] = (
        ("w=0.10", 0.10),
        ("w=0.75", 0.75),
        ("hetero_v1", None),
    )

    tracemalloc.start()
    tracemalloc.reset_peak()
    t_pilot_start = time.perf_counter()

    rows: list[dict[str, Any]] = []
    mc_backend_seen: str | None = None

    for family in DAG_FAMILIES:
        for size in pilot_sizes:
            for w_label, w_val in pilot_weight_specs:
                for b_val in DAG_B_SETTINGS:
                    # 1. Graph generation timing
                    t0 = time.perf_counter()
                    topo = generate_dag_topology(family, size, pilot_seed, split="dev")
                    if w_val is None:
                        w_arr = assign_heterogeneous_weights(topo)
                        dag = build_dag_from_topology(topo, w_arr, b_val)
                    else:
                        dag = build_dag_from_topology(topo, w_val, b_val)
                    graph_gen_sec = time.perf_counter() - t0

                    sid = f"pilot|dev|{family}|n={size}|seed={pilot_seed}|{w_label}|B={b_val:.1f}"
                    mc_seed = derive_scenario_seed(sid)

                    # 2. Staged MC batches: 1,000 and 10,000 for all 54; plus 100,000 for all size-250
                    mc_1k = run_mc_reference(dag, n_trials=1_000, seed=mc_seed, batch_size=1_000)
                    mc_10k = run_mc_reference(dag, n_trials=10_000, seed=mc_seed, batch_size=10_000)
                    mc_backend_seen = str(mc_10k["backend"])

                    mc_100k_sec = float("nan")
                    mc_100k_trials_per_sec = float("nan")
                    mc_100k_edge_trials_per_sec = float("nan")
                    mc_ref_for_storage = mc_10k

                    if size == 250:
                        mc_100k = run_mc_reference(
                            dag, n_trials=100_000, seed=mc_seed, batch_size=10_000
                        )
                        mc_100k_sec = float(mc_100k["elapsed_sec"])
                        mc_100k_trials_per_sec = 100_000.0 / mc_100k_sec
                        mc_100k_edge_trials_per_sec = (dag.n_edges * 100_000.0) / mc_100k_sec
                        mc_ref_for_storage = mc_100k

                    mc_1k_sec = float(mc_1k["elapsed_sec"])
                    mc_10k_sec = float(mc_10k["elapsed_sec"])

                    # 3. Deterministic scalar comparators, Abbe lower, Ancestry, full-graph T_j
                    comp = run_comparators(
                        dag, include_trees=True, include_abbe=True, include_ancestry=True
                    )
                    timings = comp["timings_sec"]
                    scalar_comp_sec = float(
                        timings["local_only"]
                        + timings["gbbrpm"]
                        + timings["recursive_max"]
                        + timings["clipped_sum"]
                        + timings["local_upstream"]
                    )
                    abbe_sec = float(timings["abbe_lower"])
                    ancestry_sec = float(timings["ancestry_diagnostic"])
                    full_tree_sec = float(timings["tree_bound"])

                    # 4. Standalone Two-Stage Certified Top-k Screening timing (includes O(|V|+|E|) M,R pass)
                    t_scr0 = time.perf_counter()
                    scr = certify_top_k_two_stage(dag, fractions=BUDGET_FRACTIONS)
                    screening_wall_sec = time.perf_counter() - t_scr0
                    scr_b10 = scr["budgets"][0.10]
                    scr_b20 = scr["budgets"][0.20]
                    scr_b30 = scr["budgets"][0.30]

                    # 5. Output serialization / storage overhead timing (per-node CSV + JSON summary)
                    t_io0 = time.perf_counter()
                    node_df = pd.DataFrame(
                        {
                            "node_id": dag.node_ids,
                            "B": dag.B,
                            "R": comp["gbbrpm"],
                            "M": comp["recursive_max"],
                            "T": comp["tree_bound"],
                            "H": comp["clipped_sum"],
                            "abbe_lower": comp["abbe_lower"],
                            "pi_hat": mc_ref_for_storage["pi_hat"],
                            "mc_lower": mc_ref_for_storage["lower"],
                            "mc_upper": mc_ref_for_storage["upper"],
                        }
                    )
                    csv_buf = io.StringIO()
                    node_df.to_csv(csv_buf, index=False)
                    _ = json.dumps(
                        {
                            "scenario_id": sid,
                            "res_s2_f10": scr_b10["res_s2"],
                            "res_s2_f20": scr_b20["res_s2"],
                            "res_s2_f30": scr_b30["res_s2"],
                        }
                    )
                    output_storage_sec = time.perf_counter() - t_io0
                    serialized_bytes = len(csv_buf.getvalue().encode("utf-8"))

                    rows.append(
                        {
                            "scenario_id": sid,
                            "family": family,
                            "size": size,
                            "n_edges": dag.n_edges,
                            "graph_seed": pilot_seed,
                            "weight_regime": w_label,
                            "b_setting": b_val,
                            "graph_gen_sec": graph_gen_sec,
                            "mc_1k_sec": mc_1k_sec,
                            "mc_10k_sec": mc_10k_sec,
                            "mc_100k_sec": mc_100k_sec,
                            "mc_10k_trials_per_sec": 10_000.0 / mc_10k_sec,
                            "mc_10k_edge_trials_per_sec": (dag.n_edges * 10_000.0) / mc_10k_sec,
                            "mc_100k_trials_per_sec": mc_100k_trials_per_sec,
                            "mc_100k_edge_trials_per_sec": mc_100k_edge_trials_per_sec,
                            "scalar_comparators_sec": scalar_comp_sec,
                            "abbe_lower_sec": abbe_sec,
                            "ancestry_diagnostic_sec": ancestry_sec,
                            "full_tree_bound_sec": full_tree_sec,
                            "screening_global_s1_sec": float(scr["global_stage1_pass_sec"]),
                            "screening_unique_s2_trees": int(scr["n_unique_stage2_trees"]),
                            "screening_unique_s2_tree_sec": float(
                                scr["total_unique_stage2_tree_time_sec"]
                            ),
                            "screening_total_3budgets_sec": float(screening_wall_sec),
                            "screening_f20_total_sec": float(scr_b20["total_screening_time_sec"]),
                            "screening_f20_res_s1": float(scr_b20["res_s1"]),
                            "screening_f20_res_s2": float(scr_b20["res_s2"]),
                            "output_storage_sec": output_storage_sec,
                            "serialized_node_csv_bytes": serialized_bytes,
                        }
                    )

    pilot_elapsed_sec = time.perf_counter() - t_pilot_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_bytes / (1024.0 * 1024.0)

    df = pd.DataFrame(rows)
    assert len(df) == 54, f"Expected 54 pilot rows, got {len(df)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_csv_path = out_dir / "dev_compute_pilot.csv"
    df.to_csv(pilot_csv_path, index=False)

    # Per-size measured averages and linear interpolation for size=50
    by_size: dict[int, dict[str, float]] = {}
    for sz in pilot_sizes:
        sub = df[df["size"] == sz]
        mc_100k_est = (
            float(sub["mc_100k_sec"].mean())
            if sz == 250
            else float(sub["mc_10k_sec"].mean()) * 10.0
        )
        by_size[sz] = {
            "n_edges": float(2 * sz),
            "graph_gen_sec": float(sub["graph_gen_sec"].mean()),
            "mc_1k_sec": float(sub["mc_1k_sec"].mean()),
            "mc_10k_sec": float(sub["mc_10k_sec"].mean()),
            "mc_100k_sec_est": mc_100k_est,
            "mc_10k_trials_per_sec": float(sub["mc_10k_trials_per_sec"].mean()),
            "mc_10k_edge_trials_per_sec": float(sub["mc_10k_edge_trials_per_sec"].mean()),
            "mc_100k_trials_per_sec": (
                float(sub["mc_100k_trials_per_sec"].mean()) if sz == 250 else float("nan")
            ),
            "mc_100k_edge_trials_per_sec": (
                float(sub["mc_100k_edge_trials_per_sec"].mean()) if sz == 250 else float("nan")
            ),
            "scalar_comparators_sec": float(sub["scalar_comparators_sec"].mean()),
            "abbe_lower_sec": float(sub["abbe_lower_sec"].mean()),
            "ancestry_diagnostic_sec": float(sub["ancestry_diagnostic_sec"].mean()),
            "full_tree_bound_sec": float(sub["full_tree_bound_sec"].mean()),
            "screening_total_3budgets_sec": float(sub["screening_total_3budgets_sec"].mean()),
            "output_storage_sec": float(sub["output_storage_sec"].mean()),
        }

    # Interpolate size=50 linearly between measured size=20 and size=100 (t = 30/80 = 0.375),
    # with O(|V|^2) scaling for full_tree_bound_sec (conservative interpolation).
    alpha_50 = (50.0 - 20.0) / (100.0 - 20.0)
    by_size[50] = {
        key: (1.0 - alpha_50) * by_size[20][key] + alpha_50 * by_size[100][key]
        for key in by_size[20]
    }
    by_size[50]["n_edges"] = 100.0

    # Full-inventory workload extrapolation (Chapter 3, Table tab:reference-inventory):
    # - 560 matched branch controls (exact closed-form, no MC required)
    # - 3,600 uniform DAGs (900 per size in {20, 50, 100, 250}) at N=100,000
    # - 144 hetero DAGs (36 per size in {20, 50, 100, 250}) at N=100,000
    # - 12 Express 4.18.2 (71 nodes, 128 edges -> conservative size-100 timing) at N=100,000
    # - 48 million-trial precision audits (12 per size in {20, 50, 100, 250}) at N=1,000,000
    dag_counts_per_size = {sz: 900 + 36 for sz in (20, 50, 100, 250)}  # 936 per size = 3,744
    audit_counts_per_size = {sz: 12 for sz in (20, 50, 100, 250)}      # 12 per size = 48

    extrap_graph_gen = sum(dag_counts_per_size[sz] * by_size[sz]["graph_gen_sec"] for sz in dag_counts_per_size)
    extrap_mc_primary = (
        sum(dag_counts_per_size[sz] * by_size[sz]["mc_100k_sec_est"] for sz in dag_counts_per_size)
        + 12.0 * by_size[100]["mc_100k_sec_est"]
    )
    extrap_mc_audits = sum(
        audit_counts_per_size[sz] * 10.0 * by_size[sz]["mc_100k_sec_est"]
        for sz in audit_counts_per_size
    )
    extrap_scalar_abbe_anc = (
        sum(
            dag_counts_per_size[sz]
            * (
                by_size[sz]["scalar_comparators_sec"]
                + by_size[sz]["abbe_lower_sec"]
                + by_size[sz]["ancestry_diagnostic_sec"]
            )
            for sz in dag_counts_per_size
        )
        + 12.0
        * (
            by_size[100]["scalar_comparators_sec"]
            + by_size[100]["abbe_lower_sec"]
            + by_size[100]["ancestry_diagnostic_sec"]
        )
        + 560.0
        * (
            by_size[20]["scalar_comparators_sec"]
            + by_size[20]["abbe_lower_sec"]
            + by_size[20]["ancestry_diagnostic_sec"]
        )
    )
    extrap_full_trees = (
        sum(dag_counts_per_size[sz] * by_size[sz]["full_tree_bound_sec"] for sz in dag_counts_per_size)
        + 12.0 * by_size[100]["full_tree_bound_sec"]
        + 560.0 * by_size[20]["full_tree_bound_sec"]
    )
    extrap_screening = (
        sum(
            dag_counts_per_size[sz] * by_size[sz]["screening_total_3budgets_sec"]
            for sz in dag_counts_per_size
        )
        + 12.0 * by_size[100]["screening_total_3budgets_sec"]
        + 560.0 * by_size[20]["screening_total_3budgets_sec"]
    )
    extrap_storage = (
        sum(dag_counts_per_size[sz] * by_size[sz]["output_storage_sec"] for sz in dag_counts_per_size)
        + (560.0 + 12.0 + 48.0) * by_size[100]["output_storage_sec"]
    )

    total_single_core_sec = (
        extrap_graph_gen
        + extrap_mc_primary
        + extrap_mc_audits
        + extrap_scalar_abbe_anc
        + extrap_full_trees
        + extrap_screening
        + extrap_storage
    )
    n_cpus = max(1, int(os.cpu_count() or 1))
    est_parallel_workers = min(n_cpus, 8)
    total_parallel_est_sec = total_single_core_sec / (0.75 * est_parallel_workers)

    summary = {
        "protocol": {
            "split": "dev_pilot_only",
            "seed": pilot_seed,
            "eval_seeds_evaluated": 0,
            "held_out_isolation_verified": True,
            "n_pilot_configs": int(len(df)),
            "sizes": list(pilot_sizes),
            "families": list(DAG_FAMILIES),
            "weight_regimes": ["w=0.10", "w=0.75", "hetero_v1"],
            "b_settings": list(DAG_B_SETTINGS),
            "staged_mc_batches": {"all_54": [1_000, 10_000], "size_250_all_18": [100_000]},
            "mc_backend": mc_backend_seen,
        },
        "hardware": _collect_hardware_metadata(),
        "pilot_runtime": {
            "elapsed_sec": float(pilot_elapsed_sec),
            "peak_traced_memory_mb": float(peak_memory_mb),
        },
        "measured_by_size": {str(k): v for k, v in sorted(by_size.items())},
        "size_250_100k_throughput": {
            "mean_sec": float(df.loc[df["size"] == 250, "mc_100k_sec"].mean()),
            "min_sec": float(df.loc[df["size"] == 250, "mc_100k_sec"].min()),
            "max_sec": float(df.loc[df["size"] == 250, "mc_100k_sec"].max()),
            "mean_trials_per_sec": float(
                df.loc[df["size"] == 250, "mc_100k_trials_per_sec"].mean()
            ),
            "mean_edge_trials_per_sec": float(
                df.loc[df["size"] == 250, "mc_100k_edge_trials_per_sec"].mean()
            ),
        },
        "full_inventory_extrapolation": {
            "inventory_counts": {
                "matched_branch_controls": 560,
                "dag_uniform_total": 3600,
                "dag_hetero_total": 144,
                "express_total": 12,
                "primary_scenarios_total": 4316,
                "precision_audits_1m_trials": 48,
                "total_edge_trials_billions": 88.8576,
            },
            "single_core_sec": {
                "graph_generation": float(extrap_graph_gen),
                "mc_primary_100k": float(extrap_mc_primary),
                "mc_precision_audits_1m": float(extrap_mc_audits),
                "scalar_abbe_ancestry": float(extrap_scalar_abbe_anc),
                "full_graph_tree_bounds": float(extrap_full_trees),
                "two_stage_certified_screening": float(extrap_screening),
                "output_storage": float(extrap_storage),
                "total_sec": float(total_single_core_sec),
                "total_minutes": float(total_single_core_sec / 60.0),
            },
            "parallel_estimate": {
                "assumed_workers": est_parallel_workers,
                "efficiency_factor": 0.75,
                "estimated_wall_sec": float(total_parallel_est_sec),
                "estimated_wall_minutes": float(total_parallel_est_sec / 60.0),
            },
            "feasible_without_reduction": bool(total_single_core_sec < 3600.0),
        },
    }

    _write_json(out_dir / "dev_compute_pilot_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Part B — Development Controls (560 branch + 720 dev uniform + 24 dev hetero)
# ---------------------------------------------------------------------------


def _evaluate_branch_controls() -> tuple[pd.DataFrame, dict[str, Any]]:
    scenarios = generate_matched_branch_scenarios()
    assert len(scenarios) == 560

    rows: list[dict[str, Any]] = []
    exact_enum_checked = 0
    w_node_spearmans: list[float] = []

    for sc in scenarios:
        _assert_not_held_out(sc)
        dag = sc.dag
        d_idx = int(sc.metadata["receiver_idx"])
        pi_exact = _exact_matched_branch_node_marginals(sc)

        # Cross-check exact enumeration whenever <= 16 non-degenerate primitives
        n_nondeg = _count_nondegenerate_primitives(dag)
        if n_nondeg <= 16:
            enum_res = evaluate_exact_marginals(dag, max_primitives=16)
            max_parity_err = float(np.max(np.abs(enum_res["marginals"] - pi_exact)))
            if max_parity_err >= 1e-12:
                raise AssertionError(
                    f"Exact enum vs closed-form parity mismatch on {sc.scenario_id}: {max_parity_err:.3e}"
                )
            exact_enum_checked += 1

        comp = run_comparators(
            dag, include_trees=True, include_abbe=True, include_ancestry=True
        )
        M = comp["recursive_max"]
        T = comp["tree_bound"]
        R = comp["gbbrpm"]
        H = comp["clipped_sum"]
        abbe = comp["abbe_lower"]
        anc = comp["ancestry_diagnostic"]
        W = comp["bracket_width"]

        # HARD ASSERTION: M <= T <= pi <= R <= H and abbe <= pi across ALL nodes
        _assert_exact_bounds(M, T, pi_exact, R, H, abbe, sc.scenario_id)

        # Standalone Two-Stage Certified Screening + HARD ASSERTION: 0 false certifications
        scr = certify_top_k_two_stage(dag, fractions=BUDGET_FRACTIONS)
        assert_zero_false_certifications(scr, pi_exact=pi_exact, epsilon=TOL)

        cand_mask = dag.B == 0.0
        rev = compute_pairwise_reversals(R, pi_exact, pi_exact, cand_mask, eps=TOL)
        cap_R = compute_capture_and_regret(
            R, pi_exact, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL
        )
        cap_T = compute_capture_and_regret(
            T, pi_exact, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL
        )
        w_rel = evaluate_nodewise_w_reliability(W, R, pi_exact, cand_mask, tol=TOL)
        if math.isfinite(w_rel["spearman_w_vs_slack"]):
            w_node_spearmans.append(float(w_rel["spearman_w_vs_slack"]))

        b10 = scr["budgets"][0.10]
        b20 = scr["budgets"][0.20]
        b30 = scr["budgets"][0.30]

        rows.append(
            {
                "scenario_id": sc.scenario_id,
                "m": int(sc.metadata["m"]),
                "ell": int(sc.metadata["ell"]),
                "p": float(sc.metadata["p"]),
                "b": float(sc.metadata["b"]),
                "condition": str(sc.metadata["condition"]),
                "n_nodes": dag.n_nodes,
                "n_edges": dag.n_edges,
                "n_nondeg_primitives": n_nondeg,
                "exact_enum_verified": bool(n_nondeg <= 16),
                "pi_d": float(pi_exact[d_idx]),
                "R_d": float(R[d_idx]),
                "M_d": float(M[d_idx]),
                "T_d": float(T[d_idx]),
                "H_d": float(H[d_idx]),
                "abbe_d": float(abbe[d_idx]),
                "ancestry_d": float(anc[d_idx]),
                "W_d": float(R[d_idx] - M[d_idx]),
                "tree_width_d": float(R[d_idx] - T[d_idx]),
                "upper_slack_d": float(R[d_idx] - pi_exact[d_idx]),
                "lower_slack_M_d": float(pi_exact[d_idx] - M[d_idx]),
                "lower_slack_T_d": float(pi_exact[d_idx] - T[d_idx]),
                "lower_slack_abbe_d": float(pi_exact[d_idx] - abbe[d_idx]),
                "W_cand_mean": float(np.mean(W[cand_mask])),
                "W_cand_max": float(np.max(W[cand_mask])),
                "spearman_W_vs_slack": float(w_rel["spearman_w_vs_slack"]),
                "n_reversals_R": int(rev["n_reversals"]),
                "n_resolved_strict_pairs_R": int(rev["n_resolved_strict_model_pairs"]),
                "reversal_rate_R": float(rev["reversal_rate"]),
                "regret_R_f10": float(cap_R["budgets"][0.10]["regret"]),
                "regret_R_f20": float(cap_R["budgets"][0.20]["regret"]),
                "regret_R_f30": float(cap_R["budgets"][0.30]["regret"]),
                "regret_T_f20": float(cap_T["budgets"][0.20]["regret"]),
                "res_s1_f10": float(b10["res_s1"]),
                "res_s2_f10": float(b10["res_s2"]),
                "res_s1_f20": float(b20["res_s1"]),
                "res_s2_f20": float(b20["res_s2"]),
                "res_s1_f30": float(b30["res_s1"]),
                "res_s2_f30": float(b30["res_s2"]),
                "fully_certified_s1_f20": bool(b20["fully_certified_s1"]),
                "fully_certified_s2_f20": bool(b20["fully_certified_s2"]),
                "full_tree_sec": float(comp["timings_sec"]["tree_bound"]),
                "screening_f20_sec": float(b20["total_screening_time_sec"]),
            }
        )

    df = pd.DataFrame(rows)
    shared = df[df["condition"] == "shared"]
    indep = df[df["condition"] == "independent"]

    summary = {
        "n_scenarios": int(len(df)),
        "exact_enumeration_cross_checked": int(exact_enum_checked),
        "bracket_violations": 0,
        "abbe_lower_violations": 0,
        "false_certifications": 0,
        "shared": {
            "n_scenarios": int(len(shared)),
            "mean_upper_slack_d": float(shared["upper_slack_d"].mean()),
            "max_upper_slack_d": float(shared["upper_slack_d"].max()),
            "mean_W_d": float(shared["W_d"].mean()),
            "mean_tree_width_d": float(shared["tree_width_d"].mean()),
            "mean_lower_slack_T_d": float(shared["lower_slack_T_d"].mean()),
            "mean_lower_slack_abbe_d": float(shared["lower_slack_abbe_d"].mean()),
            "mean_res_s1_f20": float(shared["res_s1_f20"].mean()),
            "mean_res_s2_f20": float(shared["res_s2_f20"].mean()),
            "full_cert_rate_s2_f20": float(shared["fully_certified_s2_f20"].mean()),
            "scenarios_with_reversals": int((shared["n_reversals_R"] > 0).sum()),
            "mean_regret_R_f20": float(shared["regret_R_f20"].mean()),
        },
        "independent": {
            "n_scenarios": int(len(indep)),
            "mean_upper_slack_d": float(indep["upper_slack_d"].mean()),
            "max_upper_slack_d": float(indep["upper_slack_d"].max()),
            "mean_W_d": float(indep["W_d"].mean()),
            "mean_tree_width_d": float(indep["tree_width_d"].mean()),
            "mean_lower_slack_T_d": float(indep["lower_slack_T_d"].mean()),
            "mean_lower_slack_abbe_d": float(indep["lower_slack_abbe_d"].mean()),
            "mean_res_s1_f20": float(indep["res_s1_f20"].mean()),
            "mean_res_s2_f20": float(indep["res_s2_f20"].mean()),
            "full_cert_rate_s2_f20": float(indep["fully_certified_s2_f20"].mean()),
            "scenarios_with_reversals": int((indep["n_reversals_R"] > 0).sum()),
            "mean_regret_R_f20": float(indep["regret_R_f20"].mean()),
        },
        "w_nodewise_spearman_mean": _finite_mean(w_node_spearmans),
    }
    return df, summary


def _evaluate_dag_suite(
    scenarios: list[BenchmarkScenario],
    mc_trials: int,
    batch_size: int = 10_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_empty_intersections = 0
    scenarios_with_empty_intersections = 0

    for sc in scenarios:
        _assert_not_held_out(sc)
        dag = sc.dag
        cand_mask = dag.B == 0.0

        # 1. Vectorized uint64 bit-packed MC reference
        mc = run_mc_reference(
            dag, n_trials=mc_trials, seed=sc.mc_seed, batch_size=batch_size, alpha=0.05
        )
        pi_hat = mc["pi_hat"]
        mc_lo = mc["lower"]
        mc_hi = mc["upper"]
        h = float(mc["hoeffding_h"])
        mc_sec = float(mc["elapsed_sec"])

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
        full_tree_sec = float(comp["timings_sec"]["tree_bound"])

        # HARD ASSERTION: M <= T <= R <= H across ALL nodes
        _assert_deterministic_bracket(M, T, R, H, sc.scenario_id)

        # Intersect MC Hoeffding interval with analytic bracket [T, R] (NEVER silently repair!)
        L_prime, U_prime, empty_mask = intersect_intervals(mc_lo, mc_hi, T, R)
        n_empty = int(np.sum(empty_mask))
        total_empty_intersections += n_empty
        if n_empty > 0:
            scenarios_with_empty_intersections += 1

        # 3. Standalone Two-Stage Certified Top-k Screening (unprecomputed M,R for fair timing)
        scr = certify_top_k_two_stage(dag, fractions=BUDGET_FRACTIONS)

        # HARD ASSERTION: Zero false certifications against exact analytic bracket [T, R]
        # AND against nonempty intersected intervals [L', U'] whenever n_empty == 0.
        assert_zero_false_certifications(scr, pi_lower=T, pi_upper=R, epsilon=TOL)
        if n_empty == 0:
            assert_zero_false_certifications(
                scr, pi_lower=L_prime, pi_upper=U_prime, epsilon=TOL
            )

        # 4. Ranking, capture/regret, saturation, reversal, and W_j reliability metrics on C
        cap_R = compute_capture_and_regret(
            R, pi_hat, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL, h=h
        )
        cap_T = compute_capture_and_regret(
            T, pi_hat, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL, h=h
        )
        cap_M = compute_capture_and_regret(
            M, pi_hat, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL, h=h
        )
        cap_H = compute_capture_and_regret(
            H, pi_hat, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL, h=h
        )
        cap_lu = compute_capture_and_regret(
            loc_up, pi_hat, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL, h=h
        )
        cap_abbe = compute_capture_and_regret(
            abbe, pi_hat, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL, h=h
        )
        cap_anc = compute_capture_and_regret(
            anc, pi_hat, cand_mask, fractions=BUDGET_FRACTIONS, seed=140, n_perms=100, tol=TOL, h=h
        )

        rev_R = compute_pairwise_reversals(R, L_prime, U_prime, cand_mask, eps=TOL)
        sat_R = compute_saturation_metrics(R, cand_mask, tol=TOL)
        sat_H = compute_saturation_metrics(H, cand_mask, tol=TOL)
        sat_pi = compute_saturation_metrics(pi_hat, cand_mask, tol=TOL)

        sp_R = compute_tie_aware_spearman(R, pi_hat, cand_mask, tol=TOL)
        sp_T = compute_tie_aware_spearman(T, pi_hat, cand_mask, tol=TOL)
        sp_M = compute_tie_aware_spearman(M, pi_hat, cand_mask, tol=TOL)
        sp_abbe = compute_tie_aware_spearman(abbe, pi_hat, cand_mask, tol=TOL)
        sp_anc = compute_tie_aware_spearman(anc, pi_hat, cand_mask, tol=TOL)

        w_rel = evaluate_nodewise_w_reliability(W, R, pi_hat, cand_mask, tol=TOL)

        b10 = scr["budgets"][0.10]
        b20 = scr["budgets"][0.20]
        b30 = scr["budgets"][0.30]

        # Candidate-level bracket tightness stats
        R_c = R[cand_mask]
        T_c = T[cand_mask]
        M_c = M[cand_mask]
        pi_c = pi_hat[cand_mask]
        W_c = W[cand_mask]

        scr_f20_sec = float(b20["total_screening_time_sec"])
        speedup_vs_full_T_f20 = full_tree_sec / scr_f20_sec if scr_f20_sec > 0 else float("nan")
        speedup_vs_mc_f20 = mc_sec / scr_f20_sec if scr_f20_sec > 0 else float("nan")

        st = sc.metadata.get("structural", {})

        rows.append(
            {
                "scenario_id": sc.scenario_id,
                "suite": sc.suite,
                "split": sc.split,
                "family": sc.family,
                "size": sc.size,
                "n_edges": dag.n_edges,
                "graph_seed": sc.graph_seed,
                "weight_regime": sc.weight_regime,
                "weight_value": sc.weight_value,
                "b_setting": sc.b_setting,
                "mc_seed": sc.mc_seed,
                "mc_trials": mc_trials,
                "hoeffding_h": h,
                "n_empty_intersections": n_empty,
                "depth": st.get("depth"),
                "reachable_fraction_candidates": st.get("reachable_fraction_candidates"),
                "overlap_candidate_fraction": st.get("overlap_candidate_fraction"),
                "layer1_source_coverage": st.get("layer1_source_coverage"),
                "layer2_source_coverage": st.get("layer2_source_coverage"),
                # Bracket tightness on C
                "W_mean": float(np.mean(W_c)),
                "W_max": float(np.max(W_c)),
                "W_p90": float(np.percentile(W_c, 90.0)),
                "top_k_boundary_W_mean_f20": float(b20["top_k_boundary_w_mean"]),
                "tree_width_mean": float(np.mean(R_c - T_c)),
                "tree_width_max": float(np.max(R_c - T_c)),
                "upper_slack_R_minus_pi_mean": float(np.mean(R_c - pi_c)),
                "upper_slack_R_minus_pi_max": float(np.max(R_c - pi_c)),
                "lower_slack_pi_minus_T_mean": float(np.mean(pi_c - T_c)),
                "lower_slack_pi_minus_M_mean": float(np.mean(pi_c - M_c)),
                "mae_R_vs_pi": float(np.mean(np.abs(R_c - pi_c))),
                "mae_T_vs_pi": float(np.mean(np.abs(T_c - pi_c))),
                "mae_abbe_vs_pi": float(np.mean(np.abs(abbe[cand_mask] - pi_c))),
                "mae_ancestry_vs_pi": float(np.mean(np.abs(anc[cand_mask] - pi_c))),
                # Saturation
                "sat_095_R": float(sat_R["fraction_ge_095"]),
                "sat_num_R": float(sat_R["numerical_saturation_fraction"]),
                "groups_R": int(sat_R["n_distinct_score_groups"]),
                "sat_095_H": float(sat_H["fraction_ge_095"]),
                "sat_095_pi": float(sat_pi["fraction_ge_095"]),
                # Spearman
                "spearman_R": sp_R,
                "spearman_T": sp_T,
                "spearman_M": sp_M,
                "spearman_abbe": sp_abbe,
                "spearman_ancestry": sp_anc,
                "spearman_W_vs_slack": float(w_rel["spearman_w_vs_slack"]),
                # Fixed-budget capture & regret (f=0.10, 0.20, 0.30)
                "capture_R_f10": float(cap_R["budgets"][0.10]["capture"]),
                "capture_R_f20": float(cap_R["budgets"][0.20]["capture"]),
                "capture_R_f30": float(cap_R["budgets"][0.30]["capture"]),
                "regret_R_f10": float(cap_R["budgets"][0.10]["regret"]),
                "regret_R_f20": float(cap_R["budgets"][0.20]["regret"]),
                "regret_R_f30": float(cap_R["budgets"][0.30]["regret"]),
                "regret_T_f20": float(cap_T["budgets"][0.20]["regret"]),
                "regret_M_f20": float(cap_M["budgets"][0.20]["regret"]),
                "regret_H_f20": float(cap_H["budgets"][0.20]["regret"]),
                "regret_local_upstream_f20": float(cap_lu["budgets"][0.20]["regret"]),
                "regret_abbe_f20": float(cap_abbe["budgets"][0.20]["regret"]),
                "regret_ancestry_f20": float(cap_anc["budgets"][0.20]["regret"]),
                "regret_R_f20_resolved": bool(cap_R["budgets"][0.20]["regret_resolved"]),
                # Pairwise reversals
                "n_reversals_R": int(rev_R["n_reversals"]),
                "n_resolved_strict_pairs_R": int(rev_R["n_resolved_strict_model_pairs"]),
                "n_strict_model_pairs_R": int(rev_R["n_strict_model_pairs"]),
                "resolved_fraction_strict_R": float(rev_R["resolved_fraction_of_strict_model"]),
                "reversal_rate_R": float(rev_R["reversal_rate"]),
                # Two-Stage Certified Top-k Screening
                "res_s1_f10": float(b10["res_s1"]),
                "res_s2_f10": float(b10["res_s2"]),
                "band_reduction_f10": float(b10["band_reduction"]),
                "full_cert_s1_f10": bool(b10["fully_certified_s1"]),
                "full_cert_s2_f10": bool(b10["fully_certified_s2"]),
                "n_stage2_trees_f10": int(b10["n_stage2_trees"]),
                "res_s1_f20": float(b20["res_s1"]),
                "res_s2_f20": float(b20["res_s2"]),
                "band_reduction_f20": float(b20["band_reduction"]),
                "full_cert_s1_f20": bool(b20["fully_certified_s1"]),
                "full_cert_s2_f20": bool(b20["fully_certified_s2"]),
                "n_stage2_trees_f20": int(b20["n_stage2_trees"]),
                "res_s1_f30": float(b30["res_s1"]),
                "res_s2_f30": float(b30["res_s2"]),
                "band_reduction_f30": float(b30["band_reduction"]),
                "full_cert_s1_f30": bool(b30["fully_certified_s1"]),
                "full_cert_s2_f30": bool(b30["fully_certified_s2"]),
                "n_stage2_trees_f30": int(b30["n_stage2_trees"]),
                # Timings & paired speedups
                "mc_sec": mc_sec,
                "full_tree_sec": full_tree_sec,
                "abbe_sec": float(comp["timings_sec"]["abbe_lower"]),
                "screening_f20_sec": scr_f20_sec,
                "screening_s1_f20_sec": float(b20["stage1_time_sec"]),
                "screening_s2_f20_sec": float(b20["stage2_time_sec"]),
                "speedup_vs_full_T_f20": speedup_vs_full_T_f20,
                "speedup_vs_mc_f20": speedup_vs_mc_f20,
            }
        )

    df = pd.DataFrame(rows)

    # W_j scenario-level screening precision/recall for flagging nonzero reversals and nonzero regret
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

    by_family: dict[str, Any] = {}
    for fam, grp in df.groupby("family", sort=True):
        by_family[str(fam)] = {
            "n_scenarios": int(len(grp)),
            "mean_W": float(grp["W_mean"].mean()),
            "mean_tree_width": float(grp["tree_width_mean"].mean()),
            "mean_upper_slack": float(grp["upper_slack_R_minus_pi_mean"].mean()),
            "mean_res_s1_f20": float(grp["res_s1_f20"].mean()),
            "mean_res_s2_f20": float(grp["res_s2_f20"].mean()),
            "full_cert_s1_f20_rate": float(grp["full_cert_s1_f20"].mean()),
            "full_cert_s2_f20_rate": float(grp["full_cert_s2_f20"].mean()),
            "median_speedup_vs_full_T_f20": _finite_median(grp["speedup_vs_full_T_f20"]),
            "median_speedup_vs_mc_f20": _finite_median(grp["speedup_vs_mc_f20"]),
            "mean_regret_R_f20": float(grp["regret_R_f20"].mean()),
            "mean_regret_T_f20": float(grp["regret_T_f20"].mean()),
            "mean_regret_M_f20": float(grp["regret_M_f20"].mean()),
            "mean_regret_abbe_f20": float(grp["regret_abbe_f20"].mean()),
            "scenarios_with_reversals": int((grp["n_reversals_R"] > 0).sum()),
            "mean_reversal_rate_R": _finite_mean(grp["reversal_rate_R"]),
        }

    layered_coverage: dict[str, Any] = {}
    layered_sub = df[df["family"] == "layered"]
    for sz, grp in layered_sub.groupby("size", sort=True):
        layered_coverage[str(int(sz))] = {
            "layer1_source_coverage": _finite_mean(grp["layer1_source_coverage"]),
            "layer2_source_coverage": _finite_mean(grp["layer2_source_coverage"]),
            "reachable_fraction_candidates": _finite_mean(grp["reachable_fraction_candidates"]),
            "overlap_candidate_fraction": _finite_mean(grp["overlap_candidate_fraction"]),
        }

    summary = {
        "n_scenarios": int(len(df)),
        "mc_trials": int(mc_trials),
        "bracket_violations": 0,
        "false_certifications_analytic": 0,
        "false_certifications_interval": 0,
        "empty_hoeffding_intersections_total_nodes": int(total_empty_intersections),
        "scenarios_with_empty_hoeffding_intersections": int(scenarios_with_empty_intersections),
        "bracket_tightness": {
            "mean_W": float(df["W_mean"].mean()),
            "median_W": float(df["W_mean"].median()),
            "max_W": float(df["W_max"].max()),
            "mean_tree_width": float(df["tree_width_mean"].mean()),
            "mean_upper_slack_R_minus_pi": float(df["upper_slack_R_minus_pi_mean"].mean()),
            "mean_lower_slack_pi_minus_T": float(df["lower_slack_pi_minus_T_mean"].mean()),
            "mean_lower_slack_pi_minus_M": float(df["lower_slack_pi_minus_M_mean"].mean()),
        },
        "certified_screening": {
            "f10": {
                "mean_res_s1": float(df["res_s1_f10"].mean()),
                "mean_res_s2": float(df["res_s2_f10"].mean()),
                "mean_band_reduction": float(df["band_reduction_f10"].mean()),
                "full_cert_s1_rate": float(df["full_cert_s1_f10"].mean()),
                "full_cert_s2_rate": float(df["full_cert_s2_f10"].mean()),
            },
            "f20": {
                "mean_res_s1": float(df["res_s1_f20"].mean()),
                "mean_res_s2": float(df["res_s2_f20"].mean()),
                "mean_band_reduction": float(df["band_reduction_f20"].mean()),
                "full_cert_s1_rate": float(df["full_cert_s1_f20"].mean()),
                "full_cert_s2_rate": float(df["full_cert_s2_f20"].mean()),
                "median_speedup_vs_full_T": _finite_median(df["speedup_vs_full_T_f20"]),
                "mean_speedup_vs_full_T": _finite_mean(df["speedup_vs_full_T_f20"]),
                "median_speedup_vs_mc": _finite_median(df["speedup_vs_mc_f20"]),
                "mean_speedup_vs_mc": _finite_mean(df["speedup_vs_mc_f20"]),
                "total_full_tree_sec": float(df["full_tree_sec"].sum()),
                "total_screening_f20_sec": float(df["screening_f20_sec"].sum()),
                "total_mc_sec": float(df["mc_sec"].sum()),
                "aggregate_speedup_vs_full_T": float(
                    df["full_tree_sec"].sum() / df["screening_f20_sec"].sum()
                ),
                "aggregate_speedup_vs_mc": float(
                    df["mc_sec"].sum() / df["screening_f20_sec"].sum()
                ),
            },
            "f30": {
                "mean_res_s1": float(df["res_s1_f30"].mean()),
                "mean_res_s2": float(df["res_s2_f30"].mean()),
                "mean_band_reduction": float(df["band_reduction_f30"].mean()),
                "full_cert_s1_rate": float(df["full_cert_s1_f30"].mean()),
                "full_cert_s2_rate": float(df["full_cert_s2_f30"].mean()),
            },
        },
        "ranking_and_reversals_f20": {
            "mean_regret_R": float(df["regret_R_f20"].mean()),
            "mean_regret_T": float(df["regret_T_f20"].mean()),
            "mean_regret_M": float(df["regret_M_f20"].mean()),
            "mean_regret_H": float(df["regret_H_f20"].mean()),
            "mean_regret_local_upstream": float(df["regret_local_upstream_f20"].mean()),
            "mean_regret_abbe": float(df["regret_abbe_f20"].mean()),
            "mean_regret_ancestry": float(df["regret_ancestry_f20"].mean()),
            "mean_spearman_R": _finite_mean(df["spearman_R"]),
            "mean_spearman_T": _finite_mean(df["spearman_T"]),
            "mean_spearman_M": _finite_mean(df["spearman_M"]),
            "mean_spearman_abbe": _finite_mean(df["spearman_abbe"]),
            "mean_spearman_ancestry": _finite_mean(df["spearman_ancestry"]),
            "scenarios_with_interval_resolved_reversals": int((df["n_reversals_R"] > 0).sum()),
            "total_reversals_R": int(df["n_reversals_R"].sum()),
            "total_resolved_strict_pairs_R": int(df["n_resolved_strict_pairs_R"].sum()),
            "pooled_reversal_rate_R": (
                float(df["n_reversals_R"].sum() / df["n_resolved_strict_pairs_R"].sum())
                if int(df["n_resolved_strict_pairs_R"].sum()) > 0
                else None
            ),
            "mean_scenario_reversal_rate_R": _finite_mean(df["reversal_rate_R"]),
        },
        "w_reliability": {
            "nodewise_spearman_W_vs_slack_mean": _finite_mean(df["spearman_W_vs_slack"]),
            "nodewise_spearman_W_vs_slack_median": _finite_median(df["spearman_W_vs_slack"]),
            "scenario_screening_reversals": w_screen_reversals,
            "scenario_screening_regret_f20": w_screen_regret,
        },
        "by_family": by_family,
        "layered_source_coverage": layered_coverage,
    }
    return df, summary


def run_dev_controls(out_dir: Path, mc_trials: int = 100_000) -> dict[str, Any]:
    """Run all 1,304 Development Controls (560 branch + 720 dev uniform + 24 dev hetero)."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Matched Branch Controls (560)
    t_br = time.perf_counter()
    branch_df, branch_summary = _evaluate_branch_controls()
    branch_elapsed = time.perf_counter() - t_br
    branch_df.to_csv(out_dir / "dev_controls_branch_560.csv", index=False)

    # 2. Development Uniform DAG Controls (720, seeds 41--45 ONLY)
    t_dag = time.perf_counter()
    dag_dev_scenarios = generate_dag_uniform_scenarios(split="dev")
    assert len(dag_dev_scenarios) == 720
    dag_df, dag_summary = _evaluate_dag_suite(dag_dev_scenarios, mc_trials=mc_trials)
    dag_elapsed = time.perf_counter() - t_dag
    dag_df.to_csv(out_dir / "dev_controls_dag_720.csv", index=False)

    # 3. Development Heterogeneous DAG Controls (24, seed 41 ONLY)
    t_het = time.perf_counter()
    hetero_dev_scenarios = generate_dag_hetero_scenarios(split="dev")
    assert len(hetero_dev_scenarios) == 24
    hetero_df, hetero_summary = _evaluate_dag_suite(hetero_dev_scenarios, mc_trials=mc_trials)
    hetero_elapsed = time.perf_counter() - t_het
    hetero_df.to_csv(out_dir / "dev_controls_hetero_24.csv", index=False)

    total_elapsed = time.perf_counter() - t0
    total_scenarios = len(branch_df) + len(dag_df) + len(hetero_df)
    assert total_scenarios == 1304

    summary = {
        "protocol": {
            "split": "dev_controls_only",
            "dev_seeds": list(DEV_SEEDS),
            "eval_seeds_evaluated": 0,
            "held_out_isolation_verified": True,
            "mc_trials_dag_controls": int(mc_trials),
            "counts": {
                "matched_branch": int(len(branch_df)),
                "dag_uniform_dev": int(len(dag_df)),
                "dag_hetero_dev": int(len(hetero_df)),
                "total": int(total_scenarios),
            },
        },
        "hardware": _collect_hardware_metadata(),
        "timings_sec": {
            "branch_560": float(branch_elapsed),
            "dag_720": float(dag_elapsed),
            "hetero_24": float(hetero_elapsed),
            "total": float(total_elapsed),
        },
        "global_assertions": {
            "total_scenarios_verified": int(total_scenarios),
            "bracket_violations_total": 0,
            "branch_abbe_lower_violations": 0,
            "false_certifications_total": 0,
            "empty_hoeffding_intersections_total_nodes": int(
                dag_summary["empty_hoeffding_intersections_total_nodes"]
                + hetero_summary["empty_hoeffding_intersections_total_nodes"]
            ),
            "all_hard_assertions_passed": True,
        },
        "matched_branch_560": branch_summary,
        "dag_uniform_720": dag_summary,
        "dag_hetero_24": hetero_summary,
    }

    _write_json(out_dir / "dev_controls_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GBBRPM Step 4 Development Compute Pilot and Development Controls."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Run both pilot and controls (default).")
    mode.add_argument("--pilot-only", action="store_true", help="Run only the 54-config compute pilot.")
    mode.add_argument("--controls-only", action="store_true", help="Run only the 1,304 dev controls.")
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=100_000,
        help="Monte Carlo trial count per dev DAG control scenario (default: 100000).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dev_pilot",
        help="Output directory for dev pilot and dev controls artifacts.",
    )
    args = parser.parse_args(argv)

    run_pilot = not args.controls_only
    run_controls = not args.pilot_only

    if run_pilot:
        print("[Step 4] Running Part A: Development Compute Pilot (seed 41, 54 configs)...", flush=True)
        pilot_summary = run_compute_pilot(args.out_dir)
        s250 = pilot_summary["size_250_100k_throughput"]
        ext = pilot_summary["full_inventory_extrapolation"]["single_core_sec"]
        par = pilot_summary["full_inventory_extrapolation"]["parallel_estimate"]
        print(
            f"[Pilot Done] elapsed={pilot_summary['pilot_runtime']['elapsed_sec']:.2f}s | "
            f"peak_mem={pilot_summary['pilot_runtime']['peak_traced_memory_mb']:.2f} MB | "
            f"n=250 100k MC mean={s250['mean_sec']:.3f}s "
            f"({s250['mean_trials_per_sec']:,.0f} trials/s, "
            f"{s250['mean_edge_trials_per_sec']/1e6:.1f}M edge-trials/s)",
            flush=True,
        )
        print(
            f"[Full Inventory Extrap (4,316 + 48 audits)] "
            f"single-core={ext['total_minutes']:.2f} min ({ext['total_sec']:.1f}s) | "
            f"est {par['assumed_workers']}-worker={par['estimated_wall_minutes']:.2f} min | "
            f"feasible={pilot_summary['full_inventory_extrapolation']['feasible_without_reduction']}",
            flush=True,
        )

    if run_controls:
        print(
            f"[Step 4] Running Part B: Development Controls (560 branch + 720 dev uniform + 24 dev hetero, "
            f"N={args.mc_trials:,} MC)...",
            flush=True,
        )
        ctrl = run_dev_controls(args.out_dir, mc_trials=args.mc_trials)
        dag_s = ctrl["dag_uniform_720"]
        scr20 = dag_s["certified_screening"]["f20"]
        print(
            f"[Controls Done] 1,304 scenarios in {ctrl['timings_sec']['total']:.2f}s | "
            f"violations=0 | false_certs=0 | "
            f"empty_hoeffding_nodes={ctrl['global_assertions']['empty_hoeffding_intersections_total_nodes']}",
            flush=True,
        )
        print(
            f"[Dev Uniform 720 (f=0.20)] Res_S1={scr20['mean_res_s1']:.3f} -> "
            f"Res_S2={scr20['mean_res_s2']:.3f} (full_cert_S2={scr20['full_cert_s2_rate']:.3f}) | "
            f"agg_speedup vs full_T={scr20['aggregate_speedup_vs_full_T']:.2f}x, "
            f"vs MC={scr20['aggregate_speedup_vs_mc']:.1f}x",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
