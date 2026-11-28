#!/usr/bin/env python3
"""Step 6 Task A: Sparse & Dense Scalability, Memory, and Paired MC Reference Benchmark.

Implements Chapter 3 Section 3.8 (Computational and Input-Cost Assessment) without
modifying the frozen ``scalability.py`` baseline or any frozen package module.

Series evaluated:
1. Sparse DAG series: |V| in {20, 50, 100, 250, 500, 1000}, target 3|V| forward
   edges, seeded random topological ordering (seed=41), bounded [0, 1] inputs.
2. Dense DAG series: |V| in {20, 50, 100, 250}, |E| = floor(0.25 * |V|*(|V|-1)/2),
   seeded random topological ordering (seed=41), bounded [0, 1] inputs.
3. Paired Reference Timing series: forward family, seed=1001, w=0.5, B=0.5,
   |V| in {20, 50, 100, 250}, pairing 5 repeated full N=100,000-trial MC solves
   against 30 repetitions of all deterministic comparators and Two-Stage Screening.

Outputs (all tracked, non-restricted):
- results/heldout_eval/scalability_v2.csv
- results/heldout_eval/scalability_v2_summary.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbbrpm.benchmark_comparators import (  # noqa: E402
    compute_abbe_lower,
    compute_all_tree_bounds,
    compute_ancestry_diagnostic,
    compute_clipped_sum,
    compute_gbbrpm_fast,
    compute_local_upstream,
    compute_recursive_max,
)
from gbbrpm.benchmark_metrics import assign_anchored_tie_groups  # noqa: E402
from gbbrpm.certified_screening import certify_top_k_two_stage  # noqa: E402
from gbbrpm.core import evaluate_gbbrpm  # noqa: E402
from gbbrpm.csr_graph import PreparedDAG, prepare_dag  # noqa: E402
from gbbrpm.generators import generate_dag_uniform_scenarios  # noqa: E402
from gbbrpm.mc_ref import run_mc_reference  # noqa: E402

SPARSE_SIZES: tuple[int, ...] = (20, 50, 100, 250, 500, 1000)
DENSE_SIZES: tuple[int, ...] = (20, 50, 100, 250)
PAIRED_MC_SIZES: tuple[int, ...] = (20, 50, 100, 250)

DEFAULT_SEED = 41
DEFAULT_WARMUPS = 5
DEFAULT_REPEATS = 30
DEFAULT_MC_WARMUPS = 1
DEFAULT_MC_REPEATS = 5
DEFAULT_MC_TRIALS = 100_000
DEFAULT_MC_BATCH_SIZE = 10_000
TREE_SINGLE_RUN_CAP_SEC = 2.0
TREE_CAPPED_REPEATS = 3


def _derive_rng(series: str, n_nodes: int, seed: int) -> np.random.Generator:
    payload = f"scalability_v2|{series}|n={int(n_nodes)}|seed={int(seed)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:16], byteorder="big", signed=False)
    return np.random.Generator(np.random.PCG64(seed_int))


def build_synthetic_scalability_dag(
    n_nodes: int,
    target_edges: int,
    series: str,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, PreparedDAG]:
    """Construct a seeded random-topological-order DAG with exact unique forward edges.

    Inputs stay strictly bounded in [0, 1]:
    - First ceil(0.10 * |V|) nodes in the random topological ordering get B = 0.50;
      all remaining nodes get B = 0.00.
    - Edges receive primitive C in [80, 120], L in [0.2, 0.9] * C, tau = 1.0 so
      S_ij = L / C in [0.2, 0.9].
    """
    n = int(n_nodes)
    max_forward = n * (n - 1) // 2
    m = min(int(target_edges), max_forward)
    if m < 1:
        raise ValueError(f"target_edges must be >= 1, got {target_edges}")

    rng = _derive_rng(series, n, seed)
    topo_order = rng.permutation(n)

    # Sample exact m unique forward pairs (i, j) with 0 <= i < j < n
    if m * 4 >= max_forward:
        all_u, all_v = np.triu_indices(n, k=1)
        chosen = rng.choice(max_forward, size=m, replace=False)
        chosen.sort()
        pos_u = all_u[chosen]
        pos_v = all_v[chosen]
    else:
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < m:
            need = m - len(pairs)
            batch = max(need * 2, 64)
            a = rng.integers(0, n, size=batch)
            b = rng.integers(0, n, size=batch)
            valid = a != b
            a = a[valid]
            b = b[valid]
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            for u_pos, v_pos in zip(lo.tolist(), hi.tolist()):
                pairs.add((u_pos, v_pos))
                if len(pairs) == m:
                    break
        sorted_pairs = sorted(pairs)
        pos_u = np.fromiter((p[0] for p in sorted_pairs), dtype=np.int32, count=m)
        pos_v = np.fromiter((p[1] for p in sorted_pairs), dtype=np.int32, count=m)

    # Active local disturbances on the first ceil(0.10 * |V|) topological nodes
    n_sources = max(1, int(math.ceil(0.10 * n)))
    b_by_node = np.zeros(n, dtype=np.float64)
    b_by_node[topo_order[:n_sources]] = 0.50

    nodes_df = pd.DataFrame(
        {
            "node": [f"v{i}" for i in range(n)],
            "B": b_by_node,
        }
    )

    src_ids = [f"v{int(topo_order[int(u)])}" for u in pos_u]
    dst_ids = [f"v{int(topo_order[int(v)])}" for v in pos_v]
    C_vals = rng.uniform(80.0, 120.0, size=m)
    L_vals = rng.uniform(0.20, 0.90, size=m) * C_vals
    tau_vals = np.ones(m, dtype=np.float64)

    edges_df = pd.DataFrame(
        {
            "source": src_ids,
            "target": dst_ids,
            "C": C_vals,
            "L": L_vals,
            "tau": tau_vals,
        }
    )
    dag = prepare_dag(nodes_df, edges_df)
    assert dag.n_nodes == n and dag.n_edges == m
    return nodes_df, edges_df, dag


def _dag_to_dataframes(dag: PreparedDAG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a PreparedDAG into canonical (nodes_df, edges_df) for DF-level timing."""
    nodes_df = pd.DataFrame(
        {
            "node": list(dag.node_ids),
            "B": dag.B.astype(np.float64, copy=True),
        }
    )
    sources: list[str] = []
    targets: list[str] = []
    weights: list[float] = []
    for target_idx in range(dag.n_nodes):
        t_id = dag.node_ids[target_idx]
        start = int(dag.in_indptr[target_idx])
        end = int(dag.in_indptr[target_idx + 1])
        for e in range(start, end):
            sources.append(dag.node_ids[int(dag.in_indices[e])])
            targets.append(t_id)
            weights.append(float(dag.in_weights[e]))
    edges_df = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "S": weights,
            "tau": np.ones(len(weights), dtype=np.float64),
        }
    )
    return nodes_df, edges_df


def _measure_peak_memory_kb(fn: Callable[[], Any]) -> float:
    """Measure isolated Python heap peak allocation (KB) via tracemalloc."""
    gc.collect()
    tracemalloc.start()
    try:
        _ = fn()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return float(peak_bytes) / 1024.0


def _benchmark_callable(
    fn: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    single_run_cap_sec: float | None = None,
    capped_repeats: int = TREE_CAPPED_REPEATS,
) -> dict[str, Any]:
    """Run warm-ups + measured repetitions (perf_counter) plus isolated peak memory."""
    capped = False
    actual_warmups = int(warmups)
    actual_repeats = int(repeats)

    if single_run_cap_sec is not None:
        t0 = time.perf_counter()
        last_result = fn()
        probe_sec = time.perf_counter() - t0
        if probe_sec > single_run_cap_sec:
            capped = True
            actual_warmups = 0
            actual_repeats = 1 if probe_sec > 5.0 else min(int(capped_repeats), int(repeats))
            times_ms = [probe_sec * 1000.0]
            for _ in range(actual_repeats - 1):
                t1 = time.perf_counter()
                last_result = fn()
                times_ms.append((time.perf_counter() - t1) * 1000.0)
            peak_kb = _measure_peak_memory_kb(fn)
            arr = np.asarray(times_ms, dtype=np.float64)
            q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            return {
                "repeats": actual_repeats,
                "warmups": actual_warmups,
                "capped": capped,
                "median_ms": float(np.median(arr)),
                "q25_ms": q25,
                "q75_ms": q75,
                "iqr_ms": float(q75 - q25),
                "mean_ms": float(np.mean(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "peak_memory_kb": peak_kb,
                "last_result": last_result,
            }
        # Probe already served as 1 warm-up
        rem_warmups = max(0, actual_warmups - 1)
    else:
        rem_warmups = actual_warmups
        last_result = None

    for _ in range(rem_warmups):
        last_result = fn()

    times_ms = []
    for _ in range(actual_repeats):
        t0 = time.perf_counter()
        last_result = fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_kb = _measure_peak_memory_kb(fn)
    arr = np.asarray(times_ms, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "repeats": actual_repeats,
        "warmups": actual_warmups,
        "capped": capped,
        "median_ms": float(np.median(arr)),
        "q25_ms": q25,
        "q75_ms": q75,
        "iqr_ms": float(q75 - q25),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "peak_memory_kb": peak_kb,
        "last_result": last_result,
    }


def _ranking_sort_op(scores: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(-scores, kind="mergesort")
    groups, n_groups = assign_anchored_tie_groups(scores[order], tol=1e-12)
    return groups, n_groups


def _benchmark_deterministic_suite(
    series: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    dag: PreparedDAG,
    *,
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    """Benchmark all deterministic comparators and component stages on one DAG."""
    n = dag.n_nodes
    m = dag.n_edges
    max_possible = n * (n - 1) // 2
    density = float(m / max_possible) if max_possible > 0 else 0.0

    # Pre-evaluate R once for the standalone ranking_sort component
    R_once = compute_gbbrpm_fast(dag)

    # Verify exact parity between DataFrame evaluate_gbbrpm and CSR compute_gbbrpm_fast
    df_risk, _ = evaluate_gbbrpm(nodes_df, edges_df)
    df_vec = np.array([df_risk[nid] for nid in dag.node_ids], dtype=np.float64)
    parity_err = float(np.max(np.abs(df_vec - R_once)))
    if parity_err >= 1e-12:
        raise AssertionError(
            f"Parity error between evaluate_gbbrpm and compute_gbbrpm_fast on {series} |V|={n}: {parity_err}"
        )

    specs: list[tuple[str, Callable[[], Any], float | None]] = [
        ("graph_validation_and_prep", lambda: prepare_dag(nodes_df, edges_df), None),
        ("gbbrpm_csr", lambda: compute_gbbrpm_fast(dag), None),
        ("ranking_sort", lambda: _ranking_sort_op(R_once), None),
        ("gbbrpm_core_df", lambda: evaluate_gbbrpm(nodes_df, edges_df), None),
        ("recursive_max_M", lambda: compute_recursive_max(dag), None),
        ("clipped_sum_H", lambda: compute_clipped_sum(dag), None),
        ("local_upstream", lambda: compute_local_upstream(dag), None),
        ("abbe_lower", lambda: compute_abbe_lower(dag), None),
        ("ancestry_diagnostic", lambda: compute_ancestry_diagnostic(dag), None),
        ("full_tree_T", lambda: compute_all_tree_bounds(dag), TREE_SINGLE_RUN_CAP_SEC),
        (
            "two_stage_screening_f20",
            lambda: certify_top_k_two_stage(dag, fractions=(0.20,)),
            None,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for method_name, fn, cap_sec in specs:
        res = _benchmark_callable(
            fn,
            warmups=warmups,
            repeats=repeats,
            single_run_cap_sec=cap_sec,
        )
        n_stage2_trees = np.nan
        res_s1 = np.nan
        res_s2 = np.nan
        if method_name == "two_stage_screening_f20" and isinstance(res["last_result"], dict):
            b20 = res["last_result"]["budgets"][0.20]
            n_stage2_trees = int(b20["n_stage2_trees"])
            res_s1 = float(b20["res_s1"])
            res_s2 = float(b20["res_s2"])

        rows.append(
            {
                "series": series,
                "n_nodes": n,
                "n_edges": m,
                "density": density,
                "method": method_name,
                "repeats": res["repeats"],
                "warmups": res["warmups"],
                "capped": bool(res["capped"]),
                "median_ms": res["median_ms"],
                "q25_ms": res["q25_ms"],
                "q75_ms": res["q75_ms"],
                "iqr_ms": res["iqr_ms"],
                "mean_ms": res["mean_ms"],
                "min_ms": res["min_ms"],
                "max_ms": res["max_ms"],
                "peak_memory_kb": res["peak_memory_kb"],
                "trials": np.nan,
                "trials_per_sec": np.nan,
                "edge_trials_per_sec": np.nan,
                "n_stage2_trees": n_stage2_trees,
                "res_s1": res_s1,
                "res_s2": res_s2,
            }
        )
    return rows


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit log(y) = a + b * log(x) to quantify empirical scaling exponent."""
    lx = np.log(np.asarray(x, dtype=np.float64))
    ly = np.log(np.asarray(y, dtype=np.float64))
    slope, intercept = np.polyfit(lx, ly, 1)
    y_hat = intercept + slope * lx
    ss_res = float(np.sum((ly - y_hat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "exponent": float(slope),
        "intercept_log": float(intercept),
        "r2_log_log": float(r2),
    }


def run_benchmark(
    *,
    seed: int = DEFAULT_SEED,
    warmups: int = DEFAULT_WARMUPS,
    repeats: int = DEFAULT_REPEATS,
    mc_warmups: int = DEFAULT_MC_WARMUPS,
    mc_repeats: int = DEFAULT_MC_REPEATS,
    mc_trials: int = DEFAULT_MC_TRIALS,
    mc_batch_size: int = DEFAULT_MC_BATCH_SIZE,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    t_total_start = time.perf_counter()
    out_dir = output_dir or (ROOT / "results" / "heldout_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    # 1. Sparse series (|V| in {20, 50, 100, 250, 500, 1000}, target 3|V|)
    print("[Scalability v2] Running sparse series...", flush=True)
    for n in SPARSE_SIZES:
        nodes_df, edges_df, dag = build_synthetic_scalability_dag(
            n_nodes=n, target_edges=3 * n, series="sparse", seed=seed
        )
        rows = _benchmark_deterministic_suite(
            "sparse", nodes_df, edges_df, dag, warmups=warmups, repeats=repeats
        )
        all_rows.extend(rows)
        by_m = {r["method"]: r for r in rows}
        print(
            f"  sparse |V|={n:4d}, |E|={dag.n_edges:4d}: "
            f"csr={by_m['gbbrpm_csr']['median_ms']:.4f}ms, "
            f"df={by_m['gbbrpm_core_df']['median_ms']:.4f}ms, "
            f"tree={by_m['full_tree_T']['median_ms']:.3f}ms (cap={by_m['full_tree_T']['capped']}), "
            f"2stage={by_m['two_stage_screening_f20']['median_ms']:.3f}ms",
            flush=True,
        )

    # 2. Dense series (|V| in {20, 50, 100, 250}, |E| = floor(0.25*|V|*(|V|-1)/2))
    print("[Scalability v2] Running dense series...", flush=True)
    for n in DENSE_SIZES:
        target_e = int(math.floor(0.25 * n * (n - 1) / 2.0))
        nodes_df, edges_df, dag = build_synthetic_scalability_dag(
            n_nodes=n, target_edges=target_e, series="dense", seed=seed
        )
        rows = _benchmark_deterministic_suite(
            "dense", nodes_df, edges_df, dag, warmups=warmups, repeats=repeats
        )
        all_rows.extend(rows)
        by_m = {r["method"]: r for r in rows}
        print(
            f"  dense  |V|={n:4d}, |E|={dag.n_edges:4d}: "
            f"csr={by_m['gbbrpm_csr']['median_ms']:.4f}ms, "
            f"df={by_m['gbbrpm_core_df']['median_ms']:.4f}ms, "
            f"tree={by_m['full_tree_T']['median_ms']:.3f}ms (cap={by_m['full_tree_T']['capped']}), "
            f"2stage={by_m['two_stage_screening_f20']['median_ms']:.3f}ms",
            flush=True,
        )

    # 3. Paired Reference Timing (forward family, seed=1001, w=0.5, B=0.5)
    print("[Scalability v2] Running paired MC reference series (forward, seed=1001, w=0.5, B=0.5)...", flush=True)
    paired_scenarios = generate_dag_uniform_scenarios(
        split="eval",
        families=("forward",),
        sizes=PAIRED_MC_SIZES,
        seeds=(1001,),
        weights=(0.50,),
        b_settings=(0.5,),
        include_structural_metadata=False,
    )
    if len(paired_scenarios) != len(PAIRED_MC_SIZES):
        raise RuntimeError(f"Expected {len(PAIRED_MC_SIZES)} paired scenarios, got {len(paired_scenarios)}")

    mc_backend_seen = None
    paired_summary: dict[str, Any] = {}

    for sc in paired_scenarios:
        dag = sc.dag
        n = dag.n_nodes
        m = dag.n_edges
        max_possible = n * (n - 1) // 2
        density = float(m / max_possible) if max_possible > 0 else 0.0

        nodes_df, edges_df = _dag_to_dataframes(dag)
        det_rows = _benchmark_deterministic_suite(
            "paired_mc_reference", nodes_df, edges_df, dag, warmups=warmups, repeats=repeats
        )
        all_rows.extend(det_rows)

        # 5 repeated full N=100,000-trial MC solves
        mc_bench = _benchmark_callable(
            lambda d=dag, s=sc.mc_seed: run_mc_reference(
                d,
                n_trials=mc_trials,
                seed=s,
                batch_size=mc_batch_size,
                alpha=0.05,
            ),
            warmups=mc_warmups,
            repeats=mc_repeats,
            single_run_cap_sec=None,
        )
        mc_res = mc_bench["last_result"]
        mc_backend_seen = mc_res["backend"]
        median_sec = mc_bench["median_ms"] / 1000.0
        trials_per_sec = float(mc_trials / median_sec) if median_sec > 0 else float("nan")
        edge_trials_per_sec = float((mc_trials * m) / median_sec) if median_sec > 0 else float("nan")

        # Separate ranking sort timing on MC pi_hat
        mc_rank_bench = _benchmark_callable(
            lambda p=mc_res["pi_hat"]: _ranking_sort_op(p),
            warmups=warmups,
            repeats=repeats,
        )

        mc_row = {
            "series": "paired_mc_reference",
            "n_nodes": n,
            "n_edges": m,
            "density": density,
            "method": "mc_ref_100k",
            "repeats": mc_bench["repeats"],
            "warmups": mc_bench["warmups"],
            "capped": False,
            "median_ms": mc_bench["median_ms"],
            "q25_ms": mc_bench["q25_ms"],
            "q75_ms": mc_bench["q75_ms"],
            "iqr_ms": mc_bench["iqr_ms"],
            "mean_ms": mc_bench["mean_ms"],
            "min_ms": mc_bench["min_ms"],
            "max_ms": mc_bench["max_ms"],
            "peak_memory_kb": mc_bench["peak_memory_kb"],
            "trials": int(mc_trials),
            "trials_per_sec": trials_per_sec,
            "edge_trials_per_sec": edge_trials_per_sec,
            "n_stage2_trees": np.nan,
            "res_s1": np.nan,
            "res_s2": np.nan,
        }
        all_rows.append(mc_row)

        by_m = {r["method"]: r for r in det_rows}
        csr_ms = by_m["gbbrpm_csr"]["median_ms"]
        df_ms = by_m["gbbrpm_core_df"]["median_ms"]
        two_stage_ms = by_m["two_stage_screening_f20"]["median_ms"]
        tree_ms = by_m["full_tree_T"]["median_ms"]
        prep_ms = by_m["graph_validation_and_prep"]["median_ms"]

        paired_summary[str(n)] = {
            "scenario_id": sc.scenario_id,
            "n_nodes": n,
            "n_edges": m,
            "mc_seed": int(sc.mc_seed),
            "hoeffding_h": float(mc_res["hoeffding_h"]),
            "mc_ref_100k_median_ms": mc_bench["median_ms"],
            "mc_ref_100k_iqr_ms": mc_bench["iqr_ms"],
            "mc_ref_100k_mean_ms": mc_bench["mean_ms"],
            "mc_ref_100k_peak_memory_kb": mc_bench["peak_memory_kb"],
            "mc_prep_median_ms": prep_ms,
            "mc_ranking_sort_median_ms": mc_rank_bench["median_ms"],
            "trials_per_sec": trials_per_sec,
            "edge_trials_per_sec": edge_trials_per_sec,
            "gbbrpm_csr_median_ms": csr_ms,
            "gbbrpm_core_df_median_ms": df_ms,
            "two_stage_screening_f20_median_ms": two_stage_ms,
            "full_tree_T_median_ms": tree_ms,
            "n_stage2_trees_f20": int(by_m["two_stage_screening_f20"]["n_stage2_trees"]),
            "res_s1_f20": float(by_m["two_stage_screening_f20"]["res_s1"]),
            "res_s2_f20": float(by_m["two_stage_screening_f20"]["res_s2"]),
            "speedup_mc_vs_gbbrpm_csr": float(mc_bench["median_ms"] / csr_ms) if csr_ms > 0 else float("nan"),
            "speedup_mc_vs_gbbrpm_core_df": float(mc_bench["median_ms"] / df_ms) if df_ms > 0 else float("nan"),
            "speedup_mc_vs_two_stage_f20": float(mc_bench["median_ms"] / two_stage_ms) if two_stage_ms > 0 else float("nan"),
            "speedup_full_tree_vs_two_stage_f20": float(tree_ms / two_stage_ms) if two_stage_ms > 0 else float("nan"),
        }
        print(
            f"  paired |V|={n:4d}, |E|={m:4d}: "
            f"mc100k={mc_bench['median_ms']:.2f}ms ({trials_per_sec/1e6:.2f}M tr/s), "
            f"csr={csr_ms:.4f}ms ({mc_bench['median_ms']/csr_ms:.0f}x), "
            f"2stage={two_stage_ms:.3f}ms ({mc_bench['median_ms']/two_stage_ms:.1f}x)",
            flush=True,
        )

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "scalability_v2.csv"
    df.to_csv(csv_path, index=False, lineterminator="\n")

    # Build structured summary JSON
    def _series_table(series_name: str) -> dict[str, Any]:
        sub = df[df["series"] == series_name]
        out: dict[str, Any] = {}
        for n_val, grp in sub.groupby("n_nodes", sort=True):
            n_key = str(int(n_val))
            out[n_key] = {
                "n_nodes": int(n_val),
                "n_edges": int(grp["n_edges"].iloc[0]),
                "density": float(grp["density"].iloc[0]),
                "methods": {},
            }
            for row in grp.itertuples(index=False):
                entry: dict[str, Any] = {
                    "repeats": int(row.repeats),
                    "warmups": int(row.warmups),
                    "capped": bool(row.capped),
                    "median_ms": float(row.median_ms),
                    "q25_ms": float(row.q25_ms),
                    "q75_ms": float(row.q75_ms),
                    "iqr_ms": float(row.iqr_ms),
                    "mean_ms": float(row.mean_ms),
                    "min_ms": float(row.min_ms),
                    "max_ms": float(row.max_ms),
                    "peak_memory_kb": float(row.peak_memory_kb),
                }
                if pd.notna(row.trials):
                    entry["trials"] = int(row.trials)
                    entry["trials_per_sec"] = float(row.trials_per_sec)
                    entry["edge_trials_per_sec"] = float(row.edge_trials_per_sec)
                if pd.notna(row.n_stage2_trees):
                    entry["n_stage2_trees"] = int(row.n_stage2_trees)
                    entry["res_s1"] = float(row.res_s1)
                    entry["res_s2"] = float(row.res_s2)
                out[n_key]["methods"][str(row.method)] = entry
        return out

    # Empirical scaling fits vs |V| + |E|
    scaling_fits: dict[str, Any] = {}
    for series_name in ("sparse", "dense"):
        scaling_fits[series_name] = {}
        for m_name in ("gbbrpm_csr", "gbbrpm_core_df", "recursive_max_M", "abbe_lower", "ancestry_diagnostic", "full_tree_T"):
            sub = df[(df["series"] == series_name) & (df["method"] == m_name)].sort_values("n_nodes")
            x_ve = (sub["n_nodes"] + sub["n_edges"]).to_numpy(dtype=np.float64)
            y_ms = sub["median_ms"].to_numpy(dtype=np.float64)
            scaling_fits[series_name][m_name] = _fit_power_law(x_ve, y_ms)

    import scipy  # local import for version logging

    summary: dict[str, Any] = {
        "benchmark_version": "2.0.0",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_runtime_sec": float(time.perf_counter() - t_total_start),
        "protocol": {
            "sparse_sizes": list(SPARSE_SIZES),
            "sparse_target_edges": "3 * |V|",
            "dense_sizes": list(DENSE_SIZES),
            "dense_target_edges": "floor(0.25 * |V| * (|V| - 1) / 2)",
            "seed": int(seed),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "paired_mc_family": "forward",
            "paired_mc_seed": 1001,
            "paired_mc_w": 0.50,
            "paired_mc_B": 0.50,
            "paired_mc_sizes": list(PAIRED_MC_SIZES),
            "mc_trials": int(mc_trials),
            "mc_batch_size": int(mc_batch_size),
            "mc_warmups": int(mc_warmups),
            "mc_repeats": int(mc_repeats),
            "mc_backend": mc_backend_seen,
            "tree_single_run_cap_sec": float(TREE_SINGLE_RUN_CAP_SEC),
            "timer_instrumentation": "time.perf_counter() in milliseconds across separate non-traced repetitions",
            "memory_instrumentation": (
                "tracemalloc.get_traced_memory() peak Python heap allocation (KB) "
                "measured in a dedicated isolated post-timing pass after gc.collect() "
                "so profiler overhead does not affect wall-clock timings"
            ),
        },
        "environment": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "networkx": nx.__version__,
                "scipy": scipy.__version__,
            },
        },
        "row_count": int(len(df)),
        "sparse_series": _series_table("sparse"),
        "dense_series": _series_table("dense"),
        "paired_mc_reference": paired_summary,
        "scaling_fits_vs_V_plus_E": scaling_fits,
    }

    summary_path = out_dir / "scalability_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[Scalability v2] Wrote {csv_path} ({len(df)} rows) and {summary_path}", flush=True)
    return df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--mc-warmups", type=int, default=DEFAULT_MC_WARMUPS)
    parser.add_argument("--mc-repeats", type=int, default=DEFAULT_MC_REPEATS)
    parser.add_argument("--mc-trials", type=int, default=DEFAULT_MC_TRIALS)
    parser.add_argument("--mc-batch-size", type=int, default=DEFAULT_MC_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        seed=args.seed,
        warmups=args.warmups,
        repeats=args.repeats,
        mc_warmups=args.mc_warmups,
        mc_repeats=args.mc_repeats,
        mc_trials=args.mc_trials,
        mc_batch_size=args.mc_batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
