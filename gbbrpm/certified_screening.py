"""Two-Stage Deterministic Certified Top-k Screening (Upgrades 1 & 3).

Implements Equations (3-14a), (3-14b), (3-14c), and (3-17a) from Chapter 3
(Section 3.6.3 and Section 3.6.6):

- Candidate set ``C = {j in V : B_j == 0.0}``.
- Stage 1 (``O(|V| + |E|)`` scalar pass + sort): uses ``[M_j, R_j]`` and
  tolerance ``epsilon = 1e-12`` to partition ``C`` into certified-IN ``I_k^(1)``,
  certified-OUT ``O_k^(1)``, and uncertified boundary band ``U_k^(1)``.
- Stage 2 (targeted ``T_j`` refinement): computes the unthresholded target-specific
  in-arborescence lower bound ``T_j`` STRICTLY on ``j in U_k^(1)`` only,
  tightening the partition to ``I_k^(2), O_k^(2), U_k^(2)`` with verified
  monotonic refinement ``I_k^(1) <= I_k^(2)``, ``O_k^(1) <= O_k^(2)``,
  ``U_k^(2) <= U_k^(1)``.
- Zero-cost scalar bracket width ``W_j = R_j - M_j`` (Eq. 3-14c).
- ``assert_zero_false_certifications``: automated fatal check against exact
  ``pi_j`` or nonempty intersected intervals ``[L'_j, U'_j]``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from .benchmark_comparators import (
    compute_gbbrpm_fast,
    compute_recursive_max,
    compute_target_tree_bound,
)
from .csr_graph import PreparedDAG


def get_candidate_indices(dag: PreparedDAG) -> np.ndarray:
    """Return topological indices of the receiving candidate set ``C = {j : B_j == 0}``."""
    return np.flatnonzero(dag.B == 0.0).astype(np.int32)


def certify_top_k_two_stage(
    dag: PreparedDAG,
    fractions: Sequence[float] = (0.10, 0.20, 0.30),
    epsilon: float = 1e-12,
    precomputed_M: np.ndarray | None = None,
    precomputed_R: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run Two-Stage Deterministic Certified Top-k Screening across fixed budgets.

    Parameters
    ----------
    dag:
        Prepared topological DAG.
    fractions:
        Budget fractions ``f`` in ``(0, 1]``; each yields ``k = max(1, ceil(f * |C|))``.
    epsilon:
        Strict-inequality numerical tolerance guard (default ``1e-12``).
    precomputed_M, precomputed_R:
        Optional precomputed ``M_j`` and ``R_j`` arrays of shape ``(dag.n_nodes,)``
        in topological index order.

    Returns
    -------
    dict with candidate metadata, global ``M``, ``R``, ``W = R - M`` summaries,
    and per-fraction certification dictionaries under ``budgets``.
    """
    eps = float(epsilon)
    if not (math.isfinite(eps) and eps >= 0.0):
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon!r}")

    frac_list = [float(f) for f in fractions]
    for f in frac_list:
        if not (math.isfinite(f) and 0.0 < f <= 1.0):
            raise ValueError(f"Each fraction must be in (0, 1], got {f!r}")

    # Stage 1 global O(|V| + |E|) scalar pass
    t_global = time.perf_counter()
    if precomputed_M is None:
        M = compute_recursive_max(dag)
    else:
        M = np.asarray(precomputed_M, dtype=np.float64)
        if M.shape != (dag.n_nodes,):
            raise ValueError(
                f"precomputed_M must have shape ({dag.n_nodes},), got {M.shape}"
            )
    if precomputed_R is None:
        R = compute_gbbrpm_fast(dag)
    else:
        R = np.asarray(precomputed_R, dtype=np.float64)
        if R.shape != (dag.n_nodes,):
            raise ValueError(
                f"precomputed_R must have shape ({dag.n_nodes},), got {R.shape}"
            )

    W = R - M
    global_stage1_pass_sec = time.perf_counter() - t_global

    cand_idx = get_candidate_indices(dag)
    n_cand = int(cand_idx.size)
    M_c = M[cand_idx]
    R_c = R[cand_idx]
    W_c = W[cand_idx]

    if n_cand > 0:
        w_summary = {
            "mean": float(np.mean(W_c)),
            "max": float(np.max(W_c)),
            "p90": float(np.percentile(W_c, 90.0)),
        }
        R_desc_order = np.argsort(-R_c, kind="mergesort")
        R_sorted = R_c[R_desc_order]
        M_sorted = np.sort(M_c)[::-1]
    else:
        w_summary = {
            "mean": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
        }
        R_desc_order = np.empty(0, dtype=np.int64)
        R_sorted = np.empty(0, dtype=np.float64)
        M_sorted = np.empty(0, dtype=np.float64)

    # Cache (T_j, single_tree_elapsed_sec) across budgets within this call while
    # attributing exact standalone tree compute time to every budget.
    tree_cache: dict[int, tuple[float, float]] = {}
    budgets: dict[float, dict[str, Any]] = {}

    for f in frac_list:
        if n_cand == 0:
            empty_i32 = np.empty(0, dtype=np.int32)
            budgets[f] = {
                "fraction": f,
                "k": 0,
                "n_candidates": 0,
                "trivial": True,
                "R_k_plus_1": float("nan"),
                "M_k": float("nan"),
                "ell_2_k": float("nan"),
                "I_1": empty_i32.copy(),
                "O_1": empty_i32.copy(),
                "U_1": empty_i32.copy(),
                "I_2": empty_i32.copy(),
                "O_2": empty_i32.copy(),
                "U_2": empty_i32.copy(),
                "n_I_1": 0,
                "n_O_1": 0,
                "n_U_1": 0,
                "n_I_2": 0,
                "n_O_2": 0,
                "n_U_2": 0,
                "res_s1": 1.0,
                "band_s1": 0.0,
                "res_s2": 1.0,
                "band_s2": 0.0,
                "band_reduction": 0.0,
                "fully_certified_s1": True,
                "fully_certified_s2": True,
                "stage1_time_sec": float(global_stage1_pass_sec),
                "stage2_time_sec": 0.0,
                "total_screening_time_sec": float(global_stage1_pass_sec),
                "n_stage2_trees": 0,
                "w_mean": float("nan"),
                "w_max": float("nan"),
                "w_p90": float("nan"),
                "top_k_boundary_w_mean": float("nan"),
                "boundary_band_w_mean": float("nan"),
            }
            continue

        k = max(1, int(math.ceil(f * n_cand)))
        if k >= n_cand:
            # Trivial budget: all candidates certified IN
            empty_i32 = np.empty(0, dtype=np.int32)
            all_in = cand_idx.copy()
            budgets[f] = {
                "fraction": f,
                "k": n_cand,
                "n_candidates": n_cand,
                "trivial": True,
                "R_k_plus_1": float("nan"),
                "M_k": float(M_sorted[-1]),
                "ell_2_k": float(M_sorted[-1]),
                "I_1": all_in.copy(),
                "O_1": empty_i32.copy(),
                "U_1": empty_i32.copy(),
                "I_2": all_in.copy(),
                "O_2": empty_i32.copy(),
                "U_2": empty_i32.copy(),
                "n_I_1": n_cand,
                "n_O_1": 0,
                "n_U_1": 0,
                "n_I_2": n_cand,
                "n_O_2": 0,
                "n_U_2": 0,
                "res_s1": 1.0,
                "band_s1": 0.0,
                "res_s2": 1.0,
                "band_s2": 0.0,
                "band_reduction": 0.0,
                "fully_certified_s1": True,
                "fully_certified_s2": True,
                "stage1_time_sec": float(global_stage1_pass_sec),
                "stage2_time_sec": 0.0,
                "total_screening_time_sec": float(global_stage1_pass_sec),
                "n_stage2_trees": 0,
                "w_mean": w_summary["mean"],
                "w_max": w_summary["max"],
                "w_p90": w_summary["p90"],
                "top_k_boundary_w_mean": w_summary["mean"],
                "boundary_band_w_mean": 0.0,
            }
            continue

        # Nontrivial budget: 1 <= k < n_cand
        t_s1 = time.perf_counter()
        R_k_plus_1 = float(R_sorted[k])
        M_k = float(M_sorted[k - 1])

        in_mask_1 = M_c > (R_k_plus_1 + eps)
        out_mask_1 = R_c < (M_k - eps)
        uncert_mask_1 = ~(in_mask_1 | out_mask_1)
        s1_part_sec = time.perf_counter() - t_s1
        stage1_time_sec = float(global_stage1_pass_sec + s1_part_sec)

        # Stage 2: compute T_j STRICTLY for j in U_k^(1)
        uncert_positions_1 = np.flatnonzero(uncert_mask_1)
        ell_2 = M_c.copy()
        s2_tree_sec = 0.0

        for pos in uncert_positions_1:
            node_idx = int(cand_idx[int(pos)])
            cached = tree_cache.get(node_idx)
            if cached is None:
                t_tr = time.perf_counter()
                t_val = compute_target_tree_bound(dag, node_idx)
                dt_tr = time.perf_counter() - t_tr
                cached = (t_val, dt_tr)
                tree_cache[node_idx] = cached
            t_val, dt_tr = cached
            s2_tree_sec += dt_tr
            m_val = float(M_c[int(pos)])
            if t_val < m_val - 1e-12:
                raise AssertionError(
                    f"Tree lower bound violated M_j <= T_j at node {dag.node_ids[node_idx]}: "
                    f"T_j={t_val:.15g} < M_j={m_val:.15g}"
                )
            # Guard 1-ULP floating-point rounding without masking any real violation
            ell_2[int(pos)] = t_val if t_val >= m_val else m_val

        t_s2_part = time.perf_counter()
        ell_2_sorted = np.sort(ell_2)[::-1]
        ell_2_k = float(ell_2_sorted[k - 1])

        in_mask_2 = ell_2 > (R_k_plus_1 + eps)
        out_mask_2 = R_c < (ell_2_k - eps)
        uncert_mask_2 = ~(in_mask_2 | out_mask_2)
        s2_part_sec = time.perf_counter() - t_s2_part
        stage2_time_sec = float(s2_tree_sec + s2_part_sec)

        # Mandatory structural invariants (Eqs. 3-14a & 3-14b)
        if np.any(in_mask_1 & out_mask_1):
            raise AssertionError("Stage 1 certified-IN and certified-OUT sets overlap")
        if np.any(in_mask_2 & out_mask_2):
            raise AssertionError("Stage 2 certified-IN and certified-OUT sets overlap")
        if not np.all(in_mask_1 <= in_mask_2):
            raise AssertionError("Stage 2 violated monotonic refinement: I_1 not subset of I_2")
        if not np.all(out_mask_1 <= out_mask_2):
            raise AssertionError("Stage 2 violated monotonic refinement: O_1 not subset of O_2")
        if not np.all(uncert_mask_2 <= uncert_mask_1):
            raise AssertionError("Stage 2 violated monotonic refinement: U_2 not subset of U_1")

        I_1 = cand_idx[in_mask_1]
        O_1 = cand_idx[out_mask_1]
        U_1 = cand_idx[uncert_mask_1]
        I_2 = cand_idx[in_mask_2]
        O_2 = cand_idx[out_mask_2]
        U_2 = cand_idx[uncert_mask_2]

        n_I_1 = int(I_1.size)
        n_O_1 = int(O_1.size)
        n_U_1 = int(U_1.size)
        n_I_2 = int(I_2.size)
        n_O_2 = int(O_2.size)
        n_U_2 = int(U_2.size)

        res_s1 = float((n_I_1 + n_O_1) / n_cand)
        band_s1 = float(n_U_1 / n_cand)
        res_s2 = float((n_I_2 + n_O_2) / n_cand)
        band_s2 = float(n_U_2 / n_cand)

        if n_U_1 > 0:
            boundary_band_w_mean = float(np.mean(W_c[uncert_mask_1]))
            top_k_boundary_w_mean = boundary_band_w_mean
        else:
            boundary_band_w_mean = 0.0
            # Boundary pair around nominal cutoff k in descending R order
            b_pos = R_desc_order[k - 1 : k + 1]
            top_k_boundary_w_mean = float(np.mean(W_c[b_pos]))

        budgets[f] = {
            "fraction": f,
            "k": int(k),
            "n_candidates": n_cand,
            "trivial": False,
            "R_k_plus_1": R_k_plus_1,
            "M_k": M_k,
            "ell_2_k": ell_2_k,
            "I_1": I_1,
            "O_1": O_1,
            "U_1": U_1,
            "I_2": I_2,
            "O_2": O_2,
            "U_2": U_2,
            "n_I_1": n_I_1,
            "n_O_1": n_O_1,
            "n_U_1": n_U_1,
            "n_I_2": n_I_2,
            "n_O_2": n_O_2,
            "n_U_2": n_U_2,
            "res_s1": res_s1,
            "band_s1": band_s1,
            "res_s2": res_s2,
            "band_s2": band_s2,
            "band_reduction": float(band_s1 - band_s2),
            "fully_certified_s1": bool(n_U_1 == 0),
            "fully_certified_s2": bool(n_U_2 == 0),
            "stage1_time_sec": stage1_time_sec,
            "stage2_time_sec": stage2_time_sec,
            "total_screening_time_sec": float(stage1_time_sec + stage2_time_sec),
            "n_stage2_trees": n_U_1,
            "w_mean": w_summary["mean"],
            "w_max": w_summary["max"],
            "w_p90": w_summary["p90"],
            "top_k_boundary_w_mean": top_k_boundary_w_mean,
            "boundary_band_w_mean": boundary_band_w_mean,
        }

    total_unique_tree_time = float(sum(dt for _, dt in tree_cache.values()))

    return {
        "candidate_indices": cand_idx,
        "n_candidates": n_cand,
        "epsilon": eps,
        "M": M,
        "R": R,
        "W": W,
        "w_summary": w_summary,
        "global_stage1_pass_sec": float(global_stage1_pass_sec),
        "n_unique_stage2_trees": int(len(tree_cache)),
        "total_unique_stage2_tree_time_sec": total_unique_tree_time,
        "cached_tree_values": {idx: val for idx, (val, _) in tree_cache.items()},
        "budgets": budgets,
    }


def assert_zero_false_certifications(
    screening_result: dict[str, Any],
    pi_exact: np.ndarray | None = None,
    pi_lower: np.ndarray | None = None,
    pi_upper: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Verify zero false certifications against exact ``pi`` and/or reference intervals.

    Raises ``AssertionError`` if:
    - Any certified-IN node could fall outside an exact top-``k`` set (exact mode:
      ``pi_a <= pi_{(k+1)}``; interval mode: ``U'_a < L'_{(k)} - epsilon``), OR
    - Any certified-OUT node could fall inside an exact top-``k`` set (exact mode:
      ``pi_b >= pi_{(k)}``; interval mode: ``L'_b > U'_{(k+1)} + epsilon``), OR
    - Any reference interval is empty (``pi_lower > pi_upper``).
    """
    if pi_exact is None and (pi_lower is None or pi_upper is None):
        raise ValueError(
            "Provide pi_exact and/or both pi_lower and pi_upper for verification"
        )
    if (pi_lower is None) != (pi_upper is None):
        raise ValueError("pi_lower and pi_upper must be provided together")

    eps = float(epsilon)
    cand_idx = np.asarray(screening_result["candidate_indices"], dtype=np.int32)
    n_cand = int(cand_idx.size)
    budgets = screening_result["budgets"]

    exact_arr: np.ndarray | None = None
    if pi_exact is not None:
        exact_arr = np.asarray(pi_exact, dtype=np.float64)
        if exact_arr.ndim != 1:
            raise ValueError("pi_exact must be a 1-D array")
        if not np.all(np.isfinite(exact_arr)):
            raise AssertionError("pi_exact contains non-finite values")

    lo_arr: np.ndarray | None = None
    hi_arr: np.ndarray | None = None
    if pi_lower is not None and pi_upper is not None:
        lo_arr = np.asarray(pi_lower, dtype=np.float64)
        hi_arr = np.asarray(pi_upper, dtype=np.float64)
        if lo_arr.shape != hi_arr.shape or lo_arr.ndim != 1:
            raise ValueError("pi_lower and pi_upper must be matching 1-D arrays")
        if not (np.all(np.isfinite(lo_arr)) and np.all(np.isfinite(hi_arr))):
            raise AssertionError("Reference intervals contain non-finite values")
        if n_cand > 0 and np.any(lo_arr[cand_idx] > hi_arr[cand_idx]):
            n_empty = int(np.sum(lo_arr[cand_idx] > hi_arr[cand_idx]))
            raise AssertionError(
                f"Cannot certify against {n_empty} empty reference interval(s) (L' > U')"
            )

    checked_budgets = 0
    for frac, entry in budgets.items():
        checked_budgets += 1
        k = int(entry["k"])
        if n_cand == 0 or k >= n_cand:
            continue

        if exact_arr is not None:
            pi_c = exact_arr[cand_idx]
            pi_sorted = np.sort(pi_c)[::-1]
            pi_k = float(pi_sorted[k - 1])
            pi_k_plus_1 = float(pi_sorted[k])

            for stage in (1, 2):
                in_nodes = np.asarray(entry[f"I_{stage}"], dtype=np.int32)
                out_nodes = np.asarray(entry[f"O_{stage}"], dtype=np.int32)

                for a in in_nodes:
                    pa = float(exact_arr[int(a)])
                    if pa <= pi_k_plus_1:
                        raise AssertionError(
                            f"False certified-IN at f={frac}, stage={stage}, "
                            f"node_idx={int(a)}: pi_a={pa:.15g} <= pi_(k+1)={pi_k_plus_1:.15g}"
                        )
                for b in out_nodes:
                    pb = float(exact_arr[int(b)])
                    if pb >= pi_k:
                        raise AssertionError(
                            f"False certified-OUT at f={frac}, stage={stage}, "
                            f"node_idx={int(b)}: pi_b={pb:.15g} >= pi_(k)={pi_k:.15g}"
                        )

        if lo_arr is not None and hi_arr is not None:
            lo_c = lo_arr[cand_idx]
            hi_c = hi_arr[cand_idx]
            lo_sorted = np.sort(lo_c)[::-1]
            hi_sorted = np.sort(hi_c)[::-1]
            lo_k = float(lo_sorted[k - 1])
            hi_k_plus_1 = float(hi_sorted[k])

            for stage in (1, 2):
                in_nodes = np.asarray(entry[f"I_{stage}"], dtype=np.int32)
                out_nodes = np.asarray(entry[f"O_{stage}"], dtype=np.int32)

                for a in in_nodes:
                    ua = float(hi_arr[int(a)])
                    if ua < lo_k - eps:
                        raise AssertionError(
                            f"Interval-contradicted certified-IN at f={frac}, stage={stage}, "
                            f"node_idx={int(a)}: U'_a={ua:.15g} < L'_(k)-eps={lo_k - eps:.15g}"
                        )
                for b in out_nodes:
                    lb_val = float(lo_arr[int(b)])
                    if lb_val > hi_k_plus_1 + eps:
                        raise AssertionError(
                            f"Interval-contradicted certified-OUT at f={frac}, stage={stage}, "
                            f"node_idx={int(b)}: L'_b={lb_val:.15g} > U'_(k+1)+eps={hi_k_plus_1 + eps:.15g}"
                        )

    return {
        "ok": True,
        "checked_budgets": checked_budgets,
        "n_false_in": 0,
        "n_false_out": 0,
    }
