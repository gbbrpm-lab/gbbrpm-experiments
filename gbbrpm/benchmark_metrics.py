"""Bound tightness, fixed-budget ranking, reversal, saturation, and W_j reliability metrics.

Implements the pre-registered evaluation metrics in Chapter 3
(Section 3.6.6 and Section 3.11.1):

- Anchored ``1e-12`` descending score-tie grouping (``assign_anchored_tie_groups``).
- Fixed-budget top-``k`` selection averaged over 100 deterministic random node-order
  permutations (``seed=140``, shared across methods and independent of reference
  outcomes), selecting **exactly** ``k`` nodes per permutation.
- Equation (3-17) captured activation mass and selection regret with explicit
  all-zero denominator handling and conservative symmetric-difference uncertainty
  bound ``h * |sym_diff|``.
- Non-repaired interval intersection ``intersect_intervals`` between Monte Carlo
  Hoeffding bounds and analytic brackets.
- Strict and interval-resolved pairwise reversal analysis (``compute_pairwise_reversals``).
- Saturation metrics (``>= 0.95``, ``>= 1 - 1e-12``, distinct anchored score groups)
  and tie-aware Spearman correlation (``nan`` on constant rank vectors).
- Self-diagnosing ``W_j = R_j - M_j`` reliability evaluation (nodewise Spearman vs.
  actual slack and scenario-level threshold precision/recall).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def assign_anchored_tie_groups(
    scores: np.ndarray | Sequence[float], tol: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Group scores in descending order using each group's highest score as anchor.

    Starts a new group whenever ``anchor - score > tol``. Returns ``(group_ids, n_groups)``
    where ``group_ids[i]`` is the 0-based descending group index of item ``i``
    (``0`` = highest score group) and ``n_groups`` is the number of distinct groups.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"scores must be a 1-D array, got shape {arr.shape}")
    t = float(tol)
    if not (math.isfinite(t) and t >= 0.0):
        raise ValueError(f"tol must be finite and >= 0, got {tol!r}")

    m = int(arr.size)
    if m == 0:
        return np.empty(0, dtype=np.int32), 0
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must contain strictly finite numeric values")

    order = np.argsort(-arr, kind="mergesort")
    groups = np.empty(m, dtype=np.int32)
    gid = 0
    anchor = float(arr[int(order[0])])
    groups[int(order[0])] = 0

    for pos in range(1, m):
        idx = int(order[pos])
        val = float(arr[idx])
        if anchor - val > t:
            gid += 1
            anchor = val
        groups[idx] = gid

    return groups, gid + 1


def generate_tie_permutations(
    n_items: int, n_perms: int = 100, seed: int = 140
) -> np.ndarray:
    """Generate ``n_perms`` deterministic random node-order permutations with ``PCG64(seed)``.

    Returns an array of shape ``(n_perms, n_items)`` (dtype ``int32``) where row ``p``
    gives the permutation priority rank ``rank[item]`` of each item in ``0 .. n_items - 1``.
    Shared across all methods and completely independent of reference outcomes.
    """
    if n_items < 0:
        raise ValueError(f"n_items must be >= 0, got {n_items}")
    if n_perms < 1:
        raise ValueError(f"n_perms must be >= 1, got {n_perms}")

    perms = np.empty((int(n_perms), int(n_items)), dtype=np.int32)
    if n_items == 0:
        return perms

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for p in range(int(n_perms)):
        perms[p] = rng.permutation(n_items).astype(np.int32, copy=False)
    return perms


def select_fixed_budget_topk_sets(
    scores: np.ndarray | Sequence[float],
    k: int,
    perms: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """Select **exactly** ``k`` candidate indices per permutation using anchored tie groups.

    Parameters
    ----------
    scores:
        1-D score vector over the candidate set of length ``m``.
    k:
        Fixed selection budget (``0 <= k <= m``).
    perms:
        Permutation matrix of shape ``(n_perms, m)`` from ``generate_tie_permutations``.
        Entry ``perms[p, i]`` is the tie-breaking rank (smaller wins) of item ``i``
        under permutation ``p``.
    tol:
        Anchored tie-group tolerance (default ``1e-12``).

    Returns
    -------
    np.ndarray of shape ``(n_perms, k)`` (dtype ``int32``) containing selected
    candidate indices for each permutation.
    """
    arr = np.asarray(scores, dtype=np.float64)
    m = int(arr.size)
    if k < 0 or k > m:
        raise ValueError(f"k must satisfy 0 <= k <= {m}, got {k}")
    n_perms = int(perms.shape[0])
    if perms.ndim != 2 or perms.shape[1] != m:
        raise ValueError(
            f"perms must have shape (n_perms, {m}), got {perms.shape}"
        )
    if k == 0:
        return np.empty((n_perms, 0), dtype=np.int32)
    if k == m:
        base = np.arange(m, dtype=np.int32)
        return np.broadcast_to(base[None, :], (n_perms, m)).copy()

    groups, n_groups = assign_anchored_tie_groups(arr, tol=tol)
    counts = np.bincount(groups, minlength=n_groups)
    cum_counts = np.cumsum(counts)
    boundary_gid = int(np.searchsorted(cum_counts, k, side="left"))

    strict_above = np.flatnonzero(groups < boundary_gid).astype(np.int32)
    need = k - int(strict_above.size)

    if need == 0:
        return np.broadcast_to(strict_above[None, :], (n_perms, k)).copy()

    boundary_items = np.flatnonzero(groups == boundary_gid).astype(np.int32)
    if need == int(boundary_items.size):
        combined = np.concatenate((strict_above, boundary_items))
        return np.broadcast_to(combined[None, :], (n_perms, k)).copy()

    # Break ties strictly within boundary_items using perms[:, boundary_items]
    out = np.empty((n_perms, k), dtype=np.int32)
    n_strict = int(strict_above.size)
    if n_strict > 0:
        out[:, :n_strict] = strict_above[None, :]

    b_ranks = perms[:, boundary_items]
    # Argpartition or argsort ascending by permutation rank
    chosen_local = np.argsort(b_ranks, axis=1, kind="mergesort")[:, :need]
    out[:, n_strict:] = boundary_items[chosen_local]
    return out


def symmetric_difference_mass_bound(
    set_a: np.ndarray | Sequence[int],
    set_b: np.ndarray | Sequence[int],
    h: float,
) -> float:
    """Conservative mass-difference uncertainty bound ``h * |A symmetric_diff B|``."""
    hv = float(h)
    if not (math.isfinite(hv) and hv >= 0.0):
        raise ValueError(f"h must be finite and >= 0, got {h!r}")
    sa = {int(x) for x in set_a}
    sb = {int(x) for x in set_b}
    sym_diff = len(sa ^ sb)
    return float(hv * sym_diff)


def compute_capture_and_regret(
    scores: np.ndarray | Sequence[float],
    ref_pi: np.ndarray | Sequence[float],
    candidate_mask: np.ndarray | Sequence[bool],
    fractions: Sequence[float] = (0.10, 0.20, 0.30),
    seed: int = 140,
    n_perms: int = 100,
    tol: float = 1e-12,
    h: float | None = None,
) -> dict[str, Any]:
    """Compute Eq. (3-17) captured activation mass and fixed-budget selection regret.

    Evaluates over candidate set ``C`` defined by ``candidate_mask``, averaging
    over ``n_perms`` deterministic permutations (``seed=140``) selecting exactly
    ``k = max(1, ceil(f * |C|))`` nodes per permutation using anchored ``1e-12``
    tie groups. Explicitly flags ``all_zero_reference=True`` with ``capture=nan``
    when ``sum_{j in C} ref_pi[j] == 0``.
    """
    s_all = np.asarray(scores, dtype=np.float64)
    pi_all = np.asarray(ref_pi, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=np.bool_)
    if s_all.ndim != 1 or pi_all.shape != s_all.shape or mask.shape != s_all.shape:
        raise ValueError(
            "scores, ref_pi, and candidate_mask must be 1-D arrays of equal length"
        )
    if h is not None and not (math.isfinite(float(h)) and float(h) >= 0.0):
        raise ValueError(f"h must be finite and >= 0, got {h!r}")

    cand_idx = np.flatnonzero(mask)
    m = int(cand_idx.size)
    s_c = s_all[cand_idx]
    pi_c = pi_all[cand_idx]

    if m > 0:
        if not (np.all(np.isfinite(s_c)) and np.all(np.isfinite(pi_c))):
            raise ValueError("Candidate scores and reference probabilities must be finite")

    total_mass = float(np.sum(pi_c)) if m > 0 else 0.0
    all_zero_reference = bool(total_mass <= 0.0)
    perms = generate_tie_permutations(m, n_perms=n_perms, seed=seed)
    pi_desc = np.sort(pi_c)[::-1] if m > 0 else np.empty(0, dtype=np.float64)

    budgets: dict[float, dict[str, Any]] = {}
    for raw_f in fractions:
        f = float(raw_f)
        if not (math.isfinite(f) and 0.0 < f <= 1.0):
            raise ValueError(f"Each fraction must be in (0, 1], got {raw_f!r}")

        if m == 0:
            budgets[f] = {
                "fraction": f,
                "k": 0,
                "n_candidates": 0,
                "selected_mass": 0.0,
                "oracle_mass": 0.0,
                "total_candidate_mass": 0.0,
                "capture": float("nan"),
                "regret": 0.0,
                "all_zero_reference": True,
                "mean_sym_diff_to_oracle": 0.0,
                "mass_uncertainty_bound": 0.0 if h is not None else float("nan"),
                "regret_resolved": False if h is not None else None,
            }
            continue

        k = max(1, min(m, int(math.ceil(f * m))))
        sel_sets = select_fixed_budget_topk_sets(s_c, k, perms, tol=tol)
        sel_masses = pi_c[sel_sets].sum(axis=1)
        mean_sel_mass = float(np.mean(sel_masses))
        oracle_mass = float(np.sum(pi_desc[:k]))

        raw_regret = oracle_mass - mean_sel_mass
        if -1e-14 < raw_regret < 0.0:
            raw_regret = 0.0
        regret = float(raw_regret)
        capture = float("nan") if all_zero_reference else float(mean_sel_mass / total_mass)

        # Symmetric difference against oracle sets (using same tie perms)
        oracle_sets = select_fixed_budget_topk_sets(pi_c, k, perms, tol=tol)
        sym_diffs = np.empty(int(n_perms), dtype=np.float64)
        for p in range(int(n_perms)):
            inter = np.intersect1d(sel_sets[p], oracle_sets[p], assume_unique=True).size
            sym_diffs[p] = float(2 * (k - int(inter)))
        mean_sym_diff = float(np.mean(sym_diffs))

        if h is not None:
            mass_unc = float(float(h) * mean_sym_diff)
            regret_resolved: bool | None = bool(regret > mass_unc)
        else:
            mass_unc = float("nan")
            regret_resolved = None

        budgets[f] = {
            "fraction": f,
            "k": int(k),
            "n_candidates": m,
            "selected_mass": mean_sel_mass,
            "oracle_mass": oracle_mass,
            "total_candidate_mass": total_mass,
            "capture": capture,
            "regret": regret,
            "all_zero_reference": all_zero_reference,
            "mean_sym_diff_to_oracle": mean_sym_diff,
            "mass_uncertainty_bound": mass_unc,
            "regret_resolved": regret_resolved,
        }

    return {
        "n_candidates": m,
        "total_candidate_mass": total_mass,
        "all_zero_reference": all_zero_reference,
        "seed": int(seed),
        "n_perms": int(n_perms),
        "tol": float(tol),
        "hoeffding_h": float(h) if h is not None else None,
        "budgets": budgets,
    }


def intersect_intervals(
    mc_lower: np.ndarray | Sequence[float],
    mc_upper: np.ndarray | Sequence[float],
    analytic_lower: np.ndarray | Sequence[float],
    analytic_upper: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect Monte Carlo and analytic intervals WITHOUT silently repairing empty ones.

    Returns ``(L_prime, U_prime, empty_mask)`` where ``empty_mask = L_prime > U_prime``.
    """
    mc_lo = np.asarray(mc_lower, dtype=np.float64)
    mc_hi = np.asarray(mc_upper, dtype=np.float64)
    an_lo = np.asarray(analytic_lower, dtype=np.float64)
    an_hi = np.asarray(analytic_upper, dtype=np.float64)

    if not (mc_lo.shape == mc_hi.shape == an_lo.shape == an_hi.shape):
        raise ValueError("All four interval bound arrays must have identical shapes")

    L_prime = np.maximum(mc_lo, an_lo)
    U_prime = np.minimum(mc_hi, an_hi)
    empty_mask = L_prime > U_prime
    return L_prime, U_prime, empty_mask


def compute_pairwise_reversals(
    model_scores: np.ndarray | Sequence[float],
    ref_lower: np.ndarray | Sequence[float],
    ref_upper: np.ndarray | Sequence[float],
    candidate_mask: np.ndarray | Sequence[bool],
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Analyze strict and interval-resolved pairwise ordering reversals on candidate set ``C``.

    A pair ``(a, b)`` with nonempty reference intervals is:
    - Strict model pair: ``R_a > R_b + eps`` (or vice versa).
    - Resolved reference pair: ``U'_a < L'_b - eps`` or ``L'_a > U'_b + eps``.
    - Strict reversal: ``R_a > R_b + eps`` and ``U'_a < L'_b - eps`` (or vice versa).

    For exact references, pass ``ref_lower = ref_upper = pi_exact``.
    """
    r_all = np.asarray(model_scores, dtype=np.float64)
    lo_all = np.asarray(ref_lower, dtype=np.float64)
    hi_all = np.asarray(ref_upper, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=np.bool_)

    if not (r_all.ndim == 1 and r_all.shape == lo_all.shape == hi_all.shape == mask.shape):
        raise ValueError(
            "model_scores, ref_lower, ref_upper, and candidate_mask must be matching 1-D arrays"
        )

    e = float(eps)
    if not (math.isfinite(e) and e >= 0.0):
        raise ValueError(f"eps must be finite and >= 0, got {eps!r}")

    cand_idx = np.flatnonzero(mask)
    m = int(cand_idx.size)
    n_total_pairs = m * (m - 1) // 2

    if m < 2:
        n_empty = int(np.sum(lo_all[cand_idx] > hi_all[cand_idx])) if m == 1 else 0
        return {
            "n_candidates": m,
            "n_total_pairs": 0,
            "n_valid_pairs": 0,
            "n_empty_intervals": n_empty,
            "n_strict_model_pairs": 0,
            "n_model_tie_pairs": 0,
            "n_resolved_ref_pairs": 0,
            "n_unresolved_ref_pairs": 0,
            "n_resolved_strict_model_pairs": 0,
            "n_unresolved_strict_model_pairs": 0,
            "n_concordant": 0,
            "n_reversals": 0,
            "resolved_fraction_of_strict_model": float("nan"),
            "reversal_rate": float("nan"),
        }

    r = r_all[cand_idx]
    lo = lo_all[cand_idx]
    hi = hi_all[cand_idx]
    valid = (lo <= hi) & np.isfinite(r) & np.isfinite(lo) & np.isfinite(hi)
    n_empty = int(np.sum(lo > hi))

    iu, ju = np.triu_indices(m, k=1)
    pair_valid = valid[iu] & valid[ju]
    n_valid_pairs = int(np.sum(pair_valid))

    ri = r[iu]
    rj = r[ju]
    loi = lo[iu]
    loj = lo[ju]
    hii = hi[iu]
    hij = hi[ju]

    model_i_gt_j = (ri > rj + e) & pair_valid
    model_j_gt_i = (rj > ri + e) & pair_valid
    model_strict = model_i_gt_j | model_j_gt_i
    model_tie = pair_valid & (~model_strict)

    ref_i_gt_j = (loi > hij + e) & pair_valid
    ref_j_gt_i = (loj > hii + e) & pair_valid
    ref_resolved = ref_i_gt_j | ref_j_gt_i
    ref_unresolved = pair_valid & (~ref_resolved)

    concordant = (model_i_gt_j & ref_i_gt_j) | (model_j_gt_i & ref_j_gt_i)
    reversals = (model_i_gt_j & ref_j_gt_i) | (model_j_gt_i & ref_i_gt_j)
    resolved_strict = model_strict & ref_resolved
    unresolved_strict = model_strict & (~ref_resolved)

    n_strict_model = int(np.sum(model_strict))
    n_model_ties = int(np.sum(model_tie))
    n_resolved_ref = int(np.sum(ref_resolved))
    n_unresolved_ref = int(np.sum(ref_unresolved))
    n_resolved_strict = int(np.sum(resolved_strict))
    n_unresolved_strict = int(np.sum(unresolved_strict))
    n_concordant = int(np.sum(concordant))
    n_reversals = int(np.sum(reversals))

    resolved_frac = (
        float(n_resolved_strict / n_strict_model)
        if n_strict_model > 0
        else float("nan")
    )
    reversal_rate = (
        float(n_reversals / n_resolved_strict)
        if n_resolved_strict > 0
        else float("nan")
    )

    return {
        "n_candidates": m,
        "n_total_pairs": int(n_total_pairs),
        "n_valid_pairs": n_valid_pairs,
        "n_empty_intervals": n_empty,
        "n_strict_model_pairs": n_strict_model,
        "n_model_tie_pairs": n_model_ties,
        "n_resolved_ref_pairs": n_resolved_ref,
        "n_unresolved_ref_pairs": n_unresolved_ref,
        "n_resolved_strict_model_pairs": n_resolved_strict,
        "n_unresolved_strict_model_pairs": n_unresolved_strict,
        "n_concordant": n_concordant,
        "n_reversals": n_reversals,
        "resolved_fraction_of_strict_model": resolved_frac,
        "reversal_rate": reversal_rate,
    }


def compute_saturation_metrics(
    scores: np.ndarray | Sequence[float],
    mask: np.ndarray | Sequence[bool] | None = None,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Compute near-saturation (``>= 0.95``), numerical saturation (``>= 1 - tol``), and score groups."""
    arr = np.asarray(scores, dtype=np.float64)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=np.bool_)]
    m = int(arr.size)
    if m == 0:
        return {
            "n_items": 0,
            "fraction_ge_095": float("nan"),
            "numerical_saturation_fraction": float("nan"),
            "n_distinct_score_groups": 0,
        }

    _, n_groups = assign_anchored_tie_groups(arr, tol=tol)
    return {
        "n_items": m,
        "fraction_ge_095": float(np.mean(arr >= 0.95)),
        "numerical_saturation_fraction": float(np.mean(arr >= 1.0 - float(tol))),
        "n_distinct_score_groups": int(n_groups),
    }


def _average_ranks_from_anchored_groups(
    values: np.ndarray, tol: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Compute ascending average ranks respecting anchored ``tol`` tie groups."""
    groups_desc, n_groups = assign_anchored_tie_groups(values, tol=tol)
    m = int(values.size)
    if n_groups <= 1:
        return np.full(m, 0.5 * (m + 1), dtype=np.float64), n_groups

    # Convert descending group IDs (0 = highest) to ascending group IDs (0 = lowest)
    groups_asc = (n_groups - 1) - groups_desc
    counts = np.bincount(groups_asc, minlength=n_groups)
    starts = np.zeros(n_groups, dtype=np.float64)
    if n_groups > 1:
        starts[1:] = np.cumsum(counts[:-1], dtype=np.float64)
    # 1-based average rank for each ascending group
    avg_rank_per_group = starts + 0.5 * (counts.astype(np.float64) + 1.0)
    return avg_rank_per_group[groups_asc], n_groups


def compute_tie_aware_spearman(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    mask: np.ndarray | Sequence[bool] | None = None,
    tol: float = 1e-12,
) -> float:
    """Compute average-rank Spearman correlation with anchored ``tol`` tie grouping.

    Returns ``np.nan`` whenever fewer than 2 items remain or either rank vector
    is constant (Section 3.11.1).
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.shape != xa.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if mask is not None:
        m_arr = np.asarray(mask, dtype=np.bool_)
        xa = xa[m_arr]
        ya = ya[m_arr]

    if xa.size < 2:
        return float("nan")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        return float("nan")

    rx, nx = _average_ranks_from_anchored_groups(xa, tol=tol)
    ry, ny = _average_ranks_from_anchored_groups(ya, tol=tol)
    if nx <= 1 or ny <= 1:
        return float("nan")

    dx = rx - float(np.mean(rx))
    dy = ry - float(np.mean(ry))
    denom = math.sqrt(float(np.dot(dx, dx)) * float(np.dot(dy, dy)))
    if denom <= 0.0:
        return float("nan")
    corr = float(np.dot(dx, dy) / denom)
    return float(max(-1.0, min(1.0, corr)))


def evaluate_nodewise_w_reliability(
    W: np.ndarray | Sequence[float],
    R: np.ndarray | Sequence[float],
    pi_ref: np.ndarray | Sequence[float],
    candidate_mask: np.ndarray | Sequence[bool] | None = None,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate how reliably zero-cost ``W_j = R_j - M_j`` tracks actual upper slack ``R_j - pi_j``."""
    w_arr = np.asarray(W, dtype=np.float64)
    r_arr = np.asarray(R, dtype=np.float64)
    p_arr = np.asarray(pi_ref, dtype=np.float64)
    if not (w_arr.ndim == 1 and w_arr.shape == r_arr.shape == p_arr.shape):
        raise ValueError("W, R, and pi_ref must be matching 1-D arrays")

    slack = r_arr - p_arr
    if candidate_mask is not None:
        m_arr = np.asarray(candidate_mask, dtype=np.bool_)
        w_sub = w_arr[m_arr]
        s_sub = slack[m_arr]
    else:
        w_sub = w_arr
        s_sub = slack

    if w_sub.size == 0:
        return {
            "n_items": 0,
            "spearman_w_vs_slack": float("nan"),
            "mean_w": float("nan"),
            "max_w": float("nan"),
            "mean_slack": float("nan"),
            "max_slack": float("nan"),
        }

    rho = compute_tie_aware_spearman(w_sub, s_sub, tol=tol)
    return {
        "n_items": int(w_sub.size),
        "spearman_w_vs_slack": rho,
        "mean_w": float(np.mean(w_sub)),
        "max_w": float(np.max(w_sub)),
        "mean_slack": float(np.mean(s_sub)),
        "max_slack": float(np.max(s_sub)),
    }


def evaluate_scenario_w_screening(
    w_indicator: np.ndarray | Sequence[float],
    positive_flag: np.ndarray | Sequence[bool],
    thresholds: Sequence[float] = (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.10),
    continuous_target: np.ndarray | Sequence[float] | None = None,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate scenario-level ``W`` threshold screening precision/recall for flagging reversals or regret.

    For ``tau == 0.0``, flags scenarios with ``w_indicator > tol``; for ``tau > 0.0``,
    flags scenarios with ``w_indicator >= tau``.
    """
    w_arr = np.asarray(w_indicator, dtype=np.float64)
    pos_arr = np.asarray(positive_flag, dtype=np.bool_)
    if w_arr.ndim != 1 or pos_arr.shape != w_arr.shape:
        raise ValueError("w_indicator and positive_flag must be matching 1-D arrays")

    n = int(w_arr.size)
    n_pos = int(np.sum(pos_arr))
    n_neg = n - n_pos

    threshold_results: list[dict[str, Any]] = []
    for raw_tau in thresholds:
        tau = float(raw_tau)
        if tau == 0.0:
            flagged = w_arr > float(tol)
        else:
            flagged = w_arr >= tau

        tp = int(np.sum(flagged & pos_arr))
        fp = int(np.sum(flagged & (~pos_arr)))
        fn = int(np.sum((~flagged) & pos_arr))
        tn = int(np.sum((~flagged) & (~pos_arr)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = float("nan")

        threshold_results.append(
            {
                "threshold": tau,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "flagged_fraction": float(np.mean(flagged)) if n > 0 else float("nan"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    spearman_vs_target: float | None = None
    if continuous_target is not None:
        spearman_vs_target = compute_tie_aware_spearman(
            w_arr, np.asarray(continuous_target, dtype=np.float64), tol=tol
        )

    return {
        "n_scenarios": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": float(n_pos / n) if n > 0 else float("nan"),
        "spearman_vs_continuous_target": spearman_vs_target,
        "thresholds": threshold_results,
    }
