"""Exact independent-primitive enumeration solver (Equation 3-10) and Eq. 3-15 closed forms."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .csr_graph import PreparedDAG, prepare_dag


def closed_form_matched_branch(
    m: int, ell: int, p: float, b: float
) -> dict[str, float]:
    """Exact Eq. (3-15) receiver activation probabilities for the matched-branch family.

    Parameters
    ----------
    m:
        Number of edge-disjoint branches (``m >= 1``).
    ell:
        Path length in edges per branch (``ell >= 1``).
    p:
        Uniform edge transmission probability along each branch (``p in [0, 1]``).
    b:
        Common source activation probability (``b in [0, 1]``).

    Returns
    -------
    dict with keys ``"shared"`` and ``"independent"``:
        ``shared = b * (1 - (1 - p**ell)**m)``
        ``independent = 1 - (1 - b * (p**ell))**m``
    """
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError(f"m must be an integer >= 1, got {m!r}")
    if not isinstance(ell, (int, np.integer)) or isinstance(ell, bool) or ell < 1:
        raise ValueError(f"ell must be an integer >= 1, got {ell!r}")
    p_val = float(p)
    b_val = float(b)
    if not (math.isfinite(p_val) and 0.0 <= p_val <= 1.0):
        raise ValueError(f"p must be finite and in [0, 1], got {p!r}")
    if not (math.isfinite(b_val) and 0.0 <= b_val <= 1.0):
        raise ValueError(f"b must be finite and in [0, 1], got {b!r}")

    u = p_val ** int(ell)
    shared = b_val * (1.0 - (1.0 - u) ** int(m))
    independent = 1.0 - (1.0 - b_val * u) ** int(m)
    return {
        "shared": float(shared),
        "independent": float(independent),
    }


def evaluate_exact_marginals(
    dag: PreparedDAG, max_primitives: int = 16
) -> dict[str, Any]:
    """Evaluate exact Eq. (3-10) node activation marginals by binary enumeration.

    Primitives with probability ``0.0`` or ``1.0`` are fixed deterministically
    and are NOT counted toward ``max_primitives`` or enumerated. Raises
    ``ValueError`` if the number of non-degenerate primitives (``0 < p < 1``)
    exceeds ``max_primitives``.
    """
    if max_primitives < 0:
        raise ValueError(f"max_primitives must be >= 0, got {max_primitives}")

    primitive_probs: list[float] = []
    node_bit = np.full(dag.n_nodes, -1, dtype=np.int32)
    edge_bit = np.full(dag.n_edges, -1, dtype=np.int32)

    for j in range(dag.n_nodes):
        bj = float(dag.B[j])
        if 0.0 < bj < 1.0:
            node_bit[j] = len(primitive_probs)
            primitive_probs.append(bj)

    for e in range(dag.n_edges):
        we = float(dag.in_weights[e])
        if 0.0 < we < 1.0:
            edge_bit[e] = len(primitive_probs)
            primitive_probs.append(we)

    m = len(primitive_probs)
    if m > max_primitives:
        raise ValueError(
            f"Exact enumeration cap exceeded: {m} non-degenerate primitives "
            f"> max_primitives={max_primitives}"
        )

    n_states = 1 << m
    # Exact probability mass vector over all 2^m configurations.
    # For configuration index s, bit k == ((s >> k) & 1) encodes primitive k.
    state_probs = np.array([1.0], dtype=np.float64)
    for p in primitive_probs:
        state_probs = np.concatenate((state_probs * (1.0 - p), state_probs * p))

    total_mass = float(state_probs.sum())
    if abs(total_mass - 1.0) >= 1e-12:
        raise RuntimeError(
            f"Exact configuration probability mass deviated from 1.0: {total_mass!r}"
        )

    states = np.arange(n_states, dtype=np.uint64)
    active = np.zeros((dag.n_nodes, n_states), dtype=np.bool_)

    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    in_weights = dag.in_weights
    B = dag.B

    for j in range(dag.n_nodes):
        bj = float(B[j])
        if bj >= 1.0:
            active[j, :] = True
            continue

        if bj > 0.0:
            bit_j = int(node_bit[j])
            xj = ((states >> np.uint64(bit_j)) & np.uint64(1)) != 0
        else:
            xj = np.zeros(n_states, dtype=np.bool_)

        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])
        for e in range(start, end):
            we = float(in_weights[e])
            if we <= 0.0:
                continue
            src = int(in_indices[e])
            if we >= 1.0:
                xj |= active[src]
            else:
                bit_e = int(edge_bit[e])
                edge_on = ((states >> np.uint64(bit_e)) & np.uint64(1)) != 0
                xj |= active[src] & edge_on

        active[j] = xj

    marginals = np.empty(dag.n_nodes, dtype=np.float64)
    for j in range(dag.n_nodes):
        val = float(state_probs[active[j]].sum())
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        marginals[j] = val

    marginals_by_node = {
        dag.node_ids[j]: float(marginals[j]) for j in range(dag.n_nodes)
    }

    return {
        "marginals": marginals,
        "marginals_by_node": marginals_by_node,
        "n_nondegenerate_primitives": m,
        "total_probability_mass": total_mass,
    }


def evaluate_exact_marginals_df(
    nodes: pd.DataFrame, edges: pd.DataFrame, max_primitives: int = 16
) -> dict[str, Any]:
    """Convenience DataFrame wrapper around ``evaluate_exact_marginals``."""
    dag = prepare_dag(nodes, edges)
    return evaluate_exact_marginals(dag, max_primitives=max_primitives)
