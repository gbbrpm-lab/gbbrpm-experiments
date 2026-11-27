"""Comprehensive unit, mathematical-invariant, and parity tests for Step 4
comparators, Two-Stage Certified Top-k Screening, and benchmark metrics.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbbrpm.benchmark_comparators import (
    compute_abbe_lower,
    compute_all_tree_bounds,
    compute_ancestry_diagnostic,
    compute_clipped_sum,
    compute_gbbrpm_fast,
    compute_local_only,
    compute_local_upstream,
    compute_recursive_max,
    compute_target_tree_bound,
    run_comparators,
)
from gbbrpm.benchmark_metrics import (
    assign_anchored_tie_groups,
    compute_capture_and_regret,
    compute_pairwise_reversals,
    compute_saturation_metrics,
    compute_tie_aware_spearman,
    evaluate_nodewise_w_reliability,
    evaluate_scenario_w_screening,
    generate_tie_permutations,
    intersect_intervals,
    select_fixed_budget_topk_sets,
    symmetric_difference_mass_bound,
)
from gbbrpm.certified_screening import (
    assert_zero_false_certifications,
    certify_top_k_two_stage,
    get_candidate_indices,
)
from gbbrpm.core import evaluate_gbbrpm
from gbbrpm.csr_graph import PreparedDAG, prepare_dag, prepare_dag_from_arrays
from gbbrpm.exact_ref import closed_form_matched_branch, evaluate_exact_marginals
from gbbrpm.mc_ref import run_mc_reference
from model import load_network
from reconvergence import evaluate_path_pruned_reference


RECONSTRUCTED_DIR = ROOT / "data" / "reconstructed"


def _load_reconstructed(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(RECONSTRUCTED_DIR / f"{name}_nodes.csv")
    edges = pd.read_csv(RECONSTRUCTED_DIR / f"{name}_edges.csv")
    return nodes, edges


def _build_matched_branch_dag(
    m: int, ell: int, p: float, b: float, shared: bool
) -> PreparedDAG:
    """Build Eq. (3-15) matched-branch graph (root a -> s_r -> ... -> d)."""
    node_ids: list[str] = ["a"]
    B_vals: list[float] = [b if shared else 0.0]
    sources: list[str] = []
    targets: list[str] = []
    weights: list[float] = []

    for r in range(m):
        s_node = f"s{r}"
        node_ids.append(s_node)
        B_vals.append(0.0 if shared else b)
        sources.append("a")
        targets.append(s_node)
        weights.append(1.0)

        prev = s_node
        for step in range(1, ell):
            mid = f"m{r}_{step}"
            node_ids.append(mid)
            B_vals.append(0.0)
            sources.append(prev)
            targets.append(mid)
            weights.append(p)
            prev = mid

        sources.append(prev)
        targets.append("d")
        weights.append(p)

    node_ids.append("d")
    B_vals.append(0.0)

    return prepare_dag_from_arrays(
        node_ids=node_ids,
        B=B_vals,
        sources=sources,
        targets=targets,
        weights=weights,
        assume_topological=False,
    )


def _generate_random_small_dag(
    rng: np.random.Generator,
    n_nodes: int = 7,
    max_nondeg: int = 14,
    include_extremes: bool = True,
) -> PreparedDAG:
    """Generate a random small DAG guaranteed solvable by exact enumeration."""
    node_ids = [f"v{i}" for i in range(n_nodes)]
    B = np.zeros(n_nodes, dtype=np.float64)
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    nondeg_budget = max_nondeg
    # Assign local disturbances (some interior nodes too, to stress Abbe MDAS edge deletions!)
    n_sources = int(rng.integers(1, max(2, n_nodes // 2 + 1)))
    src_nodes = rng.choice(n_nodes, size=n_sources, replace=False)
    for idx in src_nodes:
        if include_extremes and rng.random() < 0.25:
            B[int(idx)] = 1.0
        elif nondeg_budget > 0:
            B[int(idx)] = float(np.round(rng.uniform(0.05, 0.95), 4))
            nondeg_budget -= 1
        else:
            B[int(idx)] = 1.0

    # Candidate forward edges
    all_pairs = [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes)]
    rng.shuffle(all_pairs)
    target_edges = int(rng.integers(n_nodes, min(len(all_pairs), 2 * n_nodes) + 1))

    for u, v in all_pairs[:target_edges]:
        r = rng.random()
        if include_extremes and r < 0.15:
            w = 0.0
        elif include_extremes and r < 0.35:
            w = 1.0
        elif nondeg_budget > 0:
            w = float(np.round(rng.uniform(0.05, 0.95), 4))
            nondeg_budget -= 1
        else:
            w = 1.0 if rng.random() < 0.5 else 0.0
        sources.append(u)
        targets.append(v)
        weights.append(w)

    return prepare_dag_from_arrays(
        node_ids=node_ids,
        B=B,
        sources=sources,
        targets=targets,
        weights=weights,
        assume_topological=False,
    )


# ---------------------------------------------------------------------------
# 1. Comparator Parity & Mathematical Bracket Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("network", ["N1", "N2", "N3", "N4", "N5"])
def test_gbbrpm_and_ancestry_exact_parity_on_reconstructed_and_historical(network: str):
    """compute_gbbrpm_fast matches core.evaluate_gbbrpm and compute_ancestry_diagnostic
    matches reconvergence.evaluate_path_pruned_reference to < 1e-12."""
    for nodes_df, edges_df in (
        _load_reconstructed(network),
        load_network(network, source="historical"),
    ):
        dag = prepare_dag(nodes_df, edges_df)

        # GBBRPM fast parity
        r_fast = compute_gbbrpm_fast(dag)
        r_core_dict, _ = evaluate_gbbrpm(nodes_df, edges_df)
        r_core = np.array([r_core_dict[nid] for nid in dag.node_ids], dtype=np.float64)
        assert np.max(np.abs(r_fast - r_core)) < 1e-12

        # Ancestry diagnostic parity with fixed reconvergence.py
        anc_out = compute_ancestry_diagnostic(dag)
        pruned_dict = evaluate_path_pruned_reference(nodes_df, edges_df)
        pruned_arr = np.array(
            [pruned_dict[nid] for nid in dag.node_ids], dtype=np.float64
        )
        assert np.max(np.abs(anc_out["scores"] - pruned_arr)) < 1e-12
        assert 0.0 <= anc_out["overlap_receiver_fraction"] <= 1.0
        assert anc_out["max_lineage_size"] >= 0
        assert anc_out["overlap_check_count"] >= 0


@pytest.mark.parametrize("network", ["N1", "N2", "N3", "N4"])
def test_bracket_and_abbe_bounds_on_exact_reconstructed_fixtures(network: str):
    """Verify M_j <= T_j <= pi_j <= R_j <= H_j and 0 <= abbe_lower <= pi_j on N1-N4."""
    nodes_df, edges_df = _load_reconstructed(network)
    dag = prepare_dag(nodes_df, edges_df)
    exact = evaluate_exact_marginals(dag, max_primitives=16)
    pi = exact["marginals"]

    comp = run_comparators(dag)
    M = comp["recursive_max"]
    T = comp["tree_bound"]
    R = comp["gbbrpm"]
    H = comp["clipped_sum"]
    abbe = comp["abbe_lower"]

    tol = 1e-12
    assert np.all(M <= T + tol)
    assert np.all(T <= pi + tol)
    assert np.all(pi <= R + tol)
    assert np.all(R <= H + tol)
    assert np.all(abbe >= -tol)
    assert np.all(abbe <= pi + tol)
    assert np.allclose(comp["local_only"], dag.B, atol=0.0)


def test_matched_branch_closed_form_and_bracket():
    """Verify Eq. (3-15) exact formulas, bracket ordering, and Abbe lower bound on matched branches."""
    tol = 1e-12
    for m in (1, 2, 4):
        for ell in (2, 4):
            for p in (0.10, 0.25, 0.50, 1.0):
                for b in (0.10, 0.50, 0.75, 1.0):
                    cf = closed_form_matched_branch(m=m, ell=ell, p=p, b=b)
                    u = p**ell
                    expected_R = 1.0 - (1.0 - b * u) ** m
                    expected_M = b * u

                    for shared in (True, False):
                        dag = _build_matched_branch_dag(m, ell, p, b, shared=shared)
                        d_idx = dag.node_to_idx["d"]
                        pi_d = cf["shared"] if shared else cf["independent"]

                        M = compute_recursive_max(dag)
                        T = compute_all_tree_bounds(dag)
                        R = compute_gbbrpm_fast(dag)
                        H = compute_clipped_sum(dag)
                        abbe = compute_abbe_lower(dag)

                        assert abs(R[d_idx] - expected_R) < tol
                        assert abs(M[d_idx] - expected_M) < tol
                        assert M[d_idx] <= T[d_idx] + tol
                        assert T[d_idx] <= pi_d + tol
                        assert pi_d <= R[d_idx] + tol
                        assert R[d_idx] <= H[d_idx] + tol
                        assert -tol <= abbe[d_idx] <= pi_d + tol

                        # On independent-origin branches, incoming subtrees are disjoint so T_d == pi_d == R_d
                        if not shared:
                            assert abs(T[d_idx] - pi_d) < tol
                            assert abs(R[d_idx] - pi_d) < tol
                        else:
                            # On shared-origin branches, in-arborescence keeps 1 edge out of root a -> T_d == M_d
                            assert abs(T[d_idx] - expected_M) < tol


def test_bracket_and_abbe_zero_violations_on_120_random_and_adversarial_dags():
    """Verify M <= T <= pi <= R <= H and abbe_lower <= pi with 0 violations on 120 random DAGs."""
    rng = np.random.Generator(np.random.PCG64(202604))
    tol = 1e-12

    for trial in range(120):
        n_nodes = int(rng.integers(4, 10))
        dag = _generate_random_small_dag(
            rng, n_nodes=n_nodes, max_nondeg=14, include_extremes=(trial % 2 == 0)
        )
        pi = evaluate_exact_marginals(dag, max_primitives=16)["marginals"]

        M = compute_recursive_max(dag)
        T = compute_all_tree_bounds(dag)
        R = compute_gbbrpm_fast(dag)
        H = compute_clipped_sum(dag)
        abbe = compute_abbe_lower(dag)
        loc_up = compute_local_upstream(dag)

        assert np.all(M <= T + tol), f"M > T on trial {trial}"
        assert np.all(T <= pi + tol), f"T > pi on trial {trial}"
        assert np.all(pi <= R + tol), f"pi > R on trial {trial}"
        assert np.all(R <= H + tol), f"R > H on trial {trial}"
        assert np.all(abbe >= -tol), f"abbe < 0 on trial {trial}"
        assert np.all(abbe <= pi + tol), f"abbe > pi on trial {trial}"
        assert np.all((loc_up >= -tol) & (loc_up <= 1.0 + tol))

        # Verify parity with core.evaluate_gbbrpm and reconvergence.py on a subset
        if trial < 20:
            nodes_df = pd.DataFrame({"node": list(dag.node_ids), "B": dag.B})
            edge_rows = []
            for j in range(dag.n_nodes):
                for e in range(int(dag.in_indptr[j]), int(dag.in_indptr[j + 1])):
                    edge_rows.append(
                        {
                            "source": dag.node_ids[int(dag.in_indices[e])],
                            "target": dag.node_ids[j],
                            "S": float(dag.in_weights[e]),
                            "tau": 1.0,
                        }
                    )
            edges_df = (
                pd.DataFrame(edge_rows)
                if edge_rows
                else pd.DataFrame(columns=["source", "target", "S", "tau"])
            )
            if len(edges_df) > 0:
                r_core_dict, _ = evaluate_gbbrpm(nodes_df, edges_df)
                r_core = np.array([r_core_dict[nid] for nid in dag.node_ids])
                assert np.max(np.abs(R - r_core)) < tol

                anc_fast = compute_ancestry_diagnostic(dag)["scores"]
                anc_ref_dict = evaluate_path_pruned_reference(nodes_df, edges_df)
                anc_ref = np.array([anc_ref_dict[nid] for nid in dag.node_ids])
                assert np.max(np.abs(anc_fast - anc_ref)) < tol


def test_abbe_lower_exact_worked_examples_and_edge_cases():
    """Verify Abbe (2020) Sec. 4.1 Eq. (22) on Sec. 4.3 worked example, MDAS edge deletion, and m* truncation."""
    # Case 1: Abbe (2020) Sec. 4.3 worked example: seed a (B=1), edges a->b, a->c, b->c, c->d with weight p=0.4
    p = 0.4
    dag = prepare_dag_from_arrays(
        node_ids=["a", "b", "c", "d"],
        B=[1.0, 0.0, 0.0, 0.0],
        sources=["a", "a", "b", "c"],
        targets=["b", "c", "c", "d"],
        weights=[p, p, p, p],
    )
    # Distances from aux: dist(a)=1, dist(b)=2, dist(c)=2, dist(d)=3.
    # MDAS order: a, b, c, d -> retains b->c (same distance, topo b < c).
    # LB(a) = 1.0
    # LB(b) = p * 1 = p
    # LB(c) = p * 1 + p * p * (1 - p) = p + p^2 - p^3
    # LB(d) = p * LB(c) = p^2 + p^3 - p^4
    abbe = compute_abbe_lower(dag)
    assert abs(abbe[dag.node_to_idx["a"]] - 1.0) < 1e-12
    assert abs(abbe[dag.node_to_idx["b"]] - p) < 1e-12
    assert abs(abbe[dag.node_to_idx["c"]] - (p + p**2 - p**3)) < 1e-12
    assert abs(abbe[dag.node_to_idx["d"]] - (p**2 + p**3 - p**4)) < 1e-12

    # Case 2: Original DAG edge (u -> v) with dist[u] > dist[v] MUST be deleted in MDAS!
    # a(B=0.5, d=1) -> u(B=0, d=2) -> v(B=0.3, d=1). Since dist[u]=2 > dist[v]=1, (u, v) is dropped.
    dag_drop = prepare_dag_from_arrays(
        node_ids=["a", "u", "v"],
        B=[0.5, 0.0, 0.3],
        sources=["a", "u"],
        targets=["u", "v"],
        weights=[0.8, 0.9],
    )
    abbe_drop = compute_abbe_lower(dag_drop)
    assert abs(abbe_drop[dag_drop.node_to_idx["v"]] - 0.3) < 1e-12
    pi_drop = evaluate_exact_marginals(dag_drop)["marginals"]
    assert np.all(abbe_drop <= pi_drop + 1e-12)

    # Case 3: Incoming edge weights summing > 1 tests m* truncation in Eq. (22)
    dag_mstar = prepare_dag_from_arrays(
        node_ids=["s1", "s2", "s3", "t"],
        B=[0.8, 0.8, 0.8, 0.0],
        sources=["s1", "s2", "s3"],
        targets=["t", "t", "t"],
        weights=[0.6, 0.6, 0.6],
    )
    # At t: incoming weights (0.6, 0.6, 0.6) -> prefix sums (0.0, 0.6, 1.2) -> m*=2!
    # LB(t) = 0.6 * 0.8 * (1 - 0) + 0.6 * 0.8 * (1 - 0.6) = 0.48 + 0.192 = 0.672
    abbe_ms = compute_abbe_lower(dag_mstar)
    assert abs(abbe_ms[dag_mstar.node_to_idx["t"]] - 0.672) < 1e-12
    pi_ms = evaluate_exact_marginals(dag_mstar)["marginals"]
    assert np.all(abbe_ms <= pi_ms + 1e-12)


def test_local_upstream_and_tree_tie_breaking_and_ancestry_zero_edge_isolation():
    """Verify Eq. (3-16) local-upstream ablation, T_j node_id tie-breaking, and Q_ij > 0 ancestry."""
    # Eq. 3-16: 3-node chain a(B=0.5) -> b(B=0) -> c(B=0) with w=0.8
    dag_chain = prepare_dag_from_arrays(
        node_ids=["a", "b", "c"],
        B=[0.5, 0.0, 0.0],
        sources=["a", "b"],
        targets=["b", "c"],
        weights=[0.8, 0.8],
    )
    lu = compute_local_upstream(dag_chain)
    assert abs(lu[0] - 0.5) < 1e-12
    assert abs(lu[1] - 0.4) < 1e-12
    assert abs(lu[2] - 0.0) < 1e-12  # 2-hop target gets 0 since B_b == 0!

    # Tree tie-breaking by target node_ids[v]:
    # a(B=0.5) -> z_mid(B=0.9) -> t(B=0) with path weight 0.5 * 0.5 = 0.25
    # a(B=0.5) -> a_mid(B=0.0) -> t(B=0) with path weight 0.5 * 0.5 = 0.25
    # Exact tie in best_prob from a to t (both 0.25); lexicographically "a_mid" < "z_mid",
    # so a MUST attach to "a_mid", NOT "z_mid"!
    dag_tie = prepare_dag_from_arrays(
        node_ids=["a", "z_mid", "a_mid", "t"],
        B=[0.5, 0.9, 0.0, 0.0],
        sources=["a", "a", "z_mid", "a_mid"],
        targets=["z_mid", "a_mid", "t", "t"],
        weights=[0.5, 0.5, 0.5, 0.5],
    )
    t_idx = dag_tie.node_to_idx["t"]
    # If a attaches to a_mid: r(a_mid) = 0.25 -> q(a_mid->t) = 0.125; r(z_mid) = 0.9 -> q(z_mid->t) = 0.45
    # T_t = 1 - (1 - 0.125)*(1 - 0.45) = 1 - 0.875 * 0.55 = 0.51875
    # (Whereas attaching to z_mid would give r(z_mid)=1-(1-0.9)(1-0.25)=0.925 -> T_t = 0.4625!)
    t_val = compute_target_tree_bound(dag_tie, t_idx)
    assert abs(t_val - 0.51875) < 1e-12

    # compute_all_tree_bounds with partial targets mask / list
    t_partial = compute_all_tree_bounds(dag_tie, targets=[t_idx])
    assert t_partial.shape == (dag_tie.n_nodes,)
    assert abs(t_partial[t_idx] - 0.51875) < 1e-12
    assert np.isnan(t_partial[0])

    # Ancestry Q_ij == 0 edge must NOT bridge two independent lineages!
    # s1(B=0.5) -> b1(B=0), s2(B=0.5) -> b2(B=0), plus ZERO-weight bridge s1 -> b2 (w=0)!
    # b1 -> d, b2 -> d (w=0.8). Without Q_ij > 0 filtering, s1->b2 would falsely merge b1 and b2!
    dag_zero_bridge = prepare_dag_from_arrays(
        node_ids=["s1", "s2", "b1", "b2", "d"],
        B=[0.5, 0.5, 0.0, 0.0, 0.0],
        sources=["s1", "s2", "s1", "b1", "b2"],
        targets=["b1", "b2", "b2", "d", "d"],
        weights=[0.8, 0.8, 0.0, 0.8, 0.8],
    )
    anc = compute_ancestry_diagnostic(dag_zero_bridge)
    d_idx = dag_zero_bridge.node_to_idx["d"]
    # Independent branches: q = 0.5 * 0.8 * 0.8 = 0.32 each -> 1 - (1 - 0.32)^2 = 0.5376
    assert abs(anc["scores"][d_idx] - (1.0 - (1.0 - 0.32) ** 2)) < 1e-12
    assert anc["overlap_receiver_count"] == 0
    assert anc["overlap_receiver_fraction"] == 0.0


# ---------------------------------------------------------------------------
# 2. Two-Stage Deterministic Certified Top-k Screening Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("network", ["N1", "N2", "N3", "N4"])
def test_certified_screening_exact_fixtures(network: str):
    """Two-stage screening satisfies monotonic refinement and 0 false certifications on N1-N4."""
    nodes_df, edges_df = _load_reconstructed(network)
    dag = prepare_dag(nodes_df, edges_df)
    pi = evaluate_exact_marginals(dag, max_primitives=16)["marginals"]

    res = certify_top_k_two_stage(dag, fractions=(0.10, 0.20, 0.30))
    check = assert_zero_false_certifications(res, pi_exact=pi)
    assert check["ok"] is True
    assert check["n_false_in"] == 0
    assert check["n_false_out"] == 0

    for frac, b in res["budgets"].items():
        assert b["res_s1"] + b["band_s1"] == pytest.approx(1.0, abs=1e-12)
        assert b["res_s2"] + b["band_s2"] == pytest.approx(1.0, abs=1e-12)
        assert b["res_s2"] >= b["res_s1"] - 1e-12
        assert b["n_stage2_trees"] == b["n_U_1"]


def test_certified_screening_60_random_exact_dags_and_stage2_targeted_calls(monkeypatch):
    """Verify 0 false certifications across 60 random exact DAGs and confirm Stage 2 computes T_j ONLY on U_1."""
    rng = np.random.Generator(np.random.PCG64(90210))

    for trial in range(60):
        n_nodes = int(rng.integers(5, 10))
        dag = _generate_random_small_dag(rng, n_nodes=n_nodes, max_nondeg=14)
        pi = evaluate_exact_marginals(dag, max_primitives=16)["marginals"]

        res = certify_top_k_two_stage(dag, fractions=(0.10, 0.20, 0.30))
        assert_zero_false_certifications(res, pi_exact=pi)

    # Spy test: verify compute_target_tree_bound is called ONLY for nodes in U_1!
    import gbbrpm.certified_screening as cs_mod

    called_nodes: list[int] = []
    orig_fn = cs_mod.compute_target_tree_bound

    def _spy(dag_arg: PreparedDAG, target_idx: int) -> float:
        called_nodes.append(int(target_idx))
        return orig_fn(dag_arg, target_idx)

    monkeypatch.setattr(cs_mod, "compute_target_tree_bound", _spy)

    # Construct a DAG where Stage 2 strictly shrinks U_1 -> U_2:
    # Node `top` has TWO independent sources s1, s2 (B=0.5, w=0.5) -> M=0.25, T=R=0.4375!
    # Nodes `c1`, `c2`, `c3` each have a single source with B=0.35, w=1.0 -> M=T=R=0.35.
    # For k=1 (f=0.25 over 4 candidates):
    # Stage 1: R_sorted=[0.4375, 0.35, 0.35, 0.35] -> R_(2)=0.35; M_sorted=[0.35, 0.35, 0.35, 0.25] -> M_(1)=0.35.
    #   I_1 = empty (no M_a > 0.35), O_1 = empty (no R_b < 0.35), U_1 = all 4!
    # Stage 2: T(top) = 0.4375 > R_(2)=0.35 -> top certified IN!
    #   ell_2_(1) = 0.4375 -> c1, c2, c3 all have R=0.35 < 0.4375 -> ALL 3 certified OUT!
    #   U_2 = empty (100% deterministic certification via Stage 2)!
    dag_refine = prepare_dag_from_arrays(
        node_ids=["s1", "s2", "s3", "top", "c1", "c2", "c3"],
        B=[0.5, 0.5, 0.35, 0.0, 0.0, 0.0, 0.0],
        sources=["s1", "s2", "s3", "s3", "s3"],
        targets=["top", "top", "c1", "c2", "c3"],
        weights=[0.5, 0.5, 1.0, 1.0, 1.0],
    )
    res_ref = certify_top_k_two_stage(dag_refine, fractions=(0.25,))
    b = res_ref["budgets"][0.25]
    assert b["n_U_1"] == 4
    assert b["n_U_2"] == 0
    assert b["fully_certified_s1"] is False
    assert b["fully_certified_s2"] is True
    assert set(called_nodes) == set(int(x) for x in b["U_1"])
    assert len(called_nodes) == len(b["U_1"])

    pi_ref = evaluate_exact_marginals(dag_refine)["marginals"]
    assert_zero_false_certifications(res_ref, pi_exact=pi_ref)


def test_certified_screening_edge_cases_and_negative_false_certification_checks():
    """Verify edge cases (|C|=0, k=|C|, all-zero C, ties) and negative assertion failures."""
    # Edge case 1: |C| == 0 (all B > 0)
    dag_no_cand = prepare_dag_from_arrays(
        node_ids=["a", "b"], B=[0.5, 0.2], sources=["a"], targets=["b"], weights=[0.5]
    )
    res0 = certify_top_k_two_stage(dag_no_cand, fractions=(0.10, 0.20, 0.30))
    assert res0["n_candidates"] == 0
    assert all(b["fully_certified_s2"] for b in res0["budgets"].values())
    assert_zero_false_certifications(
        res0, pi_exact=evaluate_exact_marginals(dag_no_cand)["marginals"]
    )

    # Edge case 2: k == |C| (1 candidate -> trivial full certification)
    dag_one_cand = prepare_dag_from_arrays(
        node_ids=["a", "b"], B=[0.5, 0.0], sources=["a"], targets=["b"], weights=[0.5]
    )
    res1 = certify_top_k_two_stage(dag_one_cand, fractions=(0.10, 1.0))
    for b in res1["budgets"].values():
        assert b["trivial"] is True
        assert b["n_I_1"] == 1
        assert b["n_U_2"] == 0

    # Edge case 3: All-zero candidates (tied at 0.0 -> none certified IN or OUT!)
    dag_zero = prepare_dag_from_arrays(
        node_ids=["s", "c1", "c2", "c3", "c4"],
        B=[0.5, 0.0, 0.0, 0.0, 0.0],
        sources=["s"],
        targets=["c1"],
        weights=[0.0],
    )
    res_z = certify_top_k_two_stage(dag_zero, fractions=(0.25, 0.50))
    for b in res_z["budgets"].values():
        assert b["n_I_2"] == 0
        assert b["n_O_2"] == 0
        assert b["n_U_2"] == 4
    assert_zero_false_certifications(
        res_z, pi_exact=evaluate_exact_marginals(dag_zero)["marginals"]
    )

    # Edge case 4: MC interval mode with intersected intervals
    dag_mc = prepare_dag_from_arrays(
        node_ids=["s", "hi", "lo1", "lo2", "lo3"],
        B=[0.8, 0.0, 0.0, 0.0, 0.0],
        sources=["s", "s", "s", "s"],
        targets=["hi", "lo1", "lo2", "lo3"],
        weights=[0.9, 0.1, 0.1, 0.1],
    )
    mc = run_mc_reference(dag_mc, n_trials=20_000, seed=41, batch_size=10_000)
    res_mc = certify_top_k_two_stage(dag_mc, fractions=(0.25,))
    T_all = compute_all_tree_bounds(dag_mc)
    L_p, U_p, empty = intersect_intervals(mc["lower"], mc["upper"], T_all, res_mc["R"])
    assert not np.any(empty)
    assert_zero_false_certifications(res_mc, pi_lower=L_p, pi_upper=U_p)

    # Negative check 1: Tampering certified-IN must trigger AssertionError (exact mode)
    bad = dict(res_mc)
    bad_b = dict(res_mc["budgets"][0.25])
    bad_b["I_1"] = np.array([dag_mc.node_to_idx["lo1"]], dtype=np.int32)
    bad["budgets"] = {0.25: bad_b}
    with pytest.raises(AssertionError, match="False certified-IN"):
        assert_zero_false_certifications(
            bad, pi_exact=evaluate_exact_marginals(dag_mc)["marginals"]
        )

    # Negative check 2: Tampering certified-IN must trigger AssertionError (interval mode)
    with pytest.raises(AssertionError, match="Interval-contradicted certified-IN"):
        assert_zero_false_certifications(bad, pi_lower=L_p, pi_upper=U_p)

    # Negative check 3: Empty reference interval must trigger AssertionError
    bad_lo = L_p.copy()
    bad_hi = U_p.copy()
    bad_lo[1] = 0.9
    bad_hi[1] = 0.1
    with pytest.raises(AssertionError, match="empty reference interval"):
        assert_zero_false_certifications(res_mc, pi_lower=bad_lo, pi_upper=bad_hi)


# ---------------------------------------------------------------------------
# 3. Benchmark Metrics Tests
# ---------------------------------------------------------------------------


def test_anchored_tie_groups_and_seed140_permutations():
    """Verify anchored 1e-12 tie groups (non-chained!) and seed-140 exact-k permutations."""
    # Anchor test: 1.0, 1.0 - 0.8e-12 (within tol of anchor 1.0 -> group 0),
    # 1.0 - 1.5e-12 (within 0.7e-12 of item 1, BUT > 1e-12 from anchor 1.0 -> NEW group 1!)
    scores = np.array([1.0 - 1.5e-12, 1.0, 1.0 - 0.8e-12, 0.5], dtype=np.float64)
    groups, n_groups = assign_anchored_tie_groups(scores, tol=1e-12)
    assert n_groups == 3
    assert groups[1] == 0
    assert groups[2] == 0
    assert groups[0] == 1
    assert groups[3] == 2

    # Seed 140 determinism & exact cardinality k
    p1 = generate_tie_permutations(10, n_perms=100, seed=140)
    p2 = generate_tie_permutations(10, n_perms=100, seed=140)
    assert np.array_equal(p1, p2)
    assert p1.shape == (100, 10)

    # Boundary tie splitting: item 0 strictly top (0.9), items 1..4 tied at 0.5, item 5 at 0.1
    s_tie = np.array([0.9, 0.5, 0.5 + 1e-13, 0.5, 0.5 - 1e-13, 0.1], dtype=np.float64)
    perms = generate_tie_permutations(6, n_perms=100, seed=140)
    sel = select_fixed_budget_topk_sets(s_tie, k=3, perms=perms, tol=1e-12)
    assert sel.shape == (100, 3)
    # Every permutation must select item 0, never item 5, and 2 distinct items from {1, 2, 3, 4}
    for row in sel:
        row_set = set(int(x) for x in row)
        assert len(row_set) == 3
        assert 0 in row_set
        assert 5 not in row_set
    # All 4 tied items should appear across the 100 permutations
    counts = np.bincount(sel.ravel(), minlength=6)
    assert counts[0] == 100
    assert counts[5] == 0
    assert all(counts[i] > 0 for i in (1, 2, 3, 4))


def test_capture_regret_and_all_zero_denominator_and_sym_diff_bound():
    """Verify Eq. (3-17) capture, regret, all-zero reference handling, and h*|sym_diff| bound."""
    mask = np.array([False, True, True, True, True], dtype=bool)
    ref_pi = np.array([1.0, 0.8, 0.5, 0.2, 0.1], dtype=np.float64)

    # Perfect ordering scores -> regret == 0, capture == exact oracle ratio
    perfect = np.array([1.0, 0.9, 0.6, 0.3, 0.15], dtype=np.float64)
    res_perf = compute_capture_and_regret(
        perfect, ref_pi, mask, fractions=(0.25, 0.50), seed=140, n_perms=100, h=0.01
    )
    assert res_perf["all_zero_reference"] is False
    b25 = res_perf["budgets"][0.25]
    assert b25["k"] == 1
    assert abs(b25["regret"] - 0.0) < 1e-12
    assert abs(b25["capture"] - (0.8 / 1.6)) < 1e-12
    assert b25["mean_sym_diff_to_oracle"] == 0.0
    assert b25["mass_uncertainty_bound"] == 0.0

    # Reversed ordering scores -> exact positive regret!
    rev = np.array([1.0, 0.1, 0.2, 0.5, 0.9], dtype=np.float64)
    res_rev = compute_capture_and_regret(
        rev, ref_pi, mask, fractions=(0.25,), seed=140, n_perms=100, h=0.05
    )
    b_rev = res_rev["budgets"][0.25]
    # Selects item 4 (pi=0.1) instead of item 1 (pi=0.8): regret=0.7, sym_diff=2, unc=0.10 -> resolved!
    assert abs(b_rev["regret"] - 0.7) < 1e-12
    assert abs(b_rev["capture"] - (0.1 / 1.6)) < 1e-12
    assert abs(b_rev["mean_sym_diff_to_oracle"] - 2.0) < 1e-12
    assert abs(b_rev["mass_uncertainty_bound"] - 0.10) < 1e-12
    assert b_rev["regret_resolved"] is True

    # Standalone symmetric_difference_mass_bound helper
    assert abs(symmetric_difference_mass_bound([1, 2], [2, 3], h=0.05) - 0.10) < 1e-12

    # All-zero candidate reference -> capture is NaN and all_zero_reference is True
    zero_pi = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    res_zero = compute_capture_and_regret(perfect, zero_pi, mask, fractions=(0.25,))
    assert res_zero["all_zero_reference"] is True
    assert math.isnan(res_zero["budgets"][0.25]["capture"])
    assert res_zero["budgets"][0.25]["regret"] == 0.0


def test_interval_intersection_reversals_saturation_spearman_and_w_reliability():
    """Verify non-repaired interval intersection, strict reversals, saturation, Spearman, and W_j reliability."""
    # Interval intersection NEVER silently repairs empty intervals
    L_p, U_p, empty = intersect_intervals(
        mc_lower=[0.1, 0.7, 0.2],
        mc_upper=[0.4, 0.9, 0.5],
        analytic_lower=[0.2, 0.1, 0.6],
        analytic_upper=[0.3, 0.5, 0.8],
    )
    assert np.allclose(L_p, [0.2, 0.7, 0.6])
    assert np.allclose(U_p, [0.3, 0.5, 0.5])
    assert np.array_equal(empty, np.array([False, True, True]))

    # Pairwise reversals:
    # 4 candidates:
    #   0: R=0.80, ref=[0.20, 0.30]
    #   1: R=0.60, ref=[0.50, 0.55]  -> (0, 1): R_0 > R_1, U'_0=0.30 < L'_1=0.50 => REVERSAL!
    #   2: R=0.40, ref=[0.25, 0.45]  -> (0, 2): unresolved; (1, 2): R_1 > R_2, L'_1=0.50 > U'_2=0.45 => CONCORDANT!
    #   3: R=0.40 + 1e-13, ref=[0.05, 0.10] -> (2, 3): model TIE; (0, 3) & (1, 3): CONCORDANT!
    mask = np.array([True, True, True, True], dtype=bool)
    R = np.array([0.80, 0.60, 0.40, 0.40 + 1e-13], dtype=np.float64)
    lo = np.array([0.20, 0.50, 0.25, 0.05], dtype=np.float64)
    hi = np.array([0.30, 0.55, 0.45, 0.10], dtype=np.float64)
    rev = compute_pairwise_reversals(R, lo, hi, mask, eps=1e-12)
    assert rev["n_total_pairs"] == 6
    assert rev["n_model_tie_pairs"] == 1  # (2, 3)
    assert rev["n_strict_model_pairs"] == 5
    assert rev["n_reversals"] == 1  # (0, 1)
    assert rev["n_concordant"] == 3  # (1, 2), (0, 3), (1, 3)
    assert rev["n_resolved_strict_model_pairs"] == 4
    assert rev["n_unresolved_strict_model_pairs"] == 1  # (0, 2)
    assert abs(rev["reversal_rate"] - 0.25) < 1e-12
    assert abs(rev["resolved_fraction_of_strict_model"] - 0.80) < 1e-12

    # Saturation & Spearman
    sat = compute_saturation_metrics([1.0, 1.0 - 5e-13, 0.96, 0.40], tol=1e-12)
    assert abs(sat["fraction_ge_095"] - 0.75) < 1e-12
    assert abs(sat["numerical_saturation_fraction"] - 0.50) < 1e-12
    assert sat["n_distinct_score_groups"] == 3

    # Constant rank vector -> Spearman is NaN
    assert math.isnan(compute_tie_aware_spearman([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]))
    # Monotone -> +1.0, reversed -> -1.0
    assert abs(compute_tie_aware_spearman([0.1, 0.3, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]) - 1.0) < 1e-12
    assert abs(compute_tie_aware_spearman([0.1, 0.3, 0.7, 0.9], [0.8, 0.6, 0.4, 0.2]) - (-1.0)) < 1e-12

    # W_j reliability (nodewise and scenario-level threshold precision/recall)
    w_node = evaluate_nodewise_w_reliability(
        W=[0.0, 0.1, 0.3, 0.5],
        R=[0.2, 0.4, 0.7, 0.9],
        pi_ref=[0.2, 0.35, 0.5, 0.6],  # actual slack = [0.0, 0.05, 0.2, 0.3]
    )
    assert abs(w_node["spearman_w_vs_slack"] - 1.0) < 1e-12

    w_scen = evaluate_scenario_w_screening(
        w_indicator=[0.0, 0.002, 0.02, 0.15, 0.25],
        positive_flag=[False, False, True, True, True],
        thresholds=(0.0, 0.01, 0.10),
        continuous_target=[0.0, 0.0, 0.05, 0.10, 0.20],
    )
    t_001 = [t for t in w_scen["thresholds"] if t["threshold"] == 0.01][0]
    assert t_001["tp"] == 3 and t_001["fp"] == 0 and t_001["fn"] == 0 and t_001["tn"] == 2
    assert abs(t_001["precision"] - 1.0) < 1e-12
    assert abs(t_001["recall"] - 1.0) < 1e-12
    assert w_scen["spearman_vs_continuous_target"] > 0.9
