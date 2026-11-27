"""Comprehensive test suite for CSR DAG preparation, exact enumeration, and MC solvers."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gbbrpm.csr_graph import PreparedDAG, prepare_dag, prepare_dag_from_arrays
from gbbrpm.exact_ref import (
    closed_form_matched_branch,
    evaluate_exact_marginals,
    evaluate_exact_marginals_df,
)
from gbbrpm.mc_ref import (
    evaluate_bitpacked_draws,
    evaluate_scalar_draws,
    hoeffding_half_width,
    run_mc_reference,
    run_mc_scalar_reference,
)

RECONSTRUCTED_DIR = REPO_ROOT / "data" / "reconstructed"


def _build_matched_branch_dag(
    m: int, ell: int, p: float, b: float, shared: bool
) -> tuple[PreparedDAG, str]:
    """Construct the Eq. (3-15) matched-branch graph with root a, m branches of length ell, receiver d."""
    node_ids = ["a"]
    B = [b if shared else 0.0]
    sources: list[str] = []
    targets: list[str] = []
    weights: list[float] = []

    for r in range(1, m + 1):
        s_name = f"s{r}"
        node_ids.append(s_name)
        B.append(0.0 if shared else b)
        sources.append("a")
        targets.append(s_name)
        weights.append(1.0)

        prev = s_name
        for step in range(1, ell):
            mid = f"m{r}_{step}"
            node_ids.append(mid)
            B.append(0.0)
            sources.append(prev)
            targets.append(mid)
            weights.append(p)
            prev = mid

        sources.append(prev)
        targets.append("d")
        weights.append(p)

    node_ids.append("d")
    B.append(0.0)

    dag = prepare_dag_from_arrays(
        node_ids=node_ids,
        B=B,
        sources=sources,
        targets=targets,
        weights=weights,
        assume_topological=True,
    )
    return dag, "d"


# ---------------------------------------------------------------------------
# CSR DAG preparation tests
# ---------------------------------------------------------------------------


def test_prepare_dag_topological_ordering_and_csr_symmetry():
    # Intentionally unsorted node order: C -> B -> A reversed as input
    nodes = pd.DataFrame({"node": ["C", "A", "B"], "B": [0.0, 0.7, 0.2]})
    edges = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "C", "C"],
            "S": [0.5, 0.25, 0.8],
            "tau": [1.0, 0.5, 0.5],
        }
    )
    dag = prepare_dag(nodes, edges)
    assert dag.n_nodes == 3
    assert dag.n_edges == 3
    assert dag.node_ids == ("A", "B", "C")
    assert dag.node_to_idx == {"A": 0, "B": 1, "C": 2}
    np.testing.assert_allclose(dag.B, [0.7, 0.2, 0.0])

    # Predecessors of C (idx 2): A (0) weight 0.125, B (1) weight 0.4
    c_start, c_end = dag.in_indptr[2], dag.in_indptr[3]
    assert list(dag.in_indices[c_start:c_end]) == [0, 1]
    np.testing.assert_allclose(dag.in_weights[c_start:c_end], [0.125, 0.4])

    # Successors of A (idx 0): B (1) weight 0.5, C (2) weight 0.125
    a_start, a_end = dag.out_indptr[0], dag.out_indptr[1]
    assert list(dag.out_indices[a_start:a_end]) == [1, 2]
    np.testing.assert_allclose(dag.out_weights[a_start:a_end], [0.5, 0.125])


def test_prepare_dag_fail_closed_validation():
    # Cycle
    with pytest.raises(ValueError, match="DAG"):
        prepare_dag_from_arrays(["A", "B"], [0.5, 0.0], [0, 1], [1, 0], [0.5, 0.5])

    # Self-loop
    with pytest.raises(ValueError, match="Self-loop"):
        prepare_dag_from_arrays(["A"], [0.5], [0], [0], [0.5])

    # Duplicate edge
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_dag_from_arrays(["A", "B"], [0.5, 0.0], [0, 0], [1, 1], [0.5, 0.4])

    # Non-finite B / weights
    with pytest.raises(ValueError, match="finite"):
        prepare_dag_from_arrays(["A"], [float("nan")], [], [], [])
    with pytest.raises(ValueError, match="finite"):
        prepare_dag_from_arrays(["A", "B"], [0.5, 0.0], [0], [1], [float("inf")])

    # Out-of-bounds B / weights
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_dag_from_arrays(["A"], [1.1], [], [], [])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_dag_from_arrays(["A", "B"], [0.5, 0.0], [0], [1], [-0.01])

    # assume_topological=True still rejects non-topological/cyclic edges
    with pytest.raises(ValueError, match="topological"):
        prepare_dag_from_arrays(
            ["A", "B"], [0.5, 0.0], [1], [0], [0.5], assume_topological=True
        )


# ---------------------------------------------------------------------------
# Exact enumeration solver tests
# ---------------------------------------------------------------------------


def test_exact_isolated_chain_convergence_and_diamond():
    # 1. Isolated node
    iso = prepare_dag_from_arrays(["A"], [0.37], [], [], [], assume_topological=True)
    res_iso = evaluate_exact_marginals(iso)
    assert abs(res_iso["total_probability_mass"] - 1.0) < 1e-12
    assert res_iso["n_nondegenerate_primitives"] == 1
    assert abs(res_iso["marginals_by_node"]["A"] - 0.37) < 1e-12

    # 2. Chain A -> B -> C
    chain = prepare_dag_from_arrays(
        ["A", "B", "C"], [0.8, 0.1, 0.2], [0, 1], [1, 2], [0.5, 0.6], assume_topological=True
    )
    res_chain = evaluate_exact_marginals(chain)
    assert abs(res_chain["total_probability_mass"] - 1.0) < 1e-12
    pi_a = 0.8
    pi_b = 1.0 - (1.0 - 0.1) * (1.0 - 0.5 * pi_a)
    pi_c = 1.0 - (1.0 - 0.2) * (1.0 - 0.6 * pi_b)
    np.testing.assert_allclose(res_chain["marginals"], [pi_a, pi_b, pi_c], atol=1e-12)

    # 3. Independent convergence: A -> C, B -> C (disjoint ancestry -> matches complement product)
    conv = prepare_dag_from_arrays(
        ["A", "B", "C"], [0.6, 0.4, 0.1], [0, 1], [2, 2], [0.5, 0.75], assume_topological=True
    )
    res_conv = evaluate_exact_marginals(conv)
    assert abs(res_conv["total_probability_mass"] - 1.0) < 1e-12
    pi_c_conv = 1.0 - (1.0 - 0.1) * (1.0 - 0.6 * 0.5) * (1.0 - 0.4 * 0.75)
    assert abs(res_conv["marginals_by_node"]["C"] - pi_c_conv) < 1e-12

    # 4. Diamond: A -> B, A -> C, B -> D, C -> D (shared ancestry!)
    diamond = prepare_dag_from_arrays(
        ["A", "B", "C", "D"],
        [0.5, 0.0, 0.0, 0.0],
        [0, 0, 1, 2],
        [1, 2, 3, 3],
        [0.8, 0.6, 0.5, 0.7],
        assume_topological=True,
    )
    res_dia = evaluate_exact_marginals(diamond)
    assert abs(res_dia["total_probability_mass"] - 1.0) < 1e-12
    # Exact: conditioned on Z_A=1 (prob 0.5), path1 = 0.8*0.5=0.4, path2 = 0.6*0.7=0.42
    expected_d = 0.5 * (1.0 - (1.0 - 0.4) * (1.0 - 0.42))
    assert abs(res_dia["marginals_by_node"]["D"] - expected_d) < 1e-12


@pytest.mark.parametrize(
    "m,ell,p,b",
    [
        (1, 2, 0.5, 0.5),
        (2, 2, 0.5, 0.5),
        (2, 2, 1.0, 0.5),
        (2, 2, 0.5, 1.0),
        (2, 4, 0.25, 0.75),
        (4, 2, 0.20, 0.50),
        (4, 3, 0.75, 0.10),  # 12 non-deg edges + 1 or 4 non-deg B <= 16!
    ],
)
def test_exact_matches_eq_3_15_closed_forms(m: int, ell: int, p: float, b: float):
    closed = closed_form_matched_branch(m=m, ell=ell, p=p, b=b)

    for cond, shared in (("shared", True), ("independent", False)):
        dag, receiver = _build_matched_branch_dag(m=m, ell=ell, p=p, b=b, shared=shared)
        res = evaluate_exact_marginals(dag, max_primitives=16)
        assert abs(res["total_probability_mass"] - 1.0) < 1e-12
        assert abs(res["marginals_by_node"][receiver] - closed[cond]) < 1e-12


def test_exact_fixed_zero_one_primitives_not_counted_and_cap_enforced():
    # 20 edges with weights 0.0 and 1.0, plus only 3 non-degenerate primitives -> succeeds!
    n_nodes = 22
    node_ids = [f"v{i}" for i in range(n_nodes)]
    B = [0.0] * n_nodes
    B[0] = 0.5
    B[1] = 1.0
    sources = list(range(n_nodes - 1))
    targets = list(range(1, n_nodes))
    weights = [1.0 if i % 2 == 0 else 0.0 for i in range(n_nodes - 1)]
    weights[0] = 0.4
    weights[2] = 0.6
    dag = prepare_dag_from_arrays(
        node_ids, B, sources, targets, weights, assume_topological=True
    )
    res = evaluate_exact_marginals(dag, max_primitives=16)
    assert res["n_nondegenerate_primitives"] == 3
    assert abs(res["total_probability_mass"] - 1.0) < 1e-12

    # 17 non-degenerate primitives -> MUST raise ValueError
    node_ids_17 = [f"u{i}" for i in range(18)]
    B_17 = [0.5] + [0.0] * 17
    dag_17 = prepare_dag_from_arrays(
        node_ids_17,
        B_17,
        list(range(17)),
        list(range(1, 18)),
        [0.5] * 17,
        assume_topological=True,
    )
    with pytest.raises(ValueError, match="cap exceeded"):
        evaluate_exact_marginals(dag_17, max_primitives=16)


# ---------------------------------------------------------------------------
# Monte Carlo parity, reproducibility, Hoeffding coverage, and speed tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_trials,batch_size", [(1, 10), (63, 63), (64, 64), (65, 64), (100, 37), (10_000, 10_000)])
def test_mc_bitpacked_vs_scalar_exact_parity(n_trials: int, batch_size: int):
    # Mix of B=0, B=1, 0<B<1, w=0, w=1, 0<w<1 with reconvergence
    dag = prepare_dag_from_arrays(
        node_ids=["A", "B", "C", "D", "E", "F"],
        B=[0.6, 1.0, 0.0, 0.25, 0.0, 0.0],
        sources=[0, 0, 1, 2, 2, 3, 4],
        targets=[2, 3, 4, 4, 5, 5, 5],
        weights=[0.7, 0.0, 1.0, 0.5, 0.8, 0.4, 0.9],
        assume_topological=True,
    )

    # 1. Shared PCG64 RNG stream parity
    bp = run_mc_reference(dag, n_trials=n_trials, seed=123, batch_size=batch_size, alpha=0.05)
    sc = run_mc_scalar_reference(dag, n_trials=n_trials, seed=123, batch_size=batch_size, alpha=0.05)

    np.testing.assert_array_equal(bp["activation_counts"], sc["activation_counts"])
    assert np.max(np.abs(bp["pi_hat"] - sc["pi_hat"])) < 1e-12
    assert np.max(np.abs(bp["lower"] - sc["lower"])) < 1e-12
    assert np.max(np.abs(bp["upper"] - sc["upper"])) < 1e-12
    assert bp["backend"] in {"numpy_uint64_bitpacked", "numba_uint64_bitpacked"}
    assert sc["backend"] == "python_scalar_reference"

    # 2. Direct shared boolean draw array parity
    if n_trials <= 100:
        rng = np.random.default_rng(999)
        z_draws = rng.random((dag.n_nodes, n_trials)) < dag.B[:, None]
        e_draws = rng.random((dag.n_edges, n_trials)) < dag.in_weights[:, None]
        c_bp = evaluate_bitpacked_draws(dag, z_draws, e_draws)
        c_sc = evaluate_scalar_draws(dag, z_draws, e_draws)
        np.testing.assert_array_equal(c_bp, c_sc)


def test_mc_reproducibility_and_hoeffding_formula():
    dag, _ = _build_matched_branch_dag(m=4, ell=2, p=0.5, b=0.5, shared=True)
    r1 = run_mc_reference(dag, n_trials=5_000, seed=41, batch_size=1_000, alpha=0.05)
    r2 = run_mc_reference(dag, n_trials=5_000, seed=41, batch_size=1_000, alpha=0.05)

    np.testing.assert_array_equal(r1["activation_counts"], r2["activation_counts"])
    np.testing.assert_array_equal(r1["pi_hat"], r2["pi_hat"])
    np.testing.assert_array_equal(r1["lower"], r2["lower"])
    np.testing.assert_array_equal(r1["upper"], r2["upper"])

    expected_h = math.sqrt(math.log(2.0 * dag.n_nodes / 0.05) / (2.0 * 5_000))
    assert abs(r1["hoeffding_h"] - expected_h) < 1e-15
    assert abs(hoeffding_half_width(dag.n_nodes, 5_000, 0.05) - expected_h) < 1e-15


@pytest.mark.parametrize("fixture", ["N1", "N2", "N3", "N4"])
def test_mc_hoeffding_coverage_on_reconstructed_n1_n4(fixture: str):
    nodes = pd.read_csv(RECONSTRUCTED_DIR / f"{fixture}_nodes.csv")
    edges = pd.read_csv(RECONSTRUCTED_DIR / f"{fixture}_edges.csv")

    exact = evaluate_exact_marginals_df(nodes, edges, max_primitives=16)
    dag = prepare_dag(nodes, edges)
    mc = run_mc_reference(dag, n_trials=50_000, seed=41, batch_size=10_000, alpha=0.05)

    assert abs(exact["total_probability_mass"] - 1.0) < 1e-12
    assert np.all(exact["marginals"] >= mc["lower"] - 1e-12)
    assert np.all(exact["marginals"] <= mc["upper"] + 1e-12)


def test_mc_hoeffding_coverage_on_matched_branch_fixtures():
    for shared in (True, False):
        dag, receiver = _build_matched_branch_dag(m=4, ell=2, p=0.5, b=0.5, shared=shared)
        exact = evaluate_exact_marginals(dag, max_primitives=16)
        closed = closed_form_matched_branch(m=4, ell=2, p=0.5, b=0.5)
        key = "shared" if shared else "independent"
        assert abs(exact["marginals_by_node"][receiver] - closed[key]) < 1e-12

        mc = run_mc_reference(dag, n_trials=50_000, seed=41, batch_size=10_000, alpha=0.05)
        assert np.all(exact["marginals"] >= mc["lower"] - 1e-12)
        assert np.all(exact["marginals"] <= mc["upper"] + 1e-12)


def test_mc_throughput_on_250_node_500_edge_dag():
    n_nodes, n_edges = 250, 500
    rng = np.random.default_rng(41)
    pairs: set[tuple[int, int]] = set()
    srcs: list[int] = []
    dsts: list[int] = []
    while len(pairs) < n_edges:
        u = int(rng.integers(0, n_nodes - 1))
        v = int(rng.integers(u + 1, n_nodes))
        if (u, v) not in pairs:
            pairs.add((u, v))
            srcs.append(u)
            dsts.append(v)

    B = np.zeros(n_nodes, dtype=np.float64)
    B[:25] = 0.5
    dag = prepare_dag_from_arrays(
        node_ids=[f"v{i}" for i in range(n_nodes)],
        B=B,
        sources=np.asarray(srcs, dtype=np.int32),
        targets=np.asarray(dsts, dtype=np.int32),
        weights=np.full(n_edges, 0.25, dtype=np.float64),
        assume_topological=True,
    )

    res = run_mc_reference(dag, n_trials=100_000, seed=41, batch_size=10_000, alpha=0.05)
    assert res["n_trials"] == 100_000
    assert res["pi_hat"].shape == (n_nodes,)
    assert np.all((res["pi_hat"] >= 0.0) & (res["pi_hat"] <= 1.0))
    assert np.all(res["lower"] <= res["pi_hat"] + 1e-15)
    assert np.all(res["upper"] >= res["pi_hat"] - 1e-15)
    # Must stay fast (comfortably under 5s even under heavy parallel load)
    assert res["elapsed_sec"] < 5.0
