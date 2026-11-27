"""Unit tests for gbbrpm.generators (structural and protocol invariants only).

Note: Held-out evaluation seeds (1001--1020) are tested strictly for structural
topology, scenario count, and determinism invariants — NEVER for benchmark
outcomes, rankings, or solver comparisons.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gbbrpm.exact_ref import closed_form_matched_branch, evaluate_exact_marginals
from gbbrpm.generators import (
    BRANCH_B,
    BRANCH_CONDITIONS,
    BRANCH_ELL,
    BRANCH_M,
    BRANCH_P,
    DAG_B_SETTINGS,
    DAG_FAMILIES,
    DAG_SIZES,
    DAG_WEIGHTS,
    DEV_SEEDS,
    EVAL_SEEDS,
    HETERO_DEV_SEEDS,
    HETERO_EVAL_SEEDS,
    assign_heterogeneous_weights,
    build_matched_branch_scenario,
    compute_dag_structural_metadata,
    compute_layered_partition,
    derive_scenario_seed,
    generate_dag_hetero_scenarios,
    generate_dag_topology,
    generate_dag_uniform_scenarios,
    generate_express_scenarios,
    generate_matched_branch_scenarios,
)


def test_seed_disjointness_and_mc_seed_determinism():
    assert len(DEV_SEEDS) == 5
    assert DEV_SEEDS == (41, 42, 43, 44, 45)
    assert len(EVAL_SEEDS) == 20
    assert EVAL_SEEDS == tuple(range(1001, 1021))
    assert set(DEV_SEEDS).isdisjoint(set(EVAL_SEEDS))
    assert set(HETERO_DEV_SEEDS).issubset(set(DEV_SEEDS))
    assert set(HETERO_EVAL_SEEDS).issubset(set(EVAL_SEEDS))
    assert set(HETERO_DEV_SEEDS).isdisjoint(set(HETERO_EVAL_SEEDS))

    s1 = derive_scenario_seed("dag_uniform|dev|forward|n=20|seed=41|w=0.10|B=0.5")
    s2 = derive_scenario_seed("dag_uniform|dev|forward|n=20|seed=41|w=0.10|B=0.5")
    s3 = derive_scenario_seed("dag_uniform|dev|forward|n=20|seed=41|w=0.10|B=1.0")
    assert s1 == s2
    assert s1 != s3
    assert 0 <= s1 < 2**32


def test_split_isolation_rejects_cross_split_seeds():
    with pytest.raises(ValueError, match="Split isolation violation"):
        generate_dag_uniform_scenarios(split="dev", seeds=[1001])
    with pytest.raises(ValueError, match="Split isolation violation"):
        generate_dag_uniform_scenarios(split="eval", seeds=[41])
    with pytest.raises(ValueError, match="Split isolation violation"):
        generate_dag_hetero_scenarios(split="dev", seeds=[1002])
    with pytest.raises(ValueError, match="split must be"):
        generate_dag_uniform_scenarios(split="invalid")


def test_matched_branch_560_scenarios_and_exact_closed_forms():
    scenarios = generate_matched_branch_scenarios()
    assert len(scenarios) == 560
    assert len({sc.scenario_id for sc in scenarios}) == 560

    # Check exact structural invariants on all 560 scenarios
    for sc in scenarios:
        m = sc.metadata["m"]
        ell = sc.metadata["ell"]
        p = sc.metadata["p"]
        b = sc.metadata["b"]
        cond = sc.metadata["condition"]
        dag = sc.dag

        assert dag.n_nodes == 2 + m * ell
        assert dag.n_edges == m * (ell + 1)
        assert "a" in dag.node_to_idx
        assert "d" in dag.node_to_idx

        a_idx = dag.node_to_idx["a"]
        d_idx = dag.node_to_idx["d"]
        assert dag.B[d_idx] == 0.0
        if cond == "shared":
            assert dag.B[a_idx] == pytest.approx(b)
            for r in range(1, m + 1):
                assert dag.B[dag.node_to_idx[f"s_{r}"]] == 0.0
        else:
            # Inactive root `a` MUST be retained in the node inventory
            assert dag.B[a_idx] == 0.0
            for r in range(1, m + 1):
                assert dag.B[dag.node_to_idx[f"s_{r}"]] == pytest.approx(b)

        expected = closed_form_matched_branch(m, ell, p, b)[cond]
        assert sc.metadata["exact_receiver_pi"] == pytest.approx(expected, abs=1e-15)

    # Cross-check exact enumeration solver against Eq. (3-15) on small fixtures
    for m in (1, 2, 4):
        for cond in BRANCH_CONDITIONS:
            sc = build_matched_branch_scenario(m=m, ell=2, p=0.5, b=0.5, condition=cond)
            exact = evaluate_exact_marginals(sc.dag, max_primitives=16)
            d_idx = sc.metadata["receiver_idx"]
            assert abs(exact["marginals"][d_idx] - sc.metadata["exact_receiver_pi"]) < 1e-12


def test_layered_available_pair_counts_exact_and_fail_closed():
    expected = {
        20: ((4, 4, 4, 4, 4), 64),
        50: ((7, 7, 6, 6, 6, 6, 6, 6), 271),
        100: ((10,) * 10, 900),
        250: ((16,) * 10 + (15,) * 6, 3669),
    }
    for size, (exp_layers, exp_pairs) in expected.items():
        layers, pairs = compute_layered_partition(size)
        assert layers == exp_layers
        assert sum(layers) == size
        assert pairs == exp_pairs
        assert pairs >= 2 * size

    # Infeasible layered size (e.g., |V|=5 -> layers (2, 2, 1) -> 6 pairs < 10 edges) MUST fail closed
    with pytest.raises(ValueError, match="Infeasible layered DAG"):
        generate_dag_topology("layered", size=5, seed=41)


def test_dag_topologies_exact_2v_edges_determinism_and_layer_coverage():
    for family in DAG_FAMILIES:
        for size in DAG_SIZES:
            t1 = generate_dag_topology(family, size, seed=41, split="dev")
            t2 = generate_dag_topology(family, size, seed=41, split="dev")
            assert t1.topo_node_ints == t2.topo_node_ints
            assert t1.edges_int == t2.edges_int
            assert len(t1.topo_node_ints) == size
            assert len(set(t1.topo_node_ints)) == size
            assert len(t1.edges_int) == 2 * size
            assert len(set(t1.edges_int)) == 2 * size
            assert np.all(t1.edge_src_pos < t1.edge_dst_pos)

            # Different seed gives a distinct edge set
            t_other = generate_dag_topology(family, size, seed=42, split="dev")
            assert t_other.edges_int != t1.edges_int

    # Verify layered source-layer coverage shifts match Chapter 3 Sec. dependence-controls
    expected_layer_coverage = {
        20: (4, 2, 4, 0),    # 2 of 4 Layer-1 nodes active, 0 Layer-2
        50: (7, 5, 7, 0),    # 5 of 7 Layer-1 nodes active, 0 Layer-2
        100: (10, 10, 10, 0),  # all 10 Layer-1 nodes active, 0 Layer-2
        250: (16, 16, 16, 9),  # all 16 Layer-1 + 9 Layer-2 active
    }
    for size, (l1_sz, l1_act, l2_sz, l2_act) in expected_layer_coverage.items():
        sc_list = generate_dag_uniform_scenarios(
            split="dev",
            families=("layered",),
            sizes=(size,),
            seeds=(41,),
            weights=(0.20,),
            b_settings=(0.5,),
        )
        assert len(sc_list) == 1
        st = sc_list[0].metadata["structural"]
        assert st["layer1_size"] == l1_sz
        assert st["layer1_active_sources"] == l1_act
        assert st["layer2_size"] == l2_sz
        assert st["layer2_active_sources"] == l2_act
        assert st["n_active_sources"] == math.ceil(0.10 * size)
        assert 0.0 <= st["overlap_candidate_fraction"] <= 1.0


def test_dag_uniform_dev_720_and_eval_2880_structural_counts():
    dev_scenarios = generate_dag_uniform_scenarios(
        split="dev", include_structural_metadata=True
    )
    assert len(dev_scenarios) == 720
    assert len({sc.scenario_id for sc in dev_scenarios}) == 720
    assert all(sc.split == "dev" and sc.graph_seed in DEV_SEEDS for sc in dev_scenarios)

    # Structural-only check on eval split (NO solver evaluation!)
    eval_scenarios = generate_dag_uniform_scenarios(
        split="eval", include_structural_metadata=False
    )
    assert len(eval_scenarios) == 2880
    assert len({sc.scenario_id for sc in eval_scenarios}) == 2880
    assert all(sc.split == "eval" and sc.graph_seed in EVAL_SEEDS for sc in eval_scenarios)
    assert all(sc.dag.n_edges == 2 * sc.dag.n_nodes for sc in eval_scenarios)

    # Check total distinct topologies = 300 (60 dev + 240 eval)
    dev_topos = {(sc.family, sc.size, sc.graph_seed) for sc in dev_scenarios}
    eval_topos = {(sc.family, sc.size, sc.graph_seed) for sc in eval_scenarios}
    assert len(dev_topos) == 60
    assert len(eval_topos) == 240
    assert dev_topos.isdisjoint(eval_topos)


def test_heterogeneous_weight_assignment_24_dev_120_eval_and_exact_mean():
    dev_hetero = generate_dag_hetero_scenarios(split="dev")
    eval_hetero = generate_dag_hetero_scenarios(
        split="eval", include_structural_metadata=False
    )
    assert len(dev_hetero) == 24
    assert len(eval_hetero) == 120
    assert len({sc.scenario_id for sc in dev_hetero}) == 24
    assert len({sc.scenario_id for sc in eval_hetero}) == 120

    for sc in dev_hetero + eval_hetero:
        w = sc.dag.in_weights
        n_e = sc.dag.n_edges
        assert n_e == 2 * sc.dag.n_nodes
        q = n_e // 4
        assert int(np.sum(np.isclose(w, 0.10))) == q
        assert int(np.sum(np.isclose(w, 0.20))) == 2 * q
        assert int(np.sum(np.isclose(w, 0.30))) == q
        assert float(np.mean(w)) == pytest.approx(0.20, abs=1e-15)

    # Check determinism of SHA-256 weight assignment across repeated calls
    topo = generate_dag_topology("preferential_attachment", 50, 41, split="dev")
    w1 = assign_heterogeneous_weights(topo)
    w2 = assign_heterogeneous_weights(topo)
    np.testing.assert_array_equal(w1, w2)


def test_express_12_scenarios_sha256_and_7_advisory_roots():
    scenarios = generate_express_scenarios(verify_hashes=True)
    assert len(scenarios) == 12
    assert len({sc.scenario_id for sc in scenarios}) == 12

    for sc in scenarios:
        assert sc.dag.n_nodes == 71
        assert sc.dag.n_edges == 128
        assert sc.metadata["n_advisory_roots"] == 7
        assert int(np.sum(sc.dag.B > 0)) == 7
        assert np.allclose(sc.dag.B[sc.dag.B > 0], sc.b_setting)
        assert np.allclose(sc.dag.in_weights, sc.weight_value)
        assert sc.metadata["structural"]["n_active_sources"] == 7
        assert sc.metadata["structural"]["n_candidates"] == 64
