"""Contract regression tests for core validation, reconvergence, and frozen N1-N5 fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbbrpm import derive_susceptibility, evaluate_gbbrpm  # noqa: E402
from reconvergence import evaluate_path_pruned_reference  # noqa: E402


NON_FINITE_VALUES = [float("inf"), float("-inf"), float("nan")]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_nodes():
    return pd.DataFrame({"node": ["A", "B"], "B": [0.5, 0.0]})


@pytest.mark.parametrize("bad_value", NON_FINITE_VALUES)
def test_nonfinite_L_rejected_before_clipping(bad_value):
    nodes = _base_nodes()
    edges = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "L": [bad_value], "C": [100.0], "tau": [1.0]}
    )
    with pytest.raises(ValueError, match="L"):
        derive_susceptibility(edges)
    with pytest.raises(ValueError, match="L"):
        evaluate_gbbrpm(nodes, edges)
    with pytest.raises(ValueError, match="L"):
        evaluate_path_pruned_reference(nodes, edges)


@pytest.mark.parametrize("bad_value", NON_FINITE_VALUES)
def test_nonfinite_C_rejected_before_clipping(bad_value):
    nodes = _base_nodes()
    edges = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "L": [50.0], "C": [bad_value], "tau": [1.0]}
    )
    with pytest.raises(ValueError, match="C"):
        derive_susceptibility(edges)
    with pytest.raises(ValueError, match="C"):
        evaluate_gbbrpm(nodes, edges)
    with pytest.raises(ValueError, match="C"):
        evaluate_path_pruned_reference(nodes, edges)


@pytest.mark.parametrize("bad_value", NON_FINITE_VALUES)
def test_nonfinite_S_rejected(bad_value):
    nodes = _base_nodes()
    edges = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "S": [bad_value], "tau": [1.0]}
    )
    with pytest.raises(ValueError, match="S"):
        derive_susceptibility(edges)
    with pytest.raises(ValueError, match="S"):
        evaluate_gbbrpm(nodes, edges)
    with pytest.raises(ValueError, match="S"):
        evaluate_path_pruned_reference(nodes, edges)


@pytest.mark.parametrize("bad_value", NON_FINITE_VALUES)
def test_nonfinite_tau_rejected(bad_value):
    nodes = _base_nodes()
    edges = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "S": [0.5], "tau": [bad_value]}
    )
    with pytest.raises(ValueError, match="tau"):
        evaluate_gbbrpm(nodes, edges)
    with pytest.raises(ValueError, match="tau"):
        evaluate_path_pruned_reference(nodes, edges)


@pytest.mark.parametrize("bad_value", NON_FINITE_VALUES)
def test_nonfinite_B_rejected(bad_value):
    nodes = pd.DataFrame({"node": ["A", "B"], "B": [bad_value, 0.0]})
    edges = pd.DataFrame({"source": ["A"], "target": ["B"], "S": [0.5], "tau": [1.0]})
    with pytest.raises(ValueError, match="B"):
        evaluate_gbbrpm(nodes, edges)
    with pytest.raises(ValueError, match="B"):
        evaluate_path_pruned_reference(nodes, edges)


def test_dual_S_and_LC_ambiguity_rejected():
    nodes = _base_nodes()
    ambiguous_both = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "L": [50.0], "C": [100.0], "S": [0.5]}
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        derive_susceptibility(ambiguous_both)
    with pytest.raises(ValueError, match="Ambiguous"):
        evaluate_gbbrpm(nodes, ambiguous_both)
    with pytest.raises(ValueError, match="Ambiguous"):
        evaluate_path_pruned_reference(nodes, ambiguous_both)

    ambiguous_s_and_l = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "L": [50.0], "S": [0.5]}
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        derive_susceptibility(ambiguous_s_and_l)

    ambiguous_s_and_c = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "C": [100.0], "S": [0.5]}
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        derive_susceptibility(ambiguous_s_and_c)


def test_partial_L_or_C_without_S_rejected():
    only_l = pd.DataFrame({"source": ["A"], "target": ["B"], "L": [50.0]})
    only_c = pd.DataFrame({"source": ["A"], "target": ["B"], "C": [100.0]})
    with pytest.raises(ValueError):
        derive_susceptibility(only_l)
    with pytest.raises(ValueError):
        derive_susceptibility(only_c)


def test_explicit_S_with_all_nan_LC_and_disjoint_mixed_rows_allowed():
    nodes = pd.DataFrame({"node": ["A", "B", "C"], "B": [0.8, 0.0, 0.0]})
    explicit_edges = pd.DataFrame(
        {"source": ["A", "B"], "target": ["B", "C"], "S": [0.5, 0.5]}
    )
    derived = derive_susceptibility(explicit_edges)
    assert derived["L"].isna().all()
    assert derived["C"].isna().all()
    risk_first, _ = evaluate_gbbrpm(nodes, explicit_edges)
    risk_second, _ = evaluate_gbbrpm(nodes, derived)
    assert risk_first == risk_second

    mixed_edges = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "S": [0.5, np.nan],
            "L": [np.nan, 50.0],
            "C": [np.nan, 100.0],
        }
    )
    risk_mixed, _ = evaluate_gbbrpm(nodes, mixed_edges)
    for node, expected in risk_first.items():
        assert np.isclose(risk_mixed[node], expected, atol=1e-12)


def test_reconvergence_zero_contribution_does_not_propagate_ancestry():
    # A->M2 has S=0 (Q=0), so A's ancestry must NOT leak into M2.
    # M1 carries {A} (Q=0.2) and M2 carries {M2} (Q=0.3); they remain independent.
    nodes = pd.DataFrame(
        {"node": ["A", "M1", "M2", "T"], "B": [0.8, 0.0, 0.6, 0.0]}
    )
    edges = pd.DataFrame(
        {
            "source": ["A", "A", "M1", "M2"],
            "target": ["M1", "M2", "T", "T"],
            "S": [0.5, 0.0, 0.5, 0.5],
        }
    )
    pruned = evaluate_path_pruned_reference(nodes, edges)
    expected_t = 1.0 - (1.0 - 0.2) * (1.0 - 0.3)
    assert np.isclose(pruned["T"], expected_t, atol=1e-12)


def test_reconvergence_zero_contribution_bridge_cannot_merge_independent_groups():
    # U carries {A} (Q=0.3 into T), V carries {B} (Q=0.2 into T).
    # W carries {A, B}, but W->T is gated (tau=0 => Q=0), so W->T must not
    # enter the overlap graph or bridge {A} and {B}.
    nodes = pd.DataFrame(
        {"node": ["A", "B", "U", "V", "W", "T"], "B": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]}
    )
    edges = pd.DataFrame(
        {
            "source": ["A", "B", "U", "V", "U", "V", "W"],
            "target": ["U", "V", "W", "W", "T", "T", "T"],
            "S": [1.0, 1.0, 0.5, 0.5, 0.6, 0.4, 1.0],
            "tau": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    pruned = evaluate_path_pruned_reference(nodes, edges)
    expected_t = 1.0 - (1.0 - 0.3) * (1.0 - 0.2)
    assert np.isclose(pruned["T"], expected_t, atol=1e-12)


def test_reconvergence_transitive_overlap_on_positive_contributions():
    # X carries {A}, M carries {A, B}, Y carries {B}; all three enter T with Q > 0.
    # Transitive overlap {A} - {A, B} - {B} forms one component, keeping max(Q).
    nodes = pd.DataFrame(
        {"node": ["A", "B", "X", "M", "Y", "T"], "B": [0.8, 0.6, 0.0, 0.0, 0.0, 0.0]}
    )
    edges = pd.DataFrame(
        {
            "source": ["A", "B", "A", "B", "X", "M", "Y"],
            "target": ["X", "Y", "M", "M", "T", "T", "T"],
            "S": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    pruned = evaluate_path_pruned_reference(nodes, edges)
    q_x = 0.8 * 0.5 * 0.5  # 0.20
    q_y = 0.6 * 0.5 * 0.5  # 0.15
    r_m = 1.0 - (1.0 - 0.4) * (1.0 - 0.3)  # 0.58 (independent A and B)
    q_m = r_m * 0.5  # 0.29
    assert np.isclose(pruned["T"], max(q_x, q_m, q_y), atol=1e-12)


def test_reconstructed_n1_to_n5_manifest_hashes_and_evaluation():
    recon_dir = ROOT / "data" / "reconstructed"
    manifest = json.loads((recon_dir / "network_manifest.json").read_text(encoding="utf-8"))
    assert list(manifest.keys()) == ["N1", "N2", "N3", "N4", "N5"]

    for network, entry in manifest.items():
        assert entry["fixture_status"] == "reconstructed_fixture_not_original_historical_LC_table"
        assert entry["outlet"]
        nodes_path = recon_dir / entry["nodes_file"]
        edges_path = recon_dir / entry["edges_file"]
        assert nodes_path.exists()
        assert edges_path.exists()
        assert _sha256(nodes_path) == entry["nodes_sha256"]
        assert _sha256(edges_path) == entry["edges_sha256"]

        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
        assert len(nodes_df) == entry["node_count"]
        assert len(edges_df) == entry["edge_count"]

        risk, _ = evaluate_gbbrpm(nodes_df, edges_df)
        pruned = evaluate_path_pruned_reference(nodes_df, edges_df)
        assert entry["outlet"] in risk
        assert all(0.0 <= value <= 1.0 for value in risk.values())
        assert all(0.0 <= value <= 1.0 for value in pruned.values())
