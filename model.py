"""Repository data adapter and backwards-compatible model imports.

New applications should import the calculation engine from :mod:`gbbrpm`.
The experiment scripts continue importing this module so their historical and
recovered dataset paths remain repository-local.
"""

from pathlib import Path

import pandas as pd

from gbbrpm import derive_susceptibility, evaluate_gbbrpm

__all__ = ["derive_susceptibility", "evaluate_gbbrpm", "load_network"]

DATA_DIR = Path(__file__).resolve().parent / "data"

HISTORICAL_DIR = DATA_DIR / "historical"
RECOVERED_DIR = DATA_DIR / "recovered_candidate"

HISTORICAL_EDGE_FILE = HISTORICAL_DIR / "validation_reconstructed_edges.csv"
HISTORICAL_NODE_FILE = HISTORICAL_DIR / "validation_reconstructed_nodes.csv"

RECOVERED_EDGE_FILE = RECOVERED_DIR / "N1-N5_edge_dataset.csv"
RECOVERED_NODE_FILE = RECOVERED_DIR / "N1-N5_node_dataset.csv"

RECOVERED_NETWORK_MAP = {
    "N1": "N1_Linear",
    "N2": "N2_Branching",
    "N3": "N3_Converging",
    "N4": "N4_Diamond",
    "N5": "N5_Mixed20",
}


def load_network(network, source="historical"):
    """Load one frozen experiment network from this repository."""

    if source == "historical":
        nodes = pd.read_csv(HISTORICAL_NODE_FILE)
        edges = pd.read_csv(HISTORICAL_EDGE_FILE)

        nodes = nodes[nodes["Network"] == network].copy()
        edges = edges[edges["Network"] == network].copy()
        nodes = nodes.rename(columns={"Node": "node"})
        edges = edges.rename(columns={"src": "source", "dst": "target"})

        if {"L", "C"}.issubset(edges.columns):
            if "u" in edges.columns:
                expected_u = edges["L"] / edges["C"]
                if not expected_u.round(10).equals(edges["u"].round(10)):
                    raise ValueError("Stored u does not match L/C")
                edges = edges.rename(columns={"u": "Utilization_u"})

            if "S" in edges.columns:
                expected_s = (edges["L"] / edges["C"]).clip(0, 1)
                if not expected_s.round(10).equals(edges["S"].round(10)):
                    raise ValueError("Stored S does not match min(1, L/C)")
                edges = edges.rename(columns={"S": "Susceptibility_S"})

    elif source == "recovered":
        nodes = pd.read_csv(RECOVERED_NODE_FILE)
        edges = pd.read_csv(RECOVERED_EDGE_FILE)

        if network not in RECOVERED_NETWORK_MAP:
            raise ValueError(f"Unknown network: {network}")

        recovered_name = RECOVERED_NETWORK_MAP[network]

        nodes = nodes[nodes["Network"] == recovered_name].copy()
        edges = edges[edges["Network"] == recovered_name].copy()
        nodes = nodes.rename(columns={"Node": "node", "Local_Blockage_B": "B"})
        edges = edges.rename(
            columns={
                "From": "source",
                "To": "target",
                "Capacity_C": "C",
                "Load_L": "L",
                "Tau_baseline": "tau",
            }
        )

        if "Utilization_u" in edges.columns:
            expected_u = edges["L"] / edges["C"]
            if not expected_u.round(10).equals(edges["Utilization_u"].round(10)):
                raise ValueError("Stored Utilization_u does not match L/C")

        if "Susceptibility_S" in edges.columns:
            expected_s = (edges["L"] / edges["C"]).clip(0, 1)
            if not expected_s.round(10).equals(
                edges["Susceptibility_S"].round(10)
            ):
                raise ValueError("Stored Susceptibility_S does not match min(1, L/C)")

    else:
        raise ValueError(
            f"Unknown data source: {source}. Use 'historical' or 'recovered'."
        )

    if nodes.empty:
        raise ValueError(f"No nodes found for network {network} using source {source}")

    if edges.empty:
        raise ValueError(f"No edges found for network {network} using source {source}")

    if "tau" not in edges.columns:
        edges["tau"] = 1.0

    return nodes, edges
