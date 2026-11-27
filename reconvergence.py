"""Shared-origin reconvergence diagnostic for GBBRPM v1.

The path-pruned reference is a counterfactual diagnostic, not a replacement
model or ground truth. Independent source lineages retain bounded noisy-OR
aggregation. Incoming branches whose ancestry overlaps are grouped, and only
the largest contribution in each overlapping group is retained.
"""

import networkx as nx
import numpy as np
import pandas as pd

from model import derive_susceptibility, evaluate_gbbrpm, load_network


def evaluate_path_pruned_reference(nodes, edges):
    """Evaluate the Section 3.7.1 ancestry-aware overlap-grouped diagnostic.

    A positive local disturbance ``B_j > 0`` introduces ``j`` into its own
    local-origin label set. Only positive incoming contributions
    (``Q_ij = clip(S_ij * tau_ij * R_i, 0, 1) > 0``) propagate predecessor
    ancestry labels and enter the receiver's overlap graph. Connected
    components of intersecting label sets (including transitive overlap) form
    groups; only the largest contribution in each group is retained and
    combined with ``B_j`` by complement product.
    """

    required_node_columns = {"node", "B"}
    required_edge_columns = {"source", "target"}
    if not required_node_columns.issubset(nodes.columns):
        missing = sorted(required_node_columns - set(nodes.columns))
        raise ValueError(f"Missing node columns: {missing}")
    if not required_edge_columns.issubset(edges.columns):
        missing = sorted(required_edge_columns - set(edges.columns))
        raise ValueError(f"Missing edge columns: {missing}")
    if "S" not in edges.columns and not {"C", "L"}.issubset(edges.columns):
        raise ValueError("Edges require either explicit S or both C and L")

    node_ids = nodes["node"].astype(str)
    if node_ids.duplicated().any():
        duplicates = sorted(node_ids[node_ids.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate node identifiers: {duplicates}")

    node_set = set(node_ids)
    edge_endpoints = set(edges["source"].astype(str)) | set(
        edges["target"].astype(str)
    )
    missing_endpoints = sorted(edge_endpoints - node_set)
    if missing_endpoints:
        raise ValueError(f"Edge endpoints missing from node table: {missing_endpoints}")

    edge_pairs = edges[["source", "target"]].astype(str)
    if edge_pairs.duplicated().any():
        duplicates = edge_pairs[edge_pairs.duplicated(keep=False)].drop_duplicates()
        duplicate_text = [
            f"{row.source}->{row.target}" for row in duplicates.itertuples()
        ]
        raise ValueError(f"Duplicate directed edges: {duplicate_text}")

    b_numeric = pd.to_numeric(nodes["B"], errors="coerce")
    b_values = b_numeric.to_numpy(dtype=float)
    if (~np.isfinite(b_values)).any() or ((b_values < 0.0) | (b_values > 1.0)).any():
        raise ValueError("B must be finite numeric and in [0,1]")

    derived_edges = derive_susceptibility(edges)
    if "tau" not in derived_edges.columns:
        derived_edges["tau"] = 1.0

    tau_numeric = pd.to_numeric(derived_edges["tau"], errors="coerce")
    tau_values = tau_numeric.to_numpy(dtype=float)
    if (~np.isfinite(tau_values)).any() or (
        (tau_values < 0.0) | (tau_values > 1.0)
    ).any():
        raise ValueError("tau must be finite numeric and in [0,1]")
    derived_edges["tau"] = tau_numeric.astype(float)

    graph = nx.DiGraph()
    local = {}
    parameters = {}

    for node, blockage in zip(node_ids, b_values):
        graph.add_node(node)
        local[node] = float(blockage)

    for row in derived_edges.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        graph.add_edge(source, target)
        parameters[(source, target)] = float(row.S) * float(row.tau)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The path-pruned diagnostic requires a DAG")

    risk = {}
    ancestry = {}

    for target in nx.topological_sort(graph):
        positive_incoming = []
        for source in graph.predecessors(target):
            contribution = max(
                0.0,
                min(1.0, risk[source] * parameters[(source, target)]),
            )
            if contribution > 0.0:
                positive_incoming.append((contribution, set(ancestry[source])))

        # Build overlap-graph connected components across positive incoming
        # contributions (including transitive label overlap).
        groups = []
        for contribution, sources in positive_incoming:
            overlapping = [
                index
                for index, (_, existing_sources) in enumerate(groups)
                if sources & existing_sources
            ]

            if not overlapping:
                groups.append([contribution, set(sources)])
                continue

            retained = max(
                [contribution] + [groups[index][0] for index in overlapping]
            )
            merged_sources = set(sources)
            for index in reversed(overlapping):
                merged_sources.update(groups[index][1])
                groups.pop(index)
            groups.append([retained, merged_sources])

        complement = 1.0 - local[target]
        for contribution in sorted(group[0] for group in groups):
            complement *= 1.0 - contribution
        risk[target] = float(np.clip(1.0 - complement, 0.0, 1.0))

        target_ancestry = set()
        for _, sources in positive_incoming:
            target_ancestry.update(sources)
        if local[target] > 0.0:
            target_ancestry.add(target)
        ancestry[target] = target_ancestry

    return risk


def reconvergence_diagnostic(network, source="historical", outlet=None):
    nodes, edges = load_network(network, source=source)
    ordinary, _ = evaluate_gbbrpm(nodes, edges)
    pruned = evaluate_path_pruned_reference(nodes, edges)

    node_rows = []
    for node in ordinary:
        node_rows.append(
            {
                "network": network,
                "node": node,
                "gbbrpm_risk": ordinary[node],
                "path_pruned_risk": pruned[node],
                "overlap_inflation": ordinary[node] - pruned[node],
            }
        )

    node_results = pd.DataFrame(node_rows)
    outlet_row = node_results[node_results.node == outlet]
    outlet_oi = float(outlet_row.overlap_inflation.iloc[0]) if outlet else np.nan
    outlet_gbbrpm = float(outlet_row.gbbrpm_risk.iloc[0]) if outlet else np.nan
    outlet_pruned = float(outlet_row.path_pruned_risk.iloc[0]) if outlet else np.nan

    summary = {
        "network": network,
        "maximum_oi": float(node_results.overlap_inflation.max()),
        "mean_oi": float(node_results.overlap_inflation.mean()),
        "outlet_oi": outlet_oi,
        "outlet_gbbrpm": outlet_gbbrpm,
        "outlet_path_pruned": outlet_pruned,
    }
    return node_results, summary
