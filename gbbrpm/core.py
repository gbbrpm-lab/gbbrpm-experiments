"""Domain-agnostic GBBRPM calculation engine.

The public functions operate on pandas DataFrames and intentionally contain no
repository-specific file paths. Applications can therefore supply synthetic,
operational, or imported networks through the same interface.
"""

import networkx as nx
import numpy as np
import pandas as pd


def _validate_finite_series(series, name, *, lower=None, upper=None, strict_lower=False):
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if (~np.isfinite(values)).any():
        if strict_lower and lower is not None:
            raise ValueError(f"{name} must be finite and > {lower:g}")
        if lower is not None and upper is not None:
            raise ValueError(f"{name} must be finite numeric and in [{lower:g},{upper:g}]")
        if lower is not None:
            raise ValueError(f"{name} must be finite and >= {lower:g}")
        raise ValueError(f"{name} must be finite numeric")

    if lower is not None:
        if strict_lower and (values <= lower).any():
            raise ValueError(f"{name} must be > {lower:g}")
        if not strict_lower and (values < lower).any():
            if upper is not None:
                raise ValueError(f"{name} must be finite numeric and in [{lower:g},{upper:g}]")
            raise ValueError(f"{name} must be >= {lower:g}")
    if upper is not None and (values > upper).any():
        if lower is not None:
            raise ValueError(f"{name} must be finite numeric and in [{lower:g},{upper:g}]")
        raise ValueError(f"{name} must be <= {upper:g}")

    return numeric.astype(float)


def derive_susceptibility(edges):
    """Return an edge table with a bounded susceptibility column ``S``.

    Drainage-style inputs may provide primitive ``L`` and ``C`` columns, while
    other domains may provide an explicit normalized ``S`` column. Providing
    both explicit ``S`` and primitive ``L``/``C`` on the same edge is rejected
    fail-closed under the GBBRPM input contract.
    """

    e = edges.copy()
    has_s_col = "S" in e.columns
    has_l_col = "L" in e.columns
    has_c_col = "C" in e.columns

    if not has_s_col and not (has_l_col and has_c_col):
        raise ValueError("Edges require either explicit S or both C and L")

    s_present = e["S"].notna() if has_s_col else pd.Series(False, index=e.index)
    l_present = e["L"].notna() if has_l_col else pd.Series(False, index=e.index)
    c_present = e["C"].notna() if has_c_col else pd.Series(False, index=e.index)

    if (s_present & (l_present | c_present)).any():
        raise ValueError(
            "Ambiguous edge susceptibility input: provide either explicit S or "
            "primitive L and C, not both"
        )

    if len(e) == 0:
        for col in ("S", "L", "C", "u"):
            if col not in e.columns:
                e[col] = pd.Series(dtype=float)
        return e

    if has_s_col and not l_present.any() and not c_present.any():
        e["S"] = _validate_finite_series(e["S"], "S", lower=0.0, upper=1.0)
        if not has_l_col:
            e["L"] = float("nan")
        if not has_c_col:
            e["C"] = float("nan")
        if "u" not in e.columns:
            e["u"] = float("nan")
        return e

    if has_l_col and has_c_col and not s_present.any():
        e["C"] = _validate_finite_series(e["C"], "C", lower=0.0, strict_lower=True)
        e["L"] = _validate_finite_series(e["L"], "L", lower=0.0)
        e["u"] = e["L"] / e["C"]
        e["S"] = e["u"].clip(0.0, 1.0)
        return e

    if (l_present != c_present).any():
        raise ValueError("L and C must be provided together and be finite")

    lc_present = l_present & c_present
    if (~s_present & ~lc_present).any():
        raise ValueError(
            "Each edge must provide finite explicit S or finite primitive L and C"
        )

    s_out = pd.Series(np.nan, index=e.index, dtype=float)
    l_out = pd.Series(np.nan, index=e.index, dtype=float)
    c_out = pd.Series(np.nan, index=e.index, dtype=float)
    u_out = pd.Series(np.nan, index=e.index, dtype=float)

    if s_present.any():
        s_out.loc[s_present] = _validate_finite_series(
            e.loc[s_present, "S"], "S", lower=0.0, upper=1.0
        )

    if lc_present.any():
        c_valid = _validate_finite_series(
            e.loc[lc_present, "C"], "C", lower=0.0, strict_lower=True
        )
        l_valid = _validate_finite_series(
            e.loc[lc_present, "L"], "L", lower=0.0
        )
        u_valid = l_valid / c_valid
        c_out.loc[lc_present] = c_valid
        l_out.loc[lc_present] = l_valid
        u_out.loc[lc_present] = u_valid
        s_out.loc[lc_present] = u_valid.clip(0.0, 1.0)

    e["S"] = s_out
    e["L"] = l_out
    e["C"] = c_out
    e["u"] = u_out
    return e


def evaluate_gbbrpm(nodes, edges):
    """Evaluate one directed acyclic network and return node and edge risks.

    ``nodes`` requires unique ``node`` identifiers and local disturbance ``B``.
    ``edges`` requires ``source`` and ``target`` plus either explicit ``S`` or
    primitive ``L`` and ``C``. ``tau`` is optional and defaults to ``1.0``.
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
        duplicate_text = [f"{row.source}->{row.target}" for row in duplicates.itertuples()]
        raise ValueError(f"Duplicate directed edges: {duplicate_text}")

    e = derive_susceptibility(edges)
    if "tau" not in e.columns:
        e["tau"] = 1.0

    e["tau"] = _validate_finite_series(e["tau"], "tau", lower=0.0, upper=1.0)
    b_values = _validate_finite_series(nodes["B"], "B", lower=0.0, upper=1.0)

    graph = nx.DiGraph()
    local_disturbance = {}

    for node, blockage in zip(node_ids, b_values):
        graph.add_node(node)
        local_disturbance[node] = float(blockage)

    lookup = {}
    for row in e.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        graph.add_edge(source, target)
        lookup[(source, target)] = {
            "S": float(row.S),
            "tau": float(row.tau),
            "L": float(row.L),
            "C": float(row.C),
            "u": float(row.u),
        }

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("GBBRPM v1 requires a DAG")

    risk = {}
    contributions = []
    for target in nx.topological_sort(graph):
        complement_product = 1.0
        for source in graph.predecessors(target):
            edge = lookup[(source, target)]
            contribution = max(
                0.0,
                min(1.0, edge["S"] * edge["tau"] * risk[source]),
            )
            complement_product *= 1 - contribution
            contributions.append(
                {
                    "source": source,
                    "target": target,
                    "R_source": risk[source],
                    "S": edge["S"],
                    "tau": edge["tau"],
                    "Q": contribution,
                    "L": edge["L"],
                    "C": edge["C"],
                    "u": edge["u"],
                }
            )

        risk[target] = max(
            0.0,
            min(1.0, 1 - (1 - local_disturbance[target]) * complement_product),
        )

    return risk, pd.DataFrame(contributions)
