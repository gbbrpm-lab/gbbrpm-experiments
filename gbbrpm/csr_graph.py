"""Lightweight validated CSR DAG representation in topological order.

Shared by the exact enumeration solver, bit-packed Monte Carlo reference,
deterministic benchmark comparators, and Two-Stage Certified Top-k Screening.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Sequence

import numpy as np
import pandas as pd

from .core import derive_susceptibility


@dataclass(frozen=True)
class PreparedDAG:
    """Validated directed acyclic graph stored in topological order ``0 .. n_nodes - 1``.

    Attributes
    ----------
    node_ids:
        Tuple mapping topological index ``i`` to canonical string node identifier.
    node_to_idx:
        Mapping from string node identifier to topological index ``0 .. n_nodes - 1``.
    B:
        Float64 array of shape ``(n_nodes,)`` with strictly finite local activation
        probabilities in ``[0, 1]``.
    in_indptr, in_indices, in_weights:
        Predecessor CSR arrays (incoming edges to each target ``j``). Predecessors
        within each row are sorted by ascending topological index ``i < j``.
    out_indptr, out_indices, out_weights:
        Successor CSR arrays (outgoing edges from each source ``i``). Successors
        within each row are sorted by ascending topological index ``j > i``.
    n_nodes:
        Number of nodes ``|V|``.
    n_edges:
        Number of directed edges ``|E|``.
    """

    node_ids: tuple[str, ...]
    node_to_idx: dict[str, int]
    B: np.ndarray
    in_indptr: np.ndarray
    in_indices: np.ndarray
    in_weights: np.ndarray
    out_indptr: np.ndarray
    out_indices: np.ndarray
    out_weights: np.ndarray
    n_nodes: int
    n_edges: int


def _validate_unit_interval_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if arr.size > 0:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain strictly finite numeric values")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"{name} must be in [0, 1]")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _resolve_endpoint_indices(
    endpoints: Sequence[int | str] | np.ndarray,
    raw_to_idx: dict[str, int],
    n_nodes: int,
    label: str,
) -> np.ndarray:
    if isinstance(endpoints, np.ndarray):
        if endpoints.ndim != 1:
            raise ValueError(f"{label} must be a 1-D sequence")
        if endpoints.size == 0:
            return np.empty(0, dtype=np.int32)
        if np.issubdtype(endpoints.dtype, np.integer):
            idx = endpoints.astype(np.int64, copy=False)
            if np.any((idx < 0) | (idx >= n_nodes)):
                raise ValueError(
                    f"{label} integer indices out of range [0, {n_nodes - 1}]"
                )
            return np.ascontiguousarray(idx, dtype=np.int32)
        items = endpoints.tolist()
    else:
        items = list(endpoints)
        if not items:
            return np.empty(0, dtype=np.int32)
        if all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in items):
            idx = np.asarray(items, dtype=np.int64)
            if np.any((idx < 0) | (idx >= n_nodes)):
                raise ValueError(
                    f"{label} integer indices out of range [0, {n_nodes - 1}]"
                )
            return np.ascontiguousarray(idx, dtype=np.int32)

    resolved = np.empty(len(items), dtype=np.int32)
    missing: list[str] = []
    for k, raw in enumerate(items):
        key = str(raw)
        pos = raw_to_idx.get(key)
        if pos is None:
            missing.append(key)
        else:
            resolved[k] = pos
    if missing:
        raise ValueError(
            f"Edge {label} endpoints missing from node table: {sorted(set(missing))}"
        )
    return resolved


def _kahn_topological_order(
    n_nodes: int, src_idx: np.ndarray, dst_idx: np.ndarray
) -> np.ndarray:
    """Deterministic Kahn topological sort preserving input order on ties."""
    if n_nodes == 0:
        return np.empty(0, dtype=np.int32)

    n_edges = int(src_idx.size)
    if n_edges == 0:
        return np.arange(n_nodes, dtype=np.int32)

    in_deg = np.bincount(dst_idx, minlength=n_nodes).astype(np.int32, copy=False)
    out_order = np.argsort(src_idx, kind="mergesort")
    src_sorted = src_idx[out_order]
    dst_sorted = dst_idx[out_order]
    out_counts = np.bincount(src_sorted, minlength=n_nodes)
    out_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(out_counts, out=out_indptr[1:])

    heap = [int(i) for i in np.flatnonzero(in_deg == 0)]
    heapq.heapify(heap)
    order = np.empty(n_nodes, dtype=np.int32)
    written = 0

    while heap:
        u = heapq.heappop(heap)
        order[written] = u
        written += 1
        start = int(out_indptr[u])
        end = int(out_indptr[u + 1])
        for e in range(start, end):
            v = int(dst_sorted[e])
            in_deg[v] -= 1
            if in_deg[v] == 0:
                heapq.heappush(heap, v)

    if written != n_nodes:
        raise ValueError("GBBRPM v1 requires a DAG")
    return order


def prepare_dag_from_arrays(
    node_ids: Sequence[str],
    B: Sequence[float] | np.ndarray,
    sources: Sequence[int | str] | np.ndarray,
    targets: Sequence[int | str] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    assume_topological: bool = False,
) -> PreparedDAG:
    """Build and validate a ``PreparedDAG`` directly from array inputs.

    ``sources`` and ``targets`` may be either 0-based integer indices into
    ``node_ids`` or string node identifiers matching ``node_ids``.
    Even when ``assume_topological=True``, acyclicity, finite ``[0, 1]`` bounds,
    self-loops, and duplicate directed edges are strictly validated fail-closed.
    """
    raw_node_ids = tuple(str(x) for x in node_ids)
    n_nodes = len(raw_node_ids)
    raw_to_idx: dict[str, int] = {}
    duplicates: list[str] = []
    for i, nid in enumerate(raw_node_ids):
        if nid in raw_to_idx:
            duplicates.append(nid)
        else:
            raw_to_idx[nid] = i
    if duplicates:
        raise ValueError(f"Duplicate node identifiers: {sorted(set(duplicates))}")

    B_arr = _validate_unit_interval_array(np.asarray(B, dtype=np.float64), "B")
    if B_arr.shape[0] != n_nodes:
        raise ValueError(
            f"Length of B ({B_arr.shape[0]}) must match number of nodes ({n_nodes})"
        )

    src_idx = _resolve_endpoint_indices(sources, raw_to_idx, n_nodes, "source")
    dst_idx = _resolve_endpoint_indices(targets, raw_to_idx, n_nodes, "target")
    w_arr = _validate_unit_interval_array(
        np.asarray(weights, dtype=np.float64), "weights"
    )

    n_edges = int(src_idx.size)
    if dst_idx.size != n_edges or w_arr.size != n_edges:
        raise ValueError(
            f"Edge array lengths mismatch: sources={n_edges}, "
            f"targets={dst_idx.size}, weights={w_arr.size}"
        )

    if n_edges > 0:
        if np.any(src_idx == dst_idx):
            bad = int(np.flatnonzero(src_idx == dst_idx)[0])
            nid = raw_node_ids[int(src_idx[bad])]
            raise ValueError(f"Self-loop detected on node {nid!r}; GBBRPM v1 requires a DAG")

        edge_keys = src_idx.astype(np.int64) * np.int64(n_nodes) + dst_idx.astype(np.int64)
        _, counts = np.unique(edge_keys, return_counts=True)
        if np.any(counts > 1):
            raise ValueError("Duplicate directed edges detected")

    if assume_topological:
        if n_edges > 0 and not np.all(src_idx < dst_idx):
            raise ValueError(
                "Edges are not in topological order (source >= target); "
                "GBBRPM v1 requires a DAG"
            )
        topo_node_ids = raw_node_ids
        topo_to_idx = raw_to_idx
        topo_B = B_arr
        topo_src = src_idx
        topo_dst = dst_idx
    else:
        order = _kahn_topological_order(n_nodes, src_idx, dst_idx)
        inv_order = np.empty(n_nodes, dtype=np.int32)
        inv_order[order] = np.arange(n_nodes, dtype=np.int32)

        topo_node_ids = tuple(raw_node_ids[int(i)] for i in order)
        topo_to_idx = {nid: idx for idx, nid in enumerate(topo_node_ids)}
        topo_B = np.ascontiguousarray(B_arr[order], dtype=np.float64)
        topo_src = inv_order[src_idx]
        topo_dst = inv_order[dst_idx]

    in_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    out_indptr = np.zeros(n_nodes + 1, dtype=np.int32)

    if n_edges == 0:
        empty_i32 = np.empty(0, dtype=np.int32)
        empty_f64 = np.empty(0, dtype=np.float64)
        return PreparedDAG(
            node_ids=topo_node_ids,
            node_to_idx=topo_to_idx,
            B=topo_B,
            in_indptr=in_indptr,
            in_indices=empty_i32,
            in_weights=empty_f64,
            out_indptr=out_indptr,
            out_indices=empty_i32.copy(),
            out_weights=empty_f64.copy(),
            n_nodes=n_nodes,
            n_edges=0,
        )

    # Predecessor CSR sorted by (target, source)
    in_perm = np.lexsort((topo_src, topo_dst))
    in_indices = np.ascontiguousarray(topo_src[in_perm], dtype=np.int32)
    in_weights = np.ascontiguousarray(w_arr[in_perm], dtype=np.float64)
    in_counts = np.bincount(topo_dst[in_perm], minlength=n_nodes)
    np.cumsum(in_counts, out=in_indptr[1:])

    # Successor CSR sorted by (source, target)
    out_perm = np.lexsort((topo_dst, topo_src))
    out_indices = np.ascontiguousarray(topo_dst[out_perm], dtype=np.int32)
    out_weights = np.ascontiguousarray(w_arr[out_perm], dtype=np.float64)
    out_counts = np.bincount(topo_src[out_perm], minlength=n_nodes)
    np.cumsum(out_counts, out=out_indptr[1:])

    return PreparedDAG(
        node_ids=topo_node_ids,
        node_to_idx=topo_to_idx,
        B=topo_B,
        in_indptr=in_indptr,
        in_indices=in_indices,
        in_weights=in_weights,
        out_indptr=out_indptr,
        out_indices=out_indices,
        out_weights=out_weights,
        n_nodes=n_nodes,
        n_edges=n_edges,
    )


def prepare_dag(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> PreparedDAG:
    """Validate pandas node/edge tables and construct a ``PreparedDAG``.

    Follows the GBBRPM v1 contract:
    - ``nodes_df`` requires unique non-null ``node`` identifiers and finite ``B`` in ``[0, 1]``.
    - ``edges_df`` requires ``source`` and ``target`` plus either explicit ``S`` or
      primitive ``L`` and ``C`` (derived via ``gbbrpm.core.derive_susceptibility``).
      ``tau`` is optional and defaults to ``1.0``.
    - Non-finite values, out-of-range inputs, missing endpoints, self-loops,
      duplicate directed edges, and cycles are rejected fail-closed with ``ValueError``.
    """
    required_node_columns = {"node", "B"}
    required_edge_columns = {"source", "target"}
    if not required_node_columns.issubset(nodes_df.columns):
        missing = sorted(required_node_columns - set(nodes_df.columns))
        raise ValueError(f"Missing node columns: {missing}")
    if not required_edge_columns.issubset(edges_df.columns):
        missing = sorted(required_edge_columns - set(edges_df.columns))
        raise ValueError(f"Missing edge columns: {missing}")

    if nodes_df["node"].isna().any():
        raise ValueError("Node identifiers must not be null")

    node_ids = nodes_df["node"].astype(str).tolist()
    b_series = pd.to_numeric(nodes_df["B"], errors="coerce")
    if b_series.isna().any():
        raise ValueError("B must be strictly finite numeric and in [0, 1]")
    B_arr = _validate_unit_interval_array(b_series.to_numpy(dtype=np.float64), "B")

    if len(edges_df) == 0:
        return prepare_dag_from_arrays(
            node_ids=node_ids,
            B=B_arr,
            sources=np.empty(0, dtype=np.int32),
            targets=np.empty(0, dtype=np.int32),
            weights=np.empty(0, dtype=np.float64),
            assume_topological=False,
        )

    if edges_df["source"].isna().any() or edges_df["target"].isna().any():
        raise ValueError("Edge source and target identifiers must not be null")

    # Enforce strict finite checks on raw L/C/S prior to susceptibility derivation
    has_s = "S" in edges_df.columns and edges_df["S"].notna().any()
    has_l = "L" in edges_df.columns and edges_df["L"].notna().any()
    has_c = "C" in edges_df.columns and edges_df["C"].notna().any()
    if has_s and (has_l or has_c):
        s_nonnull = edges_df["S"].notna()
        lc_nonnull = (
            (edges_df["L"].notna() if "L" in edges_df.columns else False)
            | (edges_df["C"].notna() if "C" in edges_df.columns else False)
        )
        if bool((s_nonnull & lc_nonnull).any()):
            raise ValueError(
                "Ambiguous edge susceptibility: provide either explicit S or L and C, not both"
            )
    if not has_s and (has_l != has_c):
        raise ValueError("Edges require both L and C when explicit S is absent")
    if has_l and has_c:
        l_num = pd.to_numeric(edges_df["L"], errors="coerce").to_numpy(dtype=np.float64)
        c_num = pd.to_numeric(edges_df["C"], errors="coerce").to_numpy(dtype=np.float64)
        if not (np.all(np.isfinite(l_num)) and np.all(np.isfinite(c_num))):
            raise ValueError("L and C must be strictly finite numeric values")

    derived = derive_susceptibility(edges_df)
    s_arr = _validate_unit_interval_array(
        pd.to_numeric(derived["S"], errors="coerce").to_numpy(dtype=np.float64), "S"
    )

    if "tau" in derived.columns:
        tau_series = pd.to_numeric(derived["tau"], errors="coerce")
        if tau_series.isna().any():
            raise ValueError("tau must be strictly finite numeric and in [0, 1]")
        tau_arr = _validate_unit_interval_array(
            tau_series.to_numpy(dtype=np.float64), "tau"
        )
    else:
        tau_arr = np.ones(len(derived), dtype=np.float64)

    weights = _validate_unit_interval_array(s_arr * tau_arr, "weights")
    sources = derived["source"].astype(str).tolist()
    targets = derived["target"].astype(str).tolist()

    return prepare_dag_from_arrays(
        node_ids=node_ids,
        B=B_arr,
        sources=sources,
        targets=targets,
        weights=weights,
        assume_topological=False,
    )
