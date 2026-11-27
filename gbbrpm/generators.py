"""Deterministic benchmark topology and scenario generators for GBBRPM.

Implements the four planned reference-benchmark families from Chapter 3
(Section ``dependence-controls``):
1. Matched shared-versus-independent branch controls (560 scenarios) with
   exact Equation (3-15) closed-form receiver marginals.
2. Synthetic DAG uniform-weight benchmark across ``forward``, ``layered``,
   and ``preferential_attachment`` families (300 topologies: 720 development
   scenarios on seeds 41--45 and 2,880 held-out evaluation scenarios on seeds
   1001--1020).
3. Heterogeneous-weight synthetic DAG subset with SHA-256 deterministic edge
   ordering (24 development scenarios on seed 41 and 120 held-out evaluation
   scenarios on seeds 1001--1005; exact weight mean 0.20).
4. Express 4.18.2 real-topology stochastic suite (12 scenarios) with
   fail-closed SHA-256 verification against the frozen repository manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .csr_graph import PreparedDAG, prepare_dag, prepare_dag_from_arrays
from .exact_ref import closed_form_matched_branch

# ---------------------------------------------------------------------------
# Frozen protocol constants (Chapter 3, Table tab:reference-inventory)
# ---------------------------------------------------------------------------

BRANCH_M: tuple[int, ...] = (1, 2, 4, 8)
BRANCH_ELL: tuple[int, ...] = (2, 4)
BRANCH_P: tuple[float, ...] = (0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.0)
BRANCH_B: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)
BRANCH_CONDITIONS: tuple[str, ...] = ("shared", "independent")

DAG_FAMILIES: tuple[str, ...] = ("forward", "layered", "preferential_attachment")
DAG_SIZES: tuple[int, ...] = (20, 50, 100, 250)
DAG_WEIGHTS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.25, 0.50, 0.75)
DAG_B_SETTINGS: tuple[float, ...] = (0.5, 1.0)

DEV_SEEDS: tuple[int, ...] = (41, 42, 43, 44, 45)
EVAL_SEEDS: tuple[int, ...] = tuple(range(1001, 1021))

HETERO_DEV_SEEDS: tuple[int, ...] = (41,)
HETERO_EVAL_SEEDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)
HETERO_WEIGHT_LEVELS: tuple[float, float, float] = (0.10, 0.20, 0.30)

EXPRESS_WEIGHTS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.25, 0.50, 0.75)
EXPRESS_B_SETTINGS: tuple[float, ...] = (0.5, 1.0)
EXPRESS_EXPECTED_NODES: int = 71
EXPRESS_EXPECTED_EDGES: int = 128
EXPRESS_EXPECTED_ROOTS: int = 7


@dataclass(frozen=True)
class GeneratedTopology:
    """One deterministic synthetic DAG topology with recorded topological order."""

    family: str
    size: int
    seed: int
    split: str
    topo_node_ints: tuple[int, ...]
    edges_int: tuple[tuple[int, int], ...]
    edge_src_pos: np.ndarray
    edge_dst_pos: np.ndarray
    layer_sizes: tuple[int, ...] | None = None
    available_adjacent_pairs: int | None = None


@dataclass(frozen=True)
class BenchmarkScenario:
    """One self-contained benchmark scenario ready for deterministic/MC solvers."""

    scenario_id: str
    suite: str
    split: str
    family: str
    size: int
    graph_seed: int | None
    weight_regime: str
    weight_value: float | None
    b_setting: float
    mc_seed: int
    dag: PreparedDAG
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic seed and hash helpers
# ---------------------------------------------------------------------------


def derive_scenario_seed(scenario_id: str) -> int:
    """Derive a deterministic uint32 Monte Carlo seed from ``scenario_id`` via SHA-256."""
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _derive_topology_rng(family: str, size: int, seed: int) -> np.random.Generator:
    """Create an explicit seeded PCG64 Generator for ``(family, size, seed)``."""
    payload = f"{family}|{int(size)}|{int(seed)}|dag-topology-v1".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:16], byteorder="big", signed=False)
    return np.random.Generator(np.random.PCG64(seed_int))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Family 1: Matched shared-versus-independent branch controls (560 scenarios)
# ---------------------------------------------------------------------------


def build_matched_branch_scenario(
    m: int,
    ell: int,
    p: float,
    b: float,
    condition: str,
) -> BenchmarkScenario:
    """Construct one matched shared-vs-independent branch control scenario.

    Topology:
    - Root ``a`` connects with weight ``1.0`` to ``m`` branch-start nodes ``s_1 .. s_m``.
    - Each ``s_r`` connects to receiver ``d`` via an edge-disjoint path of ``ell``
      edges each of weight ``p``.
    - Intermediate local disturbances and ``B_d`` are ``0.0``.
    - ``shared``: ``B_a = b``, ``B_{s_r} = 0.0``.
    - ``independent``: ``B_a = 0.0`` (inactive root retained), ``B_{s_r} = b``.
    """
    if condition not in BRANCH_CONDITIONS:
        raise ValueError(f"condition must be in {BRANCH_CONDITIONS}, got {condition!r}")
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError(f"m must be an integer >= 1, got {m!r}")
    if not isinstance(ell, (int, np.integer)) or isinstance(ell, bool) or ell < 1:
        raise ValueError(f"ell must be an integer >= 1, got {ell!r}")

    m_int = int(m)
    ell_int = int(ell)
    p_val = float(p)
    b_val = float(b)

    node_ids: list[str] = ["a"]
    for r in range(1, m_int + 1):
        node_ids.append(f"s_{r}")
    for r in range(1, m_int + 1):
        for step in range(1, ell_int):
            node_ids.append(f"b{r}_{step}")
    node_ids.append("d")

    node_to_pos = {nid: idx for idx, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    B = np.zeros(n_nodes, dtype=np.float64)
    if condition == "shared":
        B[node_to_pos["a"]] = b_val
    else:
        for r in range(1, m_int + 1):
            B[node_to_pos[f"s_{r}"]] = b_val

    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    a_pos = node_to_pos["a"]
    d_pos = node_to_pos["d"]
    for r in range(1, m_int + 1):
        sr_pos = node_to_pos[f"s_{r}"]
        sources.append(a_pos)
        targets.append(sr_pos)
        weights.append(1.0)

        prev = sr_pos
        for step in range(1, ell_int):
            nxt = node_to_pos[f"b{r}_{step}"]
            sources.append(prev)
            targets.append(nxt)
            weights.append(p_val)
            prev = nxt

        sources.append(prev)
        targets.append(d_pos)
        weights.append(p_val)

    dag = prepare_dag_from_arrays(
        node_ids=node_ids,
        B=B,
        sources=np.asarray(sources, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.int32),
        weights=np.asarray(weights, dtype=np.float64),
        assume_topological=True,
    )

    exact_forms = closed_form_matched_branch(m_int, ell_int, p_val, b_val)
    exact_receiver_pi = float(exact_forms[condition])

    scenario_id = (
        f"branch|m={m_int}|ell={ell_int}|p={p_val:.2f}|b={b_val:.2f}|{condition}"
    )
    return BenchmarkScenario(
        scenario_id=scenario_id,
        suite="matched_branch",
        split="control",
        family="matched_branch",
        size=dag.n_nodes,
        graph_seed=None,
        weight_regime=f"p={p_val:.2f}",
        weight_value=p_val,
        b_setting=b_val,
        mc_seed=derive_scenario_seed(scenario_id),
        dag=dag,
        metadata={
            "m": m_int,
            "ell": ell_int,
            "p": p_val,
            "b": b_val,
            "condition": condition,
            "receiver_id": "d",
            "receiver_idx": dag.node_to_idx["d"],
            "exact_receiver_pi": exact_receiver_pi,
            "exact_shared_pi": float(exact_forms["shared"]),
            "exact_independent_pi": float(exact_forms["independent"]),
        },
    )


def generate_matched_branch_scenarios(
    m_values: Sequence[int] = BRANCH_M,
    ell_values: Sequence[int] = BRANCH_ELL,
    p_values: Sequence[float] = BRANCH_P,
    b_values: Sequence[float] = BRANCH_B,
    conditions: Sequence[str] = BRANCH_CONDITIONS,
) -> list[BenchmarkScenario]:
    """Generate the full 560-scenario matched shared-vs-independent branch grid."""
    scenarios: list[BenchmarkScenario] = []
    for m in m_values:
        for ell in ell_values:
            for p in p_values:
                for b in b_values:
                    for condition in conditions:
                        scenarios.append(
                            build_matched_branch_scenario(m, ell, p, b, condition)
                        )
    return scenarios


# ---------------------------------------------------------------------------
# Family 2: Synthetic DAG topologies and uniform-weight scenarios
# ---------------------------------------------------------------------------


def compute_layered_partition(size: int) -> tuple[tuple[int, ...], int]:
    """Compute front-loaded even layer sizes over ``L = ceil(sqrt(|V|))`` layers.

    Returns ``(layer_sizes, available_adjacent_pairs)``.
    """
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 2:
        raise ValueError(f"size must be an integer >= 2, got {size!r}")
    n = int(size)
    n_layers = math.isqrt(n)
    if n_layers * n_layers < n:
        n_layers += 1

    base = n // n_layers
    rem = n % n_layers
    layer_sizes = tuple(base + 1 if k < rem else base for k in range(n_layers))
    available_pairs = sum(
        layer_sizes[k] * layer_sizes[k + 1] for k in range(n_layers - 1)
    )
    return layer_sizes, int(available_pairs)


def _infer_split_from_seed(seed: int) -> str:
    if seed in DEV_SEEDS:
        return "dev"
    if seed in EVAL_SEEDS:
        return "eval"
    return "custom"


def generate_dag_topology(
    family: str,
    size: int,
    seed: int,
    split: str | None = None,
) -> GeneratedTopology:
    """Generate one deterministic synthetic DAG topology with exactly ``2 * |V|`` edges."""
    if family not in DAG_FAMILIES:
        raise ValueError(f"Unsupported DAG family {family!r}; expected one of {DAG_FAMILIES}")
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 5:
        raise ValueError(f"size must be an integer >= 5, got {size!r}")

    n = int(size)
    target_edges = 2 * n
    seed_int = int(seed)
    resolved_split = split if split is not None else _infer_split_from_seed(seed_int)
    rng = _derive_topology_rng(family, n, seed_int)

    layer_sizes: tuple[int, ...] | None = None
    available_pairs: int | None = None

    if family == "forward":
        total_pairs = n * (n - 1) // 2
        if total_pairs < target_edges:
            raise ValueError(
                f"Infeasible forward DAG: size={n} has only {total_pairs} ordered pairs "
                f"< required {target_edges} edges"
            )
        order = rng.permutation(n)
        topo_node_ints = tuple(int(x) for x in order)

        # Enumerate all ordered forward position pairs (i < j) in canonical order
        src_all, dst_all = np.triu_indices(n, k=1)
        chosen_idx = rng.choice(total_pairs, size=target_edges, replace=False)
        chosen_idx.sort()
        edge_src_pos = src_all[chosen_idx].astype(np.int32, copy=False)
        edge_dst_pos = dst_all[chosen_idx].astype(np.int32, copy=False)

    elif family == "layered":
        layer_sizes, available_pairs = compute_layered_partition(n)
        if available_pairs < target_edges:
            raise ValueError(
                f"Infeasible layered DAG: size={n} with layer_sizes={layer_sizes} "
                f"has only {available_pairs} adjacent-layer pairs < required {target_edges} edges"
            )

        topo_list: list[int] = []
        layer_ranges: list[tuple[int, int]] = []
        cursor = 0
        for sz in layer_sizes:
            layer_nodes = np.arange(cursor, cursor + sz, dtype=np.int64)
            permuted = rng.permutation(layer_nodes)
            start_pos = len(topo_list)
            topo_list.extend(int(x) for x in permuted)
            layer_ranges.append((start_pos, start_pos + sz))
            cursor += sz
        topo_node_ints = tuple(topo_list)

        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []
        for k in range(len(layer_ranges) - 1):
            a0, a1 = layer_ranges[k]
            b0, b1 = layer_ranges[k + 1]
            u_grid = np.repeat(np.arange(a0, a1, dtype=np.int32), b1 - b0)
            v_grid = np.tile(np.arange(b0, b1, dtype=np.int32), a1 - a0)
            src_parts.append(u_grid)
            dst_parts.append(v_grid)

        all_src = np.concatenate(src_parts)
        all_dst = np.concatenate(dst_parts)
        assert int(all_src.size) == available_pairs

        chosen_idx = rng.choice(available_pairs, size=target_edges, replace=False)
        chosen_idx.sort()
        edge_src_pos = all_src[chosen_idx].astype(np.int32, copy=False)
        edge_dst_pos = all_dst[chosen_idx].astype(np.int32, copy=False)

    elif family == "preferential_attachment":
        topo_node_ints = tuple(range(n))
        total_deg = np.zeros(n, dtype=np.int64)
        src_list: list[int] = []
        dst_list: list[int] = []

        # Initial 10 forward edges among the first 5 ordered nodes (0..4)
        for u in range(5):
            for v in range(u + 1, 5):
                src_list.append(u)
                dst_list.append(v)
                total_deg[u] += 1
                total_deg[v] += 1

        # Each subsequent node v = 5 .. n-1 attaches to 2 distinct earlier nodes
        # without replacement proportional to (current_total_degree + 1).
        for v in range(5, n):
            weights_pa = (total_deg[:v] + 1).astype(np.float64)
            probs = weights_pa / float(weights_pa.sum())
            chosen = rng.choice(v, size=2, replace=False, p=probs)
            u0, u1 = int(chosen[0]), int(chosen[1])
            if u0 > u1:
                u0, u1 = u1, u0
            src_list.append(u0)
            dst_list.append(v)
            src_list.append(u1)
            dst_list.append(v)
            total_deg[u0] += 1
            total_deg[u1] += 1
            total_deg[v] += 2

        edge_src_pos = np.asarray(src_list, dtype=np.int32)
        edge_dst_pos = np.asarray(dst_list, dtype=np.int32)

    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unsupported DAG family: {family!r}")

    assert int(edge_src_pos.size) == target_edges
    assert int(edge_dst_pos.size) == target_edges
    assert bool(np.all(edge_src_pos < edge_dst_pos))

    edges_int = tuple(
        (topo_node_ints[int(u)], topo_node_ints[int(v)])
        for u, v in zip(edge_src_pos.tolist(), edge_dst_pos.tolist())
    )

    return GeneratedTopology(
        family=family,
        size=n,
        seed=seed_int,
        split=resolved_split,
        topo_node_ints=topo_node_ints,
        edges_int=edges_int,
        edge_src_pos=edge_src_pos,
        edge_dst_pos=edge_dst_pos,
        layer_sizes=layer_sizes,
        available_adjacent_pairs=available_pairs,
    )


def build_dag_from_topology(
    topology: GeneratedTopology,
    weights: float | Sequence[float] | np.ndarray,
    b_setting: float,
) -> PreparedDAG:
    """Instantiate a ``PreparedDAG`` from a ``GeneratedTopology``, edge weights, and ``B``."""
    b_val = float(b_setting)
    if not (math.isfinite(b_val) and 0.0 <= b_val <= 1.0):
        raise ValueError(f"b_setting must be finite and in [0, 1], got {b_setting!r}")

    n = topology.size
    n_edges = int(topology.edge_src_pos.size)
    n_active = math.ceil(0.10 * n)

    B = np.zeros(n, dtype=np.float64)
    B[:n_active] = b_val

    if np.ndim(weights) == 0:
        w_arr = np.full(n_edges, float(weights), dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape != (n_edges,):
            raise ValueError(
                f"weights shape {w_arr.shape} must match edge count ({n_edges},)"
            )

    node_ids = tuple(str(x) for x in topology.topo_node_ints)
    return prepare_dag_from_arrays(
        node_ids=node_ids,
        B=B,
        sources=topology.edge_src_pos,
        targets=topology.edge_dst_pos,
        weights=w_arr,
        assume_topological=True,
    )


def compute_dag_structural_metadata(
    dag: PreparedDAG,
    layer_sizes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute structural topology, reachability, source-layer, and lineage-overlap metadata."""
    n = dag.n_nodes
    m = dag.n_edges

    in_deg = np.diff(dag.in_indptr).astype(np.int64)
    out_deg = np.diff(dag.out_indptr).astype(np.int64)

    # Longest directed path length (in edges) by topological DP
    longest_to = np.zeros(n, dtype=np.int64)
    for j in range(n):
        start = int(dag.in_indptr[j])
        end = int(dag.in_indptr[j + 1])
        if start < end:
            preds = dag.in_indices[start:end]
            longest_to[j] = int(np.max(longest_to[preds])) + 1
    depth = int(np.max(longest_to)) if n > 0 else 0

    # Active-source reachability and overlapping incoming local-origin lineages
    # via arbitrary-precision Python integer bitmasks in topological order.
    active_mask = dag.B > 0.0
    cand_mask = ~active_mask
    n_active = int(np.sum(active_mask))
    n_candidates = int(np.sum(cand_mask))

    anc_masks: list[int] = [0] * n
    reachable = np.zeros(n, dtype=np.bool_)
    has_overlap = np.zeros(n, dtype=np.bool_)

    for j in range(n):
        own_bit = (1 << j) if bool(active_mask[j]) else 0
        seen = 0
        overlap_j = False
        start = int(dag.in_indptr[j])
        end = int(dag.in_indptr[j + 1])
        for e in range(start, end):
            pred = int(dag.in_indices[e])
            pm = anc_masks[pred]
            if pm:
                if seen & pm:
                    overlap_j = True
                seen |= pm
        total_mask = own_bit | seen
        anc_masks[j] = total_mask
        reachable[j] = total_mask != 0
        has_overlap[j] = overlap_j

    reachable_all_count = int(np.sum(reachable))
    reachable_cand_count = int(np.sum(reachable & cand_mask))
    overlap_cand_count = int(np.sum(has_overlap & cand_mask))

    meta: dict[str, Any] = {
        "n_nodes": n,
        "n_edges": m,
        "depth": depth,
        "in_degree_min": int(np.min(in_deg)) if n > 0 else 0,
        "in_degree_mean": float(np.mean(in_deg)) if n > 0 else 0.0,
        "in_degree_max": int(np.max(in_deg)) if n > 0 else 0,
        "in_degree_std": float(np.std(in_deg)) if n > 0 else 0.0,
        "n_zero_in_degree": int(np.sum(in_deg == 0)),
        "out_degree_min": int(np.min(out_deg)) if n > 0 else 0,
        "out_degree_mean": float(np.mean(out_deg)) if n > 0 else 0.0,
        "out_degree_max": int(np.max(out_deg)) if n > 0 else 0,
        "out_degree_std": float(np.std(out_deg)) if n > 0 else 0.0,
        "n_zero_out_degree": int(np.sum(out_deg == 0)),
        "n_active_sources": n_active,
        "active_source_fraction": float(n_active / n) if n > 0 else 0.0,
        "n_candidates": n_candidates,
        "reachable_all_count": reachable_all_count,
        "reachable_fraction_all": float(reachable_all_count / n) if n > 0 else 0.0,
        "reachable_candidate_count": reachable_cand_count,
        "reachable_fraction_candidates": (
            float(reachable_cand_count / n_candidates) if n_candidates > 0 else 0.0
        ),
        "unreachable_candidate_count": n_candidates - reachable_cand_count,
        "overlap_candidate_count": overlap_cand_count,
        "overlap_candidate_fraction": (
            float(overlap_cand_count / n_candidates) if n_candidates > 0 else 0.0
        ),
        "overlap_reachable_candidate_fraction": (
            float(overlap_cand_count / reachable_cand_count)
            if reachable_cand_count > 0
            else 0.0
        ),
    }

    if layer_sizes is not None:
        sizes_tuple = tuple(int(s) for s in layer_sizes)
        if sum(sizes_tuple) != n:
            raise ValueError(
                f"Sum of layer_sizes ({sum(sizes_tuple)}) does not equal n_nodes ({n})"
            )
        per_layer_coverage: list[dict[str, Any]] = []
        cursor = 0
        for idx, sz in enumerate(sizes_tuple, start=1):
            active_in_layer = int(np.sum(active_mask[cursor : cursor + sz]))
            reach_in_layer = int(np.sum(reachable[cursor : cursor + sz]))
            per_layer_coverage.append(
                {
                    "layer": idx,
                    "size": sz,
                    "active_sources": active_in_layer,
                    "source_coverage": float(active_in_layer / sz) if sz > 0 else 0.0,
                    "reachable_count": reach_in_layer,
                    "reachable_fraction": float(reach_in_layer / sz) if sz > 0 else 0.0,
                }
            )
            cursor += sz

        meta["n_layers"] = len(sizes_tuple)
        meta["layer_sizes"] = list(sizes_tuple)
        meta["layer1_size"] = per_layer_coverage[0]["size"]
        meta["layer1_active_sources"] = per_layer_coverage[0]["active_sources"]
        meta["layer1_source_coverage"] = per_layer_coverage[0]["source_coverage"]
        if len(per_layer_coverage) >= 2:
            meta["layer2_size"] = per_layer_coverage[1]["size"]
            meta["layer2_active_sources"] = per_layer_coverage[1]["active_sources"]
            meta["layer2_source_coverage"] = per_layer_coverage[1]["source_coverage"]
        else:
            meta["layer2_size"] = 0
            meta["layer2_active_sources"] = 0
            meta["layer2_source_coverage"] = 0.0
        meta["source_layer_coverage"] = per_layer_coverage
    else:
        meta["n_layers"] = None
        meta["layer_sizes"] = None
        meta["layer1_size"] = None
        meta["layer1_active_sources"] = None
        meta["layer1_source_coverage"] = None
        meta["layer2_size"] = None
        meta["layer2_active_sources"] = None
        meta["layer2_source_coverage"] = None
        meta["source_layer_coverage"] = None

    return meta


def _resolve_split_seeds(
    split: str,
    seeds: Sequence[int] | None,
    dev_default: tuple[int, ...],
    eval_default: tuple[int, ...],
) -> tuple[int, ...]:
    if split not in ("dev", "eval"):
        raise ValueError(f"split must be 'dev' or 'eval', got {split!r}")
    if seeds is None:
        return dev_default if split == "dev" else eval_default

    seed_tuple = tuple(int(s) for s in seeds)
    forbidden = set(EVAL_SEEDS) if split == "dev" else set(DEV_SEEDS)
    overlap = sorted(set(seed_tuple) & forbidden)
    if overlap:
        raise ValueError(
            f"Split isolation violation: split={split!r} received seeds from opposite split: {overlap}"
        )
    return seed_tuple


def generate_dag_uniform_scenarios(
    split: str = "dev",
    families: Sequence[str] = DAG_FAMILIES,
    sizes: Sequence[int] = DAG_SIZES,
    seeds: Sequence[int] | None = None,
    weights: Sequence[float] = DAG_WEIGHTS,
    b_settings: Sequence[float] = DAG_B_SETTINGS,
    include_structural_metadata: bool = True,
) -> list[BenchmarkScenario]:
    """Generate uniform-weight synthetic DAG scenarios (720 dev or 2,880 eval)."""
    seed_tuple = _resolve_split_seeds(split, seeds, DEV_SEEDS, EVAL_SEEDS)
    scenarios: list[BenchmarkScenario] = []

    for family in families:
        for size in sizes:
            for seed in seed_tuple:
                topo = generate_dag_topology(family, int(size), int(seed), split=split)
                # Structural metadata depends only on topology + which nodes have B > 0
                # (identical across B in {0.5, 1.0} and all w > 0), so compute once per topology.
                structural_cache: dict[str, Any] | None = None
                for w in weights:
                    w_val = float(w)
                    for b in b_settings:
                        b_val = float(b)
                        dag = build_dag_from_topology(topo, w_val, b_val)
                        if include_structural_metadata and structural_cache is None:
                            structural_cache = compute_dag_structural_metadata(
                                dag, layer_sizes=topo.layer_sizes
                            )
                        scenario_id = (
                            f"dag_uniform|{split}|{family}|n={topo.size}"
                            f"|seed={topo.seed}|w={w_val:.2f}|B={b_val:.1f}"
                        )
                        meta: dict[str, Any] = {
                            "layer_sizes": topo.layer_sizes,
                            "available_adjacent_pairs": topo.available_adjacent_pairs,
                            "topo_node_ints": topo.topo_node_ints,
                        }
                        if structural_cache is not None:
                            meta["structural"] = structural_cache
                        scenarios.append(
                            BenchmarkScenario(
                                scenario_id=scenario_id,
                                suite="dag_uniform",
                                split=split,
                                family=family,
                                size=topo.size,
                                graph_seed=topo.seed,
                                weight_regime=f"w={w_val:.2f}",
                                weight_value=w_val,
                                b_setting=b_val,
                                mc_seed=derive_scenario_seed(scenario_id),
                                dag=dag,
                                metadata=meta,
                            )
                        )
    return scenarios


# ---------------------------------------------------------------------------
# Family 3: Heterogeneous-weight synthetic DAG subset (24 dev + 120 eval)
# ---------------------------------------------------------------------------


def assign_heterogeneous_weights(topology: GeneratedTopology) -> np.ndarray:
    """Assign deterministic SHA-256 ordered heterogeneous edge weights (exact mean 0.20).

    Edges are ordered by the SHA-256 hex digest of UTF-8 string
    ``f"{family}|{size}|{seed}|{source}|{destination}|hetero-v1"``
    using literal family names and integer node identifiers from the graph
    manifest, breaking any digest tie by ``(source, destination)``.
    First quarter gets ``0.10``, middle half gets ``0.20``, last quarter gets ``0.30``.
    """
    n_edges = len(topology.edges_int)
    if n_edges % 4 != 0:
        raise ValueError(
            f"Heterogeneous quartile assignment requires edge count divisible by 4, got {n_edges}"
        )

    ranked: list[tuple[str, int, int, int]] = []
    for idx, (src_int, dst_int) in enumerate(topology.edges_int):
        key_str = (
            f"{topology.family}|{topology.size}|{topology.seed}"
            f"|{src_int}|{dst_int}|hetero-v1"
        )
        digest_hex = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        ranked.append((digest_hex, src_int, dst_int, idx))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))

    q = n_edges // 4
    w_low, w_mid, w_high = HETERO_WEIGHT_LEVELS
    weights = np.empty(n_edges, dtype=np.float64)
    for rank_pos, (_, _, _, edge_idx) in enumerate(ranked):
        if rank_pos < q:
            weights[edge_idx] = w_low
        elif rank_pos < 3 * q:
            weights[edge_idx] = w_mid
        else:
            weights[edge_idx] = w_high

    return weights


def generate_dag_hetero_scenarios(
    split: str = "dev",
    families: Sequence[str] = DAG_FAMILIES,
    sizes: Sequence[int] = DAG_SIZES,
    seeds: Sequence[int] | None = None,
    b_settings: Sequence[float] = DAG_B_SETTINGS,
    include_structural_metadata: bool = True,
) -> list[BenchmarkScenario]:
    """Generate heterogeneous-weight synthetic DAG scenarios (24 dev or 120 eval)."""
    seed_tuple = _resolve_split_seeds(split, seeds, HETERO_DEV_SEEDS, HETERO_EVAL_SEEDS)
    scenarios: list[BenchmarkScenario] = []

    for family in families:
        for size in sizes:
            for seed in seed_tuple:
                topo = generate_dag_topology(family, int(size), int(seed), split=split)
                hetero_weights = assign_heterogeneous_weights(topo)
                structural_cache: dict[str, Any] | None = None
                for b in b_settings:
                    b_val = float(b)
                    dag = build_dag_from_topology(topo, hetero_weights, b_val)
                    if include_structural_metadata and structural_cache is None:
                        structural_cache = compute_dag_structural_metadata(
                            dag, layer_sizes=topo.layer_sizes
                        )
                    scenario_id = (
                        f"dag_hetero|{split}|{family}|n={topo.size}"
                        f"|seed={topo.seed}|w=hetero_v1|B={b_val:.1f}"
                    )
                    meta: dict[str, Any] = {
                        "layer_sizes": topo.layer_sizes,
                        "available_adjacent_pairs": topo.available_adjacent_pairs,
                        "topo_node_ints": topo.topo_node_ints,
                        "weight_mean": float(np.mean(hetero_weights)),
                        "weight_levels": list(HETERO_WEIGHT_LEVELS),
                    }
                    if structural_cache is not None:
                        meta["structural"] = structural_cache
                    scenarios.append(
                        BenchmarkScenario(
                            scenario_id=scenario_id,
                            suite="dag_hetero",
                            split=split,
                            family=family,
                            size=topo.size,
                            graph_seed=topo.seed,
                            weight_regime="hetero_v1",
                            weight_value=float(np.mean(hetero_weights)),
                            b_setting=b_val,
                            mc_seed=derive_scenario_seed(scenario_id),
                            dag=dag,
                            metadata=meta,
                        )
                    )
    return scenarios


# ---------------------------------------------------------------------------
# Family 4: Express 4.18.2 real-topology stochastic suite (12 scenarios)
# ---------------------------------------------------------------------------


def generate_express_scenarios(
    repo_root: str | Path | None = None,
    weights: Sequence[float] = EXPRESS_WEIGHTS,
    b_settings: Sequence[float] = EXPRESS_B_SETTINGS,
    verify_hashes: bool = True,
    include_structural_metadata: bool = True,
) -> list[BenchmarkScenario]:
    """Load Express 4.18.2 verified topology and generate its 12 controlled stochastic scenarios."""
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    express_dir = root / "data" / "software" / "express_4.18.2"
    nodes_path = express_dir / "nodes.csv"
    edges_path = express_dir / "edges.csv"
    case_manifest_path = express_dir / "manifest.json"
    paper_manifest_path = root / "results" / "paper" / "paper_outputs_manifest.json"

    for p in (nodes_path, edges_path, case_manifest_path):
        if not p.exists():
            raise FileNotFoundError(f"Required Express input file missing: {p}")

    nodes_sha256 = _sha256_file(nodes_path)
    edges_sha256 = _sha256_file(edges_path)

    case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))

    if verify_hashes:
        if not paper_manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot verify Express SHA-256: manifest missing at {paper_manifest_path}"
            )
        paper_manifest = json.loads(paper_manifest_path.read_text(encoding="utf-8"))
        input_hashes = paper_manifest.get("input_sha256", {})
        expected_nodes_hash = input_hashes.get("data/software/express_4.18.2/nodes.csv")
        expected_edges_hash = input_hashes.get("data/software/express_4.18.2/edges.csv")
        if expected_nodes_hash != nodes_sha256:
            raise ValueError(
                f"Express nodes.csv SHA-256 mismatch: expected {expected_nodes_hash}, got {nodes_sha256}"
            )
        if expected_edges_hash != edges_sha256:
            raise ValueError(
                f"Express edges.csv SHA-256 mismatch: expected {expected_edges_hash}, got {edges_sha256}"
            )

    nodes_raw = pd.read_csv(nodes_path)
    edges_raw = pd.read_csv(edges_path)

    if len(nodes_raw) != int(case_manifest.get("node_count", EXPRESS_EXPECTED_NODES)):
        raise ValueError(
            f"Express node count mismatch: got {len(nodes_raw)}, "
            f"expected {case_manifest.get('node_count', EXPRESS_EXPECTED_NODES)}"
        )
    if len(edges_raw) != int(case_manifest.get("edge_count", EXPRESS_EXPECTED_EDGES)):
        raise ValueError(
            f"Express edge count mismatch: got {len(edges_raw)}, "
            f"expected {case_manifest.get('edge_count', EXPRESS_EXPECTED_EDGES)}"
        )

    advisory_mask = pd.to_numeric(nodes_raw["B"], errors="raise") > 0.0
    advisory_root_ids = tuple(nodes_raw.loc[advisory_mask, "node"].astype(str).tolist())
    expected_roots = int(
        case_manifest.get("vulnerable_node_count", EXPRESS_EXPECTED_ROOTS)
    )
    if len(advisory_root_ids) != expected_roots:
        raise ValueError(
            f"Express advisory-origin count mismatch: got {len(advisory_root_ids)}, "
            f"expected {expected_roots}"
        )

    scenarios: list[BenchmarkScenario] = []
    structural_cache: dict[str, Any] | None = None

    for w in weights:
        w_val = float(w)
        edges_df = edges_raw[["source", "target"]].copy()
        edges_df["S"] = w_val
        edges_df["tau"] = 1.0
        for b in b_settings:
            b_val = float(b)
            nodes_df = nodes_raw[["node"]].copy()
            nodes_df["B"] = np.where(advisory_mask, b_val, 0.0)
            dag = prepare_dag(nodes_df, edges_df)
            if include_structural_metadata and structural_cache is None:
                structural_cache = compute_dag_structural_metadata(dag)

            scenario_id = f"express_4.18.2|w={w_val:.2f}|B={b_val:.1f}"
            meta: dict[str, Any] = {
                "case_id": case_manifest.get("case_id", "software_npm_express_4.18.2"),
                "nodes_sha256": nodes_sha256,
                "edges_sha256": edges_sha256,
                "n_advisory_roots": len(advisory_root_ids),
                "advisory_root_ids": advisory_root_ids,
            }
            if structural_cache is not None:
                meta["structural"] = structural_cache

            scenarios.append(
                BenchmarkScenario(
                    scenario_id=scenario_id,
                    suite="express",
                    split="real_topology",
                    family="express_4.18.2",
                    size=dag.n_nodes,
                    graph_seed=None,
                    weight_regime=f"w={w_val:.2f}",
                    weight_value=w_val,
                    b_setting=b_val,
                    mc_seed=derive_scenario_seed(scenario_id),
                    dag=dag,
                    metadata=meta,
                )
            )
    return scenarios
