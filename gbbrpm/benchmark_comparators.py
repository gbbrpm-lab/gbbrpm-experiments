"""Deterministic benchmark comparators on ``PreparedDAG`` graphs.

Implements all task-matched deterministic receiving-state methods specified in
Chapter 3 (Sections 3.6.3, 3.6.5, and 3.7.1) over the shared CSR DAG format:

- ``compute_local_only``: local input ``B_j``.
- ``compute_gbbrpm_fast``: complement-product recursion ``R_j`` (exact parity with
  ``gbbrpm.core.evaluate_gbbrpm``).
- ``compute_recursive_max``: strongest local-origin path lower bound ``M_j`` (Eq. 3-12).
- ``compute_clipped_sum``: clipped-additive outer upper bound ``H_j`` (Eq. 3-12).
- ``compute_local_upstream``: one-hop recursion ablation ``R_j^{local-upstream}`` (Eq. 3-16).
- ``compute_target_tree_bound`` / ``compute_all_tree_bounds``: MIA-inspired
  unthresholded target-specific restricted in-arborescence lower bound ``T_j``.
- ``compute_abbe_lower``: published nodewise DAG lower bound of Abbe, Kulkarni,
  & Lee (2020, JMLR v21/18-112, Section 4.1, Definitions 11-12, Equations 21-23,
  Algorithm 4) via an always-active auxiliary source and its Minimum-Distance
  Directed Acyclic Subnetwork (MDAS).
- ``compute_ancestry_diagnostic``: prospective connected-component ancestry-label
  diagnostic restricted strictly to positive-contribution edges ``Q_{ij} > 0``
  (Section 3.7.1).
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Sequence

import numpy as np

from .csr_graph import PreparedDAG


def compute_local_only(dag: PreparedDAG) -> np.ndarray:
    """Return local-only disturbance scores ``B`` in topological order."""
    return np.array(dag.B, dtype=np.float64, copy=True)


def compute_gbbrpm_fast(dag: PreparedDAG) -> np.ndarray:
    """Evaluate ordinary GBBRPM complement-product recursion ``R_j`` on a ``PreparedDAG``.

    Exact algebraic equivalent of ``gbbrpm.core.evaluate_gbbrpm``:
    ``R_j = 1 - (1 - B_j) * prod_{i in P(j)} (1 - w_{ij} * R_i)``.
    """
    n = dag.n_nodes
    R = np.empty(n, dtype=np.float64)
    B = dag.B
    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    in_weights = dag.in_weights

    for j in range(n):
        comp = 1.0
        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])
        for e in range(start, end):
            q = float(in_weights[e] * R[int(in_indices[e])])
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            comp *= 1.0 - q
        val = 1.0 - (1.0 - float(B[j])) * comp
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        R[j] = val

    return R


def compute_recursive_max(dag: PreparedDAG) -> np.ndarray:
    """Evaluate recursive maximum lower bound ``M_j`` (Equation 3-12).

    ``M_j = max(B_j, max_{i in P(j)} w_{ij} * M_i)``.
    """
    n = dag.n_nodes
    M = np.empty(n, dtype=np.float64)
    B = dag.B
    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    in_weights = dag.in_weights

    for j in range(n):
        best = float(B[j])
        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])
        for e in range(start, end):
            cand = float(in_weights[e] * M[int(in_indices[e])])
            if cand > best:
                best = cand
        if best < 0.0:
            best = 0.0
        elif best > 1.0:
            best = 1.0
        M[j] = best

    return M


def compute_clipped_sum(dag: PreparedDAG) -> np.ndarray:
    """Evaluate recursive clipped-additive outer upper bound ``H_j`` (Equation 3-12).

    ``H_j = min(1, B_j + sum_{i in P(j)} w_{ij} * H_i)``.
    """
    n = dag.n_nodes
    H = np.empty(n, dtype=np.float64)
    B = dag.B
    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    in_weights = dag.in_weights

    for j in range(n):
        total = float(B[j])
        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])
        for e in range(start, end):
            total += float(in_weights[e] * H[int(in_indices[e])])
        if total < 0.0:
            total = 0.0
        elif total > 1.0:
            total = 1.0
        H[j] = total

    return H


def compute_local_upstream(dag: PreparedDAG) -> np.ndarray:
    """Evaluate one-hop local-upstream recursion ablation (Equation 3-16).

    ``R_j^{local-upstream} = 1 - (1 - B_j) * prod_{i in P(j)} (1 - w_{ij} * B_i)``.
    """
    n = dag.n_nodes
    out = np.empty(n, dtype=np.float64)
    B = dag.B
    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    in_weights = dag.in_weights

    for j in range(n):
        comp = 1.0
        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])
        for e in range(start, end):
            q = float(in_weights[e] * B[int(in_indices[e])])
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            comp *= 1.0 - q
        val = 1.0 - (1.0 - float(B[j])) * comp
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        out[j] = val

    return out


def compute_target_tree_bound(dag: PreparedDAG, target_idx: int) -> float:
    """Compute unthresholded target-specific in-arborescence lower bound ``T_j`` (Sec. 3.6.5).

    1. Reverse topological DP over ancestors ``u < target_idx`` computes the
       maximum positive weighted-path product ``best_prob[u]`` to ``target_idx``.
       Exact ties in ``w_{uv} * best_prob[v]`` are broken consistently by
       lexicographically smallest target node ID ``dag.node_ids[v]``.
    2. Each ancestor with ``best_prob[u] > 0`` retains exactly one outgoing edge
       ``(u, next_v[u])`` (no positive-path threshold is applied), forming a
       target-directed in-arborescence rooted at ``target_idx``.
    3. Forward topological complement-product recursion on the retained tree
       evaluates ``T_j`` exactly under independent primitives, guaranteeing
       ``M_j <= T_j <= pi_j <= R_j <= H_j``.
    """
    t = int(target_idx)
    if t < 0 or t >= dag.n_nodes:
        raise ValueError(
            f"target_idx must be in [0, {dag.n_nodes - 1}], got {target_idx}"
        )

    B = dag.B
    if float(B[t]) >= 1.0:
        return 1.0
    if t == 0:
        return float(B[0])

    out_indptr = dag.out_indptr
    out_indices = dag.out_indices
    out_weights = dag.out_weights
    node_ids = dag.node_ids

    best_prob = np.zeros(t + 1, dtype=np.float64)
    best_prob[t] = 1.0
    next_v = np.full(t + 1, -1, dtype=np.int32)
    next_w = np.zeros(t + 1, dtype=np.float64)

    for u in range(t - 1, -1, -1):
        start = int(out_indptr[u])
        end = int(out_indptr[u + 1])
        best_u = 0.0
        chosen_v = -1
        chosen_w = 0.0
        chosen_id = ""

        for e in range(start, end):
            v = int(out_indices[e])
            if v > t:
                # out_indices rows are sorted ascending by topological index
                break
            bp_v = float(best_prob[v])
            if bp_v <= 0.0:
                continue
            w_uv = float(out_weights[e])
            if w_uv <= 0.0:
                continue
            cand = w_uv * bp_v
            if cand <= 0.0:
                continue
            if cand > best_u:
                best_u = cand
                chosen_v = v
                chosen_w = w_uv
                chosen_id = node_ids[v]
            elif cand == best_u:
                vid = node_ids[v]
                if vid < chosen_id:
                    chosen_v = v
                    chosen_w = w_uv
                    chosen_id = vid

        if chosen_v >= 0:
            best_prob[u] = best_u
            next_v[u] = chosen_v
            next_w[u] = chosen_w

    # Exact complement-product evaluation on the retained in-arborescence
    comp_prod = np.ones(t + 1, dtype=np.float64)
    for u in range(t):
        v = int(next_v[u])
        if v < 0:
            continue
        r_u = 1.0 - (1.0 - float(B[u])) * float(comp_prod[u])
        if r_u < 0.0:
            r_u = 0.0
        elif r_u > 1.0:
            r_u = 1.0
        q = float(next_w[u]) * r_u
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        comp_prod[v] *= 1.0 - q

    t_val = 1.0 - (1.0 - float(B[t])) * float(comp_prod[t])
    if t_val < 0.0:
        t_val = 0.0
    elif t_val > 1.0:
        t_val = 1.0
    return float(t_val)


def compute_all_tree_bounds(
    dag: PreparedDAG,
    targets: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute target-specific in-arborescence bounds ``T_j`` in topological order.

    When ``targets is None``, evaluates ``T_j`` for every node ``0 .. n_nodes - 1``.
    When ``targets`` is provided (either a boolean mask of length ``n_nodes`` or
    an iterable of integer topological indices), evaluates ``T_j`` only on those
    targets and fills unrequested positions with ``np.nan`` so the returned array
    always preserves topological index alignment.
    """
    n = dag.n_nodes
    if targets is None:
        out = np.empty(n, dtype=np.float64)
        for j in range(n):
            out[j] = compute_target_tree_bound(dag, j)
        return out

    arr = np.asarray(targets)
    if arr.dtype == np.bool_:
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"Boolean targets mask must have shape ({n},), got {arr.shape}"
            )
        target_indices = np.flatnonzero(arr)
    else:
        target_indices = np.asarray(arr, dtype=np.int64).ravel()

    out = np.full(n, np.nan, dtype=np.float64)
    for idx in target_indices:
        j = int(idx)
        out[j] = compute_target_tree_bound(dag, j)
    return out


def compute_abbe_lower(dag: PreparedDAG) -> np.ndarray:
    """Compute Abbe, Kulkarni, & Lee (2020) nodewise DAG lower bound (Sec. 4.1, Eqs. 21-23).

    Exact implementation of Definitions 11-12, Theorem 5, and Algorithm 4 from
    Abbe, Kulkarni, & Lee (2020, *JMLR* 21(18-112)):

    1. Introduce an always-active auxiliary seed ``aux`` (``LB(aux) = 1``,
       distance 0) with directed edge ``(aux, j)`` of weight ``B_j`` for each
       node ``j`` with ``B_j > 0``, keeping all original edges ``(u, v)`` with
       ``w_{uv} > 0``.
    2. Compute unweighted BFS shortest-path distance ``dist[v]`` from ``aux``.
       Unreachable nodes receive ``LB(v) = 0``.
    3. Construct the Minimum-Distance Directed Acyclic Subnetwork (MDAS,
       Definition 11): order reachable vertices ``V^0`` by ascending
       ``(dist[v], topo_idx[v])`` so ``dist[v_i] <= dist[v_j]`` for all ``i < j``,
       and retain edges ``E^0 = {(u, v) in E^+ : pos[u] < pos[v]}``. Original
       edges with ``dist[u] > dist[v]`` are deleted even though the input is a
       DAG; same-distance edges respecting topological order are retained as in
       Definition 11 and Section 4.3.
    4. Propagate Equation (22) via Algorithm 4 in ``V^0`` order (with ``aux``
       first):
       ``LB(v_k) = sum_{i=1}^{m*} B^0_{u_i, v_k} LB(u_i) (1 - sum_{j=1}^{i-1} B^0_{u_j, v_k})``
       where ``m* = max{m_0 <= m : sum_{j=1}^{m_0 - 1} B^0_{u_j, v_k} <= 1}``.

    Runs in ``O(|V| + |E|)`` time and guarantees ``0 <= LB(v) <= pi_v`` on every
    DAG under independent Bernoulli primitives.
    """
    n = dag.n_nodes
    lb = np.zeros(n, dtype=np.float64)
    if n == 0:
        return lb

    B = dag.B
    out_indptr = dag.out_indptr
    out_indices = dag.out_indices
    out_weights = dag.out_weights

    # 1. Unweighted BFS from auxiliary source `aux` over positive-weight edges
    dist = np.full(n, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for j in range(n):
        if float(B[j]) > 0.0:
            dist[j] = 1
            queue.append(j)

    max_dist = 1 if queue else 0
    while queue:
        u = queue.popleft()
        next_d = int(dist[u]) + 1
        start = int(out_indptr[u])
        end = int(out_indptr[u + 1])
        for e in range(start, end):
            if float(out_weights[e]) <= 0.0:
                continue
            v = int(out_indices[e])
            if dist[v] == -1:
                dist[v] = next_d
                if next_d > max_dist:
                    max_dist = next_d
                queue.append(v)

    if max_dist == 0:
        return lb

    # 2. Bucket-sort reachable nodes by (dist[v], topo_idx[v]) in O(|V|)
    buckets: list[list[int]] = [[] for _ in range(max_dist + 1)]
    for v in range(n):
        dv = int(dist[v])
        if dv > 0:
            buckets[dv].append(v)

    mdas_order: list[int] = []
    pos = np.full(n, -1, dtype=np.int32)
    rank = 1  # rank 0 is reserved for virtual seed `aux`
    for d in range(1, max_dist + 1):
        for v in buckets[d]:
            pos[v] = rank
            rank += 1
            mdas_order.append(v)

    # 3. Algorithm 4 / Equation (22) forward pass in MDAS order
    prefix_sum = np.zeros(n, dtype=np.float64)

    # Process virtual seed `aux` (rank 0, LB(aux) = 1.0) first
    for j in range(n):
        bj = float(B[j])
        if bj > 0.0:
            # First incoming edge at j has empty prior sum (1 - 0 = 1)
            lb[j] = bj
            prefix_sum[j] = bj

    for u in mdas_order:
        lb_u = float(lb[u])
        if lb_u < 0.0:
            lb_u = 0.0
        elif lb_u > 1.0:
            lb_u = 1.0
        lb[u] = lb_u

        pos_u = int(pos[u])
        start = int(out_indptr[u])
        end = int(out_indptr[u + 1])
        for e in range(start, end):
            w_uv = float(out_weights[e])
            if w_uv <= 0.0:
                continue
            v = int(out_indices[e])
            if int(pos[v]) <= pos_u:
                # Edge excluded from MDAS E^0 (dist[u] > dist[v])
                continue
            ps = float(prefix_sum[v])
            if ps < 1.0:
                lb[v] += w_uv * lb_u * (1.0 - ps)
                prefix_sum[v] = ps + w_uv

    np.clip(lb, 0.0, 1.0, out=lb)
    return lb


def compute_ancestry_diagnostic(dag: PreparedDAG) -> dict[str, Any]:
    """Prospective connected-component ancestry-label diagnostic (Section 3.7.1).

    Strictly restricts ancestry propagation and overlap-graph grouping to
    positive incoming contributions ``Q_{ij} = w_{ij} * R_i^{diag} > 0``:
    - A positive local term ``B_j > 0`` introduces node ``j``'s own label.
    - Each positive incoming contribution ``Q_{ij} > 0`` carries ``ancestry[i]``
      and forms a vertex in receiver ``j``'s overlap graph.
    - Vertices are joined when their label sets intersect; connected components
      (including transitive overlap) form groups, and only the largest
      contribution in each group is retained and combined with ``B_j`` by
      complement product.
    - ``ancestry[j]`` is the union of ``ancestry[i]`` over ``Q_{ij} > 0`` edges
      only, plus ``{j}`` if ``B_j > 0``.

    Returns
    -------
    dict with keys:
        - ``scores``: ``np.ndarray`` (float64) in topological order
        - ``max_lineage_size``: maximum ``|ancestry[j]|`` across all nodes
        - ``overlap_check_count``: total pairwise label-set intersection checks
        - ``overlap_receiver_fraction``: fraction of receiving candidates
          (``B_j == 0``; falls back to all nodes if no ``B_j == 0`` nodes exist)
          having at least one overlapping incoming pair
        - ``overlap_receiver_count``: count of ``B_j == 0`` receivers with overlap
        - ``overlap_node_count``: count of all nodes with overlap
        - ``overlap_node_fraction``: ``overlap_node_count / n_nodes``
    """
    n = dag.n_nodes
    scores = np.zeros(n, dtype=np.float64)
    B = dag.B
    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    in_weights = dag.in_weights

    ancestry: list[set[int]] = [set() for _ in range(n)]
    max_lineage_size = 0
    overlap_check_count = 0
    overlap_node_count = 0
    overlap_receiver_count = 0
    n_candidates = int(np.sum(B == 0.0))

    for j in range(n):
        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])

        pos_q: list[float] = []
        pos_labels: list[set[int]] = []
        for e in range(start, end):
            src = int(in_indices[e])
            q = float(in_weights[e] * scores[src])
            if q > 1.0:
                q = 1.0
            if q > 0.0:
                pos_q.append(q)
                pos_labels.append(ancestry[src])

        k = len(pos_q)
        comp_prod = 1.0 - float(B[j])

        if k == 1:
            comp_prod *= 1.0 - pos_q[0]
        elif k > 1:
            parent = list(range(k))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            has_overlap = False
            for a in range(k):
                la = pos_labels[a]
                for b in range(a + 1, k):
                    overlap_check_count += 1
                    if not la.isdisjoint(pos_labels[b]):
                        has_overlap = True
                        ra = _find(a)
                        rb = _find(b)
                        if ra != rb:
                            parent[rb] = ra

            if has_overlap:
                overlap_node_count += 1
                if float(B[j]) == 0.0:
                    overlap_receiver_count += 1

            group_max: dict[int, float] = {}
            for a in range(k):
                root = _find(a)
                qa = pos_q[a]
                prev = group_max.get(root)
                if prev is None or qa > prev:
                    group_max[root] = qa

            for retained_q in sorted(group_max.values()):
                comp_prod *= 1.0 - retained_q

        val = 1.0 - comp_prod
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        scores[j] = val

        target_labels: set[int] = set()
        for lbls in pos_labels:
            target_labels.update(lbls)
        if float(B[j]) > 0.0:
            target_labels.add(j)
        ancestry[j] = target_labels
        if len(target_labels) > max_lineage_size:
            max_lineage_size = len(target_labels)

    if n_candidates > 0:
        overlap_receiver_fraction = float(overlap_receiver_count / n_candidates)
    elif n > 0:
        overlap_receiver_fraction = float(overlap_node_count / n)
    else:
        overlap_receiver_fraction = 0.0

    overlap_node_fraction = float(overlap_node_count / n) if n > 0 else 0.0

    return {
        "scores": scores,
        "max_lineage_size": int(max_lineage_size),
        "overlap_check_count": int(overlap_check_count),
        "overlap_receiver_fraction": overlap_receiver_fraction,
        "overlap_receiver_count": int(overlap_receiver_count),
        "overlap_node_count": int(overlap_node_count),
        "overlap_node_fraction": overlap_node_fraction,
    }


def run_comparators(
    dag: PreparedDAG,
    include_trees: bool = True,
    include_abbe: bool = True,
    include_ancestry: bool = True,
    targets: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run all requested deterministic comparators on ``dag`` with per-method timings."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    local_only = compute_local_only(dag)
    timings["local_only"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gbbrpm = compute_gbbrpm_fast(dag)
    timings["gbbrpm"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    recursive_max = compute_recursive_max(dag)
    timings["recursive_max"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clipped_sum = compute_clipped_sum(dag)
    timings["clipped_sum"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    local_upstream = compute_local_upstream(dag)
    timings["local_upstream"] = time.perf_counter() - t0

    bracket_width = gbbrpm - recursive_max

    tree_bound: np.ndarray | None = None
    if include_trees:
        t0 = time.perf_counter()
        tree_bound = compute_all_tree_bounds(dag, targets=targets)
        timings["tree_bound"] = time.perf_counter() - t0

    abbe_lower: np.ndarray | None = None
    if include_abbe:
        t0 = time.perf_counter()
        abbe_lower = compute_abbe_lower(dag)
        timings["abbe_lower"] = time.perf_counter() - t0

    ancestry_scores: np.ndarray | None = None
    ancestry_meta: dict[str, Any] | None = None
    if include_ancestry:
        t0 = time.perf_counter()
        ancestry_meta = compute_ancestry_diagnostic(dag)
        ancestry_scores = ancestry_meta["scores"]
        timings["ancestry_diagnostic"] = time.perf_counter() - t0

    return {
        "local_only": local_only,
        "gbbrpm": gbbrpm,
        "recursive_max": recursive_max,
        "clipped_sum": clipped_sum,
        "local_upstream": local_upstream,
        "bracket_width": bracket_width,
        "tree_bound": tree_bound,
        "abbe_lower": abbe_lower,
        "ancestry_diagnostic": ancestry_scores,
        "ancestry_meta": ancestry_meta,
        "timings_sec": timings,
    }
