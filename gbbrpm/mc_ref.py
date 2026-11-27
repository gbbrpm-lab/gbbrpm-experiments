"""Vectorized uint64 bit-packed CSR Monte Carlo solver (Equations 3-10 and 3-11).

Also provides an exact scalar trial-by-trial reference solver consuming the
identical PCG64 primitive stream (and direct draw-evaluation entry points) to
verify < 1e-12 algebraic parity.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .csr_graph import PreparedDAG

try:  # Optional Numba acceleration when installed in the runtime environment
    from numba import njit as _njit  # type: ignore[import-not-found]

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    _njit = None
    _HAS_NUMBA = False

_PRIM_BLOCK_SIZE = 512


if _HAS_NUMBA:  # pragma: no cover - exercised only when numba is installed

    @_njit(cache=True)
    def _propagate_uint64_numba(
        n_nodes: int,
        n_words: int,
        B: np.ndarray,
        in_indptr: np.ndarray,
        in_indices: np.ndarray,
        in_weights: np.ndarray,
        n_slot: np.ndarray,
        e_slot: np.ndarray,
        zn_bits: np.ndarray,
        en_bits: np.ndarray,
        ones: np.ndarray,
    ) -> np.ndarray:
        state = np.zeros((n_nodes, n_words), dtype=np.uint64)
        for j in range(n_nodes):
            if B[j] >= 1.0:
                for w in range(n_words):
                    state[j, w] = ones[w]
                continue
            ns = n_slot[j]
            if ns >= 0:
                for w in range(n_words):
                    state[j, w] = zn_bits[ns, w]
            start = in_indptr[j]
            end = in_indptr[j + 1]
            for e in range(start, end):
                src = in_indices[e]
                es = e_slot[e]
                if es >= 0:
                    for w in range(n_words):
                        state[j, w] |= state[src, w] & en_bits[es, w]
                elif in_weights[e] >= 1.0:
                    for w in range(n_words):
                        state[j, w] |= state[src, w]
        return state


def hoeffding_half_width(
    n_nodes: int, n_trials: int, alpha: float = 0.05
) -> float:
    """Exact simultaneous Hoeffding half-width (Equation 3-11).

    ``h = sqrt(log(2 * |V| / alpha) / (2 * N))``
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be finite and in (0, 1), got {alpha}")
    return math.sqrt(math.log(2.0 * float(n_nodes) / float(alpha)) / (2.0 * float(n_trials)))


def _tail_mask_u64(chunk_trials: int) -> np.uint64:
    rem = chunk_trials % 64
    if rem == 0:
        return ~np.uint64(0)
    return np.uint64((1 << rem) - 1)


def _pack_bool_rows_to_u64(bool_rows: np.ndarray, chunk_trials: int) -> np.ndarray:
    """Pack a 2-D bool array ``(k, chunk_trials)`` into ``(k, ceil(chunk_trials/64))`` uint64.

    Unused trailing bits in the last word are strictly zeroed.
    """
    n_words = (chunk_trials + 63) // 64
    k = bool_rows.shape[0]
    if k == 0:
        return np.empty((0, n_words), dtype=np.uint64)

    packed_u8 = np.packbits(bool_rows, axis=1, bitorder="little")
    pad_bytes = n_words * 8 - packed_u8.shape[1]
    if pad_bytes:
        packed_u8 = np.pad(packed_u8, ((0, 0), (0, pad_bytes)), mode="constant")
    words = np.ascontiguousarray(packed_u8).view("<u8").astype(np.uint64, copy=False)
    tail_mask = _tail_mask_u64(chunk_trials)
    if chunk_trials % 64 != 0:
        words[:, -1] &= tail_mask
    return words


def _draw_packed_primitives(
    rng: np.random.Generator,
    probs: np.ndarray,
    chunk_trials: int,
    buf_flat: np.ndarray,
) -> np.ndarray:
    """Draw ``len(probs)`` independent Bernoulli rows of length ``chunk_trials`` and bit-pack."""
    k = int(probs.size)
    n_words = (chunk_trials + 63) // 64
    if k == 0:
        return np.empty((0, n_words), dtype=np.uint64)

    out_words = np.empty((k, n_words), dtype=np.uint64)
    block = _PRIM_BLOCK_SIZE
    for offset in range(0, k, block):
        end = min(offset + block, k)
        m = end - offset
        u_view = buf_flat[: m * chunk_trials].reshape(m, chunk_trials)
        rng.random(out=u_view)
        bool_block = u_view < probs[offset:end, None]
        out_words[offset:end] = _pack_bool_rows_to_u64(bool_block, chunk_trials)
    return out_words


def _propagate_uint64_numpy(
    dag: PreparedDAG,
    n_words: int,
    n_slot: np.ndarray,
    e_slot: np.ndarray,
    zn_bits: np.ndarray,
    en_bits: np.ndarray,
    ones: np.ndarray,
) -> np.ndarray:
    n_nodes = dag.n_nodes
    B = dag.B
    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    in_weights = dag.in_weights

    state = np.zeros((n_nodes, n_words), dtype=np.uint64)
    for j in range(n_nodes):
        ns = int(n_slot[j])
        if ns >= 0:
            sj = zn_bits[ns].copy()
        elif B[j] >= 1.0:
            state[j] = ones
            continue
        else:
            sj = state[j]

        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])
        for e in range(start, end):
            es = int(e_slot[e])
            if es >= 0:
                sj |= state[int(in_indices[e])] & en_bits[es]
            elif in_weights[e] >= 1.0:
                sj |= state[int(in_indices[e])]
        state[j] = sj
    return state


def evaluate_bitpacked_draws(
    dag: PreparedDAG, z_draws: np.ndarray, e_draws: np.ndarray
) -> np.ndarray:
    """Evaluate exact Eq. (3-10) activation counts via uint64 bit-packing on supplied draws.

    Parameters
    ----------
    dag:
        Prepared topological DAG.
    z_draws:
        Boolean array of shape ``(dag.n_nodes, n_trials)``.
    e_draws:
        Boolean array of shape ``(dag.n_edges, n_trials)`` ordered identically to
        ``dag.in_indices`` / ``dag.in_weights``.
    """
    z_arr = np.asarray(z_draws, dtype=np.bool_)
    e_arr = np.asarray(e_draws, dtype=np.bool_)
    if z_arr.ndim != 2 or z_arr.shape[0] != dag.n_nodes:
        raise ValueError(
            f"z_draws must have shape ({dag.n_nodes}, n_trials), got {z_arr.shape}"
        )
    n_trials = int(z_arr.shape[1])
    if e_arr.ndim != 2 or e_arr.shape != (dag.n_edges, n_trials):
        raise ValueError(
            f"e_draws must have shape ({dag.n_edges}, {n_trials}), got {e_arr.shape}"
        )
    if n_trials == 0:
        return np.zeros(dag.n_nodes, dtype=np.int64)

    n_words = (n_trials + 63) // 64
    z_bits = _pack_bool_rows_to_u64(z_arr, n_trials)
    e_bits = _pack_bool_rows_to_u64(e_arr, n_trials)

    state = np.zeros((dag.n_nodes, n_words), dtype=np.uint64)
    in_indptr = dag.in_indptr
    in_indices = dag.in_indices

    for j in range(dag.n_nodes):
        sj = z_bits[j].copy()
        start = int(in_indptr[j])
        end = int(in_indptr[j + 1])
        for e in range(start, end):
            sj |= state[int(in_indices[e])] & e_bits[e]
        state[j] = sj

    return np.bitwise_count(state).sum(axis=1, dtype=np.int64)


def evaluate_scalar_draws(
    dag: PreparedDAG, z_draws: np.ndarray, e_draws: np.ndarray
) -> np.ndarray:
    """Evaluate exact Eq. (3-10) activation counts via scalar trial-by-trial recursion.

    Parameters
    ----------
    dag:
        Prepared topological DAG.
    z_draws:
        Boolean array of shape ``(dag.n_nodes, n_trials)``.
    e_draws:
        Boolean array of shape ``(dag.n_edges, n_trials)`` ordered identically to
        ``dag.in_indices`` / ``dag.in_weights``.
    """
    z_arr = np.asarray(z_draws, dtype=np.bool_)
    e_arr = np.asarray(e_draws, dtype=np.bool_)
    if z_arr.ndim != 2 or z_arr.shape[0] != dag.n_nodes:
        raise ValueError(
            f"z_draws must have shape ({dag.n_nodes}, n_trials), got {z_arr.shape}"
        )
    n_trials = int(z_arr.shape[1])
    if e_arr.ndim != 2 or e_arr.shape != (dag.n_edges, n_trials):
        raise ValueError(
            f"e_draws must have shape ({dag.n_edges}, {n_trials}), got {e_arr.shape}"
        )

    counts = np.zeros(dag.n_nodes, dtype=np.int64)
    if n_trials == 0 or dag.n_nodes == 0:
        return counts

    in_indptr = dag.in_indptr
    in_indices = dag.in_indices
    n_nodes = dag.n_nodes

    for t in range(n_trials):
        x = [False] * n_nodes
        for j in range(n_nodes):
            val = bool(z_arr[j, t])
            if not val:
                start = int(in_indptr[j])
                end = int(in_indptr[j + 1])
                for e in range(start, end):
                    if x[int(in_indices[e])] and bool(e_arr[e, t]):
                        val = True
                        break
            x[j] = val
            if val:
                counts[j] += 1

    return counts


def _format_mc_result(
    dag: PreparedDAG,
    counts: np.ndarray,
    n_trials: int,
    seed: int,
    batch_size: int,
    alpha: float,
    elapsed_sec: float,
    backend: str,
) -> dict[str, Any]:
    pi_hat = counts.astype(np.float64) / float(n_trials)
    h = hoeffding_half_width(dag.n_nodes, n_trials, alpha=alpha)
    lower = np.clip(pi_hat - h, 0.0, 1.0)
    upper = np.clip(pi_hat + h, 0.0, 1.0)

    by_node = {
        dag.node_ids[j]: {
            "pi_hat": float(pi_hat[j]),
            "lower": float(lower[j]),
            "upper": float(upper[j]),
            "activation_count": int(counts[j]),
        }
        for j in range(dag.n_nodes)
    }

    return {
        "pi_hat": pi_hat,
        "lower": lower,
        "upper": upper,
        "hoeffding_h": float(h),
        "activation_counts": counts,
        "n_trials": int(n_trials),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "alpha": float(alpha),
        "elapsed_sec": float(elapsed_sec),
        "backend": backend,
        "by_node": by_node,
    }


def run_mc_reference(
    dag: PreparedDAG,
    n_trials: int = 100_000,
    seed: int = 41,
    batch_size: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run vectorized uint64 bit-packed Monte Carlo reference on a ``PreparedDAG``.

    Uses deterministic ``PCG64(seed)`` in memory-bounded batches of ``batch_size``
    trials, packing 64 independent Bernoulli trials per ``uint64`` register word
    and masking unused trailing bits in the final word of each batch.
    """
    if dag.n_nodes < 1:
        raise ValueError("DAG must contain at least one node")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    # Validate alpha eagerly
    hoeffding_half_width(dag.n_nodes, n_trials, alpha=alpha)

    t0 = time.perf_counter()

    nondeg_nodes = np.flatnonzero((dag.B > 0.0) & (dag.B < 1.0))
    nondeg_edges = np.flatnonzero((dag.in_weights > 0.0) & (dag.in_weights < 1.0))

    n_slot = np.full(dag.n_nodes, -1, dtype=np.int32)
    n_slot[nondeg_nodes] = np.arange(nondeg_nodes.size, dtype=np.int32)
    e_slot = np.full(dag.n_edges, -1, dtype=np.int32)
    e_slot[nondeg_edges] = np.arange(nondeg_edges.size, dtype=np.int32)

    node_probs = np.ascontiguousarray(dag.B[nondeg_nodes], dtype=np.float64)
    edge_probs = np.ascontiguousarray(dag.in_weights[nondeg_edges], dtype=np.float64)

    effective_batch = min(int(batch_size), int(n_trials))
    max_block_prims = min(
        _PRIM_BLOCK_SIZE,
        max(int(nondeg_nodes.size), int(nondeg_edges.size), 1),
    )
    buf_flat = np.empty(max_block_prims * effective_batch, dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = np.zeros(dag.n_nodes, dtype=np.int64)
    use_numba = _HAS_NUMBA

    for start in range(0, int(n_trials), effective_batch):
        chunk = min(effective_batch, int(n_trials) - start)
        n_words = (chunk + 63) // 64
        tail_mask = _tail_mask_u64(chunk)

        ones = np.full(n_words, ~np.uint64(0), dtype=np.uint64)
        ones[-1] = tail_mask

        zn_bits = _draw_packed_primitives(rng, node_probs, chunk, buf_flat)
        en_bits = _draw_packed_primitives(rng, edge_probs, chunk, buf_flat)

        if use_numba:  # pragma: no cover - optional dependency
            state = _propagate_uint64_numba(
                dag.n_nodes,
                n_words,
                dag.B,
                dag.in_indptr,
                dag.in_indices,
                dag.in_weights,
                n_slot,
                e_slot,
                zn_bits,
                en_bits,
                ones,
            )
        else:
            state = _propagate_uint64_numpy(
                dag,
                n_words,
                n_slot,
                e_slot,
                zn_bits,
                en_bits,
                ones,
            )

        counts += np.bitwise_count(state).sum(axis=1, dtype=np.int64)

    elapsed = time.perf_counter() - t0
    backend = "numba_uint64_bitpacked" if use_numba else "numpy_uint64_bitpacked"
    return _format_mc_result(
        dag=dag,
        counts=counts,
        n_trials=int(n_trials),
        seed=int(seed),
        batch_size=int(batch_size),
        alpha=float(alpha),
        elapsed_sec=elapsed,
        backend=backend,
    )


def run_mc_scalar_reference(
    dag: PreparedDAG,
    n_trials: int = 10_000,
    seed: int = 41,
    batch_size: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Literal scalar trial-by-trial Monte Carlo reference sharing exact RNG draws.

    Consumes the identical ``PCG64(seed)`` uniform draw sequence as
    ``run_mc_reference`` (batch by batch, non-degenerate nodes in topological
    order followed by non-degenerate edges in predecessor CSR order), enabling
    exact ``< 1e-12`` parity checks against the uint64 bit-packed kernel.
    """
    if dag.n_nodes < 1:
        raise ValueError("DAG must contain at least one node")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    hoeffding_half_width(dag.n_nodes, n_trials, alpha=alpha)

    t0 = time.perf_counter()

    nondeg_nodes = np.flatnonzero((dag.B > 0.0) & (dag.B < 1.0))
    nondeg_edges = np.flatnonzero((dag.in_weights > 0.0) & (dag.in_weights < 1.0))
    node_probs = dag.B[nondeg_nodes]
    edge_probs = dag.in_weights[nondeg_edges]

    effective_batch = min(int(batch_size), int(n_trials))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = np.zeros(dag.n_nodes, dtype=np.int64)

    for start in range(0, int(n_trials), effective_batch):
        chunk = min(effective_batch, int(n_trials) - start)

        z_draws = np.zeros((dag.n_nodes, chunk), dtype=np.bool_)
        z_draws[dag.B >= 1.0, :] = True
        if nondeg_nodes.size > 0:
            for offset in range(0, int(nondeg_nodes.size), _PRIM_BLOCK_SIZE):
                end = min(offset + _PRIM_BLOCK_SIZE, int(nondeg_nodes.size))
                m = end - offset
                u = rng.random((m, chunk))
                z_draws[nondeg_nodes[offset:end]] = u < node_probs[offset:end, None]

        e_draws = np.zeros((dag.n_edges, chunk), dtype=np.bool_)
        e_draws[dag.in_weights >= 1.0, :] = True
        if nondeg_edges.size > 0:
            for offset in range(0, int(nondeg_edges.size), _PRIM_BLOCK_SIZE):
                end = min(offset + _PRIM_BLOCK_SIZE, int(nondeg_edges.size))
                m = end - offset
                u = rng.random((m, chunk))
                e_draws[nondeg_edges[offset:end]] = u < edge_probs[offset:end, None]

        counts += evaluate_scalar_draws(dag, z_draws, e_draws)

    elapsed = time.perf_counter() - t0
    return _format_mc_result(
        dag=dag,
        counts=counts,
        n_trials=int(n_trials),
        seed=int(seed),
        batch_size=int(batch_size),
        alpha=float(alpha),
        elapsed_sec=elapsed,
        backend="python_scalar_reference",
    )
