"""Build a model-ready Caltech Component 2 electrical case.

Raw Caltech measurements remain external to the repository.  This module
combines a frozen topology source with a user-supplied five-minute probe and
returns provenance-rich tables for the complete component and its observable
GBBRPM subgraph.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile

import networkx as nx
import numpy as np
import pandas as pd

from electrical_case import OPEN_STATUS, TopologySource, load_topology_source, status_at


COMPONENT_GRID_SOURCE = "gp_1"
SOURCE_METER = "egauge_21"
EXPECTED_LOAD_METERS = ("egauge_3", "egauge_7", "egauge_9", "egauge_11", "egauge_13")
KNOWN_UNAVAILABLE_METER = "egauge_15"
KNOWN_NEIGHBOR_METER = "egauge_19"
OPEN_CLOSE_TYPES = {"Switch", "SwitchMultiPosition", "CB", "Fuse", "VFI"}
TRANSFER_TYPES = ("Line", "Transformer", "Switch", "SwitchMultiPosition", "CB", "Fuse", "VFI")


@dataclass(frozen=True)
class ProbeSource:
    """Read probe members from either a directory or ZIP without extracting."""

    path: Path

    def members(self) -> list[str]:
        if self.path.is_dir():
            return sorted(
                item.relative_to(self.path).as_posix()
                for item in self.path.rglob("*")
                if item.is_file()
            )
        with zipfile.ZipFile(self.path) as archive:
            return sorted(name for name in archive.namelist() if not name.endswith("/"))

    def _resolve(self, suffix: str) -> str:
        normalized = suffix.replace("\\", "/").lstrip("/")
        matches = [
            member
            for member in self.members()
            if member == normalized or member.endswith("/" + normalized)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one probe member ending in {normalized!r}; found {matches}"
            )
        return matches[0]

    def exists(self, suffix: str) -> bool:
        try:
            self._resolve(suffix)
        except FileNotFoundError:
            return False
        return True

    def read_bytes(self, suffix: str) -> bytes:
        member = self._resolve(suffix)
        if self.path.is_dir():
            return (self.path / Path(member)).read_bytes()
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(member)

    def read_csv(self, suffix: str) -> pd.DataFrame:
        raw = self.read_bytes(suffix)
        if not raw.strip():
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(raw))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        logical_members = []
        for member in self.members():
            normalized = member.replace("\\", "/")
            for marker in ("magnitudes/", "phasors/"):
                marker_index = normalized.find(marker)
                if marker_index >= 0:
                    normalized = normalized[marker_index:]
                    break
            else:
                if normalized.endswith("/request.json"):
                    normalized = "request.json"
            logical_members.append((normalized, member))
        for normalized, member in sorted(logical_members):
            digest.update(normalized.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self.read_bytes(member))
            digest.update(b"\0")
        return digest.hexdigest()


class UnionFind:
    def __init__(self, members: list[str]):
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        parent = self.parent[member]
        if parent != member:
            self.parent[member] = self.find(parent)
        return self.parent[member]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def _status_for_port(
    source: TopologySource,
    element: dict,
    port_index: int,
    snapshot: pd.Timestamp,
) -> str:
    target = element["tbus"][port_index]
    raw = target.get("status", element.get("status"))
    filename = None
    if isinstance(raw, str) and raw.startswith("file:"):
        filename = raw.removeprefix("file:")
    else:
        suffix = f"-{port_index + 1}" if port_index else ""
        candidate = f"{element['name']}{suffix}-tbus_status.csv"
        if candidate in source.status_series:
            filename = candidate
    if filename:
        resolved = status_at(source.status_series.get(filename, []), snapshot)
        if resolved is not None:
            raw = resolved
    if raw is None:
        raw = target.get("nominal_status", element.get("nominal_status"))
    return str(raw or "UNSPECIFIED")


def _line_is_zero_impedance(element: dict) -> bool:
    rating = element.get("line_rating")
    return rating in (None, "None") or not element.get("length")


def _node_for_bus(bus_to_node: dict[str, str], bus: str) -> str:
    try:
        return bus_to_node[bus]
    except KeyError as exc:
        raise ValueError(f"Topology references unknown bus {bus}") from exc


def build_reduced_topology(
    topology_source: Path,
    snapshot_time: str = "2024-11-14T07:00:00",
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, str]]:
    """Reproduce the documented electrical reduction for the gp_1 component."""

    source = load_topology_source(topology_source)
    network = source.network
    snapshot = pd.Timestamp(snapshot_time)
    buses = {item["name"]: item for item in network.get("Bus", [])}
    union = UnionFind(sorted(buses))
    retained: list[tuple[str, dict, str, str, int]] = []

    for element_type in TRANSFER_TYPES:
        for element in network.get(element_type, []):
            fbus = element.get("fbus")
            if not fbus:
                continue
            for port_index, target in enumerate(element.get("tbus", [])):
                tbus = target.get("name")
                if not tbus:
                    continue
                status = _status_for_port(source, element, port_index, snapshot)
                if status == OPEN_STATUS:
                    continue
                collapsible = element_type in OPEN_CLOSE_TYPES or (
                    element_type == "Line" and _line_is_zero_impedance(element)
                )
                if collapsible:
                    union.union(fbus, tbus)
                else:
                    retained.append((element_type, element, fbus, tbus, port_index))

    grouped: dict[str, list[str]] = defaultdict(list)
    for bus in sorted(buses):
        grouped[union.find(bus)].append(bus)
    canonical = {
        root: "|".join(sorted(members)) for root, members in grouped.items()
    }
    bus_to_node = {
        bus: canonical[union.find(bus)] for bus in buses
    }

    raw_edges = []
    for element_type, element, fbus, tbus, port_index in retained:
        source_node = _node_for_bus(bus_to_node, fbus)
        target_node = _node_for_bus(bus_to_node, tbus)
        if source_node == target_node:
            continue
        raw_edges.append(
            {
                "source_key": source_node,
                "target_key": target_node,
                "element": element["name"],
                "element_type": element_type,
                "port_index": port_index + 1,
                "element_data": element,
            }
        )

    graph = nx.DiGraph()
    graph.add_edges_from((row["source_key"], row["target_key"]) for row in raw_edges)
    grid = next(
        item for item in network.get("GridPower", []) if item.get("name") == COMPONENT_GRID_SOURCE
    )
    grid_node = bus_to_node[grid["bus"]]
    component_members = nx.node_connected_component(graph.to_undirected(), grid_node)
    component_edges = [
        row
        for row in raw_edges
        if row["source_key"] in component_members and row["target_key"] in component_members
    ]
    component_graph = nx.DiGraph()
    component_graph.add_edges_from(
        (row["source_key"], row["target_key"]) for row in component_edges
    )
    if not nx.is_directed_acyclic_graph(component_graph):
        raise ValueError("Reduced Component 2 topology is not a DAG")
    if component_graph.in_degree(grid_node) != 0:
        raise ValueError("gp_1 does not resolve to a root of Component 2")

    ordered_keys = list(nx.lexicographical_topological_sort(component_graph, key=str))
    alias = {key: f"C2_N{index:02d}" for index, key in enumerate(ordered_keys, start=1)}
    node_rows = []
    for key in ordered_keys:
        members = key.split("|")
        nominal_values = sorted(
            {
                float(buses[member]["nominal_voltage"])
                for member in members
                if buses[member].get("nominal_voltage") is not None
            }
        )
        node_rows.append(
            {
                "node": alias[key],
                "B": 0.0,
                "topology_key": key,
                "bus_members": "|".join(members),
                "is_grid_root": key == grid_node,
                "nominal_voltage_values": "|".join(f"{value:g}" for value in nominal_values),
            }
        )

    edge_rows = []
    for row in component_edges:
        element = row.pop("element_data")
        capacity, basis, candidates = _capacity_kva(element, row["element_type"], buses)
        edge_rows.append(
            {
                "source": alias[row["source_key"]],
                "target": alias[row["target_key"]],
                "element": row["element"],
                "element_type": row["element_type"],
                "capacity_kva": capacity,
                "capacity_basis": basis,
                "capacity_candidates_kva": candidates,
                "meter_coverage": "unresolved",
            }
        )
    edges = pd.DataFrame(edge_rows).sort_values(["source", "target", "element"]).reset_index(drop=True)
    return pd.DataFrame(node_rows), edges, network, bus_to_node


def _capacity_kva(element: dict, element_type: str, buses: dict[str, dict]) -> tuple[float, str, str]:
    if element_type == "Transformer":
        candidates = [float(value) for value in element.get("kva", []) if value is not None]
        if not candidates:
            return np.nan, "missing_transformer_rating", ""
        return min(candidates), "minimum_declared_transformer_kva", "|".join(f"{x:g}" for x in candidates)
    if element_type == "Line":
        phase = (element.get("line_rating") or {}).get("phase", {})
        current = phase.get("I_rating")
        voltage = buses.get(element.get("fbus"), {}).get("nominal_voltage")
        if current is None or voltage is None:
            return np.nan, "missing_line_current_or_voltage", ""
        capacity = 3.0 * float(voltage) * float(current) / 1000.0
        return capacity, "3_times_phase_voltage_times_declared_current_rating", f"{capacity:g}"
    return np.nan, "unsupported_retained_element", ""


def _meter_voltage_bus(network: dict, meter_name: str) -> str | None:
    meter = next((item for item in network.get("EgaugeMeter", []) if item.get("name") == meter_name), None)
    if meter is None:
        return None
    for register in meter.get("registers", []):
        if register.get("name") in {"L1", "L2", "L3"} and register.get("element"):
            return str(register["element"]).split(".", 1)[0]
    return None


def _nearest_indices(sample_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sample_times, reference_times)
    positions = np.clip(positions, 0, len(sample_times) - 1)
    before = np.clip(positions - 1, 0, len(sample_times) - 1)
    choose_before = np.abs(reference_times - sample_times[before]) <= np.abs(
        sample_times[positions] - reference_times
    )
    return np.where(choose_before, before, positions)


def align_phasor_channel(
    frame: pd.DataFrame,
    reference_times: np.ndarray,
    *,
    max_offset_seconds: float = 0.1,
) -> pd.DataFrame:
    required = {"t", "frequency", "rms", "phase_angle_harmonic_0"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Phasor channel requires columns {sorted(required)}")
    clean = frame.copy()
    clean["t"] = pd.to_datetime(clean["t"], errors="coerce")
    for column in required - {"t"}:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=list(required)).sort_values("t").drop_duplicates("t")
    times = clean["t"].to_numpy(dtype="datetime64[ns]")
    indices = _nearest_indices(times, reference_times)
    selected = clean.iloc[indices].reset_index(drop=True)
    selected_times = selected["t"].to_numpy(dtype="datetime64[ns]")
    offsets = (reference_times - selected_times) / np.timedelta64(1, "s")
    if np.max(np.abs(offsets)) > max_offset_seconds:
        raise ValueError(
            f"Phasor alignment offset {np.max(np.abs(offsets)):.6f}s exceeds "
            f"threshold {max_offset_seconds:.6f}s"
        )
    phase = selected["phase_angle_harmonic_0"].to_numpy(float)
    frequency = selected["frequency"].to_numpy(float)
    corrected = (phase + 360.0 * frequency * offsets + 180.0) % 360.0 - 180.0
    return pd.DataFrame(
        {
            "t": pd.to_datetime(reference_times),
            "rms": selected["rms"].to_numpy(float),
            "phase": corrected,
            "alignment_offset_seconds": offsets,
        }
    )


def reconstruct_meter_power(
    probe_source: Path,
    meter_names: tuple[str, ...] = EXPECTED_LOAD_METERS + (SOURCE_METER, KNOWN_UNAVAILABLE_METER, KNOWN_NEIGHBOR_METER),
    *,
    max_offset_seconds: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned per-snapshot power and one-row-per-meter summaries."""

    probe = ProbeSource(probe_source)
    t_frame = probe.read_csv("phasors/t.csv")
    if "t" not in t_frame.columns:
        raise ValueError("Probe phasors/t.csv must contain column t")
    reference_times = pd.to_datetime(t_frame["t"], errors="coerce").dropna().to_numpy(dtype="datetime64[ns]")
    if len(reference_times) == 0:
        raise ValueError("Probe contains no common phasor timestamps")

    rows = []
    summaries = []
    for meter in meter_names:
        channel_names = [f"L{index}" for index in range(1, 4)] + [f"S{index}" for index in range(1, 4)]
        suffixes = {channel: f"phasors/{meter}-{channel}.csv" for channel in channel_names}
        populated = {
            channel: probe.exists(suffix) and bool(probe.read_bytes(suffix).strip())
            for channel, suffix in suffixes.items()
        }
        if not all(populated.values()):
            summaries.append(
                {
                    "meter": meter,
                    "data_status": "unavailable_incomplete_phasors",
                    "snapshot_count": 0,
                }
            )
            continue
        aligned = {
            channel: align_phasor_channel(
                probe.read_csv(suffix), reference_times, max_offset_seconds=max_offset_seconds
            )
            for channel, suffix in suffixes.items()
        }
        complex_channels = {
            channel: frame["rms"].to_numpy(float)
            * np.exp(1j * np.deg2rad(frame["phase"].to_numpy(float)))
            for channel, frame in aligned.items()
        }
        power = sum(
            complex_channels[f"L{index}"] * np.conj(complex_channels[f"S{index}"])
            for index in range(1, 4)
        ) / 1000.0
        voltage = np.column_stack([aligned[f"L{index}"]["rms"] for index in range(1, 4)])
        max_offset = max(
            float(np.max(np.abs(frame["alignment_offset_seconds"]))) for frame in aligned.values()
        )
        register_mean = register_bias = register_mape = register_correlation = np.nan
        register_suffix = f"magnitudes/{meter}-Mains_Power.csv"
        if probe.exists(register_suffix) and probe.read_bytes(register_suffix).strip():
            register = probe.read_csv(register_suffix)
            if {"t", "v"}.issubset(register.columns):
                register["t"] = pd.to_datetime(register["t"], errors="coerce")
                register["v"] = pd.to_numeric(register["v"], errors="coerce")
                register = register.dropna(subset=["t", "v"]).sort_values("t").drop_duplicates("t")
                if len(register):
                    register_times = register["t"].to_numpy(dtype="datetime64[ns]")
                    register_indices = _nearest_indices(register_times, reference_times)
                    register_values = register["v"].to_numpy(float)[register_indices]
                    register_mean = float(register_values.mean())
                    register_bias = float(np.mean(power.real - register_values))
                    register_mape = float(
                        np.mean(
                            np.abs(power.real - register_values)
                            / np.maximum(np.abs(register_values), 1e-9)
                        )
                        * 100.0
                    )
                    if np.std(power.real) > 0 and np.std(register_values) > 0:
                        register_correlation = float(np.corrcoef(power.real, register_values)[0, 1])
        for index, timestamp in enumerate(reference_times):
            rows.append(
                {
                    "t": pd.Timestamp(timestamp).isoformat(),
                    "meter": meter,
                    "p_kw": float(power[index].real),
                    "q_kvar": float(power[index].imag),
                    "s_kva": float(abs(power[index])),
                    "voltage_l1_v": float(voltage[index, 0]),
                    "voltage_l2_v": float(voltage[index, 1]),
                    "voltage_l3_v": float(voltage[index, 2]),
                }
            )
        summaries.append(
            {
                "meter": meter,
                "data_status": "available",
                "snapshot_count": len(reference_times),
                "mean_p_kw": float(power.real.mean()),
                "mean_q_kvar": float(power.imag.mean()),
                "mean_s_kva": float(np.abs(power).mean()),
                "mean_voltage_v": float(voltage.mean()),
                "maximum_alignment_offset_seconds": max_offset,
                "mains_power_matched_mean_raw": register_mean,
                "phasor_minus_mains_bias_kw_assuming_raw_is_kw": register_bias,
                "mains_mape_percent_assuming_raw_is_kw": register_mape,
                "mains_correlation": register_correlation,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def build_component2_case(
    topology_source: Path,
    probe_source: Path,
    *,
    snapshot_time: str = "2024-11-14T07:00:00",
    max_offset_seconds: float = 0.1,
) -> dict[str, object]:
    """Build full Component 2 tables and an observable GBBRPM input subgraph."""

    topology_source = Path(topology_source).resolve()
    probe_source = Path(probe_source).resolve()
    nodes, edges, network, bus_mapping = build_reduced_topology(topology_source, snapshot_time)
    power, meter_summary = reconstruct_meter_power(
        probe_source, max_offset_seconds=max_offset_seconds
    )
    available = set(meter_summary.loc[meter_summary["data_status"] == "available", "meter"])
    if SOURCE_METER not in available:
        raise ValueError(f"Required source meter {SOURCE_METER} is unavailable")
    missing_loads = sorted(set(EXPECTED_LOAD_METERS) - available)
    if missing_loads:
        raise ValueError(f"Required Component 2 load meters unavailable: {missing_loads}")

    topology_key_to_alias = dict(zip(nodes["topology_key"], nodes["node"]))
    assignments = []
    for meter in EXPECTED_LOAD_METERS + (SOURCE_METER, KNOWN_UNAVAILABLE_METER, KNOWN_NEIGHBOR_METER):
        bus = _meter_voltage_bus(network, meter)
        topology_key = bus_mapping.get(bus, "") if bus else ""
        node = topology_key_to_alias.get(topology_key, "")
        if meter == SOURCE_METER:
            role = "component_source"
        elif meter == KNOWN_NEIGHBOR_METER:
            role = "excluded_neighbor_feeder"
        elif meter == KNOWN_UNAVAILABLE_METER:
            role = "unavailable_redundant_or_auxiliary"
        elif node:
            role = "terminal_load"
        else:
            role = "outside_component"
        status_rows = meter_summary.loc[meter_summary["meter"] == meter, "data_status"]
        assignments.append(
            {
                "meter": meter,
                "role": role,
                "voltage_bus": bus or "",
                "node": node,
                "data_status": status_rows.iloc[0] if len(status_rows) else "not_requested",
            }
        )
    meter_assignments = pd.DataFrame(assignments)

    graph = nx.from_pandas_edgelist(edges, "source", "target", edge_attr=True, create_using=nx.DiGraph)
    meter_node = dict(
        meter_assignments.loc[meter_assignments["role"] == "terminal_load", ["meter", "node"]].itertuples(index=False, name=None)
    )
    node_meter = {node: meter for meter, node in meter_node.items()}
    source_series = power.loc[power["meter"] == SOURCE_METER].sort_values("t")
    leaf_series = {
        meter: power.loc[power["meter"] == meter].sort_values("t")
        for meter in EXPECTED_LOAD_METERS
    }
    source_complex = source_series["p_kw"].to_numpy() + 1j * source_series["q_kvar"].to_numpy()
    leaf_total = sum(
        frame["p_kw"].to_numpy() + 1j * frame["q_kvar"].to_numpy()
        for frame in leaf_series.values()
    )
    residual = source_complex - leaf_total

    edge_records = []
    unobserved_elements = []
    for row in edges.to_dict("records"):
        descendants = nx.descendants(graph, row["target"]) | {row["target"]}
        downstream_meters = sorted(node_meter[node] for node in descendants if node in node_meter)
        if row["element"] == "tr_117":
            load = float(source_series["s_kva"].mean())
            p_kw = float(source_complex.real.mean())
            q_kvar = float(source_complex.imag.mean())
            coverage = "direct_source_meter"
        elif downstream_meters:
            combined = sum(
                leaf_series[meter]["p_kw"].to_numpy()
                + 1j * leaf_series[meter]["q_kvar"].to_numpy()
                for meter in downstream_meters
            )
            load = float(np.abs(combined).mean())
            p_kw = float(combined.real.mean())
            q_kvar = float(combined.imag.mean())
            coverage = "downstream_meter_aggregation"
        else:
            load = p_kw = q_kvar = np.nan
            coverage = "unobserved_branch"
            unobserved_elements.append(row["element"])
        capacity = float(row["capacity_kva"])
        utilization = load / capacity if np.isfinite(load) and capacity > 0 else np.nan
        edge_records.append(
            row
            | {
                "downstream_meters": "|".join(downstream_meters),
                "meter_coverage": coverage,
                "mean_p_kw": p_kw,
                "mean_q_kvar": q_kvar,
                "L": load,
                "C": capacity,
                "u": utilization,
                "S": min(1.0, utilization) if np.isfinite(utilization) else np.nan,
                "tau": 1.0,
            }
        )
    full_edges = pd.DataFrame(edge_records)

    observed_edges = full_edges.loc[full_edges["meter_coverage"] != "unobserved_branch"].copy()
    observable_graph = nx.from_pandas_edgelist(observed_edges, "source", "target", create_using=nx.DiGraph)
    observed_nodes = set(observable_graph.nodes)
    model_nodes = nodes.loc[nodes["node"].isin(observed_nodes)].copy().reset_index(drop=True)
    model_edges = observed_edges.reset_index(drop=True)
    if model_edges[["L", "C", "S"]].isna().any().any():
        raise ValueError("Observable subgraph contains incomplete model inputs")
    if not nx.is_directed_acyclic_graph(observable_graph):
        raise ValueError("Observable Component 2 subgraph is not a DAG")

    meter_summary = meter_summary.merge(
        meter_assignments[["meter", "role", "node"]], on="meter", how="left"
    )
    nominal_by_meter = {}
    bus_index = {item["name"]: item for item in network.get("Bus", [])}
    for row in meter_assignments.itertuples(index=False):
        nominal_by_meter[row.meter] = bus_index.get(row.voltage_bus, {}).get("nominal_voltage", np.nan)
    meter_summary["nominal_voltage_v"] = meter_summary["meter"].map(nominal_by_meter)
    meter_summary["mean_voltage_pu"] = meter_summary["mean_voltage_v"] / meter_summary["nominal_voltage_v"]

    manifest = {
        "case_id": "caltech_component2_2024_11_14",
        "snapshot_time": snapshot_time,
        "component_grid_source": COMPONENT_GRID_SOURCE,
        "source_meter": SOURCE_METER,
        "load_meters": list(EXPECTED_LOAD_METERS),
        "excluded_neighbor_meter": KNOWN_NEIGHBOR_METER,
        "unavailable_meter": KNOWN_UNAVAILABLE_METER,
        "complete_topology_nodes": int(len(nodes)),
        "complete_topology_edges": int(len(full_edges)),
        "model_nodes": int(len(model_nodes)),
        "model_edges": int(len(model_edges)),
        "unobserved_elements_excluded_from_model": sorted(unobserved_elements),
        "phasor_snapshot_count": int(source_series.shape[0]),
        "maximum_alignment_offset_seconds_allowed": max_offset_seconds,
        "source_mean_p_kw": float(source_complex.real.mean()),
        "metered_load_mean_p_kw": float(leaf_total.real.mean()),
        "residual_mean_p_kw": float(residual.real.mean()),
        "residual_mean_q_kvar": float(residual.imag.mean()),
        "residual_mean_kva": float(np.abs(residual).mean()),
        "residual_policy": "reported_separately_not_allocated",
        "local_disturbance_policy": "B_initialized_to_zero_for_controlled_scenario_injection",
        "transmission_policy": "tau_initialized_to_one",
        "susceptibility_policy": "min(1, mean_apparent_power_kva/capacity_kva)",
        "transformer_capacity_policy": "minimum_declared_kva_when_multiple_values_exist",
        "line_capacity_policy": "3*phase_to_ground_nominal_voltage*declared_current_rating/1000",
        "direction_policy": "retained_topology_fbus_to_tbus_orientation",
        "mains_power_register_policy": "declared_W_but_raw_values_crosschecked_as_kW; phasor_power_is_model_input",
        "interpretation": "descriptive real-topology real-loading instantiation; not predictive event validation",
        "topology_sha256": _file_sha256(topology_source),
        "probe_sha256": ProbeSource(probe_source).fingerprint(),
    }
    return {
        "topology_nodes": nodes,
        "topology_edges": full_edges,
        "model_nodes": model_nodes,
        "model_edges": model_edges,
        "meter_assignments": meter_assignments,
        "meter_summary": meter_summary,
        "phasor_power_timeseries": power,
        "manifest": manifest,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def write_component2_case(case: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "topology_nodes",
        "topology_edges",
        "model_nodes",
        "model_edges",
        "meter_assignments",
        "meter_summary",
        "phasor_power_timeseries",
    ):
        frame = case[name]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(case["manifest"], handle, indent=2, sort_keys=True)
        handle.write("\n")
