"""Regression checks for the private-data-free Component 2 adapter."""

from pathlib import Path
import sys
import tempfile
import zipfile

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caltech_component2 import (  # noqa: E402
    EXPECTED_LOAD_METERS,
    ProbeSource,
    SOURCE_METER,
    align_phasor_channel,
    build_component2_case,
)
from gbbrpm import evaluate_gbbrpm  # noqa: E402


def write_synthetic_probe(root: Path) -> None:
    phasors = root / "phasors"
    phasors.mkdir(parents=True)
    timestamps = pd.date_range("2024-11-14T07:00:00", periods=3, freq="10s")
    pd.DataFrame({"t": timestamps}).to_csv(phasors / "t.csv", index=False)
    phase_angles = (0.0, -120.0, 120.0)
    meters = EXPECTED_LOAD_METERS + (SOURCE_METER, "egauge_19")
    for meter in meters:
        voltage = 9526.3 if meter in {SOURCE_METER, "egauge_19"} else 277.0
        current = 2.0 if meter == SOURCE_METER else 10.0
        if meter == "egauge_19":
            current = 8.0
        for phase_index, phase in enumerate(phase_angles, start=1):
            for prefix, rms in (("L", voltage), ("S", current)):
                pd.DataFrame(
                    {
                        "t": timestamps,
                        "frequency": [60.0] * len(timestamps),
                        "rms": [rms] * len(timestamps),
                        "phase_angle_harmonic_0": [phase] * len(timestamps),
                    }
                ).to_csv(phasors / f"{meter}-{prefix}{phase_index}.csv", index=False)


topology = ROOT / "data" / "electrical" / "socal28_sample" / "source_topology.zip"
with tempfile.TemporaryDirectory() as temporary:
    probe = Path(temporary) / "probe"
    write_synthetic_probe(probe)
    probe_zip = Path(temporary) / "probe.zip"
    with zipfile.ZipFile(probe_zip, "w") as archive:
        for item in sorted(candidate for candidate in probe.rglob("*") if candidate.is_file()):
            archive.write(item, Path("outer_folder") / item.relative_to(probe))
    assert ProbeSource(probe).fingerprint() == ProbeSource(probe_zip).fingerprint()
    case = build_component2_case(topology, probe)

assert len(case["topology_nodes"]) == 13
assert len(case["topology_edges"]) == 12
assert len(case["model_nodes"]) == 12
assert len(case["model_edges"]) == 11
assert case["manifest"]["unobserved_elements_excluded_from_model"] == ["line_407"]
assert case["manifest"]["residual_policy"] == "reported_separately_not_allocated"

model_edges = case["model_edges"]
assert model_edges[["L", "C", "S", "tau"]].notna().all().all()
assert model_edges["S"].between(0, 1).all()
assert (model_edges["C"] > 0).all()
assert float(model_edges.loc[model_edges["element"] == "tr_123", "C"].iloc[0]) == 750.0

graph = nx.from_pandas_edgelist(model_edges, "source", "target", create_using=nx.DiGraph)
assert nx.is_directed_acyclic_graph(graph)
assert nx.number_weakly_connected_components(graph) == 1

assignments = case["meter_assignments"].set_index("meter")
assert assignments.loc["egauge_21", "role"] == "component_source"
assert assignments.loc["egauge_19", "role"] == "excluded_neighbor_feeder"
assert assignments.loc["egauge_15", "data_status"] == "unavailable_incomplete_phasors"

scenario_nodes = case["model_nodes"][["node", "B"]].copy()
root = next(node for node in graph if graph.in_degree(node) == 0)
scenario_nodes.loc[scenario_nodes["node"] == root, "B"] = 0.5
risk, contributions = evaluate_gbbrpm(
    scenario_nodes,
    model_edges[["source", "target", "L", "C", "tau"]],
)
assert risk[root] == 0.5
assert len(contributions) == len(model_edges)
assert all(0 <= value <= 1 for value in risk.values())

offset_frame = pd.DataFrame(
    {
        "t": ["2024-11-14T07:00:00.200"],
        "frequency": [60.0],
        "rms": [1.0],
        "phase_angle_harmonic_0": [0.0],
    }
)
try:
    align_phasor_channel(
        offset_frame,
        np.array([np.datetime64("2024-11-14T07:00:00.000", "ns")]),
        max_offset_seconds=0.1,
    )
except ValueError as exc:
    assert "exceeds threshold" in str(exc)
else:
    raise AssertionError("Alignment threshold did not reject a 0.2-second offset")

print("Caltech Component 2 adapter checks passed.")
