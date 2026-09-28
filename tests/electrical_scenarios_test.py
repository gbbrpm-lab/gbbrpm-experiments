"""Regression checks for the controlled electrical scenario suite."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from electrical_scenarios import run_electrical_scenarios  # noqa: E402


nodes = pd.DataFrame(
    {
        "node": ["root", "a", "b", "c", "d", "e"],
        "B": [0.0] * 6,
    }
)
edges = pd.DataFrame(
    [
        {"source": "root", "target": "a", "element": "e1", "L": 30.0, "C": 100.0, "S": 0.3, "u": 0.3, "tau": 1.0},
        {"source": "a", "target": "b", "element": "e2", "L": 40.0, "C": 100.0, "S": 0.4, "u": 0.4, "tau": 1.0},
        {"source": "b", "target": "c", "element": "e3", "L": 50.0, "C": 100.0, "S": 0.5, "u": 0.5, "tau": 1.0},
        {"source": "root", "target": "d", "element": "e4", "L": 20.0, "C": 100.0, "S": 0.2, "u": 0.2, "tau": 1.0},
        {"source": "d", "target": "e", "element": "e5", "L": 25.0, "C": 100.0, "S": 0.25, "u": 0.25, "tau": 1.0},
    ]
)

node_results, edge_results, summary, manifest = run_electrical_scenarios(nodes, edges)

# Internal non-root nodes are a, b, and d: 3 pairs plus one all-internal case.
assert manifest["multi_source_nodes"] == ["a", "b", "d"]
assert manifest["family_counts"] == {
    "disturbance_location": 6,
    "measured_load_scale": 5,
    "multi_source": 4,
    "reference": 1,
    "root_severity": 4,
    "uniform_transmission": 5,
}
assert manifest["scenario_count"] == 25
assert len(node_results) == manifest["scenario_count"] * len(nodes)
assert len(edge_results) == manifest["scenario_count"] * len(edges)
assert node_results["R"].between(0, 1).all()
assert edge_results["Q"].between(0, 1).all()

reference = summary.set_index("scenario_id").loc["reference_zero_B"]
assert reference["risk_sum"] == 0.0
assert reference["affected_nodes"] == 0
assert reference["priority_count_with_ties"] == 0

severity = summary.loc[summary["scenario_family"] == "root_severity"].copy()
severity["x"] = pd.to_numeric(severity["parameter"])
severity = severity.sort_values("x")
assert severity["risk_sum"].is_monotonic_increasing

load = summary.loc[summary["scenario_family"] == "measured_load_scale"].copy()
load["x"] = pd.to_numeric(load["parameter"])
load = load.sort_values("x")
assert load["risk_sum"].is_monotonic_increasing

transmission = summary.loc[summary["scenario_family"] == "uniform_transmission"].copy()
transmission["x"] = pd.to_numeric(transmission["parameter"])
transmission = transmission.sort_values("x")
assert transmission["risk_sum"].is_monotonic_increasing

# The same controlled B=0.75, load scale=1, tau=1 must agree across families.
lookup = summary.set_index("scenario_id")
reference_risk = lookup.loc["root_severity_0p75", "risk_sum"]
assert np.isclose(lookup.loc["location_root", "risk_sum"], reference_risk)
assert np.isclose(lookup.loc["load_scale_1p00", "risk_sum"], reference_risk)
assert np.isclose(lookup.loc["tau_1p00", "risk_sum"], reference_risk)

# In the nested a+b case, b combines its local B=0.5 with incoming risk from a.
nested = node_results.loc[
    (node_results["scenario_id"] == "multi_a_b") & (node_results["node"] == "b")
].iloc[0]
assert nested["B"] == 0.5
assert nested["R"] > nested["B"]

terminal_locations = summary.loc[
    (summary["scenario_family"] == "disturbance_location")
    & summary["parameter"].isin(["c", "e"])
]
assert np.allclose(terminal_locations["risk_sum"], 0.75)

print("Controlled electrical scenario checks passed.")
