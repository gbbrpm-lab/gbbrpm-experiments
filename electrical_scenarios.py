"""Controlled GBBRPM scenarios for a model-ready electrical tree.

Measured loading supplies edge susceptibility.  Local disturbance ``B`` and
uniform transmission ``tau`` are changed only by explicit scenario records.
The functions are intentionally independent of Caltech raw-data access.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import pandas as pd

from gbbrpm import evaluate_gbbrpm
from metrics import percentile_priority_set, ranked_items, summarize


ROOT_SEVERITIES = (0.25, 0.50, 0.75, 1.00)
LOAD_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)
TAU_LEVELS = (0.00, 0.25, 0.50, 0.75, 1.00)
CONTROLLED_B = 0.75
MULTI_SOURCE_B = 0.50
RISK_THRESHOLD = 0.05
PRIORITY_FRACTION = 0.20


@dataclass(frozen=True)
class ElectricalScenario:
    scenario_id: str
    family: str
    parameter: str
    disturbances: dict[str, float]
    load_scale: float = 1.0
    tau: float = 1.0
    interpretation: str = "controlled"


def validate_electrical_case(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    required_nodes = {"node", "B"}
    required_edges = {"source", "target", "L", "C", "tau"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"Missing node columns: {sorted(required_nodes - set(nodes.columns))}")
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"Missing edge columns: {sorted(required_edges - set(edges.columns))}")
    if nodes["node"].astype(str).duplicated().any():
        raise ValueError("Electrical model contains duplicate node identifiers")
    numeric = edges[["L", "C", "tau"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Electrical model edges contain missing L, C, or tau")
    if (numeric["L"] < 0).any() or (numeric["C"] <= 0).any():
        raise ValueError("Electrical model requires L >= 0 and C > 0")
    graph = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    graph.add_nodes_from(nodes["node"].astype(str))
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Electrical model must be a DAG")
    if nx.number_weakly_connected_components(graph) != 1:
        raise ValueError("Electrical model must contain one observable component")
    roots = [node for node in graph if graph.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Electrical scenario suite requires one root; found {roots}")
    return graph


def define_electrical_scenarios(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[list[ElectricalScenario], dict[str, object]]:
    graph = validate_electrical_case(nodes, edges)
    root = next(node for node in graph if graph.in_degree(node) == 0)
    topological_nodes = list(nx.lexicographical_topological_sort(graph, key=str))
    sinks = sorted(node for node in graph if graph.out_degree(node) == 0)
    multi_source_nodes = sorted(
        node for node in graph if graph.out_degree(node) > 0 and node != root
    )
    scenarios = [
        ElectricalScenario(
            "reference_zero_B",
            "reference",
            "B=0;load_scale=1;tau=1",
            {},
            interpretation="zero-disturbance mathematical reference",
        )
    ]

    for severity in ROOT_SEVERITIES:
        scenarios.append(
            ElectricalScenario(
                f"root_severity_{severity:.2f}".replace(".", "p"),
                "root_severity",
                f"{severity:.2f}",
                {root: severity},
                interpretation="controlled root disturbance severity",
            )
        )

    for node in topological_nodes:
        scenarios.append(
            ElectricalScenario(
                f"location_{node}",
                "disturbance_location",
                node,
                {node: CONTROLLED_B},
                interpretation="isolated controlled disturbance location",
            )
        )

    for scale in LOAD_SCALES:
        scenarios.append(
            ElectricalScenario(
                f"load_scale_{scale:.2f}".replace(".", "p"),
                "measured_load_scale",
                f"{scale:.2f}",
                {root: CONTROLLED_B},
                load_scale=scale,
                interpretation="counterfactual uniform scaling of measured edge load",
            )
        )

    for tau in TAU_LEVELS:
        scenarios.append(
            ElectricalScenario(
                f"tau_{tau:.2f}".replace(".", "p"),
                "uniform_transmission",
                f"{tau:.2f}",
                {root: CONTROLLED_B},
                tau=tau,
                interpretation="uniform transmission sensitivity; not calibrated",
            )
        )

    for left, right in combinations(multi_source_nodes, 2):
        scenarios.append(
            ElectricalScenario(
                f"multi_{left}_{right}",
                "multi_source",
                f"{left}+{right}",
                {left: MULTI_SOURCE_B, right: MULTI_SOURCE_B},
                interpretation=(
                    "concurrent internal disturbances; nested pairs test local-plus-incoming "
                    "aggregation and parallel pairs test branch coverage"
                ),
            )
        )
    if len(multi_source_nodes) > 2:
        scenarios.append(
            ElectricalScenario(
                "multi_all_internal",
                "multi_source",
                "+".join(multi_source_nodes),
                {node: MULTI_SOURCE_B for node in multi_source_nodes},
                interpretation="concurrent disturbance at every internal non-root node",
            )
        )

    metadata = {
        "root": root,
        "sinks": sinks,
        "multi_source_nodes": multi_source_nodes,
        "topological_nodes": topological_nodes,
        "graph_is_tree": nx.is_arborescence(graph),
        "scenario_count": len(scenarios),
        "family_counts": {
            family: sum(scenario.family == family for scenario in scenarios)
            for family in sorted({scenario.family for scenario in scenarios})
        },
    }
    return scenarios, metadata


def run_electrical_scenarios(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    risk_threshold: float = RISK_THRESHOLD,
    priority_fraction: float = PRIORITY_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    graph = validate_electrical_case(nodes, edges)
    scenarios, metadata = define_electrical_scenarios(nodes, edges)
    depth = nx.single_source_shortest_path_length(graph, metadata["root"])
    sink_set = set(metadata["sinks"])
    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    summary_rows: list[dict] = []

    for scenario in scenarios:
        scenario_nodes = nodes.copy()
        scenario_nodes["B"] = 0.0
        for node, severity in scenario.disturbances.items():
            if node not in set(scenario_nodes["node"].astype(str)):
                raise ValueError(f"Scenario {scenario.scenario_id} references unknown node {node}")
            scenario_nodes.loc[scenario_nodes["node"].astype(str) == node, "B"] = severity

        scenario_edges = edges.copy()
        scenario_edges["L"] = pd.to_numeric(scenario_edges["L"]) * scenario.load_scale
        scenario_edges["tau"] = scenario.tau
        risk, contributions = evaluate_gbbrpm(
            scenario_nodes[["node", "B"]],
            scenario_edges[["source", "target", "L", "C", "tau"]],
        )
        priority, nominal_k, cutoff = percentile_priority_set(risk, priority_fraction)
        ordered = ranked_items(risk)
        ordinal_rank = {node: index for index, (node, _) in enumerate(ordered, start=1)}
        dense_ranks = (
            pd.Series(risk, dtype=float).rank(method="dense", ascending=False).astype(int).to_dict()
        )
        common = {
            "scenario_id": scenario.scenario_id,
            "scenario_family": scenario.family,
            "parameter": scenario.parameter,
            "source_nodes": "|".join(sorted(scenario.disturbances)),
            "source_B": "|".join(
                f"{node}:{severity:g}" for node, severity in sorted(scenario.disturbances.items())
            ),
            "load_scale": scenario.load_scale,
            "uniform_tau": scenario.tau,
            "interpretation": scenario.interpretation,
        }
        b_lookup = dict(
            scenario_nodes[["node", "B"]].astype({"node": str}).itertuples(index=False, name=None)
        )
        for node, value in risk.items():
            node_rows.append(
                common
                | {
                    "node": node,
                    "depth": int(depth[node]),
                    "is_sink": node in sink_set,
                    "B": float(b_lookup[node]),
                    "R": float(value),
                    "ordinal_rank": ordinal_rank[node],
                    "dense_rank": dense_ranks[node],
                    "priority_top_20pct": node in priority,
                }
            )

        edge_metadata = scenario_edges.drop(columns=["S", "u"], errors="ignore")
        contributions = contributions.merge(
            edge_metadata,
            on=["source", "target", "L", "C", "tau"],
            how="left",
            validate="one_to_one",
        )
        for row in contributions.to_dict("records"):
            edge_rows.append(common | row)

        summary = summarize(
            risk,
            threshold=risk_threshold,
            priority_fraction=priority_fraction,
        )
        values = np.array(list(risk.values()), dtype=float)
        sink_values = np.array([risk[node] for node in metadata["sinks"]], dtype=float)
        if summary["max_risk"] == 0:
            summary["top_node"] = ""
        summary_rows.append(
            common
            | summary
            | {
                "risk_sum": float(values.sum()),
                "affected_fraction": float(np.mean(values > risk_threshold)),
                "sink_max_risk": float(sink_values.max()),
                "sink_mean_risk": float(sink_values.mean()),
                "sink_affected_count": int((sink_values > risk_threshold).sum()),
                "priority_k_nominal": nominal_k,
                "priority_count_with_ties": len(priority),
                "priority_cutoff": cutoff,
            }
        )

    manifest = {
        "suite_id": "caltech_component2_controlled_scenarios_v1",
        "scenario_count": len(scenarios),
        "family_counts": metadata["family_counts"],
        "root": metadata["root"],
        "sinks": metadata["sinks"],
        "multi_source_nodes": metadata["multi_source_nodes"],
        "graph_is_tree": metadata["graph_is_tree"],
        "root_severity_levels": list(ROOT_SEVERITIES),
        "location_severity": CONTROLLED_B,
        "load_scales": list(LOAD_SCALES),
        "uniform_tau_levels": list(TAU_LEVELS),
        "multi_source_severity": MULTI_SOURCE_B,
        "risk_threshold": risk_threshold,
        "priority_fraction": priority_fraction,
        "baseline_loading": "measured phasor-derived mean apparent power",
        "B_policy": "zero unless injected by a declared controlled scenario",
        "multi_source_boundary": (
            "Component 2 is an arborescence; nested internal-source combinations test "
            "bounded local-plus-incoming aggregation, while parallel combinations test "
            "concurrent branch coverage rather than downstream reconvergence"
        ),
        "validation_boundary": (
            "mechanism and sensitivity experiment on real topology/loading; "
            "not predictive event-response validation"
        ),
    }
    return (
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
        pd.DataFrame(summary_rows),
        manifest,
    )


def _tree_positions(graph: nx.DiGraph, root: str) -> dict[str, tuple[float, float]]:
    depth = nx.single_source_shortest_path_length(graph, root)
    levels: dict[int, list[str]] = {}
    for node, level in depth.items():
        levels.setdefault(level, []).append(node)
    positions = {}
    for level, level_nodes in sorted(levels.items()):
        ordered = sorted(level_nodes)
        count = len(ordered)
        for index, node in enumerate(ordered):
            x = 0.5 if count == 1 else index / (count - 1)
            positions[node] = (x, -float(level))
    return positions


def generate_electrical_figures(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    navy, blue, amber, slate, light = "#152238", "#2878B5", "#F0A202", "#667085", "#E9EEF5"

    graph = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    graph.add_nodes_from(nodes["node"].astype(str))
    root = next(node for node in graph if graph.in_degree(node) == 0)
    positions = _tree_positions(graph, root)
    edge_s = {(row.source, row.target): float(row.S) for row in edges.itertuples()}
    values = [edge_s[edge] for edge in graph.edges]
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color=values,
        edge_cmap=plt.cm.viridis,
        edge_vmin=0,
        edge_vmax=max(values),
        width=2.4,
        arrows=True,
        arrowsize=13,
        ax=ax,
    )
    sinks = [node for node in graph if graph.out_degree(node) == 0]
    ordinary = [node for node in graph if node not in sinks and node != root]
    nx.draw_networkx_nodes(graph, positions, nodelist=ordinary, node_color=light, edgecolors=navy, node_size=620, ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=sinks, node_color=amber, edgecolors=navy, node_size=680, ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=[root], node_color=blue, edgecolors=navy, node_size=760, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_weight="bold", ax=ax)
    edge_labels = {
        (row.source, row.target): f"{row.element}\nS={row.S:.3f}"
        for row in edges.itertuples()
    }
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=6.5, rotate=False, ax=ax)
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, max(values)), cmap=plt.cm.viridis)
    fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.02, label="Measured-load susceptibility S")
    ax.set_title("Caltech Component 2 observable topology and baseline susceptibility")
    ax.axis("off")
    created += _save_figure(fig, output_dir / "fig_electrical_component2_topology")

    created += _plot_sink_family(
        node_results,
        "root_severity",
        "Root disturbance severity B",
        "Node risk R",
        "Controlled root-severity response",
        output_dir / "fig_electrical_root_severity",
        numeric_parameter=True,
    )
    created += _plot_sink_family(
        node_results,
        "measured_load_scale",
        "Measured-load scale",
        "Node risk R",
        "Sensitivity to uniform scaling of measured loading",
        output_dir / "fig_electrical_load_scale",
        numeric_parameter=True,
    )
    created += _plot_sink_family(
        node_results,
        "uniform_transmission",
        "Uniform transmission factor τ",
        "Node risk R",
        "Uniform transmission sensitivity",
        output_dir / "fig_electrical_tau_sensitivity",
        numeric_parameter=True,
    )

    location = summary.loc[summary["scenario_family"] == "disturbance_location"].copy()
    location = location.sort_values("risk_sum", ascending=False)
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.bar(location["parameter"], location["risk_sum"], color=blue, edgecolor=navy, linewidth=0.6)
    ax.set_ylabel("Network-wide risk sum")
    ax.set_xlabel("Isolated disturbance node (B = 0.75)")
    ax.set_title("Effect of disturbance location on downstream modeled risk")
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=45)
    created += _save_figure(fig, output_dir / "fig_electrical_location_effect")
    return created


def _plot_sink_family(
    node_results: pd.DataFrame,
    family: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    numeric_parameter: bool,
) -> list[Path]:
    subset = node_results.loc[
        (node_results["scenario_family"] == family) & node_results["is_sink"]
    ].copy()
    if numeric_parameter:
        subset["parameter_numeric"] = pd.to_numeric(subset["parameter"])
        x_column = "parameter_numeric"
    else:
        x_column = "parameter"
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for node, rows in subset.groupby("node"):
        rows = rows.sort_values(x_column)
        ax.plot(rows[x_column], rows["R"], marker="o", linewidth=1.8, label=node)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    return _save_figure(fig, path)


def _save_figure(fig: plt.Figure, path: Path) -> list[Path]:
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_electrical_experiments(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
    *,
    source_files: dict[str, Path] | None = None,
    case_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    node_results, edge_results, summary, manifest = run_electrical_scenarios(nodes, edges)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_results.to_csv(output_dir / "scenario_node_risk.csv", index=False)
    edge_results.to_csv(output_dir / "scenario_edge_contributions.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    figures = generate_electrical_figures(
        nodes, edges, node_results, summary, output_dir / "figures"
    )
    manifest["figure_files"] = [path.relative_to(output_dir).as_posix() for path in figures]
    if source_files:
        manifest["input_sha256"] = {
            name: _sha256(path) for name, path in sorted(source_files.items())
        }
    if case_manifest:
        manifest["source_case"] = {
            key: case_manifest[key]
            for key in (
                "case_id",
                "snapshot_time",
                "probe_sha256",
                "topology_sha256",
                "residual_policy",
                "interpretation",
            )
            if key in case_manifest
        }
    with (output_dir / "scenario_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "node_results": node_results,
        "edge_results": edge_results,
        "summary": summary,
        "manifest": manifest,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
