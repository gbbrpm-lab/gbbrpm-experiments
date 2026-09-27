#!/usr/bin/env python3
"""Run controlled GBBRPM scenarios on private Component 2 model tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from electrical_scenarios import write_electrical_experiments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=ROOT / "results" / "electrical" / "caltech_component2_private",
        help="Directory containing model_nodes.csv and model_edges.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <case-dir>/experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else case_dir / "experiments"
    )
    nodes_path = case_dir / "model_nodes.csv"
    edges_path = case_dir / "model_edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Expected model_nodes.csv and model_edges.csv under {case_dir}"
        )
    manifest_path = case_dir / "manifest.json"
    case_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    source_files = {"model_nodes.csv": nodes_path, "model_edges.csv": edges_path}
    if manifest_path.exists():
        source_files["manifest.json"] = manifest_path
    results = write_electrical_experiments(
        pd.read_csv(nodes_path),
        pd.read_csv(edges_path),
        output_dir,
        source_files=source_files,
        case_manifest=case_manifest,
    )
    manifest = results["manifest"]
    summary = results["summary"]
    strongest = summary.loc[summary["risk_sum"].idxmax()]
    print(
        f"Electrical scenarios completed: {manifest['scenario_count']} runs across "
        f"{len(manifest['family_counts'])} families."
    )
    print(
        "Largest network-wide risk: "
        f"{strongest['scenario_id']} (sum={strongest['risk_sum']:.6f}, "
        f"max={strongest['max_risk']:.6f})."
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
