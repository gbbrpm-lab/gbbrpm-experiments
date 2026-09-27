#!/usr/bin/env python3
"""Build the private Caltech Component 2 GBBRPM adapter outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caltech_component2 import build_component2_case, write_component2_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology-source",
        type=Path,
        required=True,
        help="Caltech sample_dataset ZIP or compatible topology source.",
    )
    parser.add_argument(
        "--probe-source",
        type=Path,
        required=True,
        help="Component 2 probe directory or ZIP (kept external to this repository).",
    )
    parser.add_argument(
        "--snapshot-time",
        default="2024-11-14T07:00:00",
    )
    parser.add_argument(
        "--max-alignment-offset-seconds",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "electrical" / "caltech_component2_private",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = build_component2_case(
        args.topology_source,
        args.probe_source,
        snapshot_time=args.snapshot_time,
        max_offset_seconds=args.max_alignment_offset_seconds,
    )
    write_component2_case(case, args.output_dir.resolve())
    manifest = case["manifest"]
    print(
        "Caltech Component 2 built: "
        f"{manifest['complete_topology_nodes']} nodes / "
        f"{manifest['complete_topology_edges']} edges complete; "
        f"{manifest['model_nodes']} nodes / {manifest['model_edges']} edges observable."
    )
    print(
        "Power reconciliation: "
        f"source={manifest['source_mean_p_kw']:.3f} kW, "
        f"metered loads={manifest['metered_load_mean_p_kw']:.3f} kW, "
        f"residual={manifest['residual_mean_p_kw']:.3f} kW."
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
