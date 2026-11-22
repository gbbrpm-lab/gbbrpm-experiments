#!/usr/bin/env python3
"""Reproducibly build and validate the DPWH CDO Bitan-ag-Kolambog drainage DAG.

Source authority:
    DPWH Regional Office X Freedom of Information Release FOI-REQ-2026-7383
    (approved Oct 13-22, 2026), specifically:
    4. Master Plan-Feasibility Study / PLANADES.zip
    ("Master Plan and Feasibility Study of Urban Drainage Improvement and
    Related Works in Cagayan de Oro City", UP PLANADES / DPWH Region X,
    Contract ID 15K00003, August 2016).

Governance notice:
    DPWH RO-X release letter (Oct 22, 2026) restricts use to the stated
    academic research purpose and prohibits redistribution/reproduction of raw
    agency files. Keep raw PDFs/ZIPs external to any public Git repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbbrpm.core import evaluate_gbbrpm  # noqa: E402

EXPECTED_PLANADES_SHA256 = (
    "d21769e8d0c8d2dd34f942ff2538685dfd2ba9c0ce3dc7ef8c62e6e36fe34c58"
)
DEFAULT_PLANADES_ZIP = (
    ROOT.parents[1]
    / "Dataset"
    / "DPWH_FOI_2026_7383"
    / "4_MasterPlan_Feasibility"
    / "PLANADES.zip"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "drainage" / "dpwh_cdo_bitanag_kolambog"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manning_rectangular_q(
    width_m: float,
    depth_m: float,
    slope: float,
    manning_n: float,
    closed_box: bool = False,
    barrels: int = 1,
) -> float:
    """Compute Manning's discharge Q (m^3/s) for a rectangular channel/box."""
    area = barrels * width_m * depth_m
    wetted_p = barrels * (2.0 * width_m + 2.0 * depth_m if closed_box else width_m + 2.0 * depth_m)
    hydraulic_r = area / wetted_p
    return (1.0 / manning_n) * area * (hydraulic_r ** (2.0 / 3.0)) * math.sqrt(slope)


def verify_planades_source_archive(planades_zip: Path) -> dict:
    """Verify PLANADES.zip SHA-256 and confirm verbatim source values in PDF text."""
    actual_sha = sha256_file(planades_zip)
    if actual_sha != EXPECTED_PLANADES_SHA256:
        raise ValueError(
            f"PLANADES.zip SHA-256 mismatch: expected {EXPECTED_PLANADES_SHA256}, "
            f"got {actual_sha}"
        )

    import pypdfium2 as pdfium  # local import so verification stays explicit

    with zipfile.ZipFile(planades_zip, "r") as zf:
        ch4_bytes = zf.read("PLANADES/CDO Chapter 4. Hydarulic Studies and Flood Analysis.pdf")
        ch5_bytes = zf.read("PLANADES/CDO Chapter 5. Drainage Master Plan.pdf")

    ch4_doc = pdfium.PdfDocument(ch4_bytes)
    ch4_text = " ".join(
        "\n".join(
            ch4_doc[i].get_textpage().get_text_range() for i in range(len(ch4_doc))
        ).split()
    )
    ch5_doc = pdfium.PdfDocument(ch5_bytes)
    ch5_text = " ".join(
        "\n".join(
            ch5_doc[i].get_textpage().get_text_range() for i in range(len(ch5_doc))
        ).split()
    )

    ch4_required_tokens = [
        "BCI-2C", "65.05", "77.43",
        "BCI-2A", "75.53", "111.27",
        "BCI-1B", "104.32", "146.12",
        "BCI-1A", "123.24", "181.59",
        "BCI-1T", "20.2", "29.22",
        "KCI-1", "29.42", "60.95",
        "KCI-2", "16.04", "30.55",
        "Upstream diversion", "14.91", "9.2",
    ]
    missing_ch4 = [tok for tok in ch4_required_tokens if tok not in ch4_text]
    if missing_ch4:
        raise AssertionError(f"Missing expected tokens in PLANADES Ch.4: {missing_ch4}")

    ch5_required_tokens = [
        "BDC-1",
        "20 to 40 cubic meters per second",
        "3.0m x 3.0m",
        "BDC-2",
        "Bolonsiri Road",
    ]
    missing_ch5 = [tok for tok in ch5_required_tokens if tok not in ch5_text]
    if missing_ch5:
        raise AssertionError(f"Missing expected tokens in PLANADES Ch.5: {missing_ch5}")

    return {
        "source_archive_name": planades_zip.name,
        "source_archive_sha256": actual_sha,
        "ch4_pages_verified": len(ch4_doc),
        "ch5_pages_verified": len(ch5_doc),
        "verbatim_tokens_verified": len(ch4_required_tokens) + len(ch5_required_tokens),
    }


def build_nodes_df() -> pd.DataFrame:
    """Return the 11 documented hydraulic junctions of Bitan-ag & Kolambog."""
    rows = [
        {
            "node": "J_BIT_HEAD",
            "node_name": "Upper Bitan-ag Headwater",
            "node_role": "headwater_source",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Indahag / Upper Camaman-an",
            "pcswmm_ref_node": "J64 (upstream reach)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.2 Sec.2.5.3 p.2-15; Ch.4 Table 4-6 p.4-11; Ch.5 Map 5-10 p.5-18",
        },
        {
            "node": "J_CAM_WEIR",
            "node_name": "Camaman-an Bolonsiri Diversion Weir",
            "node_role": "diversion_junction",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Camaman-an / Bolonsiri Rd",
            "pcswmm_ref_node": "J15 (diversion tap)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Table 4-8 p.4-17; Table 4-9 p.4-21; Ch.5 Sec.5.2.2.2 p.5-25; Map 5-18",
        },
        {
            "node": "J_LIMKETKAI",
            "node_name": "Mid-Bitan-ag / J.R. Borja-Limketkai Entry",
            "node_role": "intermediate_junction",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Brgy. 35 / Brgy. 31 / J.R. Borja Ext.",
            "pcswmm_ref_node": "J49 (mid-reach)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Table 4-8 p.4-17; Ch.5 Map 5-10 p.5-18; Map 5-11 p.5-19",
        },
        {
            "node": "J_AGORA_SPLIT",
            "node_name": "Agora Street Diversion Tap (Puregold / C.M. Recto)",
            "node_role": "diversion_junction",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Lapasan (Puregold) / Claro M. Recto Ave.",
            "pcswmm_ref_node": "J16 (CWSEL=3.49058m)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Fig.4-7 & Fig.4-8 p.4-19; Ch.5 Sec.5.2.2.2 p.5-23; Map 5-11 & 5-16",
        },
        {
            "node": "J_TRIB1_HEAD",
            "node_name": "Western CBD Tributary Headwater",
            "node_role": "headwater_source",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Brgy. 32 / Brgy. 30 / Brgy. 28 (Osmena-SM Downtown corridor)",
            "pcswmm_ref_node": "J48 (CWSEL=3.259308m)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Table 4-8 p.4-17; Fig.4-6 p.4-18; Ch.5 Map 5-13 p.5-21",
        },
        {
            "node": "J_TRIB1_CONF",
            "node_name": "Western Tributary-Bitan-ag Confluence",
            "node_role": "confluence_junction",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Brgy. 24 / West Limketkai",
            "pcswmm_ref_node": "J30 (CWSEL=3.151766m)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Fig.4-5 & Fig.4-6 p.4-18; Ch.5 Map 5-11 & Map 5-13",
        },
        {
            "node": "J_AGORA_CONF",
            "node_name": "Puntod-Lapasan Agora Reconvergence",
            "node_role": "reconvergence_junction",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Puntod-Lapasan (BCI-1A Sta. 0+300)",
            "pcswmm_ref_node": "J5 (H=1.952474m)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Fig.4-5 p.4-18; Fig.4-7 p.4-19; Ch.5 Map 5-11 p.5-19; Map 5-16 p.5-24",
        },
        {
            "node": "J_BIT_OUTFALL",
            "node_name": "Bitan-ag Creek Coastal Outfall",
            "node_role": "outfall_sink",
            "catchment": "Bitan-ag",
            "barangay_landmark": "Lapasan Coast / Macajalar Bay (Sta. 0+000)",
            "pcswmm_ref_node": "OF_BITANAG",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.2 Sec.2.5.3 p.2-15; Ch.5 Map 5-11 p.5-19",
        },
        {
            "node": "J_KOL_HEAD",
            "node_name": "Upper Kolambog Headwater",
            "node_role": "headwater_source",
            "catchment": "Kolambog",
            "barangay_landmark": "Upper Camaman-an",
            "pcswmm_ref_node": "KCI-2_UP",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.2 Sec.2.5.4 p.2-16; Ch.4 Table 4-6 p.4-12; Table 4-9 p.4-21",
        },
        {
            "node": "J_KOL_CONF",
            "node_name": "Kolambog-Bolonsiri Diversion Confluence",
            "node_role": "confluence_junction",
            "catchment": "Kolambog",
            "barangay_landmark": "Camaman-an / Lapasan (Bolonsiri Rd Junction)",
            "pcswmm_ref_node": "J13 (CWSEL=5.48024m)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Fig.4-9 & Fig.4-10 p.4-22; Ch.5 Map 5-14 p.5-22; Map 5-15 p.5-23",
        },
        {
            "node": "J_KOL_OUTFALL",
            "node_name": "Kolambog Creek Coastal Outfall",
            "node_role": "outfall_sink",
            "catchment": "Kolambog",
            "barangay_landmark": "Lapasan Coast / Macajalar Bay (Sta. 0+000)",
            "pcswmm_ref_node": "OF2 (CWSEL=1.11521m)",
            "B": 0.0,
            "B_record_origin": "baseline_zero_no_observed_blockage",
            "record_origin": "documented_topology",
            "source_citation": "PLANADES Ch.4 Fig.4-9 p.4-22; Ch.5 Map 5-14 p.5-22",
        },
    ]
    return pd.DataFrame(rows)


def build_edges_df() -> pd.DataFrame:
    """Return the 11 directed drainage reaches with C, L, tau (no dual S column)."""
    rows = [
        {
            "source": "J_BIT_HEAD",
            "target": "J_CAM_WEIR",
            "reach_code": "BCI-2C",
            "reach_name": "Upstream Bitan-ag Creek",
            "catchment": "Bitan-ag",
            "cross_section_type": "open_channel",
            "length_m": 933.0,
            "width_m": 6.0,
            "design_depth_m": 2.2,
            "total_depth_m": 3.0,
            "C": 77.43,
            "L": 65.05,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0100,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-6 p.4-11 (Q25=65.05); Table 4-8 p.4-17 (C=77.43); Map 5-10 (BCI-2C 488m + BCI-2D 445m)",
            "notes": "Table 4-8 rounds Runoff Q25 to 65 cms; exact Table 4-6 Q25=65.05 cms retained.",
        },
        {
            "source": "J_CAM_WEIR",
            "target": "J_LIMKETKAI",
            "reach_code": "BCI-2A",
            "reach_name": "Mid-Section Bitan-ag Creek",
            "catchment": "Bitan-ag",
            "cross_section_type": "open_channel",
            "length_m": 1210.0,
            "width_m": 8.0,
            "design_depth_m": 2.2,
            "total_depth_m": 3.0,
            "C": 111.27,
            "L": 75.53,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0100,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-6 p.4-12 (Q25=75.53); Table 4-8 p.4-17 (C=111.27); Map 5-10 (BCI-2B 596m + BCI-2A 614m)",
            "notes": "Table 4-8 rounds Runoff Q25 to 75 cms; exact Table 4-6 Q25=75.53 cms retained.",
        },
        {
            "source": "J_CAM_WEIR",
            "target": "J_KOL_CONF",
            "reach_code": "BDC-2",
            "reach_name": "Camaman-an Bolonsiri Diversion (Bitan-ag to Kolambog)",
            "catchment": "Bitan-ag-Kolambog Diversion",
            "cross_section_type": "rectangular_box_culvert",
            "length_m": 1419.0,
            "width_m": 2.4,
            "design_depth_m": 1.6,
            "total_depth_m": 2.2,
            "C": 14.91,
            "L": 9.20,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0100,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-9 p.4-21 (Upstream diversion C=14.91, Runoff=9.2); Ch.5 Table 5-3 p.5-17 (L=1419m); Map 5-18 p.5-25",
            "notes": "PCSWMM Fig.4-10 (p.4-22) shows peak diverted flow C40=12.24 cms; Table 4-9 value (9.20 cms) retained for consistency with Table 4-9 capacity (14.91 cms).",
        },
        {
            "source": "J_LIMKETKAI",
            "target": "J_AGORA_SPLIT",
            "reach_code": "BCI-1B_UP",
            "reach_name": "Central Bitan-ag Creek (Limketkai to Agora Tap)",
            "catchment": "Bitan-ag",
            "cross_section_type": "rectangular_open_channel",
            "length_m": 234.0,
            "width_m": 10.0,
            "design_depth_m": 2.2,
            "total_depth_m": 3.0,
            "C": 146.12,
            "L": 104.32,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0100,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-6 p.4-12 (Q25=104.32); Table 4-8 p.4-17 (C=146.12); Map 5-11 p.5-19 (Sta. 2+234 to 2+000)",
            "notes": "Corresponds to pre-diversion central reach entering J16 (PCSWMM Fig.4-8 C50=99.88 cms).",
        },
        {
            "source": "J_AGORA_SPLIT",
            "target": "J_TRIB1_CONF",
            "reach_code": "BCI-1B_POST",
            "reach_name": "Central Bitan-ag Post-Agora Reach (J16 to J30)",
            "catchment": "Bitan-ag",
            "cross_section_type": "rectangular_open_channel",
            "length_m": 411.0,
            "width_m": 10.0,
            "design_depth_m": 2.2,
            "total_depth_m": 3.0,
            "C": 146.12,
            "L": 67.781,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0100,
            "C_record_origin": "documented_table",
            "L_record_origin": "derived_pcswmm_mass_balance",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "derived_from_documented_dpwh_planades",
            "source_citation": "C=146.12 from PLANADES Ch.4 Table 4-8 p.4-17 (BCI-1B); L=99.880 (Fig.4-8 C50) - 32.099 (Fig.4-7 C26) = 67.781 cms; Map 5-11 (Sta. 2+000 to 1+589)",
            "notes": "Residual main-trunk flow between Agora split (J16) and Western Tributary confluence (J30); 67.781 + 20.20 = 87.98 cms matches Fig.4-5 C7=87.621 cms within 0.4%.",
        },
        {
            "source": "J_AGORA_SPLIT",
            "target": "J_AGORA_CONF",
            "reach_code": "BDC-1",
            "reach_name": "Agora Street Diversion Channel (Twin 3x3m RCBC)",
            "catchment": "Bitan-ag",
            "cross_section_type": "twin_rectangular_box_culvert",
            "length_m": 1142.0,
            "width_m": 6.0,
            "design_depth_m": 3.0,
            "total_depth_m": 3.0,
            "C": 40.00,
            "L": 33.885,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.015,
            "manning_slope_check": 0.0016,
            "C_record_origin": "documented_text_manning_verified",
            "L_record_origin": "documented_pcswmm_profile",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "C=40.00 cms from PLANADES Ch.5 Sec.5.2.2.2 p.5-23 (20-40 cms, twin 3.0x3.0m RCBC, L=1142m); L=33.885 cms from Ch.4 Fig.4-7 C17 p.4-19",
            "notes": "Manning check on twin 3.0x3.0m closed concrete RCBC (n=0.015, S0=0.0016 from Fig.4-7 bed slope 1.30m/825m) gives Q=39.62 cms, corroborating stated 40 cms capacity. Reconverges into BCI-1A at J5.",
        },
        {
            "source": "J_TRIB1_HEAD",
            "target": "J_TRIB1_CONF",
            "reach_code": "BCI-1T",
            "reach_name": "Bitan-ag Western CBD Tributary (BCI-1AB + BCI-1AA)",
            "catchment": "Bitan-ag",
            "cross_section_type": "rectangular_open_channel",
            "length_m": 933.0,
            "width_m": 6.0,
            "design_depth_m": 1.8,
            "total_depth_m": 2.4,
            "C": 29.22,
            "L": 20.20,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0025,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-6 p.4-12 (Q25=20.2); Table 4-8 p.4-17 (C=29.22); Map 5-13 p.5-21 (BCI-1AB 533m + BCI-1AA 400m)",
            "notes": "PCSWMM Fig.4-6 (p.4-18) shows C49=13.872 cms at J30 confluence; Rational Q25=20.20 cms from Table 4-6/4-8 retained.",
        },
        {
            "source": "J_TRIB1_CONF",
            "target": "J_AGORA_CONF",
            "reach_code": "BCI-1A_UP",
            "reach_name": "Downstream Bitan-ag Main Trunk (J30 to J5)",
            "catchment": "Bitan-ag",
            "cross_section_type": "rectangular_open_channel",
            "length_m": 1289.0,
            "width_m": 12.0,
            "design_depth_m": 2.2,
            "total_depth_m": 3.0,
            "C": 181.59,
            "L": 94.419,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0100,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_pcswmm_profile",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "C=181.59 from PLANADES Ch.4 Table 4-8 p.4-17 (BCI-1A); L=94.419 cms from Ch.4 Fig.4-5 C16 p.4-18; Map 5-11 (Sta. 1+589 to 0+300)",
            "notes": "Pre-reconvergence main trunk parallel to Agora diversion BDC-1; PCSWMM flows range 87.621 (C7) to 94.419 cms (C16).",
        },
        {
            "source": "J_AGORA_CONF",
            "target": "J_BIT_OUTFALL",
            "reach_code": "BCI-1A_OUT",
            "reach_name": "Bitan-ag Final Outfall Reach (Sta. 0+300 to Macajalar Bay)",
            "catchment": "Bitan-ag",
            "cross_section_type": "rectangular_open_channel",
            "length_m": 300.0,
            "width_m": 12.0,
            "design_depth_m": 2.2,
            "total_depth_m": 3.0,
            "C": 181.59,
            "L": 123.24,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0100,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-6 p.4-12 (Q25=123.24); Table 4-8 p.4-17 (C=181.59); Map 5-11 p.5-19 (Sta. 0+300 to 0+000)",
            "notes": "Post-reconvergence outfall reach; Rational Q25=123.24 cms agrees within 4% of PCSWMM sum 94.419+33.885=128.304 cms.",
        },
        {
            "source": "J_KOL_HEAD",
            "target": "J_KOL_CONF",
            "reach_code": "KCI-2",
            "reach_name": "Upstream Kolambog Creek",
            "catchment": "Kolambog",
            "cross_section_type": "open_channel",
            "length_m": 700.0,
            "width_m": 5.0,
            "design_depth_m": 2.2,
            "total_depth_m": 2.8,
            "C": 30.55,
            "L": 16.04,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0025,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-6 p.4-12 (KCI-2 Q25=16.04); Table 4-9 p.4-21 (C=30.55, Runoff=16.1); Sec.4.5.4 p.4-20 (700m upper pipe reach)",
            "notes": "Source PDF Table 4-9 row 2 contains clerical label typo 'Dagong Creek' with Kolambog KCI-2 Q25=16.1 cms (Table 4-6 KCI-2=16.04 cms); exact Table 4-6 Q25 retained.",
        },
        {
            "source": "J_KOL_CONF",
            "target": "J_KOL_OUTFALL",
            "reach_code": "KCI-1",
            "reach_name": "Downstream Kolambog Creek (KCI-1B + KCI-1A)",
            "catchment": "Kolambog",
            "cross_section_type": "rectangular_open_channel",
            "length_m": 1229.0,
            "width_m": 8.0,
            "design_depth_m": 2.2,
            "total_depth_m": 3.0,
            "C": 60.95,
            "L": 29.42,
            "tau": 1.0,
            "C_units": "m3/s",
            "L_units": "m3/s",
            "manning_n_check": 0.020,
            "manning_slope_check": 0.0030,
            "C_record_origin": "documented_table",
            "L_record_origin": "documented_table",
            "tau_record_origin": "neutral_baseline_assumed",
            "record_origin": "documented_dpwh_planades",
            "source_citation": "PLANADES Ch.4 Table 4-6 p.4-12 (KCI-1 Q25=29.42); Table 4-9 p.4-21 (C=60.95, Runoff=29.5); Map 5-14 p.5-22 (KCI-1B 269m + KCI-1A 960m)",
            "notes": "PCSWMM Fig.4-9 (p.4-22) shows 27.26-37.81 cms with diversion; Table 4-6 Q25=29.42 cms retained.",
        },
    ]
    return pd.DataFrame(rows)


def validate_manning_consistency(edges: pd.DataFrame) -> list[dict]:
    """Verify Manning's equation consistency against documented C values."""
    checks = []
    for row in edges.itertuples(index=False):
        if row.reach_code == "BDC-1":
            q_check = manning_rectangular_q(
                width_m=3.0,
                depth_m=3.0,
                slope=float(row.manning_slope_check),
                manning_n=float(row.manning_n_check),
                closed_box=True,
                barrels=2,
            )
            tol_pct = 1.5
        else:
            q_check = manning_rectangular_q(
                width_m=float(row.width_m),
                depth_m=float(row.design_depth_m),
                slope=float(row.manning_slope_check),
                manning_n=float(row.manning_n_check),
                closed_box=False,
                barrels=1,
            )
            tol_pct = 0.5
        rel_err_pct = abs(q_check - float(row.C)) / float(row.C) * 100.0
        if rel_err_pct > tol_pct:
            raise AssertionError(
                f"Manning verification failed for {row.reach_code}: "
                f"computed {q_check:.3f} vs documented {row.C:.3f} ({rel_err_pct:.2f}%)"
            )
        checks.append(
            {
                "reach_code": row.reach_code,
                "C_documented": float(row.C),
                "C_manning_check": round(q_check, 3),
                "rel_err_pct": round(rel_err_pct, 4),
            }
        )
    return checks


def validate_graph_and_engine(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Run all Gate A graph, contract, and GBBRPM engine verification checks."""
    # Strict schema & fail-closed checks
    if "S" in edges.columns:
        raise AssertionError("edges.csv must NOT contain dual S alongside C and L")
    for col in ["C", "L", "tau"]:
        vals = pd.to_numeric(edges[col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise AssertionError(f"Non-finite values in edge column {col}")
    if (edges["C"] <= 0).any() or (edges["L"] < 0).any():
        raise AssertionError("Invalid C <= 0 or L < 0")
    if ((edges["tau"] < 0) | (edges["tau"] > 1)).any():
        raise AssertionError("Invalid tau outside [0,1]")

    b_vals = pd.to_numeric(nodes["B"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(b_vals).all() or ((b_vals < 0) | (b_vals > 1)).any():
        raise AssertionError("Invalid B values")

    if nodes["record_origin"].isna().any() or edges["record_origin"].isna().any():
        raise AssertionError("100% record_origin provenance tagging required")

    # Graph checks
    G = nx.DiGraph()
    for n in nodes["node"]:
        G.add_node(str(n))
    for row in edges.itertuples(index=False):
        u, v = str(row.source), str(row.target)
        if u == v:
            raise AssertionError(f"Self-loop detected at {u}")
        if G.has_edge(u, v):
            raise AssertionError(f"Duplicate directed edge {u}->{v}")
        G.add_edge(u, v)

    if not nx.is_directed_acyclic_graph(G):
        raise AssertionError("Extracted drainage graph is NOT a DAG")
    if nx.number_of_selfloops(G) != 0:
        raise AssertionError("Self-loops found")
    if nx.number_weakly_connected_components(G) != 1:
        raise AssertionError("Expected a single weakly connected component")

    # Detect reconvergence pairs (nodes with >=2 vertex-disjoint or distinct directed paths from same ancestor)
    reconvergent_nodes = []
    for target in G.nodes():
        preds = list(G.predecessors(target))
        if len(preds) >= 2:
            pred_anc = [nx.ancestors(G, p) | {p} for p in preds]
            shared = set()
            for i in range(len(pred_anc)):
                for j in range(i + 1, len(pred_anc)):
                    shared |= pred_anc[i] & pred_anc[j]
            if shared:
                reconvergent_nodes.append({"node": target, "shared_ancestors": sorted(shared)})

    # Engine check 1: Baseline (B=0)
    risk_zero, contrib_zero = evaluate_gbbrpm(nodes, edges)
    if not all(v == 0.0 for v in risk_zero.values()):
        raise AssertionError("Baseline zero-disturbance run must yield all-zero R_j")

    # Engine check 2: Controlled Tier-2 smoke test (B=0.75 at J_BIT_HEAD)
    nodes_test = nodes.copy()
    nodes_test.loc[nodes_test["node"] == "J_BIT_HEAD", "B"] = 0.75
    risk_test, contrib_test = evaluate_gbbrpm(nodes_test, edges)
    if not all(0.0 <= v <= 1.0 for v in risk_test.values()):
        raise AssertionError("GBBRPM output out of [0,1]")

    # Verify both reconverging paths into J_AGORA_CONF carry positive Q
    in_agora = contrib_test[contrib_test["target"] == "J_AGORA_CONF"]
    if len(in_agora) != 2 or not (in_agora["Q"] > 0).all():
        raise AssertionError("Expected 2 positive incoming contributions at J_AGORA_CONF")

    u_vals = (edges["L"] / edges["C"]).to_numpy(dtype=float)
    s_vals = np.clip(u_vals, 0.0, 1.0)

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "dag": True,
        "self_loops": 0,
        "duplicate_edges": 0,
        "weak_components": nx.number_weakly_connected_components(G),
        "sources": sorted([n for n, d in G.in_degree() if d == 0]),
        "sinks": sorted([n for n, d in G.out_degree() if d == 0]),
        "branch_nodes": sorted([n for n, d in G.out_degree() if d >= 2]),
        "convergence_nodes": sorted([n for n, d in G.in_degree() if d >= 2]),
        "reconvergent_nodes": reconvergent_nodes,
        "susceptibility_min": round(float(s_vals.min()), 6),
        "susceptibility_max": round(float(s_vals.max()), 6),
        "susceptibility_mean": round(float(s_vals.mean()), 6),
        "saturated_edges_at_1x_load": int((u_vals >= 1.0).sum()),
        "smoke_test_J_BIT_HEAD_0.75": {k: round(float(v), 6) for k, v in risk_test.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planades-zip",
        type=Path,
        default=DEFAULT_PLANADES_ZIP,
        help="Path to external DPWH FOI PLANADES.zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write nodes.csv, edges.csv, and manifest.json.",
    )
    parser.add_argument(
        "--skip-source-zip-check",
        action="store_true",
        help="Skip verifying PLANADES.zip if running offline without raw FOI files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_verification = None
    if not args.skip_source_zip_check:
        if not args.planades_zip.exists():
            raise FileNotFoundError(
                f"Raw PLANADES.zip not found at {args.planades_zip}. "
                "Supply --planades-zip or --skip-source-zip-check."
            )
        source_verification = verify_planades_source_archive(args.planades_zip)

    nodes = build_nodes_df()
    edges = build_edges_df()
    manning_checks = validate_manning_consistency(edges)
    graph_stats = validate_graph_and_engine(nodes, edges)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    manifest_path = out_dir / "manifest.json"

    nodes.to_csv(nodes_path, index=False, lineterminator="\n")
    edges.to_csv(edges_path, index=False, lineterminator="\n")

    manifest = {
        "case_id": "drainage_dpwh_cdo_bitanag_kolambog_v1",
        "foi_tracking_number": "FOI-REQ-2026-7383",
        "foi_release_date": "2026-10-22",
        "foi_custodian": "DPWH Regional Office X / Cagayan de Oro City 2nd DEO",
        "primary_technical_source": (
            "UP PLANADES (Aug 20, 2016), Master Plan and Feasibility Study of "
            "Urban Drainage Improvement and Related Works in Cagayan de Oro City, "
            "Contract ID 15K00003 (PLANADES.zip)"
        ),
        "redistribution_policy": (
            "RESTRICTED ACADEMIC USE ONLY per DPWH RO-X Oct 22, 2026 release letter "
            "(modification, redistribution, and reproduction of raw agency files prohibited)."
        ),
        "gate_a_status": "PASS",
        "gate_b_status": "FAIL_EXPLORATORY_CONTEXT_ONLY",
        "gate_b_rationale": (
            "CDO 2ND DEO_FLOODING AREAS.pdf documents June 10, 2025 existing-channel "
            "flood hotspots (B01-B16, positive-only selection without non-flooded controls "
            "and without observed pre-event blockage B_i), whereas PLANADES models the "
            "2016 proposed Master Plan intervention network including diversions BDC-1/BDC-2."
        ),
        "B_policy": (
            "Tier-2 controlled scenario injections (B in {0.25, 0.50, 0.75, 1.00}, "
            "tagged simulated_scenario); baseline nodes.csv stores explicit B=0.0 because "
            "no node-level observed blockage percentage exists across the full DAG."
        ),
        "S_policy": "L/C adapter S_ij = min(1, L_ij / C_ij) from documented Q25/PCSWMM flow L and hydraulic capacity C; no dual S column stored.",
        "tau_protocol": [0.0, 0.25, 0.50, 0.75, 1.0],
        "edge_direction": "upstream_to_downstream_gravity_flow",
        "excluded_elements": [
            "BCI-3A/BCI-3B southern tributary (Map 5-10 geometry present, no documented Q25/PCSWMM discharge L in PLANADES Ch.4)",
            "BDC-1A1 vs BDC-1A2 split sub-barrels (no separate flow split documented; aggregated at BDC-1 diversion reach)",
            "BDC-2D eastern Camaman-an branch (no separate discharge L documented)",
            "As-Built Plan 22KD0016 (pure 1D linear RCBC along Claro M. Recto Ave without design L; logged as corroborating physical reference)",
            "Vol II/III Barangay Bugo ALA-08-DR..22-DR (disconnected street lines lacking per-conduit L and invert slopes)",
        ],
        "source_verification": source_verification,
        "manning_verification": manning_checks,
        "graph_verification": graph_stats,
        "output_hashes": {
            "nodes.csv": sha256_file(nodes_path),
            "edges.csv": sha256_file(edges_path),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Built {manifest['case_id']}: V={graph_stats['node_count']}, "
        f"E={graph_stats['edge_count']}, DAG={graph_stats['dag']}, "
        f"weak_comp={graph_stats['weak_components']}, "
        f"reconvergent={[r['node'] for r in graph_stats['reconvergent_nodes']]}, "
        f"S in [{graph_stats['susceptibility_min']:.4f}, {graph_stats['susceptibility_max']:.4f}]"
    )
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
