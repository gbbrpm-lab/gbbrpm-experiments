# GBBRPM v1 Experiment Package

Reproducible experiment pipeline for the **Graph-Based Blockage Risk
Propagation Model (GBBRPM) v1**.

The package runs baseline evaluation, controlled scenario sweeps,
comparators, robustness and property tests, scalability experiments, and a
separate SWMM external-reference workflow. Reporting retains the complete
ranking and uses a tie-aware top-20% priority set rather than a fixed top
three.

## Contents

- [Model](#model)
- [Data and provenance](#data-and-provenance)
- [Setup](#setup)
- [Running experiments](#running-experiments)
- [Controlled scenario suite](#controlled-scenario-suite)
- [Outputs](#outputs)
- [SWMM external reference](#swmm-external-reference)
- [Software dependency case](#software-dependency-case)
- [Electrical distribution case](#electrical-distribution-case)
- [Reproducibility notes](#reproducibility-notes)
- [Repository structure](#repository-structure)
- [Scope and limitations](#scope-and-limitations)

## Model

GBBRPM v1 requires a directed acyclic graph (DAG) and evaluates nodes in
topological order. For a directed edge `i -> j`:

$$
Q_{ij} = S_{ij}\tau_{ij}R_i
$$

where:

- `R_i` is the current propagated risk at the upstream node.
- `S_ij` is the edge susceptibility.
- `tau_ij` is the transmission factor.
- `Q_ij` is the incoming risk contribution.

For the baseline drainage instantiation:

$$
S_{ij} = \min\left(1,\frac{L_{ij}}{C_{ij}}\right),
\qquad \tau_{ij}=1
$$

The receiving-node update is:

$$
R_j = 1 - (1-B_j)\prod_{i \in P(j)}(1-Q_{ij})
$$

`B_j` is the local blockage or disturbance severity. `R` is a bounded,
normalized continuous **risk index**, not a probability.

## Data and provenance

The repository preserves three distinct data groups. The default data source
is `historical`.

| Data source | Location | Role |
| --- | --- | --- |
| Historical validation inputs | `data/historical/` | Reconstructed inputs used for the preserved historical validation behavior. |
| Recovered historical candidate | `data/recovered_candidate/` | Alternate historical artifact retained for comparison and provenance analysis. |
| Reconstructed package fixtures | `data/reconstructed/` | Fallback fixtures retained for provenance and comparison. |

### Historical validation inputs

The historical files are:

- `validation_reconstructed_edges.csv`
- `validation_reconstructed_nodes.csv`

They have been computationally verified against the preserved historical
validation behavior. With the current frozen implementation, they reproduce:

- N5 outlet risk near `0.70864`.
- The N5 ranking `N13 > N19 > N16`.
- Historical comparator behavior.
- The restored 136-scenario controlled experiment suite.

These should be described as **historical reconstructed validation inputs
that reproduce the preserved historical validation results**. They should
not be represented as the lost original raw edge-by-edge historical dataset
unless stronger provenance evidence is recovered.

### Recovered historical candidate

The files under `data/recovered_candidate/` contain combined N1-N5 node,
edge, and network-summary tables recovered from older thesis files. They
produce different numerical results from the reconstructed historical
validation inputs, so they are retained as an alternate dataset rather than
the default thesis reproduction source.

### Selecting a data source

Use `--data-source historical` for historical validation reproduction or
`--data-source recovered` for the alternate dataset. The default is
`historical`.

## Setup

### Windows / PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python run_experiments.py --data-source historical
```

### WSL / Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python run_experiments.py --data-source historical
```

The editable installation exposes the domain-agnostic calculation engine to
other applications while keeping the frozen experiment data and publication
workflow separate:

```python
from gbbrpm import derive_susceptibility, evaluate_gbbrpm
```

The public package accepts pandas node and edge tables. Nodes require `node`
and `B`; edges require `source`, `target`, and either an explicit `S` or both
`L` and `C`. The optional `tau` column defaults to `1.0`. Repository-specific
dataset loading remains available through `model.load_network` for backwards
compatibility with the experiment scripts.

## Running experiments

Run the full historical validation suite:

```bash
python run_experiments.py --data-source historical
```

Run only the controlled scenario suite:

```bash
python run_experiments.py --suite core --data-source historical
```

Run the historical suite with 600 robustness/property trials:

```bash
python run_experiments.py --data-source historical --trials 600 --seed 41
```

The scalability block uses 30 timed repetitions and 5 warm-up runs by
default. These can be changed explicitly when a quicker development check is
needed:

```bash
python run_experiments.py \
  --data-source historical \
  --scalability-repeats 5 \
  --scalability-warmups 1
```

Run selected networks only:

```bash
python run_experiments.py \
  --suite core \
  --data-source historical \
  --networks N3 N5
```

Use the recovered alternate dataset:

```bash
python run_experiments.py --data-source recovered
```

Inspect saved outputs and run the smoke test:

```bash
python inspect_results.py
python tests/package_api_test.py
python tests/smoke_test.py
python tests/reproducibility_test.py
python tests/generic_properties_test.py
python tests/software_case_test.py
python tests/electrical_case_test.py
python tests/caltech_component2_test.py
python tests/electrical_scenarios_test.py
python tests/paper_outputs_test.py
python tests/swmm_reference_test.py
```

## Controlled scenario suite

The expanded core suite contains 161 scenario runs. The first 136 preserve
the restored historical protocol; the final 25 are the post-consultation
uniform-transmission sensitivity cases.

| Scenario family | Runs | Protocol |
| --- | ---: | --- |
| Source severity | 20 | Four severity levels across five networks. |
| L/C stress | 25 | Five load-scaling levels across five networks. |
| Blockage location | 36 | Selected blockage locations from the historical protocol. |
| Source combination | 10 | Multi-source configurations for N3 and N5. |
| Intermediate blockage | 45 | Three selected nodes, three severities, across five networks. |
| Uniform transmission | 25 | Five transmission levels across five networks. |
| **Total** | **161** | **136 historical cases plus 25 transmission cases.** |

### Sweep levels

- **Source severity:** `0.25`, `0.50`, `0.75`, `1.00`.
- **L/C stress:** `0.50`, `0.75`, `1.00`, `1.25`, `1.50`.
- **Intermediate blockage:** `0.25`, `0.50`, `0.75`.
- **Uniform transmission:** `0.00`, `0.25`, `0.50`, `0.75`, `1.00`.

The blockage-location sweep evaluates selected locations rather than every
node in every network.

### Additional validation blocks

The full suite also includes:

- N1-N5 baseline evaluation.
- Local-only and uniform-susceptibility comparators.
- Perturbation robustness at +/-5%, +/-10%, and +/-20%.
- Randomized boundedness and monotonicity tests.
- `B = 1 -> R = 1` boundary checks.
- `S = 0` edge-gating checks.
- `tau = 0` edge-gating checks.
- Topological-order and graph-relabeling invariance checks.
- Zero-state consistency and explicit cycle/endpoint rejection.
- Deterministic tie handling and 10%, 20%, and 25% priority-set checks.
- Sparse-DAG scalability timing.

### Preserved validation behavior

The historical input configuration reproduces the preserved N5 comparator
behavior:

```text
local_only       ~=  0.1086
uniform_S_0.25  ~= -0.0421
uniform_S_0.50  ~=  0.1504
uniform_S_0.75  ~=  0.6962
```

The corresponding tie-aware top-20% Jaccard values are approximately
`0.142857`, `0.142857`, `0.000000`, and `0.333333`, respectively.

The historical N5 baseline also reproduces N20 outlet risk near `0.70864`,
with the key ranking `N13 > N19 > N16`.

Robustness tail statistics and timing may vary slightly with implementation
details, random sampling order, operating-system load, and runtime
environment.

## Outputs

Generated files are written under `results/`. Important outputs include:

- `core_scenarios.csv`
- `N5_comparators.csv`
- `N5_robustness_summary.csv`
- `property_tests_summary.csv`
- `reconvergence_summary.csv`
- `reconvergence_node_diagnostic.csv`
- `scalability.csv`
- `N*_baseline_node_risk.csv`
- `N*_baseline_edge_contributions.csv`

## SWMM external reference

The SWMM workflow is separate from the internal GBBRPM experiment suite.
Its files are under `data/swmm/`, with blockage scenarios in
`data/swmm/scenarios/` and extracted hydraulic results in
`data/swmm/extracted/`.

Run the SWMM workflow with:

```bash
python scripts/run_swmm_and_extract.py
python scripts/compare_swmm_reference.py
```

The first command runs the supplied SWMM input files and refreshes
`data/swmm/extracted/swmm_hydraulic_results.csv`. The second command can be
run independently from the stored hydraulic results and writes:

- `data/swmm/extracted/swmm_comparison_by_scenario.csv`
- `data/swmm/extracted/swmm_comparison_summary.csv`

The comparison protocol represents conduit area loss as a reduction in
effective GBBRPM edge capacity, defines a common comparison universe as the
union of nodes affected in either model, computes rank correspondence and
tie-aware top-20% overlap within that universe, and reports the share of
positive hydraulic change outside the strictly downstream GBBRPM scope. The
stored inputs reproduce mean
outside-scope shares near `0.7120` for maximum depth and `0.4053` for flooding
volume.

The rank-correlation and top-3 values described as historical or archived in
the thesis are not asserted as regenerated unless their original calculation
rule is recovered. The repository now reports the values produced by the
explicit top-20% protocol above instead of silently selecting an undocumented
rule that happens to reproduce an archived number.

SWMM is used as an external hydraulic reference to examine:

- Where propagated-risk rankings agree with hydraulic consequences.
- Where those rankings diverge.
- Which hydraulic behaviors lie outside the downstream-only GBBRPM v1
  formulation.

The comparison treats GBBRPM as a lightweight, interpretable downstream
risk-propagation and prioritization surrogate, not as a replacement for a
hydraulic simulator.

## Software dependency case

The frozen npm/Express case under `data/software/express_4.18.2/` examines
whether the same propagation core can operate on a nonphysical dependency
network. Its 71 exact package-version nodes and 128 dependency relations come
from a deps.dev response. Original deps.dev edges point from a dependent
package to its dependency; the case reverses them so a disturbance propagates
from a dependency toward packages that rely on it.

Exact package versions were queried through OSV and the complete returned
advisory records were frozen in `osv_snapshot.json`. The mapping is explicit:

- `B` is the maximum matched categorical severity per exact package version:
  `LOW=.25`, `MODERATE=.50`, `HIGH=.75`, and `CRITICAL=1.00`.
- `S=1` denotes an existing resolved dependency relation.
- `tau` is not claimed as calibrated; it is swept over
  `0`, `.25`, `.50`, `.75`, and `1.00`.

Rebuild the case from an existing deps.dev response:

```bash
python scripts/build_software_case.py \
  --dependencies-json path/to/express_dependencies.json
python tests/software_case_test.py
```

Omit `--dependencies-json` to retrieve the default Express 4.18.2 graph from
deps.dev. Existing frozen OSV data is reused unless `--refresh-osv` is passed.
Generated results are stored under `results/software/express_4.18.2/`.

This case demonstrates cross-domain applicability, deterministic conversion,
bounded propagation, multi-source aggregation, and transmission sensitivity.
OSV supplies the local-disturbance inputs and is therefore not treated as an
independent outcome reference. In the frozen graph, the seven vulnerable
versions are Express itself or direct dependencies, so the observed positive
propagation is concentrated at those nodes and the Express root rather than a
long transitive chain. See `docs/software_domain_case.md` for the full mapping
and interpretation boundary.

## Electrical distribution case

The frozen public SoCal sample under `data/electrical/socal28_sample/`
provides a real electrical topology, time-varying equipment states, and meter
metadata. At the declared `2024-11-14T07:00:00` snapshot, the deterministic
converter produces 187 buses, 203 transfer-equipment relations, 177 active
directed connections, 34 documented state transitions, and 222 distinct
mapped measurement filenames. The nominal `fbus -> tbus` orientation is a
DAG, but its physical flow direction remains to be checked against
measurements.

Rebuild the topology package and test the preprocessing logic:

```bash
python scripts/build_electrical_case.py \
  --topology-source path/to/sample_dataset.zip
python tests/electrical_case_test.py
```

Summarize the large magnitude archive locally without extracting it:

```bash
python scripts/summarize_electrical_baseline.py \
  --magnitudes-zip path/to/magnitudes.zip
```

The baseline output distinguishes missing values from missing timestamps and
entirely empty channel files. Per-unit voltage summaries use only 60 direct
phase-to-ground mappings. Six signed or arithmetically derived voltage
registers are preserved as raw summaries but excluded from normalization and
event-response ranking. Two explicit `CT16 -> C16` filename aliases reconcile
the corresponding archive/topology naming mismatch while preserving both
source names in the output.

After event-centered measurements are approved and downloaded, produce the
independent observed-response ranking with:

```bash
python scripts/analyze_electrical_events.py \
  --measurements path/to/event_opening path/to/event_restoration
```

The one-day magnitude archive starts after the documented November 13, 2024
switching sequence, so it supports importer and normal-baseline checks rather
than event-response validation. GBBRPM is deliberately not executed for this
case until electrical definitions of `B`, `S`, and `tau` are declared without
leaking the observed outcome into both model input and validation target. See
`docs/electrical_domain_case.md` for the complete protocol and validation
gates.

### Caltech Component 2 measured-loading adapter

An optional private-data adapter converts the five-minute Component 2 probe
into a complete reduced topology and a model-ready observable subgraph. Raw
Caltech measurements, credentials, and private probe outputs remain outside
version control. Run it against the external dataset checkout:

```powershell
python .\scripts\build_caltech_component2.py `
  --topology-source "C:\path\to\digital-twin-dataset\sample_dataset.zip" `
  --probe-source "C:\path\to\digital-twin-dataset\data_caltech_component2_probe"
```

The adapter uses `egauge_21` as the component source and meters 3, 7, 9, 11,
and 13 as terminal loads. `egauge_19` is explicitly excluded because it
belongs to the neighboring feeder, while unavailable `egauge_15` is retained
in the audit table. It reconstructs complex three-phase power from aligned
phasors and defines edge susceptibility as
`min(1, mean apparent power / declared capacity)`.

The topology metadata labels `Mains_Power` as watts, although its raw values
agree with the reconstructed phasor power only when read as kilowatts. The
adapter records that cross-check but uses phasor-derived power for model input,
so the inconsistent register label is not silently trusted.

The complete reduced component contains 13 nodes and 12 edges. The unmetered
`line_407` branch remains in the complete topology with missing loading, but
is excluded from the 12-node, 11-edge observable model subgraph. The source to
metered-load residual is reported separately and is not assigned to that
branch. Model nodes start with `B=0`, and controlled scenarios inject local
disturbance independently; measured voltage is not reused as both input and
validation outcome. This is a descriptive real-topology, real-loading
instantiation, not predictive switching-event validation.

Run the controlled Component 2 experiment suite after building the private
case:

```powershell
python .\scripts\run_caltech_component2_experiments.py
```

The suite contains 43 runs across six families:

| Family | Runs | Controlled change |
| --- | ---: | --- |
| Zero-disturbance reference | 1 | `B=0`, measured loading, `tau=1` |
| Root severity | 4 | Root `B` in 0.25, 0.50, 0.75, 1.00 |
| Disturbance location | 12 | Isolated `B=0.75` at every observable node |
| Measured-load scaling | 5 | Uniform `L` scale from 0.50 to 1.50 |
| Uniform transmission | 5 | Uniform `tau` from 0.00 to 1.00 |
| Multi-source | 16 | Internal non-root pairs plus all internal nodes |

Outputs are written beneath
`results/electrical/caltech_component2_private/experiments/` and include the
complete node risks, edge contributions, scenario summaries, a manifest, and
five PDF/PNG figure pairs. Nested multi-source cases exercise bounded
local-plus-incoming aggregation; parallel cases exercise concurrent branch
coverage. Because the observable topology is an arborescence, these cases are
not described as reconvergence experiments.

## Reproducible paper outputs

Generate all data-derived Layer 1 and cross-domain publication figures, plus
the cross-domain LaTeX tables, from frozen results with:

```bash
python scripts/generate_paper_outputs.py
python tests/paper_outputs_test.py
```

The command writes vector PDFs, 300-dpi PNG previews, `booktabs` table files,
and an input/output hash manifest under `results/paper/`. The thirteen figure
designs and eight tables cover the Layer 1, drainage, software, and electrical
results.
See `docs/paper_outputs.md` for their intended evidentiary roles and the
electrical validation boundary.

## Reproducibility notes

The generic evaluator accepts either an explicit, domain-supplied `S` or a
drainage-style `L` and `C` pair. When `S` is absent, susceptibility is
recalculated from the primitive `L` and `C` inputs:

```text
u = L / C
S = min(1, u)
tau = 1
```

The evaluator validates:

- `C > 0`
- `L >= 0`
- `B in [0,1]`
- `tau in [0,1]`
- Unique node identifiers and directed edges
- Edge endpoints present in the node table
- DAG structure
- Bounded risk contributions

Where stored `u` or `S` columns exist, they may be checked against values
recomputed from `L` and `C`.

The reconvergence diagnostic is similarly explicit. It retains bounded
noisy-OR aggregation for independent source lineages and, solely for the
path-pruned counterfactual, keeps the largest incoming contribution when
branches share upstream ancestry. This regenerates the preserved N3, N4, and
N5 overlap-inflation table; the path-pruned result is a diagnostic reference,
not ground truth or a replacement formulation.

## Repository structure

```text
gbbrpm/
├── data/
│   ├── historical/
│   ├── electrical/
│   ├── recovered_candidate/
│   ├── reconstructed/
│   ├── software/
│   └── swmm/
├── results/
│   ├── electrical/
│   └── software/
├── gbbrpm/
│   ├── __init__.py
│   └── core.py
├── scripts/
├── tests/
├── model.py              # repository data adapter and compatibility imports
├── scenarios.py
├── run_experiments.py
├── inspect_results.py
├── pyproject.toml
└── requirements.txt
```

## Branch naming convention

Use these prefixes:

```text
feature/<short-description>
fix/<short-description>
docs/<short-description>
test/<short-description>
experiment/<short-description>
analysis/<short-description>
```

Examples include `feature/result-export`, `fix/csv-loader`,
`docs/readme-cleanup`, `experiment/lc-sweep`, and
`analysis/ranking-stability`.

## Scope and limitations

GBBRPM v1 is intended as a lightweight and interpretable graph-based
risk-propagation model. It does not currently model:

- Full hydraulic dynamics.
- Backwater effects.
- Bidirectional flow.
- Surcharge and ponding feedback.
- Common-cause or shared-path dependence correction.
- Calibrated heterogeneous transmission factors.

The baseline implementation therefore supports mechanism analysis, scenario
comparison, and prioritization rather than high-fidelity physical
simulation.
