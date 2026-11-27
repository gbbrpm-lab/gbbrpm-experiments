# Reproducibility Status

Verified on 2026-09-26 from an isolated Python virtual environment.

## Commands

```bash
pip install -r requirements.txt
pip install -e .
python tests/package_api_test.py
python tests/smoke_test.py
python tests/reproducibility_test.py
python tests/generic_properties_test.py
python tests/software_case_test.py
python tests/electrical_case_test.py
python tests/paper_outputs_test.py
python tests/swmm_reference_test.py
python run_experiments.py --data-source historical --trials 600 --seed 41
python scripts/run_swmm_and_extract.py
python scripts/compare_swmm_reference.py
```

## Verified results

- All five historical networks load and produce bounded node states.
- The installable `gbbrpm` package exposes the same calculation engine used by
  the backwards-compatible repository scripts.
- The controlled scenario suite contains 161 rows: the preserved 136-case
  historical protocol plus 25 uniform-transmission sensitivity cases.
- All deterministic generic checks pass for topological-order invariance,
  graph relabeling, zero-state consistency, edge gating, cycle and endpoint
  rejection, deterministic ties, and proportional priority-set expansion.
- The N5 comparator correlations reproduce the preserved values.
- All four property checks pass in 600 of 600 randomized trials.
- The N5 comparator top-20% Jaccard values are 0.142857, 0.142857,
  0.000000, and 0.333333 for local-only and uniform susceptibility 0.25,
  0.50, and 0.75, respectively.
- Mean top-20% Jaccard robustness is 0.776000, 0.604873, and 0.440823
  under +/-5%, +/-10%, and +/-20% perturbation.
- Robustness summaries reproduce the Chapter 4 values at all three
  perturbation levels.
- The path-pruned diagnostic reproduces the N3, N4, and N5 overlap-inflation
  values to six decimal places.
- The SWMM solver runs all 12 blockage scenarios and refreshes the stored node
  hydraulic results.
- The explicit SWMM comparison reproduces mean outside-downstream-scope shares
  of approximately 0.7120 for maximum-depth worsening and 0.4053 for
  flooding-volume worsening.
- The explicit SWMM comparison produces mean tie-aware top-20% Jaccard values
  of 0.119444 for maximum-depth change and 0.000000 for flooding-volume
  change over each scenario's common affected-node comparison universe.
- The frozen Express 4.18.2 software case contains 71 exact package-version
  nodes and 128 reversed dependency edges in one weakly connected DAG.
- The frozen OSV snapshot identifies 13 advisory matches across seven exact
  package versions. The case accepts explicit domain-supplied `S`, remains
  bounded across the five declared transmission levels, and reproduces an
  Express root risk of 0.997528076171875 at `tau=1`.
- The software case is verified as cross-domain applicability and structural
  behavior, not as independent outcome validation because OSV supplies its
  local-disturbance inputs.
- The frozen electrical topology package contains 187 buses, 203 transfer
  equipment records, 177 active directed connections at the declared
  snapshot, 226 register mappings, and 34 documented status transitions.
- The electrical preprocessing tests verify snapshot structure, nominal
  acyclicity, event grouping, streaming magnitude summaries, value and
  timestamp missingness, zero-byte and header-only empty-channel
  classification, explicit source-filename aliasing, voltage-channel
  eligibility, event-window response extraction, and tie-aware top-20%
  reporting.
- The available one-day magnitude archive is classified as a baseline/import
  check rather than event-response validation because its coverage begins
  after the documented November 13, 2024 switching sequence.
- The publication-output generator deterministically produces thirteen figure
  designs in PDF and PNG plus eight `booktabs` LaTeX tables. A committed
  manifest verifies every frozen input and generated output by SHA-256.

## Archived SWMM ranking metrics

The thesis contains archived SWMM rank-correlation and top-3 Jaccard values
whose original calculation script and complete ranking rule were not present
in the recovered repository. They are not silently relabeled as regenerated.
The current `scripts/compare_swmm_reference.py` states and executes one
complete tie-aware top-20% protocol, and its generated summary is the
reproducible result of that protocol. The archived values should remain
labeled historical unless their original calculation artifact is recovered.

## Timing interpretation

`scalability.csv` now reports the median and interquartile range in addition
to the mean, minimum, and maximum over 30 timed repetitions after five warm-up
runs. Runtime values remain machine- and load-dependent; the median is the
preferred descriptive value when isolated scheduling outliers occur.

## Protocol Freeze (v2.0.0-protocol-freeze)

Frozen prior to held-out evaluation on seeds `1001–1020` (`eval_seeds_evaluated = 0`, `eval_outcomes_inspected = false`).

### Frozen Artifacts

- Pinned environment lockfile: `requirements-lock-v2.0.0.txt` (Python `3.13.14`, `numpy==2.2.0`, `scipy==1.15.2`, `pandas==2.2.3`, `networkx==3.4.2`, `pytest==9.0.3`, `matplotlib==3.10.1`, `swmm-toolkit==0.17.0`).
- Immutable freeze manifest: `results/protocol_freeze/protocol_freeze_manifest_v2.0.0.json`, built and verified by `scripts/build_protocol_freeze_manifest.py`.
- Protocol freeze summary: `docs/PROTOCOL_FREEZE_v2.0.0.md`.
- Development compute pilot & controls (`1,304` scenarios: `560` matched branch + `720` dev uniform on seeds `41–45` + `24` dev hetero on seed `41`): `results/dev_pilot/`, with `0` bracket violations, `0` false certifications, and full-inventory feasibility confirmed (`26.30 min` single-core / `4.38 min` across `8` workers for `4,316` primary + `48` million-trial precision audits).

### Verified Commands

```bash
python scripts/build_protocol_freeze_manifest.py --verify
pytest -v
python tests/package_api_test.py
python tests/smoke_test.py
python tests/generic_properties_test.py
python tests/reproducibility_test.py
```
