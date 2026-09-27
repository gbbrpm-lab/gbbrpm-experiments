# Electrical distribution case: preprocessing and validation plan

## Current evidentiary role

The public SoCal distribution sample is used as a second non-drainage domain
for testing whether the GBBRPM data contract can be instantiated on a real,
documented network. The currently available files establish topology,
time-varying switch states, meter-to-element mapping, and a normal-operation
magnitude baseline. They do **not** yet establish independent event-response
validation of GBBRPM.

The topology archive is frozen under
`data/electrical/socal28_sample/source_topology.zip`. At the declared snapshot
of `2024-11-14T07:00:00`, the converter produces 187 bus records, 203 transfer
equipment records, and 177 active directed connections. The nominal
`fbus -> tbus` orientation is acyclic. This orientation comes from the source
topology and has not yet been verified against measured power-flow direction.

## Deterministic conversion

`scripts/build_electrical_case.py` resolves each time-series switch state at
the requested snapshot, excludes normally open equipment, and writes:

- `nodes.csv`: bus identifiers and available nominal voltage metadata;
- `all_edges.csv`: every transfer-equipment relation and resolved state;
- `active_edges.csv`: relations active at the snapshot;
- `meter_mapping.csv`: measurement files mapped to elements and units;
- `status_events.csv`: documented state transitions; and
- `event_request_windows.csv`: operation groups with recommended five-minute
  download padding on both sides; and
- `manifest.json`: source hash, snapshot, counts, and interpretation limits.

The snapshot contains 226 register mappings representing 222 distinct named
measurement files. The status series contain 34 transitions grouped into 12
event windows under the declared 60-second grouping rule.

## Baseline magnitude archive

The separately downloaded magnitude ZIP covers approximately
`2024-11-14T07:00:00` through `2024-11-15T07:00:00`. Because the latest
documented switching sequence occurred on `2024-11-13`, this archive is used
only to verify ingestion, units, coverage, and normal-operation summaries. It
must not be presented as switching-event outcome validation.

Run the streaming summarizer locally without extracting the full archive:

```bash
python scripts/summarize_electrical_baseline.py \
  --magnitudes-zip path/to/magnitudes.zip
```

It writes `baseline_channel_summary.csv` and `baseline_manifest.json` under
`results/electrical/socal28_sample/`. The summary distinguishes three data
quality conditions that must not be conflated:

- `missing_value_count`: blank or nonnumeric values in rows that exist;
- `missing_timestamp_count`: absent intervals relative to the channel's
  inferred modal sampling cadence; and
- `data_status=empty`: a zero-byte, header-only, or otherwise observation-free
  channel file.

The summary also reports timestamp coverage, duplicates, out-of-order records,
quantiles, dispersion, and voltage mean in per-unit form where normalization is
eligible. Unmapped files remain in the summary so exclusions are visible.

Two explicit filename aliases reconcile a documented source inconsistency:
the topology declares `egauge_22-C16` and `egauge_23-C16`, while the magnitude
archive uses `egauge_22-CT16` and `egauge_23-CT16`. Exact declared filenames
remain the first matching rule. The importer then applies only these two
documented aliases, preserves both names in the output, and labels the match as
`mapping_method=explicit_source_filename_alias`. Neither frozen source is
renamed or modified.

### Voltage-channel eligibility

Per-unit summaries and the event-response ranking use only direct
phase-to-ground magnitude registers (`L1`, `L2`, or `L3`) with mapped nominal
voltage. Six source registers on `egauge_17` and `egauge_18` (`V_ab`, `V_bc`,
and `V_ca`) are defined by signed or arithmetic expressions in the topology
metadata. Their values are retained for traceability, but they receive
`per_unit_status=excluded_derived_or_signed_voltage` and are excluded from the
observed-response ranking. This prevents signed derived channels from being
silently interpreted as ordinary voltage magnitudes.

## Event-response analysis after approval

When the requested event-centered measurements are available, place their CSV
files in one or more directories or ZIP archives and run:

```bash
python scripts/analyze_electrical_events.py \
  --measurements path/to/event_opening path/to/event_restoration
```

For each documented operation group, the script compares median measurements
in a 60-second pre-event window and a 60-second post-event window, separated
from the operation by a five-second guard interval. Voltage responses are
normalized by nominal voltage; power responses retain both absolute and
relative change. Bus-level voltage response is the maximum change among
eligible direct phase-to-ground channels. The script reports the complete
ranking and a tie-aware top-20% priority set, consistent with the generic
reporting convention.

This observed-response ranking is an independent comparison target. A later
GBBRPM instantiation must separately declare and justify electrical meanings
for local disturbance `B`, susceptibility `S`, and transmission `tau` before
model predictions are compared with it. The topology importer intentionally
does not invent these quantities.

## Remaining validation gates

The electrical case can support a Chapter 4 validation claim only after:

1. event-centered measurements cover at least one documented switching group;
2. the direction used for propagation is checked against the event or power
   measurements;
3. the mappings for `B`, `S`, and `tau` are specified without using the same
   observed outcome later treated as ground truth;
4. the model ranking is compared with the independently computed observed
   response ranking using the complete-ranking and tie-aware top-20% rules;
5. uncovered events, missing channels, and disconnected buses are reported;
   and
6. all source files, time windows, hashes, and transformation settings are
   preserved in manifests.

Until those gates are met, this case is accurately described as a real-data
feasibility and preprocessing result, not completed predictive validation.

## Component 2 measured-loading instantiation

The private Component 2 adapter is implemented in `caltech_component2.py` and
invoked through `scripts/build_caltech_component2.py`. It accepts the frozen
topology and the authorized five-minute magnitude/phasor probe as external
inputs; neither credentials nor raw measurements are copied into this
repository.

At `2024-11-14T07:00:00`, electrical reduction produces the 13-node, 12-edge
weak component supplied by `gp_1`. The measurement roles are fixed and audited:

- `egauge_21`: source-side measurement on the Component 2 feeder;
- `egauge_3`, `egauge_7`, `egauge_9`, `egauge_11`, and `egauge_13`: terminal
  load measurements;
- `egauge_19`: excluded neighboring-feeder measurement; and
- `egauge_15`: recorded as unavailable for the selected window.

The importer aligns each voltage and current phasor to the shared capture
timestamp, rejecting a case when any nearest-sample offset exceeds the
declared threshold. Complex power is reconstructed as

$$
S_{m,t}=\sum_{p=1}^{3}V_{m,p,t}I^*_{m,p,t}.
$$

Although topology metadata declares `Mains_Power` in watts, the downloaded
raw values agree with the phasor reconstruction only when interpreted as
kilowatts. The adapter reports bias, percentage error, and correlation under
that explicit interpretation. Phasor-derived power remains the model input;
the inconsistent register is used only as a cross-check.

Terminal meter powers are aggregated over descendants to obtain branch load.
The source transformer uses the direct source measurement. Transformer
capacity uses the minimum declared kVA when the source supplies multiple
candidate ratings; line capacity uses

$$
C_{line}=3V_{phase,nominal}I_{rating}/1000.
$$

The resulting electrical susceptibility is

$$
S_{ij}=\min(1,L_{ij}/C_{ij}),
$$

where electrical `L` is mean apparent power in kVA. `tau` starts at one.
`B` starts at zero and must be injected independently by a controlled
disturbance scenario. This separation prevents measured loading and voltage
response from being silently counted twice.

The unmetered `line_407` branch is preserved in the complete topology but
excluded from the observable model subgraph. The source-minus-metered-load
residual is reported as an unallocated diagnostic because it combines
unobserved demand, losses, and measurement/calibration differences. Assigning
the entire residual to `line_407` would be an unsupported imputation.

Accordingly, this adapter supports a real-topology, real-loading GBBRPM
instantiation and controlled propagation experiments. It does not satisfy the
independent event-response validation gates above.
