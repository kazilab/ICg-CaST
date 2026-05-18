# Public-data calibration prototype

ICg-CaST is synthetic by default. The calibration prototype is an opt-in
layer that lets user-supplied local files from COSMIC, LINCS, ToxCast,
AOP-Wiki, and AOP-DB override pieces of the simulator and theory graph.
No real data is downloaded, fetched over the network, or committed to this
repository. All tests use tiny synthetic fixtures.

> Synthetic outputs from calibrated runs are still synthetic. Calibration
> swaps prior values inside the simulator; it does not turn the package into a
> clinical, regulatory, or epidemiological tool.

## What gets calibrated

| Source | Adapter | Calibrator | What it overrides |
| --- | --- | --- | --- |
| COSMIC SBS matrix | `load_cosmic_sbs_matrix` | `calibrate_signatures_from_cosmic` | Mutational-signature profiles in [signatures.py](../src/icg_cast/signatures.py). Maps file columns onto the toy keys `aging`, `SBS4_like`, `SBS24_like`, `SBS22_like`, `oxidative_like` via an optional `name_map`. |
| LINCS L1000 | `load_lincs_signatures` | `calibrate_transcript_modules_from_lincs` | Produces a long-form `(perturbagen, module, mean_score, n_genes)` prior table. Stored on the calibration bundle for downstream use; not wired into `generate_omics` in v0.1. |
| EPA ToxCast / CompTox | `load_toxcast_summary` | `calibrate_kcc_priors_from_toxcast` | Replaces `ARCHETYPE_KCC` with per-chemical 10-element KCC vectors derived from hit-call fractions over an `assay → KCC` mapping. Chemical IDs become archetype names. |
| AOP-Wiki edge export | `load_aopwiki_export` | `enrich_theory_graph` / `build_calibration_bundle` | Merges new edges (and nodes) into [graph.py](../src/icg_cast/graph.py)'s default theory graph. |
| EPA AOP-DB | `load_aopdb_export` | `enrich_theory_graph` / `build_calibration_bundle` | Attaches per-node metadata to the theory graph by matching a node-id column. |

## Quickstart (Python API)

```python
from icg_cast import (
    SimConfig,
    build_calibration_bundle,
    build_theory_graph,
    simulate_cohort,
)

bundle = build_calibration_bundle(
    cosmic_path="local/cosmic_sbs.csv",
    cosmic_name_map={"SBS4": "SBS4_like", "SBS24": "SBS24_like"},
    toxcast_path="local/toxcast_summary.csv",
    toxcast_mapping="local/assay_to_kcc.csv",
    aopwiki_path="local/aopwiki_edges.csv",
)
bundle.save("outputs/calibration/calibration_bundle.json")

cohort, _ = simulate_cohort(SimConfig(n=200, months=24, seed=7), calibration=bundle)
graph = build_theory_graph(calibration=bundle)
```

The default behaviour of `simulate_cohort` and `build_theory_graph` is
unchanged when `calibration` is `None`; existing scripts and tests are not
affected.

## Quickstart (CLI)

```bash
icg-cast calibrate \
  --cosmic local/cosmic_sbs.csv \
  --cosmic-name-map "SBS4=SBS4_like,SBS24=SBS24_like" \
  --toxcast local/toxcast_summary.csv \
  --toxcast-mapping local/assay_to_kcc.csv \
  --aopwiki local/aopwiki_edges.csv \
  --outdir outputs/calibration

icg-cast simulate \
  --calibration outputs/calibration/calibration_bundle.json \
  --n 1200 --months 72 --seed 7 \
  --outdir outputs/calibrated_demo

icg-cast graph \
  --calibration outputs/calibration/calibration_bundle.json \
  --outdir outputs/calibrated_demo
```

`icg-cast calibrate` writes two files in the chosen output directory:

* `calibration_bundle.json` — the opt-in overrides, reloadable with
  `icg_cast.load_calibration_bundle(path)`.
* `calibration_provenance.json` — the per-source provenance records (source
  name, version, retrieval date, local file path, SHA-256 digest, license/
  citation placeholders) returned by every adapter. This file is versioned
  with `schema_version: "0.1"` and validated at runtime against the same field
  contract documented in `materials/calibration_provenance.schema.json`.

## Tiny end-to-end example

[examples/run_calibration.py](../examples/run_calibration.py) writes synthetic
mock COSMIC / LINCS / ToxCast / AOP-Wiki files into a temp directory, builds a
calibration bundle, and runs the simulator and theory graph with the bundle
applied. Run it with:

```bash
python examples/run_calibration.py
```

The script never touches real data.

## Acceptance criteria

- [x] User-supplied COSMIC SBS file loader: `load_cosmic_sbs_matrix` and the
      `calibrate_signatures_from_cosmic` calibrator.
- [x] User-supplied LINCS signature loader: `load_lincs_signatures` and the
      `calibrate_transcript_modules_from_lincs` calibrator.
- [x] User-supplied ToxCast summary loader: `load_toxcast_summary` and the
      `calibrate_kcc_priors_from_toxcast` calibrator.
- [x] qAOP graph enrichment from local AOP exports: `enrich_theory_graph` plus
      bundle-driven enrichment in `build_theory_graph(calibration=...)`.
- [x] All examples run with tiny mock data: see
      [examples/run_calibration.py](../examples/run_calibration.py).
- [x] Real-data workflows are documented but not required for tests:
      [tests/test_calibration.py](../tests/test_calibration.py) uses
      `tmp_path` fixtures only.
- [x] No controlled-access or large public datasets are committed.

## Data governance reminder

Public availability does not mean unrestricted reuse. COSMIC, CTD, LINCS, EPA,
GDC, and other resources have distinct citation, license, and access terms.
Controlled-access human genomic data (e.g. dbGaP-protected GDC files) must not
be committed to this repository. See
[docs/ethics_and_limitations.md](ethics_and_limitations.md) for the data
governance policy.
