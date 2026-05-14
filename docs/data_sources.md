# Data Sources

ICg-CaST defaults to synthetic data. Optional data-source adapters are
local-file loaders for future calibration or validation workflows. They do not
download files, call remote APIs, or handle controlled-access data.

> For the Milestone 7 calibration prototype that consumes these adapters to
> override pieces of the synthetic simulator and theory graph, see
> [docs/calibration.md](calibration.md).

Each adapter returns a `DataSourceBundle` with:

- `data`: loaded `pandas.DataFrame`.
- `provenance`: source name, version, retrieval date, local file, license notes,
  citation notes, and SHA-256 digest.
- `metadata`: adapter-specific row counts and validation details.

## Supported Stub Adapters

| Source | Adapter | Expected input | Package behavior |
| --- | --- | --- | --- |
| AOP-Wiki | `load_aopwiki_export` | CSV/TSV/JSON edge or node export | Loads local file; can map `source`/`target` edge lists to a graph. |
| EPA AOP-DB | `load_aopdb_export` | CSV/TSV/JSON/SQLite-derived table | Loads local table and records provenance. |
| EPA ToxCast/CompTox | `load_toxcast_summary` | Local summary table plus optional mapping table | Keeps assay summaries separate from KCC mapping metadata. |
| COSMIC Mutational Signatures | `load_cosmic_sbs_matrix` | Local 96-channel SBS matrix | Validates 96 contexts and non-negative signature columns. |
| SigProfilerExtractor | `load_sigprofiler_activities` | Local activity table | Loads exported activity matrix; no optional dependency import. |
| LINCS L1000 | `load_lincs_signatures` | Local signature table plus optional metadata | Records perturbagen metadata row counts when provided. |
| CTD | `load_ctd_chemical_gene_disease` | Local CTD export table | Loads local curated relationship table. |
| NCI GDC/TCGA/CPTAC | `load_gdc_manifest` | Local manifest or open metadata table | Warns against committing controlled-access data. |

## Provenance

Use `materials/provenance_template.json` as the minimum record for any
user-supplied source. The maintainer must fill in source version, retrieval
date, license terms, and citation before using real data in analysis.

## Access and License Notes

Public availability does not mean unrestricted reuse. COSMIC, CTD, LINCS, EPA,
GDC, and other resources may have distinct citation, license, and access terms.
Controlled-access human genomic data must not be committed to this repository.
