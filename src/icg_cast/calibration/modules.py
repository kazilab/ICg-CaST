"""LINCS-driven transcriptomic module calibration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data_sources.common import DataSourceBundle, read_local_table


def calibrate_transcript_modules_from_lincs(
    bundle: DataSourceBundle,
    module_map: pd.DataFrame | str | Path,
    perturbagen_column: str = "perturbagen",
    gene_column: str = "gene",
    score_column: str = "score",
    module_gene_column: str = "gene",
    module_module_column: str = "module",
) -> pd.DataFrame:
    """Aggregate LINCS gene-level scores into per-(perturbagen, module) priors.

    Args:
        bundle: result of ``load_lincs_signatures(...)``. Expected long-form
            with one row per (perturbagen, gene) and a numeric score column.
        module_map: DataFrame or path to a CSV/TSV mapping genes to
            transcriptomic modules. Must contain the columns named by
            ``module_gene_column`` and ``module_module_column``.
        perturbagen_column / gene_column / score_column: column names in the
            LINCS table.

    Returns:
        Long-form DataFrame with columns
        ``[perturbagen, module, mean_score, n_genes]``.
    """
    if bundle.metadata.get("adapter") != "lincs":
        raise ValueError("calibrate_transcript_modules_from_lincs expects a LINCS adapter bundle")
    data = bundle.data.copy()
    missing = {perturbagen_column, gene_column, score_column} - set(data.columns)
    if missing:
        raise KeyError(f"LINCS bundle missing required columns: {sorted(missing)}")

    if isinstance(module_map, pd.DataFrame):
        mmap = module_map.copy()
    else:
        mmap = read_local_table(module_map)
    map_missing = {module_gene_column, module_module_column} - set(mmap.columns)
    if map_missing:
        raise KeyError(f"module_map missing required columns: {sorted(map_missing)}")
    mmap = mmap.rename(
        columns={module_gene_column: "gene_key", module_module_column: "module_key"}
    )
    data = data.rename(
        columns={perturbagen_column: "perturbagen_key", gene_column: "gene_key",
                 score_column: "score_key"}
    )
    merged = data.merge(mmap[["gene_key", "module_key"]], on="gene_key", how="inner")
    if merged.empty:
        raise ValueError("no genes shared between LINCS table and module_map")
    agg = (
        merged.groupby(["perturbagen_key", "module_key"], as_index=False)
        .agg(mean_score=("score_key", "mean"), n_genes=("gene_key", "nunique"))
        .rename(columns={"perturbagen_key": "perturbagen", "module_key": "module"})
        .sort_values(["perturbagen", "module"])
        .reset_index(drop=True)
    )
    return agg
