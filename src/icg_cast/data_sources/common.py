"""Shared helpers for optional local data-source adapters."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REMOTE_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://")


@dataclass(frozen=True)
class Provenance:
    """Minimal provenance metadata for a user-supplied local file."""

    source_name: str
    source_version: str = "user_supplied"
    retrieval_date: str = "user_supplied"
    local_file: str = ""
    license_notes: str = "maintainer must verify"
    citation: str = "maintainer must fill"
    sha256: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class DataSourceBundle:
    """Data table plus provenance and adapter-specific metadata."""

    data: pd.DataFrame
    provenance: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


def require_local_file(path: str | Path) -> Path:
    """Return a resolved local file path or raise for URLs/missing files."""
    path_str = str(path)
    if path_str.startswith(REMOTE_PREFIXES):
        raise ValueError("data-source adapters accept local files only; URLs are not fetched")
    local = Path(path).expanduser()
    if not local.exists():
        raise FileNotFoundError(local)
    if not local.is_file():
        raise ValueError(f"expected a file path, got {local}")
    return local


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 digest for provenance records."""
    local = require_local_file(path)
    digest = hashlib.sha256()
    with local.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_for(
    path: str | Path,
    source_name: str,
    source_version: str = "user_supplied",
    retrieval_date: str = "user_supplied",
    license_notes: str = "maintainer must verify",
    citation: str = "maintainer must fill",
) -> dict[str, str]:
    """Build a provenance dictionary for a local file."""
    local = require_local_file(path)
    return Provenance(
        source_name=source_name,
        source_version=source_version,
        retrieval_date=retrieval_date,
        local_file=str(local),
        license_notes=license_notes,
        citation=citation,
        sha256=sha256_file(local),
    ).to_dict()


def read_local_table(path: str | Path, table: str | None = None) -> pd.DataFrame:
    """Read a local CSV/TSV/JSON/JSONL/SQLite table into a DataFrame."""
    local = require_local_file(path)
    suffix = local.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(local)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(local, sep="\t")
    if suffix == ".json":
        with local.open(encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.json_normalize(obj)
        raise ValueError(f"unsupported JSON top-level object in {local}")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(local, lines=True)
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        with sqlite3.connect(local) as con:
            if table is None:
                names = pd.read_sql_query(
                    "select name from sqlite_master where type='table' order by name",
                    con,
                )["name"].tolist()
                if not names:
                    raise ValueError(f"SQLite file contains no tables: {local}")
                table = str(names[0])
            return pd.read_sql_query(f'select * from "{table}"', con)
    raise ValueError(f"unsupported local table format: {local.suffix}")


def make_bundle(
    path: str | Path,
    source_name: str,
    data: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    source_version: str = "user_supplied",
    retrieval_date: str = "user_supplied",
    license_notes: str = "maintainer must verify",
    citation: str = "maintainer must fill",
) -> DataSourceBundle:
    """Attach provenance to a loaded local table."""
    return DataSourceBundle(
        data=data,
        provenance=provenance_for(
            path,
            source_name=source_name,
            source_version=source_version,
            retrieval_date=retrieval_date,
            license_notes=license_notes,
            citation=citation,
        ),
        metadata=dict(metadata or {}),
    )
