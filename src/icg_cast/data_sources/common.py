"""Shared helpers for optional local data-source adapters."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REMOTE_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://")
PROVENANCE_SCHEMA_VERSION = "0.1"
PROVENANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "source_name",
    "source_version",
    "retrieval_date",
    "local_file",
    "license_notes",
    "citation",
    "sha256",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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

    def __post_init__(self) -> None:
        self.provenance = validate_provenance_record(self.provenance)


def validate_provenance_record(
    record: Mapping[str, Any],
    *,
    context: str = "provenance",
) -> dict[str, str]:
    """Validate and normalise one adapter provenance record."""
    if not isinstance(record, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = [field for field in PROVENANCE_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")

    out: dict[str, str] = {}
    for field_name in PROVENANCE_REQUIRED_FIELDS:
        value = str(record[field_name]).strip()
        if not value:
            raise ValueError(f"{context}.{field_name} must be a non-empty string")
        out[field_name] = value

    if out["local_file"].startswith(REMOTE_PREFIXES):
        raise ValueError(f"{context}.local_file must be a local path, not a remote URL")
    if not _SHA256_PATTERN.fullmatch(out["sha256"]):
        raise ValueError(f"{context}.sha256 must be a lowercase 64-character SHA-256 digest")
    return out


def validate_calibration_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a full ``calibration_provenance.json`` payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("calibration provenance must be a mapping")
    version = str(payload.get("schema_version", "")).strip()
    if version != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "calibration provenance schema_version must be "
            f"{PROVENANCE_SCHEMA_VERSION!r}; got {version!r}"
        )

    out: dict[str, Any] = {"schema_version": PROVENANCE_SCHEMA_VERSION}
    for key, value in payload.items():
        if key == "schema_version":
            continue
        if key == "coefficient_updates":
            if value is not None and not isinstance(value, Mapping):
                raise ValueError("calibration provenance coefficient_updates must be a mapping or null")
            out[key] = dict(value) if isinstance(value, Mapping) else None
            continue
        out[str(key)] = validate_provenance_record(
            value,
            context=f"calibration_provenance.{key}",
        )
    return out


def calibration_provenance_payload(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    coefficient_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a versioned calibration provenance payload."""
    if not isinstance(sources, Mapping):
        raise ValueError("calibration provenance sources must be a mapping")

    payload: dict[str, Any] = {"schema_version": PROVENANCE_SCHEMA_VERSION}
    for name, record in sources.items():
        if name in {"schema_version", "coefficient_updates"}:
            raise ValueError(f"{name!r} is reserved and cannot be used as a source name")
        payload[str(name)] = validate_provenance_record(
            record,
            context=f"calibration_provenance.{name}",
        )
    if coefficient_updates is not None:
        payload["coefficient_updates"] = dict(coefficient_updates)
    return validate_calibration_provenance(payload)


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
