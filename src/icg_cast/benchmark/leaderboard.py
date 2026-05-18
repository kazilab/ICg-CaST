"""ICg-Bench leaderboard schema and persistence helpers.

A leaderboard entry is a single CSV row + JSON document describing a
(variant, model, package_version) result. Persistence is intentionally
file-based (no DB) so the benchmark can be cited via repo URLs and replayed
deterministically.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scoring import BenchmarkResult, score_summary

LEADERBOARD_SCHEMA_VERSION = "0.1"
SUPPORTED_LEADERBOARD_SCHEMA_VERSIONS = (LEADERBOARD_SCHEMA_VERSION,)


@dataclass
class LeaderboardEntry:
    """One leaderboard row."""

    variant_name: str
    variant_hash: str
    model_name: str
    package_version: str
    submitted_at: str
    schema_version: str = LEADERBOARD_SCHEMA_VERSION
    auroc: float = float("nan")
    r2_mean: float = float("nan")
    conformity: float = float("nan")
    auroc_target: float = float("nan")
    transfer_gap: float = float("nan")
    composite: float = float("nan")
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: BenchmarkResult) -> LeaderboardEntry:
        summary = score_summary(result)
        return cls(
            variant_name=result.variant_name,
            variant_hash=result.variant_hash,
            model_name=result.model_name,
            package_version=result.package_version,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            auroc=summary["auroc"],
            r2_mean=summary["r2_mean"],
            conformity=summary["conformity"],
            auroc_target=summary["auroc_target"],
            transfer_gap=summary["transfer_gap"],
            composite=summary["composite"],
            notes=result.notes,
        )


class LeaderboardSchemaError(ValueError):
    """Raised when a leaderboard file or entry does not match the active schema."""


_FIELDS = [
    "schema_version",
    "submitted_at",
    "variant_name",
    "variant_hash",
    "model_name",
    "package_version",
    "auroc",
    "r2_mean",
    "conformity",
    "auroc_target",
    "transfer_gap",
    "composite",
    "notes",
]


def validate_leaderboard_entry(entry: LeaderboardEntry | dict[str, Any]) -> LeaderboardEntry:
    """Validate one leaderboard entry and return it as a dataclass."""
    if isinstance(entry, LeaderboardEntry):
        candidate = entry
    elif isinstance(entry, dict):
        missing = [field_name for field_name in _FIELDS if field_name not in entry]
        if missing:
            raise LeaderboardSchemaError(f"leaderboard entry missing fields: {missing}")
        candidate = LeaderboardEntry(
            variant_name=str(entry["variant_name"]),
            variant_hash=str(entry["variant_hash"]),
            model_name=str(entry["model_name"]),
            package_version=str(entry["package_version"]),
            submitted_at=str(entry["submitted_at"]),
            schema_version=str(entry["schema_version"]),
            auroc=float(entry["auroc"]),
            r2_mean=float(entry["r2_mean"]),
            conformity=float(entry["conformity"]),
            auroc_target=float(entry["auroc_target"]),
            transfer_gap=float(entry["transfer_gap"]),
            composite=float(entry["composite"]),
            notes=str(entry.get("notes", "")),
            extra=dict(entry.get("extra", {})),
        )
    else:
        raise LeaderboardSchemaError("leaderboard entry must be a LeaderboardEntry or mapping")

    if candidate.schema_version != LEADERBOARD_SCHEMA_VERSION:
        raise LeaderboardSchemaError(
            f"unsupported leaderboard schema_version {candidate.schema_version!r}; "
            f"expected {LEADERBOARD_SCHEMA_VERSION!r}"
        )
    for field_name in ("submitted_at", "variant_name", "variant_hash", "model_name", "package_version"):
        if not str(getattr(candidate, field_name)).strip():
            raise LeaderboardSchemaError(f"leaderboard entry {field_name} must be non-empty")
    return candidate


def migrate_leaderboard_entries(
    entries: list[dict[str, Any]],
    *,
    target_version: str = LEADERBOARD_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Migrate raw leaderboard rows to ``target_version``.

    Version 0.1 is the first published schema, so no older migrations exist.
    Future schema changes should add explicit stepwise migrations here instead
    of letting readers silently reinterpret old rows.
    """
    migrated: list[dict[str, Any]] = []
    for entry in entries:
        version = str(entry.get("schema_version", ""))
        if version == target_version:
            migrated.append(dict(entry))
            continue
        raise LeaderboardSchemaError(
            f"no migration registered from leaderboard schema_version {version!r} "
            f"to {target_version!r}"
        )
    return migrated


def load_leaderboard(path: str | os.PathLike[str]) -> list[LeaderboardEntry]:
    """Load and validate a leaderboard JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise LeaderboardSchemaError("leaderboard JSON must contain a list of entries")
    migrated = migrate_leaderboard_entries(raw)
    return [validate_leaderboard_entry(entry) for entry in migrated]


def append_entry(entry: LeaderboardEntry, outdir: str) -> tuple[str, str]:
    """Append one entry to `leaderboard.csv` and `leaderboard.json` in `outdir`.

    Returns the (csv_path, json_path) written. Both files are created on first
    call; CSV is append-only, JSON is rewritten with the full sequence each
    call (so the file is always valid).
    """
    os.makedirs(outdir, exist_ok=True)
    entry = validate_leaderboard_entry(entry)
    csv_path = os.path.join(outdir, "leaderboard.csv")
    json_path = os.path.join(outdir, "leaderboard.json")

    write_header = not os.path.exists(csv_path)
    row = {k: getattr(entry, k) for k in _FIELDS}
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    history: list[dict[str, Any]] = []
    if os.path.exists(json_path):
        history = [asdict(existing) for existing in load_leaderboard(json_path)]
    history.append(asdict(entry))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    return csv_path, json_path


def write_leaderboard(entries: list[LeaderboardEntry], outdir: str) -> tuple[str, str]:
    """Replace the leaderboard files with `entries`.

    Useful for regenerating from a list of results during CI.
    """
    os.makedirs(outdir, exist_ok=True)
    entries = [validate_leaderboard_entry(entry) for entry in entries]
    csv_path = os.path.join(outdir, "leaderboard.csv")
    json_path = os.path.join(outdir, "leaderboard.json")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: getattr(e, k) for k in _FIELDS})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2, default=str)
    return csv_path, json_path
