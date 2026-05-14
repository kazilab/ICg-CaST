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
from typing import Any

from .scoring import BenchmarkResult, score_summary

LEADERBOARD_SCHEMA_VERSION = "0.1"


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


def append_entry(entry: LeaderboardEntry, outdir: str) -> tuple[str, str]:
    """Append one entry to `leaderboard.csv` and `leaderboard.json` in `outdir`.

    Returns the (csv_path, json_path) written. Both files are created on first
    call; CSV is append-only, JSON is rewritten with the full sequence each
    call (so the file is always valid).
    """
    os.makedirs(outdir, exist_ok=True)
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
        with open(json_path, encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(asdict(entry))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    return csv_path, json_path


def write_leaderboard(entries: list[LeaderboardEntry], outdir: str) -> tuple[str, str]:
    """Replace the leaderboard files with `entries`.

    Useful for regenerating from a list of results during CI.
    """
    os.makedirs(outdir, exist_ok=True)
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
