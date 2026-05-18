from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from icg_cast.benchmark import (
    LeaderboardEntry,
    LeaderboardSchemaError,
    append_entry,
    load_leaderboard,
    validate_leaderboard_entry,
    write_leaderboard,
)


def _entry() -> LeaderboardEntry:
    return LeaderboardEntry(
        variant_name="linear_lowhet",
        variant_hash="abc123def456",
        model_name="smoke-model",
        package_version="0.0-test",
        submitted_at="2026-05-15T00:00:00+00:00",
        auroc=0.8,
        r2_mean=0.7,
        conformity=0.9,
        auroc_target=0.75,
        transfer_gap=0.05,
        composite=0.8,
    )


def test_leaderboard_write_and_load_validate_schema(tmp_path: Path) -> None:
    entry = _entry()
    csv_path, json_path = write_leaderboard([entry], str(tmp_path))

    assert Path(csv_path).exists()
    loaded = load_leaderboard(json_path)

    assert len(loaded) == 1
    assert loaded[0].schema_version == "0.1"
    assert loaded[0].variant_hash == entry.variant_hash


def test_leaderboard_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    entry = replace(_entry(), schema_version="9.9")

    with pytest.raises(LeaderboardSchemaError, match="unsupported leaderboard schema_version"):
        validate_leaderboard_entry(entry)

    with pytest.raises(LeaderboardSchemaError, match="unsupported leaderboard schema_version"):
        append_entry(entry, str(tmp_path))


def test_load_leaderboard_fails_closed_when_migration_missing(tmp_path: Path) -> None:
    path = tmp_path / "leaderboard.json"
    payload = [_entry().__dict__ | {"schema_version": "0.0"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LeaderboardSchemaError, match="no migration registered"):
        load_leaderboard(path)
