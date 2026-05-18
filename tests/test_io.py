from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from icg_cast.io import ensure_dir


def test_ensure_dir_rejects_unwritable_file_path(tmp_path: Path) -> None:
    path = tmp_path / "not_a_dir"
    path.write_text("already a file", encoding="utf-8")

    with pytest.raises(OSError, match="could not create or write output directory"):
        ensure_dir(path)


def test_ensure_dir_can_fallback_to_temporary_directory(tmp_path: Path) -> None:
    path = tmp_path / "not_a_dir"
    path.write_text("already a file", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallback = ensure_dir(path, fallback_prefix="icg-cast-test-")

    assert fallback.exists()
    assert fallback.is_dir()
    assert fallback != path
    assert any("using temporary directory" in str(w.message) for w in caught)
