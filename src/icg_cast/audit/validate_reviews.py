"""Coefficient review validation."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

from icg_cast.coefficients.registry import load_registry

REVIEW_HASH_PREFIX = "coefficient_cards_sha256:"


def coefficient_cards_sha256(cards_file: str = "materials/coefficient_cards.yaml") -> str:
    """Return the SHA-256 digest of the reviewed coefficient-card registry."""
    path = Path(cards_file)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_cards_hash(review_file: Path) -> str | None:
    """Read the registry digest embedded in the comment header of the review CSV."""
    for raw_line in review_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("#"):
            return None
        comment = line.lstrip("#").strip()
        if comment.startswith(REVIEW_HASH_PREFIX):
            return comment.split(":", 1)[1].strip()
    return None


def validate_coefficient_reviews(
    review_file: str = "materials/coefficient_review.csv",
    flags_file: str = "outputs/audit/coefficient_flags.csv",
    cards_file: str = "materials/coefficient_cards.yaml",
) -> bool:
    """Return True when load-bearing reviews match the current coefficient registry."""
    review_path = Path(review_file)
    flags_path = Path(flags_file)
    cards_path = Path(cards_file)

    if not review_path.exists():
        print(f"ERROR: {review_file} not found.")
        return False

    if not cards_path.exists():
        print(f"ERROR: {cards_file} not found.")
        return False

    declared_hash = _declared_cards_hash(review_path)
    current_hash = coefficient_cards_sha256(str(cards_path))
    if not declared_hash:
        print(
            f"ERROR: {review_file} is missing a '# {REVIEW_HASH_PREFIX} <sha256>' header."
        )
        return False
    if declared_hash != current_hash:
        print(
            f"ERROR: {review_file} was reviewed against coefficient_cards.yaml hash "
            f"{declared_hash}, but the current hash is {current_hash}."
        )
        return False

    if not flags_path.exists():
        print(f"WARNING: {flags_file} not found. Run `icg-cast coeffs sensitivity` first.")
        return False

    reviews = pd.read_csv(review_path, comment="#").fillna("")
    flags = pd.read_csv(flags_path)
    required = {"coefficient", "reviewed_by", "review_date", "decision", "dissent_notes"}
    missing = required - set(reviews.columns)
    if missing:
        print(f"ERROR: {review_file} missing columns: {sorted(missing)}")
        return False
    if reviews["coefficient"].duplicated().any():
        duplicates = sorted(reviews.loc[reviews["coefficient"].duplicated(), "coefficient"])
        print(f"ERROR: {review_file} has duplicate review rows: {duplicates}")
        return False

    registry_names = set(load_registry(cards_path).names())
    unknown = sorted(set(reviews["coefficient"]) - registry_names)
    if unknown:
        print(f"ERROR: {review_file} references unknown coefficients:")
        for coeff in unknown:
            print(f"   - {coeff}")
        return False

    load_bearing = flags[flags["flag"] == "LOAD_BEARING"]["coefficient"].tolist()

    missing_reviews = []
    for coeff in load_bearing:
        row = reviews[reviews["coefficient"] == coeff]
        if row.empty or not str(row["reviewed_by"].iloc[0]).strip():
            missing_reviews.append(coeff)

    if missing_reviews:
        print("ERROR: the following load-bearing coefficients are missing expert review:")
        for c in missing_reviews:
            print(f"   - {c}")
        return False
    print("All load-bearing coefficients have review entries.")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_coefficient_reviews() else 1)
