"""Milestone 13 review validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def validate_coefficient_reviews(
    review_file: str = "materials/coefficient_review.csv",
    flags_file: str = "outputs/audit/coefficient_flags.csv",
) -> bool:
    """Return True when every load-bearing coefficient has a reviewer."""
    if not Path(review_file).exists():
        print(f"ERROR: {review_file} not found.")
        return False

    if not Path(flags_file).exists():
        print(f"WARNING: {flags_file} not found. Run Milestone 12 first.")
        return False

    reviews = pd.read_csv(review_file, comment="#").fillna("")
    flags = pd.read_csv(flags_file)
    required = {"coefficient", "reviewed_by", "review_date", "decision", "dissent_notes"}
    missing = required - set(reviews.columns)
    if missing:
        print(f"ERROR: {review_file} missing columns: {sorted(missing)}")
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
