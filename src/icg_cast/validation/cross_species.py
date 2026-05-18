"""Cross-species human-relevance estimation.

PLAN.md reference: section 9.4.

The Human Relevance Transfer Index (HRTI) summarises whether key events
observed in a rodent or in-vitro assay set are likely to translate to humans
under a user-supplied conservation table::

    HRTI = conserved_human_KE_activation
          / (conserved_human_KE_activation + rodent_specific_KE_activation)

The canonical score preserves that denominator. Conserved key events that are
active in rodents but inactive in humans are reported separately and in a
coverage-adjusted score, because they can represent species divergence rather
than missing evidence.

Inputs are intentionally explicit and column-typed: this module reasons about
mechanism-level activations supplied by the caller. It does **not** wrap a
classifier, look up KE conservation databases, or make regulatory claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HRTIResult:
    """Result of an HRTI computation."""

    score: float
    """Index value in ``[0, 1]``; ``nan`` when both numerators are zero."""

    n_conserved_human: int
    """Count of key events flagged conserved between species **and** active in humans."""

    n_rodent_specific: int
    """Count of key events flagged rodent-specific and active in the rodent assay."""

    reasons: list[str]
    """Per-key-event annotations explaining the contribution."""

    n_conserved_inactive_in_human: int = 0
    """Count of conserved KEs active in rodents but inactive in humans."""

    coverage_adjusted_score: float = float("nan")
    """Score penalizing conserved rodent-active KEs that are inactive in humans."""

    def to_dict(self) -> dict[str, object]:
        return {
            "hrti_score": float(self.score),
            "coverage_adjusted_hrti_score": float(self.coverage_adjusted_score),
            "n_conserved_human": int(self.n_conserved_human),
            "n_rodent_specific": int(self.n_rodent_specific),
            "n_conserved_inactive_in_human": int(self.n_conserved_inactive_in_human),
            "reasons": list(self.reasons),
        }


_REQUIRED_COLUMNS = {"key_event", "conservation", "human_activation", "rodent_activation"}


def human_relevance_transfer_index(
    table: pd.DataFrame,
    activation_threshold: float = 0.5,
) -> HRTIResult:
    """Compute HRTI from a long-form key-event activation table.

    Args:
        table: DataFrame with columns
            ``[key_event, conservation, human_activation, rodent_activation]``.
            ``conservation`` is one of ``"conserved"`` or ``"rodent_specific"``
            (other labels are treated as ``"unknown"`` and ignored in the
            numerators).
        activation_threshold: an activation value at or above this is treated
            as "active". Values in ``[0, 1]`` are recommended.

    Returns:
        An :class:`HRTIResult` with the score, contributing counts, and a
        per-key-event reason list. ``HRTIResult.score`` is ``nan`` when both
        contributing counts are zero (insufficient evidence). Conserved key
        events that are rodent-active but human-inactive are not part of the
        canonical denominator, but are exposed as
        ``n_conserved_inactive_in_human`` and penalize
        ``coverage_adjusted_score``.
    """
    missing = _REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise KeyError(f"HRTI table missing required columns: {sorted(missing)}")

    conserved_human = 0
    rodent_specific = 0
    conserved_inactive_in_human = 0
    reasons: list[str] = []

    for row in table.to_dict(orient="records"):
        key = str(row["key_event"])
        conservation = str(row["conservation"]).strip().lower()
        human_active = float(row["human_activation"]) >= activation_threshold
        rodent_active = float(row["rodent_activation"]) >= activation_threshold

        if conservation == "conserved" and human_active:
            conserved_human += 1
            reasons.append(f"{key}: conserved KE active in human (+1 numerator)")
        elif conservation == "rodent_specific" and rodent_active:
            rodent_specific += 1
            reasons.append(f"{key}: rodent-specific KE active (+1 denominator only)")
        elif conservation == "conserved" and rodent_active and not human_active:
            conserved_inactive_in_human += 1
            reasons.append(
                f"{key}: conserved KE active in rodent but inactive in human "
                "(coverage-adjusted denominator only)"
            )
        elif conservation == "conserved" and not human_active:
            reasons.append(f"{key}: conserved but inactive in both species (ignored)")
        elif conservation == "rodent_specific" and not rodent_active:
            reasons.append(f"{key}: rodent-specific but inactive (ignored)")
        else:
            reasons.append(f"{key}: conservation '{conservation}' not scored")

    denominator = conserved_human + rodent_specific
    score = float(conserved_human / denominator) if denominator else float("nan")
    adjusted_denominator = denominator + conserved_inactive_in_human
    coverage_adjusted_score = (
        float(conserved_human / adjusted_denominator)
        if adjusted_denominator
        else float("nan")
    )
    return HRTIResult(
        score=score,
        n_conserved_human=conserved_human,
        n_rodent_specific=rodent_specific,
        n_conserved_inactive_in_human=conserved_inactive_in_human,
        coverage_adjusted_score=coverage_adjusted_score,
        reasons=reasons,
    )
