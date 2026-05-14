"""CalibrationBundle: passive container of opt-in calibration overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import KCC_NAMES


@dataclass
class CalibrationBundle:
    """Opt-in overrides for the synthetic simulator and theory graph.

    Each field is optional. Unset fields fall back to the default simulator
    constants. The bundle is JSON-serialisable so a calibration produced by the
    ``icg-cast calibrate`` CLI can be reused by later ``simulate`` / ``graph``
    runs.

    Attributes:
        signature_labels: 96 trinucleotide context labels (must match the
            simulator's mutation context order if used as a drop-in
            replacement).
        signature_profiles: mapping ``signature_name -> length-96 probability
            vector``. The simulator references the toy names ``aging``,
            ``SBS4_like``, ``SBS24_like``, ``SBS22_like``, ``oxidative_like``;
            if you provide COSMIC-named profiles you must also pass a name map
            or supply replacements for those toy keys.
        archetype_kcc: mapping ``archetype_name -> 10-tuple of KCC values in
            [0, 1]``. Overrides ``constants.ARCHETYPE_KCC`` when used.
        transcript_module_priors: optional long-form table serialised as a list
            of records, produced by ``calibrate_transcript_modules_from_lincs``.
            Stored for downstream use; not currently wired into the simulator's
            transcriptomic generation in v0.1.
        graph_edges: optional list of ``{"source": str, "target": str, ...}``
            records to merge into the default theory graph.
        graph_node_attributes: optional ``node_id -> dict`` metadata merged
            onto graph nodes.
        provenance: ``adapter_name -> provenance dict`` from each input
            ``DataSourceBundle``.
    """

    signature_labels: list[str] | None = None
    signature_profiles: dict[str, list[float]] | None = None
    archetype_kcc: dict[str, list[float]] | None = None
    transcript_module_priors: list[dict[str, Any]] | None = None
    graph_edges: list[dict[str, Any]] | None = None
    graph_node_attributes: dict[str, dict[str, Any]] | None = None
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def signature_profile_arrays(self) -> tuple[list[str], dict[str, np.ndarray]] | None:
        """Return ``(labels, profiles)`` as numpy arrays, or ``None`` if unset."""
        if self.signature_labels is None or self.signature_profiles is None:
            return None
        if len(self.signature_labels) != 96:
            raise ValueError("calibrated signature_labels must contain 96 entries")
        profiles: dict[str, np.ndarray] = {}
        for name, values in self.signature_profiles.items():
            arr = np.asarray(values, dtype=float)
            if arr.size != 96:
                raise ValueError(f"signature {name!r} must have 96 entries; got {arr.size}")
            if (arr < 0).any():
                raise ValueError(f"signature {name!r} contains negative values")
            total = float(arr.sum())
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"signature {name!r} has non-positive total mass")
            profiles[name] = arr / total
        return list(self.signature_labels), profiles

    def archetype_kcc_arrays(self) -> dict[str, tuple[float, ...]] | None:
        """Return calibrated archetype KCC vectors validated to length 10 in ``[0, 1]``."""
        if not self.archetype_kcc:
            return None
        out: dict[str, tuple[float, ...]] = {}
        for name, values in self.archetype_kcc.items():
            arr = np.asarray(values, dtype=float)
            if arr.size != len(KCC_NAMES):
                raise ValueError(
                    f"archetype {name!r} must have {len(KCC_NAMES)} KCC values; got {arr.size}"
                )
            if (arr < 0).any() or (arr > 1).any():
                raise ValueError(f"archetype {name!r} has KCC values outside [0, 1]")
            out[name] = tuple(float(v) for v in arr)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_labels": self.signature_labels,
            "signature_profiles": self.signature_profiles,
            "archetype_kcc": self.archetype_kcc,
            "transcript_module_priors": self.transcript_module_priors,
            "graph_edges": self.graph_edges,
            "graph_node_attributes": self.graph_node_attributes,
            "provenance": self.provenance,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def load_calibration_bundle(path: str | Path) -> CalibrationBundle:
    """Load a previously-saved calibration bundle from JSON."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationBundle(
        signature_labels=obj.get("signature_labels"),
        signature_profiles=obj.get("signature_profiles"),
        archetype_kcc=obj.get("archetype_kcc"),
        transcript_module_priors=obj.get("transcript_module_priors"),
        graph_edges=obj.get("graph_edges"),
        graph_node_attributes=obj.get("graph_node_attributes"),
        provenance=obj.get("provenance", {}),
    )
