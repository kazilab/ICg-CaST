"""Compatibility wrapper for the Milestone 10 calibration workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from icg_cast.calibration.coupling import calibrated_registry_from_bundle
from icg_cast.calibration.pipeline import build_calibration_bundle
from icg_cast.coefficients import registry, save_registry


def run_milestone10_calibration(
    *,
    cosmic_path: str | None = None,
    toxcast_path: str | None = None,
    toxcast_mapping: str | None = None,
    lincs_path: str | None = None,
    lincs_module_map: str | None = None,
    aopwiki_path: str | None = None,
    apply_coefficients: bool = False,
    output_dir: str = "outputs/calibration",
) -> dict[str, Any]:
    """Build a calibration bundle and optionally write calibrated coefficients."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = build_calibration_bundle(
        cosmic_path=cosmic_path,
        toxcast_path=toxcast_path,
        toxcast_mapping=toxcast_mapping,
        lincs_path=lincs_path,
        lincs_module_map=lincs_module_map,
        aopwiki_path=aopwiki_path,
    )
    bundle_path = bundle.save(outdir / "calibration_bundle.json")

    result: dict[str, Any] = {
        "bundle_path": str(bundle_path),
        "provenance": bundle.provenance,
        "coefficient_updates": None,
    }
    if apply_coefficients:
        calibrated, update_summary = calibrated_registry_from_bundle(registry(), bundle)
        coeff_path = save_registry(calibrated, outdir / "calibrated_coefficients.yaml")
        result["coefficient_updates"] = update_summary
        result["calibrated_coefficients_path"] = str(coeff_path)

    provenance_payload = dict(bundle.provenance)
    if result["coefficient_updates"] is not None:
        provenance_payload["coefficient_updates"] = result["coefficient_updates"]
    (outdir / "calibration_provenance.json").write_text(
        json.dumps(provenance_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return result

