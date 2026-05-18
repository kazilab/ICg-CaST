"""Oracle vs biology disagreement audit.

Compares the frozen reference oracle against the biological risk equation
and surfaces measurable disagreement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from icg_cast.biology.biological_risk_equation import biological_risk_equation
from icg_cast.oracle.reference_risk_oracle import reference_risk_oracle


def compute_oracle_biology_disagreement(
    cohort: pd.DataFrame,
    use_priors: bool = False,
    seed: int = 42,
    output_dir: str = "outputs/audit"
) -> dict[str, Any]:
    """
    Compare reference oracle vs biological risk equation on a cohort.
    
    Returns disagreement metrics and saves detailed audit to disk.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    oracle_risks = []
    biology_risks = []
    
    for _, row in cohort.iterrows():
        oracle = reference_risk_oracle(
            dna_adducts=row.get("state_final_DNA_adducts", 0),
            ros=row.get("state_final_ROS", 0),
            inflammation=row.get("state_final_inflammation", 0),
            epigenetic_age=row.get("state_final_epigenetic_age", 0),
            proliferation=row.get("state_final_proliferation", 0),
            mutation_rate=row.get("state_final_mutation_rate", 0),
            clone_fraction=row.get("state_final_clone_fraction", 0),
            driver_count=row.get("state_final_driver_count_proxy", 0),
            immune_clearance=row.get("state_final_immune_clearance", 0),
        )
        
        biology = biological_risk_equation(
            dna_adducts=row.get("state_final_DNA_adducts", 0),
            ros=row.get("state_final_ROS", 0),
            inflammation=row.get("state_final_inflammation", 0),
            epigenetic_age=row.get("state_final_epigenetic_age", 0),
            proliferation=row.get("state_final_proliferation", 0),
            mutation_rate=row.get("state_final_mutation_rate", 0),
            clone_fraction=row.get("state_final_clone_fraction", 0),
            driver_count=row.get("state_final_driver_count_proxy", 0),
            immune_clearance=row.get("state_final_immune_clearance", 0),
            use_priors=use_priors,
            seed=seed,
        )
        
        oracle_risks.append(oracle)
        biology_risks.append(biology)
    
    oracle_risks = np.array(oracle_risks)
    biology_risks = np.array(biology_risks)
    
    disagreement = np.abs(oracle_risks - biology_risks)
    
    metrics = {
        "mean_absolute_disagreement": float(np.mean(disagreement)),
        "max_disagreement": float(np.max(disagreement)),
        "correlation": float(np.corrcoef(oracle_risks, biology_risks)[0, 1]),
        "oracle_mean": float(np.mean(oracle_risks)),
        "biology_mean": float(np.mean(biology_risks)),
        "n_samples": len(cohort),
    }
    
    # Save detailed audit
    audit_df = pd.DataFrame({
        "oracle_risk": oracle_risks,
        "biology_risk": biology_risks,
        "disagreement": disagreement,
    })
    audit_df.to_csv(f"{output_dir}/oracle_biology_disagreement.csv", index=False)
    
    # Save summary
    with open(f"{output_dir}/oracle_biology_summary.json", "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    
    return metrics


def print_audit_summary(metrics: dict[str, Any]):
    """Pretty print audit results."""
    print("\n=== Oracle vs Biology Disagreement Audit ===")
    print(f"Mean absolute disagreement: {metrics['mean_absolute_disagreement']:.4f}")
    print(f"Max disagreement:           {metrics['max_disagreement']:.4f}")
    print(f"Pearson correlation:        {metrics['correlation']:.4f}")
    print(f"Oracle mean risk:           {metrics['oracle_mean']:.4f}")
    print(f"Biology mean risk:          {metrics['biology_mean']:.4f}")
    print(f"Samples audited:            {metrics['n_samples']}")
    print("Audit saved to: outputs/audit/")