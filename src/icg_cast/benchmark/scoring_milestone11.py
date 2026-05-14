"""
Updated Benchmark Scoring with Oracle vs Biology Comparison (Milestone 11)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from icg_cast.biology.biological_risk_equation import biological_risk_equation


def evaluate_with_oracle_and_biology(
    cohort: pd.DataFrame,
    predictions: pd.Series,
    use_priors: bool = False,
) -> dict[str, Any]:
    """
    Evaluate model predictions against both the frozen oracle and biological equation.
    """
    # Original oracle-based labels
    oracle_labels = cohort["future_cancer_transition_event"]
    
    # Recompute labels using biological equation
    biology_labels = cohort.apply(
        lambda row: 1 if biological_risk_equation(
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
        ) > 0.5 else 0,
        axis=1
    )
    
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    results = {
        "oracle_auroc": roc_auc_score(oracle_labels, predictions),
        "biology_auroc": roc_auc_score(biology_labels, predictions),
        "oracle_ap": average_precision_score(oracle_labels, predictions),
        "biology_ap": average_precision_score(biology_labels, predictions),
        "label_disagreement_rate": (oracle_labels != biology_labels).mean(),
    }
    
    return results