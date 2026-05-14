# Ethics and Limitations

ICg-CaST is for synthetic theory development, software testing, and benchmark
construction. It must not be used as a clinical diagnostic system, medical
device, individual risk predictor, chemical safety classifier, regulatory
carcinogenicity classifier, or substitute for toxicology and epidemiology.

## Synthetic Scope

Default workflows generate synthetic cohorts. A synthetic event label is not a
clinical outcome, and a synthetic exposure archetype is not evidence about a real
chemical.

## Claims to Avoid

Do not use outputs from this package to claim:

- A real person has elevated cancer risk.
- A real chemical is safe or unsafe.
- A model is clinically validated.
- Synthetic benchmark performance transfers to real biology.
- Counterfactual feature perturbations estimate real treatment effects.

## Real Data

Real-data connectors, if added, should remain optional and disabled by default.
Each connector should document source, retrieval date, version, license terms,
citation notes, transformation steps, and known limitations.

## Bias and Validity Risks

The simulated parameter values reflect modeling choices. They may overstate
mechanistic separability, omit confounding, compress tissue context, simplify
dose timing, and make omic readouts cleaner than real measurements. These risks
should be surfaced in model cards and manuscripts that use the package.
