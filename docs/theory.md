# Theory

ICg-CaST implements a synthetic causal scaffold for AI-integrated
carcinogenomics. The core hypothesis is that chemical exposure archetypes
perturb Key Characteristics of Carcinogens (KCCs), these perturbations move
quantitative adverse-outcome-pathway (qAOP) state variables, the states generate
multi-omic readouts, and those readouts are associated with a future synthetic
cancer-transition endpoint.

This is a theory-development simulator, not a clinical, regulatory, or chemical
safety classifier.

## Modeled Layers

The package currently represents five linked layers:

1. Exposure archetype and dose.
2. Ten KCC coordinates with archetype-specific priors and stochastic variation.
3. Host susceptibility factors for repair, antioxidant capacity, immune
   surveillance, detox balance, and baseline proliferation.
4. Monthly qAOP-like state trajectories for DNA adducts, ROS, inflammation,
   epigenetic age, proliferation, mutation rate, clone fraction, driver-count
   proxy, immune clearance, and latent risk.
5. Multi-omic readouts including transcript modules, epigenomic modules,
   mutational signature activities, 96-channel synthetic mutation contexts, and
   mutation burden.

The event label is generated from the latent state trajectory. Modeling helpers
therefore explicitly exclude latent-risk summaries and future endpoint columns
from feature sets.

## Causal Interpretation

The simulator encodes a proposed direction of mechanism:

`exposure -> KCC perturbation -> qAOP states -> omic readouts -> synthetic future transition`

The baseline models learn associations from generated features to the synthetic
endpoint. Their performance is useful for checking whether the simulated signal
is recoverable, but it is not evidence that a real exposure causes cancer.

## Counterfactual Checks

Counterfactual tests perturb feature groups that correspond to mechanism-level
interventions:

- DNA repair rescue.
- ROS and inflammation blockade.
- Epigenetic memory reset.
- Proliferation suppression.
- Immune surveillance restore.

These tests are model stress tests. They ask whether a trained model responds in
the expected direction when mechanism-linked features are shifted. They do not
estimate real treatment effects.
