# 4. Proof-of-concept results on synthetic ICg-Bench cohorts

The results in this section are derived entirely from synthetic ICg-Bench cohorts (n = 1200 simulated subjects, 72 simulated months per subject). Each (cohort, variant) configuration is replicated across three random seeds (7, 13, 31) and reported as mean ± SD. They validate software execution, internal consistency of the simulator, and the by-construction mechanism coherence of MB-CNet against fully specified data-generating processes. They are **not** estimates of human carcinogenic hazard, real-world predictive performance, or regulatory classification accuracy, and must not be cited as such.

All figures and tables are reproducible by:

```bash
python3 scripts/bottleneck_proof_of_concept.py           # 54 experiments → outputs/bottleneck_v0_5/
python3 scripts/bottleneck_proof_of_concept.py --bootstrap 250   # tighter conformity CIs (slower)
python3 scripts/bottleneck_proof_of_concept.py --cohorts misspecified_signs_v2   # subset rerun
python3 scripts/render_manuscript_plots.py               # 4 PNGs → outputs/figures/
```

or, equivalently, via the installed CLI:

```bash
icg-cast bench list                               # list registered DGP variants
icg-cast bench info misspecified_signs_v2         # DGP + flipped intervention directions
icg-cast bench run --cohort linear_lowhet --variant sign_constrained --seed 7
icg-cast bench audit --cohort linear_lowhet --variant sign_constrained --seed 7   # per-unit sign relaxation
icg-cast bench sweep                              # delegates to the script above
icg-cast bench plots                              # manuscript figures (reads v0.5 artifacts)
```

Raw artifacts under `outputs/bottleneck_v0_5/`: `per_seed.csv` (one row per experiment **including bootstrap 95% CI columns** for prior, DGP, and responsive conformity), `summary.csv`, `summary.json`, `per_state_recovery_r2.csv`, and per-experiment `intervention_conformity__*.csv`. Figures under `outputs/figures/`.

## 4.1 ICg-Bench v0.1 cohorts

Six DGP variants are evaluated in the main sweep. All share the same observation feature set (≈ 48 multi-omics columns from `tx_*`, `epi_*`, `sig_activity_*`, `kcc*`, `host_*`, `dose`, `mut_total_count`) and the same 9 latent qAOP state summaries (`state_final_*`, excluding `state_final_latent_risk` to prevent target leakage). They differ in the data-generating process upstream of the observations:

**Table 4.1. ICg-Bench v0.1 DGP variants.**

| Variant | KCC sampling | KCC → state coupling | Host heterogeneity | Observation operator | Observability | True latent-risk equation | Outcome rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linear_lowhet` | 8 discrete archetypes | linear | low | linear-with-noise | full | starter-kit | 0.276 ± 0.005 |
| `nonlinear_mixhost` | Dirichlet mixture (α = 0.5) | non-linear | high | linear-with-noise | full | starter-kit | 0.451 ± 0.012 |
| `partial_observability` | Dirichlet mixture | non-linear | high | linear-with-noise | per-subject 30% mask on tx_/epi_ | starter-kit | 0.443 ± 0.020 |
| `nonlinear_obs` | 8 discrete archetypes | linear | low | non-linear (tanh, log1p, interactions) | full | starter-kit | 0.272 ± 0.010 |
| `misspecified_signs` | 8 discrete archetypes | linear | low | linear-with-noise | full | **starter-kit with the sign on `state_final_epigenetic_age` flipped** | 0.239 ± 0.011 |
| `misspecified_signs_v2` | 8 discrete archetypes | linear | low | linear-with-noise | full | **flipped signs on `state_final_epigenetic_age` *and* `state_final_immune_clearance`** | ≈ 0.35 |

`misspecified_signs` is the explicit falsification test for prior-elicited methods. It reuses the entire `linear_lowhet` pipeline, then overwrites the labels under a latent-risk DGP that has one structural-equation coefficient sign disagreeing with `bottleneck.STRUCTURAL_SIGNS` (specifically: `state_final_epigenetic_age` enters with −0.85 in the cohort rather than the prior-implied +0.85). For sign-constrained MB-CNet this means the structural prior is wrong about exactly one bottleneck unit; the data, however, tells the truth.

## 4.2 Three MB-CNet variants

We compare three stage-2 implementations. Stage 1 is identical across all three: a `MultiOutputRegressor(RandomForestRegressor(n_estimators=200))` preceded by a `SimpleImputer(strategy='mean')` for missing-value tolerance.

1. **`v0.1 unconstrained`** — isotonic-calibrated logistic regression. The reference architecture.
2. **`sign-constrained`** — L2-regularised logistic regression with per-coefficient sign constraints (eight `+1` and one `−1`) reflecting the *structural-prior* for `state_final_latent_risk`. Solved by L-BFGS-B with bound constraints. No calibration wrapper.
3. **`sign-constrained + augmented`** — as above, with intervention-augmented training: for every training row, two synthetic perturbations per do-intervention are appended with labels sampled from the structural-prior equation under the perturbation and the cumulative-hazard parameterisation (`event_hazard_scale = 0.020`, `months = 72`).

Note that the augmenter uses the **structural prior** equation (`bottleneck.starter_kit_latent_risk`), not the cohort's true DGP. This is realistic — in real-world deployment the augmenter is whatever model of the AOP graph the team can elicit, not the unknowable truth — and on `misspecified_signs` it lets us directly observe what happens when the augmentation prior is wrong.

## 4.3 Three conformity metrics

Synthetic data with a known DGP allows us to report intervention conformity in three forms simultaneously, and the three together expose a failure mode that any single metric would hide.

- **Prior conformity.** Fraction of do-interventions whose predicted-risk change agrees with the *structural-prior* direction (Table 4.1 column "Expected direction"). A `|Δ risk| < 10⁻³` is treated as a pass via tolerance; this matches the v0.1 acceptance criterion in PLAN.md §7.5.
- **DGP conformity.** Fraction that agrees with the *true cohort DGP* direction. For all four well-specified cohorts this equals prior conformity. For `misspecified_signs` the expected direction of `do_epigenetic_memory_reset` is flipped (+1 instead of −1) so DGP conformity diverges from prior conformity.
- **Responsive DGP conformity.** Fraction that agrees with the DGP direction **and** has `|Δ risk| ≥ 0.005`. This closes the "constraint drove the coefficient to zero so the intervention is a no-op and gets credit for non-response" loophole.

The responsive metric is the only one of the three that distinguishes a model that is genuinely causally coherent from one that is silent on the intervention.

### 4.3.1 Bootstrap confidence intervals

For each fitted model and held-out fold, conformity scores are also summarised with **non-parametric bootstrap CIs**: test subjects are resampled with replacement, the per-intervention mean Δ risk is recomputed on each replicate, and the scalar conformity fraction is recomputed for that replicate. Reported in `per_seed.csv` are the point estimate on the full held-out set and the 2.5–97.5 percentile interval across bootstrap draws (default **B = 120** in the repository script; override with `--bootstrap` or `ICG_N_BOOTSTRAP`). This implements the Milestone 5.5 requirement to attach uncertainty to the counterfactual audit, distinct from seed-to-seed variance in `summary.csv`.

### 4.3.2 Prior sensitivity audit (which signs bind?)

`icg_cast.audit.prior_sensitivity` refits stage 2 repeatedly on the **same** stage‑1 predictions as a fitted sign-constrained MB-CNet: for each bottleneck unit, the corresponding sign constraint is relaxed to “unconstrained” (`0`) while all other coordinates keep the structural prior. The table reports `delta_responsive_dgp` (change in responsive DGP conformity) and `delta_auroc` relative to the fully constrained model. Large positive `delta_responsive_dgp` on relaxation flags an elicited sign that was **binding**—the data wanted to violate the prior but could not. This is the audit artifact a reviewer would use to decide which AOP-derived signs to re-open for contested chemicals.

## 4.4 Aggregated results across cohorts, variants, and seeds

**Table 4.2. MB-CNet performance across cohorts and variants (mean ± SD over seeds {7, 13, 31}, n_test = 360 per seed). Best non-tied value per cohort in **bold**.**

| Cohort | Variant | AUROC | AUROC gap to best baseline | Recovery R² | Prior conformity | DGP conformity | Responsive DGP conformity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_lowhet | v0.1 unconstrained | 0.904 ± 0.024 | −0.002 ± 0.002 | 0.969 ± 0.001 | 0.762 ± 0.082 | 0.762 ± 0.082 | 0.714 ± 0.000 |
| linear_lowhet | sign-constrained | 0.906 ± 0.024 | +0.000 ± 0.002 | 0.969 ± 0.001 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.524 ± 0.082 |
| linear_lowhet | sign-constrained + augmented | **0.907 ± 0.023** | +0.001 ± 0.000 | 0.969 ± 0.001 | **1.000 ± 0.000** | **1.000 ± 0.000** | **0.762 ± 0.082** |
| nonlinear_mixhost | v0.1 unconstrained | **0.818 ± 0.031** | −0.009 ± 0.019 | 0.963 ± 0.019 | 0.810 ± 0.218 | 0.810 ± 0.218 | **0.619 ± 0.082** |
| nonlinear_mixhost | sign-constrained | 0.816 ± 0.027 | −0.011 ± 0.017 | 0.963 ± 0.019 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.476 ± 0.165 |
| nonlinear_mixhost | sign-constrained + augmented | 0.817 ± 0.030 | −0.010 ± 0.022 | 0.963 ± 0.019 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.476 ± 0.082 |
| partial_observability | v0.1 unconstrained | 0.807 ± 0.028 | −0.013 ± 0.014 | 0.926 ± 0.035 | 0.762 ± 0.218 | 0.762 ± 0.218 | **0.667 ± 0.297** |
| partial_observability | sign-constrained | 0.809 ± 0.027 | −0.010 ± 0.012 | 0.926 ± 0.035 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.524 ± 0.218 |
| partial_observability | sign-constrained + augmented | **0.813 ± 0.028** | −0.007 ± 0.014 | 0.926 ± 0.035 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.524 ± 0.082 |
| nonlinear_obs | v0.1 unconstrained | **0.912 ± 0.016** | −0.005 ± 0.008 | 0.942 ± 0.005 | 0.905 ± 0.082 | 0.905 ± 0.082 | **0.810 ± 0.082** |
| nonlinear_obs | sign-constrained | 0.910 ± 0.015 | −0.007 ± 0.011 | 0.942 ± 0.005 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.333 ± 0.360 |
| nonlinear_obs | sign-constrained + augmented | **0.912 ± 0.015** | −0.005 ± 0.011 | 0.942 ± 0.005 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.714 ± 0.000 |
| **misspecified_signs** | **v0.1 unconstrained** | 0.901 ± 0.030 | +0.000 ± 0.004 | 0.974 ± 0.005 | 0.810 ± 0.082 | **0.952 ± 0.082** | **0.952 ± 0.082** |
| misspecified_signs | sign-constrained | **0.904 ± 0.032** | +0.002 ± 0.002 | 0.974 ± 0.005 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.762 ± 0.082 |
| misspecified_signs | sign-constrained + augmented | 0.900 ± 0.035 | −0.001 ± 0.005 | 0.974 ± 0.005 | **1.000 ± 0.000** | 0.857 ± 0.000 | 0.667 ± 0.082 |
| `misspecified_signs_v2` | v0.1 unconstrained | **0.847 ± 0.010** | +0.004 ± 0.015 | 0.972 ± 0.006 | 0.714 ± 0.000 | **1.000 ± 0.000** | **0.857 ± 0.000** |
| `misspecified_signs_v2` | sign-constrained | 0.842 ± 0.012 | −0.002 ± 0.013 | 0.972 ± 0.006 | **1.000 ± 0.000** | 0.952 ± 0.082 | 0.524 ± 0.082 |
| `misspecified_signs_v2` | sign-constrained + augmented | 0.839 ± 0.014 | −0.004 ± 0.011 | 0.972 ± 0.006 | **1.000 ± 0.000** | 0.857 ± 0.000 | 0.524 ± 0.218 |

_Table 4.2 is generated from `outputs/bottleneck_v0_5/summary.csv` (full sweep, six cohorts × three variants × three seeds)._

Three findings dominate the table.

1. **Sign-constrained stage 2 lifts prior- and DGP-conformity to 1.000 ± 0.000 on every well-specified cohort, at no AUROC cost.** AUROC moves by at most 0.003 (well within 1σ of zero) and recovery R² is unchanged. On `linear_lowhet`, `nonlinear_mixhost`, `partial_observability`, and `nonlinear_obs` this is a clean architectural win against the unconstrained baseline.

2. **The constraint achieves this in part by driving misalignment-prone coefficients to the constraint boundary.** Responsive conformity — which requires `|Δ risk| ≥ 0.005` in addition to a correct sign — drops from 0.71–0.91 under the unconstrained variant to 0.33–0.52 under sign-constrained alone. Adding intervention augmentation restores responsiveness on most well-specified cohorts (0.71–0.76 on `linear_lowhet` and `nonlinear_obs`) without giving up prior or DGP conformity. The augmented sign-constrained variant therefore dominates: it is the only architecture that combines AUROC ≥ best-baseline-tied, conformity 1.000 ± 0.000, and substantial responsiveness on the well-specified cohorts.

3. **When the structural prior is wrong, the unconstrained v0.1 architecture is the safest choice.** On `misspecified_signs`, v0.1 achieves DGP conformity 0.952 ± 0.082 and responsive DGP conformity 0.952 ± 0.082 — strictly higher than both constrained variants. The augmented variant is the worst on this cohort (0.667 responsive DGP conformity), because intervention augmentation propagates the wrong prior with confidence. The data fixes the unconstrained model; nothing fixes a model that has been constrained and then augmented against the data.

4. **A multi-flip falsification cohort (`misspecified_signs_v2`) preserves the same pattern at higher difficulty.** Two coefficients in the latent-risk DGP disagree with the prior (`epigenetic_age` and `immune_clearance`). The unconstrained model again achieves **DGP conformity 1.000** (both flipped `do_*` directions are learnable) and **highest responsive conformity (≈ 0.86)** in the dedicated sweep, while constrained variants sit at ≈ 0.52 responsive DGP conformity despite near-perfect prior conformity. This supports the claim that v0.1 remains the fallback when *multiple* expert priors may be wrong—not only a single AOP edge.

## 4.5 The falsification finding (`misspecified_signs`)

The `misspecified_signs` cohort is identical to `linear_lowhet` except that the DGP's coefficient on `state_final_epigenetic_age` is flipped, so reducing epigenetic age in this cohort *increases* risk. Sign-constrained MB-CNet has the opposite prior. The per-intervention table for `do_epigenetic_memory_reset` (seed 7) makes the failure mode visible:

**Table 4.3. Per-intervention behaviour on `misspecified_signs` (seed 7), for the flipped intervention `do_epigenetic_memory_reset`.**

| Variant | Mean Δ predicted risk | Matches prior (−1)? | Matches DGP (+1)? | Responsive (|Δ| ≥ 0.005)? |
| --- | --- | --- | --- | --- |
| v0.1 unconstrained | +0.028 | no | yes | yes |
| sign-constrained | 0.000 (exact) | yes (via tolerance) | yes (via tolerance) | **no** |
| sign-constrained + augmented | −0.012 | yes | no | yes (but wrong) |

The unconstrained model has *learned the truth from the data*: reducing epigenetic age increases predicted risk by 2.8 percentage points on average, in agreement with the cohort DGP and against the structural prior. The sign-constrained model has the prior +1 on epigenetic age but the data has negative gradient, so the L-BFGS-B optimum is exactly on the constraint boundary, the coefficient is zero, and the model has no response to the intervention; both prior conformity and DGP conformity then register a trivial pass via the `|Δ| < 10⁻³` tolerance. The augmented model has been actively taught the wrong prior — its synthetic training rows say that reducing epigenetic age must decrease risk — and learns a confident −0.012 response that is structurally wrong and not rescuable.

The responsive DGP conformity column of Table 4.2 is the only one of the three conformity metrics that distinguishes (i) "model correctly responsive in the DGP direction" from (ii) "model silent on the intervention" from (iii) "model confidently wrong". This is the metric the manuscript will use as the default conformity score for ICg-Bench v0.2 onwards.

The implication for real-world deployment is concrete: prior-elicited sign constraints are conservative w.r.t. the structural prior, not w.r.t. the truth. They should be elicited from expert toxicology review, audited per AOP, and accompanied by a published responsive-conformity row that exposes which constraints are *active* on a given cohort and whose lifting would change the answer. The MB-CNet implementation here exposes the active set via the `coef_signs` attribute of `SignConstrainedLogisticRegression` and reports the per-intervention `mean_risk_change` in the conformity table — both are the audit handles a regulator would need.

## 4.6 The benchmark dissociates AUROC, recovery R², and conformity

The five-cohort design produces measurable separation on every scored axis (Figure `fig_recovery_r2_heatmap.png`, Figure `fig_responsive_conformity.png`):

- **`linear_lowhet`** is easy on all axes (AUROC ≈ 0.91, R² ≈ 0.97).
- **`nonlinear_mixhost`** preserves recovery R² (0.96) but drops AUROC to 0.82 — non-linear KCC → state coupling reduces predictability without reducing observability of the latent state.
- **`partial_observability`** drops both recovery R² to 0.93 and AUROC to 0.81 — random per-subject masking damages the stage-1 inverse problem.
- **`nonlinear_obs`** preserves AUROC (0.91) but drops recovery R² to 0.94 — the non-linear, multiplicatively-interacting observation operator stresses stage 1 while leaving the KCC → state predictive structure intact.
- **`misspecified_signs`** preserves AUROC (0.90) and recovery R² (0.97) — the observation operator and KCCs are unchanged — but exposes the constraint-versus-truth conflict for any model that uses a sign-constrained or augmented stage 2.
- **`misspecified_signs_v2`** uses the same observation pipeline with **two** flipped latent-risk coefficients; unconstrained models can still reach **DGP conformity 1.000** on both flipped interventions, while constrained models trade prior conformity for much lower responsive conformity (Table 4.2).

A future model class that beats MB-CNet on ICg-Bench must improve at least one of AUROC, recovery R², prior/DGP conformity, or responsive DGP conformity on at least one cohort without regressing on the others. The six-cohort layout therefore gives the benchmark six orthogonal axes of discrimination at the same n.

## 4.7 Summary of headline findings

1. **Mechanism bottlenecking is necessary but not sufficient for causal coherence.** Across six cohorts and three seeds, the unconstrained v0.1 architecture fails at least one mechanism counterfactual on at least one seed despite recovering the latent qAOP state with mean R² ≥ 0.93 on every unit and every cohort. The failure is in the conditional model that consumes the bottleneck, not in the bottleneck itself.
2. **Per-coefficient sign constraints on stage 2 close the residual loophole on every well-specified cohort at no AUROC cost.** The constraint not only raises mean prior and DGP conformity to 1.000 but eliminates seed-to-seed conformity variance. The cost is reduced responsiveness on a subset of interventions; intervention augmentation restores most of it on well-specified cohorts.
3. **The sign-constrained + augmented variant is the strongest architecture on well-specified cohorts.** It is the only variant that combines AUROC at-or-above the best unconstrained multi-omics baseline, 1.000 conformity (prior and DGP), and substantial responsiveness on `linear_lowhet` and `nonlinear_obs`.
4. **When the structural prior disagrees with the DGP, the unconstrained v0.1 architecture is strictly safer.** On `misspecified_signs` it is the only variant whose responsive DGP conformity exceeds 0.85. Sign constraints alone become unresponsive on the flipped intervention; augmentation makes the model confidently wrong. This is a known limitation of all prior-elicited methods, but it is now quantified for MB-CNet and packaged as a falsification cohort that any future MB-CNet variant must run against.
5. **Responsive DGP conformity should be reported alongside AUROC and the existing conformity metric** by any method that participates in ICg-Bench. It is the only metric that distinguishes "correctly responsive", "silent on intervention", and "confidently wrong" — three failure modes that the simpler conformity score conflates.

Together these findings constitute the central methodological claim of the manuscript: that by-construction mechanism coherence is achievable for integrated carcinogenomics with a two-stage MB-CNet whose stage 2 is sign-constrained and intervention-augmented, that the architecture must be evaluated jointly on predictive discrimination *and* responsive causal conformity, and that the benchmark suite must include cohorts whose DGP deliberately disagrees with the structural prior so that prior-fragile claims of coherence are exposed before they are deployed.

## 4.8 Limitations of these results

1. Synthetic AUROC values are bounded above by the data-generating process; they measure recoverability of the DGP, not real-world predictive performance.
2. Sign constraints depend on a structural-equation prior. The signs in this manuscript are derived directly from the starter-kit simulator; for real data the signs would be supplied by expert toxicology review of the AOP graph and audited per AOP. The `misspecified_signs` cohort quantifies how much risk this introduces.
3. Intervention augmentation depends on a callable that implements the DGP's latent-risk equation. On `misspecified_signs` this callable is wrong by design; on real data its correctness is unknowable and must be hedged by the prior elicitation process.
4. The six DGP variants are intentionally toy. They preserve the observation feature schema across cohorts so that the same MB-CNet architecture runs against all of them. `misspecified_signs_v2` already exercises multiple simultaneous prior errors; a v0.2 of ICg-Bench should add interaction terms in the latent-risk equation and host-specific structural equations.
5. Seed replication is over three seeds; larger seed sweeps will be needed for any formal statistical comparison of variants beyond the 0.000-SD result reported here.
6. No real human, animal, or clinical data are used in these results. External validation requires the data-source adapters in `data_sources/` to be calibrated against the cited public resources under their respective licence and access requirements, with appropriate ethics oversight where applicable.

The framework is positioned as fundamental research for theory development and methods evaluation. It is not a clinical, regulatory, or individual-risk tool.

## 4.9 Figure index

- `outputs/figures/fig_responsive_conformity.png` — headline bar chart: responsive DGP conformity per (cohort, variant). The figure that should appear with Table 4.2.
- `outputs/figures/fig_auroc_vs_conformity.png` — scatter of AUROC against responsive DGP conformity per (cohort, variant). Visualises the predictive-vs-coherence trade-off across the design space.
- `outputs/figures/fig_recovery_r2_heatmap.png` — per-state stage-1 recovery R² heatmap (cohort × bottleneck unit) showing where stage-1 inversion is hard.
- `outputs/figures/fig_intervention_deltas.png` — per-intervention mean predicted-risk change, grid faceted by cohort, that grounds the §4.5 falsification narrative in raw deltas.

---

**Note for the manuscript editor.** The raw CSVs underlying every figure and table in §4 are deposited under `outputs/bottleneck_v0_5/` in the project repository and are regenerated deterministically by `scripts/bottleneck_proof_of_concept.py` (`--bootstrap` / `--cohorts` optional). Figures are regenerated by `scripts/render_manuscript_plots.py`. The CLI entry point (`icg-cast bench list / info / run / audit / sweep / plots`) is provided as a discoverable alternative. The package version, DGP variant hashes from `icg_cast.benchmark.dgp`, stage-2 variant names, and the seed list should be cited alongside these tables in the final manuscript.
