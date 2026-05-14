# 1.3 Differentiation from prior art

A theory of integrated carcinogenomics must be explicit about which of its components are genuinely novel and which assemble well-established prior work into a new framework. We separate these layers here, both to credit the literature on which the framework rests and to make our novelty claims falsifiable rather than rhetorical.

## 1.3.1 Components inherited from established literature

**Key Characteristics of Carcinogens (KCCs).** The ten KCCs were introduced by Smith et al. [1] and operationalised in IARC monograph practice [2,3]. Subsequent work has applied KCC-style mechanistic scoring to systematic reviews of individual chemicals [4,5] and to high-throughput in vitro evidence [6]. ICg-CaST inherits the ten-element KCC vector as a compact mechanistic coordinate system. We do not claim novelty for the KCC framework itself; our contribution is to treat the vector as a continuous, host-modulated input to a state-space simulator, rather than as a binary tally of mechanistic evidence.

**Quantitative Adverse Outcome Pathways (qAOPs).** The AOP construct of Molecular Initiating Events, Key Events, Key Event Relationships, and Adverse Outcomes was formalised by Ankley et al. [7] and is curated at AOP-Wiki [8] and the EPA AOP-DB [9]. Quantitative AOP modelling has been developed by Conolly et al. [10], Spinu et al. [11], Wittwehr et al. [12] and integrated into NAMs programmes such as EU-ToxRisk and PARC [13,14]. Our state-space simulator is recognisably a qAOP in this tradition: it propagates a perturbation through latent biological state variables under mechanistic sign constraints. We extend the qAOP idea by coupling it directly to a multi-omics observation model and a clonal-ecology hazard, and by exposing the state-transition function to differentiable interventions.

**Mutational signatures.** Mutational signature analysis was established by Alexandrov, Stratton and colleagues [15-17], curated in COSMIC [18], and made widely accessible by SigProfilerExtractor [19] and related tools. The starter-kit simulator emits 96-channel SBS-like profiles as toy approximations; we explicitly label these as synthetic and provide a `COSMICSignatureLoader` adapter for real reference profiles in `data_sources/cosmic.py`. No methodological novelty is claimed at this layer.

**Multi-omics AI for cancer and toxicology.** Multimodal machine learning over genomic, transcriptomic, and epigenomic data is by now standard, with prominent examples in DeepTox [20], MoleculeNet [21], the Tox21 challenge [22], CarcinoPred-EL [23] and pan-cancer integration studies on TCGA [24] and CPTAC [25]. Our observation model and baseline classifiers operate within this tradition.

**Clonal evolution of normal tissue.** The view of normal tissue as a mosaic of clones under selection is now well-supported by Martincorena et al. [26,27], Yokoyama et al. [28], Cagan et al. [29] and reviews by Vendramin et al. [30]. Our `clone_fraction` and `driver_count_proxy` states approximate this dynamic in a deliberately low-dimensional form for v0.1; a single-cell clonal-ecology module replacing the scalar abstraction is planned for a future release.

## 1.3.2 Specifically new contributions

We claim novelty for the following four contributions, each of which is testable against the prior work cited above.

1. **A single end-to-end coupled simulator** linking KCC vector → qAOP latent state → multi-omics observations → clonal ecology → cancer-transition hazard, with shared random state, a stable feature schema, and explicit provenance metadata. To our knowledge, no published open-source package provides all five layers in one differentiable pipeline. Existing qAOP simulators address one or two layers [10,11]; existing multi-omics AI-tox tools omit the qAOP latent layer [20-23]; existing clonal-evolution simulators are not coupled to exposure mechanisms [26,29].

2. **Counterfactual biological-coherence as a falsification protocol for integrated carcinogenomics.** We define a packaged evaluation protocol in which models are subjected to mechanism-specific do-interventions (e.g. `do_DNA_repair_rescue`, `do_ROS_inflammation_blockade`, `do_immune_surveillance_restore`) and scored by the fraction of interventions whose predicted-risk change has the expected sign. Related ideas exist in causal abstraction theory [31], TCAV-style concept attribution [32], and the DoWhy/EconML causal ML stack [33], but to our knowledge no equivalent protocol has been instantiated for integrated carcinogenomics or made part of a model evaluation pipeline.

3. **Mechanism-Bottleneck Causal Networks (MB-CNet).** We introduce a model class whose architecture forces predictions of `future_cancer_transition_event` to flow through a hidden layer pinned to the qAOP latent state vector. Counterfactual interventions then become do-operations on bottleneck units rather than ad-hoc feature scaling. The construction adapts Concept Bottleneck Models [34] and the causal-abstraction framework [31] to the AI-tox setting. The result is mechanism coherence *by construction* rather than as a post-hoc evaluation, which we argue is the appropriate failure modality for AI models that will be considered for use in regulatory toxicology under the NAMs paradigm.

4. **ICg-Bench: a public synthetic causal benchmark with a known data-generating process.** ICg-Bench releases versioned synthetic cohorts spanning linear and non-linear coupling, discrete and continuous KCC mixtures, low and high host heterogeneity, and full and partial multi-omics observability. It scores models on four tasks: (a) risk prediction, (b) latent-state recovery R², (c) intervention conformity, and (d) cross-host generalisation. The benchmark complements predictive challenges such as Tox21 [22] and Tox24 by adding latent-recovery and intervention tasks that are only well-defined when the DGP is known. We are not aware of a comparable causal benchmark in toxicology; the closest analogues are domain-agnostic causal-inference benchmarks such as IHDP, ACIC, and the Atlantic Causal Inference Challenge [35], none of which are biology-aware.

## 1.3.3 Summary table

| Layer | Prior art | ICg-CaST contribution | Novelty |
| --- | --- | --- | --- |
| KCC vector | Smith 2016 [1]; Guyton 2018 [4]; IARC monograph practice [2,3] | Continuous 10-dim vector, host-modulated, exposed as a causal graph node | Operationalisation, not a new theory |
| qAOP state-space | Conolly 2017 [10]; Spinu 2020 [11]; Wittwehr 2017 [12] | Transparent state-transition recursion coupled to omics and clones | Coupling, not the modelling family |
| Mutational signatures | Alexandrov 2013-2020 [15-18]; SigProfiler [19] | Toy 96-channel profiles + adapter for real COSMIC SBS | None at this layer |
| Multi-omics observation | DeepTox [20]; MoleculeNet [21]; TCGA/CPTAC [24,25] | Synthetic transcriptomic/epigenomic modules generated from latent state | None at this layer |
| Clonal ecology | Martincorena 2018/2019 [26,27]; Cagan 2022 [29] | Scalar clone-fraction with Moran-style update coupled to immune surveillance | Coupling to exposure, not the ecology model |
| End-to-end coupling | No published open-source equivalent identified | KCC → qAOP → omics → clones → hazard in one package with shared seed and schema | **New** |
| Counterfactual mechanism evaluation | DoWhy / EconML [33]; TCAV [32]; causal abstraction [31] | Packaged biological-coherence protocol with seven canonical do-interventions and a coherence score | **New** for integrated carcinogenomics |
| Mechanism-Bottleneck Network | Concept Bottleneck Models [34]; causal abstraction [31] | Two-stage CBM pinned to the qAOP latent state, with do-operations on bottleneck units | **New** for integrated carcinogenomics |
| Public causal benchmark | Tox21 [22]; IHDP / ACIC [35] | Versioned synthetic DGPs scored on four causal tasks | **New** for AI toxicology |

## 1.3.4 Boundary of the claim

We do not claim novelty for any of the underlying biological theories or for any individual machine-learning algorithm we use. The novelty is the combination — a single reproducible package with a coupled simulator, a by-construction mechanism-coherent model class, and a versioned causal benchmark — and, distinct from that combination, the two methodological objects (MB-CNet and ICg-Bench) which we believe are publishable on their own. The framework is offered as fundamental research for theory development and methods evaluation; it is not a clinical, regulatory, or individual-risk tool, and the synthetic AUROC values reported in the proof-of-concept results section measure recoverability of the assumed data-generating process, not real-world predictive performance.

---

**References (to be expanded to the manuscript bibliography style).**

[1] Smith MT, Guyton KZ, Gibbons CF, et al. Key Characteristics of Carcinogens as a Basis for Organizing Data on Mechanisms of Carcinogenesis. *Environ Health Perspect*. 2016.
[2] IARC Monographs. Preamble to the IARC Monographs (amended 2019). IARC, Lyon, 2019.
[3] Guyton KZ, Rusyn I, Chiu WA, et al. Application of the key characteristics of carcinogens in cancer hazard identification. *Carcinogenesis*. 2018.
[4] Guyton KZ, Rieswijk L, Smith MT, et al. Key characteristics approach to carcinogenic hazard identification. *Chem Res Toxicol*. 2020.
[5] Caldwell JC. Endotoxin/lipopolysaccharide and the key characteristics of carcinogens. *Toxicology*. 2019.
[6] Becker RA, Dellarco V, Seed J, et al. Quantitative weight of evidence to assess confidence in potential modes of action. *Regul Toxicol Pharmacol*. 2017.
[7] Ankley GT, Bennett RS, Erickson RJ, et al. Adverse outcome pathways: a conceptual framework. *Environ Toxicol Chem*. 2010.
[8] SAAOP / OECD. AOP-Wiki. https://aopwiki.org.
[9] Pittman ME, Edwards SW, Ives C, Mortensen HM. AOP-DB: An Adverse Outcome Pathway Database for predictive toxicology. *Reprod Toxicol*. 2018.
[10] Conolly RB, Ankley GT, Cheng W, et al. Quantitative Adverse Outcome Pathways and Their Application to Predictive Toxicology. *Environ Sci Technol*. 2017.
[11] Spinu N, Cronin MTD, Enoch SJ, et al. Quantitative adverse outcome pathway (qAOP) models for toxicity prediction. *Arch Toxicol*. 2020.
[12] Wittwehr C, Aladjov H, Ankley G, et al. How adverse outcome pathways can aid the development and use of computational prediction models for regulatory toxicology. *Toxicol Sci*. 2017.
[13] Daneshian M, Kamp H, Hengstler J, et al. Highlight report: Launch of a large integrated European in vitro toxicology project: EU-ToxRisk. *Arch Toxicol*. 2016.
[14] PARC consortium. European Partnership for the Assessment of Risks from Chemicals. https://www.eu-parc.eu.
[15] Alexandrov LB, Nik-Zainal S, Wedge DC, et al. Signatures of mutational processes in human cancer. *Nature*. 2013.
[16] Alexandrov LB, Kim J, Haradhvala NJ, et al. The repertoire of mutational signatures in human cancer. *Nature*. 2020.
[17] Nik-Zainal S, Alexandrov LB, Wedge DC, et al. Mutational processes molding the genomes of 21 breast cancers. *Cell*. 2012.
[18] COSMIC Mutational Signatures. https://cancer.sanger.ac.uk/signatures.
[19] Islam SMA, Díaz-Gay M, Wu Y, et al. Uncovering novel mutational signatures by de novo extraction with SigProfilerExtractor. *Cell Genomics*. 2022.
[20] Mayr A, Klambauer G, Unterthiner T, Hochreiter S. DeepTox: Toxicity Prediction using Deep Learning. *Front Environ Sci*. 2016.
[21] Wu Z, Ramsundar B, Feinberg EN, et al. MoleculeNet: a benchmark for molecular machine learning. *Chem Sci*. 2018.
[22] Huang R, Xia M, Sakamuru S, et al. Tox21Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways as Mediated by Exposure to Environmental Chemicals and Drugs. *Front Environ Sci*. 2016.
[23] Zhang L, Ai H, Chen W, et al. CarcinoPred-EL: novel models for predicting the carcinogenicity of chemicals using molecular fingerprints and ensemble learning methods. *Sci Rep*. 2017.
[24] Hutter C, Zenklusen JC. The Cancer Genome Atlas: Creating Lasting Value beyond Its Data. *Cell*. 2018.
[25] Edwards NJ, Oberti M, Thangudu RR, et al. The CPTAC Data Portal: A Resource for Cancer Proteomics Research. *J Proteome Res*. 2015.
[26] Martincorena I, Roshan A, Gerstung M, et al. High burden and pervasive positive selection of somatic mutations in normal human skin. *Science*. 2015.
[27] Martincorena I, Fowler JC, Wabik A, et al. Somatic mutant clones colonize the human esophagus with age. *Science*. 2018.
[28] Yokoyama A, Kakiuchi N, Yoshizato T, et al. Age-related remodelling of oesophageal epithelia by mutated cancer drivers. *Nature*. 2019.
[29] Cagan A, Baez-Ortega A, Brzozowska N, et al. Somatic mutation rates scale with lifespan across mammals. *Nature*. 2022.
[30] Vendramin R, Litchfield K, Swanton C. Cancer evolution: Darwin and beyond. *EMBO J*. 2021.
[31] Geiger A, Lu H, Icard T, Potts C. Causal abstractions of neural networks. *NeurIPS*. 2021.
[32] Kim B, Wattenberg M, Gilmer J, et al. Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). *ICML*. 2018.
[33] Sharma A, Kiciman E. DoWhy: An End-to-End Library for Causal Inference. *arXiv*. 2020.
[34] Koh PW, Nguyen T, Tang YS, et al. Concept bottleneck models. *ICML*. 2020.
[35] Dorie V, Hill J, Shalit U, et al. Automated versus Do-It-Yourself Methods for Causal Inference: Lessons Learned from a Data Analysis Competition. *Statistical Science*. 2019.
