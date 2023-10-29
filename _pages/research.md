---
permalink: /
title: "Research Summary"
---

I'm a data scientist and researcher who is passionate about applying machine learning and computational methods to novel, impactful questions. My PhD research at UC Berkeley was focused on learning how supermassive black holes have evolved and interacted with their host galaxies over cosmic time. In particular, I was interested in the dynamics of stars within these galaxies, what stellar motions can tell us about the distribution of mass within these galaxies, and how computational and machine learning methods can be used to improve these inferences. 

***

### Image processing and galaxy scaling relations
<span style="font-size:0.75em;">[Quenneville et al. 2023, MNRAS stad3137](https://doi.org/10.1093/mnras/stad3137)</span>\
<span style="font-size:0.75em;">Key topics:</span>\
<span style="font-size:0.75em;"><i>Image Processing, Bayesian Inference, Linear Regression, Galaxy Formation</i></span>

By measuring the present-day properties of galaxy populations, we can make inferences about how they formed. To do this, we need accurate measurements of galaxy size and total luminosity. This requires sensitive galaxy images with a wide field-of-view, together with precise image processing. In this paper, we acquire and process such images for a volume and luminosity limited sample of nearby galaxies. We use the resulting size and luminosity measurements to model the relationship among the most luminous galaxies. We find that the properties of these galaxies exhibit distinct scaling relationships from lower mass galaxies, suggesting that these galaxies have distinct formation mechanisms.

[<img src="/assets/images/FJ_e_slow.png" width="600">](https://doi.org/10.1093/mnras/stad3137)

***

### Surrogate optimization for triaxial orbit modelling
<span style="font-size:0.75em;">[Quenneville et al. 2022, ApJ 926 (1), 30](https://doi.org/10.3847/1538-4357/ac3e68)</span>\
<span style="font-size:0.75em;">Key topics:</span>\
<span style="font-size:0.75em;"><i>Gaussian Process Regression, 3D Reconstruction, Optimization, Surrogate Models</i></span>

Many real galaxies show clear deviations from axisymmetry. This suggests that triaxial modelling is needed to accurately model their behaviour. Previous studies have shown that some black hole masses have changed drastically when triaxial models are used instead of axisymmetric models. In this paper, we identify and resolve several issues that affected previous triaxial black hole measurements. We also describe and implement a model search scheme based on a gaussian process regression surrogate model that fairly samples different regions of the triaxial shape space and requires drastically fewer models, making the search much more computationally efficient. We use these results to perform a triaxial measurement of the black hole in the center of NGC 1453.  

[<img src="/assets/images/6d_cornerplot.png" width=600>](https://doi.org/10.3847/1538-4357/ac3e68)

***

### Model validation with simulated axisymmetric galaxies
<span style="font-size:0.75em;">[Quenneville et al. 2021, ApJS 254 (2), 25](https://doi.org/10.3847/1538-4365/abe6a0)</span>\
<span style="font-size:0.75em;">Key topics:</span>\
<span style="font-size:0.75em;"><i>Simulated data, Model validation, Orbital Dynamics, Supermassive black holes</i></span>

Fully triaxial modelling leads to many complexities that aren't present for axisymmetric models. Because of this, triaxial modelling codes are sometimes used to perform "nearly" axisymmetric modelling. However, some previous papers have shown significant inconsistencies by doing this. In this paper, we show how to effectively use a triaxial orbit code in the axisymmetric limit. We also show that we can accurately recover the central black hole mass in mock data.  

[<img src="/assets/images/mock_realizations.png" width="600">](https://doi.org/10.3847/1538-4365/abe6a0)

***

### The central black hole in NGC 1453
<span style="font-size:0.75em;">[Liepold, Quenneville, et al. 2020, ApJ 891 (1), 4](https://doi.org/10.3847/1538-4357/ab6f71)</span>\
<span style="font-size:0.75em;">Key topics:</span>\
<span style="font-size:0.75em;"><i>Hyperspectral Image Processing, Inverse Problems, Supermassive black holes</i></span>

NGC 1453 is a nearby massive elliptical galaxy. It rotates quickly and the rotation appears to be nearly aligned with its projected minor axis. This makes it a natural candidate for axisymmetric modelling. We processed hyperspectral images of NGC 1453 in order to measure stellar kinematics. We then performed axisymmetric modelling of NGC 1453 and found a central black hole about 3 billion times more massive than the sun. We also found that regularizing constraints are needed in order to make the model's kinematics realistic.

[<img src="/assets/images/NGC1453_axisymmetric.png" width="600">](https://doi.org/10.3847/1538-4357/ab6f71)

***

### Model prediction and energetic efficiency
<span style="font-size:0.75em;">[Quenneville & Sivak 2018, Entropy 20 (9), 707](https://doi.org/10.3390/e20090707)</span>\
<span style="font-size:0.75em;">Key topics:</span>\
<span style="font-size:0.75em;"><i>Information Theory, Complexity, Thermodynamic Efficiency</i></span>

When a system evolves stochastically under the influence of its environment, the system state become correlated with that of the environment. If the environment exhibits temporal correlations, the system state will also be correlated with future environment state, acting as an implicit predictive model of the environment. The complexity of this model can be quantified through the amount of mutual information between the system and environment that is not predictive of future environment states. This measure of model complexity acts as a lower bound on the thermodynamic heat dissipation, relating model information efficiency and thermodynamic efficiency. In this paper, we explore how this relationship plays out for a simple two-state system within a two-state environment and find a lower-bound on the rate at which the system can learn about its environment in the steady-state limit. 

[<img src="/assets/images/entropy.png" width="600">](https://doi.org/10.3390/e20090707)

### Boosted tree models for particle physics
<span style="font-size:0.75em;">[Quenneville 2014, CERN Internal Note (not peer-reviewed)](https://cds.cern.ch/record/1951385)</span>\
<span style="font-size:0.75em;">Key topics:</span>\
<span style="font-size:0.75em;"><i>Boosting, Decision Trees, Regression, Model Validation, Particle Physics</i></span>

When a Higgs boson decays into a pair of tau leptons, multiple neutrinos are among the final decay products. These particles are extremely weakly interacting, and pass through detectors without being detected. Thus, the mass of the original Higgs boson is not possible to reconstruct directly from the observed decay products. However, statistical inferences can be made based on the observed decay products. In this report, we describe the use of boosted tree models to reconstruct the Higgs boson mass. This model achieves comparable accuracy to existing models, while requiring a tiny fraction of the computation time.

[<img src="/assets/images/higgs.png" width="600">](https://cds.cern.ch/record/1951385)