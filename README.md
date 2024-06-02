This is code for paper ``Rethking the uniformity metric in self-supervised learning", ICLR 2024. 

## WassersteinSSL

This repository encompasses four components of code. Firstly, in the “Distribution Approximation” folder, we visualize an asymptotic equivalence between a uniform spherical distribution and an isotropic Gaussian distribution. Secondly, the “Empirical Study” folder presents an empirical analysis, which includes examinations of dimensional collapse degrees, dimensions, the Feature Baby Constraint, the Feature Cloning Constraint, and the Instance Cloning Constraint. In the “Large Means” folder, we illustrate how large means can lead to severe representation collapse. Lastly, within the “code” folder, we integrate the Wasserstein distance $\mathcal{W}_{2}$ as an additional loss term in various self-supervised learning methods such as BYOL, BarlowTwins, and MoCov2, leading to enhanced performance in downstream tasks. Using this package, the empirical results presented in Table 2 of this paper can be reproduced.


## Distribution Approximation

To illustrate the asymptotic equivalence between a uniform spherical distribution (where $Y_i$ represents the $i$-th coordinate) and an isotropic Gaussian distribution ($\hat{Y}_i \sim \mathcal{N}(0, 1/m) $), we begin by randomly sampling data points and estimating their distribution using:

``` python
python ./DistributionApproximation/Density1DPlot.py or python ./DistributionApproximation/Density2DPlot.py
```

Then, we draw figures by running juypter notebook files:
``` python
 Density1DPlot.ipynb or Density2DPlot.ipynb
```

Using the estimated distributions, we visualize $Y_i$ and $\hat{Y}_i$ across different dimensions $m \in [2, 4, 8, 16, 32, 64, 128, 256]$.
<div>
<p align="center">
<img src='DistributionApproximation\MergedDistribution-1D.png' align="center" width=800>
</p>
</div>

We also analyze the joint binning densities and present 2D joint binning densities of  $(Y_i, Y_j)$ ($i \neq j$) and  $(\hat{Y}_i, \hat{Y}_j)$ ($i \neq j$). Even if $m$ is relatively small (i.e., 32), the densities of the two distributions are close.

<div>
<p align="center">
<img src='DistributionApproximation\MergedDistribution-2D.png' align="center" width=800>
</p>
</div>

## Empirical Study

We delve into the study of our proposed uniformity metric and baseline uniformity metric from five perspectives:

### On Dimensional Collapse Degrees

<div>
<p align="center">
<img src='EmpiricalStudy\DimensionalCollapseDegrees\SensitivityToCollapseLevel.png' align="center" width=800>
</p>
</div>

### On Sensitiveness of Dimensions

<div>
<p align="center">
<img src='EmpiricalStudy\Dimensions\SensitivityToDimensions.png' align="center" width=800>
</p>
</div>

### On Feature Cloning Constraint 

<div>
<p align="center">
<img src='EmpiricalStudy\FeatureCloningConstraint\AnalysisOnFCC.png' align="center" width=800>
</p>
</div>

### On Feature Baby Constraint

<div>
<p align="center">
<img src='EmpiricalStudy\FeatureBabyConstraint\AnalysisOnFBC.png' align="center" width=800>
</p>
</div>

### On Instance Cloning Constraint

<div>
<p align="center">
<img src='EmpiricalStudy\InstanceCloningConstraint\AnalysisOnICC.png' align="center" width=400>
</p>
</div>

## Large Means

To analyze the impact of mean on the uniformity, we assume $\mathbf{X}\in \mathbb{R}^2$ follows a Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{I}_2)$, let $\mathbf{Y} =  \mathbf{X} + k\cdot \mathbf{1}$ such that $\mathbf{Y}  \sim  \mathcal{N}(k \cdot\mathbf{1}, \mathbf{I}_2)$, where $\mathbf{1} \in \mathbb{R}^k$ represents a vector of all ones. We vary $k$ from $0$ to $32$ and visualize the $\ell_2$-normalized $\mathbf{Y}$'s (by generating multiple independent copies).  It is clear that an excessively large means will cause representations to collapse to a single point, even if the covariance matrix is isotropic.

<div>
<p align="center">
<img src='LargeMeans\largemean_collapse.png' align="center" width=800>
</p>
</div>

## Code


## Reference

Fang, X., Li, J., Sun, Q., and Wang, B., Rethinking the Uniformity Metric in Self-Supervised Learning. [Paper](https://arxiv.org/abs/2403.00642)











