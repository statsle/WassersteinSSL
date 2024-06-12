# WassersteinSSL: A New Uniformity Metric for Self-Supervised Learning 

</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2403.00642">Rethinking the Uniformity Metric in Self-Supervised Learning</a>
</p>

## Overview

This repository encompasses four components of code. Firstly, in the “Distribution Approximation” folder, we visualize an asymptotic equivalence between a uniform spherical distribution and an isotropic Gaussian distribution. Secondly, the “Empirical Study” folder presents an empirical analysis, which includes examinations of dimensional collapse degrees, dimensions, the Feature Baby Constraint, the Feature Cloning Constraint, and the Instance Cloning Constraint. In the “Large Means” folder, we illustrate how large means can lead to severe representation collapse. Lastly, within the “code” folder, we integrate the Wasserstein distance $\mathcal{W}_{2}$ as an additional loss term in various self-supervised learning methods such as BYOL, BarlowTwins, and MoCo v2, leading to enhanced performance in downstream tasks. Using this package, the empirical results presented in Table 2 of this paper can be reproduced.


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


We empirically compared our proposed uniformity metric $-\mathcal{W}_2$ and the baseline uniformity metric $-\mathcal{L_U}$ from five different perspectives.

### On Dimensional Collapse Degrees

To generate data reflecting varying degrees of dimensional collapse,  we sample data vectors from an isotropic Gaussian distribution, normalize them to have $\ell_2$ norms, and then zero out a proportion ($\eta$) of the coordinates by running the following code:

``` python
python ./EmpiricalStudy/DimensionalCollapseDegrees/AnalysisOnCollapseLevel.py 
```
Then we draw figures:

``` python
AnalysisOnCollapseLevel.ipynb
```
as visualized:

<div>
<p align="center">
<img src='EmpiricalStudy\DimensionalCollapseDegrees\SensitivityToCollapseLevel.png' align="center" width=800>
</p>
</div>

### On Sensitiveness of Dimensions

We also analyze the sensitiveness of dimensions. We generate data points and draw figures by:

``` python
python  ./EmpiricalStudy/Dimensions/AnalysisOnDimension.py   then  AnalysisOnDimension.ipynb
```
The analyses results can be found as follow:
<div>
<p align="center">
<img src='EmpiricalStudy\Dimensions\SensitivityToDimensions.png' align="center" width=800>
</p>
</div>

### On Feature Cloning Constraint 
We generate data points and draw figures by:

``` python
python  ./EmpiricalStudy/FeatureCloningConstraint/AnalysisOnProperty4.py   then  AnalysisOnProperty4.ipynb
```

The analyses results can be found as follow:

<div>
<p align="center">
<img src='EmpiricalStudy\FeatureCloningConstraint\AnalysisOnFCC.png' align="center" width=800>
</p>
</div>

### On Feature Baby Constraint

We generate data points and draw figures by:

``` python
python  ./EmpiricalStudy/FeatureBabyConstraint/AnalysisOnProperty5.py   then  AnalysisOnProperty5.ipynb
```

The analyses results can be found as follow:

<div>
<p align="center">
<img src='EmpiricalStudy\FeatureBabyConstraint\AnalysisOnFBC.png' align="center" width=800>
</p>
</div>

### On Instance Cloning Constraint

We generate data points and draw figures by:

``` python
python  ./EmpiricalStudy/InstanceCloningConstraint/AnalysisOnProperty3.py   then  AnalysisOnProperty3.ipynb
```

The analyses results can be found as follow:

<div>
<p align="center">
<img src='EmpiricalStudy\InstanceCloningConstraint\AnalysisOnICC.png' align="center" width=400>
</p>
</div>

## Large Means

To investigate the influence of the mean on uniformity, we consider $\mathbf{X}\in \mathbb{R}^2$ following a Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{I}_2)$, where $\mathbf{Y} =  \mathbf{X} + k\cdot \mathbf{1}$, leading to $\mathbf{Y}  \sim  \mathcal{N}(k \cdot\mathbf{1}, \mathbf{I}_2)$ with $\mathbf{1} \in \mathbb{R}^k$ representing a vector of all ones. By varying $k$ from $0$ to $32$, we generate $\mathbf{Y}$ and draw figures by:

``` python
python ./LargeMeans/PlotMean2D.py
```

as visualized: 

<div>
<p align="center">
<img src='LargeMeans\largemean_collapse.png' align="center" width=800>
</p>
</div>

It is clear that an excessively large means will cause representations to collapse to a single point, even if the covariance matrix is isotropic.
## Code

In this repository, we integrate the our proposed uniformity loss $\mathcal{W}_{2}$ as an additional loss term in the existing self-supervised learning methods such as BYOL, BarlowTwins, and MoCo v2.

### BYOL
Train and evaluate on either CIFAR-10 or CIFAR-100 dataset without incorporating our proposed uniformity loss $\mathcal{W}_{2}$:
``` python
bash ./code/BYOL/run_vanilla_byol.sh
```

Train and evaluate on either CIFAR-10 or CIFAR-100 dataset without incorporating our proposed uniformity loss $\mathcal{W}_{2}$:
``` python
bash ./code/BYOL/run_byol+w2.sh
```

### BarlowTwins
Train and evaluate on either CIFAR-10 or CIFAR-100 dataset without incorporating our proposed uniformity loss $\mathcal{W}_{2}$:
``` python
bash ./code/BarlowTwins/run_vanilla_barlowtwins.sh
```

Train and evaluate on either CIFAR-10 or CIFAR-100 dataset without incorporating our proposed uniformity loss $\mathcal{W}_{2}$:
``` python
bash ./code/BarlowTwins/run_barlowtwins+w2.sh
```

### MoCo v2
Train and evaluate on either CIFAR-10 or CIFAR-100 dataset without incorporating our proposed uniformity loss $\mathcal{W}_{2}$:
``` python
bash ./code/MoCov2/run_vanilla_moco.sh
```

Train and evaluate on either CIFAR-10 or CIFAR-100 dataset without incorporating our proposed uniformity loss $\mathcal{W}_{2}$:
``` python
bash ./code/MoCov2/run_moco+w2.sh
```

We report experimental results in this Table:
| Methods | Proj. | Pred. | CIFAR-10 Acc@1↑ | CIFAR-10 Acc@5↑ | CIFAR-10 $\mathcal{W}_{2}$↓ | CIFAR-10 $\mathcal{L_U}$↓ | CIFAR-10 $\mathcal{A}$↓ | CIFAR-100 Acc@1↑ | CIFAR-100 Acc@5↑ | CIFAR-100 $\mathcal{W}_{2}$↓ | CIFAR-100 $\mathcal{L_U}$↓ | CIFAR-100 $\mathcal{A}$↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MoCo v2 | 256 | ✘ | 90.65 | 99.81 | 1.06 | -3.75 | 0.51 | 60.27 | 86.29 | 1.07 | -3.60 | 0.46 |
| MoCo v2 + $\mathcal{L_U}$ | 256 | ✘ | 90.98 ↑₀.₃₃ | 99.67 | 0.98 ↑₀.₀₈ | -3.82 | 0.53 ↓₀.₀₂ | 61.21 ↑₀.₉₄ | 87.32 | 0.98 ↑₀.₀₉ | -3.81 | 0.52 ↓₀.₀₆ |
| MoCo v2 + $\mathcal{W}_{2}$ | 256 | ✘ | 91.41 ↑₀.₇₆ | 99.68 | 0.33 ↑₀.₇₃ | -3.84 | 0.63 ↓₀.₁₂ | 63.68 ↑₃.₄₁ | 88.48 | 0.28 ↑₀.₇₉ | -3.86 | 0.66 ↓₀.₂₀ |
| BYOL | 256 | 256 | 89.53 | 99.71 | 1.21 | -2.99 | 0.31 | 63.66 | 88.81 | 1.20 | -2.87 | 0.33 |
| BYOL + $\mathcal{L_U}$ | 256 | ✘ | 90.09 ↑₀.₅₆ | 99.75 | 1.09 ↑₀.₁₂ | -3.66 | 0.40 ↓₀.₀₉ | 62.68 ↓₀.₉₈ | 88.44 | 1.08 ↑₀.₁₂ | -3.70 | 0.51 ↓₀.₁₈ |
| BYOL + $\mathcal{W}_{2}$ | 256 | 256 | 90.31 ↑₀.₇₈ | 99.77 | 0.38 ↑₀.₈₃ | -3.90 | 0.65 ↓₀.₃₄ | 65.16 ↑₁.₅₀ | 89.25 | 0.36 ↑₀.₈₄ | -3.91 | 0.69 ↓₀.₃₆ |
| BarlowTwins | 256 | ✘ | 91.16 | 99.80 | 0.22 | -3.91 | 0.75 | 68.19 | 90.64 | 0.23 | -3.91 | 0.75 |
| BarlowTwins + $\mathcal{L_U}$ | 256 | ✘ | 91.38 ↑₀.₂₂ | 99.77 | 0.21 ↑₀.₀₁ | -3.92 | 0.76 ↓₀.₀₁ | 68.41 ↑₀.₂₂ | 90.99 | 0.22 ↑₀.₀₁ | -3.91 | 0.76 ↓₀.₀₁ |
| BarlowTwins + $\mathcal{W}_{2}$ | 256 | ✘ | 91.43 ↑₀.₂₇ | 99.78 | 0.19 ↑₀.₀₃ | -3.92 | 0.76 ↓₀.₀₁ | 68.47 ↑₀.₂₈ | 90.64 | 0.19 ↑₀.₀₄ | -3.91 | 0.79 ↓₀.₀₄ |
| Zero-CL | 256 | ✘ | 91.35 | 99.74 | 0.15 | -3.94 | 0.70 | 68.50 | 90.97 | 0.15 | -3.93 | 0.75 |
| Zero-CL + $\mathcal{L_U}$ | 256 | ✘ | 91.28 ↓₀.₀₇ | 99.74 | 0.15 | -3.94 | 0.72 ↓₀.₀₂ | 68.44 ↓₀.₀₆ | 90.91 | 0.15 | -3.93 | 0.74 ↑₀.₀₁ |
| Zero-CL + $\mathcal{W}_{2}$ | 256 | ✘ | 91.42 ↑₀.₀₇ | 99.82 | 0.14 ↑₀.₀₁ | -3.94 | 0.71 ↓₀.₀₁ | 68.55 ↑₀.₀₅ | 91.02 | 0.14 ↑₀.₀₁ | -3.94 | 0.76 ↓₀.₀₁ |


## Citation
If our [paper](https://arxiv.org/abs/2403.00642) assists your research, feel free to give us a star or cite us using:
```
@inproceedings{Fang2024RethinkingTU,
      title={Rethinking the Uniformity Metric in Self-Supervised Learning}, 
      author={Xianghong Fang and Jian Li and Qiang Sun and Benyou Wang},
      booktitle={ICLR},
      year={2024}
}
```










