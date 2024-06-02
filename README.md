This is code for paper ``Rethking the uniformity metric in self-supervised learning", ICLR 2024. 

## WassersteinSSL

This repository encompasses four components of code. Firstly, in the “Distribution Approximation” folder, we visualize an asymptotic equivalence between a uniform spherical distribution and an isotropic Gaussian distribution. Secondly, the “Empirical Study” folder presents an empirical analysis, which includes examinations of dimensional collapse degrees, dimensions, the Feature Baby Constraint, the Feature Cloning Constraint, and the Instance Cloning Constraint. In the “Large Means” folder, we illustrate how large means can lead to severe representation collapse. Lastly, within the “code” folder, we integrate the Wasserstein distance $\mathcal{W}_{2}$ as an additional loss term in various self-supervised learning methods such as BYOL, BarlowTwins, and MoCov2, leading to enhanced performance in downstream tasks. Using this package, the empirical results presented in Table 2 of this paper can be reproduced.


### Distribution Approximation

To demonstrate an asymptotic equivalence between a uniform spherical distribution ($Y_i$ be the $i$-th coordinate of the uniform spherical distribution) and an isotropic Gaussian distribution ($\hat{Y}_i \sim \mathcal{N}(0, 1/m)$)



We estimate their distribution by:
``` python
python Density1DPlot.py or python Density2DPlot.py
```


1-d visualization
<div>
<p align="center">
<img src='DistributionApproximation\MergedDistribution-1D.png' align="center" width=800>
</p>
</div>

2-d visualization

<div>
<p align="center">
<img src='DistributionApproximation\MergedDistribution-2D.png' align="center" width=800>
</p>
</div>

### Empirical Study

### Large Means

### Code


## Reference

Fang, X., Li, J., Sun, Q., and Wang, B., Rethinking the Uniformity Metric in Self-Supervised Learning. [Paper](https://arxiv.org/abs/2403.00642)











