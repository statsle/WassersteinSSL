from metric import uniform_loss, calc_wasserstein_distance
import torch
import numpy as np
import torch.nn.functional as F
import math

#collapse_level = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
collapse_level = [0.98, 0.99, 1.0]
N=20000

for level in collapse_level:
    print("level:", level)
    sub_dim_1 = int(200*(1-level))
    sub_dim_2 = int(200*level)
    A1 = torch.randn(N, sub_dim_1)
    A2 = torch.zeros(N, sub_dim_2)
    A = torch.cat((A1, A2), 1).cuda()
    B = F.normalize(A, dim=-1)
    uniformity = uniform_loss(B)
    wasserstein_distance = calc_wasserstein_distance(B)
    print("200 uniformity:",uniformity)
    print("200 wasserstein_distance:",wasserstein_distance)

    sub_dim_1 = int(500*(1-level))
    sub_dim_2 = int(500*level)
    A1 = torch.randn(N, sub_dim_1)
    A2 = torch.zeros(N, sub_dim_2)
    A = torch.cat((A1, A2), 1).cuda()
    B = F.normalize(A, dim=-1)
    uniformity = uniform_loss(B)
    wasserstein_distance = calc_wasserstein_distance(B)
    print("500 uniformity:",uniformity)
    print("500 wasserstein_distance:",wasserstein_distance)

    sub_dim_1 = int(1000*(1-level))
    sub_dim_2 = int(1000*level)
    A1 = torch.randn(N, sub_dim_1)
    A2 = torch.zeros(N, sub_dim_2)
    A = torch.cat((A1, A2), 1).cuda()
    B = F.normalize(A, dim=-1)
    uniformity = uniform_loss(B)
    wasserstein_distance = calc_wasserstein_distance(B)
    print("1000 uniformity:",uniformity)
    print("1000 wasserstein_distance:",wasserstein_distance)