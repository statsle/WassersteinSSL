from metric import uniform_loss, calc_wasserstein_distance
import torch
import numpy as np
import torch.nn.functional as F
import math
from scipy.io import savemat

collapse_level = [0.75, 0.50, 0.25]
m_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

N = 50000
D = 32


times = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
w2_results = [[0 for i in range(11)] for i in range(3)]
lu_results = [[0 for i in range(11)] for i in range(3)]

i = 0
j = 0

for level in collapse_level:
    print("level:", level)
    sub_dim_1 = int(D*(1-level))
    sub_dim_2 = int(D*level)

    A1 = torch.randn(N, sub_dim_1)
    A2 = torch.zeros(N, sub_dim_2)
    A = torch.cat((A1, A2), 1)
    B = F.normalize(A, dim=-1)
    uniformity = uniform_loss(B)
    wasserstein_distance = calc_wasserstein_distance(B)

    print("m=0 uniformity:",uniformity)
    print("m=0 wasserstein_distance:",wasserstein_distance)

    w2_results[i][j] = wasserstein_distance
    lu_results[i][j] = uniformity.item()

    B_cat = B

    for m in m_list:
        j += 1
        B_cat = torch.cat((B_cat, B), 1)
        uniformity = uniform_loss(B_cat)
        #B_cat = F.normalize(B_cat, dim=-1)
        wasserstein_distance = calc_wasserstein_distance(B_cat)

        print("m: "+str(m)+"    uniformity: "+str(uniformity))
        print("m: "+str(m)+"    wasserstein_distance:"+str(wasserstein_distance))

        w2_results[i][j] = wasserstein_distance
        lu_results[i][j] = uniformity.item()

    i += 1
    j = 0

print("w2_results:", w2_results)
print("lu_results:", lu_results)

savemat("property4.mat", {"times":times, "w2_results":w2_results, "lu_results":lu_results})
