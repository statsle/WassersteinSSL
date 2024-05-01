from metric import uniform_loss, calc_wasserstein_distance
import torch
import numpy as np
import torch.nn.functional as F
import math
from scipy.io import savemat

collapse_level = [0.50]
m_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

N = 1000
D = 32

times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
w2_results = []
lu_results = []

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
    w2_results.append(wasserstein_distance)
    lu_results.append(uniformity.item())

    print("m=0 uniformity:",uniformity)
    print("m=0 wasserstein_distance:",wasserstein_distance)

    A_cat = A

    for m in m_list:
        A_cat = torch.cat((A_cat, A), 0)
        print("A_cat size:", A_cat.size())

        B_cat = F.normalize(A_cat, dim=-1)
        uniformity = uniform_loss(B_cat)
        wasserstein_distance = calc_wasserstein_distance(B_cat)

        w2_results.append(wasserstein_distance)
        lu_results.append(uniformity.item())

        print("m: "+str(m)+"    uniformity: "+str(uniformity))
        print("m: "+str(m)+"    wasserstein_distance:"+str(wasserstein_distance))

print("w2_results:", w2_results)
print("lu_results:", lu_results)

savemat("property3.mat", {"times":times, "w2_results":w2_results, "lu_results":lu_results})

        
