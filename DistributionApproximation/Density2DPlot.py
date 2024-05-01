import numpy as np
import torch
import math
import torch.nn.functional as F
import scipy.io as sio
import os
from scipy.io import savemat

def density_estimate(num_bins, X1, X2, upper, lower):
    bin_interval = (upper-lower)/num_bins
    counter = [[0 for i in range(num_bins)] for j in range(num_bins)]

    N = X1.size(0)
    for i in range(N):
        x1 = X1[i].item()
        x2 = X2[i].item()

        if x1 <-1.0:
            y1 = 0
        elif x1>=1.0:
            y1 = num_bins-1
        else:
            y1 = int((x1+1)/bin_interval)

        if x2 <-1.0:
            y2 = 0
        elif x2>=1.0:
            y2 = num_bins-1
        else:
            y2 = int((x2+1)/bin_interval)

        counter[y1][y2] +=1

    density = [[0 for i in range(num_bins)] for j in range(num_bins)]
    for i in range(num_bins):
        for j in range(num_bins):
            density[i][j] = counter[i][j]/N

    return density

N = 2000000
bins = 51
bin_interval = 2.0/bins
dim = 32

A = torch.randn(N, dim)
B = F.normalize(A, dim=-1)

B1 = B[:,0]
B2 = B[:,1]

C = torch.randn(N, dim)/math.sqrt(dim)
C1 = C[:, 0]
C2 = C[:, 1]

source_density = density_estimate(num_bins=bins, X1=B1, X2=B2, upper=1.0, lower=-1.0)
target_density = density_estimate(num_bins=bins, X1=C1, X2=C2, upper=1.0, lower=-1.0)

value = [i*bin_interval-1.0 for i in range(bins)]

savemat("probs2D.mat", {"value":value, "source_density":source_density, "target_density":target_density})