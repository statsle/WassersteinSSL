import numpy as np
import torch
import math
import torch.nn.functional as F
import scipy.io as sio
import os
from scipy.io import savemat

def density_estimate(num_bins, data, upper, lower):
    bin_interval = (upper-lower)/num_bins
    counter = [0 for i in range(num_bins)]

    N = data.size(0)
    for i in range(N):
        val = data[i].item()
        if val<-1.0:
            index = 0
        elif val>=1.0:
            index = num_bins-1
        else:
            index = int((val+1)/bin_interval)
            counter[index] +=1

    density = [0 for i in range(num_bins)]
    for i in range(num_bins):
        density[i] = counter[i]/N

    return density


N = 200000
bins = 51
bin_interval = 2.0/bins
dim = 128

A = torch.randn(N, dim)
B = F.normalize(A, dim=-1)

source = B[:,0]
C = torch.randn(N, dim)/math.sqrt(dim)
target = C[:,0]

source_density = density_estimate(num_bins=bins, data=source, upper=1.0, lower=-1.0)
target_density = density_estimate(num_bins=bins, data=target, upper=1.0, lower=-1.0)

value = [i*bin_interval-1.0 for i in range(bins)]

savemat("probs-128.mat", {"value":value, "source_density":source_density, "target_density":target_density})