import torch
import numpy as np
import torch.nn.functional as F
import math

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def calc_wasserstein_distance(x):
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x-x_center).t(), x-x_center)/N

    mean =  x.mean(0)
    np_mean = mean.data.cpu().numpy()
    np_covariance = covariance.data.cpu().numpy()
   
    ##calculation of part1
    part1 = np.sum(np.multiply(np_mean, np_mean))

    ##calculation of part2
    eps = 1e-8 
    S, Q = np.linalg.eig(np_covariance)
    S = S + eps
    mS = np.sqrt(np.diag(S))
    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return wasserstein_distance 

def calc_singular(z):
    z = F.normalize(z, dim=1)
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)
    _, d, _ = np.linalg.svd(c)

    d = np.sort(d)

    d = d[::-1]

    print("d:", d)

    embedding_spectrum = np.log(d)

    return embedding_spectrum