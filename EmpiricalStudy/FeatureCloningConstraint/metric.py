import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn

def calc_entropy(x):
    return (-x*torch.log(x)).sum()

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
    #eps = 1e-8 
    #sigma_1 = np_sigma + eps * np.eye(dim)

    S, Q = np.linalg.eig(np_covariance)
    #print("S:", S)
    mS = np.sqrt(np.diag(abs(S)))
    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return wasserstein_distance

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def calc_correlation(x):
    N = x.size(0)
    dim = x.size(1)
    bn = nn.BatchNorm1d(dim, affine=False)

    x_bn = bn(x)
    c = torch.matmul(x_bn.t(), x_bn)/N
    correlation = off_diagonal(c).abs().mean() 
    return correlation


