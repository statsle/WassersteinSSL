from metric import uniform_loss, calc_wasserstein_distance
import torch
import numpy as np
import torch.nn.functional as F
import math

dim_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#dim_list = [512, 1024, 2048, 4096]
N = 50000



for dim in dim_list:
    print("dim:", dim)
    if dim == 2:
        sub_dim = int(dim/2)
        A1 = torch.randn(N, sub_dim)
        A2 = torch.zeros(N, sub_dim)
        A = torch.cat((A1, A2), 1).cuda()
        B = F.normalize(A, dim=-1)
        uniformity = uniform_loss(B)
        wasserstein_distance, mean, covariance = calc_wasserstein_distance(B)
        print("50% uniformity:",uniformity)
        print("50% wasserstein_distance:",wasserstein_distance)

        A = torch.randn((N, dim)).cuda()
        B = F.normalize(A, dim=-1)
        uniformity = uniform_loss(B)
        wasserstein_distance, mean, covariance = calc_wasserstein_distance(B)
        #print("100% uniformity:",uniformity)
        print("100% wasserstein_distance:", wasserstein_distance)
        print("100% mean:", mean)
        print("100% covariance:", covariance)

    else:
        sub_dim_1 = int(dim*0.25)
        sub_dim_2 = int(dim*0.75)
        A1 = torch.randn(N, sub_dim_1)
        A2 = torch.zeros(N, sub_dim_2)
        A = torch.cat((A1, A2), 1).cuda()
        B = F.normalize(A, dim=-1)
        #uniformity = uniform_loss(B)
        wasserstein_distance, mean, covariance = calc_wasserstein_distance(B)
        #print("25% uniformity:",uniformity)
        print("25% wasserstein_distance:",wasserstein_distance)

        sub_dim_1 = int(dim*0.5)
        sub_dim_2 = int(dim*0.5)
        A1 = torch.randn(N, sub_dim_1)
        A2 = torch.zeros(N, sub_dim_2)
        A = torch.cat((A1, A2), 1).cuda()
        B = F.normalize(A, dim=-1)
        #uniformity = uniform_loss(B)
        wasserstein_distance, mean, covariance = calc_wasserstein_distance(B)
        #print("50% uniformity:",uniformity)
        print("50% wasserstein_distance:",wasserstein_distance)

        sub_dim_1 = int(dim*0.75)
        sub_dim_2 = int(dim*0.25)
        A1 = torch.randn(N, sub_dim_1)
        A2 = torch.zeros(N, sub_dim_2)
        A = torch.cat((A1, A2), 1).cuda()
        B = F.normalize(A, dim=-1)
        #uniformity = uniform_loss(B)
        wasserstein_distance, mean, covariance = calc_wasserstein_distance(B)
        #print("75% uniformity:",uniformity)
        print("75% wasserstein_distance:",wasserstein_distance)
        

        A = torch.randn((N, dim))
        B = F.normalize(A, dim=-1)
        #uniformity = uniform_loss(B).cuda()
        wasserstein_distance, mean, covariance = calc_wasserstein_distance(B)
        #print("100% uniformity:",uniformity)
        print("100% wasserstein_distance:",wasserstein_distance)
    


