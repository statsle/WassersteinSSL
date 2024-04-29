import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision


mean_list = [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

N = 300
for mean in mean_list:
    A1 = torch.randn(N, 2)
    A2 = torch.ones(N, 2)

    A = ( A1+ mean *A2 ).cuda()
    B = F.normalize(A, dim=-1)

    np_B = B.data.cpu().numpy()
    x = np_B[:,0].tolist()
    y = np_B[:,1].tolist()

    plt.figure(figsize=(16, 16), dpi=300)
    tick_fontsize=30
    label_fontsize  = 35

    plt.xlim((-1.05, 1.05))
    x_ticks = [-1.0, -0.5, 0, 0.5, 1.0]
    x_labels = [-1.0, -0.5, 0, 0.5, 1.0]
    plt.xticks(x_ticks, x_labels, fontsize=tick_fontsize)

    plt.ylim((-1.05, 1.05))
    y_ticks = [-1.0, -0.5, 0, 0.5, 1.0]
    y_labels = [-1.0, -0.5, 0, 0.5, 1.0]
    plt.yticks(y_ticks, y_labels, fontsize=tick_fontsize)

    plt.gca().yaxis.grid(color='grey', linestyle='-', linewidth=1, alpha=0.3, zorder=0)
    plt.gca().xaxis.grid(color='grey', linestyle='-', linewidth=1, alpha=0.3, zorder=0)

    plt.scatter(x, y, c='blue')
    plt.savefig("distribution_"+str(mean)+".pdf", bbox_inches="tight")