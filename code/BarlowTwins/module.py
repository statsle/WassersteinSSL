import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

class LinearProjector(nn.Module):
    def __init__(self, args):
        super(LinearProjector, self).__init__()
        self.fc = nn.Linear(512, args.num_class, bias=True)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(nn.Linear(512, args.hidden_dim, bias=False), nn.BatchNorm1d(args.hidden_dim), nn.ReLU(inplace=True), 
                                    nn.Linear(args.hidden_dim, args.hidden_dim, bias=False), nn.BatchNorm1d(args.hidden_dim), nn.ReLU(inplace=True), 
                                    nn.Linear(args.hidden_dim, args.prej_dim, bias=True))
        self.bn = nn.BatchNorm1d(args.prej_dim, affine=False)

    def forward(self, x):
        ### h for downstream task; z self-supervised representation
        ### return (h, h_norm, z, z_norm)
        h = self.backbone(x)
        z = self.projector(h)
        return h, F.normalize(h, dim=-1), z, F.normalize(z, dim=-1)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def calc_cross_correlation(self, z1, z2):
        N = z1.size(0)
        D = z1.size(1)

        ### Barlow Twins
        z1_bn = self.bn(z1)
        z2_bn = self.bn(z2)

        ##cross-correlation matrix
        c = torch.matmul(z1_bn.t(), z2_bn)/N

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        cross_correlation = on_diag + self.args.lambd * off_diag
        return cross_correlation

    def calc_wasserstein_loss(self, z1, z2):
        z = torch.cat((z1, z2), 0)
        N = z.size(0)
        D = z.size(1)

        z_center = torch.mean(z, dim=0, keepdim=True)
        mean = z.mean(0)
        covariance = torch.mm((z-z_center).t(), z-z_center)/N + 1e-12*torch.eye(D).cuda()
        #############calculation of part1
        part1 = torch.sum(torch.multiply(mean, mean))

        ######################################################
        S, Q = torch.eig(covariance, eigenvectors=True)
        S = torch.abs(S[:,0])
        mS = torch.sqrt(torch.diag(S))
        covariance2 = torch.mm(torch.mm(Q, mS), Q.T)

        #############calculation of part2
        part2 = torch.trace(covariance - 2.0/math.sqrt(D)*covariance2)
        wasserstein_loss = torch.sqrt(part1+1+part2)

        return wasserstein_loss