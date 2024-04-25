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

class PartialEncoder(nn.Module):
    def __init__(self, args):
        super(PartialEncoder, self).__init__()
        self.args = args
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(512, args.hidden_dim, bias=False), nn.BatchNorm1d(args.hidden_dim), nn.ReLU(inplace=True), 
                                    nn.Linear(args.hidden_dim, args.hidden_dim, bias=False), nn.BatchNorm1d(args.hidden_dim), nn.ReLU(inplace=True), 
                                    nn.Linear(args.hidden_dim, args.prej_dim, bias=True))
    
    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, F.normalize(h, dim=-1), z, F.normalize(z, dim=-1)

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.online_encoder = PartialEncoder(args)
        self.target_encoder = PartialEncoder(args)

        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        self.predictor = nn.Sequential(nn.Linear(args.prej_dim, args.predictor_dim, bias=False), nn.BatchNorm1d(args.predictor_dim), nn.ReLU(inplace=True), 
                                    nn.Linear(args.predictor_dim, args.prej_dim))

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.args.momentum + param_o.data * (1. - self.args.momentum)

    def forward(self, x):
        _, _, oz, _ = self.online_encoder(x)
        p = self.predictor(oz)
        with torch.no_grad():
            _, _, tz, _ = self.target_encoder(x)
        return oz, tz, p

    def obtain_representation(self, x):
        h, h_norm, z, z_norm = self.online_encoder(x)
        return h, h_norm, z, z_norm

    def calc_alignment_loss(self, z1, p1, z2, p2):
        alignment_loss = 2.0*(1.0 - self.criterion(z1, p2).mean()) + 2.0*(1.0 - self.criterion(z2, p1).mean())
        return alignment_loss

    def calc_wasserstein_loss(self, z1, z2):
        z = torch.cat((z1, z2), 0)
        N = z.size(0)
        D = z.size(1)

        z_center = torch.mean(z, dim=0, keepdim=True)
        mean = z.mean(0)
        covariance = torch.mm((z-z_center).t(), z-z_center)/N
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