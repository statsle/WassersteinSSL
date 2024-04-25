import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

class Queue(nn.Module):
    def __init__(self, args):
        super(Queue, self).__init__()
        self.args = args
        self.register_buffer("queue", torch.randn(args.queue_size, args.prej_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, key):
        ## key [batch_size, feature_dim] and has been nn.functional.normalized.
        batch_size = key.shape[0]
        ptr = int(self.queue_ptr)
        assert self.args.queue_size % batch_size == 0  # for simplicity
        # replace the key at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = key
        #self.label_queue[ptr:ptr + batch_size] = label
        ptr = (ptr + batch_size) % self.args.queue_size  # move pointer
        self.queue_ptr[0] = ptr  

    @torch.no_grad()
    def obtain_negative_embed_from_queue(self):
        return self.queue.detach().clone()

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
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(512, args.hidden_dim, bias=False), nn.BatchNorm1d(args.hidden_dim), nn.ReLU(inplace=True), 
                                    nn.Linear(args.hidden_dim, args.hidden_dim, bias=False), nn.BatchNorm1d(args.hidden_dim), nn.ReLU(inplace=True), 
                                    nn.Linear(args.hidden_dim, args.prej_dim, bias=True))

    def forward(self, x):
        ### h for downstream task; z self-supervised representation
        ### return (h, h_norm, z, z_norm)
        h = self.backbone(x)
        z = self.projector(h)
        return h, F.normalize(h, dim=-1), z, F.normalize(z, dim=-1)

    def calc_contrastive_loss(self, q_norm, k_norm, negative_embed):
        batch_size = q_norm.size(0)
        ## q_norm k_norm [batch_size, dim], negative_embed[batch_size, K, feature_dim]
        pos = torch.exp(torch.sum(q_norm * k_norm, dim=-1) / self.args.temperature) ##[batch_size]

        negative_embed = negative_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        negative_embed = negative_embed.permute(0, 2, 1).contiguous()

        neg = torch.exp(torch.bmm(q_norm.unsqueeze(1), negative_embed).squeeze(1)/self.args.temperature) ##[batch_size, K]
        contrastive_loss = (- torch.log(pos / (pos + neg.sum(dim=-1)) )).mean()
        return contrastive_loss 

    def calc_wasserstein_loss(self, z1, z2):
        z = torch.cat((z1, z2), 0)
        N = z.size(0)
        D = z.size(1)

        z_center = torch.mean(z, dim=0, keepdim=True)
        mean = z.mean(0)
        covariance = torch.mm((z-z_center).t(), z-z_center)/N + 1e-4*torch.eye(D).cuda()
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
