import math
import torch
import torch.nn as nn
import sys
from torch import nn, optim
import time
import numpy as np
import random
import os
import scipy.io as sio
from torch.nn import functional as F
from scipy.io import savemat
from module import Encoder, LinearProjector
from util import Pack, LossManager, LARS, exclude_bias_and_norm, adjust_learning_rate
import pandas as pd
#from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from dataloader import CIFARPairTransform, CIFARSingleTransform, TinyImageNetSubset, TinyImageNetPairTransform, TinyImageNetSingleTransform
from dataloader import STLPairTransform, STLSingleTransform
from metric import align_loss, uniform_loss, calc_wasserstein_distance, calc_singular

class BarlowTwinsplus(nn.Module):
    def __init__(self, args):
        super(BarlowTwinsplus, self).__init__()
        self.args = args 
        self.encoder = Encoder(args).cuda() ### train stage
        self.projector = LinearProjector(args).cuda() ### linear evaluation stage

        if args.dataset == "CIFAR-10":
            ##for alignment metric calculation
            train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=CIFARPairTransform(train_transform=True), download=True)
            val_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=CIFARPairTransform(train_transform=True), download=True)
            ##for uniformity metric calculation
            memory_train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=CIFARPairTransform(train_transform=False), download=True)
            memory_val_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=CIFARPairTransform(train_transform=False), download=True)
            ##for linear evaluation
            linear_train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=CIFARSingleTransform(train_transform=True), download=True)
            linear_val_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=CIFARSingleTransform(train_transform=False), download=True)            
        elif args.dataset == "CIFAR-100":
            ##for alignment metric calculation
            train_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, transform=CIFARPairTransform(train_transform=True), download=True)
            val_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=CIFARPairTransform(train_transform=True), download=True)
            ##for uniformity metric calculation
            memory_train_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, transform=CIFARPairTransform(train_transform=False), download=True)
            memory_val_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=CIFARPairTransform(train_transform=False), download=True)
            ##for linear evaluation
            linear_train_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, transform=CIFARSingleTransform(train_transform=True), download=True)
            linear_val_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=CIFARSingleTransform(train_transform=False), download=True)
        elif args.dataset == "STL-10":
            train_data = torchvision.datasets.STL10(root=args.data_dir, split="train+unlabeled", transform=STLPairTransform(train_transform=True), download=True)
            val_data = torchvision.datasets.STL10(root=args.data_dir, split="test", transform=STLPairTransform(train_transform=True), download=True)
            ##for uniformity metric calculation
            memory_train_data = torchvision.datasets.STL10(root=args.data_dir, split="train", transform=STLPairTransform(train_transform=False), download=True)
            memory_val_data = torchvision.datasets.STL10(root=args.data_dir, split="test", transform=STLPairTransform(train_transform=False), download=True)
            ##for linear evaluation
            linear_train_data = torchvision.datasets.STL10(root=args.data_dir, split="train", transform=STLSingleTransform(train_transform=True), download=True)
            linear_val_data = torchvision.datasets.STL10(root=args.data_dir, split="test", transform=STLSingleTransform(train_transform=False), download=True)
        elif args.dataset == "Tiny-Imagenet":
            train_data = TinyImageNetSubset(args, split="train", transform=TinyImageNetPairTransform(train_transform = True))
            val_data = TinyImageNetSubset(args, split="val", transform=TinyImageNetPairTransform(train_transform = True))
            memory_train_data = TinyImageNetSubset(args, split="train", transform=TinyImageNetPairTransform(train_transform = False))
            memory_val_data = TinyImageNetSubset(args, split="val", transform=TinyImageNetPairTransform(train_transform = False))
            linear_train_data = TinyImageNetSubset(args, split="train", transform=TinyImageNetSingleTransform(train_transform = True))
            linear_val_data = TinyImageNetSubset(args, split="val", transform=TinyImageNetSingleTransform(train_transform = False))

        self.train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        self.val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.memory_train_dataloader = DataLoader(memory_train_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.memory_val_dataloader = DataLoader(memory_val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.linear_train_dataloader = DataLoader(linear_train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.linear_val_dataloader = DataLoader(linear_val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def build_model(self, x1, x2, lambd):
        batch_size = x1.size(0)

        _, _, z1, z1_norm = self.encoder(x1)
        _, _, z2, z2_norm = self.encoder(x2)
        cross_correlation = self.encoder.calc_cross_correlation(z1, z2)
        wasserstein_loss = self.encoder.calc_wasserstein_loss(z1_norm, z2_norm)
        total_loss = cross_correlation + lambd * wasserstein_loss

        train_pack = Pack(total_loss = total_loss)
        print_pack = Pack(total_loss = total_loss, cross_correlation=cross_correlation, wasserstein_loss=wasserstein_loss)
        return train_pack, print_pack

    def train_model(self, args):
        self.train()
        param_weights = []
        param_biases = []
        for param in self.encoder.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = LARS(parameters, lr=0, weight_decay=1e-4, weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)
        results = {'total_loss':[], 'cross_correlation': [], 'wasserstein_loss':[], 'test_acc@1': [], 'test_acc@5': []}
        best_acc = 0.0
        start = time.time()
        train_loss = LossManager()
        lambd_weight = (args.lambd_max-args.lambd_min)/args.epochs
        for epoch in range(1, args.epochs + 1):
            print("epoch:%d, lr:%4f"%(epoch, optimizer.param_groups[0]["lr"]))
            total_loss, cross_correlation, wasserstein_loss, total_num = 0.0, 0.0, 0.0, 0
            iter_count = 0
            lambd = args.lambd_max - lambd_weight * epoch
            for step, ((x1, x2), _) in enumerate(self.train_dataloader, start=(epoch-1)*len(self.train_dataloader)):
                iter_count += 1
                batch_size = x1.size(0)
                x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)
                adjust_learning_rate(args, optimizer, self.train_dataloader, step)
                optimizer.zero_grad() 
                train_pack, print_pack = self.build_model(x1, x2, lambd) 
                train_pack.total_loss.backward()
                optimizer.step()
                train_loss.add_loss(print_pack)

                total_num += batch_size
                total_loss += print_pack.total_loss.item() * batch_size
                cross_correlation += print_pack.cross_correlation.item() * batch_size
                wasserstein_loss += print_pack.wasserstein_loss.item() * batch_size
                if iter_count%10==0:
                    print(train_loss.pprint(window=30, prefix='Train Epoch: [{}/{}] Iters:[{}/{}] lambd:{}'.format(epoch, args.epochs, iter_count, len(self.train_dataloader), lambd)))

            train_loss.clear()
            if epoch%100 ==0: 
                torch.save(self.state_dict(), './{}/{}_{}_model.pth'.format(args.saver_dir, args.save_name_pre, epoch))
            test_acc_1, test_acc_5 = self.calc_knn_accuracy(epoch, args)

            #uniformity, wasserstein_distance = self.calc_uniformity(args)
            #alignment = self.calc_alignment(args)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            results['total_loss'].append(total_loss/total_num)
            results['cross_correlation'].append(cross_correlation/total_num)
            results['wasserstein_loss'].append(wasserstein_loss/total_num)
            #results['alignment'].append(alignment)
            #results['uniformity'].append(uniformity)
            #results['wasserstein_distance'].append(wasserstein_distance)
            
            #save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('./{}/{}_statistics.csv'.format(args.saver_dir, args.save_name_pre), index_label='epoch')
        self.eval()

    def calc_uniformity(self, args):
        self.eval()
        z_norm_bank = []
        with torch.no_grad():
            for (data, _), target in tqdm(self.memory_val_dataloader, desc='calculation of uniformity in val dataset'):
                _, _, _, z_norm = self.encoder(data.cuda(non_blocking=True))
                z_norm_bank.append(z_norm)
                
            z_norm_bank = torch.cat(z_norm_bank, dim=0).contiguous()
            uniformity = uniform_loss(z_norm_bank) 
            wasserstein_distance = calc_wasserstein_distance(z_norm_bank)
            
            print('Uniformity:{:.4f}'.format(uniformity))
            print('wasserstein_distance:{:.4f}'.format(wasserstein_distance))
        self.train()
        return uniformity.item(), wasserstein_distance

    def calc_alignment(self, args):
        self.eval()
        z1_bank, z2_bank = [], []
        with torch.no_grad():
            for (x1, x2), target in tqdm(self.val_dataloader, desc='calculation of alignment in val dataset'):
                _, _, z1, z1_norm = self.encoder(x1.cuda(non_blocking=True))
                _, _, z2, z2_norm = self.encoder(x2.cuda(non_blocking=True))
                z1_bank.append(z1_norm)
                z2_bank.append(z2_norm)

        z1_bank = torch.cat(z1_bank, dim=0).contiguous()
        z2_bank = torch.cat(z2_bank, dim=0).contiguous()
        alignment = align_loss(z1_bank, z2_bank)
        print('Alignment:{:.4f}'.format(alignment))
        self.train()
        return alignment.item()

    def dimensional_analysis(self, args):
        self.eval()
        z_norm_bank = []
        with torch.no_grad():
            for (data, _), target in tqdm(self.memory_val_dataloader, desc='dimensional_analysis'):
                _, _, _, z_norm = self.encoder(data.cuda(non_blocking=True))
                z_norm_bank.append(z_norm)
                
            z_norm_bank = torch.cat(z_norm_bank, dim=0).contiguous()

            spectrum = calc_singular(z_norm_bank)
            print("spectrum:", spectrum)
            savemat("barlowtwinsplus_"+args.dataset+".mat", {"spectrum":spectrum})

    def calc_knn_accuracy(self, epoch, args):
        self.eval()
        total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            for (data, _), target in tqdm(self.memory_train_dataloader, desc='calculation of knn accuracy'):
                _, feature, _, z_norm = self.encoder(data.cuda(non_blocking=True))
                feature_bank.append(feature)
                target_bank.append(target)

            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            #feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
            if args.dataset == "STL-10":
                feature_labels = torch.tensor(self.memory_train_dataloader.dataset.labels, device=feature_bank.device).long()
            else:
                feature_labels = torch.tensor(self.memory_train_dataloader.dataset.targets, device=feature_bank.device)

            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(self.memory_val_dataloader)
            for (data, _), target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                _, feature, _, _ = self.encoder(data)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = sim_weight.exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * args.k, args.num_class, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, args.num_class) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

        self.train()
        return total_top1 / total_num * 100, total_top5 / total_num * 100
    
    def linear_train_val(self, args, epoch, optimizer, criterion, is_train=True):
        if is_train==True:
            data_loader = self.linear_train_dataloader
        else:
            data_loader = self.linear_val_dataloader

        self.train() if is_train else self.eval()
        total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
        with (torch.enable_grad() if is_train else torch.no_grad()):
            for data, target in data_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature, _, _, _ = self.encoder(data)
                out = self.projector(feature)
                loss = criterion(out, target)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_num += data.size(0)
                total_loss += loss.item() * data.size(0)
                prediction = torch.argsort(out, dim=-1, descending=True)
                total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Val', epoch, args.linear_epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100,))

        return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

    def linear_model(self, args, epoch):
        self.train()
        results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [], 'test_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'alignment':[], 'uniformity':[], 'wasserstein_distance':[]}
        final_results = {'train_loss': 0, 'train_acc@1': 0, 'train_acc@5': 0, 'test_loss': 0, 'test_acc@1': 0, 'test_acc@5': 0}
        
        for param in self.encoder.parameters():
            param.requires_grad = False

        alignment = 0.0
        uniformity = 0.0
        wasserstein_distance = 0.0

        optimizer = optim.SGD(self.projector.parameters(), 0.3, momentum=0.9, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.linear_epochs)
        loss_criterion = nn.CrossEntropyLoss().cuda()
        save_name = os.path.join(args.saver_dir, args.save_name_pre +"_"+str(epoch)+ '_linear.csv')
    
        for epoch in range(1, args.linear_epochs + 1):
            print("epoch:%d, lr:%4f"%(epoch, optimizer.param_groups[0]["lr"]))
            train_loss, train_acc_1, train_acc_5 = self.linear_train_val(args, epoch, optimizer, loss_criterion, is_train=True)
            results['train_loss'].append(train_loss)
            results['train_acc@1'].append(train_acc_1)
            results['train_acc@5'].append(train_acc_5)
            test_loss, test_acc_1, test_acc_5 = self.linear_train_val(args, epoch, optimizer, loss_criterion, is_train=False)
            results['test_loss'].append(test_loss)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            results['alignment'].append(alignment)
            results['uniformity'].append(uniformity)
            results['wasserstein_distance'].append(wasserstein_distance)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(save_name, index_label='epoch')
            if test_acc_1>final_results['test_acc@1']:
                final_results['train_loss'] = train_loss
                final_results['train_acc@1'] = train_acc_1
                final_results['train_acc@5'] = train_acc_5
                final_results['test_loss'] = test_loss
                final_results['test_acc@1'] = test_acc_1
                final_results['test_acc@5'] = test_acc_5
            scheduler.step()

        alignment = self.calc_alignment(args)
        uniformity, wasserstein_distance = self.calc_uniformity(args)
        results['train_loss'].append(final_results['train_loss'])
        results['train_acc@1'].append(final_results['train_acc@1'])
        results['train_acc@5'].append(final_results['train_acc@5'])
        results['test_loss'].append(final_results['test_loss'])
        results['test_acc@1'].append(final_results['test_acc@1'])
        results['test_acc@5'].append(final_results['test_acc@5'])
        results['alignment'].append(alignment)
        results['uniformity'].append(uniformity)
        results['wasserstein_distance'].append(wasserstein_distance)

        data_frame = pd.DataFrame(data=results, index=range(1, args.linear_epochs + 2))
        data_frame.to_csv(save_name, index_label='epoch')