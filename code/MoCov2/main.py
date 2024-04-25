import sys
import os
import time
import numpy as np
import torch
from torch import nn, optim
from model import MoCoplus
from util import Logger
import config

def main(args):
    model = MoCoplus(args).to(args.device)
    if args.stage==0:
        ###training stage
        model.train_model(args)
    elif args.stage==1:
        epoch_list = [100, 200, 300, 400, 500]
        for epoch in epoch_list:
            model.load_state_dict(torch.load('./{}/{}_{}_model.pth'.format(args.saver_dir, args.save_name_pre, epoch), map_location='cpu'), strict=False)
            model.linear_model(args, epoch)
    elif args.stage==2:
        model.load_state_dict(torch.load('./saver/CIFAR-100/MoCoplus_12_CIFAR-100_2048_256_256_500_4096_0.2_1.0_0.0_500_model.pth', map_location='cpu'), strict=False)
        model.dimensional_analysis(args)


            

if __name__ == '__main__':
    args = config.parse_arg()
    sys.stdout = Logger(args)
    dict_args = vars(args)
    for k, v in zip(dict_args.keys(), dict_args.values()):
        print("{0}: {1}".format(k, v))
    
    main(args)