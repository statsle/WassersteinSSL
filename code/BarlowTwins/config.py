import argparse
import os
import torch
import numpy as np
import random
import string
import datetime

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch version of BarlowTwinsplus.') 
    ###job params
    parser.add_argument('--use_random', action='store_true', help='whether to randomly generate seed')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (also job id)')
    parser.add_argument('--model_name', default='BarlowTwinsplus', help='the name of models', choices=['BarlowTwinsplus'])
    parser.add_argument('--batch_size', type=int, default=256, help="the size of batch samples")
    parser.add_argument('--stage', type=int, default=0, help="0 train model; 1 linear evaluate model")
    parser.add_argument('--epochs', type=int, default=1000, help="training epochs")
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR', help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.005, type=float, metavar='LR', help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--linear_epochs', type=int, default=100, help="training epochs")
    parser.add_argument('--num_class', type=int, default=10, help="the categories of different labels")
    parser.add_argument('--hidden_dim', type=int, default=2048, help="dimension of contrastive representation")
    parser.add_argument('--prej_dim', type=int, default=256, help="dimension of contrastive representation")
    parser.add_argument('--dataset', default='CIFAR-10', help='dataset for training', choices=['CIFAR-10', 'STL-10', 'CIFAR-100', 'Tiny-Imagenet'])
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--lambd', type=float, default=0.005, help="the hyperparameter of off-diagonal cross-correlation")
    parser.add_argument('--lambd_max', type=float, default=0.0, help="the hyperparameter of Wasserstein Distance")
    parser.add_argument('--lambd_min', type=float, default=0.0, help="the hyperparameter of Wasserstein Distance")
    parser.add_argument('--data_dir', default="/data/fxh/data/", type=str, help='the directory of train or test dataset')
    parser.add_argument('--saver_dir', default="./saver", type=str, help='the directory of train or test dataset')
    args = parser.parse_args()

    args.save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.model_name, args.seed, args.dataset, args.hidden_dim, args.prej_dim, args.batch_size, args.epochs, args.lambd, args.lambd_max, args.lambd_min)
    print("save_name_pre:", args.save_name_pre)

    args.data_dir = os.path.join(args.data_dir, args.dataset)
    print("data dir:", args.data_dir)

    if args.dataset == "CIFAR-10":
        args.num_class = 10
    elif args.dataset == "CIFAR-100":
        args.num_class = 100
    elif args.datatset == "STL-10":
        args.num_class = 10

    ###process args
    if args.use_random:
        args.seed = random.randint(0, 1e8)   
    args.device = torch.device("cuda")

    args.log_dir = args.saver_dir
    if not os.path.exists(args.saver_dir):
        os.makedirs(args.saver_dir)

    ####Fixed the seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return args
