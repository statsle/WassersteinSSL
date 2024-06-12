###MoCo + w2 model

###On CIFAR-10 dataset
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-10 --dataset CIFAR-10 --use_w2 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 0 --seed 12 --queue_size 4096 --lambd_max 1.0 --lambd_min 1.0
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-10 --dataset CIFAR-10 --use_w2 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 1 --seed 12 --queue_size 4096 --lambd_max 1.0 --lambd_min 1.0

###On CIFAR-100 dataset
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-100 --dataset CIFAR-100 --use_w2 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 0 --seed 12 --queue_size 4096 --lambd_max 1.0 --lambd_min 1.0
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-100 --dataset CIFAR-100 --use_w2 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 1 --seed 12 --queue_size 4096 --lambd_max 1.0 --lambd_min 1.0
