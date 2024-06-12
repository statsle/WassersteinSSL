### Vanilla BYOL model

###On CIFAR-10 dataset
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-10 --dataset CIFAR-10 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 0 --seed 12 --queue_size 4096 --lambd_max 0.0 --lambd_min 0.0
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-10 --dataset CIFAR-10 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 1 --seed 12 --queue_size 4096 --lambd_max 0.0 --lambd_min 0.0

###On CIFAR-100 dataset
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-100 --dataset CIFAR-100 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 0 --seed 12 --queue_size 4096 --lambd_max 0.0 --lambd_min 0.0
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 500 --linear_epochs 100 --saver_dir ./saver/CIFAR-100 --dataset CIFAR-100 --hidden_dim 2048 --prej_dim 256 --batch_size 256 --stage 1 --seed 12 --queue_size 4096 --lambd_max 0.0 --lambd_min 0.0
