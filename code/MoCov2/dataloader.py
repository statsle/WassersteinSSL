import random
import torch
import os
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.datasets as datasets
import torch.utils.data as data
from glob import glob
from PIL import Image

class GaussianBlur:
    def __init__(self, sigma = [0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.solarize(img)

##For CIFAR10 and CIFAR-100(32x32) 
class CIFARPairTransform:
    def __init__(self, train_transform = True):
        if train_transform is True:
            self.transform_1 = transforms.Compose([
                transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
            self.transform_2 = transforms.Compose([
                transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            self.transform_1 = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ])
            self.transform_2 = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ])

    def __call__(self, x):
        y1 = self.transform_1(x)
        y2 = self.transform_2(x)
        return y1, y2

class CIFARSingleTransform:
    def __init__(self, train_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __call__(self, x):
        y = self.transform(x)
        return y

##For STL(96x96---32x32) 
class STLPairTransform:
    def __init__(self, train_transform = True):
        if train_transform is True:
            self.transform_1 = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
            ])
            self.transform_2 = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
            ])
        else:
            self.transform_1 = transforms.Compose([
                    transforms.Resize(32, interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                    ])
            self.transform_2 = transforms.Compose([
                    transforms.Resize(32, interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                    ])

    def __call__(self, x):
        y1 = self.transform_1(x)
        y2 = self.transform_2(x)
        return y1, y2

class STLSingleTransform:
    def __init__(self, train_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(32, interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))])

    def __call__(self, x):
        y = self.transform(x)
        return y

##For Tiny Imagenet(64x64)
class TinyImageNetPairTransform:
    def __init__(self, train_transform = True):
        if train_transform is True:
            self.transform_1 = transforms.Compose([
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([GaussianBlur()], p=0.5),
                    transforms.RandomApply([Solarization()], p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])

            self.transform_2 = transforms.Compose([
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([GaussianBlur()], p=0.5),
                    transforms.RandomApply([Solarization()], p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])
        else:
            self.transform_1 = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                    ])
            self.transform_2 = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                    ])

    def __call__(self, x):
        y1 = self.transform_1(x)
        y2 = self.transform_2(x)
        return y1, y2

class TinyImageNetSingleTransform:
    def __init__(self, train_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))])

    def __call__(self, x):
        y = self.transform(x)
        return y


class TinyImageNetSubset(data.Dataset):
    def __init__(self, args, split, transform=None):
        super(TinyImageNetSubset, self).__init__()
        data_dir = os.path.join(args.data_dir, split)
        self.transform = transform

        name = os.path.join(args.data_dir, "wnids.txt")
        with open(name, 'r') as f:
            result = f.read().splitlines()
        part_dir = result[0:args.num_class]
        self.imgs = []
        self.targets = []

        if split=="train":
            subdirs = []
            for line in part_dir:
                subdir = os.path.join(os.path.join(data_dir, line), "images")
                subdirs.append(subdir)
            label = 0
            for subdir in subdirs:
                files = sorted(glob(os.path.join(subdir, '*.JPEG')))
                for f in files:
                    self.imgs.append(f)
                    self.targets.append(label)
                label +=1

        elif split=="val":
            self.imgs = []
            self.targets = []
            label = 0
            dict_labels = {}
            for subdir in part_dir:
                dict_labels[subdir] = label
                label += 1

            path = os.path.join(data_dir, "images")
            label_name = os.path.join(data_dir, "val_annotations.txt")
            with open(label_name, 'r') as f:
                result = f.read().splitlines()

            for line in result:
                line_split = line.split("\t")
                image_name = line_split[0]
                label_name = line_split[1]
                img_path = os.path.join(path, image_name)
                if label_name in dict_labels:
                    label = dict_labels[label_name]
                    self.imgs.append(img_path)
                    self.targets.append(label) 

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        label = torch.tensor(self.targets[index]).long()
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


