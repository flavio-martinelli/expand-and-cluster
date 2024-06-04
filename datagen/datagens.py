"""
 # Created on 13.09.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Contains all data generator policies functions
 #
"""
import os
import torch
import torchvision

from PIL import Image
from datasets.cifar10 import CIFAR10
from platforms.platform import get_platform

# TODO: might be turned into a basic class to enforce function signatures
# TODO: implement augmentation for the CIFAR10 case


def mnist(augment=None, d_in=None):
    train_set = torchvision.datasets.MNIST(
        train=True, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
    X = []
    for im in train_set.data.numpy():
        im = Image.fromarray(im, mode='L')
        for t in transforms:
            im = t(im)
        X.append(im)
    X = torch.concat(X)
    return X


def mnist_conv(augment=None, d_in=None):
    X = mnist(augment, d_in)
    return X.unsqueeze(1)


def cifar10(augment=None, d_in=None):
    # augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
    train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    X = []
    for im in train_set.data:
        for t in transforms:
            im = t(im)
        X.append(im)
    X = torch.stack(X, dim=0)
    return X


def cifar10_conv(augment=None, d_in=None):
    X = cifar10(augment, d_in)
    return X


def fashion_mnist(augment=None, d_in=None):
    train_set = torchvision.datasets.FashionMNIST(
        train=True, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
    X = []
    for im in train_set.data.numpy():
        im = Image.fromarray(im, mode='L')
        for t in transforms:
            im = t(im)
        X.append(im)
    X = torch.concat(X)
    return X
