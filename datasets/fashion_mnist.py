"""
 # Created on 12.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: MNIST dataset class
 #
"""

import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset):
    """The Fashion MNIST dataset."""

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        # No augmentation for fashion MNIST.
        train_set = torchvision.datasets.FashionMNIST(
            train=True, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
        return Dataset(train_set.data, train_set.targets)

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.FashionMNIST(
            train=False, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
        return Dataset(test_set.data, test_set.targets)

    def __init__(self, examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')

    @staticmethod
    def get_boruta_mask():
        # load the fashion_mnist_boruta_mask.npy file
        import numpy as np
        return np.load(os.path.join(get_platform().boruta_root, 'fashion_mnist_boruta_mask.npy'))


DataLoader = base.DataLoader
