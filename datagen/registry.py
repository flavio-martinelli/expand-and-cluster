"""
 # Created on 13.09.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: A registry for all data generation policies
 #
"""

import torch
import numpy as np

from datagen.datagens import mnist, cifar10, fashion_mnist, mnist_conv, cifar10_conv
from foundations.hparams import DatasetHparams
from platforms.platform import get_platform

registered_datagens = {'cifar10': cifar10, 'mnist': mnist, 'fashion_mnist': fashion_mnist, 'mnist_conv': mnist_conv,
                       'cifar10_conv': cifar10_conv}


def get(dataset_hparams: DatasetHparams, model, use_augmentation):
    # Get the data generator policies and attach the resulting inputs together
    print("Generating dataset with teacher...")
    X = []
    if type(dataset_hparams.d_in) is not int:
        raise ValueError('Need to define input dimension --d_in: {} '.format(dataset_hparams.d_in))
    for policy in dataset_hparams.datagen.split("+"):
        if policy in registered_datagens:
            X.append(registered_datagens[policy](use_augmentation, dataset_hparams.d_in))
        else:
            raise ValueError('No such datagen: {} '.format(policy))
    with torch.no_grad():
        X = torch.concat(X, dim=0)
        y = model(X)
    return X, y
