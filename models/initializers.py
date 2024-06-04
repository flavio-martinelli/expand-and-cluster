"""
 # Created on 10.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Contains possible initializers
 #
"""

import torch
import models.students_mnist_lenet, models.students_cifar_lenet


def binary(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
        sigma = w.weight.data.std()
        w.weight.data = torch.sign(w.weight.data) * sigma
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
            isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule):
        torch.nn.init.kaiming_normal_(w.fc)
        sigma = w.fc.data.std()
        w.weight.data = torch.sign(w.fc.data) * sigma


def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
            isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule) or \
            isinstance(w, models.students_cifar_lenet.Model.ParallelFCModule) or \
            isinstance(w, models.students_cifar_lenet.Model.InitialParallelFCModule):
        torch.nn.init.kaiming_normal_(w.fc)


def kaiming_uniform(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(w.weight)
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
            isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule):
        torch.nn.init.kaiming_uniform_(w.fc)


def orthogonal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(w.weight)
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
            isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule):
        torch.nn.init.orthogonal_(w.fc)
