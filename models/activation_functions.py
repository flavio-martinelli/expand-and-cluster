"""
 # Created on 10.09.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: 
 #
"""

import torch
import torch.nn.functional as F


# The symmetry registry maps each activation function to its symmetry type. When adding a new activation function, add
# its name and symmetry type to the registry.
symmetry_registry = {
    'relu': "even_linear_positive_scaling",
    'sigmoid': "odd_constant",
    'tanh': "odd",
    'softplus': "even_linear",
    'gelu': "even_linear",
    'silu': "even_linear",
    'leaky_relu': "even_linear_positive_scaling",
    'g': "none",
    '_g': "none",
    'identity': "odd"
}


def get_symmetry(act_fun):
    return symmetry_registry[act_fun.__name__]


def relu():
    return F.relu


def sigmoid():
    return F.sigmoid


def tanh():
    return F.tanh


def softplus():
    return F.softplus


def silu():
    return F.silu


def gelu():
    return F.gelu


def leaky_relu():
    return F.leaky_relu


def g():
    return _g


def _g(x):
    return torch.sigmoid(4 * x) + F.softplus(x)


def identity():
    return _id


def _id(x):
    return x
