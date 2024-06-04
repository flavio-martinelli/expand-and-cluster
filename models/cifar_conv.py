"""
 # Created on 10.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Cifar conv net + fc layers
 #
"""

import torch
import torch.nn as nn

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from models.activation_functions import _id
from pruning import sparse_global


class Model(base.Model):
    """A classic conv neural network designed for MNIST."""

    class ConvModule(nn.Module):
        """A single convolutional module."""

        def __init__(self, in_filters, out_filters, act_fun):
            super(Model.ConvModule, self).__init__()
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            # self.bn = nn.BatchNorm2d(out_filters)
            self.act_fun = act_fun

        def forward(self, x):
            return self.act_fun(self.conv(x))

    class LinearModule(nn.Module):
        """A single linear module."""

        def __init__(self, in_features, out_features, act_fun):
            super(Model.LinearModule, self).__init__()
            self.fc = nn.Linear(in_features, out_features)
            self.act_fun = act_fun

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.act_fun(self.fc(x))

    def __init__(self, plan, initializer, act_fun, outputs=10):
        super(Model, self).__init__()

        self.act_fun = act_fun

        layers = []
        filters = 3  # Input channels is 1 for CIFAR

        for spec in plan:
            if spec[0] == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif spec[0] == 'C':
                layers.append(Model.ConvModule(filters, int(spec[1:]), self.act_fun))
                filters = int(spec[1:])
            elif spec[0] == 'F':
                # check size of input tensor after convolutions by sending dummy tensor through the layers
                shape = self.check_size_after_conv(layers)
                layers.append(Model.LinearModule(shape, int(spec[1:]), self.act_fun))
                filters = int(spec[1:])

        shape = self.check_size_after_conv(layers)
        layers.append(Model.LinearModule(shape, 10, _id))  # identity act_fun

        self.layers = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        return self.layers(x)

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_conv_') and
                len(model_name.split('_')) > 3)  # TODO: make it more robust

    @staticmethod
    def get_model_from_name(model_name, initializer, act_fun, outputs=10):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs or 10
        plan = model_name.split('_')[2:]

        return Model(plan, initializer, act_fun, outputs)

    @staticmethod
    def check_size_after_conv(layers):
        dummy_tensor = torch.zeros(1, 3, 32, 32)
        dummy_layers = nn.Sequential(*layers)

        dummy_tensor = dummy_layers(dummy_tensor)
        dummy_tensor = dummy_tensor.view(dummy_tensor.size(0), -1)
        shape = dummy_tensor.shape[1]
        return shape


    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_conv_C16_M_F32',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=1e-3,
            training_steps='30ep'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight'
        )

        extraction_hparams = hparams.ExtractionHparams(
            gamma=None,
            beta=None
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, extraction_hparams)
