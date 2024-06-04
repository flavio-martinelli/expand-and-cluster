"""
 # Created on 10.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: fully connected CIFAR-10 net
 #
"""

import torch.nn as nn

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    '''A LeNet fully-connected model for CIFAR-10'''

    def __init__(self, plan, initializer, act_fun, outputs=10):
        super(Model, self).__init__()

        self.act_fun = act_fun

        layers = []
        current_size = 32*32*3  # number of pixels in CIFAR-10 image.
        for size in plan:
            layers.append(nn.Linear(current_size, size))
            current_size = size

        self.fc_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(current_size, outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.fc_layers:
            x = self.act_fun(layer(x))

        return self.fc(x)

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_lenet') and
                len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]))

    @staticmethod
    def get_model_from_name(model_name, initializer, act_fun, outputs=None):
        """The name of a model is cifar_lenet_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'cifar_lenet_300_100'.
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        plan = [int(n) for n in model_name.split('_')[2:]]
        return Model(plan, initializer, act_fun, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_lenet_300_100',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            lr=0.1,
            training_steps='200ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight',
        )

        extraction_hparams = hparams.ExtractionHparams(
            gamma=None,
            beta=None
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, extraction_hparams)
