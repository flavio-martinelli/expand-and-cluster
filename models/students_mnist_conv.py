"""
 # Created on 10.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Mnist conv net + fc layers
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

    class ConvNet(nn.Module):

        class ConvModule(nn.Module):
            """A single convolutional module."""

            def __init__(self, in_filters, out_filters, act_fun):
                super(Model.ConvNet.ConvModule, self).__init__()
                self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
                # self.bn = nn.BatchNorm2d(out_filters)
                self.act_fun = act_fun

            def forward(self, x):
                return self.act_fun(self.conv(x))

        class LinearModule(nn.Module):
            """A single linear module."""

            def __init__(self, in_features, out_features, act_fun):
                super(Model.ConvNet.LinearModule, self).__init__()
                self.fc = nn.Linear(in_features, out_features)
                self.act_fun = act_fun

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.act_fun(self.fc(x))

        def __init__(self, plan, initializer, act_fun, outputs=10):
            super(Model.ConvNet, self).__init__()

            self.act_fun = act_fun
            layers = []
            filters = 1  # Input channels is 1 for MNIST
            for spec in plan:
                if spec[0] == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif spec[0] == 'C':
                    layers.append(Model.ConvNet.ConvModule(filters, int(spec[1:]), self.act_fun))
                    filters = int(spec[1:])
                elif spec[0] == 'F':
                    # check size of input tensor after convolutions by sending dummy tensor through the layers
                    shape = self.check_size_after_conv(layers)
                    layers.append(Model.ConvNet.LinearModule(shape, int(spec[1:]), self.act_fun))
                    filters = int(spec[1:])
            shape = self.check_size_after_conv(layers)
            layers.append(Model.ConvNet.LinearModule(shape, 10, _id))  # identity act_fun
            self.layers = nn.Sequential(*layers)
            self.apply(initializer)

        def forward(self, x):
            return self.layers(x)

        @staticmethod
        def check_size_after_conv(layers):
            dummy_tensor = torch.zeros(1, 1, 28, 28)
            dummy_layers = nn.Sequential(*layers)

            dummy_tensor = dummy_layers(dummy_tensor)
            dummy_tensor = dummy_tensor.view(dummy_tensor.size(0), -1)
            shape = dummy_tensor.shape[1]
            return shape

    def __init__(self, plan, initializer, act_fun, outputs=10):
        super(Model, self).__init__()

        self.act_fun = act_fun
        self.N = plan[0]
        plan = plan[1].split('_')[1:]

        self.module_list = nn.ModuleList()
        for n in range(self.N):
            self.module_list.append(Model.ConvNet(plan, initializer, act_fun, outputs))

        self.criterion = self.loss_fn

    def forward(self, x):
        out = [conv(x) for conv in self.module_list]
        return torch.stack(out, dim=-1)  # dimensions are [batch_size, 10, N]

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('students_mnist_conv') and
                len(model_name.split('_')) > 3)  # TODO: make it more robust

    @staticmethod
    def get_model_from_name(model_name, initializer, act_fun, outputs=10):
        """The name of a model is students_mnist_conv(N)_CW1[_CW2_M_FW3...].

        W1, W2, etc. are the number of neurons or channels in layer (C for convolutional, F for fully connected
        excluding the output layer (10 neurons by default). The number of nets run in parallel is set by the parameter N
        To run 5 ConvNets with 16 channels in the first hidden layer, maxpool, 100 neurons in the second hidden layer,
        and 10 output neurons -> 'students_mnist_conv(5)_C16_M_F100'. If N is omitted (i.e. mnist_lenet_students()_W1[
        _W2...] then N is set to 1).
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))
        # TODO: change '(' and ')' into some other characters. They are escape chars for the command line : (
        N = int(model_name[model_name.find("(")+1:model_name.find(")")])
        if N == "":
            N = 1
        plan = [N]
        plan.append(model_name.split(')')[-1])
        return Model(plan, initializer, act_fun, outputs)

    @staticmethod
    def loss_fn(y_hat, y):
        """y_hat is the network's output, y is the label"""
        overall_loss = Model.individual_losses(y_hat, y).sum()
        return overall_loss

    @staticmethod
    def individual_losses(y_hat, y):
        y_repeats = y.unsqueeze(2).repeat(1, 1, y_hat.shape[-1])
        return (y_hat - y_repeats).square().mean(dim=(0, 1)).squeeze()

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='mnist_conv_C16_M_F32',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
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
