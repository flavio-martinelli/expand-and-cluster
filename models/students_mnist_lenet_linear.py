"""
 # Created on 11.09.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Students of the MNIST lenet (fully connected) network types with linear component
 #
"""
import typing

import numpy as np
import torch
import torch.nn as nn

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base, students_mnist_lenet
from pruning import sparse_global
from models.activation_functions import identity, get_symmetry


class Model(base.Model):
    '''Training many LeNet fully-connected models in parallel for MNIST'''

    class InitialParallelFCModule(nn.Module):
        """A module for N linear layers run in a unique tensor (first layer only) + linear component to each layer.
        The last neuron of each hidden layer is treated as linear neuron."""

        def __init__(self, d_in, d_out, N, act_fun):
            super(Model.InitialParallelFCModule, self).__init__()
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))
            self.b = nn.Parameter(torch.zeros(d_out, N))
            self.act_fun = act_fun

        def forward(self, x):
            """ i:input size -- b:sample batch size -- h:hidden layer dim -- o:output_size -- n:number of nets """
            x = torch.einsum('bi,ihn->bhn', x, self.fc) + self.b.expand([x.shape[0]] + list(self.b.shape))
            x_hidden = self.act_fun(x[:, :-1, :])  # Apply activation function to all but last neuron
            x_lin = x[:, -1, :]  # Linear component
            return torch.cat([x_hidden, x_lin.unsqueeze(1)], dim=1)

    class ParallelFCModule(nn.Module):
        """A module for N linear layers run in a unique tensor + linear component to each layer."""

        def __init__(self, d_in, d_out, N, act_fun):
            super(Model.ParallelFCModule, self).__init__()
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))
            self.b = nn.Parameter(torch.zeros(d_out, N))
            self.act_fun = act_fun

        def forward(self, x):
            """ i:input size -- b:sample batch size -- h:hidden layer dim -- o:output_size -- n:number of nets """
            x = torch.einsum('bin,ihn->bhn', x, self.fc) + self.b.expand([x.shape[0]] + list(self.b.shape))
            x_hidden = self.act_fun(x[:, :-1, :])  # Apply activation function to all but last neuron
            x_lin = x[:, -1, :]  # Linear component
            return torch.cat([x_hidden, x_lin.unsqueeze(1)], dim=1)

    class FinalParallelFCModule(nn.Module):
        """A module for N linear layers run in a unique tensor + linear component to each layer."""

        def __init__(self, d_in, d_out, N):
            super(Model.FinalParallelFCModule, self).__init__()
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))
            self.b = nn.Parameter(torch.zeros(d_out, N))

        def forward(self, x):
            """ i:input size -- b:sample batch size -- h:hidden layer dim -- o:output_size -- n:number of nets """
            return torch.einsum('bin,ihn->bhn', x, self.fc) + self.b.expand([x.shape[0]] + list(self.b.shape))


    def __init__(self, plan, initializer, act_fun, outputs=10):
        super(Model, self).__init__()

        self.act_fun = act_fun
        self.plan = plan
        self.N = plan[0]
        self.initializer = initializer
        self.outputs = outputs

        layers = []
        current_size = 784  # 28 * 28 = number of pixels in MNIST image.
        for i, size in enumerate(self.plan[1:]):
            if i == 0:  # The first layer has a different dimensionality (input tensor is not 3D)
                layers.append(self.InitialParallelFCModule(current_size, size, self.N, self.act_fun))
            else:
                layers.append(self.ParallelFCModule(current_size, size, self.N, self.act_fun))
            current_size = size
        layers.append(self.FinalParallelFCModule(current_size, outputs, self.N))
        self.fc_layers = nn.ModuleList(layers)

        self.criterion = self.loss_fn
        self.apply(self.initializer)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.fc_layers:
            x = layer(x)
        return x

    @property
    def output_layer_names(self):
        out_name = list(self.named_modules())[-1][0]
        return [f'{out_name}.fc', f'{out_name}.b', f'{out_name}.m', f'{out_name}.q']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('students_mnist_lenet(') and
                len(model_name.split('_')) > 3 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[3:]]) and
                model_name.find(")") != -1)

    @staticmethod
    def get_model_from_name(model_name, initializer, act_fun, outputs=None):
        """The name of a model is mnist_lenet_students(N)_W1[_W2...].

        W1, W2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). The number of nets run in parallel is set by the parameter N. To run 5
        LeNets with 300 neurons in the first hidden layer, 100 neurons in the second hidden layer, and 10 output
        neurons -> 'mnist_lenet_students(5)_300_100'. If N is omitted (i.e. mnist_lenet_students()_W1[_W2...] then N
        is set to 1).
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))
        # TODO: change '(' and ')' into some other characters. They are escape chars for the command line : (
        N = int(model_name[model_name.find("(")+1:model_name.find(")")])
        if N == "":
            N = 1
        plan = [N]
        plan.extend([int(n) for n in model_name.split('_')[3:]])
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

    @staticmethod
    def load_from_student_mnist_lenet(model: students_mnist_lenet.Model):
        # Loads the parameters of a students_mnist_lenet model into a students_mnist_lenet_linear model.
        new_model = Model(model.plan, model.initializer, model.act_fun, model.outputs)
        new_model.load_state_dict(model.state_dict(), strict=False)
        # add one linear neuron to each hidden layer
        for i, param in enumerate(new_model.parameters()):
            if i == len(list(new_model.parameters())) - 2:  # output layer
                break
            if i % 2 == 0:
                param.data = torch.cat([param.data, torch.zeros(param.shape[0], 1, param.shape[2])], dim=1)
            else:
                param.data = torch.cat([param.data, torch.zeros(1, param.shape[1])], dim=0)
        return new_model

    @property
    def loss_criterion(self):
        return self.criterion


    @property
    def prunable_layer_names(self) -> typing.List[str]:
        """A list of the names of Tensors of this model that are valid for pruning.
        By default, only the weights of convolutional and linear layers are prunable, not biases.
        """

        return [name + '.fc' for name, module in self.named_modules() if
                isinstance(module, self.InitialParallelFCModule) or
                isinstance(module, self.ParallelFCModule)]


    @staticmethod
    def default_hparams():

        model_hparams = hparams.ModelHparams(
            model_name='students_mnist_lenet(10)_300_100',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=512
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.001,
            training_steps='25000ep',
            plateau_opt=True,
            delta=1/np.sqrt(10),
            patience=100,
            cooldown=200
        )

        last_hidden_idx = len(model_hparams.model_name.split('_')[1:])

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore=f'fc_layers.{last_hidden_idx}.fc'  # By default do not prune output layer
        )

        extraction_hparams = hparams.ExtractionHparams(
            gamma=0.5,
            beta=6,
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, extraction_hparams)
