"""
 # Created on 09.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Optimizers class
 #
"""

import bisect
import numpy as np
import torch

from foundations.hparams import TrainingHparams
from foundations.step import Step
from models.base import Model


def get_optimizer(training_hparams: TrainingHparams, model: Model) -> torch.optim.Optimizer:
    if training_hparams.optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=training_hparams.lr,
            momentum=training_hparams.momentum or training_hparams.nesterov_momentum or 0,
            weight_decay=training_hparams.weight_decay or 0,
            nesterov=training_hparams.nesterov_momentum is not None and training_hparams.nesterov_momentum > 0
        )
    elif training_hparams.optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=training_hparams.lr,
            weight_decay=training_hparams.weight_decay or 0
        )

    raise ValueError('No such optimizer: {}'.format(training_hparams.optimizer_name))


def get_lr_schedule(training_hparams: TrainingHparams, optimizer: torch.optim.Optimizer, iterations_per_epoch: int):
    lambdas = [lambda it: 1.0]

    # if plateau_opt is set
    if training_hparams.plateau_opt:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=training_hparams.delta,
                                                         patience=training_hparams.patience,
                                                         threshold=0.0,
                                                         cooldown=training_hparams.cooldown,
                                                         min_lr=1e-7,
                                                         verbose=True)
        return sch

    # Drop the learning rate according to delta at the specified milestones.
    if bool(training_hparams.delta) != bool(training_hparams.milestone_steps):
        raise ValueError('milestones and delta hyperparameters must both be set or not at all.')
    if training_hparams.milestone_steps:
        milestones = [Step.from_str(x, iterations_per_epoch).iteration
                      for x in training_hparams.milestone_steps.split(',')]
        lambdas.append(lambda it: training_hparams.delta ** bisect.bisect(milestones, it))

    # Add linear learning rate warmup if specified.
    if training_hparams.warmup_steps:
        warmup_iters = Step.from_str(training_hparams.warmup_steps, iterations_per_epoch).iteration
        lambdas.append(lambda it: min(1.0, it / warmup_iters))

    # Combine the lambdas.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: np.product([l(it) for l in lambdas]))
