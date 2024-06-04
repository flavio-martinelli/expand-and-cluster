"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Runner for Lottery Ticket Hypothesis experiment
 #
"""

import argparse
import os

from cli import shared_args
from dataclasses import dataclass

from extraction.expand_and_cluster import train_students
from foundations.runner import Runner
import models.registry
from foundations.step import Step
from lottery.desc import LotteryDesc
from platforms.platform import get_platform
import pruning.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
import models.students_mnist_lenet  # TODO: for the future make it more general (import any type of student net)
from training.plotting import plot_metrics
from utils.utils import find_final_model_step

sep = '='*140
sep2 = '-'*140


@dataclass
class LotteryRunner(Runner):
    global_seed: int
    levels: int
    desc: LotteryDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return 'Run a lottery ticket hypothesis experiment.'

    @staticmethod
    def _add_levels_argument(parser):
        help_text = \
            'The number of levels of iterative pruning to perform. At each level, the network is trained to ' \
            'completion, pruned, and rewound, preparing it for the next lottery ticket iteration. The full network ' \
            'is trained at level 0, and level 1 is the first level at which pruning occurs. Set this argument to 0 ' \
            'to just train the full network or to N to prune the network N times.'
        parser.add_argument('--levels', required=True, type=int, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # Get preliminary information.
        defaults = shared_args.maybe_get_default_hparams()

        # Add the job arguments.
        shared_args.JobArgs.add_args(parser)
        lottery_parser = parser.add_argument_group(
            'Lottery Ticket Hyperparameters', 'Hyperparameters that control the lottery ticket process.')
        LotteryRunner._add_levels_argument(lottery_parser)
        LotteryDesc.add_args(parser, defaults)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'LotteryRunner':
        return LotteryRunner(args.global_seed, args.levels, LotteryDesc.create_from_args(args),
                             not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.global_seed, 0))

    def run(self) -> None:
        if self.verbose and get_platform().is_primary_process:
            print(sep + f'\nLottery Ticket Experiment (Seed {self.global_seed})\n' + sep2)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.global_seed, 0)}' + '\n' + sep + '\n')

        if get_platform().is_primary_process: self.desc.save(self.desc.run_path(self.global_seed, 0))
        if self.desc.pretrain_training_hparams: self._pretrain()
        if get_platform().is_primary_process: self._establish_initial_weights()
        get_platform().barrier()

        for level in range(self.levels+1):
            if get_platform().is_primary_process: self._prune_level(level)
            get_platform().barrier()
            self._train_level(level)

    # Helper methods for running the lottery.
    def _pretrain(self):
        location = self.desc.run_path(self.global_seed, 'pretrain')
        if models.registry.exists(location, self.desc.pretrain_end_step): return

        if self.verbose and get_platform().is_primary_process: print('-'*82 + '\nPretraining\n' + '-'*82)
        model = models.registry.get(self.desc.model_hparams, outputs=self.desc.pretrain_outputs)
        train.standard_train(model, location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams,
                             verbose=self.verbose, evaluate_every_epoch=self.evaluate_every_epoch)

    def _establish_initial_weights(self):
        location = self.desc.run_path(self.global_seed, 0)
        if models.registry.exists(location, self.desc.train_start_step): return

        new_model = models.registry.get(self.desc.model_hparams, outputs=self.desc.train_outputs)

        # If there was a pretrained model, retrieve its final weights and adapt them for training.
        if self.desc.pretrain_training_hparams is not None:
            pretrain_loc = self.desc.run_path(self.global_seed, 'pretrain')
            old = models.registry.load(pretrain_loc, self.desc.pretrain_end_step,
                                       self.desc.model_hparams, self.desc.pretrain_outputs)
            state_dict = {k: v for k, v in old.state_dict().items()}

            # Select a new output layer if number of classes differs.
            if self.desc.train_outputs != self.desc.pretrain_outputs:
                state_dict.update({k: new_model.state_dict()[k] for k in new_model.output_layer_names})

            new_model.load_state_dict(state_dict)

        new_model.save(location, self.desc.train_start_step)

    def _train_level(self, level: int):
        location = self.desc.run_path(self.global_seed, level)
        if models.registry.exists(location, self.desc.train_end_step): return

        # If pruning strategy is contained in 'pruning.registry.rewinding_strategies' then load the init at level 0,
        # otherwise continue training the model from the previous level.
        if level == 0 or self.desc.pruning_hparams.pruning_strategy in pruning.registry.rewinding_strategies:
            # load initialisation
            model = models.registry.load(self.desc.run_path(self.global_seed, 0), self.desc.train_start_step,
                                         self.desc.model_hparams, self.desc.train_outputs)
        else:
            # load latest model
            old_location = self.desc.run_path(self.global_seed, level-1)
            train_end_step = find_final_model_step(old_location, self.desc.train_end_step._iterations_per_epoch)
            if train_end_step is None:
                raise FileNotFoundError("No final model found in {}".format(old_location))
            model = models.registry.load(old_location, train_end_step, self.desc.model_hparams, self.desc.train_outputs)

        pruned_model = PrunedModel(model, Mask.load(location))
        pruned_model.save(location, self.desc.train_start_step)
        if self.verbose and get_platform().is_primary_process:
            print(sep2 + '\nPruning Level {}\n'.format(level) + sep2)

        if type(pruned_model.model) == models.students_mnist_lenet.Model:
            pruned_model.N = pruned_model.model.N
            pruned_model.individual_losses = pruned_model.model.individual_losses
            train_students(pruned_model, location, self.desc.dataset_hparams, self.desc.training_hparams,
                           start_step=self.desc.train_start_step, verbose=self.verbose,
                           evaluate_every_epoch=self.evaluate_every_epoch)
            plot_metrics(folder_path=location, metric_name='train_individual_losses', logscale=True)
        else:
            train.standard_train(pruned_model, location, self.desc.dataset_hparams, self.desc.training_hparams,
                                 start_step=self.desc.train_start_step, verbose=self.verbose,
                                 evaluate_every_epoch=self.evaluate_every_epoch)
            plot_metrics(folder_path=location, metric_name='test_accuracy', vlims=[0.9, 1.0])
            plot_metrics(folder_path=location, metric_name='train_accuracy', vlims=[0.9, 1.0])
            plot_metrics(folder_path=location, metric_name='test_loss', logscale=True)
            plot_metrics(folder_path=location, metric_name='train_loss', logscale=True)

    def _prune_level(self, level: int):
        new_location = self.desc.run_path(self.global_seed, level)
        if Mask.exists(new_location): return

        if level == 0:
            Mask.ones_like(models.registry.get(self.desc.model_hparams)).save(new_location)
        else:
            old_location = self.desc.run_path(self.global_seed, level-1)

            train_end_step = find_final_model_step(old_location, self.desc.train_end_step._iterations_per_epoch)
            if train_end_step is None:
                raise FileNotFoundError("No final model found in {}".format(old_location))

            model = models.registry.load(old_location, train_end_step, self.desc.model_hparams,
                                         self.desc.train_outputs)

            pruning.registry.get(self.desc.pruning_hparams)(model, Mask.load(old_location)).save(new_location)
