"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Runner for training experiment
 #
"""

import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from training import train
from training.desc import TrainingDesc
from training.plotting import plot_metrics
from training.wandb_init import wandb_init


@dataclass
class TrainingRunner(Runner):
    global_seed: int
    desc: TrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(args.global_seed, TrainingDesc.create_from_args(args),
                              not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.global_seed))

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print(f"running on: {get_platform().device_str}")
            print('='*82 + f'\nTraining a Model (Seed {self.global_seed})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.global_seed)}' + '\n' + '='*82 + '\n')
        self.desc.save(self.desc.run_path(self.global_seed))

        wandb_successful_login = wandb_init([self.desc.model_hparams,
                                             self.desc.dataset_hparams,
                                             self.desc.training_hparams],
                                            run_type='train',
                                            summary_infos={'local_hashname': self.desc.hashname.split('_')[-1]})

        train.standard_train(
            models.registry.get(self.desc.model_hparams), self.desc.run_path(self.global_seed),
            self.desc.dataset_hparams, self.desc.training_hparams, evaluate_every_epoch=self.evaluate_every_epoch)

        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='test_accuracy', vlims=[0.9, 1.0])
        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='train_accuracy', vlims=[0.9, 1.0])
        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='test_loss', logscale=True)
        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='train_loss', logscale=True)

