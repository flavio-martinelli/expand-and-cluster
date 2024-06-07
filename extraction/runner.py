"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Runner for Expand-and-Cluster experiment
 #
"""
import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from extraction import expand_and_cluster
from extraction.desc import ExtractionDesc
from training.plotting import plot_metrics
from training.wandb_init import wandb_init

sep = '='*140
sep2 = '-'*140

@dataclass
class ExtractionRunner(Runner):
    global_seed: int
    desc: ExtractionDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Runs Expand-and-Cluster on the selected model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        ExtractionDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return ExtractionRunner(args.global_seed, ExtractionDesc.create_from_args(args),
                                not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.global_seed))

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print(f"running on: {get_platform().device_str}")
            print(sep + f'\nExtracting model\n' + sep2)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.global_seed)}' + '\n' + sep + '\n')
            print(sep2 + f'\nTraining student networks\n' + sep2)
        self.desc.save(self.desc.run_path(self.global_seed))

        wandb_successful_login = wandb_init([self.desc.model_hparams,
                                             self.desc.dataset_hparams,
                                             self.desc.training_hparams,
                                             self.desc.extraction_hparams],
                                            run_type='ec',
                                            summary_infos={'local_hashname': self.desc.hashname.split('_')[-1]})

        students, losses = expand_and_cluster.train_students(
                    models.registry.get(self.desc.model_hparams), self.desc.run_path(self.global_seed),
                    self.desc.dataset_hparams, self.desc.training_hparams,
                    evaluate_every_epoch=self.evaluate_every_epoch)

        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='train_individual_losses',
                     logscale=True)

        if self.verbose and get_platform().is_primary_process:
            print(sep + f'\nReconstruction\n' + sep2)

        print(f'Reconstruction Location: {self.desc.extraction_path(self.global_seed)}' + '\n' + sep2 + '\n')

        expand_and_cluster.reconstruct(students, losses, self.desc.extraction_path(self.global_seed),
                                       self.desc.extraction_hparams, self.desc.dataset_hparams,
                                       self.desc.training_hparams, self.desc.model_hparams,
                                       verbose=self.verbose, layer=layer)


def find_best_seed(it_per_epoch, path):
    import os
    import numpy as np
    from utils.utils import find_final_model_step
    from training.metric_logger import MetricLogger

    seed_loss = []
    for seed_path in os.listdir(path):
        if not seed_path.startswith('seed'):
            continue
        seed = int(seed_path.split('_')[-1])
        model_path = os.path.join(path, seed_path, "main")
        last_step = find_final_model_step(model_path, it_per_epoch)
        if last_step is None:
            print(f"Skipping seed {seed} because no final model was found.")
            continue
        logger = MetricLogger.create_from_file(model_path)
        current_loss = logger.get_data('train_individual_losses')[-1][1][0]
        seed_loss.append([seed, current_loss])
    seed_loss = np.array(seed_loss)
    best_seed = seed_loss[np.argmin(seed_loss[:, 1])][0]
    return int(best_seed), seed_loss[np.argmin(seed_loss[:, 1])][1]
