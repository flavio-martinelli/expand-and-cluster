"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: class for representing set of hparams for the expand-and-cluster experiments
 #
"""

import argparse
import copy
import hashlib
from dataclasses import dataclass, fields
import os
import torch

from datasets import registry as datasets_registry
from foundations import desc, hparams, paths
from foundations.hparams import Hparams, ExtractionHparams
from foundations.step import Step
from lottery.desc import LotteryDesc
from platforms.platform import get_platform


@dataclass
class ExtractionDesc(desc.Desc):
    """The hyperparameters necessary to describe an Expand-and-Cluster run."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    extraction_hparams: hparams.ExtractionHparams

    @staticmethod
    def name_prefix(): return 'ec'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: LotteryDesc = None):
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        hparams.ExtractionHparams.add_args(parser, defaults=defaults.extraction_hparams if defaults else None)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'ExtractionDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        extraction_hparams = hparams.ExtractionHparams.create_from_args(args)
        return ExtractionDesc(model_hparams, dataset_hparams, training_hparams, extraction_hparams)

    @property
    def end_step(self):
        iterations_per_epoch = datasets_registry.iterations_per_epoch(self.dataset_hparams)
        return Step.from_str(self.training_hparams.training_steps, iterations_per_epoch)

    @property
    def train_outputs(self):
        return datasets_registry.num_classes(self.dataset_hparams)

    def run_path(self, global_seed, experiment='main'):
        return os.path.join(get_platform().root, self.hashname, f"seed_{global_seed}", experiment)

    def extraction_path(self, global_seed, experiment='main'):
        return os.path.join(self.run_path(global_seed, experiment), self.extraction_hashname)

    @property
    def hashname(self) -> str:
        """The name under which experiments with these hyperparameters will be stored.
        Overwrites the property defined in the base class. Ensures that the hasname is not including the
        extraction_hparams."""
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        fields_dict.pop('extraction_hparams')
        # Remove further_training from training_hparams when computing hash (so experiment can be taken over again)
        # TODO: make this operation safer
        fields_dict_copy = copy.deepcopy(fields_dict)
        fields_dict_copy['training_hparams'].further_training = None
        hparams_strs = [str(fields_dict_copy[k]) for k in sorted(fields_dict_copy)
                        if isinstance(fields_dict_copy[k], Hparams)]
        hash_str = hashlib.blake2b(';'.join(hparams_strs).encode('utf-8'), digest_size=5).hexdigest()
        return f'{self.name_prefix()}_{hash_str}'

    @property
    def extraction_hashname(self) -> str:
        """The name under which each extraction subrun with extraction hyperparameters will be stored."""
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        fields_dict_copy = copy.deepcopy(fields_dict)
        fields_dict_copy['extraction_hparams'].conv_level = None
        extraction_hparams_strs = [str(fields_dict_copy[k]) for k in sorted(fields_dict_copy)
                                   if isinstance(fields_dict_copy[k], ExtractionHparams)]
        extraction_hash_str = hashlib.blake2b(';'.join(extraction_hparams_strs).encode('utf-8'),
                                              digest_size=5).hexdigest()
        return f'clustering_{extraction_hash_str}'



    @property
    def display(self):
        return '\n'.join([self.dataset_hparams.display, self.model_hparams.display, self.training_hparams.display,
                          self.extraction_hparams.display])

    def save(self, output_location):
        """Overriding the save method to save the extraction_hparams separately"""

        if not get_platform().is_primary_process: return
        if not get_platform().exists(output_location): get_platform().makedirs(output_location)

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        extraction_fields_dict = {"extraction_hparams": fields_dict.pop('extraction_hparams')}
        hparams_strs = [fields_dict[k].display for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
        with get_platform().open(paths.hparams(output_location), 'w') as fp:
            fp.write('\n'.join(hparams_strs))
        torch.save(fields_dict, os.path.join(output_location, "hparams_dict"))  # saves dict for ease of reloading

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        extraction_hparams_strs = [fields_dict[k].display for k in sorted(fields_dict)
                                   if isinstance(fields_dict[k], ExtractionHparams)]
        extraction_path = os.path.join(output_location, self.extraction_hashname)
        if not get_platform().exists(extraction_path): get_platform().makedirs(extraction_path)

        with get_platform().open(paths.extraction_hparams(extraction_path), 'w') as fp:
            fp.write('\n'.join(extraction_hparams_strs))
        torch.save(extraction_fields_dict, os.path.join(extraction_path, "extraction_hparams_dict"))

