"""
 # Created on 08.11.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Description: Pruning strategy: Sparse pruning conducted as in Frankle & Carbin (2019). The output layer is pruned
 # at half the rate of the rest of the network.
 #
 # TODO: Adapt to convolutional layers.
"""

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()
        output_layer = trained_model.output_layer_names[0]

        # Determine the number of weights that need to be pruned for the output layer.
        number_of_remaining_weights_output_layer = np.sum(current_mask[output_layer])
        number_of_weights_to_prune_output_layer = np.ceil(
            pruning_hparams.pruning_fraction/2 * number_of_remaining_weights_output_layer).astype(int)

        # Determine the number of weights that need to be pruned for the rest of the network.
        number_of_remaining_weights = np.sum([np.sum(v) for k, v in current_mask.items() if k != output_layer])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if (k in prunable_tensors) and (k != output_layer)}

        weights_output = {k: v.clone().cpu().detach().numpy()
                          for k, v in trained_model.state_dict().items()
                          if (k in prunable_tensors) and (k == output_layer)}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        # Create a vector of all the unpruned output weights in the model.
        weight_vector_output = np.concatenate([v[current_mask[k] == 1] for k, v in weights_output.items()])
        threshold_output = np.sort(np.abs(weight_vector_output))[number_of_weights_to_prune_output_layer]

        mask_dict = {k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()}
        mask_dict[output_layer] = np.where(np.abs(weights_output[output_layer]) > threshold_output,
                                           current_mask[output_layer], np.zeros_like(weights_output[output_layer]))
        new_mask = Mask(mask_dict)

        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
