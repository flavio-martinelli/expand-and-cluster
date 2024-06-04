"""
 # Created on 09.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: main script for experiment selection
 #
"""

import os
import torch
import random
import numpy as np

from foundations.step import Step


def set_seeds(seed):
    """
    Set the same seed for pytorch, numpy and python. If seed==-1 then sets no seed
    :param seed: the seed
    """
    if seed == -1:  # seed is not set
        return

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def find_final_model_step(location, iterations_per_epoch):
    # Look for final model in location
    for filename in os.listdir(location):
        # filter out non-models and initial model (model_ep0_it0.pth)
        if filename.split('_')[0] == 'model' and filename != 'model_ep0_it0.pth':
            ep = int(filename.split('_')[1][2:])
            it = int(filename.split('_')[2].split('.')[0][2:])
            return Step.from_epoch(ep, it, iterations_per_epoch)
    return None


