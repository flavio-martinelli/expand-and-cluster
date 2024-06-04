"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Registering all different commands-experiments
 #
"""

from foundations.runner import Runner
from training.runner import TrainingRunner
from extraction.runner import ExtractionRunner
from lottery.runner import LotteryRunner

registered_runners = {'train': TrainingRunner, 'extract': ExtractionRunner, 'lottery': LotteryRunner}


def get(runner_name: str) -> Runner:
    if runner_name not in registered_runners:
        raise ValueError('No such runner: {}'.format(runner_name))
    else:
        return registered_runners[runner_name]
