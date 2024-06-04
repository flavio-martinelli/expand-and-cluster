"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Instance class of local computer run
 #
"""

import os
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        return os.path.join(pathlib.Path().resolve(), 'data', 'sims')

    @property
    def dataset_root(self):
        return os.path.join(pathlib.Path().resolve(), 'data', 'datasets')

    @property
    def boruta_root(self):
        return os.path.join(pathlib.Path().resolve(), 'datasets', 'boruta')

    @property
    def imagenet_root(self):
        raise NotImplementedError
