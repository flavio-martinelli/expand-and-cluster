"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Instance class of platform ran by the control folder
 #
"""

import os
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        return os.path.join(pathlib.Path().resolve(), 'controls', 'data', 'sims')

    @property
    def dataset_root(self):
        return os.path.join(pathlib.Path().resolve(), 'controls', 'data', 'datasets')

    @property
    def boruta_root(self):
        return os.path.join(pathlib.Path().resolve(), 'controls', 'datasets', 'boruta')

    @property
    def imagenet_root(self):
        raise NotImplementedError
