"""
 # Created on 09.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Registry of platforms (sets up all directories of current machine and coderun).
 #
"""

from platforms import local, controls


registered_platforms = {'local': local.Platform, 'control': controls.Platform}


def get(name):
    return registered_platforms[name]
