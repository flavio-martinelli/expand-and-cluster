"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Specifies platform (e.g. local or others)
 #
"""

_PLATFORM = None


def get_platform():
    return _PLATFORM
