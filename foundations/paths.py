"""
 # Created on 27.10.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: useful paths are defined here
 #
"""

import os


def checkpoint(root): return os.path.join(root, 'checkpoint.pth')


def logger(root): return os.path.join(root, 'logger')


def mask(root): return os.path.join(root, 'mask.pth')


def sparsity_report(root): return os.path.join(root, 'sparsity_report.json')


def model(root, step): return os.path.join(root, 'model_ep{}_it{}.pth'.format(step.ep, step.it))


def hparams(root): return os.path.join(root, 'hparams.log')


def extraction_hparams(root): return os.path.join(root, 'extraction_hparams.log')
