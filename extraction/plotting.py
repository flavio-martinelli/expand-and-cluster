"""
 # Created on 29.10.2023
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: Plotting functions for extraction runs
 #
"""

import os
import numpy as np
import torch

from matplotlib import pyplot as plt
from training.metric_logger import MetricLogger


def plot_all_features(w, reshape=True, vlims=None, tridim=False):
    """

    :param w: shape [pixels, features, networks] if reshape True
    :param reshape:
    :return:
    """
    vlims = vlims or [w.min(), w.max()]
    plot_no = w.shape[1]*w.shape[2]
    grid_size = np.ceil(np.sqrt(plot_no)).astype(int)
    w = w.reshape(w.shape[0], -1).T
    if reshape:
        if tridim:
            fig_reshape = np.sqrt(w.shape[1]/3).astype(int)
            w = w.reshape(-1, fig_reshape, fig_reshape, 3)
            w = (w - vlims[0]) / (vlims[1] - vlims[0])
        else:
            fig_reshape = np.sqrt(w.shape[1]).astype(int)
            w = w.reshape(-1, fig_reshape, fig_reshape)
    else:
        w = np.expand_dims(w, axis=-1)
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10), dpi=300)
    ax = ax.flatten()
    for i in range(w.shape[0]):
        ax[i].imshow(w[i], vmin=vlims[0], vmax=vlims[1])
    for i in range(len(ax)):
        ax[i].axis('off')
    fig.tight_layout()
    return fig, ax