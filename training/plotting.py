"""
 # Created on 08.11.2023
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: plotting functions for training runs
 #
"""
import os
import torch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from training.metric_logger import MetricLogger


def plot_metrics(folder_path, metric_name='test_accuracy', vlims=None, logscale=False, plot_path=None):
    logger = MetricLogger.create_from_file(folder_path)
    if plot_path is None:
        plot_path = folder_path
    metrics = logger.get_data(metric_name)
    metrics = list(zip(*metrics))
    iterations, metrics = np.array(metrics[0]), metrics[1]
    if type(metrics[0]) == torch.Tensor:
        metrics = [t.numpy() for t in metrics]  # convert to numpy (compatibility with older versions)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.plot(iterations, metrics)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(metric_name)
    if logscale: ax.set_yscale('log')
    if vlims is not None: ax.set_ylim(vlims)
    fig.savefig(os.path.join(plot_path, f'{metric_name}.pdf'))
    plt.close(fig)
    return ax, fig
