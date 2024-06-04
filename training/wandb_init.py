"""
 # Created on 26.02.2024
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: 
 #
"""
import os

import numpy as np
import wandb

from foundations.step import Step
from platforms.platform import get_platform


def wandb_init(hyperparameters: list, run_type: str, summary_infos: dict = None):
    """
    Initialize wandb for logging
    :param hyperparameters: list of hyperparameters to log
    :param run_type: type of run subcommand (train, extraction, lottery)
    :return:
    """

    # Try to log in with wandb, if it returns an error return False
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not log in to wandb: {e}\nEC.py proceeds with local logging")
        return False

    config_dict = {}
    for hyp in hyperparameters:
        hyp_dict = hyp.__dict__
        hyp_dict = {k: v for k, v in hyp_dict.items() if not k.startswith('_')}
        config_dict.update(hyp_dict)

    wandb.init(project="expand-and-cluster", group=run_type, config=config_dict, dir=get_platform().root,
               resume="allow",  # to relaunch an experiment one would have to insert the ID
               reinit=True,  # allows recall of .init in case we are resuming a run
               mode="online",  # change to "offline" in case you do not want syncing
               )

    if run_type == "ec":
        wandb.define_metric("training/iteration_step")
        wandb.define_metric("reconstruction/iteration_step")
        wandb.define_metric("training/*", step_metric="training/iteration_step")
        wandb.define_metric("reconstruction/*", step_metric="reconstruction/iteration_step")
    else:
        wandb.define_metric("training/iteration_step")
        wandb.define_metric("training/*", step_metric="training/iteration_step")

    # adding summary infos to the project
    if summary_infos is not None:
        for key, value in summary_infos.items():
            wandb.run.summary[key] = value
    return True


def log_metric_wandb(metric_name: str, value):
    """
    Log a metric to wandb
    :param metric_name: name of the metric
    :param value: value of the metric
    :param step: current step
    """
    if wandb.run is None:
        return

    if np.isscalar(value):
        wandb.log({metric_name: value}, commit=False)
    else:  # the value is a list and each element is logged independently
        for i, v in enumerate(value):
            wandb.log({f"{metric_name}/#{i}": v}, commit=False)


def log_figure_wandb(fig, name):
    """
    Log a figure to wandb
    :param fig: matplotlib figure object to log
    :param name: name of the figure
    """
    if wandb.run is None:
        return
    wandb.log({name: fig}, commit=True)


def log_image_wandb(fig, name):
    """
    Log an image to wandb -- useful for standard images but also for complicated figures that are hard to load on the
    dashboard
    :param fig: image or matplotlib figure object
    :param name: name of the figure
    """
    if wandb.run is None:
        return
    wandb.log({name: wandb.Image(fig, file_type='jpg')}, commit=True)


def log_histogram_wandb(sequence, name, xaxis_name):
    """
    Log a histogram to wandb
    :param sequence: list or 1D np.array of values to log
    :param name: name of the histogram
    """
    if wandb.run is None:
        return
    table = wandb.Table(data=[[el] for el in sequence], columns=[xaxis_name])
    wandb.log({name: wandb.plot.histogram(table, xaxis_name, title=name)}, commit=True)


def sync_wandb():
    """
    Sync wandb to server (without logging any data)
    """
    if wandb.run is None:
        return
    wandb.log({}, commit=True)


def get_wandb_prefix(output_location: str):
    str = os.path.split(output_location)[-1].split('_')[0]
    if str == "main":
        return "training"
    elif str == "reconstructed":
        return "reconstruction"