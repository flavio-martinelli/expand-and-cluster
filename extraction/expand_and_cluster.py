"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Description: Expand-and-Cluster experiment script
 #
"""
import copy
import warnings

import torch
import os.path

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

import datasets.registry
import models.registry
import matplotlib.pyplot as plt

from datasets.teacher_dataset import Dataset
from foundations import hparams
from foundations import paths
from foundations.step import Step
from models.activation_functions import get_symmetry
from models.base import Model, DataParallel
from platforms.platform import get_platform
from training import optimizers
from training.checkpointing import restore_checkpoint
from training.metric_logger import MetricLogger
from training.plotting import plot_metrics
from training.standard_callbacks import ec_callbacks, ec_linear_callbacks
from training.train import train
from extraction.layer_reconstruction import reconstruct_layer, compare_with_teacher, compare_with_teacher_conv
from datasets.registry import registered_datasets

from extraction.plotting import plot_all_features
from models import initializers, students_mnist_lenet_linear
from training.wandb_init import log_figure_wandb, log_histogram_wandb, log_metric_wandb, sync_wandb
from utils.utils import find_final_model_step

try:
    import apex
    NO_APEX = False
except ImportError:
    NO_APEX = True

sep = '='*140
sep2 = '-'*140

def train_students(
    model: Model,
    output_location: str,
    dataset_hparams: hparams.DatasetHparams,
    training_hparams: hparams.TrainingHparams,
    start_step: Step = None,
    verbose: bool = True,
    evaluate_every_epoch: bool = True):

    """Train using the students through the standard_train procedure."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)

    train_loader = datasets.registry.get(dataset_hparams, train=True)

    if (models.registry.exists(output_location, train_end_step) and
    get_platform().exists(paths.logger(output_location))) and training_hparams.further_training is None:
        state_dict = get_platform().load_model(paths.model(output_location, train_end_step),
                                               map_location=get_platform().torch_device)
        model.load_state_dict(state_dict)
        losses = 0
        for examples, labels in train_loader:
            losses += model.individual_losses(model(examples), labels) / len(labels)
        return model, losses.detach().cpu().numpy()

    callbacks = ec_callbacks(training_hparams, train_loader, verbose=verbose, start_step=start_step,
                             evaluate_every_100_epochs=evaluate_every_epoch,
                             evaluate_every_10=hasattr(model, 'ConvNet'))  # If ConvNet, evaluate every 10 epochs

    train(training_hparams, model, train_loader, output_location, callbacks, start_step=start_step)
    model.cpu()

    losses = 0
    for examples, labels in train_loader:
        losses += model.individual_losses(model(examples), labels)/len(labels)

    return model, losses.detach().cpu().numpy()


def reconstruct(model: Model,
                losses: torch.Tensor,
                output_location: str,
                extraction_hparams: hparams.ExtractionHparams,
                dataset_hparams: hparams.DatasetHparams,
                training_hparams: hparams.TrainingHparams,
                model_hparams: hparams.ModelHparams,
                verbose: bool = True,
                layer: int = None):
    """Expand-and-Cluster procedure, for the moment is only for fully connected"""
    plots_folder = os.path.join(output_location, "ECplots")
    os.makedirs(plots_folder, exist_ok=True)
    finetune_checkpoints_folder = os.path.join(output_location, "finetune_checkpoints")
    os.makedirs(finetune_checkpoints_folder, exist_ok=True)
    reconstructed_folder = os.path.join(output_location, "reconstructed_model")
    os.makedirs(reconstructed_folder, exist_ok=True)
    alignment_reconstruction =  os.path.join(reconstructed_folder, "after_reconstruction")
    os.makedirs(alignment_reconstruction, exist_ok=True)
    alignment_tuning = os.path.join(reconstructed_folder, "after_tuning")
    os.makedirs(alignment_tuning, exist_ok=True)
    beta = np.pi/extraction_hparams.beta
    train_loader = datasets.registry.get(dataset_hparams, train=True)

    teacher_folder = os.path.join(get_platform().root, "train_" + dataset_hparams.teacher_name,
                                  "seed_" + dataset_hparams.teacher_seed, "main")
    teacher = Dataset.get_specified_model(teacher_folder)

    # Iterate through the different layers of model
    parameter_list = copy.deepcopy(list(model.parameters()))
    N = parameter_list[0].shape[-1]

    layer_no = int(len(parameter_list)/2)
    final_layer = False

    students = models.registry.get(model_hparams)
    symmetry = get_symmetry(students.act_fun)

    # losses = 0
    # for examples, labels in train_loader:
    #     losses += students.individual_losses(students(examples), labels) / len(labels)

    # if symmetry == "even_linear_positive_scaling" or symmetry == "even_linear":
    #     students = students_mnist_lenet_linear.Model.load_from_student_mnist_lenet(students)

    # load boruta mask (assumes we know the dataset the teacher was trained on)
    cluster_mask = None
    if extraction_hparams.boruta is not None:
        dataset_class = registered_datasets[extraction_hparams.boruta].Dataset
        cluster_mask = dataset_class.get_boruta_mask()
        cluster_mask = np.concatenate([cluster_mask, [1.0]])

    for l in range(layer_no):  # TODO: avoid re-clustering when it is already saved
        print(sep2 + '\n' + f'Layer {l}' + '\n' + sep2 + '\n')
        if l+1 == layer_no-1:  # if last hidden layer, set final_layer to True
            final_layer = True

        i = l*2  # parameter_list index
        w, b, a = parameter_list[i].data, parameter_list[i+1].data,  parameter_list[i+2].data
        w, b, a = [copy.deepcopy(x.cpu().numpy()) for x in [w, b, a]]

        if hasattr(students, 'ConvNet'):
            # compact 3D kernels into 1D vectors
            # w = w.reshape([w.shape[0], -1, w.shape[-1]]).transpose(1, 0, 2)
            # a = a.reshape([a.shape[0], -1, a.shape[-1]]).transpose(1, 0, 2)
            w = w.transpose(0, 4, 1, 2, 3)  # DIMS: [out_channels, in_channels, kernel_size, kernel_size, #networks]
            w = w.reshape([w.shape[0], w.shape[1], -1]).transpose(2, 0, 1)
            a = None

        # concatenate the biases to the weights
        w_cat = np.concatenate([w, b[np.newaxis, :, :]], axis=0)

        # move weight norms to a if symmetry is even_linear_positive_scaling
        if symmetry == 'even_linear_positive_scaling':
            w_norms = np.linalg.norm(w_cat[:, :, :], axis=0)
            w_cat /= w_norms
            a = np.einsum("hon,hn->hon", a, w_norms)

        # compute cosine similarity between weight vectors of w_cat
        # n = 3  # network index
        # sim = np.zeros((w_cat.shape[1], w_cat.shape[1]))
        # for j in range(w_cat.shape[1]):
        #     for i in range(w_cat.shape[1]):
        #         sim[j, i] = np.dot(w_cat[:-1, j, n], w_cat[:-1, i, n]) / \
        #                     (np.linalg.norm(w_cat[:-1, j, n]) * np.linalg.norm(w_cat[:-1, i, n]))
        # fig, ax = plt.subplots(); plt.imshow(sim, cmap='bwr'); plt.colorbar(); fig.show()

        w_rec, a_rec = reconstruct_layer(w_cat, N, extraction_hparams.gamma, beta, losses, A=a,
                                         symmetry=symmetry, verbose=verbose, cluster_mask=cluster_mask,
                                         plots_folder=plots_folder, exp_name=f"L{l+1}", final_layer=final_layer)

        # save w_rec in the extraction folder
        if hasattr(students, 'ConvNet'):
            if l == 1:  # Manually fix second layer permutation of conv (something scrambled in the process)
                # perm = [0, 1, 2, 9, 11, 12, 13, 14, 15, 3, 4, 5, 6, 10, 8, 7]
                perm = [10, 11, 8, 7, 6, 12, 0, 2, 13, 4, 5, 14, 9, 15, 1, 3]
                perm = np.argsort(perm)
                w_perm = w_rec[:-1,:].T.reshape(16, 16, 3, 3)
                w_perm = w_perm[:, perm, :, :].reshape(16, -1).T
                b_perm = w_rec[-1]
                b_perm = b_perm[perm]
                w_rec = np.concatenate([w_perm, b_perm[np.newaxis, :]])
                best_permutation = None

            np.save(os.path.join(output_location, f"w_rec_L{l+1}"), w_rec)
            best_permutation = teacher_comparison_conv(w_rec, teacher, l, symmetry, plots_folder, verbose=verbose,
                                                       permutation= best_permutation if l > 0 else None)
            np.save(os.path.join(output_location, f"best_permutation_L{l + 1}"), best_permutation)
            if l == layer:
                return
            else:
                continue

        # put the new clustered layer across the N networks
        w_rec = np.stack([w_rec]*N, axis=2)
        w_rec, b_rec = w_rec[:-1, :, :], w_rec[-1, :, :]

        students = fill_current_layer(students, i, w_rec, b_rec, a_rec, symmetry)

        if l+1 == layer_no-1:  # if last hidden layer, break
            break

        # Retrain the upper layers of the network
        start_step = Step.zero(train_loader.iterations_per_epoch)
        finetune_hparams = copy.deepcopy(training_hparams)
        finetune_hparams.training_steps = extraction_hparams.finetune_traning_steps
        finetune_hparams.lr = extraction_hparams.finetune_lr
        callbacks = ec_callbacks(finetune_hparams, train_loader, verbose=verbose, start_step=start_step,
                                 evaluate_every_100_epochs=True)
        os.makedirs(os.path.join(finetune_checkpoints_folder, f"L{l + 1}"), exist_ok=True)
        train(finetune_hparams, students, train_loader,
              os.path.join(finetune_checkpoints_folder, f"L{l + 1}"),
              callbacks, start_step=start_step)

        plot_metrics(folder_path=os.path.join(finetune_checkpoints_folder, f"L{l + 1}"),
                     metric_name='train_individual_losses', logscale=True)

        if l==0: cluster_mask = None  # remove cluster_mask after first layer

        # TODO: add filtering of neurons with non-aligned biases (how do we deal with the last layer? there's no OP
        #  after that)

    # FINAL FINE-TUNING
    # condense in one network and make all parameters trainable
    for i, param in enumerate(students.parameters()):
        if i % 2 == 0 and i != 2*(layer_no-1):  # weight params except the output layer
            param.data = torch.mean(param.data, dim=2).unsqueeze(2)  # this average is not doing anything (note the above np.stack)
        elif i == 2*(layer_no-1):  # output layer
            param.data = param.data.unsqueeze(2)
        elif i % 2 == 1:  # bias params
            param.data = torch.mean(param.data, dim=1).unsqueeze(1)
        param.requires_grad = True

    students.N = 1
    students.to(get_platform().torch_device)

    if symmetry == 'even_linear_positive_scaling' or symmetry == 'even_linear':
        frozen_students = copy.deepcopy(students)
        sample_no = train_loader.dataset._labels.shape[0]
        with torch.no_grad():
            y_frozen = []
            list_examples = []
            list_labels = []
            for examples, labels in train_loader:
                examples = examples.to(device=get_platform().torch_device)
                labels = labels.to(device=get_platform().torch_device)
                y_frozen.append(frozen_students(examples))
                list_labels.append(labels)
                list_examples.append(examples)
            y_frozen = torch.cat(y_frozen, dim=0).squeeze()
            y = (torch.vstack(list_labels) - y_frozen).cpu().numpy()
            examples = torch.vstack(list_examples).cpu().numpy()
        x = np.concatenate([examples.reshape(sample_no, -1), np.ones([sample_no, 1])], axis=1)
        thetas = np.linalg.lstsq(x, y, rcond=None)[0]
        print(f"MSE of linear component after reconstruction {(((x @ thetas) - y)**2).mean()}")

    teacher_comparison(teacher, students, symmetry, cluster_mask, alignment_reconstruction)

    start_step = Step.zero(train_loader.iterations_per_epoch)
    finetune_hparams = copy.deepcopy(training_hparams)
    finetune_hparams.training_steps = extraction_hparams.finetune_traning_steps
    finetune_hparams.lr = extraction_hparams.finetune_lr

    if symmetry == 'even_linear_positive_scaling' or symmetry == 'even_linear':
        callbacks = ec_linear_callbacks(finetune_hparams, train_loader, verbose=verbose, start_step=start_step,
                                        evaluate_every_100_epochs=True)
        parallel_train(finetune_hparams, students, train_loader, reconstructed_folder, callbacks,
                       thetas, start_step=start_step)
        # TODO: Implement proper linear parallel model and avoid specifying a different train function.
        # TODO: what about the align_final_bias?
    else:
        callbacks = ec_callbacks(finetune_hparams, train_loader, verbose=verbose, start_step=start_step,
                                 evaluate_every_100_epochs=True)
        align_final_bias(students, train_loader, verbose=verbose)
        train(finetune_hparams, students, train_loader, reconstructed_folder, callbacks, start_step=start_step)

    teacher_comparison(teacher, students, symmetry, cluster_mask, alignment_tuning)
    plot_metrics(folder_path=reconstructed_folder,
                 metric_name='train_individual_losses', logscale=True)


def align_final_bias(model, loader, verbose=False):
    """
    Aligns the final bias of the model to the dataset bias.
    :param model:
    :param trainloader:
    :return:
    """
    example_count = torch.tensor(0.0).to(get_platform().torch_device)
    total_loss = torch.tensor(0.0).to(get_platform().torch_device)
    total_output = 0.0
    dataset_bias = 0.0

    model.eval()
    with torch.no_grad():
        for examples, labels in loader:
            examples = examples.to(get_platform().torch_device)
            labels = labels.squeeze().to(get_platform().torch_device)
            output = model(examples)

            labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
            example_count += labels_size
            total_loss += model.loss_criterion(output, labels) * labels_size
            total_output += output.sum(dim=0)
            dataset_bias += labels.sum(dim=0)

    example_count = example_count.cpu().item()
    total_loss = total_loss.cpu().item() / example_count
    total_output /= example_count
    dataset_bias /= example_count

    if verbose: print(f"Loss before bias alignment: {total_loss:.3e}")

    list(model.parameters())[-1].data += -total_output + dataset_bias[:, None]

    total_loss = torch.tensor(0.0).to(get_platform().torch_device)
    example_count = torch.tensor(0.0).to(get_platform().torch_device)
    with torch.no_grad():
        for examples, labels in loader:
            examples = examples.to(get_platform().torch_device)
            labels = labels.squeeze().to(get_platform().torch_device)
            output = model(examples)

            labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
            example_count += labels_size
            total_loss += model.loss_criterion(output, labels) * labels_size

    total_loss = total_loss.cpu().item() / example_count
    if verbose: print(f"Loss after bias alignment: {total_loss:.3e}")


def fill_current_layer(students, i, w_rec, b_rec, a_rec, symmetry):
    param_list = list(students.parameters())
    param_list[i].data = torch.Tensor(w_rec)
    param_list[i+1].data = torch.Tensor(b_rec)
    param_list[i+2].data = torch.Tensor(a_rec)
    return students


def teacher_comparison(teacher, students, symmetry, cluster_mask, plots_folder):
    teacher_params = copy.deepcopy(list(teacher.parameters()))
    students_params = copy.deepcopy(list(students.parameters()))

    wt, bt, at = [teacher_params[i].data.cpu().numpy().squeeze().T for i in range(3)]
    ws, bs, as_ = [students_params[i].data.cpu().numpy().squeeze() for i in range(3)]

    out = compare_with_teacher(wt, bt, at, ws, bs, as_, symmetry, cluster_mask=cluster_mask, verbose=True)
    fig, best_sims_w, best_sims_a, (student_size, teacher_size) = out
    log_histogram_wandb(np.log10(best_sims_w)+1e-12, "clustering/student_alignment_W", "log cosine distance")
    log_histogram_wandb(np.log10(best_sims_a)+1e-16, "clustering/student_alignment_A", "log cosine distance")
    log_metric_wandb("clustering/student_size", student_size)
    log_metric_wandb("clustering/teacher_size", teacher_size)
    sync_wandb()

    fig.savefig(os.path.join(plots_folder, f"student_alignments.pdf"))
    out = compare_with_teacher(wt, bt, at, ws, bs, as_, symmetry, cluster_mask=cluster_mask, log=False)
    fig, best_sims_w, best_sims_a, (student_size, teacher_size) = out

    fig.savefig(os.path.join(plots_folder, f"student_alignments_nolog.pdf"))
    plt.close(fig)


def teacher_comparison_conv(w_rec, teacher, l, symmetry, plots_folder, permutation=None, verbose=False):
    wt = copy.deepcopy(list(teacher.parameters())[l*2].data.cpu().numpy().squeeze())
    inverse_permutation = np.argsort(permutation) if permutation is not None else None
    if inverse_permutation is not None:
        wt = wt[:, inverse_permutation, :, :]  # Align input channels to previous best permutation
    wt = wt.reshape(wt.shape[0], -1).T

    bt = copy.deepcopy(list(teacher.parameters())[l*2+1].data.cpu().numpy().squeeze())
    ws = copy.deepcopy(w_rec)
    ws, bs = ws[:-1], ws[-1]

    out = compare_with_teacher_conv(wt, bt, ws, bs, symmetry, verbose=True)
    fig, best_sims_w, best_permutation, (student_size, teacher_size) = out
    log_histogram_wandb(np.log10(best_sims_w)+1e-12, "clustering/student_alignment_W", "log cosine distance")
    log_metric_wandb("clustering/student_size", student_size)
    log_metric_wandb("clustering/teacher_size", teacher_size)
    sync_wandb()
    fig.savefig(os.path.join(plots_folder, f"student_alignments_nolog_L{l+1}.pdf"))
    plt.close(fig)

    out = compare_with_teacher_conv(wt, bt, ws, bs, symmetry, verbose=False, log=True)
    fig, _, _, (_, _) = out
    fig.savefig(os.path.join(plots_folder, f"student_alignments_L{l+1}.pdf"))
    plt.close(fig)

    return best_permutation


def parallel_train(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader,
    output_location: str,
    callbacks,
    thetas,
    start_step: Step = None,
    end_step: Step = None,
):

    """The main training loop modified to allow for parallel linear path.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. They provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """

    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model.to(get_platform().torch_device)
    thetas = torch.nn.Parameter(torch.Tensor(thetas).to(get_platform().torch_device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [thetas], lr=training_hparams.lr*10)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model, step_optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    # if get_platform().is_distributed:
    #     model = DistributedDataParallel(model, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model = DataParallel(model)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()

    if not isinstance(lr_schedule, ReduceLROnPlateau):
        with warnings.catch_warnings():  # Filter unnecessary warning.
            # warnings.filterwarnings("ignore", category=UserWarning)
            for _ in range(start_step.iteration): lr_schedule.step()

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return

    # if training was stopped prematurely (lr too low) then skip training.
    maybe_train_end_step = find_final_model_step(output_location, train_loader.iterations_per_epoch)
    if maybe_train_end_step is not None:  # load saved model before returning
        state_dict = get_platform().load_model(paths.model(output_location, maybe_train_end_step),
                                               map_location=get_platform().torch_device)
        model = model.load_state_dict(state_dict)
        return

    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))
        loss_ep = 0

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, [model, thetas], optimizer, logger)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            x = torch.cat([examples.reshape(examples.shape[0], -1),
                           torch.ones(examples.shape[0], 1).to(get_platform().torch_device)], dim=1)

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples) + (x@thetas).unsqueeze(-1), labels)
            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            loss_ep += loss.item()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            if not isinstance(lr_schedule, ReduceLROnPlateau):
                with warnings.catch_warnings():  # Filter unnecessary warning.
                    warnings.filterwarnings("ignore", category=UserWarning)
                    lr_schedule.step()

        loss_ep /= train_loader.iterations_per_epoch
        if isinstance(lr_schedule, ReduceLROnPlateau):
            if step_optimizer.param_groups[0]["lr"] < lr_schedule.min_lrs[0]+lr_schedule.eps:  # End if lr is minimal
                current_step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
                model.save(output_location, current_step)
                logger.save(output_location)
                break
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step(loss_ep)

    get_platform().barrier()
