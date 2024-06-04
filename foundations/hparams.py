"""
 # Created on 09.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Hyperparameters class. Add new hyperparameters in this file.
 #
"""

import abc
import argparse
import copy
from dataclasses import dataclass, fields, MISSING
from typing import Tuple


@dataclass
class Hparams(abc.ABC):
    """A collection of hyperparameters.

    Add desired hyperparameters with their types as fields. Provide default values where desired.
    You must provide default values for _name and _description. Help text for field `f` is
    optionally provided in the field `_f`,
    """

    def __post_init__(self):
        if not hasattr(self, '_name'): raise ValueError('Must have field _name with string value.')
        if not hasattr(self, '_description'): raise ValueError('Must have field _name with string value.')

    @classmethod
    def add_args(cls, parser, defaults: 'Hparams' = None, prefix: str = None,
                 name: str = None, description: str = None, create_group: bool = True):
        if defaults and not isinstance(defaults, cls):
            raise ValueError(f'defaults must also be type {cls}.')

        if create_group:
            parser = parser.add_argument_group(name or cls._name, description or cls._description)

        for field in fields(cls):
            if field.name.startswith('_'): continue
            arg_name = f'--{field.name}' if prefix is None else f'--{prefix}_{field.name}'
            helptext = getattr(cls, f'_{field.name}') if hasattr(cls, f'_{field.name}') else ''

            if defaults: default = copy.deepcopy(getattr(defaults, field.name, None))
            elif field.default != MISSING: default = copy.deepcopy(field.default)
            else: default = None

            if field.type == bool:
                # if (defaults and getattr(defaults, field.name) is not False) or field.default is not False:
                #     raise ValueError(f'Boolean hyperparameters must default to False: {field.name}.')
                parser.add_argument(arg_name, action='store_true', default=default, help='(optional) ' + helptext)

            elif field.type in [str, float, int]:
                required = field.default is MISSING and (not defaults or not getattr(defaults, field.name))
                if required:  helptext = '(required: %(type)s) ' + helptext
                elif default: helptext = f'(default: {default}) ' + helptext
                else:         helptext = '(optional: %(type)s) ' + helptext
                parser.add_argument(arg_name, type=field.type, default=default, required=required, help=helptext)

            # If it is a nested hparams, use the field name as the prefix and add all arguments.
            elif isinstance(field.type, type) and issubclass(field.type, Hparams):
                subprefix = f'{prefix}_{field.name}' if prefix else field.name
                field.type.add_args(parser, defaults=default, prefix=subprefix, create_group=False)

            else: raise ValueError(f'Invalid field type {field.type} for hparams.')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> 'Hparams':
        d = {}
        for field in fields(cls):
            if field.name.startswith('_'): continue

            # Base types.
            if field.type in [bool, str, float, int]:
                arg_name = f'{field.name}' if prefix is None else f'{prefix}_{field.name}'
                if not hasattr(args, arg_name): raise ValueError(f'Missing argument: {arg_name}.')
                d[field.name] = getattr(args, arg_name)

            # Nested hparams.
            elif isinstance(field.type, type) and issubclass(field.type, Hparams):
                subprefix = f'{prefix}_{field.name}' if prefix else field.name
                d[field.name] = field.type.create_from_args(args, subprefix)

            else: raise ValueError(f'Invalid field type {field.type} for hparams.')

        return cls(**d)

    @property
    def display(self):
        nondefault_fields = [f for f in fields(self)
                             if not f.name.startswith('_') and ((f.default is MISSING) or getattr(self, f.name))]
        s = self._name + '\n'
        return s + '\n'.join(f'    * {f.name} => {getattr(self, f.name)}' for f in nondefault_fields)

    def __str__(self):
        fs = {}
        for f in fields(self):
            if f.name.startswith('_'): continue
            if f.default is MISSING or (getattr(self, f.name) != f.default):
                value = getattr(self, f.name)
                if isinstance(value, str): value = "'" + value + "'"
                if isinstance(value, Hparams): value = str(value)
                if isinstance(value, Tuple): value = 'Tuple(' + ','.join(str(h) for h in value) + ')'
                fs[f.name] = value
        elements = [f'{name}={fs[name]}' for name in sorted(fs.keys())]
        return 'Hparams(' + ', '.join(elements) + ')'


@dataclass
class DatasetHparams(Hparams):
    dataset_name: str
    batch_size: int
    do_not_augment: bool = False
    transformation_seed: int = None
    subsample_fraction: float = None
    random_labels_fraction: float = None
    unsupervised_labels: str = None
    blur_factor: int = None
    #### ONLY FOR TEACHER DATASET  # TODO: consider making it into a subgroup of args
    teacher_name: str = None
    teacher_seed: str = None
    samples: int = None
    datagen: str = None
    d_in: int = None

    _name: str = 'Dataset Hyperparameters'
    _description: str = 'Hyperparameters that select the dataset, data augmentation, and other data transformations.'
    _dataset_name: str = 'The name of the dataset. Examples: mnist, cifar10. For a teacher generated dataset select ' \
                         '"teacher_dataset" and specify in the argument "teacher_name" the name of the folder ' \
                         'containing that teacher experiment OR the .npy file containing the dataset.'
    _batch_size: str = 'The size of the mini-batches on which to train. Example: 64'
    _do_not_augment: str = 'If True, data augmentation is disabled. It is enabled by default.'
    _transformation_seed: str = 'The random seed that controls dataset transformations like ' \
                                'random labels, subsampling, and unsupervised labels.'
    _subsample_fraction: str = 'Subsample the training set, retaining the specified fraction: float in (0, 1]'
    _random_labels_fraction: str = 'Apply random labels to a fraction of the training set: float in (0, 1]'
    _unsupervised_labels: str = 'Replace the standard labels with alternative, unsupervised labels. Example: rotation'
    _blur_factor: str = 'Blur the training set by downsampling and then upsampling by this multiple.'
    #### ONLY FOR TEACHER DATASET  # TODO: consider making it into a subgroup of args
    _teacher_name: str = 'The name of the folder (e.g. 39689376sf982n29 or /path/to/dataset.npy).'
    _teacher_seed: str = 'The seed folder of the teacher (specify only in case of e.g. --teacher_name 39689376sf982n29)'
    _samples: str = 'The amount of samples to extract from the teacher'
    _datagen: str = 'A list of datagen policies separated by a "+" character (e.g. --datagen mnist+random+grid)'
    _d_in: str = 'The input dimensionality of the teacher'

@dataclass
class ModelHparams(Hparams):
    model_name: str
    model_init: str
    batchnorm_init: str
    batchnorm_frozen: bool = False
    output_frozen: bool = False
    others_frozen: bool = False
    others_frozen_exceptions: str = None
    act_fun: str = "relu"

    _name: str = 'Model Hyperparameters'
    _description: str = 'Hyperparameters that select the model, initialization, and weight freezing.'
    _model_name: str = 'The name of the model. Examples: mnist_lenet, cifar_resnet_20, cifar_vgg_16'
    _model_init: str = 'The model initializer. Examples: kaiming_normal, kaiming_uniform, binary, orthogonal'
    _batchnorm_init: str = 'The batchnorm initializer. Examples: uniform, fixed'
    _batchnorm_frozen: str = 'If True, all batch normalization parameters are frozen at initialization.'
    _output_frozen: str = 'If True, all outputt layer parameters are frozen at initialization.'
    _others_frozen: str = 'If true, all other (non-output, non-batchnorm) parameters are frozen at initialization.'
    _others_frozen_exceptions: str = 'A comma-separated list of any tensors that should not be frozen.'
    _act_fun: str = "Activation function"


@dataclass
class TrainingHparams(Hparams):
    optimizer_name: str
    lr: float
    training_steps: str
    data_order_seed: int = None
    momentum: float = 0.0
    nesterov_momentum: float = 0.0
    milestone_steps: str = None
    delta: float = None
    plateau_opt: bool = False
    patience: int = None
    cooldown: int = None
    warmup_steps: str = None
    weight_decay: float = None
    apex_fp16: bool = False
    further_training: str = None

    _name: str = 'Training Hyperparameters'
    _description: str = 'Hyperparameters that determine how the model is trained.'
    _optimizer_name: str = 'The opimizer with which to train the network. Examples: sgd, adam'
    _lr: str = 'The learning rate'
    _training_steps: str = 'The number of steps to train as epochs (\'160ep\') or iterations (\'50000it\').'
    _momentum: str = 'The momentum to use with the SGD optimizer.'
    _nesterov: bool = 'The nesterov momentum to use with the SGD optimizer. Cannot set both momentum and nesterov.'
    _milestone_steps: str = 'Steps when the learning rate drops by a factor of delta. Written as comma-separated ' \
                            'steps (80ep,160ep,240ep) where steps are epochs (\'160ep\') or iterations (\'50000it\').'
    _delta: str = 'The factor at which to drop the learning rate at each milestone.'
    _plateau_opt: str = 'If true, selects the ReduceLRonPlateau optimizer, requires: --delta, --patience, --cooldown'
    _patience: str = 'Only necessary with --plateau_opt, sets the patience argument of ' \
                     'torch.optim.lr_scheduler.ReduceLRonPlateau'
    _cooldown: str = 'Only necessary with --plateau_opt, sets the cooldown argument of ' \
                     'torch.optim.lr_scheduler.ReduceLRonPlateau'
    _data_order_seed: str = 'The random seed for the data order. If not set, the data order is random and unrepeatable.'
    _warmup_steps: str = "Steps of linear lr warmup at the start of training as epochs ('20ep') or iterations ('800it')"
    _weight_decay: str = 'The L2 penalty to apply to the weights.'
    _apex_fp16: bool = 'Whether to train the model in float16 using the NVIDIA Apex library.'
    _further_training: str = 'Overwrites the final steps to perform to the same model after it has been trained ' \
                             'already (does not modify the hashname of the simulation). Example: write 30000ep if you '\
                             'want your model to be trained from where it stopped to 30000 epochs.' \


@dataclass
class PruningHparams(Hparams):
    pruning_strategy: str

    _name: str = 'Pruning Hyperparameters'
    _description: str = 'Hyperparameters that determine how the model is pruned. ' \
                        'More hyperparameters will appear once the pruning strategy is selected.'
    _pruning_strategy: str = 'The pruning strategy to use.'


@dataclass
class ExtractionHparams(Hparams):
    gamma: float
    beta: float
    finetune_traning_steps: str = "5000ep"  # TODO: fix typo but after sims have run!
    finetune_lr: float = 1e-4
    boruta: str = None
    conv_level: int = None

    _name: str = 'Extraction Hyperparameters'
    _description: str = 'Hyperparameters describing the network extraction with Expand-and-Cluster.'
    _gamma: str = 'The gamma parameter of Expand-and-Cluster: clusters of size <= gamma*N are filtered out (' \
                  '0<<gamma<=1)'
    _beta: str = 'The beta parameter of Expand-and-Cluster: clusters of vectors whose median angle is > beta are ' \
                 'filtered out. For simplicity, the Beta in the paper is specified here as np.pi/beta (i.e. you need ' \
                 'to provide *only* the value that divides pi).'
    _finetune_traning_steps: str = 'The number of steps to finetune the extracted network after every layer extraction.'
    _finetune_lr: str = 'The learning rate for the finetuning step.'
    _boruta: str = 'The dataset from which to take the boruta mask to use for input weight filtering. e.g. "mnist"'
    _conv_level: str = 'Indicates the level at which the bottom convolutional weights need to be loaded from previous' \
                       ' layers reconstructions '
