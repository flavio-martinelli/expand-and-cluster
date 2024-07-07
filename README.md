# Expand-and-Cluster

### Welcome

This framework allows to experiment on parameter identification of neural networks. It is the codebase of the following work:  

* [_Expand-and-Cluster: Parameter Recovery of Neural Networks._](https://arxiv.org/abs/2304.12794) Flavio Martinelli, Berfin Şimşek, Wulfram Gerstner and Johanni Brea. ArXiv 2023.

Created by [Flavio Martinelli](https://scholar.google.com/citations?user=DabSKBgAAAAJ&hl=it) working in the [Laboratory of Computational Neuroscience, EPFL](https://www.epfl.ch/labs/lcn). 

The skeleton of the codebase is heavily inspired and adapted from [OpenLTH: A Framework for Lottery Tickets and Beyond](https://github.com/facebookresearch/open_lth), developed by [Jonathan Frankle](http://www.jfrankle.com).

### How to Cite

If you use this library in a research paper, please cite the main paper:

> Martinelli, F., Simsek, B., Gerstner, W., & Brea, J. Expand-and-Cluster: Parameter Recovery of Neural Networks. In Forty-first International Conference on Machine Learning.
```
@inproceedings{martinelliexpand,
  title={Expand-and-Cluster: Parameter Recovery of Neural Networks},
  author={Martinelli, Flavio and Simsek, Berfin and Gerstner, Wulfram and Brea, Johanni},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

### License

Licensed under the MIT license, as found in the LICENSE file.

## Table of Contents

### [1 Overview](#overview)
### [2 Getting Started](#getting-started)
### [3 Internals](#internals)
### [4 Extending the Framework](#extending)
### [5 Acknowledgements](#acknowledgements)

## <a name=overview></a>1 Overview

### 1.1 Purpose

This framework is designed with four goals in mind:

1. Train standard neural networks for image classification tasks.
2. Train student networks to imitate a standard network or specific dataset.
3. Run parameter identification experiments on student networks.
4. Run pruning experiments on standard and student networks.
5. Automatically manage experiments and results without any manual intervention.
6. Make it easy to add new datasets, models, and other modifications.

### 1.2 Why the Need of a Framework?

* **Hyperparameter management.** Default hyperparameters are easy to modify, and results are automatically indexed by the hyperparameters (so you never need to worry about naming experiments).
* **Extensibility.** Consistent abstractions for models and datasets make it easy to add new ones in a modular way.
* **Enabling new experiments.** Each hyperparameter automatically surfaces on the command-line and integrates into experiment naming in a backwards-compatible way.
* **Re-training and ablation.** Such experiments can be created by writing a single function, after which they are automatically available to be called from the command line and stored in a standard way.
* **Platform flexibility.** The notion of a "platform" is a first-class abstraction, making it easy to adapt the framework to new settings or multiple settings.

## <a name='getting-started'></a>2 Getting Started

### 2.1 Versions

* Python 3.7 or greater (this framework extensively uses features from Python 3.7)
* [PyTorch 1.4 or greater](https://www.pytorch.org)
* TorchVision 0.5.0 or greater
<!-- * [NVIDIA Apex](https://anaconda.org/conda-forge/nvidia-apex) (optional, but required for 16-bit training) -->

### 2.2 Setup

0. Make sure you have the appropriate Python version.
1. Install the Python dependencies from `requirements.txt`.
2. Clone this repository.
3. Modify `platforms/local.py` so that it contains the paths where you want datasets and results to be stored. By default, they will be stored in `./data/sims/` and `./data/datasets/`. To train with ImageNet, you will need to specify the path where ImageNet is stored.

### 2.3 The Command-Line Interface

All interactions with the framework occur through the command-line interface:

```
python EC.py
```

In response, you will see the following message.

```
================================================================================
Welcome to Expand-and-Cluster!
================================================================================
Choose a command to run:
    * EC.py train [...] => Train a model.
    * EC.py extract [...] => Runs Expand-and-Cluster on the selected model.
    * EC.py lottery [...] => Run a pruning experiment.
================================================================================
```

This framework has three sub-commands for its three experimental workflows: `train` (for training a network), `extract` (for running Expand-and-Cluster), and `lottery` (for running a pruning experiment). To learn about adding more experimental workflows, see `foundations/README.md`.

### 2.4 Training a Network

To train a network, use the `train` subcommand. You will need to specify the model to be trained, the dataset on which to train it, and other standard hyperparameters (e.g., batch size, learning rate, training steps). There are two ways to do so:

* **Specify these hyperparameters by hand.** Each hyperparameter is controlled by a command-line argument. To see the complete list, run `python EC.py train --help`. Many hyperparameters are required (e.g., `--dataset_name`, `--model_name`, `--lr`). Others are optional (e.g., `--momentum`, `--random_labels_fraction`).
* **Use the defaults.** Each model comes with a set of defaults that achieve standard performance. You can specify the name of the model you wish to train using the `--default_hparams` argument and load the default hyperparameters for that model.  You can still override any default using the individual arguments for each hyperparameter.

In practice, you will almost always begin with a set of defaults and optionally modify individual hyperparameters as desired. To view the default hyperparameters for ResNet-20 on CIFAR-10, use the following command. (For a full list of available models, see 2.11.) Each of the hyperparameters from before will be updated with its default value.

```
python EC.py train --default_hparams=mnist_lenet_20 --help
```

To train with these default hyperparameters, use the following command (that is, leave off `--help`):

```
python EC.py train --default_hparams=mnist_lenet_20
```

The training process will then begin. The framework will print the required and non-default hyperparameters for the training run and the location where the resulting model will be stored.

```
==================================================================================
Training a Model (Seed -1)
----------------------------------------------------------------------------------
Dataset Hyperparameters
    * dataset_name => mnist
    * batch_size => 128
Model Hyperparameters
    * model_name => mnist_lenet_20
    * model_init => kaiming_normal
    * batchnorm_init => uniform
    * act_fun => relu
Training Hyperparameters
    * optimizer_name => sgd
    * lr => 0.1
    * training_steps => 40ep
Output Location: expand-and-cluster/data/sims/train_4e06f40307/seed_-1/main
==================================================================================
```

Before each epoch, it will print the test error and loss.

```
train   ep 000  it 000  loss 2.816      acc 13.26%      ex 60000        time 0.00s
test    ep 000  it 000  loss 2.827      acc 13.52%      ex 10000        time 0.00s
```

To suppress these messages, use the `--quiet` command-line argument.

To override any default hyperparameter values, use the corresponding hyperparameter arguments. For example, to increase the batch size and learning rate and add 10 epochs of learning rate warmup:

```
python EC.py train --default_hparams=mnist_lenet_20 --batch_size=1024 --lr=0.8 --warmup_steps=10ep
```

A train experiment will have the following folder structure:
```
expand-and-cluster/
    data/
        sims/
            "train_{hasname}"/
                seed_1/
                seed_2/
                    main/
                        checkpoint.pth
                        hparams_dict
                        hparams.log
                        logger
                        model_ep0_it0.pth
                        model_ep40_it0.pth
```

### 2.5 Running an Expand-and-Cluster Experiment

Expand-and-Cluster is an algorithm to perform parameter identification of a target network. The target network (or teacher network) can only provide information in form of a dataset of input-output pairs. The procedure consists in training a bigger (overparameterised) network to imitate the target network function by minimisation of a mean square error loss.

To run an Expand-and-Cluster experiment, use the `extract` subcommand. You will need to specify hyperparameters for student models, for the dataset generation (from which target network the data must be generated, with which policy, etc...), for the training of such students, and for the extraction of the teacher parameters (the hyperparameters of Expand-and-Cluster).

```
python EC.py extract --help
```

A pruning experiment will have the following folder structure:
```
expand-and-cluster/
    data/
        sims/
            "extract_{hasname}"/
                seed_1/
                    main/
                        checkpoint.pth
                        hparams_dict
                        hparams.log
                        logger
                        model_ep0_it0.pth
                        model_ep10000_it0.pth
                        "clustering_{hasname}"/
                            extraction_hparams_dict
                            extraction_hparams.log
                            ECplots/
                            finetune_checkpoints/
                            reconstructed_model/
                                checkpoint.pth
                                logger
                                model_ep0_it0.pth
                                model_ep5000_it0.pth
```
The hashnames are computed twice: one for including all parameters related to training (extract hashname) and one computed just for the Expand-and-Cluster hyperparameters (clustering hasname). This allows for testing different Expand-and-Cluster hyperparameters on the same trained overparameterised students (without having ot re-run the computationally heavy training).

#### 2.5.1 Student model naming convention
Student models differ from classic networks models in two ways: 
- Some implementations allow training of multiple networks in parallel
- They bring their own specialised loss functions to deal with the above

To specify a student model is sufficient to append the prefix `student` to the model specification. Some models (see section 2.11) allow for parallel training of multiple networks (within a unique tensor): to specify the amount of parallel networks N, append `(N)` before the specification of the layer sizes, e.g. `--model_name 'students_mnist_lenet(10)_20'`

#### 2.5.2 Dataset generation
Overparameterised students will be trained on a dataset that can either be generated from a teacher network trained with the `train` command or selected from a file (latter to be implemented).

To specify the teacher network folder, just indicate the hashname of the experiment in the field `--teacher_name` and the corresponding seed run in `--teacher_seed` (see section 2.4 for the teacher folder structure), finally the input dimension of the teacher must be specified in `--d_in`.

The teacher network can be probed with different types of input data, the `--dataset_name` command allows choosing a dataset and the `--datagen` can be used to add transformation strategies separated by a `+` symbol. For example `--datagen mnist+random+grid` (to be implemented). 

#### 2.5.3 Extraction hyperparameters
The extraction hyperparameters allow to specify the $\beta$ (`--beta`) and $\gamma$ (`--gamma`) parameters of Expand-and-Cluster, along with other parameters for finetuning (`--finetune_training_steps`, `--finetune_lr`) or a dataset-dependent mask that disregards some connections of the input layer deemed uninformative (e.g. `--boruta mnist`).

### 2.6 Running a Pruning Experiment

A pruning experiment involves repeatedly training the network to completion, pruning weights and potentially _rewinding_ unpruned weights to their value at initialization, and retraining. For more details, see [_The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks_](https://arxiv.org/abs/1803.03635).

To run a lottery ticket experiment, use the `lottery` subcommand. You will need to specify all the same hyperparameters required for training. In addition, you will need to specify hyperparameters for pruning the network. To see the complete set of hyperparameters, run:

```
python EC.py lottery --help
```

For pruning, you will need to specify a value for the `--pruning_strategy` hyperparameter. The framework includes three classic pruning strategies: pruning the lowest magnitude weights globally in a sparse fashion (`sparse_global`), pruning the lowest magnitude weights by following the specifications of [Frankle & Carbin, 2019](https://arxiv.org/abs/1803.03635) or the classic magnitude pruning alternated with retraining (no rewinding) of [Han et al.](https://arxiv.org/abs/1510.00149) (For instructions on adding new pruning strategies, see Section 4.7.)

Once again, it is easiest to load the default hyperparameters for a model (which includes the pruning strategy and other pruning details) using the `--default_hparams` argument. In addition, you will need to specify the number of times that the network should be pruned, rewound, and retrained, known as the number of pruning _levels_. To do so, use the `--levels` flag. Level 0 is training the full network; specifying `--levels=3` would then prune, rewind, and retrain a further three times.

To run a lottery ticket experiment with the default hyperparameters for ResNet-20 on CIFAR-10 with three pruning levels, use the following command:

```
python EC.py lottery --default_hparams=cifar_resnet_20 --levels=3
```

### 2.7 Running a Lottery Ticket Experiment with Rewinding

In the original paper on the lottery ticket hypothesis, unpruned weights were always rewound to their values from initialization before the retraining phase. In recent work ([_Stabilizing the LTH/The LTH at Scale_](https://arxiv.org/abs/1903.01611), [_Linear Mode Connectivity and the LTH_](https://arxiv.org/abs/1912.05671) and [_The Early Phase of Neural Network Training_](https://arxiv.org/abs/2002.10365)), weights are typically rewound to their values from step `k` during training (rather than from initialization).

This framework incorporates that concept through a broader feature called _pretraining_. Optionally, the full network can be _pretrained_ for `k` steps, with the resulting weights used as the starting point for the lottery ticket procedure. Rewinding to step `k` and pretraining for `k` steps are functionally identical, but pretraining offers increased flexibility. For example, you can pretrain using a different training set, batch size, or loss function; this is precisely the experiment performed in Section 5 of [_The Early Phase of Neural Network Training_](https://arxiv.org/abs/2002.10365).

If you wish to use the same hyperparameters for pretraining and the training process itself (i.e., perform standard lottery ticket rewinding), you can set the argument `--rewinding_steps`. For example, to rewind to iteration 500 after pruning (or, equivalently, to pretrain for 500 iterations), use the following command:

```
python EC.py lottery --default_hparams=cifar_resnet_20 --levels=3 --rewinding_steps=500it
```

If you wish to have different behavior during the pre-training phase (e.g., to pre-train with self-supervised rotation or even on a different task), use the `--pretrain` argument.  After doing so, the `--help` interface will offer the full suite of pretraining hyperparameters, including learning rate, batch size, dataset, etc. By default, it will pretrain using the same hyperparameters as specified for standard training. You will noueed to set the `--pretrain_training_steps` argument to the number of steps for which you wish to pretrain. Note that the network will still only train for the number of steps specified in `--training_steps`. Any steps specified in `--pretrain_training_steps` will be subtracted from `--training_steps`. In addition, the main phase of training will start from step `--pretrain_training_steps`, including the learning rate and state of the dataset at that step.


### 2.8 Accessing Results

All experiments are automatically named according to their hyperparameters. Specifically, all required hyperparameters and all optional hyperparameters that are specified are combined in a canonical fashion and hashed. This hash is the name under which the experiment is stored. The results of a training run are then stored under:

```
<root>/train_<hash>/seed_<replicate>/main
```

`<root>` is the data root directory stored in `platforms/local.py`; it defaults to `expand-and-cluster/data/sims`.

The results themselves are stored in a file called `logger`, which only appears after training is complete. This file is a lightweight CSV where each line is one piece of telemetry data about the model. A line consists of $-separated values: the name of the kind of telemetry (e.g., `test-accuracy`), the iteration of training at which the telemetry data was collected (e.g., `391`), and the value itself. You can parse this file manually, or use `training/metric_logger.py`, which is used by the framework to read and write these files.


### 2.9 Checkpointing and Re-running

Each experiment will automatically checkpoint after every epoch. If you re-launch a job, it will automatically pick up from there. If a job has already completed, it will not run again unless you manually delete the associated results.

If you wish to run multiple copies of an experiment (which is good scientific practice), use the `--global_seed` argument. This optional argument specifies the seed of an experiment, e.g. `--global_seed 5` will produce the following:

<pre>
expand-and-cluster/data/sims/train_71bc92a97/seed_<b>5</b>/main
</pre>

rather than 

<pre>
expand-and-cluster/data/sims/train_71bc92a97/seed_<b>-1</b>/main
</pre>

Where -1 is the default global seed.

### 2.10 Modifying the Training Environment

To suppress the outputs, use the `--quiet` argument.

To specify the number of PyTorch worker threads used to load data, use the `--num_workers` argument. This value defaults to 0, although you will need to specify a value for training with ImageNet.

The framework will automatically use all available GPUs. To change this behavior, you will need to modify the number of visible GPUs using the `CUDA_VISIBLE_DEVICES` environment variable.

### 2.11 Available Models

#### 2.11.1 Standard models

The framework includes the following standard models. Each model family shares the same default hyperparameters.

|Model|Description|Name in Framework|Example|
|--|--|--|--|
|LeNet| A fully-connected MLP. | `mnist_mnist_P_Q_L...` where P, Q, and L are the number of neurons per layer. You can add as many layers as you like in this way. You do not need to include the output layer. | `mnist_lenet_300_100` |
|VGG for CIFAR-10| A convolutional network with max pooling in the style of VGG. | `cifar_vgg_D`, where `D` is the depth of the network (valid choices are 11, 13, 16 or 19). | `cifar_vgg_16` |
|ResNet for CIFAR-10| A residual network for CIFAR-10. This is a different family of architectures than those designed for ImageNet. | `cifar_resnet_D`, where `D` is the depth of the network. `D-2` must be divisible by 6 to be a valid ResNet | `cifar_resnet_20`, `cifar_resnet_110` |
|Wide ResNet for CIFAR-10| The ResNets for CIFAR-10 in which the width of the network can be varied. | `cifar_resnet_D_W`, where `D` is the depth of the network as above, and `W` is the number of convolutional filters in the first block of the network. If `W` is 16, then this network is equivalent to `cifar_resnet_D`; to double the width, set `W` to 32. | `cifar_resnet_20_128` |
|ResNet for ImageNet| A residual network for ImageNet. This is a different family of architectures than those designed for CIFAR-10, although they can be trained on CIFAR-10. | `imagenet_resnet_D`, where `D` is the depth of the network (valid choices are 18, 34, 101, 152, and 200) | `imagenet_resnet_50` |
|Wide ResNet for ImageNet| The ResNets for ImageNet in which the width of the network can be varied. | `imagenet_resnet_D_W`, where `D` is the depth of the network as above, and `W` is the number of convolutional filters in the first block of the network. If `W` is 64, then this network is equivalent to `imagenet_resnet_D`; to double the width, set `W` to 128. | `imagenet_resnet_50_128` |

#### 2.11.1 Student models
And their student version:

|Model|Description|Name in Framework|Example|
|--|--|--|--|
|LeNet| A fully-connected MLP. | `students_mnist_lenet(N)_P_Q_L...` where P, Q, and L work as specified above. N specifies the number of networks to be trained in parallel with an efficient implementation. | `students_mnist_lenet(10)_300_100` |


### 2.12 ImageNet

This framework includes standard ResNet models for ImageNet and a standard data preprocessing for ImageNet. Using the default hyperparameters and 16-bit precision training, `imagenet_resnet_50` trains to 76.1% top-1 accuracy in 22 hours on four V100-16GB GPUs. To use ImageNet, you will have to take additional steps.

1. Prepare the ImageNet dataset.
    1. Create two folders, `train`, and `val`, each of which has one subfolder for each class containing the JPEG images of the examples in that class.
    2. Modify `imagenet_root()` in `platforms/local.py` to return this location.
2. If you wish to train with 16-bit precision, you will need to install the [NVIDIA Apex](https://anaconda.org/conda-forge/nvidia-apex) and add the `--apex_fp16` argument to the training command.

(I think the apex feature is broken for the moment -- need to update this to newer python / torch versions).

## <a name=internals></a>3 Internals

This framework is designed to be extensible, making it easy to add new datasets, models, initializers, optimizers, extraction and pruning strategies, hyperparameters, workflows, and other customizations. This section discusses the internals. Section 4 is a how-to guide for extending the framework.

Note that this framework makes extensive use of Python [Data Classes](https://docs.python.org/3/library/dataclasses.html), a feature introduced in Python 3.7. You will need to understand this feature before you dive into the code. This framework also makes extensive use of object oriented subclassing with the help of the Python [ABC library](https://docs.python.org/3/library/abc.html).

### 3.1 Hyperparameters Abstraction

The lowest-level abstraction in the framework is an object that stores a bundle of hyperparameters. The abstract base class for all such bundles of hyperparameters is the `Hparams` Data Class, which can be found in `foundations/hparams.py`. This file also includes four subclasses of `Hparams` that are used extensively throughout the framework:
* `DatasetHparams` (which includes all hyperparameters necessary to specify a dataset, like its name and batch size)
* `ModelHparams` (which includes all hyperparameters necessary to specify a model, like its name and initializer)
* `TrainingHparams` (which includes all hyperparameters necessary to describe how to train a model, like the optimizer, learning rate, warmup, annealing, and number of training steps)
* `PruningHparams` (which is the base class for the hyperparameters required by each pruning strategy)
* `ExtractionHparams` (which includes all the hyperparameters of the clustering procedure + finetuning)

Each field of these dataclasses is the name of the hyperparameter. The type annotation is the type that the hyperparameter must have. If a default value is specified for the field, that is the default value for the hyperparameter; if no default is provided, then the hyperparameter is required and must be specified manually.

Each `Hparams` subclass must also set the `_name` and `_description` fields with default values that describe the nature of this bundle of hyperparameters. It may optionally include a string field `_hyperparameter` with a default string value that describes the hyperparameter and how it should be set. For example, in addition to the `lr` field, `TrainingHparams` has the `_lr` field that explains how the `lr` field should be set.

The `Hparams` subclass provides several behaviors to its subclasses. Most importantly, it has a static method `add_args` which takes as input a Python command-line `ArgumentParser` and adds each of the hyperparameters as a flag `--hyperparameter`. Since each hyperparameter has a name, type annotation, and possibly a default value and help text (the `_hyperparameter` field), it can be converted into a command-line argument automatically. This is how the per-hyperparameter command-line arguments are populated. This function optionally takes an instance of the class that overrides default values; this is how `--default_hparams` is implemented. Corresponding to the `add_args` static method is a `create_from_args` static method that creates an instance of the class from a Python `argparse.NameSpace` object that results from using the `ArgumentParser`.

Finally, the `Hparams` object has a `__str__` method that converts an instance into a string (`hashname`) in a canonical way. **During this conversion, any hyperparameters that are set to their default values are left off. This step is very important for ensuring that models are saved in a backwards compatible way as new hyperparameters are added** (i.e. when adding a new hyperparameter, add it with a defalut value that does not break all previous versions. This is accounted for in the naming conventions or `hashname` of the simulations)

### 3.2 Modules for Datasets, Models, Training, Extraction and Pruning

Running an Expand-and-Cluster experiment involves combining four largely independent components:
1. There must be a way to retrieve a dataset.
2. There must be a way to retrieve a model.
3. There must be a way to train a model on a dataset.
4. There must be a way to train students on a generated teacher dataset.
5. There must be a way to retrieve trained students and experiment different clustering parameters
4. Three must be a way to prune a model (for controls).

This framework breaks these components into distinct modules that are as independent as possible. The common specification for these modules is the `Hparams` objects. To request a dataset from the dataset module, provide a `DatasetHparams` instance. To request a model from the models module, provide a `ModelHparams` instance. To train a model, provide a dataset, a model, and a `TrainingHparams` instance. To recover parameters of a target / teacher model, provide such teacher, a dataset generation procedure and a `ExtractionHparams` instance. To prune a model, provide the model and a `PruningHparams` object. The inner workings of these modules can be understood largely independently from each other, with a few final abstractions to glue everything together.

### 3.3 The Datasets Module

Each dataset consists of two abstractions:

1. A `Dataset` that stores the dataset, labels, and any data augmentation.
2. A `DataLoader` that loads the dataset for training or testing. It must keep track of the batch size, multithreaded infrastructure for data loading, and random shuffling.

A dataset must subclass the `Dataset` and `DataLoader` abstract base classes in `datasets/base.py`. Both of these classes subclass the corresponding PyTorch `Dataset` and `DataLoader` classes, although they have a richer API to facilitate functionality in other modules and to enable build-in transformations like subsampling, random labels, and blurring.

For simple datasets that can fit in memory, these base classes provide most of the necessary functionality, so the subclasses are small. In fact, MNIST (`datasets/mnist.py`) and CIFAR-10 (`datasets/cifar10.py`) use the base `DataLoader` without modification. In contrast, ImageNet (`datasets/imagenet.py`) replaces all functionality due to the specialized needs of loading such a large dataset efficiently.

The external interface of this module is contained in `datasets/registry.py`. The registry contains a list of all existing datasets in the framework (so that they can be discovered and loaded). Its most important function is `get()`, which takes as input a `DatasetHparams` instance and a boolean specifying whether to load the train or test set; it returns the `DataLoader` object corresponding to the `DatasetHparams` (i.e., with the right batch size and additional transformations). This module also contains a function for getting the number of `iterations_per_epoch()` and the `num_classes()` corresponding to a particular `DatasetHparams`, both of which are important for other modules.

A dataset can also be constructed from a teacher network, in `datasets/teacher_dataset.py` is contained the implementation of such dataset. It consists of two extra methods to load the specified teacher from its hashname and probe the teacher with a selected dataset (see `--dataset` and `--datagen` options).

### 3.4 The Models Module

Each model is created by subclassing the `Model` abstract base class in `models/base.py`. This base class is a valid PyTorch `Module` with several additional abstract methods that support other functionality throughout the framework. In particular, any subclass must have static methods to determine whether a string model name (e.g., `cifar_resnet_20`) is valid and to create a model object from a string name, a number of outputs, and an initializer.

It must also have instance properties that return the names of the tensors in the output layer (`output_layer_names`) and all tensors that are available for pruning (`prunable_layer_names` - by default just the kernels of convolutional and linear layers). These properties are used elsewhere in the framework for transfer learning, weight freezing, and pruning.

Finally, it must have a static method that returns the set of default hyperparameters for the corresponding model family (as `Hparams` objects); doing so makes it possible to load the default hyperparameters rather than specifying them one by one on the command line.

For student network models (prefix `students_`) one extra method called `individual_losses` is required. This method takes care of computing the losses for each of the student trained in parallel and is used by some logging / plotting functions.

Otherwise, these models are identical to standard `Module`s in PyTorch.

The external interface of this module is contained in `models/registry.py`. Like `datasets/registry.py`, there is a list of all existing models in the framework so that they can be discovered and loaded. The registry similarly contains a `get()` function that, given a `ModelHparams` instance and the number of outputs, returns the corresponding `Model` as specified. In the course of creating a model, the `get()` function also loads the initializer and BatchNorm initializer specified in the `ModelHparams` instance. All initializers are functions stored in `models/initializers.py`, and no registration is necessary. All BatchNorm initializers are functions stored in `models/bn_initializers.py`, and no registration is necessary.

Finally, the registry has functions for getting the default hyperparameters for a model, loading a saved model from a path, and checking whether a path contains a saved model.

### 3.5 The Step Abstraction

In several places throughout the framework, it is necessary to keep track of a particular "step" of training. Depending on the particular framework, a step takes one of two forms: an iteration of training or an epoch number and an iteration offset into that epoch. In some places in this framework, it is easier to use one representation than the other. To make it easy to convert back and forth between these representations, all steps are stored as `Step` objects (`foundations/step.py`). A step object can be created from either representation, but it requires the number of iterations per epoch so that it can convert back and forth between these two representations.

### 3.6 The Training Module

The training module centers on a single function: the `train()` function in `training/train.py`. This function takes a `Model` and a `DataLoader` as arguments along with a `TrainingHparams` instance. It then trains the `Model` on the dataset provided by the `DataLoader` as specified in the training hyperparameters.

The `train()` function takes four other arguments. These include the `output_location` where all results should be stored, the optional `Step` object at which training should begin (the learning rate schedule and training set are advanced to this point), and the optional `Step` object at which training should end (if not the default value specified in the `TrainingHparams` instance).

Most importantly, it takes an argument called `callbacks`. This argument requires some explaining. A key design goal of the training module is to keep the main training loop in `train()` as clean as possible. This means that the loop should only contain standard setup, training, checkpointing behavior, and a `MetricLogger` to record telemetry. The loop should not be modified to add other behaviors, like saving the network state, running the test set, adding if-statements for new experiments, etc.

Instead, the behavior of the loop is modified by providing _callback_ functions (known as _hooks_ in other frameworks). These callbacks are called before every training step and after the final training step. They are provided with the current training step, the model, the optimizer, the output location, and the logger, and they can perform functions like saving the model state, running the test set, checkpointing, printing useful debugging information, etc. As new functionality is needed in the training loop, simply create new callbacks. 

The file `training/standard_callbacks.py` contains a set of the most common callbacks you are likely to use, like evaluating the model on a `DataLoader` or saving the model state. It also contains a set of higher-order functions that modify a callback to run at a certain step or interval. Finally, it includes a set of standard callbacks for a training run:
* Save the network before and after training
* Run the test set every epoch
* Update the checkpoint every epoch
* Save the logger after training
* Student specific callbacks

The file `training/train.py` contains a function called `standard_train()` that takes a model, dataset and training hyperparameters, and an output location as inputs and trains the model using the standard callbacks and the main training loop. This function is used by the `train`, `extract`, and `lottery` subcommands.

To create optimizers and learning rate scheduler objects, `train()` calls the `get_optimizer()` and `get_lr_schedule()` functions in `training/optimizers.py`, which serve as small-scale registries for these objects.

### 3.7 The Extraction Module

The extraction module is in the `extraction/` directory. It follows the same structure as the train and pruning modules. The `runner.py` file implements the `run()` function that takes care of the high-level steps of Expand-and-Cluster, namely: training overparameterised students and clustering them to reconstruct a teacher network.
The `expand-and-cluster.py` module implements the above mentioned steps with the functions:
* `train_students()` which simply takes care of calling the train function described in the previous section for student networks.
* `reconstruct()` which implements performs the layer-wise clustering followed by fine-tuning steps.

The bulk of this module is the `reconstruct()` function which takes as input all the hyperparameters. 
The `layer_reconstruction.py` implements the low level details of the clustering procedure.

### 3.8 The Pruning Module

The pruning module is in the `pruning/` directory. It contains a `Mask` abstraction, which keeps track of the binary mask `Tensor` for each prunable layer in a model. The module keeps track of different pruning strategy classes (subclasses of the abstract base class `Strategy` in `pruning/base.py`. Each pruning strategy has two members:
1. A static method `get_pruning_hparams()` that returns a subclass (_not_ an instance) of the `PruningHparams` class from `foundations/hparams.py`. Since different pruning methods may require different hyperparameters, each pruning method is permitted to specify its own `PruningHparams` object. This object is used to generate the command-line arguments for the pruning strategy specified by the `--pruning_strategy` argument.
2. A static method `prune` that takes a `PruningHparams` instance, a trained `Model`, and the current `Mask` and returns a new mask representing one further pruning step according to the hyperparameters.

The external interface of this module is contained in `pruning/registry.py`. Like the other registries, it has a `get()` function for getting a pruning class from a `PruningHparams` instance. It also has a `get_pruning_hparams` function for getting the `PruningHparams` subclass for a particular pruning strategy.

Finally, this module contains a `PrunedModel` class (in `pruning/pruned_model.py`). This class is a wrapper around a `Model` (from `models/base.py`) that applies a `Mask` object to prune weights. This class is used heavily by the lottery ticket and branch experiments to effectuate pruning.

### 3.9 Descriptors

These individual components (datasets, models, training, extraction, and pruning) come together to allow for training workflows. The framework currently has three training workflows: training a model normally (the `train` subcommand), identifying the parameters of a teacher network (the `extraction` subcommand) and running a pruning experiment (the `lottery` subcommand).

Each of these workflows requires a slightly different set of hyperparameters. Training a model requires `DatasetHparams`, `ModelHparams`, and `TrainingHparams` instances (but notably no `ExtractionHparams` or `PruningHparams`). An Expand-and-Cluster experiment requires `DatasetHparams`, `ModelHparams`, `TrainingHparams`, and `ExtractionHparams`. A pruning experiment needs `DatasetHparams`, `ModelHparams`, `TrainingHparams`,`PruningHparams` and, optionally, a separate set of `DatasetHparams` and `TrainingHparams` for pre-training.

In summary, each workflow needs a "bundle" of `Hparams` objects of different kinds. The framework represents this abstraction with a _descriptor_ object, which describes everything necessary to conduct the workflow (training a network, identifying a target network or running pruning experiment). These objects descend from `foundations/desc.py`, which contains the abstract base class `Desc`. This class is a Python dataclass whose fields are `Hparams` objects. It requires subclasses to implement a `add_args` and `create_from_args` static methods that create the necessary command-line arguments like the similar methods in the `Hparams` base class; typically, these functions will simply call the corresponding ones in the constituent `Hparams` instances.

Importantly, the `Desc` base class contains the function that makes automatic experiment naming possible. It has a property called `hashname` that combines all `Hparams` objects in its fields into a single string in a canonical way and returns the MD5 hash. __This hash later becomes the name under which each experiment is stored. It is therefore important to be careful when modifying `Hparams` or `Desc` objects, as doing so may break backwards compatibility with the hashes of pre-existing experiments.__

The training, extraction and lottery workflows contain subclasses of `Desc` in `training/desc.py` and `lottery/desc.py`. Each of these subclasses contains the requisite fields and implements the required abstract methods. They also include other useful properties derived from their constituent hyperparameters for the convenience of higher-level abstractions.

### 3.10 Wiring Everything Together with Runners

A descriptor has everything necessary to specify how a particular network should be trained, but it is missing other meta-data necessary to fully describe a run. For example, a training run needs a replicate number, and a lottery run needs to know the number of pruning levels for which to run. This information is captured in a higher-level abstraction known as a `Runner`.

Each runner (subclasses of `Runner` in `foundations/runner.py`) has static `add_args` and `create_from_args` methods that interact with command-line arguments, calling the same methods on their requisite descriptors and adding other runner-level arguments. Once a `Runner` instance has been created, the `run()` method (which takes no arguments) initiates the run. This includes creating a model and making one or more calls to `train()` depending on the details of the runner. For example, the runner for the `train` subcommand (found in `training/runner.py`) performs a single training run; the runner for the `extraction` subcommand (found in `extraction/runner.py`) implements the Expand-and-Cluster algorithm which consists in training $N$ overparameterised students and clustering their weights to recover the teacher netowkr; and finally the `lottery` subcommand (found in `lottery/runner.py`) pretrains a network, trains it, prunes it, and then repeatedly re-trains and prunes it using the `PrunedModel` class and a pruning `Strategy`.

The runners are the highest level of abstraction, connecting directly to the command-line interface. Each runner must be registered in `cli/runner_registry.py`. The name under which it is added is the name of the subcommand used to access it on the command-line.

### 3.11 Platforms

It is typical to use the same codebase on many different infrastructures (such as a local machine, a cluster, and one or more cloud providers). Each of these infrastructures will have different locations where results and datasets will be stored and different ways of accessing filesystems. They may even need to call the runner's `run()` functions in a different fashion.

To make it easy to run this framework on multiple infrastructures, it includes a `Platform` abstraction. Each `Platform` class describes where to find resources (like datasets), where to store results, what hardware is available (if there are GPUs and how many if so) and how to run a job on the platform. Arguments may be required to create a `Platform` instance, for example the number of worker threads to use.

To enable this behavior, each `Platform` object is a dataclass that descends from `Hparams`; this makes it possible for its fields to be converted into command-line arguments and for an instance to be created from these arguments. The abstract base `Platform` class that all others subclass (found in `platforms/base.py`) contains a field for the number of worker threads to use for data loading.
It also has abstract properties that specify where data should be found and results stored; these must be implemented by each subclass.
Finally, it has a series of static methods that mediate access to the filesystem; by default, these are set to use the standard Python commands for the local filesystem, but it may be important to override them on certain infrastructures.

Finally, it has a method called `run_job()` that receives a function `f` as an argument, performs any pre-job setup, and calls the function. Most importantly, this function _installs_ the `Platform` instance as the global platform for the entire codebase. In practice, this entails modifying the global variable `_PLATFORM` in `platforms/platform.py`. Throughout the codebase, modules look to this global variable (accessed through the `get_platform()` function in `platforms/platform`) to determine where data is stored, the hardware on which to run a job, etc. It was cleaner to make the current platform instance a global rather than to carry it along through every function call in the codebase.

The included `local` platform will automatically use all GPUs available using PyTorch `DataParallel`. If you choose to do distributed training, the `base` platform includes primitives for distributed training like `rank`,  `world_size`, `is_primary_process`, and `barrier`; the codebase calls all of these functions in the proper places so that it is forward-compatible with distributed training should you choose to use it.

All platform subclasses must be registered in `platforms/registry.py`, which makes them available for use at the command line using the `--platform` argument. By default the `local` platform (which runs on the local machine) is used.

## <a name=extending></a>4 Extending the Framework

Please read Section 3 before trying to extend the framework. Careless changes can have unexpected consequences, such as breaking backwards compatibility and making it impossible for the framework to access your existing models.

### 4.1 Adding a New Dataset

Create a new file in the `datasets` directory that subclasses the abstract base classes `Dataset` and `DataLoader` in `datasets/base.py` with classes that are also called `Dataset` and `DataLoader`. Modify `datasets/registry.py` to import this module and add the module (_not_ the classes in the module) to the dictionary of `registered_datasets` with the name that you wish for it to be called. For small datasets that fit in memory (e.g., SVHN), use `datasets/cifar10.py` as a template and take advantage of functionality built into the base classes. For larger datasets (e.g., Places), use `datasets/imagenet.py` as a template; you may need to throw away functionality in the base classes.

### 4.2 Adding a New Model

Create a new file in the `models` directory that subclasses the abstract base class `Model` in `models/base.py`. Modify `models/registry.py` to import this module and add the class (_not_ the module containing the class) to the list of `registered_models`. As a template, use `models/cifar_resnet.py`.

### 4.3 Adding a New Initializer

Add the new initializer function to `models/initializers.py` under the name that you want it to be called. To add a new BatchNorm initializer, do the same in `models/bn_initializers.py`. No registration is necessary in either case.

### 4.4 Adding a New Optimizer

Modify the if-statement in the `get_optimizer` function of `training/optimizers.py` to create the new optimizer when the appropriate hyperparameters are specified.

### 4.5 Adding a New Hyperparameter

Modify the appropriate set of hyperparameters in `foundations/hparams.py` to include the desired hyperparameter. **The hyperparameter must have a default value, and this default value must eliminate the effect of the hyperparameter.** The goal is to ensure that adding this hyperparameter is backwards compatible. This default value should ensure that all preexisting models would train in the same way if this hyperparameter had been present and set to its default value.

If the new hyperparameter doesn't have a default value, then it will change the way results directory names are computed for all preexisting models, making it impossible for the framework to find them. If the default value is not a no-op, then all preexisting models (where were trained under the implicit assumption that this hyperparameter was set to its default value) will no longer be valid.

The unit tests include a regression test for the directory names generated by the framework to ensure that new hyperparameters have not inadvertently changed existing names. Be sure to run the unit tests after adding a new hyperparameter.

### 4.6 Modifying the Training Loop

Where possible, try to modify the training loop by creating a new kind of optimizer, a new kind of loss function, or a new callback. New callbacks can be added to `standard_train()` in `training/train.py`, gated by a new hyperparameter. The training loop is designed to be as clean and pared down as possible and to use callbacks and the other objects to abstract away the complicated parts, so try to avoid modifying the loop if at all possible. If you need to access the gradients, consider adding a second set of `post_gradient_callbacks` that are called after the gradients are computed but before the optimizer steps. This would be a new argument for `train()` and possibly `standard_train()` in `training/train.py`.

### 4.7 Adding a New Pruning Strategy

Create a new file in the `pruning` directory that subclasses the abstract base class `Strategy` in `pruning/base.py`. The new pruning strategy needs a static method that returns the hyperparameters it requires (recall that each pruning method can have a different set of hyperparameters). Modify `pruning/registry.py` to import this module and add the class (_not_ the module containing the class) to the dictionary of `registered_strategies` under the key that you want to use to describe this strategy going forward.

### 4.8 Adding a New Workflow

1. Create a new directory to store the workflow.
2. Create a file with a descriptor data class that subclasses from `Desc` in `foundations/desc.py`; it should have fields for any `Hparams` objects necessary to describe the workflow. It should implement the `add_args`, `create_from_args`, and `name_prefix` static methods as necessary for the desired behavior.
3. Create a file with a runner class that subclasses from `Runner` in `foundations/runner.py`. Create a constructor or make the runner a data class. Implement the `add_args` and `create_from_args` static methods to interact with the command line. Implement the `description` static method to describe the runner. Implement the `display_output_location` instance method to respond to the `--display_output_location` command-line argument. Finally, create the `run` instance method with the logic for performing any training necessary for the workflow.
4. Register the runner in `cli/runner_registry.py`.

### 4.9 Adding a New Platform

Subclass the `Platform` class (from `platforms/base.py`) in a new file in the `platforms` directory. Be sure to make it a dataclass. Add any additional fields and, optionally, help strings for these fields (named `_f` for a field `f`). Implement all the required abstract properties (`root`, `dataset_root`, and `imagenet_root` if ImageNet is available). Finally, override `run_job()` if different behavior is needed for the platform; be sure to ensure that the modified `run_job()` method still installs the platform instance before calling the job function `f`.


## <a name=acknowledgements></a>5 Acknowledgements

I would like to thank Christos Sourmpis for the fruitful discussions over implementation strategies for running systematic experiments, and Johnatan Frankle for making the [OpenLTH](https://github.com/facebookresearch/open_lth) framework available on GitHub!
