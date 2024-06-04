"""
 # Created on 12.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: class for any teacher-generated dataset
 #
"""

import os
import torch
import pickle
import models
import numpy as np

import torchvision.transforms as T

import models.registry

from datasets import base
from platforms.platform import get_platform
from foundations.hparams import DatasetHparams, ModelHparams
from datagen import registry


class Dataset(base.Dataset):
    """The teacher generated dataset."""

    def num_train_examples(self): return self._num_examples

    def num_test_examples(self): return self._num_examples

    def num_classes(self): return self._num_classes

    @staticmethod
    def get_train_set(use_augmentation):
        pass

    @staticmethod
    def get_test_set():
        pass

    def __init__(self, dataset_hparams: DatasetHparams, use_augmentation):
        if dataset_hparams.teacher_name is not None:
            examples, labels = self.extract_teacher_data(dataset_hparams, use_augmentation)
        elif dataset_hparams.teacher_file is not None:  # load just a dataset of input-output pairs
            data_file = os.path.join(get_platform(), "data", "train_" + dataset_hparams.teacher_name)
            dataset = np.load(data_file)
            examples = dataset.X
            labels = dataset.y

        self._num_examples = examples.shape[0]
        self._num_classes = labels.shape[1]
        super(Dataset, self).__init__(examples, labels)

    def extract_teacher_data(self, dataset_hparams, use_augmentation):
        if not hasattr(dataset_hparams, "teacher_seed"):
            raise ValueError('No --teacher_seed specified!')
        teacher_folder = os.path.join(get_platform().root, "train_" + dataset_hparams.teacher_name,
                                      "seed_" + dataset_hparams.teacher_seed, "main")
        model = self.get_specified_model(teacher_folder)
        return registry.get(dataset_hparams, model, use_augmentation)

    @staticmethod
    def get_specified_model(directory):
        """ Returns the final saved model from the directory """
        all_models = [d for d in os.listdir(directory) if d.startswith("model_")]
        eps = [int(model_name.split("_")[1][2:]) for model_name in all_models]
        id_max_ep = np.argmax(eps)
        model_state_dict = torch.load(os.path.join(directory, all_models[id_max_ep]),
                                      map_location=get_platform().torch_device)
        loaded_model_hparams = torch.load(os.path.join(directory, "hparams_dict"))["model_hparams"]
        loaded_model_hparams = ModelHparams.create_from_args(loaded_model_hparams)
        model = models.registry.get(loaded_model_hparams)
        model.load_state_dict(model_state_dict)
        return model

    def __getitem__(self, index):
        example, label = self._examples[index], self._labels[index]
        return example, label

    # IMPORTANT: the base class implements this as labels.size (since labels are scalar digits), while here the
    # labels are 10-dimensional vectors of floats
    def __len__(self):
        return self.num_train_examples()


DataLoader = base.DataLoader
