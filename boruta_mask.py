import os

import numpy as np
import matplotlib.pyplot as plt

import boruta

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

from foundations.hparams import DatasetHparams, Hparams
import platforms.platform
import platforms.registry
import datasets.registry


def compute_pixel_mask(X, y):
    boruta = BorutaPy(
        estimator=RandomForestClassifier(max_depth=5, n_jobs=-1),
        n_estimators='auto',
        max_iter=100,
        verbose=1
    ).fit(X, y)
    certain_mask = boruta.support_
    uncertain_mask = boruta.support_weak_
    return certain_mask, uncertain_mask


dataset_names = ["cifar10", "fashion_mnist", "mnist"]

platform = platforms.registry.get('local')(num_workers=0)
platforms.platform._PLATFORM = platform

for dataset_name in dataset_names:

    dataloader = datasets.registry.get(DatasetHparams(dataset_name=dataset_name, batch_size=100, do_not_augment=True),
                                       train=True)

    try:
        X = dataloader.dataset._examples.numpy()
    except:
        X = dataloader.dataset._examples

    shape_data = X.shape[1:]
    X = X.reshape(X.shape[0], -1)
    y = dataloader.dataset._labels

    certain_mask, uncertain_mask = compute_pixel_mask(X, y)
    np.save(os.path.join(platform.boruta_root, f"{dataset_name}_boruta_mask.npy"), certain_mask)

    # certain_mask = np.load(os.path.join(platform.boruta_root, f"{dataset_name}_boruta_mask.npy"))

    if len(shape_data) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        m = np.array(certain_mask.reshape(shape_data), dtype=np.float32)
        ax.imshow(certain_mask.reshape(shape_data), vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle(f"{dataset_name} Boruta mask")
        fig.savefig(os.path.join(platform.boruta_root, f"{dataset_name}_boruta_certain.png"))

    elif len(shape_data) == 3:
        fig, ax = plt.subplots(1, 3, figsize=(5, 2))
        for i in range(3):
            m = np.array(certain_mask.reshape(shape_data)[:, :, i], dtype=np.float32)
            im = ax[i].imshow(certain_mask.reshape(shape_data)[:, :, i], vmin=0, vmax=1)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        fig.suptitle(f"{dataset_name} Boruta mask")
        fig.savefig(os.path.join(platform.boruta_root, f"{dataset_name}_boruta_certain.png"))
