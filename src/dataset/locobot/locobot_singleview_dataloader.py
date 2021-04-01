import os
import random
import ipdb
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from torchvision.datasets.folder import has_file_allowed_extension

# from src.dataset.locobot.locobot_dataset import LocobotDataset
from src.dataset.robonet.robonet_dataset import RoboNetDataset

def create_loaders(config):
    file_type = "hdf5"
    files = []
    file_labels = []

    data_path = os.path.join(config.data_root, "locobot", "c0")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            files.append(d.path)
            file_labels.append("locobot")

    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    # TODO: change dataset splitting
    n_test = 200
    n_train = 1000

    X_test = files[:n_test]
    y_test = file_labels[:n_test]

    X_train = files[n_test: n_test + n_train]
    y_train = file_labels[n_test: n_test + n_train]
    print("loaded locobot data", len(X_train) + len(X_test))

    augment_img = config.img_augmentation
    train_data = RoboNetDataset(X_train, y_train, config, augment_img=augment_img)
    test_data = RoboNetDataset(X_test, y_test, config)

    train_loader = DataLoader(
        train_data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_loader = DataLoader(
        test_data,
        num_workers=config.data_threads,
        batch_size=config.test_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    return train_loader, test_loader


if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    # config.data_root = "/mnt/ssd1/pallab/locobot_data"
    config.batch_size = 16  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    config.data_threads = 0
    config.action_dim = 5
    config.model_use_heatmap = True

    train_loader, test_loader = create_loaders(config)

    for data in train_loader:
        images = data["images"]
        states = data["states"]
