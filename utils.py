#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from copy import deepcopy

from trainer import Trainer
from custom_dataset_classes import CachedDataset


def flip(x, dim):
    """
    Flips a dimension (reverse order). BidiLSTM for example uses this feature
    to apply the a LSTM with reversed time step (opposite direction).

    :param x: Tensor, which has a dimension to be flipped. The dimensions of x
        can be arbitrary.
    :param dim: The dimension/axis to be flipped.
    :return: New tensor with flipped dimension

    :example:
        >>> flip([[1,2,3], [4,5,6], [7,8,9]], 0)
        [[7,8,9], [4,5,6], [1,2,3]]
    """
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]


def get_partitions(height, width, depth):
    """
    Used by ReNet random search to get all possible window size sequences. For
    example for CIFAR-10, the images have the shape 32x32, so it is possible to
    use a window size of 2, 4, etc. This function finds all possible window
    sizes, depending on the amount of ReNet Layers.

    :param width: Width of the image
    :param height: Height of the image
    :param depth: How many ReNet Layers
    :return: Sequence of all possible window sizes.
    """
    width_kernels = set([])
    for i in range(2, int(width / 2) + 1):
        if width % i == 0:
            width_kernels.add(i)
    if width > 1:
        width_kernels.add(width)

    height_kernels = set([])
    for i in range(2, int(height / 2) + 1):
        if height % i == 0:
            height_kernels.add(i)
    if height > 1:
        height_kernels.add(height)

    intersection_kernels = width_kernels.intersection(height_kernels)
    if depth == 1:
        return [[element] for element in intersection_kernels]

    possible_kernels = []
    for kernel_size in intersection_kernels:
        temp_kernels = get_partitions(
            int(width / kernel_size),
            int(height / kernel_size),
            depth - 1
        )
        if len(temp_kernels) != 0:
            for lst in temp_kernels:
                possible_kernels.append([kernel_size] + deepcopy(lst))
    return possible_kernels


def loguniform(low=0, high=1, size=None):
    """
    Needed for choosing values for random search.

    :param low: Lower boundary of the output interval.
        All values generated will be greater than or equal to low.
        The default value is 0.
    :param high: Upper boundary of the output interval.
        All values generated will be less than high. The default value is 1.0.
    :param size: Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. If size is None (default),
        a single value is returned if low and high are both scalars.
    :return: Value between lower and upper bound.
    """
    return np.exp(np.random.uniform(low, high, size))


def get_dataset(filename, batch_size, shuffle=True, training=True, pin_memory=True, num_workers=2, random_subset=1.0):
    """
    Returns an iterable dataset.

    :param num_workers: Amount of workers for the dataset.
    :param training: Flag to signalize that the model is training or not.
        This is for the num_workers parameter. For visualization and
        num_workers=2, a "broken pipe" error can appear.
    :param filename: Dataset-filename
    :param batch_size: Batch Size for training
    :param shuffle: Boolean, if dataset set should be shuffled
    :return: Iterable dataset.
    """
    infile = open(os.path.abspath(filename), 'rb')
    ds = pickle.load(infile)
    infile.close()

    loader = torch.utils.data.DataLoader(dataset=CachedDataset(ds),
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                             list(range(int(random_subset*len(ds))))
                                         ) if random_subset < 1.0 else None,
                                         pin_memory=pin_memory,
                                         num_workers=num_workers if training else 0)
    return loader


def start_training(*args, **kwargs):
    """
    Instantiate Trainer and starts the training
    :param args: Arbitrary arguments for training
    """
    trainer = Trainer(*args, **kwargs)
    print("Trainer instantiated.")

    # Starting the training with keyboard interrupt catch
    # (saving current experiment)
    try:
        trainer.train()
    except KeyboardInterrupt:
        out_fname = trainer.filename + "_interrupted"
        trainer.save_progress(trainer.output_directory, out_fname, 'NONE')
