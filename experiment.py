#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import sys
import os
import math
import random
import numpy as np
import torch

from sacred import Experiment

sys.path.insert(0, os.path.abspath('..'))  # noqa

from lucky_trainer.utils import loguniform, get_dataset, get_partitions
from lucky_trainer.trainer import Trainer
from src.models.bidi_lstm import BidiLSTM
from src.models.n_renet import N_ReNet
from src.models.conv_lstm import ConvLSTM

# Device configuration (DO NOT EDIT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate Experiment
ex = Experiment('experiments_cifar10')


def get_model(model_params, train_loader):
    """
    Loads the model given by the entry <model> in the dictionary model_params.

    :param model_params: Dictionary with the parameters and name of the model
    :param train_loader: Train dataset, which is needed to get important
            dimensions like Time-Steps
    :return:  A Model (BidiLSTM, ReNet, ConvLSTM, MD-LSTM, pyraMiD-LSTM)
    """
    if model_params['model'] == 'bidi_lstm':
        return BidiLSTM(next(iter(train_loader))[0][0].shape[1], model_params)
    elif model_params['model'] == 'renet':
        assert next(iter(train_loader))[0][0].shape[0] == \
               model_params['input_dim'][0] and \
               next(iter(train_loader))[0][0].shape[1] == \
               model_params['input_dim'][1] and \
               next(iter(train_loader))[0][0].shape[2] == \
               model_params['input_dim'][2], \
               "<input_dim> does not match dataset"
        return N_ReNet(model_params['input_dim'], model_params)
    elif model_params['model'] == 'conv_lstm':
        return ConvLSTM(next(iter(train_loader))[0][0].shape, model_params)
    elif model_params['model'] == 'md_lstm':
        pass  # todo
    elif model_params['model'] == 'pyramid_lstm':
        pass  # todo
    else:
        raise NameError(
            model_params['model'] + ' is not defined.'
            + ' Choose one of the following: bidi_lstm, renet, '
            + 'conv_net, md_lstm or pyramid_lstm.'
        )


@ex.named_config
def conv_lstm_dataset():
    raise NotImplementedError


@ex.named_config
def md_lstm_dataset():
    raise NotImplementedError


@ex.named_config
def pyramid_lstm_dataset():
    raise NotImplementedError


@ex.named_config
def renet_dataset_cifar10():
    """
    Dataset file names for ReNet.
    """
    dataset = {
        'train_filename':
            '../data/CIFAR10/input_data/train_renet_cifar10',
        'validation_filename':
            '../data/CIFAR10/input_data/valid_renet_cifar10',
        'test_filename':
            '../data/CIFAR10/input_data/test_renet_cifar10'
    }


@ex.named_config
def renet_dataset_tiny_imagenet():
    """
    Dataset file names for ReNet.
    """
    dataset = {
        'train_filename':
            '../data/tiny_ImageNet/input_data/train_renet_tiny_imagenet',
        'validation_filename':
            '../data/tiny_ImageNet/input_data/valid_renet_tiny_imagenet'
    }


@ex.named_config
def bidi_lstm_datasets_row():
    raise NotImplementedError


@ex.named_config
def bidi_lstm_datasets_column():
    raise NotImplementedError


@ex.named_config
def bidi_lstm_dataset_block_8x8():
    raise NotImplementedError


@ex.named_config
def bidi_lstm_dataset_block_4x4():
    raise NotImplementedError


@ex.named_config
def random_search_bidi_lstm():
    """
    Configuration for the BidiLSTM model.
    """
    model_params = {
        'model': 'bidi_lstm',
        'layer_dim': random.choice(range(1, 5)),
        'hidden_dim': [],
        'output_dim': 10,
        'dropout_rate': []
    }
    model_params['hidden_dim'] = [int(loguniform(math.log(100), math.log(300)))
                                  for _ in range(model_params['layer_dim'])]
    model_params['dropout_rate'] = [
        float(loguniform(math.log(0.25), math.log(0.75)))
        for _ in range(model_params['layer_dim'])]

    train_params = {
        'max_epochs': 200,
        'batch_size': 128,
        'early_stopping_patience': 10,
        'acc_metric': 'classification',
        'class_dim': 1,
        'loss': random.choice(['CrossEntropyLoss']),
        'optimizer': random.choice(['RMSprop', 'Adam', 'Adadelta']),
        'opt_lr': float(10 ** np.random.uniform(-3, 0))
    }


@ex.named_config
def random_search_renet():
    """
    Configuration for the ReNet model.
    """
    model_params = {
        'model': 'renet',
        'reNet_layer_dim': random.choice(range(1, 5)),
        'linear_layer_dim': random.choice(range(1, 5)),
        'reNet_hidden_dim': [],
        'linear_hidden_dim': [],
        'dropout_rate': [],
        'input_dim': (32, 32, 3),  # H, W, C
        'rnn_types': [],
        'window_size': [],
        'output_dim': 10
    }
    model_params['reNet_hidden_dim'] = [
        int(loguniform(math.log(100), math.log(300)))
        for _ in range(model_params['reNet_layer_dim'])]

    model_params['dropout_rate'] = \
        [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))] + \
        [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))
         for _ in range(model_params['reNet_layer_dim'])] + \
        [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))
         for _ in range(model_params['linear_layer_dim'])]
    model_params['rnn_types'] = [random.choice(['GRU', 'LSTM', 'RNN'])
                                 for _ in
                                 range(model_params['reNet_layer_dim'])]
    try:
        model_params['window_size'] = random.choice(
            get_partitions(model_params['input_dim'][0],
                           model_params['input_dim'][1],
                           model_params['reNet_layer_dim']))
    except IndexError:
        raise IndexError(
            "<reNet_layer_dim> can not be used "
            + "to create partitions out of <input_dim>")
    model_params['linear_hidden_dim'] = [
        int(loguniform(math.log(200), math.log(500)))
        for _ in range(model_params['linear_layer_dim'])]

    train_params = {
        'max_epochs': 200,
        'batch_size': random.choice([32, 64, 128, 256]),
        'early_stopping_patience': 20,
        'acc_metric': 'classification',
        'class_dim': 1,
        'loss': random.choice(['CrossEntropyLoss']),
        'optimizer': random.choice(['RMSprop', 'Adam', 'Adadelta']),
        'opt_lr': float(10 ** np.random.uniform(-3, 0))
    }


@ex.named_config
def random_search_renet_tiny_imagenet():
    """
    Configuration for the ReNet model.
    """
    model_params = {
        'model': 'renet',
        'reNet_layer_dim': random.choice(range(5, 10)),
        'linear_layer_dim': random.choice(range(1, 5)),
        'reNet_hidden_dim': [],
        'linear_hidden_dim': [],
        'dropout_rate': [],
        'input_dim': (64, 64, 3),  # H, W, C
        'rnn_types': [],
        'window_size': [],
        'output_dim': 200
    }
    model_params['reNet_hidden_dim'] = [
        int(loguniform(math.log(500), math.log(750)))
        for _ in range(model_params['reNet_layer_dim'])]

    model_params['dropout_rate'] = \
        [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))] + \
        [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))
         for _ in range(model_params['reNet_layer_dim'])] + \
        [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))
         for _ in range(model_params['linear_layer_dim'])]
    model_params['rnn_types'] = [random.choice(['GRU', 'LSTM', 'RNN'])] \
        * model_params['reNet_layer_dim']
    try:
        model_params['window_size'] = random.choice(
            get_partitions(model_params['input_dim'][0],
                           model_params['input_dim'][1],
                           model_params['reNet_layer_dim']))
    except IndexError:
        raise IndexError(
            "<reNet_layer_dim> can not be used "
            + "to create partitions out of <input_dim>")
    model_params['linear_hidden_dim'] = [
        4096 #int(loguniform(math.log(2500), math.log(7000)))
        for _ in range(model_params['linear_layer_dim'])]

    train_params = {
        'skip_test': True,
        'max_epochs': 200,
        'batch_size': 8,
        'early_stopping_patience': 20,
        'acc_metric': 'classification',
        'class_dim': 1,
        'top_k': 1,
        'loss': random.choice(['CrossEntropyLoss']),
        'optimizer': random.choice(['RMSprop', 'Adam', 'Adadelta']),
        'opt_lr': float(10 ** np.random.uniform(-3, -1))
    }

@ex.named_config
def vgg16_like_renet_tiny_imagenet():
    """
    Configuration for the ReNet model.
    """
    model_params = {
        'model': 'renet',
        'reNet_layer_dim': random.choice(range(5, 10)),
        'linear_layer_dim': 2,  # random.choice(range(1, 5)),
        'reNet_hidden_dim': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        'linear_hidden_dim': [4096, 4096, 4096],
        'dropout_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
        'input_dim': (64, 64, 3),  # H, W, C
        'rnn_types': ['LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 'LSTM'],
        'window_size': [1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2],
        'output_dim': 200
    }
    #model_params['reNet_hidden_dim'] = [
    #    int(loguniform(math.log(500), math.log(750)))
    #    for _ in range(model_params['reNet_layer_dim'])]

    #model_params['dropout_rate'] = \
    #    [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))] + \
    #    [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))
    #     for _ in range(model_params['reNet_layer_dim'])] + \
    #    [float("{0:.1f}".format(loguniform(math.log(0.2), math.log(0.6))))
    #     for _ in range(model_params['linear_layer_dim'])]
    #model_params['rnn_types'] = [random.choice(['GRU', 'LSTM', 'RNN'])] \
    #    * model_params['reNet_layer_dim']
    #try:
    #    model_params['window_size'] = random.choice(
    #        get_partitions(model_params['input_dim'][0],
    #                       model_params['input_dim'][1],
    #                       model_params['reNet_layer_dim']))
    #except IndexError:
    #    raise IndexError(
    #        "<reNet_layer_dim> can not be used "
    #        + "to create partitions out of <input_dim>")
    #model_params['linear_hidden_dim'] = [
    #    4096 #int(loguniform(math.log(2500), math.log(7000)))
    #    for _ in range(model_params['linear_layer_dim'])]

    train_params = {
        'skip_test': True,
        'max_epochs': 200,
        'batch_size': 8,
        'early_stopping_patience': 20,
        'acc_metric': 'classification',
        'class_dim': 1,
        'top_k': 1,
        'loss': random.choice(['CrossEntropyLoss']),
        'optimizer': 'Adam',  # random.choice(['RMSprop', 'Adam', 'Adadelta']),
        'opt_lr': 0.1  # float(10 ** np.random.uniform(-3, -1))
    }


@ex.named_config
def random_search_conv_lstm():
    """
    Configuration for the ConvLSTM model.
    """
    model_params = {  # todo
        'conv_layer_dim': 3,
        'conv_hidden_dim': [128, 64, 64],
        'patch_size': (4, 4),
        'input_kernel_size': 5,
        'kernel_size': [5, 5, 5]
    }
    train_params = {
        'max_epochs': 200,
        'batch_size': 128,
        'early_stopping_patience': 20,
        'loss': random.choice(['CrossEntropyLoss']),
        'optimizer': random.choice(['RMSprop', 'Adam', 'Adadelta']),
        'opt_lr': float(10 ** np.random.uniform(-3, 0))
    }


@ex.named_config
def random_search_md_lstm():
    """
    Configuration for the MDLSTM model.
    """
    train_params = {
        'max_epochs': 2,
        'batch_size': 128,
        'early_stopping_patience': 3
    }


@ex.named_config
def random_search_pyramid_lstm():
    """
    Configuration for the PyraMiD-LSTM model.
    """
    train_params = {
        'max_epochs': 2,
        'batch_size': 128,
        'early_stopping_patience': 3
    }


@ex.automain
def run(dataset, model_params, train_params):
    """
    Runs the experiment with given model parameters and dataset

    :param dataset: Dataset-filename.
    :param model_params: Parameters of the model
    :param train_params: Parameters, which contains max. epoch amount, batch
        size and the patience value for the early stopping mechanism.
    :return: It saves the best model during the training and returns the
        corresponding stats like validation loss.
    """
    # Load datasets
    train_loader = get_dataset(
        dataset['train_filename'],
        train_params['batch_size']
    )
    validation_loader = get_dataset(
        dataset['validation_filename'],
        train_params['batch_size'],
        shuffle=False
    )
    if 'skip_test' in train_params.keys():
        skip_test = train_params['skip_test']
    else:
        skip_test = False
    if not skip_test:
        test_loader = get_dataset(
            dataset['test_filename'],
            train_params['batch_size'],
            shuffle=False
        )
    else:
        test_loader = None

    # Instantiate model
    model = get_model(model_params, train_loader).to(device)
    print(model)

    output_directory = ex.observers[0].dir   # todo: get exp dir
    filename = 'model_state_dict.pth'

    # Train the model
    trainer = Trainer(
        model,
        model_params,
        train_params,
        dataset,
        train_loader,
        validation_loader,
        test_loader,
        output_directory,
        filename
    )
    try:
        if 'acc_metric' in train_params.keys():
            tr_loss, tr_acc, test_loss, test_acc = trainer.train()
        else:
            tr_loss, test_loss = trainer.train()
            tr_acc, test_acc = "/", "/"
    except KeyboardInterrupt:
        trainer.save_progress(output_directory, filename)

    if not skip_test:
        result = {
            'train_loss': tr_loss,
            'train_accuracy': tr_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }
    else:
        result = {
            'train_loss': tr_loss,
            'train_accuracy': tr_acc,
            'val_loss': test_loss,
            'val_accuracy': test_acc
        }
    return result
