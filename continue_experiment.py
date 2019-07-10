#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))  # noqa

from src.models.bidi_lstm import BidiLSTM
from src.models.n_renet import N_ReNet
from src.models.conv_lstm import ConvLSTM
from src.models.md_lstm import MDMDLSTMModel
from lucky_trainer.utils import get_dataset
from lucky_trainer.trainer import Trainer

# Device configuration (DO NOT EDIT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-d",
                    "--directory",
                    type=str,
                    required=True,
                    help="file path of the state without last /")
parser.add_argument("-f",
                    "--filename",
                    type=str,
                    required=True,
                    help="file name of the state")
args = parser.parse_args()
args_directory = args.directory
args_filename = args.filename


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
        return MDMDLSTMModel(
            next(iter(train_loader))[0][0].shape[0],
            model_params
        )
    elif model_params['model'] == 'pyramid_lstm':
        pass  # todo
    else:
        raise NameError(
            model_params['model'] + ' is not defined.'
            + ' Choose one of the following: bidi_lstm, renet, '
            + 'conv_net, md_lstm or pyramid_lstm.'
        )


def main(_args_directory, _args_filename):
    state = torch.load(_args_directory + "/" + _args_filename)
    model_params = state['model_params']
    train_params = state['train_params']
    dataset_params = state['dataset_params']
    model_state = state['model_state']
    optimizer_state = state['optimizer_state']
    epoch = state['epoch']

    # Load iterable datasets
    train_loader = get_dataset(
        dataset_params['train_filename'],
        train_params['batch_size']
    )
    validation_loader = get_dataset(
        dataset_params['validation_filename'],
        train_params['batch_size'],
        shuffle=False
    )
    test_loader = get_dataset(
        dataset_params['test_filename'],
        train_params['batch_size'],
        shuffle=False
    )

    model = get_model(model_params, train_loader).to(device)
    model.load_state_dict(model_state)

    # Train the model
    trainer = Trainer(
        model,
        model_params,
        train_params,
        dataset_params,
        train_loader,
        validation_loader,
        test_loader,
        _args_directory,
        _args_filename,
        current_epoch=epoch
    )
    trainer.optimizer.load_state_dict(optimizer_state)
    print("Welcome back. Continuing Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_progress(_args_directory, _args_filename)


if __name__ == "__main__":
    main(args_directory, args_filename)
