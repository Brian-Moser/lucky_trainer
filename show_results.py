#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import os
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Recurrent CV Experiments Vis.')
parser.add_argument('--root_folder', required=True,
                    help='Root folder of sacred runs of a experiment')
parser.add_argument('--flag', required=True,
                    help='Root folder of sacred runs of a experiment')
args = parser.parse_args()
args_root_folder = args.root_folder
args_flag = args.flag


def main(root_folder, flag):
    lst_folder = [x for x in os.listdir(root_folder) if x.isdigit()]
    lst_result = []

    for each_folder in tqdm(lst_folder):
        config_json = pd.read_json(
            os.path.join(root_folder, '{}/config.json'.format(each_folder)),
            typ='Series', orient='columns')
        run_json = pd.read_json(
            os.path.join(root_folder, '{}/run.json'.format(each_folder)),
            typ='Series', orient='columns')

        if run_json.status != 'COMPLETED' or \
                config_json.model_params['model'] != flag:
            continue

        model_params = {k: v for k, v in config_json.model_params.items()
                        if k not in ['input_dim']}

        train_params = {k: v for k, v in config_json.train_params.items()
                        if k not in ['input_dim']}

        setup_data = {'folder': each_folder}

        run_data = {k: v for k, v in run_json.result.items()
                    if 'test' in k or 'train' in k or 'val' in k}

        lst_result.append({k: v
                           for d in [model_params,
                                     setup_data,
                                     train_params,
                                     run_data]
                           for k, v in d.items()})

    if flag == 'renet':
        df = pd.DataFrame(
            lst_result,
            columns=['reNet_hidden_dim',
                     'rnn_types',
                     'window_size',
                     'dropout_rate',
                     'linear_hidden_dim',
                     'batch_size',
                     'optimizer',
                     'opt_lr',
                     'train_loss',
                     'train_accuracy',
                     'val_loss',
                     'val_accuracy',
                     'test_loss',
                     'test_accuracy',
                     'folder']
        )
    elif flag == 'bidi_lstm':
        df = pd.DataFrame(
            lst_result,
            columns=[
                'hidden_dim',
                'dropout_rate',
                'batch_size',
                'optimizer',
                'opt_lr',
                'train_loss',
                'train_accuracy',
                'validation_loss',
                'validation_accuracy',
                'folder']
        )
    else:
        raise ValueError('flat option is not available. \
        Support only renet/bidi_lstm')

    df.sort_values(axis=0, by=['test_accuracy', 'train_accuracy'],
                   ascending=False, inplace=True)
    df.to_html(os.path.join(root_folder, 'result.html'))


# Example call:
# python show_results.py --root_folder ../data/CIFAR10/output_data --flag renet
if __name__ == "__main__":
    main(args_root_folder, args_flag)
