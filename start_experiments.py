#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import sys
import os
import logging

sys.path.insert(0, os.path.abspath('..'))   # noqa

from sacred.observers import FileStorageObserver
from lucky_trainer.experiment import ex

"""
parser = argparse.ArgumentParser(description='Recurrent CV Experiments Starter')
parser.add_argument('--model', required=True,
                    help='Random search setting like <random_search_renet>')
parser.add_argument('--dataset', required=True,
                    help='Root folder of sacred runs of a experiment')
args = parser.parse_args()
args_model = args.model
args_dataset = args.dataset
"""


def main(model, dataset):
    location = '../data/tiny_ImageNet/output_data'
    ex.observers.append(FileStorageObserver.create(location))

    logger = logging.getLogger('mylogger')
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('WARNING')
    ex.logger = logger

    for each_experiment in range(1):
        ex.run(named_configs=[model, dataset])



"""
if __name__ == "__main__":
    main(args_model, args_dataset)
"""

main("random_search_renet_tiny_imagenet", "renet_dataset_tiny_imagenet")
