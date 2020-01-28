#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

from copy import deepcopy
import torch


class EarlyStopper(object):
    """
    Custom Early Stopping Implementation.
    """

    def __init__(self, trainer, patience, min_epochs=0):
        """
        Initialization of the ValLogWithEarlyStopping class.

        :param min_epochs: Counter to make sure, that the model is training at
            least for min_epochs epochs.
        :param trainer: Train Engine of the PyTorch Ignite Framework
        :param patience: How many additional epochs should be checked before
            terminating.
        """
        self.counter = 0
        self.bestScore = None
        self.bestModelState = None
        self.bestOptimizerState = None
        self.correspondingTrainScore = None
        self.trainer = trainer
        self.patience = patience
        self.validation_loader = trainer.val_loader
        self.training_loader = trainer.train_loader
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.min_epochs = min_epochs
        self.reset()

    def get_results(self):
        """
        Returns a dictionary of the best (validation) results with the
        corresponding training results.

        :return: Dictionary of best Epoch, Validation Accuracy & Loss,
            Training Accuracy & Loss.
        """
        return {'epochs': self.bestScore[0],
                'validation_loss': self.bestScore[1],
                'train_loss': self.correspondingTrainScore}

    def get_state_dict(self):
        """
        Returns the parameter of the best validation run.

        :return: Parameters of the model.
        """
        return self.bestModelState, self.bestOptimizerState

    def reset(self):
        self.counter = 0
        self.bestScore = None
        self.bestModelState = None
        self.bestOptimizerState = None
        self.correspondingTrainScore = None
        self.trainer.early_stop_flag = False

    def update(self, epoch, train_loss, val_loss):
        """
        Calculates the accuracy and loss and checks if the results are better.
        If not and this is the case for patience many steps, then it terminates
        the trainer (early stopping).

        :return: Prints the current Accuracy & Loss results.
        """
        if self.bestScore is None or self.min_epochs > epoch:
            # Saving the current best scores
            self.bestScore = (epoch, val_loss)
            self.correspondingTrainScore = train_loss
            if torch.cuda.is_available():
                self.bestModelState = deepcopy(self.model.cpu().state_dict())
                self.model.to(self.trainer.device)
            else:
                self.bestModelState = deepcopy(self.model.state_dict())
            self.bestOptimizerState = self.optimizer.state_dict()
            self.counter = 0
        elif self.bestScore[1] > val_loss:
            # Saving the current best scores
            self.bestScore = (epoch, val_loss)
            self.correspondingTrainScore = train_loss
            if torch.cuda.is_available():
                self.bestModelState = self.model.cpu().state_dict()
                self.model.to(self.trainer.device)
            else:
                self.bestModelState = self.model.state_dict()
            self.bestOptimizerState = self.optimizer.state_dict()
            self.counter = 0
        else:
            # NOT a better performance => just count up
            self.counter += 1

        # Counter reached the patience => Stop training
        if self.counter == self.patience:
            print("Early Stopping - Best Results: Epoch: {} | ".format(
                self.bestScore[0]) + "Avg val-loss: {:.5f}".format(
                self.bestScore[1]))
            self.trainer.early_stop_flag = True
