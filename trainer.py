#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import datetime
import torch
import sys
import os
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy

from metrics import get_accuracy_metric, CustomLoss
from early_stopper import EarlyStopper


# train_one_epoch commented out
# save_progress has a return None
# calculate_loss_and_acc outputs=_target
# self.device changed to cuda(), also in early_stopper


class Trainer(object):
    """
    Class, which manages the training of the model. The main benefit of this
    class is that it extracts important information out of the train parameter
    dictionary to setup the EarlyStop Mechanism, the optimizer and so on.
    """

    def __init__(self, model, model_params, train_params, dataset_params,
                 train_loader, val_loader, test_loader,
                 output_directory, filename, current_epoch=0,
                 print_param_size=True, clip_grad_norm=True,
                 print_progress=True, checkpoint_patience=-1,
                 checkpoint_epoch_notation=True, save_whole_model=False,
                 use_cudnn_benchmark=True, optimizer=None):
        """
        Initialization of the Trainer.

        :param checkpoint_epoch_notation: Boolean, if the epoch should be appear
            on the saved checkpoints (it generates a new file).
        :param dataset_params: Parameters of the dataset (path + name).
        :param output_directory: Path to the saved file
        :param filename: Name of the saved file
        :param current_epoch: May be important if an experiment is continued.
        :param print_param_size: Boolean, prints the amount of parameters at
            the beginning of the training.
        :param clip_grad_norm: Boolean, applying grad norm since RNNs tend to
            have exploding gradients.
        :param checkpoint_patience: Int, if it's greater than 0, than it saves
            on every checkpoint_patience-th step.
        :param save_whole_model: Boolean, if you only want to save the model
            state or the whole model.
        :param print_progress: Boolean, if current loss and maybe accuracy
            should be printed after each iteration.
        :param model: The model, which has to be trained
        :param model_params: Model parameters for saving process.
        :param train_params: Train parameter dictionary with information like
            loss function, optimizer, amount of max. epochs etc.
        :param train_loader: Train dataset
        :param val_loader: Validation dataset
        :param test_loader: Test dataset
        """
        # Use CUDNN benchmark
        torch.backends.cudnn.benchmark = use_cudnn_benchmark

        # Make everything deterministic
        self.seed_everything()

        # Parameter size
        if print_param_size:
            self.print_param_size(model)

        # Create output folder for saving the model
        # and track date for saving name
        os.makedirs(output_directory, exist_ok=True)

        # Cosmetic settings like saving the whole model,
        # get start time, save name, output directory, ...
        self.start_time = self.get_start_time()
        self.save_whole_model = save_whole_model
        self.checkpoint_epoch_notation = checkpoint_epoch_notation
        self.print_progress = print_progress
        self.output_directory = os.path.abspath(output_directory)
        self.filename = filename

        # Get the parameters of the model, training and the dataset
        self.model_params, self.train_params = model_params, train_params
        self.dataset_params = dataset_params
        self.current_epoch = current_epoch
        self.max_epochs = train_params['max_epochs']

        # Track loss/accuracy
        self.history = ([], [])

        # Gradient clipping for RNNs
        self.clip_grad_norm = clip_grad_norm

        # Setting Checkpoints
        self.checkpoint_patience = checkpoint_patience

        # Send to model to GPU, if enabled
        #self.device = torch.device('cuda'
        #                           if torch.cuda.is_available()
        #                           else 'cpu')
        self.model = model.cuda()

        # Initialization of the model
        if 'init' in train_params.keys():
            for _init in train_params['init']:
                def set_init(m, init=None, nn_type=None):
                    if type(m) == nn_type:
                        getattr(torch.nn.init, init)(m.weight)

                self.model.apply(
                    lambda m: set_init(m, init=_init[0], nn_type=_init[1])
                )

        # Datasets
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Get Loss Object & Optimizer Class reference
        try:
            self.loss = getattr(torch.nn, train_params['loss'])(reduction='sum')
        except AttributeError:
            self.loss = getattr(CustomLoss, train_params['loss'])
        self.opt = getattr(torch.optim, train_params['optimizer'])
        # Creates optimizer with parameters given in params and a prefix "opt_"
        # E.g. "opt_lr": 0.0001 item will be the parameter lr=0.0001
        if optimizer is None:
            self.optimizer = self.init_optimizer()
        else:
            self.optimizer = optimizer

        # Early Stopping
        self.enable_early_stopping = False
        self.early_stop_flag = False
        if 'early_stopping_patience' in train_params.keys():
            self.enable_early_stopping = True
            min_epochs = 0
            if 'early_stopping_min_epochs' in train_params.keys():
                min_epochs = train_params['early_stopping_min_epochs']
            self.early_stopping = EarlyStopper(
                self, train_params['early_stopping_patience'], min_epochs
            )

        # If accuracy is enabled
        self.enable_accuracy_metric = False
        if 'acc_metric' in train_params.keys():
            self.enable_accuracy_metric = True
            try:
                self.acc_function, self.acc_params = get_accuracy_metric(
                    train_params
                )
            except TypeError:
                print("Accuracy metric <" + str(train_params['acc_metric'])
                      + "> is not defined. Disabling Accuracy measurement.")
                self.enable_accuracy_metric = False

        # Backpropagation through time
        self.enable_bppt = 'bppt_axis' in train_params.keys()
        self.bppt_axis = train_params['bppt_axis'] \
            if self.enable_bppt else None

        self.skip_test = train_params['skip_test'] \
            if 'skip_test' in train_params.keys() else False

        self.k_fold = train_params['k_fold'] \
            if 'k_fold' in train_params.keys() else 0

        self.enable_aux_training = train_params['aux_training'] \
            if 'aux_training' in train_params.keys() else False

        # If learn-rate decay is enabled
        self.lr_decay = False
        self.scheduler, self.lr_decay_step_size = None, None
        self.lr_decay_steps = None
        self.setup_lr_decay()

    def train(self):
        # Normal training is like training for one fold
        if self.k_fold == 0:
            return self.train_one_fold(
                self.train_loader,
                self.val_loader,
                self.test_loader
            )
        else:
            if type(self.k_fold) is int:
                ks_list = range(self.k_fold)
                fold_amount = self.k_fold
            else:
                ks_list = self.k_fold[1]
                fold_amount = self.k_fold[0]

            # Get the points to divide the dataset into folds
            fold_size = len(self.train_loader.dataset) // fold_amount
            split_size = [fold_size
                          for _ in range(fold_amount - 1)]
            split_size.append(
                len(self.train_loader.dataset) - (fold_amount - 1) * fold_size
            )
            folds = torch.utils.data.random_split(
                self.train_loader.dataset,
                lengths=split_size
            )

            # Save some configs for k-fold
            model_backup = deepcopy(self.model)
            original_filename = self.filename
            self.skip_test = True
            result = np.zeros(4 if self.enable_accuracy_metric else 2)

            # Apply K-Fold
            for i, k in enumerate(ks_list):
                # Config filename corresponding to
                # the current k-cross-validation
                self.filename = original_filename + "_K" + str(k+1) + "_"

                # Reset values for training
                if i > 0:
                    self.model = deepcopy(model_backup)
                    self.early_stopping.reset()
                    self.setup_lr_decay()
                    self.optimizer = self.init_optimizer()
                    self.current_epoch = 0

                # Assign train and test/val folds
                train_set = torch.utils.data.ConcatDataset(
                    folds[:k] + folds[k + 1:]
                )
                val_set = folds[k]

                # Make a DataLoader out of them
                train_set = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=self.train_loader.batch_size,
                    shuffle=True,
                    num_workers=self.train_loader.num_workers
                )
                val_set = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=self.train_loader.batch_size,
                    shuffle=False,
                    num_workers=self.train_loader.num_workers
                )

                # Apply one fold
                result += np.array(self.train_one_fold(
                    train_set,
                    val_set,
                    None
                ))
                print("Fold", str(k), "completed.")

            # Calculate overall performance of K-Fold
            result /= len(ks_list)
            print("Average Performance of "
                  + str(self.k_fold)
                  + "-Fold Cross-Validation:"
                  + str(result))

            # Releases all unoccupied cached memory currently held by
            # the caching allocator so that those can be used in other
            # GPU application and visible in nvidia-smi
            torch.cuda.empty_cache()

    def train_one_fold(self, train_set, val_set, test_set):
        """
        Trains the model given by the params of the initialization.

        :return: Returns nothing, but the model is now trained.
        """
        train_result, val_result = None, None
        epoch, checkpoint_counter = 0, 0
        for epoch in range(self.current_epoch, self.max_epochs):
            # Check if Early Stop is activated
            if self.early_stop_flag:
                break

            # Learn-rate counter
            if self.lr_decay:
                if self.lr_decay_steps is not None:
                    if len(self.lr_decay_steps) > 0:
                        if epoch == self.lr_decay_steps[0][0]:
                            for params in self.optimizer.param_groups:
                                params['lr'] = self.lr_decay_steps[0][1]
                            del self.lr_decay_steps[0]
                elif (type(self.lr_decay_step_size) is int
                        or epoch in self.lr_decay_step_size):
                    self.scheduler.step()

            # Train one epoch
            train_result, val_result = [[0, 1], [1, 2]]#self.train_one_epoch(train_set, val_set)

            # Print result of one epoch
            self.print_result('train', train_result, epoch)
            self.print_result('val', val_result, epoch)

            # Update Early Stopping Class, if enabled
            if self.enable_early_stopping:
                self.early_stopping.update(
                    epoch,
                    train_result[0],
                    val_result[0]
                )

            # Checkpoint Counter for saving the model during training
            if self.checkpoint_patience > -1:
                if checkpoint_counter == self.checkpoint_patience:
                    checkpoint_counter = 0

                    # For saving with epoch in filename
                    overhead = "_epoch_" + str(epoch) \
                        if self.checkpoint_epoch_notation else ""

                    # For saving with accuracy
                    if self.enable_accuracy_metric:
                        save_name = str(val_result[1]) + "_val_acc"
                    else:
                        save_name = str(val_result[0]) + "_val_loss"

                    # Save a checkpoint
                    self.save_progress(self.output_directory,
                                       self.filename + overhead,
                                       save_name)
                else:
                    checkpoint_counter += 1

        # Test Phase
        if self.enable_early_stopping:
            # Load the best working model
            self.model.load_state_dict(self.early_stopping.bestModelState)
        self.model.eval()

        # Calculate and print performance in the end
        if not self.skip_test:
            test_result = self.calculate_loss_and_acc(test_set)
        else:
            test_result = self.calculate_loss_and_acc(val_set)
        self.print_result('test', test_result, epoch)

        # Save the last model with its performance
        if self.enable_accuracy_metric:
            self.save_progress(
                self.output_directory, self.filename, test_result[1]
            )
            return train_result[0], train_result[1], test_result[0], test_result[1]
        else:
            self.save_progress(
                self.output_directory, self.filename, test_result[0]
            )
            return train_result[0], test_result[0]

    def calculate_loss_and_acc(self, loader):
        """
        Calculates the loss and accuracy of the model applied on a given
        dataset (Training or Validation).

        :param loader: The (Training or Validation) dataset
        :return: Loss and Accuracy of the model of the current epoch.
        """
        with torch.no_grad():
            loss_total = 0
            total = 0
            sum_correct = 0
            for _input, _target in loader:
                _target = _target.cuda()
                _input = _input.cuda()

                outputs = self.model(_input)
                if self.enable_aux_training:
                    outputs = outputs[0]
                total += _target.size(0)

                # Count for accuracy, if enabled
                if self.enable_accuracy_metric:
                    sum_correct += self.acc_function(
                        outputs,
                        _target,
                        **self.acc_params
                    )
                loss_total += self.loss(outputs, _target).item()
        loss = loss_total / total
        acc = None
        if self.enable_accuracy_metric:
            acc = 100 * sum_correct / total
        return loss, acc

    def train_one_epoch(self, train_set, val_set):
        # Train Phase
        self.model.train()
        loss_total, total, sum_correct = 0, 0, 0
        pbar = tqdm(train_set, leave=False,
                    file=sys.stdout, ascii=True)
        for _input, _target in pbar:
            # Load input and target to device (like GPU)
            _input = _input.to(self.device)
            _target = _target.to(self.device)

            # Count amount of images
            total += _target.size(0)

            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # Calculate output
            outputs = self.model(_input)

            # Calculate Loss
            # Backpropagation through time
            if self.enable_bppt:
                loss = 0
                for timestep in range(_target.shape[self.bppt_axis]):
                    timestep_loss = self.loss(
                        outputs.select(dim=self.bppt_axis, index=timestep),
                        _target.select(dim=self.bppt_axis, index=timestep)
                    )
                    loss += timestep_loss
            # If auxiliary classifiers (multiple outputs)
            elif self.enable_aux_training:
                loss = 0
                for output in outputs:
                    loss += self.loss(output, _target)
                outputs = outputs[0]
            # Simple loss calculation
            else:
                loss = self.loss(outputs, _target)
            loss_total += loss.item()

            # Calculate accuracy on train
            if self.enable_accuracy_metric:
                sum_correct += self.acc_function(
                    outputs,
                    _target,
                    **self.acc_params
                )

            # Getting gradients w.r.t. parameters
            loss.backward()

            # RNNs tend to have exploding gradients
            # (see https://arxiv.org/pdf/1211.5063.pdf)
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            # Updating parameters
            self.optimizer.step()

            # tqdm with current loss & acc
            if self.print_progress:
                desc = "[loss:" + str(loss_total / total)
                if self.enable_accuracy_metric:
                    desc += ", acc:" + str(100 * sum_correct / total)
                desc += "]"
                pbar.set_description(desc)

        # End of train, calculating validation and increase epoch
        self.current_epoch += 1
        self.model.eval()

        # Calculate Avg. Loss & Avg. Accuracy for train
        train_loss = loss_total / total
        train_acc = None
        if self.enable_accuracy_metric:
            train_acc = 100 * sum_correct / total

        # Calculate Avg. Loss & Avg. Accuracy for val
        val_loss, val_acc = self.calculate_loss_and_acc(val_set)

        # Save losses and accuracies
        self.history[0].append((train_loss, train_acc))
        self.history[1].append((val_loss, val_acc))

        return [train_loss, train_acc], [val_loss, val_acc]

    def get_state_dict(self):
        """
        Returns the state dict. of the model.

        :return: State dict. of the model
        """
        if self.enable_early_stopping:
            return self.early_stopping.get_state_dict()
        else:
            if torch.cuda.is_available():
                model_state = self.model.cpu().state_dict()
                self.model.to(self.device)
                return model_state, self.optimizer.state_dict()
            else:
                return self.model.state_dict(), self.optimizer.state_dict()

    def save_progress(self, output_directory, file_name, performance):
        """
        Saving function. It splits up in two parts: whole model or not (saving
        just the weights of the model).

        :param output_directory: Output directory (path to it)
        :param file_name: Name of the saving.
        :param performance: It will be appended on the name. Accuracy or loss,
            if accuracy is not given.
        """
        # Loading the results and the states of the training (model and
        # optimizer state).

        return None

        model_state, optimizer_state = self.get_state_dict()
        if self.enable_early_stopping:
            epoch = self.early_stopping.get_results()['epochs']
        else:
            epoch = self.current_epoch

        # Saving the whole model.
        if self.save_whole_model:
            save_dict = {
                'model': self.model.cpu(),
                'train_params': self.train_params,
                'dataset_params': self.dataset_params,
                'history': self.history,
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'epoch': epoch + 1
            }
            file_name += "_wholeModel_"
            self.model.to(self.device)
        # Saving just the weights and the params to build the model.
        else:
            save_dict = {
                'model_params': self.model_params,
                'train_params': self.train_params,
                'dataset_params': self.dataset_params,
                'history': self.history,
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'epoch': epoch + 1
            }

        # Actual saving process
        print("\nSaving Model in "
              + output_directory + '/'
              + file_name + " ...")
        torch.save(save_dict, output_directory + '/' + file_name
                   + "_" + str(performance)
                   + "_" + self.start_time
                   + ".pth")
        print("Saved.")

    def init_optimizer(self):
        return self.opt(
            self.model.parameters(),
            **dict((key[4:], value)
                   for key, value in self.train_params.items()
                   if key[:4] == "opt_")
        )

    def setup_lr_decay(self):
        if 'lr_cosine_decay_min' in self.train_params.keys():
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                self.train_params['max_epochs'],
                eta_min=self.train_params['lr_cosine_decay_min'])
        elif 'lr_decay_patience' in self.train_params.keys() and \
                'lr_decay_gamma' in self.train_params.keys():
            threshold = 0.0001
            if 'lr_threshold' in self.train_params.keys():
                threshold = self.train_params['lr_threshold']
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.train_params['lr_decay_gamma'],
                mode='min',
                threshold=threshold,
                patience=self.train_params['lr_decay_patience']
            )
        elif 'lr_decay_gamma' in self.train_params.keys() and \
                'lr_decay_step_size' in self.train_params.keys():
            self.lr_decay = True
            if type(self.train_params['lr_decay_step_size']) is int:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    self.train_params['lr_decay_step_size'],
                    self.train_params['lr_decay_gamma']
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    1,
                    self.train_params['lr_decay_gamma']
                )
            self.lr_decay_step_size = self.train_params['lr_decay_step_size']
        elif 'lr_decay_steps' in self.train_params.keys():
            self.lr_decay = True
            self.lr_decay_steps = self.train_params['lr_decay_steps']

    @staticmethod
    def print_result(text_type, result, epoch):
        text_snippets = {
            'train': [
                'Training Results - Epoch: {} | ',
                'Avg train-loss: {:.5f}',
                ' | Avg train-accuracy: '
            ],
            'val': [
                'Validation Results - Epoch: {} | ',
                'Avg val-loss: {:.5f}',
                ' | Avg val-accuracy: '
            ],
            'test': [
                'Test Results - Epoch: {} | ',
                'Avg test-loss: {:.5f}',
                ' | Avg test-accuracy: '
            ]
        }
        (loss, acc) = result
        text_snippet = text_snippets[text_type]
        result = text_snippet[0].format(epoch)
        result += text_snippet[1].format(loss)
        if acc is not None:
            result += text_snippet[2] + str(acc)
        print(result)

    @staticmethod
    def seed_everything(seed=1337):
        """
        Makes the model nearly deterministic (it's not deterministic because
        of the cudnn.benchmark=True statement in the beginning of the script).

        :param seed: Seed number like for random. Default: 1337
        """
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def get_start_time():
        currentDT = datetime.datetime.now()
        return "{0}-{1}-{2}--{3}-{4}".format(
            str(currentDT.year), str(currentDT.month), str(currentDT.day),
            str(currentDT.hour), str(currentDT.minute))

    @staticmethod
    def print_param_size(model):
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters()
        )
        params = sum([
            np.prod(p.size()) for p in model_parameters
        ])
        print("Param-Size:", params)
