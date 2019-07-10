#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import torch
import torch.nn.functional as F
import numpy as np
from math import log10

# Device configuration (DO NOT EDIT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def classification_accuracy(outputs, target, class_dim, top_k=1):
    """
    Returns an Top-K accuracy metric for a given input and output. Dimension
    for the classes have to be defined. E.g. (Batch-Size, Class) results in
    class_dim=1.

    :param outputs: The predicted output
    :param target: The desired output
    :param class_dim: The dimension of the classes in the output
    :param top_k: Integer value K for Top-K accuracy. Default: 1
    :return: Top-K accuracy between outputs and target (predicted and desired)
    """
    predicted = outputs.topk(k=top_k, dim=class_dim, largest=True, sorted=True)
    predicted = predicted[1]
    target = target.view(-1, 1).expand_as(predicted)

    correct = torch.eq(predicted, target).view(-1)
    sum_correct = torch.sum(correct).item()
    return sum_correct


def classification_accuracy_top_1_and_5(outputs, target, class_dim):
    """
    Returns an Top-1 and 5 accuracy metric for a given input and output.
    Combination of
    classification_accuracy(..., top_k=1) and
    classification_accuracy(..., top_k=5).

    :param outputs: The predicted output
    :param target: The desired output
    :param class_dim: The dimension of the classes in the output
    :return: Top-1 and 5 accuracy between outputs and target (predicted and desired)
    """
    return np.array([classification_accuracy(outputs, target, class_dim, top_k=1),
                    classification_accuracy(outputs, target, class_dim, top_k=5)])


def classification_accuracy_2d(outputs, target, class_dim):
    """
    todo
    :param outputs:
    :param target:
    :param class_dim:
    :return:
    """
    predicted = torch.max(outputs.data, class_dim)[1]
    correct = torch.eq(predicted, target).view(-1)
    sum_correct = torch.sum(correct).item()/(target.shape[-1]*target.shape[-2])
    return sum_correct


def classification_mse(outputs, target):
    """
    Use MSE as "accuracy" for the custom Trainer in this project.
    Useful, if you want to train on BCE (on images) and track MSE parallel as
    "accuracy"

    :param outputs: The predicted output
    :param target: The desired output
    :return: MSE loss as "accuracy"
    """
    mse = torch.nn.MSELoss(reduction='sum')(outputs, target).item()/100
    return mse


def psnr(outputs, target):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR),
    e.g. used for Super-Resolution

    :param outputs: The predicted output
    :param target: The desired output
    :return: PSNR
    """
    mse = torch.nn.MSELoss(reduction='sum')(outputs, target)
    psnr = 10 * log10(1 / mse.item())

    return psnr


def get_accuracy_metric(params):
    """
    Returns the accuracy metric given by a string and extracts the corresponding parameters from the params parameter.

    :param params: Training parameters
    :return: Accuracy metric and parameters for the metric extracted from the training parameters.
    """
    if params['acc_metric'] == "classification":
        kwargs = {'class_dim': params['class_dim']}
        if 'top_k' in params.keys():
            kwargs.update({'top_k': params['top_k']})
        return classification_accuracy, kwargs
    elif params['acc_metric'] == "classification_1_and_5":
        kwargs = {'class_dim': params['class_dim']}
        return classification_accuracy_top_1_and_5, kwargs
    elif params['acc_metric'] == "classification_2d":
        kwargs = {'class_dim': params['class_dim']}
        return classification_accuracy_2d, kwargs
    elif params['acc_metric'] == "MSELoss":
        return classification_mse, {}
    elif params['acc_metric'] == "PSNR":
        return psnr, {}


class CustomLoss(object):
    """
    Class with custom loss definitions.
    """
    @staticmethod
    def binary_cross_entropy_2d(_input, _target):
        """
        todo
        :param _input:
        :param _target:
        :return:
        """
        return F.binary_cross_entropy(_input, _target, reduction='sum')

    @staticmethod
    def cross_entropy2d(_input, _target, weight=None):
        """
        todo
        :param _input:
        :param _target:
        :param weight:
        :return:
        """
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = _input.size()
        # log_p: (n, c, h, w)
        # if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        #    log_p = F.log_softmax(input)
        # else:
        # >=0.3
        log_p = F.log_softmax(_input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[_target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = _target >= 0
        target = _target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
        return loss
