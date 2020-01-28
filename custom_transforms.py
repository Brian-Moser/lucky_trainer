#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import random
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageCms
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.decomposition import PCA


class rbg2lab(object):
    def __init__(self):
        pass

    def __call__(self, img):
        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")

        rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
        lab_im = ImageCms.applyTransform(img, rgb2lab_transform)

        return lab_im


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandomHorizontalOrVerticalFlip(object):
    """
    Applies a horizontal and vertical flip with given probabilities.
    """
    def __init__(self, p_h=0.25, p_v=0.25):
        """
        Constructor of the transform.

        :param p_h: Probability for a horizontal flip
        :param p_v: Probability for a vertical flip
        """
        self.p_h = p_h
        self.p_v = p_v
        self.h_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.v_flip = transforms.RandomVerticalFlip(p=1.0)

    def __call__(self, img):
        """
        Applies the transform to a given image.

        :param img: Input image
        :return: Flipped image
        """
        p = random.random()
        result = img
        if p < self.p_h:
            result = self.h_flip(result)
        p = random.random()
        if p < self.p_v:
            result = self.v_flip(result)
        return result


class ConvZCAOT(object):
    """
    todo
    """
    def __init__(self, train, patch_size=(9, 9)):
        self.patch_size = patch_size
        X = []
        toTensor = transforms.ToTensor()
        for _input, _ in train:
            X.append(toTensor(_input).permute(1, 2, 0).numpy())
        X = np.array(X)

        self.mean = (X.mean(axis=(0, 1, 2)))
        X = np.add(X, -self.mean)
        self.mean = torch.from_numpy(
            self.mean.reshape(1, self.mean.shape[0], 1, 1)
        )
        _, _, _, n_channels = X.shape

        # 1. Sample 10M random image patches (each with 3 colors)
        patches = PatchExtractor(patch_size=self.patch_size,
                                 max_patches=int(2.5e2)).transform(X)

        # 2. Perform PCA on these to get eigenvectors V and eigenvalues D.
        pca = PCA()
        pca.fit(patches.reshape(patches.shape[0], -1))

        dim = (-1,) + self.patch_size + (n_channels,)
        eigenvectors = torch.from_numpy(
            pca.components_.reshape(dim).transpose(0, 3, 1, 2).astype(
                X.dtype)
        )
        eigenvalues = torch.from_numpy(
            np.diag(1. / np.sqrt(pca.explained_variance_))
        )
        # 4. Construct the whitening kernel k:
        # for each pair of colors (ci,cj),
        # set k[j,i, :, :] = V[:, j, x0, y0]^T * D^{-1/2} * V[:, i, :, :]
        # where (x0, y0) is the center pixel location
        # (e.g. (5,5) for a 9x9 kernel)
        x_0 = int(np.floor(self.patch_size[0] / 2))
        y_0 = int(np.floor(self.patch_size[1] / 2))
        filter_shape = (n_channels,
                        n_channels,
                        self.patch_size[0],
                        self.patch_size[1])
        self.kernel = torch.zeros(filter_shape)
        eigenvectorsT = eigenvectors.permute(2, 3, 1, 0)
        # build the kernel
        for i in range(n_channels):
            for j in range(n_channels):
                a = torch.mm(
                    eigenvectorsT[x_0, y_0, j, :].contiguous().view(1, -1),
                    eigenvalues.float()
                )
                b = eigenvectors[:, i, :, :].contiguous().view(
                    -1, self.patch_size[0] * self.patch_size[1]
                )
                c = torch.mm(a, b).contiguous().view(self.patch_size[0],
                                                     self.patch_size[1])
                self.kernel[j, i, :, :] = c
        self.padding = (self.patch_size[0] - 1), (self.patch_size[1] - 1)

    def __call__(self, _input):
        input_tensor = _input.contiguous().view(
            1, _input.shape[0], _input.shape[1], _input.shape[2]
        ) - self.mean

        self.conv_whitening = torch.nn.functional.conv2d(
            input=input_tensor,
            weight=self.kernel,
            padding=self.padding
        )
        s_crop = [(self.patch_size[0] - 1) // 2, (self.patch_size[1] - 1) // 2]
        conv_whitening = self.conv_whitening[
                         :, :, s_crop[0]:-s_crop[0], s_crop[1]:-s_crop[1]
                         ]

        return conv_whitening.view(conv_whitening.shape[1],
                                   conv_whitening.shape[2],
                                   conv_whitening.shape[3])


class ConvZCA(object):
    """
    todo
    """
    def __init__(self, patch_size=(3, 3)):
        self.patch_size = patch_size

    def __call__(self, _input):
        _input = _input.permute(1, 2, 0).numpy()
        _input = _input.reshape(1,
                                _input.shape[0],
                                _input.shape[1],
                                _input.shape[2])

        mean = (_input.mean(axis=(0, 1, 2)))
        _input = np.add(_input, -mean)
        _, _, _, n_channels = _input.shape

        # 1. Sample 10M random image patches (each with 3 colors)
        patches = PatchExtractor(patch_size=self.patch_size).transform(_input)
        # 2. Perform PCA on these to get eigenvectors V and eigenvalues D.
        pca = PCA()
        pca.fit(patches.reshape(patches.shape[0], -1))

        dim = (-1,) + self.patch_size + (n_channels,)
        eigenvectors = torch.from_numpy(
            pca.components_.reshape(dim).transpose(0, 3, 1, 2).astype(
                _input.dtype)
        )
        eigenvalues = torch.from_numpy(
            np.diag(1. / np.sqrt(pca.explained_variance_))
        )
        # 4. Construct the whitening kernel k:
        # for each pair of colors (ci,cj),
        # set k[j,i, :, :] = V[:, j, x0, y0]^T * D^{-1/2} * V[:, i, :, :]
        # where (x0, y0) is the center pixel location
        # (e.g. (5,5) for a 9x9 kernel)
        x_0 = int(np.floor(self.patch_size[0] / 2))
        y_0 = int(np.floor(self.patch_size[1] / 2))
        filter_shape = (n_channels,
                        n_channels,
                        self.patch_size[0],
                        self.patch_size[1])
        kernel = torch.zeros(filter_shape)
        eigenvectorsT = eigenvectors.permute(2, 3, 1, 0)
        # build the kernel
        for i in range(n_channels):
            for j in range(n_channels):
                a = torch.mm(
                    eigenvectorsT[x_0, y_0, j, :].contiguous().view(1, -1),
                    eigenvalues.float()
                )
                b = eigenvectors[:, i, :, :].contiguous().view(
                    -1, self.patch_size[0] * self.patch_size[1]
                )
                c = torch.mm(a, b).contiguous().view(self.patch_size[0],
                                                     self.patch_size[1])
                kernel[j, i, :, :] = c
        padding = (self.patch_size[0] - 1), (self.patch_size[1] - 1)
        input_tensor = torch.from_numpy(_input).permute(0, 3, 1, 2)
        conv_whitening = torch.nn.functional.conv2d(
            input=input_tensor,
            weight=kernel,
            padding=padding
        )
        s_crop = [(self.patch_size[0] - 1) // 2, (self.patch_size[1] - 1) // 2]
        conv_whitening = conv_whitening[
                         :, :, s_crop[0]:-s_crop[0], s_crop[1]:-s_crop[1]
                         ]

        return conv_whitening.view(conv_whitening.shape[1],
                                   conv_whitening.shape[2],
                                   conv_whitening.shape[3])


class RandomHorizontalOrVerticalShift(object):
    """
    todo
    """
    def __init__(self, p_shift, shift_values):
        self.p_shift = p_shift
        self.shift_values = shift_values

    def __call__(self, tensor):
        output = tensor

        # horizontal shift
        p_h = random.random()
        if p_h < self.p_shift['r_shift']:
            zero_mat = torch.zeros(
                output.shape[0],
                output.shape[1],
                self.shift_values['r_shift']
            )
            output = torch.cat(
                (zero_mat, tensor),
                2)[:, :, :-self.shift_values['r_shift']]
        elif p_h < self.p_shift['r_shift'] + self.p_shift['l_shift']:
            zero_mat = torch.zeros(
                output.shape[0],
                output.shape[1],
                self.shift_values['l_shift']
            )
            output = torch.cat(
                (tensor, zero_mat),
                2)[:, :, self.shift_values['l_shift']:]

        # vertical shift
        p_v = random.random()
        if p_v < self.p_shift['b_shift']:
            zero_mat = torch.zeros(
                output.shape[0],
                self.shift_values['b_shift'],
                output.shape[2])
            output = torch.cat(
                (zero_mat, tensor),
                1)[:, :-self.shift_values['b_shift'], :]
        elif p_v < self.p_shift['b_shift'] + self.p_shift['t_shift']:
            zero_mat = torch.zeros(
                output.shape[0],
                self.shift_values['t_shift'],
                output.shape[2])
            output = torch.cat(
                (tensor, zero_mat),
                1)[:, self.shift_values['t_shift']:, :]

        return output
