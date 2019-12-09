#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

from torch.utils.data import Dataset
import numpy as np
import random
import torch
import math
from PIL import Image
from PIL import ImageOps
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class CustomDatasetWithTransform(Dataset):
    """
    Custom Dataset that gets a PyTorch datasets, loads items from there and
    applies custom transforms on them.
    """
    def __init__(self, dataset, transform=None):
        """
        Constructor for the Custom Dataset with custom transform

        :param dataset: PyTorch dataset
        :param transform: Composition of transforms. If it stays None, than the
        whole class operates like a standard PyTorch dataset
        """
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Loads the data from the original dataset and applies the given transform
        (given in constructor) on them. If not set, then it operates like the
        __getitem__ function of a standard PyTorch dataset

        :param index: Index of the data
        :return: Input and output data from the dataset
        """
        if self.transform is not None:
            return self.transform(self.dataset[index][0]), \
                   self.dataset[index][1]
        return self.dataset[index]

    def __len__(self):
        """
        Standard length-getter function

        :return: Length of the dataset.
        """
        return len(self.dataset)


class NumpyDataset(Dataset):
    """
    Custom PyTorch Dataset Class for the experiments. It fulfills the minimal
    implementation you can have.
    """
    def __init__(self, data, target, labels=None):
        """
        Constructor for saving a list of data and targets and maybe a list of
        labels (e.g. for overlapping MNIST, where input and targets are images
        and labels is listing the labels of the images).

        :param data: Input data of the dataset
        :param target: Target data of the dataset
        :param labels: Optional label list, which can be accessed by getLabel().
        """
        if isinstance(data[0], np.ndarray):
            self.data = []
            for _input in data:
                self.data.append(torch.from_numpy(_input))
        else:
            self.data = data
        self.target = target
        self.labels = labels

    def __getitem__(self, index):
        """
        Getter-function for getting the data and labels by index.

        :param index: Index of the data
        :return: Returns the data and the labels accordingly
        """
        return self.data[index], self.target[index]

    def __len__(self):
        """
        Standard length-getter function

        :return: Length of the dataset.
        """
        return len(self.data)

    def get_label(self, index):
        """
        Optional: like for Overlapping MNIST, where input and targets are images
        and for further analysis, the labels are saved.

        :param index: The index of the labels
        :return: Labels at a given index
        """
        return self.labels[index]


class ImageAEDataset(Dataset):
    """
    Auto-Encoder Dataset, where input and output images are the same. Note that
    the transforms (like normalization with subtracting mean and dividing by
    std) may be input only.
    """
    def __init__(self, img_path, transform=None):
        """
        Constructor for the Auto-Encoder Dataset

        :param img_path: List of paths to the .jpg-images
        :param transform: Composition of transforms for the input data
        """
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        """
        Loads the images and returns the image with input transform and a simple
        output transform (resizing and toTensor).

        :param index: The index of the item
        :return: Input and output images (essentially the same)
        """
        transform_target = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.ToTensor()])
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.transform is not None:
            return self.transform(img), transform_target(img)
        else:
            return img, img

    def __len__(self):
        """
        Standard length-getter function

        :return: Length of the dataset.
        """
        return len(self.img_path)


class ImageHTPDataset(Dataset):
    """
    todo
    """
    def __init__(self, img_path_input, img_path_target, pad_val=2560,
                 transform_input=None, transform_target=None):
        """
        todo

        :param img_path_input:
        :param img_path_target:
        :param transform_input:
        :param transform_target:
        """
        self.img_path_input = img_path_input
        self.img_path_target = img_path_target
        self.input_transform = transform_input
        self.target_transform = transform_target
        self.pad_val = pad_val

    def __getitem__(self, index):
        """
        todo

        :param index:
        :return:
        """
        twoPotList = [1280, 1440, 1600, 1760, 1920, 2080, 2240, 2400, 2560]
        input_img = Image.open(self.img_path_input[index][0]).convert('L')
        target_img = Image.open(self.img_path_target[index][0]).convert('L')
        width = min(range(len(twoPotList)), key=lambda i:
                    abs(twoPotList[i]-max(
                        input_img.size[0], target_img.size[0])
                        )
                    )
        if twoPotList[width] - max(input_img.size[0], target_img.size[0]) < 0:
            width += 1

        pad = (0, 8, twoPotList[width] - input_img.size[0], 8)
        input_img = ImageOps.expand(input_img, pad)
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        pad = (0, 8, twoPotList[width] - target_img.size[0], 8)
        target_img = ImageOps.expand(target_img, pad)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return input_img, target_img

    def __len__(self):
        """
        Standard length-getter function

        :return: Length of the dataset.
        """
        return len(self.img_path_input)


class ImageAugHTPFixedDataset(Dataset):
    """
    todo
    """
    def __init__(self, img_path_input, img_path_target,
                 transform_input=None, transform_target=None,
                 training=False):
        """
        todo

        :param img_path_input:
        :param img_path_target:
        :param transform_input:
        :param transform_target:
        """
        self.img_path_input = []
        self.img_path_target = []
        self.input_transform = transform_input
        self.target_transform = transform_target
        self.training = training

        for i in range(len(img_path_input)):
            input_img = Image.open(img_path_input[i][0]).convert('L')
            target_img = Image.open(img_path_target[i][0]).convert('L')
            if input_img.size[0] >= 250 and target_img.size[0] >= 250:
                if input_img.size[0] <= 640 and target_img.size[0] <= 640:
                    self.img_path_input.append(img_path_input[i])
                    self.img_path_target.append(img_path_target[i])

    def __getitem__(self, index):
        """
        todo

        :param index:
        :return:
        """
        max_width = 1600
        input_img = Image.open(self.img_path_input[index][0]).convert('L')
        target_img = Image.open(self.img_path_target[index][0]).convert('L')
        if self.training:
            # h_flip = transforms.RandomHorizontalFlip(p=1.0)
            # v_flip = transforms.RandomVerticalFlip(p=1.0)
            # p = random.random()
            # if p < 0.25:
            #     input_img = h_flip(input_img)
            #     target_img = h_flip(target_img)
            # p = random.random()
            # if p < 0.25:
            #     input_img = v_flip(input_img)
            #     target_img = v_flip(target_img)
            p = random.random()
            if p < 0.5:
                input_img = F.adjust_brightness(input_img, random.randint(0, 2))
            p = random.random()
            if p < 0.5:
                input_img = F.adjust_contrast(input_img, random.randint(0, 2))
            p = random()
            if p < 0.5:
                input_img = transforms.RandomRotation((-5, 5))(input_img)
        padding = max_width - input_img.size[0]
        left_padding = random.randint(0, padding)
        right_padding = padding - left_padding
        pad = (0, 0, left_padding + right_padding, 0)
        input_img = ImageOps.expand(input_img, pad)
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        if self.training:
            input_img = F.dropout2d(input_img, p=0.25)
        left_side = math.ceil((max_width - target_img.size[0])/2)
        pad = (0, 0, int(left_side) + int((max_width - target_img.size[0])/2), 0)
        target_img = ImageOps.expand(target_img, pad)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return input_img, target_img

    def __len__(self):
        """
        Standard length-getter function

        :return: Length of the dataset.
        """
        return len(self.img_path_input)


class ImageHTPFixedDataset(Dataset):
    """
    todo
    """
    def __init__(self, img_path_input, img_path_target,
                 transform_input=None, transform_target=None):
        """
        todo

        :param img_path_input:
        :param img_path_target:
        :param transform_input:
        :param transform_target:
        """
        self.img_path_input = []
        self.img_path_target = []
        self.input_transform = transform_input
        self.target_transform = transform_target

        for i in range(len(img_path_input)):
            input_img = Image.open(img_path_input[i][0]).convert('L')
            target_img = Image.open(img_path_target[i][0]).convert('L')
            if input_img.size[0] >= 640 and target_img.size[0] >= 640:
                if input_img.size[0] <= 1280 and target_img.size[0] <= 1280:
                    self.img_path_input.append(img_path_input[i])
                    self.img_path_target.append(img_path_target[i])

    def __getitem__(self, index):
        """
        todo

        :param index:
        :return:
        """
        max_width = 1280
        input_img = Image.open(self.img_path_input[index][0]).convert('L')
        target_img = Image.open(self.img_path_target[index][0]).convert('L')

        padding = max_width - input_img.size[0]
        left_padding = random.randint(0, padding)
        right_padding = padding - left_padding
        pad = (0, 0, left_padding + right_padding, 0)

        input_img = ImageOps.expand(input_img, pad)
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        left_side = math.ceil((max_width - target_img.size[0])/2)
        pad = (0, 0, int(left_side) + int((max_width - target_img.size[0])/2), 0)
        target_img = ImageOps.expand(target_img, pad)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return input_img, target_img

    def __len__(self):
        """
        Standard length-getter function

        :return: Length of the dataset.
        """
        return len(self.img_path_input)


class ImageHTHFixedDataset(Dataset):
    """
    todo
    """
    def __init__(self, img_path_input, img_path_target,
                 transform_input=None, transform_target=None):
        """
        todo

        :param img_path_input:
        :param img_path_target:
        :param transform_input:
        :param transform_target:
        """
        self.img_path_input = []
        self.img_path_target = []
        self.input_transform = transform_input
        self.target_transform = transform_target

        for i in range(len(img_path_input)):
            input_img = Image.open(img_path_input[i][0]).convert('L')
            target_img = Image.open(img_path_target[i][0]).convert('L')
            if input_img.size[0] >= 250 and target_img.size[0] >= 250:
                if input_img.size[0] <= 640 and target_img.size[0] <= 640:
                    self.img_path_input.append(img_path_input[i])
                    self.img_path_target.append(img_path_target[i])

    def __getitem__(self, index):
        """
        todo

        :param index:
        :return:
        """
        max_width = 640
        input_img = Image.open(self.img_path_input[index][0]).convert('L')
        target_img = Image.open(self.img_path_target[index][0]).convert('L')

        padding = max_width - input_img.size[0]
        left_padding = random.randint(0, padding)
        right_padding = padding - left_padding
        pad = (0, 0, left_padding + right_padding, 0)

        input_img = ImageOps.expand(input_img, pad)
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        left_side = math.ceil((max_width - target_img.size[0])/2)
        pad = (0, 0, int(left_side)+int((max_width - target_img.size[0])/2), 0)
        target_img = transforms.ToPILImage()(
            self.flip(transforms.ToTensor()(target_img), 2)
        )
        target_img = ImageOps.expand(target_img, pad)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return input_img, target_img

    def __len__(self):
        """
        Standard length-getter function

        :return: Length of the dataset.
        """
        return len(self.img_path_input)


class H5Dataset(Dataset):
    def __init__(self, in_file, transform):
        import h5py
        self.file = h5py.File(in_file, 'r')
        self.n_images, _, _, _ = self.file['images'].shape
        self.transform = transform

    def __getitem__(self, index):
        input = self.file['images'][index, :, :, :]
        print(np.array(self.file['labels'][index].item(0)))
        return self.transform(input.astype('float32')), np.array(self.file['labels'][index].item(0))

    def __len__(self):
        return self.n_images

@staticmethod
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
                 else x.new(
        torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]
