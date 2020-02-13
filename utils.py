from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import glob
from dataset import SegmentationDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, image_resize, mask_resize):
        self.image_resize = image_resize
        self.mask_resize = mask_resize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, self.mask_resize, cv2.INTER_AREA)
        image = cv2.resize(image, self.image_resize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, mask_resize=None, image_resize=None):
        image, mask = sample['image'], sample['mask']
        image = image.transpose(2, 0, 1)

        if len(mask.shape) == 2:
            mask = mask.reshape((1,) + mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,) + image.shape)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class Normalize(object):
    """Normalize image"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor) / 255,
                'mask': mask.type(torch.FloatTensor) / 255}


class HorizontalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return {'image': image,
                'mask': mask}


class ApplyClaheColor(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        return {'image': img_output,
                'mask': mask}


class ApplyClahe(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        clahe = cv2.createCLAHE(clipLimit=2.0)
        image = clahe.apply(image)

        return {'image': image,
                'mask': mask}


class Color2Gray(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {'image': image,
                'mask': mask}


class Denoise(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = cv2.bilateralFilter(image, d=5, sigmaColor=9, sigmaSpace=9)
        return {'image': image,
                'mask': mask}


def get_data_loaders(data_dir, image_folder='training/images', mask_folder='training/1st_manual', batch_size=4):
    data_transforms = {
        # Resize((592, 576), (592, 576)),
        'training': transforms.Compose([HorizontalFlip(), ApplyClaheColor(), Denoise(), ToTensor(), Normalize()]),
        'test': transforms.Compose([HorizontalFlip(), ApplyClaheColor(), Denoise(), ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegmentationDataset(root_dir=data_dir,
                                             transform=data_transforms[x],
                                             image_folder=image_folder,
                                             mask_folder=mask_folder,
                                             subset=x)
                      for x in ['training', 'test']}

    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                    for x in ['training', 'test']}

    return data_loaders


def plot_batch_from_dataloader(dataloaders, batch_size):
    """

    :param dataloaders: dataset dataloaders
    :param batch_size: size of the batch to plot
    :return: void
    """
    batch = next(iter(dataloaders['training']))

    for i in range(batch_size):

        np_img = batch['image'][i].numpy()
        np_mask = batch['mask'][i].numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # color image
        if np_img.shape[1] == 3:
            ax[0].imshow(np.transpose(np_img, (1, 2, 0)))
        else:
            np_img = np_img.astype(float)
            np_img *= 255.0
            ax[0].imshow(np.squeeze(np_img), cmap=plt.get_cmap('gray'))
        ax[1].imshow(np.squeeze(np.transpose(np_mask, (1, 2, 0))), cmap='gray')
        plt.show()


def myimshow(img, unnormalize=False):
    """

    :param img: tensor of images, first dimension is number of images in the batch
    :param unnormalize: whenever to unnormalize the image before plotting
    :return: void
    """
    if unnormalize:
        img = img * 255

    np_img = img.numpy()
    plt.imshow(np.transpose(np_img[0], (1, 2, 0)))
    plt.show()


def images_generator(path):
    for img_name in glob.glob(os.path.join(path, '*')):
        image = np.array(Image.open(img_name)).transpose(2, 0, 1)
        yield image


def training_plots(train_losses, test_losses):
    train_plot, = plt.plot(range(len(train_losses)), train_losses, label='train loss')
    test_plot, = plt.plot(range(len(test_losses)), test_losses, label='test loss')
    plt.legend(handles=[train_plot, test_plot])
    plt.show()
