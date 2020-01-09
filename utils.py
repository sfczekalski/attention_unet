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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class SegmentationDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, 
                 root_dir, 
                 image_folder, 
                 mask_folder, 
                 transform=None, 
                 image_colormode='rgb', 
                 mask_colormode='grayscale',
                 seed=None,
                 fraction=None,
                 subset=None):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            image_folder (string) = 'Images' : Name of the folder which contains the Images.
            mask_folder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed: Specify a seed for the train and test split
            fraction: A float value from 0 to 1 which specifies the validation split fraction
            subset: 'Train' or 'Test' to select the appropriate set.
            image_colormode: 'rgb' or 'grayscale'
            mask_colormode: 'rgb' or 'grayscale'
        """
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert (image_colormode in ['rgb', 'grayscale'])
        assert (mask_colormode in ['rgb', 'grayscale'])

        self.imagecolorflag = self.color_dict[image_colormode]
        self.maskcolorflag = self.color_dict[mask_colormode]
        self.root_dir = root_dir
        self.transform = transform

        if not fraction:
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, image_folder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, mask_folder, '*')))
        else:
            assert (subset in ['training', 'test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, image_folder, '*'))))
            self.mask_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, mask_folder, '*'))))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        if self.imagecolorflag:
            image = np.array(Image.open(img_name)).transpose(2, 0, 1)
        else:
            image = np.array(Image.open(img_name))
        msk_name = self.mask_names[idx]
        if self.maskcolorflag:
            mask = np.array(Image.open(msk_name)).transpose(2, 0, 1)
        else:
            mask = np.array(Image.open(msk_name))

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


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


def get_data_loader(data_dir, image_folder='images', mask_folder='mask', batch_size=1):
    data_transforms = {
        'training': transforms.Compose([ToTensor(), Normalize()]),
        'test': transforms.Compose([ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegmentationDataset(root_dir=os.path.join(data_dir, x),
                                             transform=data_transforms[x],
                                             image_folder=image_folder,
                                             mask_folder=mask_folder)
                      for x in ['training', 'test']}

    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                    for x in ['training', 'test']}

    return data_loaders


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
