import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import cv2


class SegmentationDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self,
                 root_dir,
                 image_folder,
                 mask_folder,
                 transform=None,
                 image_colormode='rgb',
                 mask_colormode='grayscale',
                 seed=42,
                 fraction=0.7,
                 subset=None):
        """
        Args:
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
            if subset == 'training':
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
            image = cv2.imread(img_name)
            image = cv2.copyMakeBorder(image, top=4, bottom=4, left=6, right=5,
                                       borderType=cv2.BORDER_CONSTANT)

        else:
            image = cv2.imread(img_name)
            image = cv2.copyMakeBorder(image, top=4, bottom=4, left=6, right=5,
                                       borderType=cv2.BORDER_CONSTANT)

        msk_name = self.mask_names[idx]
        if self.maskcolorflag:
            mask = np.array(Image.open(msk_name)).transpose(2, 0, 1)
            mask = cv2.copyMakeBorder(mask, top=4, bottom=4, left=6, right=5,
                                      borderType=cv2.BORDER_CONSTANT)
        else:
            mask = np.array(Image.open(msk_name))
            mask = cv2.copyMakeBorder(mask, top=4, bottom=4, left=6, right=5,
                                      borderType=cv2.BORDER_CONSTANT)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
