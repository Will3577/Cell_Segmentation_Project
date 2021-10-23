# Code skeleton adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#                            https://github.com/cosmic-cortex/pytorch-UNet

import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import utils
from torchvision.transforms import functional as F

class TransformData:
    def __init__(self,crop=None):
        self.crop = crop
    
    def __call__(self,image,mask):
        image = F.to_pil_image(image)
        mask = F.to_pil_image(mask)
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        return image, mask

class CellDataset(Dataset):
    """ Cell segmentation dataset

        dataset_folder (train/val/test)
            |-- images (01)
                |-- t000.tif
                |-- t001.tif
                |-- ...
            |-- masks (01_GT)
                |-- SEG
                    |-- t000.tif
                    |-- t001.tif
                    |-- ...
    
    """

    def __init__(self, dataset_folder, transform=None):
        self.input_path = dataset_folder+"images/"
        self.mask_path = dataset_folder+"masks/SEG/"
        self.image_list = os.listdir(self.mask_path)

        if transform:
            self.transform = transform
        else:
            to_tensor = T.ToTensor()
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.mask_path))
    
    def __getitem__(self, idx):
        mask_name = self.image_list[idx]
        img_name = 't'+mask_name[-7:]
        image = cv2.imread(self.input_path+img_name)
        mask = cv2.imread(self.mask_path+mask_name, 0)
        # print(self.input_path+img_name, mask_name)
        print(image.shape,mask.shape)

        if self.transform:
            image, mask = self.transform(image, mask)

        # To tensor
        to_tensor = T.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        # To long tensor
        mask = mask.long()

        return image, mask, img_name



