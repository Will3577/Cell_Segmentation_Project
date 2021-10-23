# Code skeleton adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#                            https://github.com/cosmic-cortex/pytorch-UNet

import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import utils

class TransformData:
    def __init__(self):
        return 0
    
    def __call__(self,image,mask):
        return 0

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
        return len(os.listdir(self.input_path))
    
    def __getitem__(self, idx):
        mask_name = self.image_list[idx]
        img_name = 't0'+mask_name.split('0')[-1]
        image = cv2.imread(self.input_path+img_name)
        mask = cv2.imread(self.mask_path+mask_name, 0)
        print(self.input_path+img_name)
        print(image.shape,mask.shape)

        image, mask = self.transform(image, mask)

        # To long tensor
        mask = mask.long()

        return image, mask, img_name



