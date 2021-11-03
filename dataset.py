# Code skeleton adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#                            https://github.com/cosmic-cortex/pytorch-UNet

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from numpy.random import random_sample
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import utils
from torchvision.transforms import functional as F

class TransformData:
    def __init__(self,crop=None):
        # TODO add more operations to improve model performance
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
        Folder structure:
        Dataset_folder (train/val/test)
            |-- images (01)
                |-- t000.tif
                |-- t001.tif
                |-- ...
            |-- masks (01_GT)
                |-- SEG
                    |-- man_seg021.tif
                    |-- man_seg022.tif
                    |-- ...
    """

    def __init__(self, dataset_folder, transform=None):
        self.input_path = dataset_folder+"images/"
        self.mask_path = dataset_folder+"masks/"
        self.image_list = os.listdir(self.mask_path)

        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.mask_path))
    
    def __getitem__(self, idx):
        mask_name = self.image_list[idx]
        img_name = mask_name.split('.')[0]+'.png'#t'+mask_name[-7:]
        image = cv2.imread(self.input_path+img_name, 0)
        mask = cv2.imread(self.mask_path+mask_name, 0)
        # print(self.input_path+img_name, mask_name)
        # print(image.shape,mask.shape,np.amax(image))
        # cv2.imwrite('/content/test_img.png',image)
        # cv2.imwrite('/content/test_mask.png',mask)
        # print(image.shape,mask.shape)

        if self.transform:
            image, mask = self.transform(image, mask)
        
        # image.save('/content/test_img.png')
        # mask.save('/content/test_mask.png')

        
        # To tensor
        to_tensor = T.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)
        # print(image.shape,mask.shape)

        # To long tensor
        mask = mask.long()

        return image, mask, img_name

class TransformMitosis:
    def __init__(self,flip_rate=0.5,mirror_rate=0.5):
        # TODO add more operations to improve model performance
        # self.crop = crop
        self.flip_rate = flip_rate
        self.mirror_rate = mirror_rate
    
    def __call__(self,curr,next):
        curr = F.to_pil_image(curr)
        next = F.to_pil_image(next)
        if random_sample() < self.flip_rate:
            curr = ImageOps.flip(curr)
            next = ImageOps.flip(next)
        if random_sample() < self.mirror_rate:
            curr = ImageOps.mirror(curr)
            next = ImageOps.mirror(next)
        

        return curr, next
import torch
class MitosisDataset(Dataset):
    """ Cell segmentation dataset
        Folder structure:
        Dataset_folder (train/val/test)
            |-- images (01)
                |-- t000.tif
                |-- t001.tif
                |-- ...
            |-- masks (01_GT)
                |-- SEG
                    |-- man_seg021.tif
                    |-- man_seg022.tif
                    |-- ...
    """

    def __init__(self, dataset_folder,in_channels, transform=None):
        self.curr_path = dataset_folder+"curr/"
        self.next_path = dataset_folder+"next/"
        self.image_list = os.listdir(self.curr_path)
        self.in_channels - in_channels
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.curr_path))
    
    def __getitem__(self, idx):
        
        img_name = self.image_list[idx]
        class_type = int(img_name.split('_')[-1].split('.')[0])
        # img_name = img_name.split('.')[0]+'.jpg'#t'+mask_name[-7:]
        curr = cv2.imread(self.curr_path+img_name, 0)
        next = cv2.imread(self.next_path+img_name, 0)
        # print(np.amax(curr))
        # print(self.input_path+img_name, mask_name)
        # print(image.shape,mask.shape,np.amax(image))
        # cv2.imwrite('/content/test_img.png',image)
        # cv2.imwrite('/content/test_mask.png',mask)
        # print(image.shape,mask.shape)

        if self.transform:
            curr, next = self.transform(curr, next)
        
        # curr.save('/content/test_img.png')
        # next.save('/content/test_mask.png')

        
        # To tensor
        # curr = np.array([curr,next])
        to_tensor = T.ToTensor()
        curr = to_tensor(curr)
        next = to_tensor(next)
        if self.in_channels==2:
            curr = torch.cat((curr, next), 0)
        # print(torch.amax(curr))

        # class_type = np.array([[class_type]]))
        # print(image.shape,mask.shape)

        # To long tensor
        # class_type = class_type.long()
        # print(curr.shape)
        return curr, torch.tensor(class_type), img_name