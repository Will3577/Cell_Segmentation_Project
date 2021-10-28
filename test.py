
import os
import cv2
import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from skimage.util.shape import view_as_windows
from argparse import ArgumentParser
from functools import partial

from dataset import *

def crop_batch(input_batch: torch.tensor, patch_size: int):
    # crop_size = (args.crop_size,args.crop_size)
    # crop = 256
    input_batch = input_batch.detach().cpu().numpy()
    print(input_batch.shape)
    input_batch = np.transpose(input_batch[0],(1,2,0))
    im_H = input_batch.shape[0]
    im_W = input_batch.shape[1]
    height_padding = (math.ceil(im_H/patch_size)*patch_size-im_H)//2
    width_padding = (math.ceil(im_W/patch_size)*patch_size-im_W)//2
    print(im_H,im_W,height_padding,width_padding)
    image = cv2.copyMakeBorder(input_batch,height_padding,height_padding,width_padding,width_padding,cv2.BORDER_REFLECT)
    print(image.shape)

    tiles = np.array([image[x:x+patch_size,y:y+patch_size] for x in range(0,image.shape[0],patch_size) for y in range(0,image.shape[1],patch_size)])
    tiles = torch.tensor(tiles)
    tiles = tiles[:,None,:,:]
    print(tiles.shape,image.shape)

    return tiles, image.shape, height_padding, width_padding

def compose_pred(pred: torch.tensor, pseudo_shape: tuple, height_padding: int, width_padding: int):
    # pred (15,256,256)
    # target (1,700,1100)
    pred = pred.detach().cpu().numpy()
    height_padding //= 2
    width_padding //= 2
    print(pred.shape,pseudo_shape,height_padding,width_padding)
    patch_size = pred[0].shape[1]
    n_batches = pred.shape[0]
    output = np.zeros((2,pseudo_shape[0],pseudo_shape[1]))
    num_H = int(math.ceil(pseudo_shape[0]/patch_size))
    num_W = int(math.ceil(pseudo_shape[1]/patch_size))
    assert num_H*num_W == n_batches
    # i = 0
    print(num_H,num_W)
    for idx_h in range(num_H):
        for idx_w in range(num_W):
            output[:,idx_h*patch_size:(idx_h+1)*patch_size,idx_w*patch_size:(idx_w+1)*patch_size] = pred[num_W*idx_h+idx_w]
            print(num_W*idx_h+idx_w)
        # i+=1
    res = output[:,height_padding:pseudo_shape[0]-height_padding,width_padding:pseudo_shape[1]-width_padding]
    res = torch.tensor(res)
    res = res[None,:,:,:]
    print("output shape: ",res.shape)
    return res

parser = ArgumentParser()
parser.add_argument('--test_folder', required=True, type=str)
parser.add_argument('--save_path', required=True, type=str)
parser.add_argument('--weights', required=True, type=str)
parser.add_argument('--patch_size', required=True, type=int)
parser.add_argument('--device', default='cpu', type=str)
# parser.add_argument('--threshold', default=0.5, type=float)
args = parser.parse_args()

# Set random seed for consistent result
seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Check and create checkpoint path
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

# Load trained model
net = torch.load(args.weights)
net.to(device=args.device)
net.train(False)

criterion = nn.CrossEntropyLoss()


# Training data transform func
test_tf = None #TODO
test_dataset = CellDataset(args.test_folder,transform=test_tf)
train_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

running_loss = 0.0
for i, data in enumerate(train_loader, 0):
    X_batch, y_batch, image_name = data

    # crop test image into batches into size of (patch_size x patch_size)
    X_batch,pseudo_shape,height_padding,width_padding = crop_batch(X_batch,args.patch_size)

    X_batch = Variable(X_batch.to(device=args.device))
    y_batch = Variable(y_batch.to(device=args.device))

    y_pred = net(X_batch)
    #(10,3,256,256)
    # Compose prediction batch into a single image
    pred_img = compose_pred(y_pred,pseudo_shape,height_padding,width_padding)
    pred_img = Variable(pred_img.to(device=args.device))
    # Calculate test loss and report
    loss = criterion(pred_img, y_batch)
    running_loss += loss
    print("test_loss: "+str(running_loss/(i+1)))
    # TODO Save predictions

    # Thresholding class 1 to 255, class 0 to 0



    cv2.imwrite(args.results_path+image_name, pred_img)










