
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
from skimage.util.shape import view_as_windows
from argparse import ArgumentParser
from functools import partial

from dataset import *

def crop_batch(input_batch, patch_size):
    crop_size = (args.crop_size,args.crop_size)

    return input_batch

def compose_pred(pred, img_shape):
    batch_size = pred[0].shape

    return pred

parser = ArgumentParser()
parser.add_argument('--test_folder', required=True, type=str)
parser.add_argument('--results_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
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
if not os.path.isdir(args.results_path):
    os.makedirs(args.results_path)

# Load trained model
net = torch.load(args.model_path)
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
    X_batch = crop_batch(X_batch,args.patch_size)

    X_batch = Variable(X_batch.to(device=args.device))
    y_batch = Variable(y_batch.to(device=args.device))

    y_pred = net(X_batch)

    # Compose prediction batch into a single image
    pred_img = compose_pred(y_pred,y_batch[0].shape)

    # Calculate test loss and report
    loss = criterion(pred_img, y_batch)
    running_loss += loss
    print("test_loss: "+str(running_loss/(i+1)))
    # TODO Save predictions

    # Thresholding class 1 to 255, class 0 to 0



    cv2.imwrite(args.results_path+image_name, pred_img)










