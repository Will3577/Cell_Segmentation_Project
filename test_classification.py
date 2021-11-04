# Code partially adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#                             https://github.com/cosmic-cortex/pytorch-UNet

# TODO Add object-wise criteration(dice-coef, f1) in metrics.py
# TODO Write test pipeline in test.py

import os
import csv
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
# from sklearn.metrics import roc_auc_score

from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import *
from utils import *
from models import *
from metrics import *

parser = ArgumentParser()
# parser.add_argument('--train_folder', required=True, type=str)
parser.add_argument('--weights', required=True, type=str)
parser.add_argument('--val_folder', required=True, type=str)

parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--in_channels', type=int, default=1)

args = parser.parse_args()

# Set random seed for consistent result
seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Decided to save best model based on val_loss or train_loss
saving_target = 'val_acc'

# crop_size = args.crop_size
# Define dataloader
# Training data transform func
# train_tf = TransformMitosis()
# train_dataset = MitosisDataset(args.train_folder,args.in_channels,transform=train_tf)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# Val data transform func
val_tf = TransformMitosis(flip_rate=0,mirror_rate=0)
val_dataset = MitosisDataset(args.val_folder,args.in_channels,transform=val_tf)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)


# Define net
net = torch.load(args.weights)
print(">>> Pretrained weights successfully loaded from "+args.weights)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device=args.device)


# Define optimizer and criteration
criterion = cross_entropy

max_auc = -np.inf

execution_log = []
for epoch in range(1,args.epochs+1):  # loop over the dataset multiple times
    
    # net.train(True)
    # dict used to save metrics for every epoch
    log_dict = {}
    log_dict['epoch'] = epoch

    # Val data
    # Disable training first
    net.train(False)
    val_running_loss = 0.0
    val_auc = 0.0
    val_acc = 0.0
    if args.val_folder:
        for i, data in enumerate(val_loader, 0):
            X_batch, y_batch, image_name = data

            # Send batch to corresponding device
            X_batch = Variable(X_batch.to(device=args.device))
            y_batch = Variable(y_batch.to(device=args.device))

            # Predict on val data
            y_pred = net(X_batch)
            y_pred = torch.softmax(y_pred,dim=1)
            
            # Calculate val loss
            loss = criterion(y_pred.float(), y_batch.long())
            # aucroc = roc_auc_compute_fn(y_pred.float(), y_batch.long())
            acc = accuracy_compute_fn(y_pred.float(), y_batch.long())
            del X_batch, y_batch

            val_running_loss += loss.item()
            # val_auc += aucroc
            val_acc += acc
        
        # print('[%d] val_loss: %.3f' %
        #           (epoch, val_running_loss / (i+1)))
        log_dict['val_loss'] = val_running_loss / (i+1)
        # log_dict['val_auc'] = val_auc / (i+1)
        log_dict['val_acc'] = val_acc / (i+1)
    print(log_dict)
    execution_log.append(log_dict)

print('Finished Testing')

