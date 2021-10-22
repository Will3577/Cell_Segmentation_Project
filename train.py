# Code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#                   https://github.com/cosmic-cortex/pytorch-UNet


import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from argparse import ArgumentParser
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--checkpoint_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--model_name', type=str, default='unet')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--crop_size', type=int, default=256)
args = parser.parse_args()


# define dataloader
train_loader = None
val_loader = None

# define net
net = None


# define optimizer and criteration
criterion = nn.CrossEntropyLoss()
optimizer = optim.adam(net.parameters(), lr=args.learning_rate, momentum=args.momentum)


for epoch in range(1,args.epochs+1):  # loop over the dataset multiple times
    net.train(True)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X_batch, Y_batch = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_batch = net(X_batch)
        loss = criterion(pred_batch, Y_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('[%d] train_loss: %.3f' %
                  (epoch, running_loss / (i+1)))
    
    # Val data
    net.train(False)
    if args.val_dataset:
        print('Val!')

    # Save best model



    # Save log as .csv file for plot generation in future


print('Finished Training')

