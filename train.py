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

from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import *
from unet.unet import UNet

parser = ArgumentParser()
parser.add_argument('--train_folder', required=True, type=str)
parser.add_argument('--val_folder', type=str)
parser.add_argument('--checkpoint_folder', required=True, type=str)

parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--model_name', type=str, default='unet')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--crop_size', type=int, default=256)
args = parser.parse_args()

# Set random seed for consistent result
seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Check and create checkpoint path
if not os.path.isdir(args.checkpoint_folder):
    os.makedirs(args.checkpoint_folder)

# Decided to save best model based on val_loss or train_loss
if args.val_folder:
    saving_target = 'val_loss'
else:
    saving_target = 'train_loss'

# Define dataloader
# Training data transform func
train_tf = None #TODO
train_dataset = CellDataset(args.train_folder,transform=train_tf)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.val_folder:
    # Val data transform func
    val_tf = None #TODO
    val_dataset = CellDataset(args.val_folder,transform=val_tf)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Define net
net = UNet(in_channel=3,out_channel=2)



# Define optimizer and criteration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

min_loss = np.inf

execution_log = []
for epoch in range(1,args.epochs+1):  # loop over the dataset multiple times
    
    net.train(True)
    # dict used to save metrics for every epoch
    log_dict = {}
    log_dict['eopch'] = epoch
    train_running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs; data is a list of [image_batch, mask_batch]
        X_batch, y_batch, image_name = data

        # Send batch to corresponding device
        X_batch = Variable(X_batch.to(device=args.device))
        y_batch = Variable(y_batch.to(device=args.device))

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        y_pred = net(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        # Free memory space
        del X_batch, y_batch

        train_running_loss += loss.item()
    
    # print statistics
    print('[%d] train_loss: %.3f' %
                  (epoch, train_running_loss / (i+1)))
    log_dict['train_loss'] = train_running_loss / (i+1)


    # Val data
    # Disable training first
    net.train(False)
    val_running_loss = 0.0
    if args.val_dataset:
        print('Val!')
        for i, data in enumerate(train_loader, 0):
            X_batch, y_batch, image_name = data

            # Send batch to corresponding device
            X_batch = Variable(X_batch.to(device=args.device))
            y_batch = Variable(y_batch.to(device=args.device))

            # Predict on val data
            y_pred = net(X_batch)
            
            # Calculate val loss
            loss = criterion(y_pred, y_batch)

            del X_batch, y_batch

            val_running_loss += loss.item()
        
        print('[%d] val_loss: %.3f' %
                  (epoch, val_running_loss / (i+1)))
        log_dict['val_loss'] = val_running_loss / (i+1)

    scheduler.step(log_dict[saving_target])

    execution_log.append(log_dict)

    # Save best model
    if log_dict[saving_target]<min_loss:
        print(saving_target+' improved from '+str(min_loss)+' to '+str(log_dict[saving_target])+', saving model')
        min_loss = log_dict[saving_target]
        torch.save(net, args.checkpoint_folder+'best_model.pt')
    # Save latest model
    torch.save(net, args.checkpoint_folder+'latest_model.pt')

    # Save predicted images for every k epochs
    if args.save_freq > 0:
        if epoch%args.save_freq == 0:
            # TODO
            print("Saving predictions")

# Save log as .csv file for plot generation in future
keys = execution_log[0].keys()
with open(args.checkpoint_folder+'log.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(execution_log)

print('Finished Training')

