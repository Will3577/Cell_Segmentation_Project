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
parser.add_argument('--train_folder', required=True, type=str)
parser.add_argument('--val_folder', type=str)
parser.add_argument('--checkpoint_folder', required=True, type=str)

parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=4, type=int)
# parser.add_argument('--model_name', type=str, default='unet')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--weights', type=str)
parser.add_argument('--in_channels', type=int, default=1)

args = parser.parse_args()

# Set random seed for consistent result
seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Check and create checkpoint path
mk_dirs(args.checkpoint_folder)

pred_folder = 'pred/'
mk_dirs(args.checkpoint_folder+pred_folder)

# Decided to save best model based on val_loss or train_loss
# if args.val_folder:
#     saving_target = 'val_loss'
# else:
#     saving_target = 'train_loss'
saving_target = 'val_auc'

# crop_size = args.crop_size
# Define dataloader
# Training data transform func
train_tf = TransformMitosis()
train_dataset = MitosisDataset(args.train_folder,args.in_channels,transform=train_tf)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

if args.val_folder:
    # Val data transform func
    val_tf = TransformMitosis(flip_rate=0,mirror_rate=0)
    val_dataset = MitosisDataset(args.val_folder,args.in_channels,transform=val_tf)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)


# Define net
if args.weights:
    net = torch.load(args.weights)
    print(">>> Pretrained weights successfully loaded from "+args.weights)
else:
    net = VGG_net(in_channels=args.in_channels,num_classes=2)
    # net = Network()
    
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device=args.device)


# Define optimizer and criteration
criterion = cross_entropy
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

max_auc = -np.inf

execution_log = []
for epoch in range(1,args.epochs+1):  # loop over the dataset multiple times
    
    net.train(True)
    # dict used to save metrics for every epoch
    log_dict = {}
    log_dict['epoch'] = epoch
    train_running_loss = 0.0
    train_auc = 0.0
    train_acc = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs; data is a list of [image_batch, mask_batch]
        X_batch, y_batch, image_name = data
        # print(X_batch.shape,y_batch.shape,y_batch[0],torch.amin(X_batch))
        # print(X_batch.shape, y_batch.shape)
        # Send batch to corresponding device
        X_batch = Variable(X_batch.to(device=args.device))
        y_batch = Variable(y_batch.to(device=args.device))
        # print(torch.amax(X_batch))
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        y_pred = net(X_batch)
        # y_pred = torch.softmax(y_pred,dim=1)
        # print(y_pred.shape,y_pred)

        # y_pred = torch.argmax(y_pred, dim=1)
        # y_pred = torch.log(y_pred+1e-32)
        # loss = criterion(y_pred, y_batch)
        # print(y_pred.shape,y_batch.shape)
        loss = criterion(y_pred.float(), y_batch.long())
        aucroc = roc_auc_compute_fn(y_pred.float(), y_batch.long())
        acc = accuracy_compute_fn(y_pred.float(), y_batch.long())
        loss.backward()
        optimizer.step()

        # Free memory space
        del X_batch, y_batch

        train_running_loss += loss.item()
        train_auc += aucroc
        train_acc += acc
    
    # print statistics
    # print('[%d] train_loss: %.3f' %
    #               (epoch, train_running_loss / (i+1)))
    log_dict['train_loss'] = train_running_loss / (i+1)
    log_dict['train_auc'] = train_auc / (i+1)
    log_dict['train_acc'] = train_acc / (i+1)

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
            aucroc = roc_auc_compute_fn(y_pred.float(), y_batch.long())
            acc = accuracy_compute_fn(y_pred.float(), y_batch.long())
            del X_batch, y_batch

            val_running_loss += loss.item()
            val_auc += aucroc
            val_acc += acc
        
        # print('[%d] val_loss: %.3f' %
        #           (epoch, val_running_loss / (i+1)))
        log_dict['val_loss'] = val_running_loss / (i+1)
        log_dict['val_auc'] = val_auc / (i+1)
        log_dict['val_acc'] = val_acc / (i+1)
    print(log_dict)
    scheduler.step(log_dict[saving_target])

    execution_log.append(log_dict)

    # Save best model
    if log_dict[saving_target]>max_auc:
        print(saving_target+' improved from '+str(max_auc)+' to '+str(log_dict[saving_target])+', saving model')
        max_auc = log_dict[saving_target]
        torch.save(net, args.checkpoint_folder+'best_model.pt')
    # print(args.checkpoint_folder+'best_model.pt')
    # Save latest model
    torch.save(net, args.checkpoint_folder+'latest_model.pt')


# Save log as .csv file for plot generation in future
keys = execution_log[0].keys()
with open(args.checkpoint_folder+'log.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(execution_log)

print('Finished Training')

