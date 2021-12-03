import argparse
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import datetime

import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
import numpy as np
import pickle
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from augmentation import get_training_augmentation

from utils import *

from models import caranet
from unet import pretrained_unet

from metrics import DiceLoss


parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--random_seed', default=42)

parser.add_argument('--data_type', type=str, choices=['A2C', 'A4C', 'both', 'A2C_new', 'A4C_new', 'both_new'], default='both')
parser.add_argument('--augmentation_type', type=int, choices=[0, 1, 2], default=0)

parser.add_argument('--model_type', type=str, default='unet')
parser.add_argument('--pretrained_path', type=str)
parser.add_argument('--memo', type=str)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
date_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

if args.model_type == 'unet':
    mode = 'base'
    model = pretrained_unet(True).to(device)
elif args.model_type == 'caranet':
    mode = 'caranet'
    model = caranet().to(device)

if args.pretrained_path is not None:
    path = f'models/{args.pretrained_path}/model.pt'
    model.load_state_dict(torch.load(path))
    train_mode = f'{args.pretrained_path}_fine'
    args.learning_rate = args.learning_rate / 10
else:
    train_mode = 'pre'

if args.memo is None:
    model_path = f'{train_mode}_{args.data_type}_{args.augmentation_type}_{args.model_type}'
else:
    model_path = f'{train_mode}_{args.data_type}_{args.augmentation_type}_{args.model_type}_{args.memo}'

folder_name = f'models/{model_path}'
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

filename = f'{folder_name}/model'

# print(model_path)
# print(filename)

augmentation = get_training_augmentation(type=args.augmentation_type)

train_loader, val_loader = make_dataloader(args.data_type, augmentation, args.random_seed, args.batch_size, mode)

optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=args.learning_rate/1000, verbose=False)
criterion = DiceLoss()

early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=False, path = filename)
loss_dict = {'train': [], 'val': []}
# print(f'current learning rate: {args.learning_rate}')
for epoch in range(args.num_epochs):
    model.train()
    
    train_losses = []
    for it_1, (img, mask) in enumerate(train_loader):
        #print(train_img)
        img = img.to(device).float()
        mask = mask.to(device).float()
        #print(train_label)
        if mode == 'base':
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif mode == 'caranet':
            train_lrmap_5,train_lrmap_3,train_lrmap_2,train_lrmap_1 = model(img)
            train_loss5 = structure_loss(train_lrmap_5, mask)
            train_loss3 = structure_loss(train_lrmap_3, mask)
            train_loss2 = structure_loss(train_lrmap_2, mask)
            train_loss1 = structure_loss(train_lrmap_1, mask)
        
            loss = train_loss5 + train_loss3 + train_loss2 + train_loss1
        
            train_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, 0.5)
            optimizer.step()
    
    train_loss = np.average(train_losses)

    loss_dict['train'].append(train_loss)

    model.eval()
    with torch.no_grad():
        valid_losses = []
        for it_2, (img, mask) in enumerate(val_loader):
            img = img.float().to(device)
            mask = mask.to(device).float()
            #print(train_label)
            if mode == 'base':
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                valid_losses.append(loss.item())

            elif mode == 'caranet':
                lrmap_5, lrmap_3, lrmap_2, lrmap_1 = model(img)
                loss5 = structure_loss(lrmap_5, mask)
                loss3 = structure_loss(lrmap_3, mask)
                loss2 = structure_loss(lrmap_2, mask)
                loss1 = structure_loss(lrmap_1, mask)
            
                loss = loss5 + loss3 + loss2 + loss1
                valid_losses.append(loss.item())
        
        valid_loss = np.average(valid_losses)
        scheduler.step(valid_loss)
        
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
#            scheduler.step(float(val_loss))
        loss_dict['val'].append(valid_loss)
    if epoch%5==4:
        torch.save(model.state_dict(), f'{filename}_{epoch}.pt')
            
    print(f'Epoch {epoch} train_loss: {train_loss:0.5f}   val_loss: {valid_loss:0.5f}')


# evaluation

model.load_state_dict(torch.load(f'{filename}.pt'))
DS, JS = evaluate(model, val_loader, mode)

with open(f'{folder_name}/results.txt', 'w') as f:
    f.write(f'Dice Similarity: {DS:0.4f}\n')
    f.write(f'Jaccard Similarity: {JS:0.4f}')

with open('result.csv', 'a', newline='') as f:
    wr=csv.writer(f)
    wr.writerow([model_path, np.round(DS,5), np.round(JS,5)])