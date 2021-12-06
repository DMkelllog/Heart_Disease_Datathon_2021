import argparse
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import datetime

import torch

import numpy as np
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *

from models import caranet, pretrained_unet

import time

from final_utils import *

# Input: test set images, trained model
# Output: prediction on test set
# Functions: Resize, TTA, Ensemble


parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--random_seed', default=42)

parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--data_type', type=str, choices=['A2C', 'A4C', 'both'], default='both')
parser.add_argument('--augmentation_type', type=int, choices=[0, 1, 2, 3], default=0)

parser.add_argument('--model_type', type=str, default='unet')
parser.add_argument('--pretrained_path', type=str)
parser.add_argument('--memo', type=str)

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = A.ToTensorV2()

size_dict, dataset = make_dataset(args.data_path, args.data_type, transform)

#data (TTA에 따라 달라짐)
#model (ensemble??)

if args.model_type == 'unet':
    mode = 'base'
    model = pretrained_unet(True).to(device)
elif args.model_type == 'caranet':
    mode = 'caranet'
    model = caranet().to(device)

if args.pretrained_path is not None:
    if args.pretrained_epoch == 'full':
        path = f'models/{args.pretrained_path}/model.pt'
    else:
        path = f'models/{args.pretrained_path}/model_{args.pretrained_epoch}.pt'
    model.load_state_dict(torch.load(path))
    # train_mode = f'{args.pretrained_path}_fine'

final_evaluate(model, dataset, size_dict, mode)
