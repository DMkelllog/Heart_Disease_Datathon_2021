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

from torch.utils.data import DataLoader
from models import caranet, pretrained_unet

import time

from final_utils import *

# Input: test set images, trained model
# Output: prediction on test set
# Functions: Resize, TTA, Ensemble


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--data_type', type=str, choices=['A2C', 'A4C'], default='A2C')

parser.add_argument('--model_type', type=str, default='caranet')
parser.add_argument('--pretrained_path', type=str)
parser.add_argument('--memo', type=str)

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_transform = ToTensorV2()

tta_transform = A.Compose([
            A.RandomContrast(p=1),
            ToTensorV2(transpose_mask=True)
            ])

size_dict, dataset = make_test_dataset(args.data_path, args.data_type, base_transform)
size_dict, dataset_tta = make_test_dataset_tta(args.data_path, args.data_type, tta_transform)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

#data (TTA에 따라 달라짐)
#model (ensemble??)

if args.model_type == 'unet':
    mode = 'base'
    model = pretrained_unet(True).to(device)
elif args.model_type == 'caranet':
    mode = 'caranet'
    model = caranet().to(device)

model.load_state_dict(torch.load(args.pretrained_path))
# train_mode = f'{args.pretrained_path}_fine'

final_evaluate(model, data_loader, size_dict, 0.5, mode)
