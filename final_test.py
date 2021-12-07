import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

# from utils import *

from torch.utils.data import DataLoader
from models import caranet, pretrained_unet

from final_utils import *


# Input: test set images, trained model
# Output: prediction on test set
# Functions: Resize, TTA, Ensemble


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="data/echocardiography/validation")
parser.add_argument('--data_type', type=str, choices=['A2C', 'A4C'], default='A2C')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_transform = ToTensorV2()

tta_transform = A.Compose([
            A.RandomContrast(limit=0, p=1),
            ToTensorV2(transpose_mask=True)
            ])

threshold = 0.5
num_tta = 10


ensemble_list = []
gt_mask_list = []

# size_dict, dataset = make_test_dataset(args.data_path, args.data_type, base_transform, mode='base')
# size_dict, dataset_tta = make_test_dataset(args.data_path, args.data_type, tta_transform, mode='base')

# data_loader_base = DataLoader(dataset, batch_size=1, shuffle=False)
# data_loader_tta = DataLoader(dataset_tta, batch_size=1, shuffle=False)

# for m in range(2):
#     print('unet:', i+1)

#     model = pretrained_unet(False).to(device)
#     model.load_state_dict(torch.load(f'models/{args.data_type}_unet_{i+1}.pt'))
#     total_list =  []
#     model.eval()
#     with torch.no_grad():
#         pred_list = []
#         for i, (img, gt_mask) in enumerate(data_loader_base):
#             output = model(img.cuda().float())
#             predict = ((output > threshold) + 0)
#             predict = resize_return(predict.squeeze(0), size_dict["cut_off"][i], size_dict["size"][i], 100).squeeze().cpu().numpy()
#             pred_list.append(predict)
#         total_list.append(pred_list)

#         for k in range(num_tta):
#             pred_list = []
#             for i, (img, gt_mask) in enumerate(data_loader_tta):
#                 output = model(img.cuda().float())
#                 predict = ((output > threshold) + 0)
#                 predict = resize_return(predict.squeeze(0), size_dict["cut_off"][i], size_dict["size"][i], 100).squeeze().cpu().numpy()
#                 pred_list.append(predict)
#             total_list.append(pred_list)
#     total_list = np.mean(np.vstack(total_list), axis=0)
#     total_list = [(a>threshold)+0 for a in total_list]
#     assert np.sum((total_list[0] > 0) & (total_list[0]<1)) == 0
#     ensemble_list.append(total_list)

size_dict, dataset = make_test_dataset(args.data_path, args.data_type, base_transform)
size_dict, dataset_tta = make_test_dataset(args.data_path, args.data_type, tta_transform)

data_loader_base = DataLoader(dataset, batch_size=1, shuffle=False)
data_loader_tta = DataLoader(dataset_tta, batch_size=1, shuffle=False)

for m in range(1,2):
    print('caranet:', m+1)
    model = caranet().to(device)
    model.load_state_dict(torch.load(f'models/{args.data_type}_caranet_{m+1}.pt'))
    total_list =  []
    model.eval()
    with torch.no_grad():
        pred_list = []
        for i, (img, gt_mask) in enumerate(data_loader_base):
            output = model(img.cuda().float())
            predict = ((output[0].sigmoid() > threshold) + 0)
            predict = resize_return(predict.squeeze(0), size_dict["cut_off"][i], size_dict["size"][i], 88).squeeze().cpu().numpy()
            pred_list.append(predict)
            gt_mask_list.append(gt_mask.squeeze(0).cpu().numpy())

        total_list.append(pred_list)

        for k in range(num_tta):
            pred_list = []
            for i, (img, gt_mask) in enumerate(data_loader_tta):
                output = model(img.cuda().float())
                predict = ((output[0].sigmoid() > threshold) + 0)
                predict = resize_return(predict.squeeze(0), size_dict["cut_off"][i], size_dict["size"][i], 88).squeeze().cpu().numpy()
                pred_list.append(predict)
            total_list.append(pred_list)
    total_list = np.mean(np.vstack(total_list), axis=0)
    total_list = [(a>threshold)+0 for a in total_list]
    assert np.sum((total_list[0] > 0) & (total_list[0]<1)) == 0
    ensemble_list.append(total_list)

ensemble_list = np.vstack(ensemble_list)
ensemble_list = np.mean(ensemble_list, axis=0)
ensemble_list = [(k>threshold)+0 for k in ensemble_list]

DS_list, JS_list = [], []
for i in range(len(dataset)):
    DS, JS = metrics(ensemble_list[i], gt_mask_list[i])
    DS_list.append(DS)
    JS_list.append(JS)
print(f'{args.data_type}')
print(f'DS: {np.mean(DS_list)}')
print(f'JS: {np.mean(JS_list)}')