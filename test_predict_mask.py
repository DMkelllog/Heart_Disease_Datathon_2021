# Test Prediction

# Predict test data & create predicted mask

# Input: test_path, trained_model  
# OutPut: predicted masks

# Load
# from final_utils import *
import argparse

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import torch

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *

from models import caranet, pretrained_unet

class TestDataset(Dataset):
    def __init__(self, X, transform=False):
        
        self.X = X
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.X[index]
        img = (img*255).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        return img

    def __len__(self):
        return len(self.X)

def resize_crop(img, mask=False, resize_h=352, resize_w=528, crop_size=88):
    resize = transforms.Compose([transforms.ToTensor(), transforms.Resize((resize_h, resize_w))])

    img = resize(img)
    img = img[:, :, crop_size:-crop_size]
    if mask:
        mask = resize(mask)
        mask = mask[:, :, crop_size:-crop_size]
        return img, mask
    else:
        return img

def remove_topnoise(img, mask=False):
    cand1 = int(img.shape[1] / 8 * 7)
    cand2 = int(img.shape[1] / 8 * 6)

    cutoff1 = np.where(img[:, cand1, :] < 0.01)[0][0]
    cutoff2 = np.where(img[:, cand2, :] < 0.01)[0][0]


    cutoff_min = np.min([cutoff1, cutoff2])
    if mask:
        return cutoff_min, img[cutoff_min:, ], mask[cutoff_min:, ]
    else:
        return cutoff_min, img[cutoff_min:, ]

def resize_return(y_pred, cutoff, originsize, pad_size): ## cutoff, originsize(tuple)
    h, w = originsize
    h -= cutoff

    inverse = transforms.Compose([
                                transforms.Pad(padding=(pad_size, 0), fill=0),
                                transforms.Resize((h,w)),
                                transforms.Pad(padding=(0,cutoff,0,0), fill=0)
                                 ])
    
    y_pred_inverse = inverse(y_pred)
    return y_pred_inverse



device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = A.pytorch.ToTensorV2()

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="data/echocardiography/test")
parser.add_argument('--data_type', type=str, choices=['A2C', 'A4C'], default='A2C')

parser.add_argument('--model_type', type=str, default='unet')
parser.add_argument('--pretrained_path', type=str, help='pretrained model path txt file')
parser.add_argument('--memo', type=str)

args = parser.parse_args()

threshold = 0.5
ensemble_list = []

################################ Unet ###############################################
d_path = os.path.join(args.data_path, args.data_type)
img_list = sorted(os.listdir(d_path))
X_test = []
size_dict = {"size":[], "cut_off":[]}
for img_name in img_list:
    if img_name.endswith('.png'):
        img = plt.imread(d_path + '/' + img_name)
        h, w = img.shape[0], img.shape[1]
        cutoff, img = remove_topnoise(img)
        img = resize_crop(img, mask=False, resize_h=400, resize_w=600, crop_size=100)
        img = img.numpy().transpose(1, 2, 0)
        X_test.append(img[:, :, :3])
        size_dict["size"].append((h,w))
        size_dict["cut_off"].append(cutoff)
        
dataset_test = TestDataset(X_test, transform = transform)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

model = pretrained_unet(False).to(device)
mode = 'base'
pad_size = 100

for i in range(2):
    print('unet ', i+1)
    path = f'models/{args.data_type}_unet_{i+1}.pt'
    model.load_state_dict(torch.load(path))

    # Predict mask
    predicted_masks = []
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(dataloader_test):
            output = model(img.cuda().float())
            if mode == 'base': 
                # unet
                output_binary = ((output > threshold) + 0).cpu()
            else: 
                # caranet
                output_binary = ((output[0].sigmoid() > threshold) + 0).cpu()
            predicted_masks.append(output_binary)
    predicted_masks = torch.vstack(predicted_masks)

    y_inverse_list = []
    for i, mask_pred in enumerate(predicted_masks):
        y_inverse = resize_return(mask_pred, size_dict["cut_off"][i], size_dict["size"][i], pad_size)
        y_predicted = y_inverse.squeeze().numpy()
        y_inverse_list.append(y_inverse.squeeze().numpy())
    ensemble_list.append(y_inverse_list)
#####################################################################################


################################ caranet ############################################

d_path = os.path.join(args.data_path, args.data_type)
img_list = sorted(os.listdir(d_path))
X_test = []
size_dict = {"size":[], "cut_off":[]}
for img_name in img_list:
    if img_name.endswith('.png'):
        img = plt.imread(d_path + '/' + img_name)
        h, w = img.shape[0], img.shape[1]
        cutoff, img = remove_topnoise(img)
        img = resize_crop(img)
        img = img.numpy().transpose(1, 2, 0)
        X_test.append(img[:, :, :3])
        size_dict["size"].append((h,w))
        size_dict["cut_off"].append(cutoff)
        
dataset_test = TestDataset(X_test, transform = transform)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

model = caranet().to(device)
mode = 'caranet'
pad_size = 88

for i in range(2):
    print('caranet ', i+1)
    path = f'models/{args.data_type}_caranet_{i+1}.pt'
    model.load_state_dict(torch.load(path))

    # Predict mask
    predicted_masks = []
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(dataloader_test):
            output = model(img.cuda().float())
            if mode == 'base': 
                # unet
                output_binary = ((output > threshold) + 0).cpu()
            else: 
                # caranet
                output_binary = ((output[0].sigmoid() > threshold) + 0).cpu()
            predicted_masks.append(output_binary)
    predicted_masks = torch.vstack(predicted_masks)

    y_inverse_list = []
    for i, mask_pred in enumerate(predicted_masks):
        y_inverse = resize_return(mask_pred, size_dict["cut_off"][i], size_dict["size"][i], pad_size)
        y_predicted = y_inverse.squeeze().numpy()
        y_inverse_list.append(y_inverse.squeeze().numpy())
    ensemble_list.append(y_inverse_list)
#####################################################################################

ensemble_list = np.vstack(ensemble_list)
ensemble_list = np.mean(ensemble_list, axis=0)
ensemble_list = [(ensemble > 0.5) + 0 for ensemble in ensemble_list]

# Save predicted masks
test_mask_filenames = [npy_file.replace('png','npy') for npy_file in img_list if npy_file.endswith('.png')]

for i, mask_pred in enumerate(ensemble_list):
    np.save(d_path +'/'+ test_mask_filenames[i], mask_pred)
