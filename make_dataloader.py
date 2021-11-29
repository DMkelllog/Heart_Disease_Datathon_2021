import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

# make pickle
for mode in ['train', 'validation']:
    for version in ['A2C', 'A4C']:
        X_list = []
        y_list = []

        PATH = f'data/resize_crop/{mode}/{version}'
        img_list = sorted(os.listdir(f'{PATH}/img/'))

        for img_name in img_list:
            img = plt.imread(f'{PATH}/img/{img_name}')
            mask = np.load(f'{PATH}/mask/{img_name[:-4]}.npy')

            X_list.append(img[:, :, :3])
            y_list.append(mask)

        with open(f'data/{mode}_{version}.pickle', 'wb') as f:
            pickle.dump([np.array(X_list), np.array(y_list)], f)

# data loader
class CustomDataset(Dataset):
    def __init__(self, mode, version, transform=False):
        with open(f'data/{mode}_{version}.pickle', 'rb') as f:
            X, y = pickle.load(f)
        
        self.X = X
        self.y = y

        self.transform = transform
        
    def __getitem__(self, index):
        img, mask = self.X[index], self.y[index]

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    train_2_dataset = CustomDataset('train', 'A2C')
    train_4_dataset = CustomDataset('train', 'A4C')
    val_2_dataset = CustomDataset('validation', 'A2C')
    val_4_dataset = CustomDataset('validation', 'A4C')

    train_2_dataloader = DataLoader(train_2_dataset, batch_size=40, shuffle=False)
    train_4_dataloader = DataLoader(train_2_dataset, batch_size=40, shuffle=False)
    val_2_dataloader = DataLoader(val_2_dataset, batch_size=40, shuffle=False)
    val_4_dataloader = DataLoader(val_4_dataset, batch_size=40, shuffle=False)