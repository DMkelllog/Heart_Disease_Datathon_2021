
from metrics import get_DC, get_JS, DiceLoss
from tqdm import tqdm
import torchvision
import os
import numpy as np
import torch
from make_dataloader import CustomDataset
from torch.utils.data import Dataset, DataLoader
import argparse
from unet import pretrained_unet
import torch.nn as nn

train_2_dataset = CustomDataset('train', 'A2C',transform = torchvision.transforms.ToTensor())
train_4_dataset = CustomDataset('train', 'A4C', transform = torchvision.transforms.ToTensor())
val_2_dataset = CustomDataset('validation', 'A2C', transform = torchvision.transforms.ToTensor())
val_4_dataset = CustomDataset('validation', 'A4C', transform = torchvision.transforms.ToTensor())
train_2_4_dataset= []
test_2_4_dataset = []
for i in range(len(train_2_dataset)):
    train_2_4_dataset.append(train_2_dataset[i])
    train_2_4_dataset.append(train_4_dataset[i])
for j in range(len(val_2_dataset)):
    test_2_4_dataset.append(val_2_dataset[j])
    test_2_4_dataset.append(val_4_dataset[j])
    

train_loader= DataLoader(train_2_4_dataset, batch_size=8)
valid_loader= DataLoader(test_2_4_dataset, batch_size=8)

def main(args):

    device = torch.device("cuda:0")

    loaders = {"train": train_loader, "valid": valid_loader}
    unet = pretrained_unet(True)
    unet.to(device)

    best_validation_dsc =np.inf 

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    # criterion = DiceLoss()
    criterion = nn.BCELoss()
    loss_valid = []

    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            total_num, total_loss = 0,0

            for i, data in enumerate(loaders[phase]):

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                y_pred = unet(x)

                loss = criterion(y_pred, y_true)
                total_num +=x.size(0)
                total_loss += loss.item()

                if phase == "train":
                    loss.backward()
                    optimizer.step()
            train_loss = total_loss/total_num
            print(f"{phase} {epoch}/{args.epochs} {train_loss:.6f}")

            if phase == "valid":
                if train_loss < best_validation_dsc:
                    best_validation_dsc = train_loss 
                    torch.save(unet.state_dict(), "unet.pth")

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)",
    )
    args = parser.parse_args()
    main(args)