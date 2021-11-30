
from metrics import get_DC, get_JS, DiceLoss
from tqdm import tqdm
import os
import numpy as np
import torch
from make_dataloader import CustomDataset
from torch.utils.data import Dataset, DataLoader
import argparse

train_2_dataset = CustomDataset('train', 'A2C')
train_4_dataset = CustomDataset('train', 'A4C')
val_2_dataset = CustomDataset('validation', 'A2C')
val_4_dataset = CustomDataset('validation', 'A4C')
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
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)

    unet.to(device)

    best_validation_dsc =np.inf 

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    loss_train = []
    loss_valid = []
    criterion = DiceLoss()

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()


            for i, data in enumerate(loaders[phase]):

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = criterion(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if loss_valid[-1]< best_validation_dsc:
                    best_validation_dsc = loss_valid[-1] 
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
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    args = parser.parse_args()
    main(args)