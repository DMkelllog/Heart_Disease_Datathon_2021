import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(type=2):
    
    if type == 0:
        train_transform = [
            ToTensorV2(transpose_mask=True)
        ]

    if type == 1:
        train_transform = [
            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(p=1),
                    A.RandomContrast(p=1),
                ],
                p=0.5,
            ),
            A.ShiftScaleRotate(scale_limit=0, rotate_limit=0.1, shift_limit=0.1, p=0.5, border_mode=0),
            ToTensorV2(transpose_mask=True)
        ]

    if type == 2:
        train_transform = [
             A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(p=1),
                    A.RandomContrast(p=1),
                ],
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.Rotate(limit=20),
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0, rotate_limit=0, border_mode=0),
            ToTensorV2(transpose_mask=True)
        ]

    if type == 3:
        train_transform = [
             A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(limit=(-0.05, 0.05), p=1),
                    A.RandomContrast(limit=(-0.1, 0.1), p=1),
                ],
                p=0.5,
            ),
            A.Rotate(limit=(-7, 7), border_mode=0),
            A.ShiftScaleRotate(shift_limit_x=(-0.05, 0.05), shift_limit_y=(-0.05, 0.05), rotate_limit=(0, 0), scale_limit=(-0.1, 0.1), border_mode=0),
            ToTensorV2(transpose_mask=True),
        ]

    return A.Compose(train_transform)

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=False):
        
        self.X = X
        self.y = y

        self.transform = transform
        
    def __getitem__(self, index):
        img, mask = self.X[index], self.y[index]
        img = (img*255).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask

    def __len__(self):
        return len(self.X)

def make_dataloader(data_type, transform, random_seed=42, batch_size=16, mode='base'):
    # for train & val
    base_transform = A.Compose([
        ToTensorV2(transpose_mask=True)
    ])

    if mode == 'base':
        filename = ''

    elif mode == 'caranet':
        filename = '_2'

    if data_type != 'both':
        if data_type == 'A2C' or data_type == 'A4C':
            load_dir = ''

            with open(f'data/{load_dir}/validation_{data_type[:3]}{filename}.pickle', 'rb') as f:
                test_img, test_mask = pickle.load(f)

        with open(f'data/{load_dir}/train_{data_type[:3]}{filename}.pickle', 'rb') as f:
            train_img, train_mask = pickle.load(f)

    else:
        if data_type == 'both':
            load_dir = ''

            both_ts_img, both_ts_mask = [], []
            
            for version in ['A2C', 'A4C']:
                with open(f'data/{load_dir}/validation_{version}{filename}.pickle', 'rb') as f:
                    test_img, test_mask = pickle.load(f)
                    both_ts_img.extend(test_img)
                    both_ts_mask.extend(test_mask)

            test_img = np.array(both_ts_img)
            test_mask = np.array(both_ts_mask)

        both_tr_img, both_tr_mask = [], []

        for version in ['A2C', 'A4C']:
            with open(f'data/{load_dir}/train_{version}{filename}.pickle', 'rb') as f:
                train_img, train_mask = pickle.load(f)
                both_tr_img.extend(train_img)
                both_tr_mask.extend(train_mask)

        train_img = np.array(both_tr_img)
        train_mask = np.array(both_tr_mask)

    print(f'train img shape : {train_img.shape}, train mask shape : {train_mask.shape}')
    print(f'test img shape : {test_img.shape}, test mask shape : {test_mask.shape}')

    train_data = CustomDataset(train_img, train_mask, transform=transform)
    test_data = CustomDataset(test_img, test_mask, transform=base_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=5, verbose=False, delta=0, path='flower10.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def evaluate(model, testloader, mode='base'):
    img_list = []
    pred_mask_list = []
    gt_mask_list = []
    model.eval()
    with torch.no_grad():
        for img, gt_mask in testloader:

            output = model(img.cuda().float())
            if mode=='base': # 일반적인 모델
                pred_mask_list.append(output.cpu().numpy())
            elif mode=='caranet': # 종욱이 모델
                pred_mask_list.append(output[0].sigmoid().cpu().numpy()) 


            gt_mask_list.append(gt_mask.numpy())
    pred_mask_list = np.vstack(pred_mask_list)
    gt_mask_list = np.vstack(gt_mask_list)
    pred_mask_list_hard = ((pred_mask_list > 0.5) + 0)
    # print(gt_mask_list.shape, pred_mask_list_hard.shape)
    DS_list = []
    JS_list = []

    for i, gt_mask in enumerate(gt_mask_list):
        Inter = np.sum((pred_mask_list_hard[i] + gt_mask) == 2)
        DS_Union = np.sum(pred_mask_list_hard[i]) + np.sum(gt_mask)
        Union = np.sum((pred_mask_list_hard[i] + gt_mask) >= 1)
        DS = (Inter*2) / (DS_Union + 1e-8)
        JS = Inter/(Union + 1e-8)
        DS_list.append(DS)
        JS_list.append(JS)
    DS_mean = np.mean(DS_list)
    JS_mean = np.mean(JS_list)
    print(f'Dice Similarity:    {DS_mean:0.4f} \nJaccard Similarity: {JS_mean:0.4f}')
    return DS_mean, JS_mean

def precision_recall(pred_mask_list_hard, gt_mask_list):
    DS_list = []
    JS_list = []
    RC_list = []
    PC_list = []

    for i, gt_mask in enumerate(gt_mask_list):      
        Inter = np.sum((pred_mask_list_hard[i] + gt_mask) == 2) ## True positive
        FN = np.sum(((1-pred_mask_list_hard[i]) + gt_mask) == 2) ## False negative
        FP = np.sum((pred_mask_list_hard[i] + (1 - gt_mask)) == 2) ## False positive
        DS_Union = np.sum(pred_mask_list_hard[i]) + np.sum(gt_mask)
        Union = np.sum((pred_mask_list_hard[i] + gt_mask) >= 1)
        DS = (Inter*2) / (DS_Union + 1e-8)
        JS = Inter/(Union + 1e-8)
        RC = Inter/(Inter + FN + 1e-8) ## Recall
        PC = Inter/(Inter + FP + 1e-8) ## precision
        DS_list.append(DS)
        JS_list.append(JS)
        RC_list.append(RC)
        PC_list.append(PC)
        
    DS_mean = np.mean(DS_list)
    JS_mean = np.mean(JS_list)
    RC_mean = np.mean(RC_list)
    PC_mean = np.mean(PC_list)

