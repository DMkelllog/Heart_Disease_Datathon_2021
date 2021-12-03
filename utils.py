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