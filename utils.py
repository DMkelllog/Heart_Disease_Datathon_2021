import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle
import torchvision.transforms as transforms
import albumentations as A
from sklearn.model_selection import train_test_split
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=False):
        
        self.X = X
        self.y = y

        self.transform = transform
        
    def __getitem__(self, index):
        img, mask = self.X[index], self.y[index]

        if type(self.transform) == transforms.Compose:
            img = self.transform(Image.fromarray((img*255).astype(np.uint8)))
            mask = self.transform(Image.fromarray((np.squeeze(mask, axis=2)*255).astype(np.uint8)))

        elif type(self.transform) == A.Compose:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask

    def __len__(self):
        return len(self.X)

def create_loader(transform, random_seed=42, batch_size=16, mode='base'):
    #for train & val
    base_transform = A.Compose([
        A.pytorch.ToTensorV2(transpose_mask=True)
    ])
    train_mode = 'train'
    img, mask = [], []
    if mode == 'base':
        filename = ''
    elif mode == 'caranet':
        filename = '_2'
        
    for version in ['A2C', 'A4C']:
        with open(f'data/{train_mode}_{version}{filename}.pickle', 'rb') as f:
            X, y = pickle.load(f)
        img.append(X)
        mask.append(y)
    train_2 = [img[0], mask[0]]
    train_4 = [img[1], mask[1]]
    train_2_4 = [np.array(img).reshape(-1, X[0].shape[1], X[0].shape[1], 3), np.array(mask).reshape(-1, X[0].shape[1], X[0].shape[1], 1)]
    print(f'train image shape: {train_2_4[0].shape} \ntrain mask shape: {train_2_4[1].shape}')

    # for test
    train_mode = 'validation'
    img, mask = [], []
    for version in ['A2C', 'A4C']:
        with open(f'data/{train_mode}_{version}{filename}.pickle', 'rb') as f:
            X, y = pickle.load(f)
        img.append(X)
        mask.append(y)
    test_2 = [img[0], mask[0]]
    test_4 = [img[1], mask[1]]
    test_2_4 = [np.array(img).reshape(-1, X[0].shape[1], X[0].shape[1], 3), np.array(mask).reshape(-1, X[0].shape[1], X[0].shape[1], 1)]
    print(f'test image shape: {test_2_4[0].shape} \ntest mask shape: {test_2_4[1].shape}')

    train_2_4_img, val_2_4_img = train_test_split(train_2_4[0], test_size=0.125, random_state=random_seed)
    train_2_4_mask, val_2_4_mask = train_test_split(train_2_4[1], test_size=0.125, random_state=random_seed)

    train_2_img, val_2_img = train_test_split(train_2[0], test_size=0.125, random_state=random_seed)
    train_2_mask, val_2_mask = train_test_split(train_2[1], test_size=0.125, random_state=random_seed)

    train_4_img, val_4_img = train_test_split(train_4[0], test_size=0.125, random_state=random_seed)
    train_4_mask, val_4_mask = train_test_split(train_4[1], test_size=0.125, random_state=random_seed)

    train_2_4_dataset = CustomDataset(train_2_4_img, train_2_4_mask, transform=transform)
    val_2_4_dataset = CustomDataset(val_2_4_img, val_2_4_mask, transform=base_transform)
    test_2_4_dataset = CustomDataset(test_2_4[0], test_2_4[1], transform=base_transform)

    train_2_dataset = CustomDataset(train_2_img, train_2_mask, transform=transform)
    val_2_dataset = CustomDataset(val_2_img, val_2_mask, transform=base_transform)
    test_2_dataset = CustomDataset(test_2[0], test_2[1], transform=base_transform)

    train_4_dataset = CustomDataset(train_4_img, train_4_mask, transform=transform)
    val_4_dataset = CustomDataset(val_4_img, val_4_mask, transform=base_transform)
    test_4_dataset = CustomDataset(test_4[0], test_4[1], transform=base_transform)

    train_2_4_loader = DataLoader(train_2_4_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_2_4_loader = DataLoader(val_2_4_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_2_4_loader = DataLoader(test_2_4_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_2_loader = DataLoader(train_2_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_2_loader = DataLoader(val_2_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_2_loader = DataLoader(test_2_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_4_loader = DataLoader(train_4_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_4_loader = DataLoader(val_4_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_4_loader = DataLoader(test_4_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return (train_2_4_loader, val_2_4_loader, test_2_4_loader), (train_2_loader, val_2_loader, test_2_loader), (train_4_loader, val_4_loader, test_4_loader)

def make_dataloader(dataset, batch_size, transform=False, shuffle=False):
    loader = DataLoader(dataset, batch_size, shuffle=shuffle)
    return loader

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
            output = model(img.cuda())
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