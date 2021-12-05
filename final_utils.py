import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
import pickle
from preprocess import *
from torch.utils.data import Dataset
class TestDataset(Dataset):
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
        return img, mask

    def __len__(self):
        return len(self.X)

def make_dataset(data_path, data_type, transform):
    data_p = f'{data_path}/{data_type}'
    img_list = sorted(os.listdir(data_p))
    size_dict = {"size":[], "cut_off":[]}
    for img_name in img_list:
                # print(img_name)
        if img_name.endswith('.png'):
            img = plt.imread(f'{data_p}/' + img_name)
            mask = np.load(f'{data_p}/' + img_name.replace('png', 'npy')) * 255
            h, w = mask.shape[0], mask.shape[1]
            cutoff, img, mask = remove_topnoise(img, mask)

            size_dict["size"].append((h,w))
            size_dict["cut_off"].append(cutoff)

            img, mask = resize_crop(img, mask)

            img = img.numpy()
            img = img.transpose(1, 2, 0)

            mask = mask.numpy()
            mask = mask.transpose(1, 2, 0)
            mask = (mask>0.5)+0
            
            plt.imsave(f'{data_p}/img/' + img_name, img[:,:,:3])
            np.save(f'{data_p}/mask/' + img_name.replace('png', 'npy'), mask)
    

    # make pickle

    img_list = sorted(os.listdir(f'{data_p}/img/'))

    for img_name in img_list:
        img = plt.imread(f'{data_p}/img/{img_name}')
        mask = np.load(f'{data_p}/mask/{img_name[:-4]}.npy')

        X_list.append(img[:, :, :3])
        y_list.append(mask)
    dataset = TestDataset(X_list, y_list, transform = transform)
    return size_dict, dataset

def final_evaluate(model, test_dataset, size_dict, mode='base'):
    DS_list = []
    JS_list = []
    model.eval()
    with torch.no_grad():
        for i, img, gt_mask in enumerate(test_dataset):

            output = model(img.cuda().float())

            if mode=='base': # 일반적인 모델
                pred_mask = resize_return(output, size_dict["cutoff"][i], size_dict["size"][i], 100)

            elif mode=='caranet': # 종욱이 모델
                pred_mask = resize_return(output[0], size_dict["cutoff"][i], size_dict["size"][i], 88)
            pred_mask_hard = ((pred_mask > 0.5) + 0)
            DS, JS = metrics(pred_mask_hard, gt_mask)
            DS_list.append(DS)
            JS_list.append(JS)

    DS_mean = np.mean(DS_list)
    JS_mean = np.mean(JS_list)
    print(f'Dice Similarity:    {DS_mean:0.4f} \nJaccard Similarity: {JS_mean:0.4f}')
    return DS_mean, JS_mean


def metrics(pred_mask_hard, gt_mask):

    Inter = np.sum((pred_mask_hard + gt_mask) == 2)
    DS_Union = np.sum(pred_mask_hard) + np.sum(gt_mask)
    Union = np.sum((pred_mask_hard + gt_mask) >= 1)
    DS = (Inter*2) / (DS_Union + 1e-8)
    JS = Inter/(Union + 1e-8)
    return DS, JS


def resize_return(y_pred, cutoff, originsize, pad_size): ## cutoff, originsize(tuple)
    h, w = originsize
    h -= cutoff

    inverse = transforms.Compose([
                                transforms.Pad(padding=(pad_size, 0), fill=0),
                                transforms.Resize((h,w), interpolation=InterpolationMode.NEAREST),
                                transforms.Pad(padding=(0,cutoff,0,0), fill=0)
                                 ])
    
    y_pred_inverse = inverse(y_pred)

    return y_pred_inverse
