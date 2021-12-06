import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
import pickle
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestDataset(Dataset):
    def __init__(self, X, y, transform=False):
        self.X = X
        self.y = y

        self.transform = transform
        
    def __getitem__(self, index):
        img, mask = self.X[index], self.y[index]
        img = (img*255).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        return img, mask

    def __len__(self):
        return len(self.X)

def remove_topnoise(img, mask=False):
    cand1 = int(img.shape[1] / 8 * 7)
    cand2 = int(img.shape[1] / 8 * 6)

    cutoff1 = np.where(img[:, cand1, :] < 0.01)[0][0]
    cutoff2 = np.where(img[:, cand2, :] < 0.01)[0][0]


    cutoff_min = np.min([cutoff1, cutoff2])
    if mask is not False:
        return cutoff_min, img[cutoff_min:, ], mask[cutoff_min:, ]
    else:
        return cutoff_min, img[cutoff_min:, ]
    
def resize_crop(img, mask=False, resize_h=352, resize_w=528, crop_size=88):
    resize = transforms.Compose([transforms.ToTensor(), transforms.Resize((resize_h, resize_w))])

    img = resize(img)
    img = img[:, :, crop_size:-crop_size]
    if mask is not False:
        mask = resize(mask)
        mask = mask[:, :, crop_size:-crop_size]
        return img, mask
    else:
        return img


def make_test_dataset(data_path, data_type, transform, mode='caranet'):
    data_p = f'{data_path}/{data_type}'
    img_list = sorted(os.listdir(data_p))
    size_dict = {"size":[], "cut_off":[]}
    for img_name in img_list:
        # print(img_name)
        if img_name.endswith('.png'):
            img = plt.imread(f'{data_p}/' + img_name)
            h, w = img.shape[0], img.shape[1]
            cutoff, img = remove_topnoise(img)

            size_dict["size"].append((h,w))
            size_dict["cut_off"].append(cutoff)
            if mode == 'caranet':
                img = resize_crop(img)
            else:
                img = resize_crop(img, resize_h=400, resize_w=600, crop_size=100)


            img = img.numpy()
            img = img.transpose(1, 2, 0)
            
            os.makedirs(f'{data_p}/img/', exist_ok=True)
            plt.imsave(f'{data_p}/img/' + img_name, img[:,:,:3])
            # np.save(f'{data_p}/mask/' + img_name.replace('png', 'npy'), mask)
    

    # make pickle
    X_list=[]
    y_list=[]
    img_list = sorted(os.listdir(f'{data_p}/img/'))

    for img_name in img_list:
        img = plt.imread(f'{data_p}/img/{img_name}')
        mask = np.load(f"{data_p}/{img_name.replace('png','npy')}")
        X_list.append(img[:, :, :3])
        y_list.append(mask)

    dataset = TestDataset(X_list, y_list, transform=transform)
    return size_dict, dataset

def TTA_caranet(image, output, model, num):
    img_list = []
    pred_mask_list = []
  
    train_transform = A.Compose([
            A.OneOf(
                [
                    A.RandomContrast(p=1, ),
                ],
                p=1,
            ),
            ToTensorV2(transpose_mask=True)
    ])
    predict = output.cpu().sigmoid().numpy()
    for i in range(num):
        b = train_transform(image = np.array(image.squeeze().permute(1,2,0)))
 
        sub_predict = model(b["image"].view(1,3,352,352).float().cuda())
        predict = predict + sub_predict[0].sigmoid().cpu().numpy()
    
    mean_predict = predict/(num+1)
    
    return mean_predict

def final_evaluate(model, test_dataset, size_dict, threshold=0.5, mode='base'):
    DS_list = []
    JS_list = []
    model.eval()
    with torch.no_grad():
        for i, (img, gt_mask) in enumerate(test_dataset):
            
            output = model(img.cuda().float())
            #print(output[0].shape)

            if mode=='base': # 일반적인 모델
                output = ((output > threshold) + 0)
                pred_mask = resize_return(output, size_dict["cut_off"][i], size_dict["size"][i], 100)

            elif mode=='caranet': # 종욱이 모델
                output2 = torch.Tensor((TTA_caranet(img, output[0], model, 30)>threshold)+0)
                pred_mask = resize_return(output2.squeeze(0), size_dict["cut_off"][i], size_dict["size"][i], 88)
            
            pred_mask_hard = ((pred_mask > threshold) + 0)
            
            DS, JS = metrics(pred_mask_hard.cpu().numpy(), gt_mask.numpy())
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
                                transforms.Resize((h,w)),
                                transforms.Pad(padding=(0,cutoff,0,0), fill=0)
                                 ])
    
    y_pred_inverse = inverse(y_pred)

    return y_pred_inverse
