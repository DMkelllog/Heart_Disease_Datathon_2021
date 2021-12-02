import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
import pickle

def remove_topnoise(img, mask):
    cand1 = int(img.shape[1] / 8 * 7)
    cand2 = int(img.shape[1] / 8 * 6)

    cutoff1 = np.where(img[:, cand1, :] < 0.01)[0][0]
    cutoff2 = np.where(img[:, cand2, :] < 0.01)[0][0]


    cutoff_min = np.min([cutoff1, cutoff2])

    return img[cutoff_min:, ], mask[cutoff_min:, ] 

def resize_crop(img, mask, resize_h=352, resize_w=528, crop_size=88):
    resize = transforms.Compose([transforms.ToTensor(), transforms.Resize((resize_h, resize_w))])

    img = resize(img)
    mask = resize(mask)

    img = img[:, :, crop_size:-crop_size]
    mask = mask[:, :, crop_size:-crop_size]

    return img, mask

for mode in ['train', 'validation']:
    for version in ['A2C', 'A4C']:
        os.makedirs(f'data/resize_crop/{mode}/{version}_2/img', exist_ok=True)
        os.makedirs(f'data/resize_crop/{mode}/{version}_2/mask', exist_ok=True)
        print(f'{mode} {version}')
        img_list = sorted(os.listdir(f'data/original/{mode}/{version}'))
        for img_name in img_list:
            # print(img_name)
            if img_name.endswith('.png'):
                img = plt.imread(f'data/original/{mode}/{version}/' + img_name)
                mask = np.load(f'data/original/{mode}/{version}/' + img_name.replace('png', 'npy')) * 255
                img, mask = remove_topnoise(img, mask)
                img, mask = resize_crop(img, mask)

                img = img.numpy()
                img = img.transpose(1, 2, 0)

                mask = mask.numpy()
                mask = mask.transpose(1, 2, 0)
                mask = (mask>0.5)+0
                
                plt.imsave(f'data/resize_crop/{mode}/{version}_2/img/' + img_name, img[:,:,:3])
                np.save(f'data/resize_crop/{mode}/{version}_2/mask/' + img_name.replace('png', 'npy'), mask)

# make pickle
for mode in ['train', 'validation']:
    for version in ['A2C', 'A4C']:
        X_list = []
        y_list = []

        PATH = f'data/resize_crop/{mode}/{version}_2'
        img_list = sorted(os.listdir(f'{PATH}/img/'))

        for img_name in img_list:
            img = plt.imread(f'{PATH}/img/{img_name}')
            mask = np.load(f'{PATH}/mask/{img_name[:-4]}.npy')

            X_list.append(img[:, :, :3])
            y_list.append(mask)

        with open(f'data/{mode}_{version}_2.pickle', 'wb') as f:
            pickle.dump([np.array(X_list), np.array(y_list)], f)

print('pickle files created')