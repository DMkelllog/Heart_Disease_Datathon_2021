import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms


def remove_topnoise(img, mask):
    cand1 = int(img.shape[1] / 8 * 7)
    cand2 = int(img.shape[1] / 8 * 6)

    cutoff1 = np.where(img[:, cand1, :] < 0.01)[0][0]
    cutoff2 = np.where(img[:, cand2, :] < 0.01)[0][0]


    cutoff_min = np.min([cutoff1, cutoff2])

    return img[cutoff_min:, ], mask[cutoff_min:, ] 

def resize_crop(img, mask, resize_h=400, resize_w=600, crop_size=100):
    resize = transforms.Compose([transforms.ToTensor(), transforms.Resize((resize_h, resize_w))])

    img = resize(img)
    mask = resize(mask)

    img = img[:, :, crop_size:-crop_size]
    mask = mask[:, :, crop_size:-crop_size]

    return img, mask

for mode in ['train', 'validation']:
    for version in ['A2C', 'A4C']:
        os.makedirs(f'data/resize_crop/{mode}/{version}/img', exist_ok=True)
        os.makedirs(f'data/resize_crop/{mode}/{version}/mask', exist_ok=True)
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
                mask = mask.transpose(1, 2, 0) * 255
                
                plt.imsave(f'data/resize_crop/{mode}/{version}/img/' + img_name, img[:,:,:3])
                np.save(f'data/resize_crop/{mode}/{version}/mask/' + img_name.replace('png', 'npy'), mask)