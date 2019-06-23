from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torchvision
import random
import pdb
import numpy as np

class RotatedCIFAR10Dataset(Dataset):
    def __init__(self, is_train=True, supervised = True, data_transforms = None):
        self.data = list(torchvision.datasets.CIFAR10("data/", train = True, transform = data_transforms, download = True))
        random.shuffle(self.data)
        if is_train:
            if supervised:
                self.data = self.data[0:4000]
            else:
                self.data = self.data[4000:45000]
        else:
            self.data = self.data[45000:50000]            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        rotated_imgs = [
            torch.from_numpy(self.rotate_img(img,   0)),
            torch.from_numpy(self.rotate_img(img,  90).transpose(1,2,0)),
            torch.from_numpy(self.rotate_img(img, 180)),
            torch.from_numpy(self.rotate_img(img, 270).transpose(1,2,0))
        ]
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        rotated_imgs = torch.stack(rotated_imgs, dim=0)
        return rotated_imgs, rotation_labels

    def rotate_img(self, img, rot):
        if rot == 0: # 0 degrees rotation
            return img.numpy()
        elif rot == 90: # 90 degrees rotation
            return np.flipud(np.transpose(img, (1,0,2))).copy()
        elif rot == 180: # 90 degrees rotation
            return np.fliplr(np.flipud(img)).copy()
        elif rot == 270: # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img).copy(), (1,0,2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def rotatedcifar10(**kwargs):
    return RotatedCIFAR10Dataset(**kwargs)

rotatedcifar10.nclasses = 4  # ugly but works