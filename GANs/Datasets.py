import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np

def MnistLabel(class_num):
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    class_tot = [0] * 10
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))

def MnistUnlabel():
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    return raw_dataset

def MnistVal():
    return datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))

if __name__ == '__main__':
    print (dir(MnistVal()))

# image_size = 64
#
# def ImageNetLabel(class_num, n_sample):
#     raw_dataset = datasets.ImageFolder('../SSL-Image-Classification/data/ssl_data_96/supervised/train',
#                    transform=transforms.Compose([
#                        transforms.Resize(image_size),
#                        transforms.CenterCrop(image_size),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                    ]))
#     class_tot = [0] * class_num
#     data = []
#     labels = []
#     positive_tot = 0
#     tot = 0
#     perm = np.random.permutation(raw_dataset.__len__())
#     for i in range(raw_dataset.__len__()):
#         datum, label = raw_dataset.__getitem__(perm[i])
#         if class_tot[label] < n_sample:
#             data.append(datum.numpy())
#             labels.append(label)
#             class_tot[label] += 1
#             tot += 1
#             if tot >= n_sample * class_num:
#                 break
#     return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))
#
# def ImageNetUnlabel():
#     return datasets.ImageFolder('../SSL-Image-Classification/data/ssl_data_96/supervised/train',
#                    transform=transforms.Compose([
#                        transforms.Resize(image_size),
#                        transforms.CenterCrop(image_size),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                    ]))
#
# def ImageNetVal():
#     return datasets.ImageFolder('../SSL-Image-Classification/data/ssl_data_96/supervised/val',
#                    transform=transforms.Compose([
#                        transforms.Resize(image_size),
#                        transforms.CenterCrop(image_size),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                    ]))
#
# if __name__ == '__main__':
#     print (dir(ImageNetVal()))