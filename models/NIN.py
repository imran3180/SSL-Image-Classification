import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb

## network and optimizer
class GlobalAveragePool(nn.Module):
    def __init__(self):
        super(GlobalAveragePool, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

class NIN(nn.Module):
    def __init__(self, num_classes):
        super(NIN, self).__init__()
        self.modules = [None for i in range(5)]
        self.modules[0] = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=160, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=96, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.modules[1] = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.modules[2] = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.modules[3] = GlobalAveragePool()
        self.modules[4] = nn.Linear(192, num_classes)
        self.model_list = nn.ModuleList(self.modules)

    def forward(self, input_x):
        for i, l in enumerate(self.model_list):
            input_x = self.model_list[i](input_x)
            # print(input_x.shape)
        return input_x

def rotnet(**kwargs):
    model = NIN(kwargs['nclasses'])
    return model