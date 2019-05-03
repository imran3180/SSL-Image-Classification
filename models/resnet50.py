import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb

__all__ = ['resnet50']

class ResNet50(nn.Module):
    def __init__(self, nclasses):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50()

    def forward(self, x):
        x = self.resnet(x)
        return x

# model-name - should be same as value provided in the argument for model
def resnet50(**kwargs):
    model = ResNet50(kwargs['nclasses'])
    return model