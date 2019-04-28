import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb

__all__ = ['resnet18']

class ResNet18(nn.Module):
    def __init__(self, nclasses):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet34()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)

# model-name - should be same as value provided in the argument for model
def resnet18(**kwargs):
    model = ResNet18(kwargs['nclasses'])
    return model