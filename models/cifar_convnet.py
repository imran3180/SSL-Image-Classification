import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

__all__ = ['cifar_convnet']

class CIFAR10ConvNet(nn.Module):
    def __init__(self, nclasses):
        super(CIFAR10ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=7)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=4)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2250, 100)
        self.fc2 = nn.Linear(100, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, 2250)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# model-name - should be same as value provided in the argument for model
def cifar_convnet(**kwargs):
    model = CIFAR10ConvNet(kwargs['nclasses'])
    return model