import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

__all__ = ['conv_net']

class ConvNet(nn.Module):
    def __init__(self, nclasses):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(25600, 2048)
        self.fc2 = nn.Linear(2048, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, 25600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# model-name - should be same as value provided in the argument for model
def conv_net(**kwargs):
    model = ConvNet(kwargs['nclasses'])
    return model