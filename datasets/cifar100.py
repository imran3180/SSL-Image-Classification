from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pdb
import torchvision
import random

class CIFAR100Dataset(Dataset):
	def __init__(self, is_train = True, supervised = True, data_transforms = None):
		self.data = list(torchvision.datasets.CIFAR100("data/", train = True, transform = data_transforms, download = False))
		random.shuffle(self.data)
		if is_train:
			if supervised:
				self.data = self.data[0:4000]
			else:
				self.data = self.data[4000:45000]
		else:
			# return validation data is_train = False => Validation data
			self.data = self.data[45000:50000]
			

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

def cifar100(**kwargs):
    return CIFAR100Dataset(**kwargs)

cifar100.nclasses = 100  # ugly but works
