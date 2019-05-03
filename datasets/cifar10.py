from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pdb
import torchvision
import random

class CIFAR10Dataset(Dataset):
	storage = None

	def __init__(self, is_train = True, supervised = True, data_transforms = None):
		if CIFAR10Dataset.storage is None:
			CIFAR10Dataset.storage = list(torchvision.datasets.CIFAR10("data/", train = True, transform = data_transforms, download = False))
			random.shuffle(CIFAR10Dataset.storage)
		
		self.data = CIFAR10Dataset.storage
		
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

def cifar10(**kwargs):
    return CIFAR10Dataset(**kwargs)

cifar10.nclasses = 10  # ugly but works
