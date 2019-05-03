from random import shuffle

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pdb
import torchvision

class ProxyDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]