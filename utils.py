# Just some utility functions
import torch.optim as optim
import datasets
import torch
import torch.nn as nn
import data_transformations
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset(args):
    data_transforms = data_transformations.__dict__[args.data_transforms]
    train_supervised_dataset = datasets.__dict__[args.dataset](is_train = True, supervised = True, data_transforms = data_transforms)
    train_unsupervised_dataset = datasets.__dict__[args.dataset](is_train = True, supervised = False, data_transforms = data_transforms)
    val_dataset = datasets.__dict__[args.dataset](is_train = False, data_transforms = data_transforms)
    return train_supervised_dataset, train_unsupervised_dataset, val_dataset

def make_loader(args):
    train_supervised_dataset, train_unsupervised_dataset, val_dataset = get_dataset(args)
    train_supervised_loader = torch.utils.data.DataLoader(train_supervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_unsupervised_loader = torch.utils.data.DataLoader(train_unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return train_supervised_loader, train_unsupervised_loader, val_loader

def select_optimizer(args, model):
    if args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return optimizer

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).sum()
            res.append(correct_k)
        return res

def compute_mean_and_std():
	# dataset = datasets.ssl_data(is_train = True, supervised = True, data_transforms = data_transformations.tensor_transform)
	dataset = datasets.cifar10(is_train = True, supervised = True, data_transforms = data_transformations.tensor_transform)
	loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

	mean = 0.
	std = 0.

	for batch_idx, (data, _) in enumerate(loader):
		images = torch.tensor(data.to(device))
		batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
		images = images.view(batch_samples, images.size(1), -1)
		mean += images.mean(2).sum(0)
		std += images.std(2).sum(0)

	mean /= len(loader.dataset)
	std /= len(loader.dataset)
	print(mean)
	print(std)