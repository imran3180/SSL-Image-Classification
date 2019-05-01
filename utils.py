# Just some utility functions
import datasets
import torch
import data_transformations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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