from torchvision import datasets

def ssl_data(**kwargs):
	if kwargs['is_train']:
		if kwargs['supervised']:
			dataset = datasets.ImageFolder('data/ssl_data_96/supervised/train', transform = kwargs['data_transforms'])
		else:
			dataset = datasets.ImageFolder('data/ssl_data_96/unsupervised', transform = kwargs['data_transforms'])	
	else:
		dataset = datasets.ImageFolder('data/ssl_data_96/supervised/val', transform = kwargs['data_transforms'])
	return dataset

ssl_data.nclasses = 1000  # ugly but works