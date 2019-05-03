import torchvision.transforms as transforms

__all__ = ['tensor_transform']

# custom class for data transformation can also be written

# Use for the SSL dataset
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5032, 0.4746, 0.4275), (0.2276, 0.2228, 0.2265))
])

# Use for the SSL dataset

def resnet_input_transform(training = True):
    if training:
        return transforms.Compose([
            transforms.RandomAffine((-5,5), translate=(0.1,0.1), scale=(0.9, 1.1), shear=None, resample=False, fillcolor=0),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5032, 0.4746, 0.4275), (0.2276, 0.2228, 0.2265))
        ])
    else:   # confirm it again no need of data augmentation in validation
        return transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5032, 0.4746, 0.4275), (0.2276, 0.2228, 0.2265))
        ])

# use only for the CIFAR Dataset
def cifar10_input_transform(training = True):
    if training:
        return transforms.Compose([
            transforms.RandomAffine((-5,5), translate=(0.1,0.1), scale=(0.9, 1.1), shear=None, resample=False, fillcolor=0),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.4925, 0.4828, 0.4464), (0.2024, 0.1998, 0.2006))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.4925, 0.4828, 0.4464), (0.2024, 0.1998, 0.2006))
        ])
# SSL Data
# transforms.Normalize((0.5032, 0.4746, 0.4275), (0.2276, 0.2228, 0.2265))

# for CIFAR data
# transforms.Normalize((0.4925, 0.4828, 0.4464), (0.2024, 0.1998, 0.2006))
