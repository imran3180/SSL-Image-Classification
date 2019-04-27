import torchvision.transforms as transforms

__all__ = ['tensor_transform']

# custom class for data transformation can also be written
tensor_transform = transforms.Compose([
    transforms.ToTensor()
])