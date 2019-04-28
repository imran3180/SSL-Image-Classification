import torchvision.transforms as transforms

__all__ = ['tensor_transform']

# custom class for data transformation can also be written
tensor_transform = transforms.Compose([
    transforms.ToTensor()
])

resnet_input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

cifar10_input_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])