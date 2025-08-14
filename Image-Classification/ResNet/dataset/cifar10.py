from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.build_transfoms import get_transform_cifar10

def get_cifar10_loader(batch_size=64):
    transform_train, transform_test = get_transform_cifar10()

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    print(f"CIFAR10 训练数据集大小：{len(train_dataset)}")
    print(f"CIFAR10 测试数据集大小：{len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader

def get_cifar10_dataset():
    transform_train, transform_test = get_transform_cifar10()

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset