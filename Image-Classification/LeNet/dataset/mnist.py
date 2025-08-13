from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(f"MNIST 训练数据集大小：{len(train_dataset)}")
    print(f"MNIST 测试数据集大小：{len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader

def get_mnist_dataset():
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset