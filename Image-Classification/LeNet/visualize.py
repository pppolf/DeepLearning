import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from model import LeNet
import argparse
import yaml
from dataset.mnist import get_mnist_dataset
from dataset.cifar10 import get_cifar10_dataset
import numpy as np
import os
import sys

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def get_model(config):
    data = config['data']
    if data == 'mnist':
        return LeNet(1)
    elif data == 'cifar10':
        return LeNet(3)
    else:
        raise Exception('Vaild dataset name!')

def get_dataset(config):
    if config['data'] == 'mnist':
        return get_mnist_dataset()
    elif config['data'] == 'cifar10':
        return get_cifar10_dataset()
    

def denormalize(img_tensor):
    """
    将标准化后的图像 Tensor 转为 [0,1] 范围的 numpy 图像 (H,W,C)
    """
    # CIFAR-10 反归一化用的 mean 和 std
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img_tensor.cpu().numpy()
    if img.shape[0] == 1:
        # MNIST: 单通道，不反归一化
        img = img.squeeze(0)
        return img, 'gray'
    elif img.shape[0] == 3:
        # CIFAR-10: 反标准化
        img = img.transpose(1, 2, 0)  # C,H,W -> H,W,C
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img, None
    else:
        raise ValueError(f"Unsupported channel shape: {img.shape}")
    
def get_label_name(label, dataset_name="mnist"):
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    if dataset_name == "mnist":
        return str(label)
    elif dataset_name == "cifar10":
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        return cifar10_classes[label]

def main(config):
    RESULT_PATH = config['result_path']
    data = config['data']
    model = get_model(config).to(device)

    train_dataset, test_dataset = get_dataset(config)

    model.load_state_dict(torch.load(f'{RESULT_PATH}/lenet_{data}.pth', map_location=device))
    model.eval()

    # 随机选取5个测试样本
    indices = random.sample(range(len(test_dataset)), 5)
    samples = [test_dataset[i] for i in indices]
    images = [img for img, _ in samples]
    labels = [label for _, label in samples]

    # 图像转 batch，送入模型
    input_batch = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = model(input_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()

    # 展示图像、真实标签和预测结果
    plt.figure(figsize=(12, 3))
    for i in range(5):
        img, cmap = denormalize(images[i])
        plt.subplot(1, 5, i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(f"Label: {get_label_name(labels[i], config['data'])}\nPred: {get_label_name(preds[i], config['data'])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_config():
    parser = argparse.ArgumentParser()
    ######################### 加载config.yml配置文件 #############################
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    ######################### 加载手动配置 #############################
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--data', default='mnist', type=str, help='Dataset name', choices=['mnist', 'cifar10'])
    parser.add_argument('--result_path', default='./result', type=str, help='Path to save results')

    args, unknown = parser.parse_known_args()

    user_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            user_args.add(arg.lstrip('-').split('=')[0])

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if key not in user_args:
                setattr(args, key, value)

    return vars(args)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    config = get_config()
    main(config)