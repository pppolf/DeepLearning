import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
from dataset.mnist import get_mnist_loader
from dataset.cifar10 import get_cifar10_loader
from dataset.cifar100 import get_cifar100_loader
from tqdm import tqdm
import os
import sys
from model.resnet import resnet18, resnet34, resnet50, resnet101
    
def get_model(config, num_classes):
    if config['model'] == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif config['model'] == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif config['model'] == 'resnet50':
        return resnet50(num_classes=num_classes)
    elif config['model'] == 'resnet101':
        return resnet101(num_classes=num_classes)
    
def get_loader(config):
    if config['data'] == 'mnist':
        return get_mnist_loader(config['batch_size'])
    elif config['data'] == 'cifar10':
        return get_cifar10_loader(config['batch_size'])
    elif config['data'] == 'cifar100':
        return get_cifar100_loader(config['batch_size'])

# 训练过程
def train(model, loader, optimizer, criterion, epoch, epochs):
    '''
    model: 模型
    loader: 数据加载器
    optimizer: 优化器
    criterion: 损失函数
    epoch: 当前训练轮数
    epochs: 总训练轮数
    '''
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    # 使用 tqdm 显示训练进度条
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
    for input, labels in pbar:
        input, labels = input.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_train += (output.argmax(1) == labels).sum().item()
        total_train += labels.size(0)

        pbar.set_postfix({
            "loss": loss.item(),
            "acc": 100 * correct_train / total_train if total_train else 0
        })

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")

# 测试过程
def test(model, loader, criterion, epoch, epochs):
    '''
    model: 模型
    loader: 数据加载器
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for input, labels in pbar:
            input, labels = input.to(device), labels.to(device)
            output = model(input)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            loss = criterion(output, labels)
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": 100 * correct / total if total else 0
            })
    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc

def main(config):
    RESULT_PATH = config['result_path']
    os.makedirs(RESULT_PATH, exist_ok=True)

    num_classes = 100 if config['data'] == 'cifar100' else 10

    model = get_model(config, num_classes).to(device)
    train_loader, test_loader = get_loader(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_acc = 0.0
    epochs = config['epochs']
    for epoch in range(1, epochs+1):
        train(model, train_loader, optimizer, criterion, epoch, epochs)
        acc = test(model, test_loader, criterion, epoch, epochs)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{RESULT_PATH}/{config['model']}_{config['data']}.pth")
            print(">> Saved best model.")


def get_config():
    parser = argparse.ArgumentParser()
    ######################### 加载config.yml配置文件 #############################
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    ######################### 加载手动配置 #############################
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--data', default='mnist', type=str, help='Dataset name', choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--model', default='resnet18', type=str, help='Model name', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
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