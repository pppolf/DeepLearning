## ResNet 经典卷积神经网络

> 此项目使用 ResNet 网络进行 MNIST、CIFAR10、CIFAR100 数据集进行演示

### 主要文件说明

- train.py: 训练模块
- dataset: 数据集模块
    - mnist.py: 加载 MNIST 数据集
    - cifar10.py: 加载 CIFAR10 数据集
    - cifar100.py: 加载 CIFAR100 数据集
- model/resnet.py: ResNet神经网络的结构
- config.yml: 参数配置文件(可选)

### config.yml 文件说明

此文件中用于调整各种训练参数

- batch_size: 一次传入的图片数量
- epochs: 总训练次数
- lr: 学习率
- data: 数据集选择，可选('mnist', 'cifar10', 'cifar100')
- model: 使用的模型，可选('resnet18', 'resnet34', 'resnet50', 'resnet101')
- result_path: 模型保存路径，默认('./result')

### 训练过程中产生的中间文件夹

- data: 下载的数据集
- result: 训练好的模型存放路径

### 运行脚本

- 训练脚本
  
1. 使用配置文件运行(config.yml)
`python train.py --config config.yml`
2. 使用命令行配置
`python train.py --batch_size=64 --epochs=20 --lr=0.001 --data='mnist' --model='resnet18' --result_path='./result'`
