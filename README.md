# DeepLearning

> 存放深度学习的各种模型

## 环境配置

- conda虚拟环境 python=3.8
- torch版本1.13.1
- 安装GPU（CPU）版本，安装GPU版本之前需要安装CUDA
  - GPU：`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
  - CPU：`pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu`
- 附：python版本和torch版本的对应关系：https://blog.csdn.net/ttrr27/article/details/144162171
- 项目所使用的各种包，位于 `requirement.txt`，直接在环境中运行 `pip install -r requirements.txt` 即可安装。

## 目录分类

- Image-Classification: 图像分类模型(LeNet, ResNet)
- SequenceModeling: 时序建模(RNN, LSTM, Transformer)
