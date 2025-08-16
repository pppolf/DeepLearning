## Transformer 翻译(英 -> 中)

### 环境配置

安装分词器：`pip install transformer=4.36.0`，一定要是这个版本才行！

### 模型运行

- `train.py` 直接在目录运行 `python train.py` 即可进行 Transformer 模型的训练
- `inference.py` 翻译模块，在训练完成模型后会保存一个 `model.pth`，之后在翻译模块里自己输入想要翻译的英文，然后运行 `python inference.py` 就可以翻译了