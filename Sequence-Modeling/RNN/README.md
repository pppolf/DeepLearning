## RNN 时序建模

> 本项目基于 RNN 循环神经网络，主要探讨 情感分析、时间序列预测、语言建模 方面的知识

### 情感分析

> 给定一段文本，判断其情感是正面、负面（或中性）。

- 数据集：IMDb 电影评论情感分类（二分类）下载地址：[Kaggle数据集](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)
- 使用基础 RNN 模型，训练 20 轮，数据集划分 80% 训练集、20% 测试集。
- 最好测试集 ACC=72.83%
- AI生成几句英文进行预测该句的情感极性
- 具体 Demo 可在 `./rnn_imdb.ipynb` 找到

### 时间序列预测
  - 数据集：