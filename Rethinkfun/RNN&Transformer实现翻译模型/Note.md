# 项目结构的简要说明
## RNN模型
- 文件`translator.py`基于RNN，训练速度较慢，使用NVIDIA GeForce RTX 4060 Laptop GPU可能要练两三天
## Transformer模型
1. 文件`transformer.py`包含整个transformer架构的所有nn.Module,以及这些组件的组装
2. 文件`train.py`基于`transformer.py`，运行后输出一个模型`transformer.pt`
3. 文件`inference.py`可以实时将英文翻译为中文
4. 文件`BLEU_score.py`用于评估整个模型的BLEU分数
