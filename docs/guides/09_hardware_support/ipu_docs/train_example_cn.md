# 飞桨框架IPU版训练示例

## BERT-Base训练示例

示例将默认用户已安装飞桨框架IPU版，并且已经配置运行时需要的环境(建议在Docker环境中使用飞桨框架IPU版)。

示例代码位于 [Paddle-BERT with Graphcore IPUs](https://github.com/PaddlePaddle/PaddleNLP/examples/language_model/bert/static_ipu)

**第一步**：下载源码并安装依赖

```
# 下载源码
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd paddlenlp/examples/language_model/bert/static_ipu

# 安装依赖
pip install -r requirements.txt
```

**第二步**：准备数据集

按照 `README.md` 的描述准备用于预训练的数据集。

**第二步**：执行模型训练

按照 `README.md` 的描述开始BERT-Base模型的预训练和在SQuAD v1.1数据集上的模型微调。
