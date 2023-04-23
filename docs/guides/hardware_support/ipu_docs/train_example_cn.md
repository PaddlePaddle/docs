# 飞桨框架 IPU 版训练示例

## BERT-Base 训练示例

示例将默认用户已安装飞桨框架 IPU 版，并且已经配置运行时需要的环境(建议在 Docker 环境中使用飞桨框架 IPU 版)。

示例代码位于 [Paddle-BERT with Graphcore IPUs](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/bert/static_ipu)

**第一步**：下载源码并安装依赖

```
# 下载源码
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd model_zoo/bert/static_ipu/

# 安装依赖
pip install -r requirements.txt
```

**第二步**：准备数据集

按照 `README.md` 的描述准备用于预训练的数据集。

**第三步**：执行模型训练

按照 `README.md` 的描述开始 BERT-Base 模型的预训练和在 SQuAD v1.1 数据集上的模型微调。
