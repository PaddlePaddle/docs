## [ torch 参数更多 ]torch.nn.TransformerDecoderLayer
### [torch.nn.TransformerDecoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html?highlight=transformerdecoderlayer#torch.nn.TransformerDecoderLayer)

```python
torch.nn.TransformerDecoderLayer(d_model,
                                 nhead,
                                 dim_feedforward=2048,
                                 dropout=0.1,
                                 activation="relu',
                                 layer_norm_eps=1e-05,
                                 batch_first=False,
                                 norm_first=False,
                                 device=None,
                                 dtype=None)
```
### [paddle.nn.TransformerDecoderLayer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/TransformerDecoderLayer_cn.html#transformerdecoderlayer)

```python
paddle.nn.TransformerDecoderLayer(d_model,
                                  nhead,
                                  dim_feedforward=2048,
                                  dropout=0.1,
                                  activation="relu',
                                  attn_dropout=None,
                                  act_dropout=None,
                                  normalize_before=False,
                                  weight_attr=None,
                                  bias_attr=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| d_model     |      d_model       | 表示输入的维度。  |
| nhead     | nhead            | 表示多头注意力机制的 head 数量。  |
| dim_feedforward     | dim_feedforward            | 前馈神经网络中隐藏层的大小。  |
| dropout      | dropout            | dropout 值。  |
| activation     | activation           | 前馈神经网络的激活函数。  |
| layer_norm_eps | -       | eps 值，Paddle 无此参数，暂无转写方式。  |
| batch_first     | -      | 输入和输出 tensor 的 shape，Paddle 无此参数，暂无转写方式  |
| norm_first             | normalize_before  | 设置对每个子层的输入输出的处理。如果为 True，则对每个子层的输入进行层标准化（Layer Normalization），对每个子层的输出进行 dropout 和残差连接（residual connection）。否则（即为 False），则对每个子层的输入不进行处理，只对每个子层的输出进行 dropout、残差连接（residual connection）和层标准化（Layer Normalization）。默认值：False。  仅参数名不一致|
| device        | -            | 设备类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。        |
| dtype         | -            | 参数类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。        |
| -             | weight_attr  | 指定权重参数的属性，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | bias_attr    | 指定偏置参数的属性, PyTorch 无此参数，Paddle 保持默认即可。 |
