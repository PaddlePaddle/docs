## [torch 参数更多]torch.nn.TransformerEncoderLayer

### [torch.nn.TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer)

```python
torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
```

### [paddle.nn.TransformerEncoderLayer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/TransformerEncoderLayer_cn.html)

```python
paddle.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, activation='relu', attn_dropout=None, act_dropout=None, normalize_before=False, weight_attr=None, bias_attr=None, layer_norm_eps=1e-05)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch         | PaddlePaddle     | 备注                                                                                |
| --------------- | ---------------- | ----------------------------------------------------------------------------------- |
| d_model         | d_model          | 输入输出的维度。                                                                    |
| nhead           | nhead            | 多头注意力机制的 Head 数量。                                                        |
| dim_feedforward | dim_feedforward  | 前馈神经网络中隐藏层的大小。                                                        |
| dropout         | dropout          | 对两个子层的输出进行处理的 dropout 值。                                             |
| activation      | activation       | 前馈神经网络的激活函数。                                                            |
| layer_norm_eps  | layer_norm_eps   | 层 normalization 组件的 eps 值。                                                  |
| batch_first     | -                | 表示输入数据的第 0 维是否代表 batch_size，Paddle 无此参数，暂无转写方式。           |
| norm_first      | normalize_before | 是否 LayerNorms 操作在 attention 和 feedforward 前，仅参数名不一致。                |
| device          | -                | Tensor 的设备，Paddle 无此参数，需要转写。                                      |
| dtype           | -                | Tensor 的数据类型，Paddle 无此参数，需要转写。                                  |
| -               | attn_dropout     | 多头自注意力机制中对注意力目标的随机失活率，PyTorch 无此参数，Paddle 保持默认即可。 |
| -               | act_dropout      | 前馈神经网络的激活函数后的 dropout，PyTorch 无此参数，Paddle 保持默认即可。         |
| -               | weight_attr      | 指定权重参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。                     |
| -               | bias_attr        | 指定偏置参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。                     |

### 转写示例

#### device：Tensor 的设备

```python
# PyTorch 写法
m = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward，device=torch.device('cpu'))
y = m(x)

# Paddle 写法
m = paddle.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
y = m(x).cpu()
```

#### dtype：Tensor 的数据类型

```python
# PyTorch 写法
m = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward，dtype=torch.float32)
y = m(x)

# Paddle 写法
m = paddle.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
y = m(x).astype(paddle.float32)
```
