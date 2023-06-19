## [ torch 参数更多 ]torch.nn.GRU
### [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html?highlight=torch+nn+multiheadattention#torch.nn.MultiheadAttention)
```python
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
```

### [paddle.nn.MultiHeadAttention](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MultiHeadAttention_cn.html)
```python
paddle.nn.MultiHeadAttention(embed_dim, num_heads, dropout=0.0, kdim=None, vdim=None, need_weights=False, weight_attr=None, bias_attr=None)
```

Pytoch 参数更多，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| embed_dim          | embed_dim            | 表示输入 tensor 的维度。  |
| num_heads          | num_heads            | 表示 head 的数量。  |
| dropout          | dropout           | 表示 dropout 概率。  |
| bias          | bias_attr  | 是否使用偏置， Paddle 支持自定义偏置属性， torch 不支持。  |
| add_bias_kv   | -   | 是否在 key 和 value tensor 的`0`维添加 bias， Paddle 无此参数，暂无转写方式。 |
| add_zero_attn   | -   | 是否在 key 和 value tensor 的`1`维添加 zeros batch， Paddle 无此参数，暂无转写方式。  |
| kdim | kdim    | Key tenosr 的维度。 |
| vdim             |vdim| Value tensor 的维度。  |
| batch_first             |-| 是否使用第一维表示 batch_size， Paddle 无此参数，暂无转写方式。  |
| device            |-| 表示网络层的硬件位置，Paddle 无此参数， 一般对网络训练结果影响不大，可直接删除。  |
| dtype             |-| Tensor 的所需数据类型，Paddle 无此参数，暂无转写方式。  |
| -             |need_weights| 是否返回权重 tensor， Pytorch 无此参数， Paddle 保持默认即可。  |
| -            |weight_attr| 自定义权重属性， Pytorch 无此参数， Paddle 保持默认即可。  |
