## [torch 参数更多]torch.nn.MultiheadAttention

### [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention)

```python
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
```

### [paddle.nn.MultiHeadAttention](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/MultiHeadAttention_cn.html)

```python
paddle.nn.MultiHeadAttention(embed_dim, num_heads, dropout=0.0, kdim=None, vdim=None, need_weights=False, weight_attr=None, bias_attr=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                      |
| ------------- | ------------ | ------------------------------------------------------------------------- |
| embed_dim     | embed_dim    | 输入输出的维度。                                                          |
| num_heads     | num_heads    | 多头注意力机制的 Head 数量。                                              |
| dropout       | dropout      | 注意力目标的随机失活率。                                                  |
| bias          | bias_attr    | 指定偏置参数属性的对象，Paddle 支持更多功能，同时支持 bool 用法。         |
| add_bias_kv   | -            | 在 dim=0 增加 bias，Paddle 无此参数，暂无转写方式。                       |
| add_zero_attn | -            | 在 dim=1 增加批量 0 值，Paddle 无此参数，暂无转写方式。                   |
| kdim          | kdim         | 键值对中 key 的维度。                                                     |
| vdim          | vdim         | 键值对中 value 的维度。                                                   |
| batch_first   | -            | 表示输入数据的第 0 维是否代表 batch_size，Paddle 无此参数，暂无转写方式。 |
| device        | -            | Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                            |
| dtype         | -            | Tensor 的数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                        |
| -             | need_weights | 表明是否返回注意力权重，PyTorch 无此参数，Paddle 保持默认即可。           |
| -             | weight_attr  | 指定权重参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。           |
