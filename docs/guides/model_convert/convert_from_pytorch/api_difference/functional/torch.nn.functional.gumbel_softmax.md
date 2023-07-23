## [torch 参数更多]torch.nn.functional.gumbel_softmax

### [torch.nn.functional.gumbel_softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#torch.nn.functional.gumbel_softmax)

```python
torch.nn.functional.gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=- 1)
```

### [paddle.nn.functional.gumbel_softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gumbel_softmax_cn.html)

```python
paddle.nn.functional.gumbel_softmax(x, temperature=1.0, hard=False, axis=- 1, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                              |
| ------- | ------------ | ------------------------------------------------------------------------------------------------- |
| logits  | x            | 一个 N-D Tensor，前 N-1 维用于独立分布 batch 的索引，最后一维表示每个类别的概率，仅参数名不一致。 |
| tau     | temperature  | 大于 0 的标量，仅参数名不一致。                                                                   |
| hard    | hard         | 如果是 True，返回离散的 one-hot 向量。如果是 False，返回软样本。                                  |
| eps     | -            | eps 参数，Paddle 无此参数，暂无转写方式。                                                         |
| dim     | axis         | 按照维度 axis 计算 softmax，仅参数名不一致。                                                      |
