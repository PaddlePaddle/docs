## [ 仅 paddle 参数更多 ]torch.Tensor.logcumsumexp

### [torch.Tensor.logcumsumexp](https://pytorch.org/docs/stable/generated/torch.Tensor.logcumsumexp.html?highlight=logcumsumexp#torch.Tensor.logcumsumexp)

```python
torch.Tensor.logcumsumexp(dim)
```

### [paddle.Tensor.logcumsumexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logcumsumexp_cn.html#logcumsumexp)

```python
paddle.Tensor.logcumsumexp(axis=None, dtype=None, name=None)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                                                           |
| ------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| dim     | axis         | 指明需要计算的维，仅参数名不同。 paddle 中默认 None。                                                                                          |
| -       | dtype        | 输出 Tensor 的数据类型，支持 float32、float64。如果指定了，那么在执行操作之前，输入张量将被转换为 dtype，PyTorch 无此参数，Paddle 保持默认即可。 |
