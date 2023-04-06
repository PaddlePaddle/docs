## [ 仅 paddle 参数更多 ] torch.Tensor.outer

### [torch.Tensor.outer](https://pytorch.org/docs/stable/generated/torch.Tensor.outer.html?highlight=outer#torch.Tensor.outer)

```python
torch.Tensor.outer(vec2)
```

### [paddle.outer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/outer_cn.html)

```python
paddle.outer(x, y)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                 |
| ------- | ------------ | -------------------------------------------------------------------- |
| -    | x            | 一个 N 维 Tensor 或者标量 Tensor。 |
| vec2    | y            | 如果类型是多维 `Tensor`，其数据类型应该和 `x` 相同，仅参数名不一致。 |
